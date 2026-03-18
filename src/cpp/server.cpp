/**
 * YOLO TensorRT 推理服务 - 方案 A: gRPC + Shared Memory
 * 
 * 数据流: Python -> 共享内存 -> TensorRT
 * 控制流: gRPC (轻量指令)
 * 
 * 构建:
 *   cd src/cpp
 *   mkdir build && cd build
 *   cmake .. -DTENSORRT_ROOT=/usr/local/TensorRT
 *   make -j4
 * 
 * 运行:
 *   ./yolo_server --engine best.trt --port 50051
 */

#include <iostream>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include <cstring>
#include <chrono>
#include <thread>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <grpcpp/grpcpp.h>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <NvInfer.h>

#include "yolo.grpc.pb.h"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;
using yolo::YoloInference;
using yolo::InferRequest;
using yolo::ImageData;
using yolo::DetectionResult;
using yolo::BoundingBox;
using yolo::HealthRequest;
using yolo::HealthResponse;
using namespace nvinfer1;

namespace {
class TrtLogger final : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TRT] " << msg << std::endl;
        }
    }
};

TrtLogger gTrtLogger;

template <typename T>
void trtDestroy(T*& p) {
    if (p) {
        p->destroy();
        p = nullptr;
    }
}
}  // namespace

// ============== 配置 ==============
struct Config {
    int input_w = 640;
    int input_h = 640;
    int input_c = 3;
    int num_classes = 1;
    float conf_threshold = 0.5f;
    float nms_threshold = 0.45f;
    int num_anchors = 8400;
};

// CUDA 错误检查
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "[CUDA] Error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            return false; \
        } \
    } while(0)

// ============== Shared Memory 读取 ==============
class SharedMemoryReader {
public:
    SharedMemoryReader() : shm_fd(-1), shm_addr(nullptr), size(0) {}
    
    ~SharedMemoryReader() { close(); }
    
    bool open(const std::string& buffer_name, size_t expected_size = 0) {
        close();
        
        // 打开共享内存
        std::string shm_path = "/dev/shm/" + buffer_name;
        shm_fd = shm_open(shm_path.c_str(), O_RDWR, 0666);
        if (shm_fd == -1) {
            std::cerr << "[Shm] Failed to open: " << shm_path << std::endl;
            return false;
        }
        
        // 获取大小
        struct stat stat_buf;
        if (fstat(shm_fd, &stat_buf) == -1) {
            std::cerr << "[Shm] Failed to get size" << std::endl;
            close();
            return false;
        }
        size = stat_buf.st_size;
        
        if (expected_size > 0 && size < expected_size) {
            std::cerr << "[Shm] Size mismatch: " << size << " < " << expected_size << std::endl;
            close();
            return false;
        }
        
        // 内存映射
        shm_addr = mmap(nullptr, size, PROT_READ, MAP_SHARED, shm_fd, 0);
        if (shm_addr == MAP_FAILED) {
            std::cerr << "[Shm] Failed to mmap" << std::endl;
            close();
            return false;
        }
        
        std::cout << "[Shm] Opened: " << shm_path << " (" << size << " bytes)" << std::endl;
        return true;
    }
    
    bool read_image(int& width, int& height, std::vector<uint8_t>& data) {
        if (!shm_addr || size < 16) {
            return false;
        }
        
        // 读取头信息 [width, height, channels, data_size]
        uint32_t header[4];
        memcpy(header, shm_addr, 16);
        
        width = header[0];
        height = header[1];
        uint32_t data_size = header[3];
        
        if (16 + data_size > size) {
            std::cerr << "[Shm] Data size error: " << data_size << " > " << (size - 16) << std::endl;
            return false;
        }
        
        // 读取图像数据
        data.resize(data_size);
        memcpy(data.data(), (uint8_t*)shm_addr + 16, data_size);
        
        return true;
    }
    
    void close() {
        if (shm_addr && shm_addr != MAP_FAILED) {
            munmap(shm_addr, size);
        }
        if (shm_fd != -1) {
            ::close(shm_fd);
        }
        shm_addr = nullptr;
        shm_fd = -1;
        size = 0;
    }
    
private:
    int shm_fd;
    void* shm_addr;
    size_t size;
};

// ============== TensorRT 引擎 ==============
class TensorRTEngine {
public:
    TensorRTEngine(const std::string& engine_path, const Config& config);
    ~TensorRTEngine();
    
    bool load();
    bool infer(const uint8_t* input_data, int width, int height,
              std::vector<BoundingBox>& boxes, float& time_ms);
    std::string getInfo() const { return info_; }
    
private:
    Config config_;
    std::string engine_path_;
    std::string info_;
    
    void* d_input_ = nullptr;
    void* d_output_ = nullptr;
    size_t input_size_ = 0;
    size_t output_size_ = 0;
    cudaStream_t stream_ = nullptr;

    IRuntime* runtime_ = nullptr;
    ICudaEngine* engine_ = nullptr;
    IExecutionContext* context_ = nullptr;
    int input_binding_idx_ = -1;
    int output_binding_idx_ = -1;
    std::vector<float> h_output_;
};

TensorRTEngine::TensorRTEngine(const std::string& engine_path, const Config& config)
    : engine_path_(engine_path), config_(config) {
    
    input_size_ = config_.input_w * config_.input_h * config_.input_c * sizeof(float);
    output_size_ = config_.num_anchors * (4 + 1 + config_.num_classes) * sizeof(float);
    
    info_ = "TensorRT: " + engine_path + 
            " | Input: " + std::to_string(config_.input_w) + "x" + std::to_string(config_.input_h);
}

TensorRTEngine::~TensorRTEngine() {
    if (d_input_) cudaFree(d_input_);
    if (d_output_) cudaFree(d_output_);
    if (stream_) cudaStreamDestroy(stream_);

    trtDestroy(context_);
    trtDestroy(engine_);
    trtDestroy(runtime_);
}

bool TensorRTEngine::load() {
    std::cout << "[TensorRT] Loading: " << engine_path_ << std::endl;
    
    std::ifstream file(engine_path_, std::ios::binary);
    if (!file) {
        std::cerr << "[TensorRT] Engine not found: " << engine_path_ << std::endl;
        std::cerr << "[TensorRT] Run: trtexec --onnx=best.onnx --saveEngine=best.trt --fp16" << std::endl;
        return false;
    }

    file.seekg(0, file.end);
    size_t size = static_cast<size_t>(file.tellg());
    file.seekg(0, file.beg);
    std::vector<char> trtModelStream(size);
    file.read(trtModelStream.data(), size);
    file.close();

    runtime_ = createInferRuntime(gTrtLogger);
    if (!runtime_) {
        std::cerr << "[TensorRT] createInferRuntime failed" << std::endl;
        return false;
    }

    engine_ = runtime_->deserializeCudaEngine(trtModelStream.data(), size);
    if (!engine_) {
        std::cerr << "[TensorRT] deserializeCudaEngine failed" << std::endl;
        return false;
    }

    context_ = engine_->createExecutionContext();
    if (!context_) {
        std::cerr << "[TensorRT] createExecutionContext failed" << std::endl;
        return false;
    }

    input_binding_idx_ = -1;
    output_binding_idx_ = -1;
    for (int i = 0; i < engine_->getNbBindings(); ++i) {
        if (engine_->bindingIsInput(i)) {
            if (input_binding_idx_ < 0) input_binding_idx_ = i;
        } else {
            if (output_binding_idx_ < 0) output_binding_idx_ = i;
        }
    }
    if (input_binding_idx_ < 0 || output_binding_idx_ < 0) {
        std::cerr << "[TensorRT] Failed to locate input/output bindings" << std::endl;
        return false;
    }
    
    // 分配 GPU 内存
    CUDA_CHECK(cudaMalloc(&d_input_, input_size_));
    CUDA_CHECK(cudaMalloc(&d_output_, output_size_));
    CUDA_CHECK(cudaStreamCreate(&stream_));

    h_output_.resize(output_size_ / sizeof(float));
    
    std::cout << "[TensorRT] Engine loaded, Input: " << input_size_ 
              << " bytes, Output: " << output_size_ << " bytes" << std::endl;
    
    return true;
}

bool TensorRTEngine::infer(const uint8_t* input_data, int width, int height,
                           std::vector<BoundingBox>& boxes, float& time_ms) {
    auto start = std::chrono::high_resolution_clock::now();
    
    if (!context_ || !engine_ || !d_input_ || !d_output_ || !stream_) {
        std::cerr << "[TensorRT] Engine not initialized" << std::endl;
        return false;
    }

    // 1) 预处理: HWC(uint8) -> NCHW(float), 并在必要时 resize 到模型输入尺寸
    cv::Mat src(height, width, CV_8UC3, const_cast<uint8_t*>(input_data));
    cv::Mat resized;
    if (width != config_.input_w || height != config_.input_h) {
        cv::resize(src, resized, cv::Size(config_.input_w, config_.input_h), 0, 0, cv::INTER_LINEAR);
    } else {
        resized = src;
    }

    const int H = config_.input_h;
    const int W = config_.input_w;
    std::vector<float> input_float(static_cast<size_t>(3) * H * W);
    for (int y = 0; y < H; ++y) {
        const cv::Vec3b* row = resized.ptr<cv::Vec3b>(y);
        for (int x = 0; x < W; ++x) {
            const cv::Vec3b& bgr = row[x];
            const size_t idx = static_cast<size_t>(y) * W + x;
            input_float[idx] = static_cast<float>(bgr[2]) / 255.0f;              // R
            input_float[static_cast<size_t>(H) * W + idx] = static_cast<float>(bgr[1]) / 255.0f;  // G
            input_float[static_cast<size_t>(2) * H * W + idx] = static_cast<float>(bgr[0]) / 255.0f;  // B
        }
    }

    // 2) H2D (异步)
    CUDA_CHECK(cudaMemcpyAsync(d_input_, input_float.data(), input_size_, cudaMemcpyHostToDevice, stream_));

    // 3) TensorRT 推理 (enqueueV2)
    std::vector<void*> bindings(static_cast<size_t>(engine_->getNbBindings()), nullptr);
    bindings[static_cast<size_t>(input_binding_idx_)] = d_input_;
    bindings[static_cast<size_t>(output_binding_idx_)] = d_output_;
    if (!context_->enqueueV2(bindings.data(), stream_, nullptr)) {
        std::cerr << "[TensorRT] enqueueV2 failed" << std::endl;
        return false;
    }

    // 4) D2H (异步)
    CUDA_CHECK(cudaMemcpyAsync(h_output_.data(), d_output_, output_size_, cudaMemcpyDeviceToHost, stream_));

    // 5) 同步等待本次推理完成
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    // 6) 后处理 (NMS) - 仍保持简化版
    // 实际应解析 h_output_ 得到 boxes
    
    // 模拟结果
    BoundingBox box;
    box.set_x1(width * 0.3f);
    box.set_y1(height * 0.3f);
    box.set_x2(width * 0.7f);
    box.set_y2(height * 0.7f);
    box.set_confidence(0.85f);
    box.set_class_id(0);
    box.set_class_name("car");
    boxes.push_back(box);
    
    auto end = std::chrono::high_resolution_clock::now();
    time_ms = std::chrono::duration<float, std::milli>(end - start).count();
    
    return true;
}

// ============== YOLO gRPC 服务 ==============
class YoloServiceImpl final : public YoloInference::Service {
public:
    YoloServiceImpl(const std::string& engine_path) 
        : config_(), engine_(engine_path, config_) {
        
        if (!engine_.load()) {
            loaded_ = false;
            std::cerr << "[Server] Failed to load engine!" << std::endl;
        } else {
            loaded_ = true;
        }
    }
    
    // 方案 A: Shared Memory 方式
    Status Infer(ServerContext* context, const InferRequest* request,
                 DetectionResult* response) override {
        
        if (!loaded_) {
            return grpc::Status(grpc::UNAVAILABLE, "Engine not loaded");
        }
        
        const std::string& buffer_name = request->buffer_name();
        int width = request->width();
        int height = request->height();
        
        std::cout << "[Server] Infer request: buffer=" << buffer_name 
                  << " size=" << width << "x" << height << std::endl;
        
        // 从 Shared Memory 读取图像
        SharedMemoryReader reader;
        if (!reader.open(buffer_name)) {
            return grpc::Status(grpc::INVALID_ARGUMENT, 
                "Failed to open shared memory: " + buffer_name);
        }
        
        int img_w, img_h;
        std::vector<uint8_t> img_data;
        if (!reader.read_image(img_w, img_h, img_data)) {
            reader.close();
            return grpc::Status(grpc::INVALID_ARGUMENT, "Failed to read image from shared memory");
        }
        
        // 验证尺寸
        if (img_w != width || img_h != height) {
            std::cerr << "[Server] Size mismatch: expected " << width << "x" << height
                      << ", got " << img_w << "x" << img_h << std::endl;
        }
        
        // 推理
        std::vector<BoundingBox> boxes;
        float inference_time_ms = 0.0f;
        
        bool success = engine_.infer(img_data.data(), width, height, boxes, inference_time_ms);
        reader.close();
        
        if (!success) {
            return grpc::Status(grpc::INTERNAL, "Inference failed");
        }
        
        // 填充响应
        response->set_inference_time_ms(inference_time_ms);
        for (const auto& box : boxes) {
            BoundingBox* b = response->add_boxes();
            *b = box;
        }
        
        return Status::OK;
    }
    
    // 方案 B: 直接传数据 (兼容)
    Status InferRaw(ServerContext* context, const ImageData* request,
                    DetectionResult* response) override {
        
        if (!loaded_) {
            return grpc::Status(grpc::UNAVAILABLE, "Engine not loaded");
        }
        
        int width = request->width();
        int height = request->height();
        const std::string& data = request->data();
        
        if (data.size() != static_cast<size_t>(width * height * 3)) {
            return grpc::Status(grpc::INVALID_ARGUMENT, "Image data size mismatch");
        }
        
        const uint8_t* input_data = reinterpret_cast<const uint8_t*>(data.data());
        std::vector<BoundingBox> boxes;
        float inference_time_ms = 0.0f;
        
        bool success = engine_.infer(input_data, width, height, boxes, inference_time_ms);
        
        if (!success) {
            return grpc::Status(grpc::INTERNAL, "Inference failed");
        }
        
        response->set_inference_time_ms(inference_time_ms);
        for (const auto& box : boxes) {
            BoundingBox* b = response->add_boxes();
            *b = box;
        }
        
        return Status::OK;
    }
    
    Status Health(ServerContext* context, const HealthRequest* request,
                  HealthResponse* response) override {
        response->set_healthy(loaded_);
        response->set_message(loaded_ ? "YOLO service running" : "Engine not loaded");
        response->set_engine_info(engine_.getInfo());
        return Status::OK;
    }
    
private:
    Config config_;
    TensorRTEngine engine_;
    bool loaded_ = false;
};

// ============== 主函数 ==============
void RunServer(const std::string& engine_path, const std::string& port) {
    std::string server_address("0.0.0.0:" + port);
    YoloServiceImpl service(engine_path);
    
    ServerBuilder builder;
    builder.SetMaxMessageSize(INT_MAX);
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);
    
    std::unique_ptr<Server> server(builder.BuildAndStart());
    
    if (!server) {
        std::cerr << "[Server] Failed to start" << std::endl;
        return;
    }
    
    std::cout << "[Server] ==============================" << std::endl;
    std::cout << "[Server] YOLO Inference Server (gRPC + Shm)" << std::endl;
    std::cout << "[Server] Listening on " << server_address << std::endl;
    std::cout << "[Server] Engine: " << engine_path << std::endl;
    std::cout << "[Server] ==============================" << std::endl;
    
    server->Wait();
}

int main(int argc, char** argv) {
    std::string engine_path = "best.trt";
    std::string port = "50051";
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--engine" && i + 1 < argc) {
            engine_path = argv[++i];
        } else if (arg == "--port" && i + 1 < argc) {
            port = argv[++i];
        } else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] 
                      << " [--engine <path>] [--port <port>]" << std::endl;
            return 0;
        }
    }
    
    RunServer(engine_path, port);
    
    return 0;
}
