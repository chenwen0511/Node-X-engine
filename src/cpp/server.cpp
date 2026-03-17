/**
 * YOLO TensorRT 推理服务
 * 
 * 构建:
 *   mkdir build && cd build
 *   cmake .. && make -j4
 * 
 * 运行:
 *   ./yolo_server --engine best.trt --port 50051
 */

#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <chrono>
#include <thread>

#include <grpcpp/grpcpp.h>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

#include "yolo.grpc.pb.h"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;
using yolo::YoloInference;
using yolo::ImageData;
using yolo::DetectionResult;
using yolo::BoundingBox;
using yolo::HealthRequest;
using yolo::HealthResponse;

// ============== TensorRT 引擎 ==============
class TensorRTEngine {
public:
    TensorRTEngine(const std::string& engine_path, bool use_fp16 = true) 
        : use_fp16_(use_fp16) {
        loadEngine(engine_path);
    }
    
    ~TensorRTEngine() {
        if (context_) {
            delete context_;
        }
        if (engine_) {
            delete engine_;
        }
        if (cuda_stream_) {
            cudaStreamDestroy(cuda_stream_);
        }
    }
    
    void loadEngine(const std::string& engine_path) {
        // 简化版: 假设 engine 已存在
        // 实际应使用 TensorRT API 加载 .trt 文件
        
        // TODO: 实现完整的 TensorRT 加载逻辑
        // nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
        // std::ifstream file(engine_path, std::ios::binary);
        // engine_ = runtime->deserializeCudaEngine(file);
        
        std::cout << "[TensorRT] Engine loaded from: " << engine_path << std::endl;
    }
    
    bool infer(const uint8_t* input_data, int width, int height, 
               std::vector<BoundingBox>& boxes, float& inference_time_ms) {
        auto start = std::chrono::high_resolution_clock::now();
        
        // TODO: 完整的推理逻辑
        // 1. 预处理 (Resize, Normalize)
        // 2. H2D (Host to Device)
        // 3. Inference
        // 4. D2H (Device to Host)
        // 5. Post-process (NMS)
        
        // 模拟推理耗时
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        
        auto end = std::chrono::high_resolution_clock::now();
        inference_time_ms = std::chrono::duration<float, std::milli>(end - start).count();
        
        return true;
    }
    
    std::string getEngineInfo() const {
        return "TensorRT Engine (placeholder)";
    }
    
private:
    void* engine_ = nullptr;
    void* context_ = nullptr;
    cudaStream_t cuda_stream_ = nullptr;
    bool use_fp16_ = true;
};

// ============== YOLO 推理服务实现 ==============
class YoloServiceImpl final : public YoloInference::Service {
public:
    YoloServiceImpl(const std::string& engine_path) 
        : engine_(engine_path) {}
    
    Status Infer(ServerContext* context, const ImageData* request,
                 DetectionResult* response) override {
        
        int width = request->width();
        int height = request->height();
        const std::string& data = request->data();
        
        // 图像数据验证
        if (data.size() != width * height * 3) {
            return grpc::Status(grpc::INVALID_ARGUMENT, 
                "Image data size mismatch");
        }
        
        // 执行推理
        std::vector<BoundingBox> boxes;
        float inference_time_ms = 0.0f;
        
        const uint8_t* input_data = reinterpret_cast<const uint8_t*>(data.data());
        bool success = engine_.infer(input_data, width, height, boxes, inference_time_ms);
        
        if (!success) {
            return grpc::Status(grpc::INTERNAL, "Inference failed");
        }
        
        // 填充响应
        response->set_inference_time_ms(inference_time_ms);
        
        for (const auto& box : boxes) {
            BoundingBox* b = response->add_boxes();
            b->set_x1(box.x1());
            b->set_y1(box.y1());
            b->set_x2(box.x2());
            b->set_y2(box.y2());
            b->set_confidence(box.confidence());
            b->set_class_id(box.class_id());
        }
        
        return Status::OK;
    }
    
    Status InferStream(ServerContext* context, 
                      ServerReaderWriter<DetectionResult, ImageData>* stream) override {
        // TODO: 实现流式推理
        return Status::UNIMPLEMENTED;
    }
    
    Status Health(ServerContext* context, const HealthRequest* request,
                  HealthResponse* response) override {
        response->set_healthy(true);
        response->set_message("YOLO service is running");
        response->set_engine_info(engine_.getEngineInfo());
        return Status::OK;
    }
    
private:
    TensorRTEngine engine_;
};

// ============== 主函数 ==============
void RunServer(const std::string& engine_path, const std::string& port) {
    std::string server_address("0.0.0.0:" + port);
    YoloServiceImpl service(engine_path);
    
    ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);
    
    std::unique_ptr<Server> server(builder.BuildAndStart());
    std::cout << "[Server] Listening on " << server_address << std::endl;
    std::cout << "[Server] Engine: " << engine_path << std::endl;
    
    server->Wait();
}

int main(int argc, char** argv) {
    std::string engine_path = "best.trt";
    std::string port = "50051";
    
    // 解析命令行参数
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
