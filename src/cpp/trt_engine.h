/**
 * TensorRT 推理核心 - 完整实现
 * 
 * 使用方法:
 * 1. 确保已安装 TensorRT SDK
 * 2. 在 CMakeLists.txt 中设置 -DTENSORRT_ROOT=/path/to/TensorRT
 * 3. 取消本文件的注释并包含到构建中
 */

#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <string>
#include <cstring>
#include <algorithm>
#include <numeric>

#include <cuda_runtime.h>

// TensorRT 头文件 (在 Orin NX 上可用)
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <NvUtils.h>

// ============== Logger ==============
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TensorRT] " << msg << std::endl;
        }
    }
} gLogger;

// ============== 配置 ==============
struct YOLOConfig {
    int input_w = 640;
    int input_h = 640;
    int input_c = 3;
    int num_classes = 1;
    float conf_threshold = 0.5f;
    float nms_threshold = 0.45f;
    
    // YOLOv8 输出配置
    int num_outputs = 1;  // YOLOv8 单输出
    int num_anchors = 8400;  // 640x640 输入的 anchor 数量
};

// ============== TensorRT 引擎类 ==============
class TRTEngine {
public:
    TRTEngine(const std::string& engine_path, const YOLOConfig& config);
    ~TRTEngine();
    
    bool build(const std::string& onnx_path);  // 从 ONNX 构建
    bool load();                                 // 加载 .trt 文件
    bool infer(const float* input, float* output, int batch_size = 1);
    
    int getInputSize() const { return input_size_; }
    int getOutputSize() const { return output_size_; }
    
private:
    std::string engine_path_;
    YOLOConfig config_;
    
    nvinfer1::IRuntime* runtime_ = nullptr;
    nvinfer1::ICudaEngine* engine_ = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;
    
    void* d_input_ = nullptr;
    void* d_output_ = nullptr;
    cudaStream_t stream_ = nullptr;
    
    size_t input_size_ = 0;
    size_t output_size_ = 0;
};

TRTEngine::TRTEngine(const std::string& engine_path, const YOLOConfig& config)
    : engine_path_(engine_path), config_(config) {
    
    // 计算内存大小
    input_size_ = config_.input_w * config_.input_h * config_.input_c * sizeof(float);
    output_size_ = config_.num_anchors * (4 + 1 + config_.num_classes) * sizeof(float);
}

TRTEngine::~TRTEngine() {
    if (context_) context_->destroy();
    if (engine_) engine_->destroy();
    if (runtime_) runtime_->destroy();
    if (d_input_) cudaFree(d_input_);
    if (d_output_) cudaFree(d_output_);
    if (stream_) cudaStreamDestroy(stream_);
}

bool TRTEngine::build(const std::string& onnx_path) {
    std::cout << "[TRT] Building engine from ONNX: " << onnx_path << std::endl;
    
    // 创建 builder
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger);
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(1);
    
    // 解析 ONNX
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);
    
    std::ifstream onnx_file(onnx_path, std::ios::binary);
    if (!onnx_file) {
        std::cerr << "[TRT] Failed to open ONNX file: " << onnx_path << std::endl;
        return false;
    }
    
    onnx_file.seekg(0, std::ifstream::end);
    size_t size = onnx_file.tellg();
    onnx_file.seekg(0, std::ifstream::beg);
    
    std::vector<char> onnx_buffer(size);
    onnx_file.read(onnx_buffer.data(), size);
    onnx_file.close();
    
    bool parsed = parser->parse(onnx_buffer.data(), size);
    if (!parsed) {
        std::cerr << "[TRT] Failed to parse ONNX" << std::endl;
        return false;
    }
    
    // 配置 builder
    builder->setMaxBatchSize(1);
    builder->setFp16Mode(true);  // FP16 加速
    
    // 构建配置
    nvinfer1::IBuilderConfig* build_config = builder->createBuilderConfig();
    build_config->setFlag(nvinfer1::BuilderFlag::kFP16);
    build_config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30);  // 1GB
    
    // 构建引擎
    std::cout << "[TRT] Building engine..." << std::endl;
    engine_ = builder->buildSerializedNetwork(*network, *build_config);
    
    if (!engine_) {
        std::cerr << "[TRT] Failed to build engine" << std::endl;
        return false;
    }
    
    // 保存引擎
    std::ofstream engine_file(engine_path_, std::ios::binary);
    engine_file.write((const char*)engine_->serialize()->data(), engine_->serialize()->size());
    engine_file.close();
    
    std::cout << "[TRT] Engine saved to: " << engine_path_ << std::endl;
    
    // 清理
    parser->destroy();
    build_config->destroy();
    network->destroy();
    builder->destroy();
    
    return load();
}

bool TRTEngine::load() {
    std::cout << "[TRT] Loading engine from: " << engine_path_ << std::endl;
    
    // 加载引擎文件
    std::ifstream file(engine_path_, std::ios::binary);
    if (!file) {
        std::cerr << "[TRT] Engine file not found" << std::endl;
        return false;
    }
    
    file.seekg(0, std::ifstream::end);
    size_t size = file.tellg();
    file.seekg(0, std::ifstream::beg);
    
    std::vector<char> buffer(size);
    file.read(buffer.data(), size);
    file.close();
    
    // 反序列化
    runtime_ = nvinfer1::createInferRuntime(gLogger);
    engine_ = runtime_->deserializeCudaEngine(buffer.data(), size);
    
    if (!engine_) {
        std::cerr << "[TRT] Failed to deserialize engine" << std::endl;
        return false;
    }
    
    // 创建执行上下文
    context_ = engine_->createExecutionContext();
    
    if (!context_) {
        std::cerr << "[TRT] Failed to create execution context" << std::endl;
        return false;
    }
    
    // 分配 GPU 内存
    cudaMalloc(&d_input_, input_size_);
    cudaMalloc(&d_output_, output_size_);
    cudaStreamCreate(&stream_);
    
    std::cout << "[TRT] Engine loaded successfully" << std::endl;
    std::cout << "[TRT] Input size: " << input_size_ << " bytes" << std::endl;
    std::cout << "[TRT] Output size: " << output_size_ << " bytes" << std::endl;
    
    return true;
}

bool TRTEngine::infer(const float* input, float* output, int batch_size) {
    // H2D
    cudaMemcpy(d_input_, input, input_size_ * batch_size, cudaMemcpyHostToDevice);
    
    // 推理
    void* bindings[] = {d_input_, d_output_};
    bool success = context_->executeV2(bindings);
    
    // D2H
    cudaMemcpy(output, d_output_, output_size_ * batch_size, cudaMemcpyDeviceToHost);
    
    return success;
}

// ============== 后处理: NMS ==============
struct Box {
    float x1, y1, x2, y2;
    float obj_conf;
    float class_conf;
    int class_id;
};

float iou(const Box& a, const Box& b) {
    float x1 = std::max(a.x1, b.x1);
    float y1 = std::max(a.y1, b.y1);
    float x2 = std::min(a.x2, b.x2);
    float y2 = std::min(a.y2, b.y2);
    
    float inter = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    
    return inter / (area_a + area_b - inter + 1e-6f);
}

std::vector<Box> nms(std::vector<Box>& boxes, float iou_threshold) {
    // 按置信度排序
    std::sort(boxes.begin(), boxes.end(), 
              [](const Box& a, const Box& b) {
                  return a.obj_conf * a.class_conf > b.obj_conf * b.class_conf;
              });
    
    std::vector<Box> result;
    std::vector<bool> suppressed(boxes.size(), false);
    
    for (size_t i = 0; i < boxes.size(); i++) {
        if (suppressed[i]) continue;
        
        result.push_back(boxes[i]);
        
        for (size_t j = i + 1; j < boxes.size(); j++) {
            if (suppressed[j]) continue;
            if (boxes[i].class_id != boxes[j].class_id) continue;
            
            if (iou(boxes[i], boxes[j]) > iou_threshold) {
                suppressed[j] = true;
            }
        }
    }
    
    return result;
}

std::vector<Box> postprocessYOLO(const float* output, int num_anchors, 
                                  int num_classes, float conf_threshold,
                                  int img_w, int img_h) {
    std::vector<Box> boxes;
    
    for (int i = 0; i < num_anchors; i++) {
        const float* pred = output + i * (4 + 1 + num_classes);
        
        // 解析预测
        float cx = pred[0];
        float cy = pred[1];
        float w = pred[2];
        float h = pred[3];
        float obj_conf = pred[4];
        
        // 找最大类别置信度
        float class_conf = 0;
        int class_id = 0;
        for (int c = 0; c < num_classes; c++) {
            if (pred[5 + c] > class_conf) {
                class_conf = pred[5 + c];
                class_id = c;
            }
        }
        
        // 总置信度
        float total_conf = obj_conf * class_conf;
        if (total_conf < conf_threshold) continue;
        
        // 转换为 xyxy 格式 (假设输出是归一化的)
        Box box;
        box.x1 = (cx - w/2) * img_w;
        box.y1 = (cy - h/2) * img_h;
        box.x2 = (cx + w/2) * img_w;
        box.y2 = (cy + h/2) * img_h;
        box.obj_conf = obj_conf;
        box.class_conf = class_conf;
        box.class_id = class_id;
        
        boxes.push_back(box);
    }
    
    // NMS
    return nms(boxes, 0.45f);
}
