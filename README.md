# YOLO 部署到 NVIDIA Orin NX - 完整指南

## 目录

1. [项目概述](#1-项目概述)
2. [硬件要求](#2-硬件要求)
3. [软件环境](#3-软件环境)
4. [模型准备](#4-模型准备)
5. [编译部署](#5-编译部署)
6. [运行测试](#6-运行测试)
7. [性能优化](#7-性能优化)
8. [常见问题](#8-常见问题)

---

## 1. 项目概述

本项目实现在 NVIDIA Orin NX 上部署 YOLOv8 推理引擎，通过 gRPC 提供推理服务。

### 系统架构

```
┌─────────────┐     gRPC      ┌─────────────┐
│   Python    │ ─────────────▶│     C++     │
│  (Client)   │               │  (Server)   │
│             │◀──────────────│             │
└─────────────┘               └──────┬──────┘
       │                                  │
       │  (两种输入模式)                   │
       ▼                                  ▼
┌─────────────┐                   ┌─────────────┐
│ 本地图片    │                   │  RTMP 流    │
│ (测试用)    │                   │ (生产环境)  │
└─────────────┘                   └──────┬──────┘
                                          │
                                   FFmpeg
                                          │
                                   ┌──────▼──────┐
                                   │  Orin NX    │
                                   │   (GPU)     │
                                   └─────────────┘
```

### 功能特性

- [x] YOLOv8 目标检测
- [x] TensorRT 推理加速 (FP16)
- [x] gRPC 通信
- [x] 本地图片输入
- [x] RTMP 视频流输入
- [x] 单帧/流式推理

---

## 2. 硬件要求

### Orin NX 配置

| 项目 | 最低要求 | 推荐配置 |
|------|----------|----------|
| 内存 | 8GB | 16GB |
| 存储 | 10GB | 32GB+ NVMe SSD |
| CUDA | 12.0 | 12.2 |
| TensorRT | 8.5 | 8.6+ |

### 主机配置 (可选)

用于开发调试，建议配置：

- GPU (可选): NVIDIA GPU (用于本地测试)
- 内存: 16GB+
- 存储: 50GB+

---

## 3. 软件环境

### Orin NX 系统要求

```bash
# 操作系统
Ubuntu 20.04 或 22.04 (JetPack)

# 必需软件包
sudo apt update
sudo apt install -y \
    build-essential \
    cmake \
    git \
    libgrpc++-dev \
    libprotobuf-dev \
    protobuf-compiler-grpc \
    libopencv-dev \
    cuda-toolkit-12-2

# Python 依赖
pip install grpcio grpcio-tools opencv-python numpy
```

### Python 环境 (开发机)

```bash
# 克隆仓库
git clone https://github.com/chenwen0511/Node-X-engine.git
cd Node-X-engine

# 安装 Python 依赖
pip install -r src/python/requirements.txt

# 安装 PyTorch (用于模型转换)
pip install torch torchvision ultralytics
```

---

## 4. 模型准备

### 4.1 导出 ONNX 模型

如果使用 PyTorch 权重，需要先导出为 ONNX：

```bash
# 方式 1: 使用 ultralytics
yolo export model=weights/best.pt format=onnx imgsz=640

# 方式 2: Python 代码
python3 << 'EOF'
from ultralytics import YOLO

model = YOLO('weights/best.pt')
model.export(format='onnx', imgsz=640)
EOF
```

### 4.2 编译 TensorRT 引擎

**方式 A: 使用 trtexec (推荐)**

```bash
# 基本编译
trtexec --onnx=weights/best.onnx \
        --saveEngine=weights/best.trt \
        --fp16

# 完整参数
trtexec --onnx=weights/best.onnx \
        --saveEngine=weights/best.trt \
        --fp16 \
        --workspace=2048 \
        --minTiming=1 \
        --avgTiming=8
```

**方式 B: 代码内构建**

修改 `server.cpp` 启用构建功能，然后运行：

```bash
./yolo_server --build weights/best.onnx --engine weights/best.trt
```

### 4.3 验证引擎

```bash
# 查看引擎信息
trtexec --loadEngine=weights/best.trt --dumpEngine

# 性能测试
trtexec --loadEngine=weights/best.trt --warmUp=1000 --duration=30
```

---

## 5. 编译部署

### 5.1 编译 C++ 服务

```bash
cd src/cpp

# 创建构建目录
mkdir build && cd build

# 配置 (根据 TensorRT 安装路径调整)
cmake .. -DTENSORRT_ROOT=/usr/local/TensorRT

# 编译 (-j4 使用 4 核并行)
make -j4

# 查看生成的可执行文件
ls -la yolo_server
```

### 5.2 目录结构

```
# 最终部署目录结构
deploy/
├── yolo_server          # 编译后的服务程序
├── weights/
│   ├── best.onnx        # ONNX 模型 (可选)
│   └── best.trt         # TensorRT 引擎 (必需)
└── config/              # 配置文件 (可选)
```

### 5.3 复制到 Orin NX

```bash
# 方式 1: SCP
scp -r deploy/ orin@192.168.1.100:~/

# 方式 2: Git
git clone https://github.com/chenwen0511/Node-X-engine.git
cd Node-X-engine
# 然后按照上述步骤编译
```

---

## 6. 运行测试

### 6.1 启动服务

```bash
# 基本启动
./yolo_server --engine weights/best.trt --port 50051

# 查看帮助
./yolo_server --help

# 输出示例:
# [Server] ==============================
# [Server] YOLO Inference Server
# [Server] listening on 0.0.0.0:50051
# [Server] Engine: weights/best.trt
# [Server] ==============================
```

### 6.2 测试客户端

```bash
# 测试本地图片
python src/python/client.py \
    --server localhost:50051 \
    --input 811W2.jpg \
    --output result.jpg

# 测试图片并显示
python src/python/client.py \
    --server localhost:50051 \
    --input 811W2.jpg \
    --show

# 测试 RTMP 流
python src/python/client.py \
    --server localhost:50051 \
    --input rtmp://192.168.1.100/live \
    --show

# 测试视频文件
python src/python/client.py \
    --server localhost:50051 \
    --input test.mp4 \
    --show
```

### 6.3 测试输出示例

```
[Client] Service healthy: YOLO service is running
[Client] Engine: TensorRT Engine: best.trt
[1] FPS: 45.2, Infer: 18.5ms, Boxes: 1
[2] FPS: 48.1, Infer: 17.2ms, Boxes: 1
[3] FPS: 46.7, Infer: 19.1ms, Boxes: 1
...
[Client] Average FPS: 46.8
```

---

## 7. 性能优化

### 7.1 TensorRT 优化参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `--fp16` | 启用 | 半精度，加速 ~2x |
| `--int8` | 可选 | INT8 量化，进一步加速 |
| `--workspace` | 2048+ | MB，编译缓存 |
| `--builderOptimizationLevel` | 5 | 最高优化 |

### 7.2 推理优化

```cpp
// 在代码中启用优化
context->setOptimizationProfileAsync(0, stream);
context->setDLACore(0);  // 使用 DLA (可选)
```

### 7.3 网络优化

- 使用 gRPC 负载均衡
- 批量推理 (需修改代码)
- 共享内存传输 (Linux only)

### 7.4 性能基准

| 配置 | FPS | 延迟 |
|------|-----|------|
| FP32 | ~25 | ~40ms |
| FP16 | ~50 | ~20ms |
| INT8 | ~70 | ~14ms |

---

## 8. 常见问题

### Q1: trtexec 找不到

```bash
# 检查 TensorRT 安装
ls /usr/local/TensorRT/bin/trtexec

# 添加到 PATH
export PATH=$PATH:/usr/local/TensorRT/bin
```

### Q2: 编译报错缺少头文件

```bash
# 安装完整 TensorRT 开发包
sudo apt install libnvinfer-dev libnvparsers-dev libnvonnxparser-dev
```

### Q3: 推理结果为空

1. 检查输入图像格式 (应为 RGB)
2. 调整置信度阈值: `--conf 0.3`
3. 检查模型是否匹配 (类别数等)

### Q4: gRPC 连接失败

```bash
# 检查端口是否开放
telnet localhost 50051

# 检查防火墙
sudo ufw allow 50051
```

### Q5: CUDA 内存不足

```bash
# 清理缓存
sudo rm -rf /tmp/*

# 减少模型批次大小
# 修改代码中的 batch_size = 1
```

### Q6: ONNX 转 TensorRT 失败

```bash
# 检查 ONNX 模型
python3 << 'EOF'
import onnx
model = onnx.load("best.onnx")
onnx.checker.check_model(model)
print("Model OK")
EOF

# 使用简化版导出
yolo export model=best.pt format=onnx imgsz=640 simplify=True
```

---

## API 参考

### gRPC 服务接口

```protobuf
service YoloInference {
  rpc Infer(ImageData) returns (DetectionResult);
  rpc InferStream(stream ImageData) returns (stream DetectionResult);
  rpc Health(HealthRequest) returns (HealthResponse);
}

message ImageData {
  int32 width = 1;
  int32 height = 2;
  bytes data = 3;    // RGB 原始数据
  string format = 4; // "rgb" 或 "bgr"
}

message DetectionResult {
  repeated BoundingBox boxes = 1;
  float inference_time_ms = 2;
}

message BoundingBox {
  float x1 = 1;
  float y1 = 2;
  float x2 = 3;
  float y2 = 4;
  float confidence = 5;
  int32 class_id = 6;
  string class_name = 7;
}
```

### Python 客户端使用

```python
from client import YOLOClient, ImageFileSource, RTMPSource

# 创建客户端
client = YOLOClient("localhost:50051")

# 加载图像
import cv2
image = cv2.imread("test.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 推理
boxes, inference_time = client.infer(image)

# 处理结果
for box in boxes:
    print(f"Box: ({box['x1']}, {box['y1']}) - ({box['x2']}, {box['y2']})")
    print(f"Confidence: {box['confidence']:.2f}")

client.close()
```

---

## 技术细节

### YOLOv8 输出格式

```
输出 shape: (1, 8400, 6)
每行: [x, y, w, h, objectness, class_0]

其中:
- x, y: 归一化中心点坐标 (0-1)
- w, h: 归一化宽高 (0-1)
- objectness: 目标置信度 (0-1)
- class_0: 类别置信度 (0-1)
```

### 坐标系说明

```
图像坐标系:
(0,0) ──────→ x
  │
  │
  ↓
  y

检测框: (x1, y1) 左上角, (x2, y2) 右下角
```

---

## 许可证

MIT License

---

## 参考资料

- [TensorRT 官方文档](https://docs.nvidia.com/deeplearning/tensorrt/)
- [YOLOv8 文档](https://docs.ultralytics.com/)
- [gRPC 官方文档](https://grpc.io/docs/)
- [NVIDIA JetPack](https://developer.nvidia.com/embedded/jetpack)
