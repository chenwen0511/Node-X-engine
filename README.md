# YOLO 部署到 NVIDIA Orin NX - 完整指南

## 目录

1. [项目概述](#1-项目概述)
2. [部署方案选择](#2-部署方案选择)
3. [硬件要求](#3-硬件要求)
4. [软件环境](#4-软件环境)
5. [模型准备](#5-模型准备)
6. [编译部署](#6-编译部署)
7. [运行测试](#7-运行测试)
8. [性能优化](#8-性能优化)
9. [常见问题](#9-常见问题)

---

## 1. 项目概述

本项目实现在 NVIDIA Orin NX 上部署 YOLOv8 推理引擎。

### 系统架构 (方案 A: gRPC + Shared Memory)

```
┌─────────────┐     gRPC      ┌─────────────┐
│   Python    │ ─────────────▶│     C++     │
│  (Client)   │  (指令信号)    │  (Server)   │
│             │◀──────────────│             │
└─────────────┘               └──────┬──────┘
       │                                  │
       │  Shared Memory (Zero-Copy)      │
       │  ┌─────────────────────────┐     │
       └──▶│ /dev/shm/image_buffer │◀────┘
          └─────────────────────────┘
                                          │
                                   TensorRT
                                          │
                                   ┌──────▼──────┐
                                   │  Orin NX    │
                                   │   (GPU)     │
                                   └─────────────┘
```

### 功能特性

- [x] YOLOv8 目标检测
- [x] TensorRT 推理加速 (FP16)
- [x] gRPC 轻量级指令传输
- [x] Shared Memory 零拷贝数据传输
- [x] 本地图片输入
- [x] RTMP 视频流输入

---

## 2. 部署方案选择

### 方案 A: gRPC + Shared Memory (当前方案)

**推荐用于**: 跨设备通信、多进程架构、需要解耦的场景

```
数据流: Python → 共享内存 → C++ (Zero-Copy)
控制流: Python → gRPC → C++ (轻量信号)
```

**架构图**:
```
┌─────────────────────────────────────────────────────────────┐
│                        Python 进程                          │
│  ┌──────────────┐    ┌─────────────────────────────────┐  │
│  │ OpenCV/RTMP  │───▶│  写入共享内存 /dev/shm/          │  │
│  │ 获取图像     │    │  image_buffer_01                │  │
│  └──────────────┘    └─────────────────────────────────┘  │
│                              │                             │
│                              ▼                             │
│                     ┌──────────────────┐                  │
│                     │  gRPC Client     │                  │
│                     │  "infer buffer01"│ ─────────────────┼──▶
│                     └──────────────────┘                  │
└─────────────────────────────────────────────────────────────┘
                                                                gRPC (轻量信号)
                                                                ▼
┌─────────────────────────────────────────────────────────────┐
│                        C++ 进程                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  gRPC Server                                        │  │
│  │  收到信号 → 读取 /dev/shm/image_buffer_01          │  │
│  └──────────────────────────────────────────────────────┘  │
│                              │                             │
│                              ▼                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  TensorRT Inference (GPU)                          │  │
│  │  - H2D: 共享内存 → GPU 显存                        │  │
│  │  - Inference                                       │  │
│  │  - D2H: GPU 显存 → 结果                            │  │
│  └──────────────────────────────────────────────────────┘  │
│                              │                             │
│                              ▼                             │
│                     ┌──────────────────┐                  │
│                     │  gRPC 返回结果   │ ◀────────────────┘
│                     └──────────────────┘
└─────────────────────────────────────────────────────────────┘
```

**优点**:
- ✅ 完美解耦: Python 和 C++ 可独立运行
- ✅ Zero-Copy: 共享内存实现零拷贝
- ✅ 跨设备: gRPC 可通过网络调用
- ✅ 成熟方案: 自动驾驶/机器人常用

**实现步骤**:

1. **创建共享内存**:
```python
import numpy as np
import mmap

# 创建共享内存区域
shm = mmap.mmap(fileno, length, mmap.ACCESS_WRITE)
```

2. **Python 端写入图像**:
```python
# 图像写入共享内存
img = cv2.imread("test.jpg")
img_flat = img.tobytes()
shm.write(img_flat)
```

3. **gRPC 发送指令**:
```python
stub.Infer(InferRequest(buffer_name="image_buffer_01"))
```

4. **C++ 端读取**:
```cpp
// 从共享内存读取
void* shm_addr = shm_open("/image_buffer_01", O_RDWR, 0666);
// mmap 获取指针
float* data = (float*)mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
```

---

### 方案 B: pybind11 (第二阶段优化)

**适用于**: 单板部署、无跨设备需求、追求极致性能

**架构图**:
```
┌────────────────────────────────────┐
│           Python 进程              │
│                                    │
│  import tensorrt_engine           │
│                                    │
│  ┌─────────────────────────────┐   │
│  │  tensorrt_engine.Infer()   │───┼──▶ 直接调用，无 IPC
│  └─────────────────────────────┘   │
│           ▲        ▲               │
│           │        │               │
│    Numpy  │        │  C++ 结果      │
│    数组   │        │  直接返回      │
│           │        │               │
└───────────┼────────┼───────────────┘
            │        │
            │ pybind11
            │ (无拷贝)
            ▼
┌────────────────────────────────────┐
│       C++ 扩展 (tensorrt.so)       │
│                                    │
│  ┌─────────────────────────────┐  │
│  │ TensorRT 推理核心           │  │
│  │ - 模型加载                  │  │
│  │ - 显存分配                  │  │
│  │ - 推理执行                  │  │
│  └─────────────────────────────┘  │
│           ▲        ▲               │
│           │        │               │
│    Numpy │        │ GPU 结果        │
│    指针  │        │                 │
└──────────┼────────┼────────────────┘
           │        │
           │        │
           ▼        ▼
┌────────────────────────────────────┐
│        Orin NX GPU                 │
└────────────────────────────────────┘
```

**优点**:
- ✅ 性能最高: 无 IPC 开销
- ✅ 开发简单: 像用 Python 库一样
- ✅ 零拷贝: Numpy 直接转 C++ 指针

**缺点**:
- ❌ 紧耦合: 必须在同一进程
- ❌ 无法跨设备: 不能网络调用

**实现方式**:

1. **C++ 类封装**:
```cpp
#include <pybind11/pybind11.h>

class TRTEngine {
public:
    void load(const std::string& engine_path);
    std::vector<Box> infer(float* input, int h, int w);
    
private:
    // TensorRT 内部成员
};

PYBIND11_MODULE(tensorrt_engine, m) {
    pybind11::class_<TRTEngine>(m, "TRTEngine")
        .def(pybind11::init<>())
        .def("load", &TRTEngine::load)
        .def("infer", &TRTEngine::infer);
}
```

2. **编译**:
```bash
# CMakeLists.txt
add_library(tensorrt_engine SHARED engine.cpp)
target_link_libraries(tensorrt_engine 
    pybind11::pybind11
    nvinfer
    cudart
)

# 编译
cmake .. && make
```

3. **Python 使用**:
```python
import tensorrt_engine

# 初始化
engine = tensorrt_engine.TRTEngine()
engine.load("best.trt")

# 推理
import cv2
img = cv2.imread("test.jpg")
img = cv2.resize(img, (640, 640))
img = img.astype(np.float32) / 255.0

boxes = engine.infer(img)
print(boxes)
```

---

### 方案对比

| 特性 | 方案 A (gRPC+Shm) | 方案 B (pybind11) |
|------|-------------------|-------------------|
| 延迟 | ~1-2ms (Zero-Copy) | ~0.1ms (无IPC) |
| 跨设备 | ✅ 支持 | ❌ 必须在同主机 |
| 解耦度 | 高 | 低 |
| 开发难度 | 中等 | 简单 |
| 适用场景 | 自动驾驶/机器人 | 单机推理 |
| 推荐阶段 | 第一阶段 | 第二阶段优化 |

---

### 本项目规划

- **第一阶段**: 实现方案 A (gRPC + Shared Memory)
- **第二阶段**: 优化为方案 B (pybind11)，如需更高性能

---

## 3. 硬件要求

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

## 4. 软件环境

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

## 5. 模型准备

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

## 6. 编译部署

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

## 7. 运行测试

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

## 8. 性能优化

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

## 9. 常见问题

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
