# YOLO 部署到 NVIDIA Orin NX - 测试方案

## 目标

在 NVIDIA Orin NX 上部署 YOLO 推理引擎，实现 Python 调用 C++ gRPC 服务进行推理。

---

## 方案概述

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
                                   FFmpeg / GStreamer
                                          │
                                   ┌──────▼──────┐
                                   │  Orin NX    │
                                   │   (GPU)     │
                                   └─────────────┘
```

---

## 步骤 1: ONNX 权重文件编译

### YOLOv8 模型导出

```bash
# 在有 PyTorch 的机器上操作 (可以是本地 Pi5 或其他机器)
pip install ultralytics

# 导出为 ONNX (固定输入尺寸)
yolo export model=weights/best.pt format=onnx imgsz=640
```

### 使用 TensorRT 进行模型优化

```bash
# 在 Orin NX 上操作
trtexec --onnx=best.onnx \
        --saveEngine=best.trt \
        --fp16  # 半精度加速
```

### YOLOv8 输出格式

YOLOv8 输出一个 tensor，shape 为 `(1, num_predictions, 5 + num_classes)`

- 前 4 个值: `x, y, w, h` (中心点 + 宽高)
- 第 5 个值: `objectness` 置信度
- 后面的值: 各类别置信度

本项目模型为 **单类别**，所以是 `(1, 8400, 6)`

### 注意事项

| 问题 | 说明 |
|------|------|
| TensorRT 版本 | 需要与 CUDA 版本匹配 (CUDA 12.x + TensorRT 8.x) |
| 动态尺寸 | 建议固定输入尺寸 (640x640)，避免动态shape问题 |
| 算子支持 | 检查 ONNX 算子是否被 TensorRT 支持 |
| 模型输出 | 确认输出格式 (YOLO 的输出层数量) |

### 预期产出

- `best.trt` - TensorRT 优化后的引擎文件

---

## 步骤 2: C++ 推理服务

### 功能模块

1. **模型加载**
   - 加载 TensorRT 引擎文件
   - 创建执行上下文 (Context)

2. **内存管理**
   ```cpp
   // 输入: Host -> Device
   cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice);
   
   // 推理
   context->executeV2(bindings);
   
   // 输出: Device -> Host
   cudaMemcpy(h_output, d_output, output_size, cudaMemcpyDeviceToHost);
   ```

3. **gRPC 服务**
   - 定义 Protobuf 接口
   - 接收图像数据 → 推理 → 返回检测结果

### Protobuf 定义 (草稿)

```protobuf
syntax = "proto3";

service YoloInference {
  rpc Infer(ImageData) returns (DetectionResult);
}

message ImageData {
  int32 width = 1;
  int32 height = 2;
  bytes data = 3;  // RGB 原始数据
}

message DetectionResult {
  repeated BoundingBox boxes = 1;
}

message BoundingBox {
  float x1 = 1;
  float y1 = 2;
  float x2 = 3;
  float y2 = 4;
  float confidence = 5;
  int32 class_id = 6;
}
```

### 依赖库

- TensorRT
- CUDA
- gRPC
- OpenCV (图像预处理)

---

## 步骤 3: Python 调用端

### 功能

1. 图像预处理 (与 C++ 端对齐)
2. gRPC 客户端
3. 结果解析与后处理 (NMS)
4. **双输入模式**: 本地图片 / RTMP 拉流

### 输入模式

```python
class InputSource(ABC):
    @abstractmethod
    def read(self) -> Optional[np.ndarray]:
        """返回 RGB 图像，None 表示结束"""
        pass

class ImageFileSource(InputSource):
    """本地图片测试"""
    def __init__(self, path: str):
        self.cap = cv2.VideoCapture(path)

class RTMPSource(InputSource):
    """RTMP 拉流"""
    def __init__(self, rtmp_url: str):
        self.cap = cv2.VideoCapture(rtmp_url)
```

### 示例代码结构

```python
import grpc
import yolo_pb2
import yolo_pb2_grpc
import cv2

# 选择输入源
# source = ImageFileSource("test.jpg")           # 本地图片
source = RTMPSource("rtmp://192.168.1.100/live")  # RTMP 流

def infer(image):
    # 预处理: 640x640 BGR->RGB
    img = cv2.resize(image, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    data = img.tobytes()
    
    # gRPC 调用
    response = stub.Infer(yolo_pb2.ImageData(
        width=640, height=640, data=data
    ))
    
    # 后处理 (NMS, 坐标映射)
    return postprocess(response.boxes)
```

---

## 待确认问题

### 1. 模型相关
- [x] YOLO 版本: **YOLOv8**
- [x] 类别数量: **1 个类别**
- [x] 原始模型: 使用现有 best.onnx

### 2. 部署环境
- [x] Orin NX CUDA: **12.6**
- [x] TensorRT: **在 Orin NX 上运行**
- [x] 当前环境: **Pi5 (仅用于开发测试)**

### 3. 性能需求
- [x] 目标 FPS: **> 24**
- [ ] 批处理: 暂不需要

---

## 备选方案

### 简化版: TensorRT Python API

如果 gRPC 不是强需求，可以直接用 TensorRT 的 Python 接口：

```python
import tensorrt as trt

# 加载引擎
with open("best.trt", "rb") as f:
    engine = trt.Runtime(logger).deserialize_cuda_engine(f)
    
# 推理
context = engine.create_execution_context()
# ... 直接调用
```

**优点**: 简单，延迟更低
**缺点**: 需要在同一个进程/机器上

---

## 在 Orin NX 上部署

### 1. 环境准备

```bash
# 安装 Python 依赖
pip install -r src/python/requirements.txt
```

### 2. 编译 TensorRT 引擎

```bash
trtexec --onnx=weights/best.onnx \
        --saveEngine=weights/best.trt \
        --fp16 \
        --workspace=2048
```

### 3. 编译 C++ 服务

```bash
cd src/cpp
mkdir build && cd build
cmake .. -DTENSORRT_ROOT=/usr/local/TensorRT
make -j4
```

### 4. 启动服务

```bash
./yolo_server --engine ../weights/best.trt --port 50051
```

### 5. 测试

```bash
# 图片
python src/python/client.py --server localhost:50051 --input 811W2.jpg

# RTMP 流
python src/python/client.py --server localhost:50051 --input rtmp://192.168.1.100/live
```

### 性能调优

| 参数 | 推荐值 |
|------|--------|
| --fp16 | 启用 |
| --workspace | 2048MB |
| batch | 1 (实时) |

---

## 项目文件结构

```
Node-X-engine/
├── README.md
├── yolo_infer.py              # 本地测试脚本 (Pi5 / CPU)
├── weights/
│   ├── best.onnx             # ONNX 权重
│   ├── best.pt               # PyTorch 权重
│   └── best.trt              # TensorRT 引擎 (Orin NX 上生成)
├── src/
│   ├── proto/
│   │   └── yolo.proto        # gRPC 定义
│   ├── cpp/
│   │   ├── server.cpp        # C++ gRPC 服务
│   │   ├── engine.cpp        # TensorRT 推理核心
│   │   └── CMakeLists.txt
│   └── python/
│       ├── client.py         # Python 客户端
│       ├── source.py         # 输入源 (图片/RTMP)
│       └── requirements.txt
└── scripts/
    └── build.sh              # 编译脚本
```

---

## 硬件要求检查表

| 组件 | 最低要求 | 推荐 |
|------|----------|------|
| Orin NX | 8GB 内存 | 16GB |
| 存储 | 10GB+ (TensorRT + 模型) | NVMe SSD |
| CUDA | 12.0+ | 12.2 |
| TensorRT | 8.5+ | 8.6 |
| FFmpeg | 4.0+ | 用于 RTMP 拉流 |

---

## 待办确认

- [ ] Orin NX CUDA 版本？
- [ ] TensorRT 已安装？
- [ ] 目标 FPS？
