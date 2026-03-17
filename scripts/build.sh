#!/bin/bash
# 构建脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== YOLO Inference Build Script ==="
echo "Project: $PROJECT_ROOT"

# 检查 CUDA
if command -v nvcc &> /dev/null; then
    echo "[Build] CUDA found: $(nvcc --version | grep release)"
else
    echo "[Build] WARNING: CUDA not found"
fi

# 检查 TensorRT
if [ -d "/usr/local/TensorRT" ]; then
    echo "[Build] TensorRT found: /usr/local/TensorRT"
elif pkg-config --exists tensorrt; then
    echo "[Build] TensorRT found via pkg-config"
else
    echo "[Build] WARNING: TensorRT not found"
fi

# 生成 Protobuf 和 gRPC 代码
echo "[Build] Generating Protobuf/gRPC code..."

cd "$PROJECT_ROOT/src/proto"

# 检查是否有 protoc 和 grpc_cpp_plugin
if ! command -v protoc &> /dev/null; then
    echo "[Build] ERROR: protoc not found"
    exit 1
fi

PROTOC_OUT_DIR="$PROJECT_ROOT/src/python"
mkdir -p "$PROTOC_OUT_DIR"

# 生成 Python 代码
protoc --python_out="$PROTOC_OUT_DIR" \
       --grpc_out="$PROTOC_OUT_DIR" \
       --plugin=protoc-gen-grpc=$(which grpc_python_plugin) \
       yolo.proto

# 重命名生成的文件
if [ -f "$PROTOC_OUT_DIR/yolo_pb2.py" ]; then
    mv "$PROTOC_OUT_DIR/yolo_pb2.py" "$PROTOC_OUT_DIR/yolo_pb2.py.bak" 2>/dev/null || true
fi
if [ -f "$PROTOC_OUT_DIR/yolo_pb2_grpc.py" ]; then
    mv "$PROTOC_OUT_DIR/yolo_pb2_grpc.py" "$PROTOC_OUT_DIR/yolo_pb2_grpc.py.bak" 2>/dev/null || true
fi

echo "[Build] Python proto generated"

# 生成 C++ 代码
cd "$PROJECT_ROOT/src/cpp"

# 暂时跳过 C++ 生成 (需要完整的 grpc 环境)
echo "[Build] C++ proto generation skipped (run manually on Orin NX)"

echo "[Build] Done!"
echo ""
echo "=== Next Steps ==="
echo "1. Install Python dependencies:"
echo "   pip install -r src/python/requirements.txt"
echo ""
echo "2. Build C++ server (on Orin NX):"
echo "   cd src/cpp && mkdir build && cd build"
echo "   cmake .. && make -j4"
echo ""
echo "3. Run:"
echo "   # Start server"
echo "   ./build/yolo_server --engine best.trt --port 50051"
echo "   # Run client"
echo "   python src/python/client.py --server localhost:50051 --input test.jpg"
