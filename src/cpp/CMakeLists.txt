cmake_minimum_required(VERSION 3.18)
project(yolo_inference)

# C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA
find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -O3 --expt-relaxed-constexpr)
include_directories(${CUDA_INCLUDE_DIRS})

# TensorRT (可选)
# 使用 -DTENSORRT_ROOT=/path/to/TensorRT 指定路径
set(TENSORRT_ROOT "/usr/local/TensorRT" CACHE PATH "TensorRT root directory")
if(EXISTS ${TENSORRT_ROOT})
    message(STATUS "TensorRT root: ${TENSORRT_ROOT}")
    set(TensorRT_FOUND ON)
    include_directories(
        ${TENSORRT_ROOT}/include
        ${TENSORRT_ROOT}/include/x86_64-linux-gnu
    )
    link_directories(${TENSORRT_ROOT}/lib)
endif()

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# gRPC
find_package(Protobuf REQUIRED)
find_package(grpc REQUIRED)
include_directories(${CMAKE_CURRENT_BINARY_DIR})

# Protobuf 生成
protobuf_generate(
    TARGET yolo_proto
    PROTOS ${CMAKE_CURRENT_SOURCE_DIR}/../proto/yolo.proto
)

# 生成 gRPC 代码
grpc_generate(
    TARGET yolo_grpc
    PROTOS ${CMAKE_CURRENT_SOURCE_DIR}/../proto/yolo.proto
)

# 可执行文件
add_executable(yolo_server server.cpp)
target_link_libraries(yolo_server
    yolo_proto
    yolo_grpc
    grpc++
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    cudart
)

# TensorRT 链接
if(TensorRT_FOUND)
    message(STATUS "Linking TensorRT libraries")
    target_link_libraries(yolo_server
        nvinfer
        nvonnxparser
        nvparsers
    )
    target_compile_definitions(yolo_server USE_TENSORRT)
endif()

# 安装
install(TARGETS yolo_server DESTINATION bin)
