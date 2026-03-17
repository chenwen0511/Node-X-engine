cmake_minimum_required(VERSION 3.18)
project(yolo_inference)

# C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA
find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -O3 --expt-relaxed-constexpr)

# TensorRT (可选)
find_package(TensorRT QUIET)
if(TensorRT_FOUND)
    message(STATUS "TensorRT found: ${TensorRT_VERSION}")
    include_directories(${TensorRT_INCLUDE_DIRS})
    link_directories(${TensorRT_LIBRARY_DIRS})
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
)

if(TensorRT_FOUND)
    target_link_libraries(yolo_server ${TensorRT_LIBRARIES})
    target_compile_definitions(yolo_server USE_TENSORRT)
endif()

# 安装
install(TARGETS yolo_server DESTINATION bin)
