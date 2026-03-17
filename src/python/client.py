"""
YOLO Python 客户端 - 方案 A: gRPC + Shared Memory
支持两种输入模式: 本地图片 / RTMP 流
"""

import argparse
import time
import os
import struct
import numpy as np
import cv2
import grpc
import mmap
from abc import ABC, abstractmethod
from typing import Optional, List, Tuple

# 生成的 gRPC 代码
import yolo_pb2
import yolo_pb2_grpc


# ============== Shared Memory 管理 ==============
class SharedMemoryManager:
    """Shared Memory 管理器"""
    
    def __init__(self, buffer_name: str = "yolo_image_buffer", size: int = 640 * 640 * 3 + 1024):
        self.buffer_name = buffer_name
        self.size = size
        self.shm_fd = None
        self.shm = None
        self.created = False
    
    def create(self):
        """创建共享内存"""
        # 创建共享内存文件
        self.shm_fd = os.open(f"/dev/shm/{self.buffer_name}", 
                              os.O_CREAT | os.O_RDWR, 0o666)
        os.ftruncate(self.shm_fd, self.size)
        
        # 内存映射
        self.shm = mmap.mmap(self.shm_fd, self.size, mmap.ACCESS_WRITE)
        self.created = True
        print(f"[Shm] Created: /dev/shm/{self.buffer_name} ({self.size} bytes)")
    
    def open(self):
        """打开已存在的共享内存"""
        self.shm_fd = os.open(f"/dev/shm/{self.buffer_name}", os.O_RDWR)
        self.shm = mmap.mmap(self.shm_fd, self.size, mmap.ACCESS_WRITE)
        print(f"[Shm] Opened: /dev/shm/{self.buffer_name}")
    
    def write_image(self, image: np.ndarray, width: int, height: int):
        """写入图像数据
        
        格式:
        - 前 16 字节: [width(4), height(4), channels(4), size(4)]
        - 后续: 图像数据 (RGB)
        """
        if self.shm is None:
            raise RuntimeError("Shared memory not initialized")
        
        # 转换为连续数组 (C order)
        img_flat = np.ascontiguousarray(image).flatten()
        data_size = img_flat.nbytes
        
        # 检查空间
        total_size = 16 + data_size
        if total_size > self.size:
            raise RuntimeError(f"Image too large: {total_size} > {self.size}")
        
        # 写入头信息
        header = struct.pack('IIII', width, height, image.shape[0], data_size)
        self.shm.seek(0)
        self.shm.write(header)
        
        # 写入图像数据
        self.shm.write(img_flat.tobytes())
        self.shm.flush()
        
        return data_size
    
    def close(self):
        """关闭共享内存"""
        if self.shm:
            self.shm.close()
        if self.shm_fd:
            os.close(self.shm_fd)
    
    def unlink(self):
        """删除共享内存"""
        try:
            os.unlink(f"/dev/shm/{self.buffer_name}")
            print(f"[Shm] Deleted: /dev/shm/{self.buffer_name}")
        except FileNotFoundError:
            pass


class InputSource(ABC):
    """输入源抽象类"""
    
    @abstractmethod
    def read(self) -> Optional[np.ndarray]:
        """读取下一帧，返回 RGB 格式的图像, None 表示结束"""
        pass
    
    @abstractmethod
    def release(self):
        """释放资源"""
        pass


class ImageFileSource(InputSource):
    """本地图片/视频文件输入"""
    
    def __init__(self, path: str):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open: {path}")
    
    def read(self) -> Optional[np.ndarray]:
        ret, frame = self.cap.read()
        if not ret:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    def release(self):
        if self.cap:
            self.cap.release()


class RTMPSource(InputSource):
    """RTMP 流输入"""
    
    def __init__(self, rtmp_url: str, retry=3):
        self.rtmp_url = rtmp_url
        self.retry = retry
        self._connect()
    
    def _connect(self):
        self.cap = cv2.VideoCapture(self.rtmp_url)
        if not self.cap.isOpened():
            raise ConnectionError(f"Cannot connect to RTMP: {self.rtmp_url}")
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    def read(self) -> Optional[np.ndarray]:
        ret, frame = self.cap.read()
        if not ret:
            if self.retry > 0:
                self.retry -= 1
                print(f"[RTMP] Reconnecting... ({self.retry} left)")
                self._connect()
                return self.read()
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    def release(self):
        if self.cap:
            self.cap.release()


class YOLOClient:
    """YOLO gRPC 客户端 (方案 A: gRPC 传指令 + Shared Memory 传数据)"""
    
    def __init__(self, address: str = "localhost:50051", 
                 buffer_name: str = "yolo_image_buffer",
                 buffer_size: int = 640 * 640 * 3 * 2):
        self.address = address
        self.buffer_name = buffer_name
        
        # 创建共享内存
        self.shm_manager = SharedMemoryManager(buffer_name, buffer_size)
        self.shm_manager.create()
        
        # gRPC 连接
        self.channel = grpc.insecure_channel(address)
        self.stub = yolo_pb2_grpc.YoloInferenceStub(self.channel)
        
        # 健康检查
        self._health_check()
    
    def _health_check(self):
        try:
            response = self.stub.Health(yolo_pb2.HealthRequest())
            print(f"[Client] Service healthy: {response.message}")
            print(f"[Client] Engine: {response.engine_info}")
        except grpc.RpcError as e:
            print(f"[Client] Health check failed: {e.code()}: {e.details()}")
            raise
    
    def infer(self, image: np.ndarray, 
              target_size: Tuple[int, int] = (640, 640)) -> Tuple[List, float]:
        """
        推理 - 方案 A: Shared Memory 传输
        
        Args:
            image: RGB 格式图像
            target_size: 目标尺寸 (width, height)
            
        Returns:
            boxes: 检测框列表
            inference_time_ms: 推理耗时
        """
        # 1. 预处理: Resize
        h, w = image.shape[:2]
        target_w, target_h = target_size
        
        resized = cv2.resize(image, (target_w, target_h))
        
        # 2. 写入共享内存 (Zero-Copy)
        self.shm_manager.write_image(resized, target_w, target_h)
        
        # 3. gRPC 发送指令 (只传 buffer 名字)
        request = yolo_pb2.InferRequest(
            buffer_name=self.buffer_name,
            width=target_w,
            height=target_h,
            format="rgb"
        )
        
        response = self.stub.Infer(request)
        
        # 4. 解析结果
        boxes = []
        for box in response.boxes:
            boxes.append({
                'x1': int(box.x1 * w / target_w),
                'y1': int(box.y1 * h / target_h),
                'x2': int(box.x2 * w / target_w),
                'y2': int(box.y2 * h / target_h),
                'confidence': box.confidence,
                'class_id': box.class_id,
                'class_name': box.class_name if box.class_name else f"class_{box.class_id}"
            })
        
        return boxes, response.inference_time_ms
    
    def close(self):
        self.channel.close()
        self.shm_manager.close()


def draw_boxes(image: np.ndarray, boxes: List, labels: bool = True) -> np.ndarray:
    """绘制检测框"""
    img = image.copy()
    for box in boxes:
        x1, y1, x2, y2 = int(box['x1']), int(box['y1']), int(box['x2']), int(box['y2'])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if labels:
            label = f"{box['class_name']}: {box['confidence']:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), (0, 255, 0), -1)
            cv2.putText(img, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return img


def main():
    parser = argparse.ArgumentParser(description="YOLO Client (gRPC + Shared Memory)")
    parser.add_argument("--server", type=str, default="localhost:50051",
                       help="gRPC server address")
    parser.add_argument("--input", type=str, required=True,
                       help="Input: image file or RTMP URL")
    parser.add_argument("--buffer-name", type=str, default="yolo_image_buffer",
                       help="Shared memory buffer name")
    parser.add_argument("--buffer-size", type=int, default=640*640*3*2,
                       help="Shared memory buffer size")
    parser.add_argument("--output", type=str, default="output.jpg",
                       help="Output image path")
    parser.add_argument("--show", action="store_true", help="Show result")
    parser.add_argument("--target-size", type=int, nargs=2, default=[640, 640],
                       help="Target inference size")
    args = parser.parse_args()
    
    # 判断输入类型
    if args.input.startswith("rtmp://") or args.input.startswith("rtsp://"):
        source = RTMPSource(args.input)
        mode = "stream"
    else:
        source = ImageFileSource(args.input)
        mode = "image" if not args.input.endswith(('.mp4', '.avi', '.mov')) else "video"
    
    # 创建客户端
    client = YOLOClient(args.server, args.buffer_name, args.buffer_size)
    
    print(f"[Client] Mode: {mode}, Input: {args.input}")
    
    frame_count = 0
    total_time = 0.0
    
    try:
        while True:
            frame = source.read()
            if frame is None:
                break
            
            # 推理
            start = time.time()
            boxes, inf_time = client.infer(frame, tuple(args.target_size))
            elapsed = time.time() - start
            
            frame_count += 1
            total_time += elapsed
            
            fps = 1.0 / elapsed if elapsed > 0 else 0
            print(f"[{frame_count}] FPS: {fps:.1f}, Infer: {inf_time:.1f}ms, Boxes: {len(boxes)}")
            
            # 绘制结果
            result = draw_boxes(frame, boxes)
            
            if mode == "image":
                result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
                cv2.imwrite(args.output, result_bgr)
                print(f"[Client] Saved: {args.output}")
                break
            elif args.show:
                cv2.imshow("YOLO", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(1) == ord('q'):
                    break
    
    except KeyboardInterrupt:
        print("\n[Client] Interrupted")
    finally:
        source.release()
        client.close()
        
        if frame_count > 0:
            print(f"\n[Client] Average FPS: {frame_count / total_time:.2f}")


if __name__ == "__main__":
    main()
