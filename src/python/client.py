"""
YOLO Python 客户端
支持两种输入模式: 本地图片 / RTMP 流
"""

import argparse
import time
import numpy as np
import cv2
import grpc
from abc import ABC, abstractmethod
from typing import Optional, List, Tuple

# 生成的 gRPC 代码
import yolo_pb2
import yolo_pb2_grpc


class InputSource(ABC):
    """输入源抽象类"""
    
    @abstractmethod
    def read(self) -> Optional[np.ndarray]:
        """
        读取下一帧
        返回: RGB 格式的图像, None 表示结束
        """
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
        # BGR -> RGB
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
        """连接 RTMP 流"""
        self.cap = cv2.VideoCapture(self.rtmp_url)
        if not self.cap.isOpened():
            raise ConnectionError(f"Cannot connect to RTMP: {self.rtmp_url}")
        
        # 设置缓冲区大小
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    def read(self) -> Optional[np.ndarray]:
        ret, frame = self.cap.read()
        
        if not ret:
            # 尝试重连
            if self.retry > 0:
                self.retry -= 1
                print(f"[RTMP] Connection lost, retrying... ({self.retry} left)")
                self._connect()
                return self.read()
            return None
        
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    def release(self):
        if self.cap:
            self.cap.release()


class YOLOClient:
    """YOLO gRPC 客户端"""
    
    def __init__(self, address: str = "localhost:50051"):
        self.channel = grpc.insecure_channel(address)
        self.stub = yolo_pb2_grpc.YoloInferenceStub(self.channel)
        
        # 连接测试
        self._health_check()
    
    def _health_check(self):
        """检查服务健康状态"""
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
        推理
        
        Args:
            image: RGB 格式图像
            target_size: 目标尺寸 (width, height)
            
        Returns:
            boxes: 检测框列表
            inference_time_ms: 推理耗时
        """
        # 预处理: Resize
        h, w = image.shape[:2]
        target_w, target_h = target_size
        
        resized = cv2.resize(image, (target_w, target_h))
        
        # 转换为 bytes
        data = resized.tobytes()
        
        # gRPC 调用
        request = yolo_pb2.ImageData(
            width=target_w,
            height=target_h,
            data=data,
            format="rgb"
        )
        
        response = self.stub.Infer(request)
        
        # 解析结果
        boxes = []
        for box in response.boxes:
            # 映射回原图尺寸
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


def draw_boxes(image: np.ndarray, boxes: List, labels: bool = True) -> np.ndarray:
    """在图像上绘制检测框"""
    img = image.copy()
    
    for box in boxes:
        x1, y1, x2, y2 = int(box['x1']), int(box['y1']), int(box['x2']), int(box['y2'])
        conf = box['confidence']
        
        # 绘制框
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 绘制标签
        if labels:
            label = f"{box['class_name']}: {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), (0, 255, 0), -1)
            cv2.putText(img, label, (x1, y1 - 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return img


def main():
    parser = argparse.ArgumentParser(description="YOLO Client")
    parser.add_argument("--server", type=str, default="localhost:50051",
                       help="gRPC server address")
    parser.add_argument("--input", type=str, required=True,
                       help="Input: image file or RTMP URL")
    parser.add_argument("--output", type=str, default="output.jpg",
                       help="Output image path (for single image mode)")
    parser.add_argument("--show", action="store_true",
                       help="Show result in window")
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
    
    # 连接服务端
    client = YOLOClient(args.server)
    
    # 统计
    frame_count = 0
    total_time = 0.0
    
    print(f"[Client] Mode: {mode}")
    print(f"[Client] Input: {args.input}")
    
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
            
            # FPS
            fps = 1.0 / elapsed if elapsed > 0 else 0
            
            print(f"[{frame_count}] FPS: {fps:.1f}, "
                  f"Infer: {inf_time:.1f}ms, "
                  f"Boxes: {len(boxes)}")
            
            # 绘制结果
            result = draw_boxes(frame, boxes)
            
            # 显示/保存
            if mode == "image":
                # 转为 BGR 保存
                result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
                cv2.imwrite(args.output, result_bgr)
                print(f"[Client] Result saved to: {args.output}")
                break
            elif args.show:
                cv2.imshow("YOLO", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(1) == ord('q'):
                    break
            
            # 流模式: 按目标 FPS 播放
            # target_fps = 24
            # delay = int(1000 / target_fps)
            # cv2.waitKey(delay)
    
    except KeyboardInterrupt:
        print("\n[Client] Interrupted")
    finally:
        source.release()
        client.close()
        
        if frame_count > 0:
            avg_fps = frame_count / total_time
            print(f"\n[Client] Average FPS: {avg_fps:.2f}")


if __name__ == "__main__":
    main()
