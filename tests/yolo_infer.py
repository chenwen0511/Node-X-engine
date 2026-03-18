"""
YOLO 推理脚本 - 适配 Pi5 环境
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

# 配置
MODEL_PATH = Path(__file__).parent / "weights" / "best.onnx"
IMAGE_PATH = Path(__file__).parent / "811W2.jpg"
CONFIDENCE_THRESHOLD = 0.5  # 调整阈值
IMG_SIZE = 640

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """图像预处理：直接缩放（简化版）"""
    # 直接 resize 到 640x640
    img_resized = cv2.resize(img, new_shape, interpolation=cv2.INTER_LINEAR)
    return img_resized, 1.0, (0, 0)

def load_model(model_path):
    """加载 ONNX 模型"""
    print(f"Loading model: {model_path}")
    
    # 使用 ONNX Runtime 如果可用
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(
            str(model_path),
            providers=['CPUExecutionProvider']
        )
        return "onnxruntime", sess
    except ImportError:
        print("ONNX Runtime not available, trying OpenCV DNN...")
    
    # 回退到 OpenCV DNN
    net = cv2.dnn.readNetFromONNX(str(model_path))
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    
    return "opencv", net

def infer_onnx_runtime(sess, img):
    """ONNX Runtime 推理"""
    import onnxruntime as ort
    
    input_name = sess.get_inputs()[0].name
    output = sess.run(None, {input_name: img})[0]
    return output

def infer_opencv(net, img):
    """OpenCV DNN 推理"""
    # img 已经是 (1, 3, 640, 640) float32 格式，值在 0-1 之间
    # 需要转为 (1, 3, 640, 640) float32 但值在 0-255 或保持 0-1 取决于模型
    # 这里假设模型期望 0-1
    net.setInput(img)
    output = net.forward()
    return output

def xywh2xyxy(x):
    """转换边界框格式"""
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # x1
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # y1
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # x2
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # y2
    return y

def nms(boxes, scores, iou_threshold=0.45):
    """非极大值抑制"""
    if len(boxes) == 0:
        return []
    
    boxes = np.array(boxes)
    scores = np.array(scores)
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        
        if len(order) == 1:
            break
            
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        
        # 保留 IOU 小于阈值的
        order = order[1:][iou <= iou_threshold]
    
    return keep

def post_process(output, conf_threshold=0.25, iou_threshold=0.45):
    """后处理 - 适配 YOLO 输出格式"""
    # output shape: (batch, 7, 8400)
    pred = np.squeeze(output[0])  # (7, 8400)
    
    # 转置为 (8400, 7)
    pred = pred.T
    
    # 这个模型只有1个类别，用 obj_conf 就够了
    obj_conf = pred[:, 4]
    confs = obj_conf  # 直接用 objectness
    
    # 过滤低置信度
    mask = obj_conf > conf_threshold
    pred = pred[mask]
    confs = confs[mask]
    
    if len(pred) == 0:
        return np.array([]), np.array([]), np.array([])
    
    # 转换框坐标 xywh -> xyxy
    boxes = np.column_stack([
        pred[:, 0] - pred[:, 2]/2,
        pred[:, 1] - pred[:, 3]/2,
        pred[:, 0] + pred[:, 2]/2,
        pred[:, 1] + pred[:, 3]/2
    ])
    
    # NMS
    keep = nms(boxes, confs, iou_threshold)
    
    # 类别都设为 0 (单类别)
    class_ids = np.zeros(len(keep), dtype=np.int64)
    
    return boxes[keep], class_ids, confs[keep]

def draw_results(img, boxes, class_ids, confs, class_names=None):
    """绘制结果"""
    if class_names is None:
        class_names = [f"Class {i}" for i in range(80)]
    
    colors = np.random.randint(0, 255, (80, 3), dtype=np.uint8)
    
    for box, cls_id, conf in zip(boxes, class_ids, confs):
        x1, y1, x2, y2 = map(int, box)
        color = tuple(map(int, colors[cls_id]))
        
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        label = f"{class_names[cls_id]}: {conf:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - h - 4), (x1 + w, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return img

def main():
    # 加载图像
    print(f"Loading image: {IMAGE_PATH}")
    img0 = cv2.imread(str(IMAGE_PATH))
    if img0 is None:
        print(f"Failed to load image: {IMAGE_PATH}")
        return
    
    print(f"Image shape: {img0.shape}")
    
    # 预处理
    img, ratio, pad = letterbox(img0, (640, 640))
    img = img.transpose(2, 0, 1)[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    
    print(f"Preprocessed shape: {img.shape}")
    
    # 加载模型
    backend, model = load_model(MODEL_PATH)
    print(f"Using backend: {backend}")
    
    # 推理
    import time
    start = time.time()
    
    if backend == "onnxruntime":
        output = infer_onnx_runtime(model, img)
    else:
        output = infer_opencv(model, img)
    
    elapsed = time.time() - start
    print(f"Inference time: {elapsed*1000:.1f} ms")
    print(f"Output shape: {output.shape}")
    
    # 后处理
    boxes, class_ids, confs = post_process(output, CONFIDENCE_THRESHOLD)
    print(f"Detections: {len(boxes)}")
    
    # 映射回原始图像尺寸
    if len(boxes) > 0:
        h0, w0 = img0.shape[:2]
        # 当前是 640x640，需要映射回原始尺寸
        scale_x = w0 / IMG_SIZE
        scale_y = h0 / IMG_SIZE
        boxes[:, [0, 2]] *= scale_x  # x1, x2
        boxes[:, [1, 3]] *= scale_y  # y1, y2
    
    # 绘制结果
    result_img = draw_results(img0.copy(), boxes, class_ids, confs)
    
    # 保存结果
    output_path = Path(__file__).parent / "result.jpg"
    cv2.imwrite(str(output_path), result_img)
    print(f"Result saved to: {output_path}")
    
    # 显示 (无 GUI 环境)
    # cv2.imshow("Result", result_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    print("Done!")

if __name__ == "__main__":
    main()
