# app/inference.py
import time
import numpy as np
import cv2
import threading
from typing import Tuple, List, Optional

try:
    import onnxruntime as ort
except Exception:
    ort = None

# Option A: ONNX inference wrapper
class ONNXModel:
    def __init__(self, onnx_path: str, providers: Optional[List[str]] = None):
        if ort is None:
            raise RuntimeError("onnxruntime not installed")
        self.onnx_path = onnx_path
        providers = providers or (["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in ort.get_available_providers() else ["CPUExecutionProvider"])
        self.sess = ort.InferenceSession(onnx_path, providers=providers)
        # infer input name
        self.input_name = self.sess.get_inputs()[0].name

    def preprocess(self, frame: np.ndarray, target_size=(640, 640)):
        h, w = frame.shape[:2]
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Resize with letterbox to keep aspect ratio
        r = min(target_size[0] / h, target_size[1] / w)
        nh, nw = int(round(h * r)), int(round(w * r))
        img_resized = cv2.resize(img, (nw, nh))
        canvas = np.full((target_size[0], target_size[1], 3), 114, dtype=np.uint8)
        dh, dw = (target_size[0] - nh) // 2, (target_size[1] - nw) // 2
        canvas[dh:dh+nh, dw:dw+nw, :] = img_resized
        img = canvas.astype(np.float32) / 255.0
        # transpose to NCHW
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, 0).astype(np.float32)
        return img, r, dw, dh

    def postprocess(self, outputs, conf_thres=0.25, iou_thres=0.45):
        # This depends on ONNX model output format (assuming YOLOv8 export format)
        # For simplicity, we assume the exported model returns detections in [N, 6] format: [x1, y1, x2, y2, score, class]
        # If your ONNX model outputs differently, adapt this.
        out = outputs[0]
        if out.size == 0:
            return []
        # Filter by conf
        dets = out[out[:, 4] >= conf_thres]
        return dets

    def predict(self, frame: np.ndarray):
        img, r, dw, dh = self.preprocess(frame)
        ort_inputs = {self.input_name: img}
        outputs = self.sess.run(None, ort_inputs)
        dets = self.postprocess(outputs)
        # Map back to original frame coords
        results = []
        for d in dets:
            x1, y1, x2, y2, score, cls = d[:6]
            # Remove padding & scale
            x1 = (x1 - dw) / r
            x2 = (x2 - dw) / r
            y1 = (y1 - dh) / r
            y2 = (y2 - dh) / r
            results.append([int(x1), int(y1), int(x2), int(y2), float(score), int(cls)])
        return results

# Option B: Ultralytics YOLO model (PyTorch)
class YOLOv8Model:
    def __init__(self, model_path: str, device: str = "cuda"):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.device = device

    def predict(self, frame: np.ndarray, conf=0.25, imgsz=640):
        # ultralytics model accept numpy BGR frames
        results = self.model.predict(source=[frame], imgsz=imgsz, conf=conf, verbose=False)
        # results is a list; take first
        r = results[0]
        dets = []
        if r.boxes is None or len(r.boxes) == 0:
            return dets
        for box, score, cls in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy()):
            x1, y1, x2, y2 = map(int, box)
            dets.append([x1, y1, x2, y2, float(score), int(cls)])
        return dets
