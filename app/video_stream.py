# app/video_stream.py
import cv2
import threading
import time
import base64
import numpy as np
from typing import Callable, Optional

class VideoCaptureAsync:
    def __init__(self, src=0):
        self.src = src
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW | cv2.CAP_ANY)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source {src}")
        self.frame = None
        self.running = False
        self.lock = threading.Lock()

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                # try to reconnect for network streams
                time.sleep(0.5)
                continue
            with self.lock:
                self.frame = frame
            time.sleep(0.001)

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        self.thread.join(timeout=1)
        try:
            self.cap.release()
        except Exception:
            pass

def draw_detections(frame, detections, class_names=None):
    for det in detections:
        x1, y1, x2, y2, score, cls = det
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{class_names[int(cls)] if class_names else cls}: {score:.2f}"
        cv2.putText(frame, label, (x1, max(15, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    return frame

def encode_frame_to_jpeg_b64(frame, quality=80):
    ret, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ret:
        return None
    b64 = base64.b64encode(jpeg.tobytes()).decode('utf-8')
    return b64
