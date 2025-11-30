# app/utils.py
import time
import cv2
import numpy as np
from collections import deque

class FPSCounter:
    """ Tracks FPS over a sliding window. """
    def __init__(self, window=30):
        self.timestamps = deque(maxlen=window)

    def update(self):
        now = time.time()
        self.timestamps.append(now)

    @property
    def fps(self):
        if len(self.timestamps) < 2:
            return 0.0
        return len(self.timestamps) / (self.timestamps[-1] - self.timestamps[0])

def warmup_model(model, img_size=640):
    """Runs a single dummy pass to warm up ONNX or Torch model."""
    dummy = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)
    try:
        model.predict(dummy)
    except Exception:
        pass

def get_color_palette(num_classes):
    """Returns unique colors for each class index."""
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(num_classes, 3), dtype="uint8")
    return colors

def safe_read_frame(cap):
    """Safe wrapper for cv2 VideoCapture().read()."""
    try:
        ret, frame = cap.read()
        if not ret:
            return None
        return frame
    except Exception:
        return None
