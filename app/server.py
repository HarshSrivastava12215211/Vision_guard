# app/server.py
import asyncio
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import numpy as np
import cv2
from typing import List

from .video_stream import VideoCaptureAsync, draw_detections, encode_frame_to_jpeg_b64
from .inference import ONNXModel, YOLOv8Model

app = FastAPI(title="VisionGuard - Real Time Inference")

# mount static client for convenience
app.mount("/client", StaticFiles(directory="app/client"), name="client")

# Config
MODEL_TYPE = "onnx"  # "onnx" or "yolo"
MODEL_PATH = "models/visionguard.onnx"  # Update path if needed
YOLO_MODEL_PATH = "yolov8n.pt"  # for ultralytics

# optional class names
CLASS_NAMES = ["person", "car", "bicycle", "motorbike", "truck", "other"]  # change to your dataset labels

# load model
if MODEL_TYPE == "onnx":
    try:
        model = ONNXModel(MODEL_PATH)
    except Exception as e:
        print(f"ONNX model load failed: {e}")
        model = None
else:
    try:
        model = YOLOv8Model(YOLO_MODEL_PATH)
    except Exception as e:
        print(f"YOLO model load failed: {e}")
        model = None

# Global video source (default webcam)
video_src = 0
video_capture = VideoCaptureAsync(video_src)
video_capture.start()

@app.get("/")
async def index():
    return HTMLResponse(open("app/client/index.html", "r").read())

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/infer")
async def infer_image(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image")
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    dets = model.predict(frame)
    annotated = draw_detections(frame.copy(), dets, class_names=CLASS_NAMES)
    _, jpeg = cv2.imencode('.jpg', annotated)
    return {"detections": dets, "image_base64": jpeg.tobytes().hex()}

@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            frame = video_capture.read()
            if frame is None:
                await asyncio.sleep(0.01)
                continue
            # run inference
            if model:
                dets = model.predict(frame)
            else:
                dets = []
            annotated = draw_detections(frame.copy(), dets, class_names=CLASS_NAMES)
            b64 = encode_frame_to_jpeg_b64(annotated)
            payload = json.dumps({"frame": b64, "fps": 0})
            await websocket.send_text(payload)
            await asyncio.sleep(0.03)  # roughly 30 FPS loop, actual speed depends on inference
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print("Stream error:", e)
        await websocket.close()

if __name__ == "__main__":
    uvicorn.run("app.server:app", host="0.0.0.0", port=8501, reload=False)
