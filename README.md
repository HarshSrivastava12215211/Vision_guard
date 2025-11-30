# VisionGuard — Real-Time Video Stream Detection & Inference

VisionGuard is a real-time computer vision inference system: object detection & anomaly detection over live video. Built to be easy to run locally or in Docker.

## Features
- YOLOv8-compatible training & inference
- ONNX export + ONNX Runtime inference
- FastAPI server with WebSocket streaming
- Async producer-consumer video pipeline
- Dockerized for easy deployment

## Repo Layout
See project tree in repo.

## Quickstart (Local, CPU)
1. Clone repo:
```bash
git clone https://github.com/HarshSrivastava12215211/visionguard.git
cd visionguard
