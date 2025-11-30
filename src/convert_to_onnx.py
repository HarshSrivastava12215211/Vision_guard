# src/convert_to_onnx.py
from ultralytics import YOLO
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="yolov8n.pt")
    parser.add_argument("--onnx", type=str, default="../models/visionguard.onnx")
    args = parser.parse_args()

    model = YOLO(args.weights)
    model.export(format="onnx", imgsz=640, dynamic=True, simplify=True, opset=12)
    print("Export completed. Check the output ONNX in the current dir or models folder.")

if __name__ == "__main__":
    main()
