# src/train_yolo.py
"""
Train YOLOv8 using ultralytics API.
Prepare a data.yaml with paths & class names:
train: /path/to/train/images
val: /path/to/val/images
nc: <num_classes>
names: ['class0','class1',...]
"""

from ultralytics import YOLO
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data.yaml", help="path to data yaml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="starting weights")
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(data=args.data, imgsz=args.imgsz, epochs=args.epochs, batch=16, device=0)  # device=0 for GPU

if __name__ == "__main__":
    main()
