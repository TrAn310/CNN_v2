from ultralytics import YOLO
model = YOLO("best.pt")
model.export(format="onnx", imgsz=416, simplify=True)