from ultralytics import YOLO
model = YOLO("/mnt/data1/workspace/wmq/ultralytics/runs/detect/train28/weights/last.pt")
path = model.export(format="onnx") 
print(path)