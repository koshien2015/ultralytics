from ultralytics import YOLO

# Load a model
#model = YOLO("yolo8m.yaml")  # build a new model from YAML
#model = YOLO("yolov8m.pt")  # load a pretrained model (recommended for training)
model = YOLO("yolov8m.yaml").load("yolov8m.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(data="data.yaml", epochs=100, imgsz=640, batch=6)