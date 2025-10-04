from ultralytics import YOLO

# Load a model
#model = YOLO("yolo11n.yaml")  # build a new model from YAML
# mpythodel = YOLO("best.pt")  # load a pretrained model (recommended for training)
model = YOLO("yolo11m.yaml").load("yolo11m.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(data="data.yaml", epochs=100, imgsz=640, patience=100, batch=4)