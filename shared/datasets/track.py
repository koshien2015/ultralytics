from ultralytics import YOLO, FastSAM
import cv2
# Load an official or custom model
# model = YOLO("yolo11x.pt")  # Load an official Detect model
# model = YOLO("yolo11n-seg.pt")  # Load an official Segment model
# model = YOLO("yolo11n-pose.pt")  # Load an official Pose model
model = YOLO("YOLO11m_20250426_epoch100.pt")  # Load a custom trained model
video_path = "test7.mp4"  # Path to the video file
# Perform tracking with the model
results = model(
    source=video_path,  # Path to the video file
    show=False, 
    save=True
)  # Tracking with ByteTrack tracker
