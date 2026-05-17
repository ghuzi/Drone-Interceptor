from ultralytics import YOLO
import cv2

# 1. YOLOv8 Nano model load karein (Ye fast chalta hai)
model = YOLO('yolov8n.pt')

# 2. Camera setup (0 built-in camera ke liye hai)
# Agar error aaye to '0' ki jagah 1 ya 2 try karein
results = model.predict(source='0', show=True, conf=0.5)

print("Camera band karne ke liye 'q' dabayein ya terminal mein Ctrl+C karein.")