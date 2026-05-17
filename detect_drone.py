from ultralytics import YOLO
import cv2

# Model load karo
model = YOLO('/home/muhammad-ghuzaif/Downloads/yolo_project/best.pt')

# Tumhari actual files
images = [
    '/home/muhammad-ghuzaif/Downloads/yolo_project/drone_pic1.webp',
    '/home/muhammad-ghuzaif/Downloads/yolo_project/drone_pic2.jpg',
    '/home/muhammad-ghuzaif/Downloads/yolo_project/drone_pic3.jpeg',
    '/home/muhammad-ghuzaif/Downloads/yolo_project/drone_pic4.jpg',
    '/home/muhammad-ghuzaif/Downloads/yolo_project/drone_pic5.avif',
]

for img_path in images:
    results = model(img_path, conf=0.3)
    annotated = results[0].plot()
    
    cv2.imshow("Drone Detection", annotated)
    print(f"✅ {img_path} - Done!")
    
    cv2.waitKey(0)  # Next image ke liye koi bhi key dabao

cv2.destroyAllWindows()
print("✅ Sab images detect ho gayi!")