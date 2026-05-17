import cv2
from ultralytics import YOLO
import numpy as np
from collections import deque

# 1. Aapka trained model load karein (95.8% accuracy wala)
model = YOLO('best.pt') 

# 2. History points save karne ke liye (Trajectory ke liye)
# 'maxlen=30' ka matlab hai ke pichli 30 frames ki movement nazar aayegi
pts = deque(maxlen=30)

# 3. Gazebo ki window capture karne ke liye ya Camera feed
# Agar aap Gazebo ki screen capture karna chahte hain toh 'mss' library use karni hogi
# Abhi test karne ke liye hum default webcam use kar rahe hain
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv8 Detection
    results = model.predict(frame, conf=0.5) # Conf 0.5 rakha hai taake pakki detection ho

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        for box in boxes:
            # Bounding box ka center nikalna (x, y)
            x_center = int((box[0] + box[2]) / 2)
            y_center = int((box[1] + box[3]) / 2)
            center = (x_center, y_center)
            
            # Points ki list mein center add karein
            pts.appendleft(center)
            
            # Drone ke gird box draw karein
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)

    # 4. Trajectory Drawing Logic
    # Pichle points ko aapas mein lines ke zariye jorna
    for i in range(1, len(pts)):
        if pts[i - 1] is None or pts[i] is None:
            continue
        
        # Line ki motai waqt ke saath kam hogi (Tailing effect)
        thickness = int(np.sqrt(30 / float(i + 1)) * 2.5)
        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

    # Output dikhayein
    cv2.imshow("Drone Trajectory & Prediction", frame)

    # 'q' dabane se band ho jayega
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()c