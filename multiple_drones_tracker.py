import cv2
from ultralytics import YOLO
import numpy as np
from collections import deque

model = YOLO('/home/muhammad-ghuzaif/Downloads/yolo_project/best.pt')

# Har drone ke liye alag trajectory
trajectories = {}
colors = [(0,0,255), (0,255,0), (255,0,0), (0,255,255), (255,0,255)]

cap = cv2.VideoCapture(0)

print("✅ Multiple Drone Tracking chal raha hai! Q dabao band karne ke liye")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, conf=0.5, verbose=False)

    # Is frame mein detected drones
    current_drones = []

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            x_center = int((x1 + x2) / 2)
            y_center = int((y1 + y2) / 2)

            # Drone ID assign karo
            drone_id = i
            color = colors[drone_id % len(colors)]

            # Trajectory initialize karo
            if drone_id not in trajectories:
                trajectories[drone_id] = deque(maxlen=30)

            trajectories[drone_id].appendleft((x_center, y_center))
            current_drones.append(drone_id)

            # Box draw karo
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Drone ID dikhao
            cv2.putText(frame, f"Drone {drone_id+1}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Confidence dikhao
            conf = float(r.boxes.conf[i])
            cv2.putText(frame, f"{conf:.2f}", (x1, y2+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Trajectory draw karo har drone ke liye
    for drone_id, pts in trajectories.items():
        color = colors[drone_id % len(colors)]
        for i in range(1, len(pts)):
            if pts[i-1] is None or pts[i] is None:
                continue
            thickness = int(np.sqrt(30 / float(i + 1)) * 2.5)
            cv2.line(frame, pts[i-1], pts[i], color, thickness)

    # Total drones count dikhao
    cv2.putText(frame, f"Drones Detected: {len(current_drones)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Multiple Drone Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()