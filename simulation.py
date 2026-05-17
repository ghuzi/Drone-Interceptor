import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque

model = YOLO('/home/muhammad-ghuzaif/Downloads/yolo_project/best.pt')

canvas_h, canvas_w = 600, 900

# Drone image load karo
drone_img = cv2.imread('/home/muhammad-ghuzaif/Downloads/yolo_project/drone_pic2.jpg')
drone_img = cv2.resize(drone_img, (120, 80))

# 3 Drones — alag positions aur speeds
drones = [
    {'x': 100, 'y': 100, 'dx': 3,  'dy': 2,  'color': (0, 0, 255),   'pts': deque(maxlen=30)},
    {'x': 400, 'y': 300, 'dx': -4, 'dy': 3,  'color': (0, 255, 0),   'pts': deque(maxlen=30)},
    {'x': 700, 'y': 200, 'dx': 2,  'dy': -3, 'color': (255, 0, 0),   'pts': deque(maxlen=30)},
]

print("✅ Multiple Drone Simulation chal rahi hai! Q dabao band karne ke liye")

while True:
    # Sky background
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:] = (135, 100, 30)

    # Har drone ko move karo
    for i, drone in enumerate(drones):
        drone['x'] += drone['dx']
        drone['y'] += drone['dy']

        # Walls se bounce
        if drone['x'] <= 0 or drone['x'] >= canvas_w - 120:
            drone['dx'] *= -1
        if drone['y'] <= 0 or drone['y'] >= canvas_h - 80:
            drone['dy'] *= -1

        # Drone draw karo
        x, y = drone['x'], drone['y']
        canvas[y:y+80, x:x+120] = drone_img

        # Center point trajectory mein add karo
        center = (x + 60, y + 40)
        drone['pts'].appendleft(center)

    # YOLO se detect karo
    results = model(canvas, conf=0.3, verbose=False)
    annotated = results[0].plot()

    # Har drone ki trajectory draw karo
    for i, drone in enumerate(drones):
        color = drone['color']
        pts = drone['pts']
        for j in range(1, len(pts)):
            if pts[j-1] is None or pts[j] is None:
                continue
            thickness = int(np.sqrt(30 / float(j + 1)) * 2.5)
            cv2.line(annotated, pts[j-1], pts[j], color, thickness)

        # Drone number dikhao
        cv2.putText(annotated, f"Drone {i+1}", 
                    (drone['x'], drone['y'] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Info dikhao
    cv2.putText(annotated, "Interceptor Drone Simulation", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(annotated, f"Drones Detected: {len(results[0].boxes)}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow("Multiple Drone Simulation", annotated)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()