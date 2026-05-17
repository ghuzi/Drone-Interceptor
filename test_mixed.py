from ultralytics import YOLO
import cv2
import os

# Model load karo
model = YOLO("best.pt")

# Test karo
results = model.predict(
    source="bird3.jpeg",
    conf=0.7,
    save=True
)

print("\n--- TEST RESULTS ---\n")

for r in results:
    img_name = os.path.basename(r.path)
    print(f"Image: {img_name}")

    if len(r.boxes) == 0:
        print("  Kuch detect nahi hua")
    else:
        for box in r.boxes:
            label = model.names[int(box.cls)]
            conf  = float(box.conf[0])
            print(f"  Detected: {label} | Confidence: {conf:.2f}")

    # Window open rakho
    img = r.plot()
    cv2.imshow("Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()