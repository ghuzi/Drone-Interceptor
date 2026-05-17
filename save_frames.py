import cv2
import os

# Folder banao jahan frames save hongi
os.makedirs("drone_frames", exist_ok=True)

cap = cv2.VideoCapture(0)  # 0 = laptop camera
saved = 0

print("✅ Camera chal raha hai!")
print("👉 SPACE dabao frame save karne ke liye")
print("👉 Q dabao band karne ke liye")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Screen pe info dikhao
    cv2.putText(frame, f"Saved Frames: {saved}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, "SPACE=Save  Q=Quit", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Drone Frame Collector", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):  # SPACE se save karo
        filename = f"drone_frames/frame_{saved:04d}.jpg"
        cv2.imwrite(filename, frame)
        saved += 1
        print(f"💾 Frame {saved} save ho gayi!")

    elif key == ord('q'):  # Q se band karo
        break

cap.release()
cv2.destroyAllWindows()
print(f"\n✅ Total {saved} frames save hui!")
print(f"📁 Location: drone_frames/ folder mein")