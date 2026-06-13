from ultralytics import YOLO
import cv2

model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot open camera")
    exit()

print("✅ Camera opened — press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Cannot read frame")
        break

    results = model(frame, verbose=False)
    annotated = results[0].plot()
    cv2.imshow('YOLO Camera Detection', annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
