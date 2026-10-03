import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load YOLO model
model = YOLO("yolov8n.pt")

# Initialize Deep SORT
tracker = DeepSort(
    max_age=30,
    n_init=2,
    max_cosine_distance=0.3,
    nn_budget=None,
    embedder="mobilenet",
    half=True
)

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Object Detection and Tracking started.")
print("Press Q to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    # YOLO detection
    results = model(frame, verbose=False)

    detections = []

    for result in results:
        for box in result.boxes:
            confidence = float(box.conf[0])

            if confidence < 0.40:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            width = x2 - x1
            height = y2 - y1

            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            detections.append(
                ([x1, y1, width, height], confidence, class_name)
            )

    # Update tracker
    tracks = tracker.update_tracks(detections, frame=frame)

    # Draw tracked objects
    for track in tracks:

        if not track.is_confirmed():
            continue

        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())

        class_name = track.get_det_class() or "Object"

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        label = f"{class_name} | ID: {track_id}"

        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    cv2.putText(
        frame,
        "CodeAlpha - Object Detection and Tracking",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow("Object Detection and Tracking", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("Application stopped.")
