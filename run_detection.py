from ultralytics import YOLO
import cv2
import csv

# Load trained model
model = YOLO("runs/detect/train/weights/best.pt")

# Open video
video_path = "path_to_video.mp4"
cap = cv2.VideoCapture(video_path)

# Get FPS to log timestamp per frame
fps = cap.get(cv2.CAP_PROP_FPS)

# Prepare CSV log
with open("people_count_log.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["frame", "timestamp_sec", "person_count"])

    frame_num = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference
        results = model.predict(frame, conf=0.4, iou=0.5, verbose=False)
        boxes = results[0].boxes
        class_ids = boxes.cls.tolist()

        # Count only 'person' detections
        person_count = sum(1 for c in class_ids if int(c) == 0)

        timestamp = frame_num / fps
        writer.writerow([frame_num, round(timestamp, 2), person_count])

        frame_num += 1

print("✅ Done! Logged per-frame counts to people_count_log.csv")
