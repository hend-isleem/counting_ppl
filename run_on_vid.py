import os
import cv2
import csv
from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")
CONF_THRESHOLD = 0.5

base_dir = "own_data"
video_input_dir = os.path.join(base_dir, "videos_input")
video_output_dir = os.path.join(base_dir, "videos_output")
log_output_dir = os.path.join(base_dir, "logs_output")

os.makedirs(video_output_dir, exist_ok=True)
os.makedirs(log_output_dir, exist_ok=True)


def process_video(video_path, output_video_path, output_csv_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_count = 0

    with open(output_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "timestamp_sec", "person_count"])

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # The actuaalllll running of the model
            results = model.predict(frame, conf=CONF_THRESHOLD, iou=0.5, verbose=False)
            boxes = results[0].boxes
            class_ids = boxes.cls.tolist()
            person_count = sum(1 for c in class_ids if int(c) == 0)

            # Actuall counting!
            # label = f"People: {person_count}"
            # cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            # Draw bounding boxes and count
            for box, cls in zip(boxes.xyxy, class_ids):
                if int(cls) == 0:  # Only draw for 'person'
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Draw total count at top
            cv2.putText(frame, f"People: {person_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)


            out.write(frame)

            # Have the logs in the terminal so I can actually see things running :)
            timestamp = frame_count / fps
            print(f"{os.path.basename(video_path)} | Frame {frame_count} | Time {timestamp:.2f}s | {person_count}")
            writer.writerow([frame_count, round(timestamp, 2), person_count])

            frame_count += 1

    cap.release()
    out.release()
    print(f"!!!!!!!!!!!!! Done: {os.path.basename(video_path)}\n  {output_video_path}\n   {output_csv_path}")


def process_all_videos():
    os.makedirs(video_output_dir, exist_ok=True)
    os.makedirs(log_output_dir, exist_ok=True)

    video_files = [f for f in os.listdir(video_input_dir) if f.lower().endswith(".mp4")]

    for video_file in video_files:
        input_path = os.path.join(video_input_dir, video_file)
        output_video_path = os.path.join(video_output_dir, f"{os.path.splitext(video_file)[0]}_out.mp4")
        output_csv_path = os.path.join(log_output_dir, f"{os.path.splitext(video_file)[0]}.csv")
        process_video(input_path, output_video_path, output_csv_path)


def main():
    print("Here we go. PLEASE WORKKK...")
    process_all_videos()
    print(">>>>>>>>>>>>> All videos processed.")


if __name__ == "__main__":
    main()
