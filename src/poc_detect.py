import cv2
from ultralytics import YOLO
import time

def analyze_basketball_video(video_path="sample.mp4"):
    print(f"Loading YOLOv8 model & Analyzing '{video_path}'...")
    model = YOLO('yolov8n.pt')

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video '{video_path}'. Make sure the file exists!")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

    # הגדרת קובץ וידאו תוצאה מעובד
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, fps, (frame_width, frame_height))

    frame_count = 0
    ball_detected_frames = 0
    max_players_detected = 0
    start_time = time.time()

    print("\n--- 🏀 Basketball Analytics Ingest Started ---")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Class 0 = Person (שחקנים), Class 32 = Sports Ball (כדור)
        results = model(frame, classes=[0, 32], conf=0.25, verbose=False)
        boxes = results[0].boxes

        num_players = 0
        num_ball = 0
        
        if boxes is not None:
            for cls in boxes.cls:
                cls_id = int(cls.item())
                if cls_id == 0:
                    num_players += 1
                elif cls_id == 32:
                    num_ball += 1

        if num_players > max_players_detected:
            max_players_detected = num_players
        if num_ball > 0:
            ball_detected_frames += 1

        # ציור התיבות על הפריים
        annotated_frame = results[0].plot()
        out.write(annotated_frame)

        # הדפסת מטריקות בזמן אמת כל 15 פריימים (בערך חצי שנייה של וידאו)
        if frame_count % 15 == 0:
            print(f"Frame {frame_count:04d} | Players: {num_players} | Ball: {'YES 🏀' if num_ball > 0 else 'NO ❌'}")
            # שמירת תמונת דגימה לצפייה בעיניים
            cv2.imwrite(f"frame_sample_{frame_count}.jpg", annotated_frame)

    cap.release()
    out.release()

    total_time = time.time() - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0
    ball_detection_rate = (ball_detected_frames / frame_count) * 100 if frame_count > 0 else 0

    print("\n================ 📊 ANALYSIS SUMMARY ================")
    print(f"Total Frames Analyzed:     {frame_count}")
    print(f"Total Processing Time:    {total_time:.2f} seconds")
    print(f"Average Inference Speed:  {avg_fps:.2f} FPS")
    print(f"Max Players Detected:     {max_players_detected}")
    print(f"Ball Detection Rate:      {ball_detection_rate:.1f}% of frames")
    print("=====================================================")
    print("📁 Saved annotated video: output.mp4")
    print("🖼️ Saved visual snapshots: 'frame_sample_X.jpg'")

if __name__ == "__main__":
    analyze_basketball_video('sample.mp4')