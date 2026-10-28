import os
import cv2
from ultralytics import YOLO
import time

def analyze_basketball_v2(video_path="sample.mp4"):
    # יצירת תיקיית פלט ייעודית ומסודרת ל-PoC 2
    output_dir = "poc2_test_frames"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading YOLOv8 Small model & Analyzing '{video_path}'...")
    # מודל Small (yolov8s) לזיהוי משופר של כדורסל
    model = YOLO('yolov8s.pt')

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video '{video_path}'")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # קובץ הוידאו המוער יישמר בתוך התיקייה
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(os.path.join(output_dir, 'poc2_output.mp4'), fourcc, fps, (frame_width, frame_height))

    frame_count = 0
    ball_detected_frames = 0
    max_players = 0
    start_time = time.time()

    # שמירת תמונה כל N פריימים כדי לא להציף בתמונות
    save_interval = max(30, total_video_frames // 8) if total_video_frames > 0 else 60

    print(f"\n--- 🏀 Basketball Analytics PoC 2 Started (Outputs -> ./{output_dir}/) ---")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # imgsz=1280 משפר זיהוי כדור קטן; conf=0.15 תופס כדור בתנועה
        results = model(frame, classes=[0, 32], conf=0.15, imgsz=1280, verbose=False)
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

        if num_players > max_players:
            max_players = num_players
        if num_ball > 0:
            ball_detected_frames += 1

        annotated_frame = results[0].plot()
        out.write(annotated_frame)

        # שמירת דגימות בודדות בלבד בתוך התיקייה המיועדת
        if frame_count % save_interval == 0 or num_ball > 0 and ball_detected_frames == 1:
            sample_path = os.path.join(output_dir, f"sample_frame_{frame_count:04d}.jpg")
            cv2.imwrite(sample_path, annotated_frame)
            print(f"Frame {frame_count:04d} | Players: {num_players} | Ball: {'YES 🏀' if num_ball > 0 else 'NO ❌'} | Saved sample to {sample_path}")

    cap.release()
    out.release()

    total_time = time.time() - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0
    ball_detection_rate = (ball_detected_frames / frame_count) * 100 if frame_count > 0 else 0

    print("\n================ 📊 ANALYSIS SUMMARY (PoC 2) ================")
    print(f"Total Frames Analyzed:     {frame_count}")
    print(f"Total Processing Time:    {total_time:.2f} seconds")
    print(f"Average Inference Speed:  {avg_fps:.2f} FPS")
    print(f"Max Players Detected:     {max_players}")
    print(f"Ball Detection Rate:      {ball_detection_rate:.1f}% of frames")
    print("=============================================================")
    print(f"📁 All clean outputs are isolated in: ./{output_dir}/")

if __name__ == "__main__":
    analyze_basketball_v2('sample.mp4')