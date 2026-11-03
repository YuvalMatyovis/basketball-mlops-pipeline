import cv2
import json
import os
from ultralytics import YOLO

# --- הגדרות נתיבים ---
# ודא ששם הסרטון תואם למה שהעלית!
VIDEO_PATH = "footvolley/data/clips/test_2min.mp4" 
OUTPUT_JSON = "footvolley/data/output/tracking_data.json"
OUTPUT_VIDEO = "footvolley/data/output/debug_perception.mp4"

# --- טעינת מודלים ---
print("Loading models...")
people_model = YOLO("models/yolo11n.pt") # מודל YOLO רגיל לאנשים
ball_model = YOLO("models/ball_v2.pt")   # המודל הייעודי שלנו לכדור

# --- הגדרות סינון גודל (מחיקת שחקנים וכדורים מהרקע) ---
MIN_PERSON_AREA = 4000 # פיקסלים מינימליים כדי להיחשב שחקן במעגל הקרוב
MIN_BALL_AREA = 20     # פיקסלים מינימליים כדי להיחשב כדור

def process_video():
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Video not found at {VIDEO_PATH}")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # הגדרת כותב וידאו לבדיקה ויזואלית
    out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    tracking_data = []
    frame_idx = 0

    print("Processing video frames...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_data = {"frame": frame_idx, "people": [], "ball": None}
        
        # 1. זיהוי אנשים וסינון גודל
        people_results = people_model(frame, classes=[0], verbose=False)[0] # class 0 = person
        for box in people_results.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = box
            area = (x2 - x1) * (y2 - y1)
            
            # הוספה רק אם האדם מספיק קרוב (גדול)
            if area > MIN_PERSON_AREA:
                frame_data["people"].append([int(x1), int(y1), int(x2), int(y2), round(conf, 2)])
                # ציור מלבן כחול סביב השחקנים הרלוונטיים
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

        # 2. זיהוי כדור, בחירת הוודאות הגבוהה ביותר וסינון גודל
        ball_results = ball_model(frame, verbose=False)[0]
        best_ball = None
        highest_conf = 0
        
        for box in ball_results.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = box
            area = (x2 - x1) * (y2 - y1)
            
            # וידוא שזה לא כדור בחוף מאחורה (גודל) ושהוא האופציה הטובה ביותר בפריים
            if area > MIN_BALL_AREA and conf > highest_conf:
                best_ball = [int(x1), int(y1), int(x2), int(y2), round(conf, 2)]
                highest_conf = conf
                
        if best_ball:
            frame_data["ball"] = best_ball
            # ציור מלבן ירוק סביב הכדור
            cv2.rectangle(frame, (best_ball[0], best_ball[1]), (best_ball[2], best_ball[3]), (0, 255, 0), 2)

        # שמירת הנתונים של הפריים
        tracking_data.append(frame_data)
        out.write(frame)
        
        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx} frames...")

    cap.release()
    out.release()

    # שמירת כל הנתונים לקובץ JSON שישמש את ה"מוח"
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(tracking_data, f, indent=4)

    print(f"\nDone! Tracking data saved to: {OUTPUT_JSON}")
    print(f"Debug video saved to: {OUTPUT_VIDEO}")

if __name__ == "__main__":
    process_video()