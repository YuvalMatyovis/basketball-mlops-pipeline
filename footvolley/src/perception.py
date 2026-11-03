import cv2
import json
import os
from ultralytics import YOLO

# --- Path settings ---
# Make sure the video file name matches the one you uploaded!
VIDEO_PATH = "footvolley/data/clips/test_2min.mp4" 
OUTPUT_JSON = "footvolley/data/output/tracking_data.json"
OUTPUT_VIDEO = "footvolley/data/output/debug_perception.mp4"

# --- Load models ---
print("Loading models...")
people_model = YOLO("models/yolo11n.pt") # Standard YOLO model for people
ball_model = YOLO("models/ball_v2.pt")   # Our dedicated ball model

# --- Size filtering settings (removes background players and balls) ---
MIN_PERSON_AREA = 4000 # Minimum pixels to count as a player in the near circle
MIN_BALL_AREA = 20     # Minimum pixels to count as a ball

def process_video():
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Video not found at {VIDEO_PATH}")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Set up a video writer for visual inspection
    out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    tracking_data = []
    frame_idx = 0

    print("Processing video frames...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_data = {"frame": frame_idx, "people": [], "ball": None}
        
        # 1. Detect people and filter by size
        people_results = people_model(frame, classes=[0], verbose=False)[0] # class 0 = person
        for box in people_results.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = box
            area = (x2 - x1) * (y2 - y1)
            
            # Add only if the person is close enough (large)
            if area > MIN_PERSON_AREA:
                frame_data["people"].append([int(x1), int(y1), int(x2), int(y2), round(conf, 2)])
                # Draw a blue rectangle around the relevant players
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

        # 2. Detect the ball, pick the highest-confidence option and filter by size
        ball_results = ball_model(frame, verbose=False)[0]
        best_ball = None
        highest_conf = 0
        
        for box in ball_results.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = box
            area = (x2 - x1) * (y2 - y1)
            
            # Make sure it is not a ball on the beach behind (size) and that it is the best option in the frame
            if area > MIN_BALL_AREA and conf > highest_conf:
                best_ball = [int(x1), int(y1), int(x2), int(y2), round(conf, 2)]
                highest_conf = conf
                
        if best_ball:
            frame_data["ball"] = best_ball
            # Draw a green rectangle around the ball
            cv2.rectangle(frame, (best_ball[0], best_ball[1]), (best_ball[2], best_ball[3]), (0, 255, 0), 2)

        # Store the frame's data
        tracking_data.append(frame_data)
        out.write(frame)
        
        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx} frames...")

    cap.release()
    out.release()

    # Save all data to a JSON file used by the analytics "brain"
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(tracking_data, f, indent=4)

    print(f"\nDone! Tracking data saved to: {OUTPUT_JSON}")
    print(f"Debug video saved to: {OUTPUT_VIDEO}")

if __name__ == "__main__":
    process_video()
