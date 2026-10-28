import time
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
VIDEO_PATH = ROOT / "data" / "sample.mp4"
MODELS_DIR = ROOT / "models"

PERSON, BALL = 0, 32
WARMUP_FRAMES = 5

CONFIGS = [
    {"name": "A (Baseline)",   "model": "yolov8n.pt", "imgsz": 640,  "conf": 0.25},
    {"name": "B (Mid-Ground)", "model": "yolov8n.pt", "imgsz": 960,  "conf": 0.25},
    {"name": "C (Heavy)",      "model": "yolov8s.pt", "imgsz": 1280, "conf": 0.15},
]


def load_frames(path, max_frames):
    """Read frames once so every config sees identical input and disk I/O isn't timed."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def run_config(cfg, frames):
    model = YOLO(str(MODELS_DIR / cfg["model"]))
    kwargs = dict(classes=[PERSON, BALL], conf=cfg["conf"], imgsz=cfg["imgsz"], verbose=False)

    for f in frames[:WARMUP_FRAMES]:  # warm-up, not measured
        model(f, **kwargs)

    ball_frames, ball_confs, players = 0, [], 0
    start = time.perf_counter()
    for f in frames:
        boxes = model(f, **kwargs)[0].boxes
        cls = boxes.cls.int().tolist()
        confs = boxes.conf.tolist()
        players += cls.count(PERSON)
        if BALL in cls:
            ball_frames += 1
            ball_confs += [c for k, c in zip(cls, confs) if k == BALL]
    elapsed = time.perf_counter() - start

    n = len(frames)
    return {
        "Config": cfg["name"],
        "Model": cfg["model"].replace(".pt", ""),
        "Imgsz": cfg["imgsz"],
        "Conf": cfg["conf"],
        "FPS": f"{n / elapsed:.2f}",
        "Ball det. rate": f"{ball_frames / n * 100:.1f}% ({ball_frames}/{n})",
        "Avg ball conf": f"{sum(ball_confs) / len(ball_confs):.2f}" if ball_confs else "-",
        "Avg players": f"{players / n:.1f}",
    }


def main(max_frames=150):
    frames = load_frames(VIDEO_PATH, max_frames)
    print(f"Benchmarking {len(frames)} frames per config\n")

    rows = []
    for cfg in CONFIGS:
        print(f"Running {cfg['name']}...")
        rows.append(run_config(cfg, frames))

    # Markdown table: prints cleanly in the terminal and pastes straight into the README
    headers = list(rows[0].keys())
    print("\n| " + " | ".join(headers) + " |")
    print("|" + "|".join("---" for _ in headers) + "|")
    for r in rows:
        print("| " + " | ".join(str(r[h]) for h in headers) + " |")


if __name__ == "__main__":
    main()