# Basketball PoC

Benchmark of 3 YOLO configs on 150 frames of a clip with 1-2 players.
Run: `python basketball/src/benchmark_basketball.py` (clip in `basketball/data/sample.mp4`).

| Config | Model | Imgsz | Conf | FPS | Ball det. rate | Avg ball conf | Avg players |
|---|---|---|---|---|---|---|---|
| A (Baseline) | yolov8n | 640 | 0.25 | 9.76 | 10.7% (16/150) | 0.41 | 1.3 |
| B (Mid-Ground) | yolov8n | 960 | 0.25 | 4.68 | 16.0% (24/150) | 0.53 | 1.3 |
| C (Heavy) | yolov8s | 1280 | 0.15 | 1.18 | 28.0% (42/150) | 0.70 | 1.5 |

Higher resolution and a bigger model improved ball detection but cost ~8x speed.
Detection rate is not accuracy (no ground truth).
