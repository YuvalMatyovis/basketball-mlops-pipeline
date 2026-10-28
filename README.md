# Basketball Video Analysis (MLOps learning project)

Detecting players and the ball in basketball clips with YOLOv8, with a
reproducible benchmark to compare model configurations.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/benchmark.py
```
Data (`data/sample.mp4`) and weights (`models/*.pt`) are not in git.

## Benchmark results
150 frames from `sample.mp4` (CPU, GitHub Codespaces).

| Config | Model | Imgsz | Conf | FPS | Ball det. rate | Avg ball conf | Avg players |
|---|---|---|---|---|---|---|---|
| A (Baseline) | yolov8n | 640 | 0.25 | 9.76 | 10.7% (16/150) | 0.41 | 1.3 |
| B (Mid-Ground) | yolov8n | 960 | 0.25 | 4.68 | 16.0% (24/150) | 0.53 | 1.3 |
| C (Heavy) | yolov8s | 1280 | 0.15 | 1.18 | 28.0% (42/150) | 0.70 | 1.5 |

Note: "Ball det. rate" is the share of frames with at least one ball
detected, not accuracy (no ground truth yet).

## Findings
- Higher resolution and a larger model improve ball detection (10.7% -> 28%),
  but cost ~8x in speed.
- The clip contains 1-2 players, so ~1.3-1.5 avg players is plausible.
- The pretrained COCO model is weak on basketballs, which motivates fine-tuning.

## Next steps
- [ ] Inspect saved frames visually (A vs C)
- [ ] Hand-label ~30 frames for ground truth (precision/recall)
- [ ] Fine-tune YOLOv8n on a Roboflow basketball dataset (Colab)
- [ ] Add tracking (ByteTrack), then MLflow and DVC
- [ ] Decide on direction: shot analysis / pose estimation# basketball-mlops-pipeline