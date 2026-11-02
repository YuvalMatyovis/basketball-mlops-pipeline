# Ball Sports Video Analytics (MLOps learning project)

Detecting players and a ball in sports clips with YOLO, with reproducible
experiments to compare models and settings.

| Project | What | Status |
|---|---|---|
| [`footvolley/`](footvolley/) | Ball-keeping circles (4 players): tracking, touches, passes, chains | active |
| [`basketball/`](basketball/) | Initial YOLO benchmark on a basketball clip | PoC done |

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
Videos (`*/data/*.mp4`) and weights (`models/*.pt`) are not in git.
Each project has its own README with run commands and results.

## 📊 Final Detector Benchmarks (Test Set: 40 frames)

| Model | Precision | Recall | F1-Score | Status / Insight |
| :--- | :--- | :--- | :--- | :--- |
| **YOLO11n (COCO)** | 1.00 | 0.09 | 0.17 | Too rigid; misses 91% of fast-moving sports balls. |
| **HSV Color (Classical CV)** | 0.00 | 0.00 | 0.00 | Failed due to rigid color thresholding vs real-world lighting. |
| **ball_v1 (External Data)** | 0.00 | 0.00 | 0.00 | Domain shift failure (trained on external public dataset). |
| **ball_v2 (Custom Fine-Tuned)** | **0.89** | **0.73** | **0.80** | **Winner:** Fine-tuned on custom annotated frames via Colab T4 GPU. |

