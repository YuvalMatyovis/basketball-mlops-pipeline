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
