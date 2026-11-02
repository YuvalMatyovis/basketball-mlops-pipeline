# ball_v1 (single-class ball detector)

- Base: yolo11n (Ultralytics), fine-tuned in Colab (T4): 50 epochs, imgsz=1280, hsv_h=0.05, ~15 min. Best checkpoint = epoch 43.
- Data: FOTEVOLLY dataset (Ultralytics Platform, public; 417 images, 1 class). License/terms not verified - credit the author before publishing.
  Re-export the .ndjson from the platform to reproduce (signed URLs expire). Prepared with footvolley/src/prepare_dataset.py
  (block split: 298 train / 119 val).
- Val results (dataset's own val, 114 instances): P 0.906, R 0.807, mAP50 0.831, mAP50-95 0.321
- Caveats: small noisy val set (mAP50 last 10 epochs: mean 0.81, std 0.016); best.pt chosen on the same val;
  domain differs from our clip (broadcast beach volleyball vs phone footvolley).
- NOT yet evaluated on foochi_test (the real test; never used for training).
- Weights: models/ball_v1.pt (not in git). Training notebook: run in Colab with args.yaml settings.
