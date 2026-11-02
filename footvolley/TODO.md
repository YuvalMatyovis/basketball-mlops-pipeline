# Next steps
1. Look at ball_v1 on foochi_test (try_ball_v1.py) - are the boxes on the ball?
2. Ground truth: extract_gt_frames.py, label ~40 frames (ball box only; none if not visible), export.
3. Eval script: precision/recall/F1 vs ground truth for COCO yolo11s, HSV red, ball_v1.
4. Decide: good enough -> ball tracker (temporal filter + proximity), integrate into footvolley.py; else more clips/colors.
5. clips.csv (clip, colorway, family, background, camera, split, source).
6. Cleanup: slim requirements.txt, update READMEs, progress printing in footvolley.py.
Later: MLflow, DVC, GitHub Actions, Docker, HPC training, more colorways, app export (ONNX).
