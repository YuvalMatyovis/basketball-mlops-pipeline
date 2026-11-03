# Footvolley Analytics Pipeline

A computer vision and data analytics pipeline for footvolley games and ball-keeping circles, built on YOLOv8/YOLO11 detection models and ByteTrack tracking.

## Project Status (Proof of Concept)

- **Perception pipeline:** Player and ball detection and tracking run end to end, extracting structured per-frame positions to `tracking_data.json`. Players are detected and tracked reliably in test clips (roughly 90% by visual check); ball detection remains the main bottleneck and is being actively improved with a custom-trained model (`ball_v2`).
- **Repository organization:** Legacy basketball files removed and the models directory consolidated (`footvolley/models`).
- **Visual verification:** Conversion pipeline to `footvolley_preview.mp4` for direct preview in Codespaces.

## Near-Term Goals (Post-Processing & Analytics)

1. **Gap filling (linear interpolation):** Estimate ball positions in frames where the ball is lost to height or motion blur.
2. **Zonal mapping:** Dynamically assign players to the four court zones (`Top`, `Bottom`, `Left`, `Right`).
3. **Touch detection:** Combine spatial proximity with sharp changes in the ball's velocity vector.

---

> **Note on `ball_v2` benchmark metrics:**
> - The current test set contains 40 frames from a single clip.
> - Results are preliminary, directional evidence only.
