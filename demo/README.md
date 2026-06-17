# Demo: try the rPPG pipeline on your own video

This is a stripped-down way to see the pipeline work, without needing the
original dissertation dataset (which can't be shared, since it contains
identifiable faces).

## Important: this needs a video FILE, not a live webcam

GitHub Codespaces runs in the cloud, so there is no way for it to see a
physical webcam attached to your computer. You'll need to upload an
existing video file (a short clip recorded on your phone works fine --
a well-lit, front-facing video of a face, 20-30 seconds is plenty).

## Steps

1. **Open this repo in a Codespace.**
   On the GitHub repo page, click the green "Code" button, then the
   "Codespaces" tab, then "Create codespace on main." Wait for it to finish
   building (this runs automatically the first time -- it's installing
   Python packages, OpenCV's system dependencies, etc., and can take a
   couple of minutes).

2. **Upload a video file.**
   In the VS Code file explorer on the left, right-click the `demo` folder
   and choose "Upload..." Select a video file from your computer. It'll
   appear inside `demo/`.

3. **Open a terminal.**
   `Terminal -> New Terminal` from the top menu (or `` Ctrl+` ``).

4. **Run the demo.**
   ```
   python demo/demo_run.py demo/your_video_filename.mp4
   ```
   You'll be asked to pick a crop mode (or pass one directly with
   `--crop_mode bbox_forehead`, see below).

5. **Check the result.**
   The console prints a summary (frames processed, estimated mean heart
   rate, etc.), and a plot is saved to `demo/output/heart_rate.png`.
   Click that file in the VS Code explorer to view it.

## Crop modes

| Mode | What it does |
|---|---|
| `face_track` | Tracks the whole face region |
| `bbox_forehead` | Crops to the forehead using the face bounding box (good default) |
| `mesh_forehead` | Crops to the forehead using a precise 478-point face mesh |
| `poly` | Uses forehead + both cheeks for maximum skin coverage |

Example passing the mode directly, skipping the interactive prompt:

```
python demo/demo_run.py demo/your_video_filename.mp4 --crop_mode mesh_forehead
```

## A note on accuracy

This demo has no ground-truth heart rate to compare against (e.g. from a
pulse oximeter or ECG), so there's no way to verify the *accuracy* of the
estimate from this alone. It demonstrates that the pipeline runs correctly
end-to-end: detects a face, extracts a colour signal, and produces a
heart-rate estimate. The dissertation itself evaluates accuracy against
ground-truth physiological data, which isn't part of this public demo.
