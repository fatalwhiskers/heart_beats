# Demo: try the rPPG pipeline on your own video

This is a stripped-down way to see the pipeline work, without needing the
original dissertation dataset (which can't be shared, since it contains
identifiable faces).

**Don't have a video to upload yet, or just want a fast sanity check?**
See `demo/synthetic_test/` -- it generates a synthetic, non-identifiable
face video and runs the pipeline against it automatically, no upload
needed. Good for quickly confirming everything's installed and working
before bothering with a real video.

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
   You'll be asked to pick a crop mode, a signal-channel category, and one
   or more HR estimation methods (or pass them directly, see below).

5. **Check the result.**
   The console prints a summary for each signal produced (mean HR, valid
   window count, etc.), and one plot per HR method is saved to
   `demo/output/` (e.g. `heart_rate_welch.png`, `heart_rate_hilbert.png`).
   Click those files in the VS Code explorer to view them.

## Crop modes

| Mode | What it does |
|---|---|
| `face_track` | Tracks the whole face region |
| `bbox_forehead` | Crops to the forehead using the face bounding box (good default) |
| `mesh_forehead` | Crops to the forehead using a precise 478-point face mesh |
| `poly` | Uses forehead + both cheeks for maximum skin coverage |

## Signal-channel categories

| Category | What it does |
|---|---|
| `R` / `G` / `B` | A single colour channel. `G` is the standard rPPG default. |
| `GREY_W` / `GREY_A` | Weighted or simple-average greyscale |
| `PCA` | Principal Component Analysis -- produces up to 3 signals |
| `ZCA` | ZCA whitening -- produces up to 3 signals |
| `ICA` | Independent Component Analysis -- produces up to 3 signals |
| `CHROM` | Chrominance-based method |
| `POS` | Plane-Orthogonal-to-Skin method |

Categories that produce multiple signals (PCA, ZCA, ICA) will plot all of
them together on the same graph, one line per signal, so you can compare
them directly.

## HR estimation methods

| Method | What it does |
|---|---|
| `welch` | Welch periodogram (via neurokit2) -- generally the most robust |
| `hilbert` | Hilbert-transform instantaneous HR, windowed to match Welch |

You can select one or both. Each selected method produces its own
separate plot, so they're never tangled together on one graph.

Example passing everything directly, skipping all interactive prompts:

```
python demo/demo_run.py demo/your_video_filename.mp4 --crop_mode mesh_forehead --channel PCA --hr_method welch --hr_method hilbert
```

## A note on video length

Both HR methods use a sliding-window approach to smooth out noise. The
demo uses a 10-second window with a 1-second step (shorter than the
30-second window used elsewhere in this project for longer dissertation
recordings), so a clip as short as ~15-20 seconds will still produce
several HR estimates. Shorter than that, and you may get very few or zero
valid windows -- if that happens, try a longer clip or a different
crop_mode.


## A note on accuracy

This demo has no ground-truth heart rate to compare against (e.g. from a
pulse oximeter or ECG), so there's no way to verify the *accuracy* of the
estimate from this alone. It demonstrates that the pipeline runs correctly
end-to-end: detects a face, extracts a colour signal, and produces a
heart-rate estimate. The dissertation itself evaluates accuracy against
ground-truth physiological data, which isn't part of this public demo.
