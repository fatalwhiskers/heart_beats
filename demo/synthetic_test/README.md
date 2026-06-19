# Synthetic test (no real person required)

This folder lets you confirm the pipeline works without uploading any
video at all -- it procedurally generates a synthetic, non-identifiable
"face" with a built-in periodic colour pulse, then runs it through the
real pipeline.

## Why this exists

The main demo (`demo/demo_run.py`) needs a real video of a face, which
can't be shared in this repo. This synthetic version sidesteps that
entirely: nothing here is or depicts a real person, so it can be safely
committed, run by anyone, anywhere, with zero setup beyond installing
the project's dependencies.

## Run it

```
python demo/synthetic_test/run_test.py
```

That's it -- no arguments needed. It generates a 25-second synthetic face
video (if one doesn't already exist), runs it through `mesh_forehead`
crop mode and the Welch HR method, and prints/plots the result. The
video is designed for a true heart rate of 75 bpm, so a working pipeline
should report a result close to that.

## Options

```
python demo/synthetic_test/run_test.py --crop_mode bbox_forehead --channel PCA --hr_method welch --hr_method hilbert
```

Same `--crop_mode`, `--channel`, and `--hr_method` options as the main
demo (see `demo/README.md`).

## A note on detection reliability

MediaPipe's face detectors are trained on real photographic faces, not
drawn ones. In testing, the 478-point face mesh (`mesh_forehead`, `poly`)
detected this synthetic face on nearly every frame, while the simpler
bounding-box detector (`face_track`, `bbox_forehead`) was noticeably less
consistent -- though the pipeline's existing Kalman-filter smoothing
still produced a usable signal in both cases. `mesh_forehead` is the
default here for that reason. If you ever see a "no face was detected"
message, that's the one genuine failure mode worth knowing about -- it
means the synthetic face wasn't detected in a single frame of the video,
which is a real (if unlikely) possibility with non-photographic input,
not a bug in the pipeline itself.
