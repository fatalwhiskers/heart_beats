"""
generate_synthetic_face.py
---------------------------
Generates a short synthetic "face" video for testing the rPPG pipeline
without needing a real person's video. No real or identifiable person
appears in the output -- every frame is procedurally drawn.

The face includes a subtle, periodic colour pulse (mimicking real blood
flow) so the demo's heart-rate estimators have something genuine to find,
not just noise.

NOTE: MediaPipe's face detector is trained on real photographic faces.
A synthetic face is not guaranteed to be detected -- this script makes a
reasonable attempt (skin-toned gradient shading, correctly-proportioned
features) but if detection fails when you run the demo on the output,
that's an expected possible outcome, not a bug. The 'manual' or 'none'
crop modes will always work on this video regardless, since they don't
need face detection at all.

Usage
-----
    python demo/synthetic_test/generate_synthetic_face.py
"""

import os
import numpy as np
import cv2

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "synthetic_face.mp4")

WIDTH, HEIGHT = 640, 480
FPS = 30
DURATION_S = 25
TRUE_HR_BPM = 75  # the heart rate this video is designed to produce


def make_frame(t: float) -> np.ndarray:
    """Draw one frame at time t (seconds). Returns a BGR uint8 image."""
    frame = np.full((HEIGHT, WIDTH, 3), (40, 35, 30), dtype=np.uint8)  # dark background

    cx, cy = WIDTH // 2, HEIGHT // 2
    face_w, face_h = 180, 230

    # Subtle periodic colour pulse mimicking blood-flow variation,
    # strongest in red/reduced in blue -- the same physiological basis
    # the real pipeline relies on.
    hr_hz = TRUE_HR_BPM / 60.0
    pulse = np.sin(2 * np.pi * hr_hz * t)
    base_skin = np.array([90, 130, 200], dtype=np.float64)  # BGR, skin-toned
    skin = base_skin + np.array([0, -3, 6]) * pulse
    skin = np.clip(skin, 0, 255).astype(np.uint8)

    # Face oval with soft gradient shading (flat colour reads as
    # "cartoon" to a detector trained on photographs)
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    cv2.ellipse(mask, (cx, cy), (face_w // 2, face_h // 2), 0, 0, 360, 255, -1)

    grad = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float64)
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH]
    dist = np.sqrt(((xx - cx) / (face_w / 2)) ** 2 + ((yy - cy) / (face_h / 2)) ** 2)
    shade = np.clip(1.0 - 0.25 * dist, 0.6, 1.0)
    for c in range(3):
        grad[:, :, c] = skin[c] * shade

    face_region = mask > 0
    frame[face_region] = grad[face_region].astype(np.uint8)

    # Eyes
    eye_y = cy - face_h // 6
    for dx in (-1, 1):
        ex = cx + dx * face_w // 4
        cv2.ellipse(frame, (ex, eye_y), (22, 12), 0, 0, 360, (245, 245, 245), -1)
        cv2.circle(frame, (ex, eye_y), 7, (40, 30, 20), -1)

    # Eyebrows
    for dx in (-1, 1):
        ex = cx + dx * face_w // 4
        cv2.ellipse(frame, (ex, eye_y - 20), (24, 6), 0, 180, 360, (60, 45, 35), -1)

    # Nose (simple shaded triangle/line)
    nose_top = (cx, cy - 10)
    nose_bottom = (cx, cy + 35)
    cv2.line(frame, nose_top, nose_bottom, tuple(int(c * 0.85) for c in skin.tolist()), 4)

    # Mouth
    mouth_y = cy + face_h // 3
    cv2.ellipse(frame, (cx, mouth_y), (38, 14), 0, 0, 180, (60, 50, 110), -1)

    # Soft blur to remove hard cartoon edges, closer to photographic softness
    frame = cv2.GaussianBlur(frame, (5, 5), 0)

    return frame


def main():
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, FPS, (WIDTH, HEIGHT))

    n_frames = int(FPS * DURATION_S)
    for i in range(n_frames):
        t = i / FPS
        frame = make_frame(t)
        writer.write(frame)

    writer.release()
    print(f"Wrote {n_frames} frames ({DURATION_S}s at {FPS}fps) to:")
    print(f"  {OUTPUT_PATH}")
    print(f"Designed true heart rate: {TRUE_HR_BPM} bpm")


if __name__ == "__main__":
    main()
