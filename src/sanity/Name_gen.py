from pathlib import Path
import csv
import random
import os
import sys
import cv2

VIDEO_ROOT = Path(r"C:\Users\samue\OneDrive\Desktop\heart_beat\heart_beats\data\Dataset2\Videos")
OUTPUT_CSV = Path(r"data\CSVFiles\dataset2.csv")
FRAME_PICK_MODE = "first"
USE_ABS_PATHS = False
VIDEO_EXTENSIONS = {".mp4", ".avi"}


def find_videos(root: Path):
    return [p for p in root.rglob("*") if p.suffix.lower() in VIDEO_EXTENSIONS]


def select_frame_index(total_frames: int, mode: str) -> int:
    if total_frames <= 0:
        return 0
    mode_key = (mode or "middle").lower()
    if mode_key == "first":
        return 0
    if mode_key == "random":
        start = max(0, int(total_frames * 0.05))
        end = max(start, int(total_frames * 0.95) - 1)
        return random.randint(start, end) if end >= start else 0
    return max(0, total_frames // 2)


def load_haar_cascade():
    cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    classifier = cv2.CascadeClassifier(cascade_path)
    if classifier.empty():
        raise RuntimeError(f"Failed to load Haar cascade at {cascade_path}")
    return classifier


def detect_primary_face(img_bgr, face_cascade):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        flags=cv2.CASCADE_SCALE_IMAGE,
        minSize=(30, 30),
    )
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
    return int(x), int(y), int(x + w), int(y + h)


def read_frame_bgr(path: Path, frame_mode: str):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None, None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = select_frame_index(total, frame_mode)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None, None
    return idx, frame


def infer_subject_id(path: Path) -> str:
    def looks_like_video_dir(name: str) -> bool:
        n = name.strip().lower()
        return n == "videos" or "video" in n

    for parent in [path.parent] + list(path.parents):
        if parent == path or parent.name == "":
            continue
        if not looks_like_video_dir(parent.name):
            return parent.name
    return path.parent.name


def detect_face_with_retry(vid: Path, face_cascade):
    modes = ["first", "middle"] if FRAME_PICK_MODE.lower() == "first" else [FRAME_PICK_MODE]
    for mode in modes:
        _, frame = read_frame_bgr(vid, mode)
        if frame is None:
            continue
        bbox = detect_primary_face(frame, face_cascade)
        if bbox is not None:
            return bbox
    return None


def main():
    videos = find_videos(VIDEO_ROOT)
    if not videos:
        print(f"No video files found under: {VIDEO_ROOT}")
        return

    try:
        face_cascade = load_haar_cascade()
    except Exception as e:
        print(f"Error loading face detector: {e}")
        return

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["subject", "video_path", "x1", "y1", "x2", "y2"])

        for vid in sorted(videos):
            subject_id = infer_subject_id(vid)
            bbox = detect_face_with_retry(vid, face_cascade)
            if bbox is None:
                x1 = y1 = x2 = y2 = -1
            else:
                x1, y1, x2, y2 = bbox

            video_path_str = str(vid if USE_ABS_PATHS else vid.relative_to(VIDEO_ROOT))
            writer.writerow([subject_id, video_path_str, x1, y1, x2, y2])

    print(f"Wrote {OUTPUT_CSV} with {len(videos)} rows.")


if __name__ == "__main__":
    main()
