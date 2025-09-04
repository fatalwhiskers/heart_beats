import cv2
import csv
import numpy as np
import os
import src.Skin_Makse as mask
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import urllib.request
from src.config import Video, fileDataset1, fileDataset2
import src.extract_wave as ext

def load_crop_settings(csv_path):
    crop_settings = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                filename = row['filename'].strip()
                file_CSV = row['file_CSV'].strip()
                x1 = int(row['x1'])
                y1 = int(row['y1'])
                x2 = int(row['x2'])
                y2 = int(row['y2'])
                crop_settings.append((filename, file_CSV, x1, y1, x2, y2))
            except (KeyError, ValueError) as e:
                print(f"Skipping row due to error: {e}")
                continue  # skip malformed rows
    return crop_settings

MODEL_DIR = "models"  # put everything under ./models

MODEL_URLS = {
    "blaze_face_short_range.tflite":
        "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
    "face_landmarker.task":
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
}


def get_model_path(filename):
    """
    Ensure the model file exists locally.
    Downloads it from Google storage if missing.
    Returns the local path.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, filename)

    if not os.path.exists(path):
        url = MODEL_URLS.get(filename)
        if url is None:
            raise ValueError(f"No URL known for {filename}")
        print(f"Downloading {filename} ...")
        urllib.request.urlretrieve(url, path)
        print(f"Saved to {path}")

    return path

def get_bbox_forehead_from_tasks_bbox(frame, bbox_px):
    h, w, _ = frame.shape
    x = int(bbox_px.origin_x)
    y = int(bbox_px.origin_y)
    width  = int(bbox_px.width)
    height = int(bbox_px.height)

    band_h = int(0.14 * height)
    top    = y + int(0.10 * height)   # shift below hairline

    fh_y1 = max(0, top)
    fh_y2 = min(h, top + band_h)
    fh_x1 = max(0, x + int(0.10 * width))            # inset left/right a bit
    fh_x2 = min(w, x + int(0.90 * width))
    return fh_x1, fh_y1, fh_x2, fh_y2


# NEW: landmarks → forehead crop using MediaPipe Tasks landmarks
def get_mesh_forehead_from_tasks_landmarks(frame, face_landmarks):
    h, w, _ = frame.shape
    FOREHEAD_LANDMARKS = [103, 67, 109, 10, 338, 297, 332, 333, 299, 337, 251, 108, 69, 104]
    points = [(int(face_landmarks[i].x * w), int(face_landmarks[i].y * h))
              for i in FOREHEAD_LANDMARKS if i < len(face_landmarks)]
    xs, ys = zip(*points)
    pad = 10
    x1 = max(0, min(xs) - pad)
    x2 = min(w, max(xs) + pad)
    y1 = max(0, min(ys) - pad)
    y2 = min(h, max(ys) + pad)
    return x1, y1, x2, y2

class FaceDetectorV2:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = get_model_path("blaze_face_short_range.tflite")
        base = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.FaceDetectorOptions(
            base_options=base,
            running_mode=mp_vision.RunningMode.VIDEO,
            min_detection_confidence=0.7
        )
        self.detector = mp_vision.FaceDetector.create_from_options(options)
        self._t = 0

    def process(self, rgb_frame, timestamp_ms: float):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        return self.detector.detect_for_video(mp_image, int(timestamp_ms))

    def close(self):
        # Be robust even if mediapipe version lacks .close()
        if getattr(self, "detector", None) is not None:
            try:
                self.detector.close()
            except AttributeError:
                pass

class FaceLandmarkerV2:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = get_model_path("face_landmarker.task")
        base = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.7,
            min_face_presence_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        self._t = 0

    def process(self, rgb_frame, timestamp_ms: float):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        return self.landmarker.detect_for_video(mp_image, int(timestamp_ms))

    def close(self):
        if getattr(self, "landmarker", None) is not None:
            try:
                self.landmarker.close()
            except AttributeError:
                pass


def get_bbox_forehead(frame, detection):
    h, w, _ = frame.shape
    bbox = detection.location_data.relative_bounding_box
    x = int(bbox.xmin * w)
    y = int(bbox.ymin * h)
    width = int(bbox.width * w)
    height = int(bbox.height * h)

    fh_y1 = max(0, y)
    fh_y2 = min(h, y + int(0.25 * height))   # <-- clamp to bottom
    fh_x1 = max(0, x)
    fh_x2 = min(w, x + width)
    return fh_x1, fh_y1, fh_x2, fh_y2

# -------------------------------
# Debug Mesh helper
# -------------------------------
def get_mesh_forehead(frame, face_landmarks):
    h, w, _ = frame.shape


    #FOREHEAD_LANDMARKS = [104,69,108,151,337,299,333,298,293,334,296,336,9,107,66,105]

    FOREHEAD_LANDMARKS = [103,67,109,10,338,297,332,333,299,337,251,108,69,104]

    points = [(int(face_landmarks.landmark[i].x * w),
               int(face_landmarks.landmark[i].y * h)) for i in FOREHEAD_LANDMARKS]

    xs, ys = zip(*points)
    pad = 10  # padding around forehead box
    x1 = max(0, min(xs) - pad)
    x2 = min(w, max(xs) + pad)
    y1 = max(0, min(ys) - pad)
    y2 = min(h, max(ys) + pad)

    return x1, y1, x2, y2


# -------------------------------
# Debug overlay helper
# -------------------------------
def draw_debug_overlay(frame, crop_mode, crop_coords=None, landmarks=None):
    debug_frame = frame.copy()

    if crop_coords:
        x1, y1, x2, y2 = crop_coords
        color_map = {
        'manual':       (0, 255, 0),    # green
        'face_track':   (255, 165, 0),  # orange
        'bbox_forehead':(255, 0, 0),    # blue
        'mesh_forehead':(0, 0, 255)     # red
    }
        color = color_map.get(crop_mode, (255, 255, 255))
        cv2.rectangle(debug_frame, (x1, y1), (x2, y2), color, 2)

    if landmarks:
        for px, py in landmarks:
            cv2.circle(debug_frame, (px, py), 1, (0, 255, 255), -1)  # yellow points

    return debug_frame


def extract_video_to_rgb(video_path, x1=0, y1=0, x2=0, y2=0,
    crop_mode='manual',
    display=False, testing=False,
    test_output_dir=r"outputs\test_frames",
    skin_mask = False,
    apply_detrend=True,
    interpolate=True
):

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video at {video_path}")
        return None, None, None, None

    if display:
        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

    if testing:
        os.makedirs(test_output_dir, exist_ok=True)

    R, G, B = [], [], []
    timestamps = []
    frame_count = 0
    face_detector = None
    face_mesh = None
    face_landmarker = None

    try:
        if crop_mode in ('face_track_old', 'bbox_forehead_old'):
            face_detector = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.7)
        if crop_mode == 'mesh_forehead_old':
            face_mesh =  mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                            refine_landmarks=True, min_detection_confidence=0.7)
        if crop_mode in ('face_track', 'bbox_forehead'):
             face_detector = FaceDetectorV2()         # auto-downloads if missing
        if crop_mode == 'mesh_forehead':
            face_landmarker = FaceLandmarkerV2()
        while cap.isOpened():
            detected = False
            ret, frame = cap.read()
            if not ret:
                break
            h, w, _ = frame.shape
            crop_frame = frame
            crop_coords = None
            landmarks = None

            if crop_mode == 'manual':
                detected = True
                crop_frame = frame[y1:y2, x1:x2]
                crop_coords = (x1, y1, x2, y2)
                

            elif crop_mode == 'none':
                detected = True
                crop_frame = frame

            elif crop_mode == 'face_track_old':
                detected = True
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_detector.process(rgb)

                if results.detections:
                    bbox = results.detections[0].location_data.relative_bounding_box
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)

                    margin_x = int(width * 0.1)
                    margin_y = int(height * 0.2)

                    x1 = max(0, x - margin_x)
                    y1 = max(0, y - margin_y)
                    x2 = min(w, x + width + margin_x)
                    y2 = min(h, y + height + margin_y)

                    crop_frame = frame[y1:y2, x1:x2]
                    crop_coords = (x1, y1, x2, y2)
                    detected = True
            
            elif crop_mode == 'face_track':
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                ts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                results = face_detector.process(rgb, ts_ms)
                if results.detections:
                    det = results.detections[0]
                    bbox = det.bounding_box  # pixel bbox
                    x = int(bbox.origin_x)
                    y = int(bbox.origin_y)
                    width = int(bbox.width)
                    height = int(bbox.height)
                    x1 = x
                    y1 = y
                    x2 = x + width
                    y2 = y + height

                    margin_x = int(width * 0.1)
                    margin_y = int(height * 0.2)

                    x1 = max(0, x1 - margin_x)
                    y1 = max(0, y1 - margin_y)
                    x2 = min(w, x2 + margin_x)
                    y2 = min(h, y2 + margin_y)

                    crop_frame = frame[y1:y2, x1:x2]
                    crop_coords = (x1, y1, x2, y2)
                    detected = True

            elif crop_mode == 'bbox_forehead_old':
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                ts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                results = face_detector.process(rgb, ts_ms)
                if results.detections:
                    x1, y1, x2, y2 = get_bbox_forehead(frame, results.detections[0])
                    crop_frame = frame[y1:y2, x1:x2]
                    crop_coords = (x1, y1, x2, y2)
                    detected = True

            elif crop_mode == 'bbox_forehead':
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                ts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                results = face_detector.process(rgb, ts_ms)
                if results.detections:
                    det = results.detections[0]
                    bbox = det.bounding_box
                    x1, y1, x2, y2 = get_bbox_forehead_from_tasks_bbox(frame, bbox)
                    crop_frame = frame[y1:y2, x1:x2]
                    crop_coords = (x1, y1, x2, y2)
                    detected = True

            elif crop_mode == 'mesh_forehead_old':
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                ts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                results = face_mesh.process(rgb, ts_ms)
                if results.multi_face_landmarks:
                    x1, y1, x2, y2 = get_mesh_forehead(frame, results.multi_face_landmarks[0])
                    crop_frame = frame[y1:y2, x1:x2]
                    crop_coords = (x1, y1, x2, y2)
                    landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in results.multi_face_landmarks[0].landmark]
                    detected = True

            elif crop_mode == 'mesh_forehead':
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_landmarker.process(rgb)
                if results.face_landmarks:
                    lms = results.face_landmarks[0]
                    x1, y1, x2, y2 = get_mesh_forehead_from_tasks_landmarks(frame, lms)
                    crop_frame = frame[y1:y2, x1:x2]
                    crop_coords = (x1, y1, x2, y2)
                    landmarks = [(int(pt.x * w), int(pt.y * h)) for pt in lms]
                    detected = True

            else:
                raise ValueError(f"Invalid crop_mode: {crop_mode}")
            if not detected:
                R.append(np.nan); G.append(np.nan); B.append(np.nan)
            else:
            # ---- Average immediately instead of storing frame ----
                if skin_mask:
                    rgb_crop = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2RGB)
                    R_val, G_val, B_val = mask.get_skin_means(rgb_crop)
                    B.append(B_val)
                    G.append(G_val)
                    R.append(R_val)
                else:
                    B.append(float(np.median(crop_frame[:, :, 0])))
                    G.append(float(np.median(crop_frame[:, :, 1])))
                    R.append(float(np.median(crop_frame[:, :, 2])))


            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            timestamps.append(timestamp)
            frame_count += 1

            if testing and frame_count <= 60:
                video_name = os.path.splitext(os.path.basename(video_path))[0]
                mode_dir = os.path.join(test_output_dir, video_name, crop_mode)
                os.makedirs(mode_dir, exist_ok=True)
                debug_frame = draw_debug_overlay(frame, crop_mode, crop_coords, landmarks)
                save_debug = os.path.join(mode_dir, f"{crop_mode}_frame_{frame_count:03d}_{video_name}.jpg")
                cv2.imwrite(save_debug, debug_frame)

            if testing and frame_count >= 60:
                break

            if display:
                cv2.imshow('frame', crop_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
    # Clean up only the ones you created
        if face_detector is not None:
            face_detector.close()
        if face_mesh is not None:
            face_mesh.close()        
        if face_landmarker is not None:       # <-- add
            face_landmarker.close()        
    cap.release()
    if display:
        cv2.destroyAllWindows()

    if interpolate:
        R_signal , t_uniform = ext.interpolate_signal_with_timestamps(R, timestamps, target_fps=None, max_gap_sec=0.5) 
        B_signal , _         = ext.interpolate_signal_with_timestamps(B, timestamps, t_uniform=t_uniform, max_gap_sec=0.5)
        G_signal , _         = ext.interpolate_signal_with_timestamps(G, timestamps, t_uniform=t_uniform, max_gap_sec=0.5)
    Video.FPS = 1.0 / np.median(np.diff(t_uniform))
    if apply_detrend:
        R_signal = ext.sliding_mean_normalize(R_signal, Video.FPS)
        G_signal = ext.sliding_mean_normalize(G_signal, Video.FPS)
        B_signal = ext.sliding_mean_normalize(B_signal, Video.FPS)




    fs = 1.0 / np.median(np.diff(t_uniform))

    return np.array(R_signal), np.array(G_signal), np.array(B_signal), np.array(t_uniform)
