import cv2
import csv
import numpy as np
import os
import src.smoother as smo
import src.not_working.Skin_Makse as mask
import src.PRV_0 as prv
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import urllib.request
from src.config import Video, fileDataset1, fileDataset2, BVP
import src.extract_wave as ext
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from collections import deque

FOREHEAD_IDX = [107, 66, 69, 109, 338, 299, 296, 336]
L_CHEEK_IDX  = [118, 119, 100, 126, 209, 49, 129, 203, 205, 50]
R_CHEEK_IDX  = [347, 348, 329, 355, 429, 279, 358, 423, 425, 280]

FOREHEAD_LANDMARKS = [103,67,109,10,338,297,332,333,299,337,251,108,69,104]

smooth = False

class RollingMedian:
    def __init__(self, window=5):
        self.window = window
        self.buf = deque(maxlen=window)
    def filter(self, x):
        self.buf.append(x)
        return float(np.median(self.buf))

def srgb_to_linear(x):
    x = x.astype(np.float32)
    if x.max() > 1: x /= 255.0
    a = 0.055
    return np.where(x <= 0.04045, x/12.92, ((x + a)/(1 + a))**2.4)

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

def load_crop_settings_D2(csv_path):
    crop_settings = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                subject = row['subject'].strip()
                video_path = row['video_path'].strip()
                file_CSV = row['file_CSV'].strip()
                x1 = int(row['x1'])
                y1 = int(row['y1'])
                x2 = int(row['x2'])
                y2 = int(row['y2'])
                crop_settings.append((subject, video_path, file_CSV, x1, y1, x2, y2))
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

def poly_from_indices(frame_shape, face_landmarks, idx_list):
    h, w = frame_shape[:2]
    pts = []
    for i in idx_list:
        if i < len(face_landmarks):
            x = int(np.clip(face_landmarks[i].x * w, 0, w-1))
            y = int(np.clip(face_landmarks[i].y * h, 0, h-1))
            pts.append((x, y))
    return np.array(pts, dtype=np.int32) if len(pts) >= 3 else None

def mean_rgb_in_polygon(rgb, poly_pts, erode_px=2):
    """Mean R,G,B inside (possibly non-convex) polygon on an RGB frame."""
    if poly_pts is None or len(poly_pts) < 3:
        return None
    h, w = rgb.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly_pts.astype(np.int32)], 255)  # supports non-convex

    if erode_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*erode_px+1, 2*erode_px+1))
        mask = cv2.erode(mask, k)

    # drop near-saturated or very dark pixels to avoid highlights/shadows
    sel = mask.astype(bool)
    if not np.any(sel):
        return None
    roi = rgb[sel]  # Nx3, RGB order
    v = roi.max(axis=1)
    keep = (v > 10) & (v < 250)
    if not np.any(keep):
        return None
    roi = roi[keep]

    R = float(roi[:, 0].mean())
    G = float(roi[:, 1].mean())
    B = float(roi[:, 2].mean())
    return (R, G, B)

def get_bbox_forehead_from_tasks_bbox(frame, bbox_px):
    h, w, _ = frame.shape
    x = int(bbox_px.origin_x)
    y = int(bbox_px.origin_y)
    width  = int(bbox_px.width)
    height = int(bbox_px.height)

    band_h = int(0.14 * height)
    top    = y - int(0.10 * height)   # shift below hairline

    fh_y1 = max(0, top)
    fh_y2 = min(h, top + band_h)
    fh_x1 = max(0, x + int(0.10 * width))            # inset left/right a bit
    fh_x2 = min(w, x + int(0.90 * width))
    return fh_x1, fh_y1, fh_x2, fh_y2

def get_mesh_forehead_from_tasks_landmarks(frame, face_landmarks):
    h, w, _ = frame.shape

    points = [(int(face_landmarks[i].x * w), int(face_landmarks[i].y * h))
              for i in FOREHEAD_IDX if i < len(face_landmarks)]
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

    points = [(int(face_landmarks.landmark[i].x * w),
               int(face_landmarks.landmark[i].y * h)) for i in FOREHEAD_IDX]

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


def extract_video_to_rgb(video_path, x1=1, y1=1, x2=1, y2=1,
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
    R_smooth, G_smooth, B_smooth = [], [], []
    Rm, Gm, Bm = [], [], []
    timestamps = []
    frame_count = 0
    face_detector = None
    face_mesh = None
    face_landmarker = None
    ROI_smoother = smo.OneEuroFilter(base_cutoff_hz=1.4, responsiveness=0.15, deriv_cutoff_hz=0.5)
   # r_filter     = smo.OneEuroFilter(base_cutoff_hz=0.8, responsiveness=0.15, deriv_cutoff_hz=0.5)
    #g_filter     = smo.OneEuroFilter(base_cutoff_hz=0.8, responsiveness=0.15, deriv_cutoff_hz=0.5)
    #b_filter     = smo.OneEuroFilter(base_cutoff_hz=0.8, responsiveness=0.15, deriv_cutoff_hz=0.5)
    kf = smo.ROIKalman()

    #r_medfilt = RollingMedian(window=5)
    #g_medfilt = RollingMedian(window=5)
    #b_medfilt = RollingMedian(window=5)
    debug_compare = False
    recording = False
    out_dir = "outputs/video"   # your desired folder
    os.makedirs(out_dir, exist_ok=True)

    if crop_mode == 'bbox_forehead_jitter':
        jm_raw = smo.JitterMeter(fps=Video.FPS, win_sec=1.0)
        jm_kf  = smo.JitterMeter(fps=Video.FPS, win_sec=1.0)
        jm_eur = smo.JitterMeter(fps=Video.FPS, win_sec=1.0)
        debug_compare = True
        recording = True
    compare_path = os.path.join(out_dir, "compare_bbox_forehead.mp4")
    compare_out = cv2.VideoWriter( compare_path ,
                                cv2.VideoWriter_fourcc(*"mp4v"),
                                15,
                                (1280, 720))
    try:
        if crop_mode in ('face_track_old', 'bbox_forehead_old'):
            face_detector = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.7)
        if crop_mode == 'mesh_forehead_old':
            face_mesh =  mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                            refine_landmarks=True, min_detection_confidence=0.7)
        if crop_mode in ('face_track', 'bbox_forehead', 'bbox_forehead_jitter'):
             face_detector = FaceDetectorV2()         # auto-downloads if missing
        if crop_mode in ('mesh_forehead' , 'poly'):            
            face_landmarker = FaceLandmarkerV2()
        while cap.isOpened():
            detected = False
            ret, frame = cap.read()
            if not ret:
                break
            h, w, _ = frame.shape
            crop_frame = frame
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            crop_coords = None
            landmarks = None

            if crop_mode == 'manual':
                detected = True
                crop_frame = rgb[y1:y2, x1:x2]
                crop_coords = (x1, y1, x2, y2)
                

            elif crop_mode == 'none':
                detected = True
                crop_frame = rgb

            elif crop_mode == 'face_track_old':
                detected = True
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

                    crop_frame = rgb[y1:y2, x1:x2]
                    crop_coords = (x1, y1, x2, y2)
                    detected = True
            
            elif crop_mode == 'face_track':
                results = face_detector.process(rgb, ts_ms)
                if results.detections:
                    det = results.detections[0]
                    bbox = det.bounding_box  # pixel bbox

                    x = int(bbox.origin_x)
                    y = int(bbox.origin_y)
                    width = int(bbox.width)
                    height = int(bbox.height)

                    x1_raw = x
                    y1_raw = y
                    x2_raw = x + width
                    y2_raw = y + height

                    raw_box = (int(x1_raw), int(y1_raw), int(x2_raw), int(y2_raw))

                    # --- KALMAN smoothing (pixel-domain xyxy) ---
                    conf = getattr(det, "score", [1.0])[0] if hasattr(det, "score") else 1.0
                    kf_box = kf.update(raw_box, conf=float(conf))
                    kx1, ky1, kx2, ky2 = kf_box

                    x1, y1, x2, y2 = kx1, ky1, kx2, ky2         

                    margin_x = int(width * 0.1)
                    margin_y = int(height * 0.2)

                    x1_F = max(0, x1 - margin_x)
                    y1_F = max(0, y1 - margin_y)
                    x2_F = min(w, x2 + margin_x)
                    y2_F = min(h, y2 + margin_y)

                    crop_frame = rgb[y1_F:y2_F, x1_F:x2_F]
                    crop_coords = (x1_F, y1_F, x2_F, y2_F)
                    detected = True

            elif crop_mode == 'bbox_forehead_old':
                ts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                results = face_detector.process(rgb, ts_ms)
                if results.detections:
                    x1, y1, x2, y2 = get_bbox_forehead(rgb, results.detections[0])
                    crop_frame = rgb[y1:y2, x1:x2]
                    crop_coords = (x1, y1, x2, y2)
                    detected = True

            elif crop_mode == 'bbox_forehead_jitter':
                ts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                t_s   = ts_ms / 1000.0
                results = face_detector.process(rgb, ts_ms)
                if results.detections:
                    det = results.detections[0]
                    bbox = det.bounding_box
                    x1_raw, y1_raw, x2_raw, y2_raw = get_bbox_forehead_from_tasks_bbox(rgb, bbox)
                    raw_box = (int(x1_raw), int(y1_raw), int(x2_raw), int(y2_raw))
                    conf = getattr(det, "score", [1.0])[0] if hasattr(det, "score") else 1.0
                    kf_box = kf.update(raw_box, conf=float(conf))
                    kx1, ky1, kx2, ky2 = kf_box

                    x1, y1, x2, y2 = kx1, ky1, kx2, ky2

                    crop_frame = rgb[y1:y2, x1:x2]
                    crop_coords = (x1, y1, x2, y2)
                    detected = True

                    cx, cy, w, h = smo.box_xyxy_to_cxcywh(*raw_box)
                    smoothed_cxcywh = ROI_smoother.filter(np.array([cx, cy, w, h], dtype=np.float32),
                              time_s=ts_ms * 1e-3)
                    eur_box = smo.box_cxcywh_to_xyxy(*smoothed_cxcywh)
                    ex1, ey1, ex2, ey2 = eur_box
                    jm_raw.add(raw_box)
                    jm_kf.add(kf_box)
                    jm_eur.add(eur_box)

                    crop_frame = rgb[y1:y2, x1:x2]
                    crop_coords = (x1, y1, x2, y2)
                    detected = True
                    
                    # --- optional: visual comparison panel ---
                    if debug_compare:  # set this True to write a side-by-side panel
                        dbg = rgb.copy()
                        cv2.rectangle(dbg, (x1_raw,y1_raw), (x2_raw,y2_raw), (0,0,255), 2)  # raw=red
                        cv2.rectangle(dbg, (kx1,ky1), (kx2,ky2), (0,255,0), 2)              # kalman=green
                        cv2.rectangle(dbg, (int(ex1),int(ey1)), (int(ex2),int(ey2)), (255,0,0), 2)  # euro=blue
                        panel = np.hstack([rgb, dbg])
                        cv2.putText(dbg, "Raw=Red | Kalman=Green | one_Euro=blue",
                                    (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

                        if recording:
                            compare_out.write(dbg)   # VideoWriter you init once
                    

            elif crop_mode == 'bbox_forehead':
                ts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                t_s   = ts_ms / 1000.0

                results = face_detector.process(rgb, ts_ms)
                if results.detections:
                    det = results.detections[0]
                    bbox = det.bounding_box

                    # --- RAW forehead box from detector bbox ---
                    x1_raw, y1_raw, x2_raw, y2_raw = get_bbox_forehead_from_tasks_bbox(rgb, bbox)
                    raw_box = (int(x1_raw), int(y1_raw), int(x2_raw), int(y2_raw))

                    # --- KALMAN smoothing (pixel-domain xyxy) ---
                    conf = getattr(det, "score", [1.0])[0] if hasattr(det, "score") else 1.0
                    kf_box = kf.update(raw_box, conf=float(conf))
                    kx1, ky1, kx2, ky2 = kf_box

                    x1, y1, x2, y2 = kx1, ky1, kx2, ky2

                    # clamp to frame
                    x1 = max(0, min(x1, rgb.shape[1]-1)); x2 = max(0, min(x2, rgb.shape[1]-1))
                    y1 = max(0, min(y1, rgb.shape[0]-1)); y2 = max(0, min(y2, rgb.shape[0]-1))

                    # crop
                    if x2 > x1 and y2 > y1:
                        crop_frame = rgb[y1:y2, x1:x2]
                        crop_coords = (x1, y1, x2, y2)
                        detected = True
                    else:
                        crop_frame = None
                        crop_coords = None
                        detected = False


            elif crop_mode == 'mesh_forehead_old':
                results = face_mesh.process(rgb, ts_ms)
                if results.multi_face_landmarks:
                    x1_raw, y1_raw, x2_raw, y2_raw = get_mesh_forehead(rgb, results.multi_face_landmarks[0])
                    state_raw = smo.box_xyxy_to_cxcywh(x1_raw, y1_raw, x2_raw, y2_raw)       # [cx,cy,w,h]
                    state_s   = None #ROI_smoother.filter(state_raw.astype(np.float32), t_s)
                    x1, y1, x2, y2 = smo.box_cxcywh_to_xyxy(*state_s)
                    crop_frame = rgb[y1:y2, x1:x2]
                    crop_coords = (x1, y1, x2, y2)
                    landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in results.multi_face_landmarks[0].landmark]
                    detected = True

            elif crop_mode == 'mesh_forehead':
                ts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                t_s = ts_ms / 1000.0
                results = face_landmarker.process(rgb, ts_ms)
                if results.face_landmarks:
                    lms = results.face_landmarks[0]
                    x1_raw, y1_raw, x2_raw, y2_raw = get_mesh_forehead_from_tasks_landmarks(rgb, lms)
                    raw_box = (int(x1_raw), int(y1_raw), int(x2_raw), int(y2_raw))

                    # --- KALMAN smoothing (pixel-domain xyxy) ---
                    conf = getattr(lms, "score", [1.0])[0] if hasattr(lms, "score") else 1.0
                    kf_box = kf.update(raw_box, conf=float(conf))
                    kx1, ky1, kx2, ky2 = kf_box
                    landmarks = [(int(pt.x * w), int(pt.y * h)) for pt in lms]
                    kx1, ky1, kx2, ky2 = kf_box

                    x1, y1, x2, y2 = kx1, ky1, kx2, ky2

                    # clamp to frame
                    x1 = max(0, min(x1, rgb.shape[1]-1)); x2 = max(0, min(x2, rgb.shape[1]-1))
                    y1 = max(0, min(y1, rgb.shape[0]-1)); y2 = max(0, min(y2, rgb.shape[0]-1))
                    detected = True
                    # crop
                    if x2 > x1 and y2 > y1:
                        crop_frame = rgb[y1:y2, x1:x2]
                        crop_coords = (x1, y1, x2, y2)
                        detected = True
                    else:
                        crop_frame = None
                        crop_coords = None
                        detected = False

            elif crop_mode == 'poly':
                results = face_landmarker.process(rgb, ts_ms)
                if results.face_landmarks:
                    lms = results.face_landmarks[0]  # Mediapipe Tasks landmarks (normalized)
                    ts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                    t_s = ts_ms / 1000.0
                    # Build polygon masks (forehead + both cheeks)
                    mF  = polygon_mask_from_indices(rgb.shape, lms, FOREHEAD_IDX)
                    mLC = polygon_mask_from_indices(rgb.shape, lms, L_CHEEK_IDX)
                    mRC = polygon_mask_from_indices(rgb.shape, lms, R_CHEEK_IDX)

                    # Union of facial ROI polygons (OR)
                    poly_mask = None
                    for m in (mF, mLC, mRC):
                        if m is not None:
                            poly_mask = m if poly_mask is None else cv2.bitwise_or(poly_mask, m)

                    if poly_mask is None:
                        detected = False
                    else:
                        # Intersect with quick skin mask to knock out hair/eyes/background
                        rgb8 = ensure_uint8_rgb(rgb)
                        skin = quick_skin_mask(rgb8, morph=True)
                        roi_mask = combine_masks(poly_mask, skin)

                        if roi_mask is not None and np.any(roi_mask):
                            meanRGB = masked_mean_rgb(rgb, roi_mask)
                            if meanRGB is not None:
                                R_mean, G_mean, B_mean = meanRGB
                                R_mean = float(R_mean)
                                G_mean = float(G_mean)
                                B_mean = float(B_mean)
                                Bm.append(B_mean)
                                Gm.append(G_mean)
                                Rm.append(R_mean)
                                detected = True
                            else:
                                detected = False
                        else:
                            detected = False
                else:
                    detected = False

            else:
                raise ValueError(f"Invalid crop_mode: {crop_mode}")
            if not detected:
                if crop_mode != 'poly':
                    smooth_box = kf.update(None, conf=0.0)
                    if smooth_box is None:
                        Rm.append(np.nan); Gm.append(np.nan); Bm.append(np.nan)
                    else:
                        x1, y1, x2, y2 = smooth_box
                        crop_frame = rgb[y1:y2, x1:x2]
                        crop_coords = (x1, y1, x2, y2)
                        Rma, Gma, Bma, skinmask = rgb_means_with_skin(crop_frame, equalize_Y=True)
                        Bm.append(Bma)
                        Gm.append(Gma)
                        Rm.append(Rma)
                else:
                    Rm.append(np.nan); Gm.append(np.nan); Bm.append(np.nan)
            else:

                if crop_mode != 'poly':
                    ts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                    t_s = ts_ms / 1000.0
                   # R_mean = float(np.mean(crop_frame[:, :, 0]))
                  #  G_mean = float(np.mean(crop_frame[:, :, 1]))
                  #  B_mean = float(np.mean(crop_frame[:, :, 2]))

                   # R_s = float(r_filter.filter(np.array([R_mean], dtype=np.float32), t_s)[0])
                   # G_s = float(g_filter.filter(np.array([G_mean], dtype=np.float32), t_s)[0])
                   # B_s = float(b_filter.filter(np.array([B_mean], dtype=np.float32), t_s)[0])

                   # R_med = r_medfilt.filter(R_mean)
                   # G_med = g_medfilt.filter(G_mean)
                   # B_med = b_medfilt.filter(B_mean)
                    
                    #R_smooth.append(R_med); G_smooth.append(G_med); B_smooth.append(B_med)
                 #   B.append(float(np.mean(crop_frame[:, :, 2])))
                #    G.append(float(np.mean(crop_frame[:, :, 1])))
                 #   R.append(float(np.mean(crop_frame[:, :, 0])))

                    Rma, Gma, Bma, skinmask = rgb_means_with_skin(crop_frame, equalize_Y=True)
                    Bm.append(Bma)
                    Gm.append(Gma)
                    Rm.append(Rma)






            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            timestamps.append(timestamp)
            frame_count += 1

            if testing and frame_count <= 0:
                video_name = os.path.splitext(os.path.basename(video_path))[0]
                mode_dir = os.path.join(test_output_dir, video_name, crop_mode)
                os.makedirs(mode_dir, exist_ok=True)
                debug_frame = draw_debug_overlay(frame, crop_mode, crop_coords, landmarks)
                save_debug = os.path.join(mode_dir, f"{crop_mode}_frame_{frame_count:03d}_{video_name}.jpg")
                cv2.imwrite(save_debug, debug_frame)

            if testing and frame_count >= Video.FPS * 30:
                break

            if display:
                cv2.imshow('frame', crop_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if display:
                # Ensure crop box is int pixels (smoother may output floats)
                box = tuple(map(int, crop_coords)) if crop_coords else None

                # 1) Full-size view with box + landmarks
                full_debug = draw_debug_overlay(frame, crop_mode, box, landmarks)
                cv2.imshow('full', full_debug)

                # 2) (optional) Also show the cropped ROI next to it
                # crop_frame is RGB; OpenCV expects BGR to look right
                if crop_frame is not None and crop_frame.size:
                    cv2.imshow('crop', cv2.cvtColor(crop_frame, cv2.COLOR_RGB2BGR))

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
    compare_out.release()
    cap.release()
    if display:
        cv2.destroyAllWindows()

    if interpolate:
       # R_signal , G_signal, B_signal, t_uniform = ext.resample_rgb_pchip(R, G, B, timestamps) 
      #  R_smooth_signal , G_smooth_signal, B_smooth_signal, t_uniform = ext.resample_rgb_pchip(R_smooth, G_smooth, B_smooth, timestamps) 
        Rm_signal , Gm_signal , Bm_signal  , t_uniform = ext.resample_rgb_pchip(Rm, Gm, Bm, timestamps)
    else:
        t_uniform = np.array(timestamps, dtype=float)
        R_signal = np.array(R, dtype=float)
        G_signal = np.array(G, dtype=float)
        B_signal = np.array(B, dtype=float)

    Rm_signal, t_uniform,  mask_valid, idx_valid, reinsert = ext.fill_short_gaps_then_drop(Rm_signal, t_uniform)
    Gm_signal, _,  mask_valid, idx_valid, reinsert = ext.fill_short_gaps_then_drop(Gm_signal, t_uniform)
    Bm_signal, _,  mask_valid, idx_valid, reinsert = ext.fill_short_gaps_then_drop(Bm_signal, t_uniform)
    #summarize_all(jm_raw, jm_kf, jm_eur, out_csv_path="outputs/video/roi_stability_summary.csv")
   # R_signal, _,  mask_valid, idx_valid, reinsert = ext.fill_short_gaps_then_drop(R_signal, t_uniform)
   # G_signal, _,  mask_valid, idx_valid, reinsert = ext.fill_short_gaps_then_drop(G_signal, t_uniform)
   # B_signal, _,  mask_valid, idx_valid, reinsert = ext.fill_short_gaps_then_drop(B_signal, t_uniform)
   # plot_traces(t_uniform, G_signal, G_smooth_signal, Gm_signal )
    if crop_mode == 'bbox_forehead_jitter':
        smo.summarize_all(jm_raw, jm_kf, jm_eur)
    return np.array(Rm_signal), np.array(Gm_signal), np.array(Bm_signal), np.array(t_uniform)
   # R_signal, G_signal, B_signal, bad = repair_impulses_rgb(R_signal, G_signal, B_signal, win_sec=0.5, n_sigma=5)  

    # plot_rgb_traces(R, G, B, timestamps)
    t_uniform = np.array(timestamps, dtype=float)
    R_signal = np.array(R, dtype=float)
    G_signal = np.array(G, dtype=float)
    B_signal = np.array(B, dtype=float)

    #plot_detrended(R_signal, G_signal, B_signal, time)

    #plot_fft(G_signal, Video.FPS, label="Green channel")
    
    peaks_red, _   = find_peaks(R_signal, prominence=5, distance=30)
    peaks_green, _ = find_peaks(G_signal, prominence=5, distance=30)
    peaks_blue, _  = find_peaks(B_signal, prominence=5, distance=30)

    # Plot with peaks marked
    fig, axes = plt.subplots(3, 1, figsize=(15, 6), sharex=True)

    axes[0].plot(R_signal, 'r')
    axes[0].plot(peaks_red, R_signal[peaks_red], "ko")
    axes[0].set_ylabel("Red")

    axes[1].plot(G_signal, 'g')
    axes[1].plot(peaks_green, G_signal[peaks_green], "ko")
    axes[1].set_ylabel("Green")

    axes[2].plot(B_signal, 'b')
    axes[2].plot(peaks_blue, B_signal[peaks_blue], "ko")
    axes[2].set_ylabel("Blue")
    axes[2].set_xlabel("Time (frames)")

    plt.show()

    detrended = ext.smoothness_priors_detrend(G_signal, lam=10)
    normalized = ext.normalize(detrended)

    normalized = np.asarray(normalized, dtype=float).ravel()
    G_smooth = np.asarray(G_smooth, dtype=float).ravel()
    G_smooth = ext.normalize(G_smooth)
    peaks_raw, _ = find_peaks(G_signal, prominence=5)
    peaks_proc, _ = find_peaks(G_smooth, prominence=5)

    # Plot comparison
    plt.figure(figsize=(12,5))
    plt.plot(normalized, label="Raw", alpha=0.6)
    plt.plot(G_smooth, label="Detrended + Smoothed", linewidth=2)
    plt.plot(peaks_raw, normalized[peaks_raw], "ro", label="Raw Peaks")
    plt.plot(peaks_proc, G_smooth[peaks_proc], "ko", label="Processed Peaks")
    plt.legend()
    plt.show()

    # Quantify improvement
    def snr(trace, peaks):
        baseline = np.delete(trace, peaks)  # exclude peaks
        peak_vals = trace[peaks]
        return np.mean(peak_vals) / np.std(baseline)

    print("SNR Raw:", snr(G_signal, peaks_raw))
    print("SNR Processed:", snr(G_smooth, peaks_proc))


    return np.array(R_signal), np.array(G_signal), np.array(B_signal), np.array(t_uniform)

from scipy.ndimage import median_filter

def hampel_fix(x, fs = Video.FPS, win_sec=0.5, n_sigma=5.0, replace='interp'):

    x = np.asarray(x, dtype=float)
    N = x.size
    k = max(1, int(round(win_sec * fs)))
    size = 2 * k + 1

    # rolling median and rolling MAD
    med = median_filter(x, size=size, mode='reflect')
    abs_dev = np.abs(x - med)
    mad = 1.4826 * median_filter(abs_dev, size=size, mode='reflect') + 1e-12

    outlier = abs_dev > n_sigma * mad

    if replace == 'none':
        return x.copy(), outlier

    x_fixed = x.copy()
    if replace == 'median':
        x_fixed[outlier] = med[outlier]
        return x_fixed, outlier

    # replace == 'interp'
    if outlier.any():
        good = ~outlier
        idx = np.arange(N)
        # np.interp holds the boundary values for leading/trailing gaps
        x_fixed[outlier] = np.interp(idx[outlier], idx[good], x_fixed[good])
    return x_fixed, outlier

def repair_impulses_rgb(R, G, B, win_sec=0.5, n_sigma=5.0):
    Rf, mR = hampel_fix(R,  win_sec, n_sigma, replace='interp')
    Gf, mG = hampel_fix(G,  win_sec, n_sigma, replace='interp')
    Bf, mB = hampel_fix(B,  win_sec, n_sigma, replace='interp')

    bad = mR | mG | mB
    if bad.any():
        idx = np.arange(len(R))
        good = ~bad
        for x in (Rf, Gf, Bf):
            x[bad] = np.interp(idx[bad], idx[good], x[good])
    return Rf, Gf, Bf, bad

def plot_rgb_traces(R, G, B, timestamps):
    fig, axs = plt.subplots(3, 1, figsize=(10, 6), sharex=True)

    axs[0].plot(timestamps, R, color='red')
    axs[0].set_ylabel("Red")

    axs[1].plot(timestamps, G, color='green')
    axs[1].set_ylabel("Green")

    axs[2].plot(timestamps, B, color='blue')
    axs[2].set_ylabel("Blue")
    axs[2].set_xlabel("Time")

    plt.tight_layout()
    plt.show()

    
def rgb_means_with_skin(rgb_frame, min_coverage=0.08, equalize_Y=True):
    """
    Returns (R_mean, G_mean, B_mean, mask) using a skin mask.
    Optionally equalizes the Y channel (illumination normalization) before masking.
    """
    x = rgb_frame

    # 1) Illumination normalization (equalize Y in YCrCb)
    if equalize_Y:
        ycrcb = cv2.cvtColor(x, cv2.COLOR_RGB2YCrCb)
        Y, Cr, Cb = cv2.split(ycrcb)
        Y = cv2.equalizeHist(Y)
        x = cv2.cvtColor(cv2.merge([Y, Cr, Cb]), cv2.COLOR_YCrCb2RGB)

    # 2) Skin mask (on the equalized image)
    mask = quick_skin_mask(x, morph=True)  # your function, returns uint8 mask (0/255)

    # 3) Fail-safe: if too few pixels, fall back to full ROI
    if mask.sum() < min_coverage * mask.size * 255:
        mask = np.ones_like(mask, dtype=np.uint8) * 255

    m = (mask > 0)
    Rm = float(x[..., 0][m].mean())
    Gm = float(x[..., 1][m].mean())
    Bm = float(x[..., 2][m].mean())
    return Rm, Gm, Bm, mask

def quick_skin_mask(img, morph=True):
    # Convert color space depending on input format
    code = cv2.COLOR_RGB2YCrCb
    ycrcb = cv2.cvtColor(img, code)

    # Skin color range in YCrCb
    lower = (0, 133, 77)
    upper = (255, 173, 127)
    mask = cv2.inRange(ycrcb, lower, upper)

    # Optional morphological filtering
    if morph:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return mask

def ensure_uint8_rgb(rgb):
    if rgb.dtype != np.uint8:
        rgb8 = np.clip(rgb, 0, 255).astype(np.uint8)
    else:
        rgb8 = rgb
    return rgb8

def masked_mean_rgb(rgb, mask):
    """
    rgb: (H,W,3) RGB uint8/float
    mask: (H,W) uint8 {0,255} or bool
    Returns (R,G,B) means over mask; None if empty.
    """
    if mask.dtype != np.bool_:
        sel = mask > 0
    else:
        sel = mask
    if not np.any(sel):
        return None
    roi = rgb[sel]
    # optional robustness: drop very dark/bright
    v = roi.max(axis=1)
    keep = (v > 10) & (v < 250)
    if not np.any(keep):
        return None
    roi = roi[keep]
    return float(roi[:, 0].mean()), float(roi[:, 1].mean()), float(roi[:, 2].mean())

def polygon_mask_from_indices(frame_shape, face_landmarks, idx_list):
    """
    Build a binary mask (H,W) for a landmark polygon.
    """
    h, w = frame_shape[:2]
    pts = []
    for i in idx_list:
        if i < len(face_landmarks):
            x = int(np.clip(face_landmarks[i].x * w, 0, w-1))
            y = int(np.clip(face_landmarks[i].y * h, 0, h-1))
            pts.append((x, y))
    if len(pts) < 3:
        return None
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 255)
    return mask

def combine_masks(*masks):
    """
    AND-combine masks that are not None. Returns None if all None.
    """
    valid = [m for m in masks if m is not None]
    if not valid:
        return None
    out = valid[0].copy()
    for m in valid[1:]:
        out = cv2.bitwise_and(out, m)
    return out

   

