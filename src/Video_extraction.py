import cv2
import csv
import numpy as np
import mediapipe as mp
import os

# -------------------------------
# CSV crop settings loader
# -------------------------------
def load_crop_settings(csv_path):
    crop_settings = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                filename = row['filename'].strip()
                x1 = int(row['x1'])
                y1 = int(row['y1'])
                x2 = int(row['x2'])
                y2 = int(row['y2'])
                crop_settings.append((filename, x1, y1, x2, y2))
            except (KeyError, ValueError) as e:
                print(f"Skipping row due to error: {e}")
                continue
    return crop_settings


# -------------------------------
# Forehead detection helpers
# -------------------------------
def get_bbox_forehead(frame, detection):
    h, w, _ = frame.shape
    bbox = detection.location_data.relative_bounding_box
    x = int(bbox.xmin * w)
    y = int(bbox.ymin * h)
    width = int(bbox.width * w)
    height = int(bbox.height * h)

    # Forehead: top 25% of the face bbox
    fh_y1 = max(0, y)
    fh_y2 = max(0, y + int(0.25 * height))
    fh_x1 = max(0, x)
    fh_x2 = min(w, x + width)

    return fh_x1, fh_y1, fh_x2, fh_y2

# -------------------------------
# Debug Mesh helper
# -------------------------------
def get_mesh_forehead(frame, face_landmarks):
    h, w, _ = frame.shape

    # Approximate forehead landmarks
    FOREHEAD_LANDMARKS = [10, 338, 297, 332, 284, 251, 389, 356]

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


# -------------------------------
# Main function
# -------------------------------
def read_video_to_array_v2(video_path, x1=0, y1=0, x2=0, y2=0, crop=True,
    crop_mode='manual', display=False, testing=False, test_output_dir="test_frames"
):
    mp_face_detection = mp.solutions.face_detection
    mp_face_mesh = mp.solutions.face_mesh

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video at {video_path}")
        return None, None

    if display:
        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

    if testing:
        os.makedirs(test_output_dir, exist_ok=True)

    frames = []
    timestamps = []
    frame_count = 0

    tracker = None
    init_tracker = False

    # Initialize MediaPipe face detection and face mesh models:
    # - FaceDetection: used to get the bounding box of the face.
    # - FaceMesh: provides 468 detailed facial landmarks (eyes, nose, mouth, forehead, etc.).
    # - model_selection=0: Optimized for short-range (e.g., webcam). Use 1 for long-range.
    # - min_detection_confidence=0.7: Only use detections with at least 70% confidence.
    # - static_image_mode=False: Faster tracking for videos (not re-detecting every frame).
    # - max_num_faces=1: Only detect one face per frame (for efficiency).
    # - refine_landmarks=True: More accurate eye and mouth landmarks (slower, but higher quality).
    # Don't remove the backslash in the 'with' — Python uses it to continue the statement.

    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.7) as face_detector, \
         mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.7) as face_mesh:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            timestamps.append(timestamp)

            h, w, _ = frame.shape
            crop_frame = frame
            crop_coords = None
            landmarks = None

            if crop:
                if crop_mode == 'manual':
                    crop_frame = frame[y1:y2, x1:x2]
                    crop_coords = (x1, y1, x2, y2)

                elif crop_mode == 'none':
                    crop_frame = frame

                elif crop_mode == 'face_track':
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    if not init_tracker:
                        results = face_detector.process(rgb)
                        if results.detections:
                            bbox = results.detections[0].location_data.relative_bounding_box
                            x = int(bbox.xmin * w)
                            y = int(bbox.ymin * h)
                            width = int(bbox.width * w)
                            height = int(bbox.height * h)
                            tracker = cv2.TrackerKCF_create()
                            tracker.init(frame, (x, y, width, height))
                            init_tracker = True
                            x1_c = max(0, x)
                            y1_c = max(0, y)
                            x2_c = min(w, x + width)
                            y2_c = min(h, y + height)
                            crop_frame = frame[y1_c:y2_c, x1_c:x2_c]
                            crop_coords = (x1_c, y1_c, x2_c, y2_c)
                        else:
                            continue
                    else:
                        success, box = tracker.update(frame)
                        if success:
                            x, y, width, height = map(int, box)
                            x1_c = max(0, x)
                            y1_c = max(0, y)
                            x2_c = min(w, x + width)
                            y2_c = min(h, y + height)
                            crop_frame = frame[y1_c:y2_c, x1_c:x2_c]
                            crop_coords = (x1_c, y1_c, x2_c, y2_c)
                        else:
                            continue

                elif crop_mode == 'bbox_forehead':
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = face_detector.process(rgb)
                    if results.detections:
                        x1_f, y1_f, x2_f, y2_f = get_bbox_forehead(frame, results.detections[0])
                        crop_frame = frame[y1_f:y2_f, x1_f:x2_f]
                        crop_coords = (x1_f, y1_f, x2_f, y2_f)
                    else:
                        continue

                elif crop_mode == 'mesh_forehead':
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = face_mesh.process(rgb)
                    if results.multi_face_landmarks:
                        x1_f, y1_f, x2_f, y2_f = get_mesh_forehead(frame, results.multi_face_landmarks[0])
                        crop_frame = frame[y1_f:y2_f, x1_f:x2_f]
                        crop_coords = (x1_f, y1_f, x2_f, y2_f)
                        landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in results.multi_face_landmarks[0].landmark]
                    else:
                        continue

                else:
                    raise ValueError(f"Invalid crop_mode: {crop_mode}")

            frames.append(crop_frame)
            frame_count += 1

            if testing and frame_count <= 30:
                debug_frame = draw_debug_overlay(frame, crop_mode, crop_coords, landmarks)
                save_debug = os.path.join(test_output_dir, f"frame_{frame_count:03d}_debug.jpg")
                cv2.imwrite(save_debug, debug_frame)

            if testing and frame_count >= 30:
                break

            if display:
                cv2.imshow('frame', crop_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    cap.release()
    if display:
        cv2.destroyAllWindows()

    frames = np.array(frames)
    return frames, timestamps
