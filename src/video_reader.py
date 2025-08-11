import cv2
import av
import numpy as np
import decord
from decord import VideoReader
from torchvision.io import read_video
from src.config import Video 
from scipy.interpolate import interp1d
import csv
import cv2
import numpy as np
import mediapipe as mp


"""
def interpolate_video_frames(frames, original_fps, target_fps): #memoary problems
    if original_fps == target_fps:
        return frames

    n_frames, h, w, c = frames.shape
    duration = n_frames / original_fps

    t_original = np.linspace(0, duration, n_frames)
    t_target = np.linspace(0, duration, int(duration * target_fps))

    frames_interp = np.empty((len(t_target), h, w, c), dtype=np.float32)

    for ch in range(c):
        for i in range(h):
            for j in range(w):
                pixel_values = frames[:, i, j, ch]
                interp_func = interp1d(t_original, pixel_values, kind='linear', bounds_error=False, fill_value='extrapolate')
                frames_interp[:, i, j, ch] = interp_func(t_target)

    return np.clip(frames_interp, 0, 255).astype(np.uint8)


def interpolate_video_frames(frames, original_fps, target_fps, use_float16=True):
    if original_fps == target_fps:
        return frames

    n_frames, h, w, c = frames.shape
    duration = n_frames / original_fps

    t_original = np.linspace(0, duration, n_frames)
    t_target = np.linspace(0, duration, int(duration * target_fps))

    dtype = np.float16 if use_float16 else np.float32

    frames_interp = np.empty((len(t_target), h, w, c), dtype=dtype)

    for ch in range(c):
        print(f"Interpolating channel {ch + 1}/{c}...")
        for i in range(h):
            pixel_row = frames[:, i, :, ch]  # shape: (n_frames, w)

            for j in range(w):
                pixel_values = pixel_row[:, j]  # shape: (n_frames,)
                interp_func = interp1d(t_original, pixel_values, kind='linear', bounds_error=False, fill_value='extrapolate')
                frames_interp[:, i, j, ch] = interp_func(t_target)

    # Clip and cast to uint8 for final video format
    return np.clip(frames_interp, 0, 255).astype(np.uint8)

def interpolate_video_frames_chunked(frames, original_fps, target_fps, chunk_size=100, use_float16=False):
    if original_fps == target_fps:
        return frames

    n_frames, h, w, c = frames.shape
    duration = n_frames / original_fps

    t_original = np.linspace(0, duration, n_frames)
    t_target_full = np.linspace(0, duration, int(duration * target_fps))

    dtype = np.float16 if use_float16 else np.float32
    output_chunks = []

    print(f"Interpolating in chunks of {chunk_size} frames...")

    # Interpolation setup for each pixel position and channel
    for start in range(0, len(t_target_full), chunk_size):
        end = min(start + chunk_size, len(t_target_full))
        t_target_chunk = t_target_full[start:end]
        chunk_interp = np.empty((len(t_target_chunk), h, w, c), dtype=dtype)

        for ch in range(c):
            for i in range(h):
                for j in range(w):
                    pixel_values = frames[:, i, j, ch]
                    interp_func = interp1d(t_original, pixel_values, kind='linear', bounds_error=False, fill_value='extrapolate')
                    chunk_interp[:, i, j, ch] = interp_func(t_target_chunk)

        output_chunks.append(chunk_interp)
        print(f"  → Chunk {start}-{end} done")

    # Concatenate all chunks into one big array
    final_result = np.concatenate(output_chunks, axis=0)
    return np.clip(final_result, 0, 255).astype(np.uint8)
"""
#loads a video into an array in cv2 bgr
#video path = string to location (must be escaperd)
#resize is optional not reallt important here
#189s
def read_video_to_array(video_path, x1=0, y1=0, x2=0, y2=0, crop=True, display=False, testing=False):

    cap = cv2.VideoCapture(video_path)
    #Video.fps = cap.get(cv2.CAP_PROP_FPS) reads the files fps but is wrong

    if not cap.isOpened():
        print(f"Error: Could not open video at {video_path}")
        return None

    if display:
        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

    frames = []
    timestamps = []  
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        timestamps.append(timestamp)

        # Optional crop:
        if crop:
          frame = frame[y1:y2, x1:x2]
          #frame = crop_frame_percent(frame, 0.55, 0.5)

        frames.append(frame)
        frame_count += 1

        if testing and frame_count >= 30:
            break

        if display:
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if display:
        cv2.destroyAllWindows()

    frames = np.array(frames)

    return frames, timestamps

#210s - 160 with cuda
#same as before but using a supposdly faster libary changed to bgr so I can swap for testing in code
def read_video_with_pyav(video_path):
    container = av.open(video_path ,options={'hwaccel': 'cuda'})
    frames = []

    for frame in container.decode(video=0):

        img = frame.to_ndarray(format='bgr24')  # bgr to keep BGR like OpenCV
        frames.append(img)

    return np.array(frames)  # (num_frames, height, width, channels)

# ran in 177s
def read_video_decord(path):
   # vr = VideoReader(path)
    vr = VideoReader(path)  # GPU decoding
    frames = vr[:]
    return np.stack(frames)

#206.232 seconds
def read_video_torchvision(video_path, resize=None, max_frames=None, to_bgr=False):
    video, _, info = read_video(video_path, pts_unit='sec')

    # video: (T, H, W, C) in RGB
    if max_frames:
        video = video[:max_frames]

    frames = video.numpy()  # convert to NumPy

    if resize:
        import cv2
        frames = [cv2.resize(f, resize) for f in frames]
        frames = np.stack(frames)

    if to_bgr:
        frames = frames[..., ::-1]  # RGB to BGR

    return frames, info

def crop_frame_percent(frame, percent_h, percent_w):

    h, w = frame.shape[:2]
    
    crop_h = int(h * percent_h)
    crop_w = int(w * percent_w)

    start_y = (h - crop_h) // 2
    start_x = (w - crop_w) // 2

    return frame[start_y:start_y + crop_h, start_x:start_x + crop_w]

def crop_frame(frame, crop_h, crop_w, position='center'):
    """
    - position (str): One of 'center', 'top-left', 'top-right', 'bottom-left', 'bottom-right'.
    """
    h, w, c = frame.shape

    if crop_h > h or crop_w > w:
        raise ValueError("Crop size exceeds frame dimensions.")

    if position == 'center':
        start_y = (h - crop_h) // 2
        start_x = (w - crop_w) // 2
    elif position == 'top-left':
        start_y = 0
        start_x = 0
    elif position == 'top-right':
        start_y = 0
        start_x = w - crop_w
    elif position == 'bottom-left':
        start_y = h - crop_h
        start_x = 0
    elif position == 'bottom-right':
        start_y = h - crop_h
        start_x = w - crop_w
    else:
        raise ValueError(f"Invalid crop position: {position}")

    return frame[start_y:start_y+crop_h, start_x:start_x+crop_w]

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
                continue  # skip malformed rows
    return crop_settings


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

def get_mesh_forehead(frame, face_landmarks):
    h, w, _ = frame.shape

    # Using approximate forehead landmarks got off internet seems to work
    FOREHEAD_LANDMARKS = [10, 338, 297, 332, 284, 251, 389, 356]  # tweakable

    points = [(int(face_landmarks.landmark[i].x * w),
               int(face_landmarks.landmark[i].y * h)) for i in FOREHEAD_LANDMARKS]

    xs, ys = zip(*points)
    pad = 10  # padding around forehead box
    x1 = max(0, min(xs) - pad)
    x2 = min(w, max(xs) + pad)
    y1 = max(0, min(ys) - pad)
    y2 = min(h, max(ys) + pad)

    return x1, y1, x2, y2

def read_video_to_array_v2(video_path, x1=0, y1=0, x2=0, y2=0, crop=True,
    crop_mode='manual',  # 'manual', 'face_track', 'bbox_forehead', 'mesh_forehead', 'none'
    display=False,
    testing=False
):
    mp_face_detection = mp.solutions.face_detection
    mp_face_mesh = mp.solutions.face_mesh

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video at {video_path}")
        return None, None

    if display:
        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

    frames = []
    timestamps = []
    frame_count = 0

    tracker = None
    init_tracker = False

    # so I rember what its do
    # Initialize MediaPipe face detection and face mesh models:
    # - FaceDetection is used to get the bounding box of the face.
    # - FaceMesh provides 468 detailed facial landmarks (eyes, nose, mouth, forehead, etc.).
    # - model_selection=0: Optimized for short-range (e.g., webcam) set to 1 for distance
    # - min_detection_confidence=0.7: Only use detections with at least 70% confidence. can be changed for differant scenarios
    # - static_image_mode=False: Enables faster tracking across video frames. since its a video its not static 
    # - max_num_faces=1: Only detect one face per frame (for efficiency). since we only have one face this will need to be changed for more
    # - refine_landmarks=True: Enables more accurate eye and mouth landmarks. this increase processing but better quality. if its live ran may need to change
    # don't remove the slash its wired python nonsense since they don't use semicolons (means carry onto next line)
    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.7) as face_detector, \
         mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.7) as face_mesh:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            timestamps.append(timestamp)

            h, w, _ = frame.shape
            crop_frame = frame  # default to full frame

            if crop:
                # 1. Manual cropping using fixed pixel coordinates
                #    Uses x1, y1, x2, y2 given by the user
                if crop_mode == 'manual':
                    crop_frame = frame[y1:y2, x1:x2]
                # 2. No cropping at all — keep full frame
                elif crop_mode == 'none':
                    crop_frame = frame
                # 3. Face tracking using bounding box from first frame
                #    - First, run face detection to locate the face
                #    - Initialize OpenCV KCF tracker
                #    - On later frames, update tracker instead of re-detecting
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
                        else:
                            continue  # skip frame

                    else:
                        success, box = tracker.update(frame)
                        if success:
                            x, y, width, height = map(int, box)
                            x1_c = max(0, x)
                            y1_c = max(0, y)
                            x2_c = min(w, x + width)
                            y2_c = min(h, y + height)
                            crop_frame = frame[y1_c:y2_c, x1_c:x2_c]
                        else:
                            continue  # skip frame
               
                 # 4. Forehead region from face detection bounding box
                    #    - Detect face with MediaPipe
                    #    - Use a helper function (get_bbox_forehead) to
                    #      extract only the forehead area from detection box                
                elif crop_mode == 'bbox_forehead':
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = face_detector.process(rgb)
                    if results.detections:
                        x1_f, y1_f, x2_f, y2_f = get_bbox_forehead(frame, results.detections[0])
                        crop_frame = frame[y1_f:y2_f, x1_f:x2_f]
                    else:
                        continue

            # 5. Forehead region using facial landmarks (Face Mesh)
            #    - Detect 468 face landmarks with MediaPipe
            #    - Use a helper function (get_mesh_forehead) to
            #      extract forehead region precisely from landmarks
                elif crop_mode == 'mesh_forehead':
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = face_mesh.process(rgb)
                    if results.multi_face_landmarks:
                        x1_f, y1_f, x2_f, y2_f = get_mesh_forehead(frame, results.multi_face_landmarks[0])
                        crop_frame = frame[y1_f:y2_f, x1_f:x2_f]
                    else:
                        continue

                else:
                    raise ValueError(f"Invalid crop_mode: {crop_mode}")

            frames.append(crop_frame)
            frame_count += 1

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

# added a fixed limit for cropping so that a univesal array can be stored.
def read_video_to_array_v3(video_path, x1=0, y1=0, x2=0, y2=0, crop=True,
    crop_mode='manual',  # 'manual', 'face_track', 'bbox_forehead', 'mesh_forehead', 'none'
    display=False,
    testing=False,
    target_size=(128, 128)  # fixed output size for all frames
):
    """
    Reads a video, crops around the face/forehead region based on the chosen mode,
    and returns the cropped frames + timestamps.

    crop_mode options:
    - 'manual'        : Use fixed pixel coordinates (x1, y1, x2, y2).
    - 'none'          : Keep the full frame, no cropping.
    - 'face_track'    : Detect face once, track across frames with KCF tracker.
    - 'bbox_forehead' : Detect face bounding box each frame, crop only forehead region.
    - 'mesh_forehead' : Detect facial landmarks, crop forehead region precisely.
    """

    import cv2
    import numpy as np
    mp_face_detection = mp.solutions.face_detection
    mp_face_mesh = mp.solutions.face_mesh

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video at {video_path}")
        return None, None

    if display:
        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

    frames = []
    timestamps = []
    frame_count = 0
    skipped_count = 0  # count how many frames were skipped

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

            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # seconds
            h, w, _ = frame.shape
            crop_frame = frame  # default to full frame
            success_crop = True  # track if we successfully cropped a face

            if crop:
                # 1. Manual cropping using fixed pixel coordinates
                if crop_mode == 'manual':
                    x1_c = max(0, min(x1, w))
                    y1_c = max(0, min(y1, h))
                    x2_c = max(0, min(x2, w))
                    y2_c = max(0, min(y2, h))
                    if x2_c <= x1_c or y2_c <= y1_c:
                        success_crop = False
                    else:
                        crop_frame = frame[y1_c:y2_c, x1_c:x2_c]

                # 2. No cropping at all — keep full frame
                elif crop_mode == 'none':
                    crop_frame = frame

                # 3. Face tracking using bounding box from first frame
                #    - First, run face detection to locate the face
                #    - Initialize OpenCV KCF tracker
                #    - On later frames, update tracker instead of re-detecting
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
                            crop_frame = frame[max(0, y):min(h, y + height), max(0, x):min(w, x + width)]
                        else:
                            success_crop = False
                    else:
                        success, box = tracker.update(frame)
                        if success:
                            x, y, width, height = map(int, box)
                            crop_frame = frame[max(0, y):min(h, y + height), max(0, x):min(w, x + width)]
                        else:
                            success_crop = False

                # 4. Forehead region from face detection bounding box
                #    - Detect face with MediaPipe
                #    - Use get_bbox_forehead() to extract only forehead area
                elif crop_mode == 'bbox_forehead':
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = face_detector.process(rgb)
                    if results.detections:
                        x1_f, y1_f, x2_f, y2_f = get_bbox_forehead(frame, results.detections[0])
                        crop_frame = frame[y1_f:y2_f, x1_f:x2_f]
                    else:
                        success_crop = False

                # 5. Forehead region using facial landmarks (Face Mesh)
                #    - Detect 468 face landmarks with MediaPipe
                #    - Use get_mesh_forehead() to extract forehead region precisely
                elif crop_mode == 'mesh_forehead':
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = face_mesh.process(rgb)
                    if results.multi_face_landmarks:
                        x1_f, y1_f, x2_f, y2_f = get_mesh_forehead(frame, results.multi_face_landmarks[0])
                        crop_frame = frame[y1_f:y2_f, x1_f:x2_f]
                    else:
                        success_crop = False

                else:
                    raise ValueError(f"Invalid crop_mode: {crop_mode}")

            if success_crop:
                # Resize to fixed size for consistency
                crop_frame = cv2.resize(crop_frame, target_size, interpolation=cv2.INTER_AREA)
                frames.append(crop_frame)
                timestamps.append(timestamp)
                frame_count += 1
            else:
                skipped_count += 1

            # Limit to first 30 frames if testing
            if testing and frame_count >= 30:
                break

            if display and success_crop:
                cv2.imshow('frame', crop_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    cap.release()
    if display:
        cv2.destroyAllWindows()

    # Log skipped frames for quality control
    print(f"Total frames processed: {frame_count}, Skipped frames: {skipped_count} "
          f"({(skipped_count / (frame_count + skipped_count) * 100):.2f}% skipped)")

    frames = np.array(frames)
    return frames, timestamps
