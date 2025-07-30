import cv2
import av
import numpy as np
import decord
from decord import VideoReader
from torchvision.io import read_video
from src.config import Video 
from scipy.interpolate import interp1d
import csv

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
"""

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
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

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

    return frames

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