import cv2
import av
import numpy as np
import decord
from decord import VideoReader
from torchvision.io import read_video
from .config import Video 
from scipy.interpolate import interp1d

def interpolate_video_frames(frames, original_fps, target_fps):
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

#loads a video into an array in cv2 bgr
#video path = string to location (must be escaperd)
#resize is optional not reallt important here
#189s
def read_video_to_array(video_path, Interpolate=True, display=False, testing=False):
    cap = cv2.VideoCapture(video_path)
    Video.fps = cap.get(cv2.CAP_PROP_FPS)

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
        # frame = crop_frame(frame, Video.ROI_HEIGHT, Video.ROI_WIDTH)
        # frame = crop_frame_percent(frame, 0.55, 0.5)

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

    if Interpolate:
        frames = interpolate_video_frames(frames, Video.fps, Video.target_FPS)

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