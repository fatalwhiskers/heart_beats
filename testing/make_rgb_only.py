import cv2
import os

input_path = r"data\Dataset2\Videos\2ea4\2ea4_Baseline.mp4"
output_dir = r"data\fakedata"

os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise FileNotFoundError(f"Could not open input video: {input_path}")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out_red   = cv2.VideoWriter(os.path.join(output_dir, "red_only.mp4"),   fourcc, fps, (width, height))
out_green = cv2.VideoWriter(os.path.join(output_dir, "green_only.mp4"), fourcc, fps, (width, height))
out_blue  = cv2.VideoWriter(os.path.join(output_dir, "blue_only.mp4"),  fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Split channels
    red_frame   = frame.copy()
    green_frame = frame.copy()
    blue_frame  = frame.copy()

    red_frame[:, :, 0] = 0   # remove blue
    red_frame[:, :, 1] = 0   # remove green
    
    green_frame[:, :, 0] = 0 # remove blue
    green_frame[:, :, 2] = 0 # remove red
    
    blue_frame[:, :, 1] = 0  # remove green
    blue_frame[:, :, 2] = 0  # remove red

    # Write frames
    out_red.write(red_frame)
    out_green.write(green_frame)
    out_blue.write(blue_frame)

cap.release()
out_red.release()
out_green.release()
out_blue.release()

print(f"   {os.path.join(output_dir, 'red_only.mp4')}")
print(f"   {os.path.join(output_dir, 'green_only.mp4')}")
print(f"   {os.path.join(output_dir, 'blue_only.mp4')}")
