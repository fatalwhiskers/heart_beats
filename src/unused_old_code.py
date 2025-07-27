def read_video_to_array(video_path, display=False, testing=False):
    cap = cv2.VideoCapture(video_path)
    Video.fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    
    if not cap.isOpened():
        print(f"Error: Could not open video at {video_path}")
        return None

    # testing video can be removed but helpful for debugin
    if display:
        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)


    frames = []

    while cap.isOpened():
        ret, frame = cap.read() # reeads frames 
        if not ret:
            break
        #frame = crop_frame(frame, Video.ROI_HEIGHT, Video.ROI_WIDTH)    
        #frame = crop_frame_percent(frame, 0.55, 0.5)
        frames.append(frame) # adds them to an array
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

    return np.array(frames)  # Shape: (num_frames, height, width, channels) (Blue Green Red)