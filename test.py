import src.video_reader as vr
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2

def test_space():
    video_path = r"data\Dataset1\vid_s28_T1.avi"
    start = time.time()
    video_array_opencv = vr.read_video_torchvision(video_path)
    end = time.time()
    print(f"OpenCV reading took {end - start:.3f} seconds")


def render_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    plt.imshow(frame_rgb)
    plt.title("Cropped Frame Preview")
    plt.axis('off')  # Hide axes
    plt.show()

def test_Video_Reader():
    video_path = r"data\Dataset1\vid_s28_T1.avi"

    start = time.time()
    video_array_opencv = vr.read_video_to_array(video_path, display=False)
    end = time.time()
    print(f"OpenCV reading took {end - start:.3f} seconds")

    start = time.time()
    video_array_pyav = vr.read_video_decord(video_path)
    end = time.time()
    print(f"PyAV reading took {end - start:.3f} seconds")

    start = time.time()
    video_array_pyav = vr.read_video_with_pyav(video_path)
    end = time.time()
    print(f"PyAV reading took {end - start:.3f} seconds")

def test_fft():
    dt = 0.001
    time = np.arange(0, 1, dt)
    freq = np.sin(2 * np.pi * 45 * time) + np.sin(2 * np.pi * 145 * time)
    org_freq = freq.copy()
    freq += 3.0 * np.random.randn(len(time))

    n = len(time)
    fhat = np.fft.fft(freq, n) #the fourier transform vectors 
    power_spec = fhat * np.conj(fhat) / n # magnitude of frequnceies units of hz/power
    freq_axis = (1/(dt*n)) * np.arange(n) # vector of frequencies
    L = np.arange(1,np.floor(n/2),dtype='int') # only positive
  
    indices = power_spec > 100
    power_spec_high = power_spec * indices
    fhat_clean = indices * fhat
    ffiilt = np.fft.ifft(fhat_clean)

    fig, axs = plt.subplots(3,1)

    plt.sca(axs[0])
    plt.plot(time, freq, color='c', linewidth=1.5, label='noisy')
    plt.plot(time, org_freq, color='k', linewidth=1.5, label='original')
    plt.xlim(time[0],time[-1])
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()

    plt.sca(axs[1])
    plt.plot(time, ffiilt, color='k', linewidth=1.5, label='filter')
    plt.xlim(time[0],time[-1])
    plt.legend()

    plt.sca(axs[2])
    plt.plot(freq_axis[L], power_spec[L], color='c', linewidth=1.5, label='noisy')
    plt.plot(freq_axis[L], power_spec_high[L], color='k', linewidth=1.5, label='filter')
    plt.xlim(freq_axis[L[0]],freq_axis[L[-1]])
    plt.legend()

    plt.show()