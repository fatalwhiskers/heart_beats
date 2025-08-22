import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from src.config import Signal
from src.config import BVP, filePaths
import os

def get_bvp_ground_truth(path_csv):
    # Example: load BVP from CSV
    video_path = os.path.join(filePaths.folder_path, path_csv)

    bvp = np.loadtxt(video_path)  
 
    min_distance = int(BVP.BVP_RATE / Signal.HR_HIGH)
    peaks, details = find_peaks(bvp, distance=min_distance)
    peak_times = peaks / BVP.BVP_RATE
    ibi = np.diff(peak_times)
    hr = 60 / ibi

    print("Average Heart Rate:", np.mean(hr), "BPM")

    plt.figure(figsize=(12,4))
    plt.plot(np.arange(len(bvp))/BVP.BVP_RATE, bvp, label="BVP")
    plt.plot(peak_times, bvp[peaks], 'ro', label="Peaks")
    plt.xlabel("Time (s)")
    plt.ylabel("BVP Amplitude")
    plt.title("BVP Signal with Detected Peaks")
    plt.legend()
    plt.show()

    return hr, peak_times[1:]


