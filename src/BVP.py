import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from src.config import Signal

def get_bvp_ground_truth(path_csv, fps= 64):
    # Example: load BVP from CSV
    bvp = np.loadtxt(path_csv)  
 
    min_distance = int(fps / Signal.HR_HIGH)  
    # Step 1: Detect peaks
    peaks, _ = find_peaks(bvp, distance=fps * min_distance)  # minimum distance  between beats

    # Step 2: Compute inter-beat intervals (IBI)
    peak_times = peaks / fps  # convert sample index to time in seconds
    ibi = np.diff(peak_times)  # intervals between successive peaks

    # Step 3: Compute heart rate
    hr = 60 / ibi  # beats per minute

    # Step 4: Print average heart rate
    print("Average Heart Rate:", np.mean(hr), "BPM")

    return hr

"""
    plt.figure(figsize=(12,4))
    plt.plot(np.arange(len(bvp))/fs, bvp, label="BVP")
    plt.plot(peak_times, bvp[peaks], 'ro', label="Peaks")
    plt.xlabel("Time (s)")
    plt.ylabel("BVP Amplitude")
    plt.title("BVP Signal with Detected Peaks")
    plt.legend()
    plt.show()
  
"""