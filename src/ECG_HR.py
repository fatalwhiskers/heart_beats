import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from src.config import Signal  # uses Signal.HR_HIGH (Hz) for min peak spacing

def ecg_hr_from_signal_500hz(ecg):
    """
    Convert raw ECG (uniformly sampled at 500 Hz) to instantaneous HR (BPM).
    Returns:
      t_mid   : times (s) where HR is defined (midpoints between R-peaks)
      hr_bpm  : instantaneous HR at t_mid
      r_peaks : sample indices of detected R-peaks
      rr_s    : RR intervals in seconds
    """
    fs = 500.0
    ecg = np.asarray(ecg, float).ravel()
    t = np.arange(ecg.size) / fs

    # QRS band-pass (robust default for fs≥100 Hz)
    b, a = butter(4, [5, 20], btype="band", fs=fs)   # you can tweak to [8,20] if needed
    x = filtfilt(b, a, ecg)

    # Peak picking (refractory based on max plausible HR from your config)
    distance = int(fs / Signal.HR_HIGH)              # e.g., HR_HIGH=4 Hz → min 125 samples
    lo, hi = np.percentile(x, [5, 95])
    prom = 0.3 * (hi - lo)                           # adjust if too many/too few peaks
    r_peaks, _ = find_peaks(x, distance=distance, prominence=prom)

    if r_peaks.size < 2:
        return np.array([]), np.array([]), r_peaks, np.array([])

    t_peaks = t[r_peaks]
    rr_s = np.diff(t_peaks)
    t_mid = 0.5 * (t_peaks[1:] + t_peaks[:-1])
    hr_bpm = 60.0 / rr_s
    return t_mid, hr_bpm, r_peaks, rr_s