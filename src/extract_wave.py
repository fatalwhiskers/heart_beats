import numpy as np
from .config import Signal
from scipy.signal import butter, filtfilt, detrend

def extract_rgb_signals_BGR(frames):
    B = frames[:, :, :, 0].mean(axis=(1, 2))
    G = frames[:, :, :, 1].mean(axis=(1, 2))
    R = frames[:, :, :, 2].mean(axis=(1, 2))
    
    return R, G, B

#since doing pca later anyway "standardization"
def zscore_normalize(signal):
    return (signal - np.mean(signal)) / (np.std(signal))

def bandpass_filter(signal, fs, lowcut=Signal.HR_LOW, highcut=Signal.HR_HIGH, order=Signal.HR_ORDER):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def get_heart_rate(signal, fs, low=0.7, high=4.0):
    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=1/fs)
    fft_values = np.abs(np.fft.rfft(signal))

    # Limit to heart rate range (in Hz)
    valid = (freqs >= low) & (freqs <= high)
    freqs = freqs[valid]
    fft_values = fft_values[valid]

    # Find peak frequency
    peak_freq = freqs[np.argmax(fft_values)]
    bpm = peak_freq * 60  # Convert Hz to BPM
    return bpm

def detrend_sig(signal):
    return detrend(signal)

def save_rgb_signals(file_path, R, G, B):
    np.savez(file_path, R=R, G=G, B=B)

def load_rgb_signals(file_path):
    data = np.load(file_path)
    return data['R'], data['G'], data['B']