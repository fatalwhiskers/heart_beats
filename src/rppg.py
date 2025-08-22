import numpy as np
from src.config import Video 
from src.config import rppg 

def sliding_fft_hr(rppg_signal, label):
    # Normalize
    signal = (rppg_signal - np.mean(rppg_signal)) / np.std(rppg_signal)
    
    n = len(signal)
    win_len = int(rppg.window_size * Video.FPS)
    step_len = int(rppg.step_size * Video.FPS)
    hann_win = np.hanning(win_len)
    
    hr_values, times = [], [] 
    psd_accum = []

    for start in range(0, n - win_len, step_len):
        segment = signal[start:start + win_len]
        segment = segment * hann_win

        # FFT
        fft_vals = np.fft.rfft(segment)
        freqs = np.fft.rfftfreq(win_len, d=1/Video.FPS)
        power = np.abs(fft_vals) ** 2

        # Restrict to HR band
        mask = (freqs >= 0.7) & (freqs <= 4.0)
        freqs_band, power_band = freqs[mask], power[mask]

        # Normalize spectrum for averaging
        power_band = power_band / np.max(power_band)
        psd_accum.append(power_band)

        # Power-weighted frequency (robust HR)
        freq_est = np.sum(freqs_band * power_band) / np.sum(power_band)
        bpm = freq_est * 60

        hr_values.append(bpm)
        times.append(start / Video.FPS)
    
    return np.array(times), np.array(hr_values), np.array(psd_accum), freqs_band