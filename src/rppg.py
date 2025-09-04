import numpy as np
from src.config import Video 
from src.config import rppg 
from scipy.signal import welch, butter, filtfilt, detrend

def sliding_fft_hr(rppg_signal):
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

def sliding_fft_hr_center(rppg_signal):

    signal = (rppg_signal - np.mean(rppg_signal)) / np.std(rppg_signal)

    n = len(signal)
    win_len = int(rppg.window_size * Video.FPS)
    step_len = int(rppg.step_size * Video.FPS)
    hann_win = np.hanning(win_len)

    # Precompute frequency grid and HR band mask once
    freqs = np.fft.rfftfreq(win_len, d=1 / Video.FPS)
    mask = (freqs >= 0.7) & (freqs <= 4.0)
    freqs_band = freqs[mask]

    hr_values, times = [], []
    psd_rows = []

    # Inclusive last window
    for start in range(0, n - win_len + 1, step_len):
        segment = signal[start:start + win_len] * hann_win

        fft_vals = np.fft.rfft(segment)
        power = np.abs(fft_vals) ** 2

        # Restrict to HR band and normalize per-window
        power_band = power[mask]
        power_band = power_band / np.max(power_band)

        # Power-weighted frequency → BPM
        freq_est = np.sum(freqs_band * power_band) / np.sum(power_band)
        bpm = freq_est * 60.0

        psd_rows.append(power_band)
        hr_values.append(bpm)

        # Use window center time
        times.append((start + win_len / 2) / Video.FPS)

    psd_accum = np.vstack(psd_rows)
    return np.array(times), np.array(hr_values), psd_accum, freqs_band

def sliding_welch_hr_center(rppg_signal):
    # Standardize signal
    signal = (rppg_signal - np.mean(rppg_signal)) / (np.std(rppg_signal) + 1e-12)

    n = len(signal)
    win_len = int(rppg.window_size * Video.FPS)
    step_len = int(rppg.step_size * Video.FPS)

    # Precompute HR band once
    hr_lo, hr_hi = 0.7, 4.0  # Hz

    hr_values, times = [], []
    psd_rows = []
    freqs_band_ref = None

    # Inclusive last window
    for start in range(0, n - win_len + 1, step_len):
        segment = signal[start:start + win_len]
        """
        # Welch PSD for this segment
        freqs, psd = welch(
            segment,
            fs=Video.FPS,
            window='hann',
            nperseg=min(256, win_len),  # adjust if needed
            noverlap=None,
            detrend='constant',
            scaling='density'
        )"""

        freqs, psd = welch(
            segment,
            fs=Video.FPS,
            window='hamming',
            nperseg=win_len,
            noverlap=win_len // 2,   # <-- paper setting
            nfft=4096,
            detrend=False,
            scaling='density'
        )

        # Restrict to HR band
        mask = (freqs >= hr_lo) & (freqs <= hr_hi)
        freqs_band = freqs[mask]
        power_band = psd[mask]

        # Normalize within this band
        if power_band.size > 0:
            power_band = power_band / np.max(power_band)

            # Power-weighted frequency centroid
            freq_est = np.sum(freqs_band * power_band) / np.sum(power_band)
            bpm = freq_est * 60.0
        else:
            power_band = np.array([])
            bpm = np.nan

        psd_rows.append(power_band)
        hr_values.append(bpm)

        # Use window center time
        times.append((start + win_len / 2) / Video.FPS)

        if freqs_band_ref is None:
            freqs_band_ref = freqs_band

    psd_accum = np.vstack(psd_rows) if psd_rows else np.zeros((0, len(freqs_band_ref) if freqs_band_ref is not None else 0))

    return np.array(times), np.array(hr_values), psd_accum, freqs_band_ref

#https://pmc.ncbi.nlm.nih.gov/articles/PMC10770840/

def sliding_welch(rppg_signal):
    
    x = np.asarray(rppg_signal, dtype=float)

    # --- Preprocessing: detrend then 6th-order Butterworth band-pass 0.65–4.0 Hz ---
    x = detrend(x, type="linear")
    b, a = butter(6, [0.65, 4.0], btype="band", fs=Video.FPS)
    xf = filtfilt(b, a, x)

    # --- Sliding windows: 10 s, no overlap ---
    win_len = int(round(10 * Video.FPS))
    step_len = win_len  # no overlap

    # Welch config: use the full 10-s window as one Welch segment (nperseg=win_len)
    # This yields a Hann-windowed periodogram equivalent but via welch().
    nperseg = win_len
    noverlap = 0

    # Frequency grid for this nperseg
    freqs = np.fft.rfftfreq(nperseg, 1 / Video.FPS)
    # HR band: 39–240 BPM  -> 0.65–4.0 Hz
    band_mask = (freqs >= 39.0 / 60.0) & (freqs <= 240.0 / 60.0)
    freqs_band = freqs[band_mask]

    times, hr_values, rows = [], [], []

    # Inclusive last window if exact multiple; otherwise drop partial tail (no overlap per spec)
    for start in range(0, len(xf) - win_len + 1, step_len):
        seg = xf[start:start + win_len]

        # Welch PSD on the 10-s segment
        f, pxx = welch(
            seg, fs=Video.FPS, window="hann",
            nperseg=nperseg, noverlap=noverlap,
            detrend=False, scaling="density", return_onesided=True
        )

        pband = pxx[band_mask]
        # Normalize for comparability/visualization
        pnorm = pband / np.max(pband)

        # HR = peak within 39–240 BPM
        peak_idx = np.argmax(pband)
        bpm = freqs_band[peak_idx] * 60.0

        rows.append(pnorm)
        hr_values.append(bpm)
        times.append((start + win_len / 2) / Video.FPS)  # window center time

    psd_accum = np.vstack(rows) 
    return np.array(times), np.array(hr_values), psd_accum, freqs_band

