import numpy as np
from src.config import Video
from src.config import rppg, Signal, PRV
from scipy.signal import welch, butter, filtfilt, detrend
import numpy as np
from scipy import signal, interpolate
import neurokit2 as nk
import numpy as np

def sliding_fft_hr(rppg_signal, timestamps):
    fs_target = PRV.FPS_RESAMPLE_RATE
    hr_low_hz = Signal.HR_LOW
    hr_high_hz = Signal.HR_HIGH
    window_size = rppg.window_size
    step_size = rppg.step_size

    dt_median = np.median(np.diff(timestamps))
    fs_input = 1.0 / dt_median
    signal_resampled = nk.signal_resample(
        signal,
        sampling_rate=fs_input,
        desired_sampling_rate=fs_target,
        method="pchip"
    )
    time_uniform = np.linspace(timestamps[0], timestamps[-1], num=len(signal_resampled))

    signal_filtered = nk.signal_filter(
        signal_resampled,
        sampling_rate=fs_target,
        lowcut=float(hr_low_hz),
        highcut=float(hr_high_hz),
        method="butterworth",
        order=Signal.HR_ORDER
    )

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
        fft_vals = np.fft.rfft(segment)
        freqs = np.fft.rfftfreq(win_len, d=1/Video.FPS)
        power = np.abs(fft_vals) ** 2
        mask = (freqs >= 0.7) & (freqs <= 4.0)
        freqs_band, power_band = freqs[mask], power[mask]
        power_band = power_band / np.max(power_band)
        psd_accum.append(power_band)
        freq_est = np.sum(freqs_band * power_band) / np.sum(power_band)
        bpm = freq_est * 60
        hr_values.append(bpm)
        times.append(start / Video.FPS)

    return np.array(times), np.array(hr_values), np.array(psd_accum), freqs_band

def estimate_hr_pyvhr_nt(timestamps, signal):
    from pyVHR.BPM import BPM

    fs_target = PRV.FPS_RESAMPLE_RATE
    hr_low_hz = Signal.HR_LOW
    hr_high_hz = Signal.HR_HIGH
    window_size = rppg.window_size
    step_size = rppg.step_size

    if fs_target is None:
        fs_target = float(PRV.FPS_RESAMPLE_RATE)

    dt_median = np.median(np.diff(timestamps))
    fs_input = 1.0 / dt_median

    # Resample to uniform fs_target
    signal_resampled = nk.signal_resample(
        signal,
        sampling_rate=fs_input,
        desired_sampling_rate=fs_target,
        method="pchip"
    )
    time_uniform = np.linspace(timestamps[0], timestamps[-1], num=len(signal_resampled))

    # Band-pass filter same as your original
    signal_filtered = nk.signal_filter(
        signal_resampled,
        sampling_rate=fs_target,
        lowcut=float(hr_low_hz),
        highcut=float(hr_high_hz),
        method="butterworth",
        order=Signal.HR_ORDER
    )

    samples_per_window = int(window_size * fs_target)
    step_samples = int(step_size * fs_target)

    hr_estimates = []
    window_centers = []

    for start_idx in range(0, len(signal_filtered) - samples_per_window, step_samples):
        segment = signal_filtered[start_idx:start_idx + samples_per_window]
        # pyVHR BPM expects shape [n_estimators, T], so wrap in [None, :]
        segment2d = segment[None, :]
        bpm_est = BPM(segment2d, fs_target).BVP_to_BPM()[0]
        hr_estimates.append(float(bpm_est))
        window_centers.append(time_uniform[start_idx + samples_per_window // 2])

    return np.array(window_centers), np.array(hr_estimates)


def estimate_hr_fft_nt(timestamps, signal):
    fs_target = PRV.FPS_RESAMPLE_RATE
    hr_low_hz = Signal.HR_LOW
    hr_high_hz = Signal.HR_HIGH
    window_size = rppg.window_size
    step_size = rppg.step_size

    if fs_target is None:
        fs_target = float(PRV.FPS_RESAMPLE_RATE)

    dt_median = np.median(np.diff(timestamps))
    fs_input = 1.0 / dt_median
    signal_resampled = nk.signal_resample(
        signal,
        sampling_rate=fs_input,
        desired_sampling_rate=fs_target,
        method="pchip"
    )
    time_uniform = np.linspace(timestamps[0], timestamps[-1], num=len(signal_resampled))

    signal_filtered = nk.signal_filter(
        signal_resampled,
        sampling_rate=fs_target,
        lowcut=float(hr_low_hz),
        highcut=float(hr_high_hz),
        method="butterworth",
        order=Signal.HR_ORDER
    )

    samples_per_window = int(window_size * fs_target)
    step_samples = int(step_size * fs_target)

    hr_estimates = []
    window_centers = []

    for start_idx in range(0, len(signal_filtered) - samples_per_window, step_samples):
        segment = signal_filtered[start_idx:start_idx + samples_per_window]
        segment = segment - np.mean(segment)
        spectrum = np.fft.rfft(segment)
        freqs = np.fft.rfftfreq(len(segment), d=1.0/fs_target)
        power = np.abs(spectrum) ** 2
        band_mask = (freqs >= float(Signal.HR_LOW)) & (freqs <= float(Signal.HR_HIGH))
        if not np.any(band_mask):
            continue
        freqs_band = freqs[band_mask]
        power_band = power[band_mask]
        peak_freq = freqs_band[np.argmax(power_band)]
        hr_bpm = 60.0 * peak_freq
        hr_estimates.append(hr_bpm)
        window_centers.append(time_uniform[start_idx + samples_per_window // 2])

    return np.array(window_centers), np.array(hr_estimates)

def next_pow2(x):
    return 1 << (int(np.ceil(np.log2(max(1, int(x))))))

def estimate_hr_welch_nk(timestamps, signal):
    fs_target = PRV.FPS_RESAMPLE_RATE
    hr_low_hz = float(Signal.HR_LOW)
    hr_high_hz = float(Signal.HR_HIGH)
    window_size = float(rppg.window_size)
    step_size = float(rppg.step_size)

    dt = np.diff(timestamps)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    fs_input = 1.0 / np.median(dt) if dt.size else fs_target

    signal_resampled = nk.signal_resample(
        signal,
        sampling_rate=fs_input,
        desired_sampling_rate=fs_target,
        method="pchip"
    )
    time_uniform = np.linspace(timestamps[0], timestamps[-1], num=len(signal_resampled))

    signal_filtered = nk.signal_filter(
        signal_resampled,
        sampling_rate=fs_target,
        lowcut=hr_low_hz,
        highcut=hr_high_hz,
        method="butterworth",
        order=Signal.HR_ORDER
    )

    samples_per_window = int(round(window_size * fs_target))
    step_samples = max(1, int(round(step_size * fs_target)))

    nperseg = max(16, int(round(0.75 * samples_per_window)))
    noverlap = max(0, int(round(0.5 * nperseg)))
    nfft = next_pow2(4 * nperseg)
    win = 'hann'

    hr_estimates = []
    window_centers = []
    freqs_band_ref = None
    band_mask = None

    if len(signal_filtered) >= samples_per_window:
        dummy = signal_filtered[:samples_per_window]
        freqs_full, _ = welch(
            dummy, fs=fs_target, window=win, nperseg=nperseg, noverlap=noverlap,
            nfft=nfft, detrend="constant", scaling="density", average="median",
            return_onesided=True
        )
        band_mask = (freqs_full >= hr_low_hz) & (freqs_full <= hr_high_hz)
        freqs_band_ref = freqs_full[band_mask] if np.any(band_mask) else None

    for start_idx in range(0, len(signal_filtered) - samples_per_window + 1, step_samples):
        seg = signal_filtered[start_idx:start_idx + samples_per_window]

        if band_mask is None or not np.any(band_mask):
            hr_estimates.append(np.nan)
            window_centers.append(time_uniform[start_idx + samples_per_window // 2])
            continue

        freqs, psd = welch(
            seg, fs=fs_target, window=win, nperseg=nperseg, noverlap=noverlap,
            nfft=nfft, detrend="constant", scaling="density", average="median",
            return_onesided=True
        )

        Pb = psd[band_mask]
        fb = freqs[band_mask]

        if Pb.size:
            k = int(np.argmax(Pb))
            f_peak = fb[k]
            hr_estimates.append(60.0 * f_peak)
        else:
            hr_estimates.append(np.nan)

        window_centers.append(time_uniform[start_idx + samples_per_window // 2])

    return np.asarray(window_centers), np.asarray(hr_estimates)

def estimate_hr_welch_nk_simple(timestamps, signal):
    fs_target = PRV.FPS_RESAMPLE_RATE
    hr_low_hz = Signal.HR_LOW
    hr_high_hz = Signal.HR_HIGH
    window_size = rppg.window_size
    step_size = rppg.step_size

    dt_median = np.median(np.diff(timestamps))
    fs_input = 1.0 / dt_median
    signal_resampled = nk.signal_resample(
        signal,
        sampling_rate=fs_input,
        desired_sampling_rate=fs_target,
        method="pchip"
    )
    time_uniform = np.linspace(timestamps[0], timestamps[-1], num=len(signal_resampled))

    signal_filtered = nk.signal_filter(
        signal_resampled,
        sampling_rate=fs_target,
        lowcut=float(hr_low_hz),
        highcut=float(hr_high_hz),
        method="butterworth",
        order=Signal.HR_ORDER
    )

    samples_per_window = int(window_size * fs_target)
    step_samples = int(step_size * fs_target)

    hr_estimates = []
    window_centers = []

    for start_idx in range(0, len(signal_filtered) - samples_per_window, step_samples):
        segment = signal_filtered[start_idx:start_idx + samples_per_window]
        freqs, psd = welch(
            segment,
            fs=fs_target,
            nperseg=samples_per_window,
            scaling="density"
        )
        band_mask = (freqs >= float(hr_low_hz)) & (freqs <= float(hr_high_hz))
        if not np.any(band_mask):
            continue
        freqs_band = freqs[band_mask]
        psd_band = psd[band_mask]
        peak_freq = freqs_band[np.argmax(psd_band)]
        hr_bpm = 60.0 * peak_freq
        hr_estimates.append(hr_bpm)
        window_centers.append(time_uniform[start_idx + samples_per_window // 2])

    return np.array(window_centers), np.array(hr_estimates)

def sliding_welch(rppg_signal):
    x = np.asarray(rppg_signal, dtype=float)
    x = detrend(x, type="linear")
    b, a = butter(6, [0.65, 4.0], btype="band", fs=Video.FPS)
    xf = filtfilt(b, a, x)
    win_len = int(round(10 * Video.FPS))
    step_len = win_len
    nperseg = win_len
    noverlap = 0
    freqs = np.fft.rfftfreq(nperseg, 1 / Video.FPS)
    band_mask = (freqs >= 39.0 / 60.0) & (freqs <= 240.0 / 60.0)
    freqs_band = freqs[band_mask]
    times, hr_values, rows = [], [], []

    for start in range(0, len(xf) - win_len + 1, step_len):
        seg = xf[start:start + win_len]
        f, pxx = welch(
            seg, fs=Video.FPS, window="hann",
            nperseg=nperseg, noverlap=noverlap,
            detrend=False, scaling="density", return_onesided=True
        )
        pband = pxx[band_mask]
        pnorm = pband / np.max(pband)
        peak_idx = np.argmax(pband)
        bpm = freqs_band[peak_idx] * 60.0
        rows.append(pnorm)
        hr_values.append(bpm)
        times.append((start + win_len / 2) / Video.FPS)

    psd_accum = np.vstack(rows)
    return np.array(times), np.array(hr_values), psd_accum, freqs_band
