import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sps
from src.config import Signal


def ecg_hr_from_signal_500hz(ecg):
    fs_hz = 500.0
    ecg_signal = np.asarray(ecg, float).ravel()
    time_s = np.arange(ecg_signal.size) / fs_hz

    b, a = sps.butter(4, [5, 15], btype="band", fs=fs_hz)
    ecg_band = sps.filtfilt(b, a, ecg_signal)

    min_samples_between_peaks = int(fs_hz / Signal.HR_HIGH)
    p5, p95 = np.percentile(ecg_band, [5, 95])
    prominence = 0.3 * (p95 - p5)
    r_peak_idx, _ = sps.find_peaks(ecg_band, distance=min_samples_between_peaks, prominence=prominence)

    if r_peak_idx.size < 2:
        return np.array([]), np.array([]), r_peak_idx, np.array([])

    r_peak_time = time_s[r_peak_idx]
    rr_intervals_s = np.diff(r_peak_time)
    hr_time_s = 0.5 * (r_peak_time[1:] + r_peak_time[:-1])
    hr_bpm = 60.0 / rr_intervals_s
    return hr_time_s, hr_bpm, r_peak_idx, rr_intervals_s


def moving_average(x, w):
    if w <= 1:
        return np.asarray(x, float).copy()
    x_arr = np.asarray(x, float).ravel()
    csum = np.cumsum(np.insert(x_arr, 0, 0.0))
    return (csum[w:] - csum[:-w]) / float(w)


def detect_r_waves(
    ecg: np.ndarray,
    time: np.ndarray,
    heightper: float = 0.7,
    distanceper: float = 1.0,
    plot: bool = True,
    *,
    fs: float = 500.0,
    qrs_band=(5.0, 15.0),
    notch_hz=None,
    min_rr_s: float = 0.25,
    hr_bounds=(30.0, 220.0)
):
    ecg_signal = np.asarray(ecg, float).ravel()
    time_s = np.asarray(time, float).ravel()
    assert ecg_signal.size == time_s.size, "ecg and time must have same length"

    x = ecg_signal.copy()
    if notch_hz:
        bw = 2.0
        b_notch, a_notch = sps.iirnotch(notch_hz, notch_hz / bw, fs=fs)
        x = sps.filtfilt(b_notch, a_notch, x)
    if qrs_band is not None:
        b_band, a_band = sps.butter(4, qrs_band, btype="band", fs=fs)
        x = sps.filtfilt(b_band, a_band, x)

    d_ecg = np.diff(x, prepend=x[0])
    peaks_d_ecg, _ = sps.find_peaks(d_ecg)

    mean_peaks = np.mean(d_ecg[peaks_d_ecg]) if peaks_d_ecg.size else np.mean(d_ecg)
    max_d = np.max(d_ecg) if d_ecg.size else 0.0
    raw_thr = np.mean([mean_peaks, max_d]) * heightper
    med = np.median(d_ecg)
    mad = np.median(np.abs(d_ecg - med)) + 1e-12
    thr = max(raw_thr, med + 4.0 * 1.4826 * mad)

    _cand_idx, _ = sps.find_peaks(d_ecg, height=thr)
    min_distance = int(round(min_rr_s * fs))
    d_ecg_idx, _ = sps.find_peaks(d_ecg, height=thr, distance=min_distance)

    r_idx = []
    qrs_window = int(round(0.12 * fs))
    for i in d_ecg_idx:
        end = min(i + qrs_window, ecg_signal.size)
        if end <= i + 1:
            continue
        local_max = i + int(np.argmax(x[i:end]))
        if not r_idx or (local_max - r_idx[-1]) >= min_distance:
            r_idx.append(local_max)
    r_idx = np.asarray(r_idx, dtype=int)

    if r_idx.size == 0:
        return r_idx, np.array([]), np.array([]), np.array([]), {
            'ecg_filt': x,
            'd_ecg': d_ecg,
            'peaks_d_ecg': peaks_d_ecg,
            'Rwave_peaks_d_ecg': d_ecg_idx,
            'threshold': thr,
            'mean_distance': np.nan,
            'avg_hr_bpm': np.nan
        }

    r_time = time_s[r_idx]
    rr_s = np.diff(r_time)
    hr_inst = 60.0 / rr_s
    hr_lo, hr_hi = hr_bounds
    keep_mask = (hr_inst >= hr_lo) & (hr_inst <= hr_hi)
    hr_bpm = hr_inst[keep_mask]
    hr_time = r_time[1:][keep_mask]
    avg_hr_bpm = float(np.mean(hr_bpm)) if hr_bpm.size else np.nan

    if plot:
        plt.figure()
        plt.plot(time_s, d_ecg, label='dECG (band-passed)')
        if peaks_d_ecg.size:
            plt.plot(time_s[peaks_d_ecg], d_ecg[peaks_d_ecg], "x", label='dECG peaks')
        plt.xlabel('Time [s]')
        plt.ylabel('Derivative []')
        plt.title('Step 1: Peaks of dECG')
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(time_s, d_ecg, label='dECG')
        plt.axhline(thr, linestyle='--', label='threshold')
        if d_ecg_idx.size:
            plt.plot(time_s[d_ecg_idx], d_ecg[d_ecg_idx], "x", label='filtered dECG peaks')
        plt.xlabel('Time [s]')
        plt.ylabel('Derivative []')
        plt.legend()
        plt.title('Step 2: Thresholded + refractory dECG peaks')
        plt.show()

        fig, ax1 = plt.subplots()
        ax1.plot(time_s, d_ecg, label='dECG')
        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel('dECG []')
        ax1.set_title('Step 3: R-wave peaks on ECG')
        ax2 = ax1.twinx()
        ax2.plot(time_s, ecg_signal, label='ECG (raw)', alpha=0.6)
        ax2.plot(time_s, x, label='ECG (band-passed)', alpha=0.8)
        ax2.plot(time_s[r_idx], x[r_idx], "x", label='R-waves')
        ax2.set_ylabel('ECG []')
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        plt.show()

        if hr_bpm.size:
            plt.figure()
            plt.plot(hr_time, hr_bpm)
            plt.xlabel('Time [s]')
            plt.ylabel('Heart Rate [BPM]')
            plt.title(f'Instantaneous HR (avg ≈ {avg_hr_bpm:.1f} BPM)')
            plt.show()

    aux = {
        'ecg_filt': x,
        'd_ecg': d_ecg,
        'peaks_d_ecg': peaks_d_ecg,
        'Rwave_peaks_d_ecg': d_ecg_idx,
        'threshold': thr,
        'mean_distance': float(min_distance),
        'avg_hr_bpm': avg_hr_bpm,
    }
    return r_idx, r_time, hr_bpm, hr_time, aux
