import os
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from scipy.ndimage import median_filter
from src.config import Signal, BVP, fileDataset1


def _bandpass(signal, sample_rate, low=0.7, high=5.0, order=4):
    nyquist = 0.5 * sample_rate
    b, a = butter(order, [low / nyquist, high / nyquist], btype="band")
    return filtfilt(b, a, signal)


def _phys_peaks(signal, sample_rate, hr_low_bpm, hr_high_bpm, prom_frac=0.2, min_width_sec=0.10):
    min_distance = max(1, int(round(sample_rate / float(hr_high_bpm))))
    min_width = max(1, int(round(min_width_sec * sample_rate)))
    p5, p95 = np.percentile(signal, [5, 95])
    amp_range = max(np.finfo(float).eps, (p95 - p5))
    prominence = max(np.finfo(float).eps, prom_frac * amp_range)
    peak_idx, props = find_peaks(signal, distance=min_distance, prominence=prominence, width=min_width)
    return peak_idx, props


def _pp_hr_from_peaks(peak_idx, sample_rate):
    t_peaks = peak_idx / float(sample_rate)
    if t_peaks.size < 2:
        return np.array([]), np.array([]), np.array([])
    pp_intervals = np.diff(t_peaks)
    t_mid = 0.5 * (t_peaks[1:] + t_peaks[:-1])
    hr_bpm = 60.0 / pp_intervals
    return pp_intervals, hr_bpm, t_mid


def _kubios_like_pp_filter(pp_intervals, L=11, thr=0.15, rel_cap=0.35):
    if pp_intervals.size == 0:
        return pp_intervals, np.zeros(0, dtype=bool)
    local_med = median_filter(pp_intervals, size=L, mode="reflect")
    abs_dev = np.abs(pp_intervals - local_med)
    outlier_mask = abs_dev > thr
    pp_corrected = np.where(outlier_mask, local_med, pp_intervals)
    pp_corrected = np.clip(pp_corrected, local_med * (1 - rel_cap), local_med * (1 + rel_cap))
    return pp_corrected, outlier_mask


def _score(peak_idx, details):
    if peak_idx.size == 0:
        return 0, 0.0, 0.0
    prominences = details.get("prominences", np.array([0.0]))
    widths = details.get("widths", np.array([0.0]))
    return int(peak_idx.size), float(np.median(prominences)), float(np.median(widths))


def get_bvp_ground_truth(path_csv):
    bvp_path = os.path.join(fileDataset1.folder_path, path_csv)
    bvp = np.loadtxt(bvp_path)
    fs = float(BVP.BVP_RATE)
    bvp_bp = _bandpass(bvp, fs, low=0.7, high=5.0, order=4)

    peaks_pos, det_pos = _phys_peaks(bvp_bp, fs, Signal.HR_LOW, Signal.HR_HIGH, prom_frac=0.2, min_width_sec=0.10)
    pp_pos, hr_pos, tmid_pos = _pp_hr_from_peaks(peaks_pos, fs)

    peaks_neg, det_neg = _phys_peaks(-bvp_bp, fs, Signal.HR_LOW, Signal.HR_HIGH, prom_frac=0.2, min_width_sec=0.10)
    pp_neg, hr_neg, tmid_neg = _pp_hr_from_peaks(peaks_neg, fs)

    use_negative = _score(peaks_neg, det_neg) > _score(peaks_pos, det_pos)

    if use_negative:
        peaks_idx = peaks_neg
        pp_raw, hr_raw, t_mid = pp_neg, hr_neg, tmid_neg
        bvp_used = -bvp_bp
        det_used = det_neg
    else:
        peaks_idx = peaks_pos
        pp_raw, hr_raw, t_mid = pp_pos, hr_pos, tmid_pos
        bvp_used = bvp_bp
        det_used = det_pos

    if hr_raw.size:
        hr_ok = (hr_raw >= float(Signal.HR_LOW)) & (hr_raw <= float(Signal.HR_HIGH))
        pp_raw = pp_raw[hr_ok]
        hr_raw = hr_raw[hr_ok]
        t_mid = t_mid[hr_ok]

    pp_clean, artifact_mask = _kubios_like_pp_filter(pp_raw, L=11, thr=0.15, rel_cap=0.35)
    hr_clean = 60.0 / pp_clean if pp_clean.size else np.array([])

    return hr_clean, t_mid
