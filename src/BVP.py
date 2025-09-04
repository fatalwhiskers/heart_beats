import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from scipy.ndimage import median_filter
from src.config import Signal, BVP, fileDataset1
import os

# -------------------------------------------------------------------
# Helpers (same names; improved internals)
# -------------------------------------------------------------------

def _bandpass(sig, fs, low=0.7, high=5.0, order=4):
    """
    Zero-phase band-pass (offline) tuned for wrist PPG.
    Defaults ~0.7–5.0 Hz (~42–300 bpm) to reduce motion/high-freq noise.
    """
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, sig)

def _phys_peaks(sig, fs, hr_low, hr_high, prom_frac=0.2, min_width_sec=0.10):
    """
    Peak picking with constraints derived from HR bounds.
    NOTE: hr_low/high are in bpm.
    """
    # Min distance in samples at the *maximum* plausible HR
    # (FIX: convert bpm → samples; previously treated bpm as Hz)
    min_dist = max(1, int(round(fs * 60.0 / float(hr_high))))

    # Width at half-prominence (samples)
    width_samp = max(1, int(round(min_width_sec * fs)))

    # Robust range → prominence threshold
    lo, hi = np.percentile(sig, [5, 95])
    amp_range = max(np.finfo(float).eps, (hi - lo))
    prom = max(np.finfo(float).eps, prom_frac * amp_range)

    peaks, props = find_peaks(sig, distance=min_dist, prominence=prom, width=width_samp)
    return peaks, props

def _pp_hr_from_peaks(peaks_idx, fs):
    """
    Returns:
      pp  : beat-to-beat intervals (s)
      hr  : instantaneous HR (bpm)
      t_mid: time at mid-interval (s from start)
    """
    t_peaks = peaks_idx / float(fs)
    if t_peaks.size < 2:
        return (np.array([]), np.array([]), np.array([]))
    pp = np.diff(t_peaks)
    t_mid = 0.5 * (t_peaks[1:] + t_peaks[:-1])
    hr = 60.0 / pp
    return pp, hr, t_mid

def _kubios_like_pp_filter(pp, L=11, thr=0.15, rel_cap=0.35):
    """
    Simple median-based correction:
      - Replace outliers |pp - med| > thr (seconds) with local median.
      - Then clamp to ±rel_cap around the local median.
    """
    if pp.size == 0:
        return pp, np.zeros(0, dtype=bool)
    med = median_filter(pp, size=L, mode='reflect')
    diff = np.abs(pp - med)
    mask = diff > thr
    pp_f = np.where(mask, med, pp)
    pp_f = np.clip(pp_f, med*(1-rel_cap), med*(1+rel_cap))
    return pp_f, mask

# -------------------------------------------------------------------
# Main entry (same name/signature; returns hr_clean, t_mid)
# -------------------------------------------------------------------

def get_bvp_ground_truth(path_csv):
    """
    Loads a single-column numeric BVP file from fileDataset1.folder_path/path_csv
    sampled at BVP.BVP_RATE (Hz), runs band-pass + peak picking (auto-polarity),
    applies Kubios-like PP correction, and returns:
        hr_clean (bpm), t_mid (s from start)
    """
    # Load single-column numeric data
    bvp_path = os.path.join(fileDataset1.folder_path, path_csv)
    bvp = np.loadtxt(bvp_path)  # ensure it’s a single numeric column

    # Sampling rate (Hz)
    fs = float(BVP.BVP_RATE)

    # Band-pass (offline)
    bvp_bp = _bandpass(bvp, fs, low=0.7, high=5.0, order=4)

    # Try both polarities; keep the one with "better" peaks
    peaks_a, det_a = _phys_peaks(bvp_bp, fs, Signal.HR_LOW, Signal.HR_HIGH,
                                 prom_frac=0.2, min_width_sec=0.10)
    pp_a, hr_a, tmid_a = _pp_hr_from_peaks(peaks_a, fs)

    peaks_b, det_b = _phys_peaks(-bvp_bp, fs, Signal.HR_LOW, Signal.HR_HIGH,
                                 prom_frac=0.2, min_width_sec=0.10)
    pp_b, hr_b, tmid_b = _pp_hr_from_peaks(peaks_b, fs)

    def _score(peaks, det):
        if peaks.size == 0:
            return (0, 0.0, 0.0)
        prom = det.get("prominences", np.array([0.0]))
        width = det.get("widths", np.array([0.0]))
        return (int(peaks.size), float(np.median(prom)), float(np.median(width)))

    use_b = _score(peaks_b, det_b) > _score(peaks_a, det_a)

    if use_b:
        peaks = peaks_b
        pp_raw, hr_raw, t_mid = pp_b, hr_b, tmid_b
        bvp_used = -bvp_bp
        det_used = det_b
    else:
        peaks = peaks_a
        pp_raw, hr_raw, t_mid = pp_a, hr_a, tmid_a
        bvp_used = bvp_bp
        det_used = det_a

    if hr_raw.size:
        ok = (hr_raw >= float(Signal.HR_LOW)) & (hr_raw <= float(Signal.HR_HIGH))
        pp_raw = pp_raw[ok]
        hr_raw = hr_raw[ok]
        t_mid = t_mid[ok]

    # Kubios-like correction on PP
    pp_clean, _mask = _kubios_like_pp_filter(pp_raw, L=11, thr=0.15, rel_cap=0.35)
    hr_clean = 60.0 / pp_clean if pp_clean.size else np.array([])

    return hr_clean, t_mid

    return {
        "hr_inst_clean": hr_clean,     # instantaneous HR (BPM) at t_mid
        "hr_inst_raw":   hr_raw,       # pre-correction
        "t_mid":         t_mid,        # times of HR values (s)
        "pp_clean":      pp_clean,     # cleaned PP (s)
        "pp_raw":        pp_raw,       # raw PP (s)
        "peaks_idx":     peaks,        # indices into bvp_used
        "bvp_filt":      bvp_used,     # filtered BVP (possibly inverted)
        "fs":            fs,
        "details":       det_used,     # find_peaks diagnostics
        "artifact_mask": mask
    }
