import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.signal import find_peaks
from src.config import Video 
from src.config import Signal
from src.config import PRV
from scipy.ndimage import median_filter


def resample_to_uniform(time_in, signal_in, max_gap_sec=0.6):
    """
    NaN-safe resampling to a uniform grid at PRV.FPS_RESAMPLE_RATE.
    - Drops non-finite pairs and duplicate timestamps.
    - Interpolates segment-by-segment with PCHIP (no extrapolation).
    - Leaves long gaps (> max_gap_sec) as NaN.
    """
    time_in   = np.asarray(time_in,  float)
    signal_in = np.asarray(signal_in, float)

    if time_in.ndim != 1 or signal_in.ndim != 1 or len(time_in) != len(signal_in):
        raise ValueError("t_in and x_in must be 1D and same length")

    # keep only finite (time, signal) pairs
    m = np.isfinite(time_in) & np.isfinite(signal_in)
    time_in, signal_in = time_in[m], signal_in[m]
    if time_in.size < 2:
        return np.array([]), np.array([])

    # sort & deduplicate timestamps (keep first)
    order = np.argsort(time_in)
    time_in, signal_in = time_in[order], signal_in[order]
    time_in, keep_idx = np.unique(time_in, return_index=True)
    signal_in = signal_in[keep_idx]

    # build uniform grid over the valid span
    dt = 1.0 / float(PRV.FPS_RESAMPLE_RATE)
    t0 = float(time_in[0]); t1 = float(time_in[-1])
    new_Time_Grid = np.arange(t0, t1 + 1e-9, dt)
    signal_out = np.full_like(new_Time_Grid, np.nan, dtype=float)

    # split input into contiguous segments (no long gaps)
    gaps = np.diff(time_in)
    cut_idx = np.where(gaps > max_gap_sec)[0]
    seg_starts = np.r_[0, cut_idx + 1]
    seg_ends   = np.r_[cut_idx + 1, time_in.size]

    for s, e in zip(seg_starts, seg_ends):
        if e - s < 2:
            continue
        t_seg = time_in[s:e]
        x_seg = signal_in[s:e]

        mask_grid = (new_Time_Grid >= t_seg[0]) & (new_Time_Grid <= t_seg[-1])
        if not mask_grid.any():
            continue

        f = PchipInterpolator(t_seg, x_seg, extrapolate=False)
        signal_out[mask_grid] = f(new_Time_Grid[mask_grid])

    return new_Time_Grid, signal_out

def detect_peaks_rppg(sig):
    """
    NaN-safe peak detection on 'sig' sampled at PRV.FPS_RESAMPLE_RATE.
    Processes each finite segment separately and concatenates results.
    """
    sig = np.asarray(sig, float)
    fs = float(PRV.FPS_RESAMPLE_RATE)

    # distances/widths in samples
    min_distance = max(1, int(np.floor(fs / float(Signal.HR_HIGH))))       # max HR
    # max_distance = int(fs / float(Signal.HR_LOW))  # (not used)
    sample_width = max(1, int(np.ceil(PRV.MIN_PEAK_WIDTH * fs)))

    sig_samples_all = []
    widths_all, prominences_all = [], []

    finite = np.isfinite(sig)
    if not finite.any():
        return np.array([], dtype=int), {"widths": np.array([]), "prominences": np.array([])}

    # contiguous finite segments
    edges = np.flatnonzero(np.diff(np.r_[False, finite, False]))
    segs = list(zip(edges[0::2], edges[1::2]))

    for s, e in segs:
        seg = sig[s:e]
        if seg.size < 3:
            continue

        # robust prominence threshold from your bounds
        lo, hi = np.nanpercentile(seg, [PRV.PROMINENCE_LOWER_BOUND, PRV.PROMINENCE_UPPER_BOUND])
        prom = PRV.PROM_THRESHOLD * (hi - lo)

        idx, props = find_peaks(seg, distance=min_distance, prominence=prom, width=sample_width)
        if idx.size:
            sig_samples_all.append(idx + s)
            if "widths" in props and props["widths"].size:
                widths_all.append(props["widths"])
            if "prominences" in props and props["prominences"].size:
                prominences_all.append(props["prominences"])

    if not sig_samples_all:
        return np.array([], dtype=int), {"widths": np.array([]), "prominences": np.array([])}

    sig_samples = np.concatenate(sig_samples_all)
    widths = np.concatenate(widths_all) if len(widths_all) else np.array([])
    prominences = np.concatenate(prominences_all) if len(prominences_all) else np.array([])

    details = {"widths": widths, "prominences": prominences}
    return sig_samples, details

def pp_intervals_from_peaks(t_peaks):
    """
        PP intervals (seconds) and mid-times of intervals.
        [0.80, 1.60, 2.43, 3.26]

        pp = diff = [0.80, 0.83, 0.83] s

        o its just the time between peaks 
        so
        hr_inst = 60.0 / pp   
        [75.0, 72.3, 72.3] BPM

        t_mid = [(0.80+1.60)/2, (1.60+2.43)/2, (2.43+3.26)/2] = [1.20, 2.015, 2.845] s
        this is just the time for each pp so it can be comaperd just the midlle of each time
    """
    t_peaks = np.asarray(t_peaks, float) #must be float since we are getting the middle

    pp = np.diff(t_peaks)                   # seconds
    time_at_mid = 0.5 * (t_peaks[1:] + t_peaks[:-1])  # time at which PP is defined
    return pp, time_at_mid

def kubios_like_pp_filter(pp):
    """
    Median-based artifact correction (Kubios-like).
    Replace PP values deviating > t_thresh (s) from running median (window L).
    """

    #running median the expexted value
    
    med = median_filter(pp, size= PRV.KUBIOS_L, mode='reflect')

    """
    pp   = [0.82, 0.81, 0.79, 1.60, 0.80, 0.83]
    med  = [0.81, 0.81, 0.81, 0.81, 0.81, 0.82]  # running median
    t_thresh = 0.15

    abs(pp - med) = [0.01, 0.00, 0.02, 0.79, 0.01, 0.01]
    mask          = [F,    F,    F,    T,    F,    F]
    pp_f          = [0.82, 0.81, 0.79, 0.81, 0.80, 0.83]
    """

    artifacts_mask = np.abs(pp - med) > PRV.KUBIOS_THRESHOLD
    clean_pp = np.where(artifacts_mask, med, pp)
    return clean_pp, artifacts_mask


def compute_prv_hr(time_stamps_raw, signal_raw):
    """
    Your compute_prv_hr, but NaN-safe. Returns the same tuple you expect:
    (pp_clean, hr_inst_clean, hr_inst_raw, time_at_mid_pp, peaks_t, artifacts_mask)
    """
    # 1) resample to uniform grid (shape-preserving), no gap bridging
    new_Time_Grid, sig = resample_to_uniform(time_stamps_raw, signal_raw, max_gap_sec=0.6)
    if new_Time_Grid.size == 0:
        return (np.array([]),)*6

    # 2) peaks (segment-wise) on 'sig'
    sig_samples, details = detect_peaks_rppg(sig)
    peaks_t = new_Time_Grid[sig_samples]

    # 3) PP intervals and mid-times
    pp_raw, time_at_mid_pp = pp_intervals_from_peaks(peaks_t)
    if pp_raw.size == 0:
        return (np.array([]), np.array([]), np.array([]), np.array([]), peaks_t, np.array([], dtype=bool))

    # 4) Kubios-like artifact correction
    pp_clean, artifacts_mask = kubios_like_pp_filter(pp_raw)

    hr_inst_raw   = 60.0 / pp_raw
    hr_inst_clean = 60.0 / pp_clean

    return (
        pp_clean,            # cleaned PP intervals (s)
        hr_inst_clean,       # cleaned instantaneous HR (BPM)
        hr_inst_raw,         # raw instantaneous HR (BPM)
        time_at_mid_pp,      # time for both HR series (s)
        peaks_t,             # peak timestamps (s)
        artifacts_mask       # which PP were corrected (bool)
    )

def prv_hr_on_times(t_pp, hr_inst, t_target):
    """
    Resample instantaneous HR (defined at t_pp) onto a desired time array (t_target).
    Uses shape-preserving interpolation; returns NaN outside support.
    """
    if t_pp.size < 2:
        return np.full_like(t_target, np.nan, dtype=float)
    f = PchipInterpolator(t_pp, hr_inst, extrapolate=False)
    return f(t_target)
