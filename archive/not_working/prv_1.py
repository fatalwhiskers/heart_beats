import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.signal import find_peaks, periodogram
from scipy.ndimage import median_filter, uniform_filter1d
import matplotlib.pyplot as plt
# ---- your config imports ----
from src.config import Video, Signal, PRV


# -----------------------------
# Utility helpers
# -----------------------------
def _samples_between_beats(fs: float, hr_high_hz: float) -> int:
    """
    Minimum expected samples between peaks at the *maximum* heart rate (Hz).
    Using Hz here because your Signal.HR_HIGH is in Hz already.
    """
    hr_hz = float(hr_high_hz)
    return max(1, int(np.floor(fs / hr_hz)))


def _segmented_finite_ranges(x: np.ndarray):
    """Yield (start, end) index pairs for contiguous finite runs in x."""
    finite = np.isfinite(x)
    if not finite.any():
        return []
    edges = np.flatnonzero(np.diff(np.r_[False, finite, False]))
    return list(zip(edges[0::2], edges[1::2]))


# -----------------------------
# 1) Resample to a uniform grid (NaN-safe, gap-aware)
# -----------------------------
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


# -----------------------------
# 2) Peak detection (segment-wise) with motion-robust guards
# -----------------------------
def detect_peaks_rppg(sig):
    """
    NaN-safe peak detection on 'sig' sampled at PRV.FPS_RESAMPLE_RATE.
    - Processes each finite segment separately.
    - Adaptive prominence using robust range + spectral SNR proxy.
    - Refractory/merge rule to avoid double detections during motion.
    """
    sig = np.asarray(sig, float)
    fs = float(PRV.FPS_RESAMPLE_RATE)

    # distances/widths in samples
    min_distance = _samples_between_beats(fs, float(Signal.HR_HIGH))  # HR_HIGH is Hz
    sample_width = max(1, int(np.ceil(PRV.MIN_PEAK_WIDTH * fs)))

    sig_samples_all = []
    widths_all, prominences_all = [], []

    for s, e in _segmented_finite_ranges(sig):
        seg = sig[s:e]
        if seg.size < 3:
            continue

        # Robust dynamic prominence from percentile range
        lo, hi = np.nanpercentile(seg, [PRV.PROMINENCE_LOWER_BOUND, PRV.PROMINENCE_UPPER_BOUND])
        base_prom = PRV.PROM_THRESHOLD * max(1e-12, (hi - lo))

        # Spectral peakiness (simple SNR proxy) in the HR band
        f, Pxx = periodogram(seg, fs, detrend='constant', scaling='density')
        band = (f >= float(Signal.HR_LOW)) & (f <= float(Signal.HR_HIGH))
        if np.any(band):
            P = Pxx[band]
            pmax = np.max(P)
            pmed = np.median(P)
            # if spectrum is flat (low SNR), raise required prominence
            snr_ratio = (pmax / (pmed + 1e-12))
            # limit range: 1..3
            snr_ratio = float(np.clip(snr_ratio, 1.0, 3.0))
            prom = base_prom * (1.2 / snr_ratio)
        else:
            prom = base_prom

        idx, props = find_peaks(
            seg,
            distance=min_distance,
            prominence=prom,
            width=sample_width
        )

        if idx.size:
            # Merge peaks that are too close (residual doubles)
            exp_samp = min_distance
            too_close = np.diff(idx) < int(0.6 * exp_samp)
            if np.any(too_close):
                keep = np.ones_like(idx, dtype=bool)
                clashes = np.where(too_close)[0]
                for k in clashes:
                    # keep the more prominent of the pair
                    a = props["prominences"][k]
                    b = props["prominences"][k + 1]
                    keep[k if a >= b else k + 1] = False
                idx = idx[keep]
                for key in ("widths", "prominences"):
                    if key in props and props[key].size:
                        props[key] = props[key][keep]

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


# -----------------------------
# 3) PP intervals and mid-times
# -----------------------------
def pp_intervals_from_peaks(t_peaks):
    """
    Compute PP intervals (seconds) and mid-times.
    """
    t_peaks = np.asarray(t_peaks, float)
    if t_peaks.size < 2:
        return np.array([]), np.array([])
    pp = np.diff(t_peaks)
    t_mid = 0.5 * (t_peaks[1:] + t_peaks[:-1])
    return pp, t_mid


# -----------------------------
# 4) Kubios-like artifact correction (median replacement)
# -----------------------------
def kubios_like_pp_filter(pp):
    """
    Median-based artifact correction.
    Replace PP values deviating > KUBIOS_THRESHOLD (seconds)
    from running median (window KUBIOS_L).
    """
    if pp.size == 0:
        return pp, np.array([], dtype=bool)

    med = median_filter(pp, size=int(PRV.KUBIOS_L), mode='reflect')
    artifacts_mask = np.abs(pp - med) > float(PRV.KUBIOS_THRESHOLD)
    clean_pp = np.where(artifacts_mask, med, pp)
    return clean_pp, artifacts_mask


# -----------------------------
# 5) Main pipeline: compute PRV HR (robust, NaN-gated, tracked)
# -----------------------------
def compute_prv_hr(time_stamps_raw, signal_raw):
    """
    Returns:
      pp_clean           : cleaned PP intervals (s)
      hr_track           : quality-gated tracked HR (BPM)
      hr_inst_raw        : raw instantaneous HR (BPM)
      time_at_mid_pp     : time for HR series (s)
      peaks_t            : peak timestamps (s)
      artifacts_mask     : which PP were corrected (bool)
    """
    # 1) resample to uniform grid (shape-preserving), no gap bridging
    new_Time_Grid, sig = resample_to_uniform(time_stamps_raw, signal_raw, max_gap_sec=0.6)
    if new_Time_Grid.size == 0:
        return (np.array([]),)*6

    # 2) peaks (segment-wise) on 'sig'
    sig_samples, _ = detect_peaks_rppg(sig)
    if sig_samples.size == 0:
        return (np.array([]),)*6
    peaks_t = new_Time_Grid[sig_samples]

    # 3) PP intervals and mid-times
    pp_raw, time_at_mid_pp = pp_intervals_from_peaks(peaks_t)
    if pp_raw.size == 0:
        return (np.array([]),)*6

    # 4) Kubios-like artifact correction (for diagnostics)
    pp_clean_median, artifacts_mask = kubios_like_pp_filter(pp_raw)

    # 5) Gate out bad PPs before smoothing (hold during motion)
    med = median_filter(pp_raw, size=int(PRV.KUBIOS_L), mode='reflect')
    good = np.abs(pp_raw - med) <= float(PRV.KUBIOS_THRESHOLD)

    # create NaN-gated series (bad -> NaN)
    pp_gate = pp_raw.astype(float).copy()
    pp_gate[~good] = np.nan

    # light inpainting for very short gaps only (<=2 consecutive NaNs)
    # (prevents tiny holes from freezing the tracker)
    pp_inp = pp_gate.copy()
    idx_finite = np.where(np.isfinite(pp_inp))[0]
    if idx_finite.size >= 2:
        f = PchipInterpolator(time_at_mid_pp[idx_finite], pp_inp[idx_finite], extrapolate=False)
        # We only fill where the gap is short; leave long gaps as NaN
        filled = f(time_at_mid_pp)
        # determine run lengths of NaNs
        isnan = ~np.isfinite(pp_inp)
        if isnan.any():
            edges = np.flatnonzero(np.diff(np.r_[0, isnan.view(np.int8), 0]))
            runs = list(zip(edges[0::2], edges[1::2]))
            for a, b in runs:
                if (b - a) <= 2:   # short gap (<=2 PP)
                    pp_inp[a:b] = filled[a:b]

    # 6) Smooth only finite values to reduce jitter
    pp_smooth = pp_inp.copy()
    mask = np.isfinite(pp_smooth)
    if mask.any():
        pp_smooth[mask] = uniform_filter1d(pp_smooth[mask], size=5)

    # 7) Convert to HR (BPM); keep NaNs through long motion bursts
    hr_inst_raw   = 60.0 / pp_raw
    hr_inst_clean = 60.0 / pp_smooth

    # 8) Simple α-β–style tracker: ignore NaNs -> hold estimate
    alpha = 0.2  # measurement blend factor
    hr_track = np.full_like(hr_inst_clean, np.nan, dtype=float)
    # seed with robust initial value
    seed = np.nanmedian(hr_inst_clean[:max(5, int(2 * Signal.HR_HIGH))])  # small initial window
    if not np.isfinite(seed):
        seed = np.nanmedian(hr_inst_raw[:max(5, int(2 * Signal.HR_HIGH))])
    if not np.isfinite(seed):
        seed = 75.0  # fallback BPM
    hr_track[0] = seed

    for i in range(1, hr_track.size):
        h = hr_inst_clean[i]
        if np.isfinite(h):
            # limit unrealistically fast jumps (e.g., ≤ 3 BPM/s)
            dt_pp = time_at_mid_pp[i] - time_at_mid_pp[i - 1]
            max_step = max(1.0, 3.0 * dt_pp)
            pred = hr_track[i - 1]
            h_clamped = np.clip(h, pred - max_step, pred + max_step)
            hr_track[i] = alpha * h_clamped + (1.0 - alpha) * pred
        else:
            # predict/hold during motion/noise
            hr_track[i] = hr_track[i - 1]

    return (
        pp_clean_median,     # cleaned PP intervals (s) by median replacement
        hr_track,            # robust tracked HR (BPM)
        hr_inst_raw,         # raw instantaneous HR (BPM)
        time_at_mid_pp,      # time for HR series (s)
        peaks_t,             # peak timestamps (s)
        artifacts_mask       # which PP were corrected (bool)
    )


# -----------------------------
# 6) Resample HR onto arbitrary timestamps
# -----------------------------
def prv_hr_on_times(t_pp, hr_inst, t_target):
    """
    Resample instantaneous HR (defined at t_pp) onto a desired time array (t_target).
    Uses shape-preserving interpolation; returns NaN outside support.
    """
    t_pp = np.asarray(t_pp, float)
    hr_inst = np.asarray(hr_inst, float)
    t_target = np.asarray(t_target, float)

    if t_pp.size < 2 or not np.isfinite(hr_inst).any():
        return np.full_like(t_target, np.nan, dtype=float)

    # use only finite HR for interpolation
    m = np.isfinite(hr_inst)
    if m.sum() < 2:
        return np.full_like(t_target, np.nan, dtype=float)

    f = PchipInterpolator(t_pp[m], hr_inst[m], extrapolate=False)
    return f(t_target)


import numpy as np
import math

def _pp_physio_bounds():
    # Convert your HR bounds (Hz) to PP bounds (s): PP = 1 / HR(Hz)
    pp_min = 1.0 / float(Signal.HR_HIGH)  # shortest plausible PP (max HR)
    pp_max = 1.0 / float(Signal.HR_LOW)   # longest plausible PP (min HR)
    return pp_min, pp_max

def pp_diagnostics(pp_raw, pp_clean, t_mid, artifacts_mask=None, verbose=True):
    """
    Compute summary diagnostics for PP intervals.
    Returns a dict with stats and masks you can log or assert on.
    """
    pp_raw   = np.asarray(pp_raw, float)
    pp_clean = np.asarray(pp_clean, float)
    t_mid    = np.asarray(t_mid,  float)

    assert pp_raw.ndim == pp_clean.ndim == t_mid.ndim == 1
    assert pp_raw.size == pp_clean.size == t_mid.size

    # Physiological bounds (derived from Signal.HR_* in Hz)
    pp_min, pp_max = _pp_physio_bounds()

    # Basic masks
    nan_mask   = ~np.isfinite(pp_raw)
    bad_range  = np.isfinite(pp_raw) & ((pp_raw < pp_min) | (pp_raw > pp_max))
    good_range = np.isfinite(pp_raw) & ~bad_range

    # Kubios-like artifacts if not provided (mirror your pipeline logic)
    if artifacts_mask is None or artifacts_mask.size != pp_raw.size:
        med = median_filter(pp_raw, size=int(PRV.KUBIOS_L), mode='reflect')
        artifacts_mask = np.abs(pp_raw - med) > float(PRV.KUBIOS_THRESHOLD)

    # Gaps in PP (consecutive NaNs) – report run lengths
    def _runs(mask):
        if mask.size == 0:
            return []
        edges = np.flatnonzero(np.diff(np.r_[0, mask.view(np.int8), 0]))
        return [b - a for a, b in zip(edges[0::2], edges[1::2])]
    nan_runs = _runs(nan_mask)

    # Stats helpers
    def _pct(x): return 100.0 * (x / max(1, pp_raw.size))

    def _summary(arr, mask):
        vals = arr[mask]
        if vals.size == 0:
            return None
        q = np.nanpercentile(vals, [5, 25, 50, 75, 95])
        return {
            "count": int(vals.size),
            "mean": float(np.nanmean(vals)),
            "std":  float(np.nanstd(vals)),
            "p5":   float(q[0]),
            "p25":  float(q[1]),
            "p50":  float(q[2]),
            "p75":  float(q[3]),
            "p95":  float(q[4]),
        }

    # Convert to BPM for a feel of distributions
    def _to_bpm(pp): 
        v = 60.0 / pp.astype(float)
        v[~np.isfinite(pp)] = np.nan
        return v

    stats = {
        "n": int(pp_raw.size),
        "nan_count": int(nan_mask.sum()),
        "nan_pct": _pct(nan_mask.sum()),
        "nan_run_lengths": nan_runs,
        "bad_range_count": int(bad_range.sum()),
        "bad_range_pct": _pct(bad_range.sum()),
        "artifact_count": int(np.sum(artifacts_mask)),
        "artifact_pct": _pct(np.sum(artifacts_mask)),
        "pp_bounds_s": {"min": pp_min, "max": pp_max},
        "pp_raw_summary_all": _summary(pp_raw, np.isfinite(pp_raw)),
        "pp_raw_summary_goodrange": _summary(pp_raw, good_range),
        "pp_clean_summary_all": _summary(pp_clean, np.isfinite(pp_clean)),
        "hr_raw_bpm_summary": _summary(_to_bpm(pp_raw), np.isfinite(pp_raw)),
        "hr_clean_bpm_summary": _summary(_to_bpm(pp_clean), np.isfinite(pp_clean)),
    }

    if verbose:
        print(
            f"PP count: {stats['n']}\n"
            f"NaNs: {stats['nan_count']} ({stats['nan_pct']:.1f}%) | "
            f"Bad range: {stats['bad_range_count']} ({stats['bad_range_pct']:.1f}%) | "
            f"Artifacts (Kubios-like): {stats['artifact_count']} ({stats['artifact_pct']:.1f}%)\n"
            f"Physio PP bounds (s): [{pp_min:.3f}, {pp_max:.3f}] "
        )
        if stats["pp_clean_summary_all"]:
            m = stats["pp_clean_summary_all"]
            print(
                f"PP_clean (s): mean={m['mean']:.3f} std={m['std']:.3f} "
                f"p50={m['p50']:.3f}  "
                f"[p5={m['p5']:.3f}, p95={m['p95']:.3f}]"
            )
        if stats["hr_clean_bpm_summary"]:
            h = stats["hr_clean_bpm_summary"]
            print(
                f"HR_clean (BPM): mean={h['mean']:.1f} std={h['std']:.1f} "
                f"p50={h['p50']:.1f}  "
                f"[p5={h['p5']:.1f}, p95={h['p95']:.1f}]"
            )
    return stats

def plot_pp_diagnostics(t_mid, pp_raw, pp_clean, artifacts_mask=None, show=True):
    """
    Lightweight plots to visually inspect PP series and artifacts.
    Only uses matplotlib; call after compute_prv_hr.
    """
   

    pp_min, pp_max = _pp_physio_bounds()
    hr_low_bpm  = 60.0 * float(Signal.HR_LOW)
    hr_high_bpm = 60.0 * float(Signal.HR_HIGH)

    # Time series of PP
    plt.figure()
    plt.plot(t_mid, pp_raw, label="PP raw")
    plt.plot(t_mid, pp_clean, label="PP clean", linewidth=2)
    if artifacts_mask is not None and artifacts_mask.any():
        plt.scatter(t_mid[artifacts_mask], pp_raw[artifacts_mask], marker="x", label="Artifacts")
    plt.hlines([pp_min, pp_max], xmin=t_mid[0], xmax=t_mid[-1], linestyles="dashed", label="Physio bounds")
    plt.xlabel("Time (s) at PP midpoints")
    plt.ylabel("PP interval (s)")
    plt.title("PP intervals")
    plt.legend()
    if show: plt.show()

    # Histogram of HR equivalents
    def _to_bpm(pp):
        v = 60.0 / pp.astype(float)
        return v[np.isfinite(v)]

    plt.figure()
    hr_raw = _to_bpm(pp_raw)
    hr_clean = _to_bpm(pp_clean)
    bins = max(10, int(math.sqrt(max(1, hr_clean.size))))
    plt.hist(hr_raw, bins=bins, alpha=0.5, label="HR raw (BPM)")
    plt.hist(hr_clean, bins=bins, alpha=0.5, label="HR clean (BPM)")
    plt.axvline(hr_low_bpm,  linestyle="dashed", label="HR low")
    plt.axvline(hr_high_bpm, linestyle="dashed", label="HR high")
    plt.xlabel("BPM")
    plt.ylabel("Count")
    plt.title("Instantaneous HR distribution")
    plt.legend()
    if show: plt.show()

def plot_pp_over_time(peak_times, t_pp_mid, pp_clean, artifacts_mask=None):
    """
    peak_times : 1D array of detected beat timestamps (s)
    t_pp_mid   : 1D array of PP mid-times (s)  -> res["t_pp"]
    pp_clean   : 1D array of cleaned PP intervals (s) -> res["pp_clean"]
    artifacts_mask : optional bool array, same length as raw PP (np.diff(peak_times))
                     True where a raw PP was corrected/replaced
    """
    # raw PP from beats
    pp_raw = np.diff(peak_times)
    t_mid_raw = 0.5 * (peak_times[1:] + peak_times[:-1])

    # physio PP bounds from your HR band (Hz)
    pp_min = 1.0 / float(Signal.HR_HIGH)
    pp_max = 1.0 / float(Signal.HR_LOW)

    # default artifacts (if not provided) using your Kubios-like rule
    if artifacts_mask is None or artifacts_mask.size != pp_raw.size:
        from scipy.ndimage import median_filter
        med = median_filter(pp_raw, size=int(PRV.KUBIOS_L), mode='reflect')
        artifacts_mask = np.abs(pp_raw - med) > float(PRV.KUBIOS_THRESHOLD)

    # plot
    plt.figure(figsize=(14,4))
    plt.plot(t_mid_raw, pp_raw, lw=1, label="PP raw")
    plt.plot(t_pp_mid,  pp_clean, lw=2, label="PP clean")
    if artifacts_mask.any():
        plt.scatter(t_mid_raw[artifacts_mask], pp_raw[artifacts_mask],
                    marker='x', label="Artifacts", zorder=3)
    plt.hlines([pp_min, pp_max], xmin=t_mid_raw[0], xmax=t_mid_raw[-1],
               linestyles="dashed", label="Physio bounds")

    # optional: a secondary x-axis in BPM for intuition
    # (PP→HR is nonlinear, so we add a small right y-axis instead)
    ax = plt.gca()
    ax2 = ax.twinx()
    yticks_s = ax.get_yticks()
    ax2.set_ylim(ax.get_ylim())
    with np.errstate(divide='ignore', invalid='ignore'):
        ax2.set_yticks(yticks_s)
        ax2.set_yticklabels([f"{(60.0/y):.0f}" if y>0 else "" for y in yticks_s])
    ax2.set_ylabel("BPM (≈ 60 / PP)")

    plt.xlabel("Time (s) at PP midpoints")
    plt.ylabel("PP interval (s)")
    plt.title("PP intervals over time")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

    return
