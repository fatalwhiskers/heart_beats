# ==========================================
# BVP → windowed HR / HRV / RR (rPPG-friendly)
# + plotting helpers
# ==========================================
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, interpolate
from src.config import rppg, BVP, PRV
# ---------- Plot helpers ----------

def plot_beats(time, signal_bp, peaks_t, title="Bandpassed signal with detected beats", zoom=None):
    """
    Plot a (bandpassed) pulse trace with detected peak markers.
      - time, signal_bp : arrays for the plotted trace (same length)
      - peaks_t         : timestamps (s) of detected peaks
      - zoom            : optional (t0, t1) seconds to zoom into a region
    """
    time = np.asarray(time, float)
    signal_bp = np.asarray(signal_bp, float)
    if time.size != signal_bp.size:
        raise ValueError("time and signal_bp must be same length")

    if zoom is not None:
        t0, t1 = zoom
        m = (time >= t0) & (time <= t1)
    else:
        m = np.ones_like(time, dtype=bool)

    plt.figure()
    plt.plot(time[m], signal_bp[m], label="Bandpassed")
    # mark peaks that fall within the plotted range
    if peaks_t is not None and len(peaks_t):
        pk_mask = (peaks_t >= time[m][0]) & (peaks_t <= time[m][-1])
        if np.any(pk_mask):
            pk_times = np.asarray(peaks_t)[pk_mask]
            pk_vals = np.interp(pk_times, time, signal_bp)
            plt.plot(pk_times, pk_vals, 'o', label="Detected peaks")
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()

def plot_hr_series(t_pp, hr_inst_raw, hr_inst_clean=None, artifacts_mask=None, title="Instantaneous HR"):
    """
    Plot instantaneous HR defined at the mid-times of PP intervals.
      - t_pp           : times (s) for HR samples (midpoint of each PP)
      - hr_inst_raw    : 60/PP_raw (BPM)
      - hr_inst_clean  : optional cleaned/smoothed HR (BPM)
      - artifacts_mask : optional boolean mask for which PP were corrected
    """
    t_pp = np.asarray(t_pp, float)
    hr_inst_raw = np.asarray(hr_inst_raw, float)
    if t_pp.size != hr_inst_raw.size:
        raise ValueError("t_pp and hr_inst_raw must be same length")

    plt.figure()
    plt.plot(t_pp, hr_inst_raw, '.', label="HR inst (raw)")
    if hr_inst_clean is not None:
        plt.plot(t_pp, hr_inst_clean, '-', alpha=0.9, label="HR inst (clean)")
    if artifacts_mask is not None and artifacts_mask.size == hr_inst_raw.size:
        plt.plot(t_pp[artifacts_mask], hr_inst_raw[artifacts_mask], 'x', label="Corrected PP", markersize=8)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("HR (BPM)")
    plt.legend()
    plt.show()

# ---------- Core windowed pipeline ----------

def analyze_bvp_windowed(
    bvp, fs= BVP.BVP_RATE,
    win_sec=rppg.window_size,          # analysis window length (e.g., 60 s)
    step_sec=rppg.step_size,          # hop/overlap (e.g., 5 s)
    interp_fs=PRV.FPS_RESAMPLE_RATE,       # interpolation rate for peak timing
    return_intermediates=False,  # if True, return bandpassed/interpolated traces & peaks
    debug=False
):
    """
    Sliding-window HR / HRV / RR from a BVP trace.

    Steps:
      1) Smooth (5-pt MA) and bandpass (0.7–4 Hz, 128-tap Hamming FIR)
      2) Interpolate to interp_fs (cubic spline)
      3) Peak detect globally (distance ~240 bpm cap; adaptive prominence)
      4) IBI = diff(peak_times), artifact-filter with median ±30% (NC-VT-like)
      5) Windowed metrics (HR, LF/HF, RR via HF peak of Lomb–Scargle)

    Returns
    -------
    results : dict
        't_center' : window center times (s)
        'HR_bpm'   : windowed heart rate (BPM)
        'LF','HF'  : HRV band powers (a.u.)
        'LF_nu','HF_nu' : normalized units (%)
        'LF_HF'    : LF/HF ratio
        'RR_bpm'   : respiratory rate from HF peak (breaths/min)
        'quality'  : simple heartband peakiness (0–1) for each window
        If return_intermediates:
          'time'     : original timebase (s)
          'bvp_bp'   : bandpassed signal
          'time_hi'  : interpolated timebase (s)
          'bvp_hi'   : interpolated signal
          'peak_times': detected beat times (s)
          't_pp'     : IBI mid-times (s)
          'pp_clean' : cleaned IBI series (s)
    """
    bvp = np.asarray(bvp, float)
    n = len(bvp)
    if n < 3:
        raise ValueError("BVP too short")

    fs = float(fs)
    t = np.arange(n) / fs

    # 1) Smooth (5-point moving average)
    smoothed = np.convolve(bvp, np.ones(5)/5, mode='same')

    # 2) Bandpass 0.7–4 Hz (128-tap Hamming FIR)
    numtaps = 128
    bp = signal.firwin(numtaps, [0.7, 4.0], pass_zero=False, fs=fs, window='hamming')
    bvp_bp = signal.filtfilt(bp, [1.0], smoothed, padlen=3*numtaps)

    # 3) Interpolate to interp_fs
    time_hi = np.arange(0, t[-1], 1.0/interp_fs)
    cs = interpolate.CubicSpline(t, bvp_bp, bc_type='natural')
    bvp_hi = cs(time_hi)

    # 4) Peak detection (global)
    min_distance = int(0.25 * interp_fs)  # ~240 bpm cap
    # adaptive prominence: percentile spread of the interpolated signal
    lo, hi = np.percentile(bvp_hi, [10, 90])
    prom = 0.25 * (hi - lo)
    peaks, _ = signal.find_peaks(bvp_hi, distance=min_distance, prominence=max(prom, 1e-6))

    peak_times = time_hi[peaks]
    if peak_times.size < 3:
        raise RuntimeError("Insufficient peaks detected")

    # IBI and mid-times
    pp = np.diff(peak_times)  # seconds (PP = IBI)
    t_pp = 0.5 * (peak_times[1:] + peak_times[:-1])

    # NC-VT-like artifact filter: ±30% of local median (win=5)
    def filter_IBI_ncvt_like(ibi, win=5, tol=0.30):
        ibi = np.asarray(ibi)
        if len(ibi) < 3:
            return ibi, np.ones_like(ibi, dtype=bool)
        valid = np.ones(len(ibi), dtype=bool)
        for i in range(len(ibi)):
            lo_i = max(0, i - win//2)
            hi_i = min(len(ibi), i + win//2 + 1)
            ref = np.median(ibi[lo_i:hi_i])
            valid[i] = ((1 - tol)*ref <= ibi[i] <= (1 + tol)*ref)
        return ibi[valid], valid

    pp_clean, mask_pp = filter_IBI_ncvt_like(pp)
    t_pp = t_pp[mask_pp]

    # Detrend & center tachogram for Lomb
    tach = signal.detrend(pp_clean, type='linear')
    tach = tach - np.mean(tach)

    # window grid
    win = int(round(win_sec * fs))
    step = int(round(step_sec * fs))
    if win <= 0 or step <= 0:
        raise ValueError("win_sec and step_sec must be > 0")

    # Simple heartband quality on bandpassed segment
    def heartband_quality(x, fs, f1=0.7, f2=4.0):
        f, Pxx = signal.welch(x, fs=fs, nperseg=min(1024, len(x)))
        hb = (f >= f1) & (f <= f2)
        if not np.any(hb):
            return 0.0
        peak = np.max(Pxx[hb])
        noise = np.median(Pxx[~hb]) if np.any(~hb) else np.median(Pxx[hb])
        q = peak / (peak + noise + 1e-12)
        return float(np.clip(q, 0, 1))

    # HRV frequency grid
    fmin, fmax = 0.01, 0.5
    freqs = np.linspace(fmin, fmax, 2000)

    def band_power(f, Pxx, f1, f2):
        m = (f >= f1) & (f <= f2)
        if not np.any(m): return 0.0
        return np.trapz(Pxx[m], f[m])

    # Helper: select tachogram points inside [a,b]
    def tach_in_window(a, b):
        sel = (t_pp >= a) & (t_pp <= b)
        return t_pp[sel], tach[sel], pp_clean[sel]  # detrended, and raw-IBI for HR

    t_centers = []
    HR_bpm_list, LF_list, HF_list, LF_nu_list, HF_nu_list, LFHF_list, RR_bpm_list, qual_list = ([] for _ in range(8))

    start = 0
    while start + win <= n:
        a = start / fs
        b = (start + win) / fs
        t_centers.append(0.5*(a+b))

        # Quality from bandpassed segment
        seg_bp = bvp_bp[start:start+win]
        qual_list.append(heartband_quality(seg_bp, fs=fs))

        # Tachogram in window
        tt, tach_win, pp_win = tach_in_window(a, b)

        if len(pp_win) < 6:
            HR_bpm_list.append(np.nan); LF_list.append(np.nan); HF_list.append(np.nan)
            LF_nu_list.append(np.nan); HF_nu_list.append(np.nan); LFHF_list.append(np.nan)
            RR_bpm_list.append(np.nan)
        else:
            # HR from mean IBI (use un-detrended IBIs in window)
            HR_bpm_list.append(60.0 / np.mean(pp_win))

            # Lomb–Scargle PSD (use detrended/centered tachogram)
            P = signal.lombscargle(tt, tach_win, 2*np.pi*freqs, precenter=False, normalize=True)

            LF = band_power(freqs, P, 0.04, 0.15)
            HF = band_power(freqs, P, 0.15, 0.40)
            if (LF + HF) > 0:
                LF_nu = 100.0 * LF / (LF + HF)
                HF_nu = 100.0 * HF / (LF + HF)
            else:
                LF_nu = HF_nu = np.nan
            LFHF = LF / HF if HF > 0 else np.inf

            # RR from HF peak (Hz → breaths/min = 60 * f)
            hf_band = (freqs >= 0.15) & (freqs <= 0.40)
            if np.any(hf_band):
                fHFpeak = freqs[hf_band][np.argmax(P[hf_band])]
                RR_bpm = 60.0 * fHFpeak
            else:
                RR_bpm = np.nan

            LF_list.append(LF); HF_list.append(HF)
            LF_nu_list.append(LF_nu); HF_nu_list.append(HF_nu)
            LFHF_list.append(LFHF); RR_bpm_list.append(RR_bpm)

        start += step

    results = {
        "t_center": np.asarray(t_centers),
        "HR_bpm":   np.asarray(HR_bpm_list),
        "LF":       np.asarray(LF_list),
        "HF":       np.asarray(HF_list),
        "LF_nu":    np.asarray(LF_nu_list),
        "HF_nu":    np.asarray(HF_nu_list),
        "LF_HF":    np.asarray(LFHF_list),
        "RR_bpm":   np.asarray(RR_bpm_list),
        "quality":  np.asarray(qual_list),
    }

    if return_intermediates:
        results.update({
            "time": t,
            "bvp_bp": bvp_bp,
            "time_hi": time_hi,
            "bvp_hi": bvp_hi,
            "peak_times": peak_times,
            "t_pp": t_pp,
            "pp_clean": pp_clean,
        })

    if debug:
        print(f"windows: {len(results['t_center'])}, "
              f"mean HR: {np.nanmean(results['HR_bpm']):.1f} bpm, "
              f"mean RR: {np.nanmean(results['RR_bpm']):.1f} brpm, "
              f"valid windows: {np.sum(np.isfinite(results['HR_bpm']))}")

    return results
