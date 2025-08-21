import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.signal import find_peaks
from src.config import Video 
from src.config import Signal

# ----------------------------
# Utilities
# ----------------------------
def resample_to_uniform(time_in, signal_in, fps_to_sample=128.0):

    time_in = np.asarray(time_in, float)
    signal_in = np.asarray(signal_in, float)

    if time_in.ndim != 1 or signal_in.ndim != 1 or len(time_in) != len(signal_in):
        # saftey check fince I made mistakes and frames can be skipped when no face detected. if this procs check the read video array and make sure it is the same as time stamps 
        raise ValueError("t_in and x_in must be 1D and same length") 
    
    t0 = float(time_in[0]) # first
    t1 = float(time_in[-1]) # last
 
    new_Time_Grid = np.arange(t0, t1, 1.0/fps_to_sample) # the grid to interpolate to

    interpolator_func = PchipInterpolator(time_in, signal_in, extrapolate=False) # shape-preserving cubic (PCHIP = Piecewise Cubic Hermite)
    signal_out = interpolator_func(new_Time_Grid)
    return new_Time_Grid, signal_out

def detect_peaks_rppg(sig, fps=Video.FPS, prom_threshold=0.3, min_peak_width=0.12):
    
    #relative prominence threshold 0.3 from papare  = filters out small/noisy peaks. # lower this for weak signals increase for lots of movement
    # min_peak_width = 0.12 from paper              = rejects narrow, spiky peaks        

    # Min distance from maximum plausible HR  pulse peaks

    min_distance = int(fps / Signal.HR_HIGH)
    max_distance = int(fps / Signal.HR_LOW)   

    # Robust prominence threshold
    lo, hi = np.percentile(sig, [5, 95]) # removes top 5 and bottom 5 outliers
    prom = prom_threshold * (hi - lo) # makes a minmum threshold for the wave length it needs to be higher than this to be used

    sample_width = max(1, int(np.ceil(min_peak_width * fps))) # physiological lower bound for how wide a real pulse peak should be. this will allow me to remove small peaks

    """
    this only keeps peaks which survive 3 tests

    distance = Don’t place beats too close (physiology: max HR).

    prominence = Only count peaks that actually stand out (SNR).
    prominence = How much the peak stands out above its surroundings

    width = Only count peaks that are broad enough to be real pulses (shape).
    """
    sig_samples, details = find_peaks(sig, distance=min_distance, prominence=prom, width=sample_width)
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
    t_mid = 0.5 * (t_peaks[1:] + t_peaks[:-1])  # time at which PP is defined
    return pp, t_mid

def kubios_like_pp_filter(pp, L=51, t_thresh=0.15):
    """
    Median-based artifact correction (Kubios-like).
    Replace PP values deviating > t_thresh (s) from running median (window L).
    """

    half = (L - 1)//2
    # Mirror padding
    if pp.size > half + 1:
        left = (2*pp[0] - pp[1:half+1][::-1])
        right = (2*pp[-1] - pp[-half-1:-1][::-1])
    else:
        left = np.full(half, pp[0])
        right = np.full(half, pp[-1])
    pad = np.r_[left, pp, right]

    # Rolling median (pure numpy implementation)
    med = np.empty_like(pad)
    for i in range(pad.size):
        s = max(0, i - half)
        e = min(pad.size, i + half + 1)
        med[i] = np.median(pad[s:e])

    med = med[half:half+pp.size]

    pp_f = pp.copy()
    mask = np.abs(pp - med) > t_thresh
    pp_f[mask] = med[mask]
    return pp_f, mask

# ----------------------------
# Main: compute PRV HR
# ----------------------------
def compute_prv_hr(
    time_stamps_raw, signal_raw,
    resample_fs=128.0, #from paper
    prom_pct=0.3, #from paper
    min_width_s=0.12, #from paper
    kubios_L=51, #from paper
    kubios_thresh=0.15 #from paper
):

    # 1) Resample to uniform grid (shape-preserving)
    t_u, sig, fs = resample_to_uniform(time_stamps_raw, signal_raw, fs_out=resample_fs)

    # 3) Peaks
    peaks_idx, _ = detect_peaks_rppg(
        sig, fs,
        hr_min=Signal.HR_LOW, hr_max=Signal.HR_HIGH,
        prom_pct=prom_pct, min_width_s=min_width_s
    )
    peaks_t = t_u[peaks_idx] if peaks_idx.size else np.array([])

    # 4) PP intervals and mid-times
    pp_raw, t_pp = pp_intervals_from_peaks(peaks_t)

    # 5) Kubios-like artifact correction on PP
    pp_clean, art_mask = kubios_like_pp_filter(pp_raw, L=kubios_L, t_thresh=kubios_thresh)

    # 6) Instantaneous HR (BPM) at t_pp
    hr_inst_raw = 60.0 / pp_raw if pp_raw.size else np.array([])
    hr_inst_clean = 60.0 / pp_clean if pp_clean.size else np.array([])

    return {
        "t_u": t_u, "pulse_u": signal_raw, "fs": fs,
        "peaks_idx": peaks_idx, "peaks_t": peaks_t,
        "pp_raw": pp_raw, "t_pp": t_pp,
        "pp_clean": pp_clean, "art_mask": art_mask,
        "hr_inst_raw": hr_inst_raw, "hr_inst_clean": hr_inst_clean
    }

# ----------------------------
# Optional: align PRV HR to an external timeline (e.g., your FFT HR times)
# ----------------------------
def prv_hr_on_times(t_pp, hr_inst, t_target):
    """
    Resample instantaneous HR (defined at t_pp) onto a desired time array (t_target).
    Uses shape-preserving interpolation; returns NaN outside support.
    """
    if t_pp.size < 2:
        return np.full_like(t_target, np.nan, dtype=float)
    f = PchipInterpolator(t_pp, hr_inst, extrapolate=False)
    return f(t_target)
