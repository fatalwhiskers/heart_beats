import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.signal import find_peaks
from src.config import Video 
from src.config import Signal
from src.config import PRV
from scipy.ndimage import median_filter


def resample_to_uniform(time_in, signal_in):

    time_in = np.asarray(time_in, float)
    signal_in = np.asarray(signal_in, float)

    if time_in.ndim != 1 or signal_in.ndim != 1 or len(time_in) != len(signal_in):
        # saftey check fince I made mistakes and frames can be skipped when no face detected. if this procs check the read video array and make sure it is the same as time stamps 
        raise ValueError("t_in and x_in must be 1D and same length") 
    
    t0 = float(time_in[0]) # first
    t1 = float(time_in[-1]) # last
 
    new_Time_Grid = np.arange(t0, t1, 1.0/PRV.FPS_SAMPLE_RATE) # the grid to interpolate to

    interpolator_func = PchipInterpolator(time_in, signal_in, extrapolate=False) # shape-preserving cubic (PCHIP = Piecewise Cubic Hermite)
    signal_out = interpolator_func(new_Time_Grid)
    return new_Time_Grid, signal_out

def detect_peaks_rppg(sig):
    
    #relative prominence threshold 0.3 from papare  = filters out small/noisy peaks. # lower this for weak signals increase for lots of movement
    # min_peak_width = 0.12 from paper              = rejects narrow, spiky peaks        

    # Min distance from maximum plausible HR  pulse peaks

    min_distance = int(Video.FPS / Signal.HR_HIGH)
    max_distance = int(Video.FPS / Signal.HR_LOW)   

    # Robust prominence threshold
    lo, hi = np.percentile(sig, [Signal.PROMINENCE_LOWER_BOUND, PRV.PROMINENCE_UPPER_BOUND]) # removes top 5 and bottom 5 outliers
    prom = PRV.PROM_THRESHOLD * (hi - lo) # makes a minmum threshold for the wave length it needs to be higher than this to be used

    sample_width = max(1, int(np.ceil(PRV.MIN_PEAK_WIDTH * Video.FPS))) # physiological lower bound for how wide a real pulse peak should be. this will allow me to remove small peaks

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

    # 1) Resample to uniform grid (shape-preserving)
    new_Time_Grid, sig = resample_to_uniform(time_stamps_raw, signal_raw)

    # 3) Peaks
    sig_samples, details = detect_peaks_rppg(sig)
    peaks_t = new_Time_Grid[sig_samples] 

    # 4) PP intervals and mid-times
    pp_raw, time_at_mid_pp = pp_intervals_from_peaks(peaks_t)

    # 5) Kubios-like artifact correction on PP
    pp_clean, artifacts_mask = kubios_like_pp_filter(pp_raw)

    hr_inst_raw = 60.0 / pp_raw 
    hr_inst_clean = 60.0 / pp_clean 

    return pp_clean, hr_inst_clean, hr_inst_raw

def prv_hr_on_times(t_pp, hr_inst, t_target):
    """
    Resample instantaneous HR (defined at t_pp) onto a desired time array (t_target).
    Uses shape-preserving interpolation; returns NaN outside support.
    """
    if t_pp.size < 2:
        return np.full_like(t_target, np.nan, dtype=float)
    f = PchipInterpolator(t_pp, hr_inst, extrapolate=False)
    return f(t_target)
