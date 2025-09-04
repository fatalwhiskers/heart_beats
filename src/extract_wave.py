import numpy as np
from .config import Signal, Video
from scipy import signal, sparse
from sklearn.decomposition import PCA, FastICA
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, detrend, sosfiltfilt
from scipy.sparse.linalg import spsolve
import numpy as np
import warnings
from sklearn.decomposition import FastICA
from sklearn.exceptions import ConvergenceWarning
from scipy import signal

def extract_rgb_signals_BGR(frames):
    B = frames[:, :, :, 0].mean(axis=(1, 2))
    G = frames[:, :, :, 1].mean(axis=(1, 2))
    R = frames[:, :, :, 2].mean(axis=(1, 2))
    
    return R, G, B

def extract_rgb_signals_BGR_list(frames):
    R, G, B = [], [], []
    for f in frames:
        B.append(f[:, :, 0].mean())
        G.append(f[:, :, 1].mean())
        R.append(f[:, :, 2].mean())
    return np.array(R), np.array(G), np.array(B)

#since doing pca later anyway "standardization"
def zscore_normalize(signal):
    return (signal - np.mean(signal)) / (np.std(signal))

def detrend_zscore(signal):
    """
    Standardize signal to zero mean and unit variance (global).
    """
    return (signal - np.mean(signal)) / (np.std(signal) ) # (np.std(signal) + 1e-8)

def detrend_running_mean(signal, fps, win_sec=1.0):
    """
    Detrend by dividing each sample by its local running mean.
    UBFC-Phys used ~1s window.
    """
    win_len = int(fps * win_sec)
    if win_len < 1:
        win_len = 1

    # Compute running mean with convolution
    kernel = np.ones(win_len) / win_len
    local_mean = np.convolve(signal, kernel, mode='same')

    # Avoid divide by zero
    local_mean[local_mean == 0] = 1e-8

    detrended = signal / local_mean
    return detrended - np.mean(detrended)  # optional re-centering

def bandpass_filter_old(signal, fs, lowcut=Signal.HR_LOW, highcut=Signal.HR_HIGH, order=Signal.HR_ORDER):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def bandpass_filter(signal, fs=Video.FPS,
                    lowcut=Signal.HR_LOW,
                    highcut=Signal.HR_HIGH,
                    order=Signal.HR_ORDER):
    """
    NaN-safe bandpass for rPPG using your variables.
    - Filters each contiguous finite segment of `signal` separately.
    - Leaves gaps as NaN.
    - Validates cutoffs based on `fs`.
    """
    signal = np.asarray(signal, float)
    filtered = np.full_like(signal, np.nan, dtype=float)

    # guard: bad fs
    if not np.isfinite(fs) or fs <= 0:
        return filtered

    nyq = 0.5 * float(fs)
    eps = 1e-6
    low = max(float(lowcut), eps)
    high = min(float(highcut), nyq - eps)
    if not (low < high):
        # invalid band for this fs → return NaNs
        return filtered

    # normalized band + reasonable order
    wn = [low / nyq, high / nyq]
    order = int(max(1, min(int(order), 6)))
    sos = butter(order, wn, btype='band', output='sos')

    # contiguous finite segments (no NaNs)
    finite = np.isfinite(signal)
    if not finite.any():
        return filtered

    edges = np.flatnonzero(np.diff(np.r_[False, finite, False]))
    segs = list(zip(edges[0::2], edges[1::2]))

    # need ~2 s minimum per segment for stable zero-phase filtering
    min_len = max(8, int(round(2.0 * fs)))

    for s, e in segs:
        if (e - s) < min_len:
            continue

        L = e - s
        seg = signal[s:e].astype(float, copy=False)
        dc  = np.nanmedian(seg)
        x   = seg - dc

        b, a = butter(order, [low/nyq, high/nyq], btype='band', output='ba')
        padlen = min(L - 1, max(1, 3 * (max(len(a), len(b)) - 1)))
        y = filtfilt(b, a, x, padlen=padlen)

        filtered[s:e] = y  # (omit +dc for a true band-pass)

    return filtered

def get_heart_rate(signal, fs, low=0.7, high=4.0):
    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=1/fs)
    fft_values = np.abs(np.fft.rfft(signal))

    # Limit to heart rate range (in Hz)
    valid = (freqs >= low) & (freqs <= high)
    freqs = freqs[valid]
    fft_values = fft_values[valid]

    # Find peak frequency
    peak_freq = freqs[np.argmax(fft_values)]
    bpm = peak_freq * 60  # Convert Hz to BPM
    return bpm

import numpy as np

def interpolate_signal_with_timestamps(signal,
                                       timestamps,
                                       target_fps=None,
                                       t_uniform=None,
                                       max_gap_sec=0.5):
    """
    Interpolate a (possibly NaN-containing) time series to a uniform time grid.

    - If t_uniform is None: builds a grid using target_fps; if target_fps is None or <=0,
      it uses the median dt from timestamps.
    - No edge extrapolation: samples outside the convex hull of valid data are NaN.
    - Long gaps (> max_gap_sec between surrounding valid samples) are set to NaN.

    Returns: x_uniform, t_uniform
    """
    x = np.asarray(signal, dtype=float)
    t = np.asarray(timestamps, dtype=float)

    if x.size != t.size or x.size < 2:
        return np.array([]), np.array([])

    # sort by time; drop non-finite timestamps
    order = np.argsort(t)
    t = t[order]
    x = x[order]
    finite_t = np.isfinite(t)
    t = t[finite_t]
    x = x[finite_t]
    if t.size < 2:
        return np.array([]), np.array([])

    # deduplicate timestamps (keep first)
    t, keep_idx = np.unique(t, return_index=True)
    x = x[keep_idx]

    # build/accept uniform grid
    if t_uniform is None:
        if target_fps is None or target_fps <= 0:
            dt = float(np.median(np.diff(t)))
        else:
            dt = 1.0 / float(target_fps)
        if not np.isfinite(dt) or dt <= 0:
            dt = 1.0 / 35.0  # safe default
        t_uniform = np.arange(t[0], t[-1] + 1e-9, dt)
    else:
        t_uniform = np.asarray(t_uniform, dtype=float)

    # use only valid (non-NaN) samples for interpolation
    valid = np.isfinite(x)
    if valid.sum() < 2:
        return np.full_like(t_uniform, np.nan, dtype=float), t_uniform

    tv = t[valid]
    xv = x[valid]

    # linear interpolation on the uniform grid
    x_uniform = np.interp(t_uniform, tv, xv)

    # mask outside valid-data span (no edge extrapolation)
    outside = (t_uniform < tv[0]) | (t_uniform > tv[-1])
    x_uniform[outside] = np.nan

    # re-NaN long gaps by checking distance between bracketing valid samples
    # ensure indices are inside [1, len(tv)-1] so we always have a pair
    idx_right = np.searchsorted(tv, t_uniform, side='left')
    idx_right = np.clip(idx_right, 1, len(tv) - 1)
    idx_left = idx_right - 1
    span = tv[idx_right] - tv[idx_left]
    x_uniform[span > max_gap_sec] = np.nan

    return x_uniform, t_uniform



def extract_pca_components(R, G, B, n_components=3):
    X = np.column_stack([R, G, B]).astype(float)        # shape (T,3)
    comps = np.full((len(X), n_components), np.nan)     # output aligned to timeline

    # rows where all three channels are finite
    mask = np.isfinite(X).all(axis=1)
    if mask.sum() < n_components + 1:
        print(f"[PCA] Not enough finite rows: {mask.sum()}")
        return comps

    pca = PCA(n_components=n_components)
    comps_mask = pca.fit_transform(X[mask])             # fit/transform only valid rows
    comps[mask] = comps_mask
    return comps


def zca_whiten(R, G, B, epsilon=1e-5):
    X = np.column_stack([R, G, B]).astype(float)        # (T,3)
    Xz = np.full_like(X, np.nan)
    mask = np.isfinite(X).all(axis=1)
    if mask.sum() < 3:
        print(f"[ZCA] Not enough finite rows: {mask.sum()}")
        return Xz

    Xm = X[mask] - np.mean(X[mask], axis=0)
    # covariance (3x3)
    sigma = np.cov(Xm, rowvar=False)
    # regularize a touch to avoid tiny/negative eigenvalues
    U, S, _ = np.linalg.svd(sigma + epsilon * np.eye(sigma.shape[0]), full_matrices=False)
    W = U @ np.diag(1.0 / np.sqrt(S)) @ U.T
    Xz[mask] = Xm @ W.T
    return Xz   


def ICA_Test(R, G, B, fs= Video.FPS, max_iter=1000):
    X = np.column_stack([R, G, B]).astype(float)  # (T,3)
    S = np.full_like(X, np.nan)                   # sources aligned to timeline

    mask = np.isfinite(X).all(axis=1)
    n_valid = int(mask.sum())
    if n_valid < 10:
        print(f"[ICA] Not enough finite rows: {n_valid}")
        return S, (None, None)

    # sklearn version differences: whiten=True (old), 'unit-variance' (new)
    try:
        ica = FastICA(n_components=3, whiten='unit-variance',
                      max_iter=max_iter, tol=1e-4, random_state=0)
    except TypeError:
        ica = FastICA(n_components=3, whiten=True,
                      max_iter=max_iter, tol=1e-4, random_state=0)

    with warnings.catch_warnings(record=True) as wlist:
        warnings.filterwarnings("always", category=ConvergenceWarning)
        try:
            S_mask = ica.fit_transform(X[mask])
        except Exception as e:
            print(f"[ICA ERROR] FastICA failed: {e}")
            return S, (None, None)
        for w in wlist:
            if issubclass(w.category, ConvergenceWarning):
                print(f"[ICA WARNING] {w.message} (n_iter={getattr(ica, 'n_iter_', 'unknown')})")

    S[mask] = S_mask

    # Welch PSD per component on the finite segment only
    n = n_valid
    # pick a reasonable segment length (e.g., ~8 s) but cap by available samples
    nperseg = max(64, min(int(round(8 * fs)), n))
    freqs, psd = signal.welch(S_mask, fs=fs, nperseg=nperseg, axis=0)
    return S, (freqs, psd)


def detrend_sig(signal):
    return detrend(signal)

def save_rgb_signals(file_path, R, G, B):
    np.savez(file_path, R=R, G=G, B=B)

def load_rgb_signals(file_path):
    data = np.load(file_path)
    return data['R'], data['G'], data['B']

def chrom_pos_windowed(R, G, B, win_sec=2, step_sec=0.8, method='CHROM', fps=35):
    if fps is None: fps = float(Video.FPS)
    R, G, B = map(lambda x: (x - np.mean(x)) / (np.std(x) + 1e-8), [R,G,B])
    n = len(R)
    w = max(8, int(win_sec * fps))
    s = max(1, int(step_sec * fps))
    out = np.zeros(n, float); wsum = np.zeros(n, float)
    hann = np.hanning(w)
    for start in range(0, n - w + 1, s):
        r = R[start:start+w]; g = G[start:start+w]; b = B[start:start+w]
        if method == 'CHROM':
            x = 3*r - 2*g
            y = 1.5*r + g - 1.5*b
            a = (np.std(x)+1e-8) / (np.std(y)+1e-8)
            y = x - a*y
        else:  # POS
            y1 = g - b
            y2 = g + b - 2*r
            a  = (np.std(y1)+1e-8) / (np.std(y2)+1e-8)
            y  = y1 + a*y2
        y = (y - np.mean(y)) / (np.std(y) + 1e-8)
        y = y * hann
        out[start:start+w] += y
        wsum[start:start+w] += hann
    return np.divide(out, np.maximum(wsum, 1e-8))

def chrom_pos_windowed_nan(R, G, B, win_sec=2, step_sec=0.8, method='CHROM', fps=35):
    """
    NaN-safe CHROM/POS windowing.

    - Ignores samples where any of R, G, or B is NaN/inf for that frame.
    - Window mean/std are computed only over valid samples.
    - Windows with <2 valid samples contribute nothing.
    - Positions with no coverage are returned as NaN.
    """
    EPS = 1e-8

    if fps is None:
        fps = float(Video.FPS)

    R = np.asarray(R, dtype=float)
    G = np.asarray(G, dtype=float)
    B = np.asarray(B, dtype=float)

    def zscore_nan(x):
        m = np.nanmean(x)
        s = np.nanstd(x)
        if not np.isfinite(m): m = 0.0
        if not np.isfinite(s) or s < 1e-12: s = 1.0
        return (x - m) / (s + EPS)

    R = zscore_nan(R)
    G = zscore_nan(G)
    B = zscore_nan(B)

    n = len(R)
    w = max(8, int(round(win_sec * fps)))
    s = max(1, int(round(step_sec * fps)))

    out  = np.zeros(n, float)
    wsum = np.zeros(n, float)
    if w <= 1 or n == 0:
        return np.full(n, np.nan, float)

    hann = np.hanning(w)

    mth = (method or 'CHROM').upper()
    if mth not in ('CHROM', 'POS'):
        raise ValueError("method must be 'CHROM' or 'POS'")

    for start in range(0, n - w + 1, s):
        sl = slice(start, start + w)
        r = R[sl]; g = G[sl]; b = B[sl]

        valid = np.isfinite(r) & np.isfinite(g) & np.isfinite(b)
        idx = np.nonzero(valid)[0]
        if idx.size < 2:
            continue

        if mth == 'CHROM':
            x = 3.0 * r - 2.0 * g
            y = 1.5 * r + 1.0 * g - 1.5 * b
            a = (np.nanstd(x[valid]) + EPS) / (np.nanstd(y[valid]) + EPS)
            y = x - a * y
        else:  # POS
            y1 = g - b
            y2 = g + b - 2.0 * r
            a  = (np.nanstd(y1[valid]) + EPS) / (np.nanstd(y2[valid]) + EPS)
            y  = y1 + a * y2

        ym = np.nanmean(y[valid])
        ys = np.nanstd(y[valid])
        if not np.isfinite(ys) or ys < 1e-12:
            continue

        yz = np.zeros_like(y)
        yz[valid] = (y[valid] - ym) / (ys + EPS)

        abs_idx = start + idx
        h_idx = hann[idx]
        out[abs_idx]  += yz[idx] * h_idx
        wsum[abs_idx] += h_idx

    # Return NaN where there was no coverage
    res = np.full(n, np.nan, float)
    mask = wsum > 0
    res[mask] = out[mask] / (wsum[mask] + EPS)
    return res

#Tarvainen, M. P., Ranta-aho, P. O., & Karjalainen, P. A. (2002). An advanced detrending method with application to HRV analysis. IEEE Transactions on Biomedical Engineering, 49(2). https://doi.org/10.1109/10.979357

def detrend_signal(signal, lambda_param=300):
    """Detrend a signal using smoothness priors (Tarvainen et al. 2002)."""
    num_samples = len(signal)

    # Identity matrix (size: num_samples x num_samples)
    identity_matrix = sparse.eye(num_samples, format='csc')

    # Second-order difference matrix D: shape (num_samples-2, num_samples)
    e = np.ones(num_samples)
    diff_matrix = sparse.diags([e, -2*e, e], [0, 1, 2],
                               shape=(num_samples-2, num_samples), format='csc')

    # Construct the smoothing operator (I + lambda^2 * D^T D)
    smoothing_matrix = identity_matrix + (lambda_param**2) * (diff_matrix.T @ diff_matrix)

    # Solve the linear system to obtain the trend estimate
    trend_estimate = spsolve(smoothing_matrix, signal)

    # Subtract the trend from the original signal
    detrended_signal = signal - trend_estimate

    return detrended_signal

def sliding_mean_normalize(x, fs, win_sec=1.5, eps=1e-8):
    n = max(1, int(round(win_sec * fs)))
    x = np.asarray(x, float)
    m = np.isfinite(x).astype(float)
    x0 = np.nan_to_num(x, nan=0.0)

    # moving sums (ignore NaNs via mask)
    kern = np.ones(n, float)
    num = np.convolve(x0, kern, mode='same')
    den = np.convolve(m,  kern, mode='same')

    ma = num / np.maximum(den, eps)
    y = (x - ma) / np.maximum(ma, eps)

    # if a window had no valid samples, keep NaN there
    y[den < 1] = np.nan
    return y