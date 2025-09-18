from __future__ import annotations

import warnings
import numpy as np
from scipy import signal as sp_signal, sparse
from scipy.interpolate import interp1d, PchipInterpolator, CubicSpline
from scipy.signal import butter, filtfilt, detrend, sosfiltfilt
from scipy.sparse.linalg import spsolve
from sklearn.decomposition import PCA, FastICA
from sklearn.exceptions import ConvergenceWarning

from .config import Signal, Video, rppg, POS, PRV



def extract_rgb_signals_BGR(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return frame-wise mean R, G, B from a 4D BGR array [T, H, W, C]."""
    b_chan = frames[:, :, :, 0].mean(axis=(1, 2))
    g_chan = frames[:, :, :, 1].mean(axis=(1, 2))
    r_chan = frames[:, :, :, 2].mean(axis=(1, 2))
    return r_chan, g_chan, b_chan


def extract_rgb_signals_BGR_list(frames: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return frame-wise mean R, G, B from a list of BGR images."""
    r_vals, g_vals, b_vals = [], [], []
    for frm in frames:
        b_vals.append(frm[:, :, 0].mean())
        g_vals.append(frm[:, :, 1].mean())
        r_vals.append(frm[:, :, 2].mean())
    return np.array(r_vals), np.array(g_vals), np.array(b_vals)


def zscore_normalize(x: np.ndarray) -> np.ndarray:
    """Return z-scored signal."""
    x = np.asarray(x, float)
    return (x - np.mean(x)) / (np.std(x) + 1e-12)


def detrend_ma_subtract(x: np.ndarray, win_sec: float = 15.0, fps: float | None = None) -> np.ndarray:
    """Return x minus its centered moving average (preserves channel ratios)."""
    fs_hz = float(fps if fps is not None else Video.FPS)
    x = np.asarray(x, float)
    win = max(1, int(round(fs_hz * win_sec)))
    if win % 2 == 0:
        win += 1
    pad = win // 2
    x_pad = np.pad(x, (pad, pad), mode="reflect")
    ma = np.convolve(x_pad, np.ones(win) / win, mode="valid")
    return x - ma


def smoothness_priors_detrend(y: np.ndarray, lam: float = 10.0) -> np.ndarray:
    """Tarvainen-style smoothness priors detrending; returns y − trend."""
    y = np.asarray(y, float).reshape(-1)
    n = y.size
    eye_n = sparse.eye(n, format="csc")
    diagonals = [np.ones(n - 2), -2 * np.ones(n - 2), np.ones(n - 2)]
    d2 = sparse.diags(diagonals, offsets=[0, 1, 2], shape=(n - 2, n), format="csc")
    a_mat = eye_n + (lam ** 2) * (d2.T @ d2)
    trend = spsolve(a_mat, y)
    return y - trend

def upsample_cubic(
    input_signal: np.ndarray,
    fs_output: float = PRV.FPS_RESAMPLE_RATE,
    bc_type: str = "natural",
) -> tuple[np.ndarray, np.ndarray]:
    """Upsample a uniformly sampled 1D signal from Video.FPS to fs_output using CubicSpline."""
    fs_input = float(Video.FPS)
    x = np.asarray(input_signal, float).ravel()
    if x.ndim != 1:
        raise ValueError("input_signal must be 1D")
    if not np.isfinite(fs_input) or fs_input <= 0:
        raise ValueError("fs_input must be positive and finite")
    if not np.isfinite(x).all():
        raise ValueError("input_signal contains non-finite values")

    n = x.size
    t_orig = np.arange(n, dtype=float) / fs_input
    dt_out = 1.0 / float(fs_output)
    t_end = t_orig[-1]
    t_resamp = np.arange(0.0, t_end + 0.5 * dt_out, dt_out)

    spl = CubicSpline(t_orig, x, bc_type=bc_type, extrapolate=False)
    y_resamp = spl(t_resamp)
    return t_resamp, y_resamp


def interpolate_signal_with_timestamps(
    x: np.ndarray,
    timestamps: np.ndarray,
    target_fps: float = 35.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate (timestamps, x) to a uniform grid at target_fps."""
    t = np.asarray(timestamps, float).ravel()
    x = np.asarray(x, float).ravel()
    t0, t1 = float(t[0]), float(t[-1])
    n_target = max(1, int((t1 - t0) * target_fps))
    t_uniform = np.linspace(t0, t1, n_target)
    f = interp1d(t, x, kind="linear", fill_value="extrapolate", assume_sorted=True)
    y_uniform = f(t_uniform)
    return y_uniform, t_uniform


def resample_rgb_pchip(
    R: np.ndarray,
    G: np.ndarray,
    B: np.ndarray,
    timestamps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Resample R,G,B to a uniform grid at config Video.target_FPS via PCHIP (NaN-safe, dedups timestamps)."""
    t = np.asarray(timestamps, float)
    R = np.asarray(R, float)
    G = np.asarray(G, float)
    B = np.asarray(B, float)

    order = np.argsort(t)
    t, R, G, B = t[order], R[order], G[order], B[order]
    m = np.isfinite(t) & np.isfinite(R) & np.isfinite(G) & np.isfinite(B)
    t, R, G, B = t[m], R[m], G[m], B[m]
    if t.size < 2:
        raise ValueError("Need at least two valid samples")

    if np.any(np.diff(t) == 0):
        uniq_t, idx, counts = np.unique(t, return_inverse=True, return_counts=True)

        def collapse(arr: np.ndarray) -> np.ndarray:
            return np.bincount(idx, weights=arr) / counts

        R, G, B, t = collapse(R), collapse(G), collapse(B), uniq_t

    fs_out = float(Video.target_FPS)
    dt = 1.0 / fs_out
    t_uniform = np.arange(t[0], t[-1] + 0.5 * dt, dt)
    t_uniform = t_uniform[t_uniform <= t[-1]]

    fR = PchipInterpolator(t, R, extrapolate=False)
    fG = PchipInterpolator(t, G, extrapolate=False)
    fB = PchipInterpolator(t, B, extrapolate=False)

    Rr = fR(t_uniform)
    Gr = fG(t_uniform)
    Br = fB(t_uniform)

    keep = np.isfinite(Rr) & np.isfinite(Gr) & np.isfinite(Br)
    return Rr[keep], Gr[keep], Br[keep], t_uniform[keep]


def normalize(x: np.ndarray) -> np.ndarray:
    """Return zero-mean, unit-variance signal."""
    x = np.asarray(x, float)
    return (x - np.mean(x)) / (np.std(x) + 1e-12)


def detrend_running_mean(x: np.ndarray, win_sec: float = 1.0) -> np.ndarray:
    """Divide by local running mean (reflect-pad), then subtract 1."""
    fs_hz = float(Video.FPS)
    eps = 1e-8
    x = np.asarray(x, float).ravel()
    win = max(1, int(round(fs_hz * win_sec)))
    if win % 2 == 0:
        win += 1
    half = win // 2
    x_pad = np.pad(x, (half, half), mode="reflect")
    kernel = np.ones(win, float) / win
    local_mean = np.convolve(x_pad, kernel, mode="valid")
    return x / (local_mean + eps) - 1.0


def detrend_sig(x: np.ndarray) -> np.ndarray:
    """Wrapper around scipy.signal.detrend."""
    return detrend(np.asarray(x, float))


def detrend_luminance_only_ma(
    rgb: np.ndarray,
    fps: float,
    win_sec: float = 15.0,
    keep_mean: bool = True,
    clip: bool = False,
) -> np.ndarray:
    """Detrend only Y in YUV via moving-average subtraction, then back to RGB."""
    rgb = np.asarray(rgb, float)
    if rgb.ndim != 2 or rgb.shape[1] != 3:
        raise ValueError("rgb must be (N, 3)")

    M_rgb2yuv = np.array(
        [[0.299, 0.587, 0.114], [-0.147, -0.289, 0.436], [0.615, -0.515, -0.100]]
    )
    M_yuv2rgb = np.linalg.inv(M_rgb2yuv)

    yuv = rgb @ M_rgb2yuv.T
    y, u, v = yuv[:, 0], yuv[:, 1], yuv[:, 2]

    y_trend = y - detrend_ma_subtract(y, win_sec=win_sec, fps=fps)
    y_detr = y - y_trend
    if keep_mean:
        y_detr += np.mean(y_trend)

    yuv_d = np.column_stack([y_detr, u, v])
    rgb_out = yuv_d @ M_yuv2rgb.T

    if clip:
        lo, hi = rgb.min(), rgb.max()
        rgb_out = np.clip(rgb_out, lo, hi)
    return rgb_out

def bandpass_filter_old(
    x: np.ndarray,
    fs: float,
    lowcut: float = Signal.HR_LOW,
    highcut: float = Signal.HR_HIGH,
    order: int = Signal.HR_ORDER,
) -> np.ndarray:
    """Classic zero-phase band-pass."""
    nyq = 0.5 * fs
    wn = [lowcut / nyq, highcut / nyq]
    b, a = butter(order, wn, btype="band")
    return filtfilt(b, a, np.asarray(x, float))


def bandpass_filter(
    x: np.ndarray,
    lowcut: float = Signal.HR_LOW,
    highcut: float = Signal.HR_HIGH,
    order: int = Signal.HR_ORDER,
) -> np.ndarray:
    """NaN-safe zero-phase band-pass on finite segments; leaves gaps as NaN."""
    fs_hz = float(Video.FPS)
    x = np.asarray(x, float)
    y = np.full_like(x, np.nan, dtype=float)
    if not np.isfinite(fs_hz) or fs_hz <= 0:
        return y

    nyq = 0.5 * fs_hz
    eps = 1e-6
    lo = max(float(lowcut), eps)
    hi = min(float(highcut), nyq - eps)
    if not (lo < hi):
        return y

    lo_n, hi_n = lo / nyq, hi / nyq
    order = int(max(1, min(int(order), 6)))
    b, a = butter(order, [lo_n, hi_n], btype="band", output="ba")

    finite = np.isfinite(x)
    if not finite.any():
        return y

    edges = np.flatnonzero(np.diff(np.r_[False, finite, False]))
    segs = list(zip(edges[0::2], edges[1::2]))
    min_len = max(8, int(round(2.0 * fs_hz)))

    for s, e in segs:
        if (e - s) < min_len:
            continue
        seg = x[s:e].astype(float, copy=False)
        seg_dc = np.nanmedian(seg)
        seg = seg - seg_dc
        padlen = min((e - s) - 1, max(1, 3 * (max(len(a), len(b)) - 1)))
        y[s:e] = filtfilt(b, a, seg, padlen=padlen)
    return y


def bandpass_filter_g(
    x: np.ndarray,
    lowcut: float = Signal.HR_LOW,
    highcut: float = Signal.HR_HIGH,
    order: int = Signal.HR_ORDER,
) -> np.ndarray:
    """NaN-safe zero-phase band-pass using SOS for numerical stability."""
    fs_hz = float(Video.FPS)
    x = np.asarray(x, float)
    y = np.full_like(x, np.nan, dtype=float)
    if not np.isfinite(fs_hz) or fs_hz <= 0:
        return y

    nyq = 0.5 * fs_hz
    eps = 1e-6
    lo = max(float(lowcut), eps)
    hi = min(float(highcut), nyq - eps)
    if not (lo < hi):
        return y

    wn = [lo / nyq, hi / nyq]
    order = int(max(1, min(int(order), 6)))
    sos = butter(order, wn, btype="band", output="sos")

    finite = np.isfinite(x)
    if not finite.any():
        return y

    edges = np.flatnonzero(np.diff(np.r_[False, finite, False]))
    segs = list(zip(edges[0::2], edges[1::2]))
    min_len = max(8, int(round(2.0 * fs_hz)))

    for s, e in segs:
        if (e - s) < min_len:
            continue
        seg = x[s:e].astype(float, copy=False)
        seg_dc = np.nanmedian(seg)
        seg = seg - seg_dc
        padlen = min((e - s) - 1, max(1, 3 * (2 * order)))
        y[s:e] = sosfiltfilt(sos, seg, padlen=padlen)
    return y


def get_heart_rate(x: np.ndarray, fs: float, low: float = 0.7, high: float = 4.0) -> float:
    """Return HR (BPM) from max FFT peak within [low, high] Hz."""
    x = np.asarray(x, float)
    n = x.size
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    spectrum = np.abs(np.fft.rfft(x))
    band = (freqs >= low) & (freqs <= high)
    if not band.any():
        return float("nan")
    peak_hz = freqs[band][np.argmax(spectrum[band])]
    return float(60.0 * peak_hz)

def extract_pca_components(R: np.ndarray, G: np.ndarray, B: np.ndarray, n_components: int = 3) -> np.ndarray:
    """Return PCA components aligned to full timeline; NaN rows are skipped for fit/transform."""
    X = np.column_stack([R, G, B]).astype(float)
    comps = np.full((len(X), n_components), np.nan)
    mask = np.isfinite(X).all(axis=1)
    if mask.sum() < n_components + 1:
        print(f"[PCA] Not enough finite rows: {mask.sum()}")
        return comps
    pca = PCA(n_components=n_components)
    comps_mask = pca.fit_transform(X[mask])
    comps[mask] = comps_mask
    return comps


def zca_whiten(R: np.ndarray, G: np.ndarray, B: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
    """Return ZCA-whitened channels aligned to full timeline; NaN rows are skipped."""
    X = np.column_stack([R, G, B]).astype(float)
    Xz = np.full_like(X, np.nan)
    mask = np.isfinite(X).all(axis=1)
    if mask.sum() < 3:
        print(f"[ZCA] Not enough finite rows: {mask.sum()}")
        return Xz
    Xm = X[mask] - np.mean(X[mask], axis=0)
    sigma = np.cov(Xm, rowvar=False)
    U, S, _ = np.linalg.svd(sigma + epsilon * np.eye(sigma.shape[0]), full_matrices=False)
    W = U @ np.diag(1.0 / np.sqrt(S)) @ U.T
    Xz[mask] = Xm @ W.T
    return Xz


def ICA_Test(R: np.ndarray, G: np.ndarray, B: np.ndarray, max_iter: int = 1000):
    """Run FastICA on RGB; return sources aligned to full timeline and Welch PSD (freqs, psd)."""
    X = np.column_stack([R, G, B]).astype(float)
    S = np.full_like(X, np.nan)
    fs_hz = float(Video.FPS)
    mask = np.isfinite(X).all(axis=1)
    n_valid = int(mask.sum())
    if n_valid < 10:
        print(f"[ICA] Not enough finite rows: {n_valid}")
        return S, (None, None)

    try:
        ica = FastICA(n_components=3, whiten="unit-variance", max_iter=max_iter, tol=1e-4, random_state=0)
    except TypeError:
        ica = FastICA(n_components=3, whiten=True, max_iter=max_iter, tol=1e-4, random_state=0)

    with warnings.catch_warnings(record=True) as wlist:
        warnings.filterwarnings("always", category=ConvergenceWarning)
        try:
            S_mask = ica.fit_transform(X[mask])
        except Exception as exc:
            print(f"[ICA ERROR] FastICA failed: {exc}")
            return S, (None, None)
        for w in wlist:
            if issubclass(w.category, ConvergenceWarning):
                print(f"[ICA WARNING] {w.message} (n_iter={getattr(ica, 'n_iter_', 'unknown')})")

    S[mask] = S_mask

    nperseg = max(64, min(int(round(8 * fs_hz)), n_valid))
    freqs, psd = sp_signal.welch(S_mask, fs=fs_hz, nperseg=nperseg, axis=0)
    return S, (freqs, psd)

def pos_windowed(R: np.ndarray, G: np.ndarray, B: np.ndarray, win_sec: float = 1.6) -> np.ndarray:
    """Overlap-accumulate POS projection using per-window mean normalization."""
    fs_hz = float(Video.FPS)
    eps = 1e-9
    R = np.asarray(R, float)
    G = np.asarray(G, float)
    B = np.asarray(B, float)
    n = len(R)
    w = max(1, min(int(round(win_sec * fs_hz)), n))
    out = np.zeros(n, float)

    for n_idx in range(w, n):
        m_idx = n_idx - w + 1
        r = R[m_idx : n_idx + 1] / (R[m_idx : n_idx + 1].mean() + eps) - 1.0
        g = G[m_idx : n_idx + 1] / (G[m_idx : n_idx + 1].mean() + eps) - 1.0
        b = B[m_idx : n_idx + 1] / (B[m_idx : n_idx + 1].mean() + eps) - 1.0
        y1 = g - b
        y2 = g + b - 2.0 * r
        alpha = np.std(y1, ddof=0) / (np.std(y2, ddof=0) + eps)
        y = (y1 + alpha * y2) - (y1 + alpha * y2).mean()
        out[m_idx : n_idx + 1] += y
    return out


def chrom_pos_windowed(
    R: np.ndarray,
    G: np.ndarray,
    B: np.ndarray,
    win_sec: float = POS.window_size,
    step_sec: float = POS.step_size,
    method: str = "CHROM",
) -> np.ndarray:
    """Windowed CHROM/POS with Hann overlap-add and per-window normalization."""
    fs_hz = float(Video.FPS)
    eps = 1e-8
    R = np.asarray(R, float)
    G = np.asarray(G, float)
    B = np.asarray(B, float)
    n = len(R)

    w = max(8, int(round(win_sec * fs_hz)))
    w = min(w, n)
    s = max(1, int(round(step_sec * fs_hz)))

    out = np.zeros(n, float)
    wsum = np.zeros(n, float)
    win = np.hanning(w)
    kind = (method or "CHROM").upper()
    if kind not in ("CHROM", "POS"):
        raise ValueError("method must be 'CHROM' or 'POS'")

    for start in range(0, n - w + 1, s):
        sl = slice(start, start + w)
        r = R[sl].copy()
        g = G[sl].copy()
        b = B[sl].copy()

        r = r / (r.mean() + eps) - 1.0
        g = g / (g.mean() + eps) - 1.0
        b = b / (b.mean() + eps) - 1.0

        if kind == "CHROM":
            x = 3.0 * r - 2.0 * g
            y = 1.5 * r + g - 1.5 * b
            a = (np.std(x, ddof=1) + eps) / (np.std(y, ddof=1) + eps)
            y = x - a * y
        else:
            y1 = g - b
            y2 = g + b - 2.0 * r
            a = (np.std(y1, ddof=1) + eps) / (np.std(y2, ddof=1) + eps)
            y = y1 + a * y2

        y = (y - y.mean()) / (np.std(y, ddof=1) + eps)
        out[sl] += y * win
        wsum[sl] += win

    pulse = np.zeros_like(out)
    nz = wsum > eps
    pulse[nz] = out[nz] / (wsum[nz] + eps)
    return pulse - pulse.mean()


def chrom_pos_windowed_nan(
    R: np.ndarray,
    G: np.ndarray,
    B: np.ndarray,
    win_sec: float = 2.0,
    step_sec: float = 0.8,
    method: str = "CHROM",
    fps: float = 35.0,
) -> np.ndarray:
    """NaN-safe CHROM/POS with Hann overlap-add; positions with no coverage return NaN."""
    eps = 1e-8
    fs_hz = float(Video.FPS)
    R = np.asarray(R, float)
    G = np.asarray(G, float)
    B = np.asarray(B, float)

    def zscore_nan(arr: np.ndarray) -> np.ndarray:
        m = np.nanmean(arr)
        s = np.nanstd(arr)
        if not np.isfinite(m):
            m = 0.0
        if not np.isfinite(s) or s < 1e-12:
            s = 1.0
        return (arr - m) / (s + eps)

    R = zscore_nan(R)
    G = zscore_nan(G)
    B = zscore_nan(B)

    n = len(R)
    w = max(8, int(round(win_sec * fs_hz)))
    s = max(1, int(round(step_sec * fs_hz)))
    if w <= 1 or n == 0:
        return np.full(n, np.nan, float)

    out = np.zeros(n, float)
    wsum = np.zeros(n, float)
    win = np.hanning(w)
    kind = (method or "CHROM").upper()
    if kind not in ("CHROM", "POS"):
        raise ValueError("method must be 'CHROM' or 'POS'")

    for start in range(0, n - w + 1, s):
        sl = slice(start, start + w)
        r = R[sl]
        g = G[sl]
        b = B[sl]
        valid = np.isfinite(r) & np.isfinite(g) & np.isfinite(b)
        idx = np.nonzero(valid)[0]
        if idx.size < 2:
            continue

        if kind == "CHROM":
            x = 3.0 * r - 2.0 * g
            y = 1.5 * r + 1.0 * g - 1.5 * b
            a = (np.nanstd(x[valid]) + eps) / (np.nanstd(y[valid]) + eps)
            y = x - a * y
        else:
            y1 = g - b
            y2 = g + b - 2.0 * r
            a = (np.nanstd(y1[valid]) + eps) / (np.nanstd(y2[valid]) + eps)
            y = y1 + a * y2

        ym = np.nanmean(y[valid])
        ys = np.nanstd(y[valid])
        if not np.isfinite(ys) or ys < 1e-12:
            continue

        yz = np.zeros_like(y)
        yz[valid] = (y[valid] - ym) / (ys + eps)

        abs_idx = start + idx
        h_idx = win[idx]
        out[abs_idx] += yz[idx] * h_idx
        wsum[abs_idx] += h_idx

    res = np.full(n, np.nan, float)
    msk = wsum > 0
    res[msk] = out[msk] / (wsum[msk] + eps)
    return res

def detrend_signal_nan(x: np.ndarray, lambda_param: float = 300.0) -> np.ndarray:
    """Smoothness-priors detrend with linear interpolation over NaNs; NaNs restored after."""
    x = np.asarray(x, float)
    n = x.size
    nan_mask = np.isnan(x)
    if nan_mask.all():
        return x
    idx = np.arange(n)
    if nan_mask.any():
        x_filled = x.copy()
        x_filled[nan_mask] = np.interp(idx[nan_mask], idx[~nan_mask], x[~nan_mask])
    else:
        x_filled = x

    I = sparse.eye(n, format="csc")
    e = np.ones(n)
    D = sparse.diags([e, -2 * e, e], [0, 1, 2], shape=(n - 2, n), format="csc")
    A = I + (lambda_param ** 2) * (D.T @ D)
    trend = spsolve(A, x_filled)
    y = x_filled - trend
    y[nan_mask] = np.nan
    return y


def detrend_signal(x: np.ndarray, lambda_param: float = 300.0) -> np.ndarray:
    """Smoothness-priors detrend without NaN handling."""
    x = np.asarray(x, float)
    n = x.size
    I = sparse.eye(n, format="csc")
    e = np.ones(n)
    D = sparse.diags([e, -2 * e, e], [0, 1, 2], shape=(n - 2, n), format="csc")
    A = I + (lambda_param ** 2) * (D.T @ D)
    trend = spsolve(A, x)
    return x - trend

def sliding_mean_normalize(x: np.ndarray, fs: float, win_sec: float = 1.5, eps: float = 1e-8) -> np.ndarray:
    """Normalize by centered moving average over a window of win_sec seconds."""
    n_win = max(1, int(round(win_sec * fs)))
    x = np.asarray(x, float)
    m = np.isfinite(x).astype(float)
    x0 = np.nan_to_num(x, nan=0.0)
    kern = np.ones(n_win, float)
    num = np.convolve(x0, kern, mode="same")
    den = np.convolve(m, kern, mode="same")
    ma = num / np.maximum(den, eps)
    return (x - ma) / np.maximum(ma, eps)


def fill_short_gaps_then_drop(
    x: np.ndarray,
    t: np.ndarray | None = None,
    max_gap_s: float = 0.2,
):
    """
    Fill short NaN gaps (≤ max_gap_s) by linear interpolation; drop longer gaps.
    Returns (x_clean, t_clean, mask_valid, idx_valid, reinsert_fn).
    """
    fs_hz = float(Video.FPS)
    x = np.asarray(x, float)
    n = x.size
    if t is None:
        t_vec = np.arange(n, dtype=float) / fs_hz
    else:
        t_vec = np.asarray(t, float)

    max_gap = int(round(max_gap_s * fs_hz))
    idx_all = np.arange(n)
    nan_mask = np.isnan(x)

    if nan_mask.any():
        starts = np.where(np.diff(np.r_[False, nan_mask, False]) == 1)[0]
        ends = np.where(np.diff(np.r_[False, nan_mask, False]) == -1)[0]
        for s, e in zip(starts, ends):
            gap_len = e - s
            if gap_len <= max_gap:
                x[s:e] = np.interp(idx_all[s:e], idx_all[~nan_mask], x[~nan_mask])

    valid_mask = ~np.isnan(x)
    idx_valid = np.where(valid_mask)[0]
    x_clean = x[valid_mask]
    t_clean = t_vec[valid_mask]

    def reinsert(y_proc: np.ndarray) -> np.ndarray:
        y_full = np.full_like(x, np.nan, dtype=float)
        y_full[idx_valid] = y_proc
        return y_full

    return x_clean, t_clean, valid_mask, idx_valid, reinsert
