import numpy as np
from scipy.signal import periodogram, welch, butter, filtfilt, detrend
import matplotlib.pyplot as plt
import src.rppg as rppg_lib


def generate_synthetic_rgb(
    duration_s: float = 60,
    fs: float = 35.0,
    hr_bpm: float = 72.0,
    amps: tuple[float, float, float] = (0.010, 0.012, 0.008),
    snr_db: float = -10.0,
    seed: int | None = None,
):
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration_s, 1.0 / fs)
    hr_hz = float(hr_bpm) / 60.0
    r_clean = amps[0] * np.sin(2 * np.pi * hr_hz * t + 0.0)
    g_clean = amps[1] * np.sin(2 * np.pi * hr_hz * t + 0.5)
    b_clean = amps[2] * np.sin(2 * np.pi * hr_hz * t + 1.0)
    sig_power = np.mean([np.mean(r_clean**2), np.mean(g_clean**2), np.mean(b_clean**2)])
    noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    sigma = np.sqrt(noise_power)
    r = r_clean + sigma * rng.standard_normal(t.size)
    g = g_clean + sigma * rng.standard_normal(t.size)
    b = b_clean + sigma * rng.standard_normal(t.size)
    return t, r, g, b, r_clean, g_clean, b_clean


def butter_bandpass(x, fs, lo=0.7, hi=4.0, order=2):
    b, a = butter(order, [lo, hi], btype="band", fs=fs)
    return filtfilt(b, a, detrend(x, type="linear"))


def parabolic_interpolation_offset(y_minus, y0, y_plus):
    denom = (y_minus - 2.0 * y0 + y_plus)
    if abs(denom) < 1e-20:
        return 0.0
    return 0.5 * (y_minus - y_plus) / denom


def estimate_hr_periodogram(x, fs, band=(0.7, 4.0), nearest_bpm=None):
    x = np.asarray(x, float)
    x = x - np.nanmean(x)
    f, Pxx = periodogram(x, fs=fs, detrend=False, scaling="density")
    mask = (f >= band[0]) & (f <= band[1])
    if not np.any(mask):
        return np.nan, None, (f, Pxx)
    f_band = f[mask]
    P_band = Pxx[mask]
    k = int(np.argmax(P_band))
    dk = parabolic_interpolation_offset(P_band[k - 1], P_band[k], P_band[k + 1]) if 0 < k < len(P_band) - 1 else 0.0
    df = (f_band[1] - f_band[0]) if len(f_band) > 1 else 0.0
    f_peak = f_band[k] + dk * df
    hr_from_max = 60.0 * f_peak
    if nearest_bpm is None:
        return hr_from_max, None, (f, Pxx)
    idx_near = np.argmin(np.abs(f_band * 60.0 - nearest_bpm))
    hr_near = 60.0 * f_band[idx_near]
    return hr_from_max, hr_near, (f, Pxx)


def estimate_hr_welch(
    x,
    fs,
    band=(0.7, 4.0),
    nperseg=None,
    noverlap=None,
    window="hann",
    nearest_bpm=None,
):
    x = np.asarray(x, float)
    x = x - np.nanmean(x)
    f, Pxx = welch(
        x,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=False,
        scaling="density",
        return_onesided=True,
    )
    mask = (f >= band[0]) & (f <= band[1])
    if not np.any(mask):
        return np.nan, None, (f, Pxx)
    f_band = f[mask]
    P_band = Pxx[mask]
    k = int(np.argmax(P_band))
    dk = parabolic_interpolation_offset(P_band[k - 1], P_band[k], P_band[k + 1]) if 0 < k < len(P_band) - 1 else 0.0
    df = (f_band[1] - f_band[0]) if len(f_band) > 1 else 0.0
    f_peak = f_band[k] + dk * df
    hr_from_max = 60.0 * f_peak
    if nearest_bpm is None:
        return hr_from_max, None, (f, Pxx)
    idx_near = np.argmin(np.abs(f_band * 60.0 - nearest_bpm))
    hr_near = 60.0 * f_band[idx_near]
    return hr_from_max, hr_near, (f, Pxx)


def sliding_welch_hr_centered(
    x,
    fs,
    win_sec=1.6,
    step_sec=0.5,
    nperseg_frac=0.5,
    noverlap_frac=0.75,
    band=(0.7, 4.0),
    nfft=4096,
    window="hamming",
):
    x = np.asarray(x, float)
    x = (x - np.mean(x)) / (np.std(x) + 1e-12)
    n = len(x)
    win_len = max(8, int(round(win_sec * fs)))
    step_len = max(1, int(round(step_sec * fs)))
    nperseg = max(8, int(round(nperseg_frac * win_len)))
    noverlap = int(round(noverlap_frac * nperseg))
    hr_vals, times, psd_rows = [], [], []
    freqs_band = None
    for start in range(0, n - win_len + 1, step_len):
        seg = x[start:start + win_len]
        f, Pxx = welch(
            seg,
            fs=fs,
            window=window,
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nfft,
            detrend=False,
            scaling="density",
            return_onesided=True,
        )
        mask = (f >= band[0]) & (f <= band[1])
        fb = f[mask]
        Pb = Pxx[mask]
        if Pb.size:
            k = int(np.argmax(Pb))
            dk = parabolic_interpolation_offset(Pb[k - 1], Pb[k], Pb[k + 1]) if 0 < k < len(Pb) - 1 else 0.0
            df = (fb[1] - fb[0]) if len(fb) > 1 else 0.0
            f_peak = fb[k] + dk * df
            hr_vals.append(60.0 * f_peak)
            psd_rows.append(Pb)
            times.append((start + win_len / 2.0) / fs)
            if freqs_band is None:
                freqs_band = fb
        else:
            hr_vals.append(np.nan)
            psd_rows.append(np.zeros_like(freqs_band) if freqs_band is not None else np.array([]))
            times.append((start + win_len / 2.0) / fs)
    psd_stack = np.vstack(psd_rows) if psd_rows and psd_rows[0].size else np.zeros((0, 0))
    return np.asarray(times), np.asarray(hr_vals), psd_stack, freqs_band


def sliding_welch_fixed10s(x, fs, band=(0.65, 4.0)):
    x = detrend(np.asarray(x, float), type="linear")
    b, a = butter(6, [band[0], band[1]], btype="band", fs=fs)
    x_filt = filtfilt(b, a, x)
    win_len = int(round(10 * fs))
    step_len = win_len
    nperseg = win_len
    noverlap = 0
    f = np.fft.rfftfreq(nperseg, 1.0 / fs)
    mask = (f >= band[0]) & (f <= band[1])
    f_band = f[mask]
    times, hr_vals, rows = [], [], []
    for start in range(0, len(x_filt) - win_len + 1, step_len):
        seg = x_filt[start:start + win_len]
        _, Pxx = welch(
            seg,
            fs=fs,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            detrend=False,
            scaling="density",
            return_onesided=True,
        )
        P_band = Pxx[mask]
        k = int(np.argmax(P_band))
        bpm = f_band[k] * 60.0
        rows.append(P_band / (np.max(P_band) + 1e-12))
        hr_vals.append(bpm)
        times.append((start + win_len / 2.0) / fs)
    psd_stack = np.vstack(rows) if rows else np.zeros((0, len(f_band)))
    return np.asarray(times), np.asarray(hr_vals), psd_stack, f_band


def compare_hr_methods(
    rppg_signal,
    fs,
    gt_bpm: float | None = None,
    band=(0.7, 4.0),
    do_prebandpass: bool = False,
    plot: bool = True,
    title: str = "PSD/FFT comparison",
):
    x = np.asarray(rppg_signal, float)
    x_proc = butter_bandpass(x, fs, lo=band[0], hi=band[1], order=2) if do_prebandpass else (x - np.mean(x))
    results = []
    hr_max_p, hr_near_p, (f_per, P_per) = estimate_hr_periodogram(x_proc, fs, band=band, nearest_bpm=gt_bpm)
    results.append(("Periodogram", hr_max_p, hr_near_p))
    nperseg = int(round(8 * fs))
    noverlap = nperseg // 2
    hr_max_w, hr_near_w, (f_welch, P_welch) = estimate_hr_welch(
        x_proc, fs, band=band, nperseg=nperseg, noverlap=noverlap, window="hann", nearest_bpm=gt_bpm
    )
    results.append(("Welch(8s,50%)", hr_max_w, hr_near_w))
    t_sw, hr_sw, psd_sw, f_sw = rppg_lib.sliding_welch_hr_center_best(x_proc)
    hr_sw_med = np.nanmedian(hr_sw) if hr_sw.size else np.nan
    results.append(("SlidingWelch(1.6s)", hr_sw_med, np.nan if gt_bpm is None else hr_sw_med))
    t_s10, hr_s10, psd_s10, f_s10 = sliding_welch_fixed10s(
        x_proc, fs, band=(max(0.65, band[0]), band[1])
    )
    hr_s10_med = np.nanmedian(hr_s10) if hr_s10.size else np.nan
    results.append(("SlidingWelch(10s)", hr_s10_med, np.nan if gt_bpm is None else hr_s10_med))
    print("\n=== PSD/FFT HR comparison{} ===".format("" if gt_bpm is None else f" (GT={gt_bpm:.1f} BPM)"))
    header = "Method".ljust(20) + "HR_max".rjust(10)
    if gt_bpm is not None:
        header += "Err_max".rjust(10) + "HR_near".rjust(10) + "Err_near".rjust(10)
    print(header)
    for name, hr_m, hr_n in results:
        if gt_bpm is None:
            print(f"{name.ljust(20)}{hr_m:10.2f}")
        else:
            err_m = abs(hr_m - gt_bpm) if np.isfinite(hr_m) else np.nan
            err_n = abs(hr_n - gt_bpm) if (hr_n is not None and np.isfinite(hr_n)) else np.nan
            hrn_val = (hr_n if hr_n is not None else np.nan)
            print(f"{name.ljust(20)}{hr_m:10.2f}{err_m:10.2f}{hrn_val:10.2f}{err_n:10.2f}")
    if plot:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        ax1, ax2, ax3, ax4 = axes.ravel()
        ax1.plot(f_per, P_per, label="Periodogram")
        ax1.set_xlim(0, 5)
        ax1.set_xlabel("Hz")
        ax1.set_ylabel("PSD")
        ax1.grid(True, alpha=0.3)
        ax1.set_title("Periodogram")
        if gt_bpm is not None:
            ax1.axvline(gt_bpm / 60.0, ls="--", alpha=0.6, label="GT")
        ax1.legend()
        ax2.plot(f_welch, P_welch, label="Welch(8s,50%)")
        ax2.set_xlim(0, 5)
        ax2.set_xlabel("Hz")
        ax2.set_ylabel("PSD")
        ax2.grid(True, alpha=0.3)
        ax2.set_title("Welch")
        if gt_bpm is not None:
            ax2.axvline(gt_bpm / 60.0, ls="--", alpha=0.6, label="GT")
        ax2.legend()
        if psd_sw.size:
            im = ax3.imshow(
                psd_sw.T,
                aspect="auto",
                origin="lower",
                extent=[t_sw[0], t_sw[-1], f_sw[0], f_sw[-1]],
            )
            ax3.set_xlabel("Time (s)")
            ax3.set_ylabel("Hz")
            ax3.set_title("Sliding Welch (1.6s) PSD")
            fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
            if gt_bpm is not None:
                ax3.axhline(gt_bpm / 60.0, ls="--", color="w", alpha=0.7)
        if psd_s10.size:
            im2 = ax4.imshow(
                psd_s10.T,
                aspect="auto",
                origin="lower",
                extent=[t_s10[0], t_s10[-1], f_s10[0], f_s10[-1]],
            )
            ax4.set_xlabel("Time (s)")
            ax4.set_ylabel("Hz")
            ax4.set_title("Sliding Welch (10s) PSD")
            fig.colorbar(im2, ax=ax4, fraction=0.046, pad=0.04)
            if gt_bpm is not None:
                ax4.axhline(gt_bpm / 60.0, ls="--", color="w", alpha=0.7)
        fig.suptitle(title)
        fig.tight_layout()
        plt.show()
    return {
        "Periodogram": hr_max_p,
        "Welch(8s,50%)": hr_max_w,
        "SlidingWelch(1.6s)": hr_sw_med,
        "SlidingWelch(10s)": hr_s10_med,
    }


if __name__ == "__main__":
    fs_demo = 35.0
    t, R, G, B, R0, G0, B0 = generate_synthetic_rgb(
        duration_s=60, fs=fs_demo, hr_bpm=72.0, snr_db=-20.0, seed=0
    )
    rppg_signal = G
    report = compare_hr_methods(
        rppg_signal,
        fs_demo,
        gt_bpm=72.0,
        band=(0.7, 4.0),
        do_prebandpass=False,
        plot=True,
        title="Synthetic rPPG PSD comparison (G channel)",
    )
    print("\nChosen HRs:", report)
