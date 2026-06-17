import os
import numpy as np
import matplotlib.pyplot as plt

from src.rppg import sliding_welch_hr_center, sliding_fft_hr_center
from archive.not_working.prv_0 import compute_prv_hr
from src.config import Video


def ensure_directory(path: str):
    os.makedirs(path, exist_ok=True)


def resample_linear(t, x, fs_target):
    t = np.asarray(t, float)
    x = np.asarray(x, float)
    t0, t1 = t[0], t[-1]
    t_uniform = np.arange(t0, t1, 1.0 / fs_target)
    x_uniform = np.interp(t_uniform, t, x)
    return t_uniform, x_uniform


def compute_aligned_mae(t_est, hr_est, t_gt_hr, hr_gt_bpm):
    t_est = np.asarray(t_est, float)
    hr_est = np.asarray(hr_est, float)
    t_gt_hr = np.asarray(t_gt_hr, float)
    hr_gt_bpm = np.asarray(hr_gt_bpm, float)

    if t_est.size == 0 or t_gt_hr.size == 0:
        return np.nan, np.array([]), np.array([])

    mask = (t_est >= t_gt_hr[0]) & (t_est <= t_gt_hr[-1])
    if not np.any(mask):
        return np.nan, np.array([]), np.array([])

    t_ok = t_est[mask]
    hr_ok = hr_est[mask]
    hr_gt_interp = np.interp(t_ok, t_gt_hr, hr_gt_bpm)
    mae = float(np.mean(np.abs(hr_gt_interp - hr_ok)))
    return mae, t_ok, hr_gt_interp


def generate_synthetic_pair(
    duration_s=300,
    fs_gt=200.0,
    fps_nominal=35,
    jitter_frac=0.12,
    drift_ppm=250.0,
    drop_frame_prob=0.03,
    noise_std=0.8,
    base_bpm=78.0,
    swing_bpm=10.0,
    swing_period_s=40.0
):
    rng = np.random.default_rng(42)

    gt_time = np.arange(0, duration_s, 1.0 / fs_gt)
    hr_profile_bpm = base_bpm + swing_bpm * np.sin(2 * np.pi * gt_time / swing_period_s)
    inst_freq_hz = hr_profile_bpm / 60.0

    dt = 1.0 / fs_gt
    phase = 2 * np.pi * np.cumsum(inst_freq_hz) * dt

    base_wave = (np.pi - ((phase + np.pi) % (2 * np.pi))) / np.pi
    gt_signal = 0.7 * base_wave + 0.3 * np.sin(2 * phase)
    gt_signal = (gt_signal - np.mean(gt_signal)) / (np.std(gt_signal) + 1e-12)

    from scipy.signal import find_peaks, butter, filtfilt
    b, a = butter(3, [0.5, 8.0], btype="band", fs=fs_gt)
    gt_signal = filtfilt(b, a, gt_signal)

    prom = 0.2 * (np.percentile(gt_signal, 95) - np.percentile(gt_signal, 5))
    peaks, _ = find_peaks(gt_signal, distance=max(1, int(fs_gt / 4.0)), prominence=max(prom, 1e-9))
    peak_times = gt_time[peaks]
    ibi = np.diff(peak_times)
    hr_time_gt = 0.5 * (peak_times[1:] + peak_times[:-1])
    hr_gt_bpm = 60.0 / ibi if ibi.size else np.array([])

    frame_dt = 1.0 / fps_nominal
    n_nom = int(np.floor(duration_s / frame_dt))
    t_nom = np.arange(n_nom) * frame_dt

    drift_scale = 1.0 + drift_ppm * 1e-6
    t_drift = t_nom * drift_scale

    jitter = rng.standard_normal(n_nom) * (jitter_frac * frame_dt)
    t_jitter = t_drift + jitter

    keep_mask = rng.random(n_nom) > drop_frame_prob
    obs_time = t_jitter[keep_mask]
    obs_time = obs_time[(obs_time >= 0) & (obs_time <= duration_s)]
    obs_time = np.maximum.accumulate(obs_time)

    obs_signal = np.interp(obs_time, gt_time, gt_signal)
    obs_signal = obs_signal + noise_std * rng.standard_normal(len(obs_signal))

    if len(obs_time) > 10:
        fs_local = 1.0 / np.median(np.diff(obs_time))
        if fs_local > 2.0:
            b2, a2 = butter(3, [0.5, 8.0], btype="band", fs=fs_local)
            obs_signal = filtfilt(b2, a2, obs_signal)

    return gt_time, gt_signal, hr_time_gt, hr_gt_bpm, obs_time, obs_signal


def main():
    output_dir = "outputs/synth"
    ensure_directory(output_dir)

    gt_time, gt_signal, hr_time_gt, hr_gt_bpm, obs_time, obs_signal = generate_synthetic_pair()

    uniform_time, uniform_signal = resample_linear(obs_time, obs_signal, fs_target=Video.FPS)

    welch_time, welch_hr, welch_psd, welch_freqs = sliding_welch_hr_center(uniform_signal)
    fft_time, fft_hr, fft_psd, fft_freqs = sliding_fft_hr_center(uniform_signal)
    pp_clean, prv_hr_clean, prv_hr_raw, prv_mid_times, _, _ = compute_prv_hr(uniform_time, uniform_signal)

    mae_welch, welch_time_ok, hr_gt_on_welch = compute_aligned_mae(welch_time, welch_hr, hr_time_gt, hr_gt_bpm)
    mae_fft, fft_time_ok, hr_gt_on_fft = compute_aligned_mae(fft_time, fft_hr, hr_time_gt, hr_gt_bpm)
    mae_prv, prv_time_ok, hr_gt_on_prv = compute_aligned_mae(prv_mid_times, prv_hr_clean, hr_time_gt, hr_gt_bpm)

    print(f"[Welch] MAE vs GT: {mae_welch:.2f} BPM   (n={len(welch_time_ok)})")
    print(f"[ FFT ] MAE vs GT: {mae_fft:.2f} BPM     (n={len(fft_time_ok)})")
    print(f"[ PRV ] MAE vs GT: {mae_prv:.2f} BPM     (n={len(prv_time_ok)})")

    plt.figure()
    plt.plot(uniform_time, uniform_signal)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Observed (resampled to Video.FPS)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "observed_resampled.png"))
    plt.close()

    if len(welch_time_ok):
        plt.figure()
        plt.plot(hr_time_gt, hr_gt_bpm, label="Ground truth HR")
        plt.scatter(welch_time, welch_hr, s=16, label="Welch HR", alpha=0.8)
        plt.xlabel("Time (s)")
        plt.ylabel("BPM")
        plt.legend()
        plt.title(f"Welch vs GT (MAE={mae_welch:.2f} BPM)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "welch_vs_gt.png"))
        plt.close()

    if len(fft_time_ok):
        plt.figure()
        plt.plot(hr_time_gt, hr_gt_bpm, label="Ground truth HR")
        plt.scatter(fft_time, fft_hr, s=16, label="FFT HR", alpha=0.8)
        plt.xlabel("Time (s)")
        plt.ylabel("BPM")
        plt.legend()
        plt.title(f"FFT vs GT (MAE={mae_fft:.2f} BPM)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "fft_vs_gt.png"))
        plt.close()

    if welch_psd.size:
        plt.figure()
        plt.imshow(
            welch_psd.T,
            aspect="auto",
            origin="lower",
            extent=[welch_time[0], welch_time[-1], welch_freqs[0] * 60, welch_freqs[-1] * 60],
        )
        plt.xlabel("Time (s)")
        plt.ylabel("BPM (freq axis)")
        plt.title("Welch band PSD over time (normalized)")
        plt.colorbar(label="Norm power")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "welch_psd_waterfall.png"))
        plt.close()

    print(f"Saved plots to: {output_dir}")


if __name__ == "__main__":
    main()
