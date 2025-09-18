import matplotlib.pyplot as plt
import numpy as np
import test as test
from src.config import Video
from src.rppg import sliding_fft_hr
from src.config import Signal, fileDataset1, fileDataset2
import pandas as pd
import src.Stats as hr_stat
from scipy.signal import welch, medfilt, savgol_filter, get_window, find_peaks
from scipy.ndimage import uniform_filter1d

HEADER = [
    "Subject ID", "Recording ID", "ROI", "Extraction Method",
    "Number of Windows",
    "MAE (bpm)", "RMSE (bpm)", "Pearson r", "Pearson p",
    "Bias (bpm)", "SD (bpm)", "LoA Lower (bpm)", "LoA Upper (bpm)",
    "Mean rPPG (bpm)", "Median rPPG (bpm)",
    "Mean Ground Truth (bpm)", "Median Ground Truth (bpm)",
    "Mean Error (bpm)", "Median Error (bpm)", "Median Absolute Error (bpm)",
]

def run(signal_processed, label, ground_truth_hr):
    estimate_hr_sliding(signal_processed, label)

def build_table(rPPG, rPPG_time, ground_truth, gt_time, subject, recording_id, signal_label, cropMode):
    stats = hr_stat.evaluate_hr_metrics(rPPG, ground_truth, rPPG_time, gt_time)
    row = hr_stat.metrics_to_row(stats, subject, recording_id, signal_label, cropMode)
    hr_stat.append_rows_to_csv(r"outputs\time_array_fix_results.csv", HEADER, [row])
    return

def build_table_ECG(rPPG, ground_truth, t_rPPG, t_ref, subject, recording_id, signal_label, cropMode):
    stats = hr_stat.evaluate_hr_metrics(rPPG, ground_truth, t_rPPG, t_ref)
    row = hr_stat.metrics_to_row(stats, subject, recording_id, signal_label, cropMode)
    hr_stat.append_rows_to_csv(r"outputs\Dataset_welch_2_20s_3s_results.csv", HEADER, [row])
    return

def build_table3(rPPG, rPPG_time, recording_id, signal_label, cropMode):
    hr_stat.export_hr_to_csv_and_plot(rPPG, rPPG_time, recording_id, signal_label, cropMode, r'outputs\mark')
    return

def compare_bvp_rppg(bvp_signal, rppg_signal, bvp_rate=64, rppg_fps=35, window_size=15, step_size=5, smooth_rppg=True):
    """
    Compare ground-truth BVP HR with sliding FFT HR from rPPG.
    """
    min_distance = int(bvp_rate / 3.0)
    peaks, _ = find_peaks(bvp_signal, distance=min_distance)
    peak_times = peaks / bvp_rate
    ibi = np.diff(peak_times)
    bvp_hr = 60 / ibi
    bvp_times = peak_times[1:]

    rppg_signal = (rppg_signal - np.mean(rppg_signal)) / np.std(rppg_signal)

    n = len(rppg_signal)
    win_len = int(window_size * rppg_fps)
    step_len = int(step_size * rppg_fps)
    hann_win = np.hanning(win_len)

    hr_values, times = [], []

    for start in range(0, n - win_len, step_len):
        segment = rppg_signal[start:start + win_len] * hann_win
        fft_vals = np.fft.rfft(segment)
        freqs = np.fft.rfftfreq(win_len, d=1/rppg_fps)
        power = np.abs(fft_vals)**2
        mask = (freqs >= 0.7) & (freqs <= 4.0)
        freqs_band, power_band = freqs[mask], power[mask]
        freq_est = np.sum(freqs_band * power_band) / np.sum(power_band)
        bpm = freq_est * 60
        hr_values.append(bpm)
        times.append(start / rppg_fps)

    hr_values = np.array(hr_values)
    times = np.array(times)

    if smooth_rppg:
        hr_values = uniform_filter1d(hr_values, size=3)

    plt.figure(figsize=(12,5))
    plt.plot(bvp_times, bvp_hr, label="BVP HR (Ground Truth)", color='blue')
    plt.plot(times, hr_values, label="rPPG HR (Sliding FFT)", color='red', alpha=0.7)
    plt.xlabel("Time (s)")
    plt.ylabel("Heart Rate (BPM)")
    plt.title("BVP vs rPPG Heart Rate Comparison")
    plt.legend()
    plt.grid(True)
    plt.show()

    return {
        "bvp_times": bvp_times,
        "bvp_hr": bvp_hr,
        "rppg_times": times,
        "rppg_hr": hr_values
    }

def plot_hr(rppg_hr, rppg_t, gt_hr, gt_t, label, title="HR comparison"):
    gt_interp = np.interp(rppg_t, gt_t, gt_hr)
    diff = rppg_hr - gt_interp
    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff**2))

    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(rppg_t, rppg_hr, label="rPPG HR", lw=2)
    ax.plot(gt_t, gt_hr, label="Ground truth HR", lw=2, alpha=0.7)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("HR (BPM)")
    ax.set_title(f"{title}\nMAE={mae:.2f} BPM, RMSE={rmse:.2f} BPM  " + label)
    ax.legend()
    plt.tight_layout()
    plt.show()

    return {"mae_bpm": mae, "rmse_bpm": rmse}

def plot_sig(rppg_hr, rppg_t, label, title="HR comparison"):
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(rppg_t, rppg_hr, label="rPPG HR", lw=2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("HR (BPM)")
    ax.set_title(f"{title} " + label)
    ax.legend()
    plt.tight_layout()
    plt.show()
    return

def estimate_hr_sliding(signal, fps=Video.FPS):
    times, hr_values, psd_accum, freqs_band = sliding_fft_hr(signal)
    avg_psd = np.mean(psd_accum, axis=0)
    peak_indices, _ = find_peaks(avg_psd, distance=5)
    peak_powers = avg_psd[peak_indices]
    sorted_idx = np.argsort(peak_powers)[::-1]
    top_peaks = peak_indices[sorted_idx[:3]]
    top_freqs = freqs_band[top_peaks]
    top_bpms = top_freqs * 60
    global_hr = top_bpms[0]
    peak_colors = ['red', 'green', 'blue']

    fig, axs = plt.subplots(3, 1, figsize=(12, 10))

    axs[0].plot(np.arange(len(signal)) / fps, signal, color='black')
    axs[0].set_title("Preprocessed rPPG Signal")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Normalized Intensity")

    axs[1].plot(times, hr_values, 'o-r', label="Sliding HR")
    axs[1].axhline(np.mean(hr_values), color='black', linestyle='--', label=f"Mean HR = {np.mean(hr_values):.1f} BPM")
    axs[1].axhline(np.median(hr_values), color='blue', linestyle='--', label=f"Median HR = {np.median(hr_values):.1f} BPM")
    axs[1].set_title("Sliding HR Over Time")
    axs[1].legend()

    axs[2].plot(freqs_band, avg_psd, color='orange')
    for i, (f, bpm) in enumerate(zip(top_freqs, top_bpms)):
        color = peak_colors[i % len(peak_colors)]
        axs[2].axvline(f, color=color, linestyle='--', label=f"Peak {i+1}: {f:.2f} Hz ({bpm:.1f} BPM)")
    axs[2].set_xlim(0.5, 4.5)
    axs[2].set_title("Averaged Power Spectrum (Top 3 Peaks)")
    axs[2].legend()

    plt.tight_layout()
    plt.show()

    return {
        "sliding_hr": np.array(hr_values),
        "times": np.array(times),
        "mean_hr": np.mean(hr_values),
        "median_hr": np.median(hr_values),
        "global_hr": global_hr,
        "top_peaks_bpm": top_bpms
    }
