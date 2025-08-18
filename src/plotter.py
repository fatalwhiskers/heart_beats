import matplotlib.pyplot as plt
import numpy as np
import src.video_reader as vr
import src.Video_extraction as VE
import test as test
import src.extract_wave as ext
import argparse
import sys
import os
from src.config import Video 
from src.config import Signal
from sklearn.decomposition import PCA
from scipy.interpolate import interp1d
import pandas as pd

def plot_signal(signal_processed, label):
    dt = 1 / Video.FPS
    n = len(signal_processed)
    time = np.arange(n) / Video.FPS

    # Compute FFT
    fhat = np.fft.fft(signal_processed)
    freqs = np.fft.fftfreq(n, d=dt)
    power_spec = np.abs(fhat)**2 / n

    # Keep only positive frequencies
    pos_mask = freqs > 0
    freqs = freqs[pos_mask]
    fft_mag = np.abs(fhat)[pos_mask]
    power_spec = power_spec[pos_mask]


    hr_band = (freqs >= Signal.HR_LOW) & (freqs <= Signal.HR_HIGH)
    peak_freq = freqs[hr_band][np.argmax(power_spec[hr_band])]
    bpm = peak_freq * 60

    fig, axs = plt.subplots(3, 1, figsize=(12, 10))

    # 1. Original signal
    axs[0].plot(time, signal_processed, color='black')
    axs[0].set_title(f"1. Time Signal - {label}")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Intensity")

    # 2. FFT
    axs[1].plot(freqs, fft_mag, color='purple')
    axs[1].set_xlim(0, 5)
    axs[1].set_title(f"2. FFT Magnitude Spectrum - {label}")
    axs[1].set_xlabel("Frequency (Hz)")
    axs[1].set_ylabel("Amplitude")

    # 3. Power Spectrum
    axs[2].plot(freqs, power_spec, color='orange')
    axs[2].axvline(peak_freq, color='red', linestyle='--', label=f'Peak: {peak_freq:.2f} Hz ({bpm:.1f} BPM)')
    axs[2].set_xlim(0, 5)
    axs[2].set_title(f"3. Power Spectrum - {label}")
    axs[2].set_xlabel("Frequency (Hz)")
    axs[2].set_ylabel("Power")
    axs[2].legend()

    plt.tight_layout()
    plt.show()



def plot_signals(G_signal, R_signal, B_signal):
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

     # Red channel
    axs[0].plot(R_signal, color='red')
    axs[0].set_title('Red Channel Average Intensity')
    axs[0].set_ylabel('Intensity')

    # Green channel
    axs[1].plot(G_signal, color='green')
    axs[1].set_title('Green Channel Average Intensity')
    axs[1].set_ylabel('Intensity')

    # Blue channel
    axs[2].plot(B_signal, color='blue')
    axs[2].set_title('Blue Channel Average Intensity')
    axs[2].set_ylabel('Intensity')
    axs[2].set_xlabel('Frame')

    plt.tight_layout()
    plt.show()



def bpm_over_time(signal_processed, label):
    window_size_sec = 10
    window_size = int(window_size_sec * Video.FPS)
    step_size = window_size // 2

    bpm_list = []
    time_bins = []

    for start in range(0, len(signal_processed) - window_size + 1, step_size):
        segment = signal_processed[start:start + window_size]
        segment = segment * np.hamming(len(segment))  # Apply Hamming window

        n = len(segment)
        fhat = np.fft.fft(segment)
        freqs = np.fft.fftfreq(n, d=1 / Video.FPS)
        power_spec = np.abs(fhat)**2 / n
        
        pos = freqs > 0
        freqs = freqs[pos]
        power_spec = power_spec[pos]
        
        # Heart rate band mask
        hr_band = (freqs >= Signal.HR_LOW) & (freqs <= Signal.HR_HIGH)
        peak_freq = freqs[hr_band][np.argmax(power_spec[hr_band])]
        bpm = peak_freq * 60
        
        bpm_list.append(bpm)
        time_bins.append(start / Video.FPS)  # time in seconds for the bin start

    # Calculate average BPM
    avg_bpm = np.mean(bpm_list) if bpm_list else 0

    # Plot heart rate over time
    plt.plot(time_bins, bpm_list, label='BPM Over Time', color='blue')
    plt.axhline(y=avg_bpm, color='red', linestyle='--', label=f'Average BPM: {avg_bpm:.2f}')
    plt.xlabel(f"Time - {label}")
    plt.ylabel('Heart Rate (BPM)')
    plt.title(f"Heart Rate Over Time - {label}")
    plt.legend()
    plt.show()


def summarize_results(signals_dict, fps=Video.FPS, ground_truth_bpm=None,
                      save_csv=True, output_file="rppg_results.csv"):
    """
    Summarizes BPM estimation results for each signal channel, 
    generates plots, and optionally saves as CSV.
    """
    results = []

    for label, sig in signals_dict.items():
        # FFT-based BPM estimation
        n = len(sig)
        fhat = np.fft.fft(sig)
        freqs = np.fft.fftfreq(n, d=1 / fps)
        power_spec = np.abs(fhat)**2 / n

        pos = freqs > 0
        freqs = freqs[pos]
        power_spec = power_spec[pos]

        hr_band = (freqs >= Signal.HR_LOW) & (freqs <= Signal.HR_HIGH)
        peak_freq = freqs[hr_band][np.argmax(power_spec[hr_band])]
        bpm_est = peak_freq * 60

        row = {"Channel": label, "Estimated_BPM": bpm_est}

        # If ground truth available, compute metrics
        if ground_truth_bpm is not None:
            gt = np.array(ground_truth_bpm)
            mae = np.mean(np.abs(bpm_est - gt))
            rmse = np.sqrt(np.mean((bpm_est - gt) ** 2))
            row["MAE"] = mae
            row["RMSE"] = rmse

        results.append(row)

    results_df = pd.DataFrame(results)
    print("\n=== RPPG Results Summary ===")
    print(results_df)

    if save_csv:
        results_df.to_csv(output_file, index=False)
        print(f"Saved results to {output_file}")

    # ==== Plot MAE per channel (if GT available) ====
    if ground_truth_bpm is not None and "MAE" in results_df.columns:
        plt.figure(figsize=(8, 5))
        plt.bar(results_df["Channel"], results_df["MAE"], color="skyblue")
        plt.ylabel("MAE (BPM)")
        plt.title("Mean Absolute Error per Channel")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()

        # ==== Bland–Altman plots ====
        for label, sig in signals_dict.items():
            n = len(sig)
            fhat = np.fft.fft(sig)
            freqs = np.fft.fftfreq(n, d=1 / fps)
            pos = freqs > 0
            freqs = freqs[pos]
            power_spec = np.abs(fhat)**2 / n
            hr_band = (freqs >= Signal.HR_LOW) & (freqs <= Signal.HR_HIGH)
            peak_freq = freqs[hr_band][np.argmax(power_spec[hr_band])]
            bpm_est = peak_freq * 60

            avg_vals = (bpm_est + np.array(ground_truth_bpm)) / 2
            diff_vals = bpm_est - np.array(ground_truth_bpm)

            mean_diff = np.mean(diff_vals)
            loa = 1.96 * np.std(diff_vals)

            plt.figure(figsize=(6, 5))
            plt.scatter(avg_vals, diff_vals, alpha=0.6)
            plt.axhline(mean_diff, color='red', linestyle='--', label=f'Mean Bias: {mean_diff:.2f}')
            plt.axhline(mean_diff + loa, color='gray', linestyle='--', label=f'+1.96 SD')
            plt.axhline(mean_diff - loa, color='gray', linestyle='--', label=f'-1.96 SD')
            plt.title(f"Bland–Altman Plot - {label}")
            plt.xlabel("Average BPM")
            plt.ylabel("Difference (Est - GT)")
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.show()

    return results_df
