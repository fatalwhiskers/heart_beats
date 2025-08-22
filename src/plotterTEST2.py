import matplotlib.pyplot as plt
import numpy as np
import test as test
from src.config import Video 
from src.rppg import sliding_fft_hr 
from src.config import Signal
import pandas as pd
from scipy.signal import welch, medfilt, savgol_filter, get_window, find_peaks
from scipy.ndimage import uniform_filter1d

def plot_signal_poer(signal_processed, label, Raw = False):
    dt = 1 / Video.FPS
    n = len(signal_processed)
    time = np.arange(n) / Video.FPS

    if Raw:
        # --- Raw FFT ---
        fhat = np.fft.fft(signal_processed)
        freqs = np.fft.fftfreq(n, d=dt)
        power_spec = np.abs(fhat)**2 / n

    else:
        # --- Windowed FFT with zero-padding ---
        window = np.hamming(n)
        signal_win = signal_processed * window
        n_fft = 2**int(np.ceil(np.log2(n*4)))  # 4x zero-padding

        fhat = np.fft.fft(signal_win, n=n_fft)
        freqs = np.fft.fftfreq(n_fft, d=dt)
        power_spec = (np.abs(fhat)**2 / n)

    
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

def plot_power_spectrum(signal_processed, label="rPPG Signal"):

    n = len(signal_processed)
    dt = 1 / FPS
    time = np.arange(n) / FPS


    # Welch’s method (smoother PSD)
    freqs, power_spec = welch(signal_processed, fs=FPS, nperseg=min(1024, n))
    fft_mag = np.sqrt(power_spec) 
    

    # Heart rate band
    hr_band = (freqs >= HR_LOW) & (freqs <= HR_HIGH)
    peak_freq = freqs[hr_band][np.argmax(power_spec[hr_band])]
    bpm = peak_freq * 60

    # Plotting
    fig, axs = plt.subplots(3, 1, figsize=(12, 10))

    # 1. Time Signal
    axs[0].plot(time, signal_processed, color='green')
    axs[0].set_title(f"1. Filtered Signal (Time Domain) - {label}")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Intensity")

    # 2. FFT Magnitude
    axs[1].plot(freqs, fft_mag, color='purple')
    axs[1].set_xlim(0, 5)
    axs[1].set_title(f"2. FFT Magnitude Spectrum - {label}")
    axs[1].set_xlabel("Frequency (Hz)")
    axs[1].set_ylabel("Amplitude")

    # 3. Power Spectrum
    axs[2].plot(freqs, power_spec, color='orange')
    axs[2].axvline(peak_freq, color='red', linestyle='--',
                   label=f'Peak: {peak_freq:.2f} Hz ({bpm:.1f} BPM)')
    axs[2].set_xlim(0, 5)
    axs[2].set_title(f"3. Power Spectrum - {label}")
    axs[2].set_xlabel("Frequency (Hz)")
    axs[2].set_ylabel("Power")
    axs[2].legend()

    plt.tight_layout()
    plt.show()

    return peak_freq, bpm

def estimate_hr_rppg(signal, label="rPPG",
                     spike_k=9, spike_nsigma=3.0,
                     detrend_sec=3.0, detrend_poly=3,
                     welch_seg_sec=8.0, welch_overlap=0.5,
                     harmonic_weights=(1.0, 0.5, 0.25),
                     smooth_bins=3,
                     plot=False):

    x = np.asarray(signal, dtype=float)
    n = len(x)
    if n < 256:
        raise ValueError("Signal too short (<256 samples).")

    # ----- 1) Spike removal (Hampel) -----
    k = int(spike_k) | 1  # ensure odd
    med = medfilt(x, kernel_size=k)
    diff = x - med
    mad = medfilt(np.abs(diff), kernel_size=k)
    sigma = 1.4826 * mad
    mask = np.abs(diff) > spike_nsigma * (sigma + 1e-12)
    x_spikefree = x.copy()
    x_spikefree[mask] = med[mask]

    # ----- 2) Detrend & normalize -----
    win_len = max(5, int(round(detrend_sec * FPS)) | 1)           # odd
    win_len = min(win_len, n - (1 - (n % 2)))                     # <= n-1 and odd
    trend = savgol_filter(x_spikefree, win_len, detrend_poly)
    x_dt = x_spikefree - trend
    x_dt = (x_dt - np.median(x_dt)) / (np.std(x_dt) + 1e-8)

    # ----- 3) Welch PSD (Hann) -----
    seg = max(256, int(round(welch_seg_sec * FPS)))
    seg = min(seg, n)
    nover = int(seg * np.clip(welch_overlap, 0, 0.95))
    window = get_window('hann', seg)
    freqs, psd = welch(x_dt, fs=FPS, nperseg=seg, noverlap=nover, window=window)

    # keep positive freqs (welch already does, but be explicit)
    pos = freqs > 0
    freqs, psd = freqs[pos], psd[pos]

    # ----- 4) Harmonic-sum selector -----
    band = (freqs >= HR_LOW) & (freqs <= HR_HIGH)
    f_band = freqs[band]
    P_band = psd[band]
    if len(f_band) < 5:
        raise ValueError("Not enough frequency bins in HR band; adjust hr_low/high or seg length.")

    # small smoothing inside band
    if smooth_bins and smooth_bins > 1:
        k = int(smooth_bins)
        ker = np.ones(k) / k
        P_band = np.convolve(P_band, ker, mode="same")

    # build harmonic-sum curve
    H = np.zeros_like(P_band)
    for h, w in enumerate(harmonic_weights, start=1):
        H += w * np.interp(h * f_band, freqs, psd, left=0.0, right=0.0)

    idx = int(np.argmax(H))
    f0 = float(f_band[idx])

    # optional subharmonic safety check
    f_half = f0 / 2.0
    if f_half >= HR_LOW:
        score_f0   = np.interp(f0,   f_band, H)
        score_half = np.interp(f_half, f_band, H)
        if score_half > 1.1 * score_f0:  # 10% margin
            f0 = f_half

    bpm = f0 * 60.0


    t = np.arange(n) / FPS
    fig, axs = plt.subplots(3, 1, figsize=(12, 9))
    axs[0].plot(t, x, label="orig", alpha=0.4)
    axs[0].plot(t, x_spikefree, label="spikefree", linewidth=1)
    axs[0].plot(t, trend, label="trend", linewidth=1)
    axs[0].set_title(f"1) Time Signal - {label}")
    axs[0].set_xlabel("Time (s)"); axs[0].set_ylabel("Intensity"); axs[0].legend(loc="upper right")

    axs[1].plot(freqs, psd)
    axs[1].axvspan(HR_LOW, HR_HIGH, alpha=0.1)
    axs[1].set_xlim(0, 5)
    axs[1].set_title(f"2) Welch PSD - {label}")
    axs[1].set_xlabel("Frequency (Hz)"); axs[1].set_ylabel("Power")

    axs[2].plot(f_band, H, label="Harmonic-sum")
    axs[2].axvline(f0, linestyle="--", label=f"{f0:.2f} Hz  ({bpm:.1f} BPM)")
    axs[2].set_xlim(0, 5)
    axs[2].set_title("3) Harmonic-Sum Score (HR band)")
    axs[2].set_xlabel("Frequency (Hz)"); axs[2].set_ylabel("Score"); axs[2].legend()
    plt.tight_layout()
    plt.show()

    return bpm, f0, freqs, psd



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

def plot_signal_sliding_avg_median(signal_processed, label, window_size=10, step_size=1):
    """
    Sliding window HR estimation using power-weighted frequency.
    Returns sliding HR values, average HR, and median HR.
    Assumes signal_processed is already filtered.
    """
    n_samples = len(signal_processed)
    time = np.arange(n_samples) / FPS

    # Sliding window parameters
    win_samples = int(window_size * FPS)
    step_samples = int(step_size * FPS)
    
    times_hr = []
    hr_estimates = []
    
    for start in range(0, n_samples - win_samples + 1, step_samples):
        segment = signal_processed[start:start + win_samples]
        windowed = segment * np.hanning(win_samples)
        
        # FFT
        fhat = np.fft.fft(windowed)
        freqs = np.fft.fftfreq(win_samples, d=1/FPS)
        power = np.abs(fhat)**2 / win_samples
        
        # Positive frequencies
        pos_mask = freqs > 0
        freqs_pos = freqs[pos_mask]
        power_pos = power[pos_mask]
        
        # Heart rate band
        hr_mask = (freqs_pos >= HR_LOW) & (freqs_pos <= HR_HIGH)
        freqs_hr = freqs_pos[hr_mask]
        power_hr = power_pos[hr_mask]
        
        if len(power_hr) > 0 and np.sum(power_hr) > 0:
            # Power-weighted mean frequency
            peak_freq = np.sum(freqs_hr * power_hr) / np.sum(power_hr)
            bpm = peak_freq * 60
            hr_estimates.append(bpm)
            times_hr.append(start / FPS + window_size / 2)
        else:
            hr_estimates.append(np.nan)
            times_hr.append(start / FPS + window_size / 2)
    
    # Compute average and median HR (ignore NaNs)
    hr_array = np.array(hr_estimates)
    avg_hr = np.nanmean(hr_array)
    median_hr = np.nanmedian(hr_array)
    
    # Plot
    fig, axs = plt.subplots(2, 1, figsize=(12, 8))
    
    # Time-domain signal with HR overlay
    axs[0].plot(time, signal_processed, color='black', label='rPPG signal')
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Intensity")
    axs[0].set_title(f"Time-domain Signal with Sliding HR - {label}")
    axs[0].twinx().plot(times_hr, hr_estimates, 'r-o', label='HR (BPM)')
    axs[0].set_ylabel("Heart Rate (BPM)")
    axs[0].legend(loc='upper right')
    
    # HR over time
    axs[1].plot(times_hr, hr_estimates, 'r-o', label='Sliding HR')
    axs[1].axhline(avg_hr, color='g', linestyle='--', label=f"Avg HR = {avg_hr:.1f} BPM")
    axs[1].axhline(median_hr, color='b', linestyle='--', label=f"Median HR = {median_hr:.1f} BPM")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Heart Rate (BPM)")
    axs[1].set_title("Estimated Heart Rate Over Time")
    axs[1].legend()
    axs[1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Average Heart Rate: {avg_hr:.2f} BPM")
    print(f"Median Heart Rate: {median_hr:.2f} BPM")
    
    return times_hr, hr_estimates, avg_hr, median_hr

def sliding_power_spectrum_hr(signal,label, fps=Video.FPS,
                              hr_low=0.7, hr_high=4.0,
                              window_size=15, step_size=5):
    """
    Compute sliding-window power spectrum and average them to get a robust HR estimate.
    
    Parameters:
    - signal: preprocessed rPPG signal (1D array)
    - fps: sampling rate
    - label: string for plotting
    - hr_low, hr_high: HR frequency range in Hz (default 0.7-4 Hz)
    - window_size: window length in seconds
    - step_size: step between windows in seconds
    
    Returns:
    - times_hr: center times of each window
    - hr_sliding: sliding HR estimates in BPM
    - avg_hr: mean HR across windows
    - median_hr: median HR across windows
    - freqs_avg: frequency bins of averaged spectrum
    - power_avg: averaged power spectrum
    """

    n_samples = len(signal)
    win_samples = int(window_size * fps)
    step_samples = int(step_size * fps)

    times_hr = []
    hr_sliding = []

    power_accum = None  # To accumulate power spectra

    # Sliding window
    for start in range(0, n_samples - win_samples + 1, step_samples):
        segment = signal[start:start + win_samples]
        windowed = segment * np.hanning(win_samples)

        # FFT
        fhat = np.fft.fft(windowed)
        freqs = np.fft.fftfreq(win_samples, d=1/fps)
        power = np.abs(fhat)**2 / win_samples

        # Positive frequencies
        pos_mask = freqs > 0
        freqs_pos = freqs[pos_mask]
        power_pos = power[pos_mask]

        # Accumulate for average spectrum
        if power_accum is None:
            power_accum = np.zeros_like(power_pos)
        power_accum += power_pos

        # HR band
        hr_mask = (freqs_pos >= hr_low) & (freqs_pos <= hr_high)
        freqs_hr = freqs_pos[hr_mask]
        power_hr = power_pos[hr_mask]

        if len(power_hr) > 0 and np.sum(power_hr) > 0:
            # Power-weighted mean frequency
            peak_freq = np.sum(freqs_hr * power_hr) / np.sum(power_hr)
            bpm = peak_freq * 60
            hr_sliding.append(bpm)
        else:
            hr_sliding.append(np.nan)

        times_hr.append(start/fps + window_size/2)

    # Average the accumulated spectrum
    power_avg = power_accum / len(range(0, n_samples - win_samples + 1, step_samples))
    freqs_avg = freqs_pos

    # Global HR from averaged spectrum in HR band
    hr_mask_avg = (freqs_avg >= hr_low) & (freqs_avg <= hr_high)
    freqs_hr_avg = freqs_avg[hr_mask_avg]
    power_hr_avg = power_avg[hr_mask_avg]

    if len(power_hr_avg) > 0 and np.sum(power_hr_avg) > 0:
        global_peak_freq = np.sum(freqs_hr_avg * power_hr_avg) / np.sum(power_hr_avg)
        global_hr = global_peak_freq * 60
    else:
        global_hr = np.nan

    # Mean/median sliding HR
    hr_array = np.array(hr_sliding)
    avg_hr = np.nanmean(hr_array)
    median_hr = np.nanmedian(hr_array)

    # Plotting
    fig, axs = plt.subplots(3,1, figsize=(12,12))

    # 1. Time signal
    axs[0].plot(np.arange(n_samples)/fps, signal, color='black')
    axs[0].set_title(f"Time-domain rPPG Signal - {label}")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Intensity")

    # 2. Sliding HR over time
    axs[1].plot(times_hr, hr_sliding, 'r-o', label='Sliding HR')
    axs[1].axhline(avg_hr, color='g', linestyle='--', label=f"Mean HR = {avg_hr:.1f} BPM")
    axs[1].axhline(median_hr, color='b', linestyle='--', label=f"Median HR = {median_hr:.1f} BPM")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("HR (BPM)")
    axs[1].set_title("Sliding HR Over Time")
    axs[1].legend()
    axs[1].grid(True)

    # 3. Averaged Power Spectrum
    axs[2].plot(freqs_avg, power_avg, color='orange', label='Averaged Power Spectrum')
    axs[2].axvline(global_peak_freq, color='red', linestyle='--',
                   label=f'Global Peak: {global_peak_freq:.2f} Hz ({global_hr:.1f} BPM)')
    axs[2].set_xlim(0,5)
    axs[2].set_xlabel("Frequency (Hz)")
    axs[2].set_ylabel("Power")
    axs[2].set_title("Averaged Power Spectrum")
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()

    print(f"Mean sliding HR: {avg_hr:.2f} BPM")
    print(f"Median sliding HR: {median_hr:.2f} BPM")
    print(f"Global HR from averaged spectrum: {global_hr:.2f} BPM")

    return times_hr, hr_sliding, avg_hr, median_hr, freqs_avg, power_avg
