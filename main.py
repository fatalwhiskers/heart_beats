import matplotlib.pyplot as plt
import numpy as np
import src.video_reader as vr
import test as test
import src.extract_wave as ext
import argparse
import sys
import os
from src.config import Video 
from src.config import Signal
from sklearn.decomposition import PCA
from scipy.interpolate import interp1d

def runLoad(channels=['G'], cropping = True, face_tracking = False, interpolate = True, Display = False, Testing = False):
    output_path = r"outputs"  
    folder_path = r"data\Dataset1"
    csv_path = r"data\CSVFiles\Settings.csv"
    crop_list = vr.load_crop_settings(csv_path)
    
    for filename, x1, y1, x2, y2 in crop_list:

        video_path = os.path.join(folder_path, filename)
        video_array, time_array = vr.read_video_to_array(video_path, x1, y1, x2, y2, cropping, Display, Testing)  # Shape: (num_frames, height, width, channels) (Blue Green Red)
        
        #test.render_frame(video_array[0])
        R_signal, G_signal, B_signal = ext.extract_rgb_signals_BGR(video_array)
        if interpolate:
            R_signal , t_uniform = interpolate_signal_with_timestamps(R_signal, time_array) 
            B_signal , t_uniform = interpolate_signal_with_timestamps(B_signal, time_array)
            plt.figure()
            plt.plot(time_array, G_signal, 'o-', label='Original G (raw)')
            G_signal , t_uniform = interpolate_signal_with_timestamps(G_signal, time_array)
            plt.plot(t_uniform, G_signal, '-x', label='Interpolated G (35 Hz)')
            plt.xlabel('Time (s)')
            plt.ylabel('Green Signal')
            plt.legend()
            plt.title('Raw vs Interpolated Green Signal')
            plt.show()
        #G_signal = ext.bandpass_filter(G_signal, Video.FPS)
        #R_signal = ext.bandpass_filter(R_signal, Video.FPS)
        #B_signal = ext.bandpass_filter(B_signal, Video.FPS)

        signals = {}

        if 'R' in channels:
            signals['R'] = ext.bandpass_filter(R_signal, Video.FPS)

        if 'G' in channels:
            signals['G'] = ext.bandpass_filter(G_signal, Video.FPS)
            signals['G_raw'] = ext.bandpass_filter(G_signal, Video.FPS)
        if 'B' in channels:
            signals['B'] = ext.bandpass_filter(B_signal, Video.FPS)
        if 'GREY_W' in channels:
            gray_w = 0.2989 * R_signal + 0.5870 * G_signal + 0.1140 * B_signal
            gray_w = ext.bandpass_filter(gray_w, Video.FPS)
            signals['GREY_W'] = gray_w
        if 'GREY_A' in channels:
            gray_a = (R_signal + G_signal + B_signal) / 3.0
            gray_a = ext.bandpass_filter(gray_a, Video.FPS)
            signals['GREY_A'] = gray_a
        if 'PCA' in channels:
            pca_components = extract_pca_components(R_signal, G_signal, B_signal)
            for i in range(min(3, pca_components.shape[1])):
                signals[f'PCA_{i+1}'] = ext.bandpass_filter(pca_components[:, i], Video.FPS)
        if 'ZCA' in channels:
            zca_components = zca_whiten(R_signal, G_signal, B_signal)
            for i in range(min(3, zca_components.shape[1])):
                signals[f'ZCA_{i+1}'] = ext.bandpass_filter(zca_components[:, i], Video.FPS)


        # loop though signals

        for label, signal_data in signals.items():
            print(f"\nAnalyzing signal: {label}")
           # print("Correlation:", np.corrcoef(G_signal, gray_w)[0, 1])
            print("First 5 values:", signal_data[:5])
            plot_signal(signal_data, label)
            bpm_over_time(signal_data, label)

   # plot_signal(G_signal)
   # bpm_over_time(G_signal)
   
def zca_whiten(R, G, B, epsilon=1e-5):
    signal_matrix = np.vstack((R, G, B)).T  # shape (time, channels)
    
    # Center the data
    X = signal_matrix - np.mean(signal_matrix, axis=0)
    
    # Compute covariance
    sigma = np.cov(X, rowvar=False)
    
    # Eigen-decomposition
    U, S, _ = np.linalg.svd(sigma)
    
    # ZCA Whitening matrix
    ZCA_matrix = U @ np.diag(1.0 / np.sqrt(S + epsilon)) @ U.T
    X_zca = X @ ZCA_matrix.T
    
    return X_zca  # shape (time, 3)

def interpolate_signal_with_timestamps(signal, timestamps, target_fps=35):
    start_time = timestamps[0]
    end_time = timestamps[-1]
    duration = end_time - start_time

    # 1. Create a uniform time grid (target sampling points)
    num_target_samples = int(duration * target_fps)
    t_target = np.linspace(start_time, end_time, num_target_samples)

    # 2. Create interpolation function from the original data
    interp_func = interp1d(timestamps, signal, kind='linear', fill_value="extrapolate")

    # 3. Evaluate the interpolation function at the uniform time points
    interpolated_signal = interp_func(t_target)

    return interpolated_signal, t_target

def extract_pca_components(R, G, B, n_components=3):
    signal_matrix = np.vstack((R, G, B)).T
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(signal_matrix)
    print("PCA explained variance ratio:", pca.explained_variance_ratio_)
    return components

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

    print(f"Estimated Heart Rate: {bpm:.2f} BPM")


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

# example usage python main.py --channels G R --face_tracking
def main():
    parser = argparse.ArgumentParser(description="Video signal processing CLI")
    parser.add_argument(
        '--channels',
        nargs='+',
        choices=['R', 'G', 'B', 'GREY_W', 'GREY_A', 'PCA', 'ZCA'],
        default=['G'],
        help='Color channels: R, G, B, GREY_W (weighted), GREY_A (average) , PCA, ZCA'
    )
    parser.add_argument(
        '--face_tracking',
        action='store_true',
        help='Enable face tracking'
    )

    # Only parse args if they exist (i.e., from command line)
    if len(sys.argv) > 1:
        args = parser.parse_args()
    else:
        # Defaults for IDE or test environment
        args = parser.parse_args(args=[])

        # Optional: manually override for testing here
        args.channels = ['G', 'PCA' , 'ZCA']
        #args.face_tracking = False

    runLoad(channels=args.channels, face_tracking=args.face_tracking)



if __name__ == "__main__":
    main()