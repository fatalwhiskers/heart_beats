import matplotlib.pyplot as plt
import numpy as np
import src.video_reader as vr
import test as test
import src.extract_wave as ext
from src.config import Video 
from src.config import Signal

def runLoad():
    video_path = r"data\Dataset1\vid_s28_T1.avi"
    output_path = r"outputs"
    video_array = vr.read_video_to_array(video_path, True, False, True)  # Shape: (num_frames, height, width, channels) (Blue Green Red)

    #test.render_frame(video_array[0])
    R_signal, G_signal, B_signal = ext.extract_rgb_signals_BGR(video_array)

    G_signal = ext.bandpass_filter(G_signal, Video.FPS)
    R_signal = ext.bandpass_filter(R_signal, Video.FPS)
    B_signal = ext.bandpass_filter(B_signal, Video.FPS)
    ext.save_rgb_signals(output_path, R_signal, G_signal, B_signal)
    #plot_signal(G_signal)
    #bpm_over_time(G_signal)
   

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

def plot_signal(signal_processed):
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
    axs[0].plot(time, signal_processed, color='green')
    axs[0].set_title("1. Filtered Green Channel (Time Domain)")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Intensity")

    # 2. FFT
    axs[1].plot(freqs, fft_mag, color='purple')
    axs[1].set_xlim(0, 5)  # Focus on heart rate band
    axs[1].set_title("2. FFT Magnitude Spectrum")
    axs[1].set_xlabel("Frequency (Hz)")
    axs[1].set_ylabel("Amplitude")

    # 3. Power Spectrum
    axs[2].plot(freqs, power_spec, color='orange')
    axs[2].axvline(peak_freq, color='red', linestyle='--', label=f'Peak: {peak_freq:.2f} Hz ({bpm:.1f} BPM)')
    axs[2].set_xlim(0, 5)
    axs[2].set_title("3. Power Spectrum")
    axs[2].set_xlabel("Frequency (Hz)")
    axs[2].set_ylabel("Power")
    axs[2].legend()

    plt.tight_layout()
    plt.show()

    print(f"Estimated Heart Rate: {bpm:.2f} BPM")


def bpm_over_time(signal_processed):
    window_size_sec = 4  
    window_size = int(window_size_sec * Video.FPS)  
    step_size = window_size - 2

    bpm_list = []
    time_bins = []

    for start in range(0, len(signal_processed) - window_size + 1, step_size):
        segment = signal_processed[start:start + window_size]
        
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

    # Plot heart rate over time
    plt.plot(time_bins, bpm_list)
    plt.xlabel('Time (s)')
    plt.ylabel('Heart Rate (BPM)')
    plt.title('Heart Rate Over Time')
    plt.show()


def main():
    runLoad()



if __name__ == "__main__":
    main()