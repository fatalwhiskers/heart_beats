import numpy as np
from .config import Signal, Video
from scipy import signal
from sklearn.decomposition import PCA, FastICA
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, detrend

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

def bandpass_filter(signal, fs, lowcut=Signal.HR_LOW, highcut=Signal.HR_HIGH, order=Signal.HR_ORDER):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

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

def ICA(R, G, B):

    RGB_array = np.vstack([R, G, B]).T.astype(float)

    ica = FastICA(n_components=3)
    # Center the data
    ICA_RGB_S = ica.fit_transform(RGB_array)

    freqs, psd = signal.welch(ICA_RGB_S, Video.FPS, nperseg=min(1024, len(ICA_RGB_S)))

    hr_mask = (freqs >= Signal.HR_LOW) & (freqs <= Signal.HR_HIGH)
    band_power = psd[:, hr_mask].sum(axis=1)
    best_idx = int(np.argmax(band_power))

    best_guess_signal = ICA_RGB_S[:, best_idx]

    
    return best_guess_signal, best_idx  # shape (time, 3)

def detrend_sig(signal):
    return detrend(signal)

def save_rgb_signals(file_path, R, G, B):
    np.savez(file_path, R=R, G=G, B=B)

def load_rgb_signals(file_path):
    data = np.load(file_path)
    return data['R'], data['G'], data['B']