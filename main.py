import src.Video_extraction as VE
import src.not_working.BVP as BVP
import src.hilbert_prv as hilly
import src.rppg as rPPG
import src.not_working.testv2BVP as vid
import pandas as pd
import src.plotter as plotter
import test as test
import numpy as np
import src.extract_wave as ext
import src.not_working.ECG_HR as ecg
import argparse
import sys
import os
import src.Stats as stat
from pathlib import Path
from src.config import Video, fileDataset1, fileDataset2, fileDataset3, BVP, rppg, Signal, PRV
from scipy.signal import welch, butter, filtfilt, get_window
import cv2
import src.not_working.testv2BVP as bvp_test
import numpy as np
from dataclasses import dataclass
import neurokit2 as nk

test = False
"""vid_s9_T1.avi, bvp_s9_T1.csv,  615, 217, 985,  641      
vid_s9_T2.avi, bvp_s9_T2.csv,  433, 125, 750,  500   
vid_s9_T3.avi, bvp_s9_T3.csv,  450, 260, 730,  520                                                                
vid_s28_T1.avi, bvp_s28_T1.csv, 520, 115, 880, 730
vid_s28_T2.avi, bvp_s28_T2.csv, 520, 115, 880, 730"""

def synth_bvp(fs=30.0, dur=120.0, hr_bpm=72.0, rr_bpm=15.0, noise=0.02):
    t = np.arange(0, dur, 1/fs)
    # make beats
    f_resp = rr_bpm/60.0
    mean_ibi = 60.0/hr_bpm
    rng = np.random.default_rng(7)
    beats = [0.5]
    while beats[-1] < dur:
        ibi = mean_ibi + 0.06*np.sin(2*np.pi*f_resp*beats[-1]) + rng.normal(0, 0.015)
        beats.append(beats[-1] + max(0.4, ibi))
    beat_times = np.array(beats[:-1])
    # pulse train -> convolve with kernel
    imp = np.zeros_like(t); imp[np.clip((beat_times*fs).astype(int), 0, len(t)-1)] = 1.0
    k_len = int(0.5*fs); xk = np.linspace(0,1,k_len)
    kernel = np.exp(-6*xk)*(1-np.exp(-40*xk)); kernel /= kernel.max()
    bvp = np.convolve(imp, kernel, mode='same')
    # noise + tiny drift
    bvp += noise*np.random.default_rng(3).standard_normal(len(t))
    bvp += 0.05*np.sin(2*np.pi*0.01*t)
    return t, bvp

import matplotlib.pyplot as plt
def synthtic_test():
    fs = 30.0
    t, bvp = synth_bvp(fs=fs, dur=180.0, hr_bpm=72, rr_bpm=15)

    # Run the windowed analyzer (30s windows / 5s hop)
    res = bvp_test.analyze_bvp_windowed(bvp, fs, win_sec=30.0, step_sec=5.0,
                            interp_fs=256.0, return_intermediates=True, debug=True)

    # Print simple checks
    print(f"Mean HR (BPM): {np.nanmean(res['HR_bpm']):.1f}")
    print(f"Mean RR (brpm): {np.nanmean(res['RR_bpm']):.1f}")
    print(f"Valid windows: {np.sum(np.isfinite(res['HR_bpm']))}/{len(res['HR_bpm'])}")

    # Plot HR over time
    plt.figure()
    plt.plot(res["t_center"], res["HR_bpm"], '-o')
    plt.xlabel("Time (s)"); plt.ylabel("HR (BPM)"); plt.title("Windowed HR (synthetic)")
    plt.show()

    # Plot RR over time
    plt.figure()
    plt.plot(res["t_center"], res["RR_bpm"], '-o')
    plt.xlabel("Time (s)"); plt.ylabel("RR (breaths/min)"); plt.title("Windowed RR (synthetic)")
    plt.show()

    

    return

def testing():
    CSV_path = os.path.join(fileDataset2.folder_path, r"Physiological\2ea4\2ea4_Baseline.txt")   
    # --- load your file and compute BPM ---
    fs = 500.0  # change if your true sampling rate is different
    df = pd.read_csv(CSV_path)   # columns: ECG, EDA, RR
    ecg_data = df["ECG"].to_numpy(dtype=float)

    peaks = ecg.detect_r_peaks(ecg_data)
    bpm_mean, bpm_series, bpm_time = ecg.bpm_from_peaks(peaks)
    t_mid_gt, hr_bpm_gt, r_peaks_gt, rr_s_gt = ecg.ecg_hr_from_signal_500hz(ecg_data)
    print(f"Detected beats: {len(peaks)}")
    print(f"Mean heart rate: {bpm_mean:.2f} BPM")
    
    return

def get_Signals(channels, R_signal, G_signal, B_signal):
    signals = {}

    if 'R' in channels or 'ALL' in channels:
        signals['R'] = R_signal

    if 'G' in channels or 'ALL' in channels:
        signals['G'] = G_signal

    if 'B' in channels or 'ALL' in channels:
        signals['B'] = B_signal

    if 'GREY_W' in channels or 'ALL' in channels:
        signals['GREY_W'] = 0.2989 * R_signal + 0.5870 * G_signal + 0.1140 * B_signal

    if 'GREY_A' in channels or 'ALL' in channels:
        signals['GREY_A'] = (R_signal + G_signal + B_signal) / 3.0

    if 'PCA' in channels or 'ALL' in channels:
        pca_components = ext.extract_pca_components(R_signal, G_signal, B_signal)
        for i in range(min(3, pca_components.shape[1])):
            signals[f'PCA_{i+1}'] = pca_components[:, i]

    if 'ZCA' in channels or 'ALL' in channels:
        zca_components = ext.zca_whiten(R_signal, G_signal, B_signal)
        for i in range(min(3, zca_components.shape[1])):
            signals[f'ZCA_{i+1}'] = zca_components[:, i]

    if 'ICA' in channels or 'ALL' in channels:
        ICA_components, best_idx = ext.ICA_Test(R_signal, G_signal, B_signal)
        for i in range(min(3, ICA_components.shape[1])):

            if i == best_idx:
                signals[f'Best_ICA_{i+1}'] = ICA_components[:, i]
            else:
                signals[f'ICA_{i+1}'] = ICA_components[:, i]
    
    if 'CHROM' in channels or 'ALL' in channels:
        signals['CHROM'] = ext.chrom_pos_windowed(R_signal, G_signal, B_signal, method='CHROM')


    if 'POS' in channels or 'ALL' in channels:
        signals['POS'] = ext.chrom_pos_windowed(R_signal, G_signal, B_signal, method='POS')

    return signals


def runDataset1(channels=['G'], cropping = True, crop_modes = "manual", interpolate = True, apply_bandpass = True,  Display = False, Testing = False):
    Video.FPS = 35
    Video.target_FPS = 35
    crop_list = VE.load_crop_settings(fileDataset1.csv_path)
    
    for filename, file_CSV, x1, y1, x2, y2 in crop_list:
        Video.Csv_path = file_CSV
        video_path = os.path.join(fileDataset1.folder_path, filename)

        for crop_mode in ([crop_modes] if isinstance(crop_modes, str) else crop_modes):
            R_signal, G_signal, B_signal, time_array = VE.extract_video_to_rgb(video_path, x1, y1, x2, y2, crop_mode, Display, Testing)        
            signals = get_Signals(channels, R_signal, G_signal, B_signal)
            for label, signal_data in signals.items():
                build_table(signal_data, time_array, label, crop_mode, filename, file_CSV)
                
def runDataset2(channels=['G'], cropping = True, crop_modes = "manual", interpolate = True, apply_bandpass = True,  Display = False, Testing = False):
    Video.FPS = 15
    Video.target_FPS = 15
    crop_list = VE.load_crop_settings_D2(fileDataset2.csv_path)
    
    for subject, filename, file_CSV, x1, y1, x2, y2 in crop_list:
        video_path = os.path.join(fileDataset2.folder_path, "Videos", filename)
        CSV_path = os.path.join(fileDataset2.folder_path, file_CSV)   
        df = pd.read_csv(CSV_path)
        ecg_signal = df["ECG"].values
        signals, info = nk.ecg_process(ecg_signal, sampling_rate=500)

        # Access results
        r_peaks = info["ECG_R_Peaks"]           # indices of R-peaks
        hr_gt = signals["ECG_Rate"]                # instantaneous heart rate (BPM)
        hr_time_gt = signals["ECG_Rate"].index/500

        fs = 500  # <-- replace with your actual sampling frequency
        time = np.arange(len(ecg_signal) ) / fs
        
        total_duration = float(np.max(time))
        windows = make_windows(int(total_duration), rppg.window_size,rppg.step_size)
        hr_t_windows, hr_gt_windows = get_window_hr(hr_time_gt, hr_gt.values, windows)

        for crop_mode in ([crop_modes] if isinstance(crop_modes, str) else crop_modes):

            R_signal, G_signal, B_signal, time_array = VE.extract_video_to_rgb(video_path, x1, y1, x2, y2, crop_mode)
            signals = get_Signals(channels, R_signal, G_signal, B_signal)
            for label, signal_data in signals.items():
                build_table_2(signal_data, time_array, crop_mode, label, subject, filename, file_CSV, hr_gt_windows, hr_t_windows)

def runDataset3(channels=['G'], cropping = True, crop_modes = "manual", interpolate = True, apply_bandpass = True,  Display = False, Testing = False):
    Video.FPS = 60
    Video.target_FPS = 60
    crop_list = VE.load_crop_settings(fileDataset3.csv_path)

    for filename, file_CSV, x1, y1, x2, y2 in crop_list:
        video_path = os.path.join(fileDataset3.folder_path, filename)
        for crop_mode in ([crop_modes] if isinstance(crop_modes, str) else crop_modes):
            R_signal, G_signal, B_signal, time_array = VE.extract_video_to_rgb(video_path, x1, y1, x2, y2, crop_mode)
            signals = get_Signals(channels, R_signal, G_signal, B_signal)
            for label, signal_data in signals.items():
                build_table_3(signal_data, time_array, crop_mode, label, filename)
    return

def smooth_signal(sig, window_size=5):
    kernel = np.ones(window_size) / window_size
    return np.convolve(sig, kernel, mode='same')


def build_table(signal_data, time_array, label, crop_mode, filename, file_CSV):

   # _, signal_data = ext.upsample_cubic(signal_data)
    window_centers_t_fft, hr_estimates_fft = rPPG.estimate_hr_pyvhr_nt(time_array, signal_data)

    window_centers_t_welch, hr_estimates_welch = rPPG.estimate_hr_welch_nk(time_array, signal_data)

    plt.figure(); 
    plt.plot(window_centers_t_fft, hr_estimates_fft,  label="FFT/BPM (pyVHR)", marker='o') 
    plt.plot(window_centers_t_welch, hr_estimates_welch,label="Welch (nk)",      marker='s')
    plt.xlabel("Time (s)"); plt.ylabel("HR (bpm)"); plt.legend(); plt.title("HR estimates"); 
    plt.show()


    pp_clean, hr_inst_clean, hr_inst_raw, t_mid_pp, _, _  = hilly.estimate_prv_hilbert_simple(time_array, signal_data)  
    total_duration = float(np.max(t_mid_pp))
    windows = make_windows(int(total_duration), rppg.window_size,rppg.step_size)
    hr_t_windows_hilbert, hr_windows_hilbert = get_window_hr(t_mid_pp, hr_inst_clean, windows)

  #  pp_clean, hr_track, hr_inst_raw, t_mid, peaks_t, artifacts_mask  = prv.compute_prv_hr(time_array, signal_data) 
    
    #plotter.plot_hr(hr_values, times, hr_inst_clean, t_mid_pp, label)
    """
    stats = prv.pp_diagnostics(
    pp_raw = np.diff(peaks_t),          # or use pp from pp_intervals_from_peaks
    pp_clean = pp_clean,
    t_mid = t_mid,
    artifacts_mask = artifacts_mask,
    verbose = True
    )  """
    bvp_path = os.path.join(fileDataset1.folder_path, file_CSV)
    ppg = np.loadtxt(bvp_path)  # ensure it’s a single numeric column

    signals_gt, info = nk.ppg_process(ppg, sampling_rate=BVP.BVP_RATE)

    hr_gt = signals_gt["PPG_Rate"]
    time_gt = np.arange(len(ppg)) / BVP.BVP_RATE
    total_duration = float(np.max(time_gt))
    windows = make_windows(int(total_duration), rppg.window_size,rppg.step_size)
    hr_t_windows, hr_gt_windows = get_window_hr(time_gt, hr_gt.values, windows)
    

    result = filename.split("_")[1]
    plotter.build_table(rPPG=hr_estimates_fft, rPPG_time=window_centers_t_fft, ground_truth=hr_gt_windows, gt_time=hr_t_windows, subject = result, recording_id= filename, signal_label=label, cropMode=crop_mode) 
    plotter.build_table(rPPG=hr_estimates_welch, rPPG_time=window_centers_t_welch, ground_truth=hr_gt_windows, gt_time=hr_t_windows, subject = result, recording_id= filename, signal_label=label, cropMode=crop_mode) 
    plotter.build_table(rPPG=hr_windows_hilbert, rPPG_time=hr_t_windows_hilbert, ground_truth=hr_gt_windows, gt_time=hr_t_windows, subject = result, recording_id= filename, signal_label=label, cropMode=crop_mode) 


    return
from scipy import signal

def build_table_2(signal_data, time_array, crop_mode, label, subject, filename, file_CSV, hr_gt_windows, hr_t_windows):
    #signal_data = smooth_signal(signal_data)
   # time_up , signal_data = ext.upsample_cubic(signal_data, 30)
    fs = 30.0
    #signal_data = signal.detrend(signal_data, type='linear')
    #signal_data = ext.bandpass_filter_old(signal_data, fs=30)
    
    #pp_clean, hr_inst_clean, hr_inst_raw, t_mid_pp, _, _  = hilly.estimate_prv_hilbert_simple(time_array, signal_data)  
   # total_duration = float(np.max(t_mid_pp))
   # windows = make_windows(int(total_duration), rppg.window_size,rppg.step_size)
    #hr_t_hilbert, hr_beats_hilbert = get_window_hr(t_mid_pp, hr_inst_clean, windows)
    #hr_res = rPPG.hr_from_psd(signal_data, fs, fmin=0.7, fmax=3.0)
   # times, hr_values , psd_accum, freqs_band_ref = rPPG.sliding_welch_hr_center_best(signal_data, 30)
    #plotter.build_table_ECG(rPPG=hr_inst_clean, ground_truth= hr_gt_windows, t_rPPG=t_mid_pp, t_ref=hr_t_windows, recording_id=file_id, subject=subject,  signal_label=label+ " hilbert", cropMode=crop_mode) 

    #times, hr_values = rPPG.estimate_hr_welch_nk(time_array, signal_data)

   # results = ecg.run_ecg_pipeline(CSV_path)
    #plotter.plot_hr(hr_beats_hilbert, hr_t_hilbert, hr_gt_windows,hr_t_windows, label)
   # plotter.plot_hr(hr_values, times, hr_gt_windows, hr_t_windows, label)

    window_centers_t_fft, hr_estimates_fft = rPPG.estimate_hr_welch_nk(time_array, signal_data)
    file_id = Path(filename).stem
    plotter.build_table_ECG(rPPG=hr_estimates_fft, ground_truth= hr_gt_windows, t_rPPG=window_centers_t_fft, t_ref=hr_t_windows, recording_id=file_id, subject=subject,  signal_label=label, cropMode=crop_mode) 
    
   # pp_clean, hr_track, hr_inst_raw, t_mid, peaks_t, artifacts_mask  = prv.compute_prv_hr(time_array, signal_data) 
   # windows = make_windows(int(total_duration), rppg.window_size,rppg.step_size)
  #  hr_t_prv, hr_beats_prv = get_window_hr( t_mid, hr_track, windows)
  #  plotter.plot_hr(hr_beats_prv, hr_t_prv, hr_gt_windows,hr_t_windows, label)
  #  plotter.build_table_ECG(rPPG=hr_beats_prv, ground_truth= hr_gt_windows, t_rPPG=hr_t_prv, t_ref=hr_t_windows, recording_id=file_id, subject=subject,  signal_label=label + " PRV", cropMode=crop_mode) 

   
    return

def build_table_3(signal_data, time_array, crop_modes, label, filename):
    window_centers_t_fft, hr_estimates_fft = rPPG.estimate_hr_fft_nt(time_array, signal_data)
    window_centers_t_welch, hr_estimates_welch = rPPG.estimate_hr_welch_nk(time_array, signal_data)
    pp_clean, hr_inst_clean, hr_inst_raw, t_mid_pp, _, _  = hilly.estimate_prv_hilbert_simple(time_array, signal_data)  
    total_duration = float(np.max(t_mid_pp))
    windows = make_windows(int(total_duration), rppg.window_size,rppg.step_size)
    hr_t_windows_hilbert, hr_windows_hilbert = get_window_hr(t_mid_pp, hr_inst_clean, windows)

    #plotter.build_table3(rPPG=hr_estimates_fft, rPPG_time=window_centers_t_fft, recording_id= filename, signal_label=label+" fft", cropMode=crop_modes) 
    #plotter.build_table3(rPPG=hr_estimates_welch, rPPG_time=window_centers_t_welch, recording_id= filename, signal_label=label+" welch", cropMode=crop_modes) 
    #plotter.build_table3(rPPG=hr_inst_clean, rPPG_time=t_mid_pp, recording_id= filename, signal_label=label+" hilbert", cropMode=crop_modes) 

    return

from scipy import interpolate
def get_window_hr(hr_time, hr_values, windows):
    hr_per_win = []
    times = []
    
    for (t0, t1) in windows:
        mask = (hr_time >= t0) & (hr_time < t1)
        if np.any(mask):
            hr_summary = np.mean(hr_values[mask])
            hr_per_win.append(hr_summary)
        else:
            hr_per_win.append(np.nan)  # mark empty windows
        
        # Use the midpoint of the window as its time
        times.append((t0 + t1) / 2.0)
    
    return np.array(times), np.array(hr_per_win)

def make_windows(total_duration: int, window_size: int, step_size: int):
    starts = np.arange(0, total_duration - window_size + 1, step_size)
    return [(int(s), int(s) + window_size) for s in starts]

def plot_hr_series_side_by_side(
    hr_r, t_r,
    hr_c, t_c, 
    resample_hz=2.0,
    max_lag_s=3.0,
    smooth_match_sec=None,
    labels=("rPPG", "cPPG")
):
    """Interpolate both HR series to common grid, apply best lag, then plot."""

    # build common time grid
    tmin = max(np.nanmin(t_r), np.nanmin(t_c))
    tmax = min(np.nanmax(t_r), np.nanmax(t_c))
    fs = resample_hz
    grid = np.arange(tmin, tmax, 1.0/fs)

    def interp_to_grid(t, y):
        m = np.isfinite(t) & np.isfinite(y)
        f = interpolate.interp1d(t[m], y[m], kind="linear",
                                 bounds_error=False, fill_value=np.nan)
        return f(grid)

    xr = interp_to_grid(t_r, hr_r)
    yc = interp_to_grid(t_c, hr_c)

    # optional smoothing (to match slower signal)
    if smooth_match_sec and smooth_match_sec > 0:
        k = max(1, int(round(smooth_match_sec * fs)))
        kernel = np.ones(k)/k
        def movavg(x): 
            num = np.convolve(np.nan_to_num(x), kernel, mode="same")
            den = np.convolve(np.isfinite(x).astype(float), kernel, mode="same")
            out = num/den; out[den < 0.5] = np.nan
            return out
        xr = movavg(xr)
        yc = movavg(yc)

    # crude lag correction: find cross-corr max
    max_lag = int(round(max_lag_s * fs))
    best_lag = 0
    best_corr = -np.inf
    for lag in range(-max_lag, max_lag+1):
        if lag >= 0:
            x_shift = np.r_[np.full(lag, np.nan), xr[:-lag or None]]
        else:
            x_shift = np.r_[xr[-lag:], np.full(-lag, np.nan)]
        m = np.isfinite(x_shift) & np.isfinite(yc)
        if np.count_nonzero(m) < 10: 
            continue
        c = np.corrcoef(x_shift[m], yc[m])[0,1]
        if c > best_corr:
            best_corr, best_lag, best_x = c, lag, x_shift

    # plot side by side
    plt.figure(figsize=(12,5))
    plt.plot(grid, best_x, label=f"{labels[0]} (lag {best_lag/fs:.2f}s)")
    plt.plot(grid, yc, label=labels[1])
    plt.xlabel("Time (s)")
    plt.ylabel("HR (BPM)")
    plt.title("HR series comparison")
    plt.legend()
    plt.show()


def make_synthetic(fs=35, secs=30, hr_bpm=90):
    t = np.arange(0, secs, 1/fs)
    x = 0.6*np.sin(2*np.pi*(hr_bpm/60.0)*t) + 0.2*np.random.randn(t.size)
    return t, x

# example usage python main.py --channels G R --face_tracking
def main():
    parser = argparse.ArgumentParser(description="Video signal processing rPPG")
    parser.add_argument(
        '--channels',
        nargs='+',
        choices=['R','G','B','GREY_W','GREY_A','PCA','ZCA','ICA','CHROM','POS','ALL'],
        default=['G'],
        help='Color channels: R, G, B, GREY_W (weighted), GREY_A (average) , PCA, ZCA, ALL'
    )
    parser.add_argument(
    '--crop_mode',
    choices=['manual', 'none', 'face_track', 'bbox_forehead', 'mesh_forehead', 'poly', 'bbox_forehead_jitter'],
    default='none',
    help=(
        "Cropping method:\n"
        "manual - use fixed coords (x1,y1,x2,y2)\n"
        "none - no cropping\n"
        "face_track - detect face once then track with KCF\n"
        "bbox_forehead - crop to forehead region using detection bbox\n"
        "mesh_forehead - crop to forehead using mesh landmarks"
        )
    )

    if test:
        testing()

    # Only parse args if they exist (i.e., from command line)
    if len(sys.argv) > 1:
        args = parser.parse_args()
    else:
        # Defaults for IDE or test environment
        args = parser.parse_args(args=[])

        # Optional: manually override for testing here
        args.channels = ['ALL']
        #args.channels = ['POS']
        args.crop_mode = 'manual', 'none', 'face_track', 'bbox_forehead', 'mesh_forehead', 'poly'
        args.crop_mode =  'bbox_forehead_jitter'
       # args.crop_mode = 'poly'
        #args.face_tracking = False

    runDataset1(channels=args.channels, crop_modes=args.crop_mode)
    #runDataset_1_csv(channels=args.channels, crop_modes=args.crop_mode)
    #runDataset2(channels=args.channels, crop_modes=args.crop_mode)
    #runDataset3(channels=args.channels, crop_modes=args.crop_mode)


from scipy.stats import pearsonr
def evaluate_hr_methods(hr_t_windows, hr_gt_windows,
                        window_centers_t_fft, hr_estimates_fft,
                        window_centers_t_welch, hr_estimates_welch,
                        t_mid_pp, hr_inst_clean):
    """Compare FFT, Welch, and Hilbert HR estimates against ground truth."""

    def interp_to(t_src, y_src, t_tgt):
        out = np.interp(t_tgt, t_src, y_src, left=np.nan, right=np.nan)
        out[(t_tgt < np.nanmin(t_src)) | (t_tgt > np.nanmax(t_src))] = np.nan
        return out

    def metrics(y_true, y_pred):
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        if not np.any(mask):
            return dict(MAE=np.nan, RMSE=np.nan, Bias=np.nan, SD_err=np.nan, r=np.nan)
        e = y_pred[mask] - y_true[mask]
        mae = np.mean(np.abs(e))
        rmse = np.sqrt(np.mean(e**2))
        bias = np.mean(e)
        sd = np.std(e, ddof=1)
        r, _ = pearsonr(y_true[mask], y_pred[mask]) if mask.sum() > 2 else (np.nan, None)
        return dict(MAE=mae, RMSE=rmse, Bias=bias, SD_err=sd, r=r)

    # align predictions to GT windows
    pred_fft   = interp_to(window_centers_t_fft,   hr_estimates_fft,   hr_t_windows)
    pred_welch = interp_to(window_centers_t_welch, hr_estimates_welch, hr_t_windows)
    pred_hilb  = interp_to(t_mid_pp,               hr_inst_clean,      hr_t_windows)

    results = {
        "FFT": metrics(hr_gt_windows, pred_fft),
        "Welch": metrics(hr_gt_windows, pred_welch),
        "Hilbert": metrics(hr_gt_windows, pred_hilb),
    }

    # print results
    print("\nMethod comparison:")
    for name, m in results.items():
        print(f"{name:8s} | MAE={m['MAE']:.2f} | RMSE={m['RMSE']:.2f} "
              f"| Bias={m['Bias']:.2f} | SD={m['SD_err']:.2f} | r={m['r']:.3f}")

    # pick best by MAE
    best = min(results.items(), key=lambda kv: kv[1]["MAE"] if np.isfinite(kv[1]["MAE"]) else np.inf)[0]
    print(f"\nBest method: {best}")

    return results, best

import os, re, numpy as np, pandas as pd

def _slug(s: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', str(s).strip().lower()).strip('_')

def _align_to_base(time_base, t_vec, y_vec):
    """Align y(t) to time_base by linear interpolation; outside range -> NaN."""
    tb = np.asarray(time_base, float)
    t  = np.asarray(t_vec, float)
    y  = np.asarray(y_vec, float)
    out = np.full_like(tb, np.nan, dtype=float)
    if len(t) < 2 or np.allclose(np.nanstd(t), 0.0):
        return out
    mask = (tb >= np.nanmin(t)) & (tb <= np.nanmax(t))
    out[mask] = np.interp(tb[mask], t, y)
    return out

class VideoCSVBuilder:
    """
    Build one wide CSV per video × ROI with columns:
      video_id, time_s, gt_hr_bpm, <tech>_<method>_<roi>, ...
    """
    def __init__(self, out_dir: str, also_long: bool = False):
        self.out_dir = out_dir
        self.also_long = also_long
        os.makedirs(self.out_dir, exist_ok=True)
        self.reset()

    def reset(self):
        self.meta = {}
        self.cols = {}   # name -> array

    def start(self, subject: str, recording_id: str, crop_mode: str,
              time_s: np.ndarray, gt_hr_bpm: np.ndarray):
        self.reset()
        self.meta['subject'] = str(subject)
        self.meta['recording_id'] = str(recording_id)
        self.meta['roi'] = str(crop_mode)
        self.meta['video_id'] = f"{self.meta['subject']}_{os.path.splitext(os.path.basename(recording_id))[0]}"
        self.time_s = np.asarray(time_s, float)
        self.cols['gt_hr_bpm'] = np.asarray(gt_hr_bpm, float)

    def add_curve(self, tech: str, method_label: str, roi: str,
                  t_vec: np.ndarray, hr_vec: np.ndarray):
        col = f"{_slug(tech)}_{_slug(method_label)}_{_slug(roi)}"
        y_aligned = _align_to_base(self.time_s, t_vec, hr_vec)
        self.cols[col] = y_aligned

    def save(self):
        # assemble wide df
        df = pd.DataFrame({'time_s': self.time_s, **self.cols})
        df.insert(0, 'video_id', self.meta['video_id'])
        out_path = os.path.join(self.out_dir, f"{self.meta['video_id']}_{_slug(self.meta['roi'])}.csv")
        df.to_csv(out_path, index=False)

        if self.also_long:
            rows = []
            for c in df.columns:
                if c in ('video_id','time_s','gt_hr_bpm'): 
                    continue
                parts = c.split('_', 2)  # tech, method, roi
                tech  = parts[0] if len(parts) > 0 else 'x'
                method= parts[1] if len(parts) > 1 else 'x'
                roi   = parts[2] if len(parts) > 2 else 'x'
                rows.append(pd.DataFrame({
                    'video_id': df['video_id'],
                    'time_s':   df['time_s'],
                    'source':   'camera',
                    'tech':     tech,
                    'method':   method,
                    'roi':      roi,
                    'hr_bpm':   df[c]
                }))
            rows.append(pd.DataFrame({
                'video_id': df['video_id'],
                'time_s':   df['time_s'],
                'source':   'gt',
                'tech':     'gt',
                'method':   'gt',
                'roi':      'na',
                'hr_bpm':   df['gt_hr_bpm']
            }))
            df_long = pd.concat(rows, ignore_index=True)
            long_path = out_path.replace('.csv', '_long.csv')
            df_long.to_csv(long_path, index=False)

        return out_path

def add_estimates_for_method(signal_data, time_array, label, crop_mode, builder: VideoCSVBuilder):
    """
    Compute FFT/Welch/Hilbert for one method (label) and add 3 columns to builder.
    Columns will be named: fft_<label>_<roi>, welch_<label>_<roi>, hilbert_<label>_<roi>
    """
    # ---- FFT
    t_fft, hr_fft = rPPG.estimate_hr_fft_nt(time_array, signal_data)

    # ---- Welch
    t_welch, hr_welch = rPPG.estimate_hr_welch_nk(time_array, signal_data)

    # ---- Hilbert (instantaneous -> windowed like GT)
    pp_clean, hr_inst_clean, hr_inst_raw, t_mid_pp, _, _ = hilly.estimate_prv_hilbert_simple(time_array, signal_data)
    total_duration = float(np.max(t_mid_pp))
    windows = make_windows(int(total_duration), rppg.window_size, rppg.step_size)
    t_hilb, hr_hilb = get_window_hr(t_mid_pp, hr_inst_clean, windows)

    # ---- Add to CSV builder (auto-aligned to GT time base)
    builder.add_curve('fft',     method_label=label, roi=crop_mode, t_vec=t_fft,   hr_vec=hr_fft)
    builder.add_curve('welch',   method_label=label, roi=crop_mode, t_vec=t_welch, hr_vec=hr_welch)
    builder.add_curve('hilbert', method_label=label, roi=crop_mode, t_vec=t_hilb,  hr_vec=hr_hilb)

def runDataset_1_csv(channels=['G'], cropping=True, crop_modes="manual",
                interpolate=True, apply_bandpass=True, Display=False, Testing=False):
    Video.FPS = 35
    Video.target_FPS = 35
    crop_list = VE.load_crop_settings(fileDataset1.csv_path)
    out_dir = os.path.join("outputs", "dset1_timeseries")

    for filename, file_CSV, x1, y1, x2, y2 in crop_list:
        Video.Csv_path = file_CSV
        video_path = os.path.join(fileDataset1.folder_path, filename)
        subject = filename.split("_")[1] if "_" in filename else "sub"

        # --- Ground truth once per VIDEO (shared across ROIs)
        bvp_path = os.path.join(fileDataset1.folder_path, file_CSV)
        ppg = np.loadtxt(bvp_path)
        signals_gt, info = nk.ppg_process(ppg, sampling_rate=BVP.BVP_RATE)
        hr_gt = signals_gt["PPG_Rate"].values
        time_gt = np.arange(len(ppg)) / BVP.BVP_RATE
        total_duration = float(np.max(time_gt))
        windows = make_windows(int(total_duration), rppg.window_size, rppg.step_size)
        hr_t_windows, hr_gt_windows = get_window_hr(time_gt, hr_gt, windows)

        # --- Start ONE builder for the whole video (all ROIs will be added)
        builder = VideoCSVBuilder(out_dir=out_dir, also_long=False)
        builder.start(subject=subject, recording_id=filename,
                      crop_mode="all_rois", time_s=hr_t_windows, gt_hr_bpm=hr_gt_windows)

        # --- Loop over ROIs and methods, add columns to the SAME builder
        for crop_mode in ([crop_modes] if isinstance(crop_modes, str) else crop_modes):
            R_signal, G_signal, B_signal, time_array = VE.extract_video_to_rgb(
                video_path, x1, y1, x2, y2, crop_mode, Display, Testing
            )
            signals = get_Signals(channels, R_signal, G_signal, B_signal)  # {label: signal}

            for label, signal_data in signals.items():
                add_estimates_for_method(signal_data, time_array, label, crop_mode, builder)

        # --- Save ONE CSV per video (contains all ROIs)
        out_path = builder.save()
        print("Saved:", out_path)


if __name__ == "__main__":
    main()
