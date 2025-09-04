import src.video_reader as vr
import src.Video_extraction as VE
import src.BVP as BVP
import src.PRV as prv
import src.rppg as rPPG
import pandas as pd
import src.plotter as plotter
import test as test
import numpy as np
import src.extract_wave as ext
import src.ECG_HR as ecg
import argparse
import sys
import os
from pathlib import Path
from src.config import Video, fileDataset1, fileDataset2, BVP

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
        signals['CHROM'] = ext.chrom_pos_windowed_nan(R_signal, G_signal, B_signal, method='CHROM')


    if 'POS' in channels or 'ALL' in channels:
        signals['POS'] = ext.chrom_pos_windowed_nan(R_signal, G_signal, B_signal, method='POS')

    return signals


def runDataset1(channels=['G'], cropping = True, crop_modes = "manual", interpolate = True, apply_bandpass = True,  Display = False, Testing = False):


    crop_list = VE.load_crop_settings(fileDataset1.csv_path)
    
    for filename, file_CSV, x1, y1, x2, y2 in crop_list:
        video_path = os.path.join(fileDataset1.folder_path, filename)
        for crop_mode in crop_modes:

            R_signal, G_signal, B_signal, time_array = VE.extract_video_to_rgb(video_path, x1, y1, x2, y2, crop_mode, Display, Testing)
        
            signals = get_Signals(channels, R_signal, G_signal, B_signal)

            for label, signal_data in signals.items():
                if apply_bandpass:
                    signal_data = ext.bandpass_filter(signal_data)
                build_table(signal_data, time_array, label, crop_mode, filename, file_CSV)
                

def runDataset2(channels=['G'], cropping = True, crop_modes = "manual", interpolate = True, apply_bandpass = True,  Display = False, Testing = False):
    crop_list = VE.load_crop_settings(fileDataset2.csv_path)
    
    for subject, filename, file_CSV, x1, y1, x2, y2 in crop_list:
        video_path = os.path.join(fileDataset2.folder_path, filename)
        for crop_mode in crop_modes:

            R_signal, G_signal, B_signal, time_array = VE.extract_video_to_rgb(video_path, x1, y1, x2, y2, crop_mode, Display, Testing)
        
            signals = get_Signals(channels, R_signal, G_signal, B_signal)

            for label, signal_data in signals.items():
                if apply_bandpass:
                    signal_data = ext.bandpass_filter(signal_data, Video.FPS)
                build_table_2(signal_data, time_array, crop_mode, label, subject, filename, file_CSV)

def build_table(signal_data, time_array, label, crop_mode, filename, file_CSV):

    pp_clean, hr_inst_clean, hr_inst_raw, t_mid_pp, _, _ = prv.compute_prv_hr(time_array, signal_data)    
    # Load single-column numeric data
    bvp_path = os.path.join(fileDataset1.folder_path, file_CSV)
    bvp = np.loadtxt(bvp_path)  # ensure it’s a single numeric column
    # Sampling rate (Hz)
    fs = float(BVP.BVP_RATE)
    n = len(bvp)  # number of samples in your signal
    t = np.arange(n) / fs
    pp_clean, ground_truth_hr, hr_inst_raw, gt_times, _, _ = prv.compute_prv_hr(t, bvp)
    result = filename.split("_")[1]
    plotter.build_table(rPPG=hr_inst_clean, rPPG_time=t_mid_pp, ground_truth=ground_truth_hr, gt_time=gt_times, subject = result, recording_id= filename, signal_label=label, cropMode=crop_mode) 
    return

def build_table_2(signal_data, time_array, crop_mode, label, subject, filename, file_CSV):
    pp_clean, hr_inst_clean, hr_inst_raw, t_mid_pp, _, _ = prv.compute_prv_hr(time_array, signal_data)
    file_id = Path(filename).stem   
    df = pd.read_csv(file_CSV)
    time_gt, hr_gt = ecg.ecg_hr_from_signal_500hz(df["ECG"].values)
    plotter.build_table_ECG(rPPG=hr_inst_clean, ground_truth=hr_gt, t_rPPG=t_mid_pp, t_ref=time_gt, recording_id=file_id, subject=subject,  signal_label=label, cropMode=crop_mode) 
    return

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
    choices=['manual', 'none', 'face_track', 'bbox_forehead', 'mesh_forehead'],
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

    # Only parse args if they exist (i.e., from command line)
    if len(sys.argv) > 1:
        args = parser.parse_args()
    else:
        # Defaults for IDE or test environment
        args = parser.parse_args(args=[])

        # Optional: manually override for testing here
        args.channels = ['ALL']
        args.crop_mode = 'manual', 'none', 'face_track', 'bbox_forehead', 'mesh_forehead'
      #
       # args.crop_mode = 'face_track' , 'mesh_forehead'
        #args.face_tracking = False

    runDataset1(channels=args.channels, crop_modes=args.crop_mode)
   # runDataset2(channels=args.channels, crop_mode=args.crop_mode)


if __name__ == "__main__":
    main()