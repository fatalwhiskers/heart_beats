import src.video_reader as vr
import src.Video_extraction as VE
import src.BVP as BVP
import src.plotter as plotter
import test as test
import numpy as np
import src.extract_wave as ext
import argparse
import sys
import os
from src.config import Video, filePaths


def runLoad(channels=['G'], cropping = True, crop_mode = "manual", interpolate = True, apply_bandpass = True,  Display = False, Testing = False):
   

    crop_list = VE.load_crop_settings(filePaths.csv_path)
    
    for filename, file_CSV, x1, y1, x2, y2 in crop_list:

        video_path = os.path.join(filePaths.folder_path, filename)
        R_signal, G_signal, B_signal, time_array = VE.extract_video_to_rgb(video_path, x1, y1, x2, y2, crop_mode, Display, Testing)

        
        if interpolate:
            R_signal , t_uniform = ext.interpolate_signal_with_timestamps(R_signal, time_array) 
            B_signal , t_uniform = ext.interpolate_signal_with_timestamps(B_signal, time_array)
            G_signal , t_uniform = ext.interpolate_signal_with_timestamps(G_signal, time_array)
    
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
        
        if 'POS' in channels or 'ALL' in channels:
            y1 = G_signal - B_signal
            y2 = G_signal + B_signal - 2*R_signal
            a = np.std(y1) / (np.std(y2))
            pos_signal = y1 + a * y2
            signals['POS'] = pos_signal

        for label, signal_data in signals.items():
            if apply_bandpass:
                signal_data = ext.bandpass_filter(signal_data, Video.FPS)
            ground_truth_hr = BVP.get_bvp_ground_truth(file_CSV)    
            plotter.run(signal_data, label, ground_truth_hr)



# example usage python main.py --channels G R --face_tracking
def main():
    parser = argparse.ArgumentParser(description="Video signal processing CLI")
    parser.add_argument(
        '--channels',
        nargs='+',
        choices=['R', 'G', 'B', 'GREY_W', 'GREY_A', 'PCA', 'ZCA', 'ALL'],
        default=['G'],
        help='Color channels: R, G, B, GREY_W (weighted), GREY_A (average) , PCA, ZCA, All'
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
        #args.channels = ['G', 'PCA' , 'ZCA']
        args.crop_mode = 'face_track' 
        #args.face_tracking = False

    runLoad(channels=args.channels, crop_mode=args.crop_mode)



if __name__ == "__main__":
    main()