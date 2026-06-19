"""
demo_run.py
-----------
A self-contained demonstration of the rPPG pipeline.

Unlike main.py, this script needs NO dataset CSVs, NO ground-truth files,
and NO manual crop coordinates -- it works on any video of a face,
using an auto-detecting crop mode.

This exists specifically so someone reviewing this project (a supervisor,
an examiner, a recruiter) can verify the pipeline genuinely works, without
needing access to the original dissertation video dataset (which can
never be shared publicly, since it contains identifiable faces).

This script intentionally duplicates a few small helper functions that
also exist in main.py (get_Signals, make_windows, get_window_hr), rather
than importing them from main.py. That's deliberate: main.py imports
several archive/ and analysis/ modules at the top of the file purely for
its own dataset-driven workflows, none of which this demo needs. Keeping
this script's import list short and independent means it can't break for
reasons unrelated to what it's actually demonstrating.

Usage
-----
    python demo/demo_run.py path/to/your_video.mp4
    python demo/demo_run.py path/to/your_video.mp4 --crop_mode mesh_forehead --channel G --hr_method welch

If --crop_mode, --channel, or --hr_method are omitted you'll be asked
interactively.

Output
------
Prints summary statistics to the console, and saves one PNG plot per
chosen HR method to demo/output/ (e.g. heart_rate_welch.png,
heart_rate_hilbert.png).
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")  # safe for headless/Codespaces environments -- no display needed
import matplotlib.pyplot as plt

# Make sure the project root is importable regardless of where this script is run from
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rppg.pipeline import VideoRGBExtractor
import src.rppg as rPPG
import src.hilbert_prv as hilly
import src.extract_wave as ext
from src.config import rppg as rppg_cfg

# Both estimate_hr_welch_nk (src/rppg.py) and the Hilbert windowing below
# read rppg_cfg.window_size / step_size directly. The dissertation's
# original 30s/3s windows assume long recordings; short demo clips
# (20-30s) would silently produce zero or one HR estimate with that
# setting. This override applies only when demo_run.py is executed
# directly (see the __main__ guard at the bottom) -- it deliberately
# mutates the shared config object, which is safe here only because this
# script is always run standalone, never imported alongside other code
# that might also rely on the original 30s/3s values.
rppg_cfg.window_size = 10
rppg_cfg.step_size = 1


# ---------------------------------------------------------------------------
# Crop mode options
# ---------------------------------------------------------------------------
CROP_MODE_CHOICES = ["face_track", "bbox_forehead", "mesh_forehead", "poly"]

CROP_MODE_DESCRIPTIONS = {
    "face_track":    "Tracks the whole face (largest region, most light, least specific)",
    "bbox_forehead": "Crops to the forehead using the face bounding box (fast, usually most stable)",
    "mesh_forehead": "Crops to the forehead using a 478-point face mesh (more precise than bbox)",
    "poly":          "Uses forehead + both cheeks via face mesh polygons (most skin coverage)",
}

# ---------------------------------------------------------------------------
# Channel category options (mirrors main.py's get_Signals categories)
# ---------------------------------------------------------------------------
CHANNEL_CHOICES = ["R", "G", "B", "GREY_W", "GREY_A", "PCA", "ZCA", "ICA", "CHROM", "POS"]

CHANNEL_DESCRIPTIONS = {
    "R":      "Red channel only",
    "G":      "Green channel only (standard rPPG default -- best pulse signal-to-noise)",
    "B":      "Blue channel only",
    "GREY_W": "Weighted greyscale (perceptual luminance)",
    "GREY_A": "Simple average of R, G, B",
    "PCA":    "Principal Component Analysis -- up to 3 components",
    "ZCA":    "ZCA whitening -- up to 3 components",
    "ICA":    "Independent Component Analysis -- up to 3 components (best one flagged)",
    "CHROM":  "Chrominance-based method (de Haan & Jeanne)",
    "POS":    "Plane-Orthogonal-to-Skin method (Wang et al.)",
}

# ---------------------------------------------------------------------------
# HR method options
# ---------------------------------------------------------------------------
HR_METHOD_CHOICES = ["welch", "hilbert"]

HR_METHOD_DESCRIPTIONS = {
    "welch":   "Welch periodogram via neurokit2 -- robust, windowed",
    "hilbert": "Hilbert-transform instantaneous HR, then windowed to match Welch's output shape",
}


# ---------------------------------------------------------------------------
# Duplicated small helpers (see module docstring for why these aren't
# imported from main.py)
# ---------------------------------------------------------------------------

def get_Signals(channel: str, R_signal, G_signal, B_signal) -> dict:
    """
    Return {label: signal_array} for the requested channel category.
    Mirrors main.py's get_Signals, restricted to a single category at a
    time (the demo asks the user to pick one category per run).
    """
    signals = {}

    if channel == 'R':
        signals['R'] = R_signal
    elif channel == 'G':
        signals['G'] = G_signal
    elif channel == 'B':
        signals['B'] = B_signal
    elif channel == 'GREY_W':
        signals['GREY_W'] = 0.2989 * R_signal + 0.5870 * G_signal + 0.1140 * B_signal
    elif channel == 'GREY_A':
        signals['GREY_A'] = (R_signal + G_signal + B_signal) / 3.0
    elif channel == 'PCA':
        pca_components = ext.extract_pca_components(R_signal, G_signal, B_signal)
        for i in range(min(3, pca_components.shape[1])):
            signals[f'PCA_{i+1}'] = pca_components[:, i]
    elif channel == 'ZCA':
        zca_components = ext.zca_whiten(R_signal, G_signal, B_signal)
        for i in range(min(3, zca_components.shape[1])):
            signals[f'ZCA_{i+1}'] = zca_components[:, i]
    elif channel == 'ICA':
        ICA_components, best_idx = ext.ICA_Test(R_signal, G_signal, B_signal)
        for i in range(min(3, ICA_components.shape[1])):
            if i == best_idx:
                signals[f'Best_ICA_{i+1}'] = ICA_components[:, i]
            else:
                signals[f'ICA_{i+1}'] = ICA_components[:, i]
    elif channel == 'CHROM':
        signals['CHROM'] = ext.chrom_pos_windowed(R_signal, G_signal, B_signal, method='CHROM')
    elif channel == 'POS':
        signals['POS'] = ext.chrom_pos_windowed(R_signal, G_signal, B_signal, method='POS')
    else:
        raise ValueError(f"Unknown channel category: {channel}")

    return signals


def make_windows(total_duration: int, window_size: int, step_size: int):
    """Build (start, end) second-pair windows covering total_duration."""
    starts = np.arange(0, total_duration - window_size + 1, step_size)
    return [(int(s), int(s) + window_size) for s in starts]


def get_window_hr(hr_time, hr_values, windows):
    """Average hr_values into each (start, end) window; NaN for empty windows."""
    hr_per_win = []
    times = []
    for (t0, t1) in windows:
        mask = (hr_time >= t0) & (hr_time < t1)
        if np.any(mask):
            hr_per_win.append(np.mean(hr_values[mask]))
        else:
            hr_per_win.append(np.nan)
        times.append((t0 + t1) / 2.0)
    return np.array(times), np.array(hr_per_win)


def estimate_hr_hilbert_windowed(t: np.ndarray, signal_data: np.ndarray):
    """
    Run the Hilbert-transform instantaneous HR estimator, then window its
    output the same way Welch's windowed output is shaped, so both methods
    can be plotted and compared consistently.

    Returns (window_centers_t, hr_estimates) -- both empty arrays if too
    few heartbeats were detected to estimate anything.
    """
    pp_clean, hr_inst_clean, hr_inst_raw, t_mid_pp, t_beats, artifact_mask = \
        hilly.estimate_prv_hilbert_simple(t, signal_data)

    if t_mid_pp.size < 2:
        # Hilbert found fewer than 2 heartbeats -- too little signal to window.
        return np.array([]), np.array([])

    total_duration = float(np.max(t_mid_pp))
    windows = make_windows(int(total_duration), rppg_cfg.window_size, rppg_cfg.step_size)
    return get_window_hr(t_mid_pp, hr_inst_clean, windows)


# ---------------------------------------------------------------------------
# Interactive prompts
# ---------------------------------------------------------------------------

def _prompt_single(choices: list, descriptions: dict, label: str, default_idx: int = 0) -> str:
    print(f"\nChoose {label}:")
    for i, choice in enumerate(choices, start=1):
        print(f"  {i}. {choice:10s} - {descriptions[choice]}")
    while True:
        raw = input(f"\nEnter a number (1-{len(choices)}) [default: {default_idx + 1}]: ").strip()
        if raw == "":
            return choices[default_idx]
        if raw.isdigit() and 1 <= int(raw) <= len(choices):
            return choices[int(raw) - 1]
        print("Not a valid choice, try again.")


def _prompt_multi(choices: list, descriptions: dict, label: str, default_indices: list) -> list:
    print(f"\nChoose {label} (comma-separated, e.g. '1,2'):")
    for i, choice in enumerate(choices, start=1):
        print(f"  {i}. {choice:10s} - {descriptions[choice]}")
    default_str = ",".join(str(i + 1) for i in default_indices)
    while True:
        raw = input(f"\nEnter number(s) [default: {default_str}]: ").strip()
        if raw == "":
            return [choices[i] for i in default_indices]
        try:
            picked_idx = [int(x.strip()) - 1 for x in raw.split(",")]
            if all(0 <= i < len(choices) for i in picked_idx):
                # de-duplicate while preserving order
                seen = []
                for i in picked_idx:
                    if choices[i] not in seen:
                        seen.append(choices[i])
                return seen
        except ValueError:
            pass
        print("Not a valid choice, try again.")


def prompt_for_crop_mode() -> str:
    return _prompt_single(CROP_MODE_CHOICES, CROP_MODE_DESCRIPTIONS, "a crop mode", default_idx=1)


def prompt_for_channel() -> str:
    return _prompt_single(CHANNEL_CHOICES, CHANNEL_DESCRIPTIONS, "a channel category", default_idx=1)


def prompt_for_hr_methods() -> list:
    return _prompt_multi(HR_METHOD_CHOICES, HR_METHOD_DESCRIPTIONS, "HR method(s)", default_indices=[0])


# ---------------------------------------------------------------------------
# Main demo logic
# ---------------------------------------------------------------------------

def run_demo(video_path: str, crop_mode: str, channel: str, hr_methods: list, output_dir: str) -> None:
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Could not find video file: {video_path}")

    os.makedirs(output_dir, exist_ok=True)

    print(f"\nProcessing '{video_path}' with crop_mode='{crop_mode}'...")
    print("This extracts a frame-by-frame skin colour signal from the video. "
          "It may take a little while depending on video length.\n")

    extractor = VideoRGBExtractor(crop_mode=crop_mode)
    R, G, B, t = extractor.extract(video_path)

    n_frames = len(t)
    duration_s = float(t[-1] - t[0]) if n_frames > 1 else 0.0

    print(f"Frames processed:        {n_frames}")
    print(f"Video duration analysed: {duration_s:.1f} s")

    if n_frames < 2:
        print("\nNot enough valid frames were extracted to estimate a heart rate.")
        print("This usually means no face was detected in the video -- "
              "try a different crop_mode, or a video with a clearer, "
              "well-lit, front-facing view of the face.")
        return

    signals = get_Signals(channel, R, G, B)
    print(f"\nChannel category '{channel}' produced {len(signals)} signal(s): "
          f"{', '.join(signals.keys())}")

    for hr_method in hr_methods:
        print(f"\n=== HR method: {hr_method} ===")
        fig, ax = plt.subplots(figsize=(9, 4.5))
        any_valid_line = False

        for label, signal_data in signals.items():
            if hr_method == "welch":
                window_centers_t, hr_estimates = rPPG.estimate_hr_welch_nk(t, signal_data)
            else:  # hilbert
                window_centers_t, hr_estimates = estimate_hr_hilbert_windowed(t, signal_data)

            valid_hr = hr_estimates[np.isfinite(hr_estimates)] if hr_estimates.size else hr_estimates

            if valid_hr.size == 0:
                print(f"  [{label}] No valid HR estimates "
                      f"(signal may be too short, noisy, or have too few detected beats).")
                continue

            print(f"  [{label}] mean HR: {np.mean(valid_hr):.1f} bpm  "
                  f"(range {np.min(valid_hr):.1f}-{np.max(valid_hr):.1f}, "
                  f"{len(valid_hr)}/{len(hr_estimates)} windows valid)")

            ax.plot(window_centers_t, hr_estimates, marker="o", linewidth=1.6, label=label)
            any_valid_line = True

        if not any_valid_line:
            print(f"  No signal produced a usable HR estimate for method '{hr_method}'. "
                  f"Skipping plot.")
            plt.close(fig)
            continue

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Heart rate (bpm)")
        ax.set_title(f"Estimated heart rate -- channel='{channel}', method='{hr_method}', "
                     f"crop_mode='{crop_mode}'")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
        fig.tight_layout()

        plot_path = os.path.join(output_dir, f"heart_rate_{hr_method}.png")
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"  Saved plot to: {plot_path}")

    print("\nDone. Open the saved PNG(s) from the VS Code file explorer to view them.")


def main():
    parser = argparse.ArgumentParser(
        description="Run the rPPG pipeline on any video of a face -- no dataset CSV needed."
    )
    parser.add_argument("video_path", help="Path to a video file (.mp4, .avi, etc.)")
    parser.add_argument(
        "--crop_mode",
        choices=CROP_MODE_CHOICES,
        default=None,
        help="Which face-region crop mode to use. If omitted, you'll be asked interactively.",
    )
    parser.add_argument(
        "--channel",
        choices=CHANNEL_CHOICES,
        default=None,
        help="Which signal-channel category to use. If omitted, you'll be asked interactively.",
    )
    parser.add_argument(
        "--hr_method",
        choices=HR_METHOD_CHOICES,
        action="append",
        default=None,
        help="HR estimation method(s) to run. Repeat the flag to select more than one "
             "(e.g. --hr_method welch --hr_method hilbert). If omitted, you'll be asked interactively.",
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join("demo", "output"),
        help="Where to save the result plot(s) (default: demo/output)",
    )
    args = parser.parse_args()

    crop_mode = args.crop_mode or prompt_for_crop_mode()
    channel = args.channel or prompt_for_channel()
    hr_methods = args.hr_method or prompt_for_hr_methods()

    run_demo(args.video_path, crop_mode, channel, hr_methods, args.output_dir)


if __name__ == "__main__":
    main()
