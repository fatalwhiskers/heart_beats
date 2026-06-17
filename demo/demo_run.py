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

Usage
-----
    python demo_run.py path/to/your_video.mp4
    python demo_run.py path/to/your_video.mp4 --crop_mode mesh_forehead

If --crop_mode is omitted you'll be prompted to choose one interactively.

Output
------
Prints summary statistics (mean HR, signal length, frame count) to the
console, and saves a PNG plot of the estimated heart rate over time to
demo/output/heart_rate.png
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


CROP_MODE_CHOICES = ["face_track", "bbox_forehead", "mesh_forehead", "poly"]

CROP_MODE_DESCRIPTIONS = {
    "face_track":    "Tracks the whole face (largest region, most light, least specific)",
    "bbox_forehead": "Crops to the forehead using the face bounding box (fast, usually most stable)",
    "mesh_forehead": "Crops to the forehead using a 478-point face mesh (more precise than bbox)",
    "poly":          "Uses forehead + both cheeks via face mesh polygons (most skin coverage)",
}


def prompt_for_crop_mode() -> str:
    """Ask the user to pick a crop mode interactively."""
    print("\nChoose a crop mode:")
    for i, mode in enumerate(CROP_MODE_CHOICES, start=1):
        print(f"  {i}. {mode:15s} - {CROP_MODE_DESCRIPTIONS[mode]}")

    while True:
        choice = input(f"\nEnter a number (1-{len(CROP_MODE_CHOICES)}) [default: 2]: ").strip()
        if choice == "":
            return CROP_MODE_CHOICES[1]  # default to bbox_forehead
        if choice.isdigit() and 1 <= int(choice) <= len(CROP_MODE_CHOICES):
            return CROP_MODE_CHOICES[int(choice) - 1]
        print("Not a valid choice, try again.")


def run_demo(video_path: str, crop_mode: str, output_dir: str) -> None:
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

    if n_frames < 2:
        print("Not enough valid frames were extracted to estimate a heart rate.")
        print("This usually means no face was detected in the video -- "
              "try a different crop_mode, or a video with a clearer, "
              "well-lit, front-facing view of the face.")
        return

    # Use the Green channel -- the standard choice for rPPG, since haemoglobin
    # absorbs green light most strongly, giving the best pulse signal-to-noise.
    window_centers_t, hr_estimates = rPPG.estimate_hr_welch_nk(t, G)

    valid_hr = hr_estimates[np.isfinite(hr_estimates)]

    print("=== Results ===")
    print(f"Frames processed:        {n_frames}")
    print(f"Video duration analysed: {duration_s:.1f} s")
    print(f"HR windows estimated:    {len(hr_estimates)} "
          f"({len(valid_hr)} valid)")
    if valid_hr.size:
        print(f"Mean estimated HR:       {np.mean(valid_hr):.1f} bpm")
        print(f"HR range:                {np.min(valid_hr):.1f} - {np.max(valid_hr):.1f} bpm")
    else:
        print("No valid HR estimates were produced -- the signal may be too "
              "noisy or the video too short for a reliable window.")

    # --- plot ---
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(window_centers_t, hr_estimates, marker="o", linewidth=1.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Heart rate (bpm)")
    ax.set_title(f"Estimated heart rate over time (crop_mode='{crop_mode}')")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()

    plot_path = os.path.join(output_dir, "heart_rate.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    print(f"\nSaved plot to: {plot_path}")
    print("Open it from the VS Code file explorer to view it.")


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
        "--output_dir",
        default=os.path.join("demo", "output"),
        help="Where to save the result plot (default: demo/output)",
    )
    args = parser.parse_args()

    crop_mode = args.crop_mode or prompt_for_crop_mode()
    run_demo(args.video_path, crop_mode, args.output_dir)


if __name__ == "__main__":
    main()
