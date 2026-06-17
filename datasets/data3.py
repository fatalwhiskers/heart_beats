import os
import re
import argparse
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import src.rppg as rPPG
import src.hilbert_prv as hilly
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rppg.pipeline import VideoRGBExtractor
from src.config import Video, fileDataset1, fileDataset2, fileDataset3, BVP, rppg, Signal, PRV
from contextlib import contextmanager
import matplotlib as mpl
import src.extract_wave as ext
import matplotlib.cm as cm
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.cm as cm
import numpy as np
import glob

@contextmanager
def apa7_style(figsize=(6.5, 4.5), dpi=300):
    old = mpl.rcParams.copy()
    try:
        mpl.rcParams.update({
            'figure.figsize': figsize,
            'figure.dpi': dpi,
            'savefig.dpi': dpi,
            'savefig.bbox': 'tight',
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
            'font.size': 12,
            'axes.titlesize': 12,
            'axes.labelsize': 12,
            'xtick.labelsize': 11,
            'ytick.labelsize': 11,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.grid': False,
            'grid.color': '0.85',
            'grid.linestyle': '--',
            'lines.linewidth': 1.6,
            'legend.frameon': False,
        })
        yield
    finally:
        mpl.rcParams.update(old)

def make_windows(total_seconds: float, win_sec: float, step_sec: float) -> List[Tuple[float, float]]:
    if total_seconds <= 0 or win_sec <= 0 or step_sec <= 0:
        return []
    starts = np.arange(0, max(0.0, total_seconds - win_sec) + 1e-9, step_sec)
    return [(float(s), float(min(s + win_sec, total_seconds))) for s in starts]

def window_centers(windows: List[Tuple[float, float]]) -> np.ndarray:
    return np.array([(a + b) / 2.0 for a, b in windows], dtype=float)

def get_window_hr(t: np.ndarray, hr_inst: np.ndarray, windows: List[Tuple[float, float]]) -> Tuple[np.ndarray, np.ndarray]:
    t = np.asarray(t, float)
    hr = np.asarray(hr_inst, float)
    centers = window_centers(windows)
    out = np.full(len(windows), np.nan, dtype=float)
    for i, (a, b) in enumerate(windows):
        m = (t >= a) & (t < b)
        if m.any():
            out[i] = np.nanmean(hr[m])
    return centers, out

def pca_from_rgb(r: np.ndarray, g: np.ndarray, b: np.ndarray, ncomp: int = 3) -> Dict[str, np.ndarray]:
    signals = {}
    pca_components = ext.extract_pca_components(r, g, b)
    for i in range(min(3, pca_components.shape[1])):
        signals[f'PCA_{i+1}'] = pca_components[:, i]
    return signals

def extract_rgb_poly(video_path: str, crop_mode: str = "poly", display: bool = False, testing: bool = False
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    video_path = os.path.join(fileDataset3.folder_path, video_path)
    extractor = VideoRGBExtractor(crop_mode=crop_mode, display=display, testing=testing)
    R, G, B, t = extractor.extract(video_path)
    return np.asarray(R), np.asarray(G), np.asarray(B), np.asarray(t)

STRESS_HINTS = ("stress", "stroop", "math", "speaking", "counting", "talk", "arousal")
NEUTRAL_HINTS = ("baseline", "relax", "video", "reading", "neutral", "rest")

def infer_label_from_name(name: str) -> str:
    s = name.lower()
    if any(h in s for h in STRESS_HINTS):
        return "stress"
    if any(h in s for h in NEUTRAL_HINTS):
        return "neutral"
    return "unknown"

def process_video_to_csv_and_plot(
    video_path: str,
    out_dir: str,
    label: Optional[str] = None,
    crop_mode: str = "poly",
    win_sec: float = 20.0,
    step_sec: float = 3.0,
    plot_component: str = "PCA_1",
    include_gt_col: Optional[str] = None
) -> Tuple[str, str]:
    video_path = str(video_path)
    vid_id = Path(video_path).stem
    if label is None:
        label = infer_label_from_name(vid_id)
    R, G, B, t = extract_rgb_poly(video_path, crop_mode=crop_mode)
    t = np.asarray(t, float)
    total_dur = float(np.nanmax(t)) if t.size else 0.0
    if total_dur <= 0:
        raise ValueError(f"No frames/time found for {video_path}")
    pcs = pca_from_rgb(R, G, B, ncomp=3)
    if plot_component not in pcs:
        raise ValueError(f"{plot_component} not available; got keys {list(pcs.keys())}")
    windows = make_windows(total_dur, win_sec, step_sec)
    t_win = window_centers(windows)
    t_fft, hr_fft = rPPG.estimate_hr_fft_nt(t, pcs[plot_component])
    if len(t_fft) >= 2:
        hr_fft_on_base = np.interp(t_win, np.asarray(t_fft, float), np.asarray(hr_fft, float), left=np.nan, right=np.nan)
    else:
        hr_fft_on_base = np.full_like(t_win, np.nan, dtype=float)
    _, hr_inst_clean, _, t_mid_pp, _, _ = hilly.estimate_prv_hilbert_simple(t, pcs[plot_component])
    t_hil, hr_hil = get_window_hr(np.asarray(t_mid_pp, float), np.asarray(hr_inst_clean, float), windows)
    df = pd.DataFrame({
        "video_id": vid_id,
        "label": label,
        "time_s": t_win,
        f"fft_{plot_component}_poly": hr_fft_on_base,
        f"hilbert_{plot_component}_poly": hr_hil
    })
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{vid_id}_timeseries_pca-poly.csv"
    df.to_csv(csv_path, index=False)
    fig_path = out_dir / f"{vid_id}_timeseries_pca-poly.png"
    plt.figure(figsize=(9, 4.2))
    if label == "baseline":
        plt.axhspan(0, 180, xmin=0, xmax=1, alpha=0.07)
    elif label == "fear":
        plt.axhspan(0, 180, xmin=0, xmax=1, alpha=0.05)
    plt.plot(df["time_s"], df[f"fft_{plot_component}_poly"], linewidth=2.0, label="FFT")
    plt.plot(df["time_s"], df[f"hilbert_{plot_component}_poly"], linewidth=2.0, linestyle=":", label="Hilbert")
    if include_gt_col and include_gt_col in df.columns:
        plt.plot(df["time_s"], df[include_gt_col], linewidth=2.2, color="k", label="Ground truth")
    plt.xlabel("Time (s)")
    plt.ylabel("Heart rate (bpm)")
    plt.title(f"{vid_id} — rPPG HR (ROI=poly, PCA component={plot_component})")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[Caption hint] ROI=poly; Methods: FFT (solid), Hilbert (dotted); Component={plot_component}; Label={label}")
    return str(csv_path), str(fig_path)

def plot_baseline_vs_fear(manifest_csv, ts_dir, component="PCA_2"):
    man = pd.read_csv(manifest_csv)
    with apa7_style():
        fig, ax = plt.subplots()
        cmap = cm.get_cmap("tab10", len(man))
        baseline_curves, fear_curves = [], []
        for i, row in man.iterrows():
            vid = Path(row["video_path"]).stem
            label = str(row["labelfile_CSV"]).lower()
            csv_path = Path(ts_dir) / f"{vid}_timeseries_pca-poly.csv"
            if not csv_path.exists():
                print(f"⚠️ Missing: {csv_path}")
                continue
            df = pd.read_csv(csv_path)
            colname = f"hilbert_{component}_poly"
            if colname not in df.columns:
                print(f"⚠️ No column {colname} in {csv_path}")
                continue
            color = cmap(i % 10)
            linestyle = "-" if label == "baseline" else "--"
            ax.plot(df["time_s"], df[colname], color=color, linestyle=linestyle, linewidth=1.6, alpha=1.0)
            if label == "baseline":
                baseline_curves.append(df.set_index("time_s")[colname])
            elif label == "fear":
                fear_curves.append(df.set_index("time_s")[colname])
        if baseline_curves:
            baseline_avg = pd.concat(baseline_curves, axis=1).mean(axis=1)
            ax.plot(baseline_avg.index, baseline_avg.values, color="black", linestyle="-", linewidth=1.6, alpha=1.0)
        if fear_curves:
            fear_avg = pd.concat(fear_curves, axis=1).mean(axis=1)
            ax.plot(fear_avg.index, fear_avg.values, color="black", linestyle="--", linewidth=1.6, alpha=1.0)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Heart rate (bpm)")
        ax.set_ylim(60, 120)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)
        fig.tight_layout()
        plt.show()

def summarize_dataset3_timeseries(data_dir="ds3_timeseries"):
    csv_files = glob.glob(os.path.join(data_dir, "*_timeseries_pca-poly.csv"))
    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        dfs.append(df)
    all_df = pd.concat(dfs, ignore_index=True)
    hilbert_cols = [c for c in all_df.columns if "hilbert" in c.lower()]
    hilbert_col = hilbert_cols[0]
    stats = (
        all_df.groupby("label")[hilbert_col]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    print("=== Dataset 3 Summary ===")
    print(stats)
    if "baseline" in stats["label"].values and "fear" in stats["label"].values:
        baseline_mean = stats.loc[stats["label"] == "baseline", "mean"].values[0]
        fear_mean = stats.loc[stats["label"] == "fear", "mean"].values[0]
        delta = fear_mean - baseline_mean
        print(f"Mean baseline HR = {baseline_mean:.2f} bpm")
        print(f"Mean fear HR     = {fear_mean:.2f} bpm")
        print(f"Difference (fear – baseline) = {delta:.2f} bpm")

summarize_dataset3_timeseries("ds3_timeseries")

def run_manifest(
    manifest_csv: str,
    out_dir: str,
    crop_mode: str = "poly",
    win_sec: float = 20.0,
    step_sec: float = 3.0,
    plot_component: str = "pca_1"
):
    man = pd.read_csv(manifest_csv)
    if "video_path" not in man.columns:
        raise ValueError("Manifest must have a 'video_path' column.")
    for i, row in man.iterrows():
        video_path = str(row["video_path"])
        label = None
        for key in ("label", "Label", "state"):
            if key in man.columns:
                label = str(row[key]) if not pd.isna(row[key]) else None
                break
        print(f"\nProcessing [{i+1}/{len(man)}]: {video_path}")
        csv_path, fig_path = process_video_to_csv_and_plot(
            video_path=video_path,
            out_dir=out_dir,
            label=label,
            crop_mode=crop_mode,
            win_sec=win_sec,
            step_sec=step_sec,
            plot_component=plot_component
        )
        print(f"  ➜ saved CSV:  {csv_path}")
        print(f"  ➜ saved plot: {fig_path}")

def main():
    p = argparse.ArgumentParser(description="Dataset 3: rPPG time-series via PCA+poly (FFT & Hilbert).")
    p.add_argument("--manifest", default="data/CSVFiles/dataset3.csv", help="CSV with column 'video_path' and optional 'label'.")
    p.add_argument("--out_dir", default="ds3_timeseries", help="Output directory for CSVs/plots.")
    p.add_argument("--crop_mode", default="poly", help="ROI/cropping mode (default: poly).")
    p.add_argument("--win_sec", type=float, default=30.0, help="Window length in seconds (default 20).")
    p.add_argument("--step_sec", type=float, default=2.0, help="Window step in seconds (default 3).")
    p.add_argument("--component", default="PCA_2", choices=["pca_1", "pca_2", "pca_3"], help="Which PCA component to plot.")
    args = p.parse_args()
    Video.FPS = 60
    Video.target_FPS = 60
    plot_baseline_vs_fear("data/CSVFiles/dataset3.csv", "ds3_timeseries", component="PCA_2")
    summarize_dataset3_timeseries("ds3_timeseries")

if __name__ == "__main__":
    main()
