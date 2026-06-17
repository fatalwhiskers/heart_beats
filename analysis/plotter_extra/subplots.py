import os
import re
import glob
from contextlib import contextmanager

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

INPUT_DIR = r"outputs\dset1_timeseries"
FILE_GLOB = "*_all_rois.csv"
ESTIMATOR = "hilbert"
OUTPUT_DIR = "figures"
SHOW_PLOTS = False
SHOW_LEGEND = False
LINE_WIDTH = 1.6

ROI_ORDER = ["manual", "none", "face_track", "bbox_forehead", "mesh_forehead", "poly"]

ROI_DISPLAY_NAMES = {
    "manual": "Manual",
    "none": "None",
    "face_track": "Face tracking",
    "bbox_forehead": "Forehead bbox",
    "mesh_forehead": "Polygonal tracking forehead",
    "poly": "Polygonal tracking forehead and cheeks",
}

BASE_METHODS = ["r", "g", "b", "grey_w", "grey_a", "chrom", "pos"]
COMPONENT_FAMILIES = ["pca", "ica", "zca"]

@contextmanager
def apa7_style(figsize=(11, 7), dpi=300):
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
            'lines.linewidth': LINE_WIDTH,
            'legend.frameon': False,
        })
        yield
    finally:
        mpl.rcParams.update(old)

COLOR_MAP = {
    "pca": "red",
    "ica": "blue",
    "zca": "green",
    "pos": "purple",
    "chrom": "orange",
    "r": "brown",
    "g": "darkgreen",
    "b": "navy",
    "grey_w": "dimgray",
    "grey_a": "#87CEFA",
}

def pick_color(name: str) -> str:
    key = name.lower()
    if key.startswith(("pca", "ica", "zca")):
        key = key.split("_")[0]
    return COLOR_MAP.get(key, "gray")

STYLE_MAP = {"pca": "--", "ica": ":", "zca": "-."}

def pick_style(name: str) -> str:
    key = name.lower()
    if key.startswith(("pca", "ica", "zca")):
        key = key.split("_")[0]
    return STYLE_MAP.get(key, "-")

def extract_source_id(path: str) -> str:
    base = os.path.basename(path)
    return re.sub(r"_all_rois\.csv$", "", base)

def load_timeseries(data_dir: str, pattern: str) -> pd.DataFrame:
    paths = glob.glob(os.path.join(data_dir, pattern))
    if not paths:
        raise FileNotFoundError(f"No CSVs found under {data_dir} with pattern {pattern}")
    frames = []
    for p in sorted(paths):
        df = pd.read_csv(p)
        df["source_id"] = extract_source_id(p)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

def component_columns(estimator: str, family: str, roi: str) -> list[str]:
    return [f"{estimator}_{family}_{k}_{roi}" for k in (1, 2, 3)]

def select_best_component_column(video_df: pd.DataFrame, estimator: str, family: str, roi: str) -> str | None:
    candidates = [c for c in component_columns(estimator, family, roi) if c in video_df.columns]
    if not candidates:
        return None
    if "gt_hr_bpm" in video_df.columns:
        maes = []
        gt = video_df["gt_hr_bpm"].to_numpy()
        for c in candidates:
            y = video_df[c].to_numpy()
            mask = np.isfinite(gt) & np.isfinite(y)
            mae = np.mean(np.abs(y[mask] - gt[mask])) if mask.any() else np.inf
            maes.append(mae)
        if np.isfinite(maes).any():
            return candidates[int(np.argmin(maes))]
    counts = [np.isfinite(video_df[c].to_numpy()).sum() for c in candidates]
    if any(counts):
        return candidates[int(np.argmax(counts))]
    return candidates[0]

def format_label(name: str) -> str | None:
    key = name.lower()
    if key.startswith("pca"): return "PCA"
    if key.startswith("ica"): return "ICA"
    if key.startswith("zca"): return "ZCA"
    if key == "pos": return "POS"
    if key == "chrom": return "CHROM"
    if key == "r": return "R channel"
    if key == "g": return "G channel"
    if key == "b": return "B channel"
    if key == "grey_w": return "Grey W"
    if key == "grey_a": return "Grey A"
    return None

def plot_single_video(video_df: pd.DataFrame, estimator_name: str = "fft", output_dir: str | None = None):
    video_df = video_df.sort_values("time_s")
    video_id = str(video_df["source_id"].iloc[0])
    n = len(ROI_ORDER)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    with apa7_style(figsize=(11, 7)):
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True)
        axes = axes.flatten()
        used_labels = set()
        for ax, roi in zip(axes, ROI_ORDER):
            t = video_df["time_s"].values
            for method in BASE_METHODS:
                col = f"{estimator_name}_{method}_{roi}"
                if col in video_df.columns:
                    color = pick_color(method)
                    style = pick_style(method)
                    label_text = format_label(method)
                    label_text = label_text if (label_text and label_text not in used_labels) else None
                    ax.plot(t, video_df[col].values, color=color, linestyle=style, label=label_text)
                    if label_text: used_labels.add(label_text)
            for family in COMPONENT_FAMILIES:
                best_col = select_best_component_column(video_df, estimator_name, family, roi)
                if best_col:
                    color = pick_color(family)
                    style = pick_style(family)
                    label_text = format_label(family)
                    label_text = label_text if (label_text and label_text not in used_labels) else None
                    ax.plot(t, video_df[best_col].values, color=color, linestyle=style, label=label_text)
                    if label_text: used_labels.add(label_text)
            if "gt_hr_bpm" in video_df.columns:
                gt_label = "Ground Truth HR"
                label_text = gt_label if gt_label not in used_labels else None
                ax.plot(t, video_df["gt_hr_bpm"].values, color="black", linestyle="--", label=label_text)
                if label_text: used_labels.add(gt_label)
            title = ROI_DISPLAY_NAMES.get(roi, roi.replace("_", " ").capitalize())
            ax.set_title(title)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("HR (bpm)")
        for j in range(len(ROI_ORDER), len(axes)):
            axes[j].set_visible(False)
        if SHOW_LEGEND:
            handles, labels = [], []
            for ax in axes:
                h, l = ax.get_legend_handles_labels()
                for hh, ll in zip(h, l):
                    if ll not in labels:
                        handles.append(hh); labels.append(ll)
            if handles:
                nlabels = len(labels)
                ncol = int(np.ceil(nlabels / 2))
                fig.legend(
                    handles, labels,
                    loc="upper center",
                    bbox_to_anchor=(0.5, 1.05),
                    ncol=ncol,
                    frameon=False, fontsize=10
                )
            fig.tight_layout(rect=[0, 0, 1, 0.9])
        else:
            fig.tight_layout(rect=[0, 0, 1, 1])
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            out_path = os.path.join(output_dir, f"{video_id}_{estimator_name}.png")
            fig.savefig(out_path)
            print(f"Saved: {out_path}")
        if SHOW_PLOTS:
            plt.show()
        else:
            plt.close(fig)

if __name__ == "__main__":
    all_ts = load_timeseries(INPUT_DIR, FILE_GLOB)
    REQUIRED_COLUMNS = {"video_id", "time_s", "gt_hr_bpm", "source_id"}
    plot_columns = set()
    for roi in ROI_ORDER:
        for method in BASE_METHODS:
            for est in ("fft", "welch", "hilbert"):
                plot_columns.add(f"{est}_{method}_{roi}")
        for family in COMPONENT_FAMILIES:
            for k in (1, 2, 3):
                for est in ("fft", "welch", "hilbert"):
                    plot_columns.add(f"{est}_{family}_{k}_{roi}")
    columns_to_keep = [c for c in all_ts.columns if (c in plot_columns) or (c in REQUIRED_COLUMNS)]
    all_ts = all_ts[columns_to_keep]
    for vid, video_df in all_ts.groupby("source_id", sort=True):
        plot_single_video(video_df, estimator_name=ESTIMATOR, output_dir=OUTPUT_DIR)
    print("Done.")
