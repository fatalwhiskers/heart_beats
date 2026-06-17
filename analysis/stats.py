import numpy as np
from src.config import rppg
from scipy import stats
import csv, os
from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Dict, Any, Optional
import pandas as pd

def format_p(p):
    """Format p-values for paper-style reporting."""
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"

# --------- metrics ----------
def mae_rmse(rPPG: np.ndarray, ground_truth: np.ndarray):
    d = np.asarray(rPPG, float) - np.asarray(ground_truth, float)
    mae = float(np.mean(np.abs(d)))
    rmse = float(np.sqrt(np.mean(d**2)))
    return mae, rmse

def pearson_corr(rPPG: np.ndarray, ground_truth: np.ndarray):
    r, p = stats.pearsonr(np.asarray(rPPG, float), np.asarray(ground_truth, float))
    return float(r), float(p)

def bland_altman(rPPG: np.ndarray, ground_truth: np.ndarray, alpha: float = 0.05, filename_prefix: str = "bland_altman",
    outdir: str = "outputs/plots"):
    """
    Differences are rPPG - ground_truth.
    Returns bias, sd, LoA (±z*sd) and their (1-α) CIs.
    """
    rPPG = np.asarray(rPPG, float).ravel()
    gt   = np.asarray(ground_truth, float).ravel()

    # drop NaNs pairwise
    m = np.isfinite(rPPG) & np.isfinite(gt)
    x = 0.5 * (rPPG[m] + gt[m])
    d = rPPG[m] - gt[m]
    n = d.size
    if n < 3:
        raise ValueError("Not enough paired samples for Bland–Altman.")

    bias = float(np.mean(d))
    sd   = float(np.std(d, ddof=1))

    z = stats.norm.ppf(1 - alpha/2.0)   # e.g., 1.96 for alpha=0.05
    loa_lower = bias - z * sd
    loa_upper = bias + z * sd

    # t critical for CIs on bias and LoA estimates
    tcrit = stats.t.ppf(1 - alpha/2.0, df=n - 1)

    # SEs
    se_bias = sd / np.sqrt(n)
    se_loa  = sd * np.sqrt( (1.0/n) + (z**2)/(2.0*(n - 1)) )
    
    #plt.show()
    #plt.close(fig)

    return dict(
        bias=bias, sd=sd,
        loa_lower=float(loa_lower), loa_upper=float(loa_upper),
        bias_ci_low=float(bias - tcrit * se_bias),
        bias_ci_high=float(bias + tcrit * se_bias),
        loa_lower_ci_low=float(loa_lower - tcrit * se_loa),
        loa_lower_ci_high=float(loa_lower + tcrit * se_loa),
        loa_upper_ci_low=float(loa_upper - tcrit * se_loa),
        loa_upper_ci_high=float(loa_upper + tcrit * se_loa),
    )
# --------- optional: align ground truth onto rPPG timeline ----------
def align_ground_truth_to_rPPG(t_rPPG, rPPG, t_ref, ground_truth):
    """
    Interpolate ground_truth onto t_rPPG and return aligned arrays.
    Points in t_rPPG outside [t_ref[0], t_ref[-1]] are dropped.
    """
    t_rPPG = np.asarray(t_rPPG, float).ravel()
    rPPG   = np.asarray(rPPG, float).ravel()
    t_ref  = np.asarray(t_ref, float).ravel()
    ground_truth = np.asarray(ground_truth, float).ravel()

    if np.any(np.diff(t_ref) < 0):
        idx = np.argsort(t_ref); t_ref, ground_truth = t_ref[idx], ground_truth[idx]
    if np.any(np.diff(t_rPPG) < 0):
        idx = np.argsort(t_rPPG); t_rPPG, rPPG = t_rPPG[idx], rPPG[idx]

    m = (t_rPPG >= t_ref[0]) & (t_rPPG <= t_ref[-1])
    t_in = t_rPPG[m]
    rPPG_in = rPPG[m]
    gt_on_est = np.interp(t_in, t_ref, ground_truth)
    return t_in, rPPG_in, gt_on_est

# --------- one-stop evaluator (now with means/medians) ----------
def evaluate_hr_metrics(rPPG, ground_truth, t_rPPG=None, t_ref=None):
    """
    If both timelines are provided, ground_truth is interpolated onto t_rPPG.
    Returns MAE, RMSE, Pearson r/p, Bland–Altman, plus means/medians.
    """
    setup_apa7_matplotlib()
    ensure_dir("outputs/plots")

    if t_rPPG is not None and t_ref is not None:
        _, rPPG_use, gt_use = align_ground_truth_to_rPPG(t_rPPG, rPPG, t_ref, ground_truth)
    else:
        rPPG_use = np.asarray(rPPG, float).ravel()
        gt_use   = np.asarray(ground_truth, float).ravel()

    # Errors (rPPG - ground_truth)
    err = rPPG_use - gt_use
    abs_err = np.abs(err)

    mae, rmse = mae_rmse(rPPG_use, gt_use)
    r, p = pearson_corr(rPPG_use, gt_use)
    ba = bland_altman(rPPG_use, gt_use)

    return dict(
        n=int(rPPG_use.size),

        # core metrics
        mae=round(float(mae), 2),
        rmse=round(float(rmse), 2),
        pearson_r=round(float(r), 3),
        pearson_p=format_p(float(p)),
        bland_altman={k: round(float(v), 2) for k, v in ba.items()},  # bias, etc.

        # descriptive stats of series
        mean_rPPG=round(float(np.mean(rPPG_use)), 2),
        median_rPPG=round(float(np.median(rPPG_use)), 2),
        mean_ground_truth=round(float(np.mean(gt_use)), 2),
        median_ground_truth=round(float(np.median(gt_use)), 2),

        # error summaries
        mean_error=round(float(np.mean(err)), 2),       # == BA bias
        median_error=round(float(np.median(err)), 2),
        median_absolute_error=round(float(np.median(abs_err)), 2),

        # return aligned series used (full precision for downstream analysis)
        rPPG=rPPG_use,
        ground_truth=gt_use,
)

def append_rows_to_csv(csv_path, header, rows):
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def metrics_to_row(metrics, subject_id, recording_id, extraction_method, roi):
    ba = metrics["bland_altman"]
    return {
        "Subject ID": subject_id,
        "Recording ID": recording_id,
        "ROI": roi,
        "Extraction Method": extraction_method,
        "Number of Windows": int(metrics["n"]),

        "MAE (bpm)": float(metrics["mae"]),
        "RMSE (bpm)": float(metrics["rmse"]),
        "Pearson r": float(metrics["pearson_r"]),
        "Pearson p": metrics["pearson_p"],
        "Bias (bpm)": float(metrics["mean_error"]),  # aka mean error
        "SD (bpm)": float(ba["sd"]),
        "LoA Lower (bpm)": float(ba["loa_lower"]),
        "LoA Upper (bpm)": float(ba["loa_upper"]),

        "Mean rPPG (bpm)": float(metrics["mean_rPPG"]),
        "Median rPPG (bpm)": float(metrics["median_rPPG"]),
        "Mean Ground Truth (bpm)": float(metrics["mean_ground_truth"]),
        "Median Ground Truth (bpm)": float(metrics["median_ground_truth"]),
        "Mean Error (bpm)": float(metrics["mean_error"]),
        "Median Error (bpm)": float(metrics["median_error"]),
        "Median Absolute Error (bpm)": float(metrics["median_absolute_error"]),
    }

def average_hr_signal(t_sig, hr_sig):

    win_len = float(rppg.window_size)
    step    = float(rppg.step_size)

    # interpolation (NaN outside support)
    f = PchipInterpolator(t_sig, hr_sig, extrapolate=False)

    # timeline centers
    t0, t1 = t_sig[0], t_sig[-1]
    t_centers = np.arange(t0 + win_len/2, t1 - win_len/2 + 1e-9, step)

    hr_avg = []
    for c in t_centers:
        mask = (t_sig >= c - win_len/2) & (t_sig < c + win_len/2)
        if mask.any():
            hr_avg.append(np.nanmean(f(t_sig[mask])))
        else:
            hr_avg.append(np.nan)

    return t_centers, np.array(hr_avg)

def save_figure(fig: mpl.figure.Figure, outdir: str, filename_base: str, also_pdf: bool = True) -> Dict[str, str]:
    """
    Save a figure to PNG (and optionally PDF) with APA-7 friendly settings.
    Returns dict of saved paths.
    """
    ensure_dir(outdir)
    saved = {}
    png_path = os.path.join(outdir, f"{filename_base}.png")
    fig.savefig(png_path, bbox_inches="tight", facecolor="white")
    saved["png"] = png_path
    if also_pdf:
        pdf_path = os.path.join(outdir, f"{filename_base}.pdf")
        fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
        saved["pdf"] = pdf_path
    return saved

def setup_apa7_matplotlib(font_family: str = "Times New Roman", base_fontsize: int = 12, dpi: int = 300) -> None:
    """
    Configure matplotlib for APA 7–style figures:
    - Times New Roman 12pt if available (fallbacks listed)
    - High DPI, tight layout, readable ticks/labels
    """
    tnr_candidates = [
        font_family,
        "Times New Roman",
        "Nimbus Roman",
        "Nimbus Roman No9 L",
        "Liberation Serif",
        "DejaVu Serif",
    ]

    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": tnr_candidates,
        "figure.dpi": dpi,
        "savefig.dpi": dpi,
        "font.size": base_fontsize,
        "axes.titlesize": base_fontsize,
        "axes.labelsize": base_fontsize,
        "xtick.labelsize": int(base_fontsize * 0.9),
        "ytick.labelsize": int(base_fontsize * 0.9),
        "legend.fontsize": int(base_fontsize * 0.9),
        "axes.grid": True,
        "grid.alpha": 0.25,
        "figure.autolayout": True,
    })


def ensure_dir(path: str = "outputs/plots") -> str:
    os.makedirs(path, exist_ok=True)
    return path

def export_hr_to_csv_and_plot(
    rPPG,
    rPPG_time,
    recording_id: str,
    signal_label: str,
    cropMode: str,
    out_dir: str = "."
):
    """
    - CSV filename: {recording_id}_{signal_label}_{cropMode}.csv
    - Columns in CSV: time, hr
    - Plot filename: {recording_id}_{signal_label}_{cropMode}.png (APA 7 style)
    """
    # --- prep data ---
    if len(rPPG) != len(rPPG_time):
        raise ValueError("rPPG and rPPG_time must be the same length.")

    df = pd.DataFrame({
        "time": rPPG_time,  # already in seconds
        "hr": pd.to_numeric(rPPG, errors="coerce")
    }).dropna(subset=["hr", "time"])

    base = f"{recording_id}_{signal_label}_{cropMode}"
    csv_path = os.path.join(out_dir, base + ".csv")
    png_path = os.path.join(out_dir, base + ".png")

    # --- save CSV ---
    df.to_csv(csv_path, index=False)

    # --- make plot ---
    fig = plt.figure(figsize=(6, 4))  # APA-friendly aspect
    ax = fig.add_subplot(111)

    ax.plot(df["time"], df["hr"], linewidth=1.5)

    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Heart rate (bpm)", fontsize=12)
    ax.set_title(f"{signal_label} ({recording_id}, {cropMode})", fontsize=12)

    ax.grid(False)
    ax.tick_params(axis="both", which="both", direction="in", length=4, labelsize=10)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return csv_path, png_path