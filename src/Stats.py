import numpy as np
from scipy import stats
import csv, os

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

def bland_altman(rPPG: np.ndarray, ground_truth: np.ndarray, alpha: float = 0.05):
    """
    Differences are rPPG - ground_truth.
    Returns bias, sd, LoA (±z*sd) and their (1-α) CIs.
    """
    rPPG = np.asarray(rPPG, float).ravel()
    gt   = np.asarray(ground_truth, float).ravel()

    # drop NaNs pairwise
    m = np.isfinite(rPPG) & np.isfinite(gt)
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