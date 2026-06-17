import cv2
import csv
import numpy as np
import os
import archive.not_working.skin_mask as mask
import archive.not_working.prv_0 as prv
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import urllib.request
from src.config import Video, fileDataset1, fileDataset2, BVP
import src.extract_wave as ext
import matplotlib.pyplot as plt

def get_PRV(signal_data, time_array):
   # pp_clean, hr_inst_clean, hr_inst_raw, t_mid_pp, _, _ = prv.compute_prv_hr(time_array, signal_data) 

    bvp_path = os.path.join(fileDataset1.folder_path, Video.Csv_path)
    bvp = np.loadtxt(bvp_path)  # ensure it’s a single numeric column
    # Sampling rate (Hz)
    fs = float(BVP.BVP_RATE)
    n = len(bvp)  # number of samples in your signal
    t = np.arange(n) / fs
  #  pp_clean, ground_truth_hr, hr_inst_raw, gt_times, _, _ = prv.compute_prv_hr(t, bvp)

    pp_clean_a, hr_inst_clean_a, hr_inst_raw_a, t_mid_pp_a, peaks_t_a, artifacts_a = prv.compute_prv_hr(time_array, signal_data)
    # B) From the second compute_prv_hr
    pp_clean_b, ground_truth_hr_b, hr_inst_raw_b, t_mid_pp_b, peaks_t_b, artifacts_b = prv.compute_prv_hr(t, bvp)


    run_comparison_hr(
    hr_inst_ref = ground_truth_hr_b,  # "reference" HR
    t_ref       = t_mid_pp_b,
    hr_inst_cmp = hr_inst_clean_a,    # "compared" HR
    t_cmp       = t_mid_pp_a,
    ref_label   = "ground_truth_hr (B)",
    cmp_label   = "hr_inst_clean (A)"
    )

    # PP comparison
    run_comparison_pp(
        pp_ref    = pp_clean_b,    # reference PP (e.g., from B)
        t_pp_ref  = t_mid_pp_b,
        pp_cmp    = pp_clean_a,    # compared PP (from A)
        t_pp_cmp  = t_mid_pp_a,
        ref_label = "pp_clean (B)",
        cmp_label = "pp_clean (A)"
    )

    return

def get_BVP(G):
    bvp_path = os.path.join(fileDataset1.folder_path, Video.Csv_path)
    bvp = np.loadtxt(bvp_path)  # ensure it’s a single numeric column
    # Sampling rate (Hz)
    fs = float(BVP.BVP_RATE)
    n = len(bvp)  # number of samples in your signal
    t = np.arange(n) / fs
    pp_clean, ground_truth_hr, hr_inst_raw, gt_times, _, _ = prv.compute_prv_hr(t, bvp)
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(t, bvp, color='purple')
    plt.title("Cleaned PPG Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    # Plot ground truth heart rate
    plt.subplot(3, 1, 2)
    plt.plot(gt_times, ground_truth_hr, 'r-', label="Ground Truth HR")
    plt.title("Ground Truth Heart Rate")
    plt.xlabel("Time (s)")
    plt.ylabel("HR (bpm)")
    plt.legend()

    # Plot instantaneous HR
    plt.subplot(3, 1, 3)
    plt.plot(gt_times, hr_inst_raw, 'b-', label="Instantaneous HR")
    plt.title("Instantaneous Heart Rate")
    plt.xlabel("Time (s)")
    plt.ylabel("HR (bpm)")
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_rgb_traces(R, G, B, t):
    plt.figure(figsize=(12, 4))
    plt.plot(t, R, label='R', alpha=0.7)
    plt.plot(t, G, label='G', alpha=0.7)
    plt.plot(t, B, label='B', alpha=0.7)
    plt.xlabel("Time (s)")
    plt.ylabel("Mean intensity")
    plt.title("Raw ROI color traces")
    plt.legend()
    plt.show()

def plot_detrended(R, G, B, t):
    plt.figure(figsize=(12, 4))
    plt.plot(t, R, label='R detrended', alpha=0.7)
    plt.plot(t, G, label='G detrended', alpha=0.7)
    plt.plot(t, B, label='B detrended', alpha=0.7)
    plt.xlabel("Time (s)")
    plt.ylabel("Normalized intensity")
    plt.title("Detrended signals")
    plt.legend()
    plt.show()

def plot_fft(sig, fs, label="signal"):
    n = len(sig)
    freqs = np.fft.rfftfreq(n, d=1/fs)
    fft = np.fft.rfft((sig - np.mean(sig)) * np.hanning(n))
    psd = np.abs(fft)**2
    
    plt.figure(figsize=(10, 4))
    plt.plot(freqs, psd)
    plt.xlim(0, 4)  # 0–4 Hz (0–240 bpm)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")
    plt.title(f"FFT power spectrum ({label})")
    plt.axvline(1.0, color="r", ls="--", alpha=0.5)  # 60 bpm
    plt.axvline(1.5, color="g", ls="--", alpha=0.5)  # 90 bpm
    plt.axvline(2.0, color="b", ls="--", alpha=0.5)  # 120 bpm
    plt.show()

def plot_with_ground_truth(sig, t, gt_hr_bpm):
    plt.figure(figsize=(12, 4))
    plt.plot(t, sig, label="rPPG (detrended)")
    plt.plot(t, gt_hr_bpm, label="Ground truth HR", alpha=0.7)
    plt.xlabel("Time (s)")
    plt.ylabel("BPM (or signal units)")
    plt.legend()
    plt.show()


import numpy as np
import math
from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt
import pandas as pd

# ---------- Utilities ----------

def interpolate_onto(t_src, v_src, t_tgt):
    """Shape-preserving interpolation of v_src(t_src) onto t_tgt. NaN outside support."""
    t_src = np.asarray(t_src, float)
    v_src = np.asarray(v_src, float)
    t_tgt = np.asarray(t_tgt, float)
    if t_src.size < 2:
        return np.full_like(t_tgt, np.nan, dtype=float)
    f = PchipInterpolator(t_src, v_src, extrapolate=False)
    return f(t_tgt)

def finite_mask(*arrs):
    m = np.ones_like(np.asarray(arrs[0], float), dtype=bool)
    for a in arrs:
        m &= np.isfinite(a)
    return m

def metrics(y_true, y_pred):
    """
    Compute overlap count, coverage, MAE, RMSE, bias, and Pearson r.
    y_true and y_pred must be same-length arrays sampled at the same times.
    """
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    m = finite_mask(y_true, y_pred)
    n = int(np.sum(m))
    if n == 0:
        return {
            "N (overlap)": 0,
            "Coverage (overlap / total true)": 0.0,
            "MAE": np.nan,
            "RMSE": np.nan,
            "Bias mean (pred-true)": np.nan,
            "Corr (Pearson r)": np.nan,
        }
    e = y_pred[m] - y_true[m]
    mae = float(np.mean(np.abs(e)))
    rmse = float(np.sqrt(np.mean(e**2)))
    bias = float(np.mean(e))
    # guard against constant series for Pearson r
    if np.allclose(y_true[m], y_true[m].mean()) or np.allclose(y_pred[m], y_pred[m].mean()):
        r = np.nan
    else:
        r = float(np.corrcoef(y_true[m], y_pred[m])[0,1])
    cov = float(n / max(1, y_true.size))
    return {
        "N (overlap)": n,
        "Coverage (overlap / total true)": cov,
        "MAE": mae,
        "RMSE": rmse,
        "Bias mean (pred-true)": bias,
        "Corr (Pearson r)": r,
    }

def _overlay_diff_scatter(t, y_ref, y_cmp, ylabel, title_prefix, ref_label="Reference", cmp_label="Compared"):
    """Three plots: overlay vs time, diff vs time, scatter with y=x."""
    m = finite_mask(t, y_ref, y_cmp)
    if np.sum(m) < 2:
        print("Not enough overlapping finite points to plot.")
        return

    # 1) Overlay plot
    plt.figure()
    plt.plot(t[m], y_ref[m], label=f"{ref_label}")
    plt.plot(t[m], y_cmp[m], label=f"{cmp_label}")
    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)
    plt.title(f"{title_prefix}: series over time")
    plt.legend()
    plt.show()

    # 2) Difference plot
    plt.figure()
    plt.plot(t[m], (y_cmp[m] - y_ref[m]))
    plt.axhline(0, linestyle="--", color="k")
    plt.xlabel("Time (s)")
    plt.ylabel(f"Δ {ylabel} = {cmp_label} − {ref_label}")
    plt.title(f"{title_prefix}: difference over time")
    plt.show()

    # 3) Scatter with y=x
    plt.figure()
    plt.scatter(y_ref[m], y_cmp[m], s=10)
    lo = float(np.nanmin([np.nanmin(y_ref[m]), np.nanmin(y_cmp[m])]))
    hi = float(np.nanmax([np.nanmax(y_ref[m]), np.nanmax(y_cmp[m])]))
    if math.isfinite(lo) and math.isfinite(hi):
        plt.plot([lo, hi], [lo, hi], color="k")
    plt.xlabel(f"{ref_label}")
    plt.ylabel(f"{cmp_label}")
    plt.title(f"{title_prefix}: agreement plot")
    plt.show()

def _print_metrics_table(results, units=None):
    df = pd.DataFrame([results])
    if units:
        df = df.rename(columns={
            "MAE": f"MAE ({units})",
            "RMSE": f"RMSE ({units})",
            "Bias mean (pred-true)": f"Bias mean (pred-true) ({units})"
        })
    print(df.to_string(index=False))

# ---------- Comparisons ----------

def run_comparison_hr(hr_inst_ref, t_ref, hr_inst_cmp, t_cmp,
                      ref_label="ground_truth_hr (interp to ref times)",
                      cmp_label="hr_inst_clean"):
    """
    Compare two HR series. hr_inst_ref at times t_ref is treated as the 'reference'
    AFTER interpolation of hr_inst_cmp onto t_ref (or the other way around—your choice).
    Here we interpolate 'hr_inst_ref' onto t_ref already (no-op) and instead
    interpolate 'hr_inst_cmp' onto t_ref for a same-time comparison.
    """
    t_ref = np.asarray(t_ref, float)
    hr_inst_ref = np.asarray(hr_inst_ref, float)
    t_cmp = np.asarray(t_cmp, float)
    hr_inst_cmp = np.asarray(hr_inst_cmp, float)

    # Interpolate compared series onto reference times
    cmp_on_ref = interpolate_onto(t_cmp, hr_inst_cmp, t_ref)

    # Metrics: y_true = reference, y_pred = compared-on-ref
    res = metrics(y_true=hr_inst_ref, y_pred=cmp_on_ref)
    _print_metrics_table(res, units="BPM")

    _overlay_diff_scatter(
        t=t_ref,
        y_ref=hr_inst_ref,
        y_cmp=cmp_on_ref,
        ylabel="HR (BPM)",
        title_prefix="HR comparison",
        ref_label=ref_label,
        cmp_label=cmp_label,
    )

def run_comparison_pp(pp_ref, t_pp_ref, pp_cmp, t_pp_cmp,
                      ref_label="pp_clean (reference)",
                      cmp_label="pp_clean (compared)"):
    """
    Compare two PP-interval series using their mid-times.
    We interpolate the compared PP onto the reference PP mid-times and compute metrics.
    Units are seconds.
    """
    t_pp_ref = np.asarray(t_pp_ref, float)
    pp_ref   = np.asarray(pp_ref, float)
    t_pp_cmp = np.asarray(t_pp_cmp, float)
    pp_cmp   = np.asarray(pp_cmp, float)

    cmp_on_ref = interpolate_onto(t_pp_cmp, pp_cmp, t_pp_ref)

    res = metrics(y_true=pp_ref, y_pred=cmp_on_ref)
    _print_metrics_table(res, units="s")

    _overlay_diff_scatter(
        t=t_pp_ref,
        y_ref=pp_ref,
        y_cmp=cmp_on_ref,
        ylabel="PP interval (s)",
        title_prefix="PP comparison",
        ref_label=ref_label,
        cmp_label=cmp_label,
    )
