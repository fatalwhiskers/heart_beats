from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Dict
import re

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

TECHNIQUES: Tuple[str, ...] = ("fft", "welch", "hilbert")
KNOWN_ROIS = {"manual", "none", "face_track", "bbox_forehead", "mesh_forehead", "poly"}

def parse_column_key(col: str) -> Optional[Tuple[str, str, str]]:
    for tech in TECHNIQUES:
        pref = f"{tech}_"
        if col.startswith(pref):
            for roi in sorted(KNOWN_ROIS, key=len, reverse=True):
                suf = f"_{roi}"
                if col.endswith(suf):
                    method = col[len(pref):-len(suf)]
                    return tech, method, roi
    return None

def base_method_name(name: str) -> str:
    name = name.lower()
    if name.startswith("pca"):
        return "pca"
    if name.startswith("ica"):
        return "ica"
    if name.startswith("zca"):
        return "zca"
    return name

def mean_abs_error(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    return float(np.nanmean(np.abs(a[mask] - b[mask]))) if mask.any() else np.nan

def safe_pearson(a: np.ndarray, b: np.ndarray, min_valid: int = 5) -> Tuple[float, float, int]:
    mask = np.isfinite(a) & np.isfinite(b)
    n = int(mask.sum())
    if n < min_valid:
        return np.nan, np.nan, n
    r, p = pearsonr(a[mask], b[mask])
    return float(r), float(p), n

def normalize_label(name: str) -> str:
    m = re.search(r"(s\d+).*?(t\d+)", name, flags=re.IGNORECASE)
    if m:
        return f"{m.group(1).upper()} {m.group(2).upper()}"
    return name

def format_pvalue(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"

def build_spread_table_for_video_label(label: str, csv_dir: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    TECH_ORDER = ["FFT", "WELCH", "HILBERT"]

    ROI_LABELS: Dict[str, str] = {
        "none":          "Full frame (baseline)",
        "manual":        "Manual crop ROI",
        "face_track":    "Face tracking (BlazeFace)",
        "bbox_forehead": "Face tracking (BlazeFace forehead)",
        "mesh_forehead": "Polygonal forehead (Landmarker)",
        "poly":          "Polygonal (Landmarker; forehead + cheeks)",
    }
    ROI_ORDER = ["none", "manual", "face_track", "bbox_forehead", "mesh_forehead", "poly"]

    METHOD_LABELS: Dict[str, str] = {
        "r":        "R",
        "g":        "G",
        "b":        "B",
        "grey_w":   "GREY Weight",
        "grey_a":   "GREY Avg",
        "pca":      "PCA",
        "zca":      "ZCA",
        "ica":      "ICA",
        "chrom":    "CHROM",
        "pos":      "POS",
    }
    METHOD_ORDER = ["r", "g", "b", "grey_w", "grey_a", "pca", "zca", "ica", "chrom", "pos"]

    matched_csv: Optional[Path] = None
    for csv_path in sorted(csv_dir.glob("*.csv")):
        if normalize_label(csv_path.stem).lower() == label.lower():
            matched_csv = csv_path
            break
    if matched_csv is None:
        raise FileNotFoundError(f"No CSV in {csv_dir} matched label: {label}")

    df = pd.read_csv(matched_csv)
    if "gt_hr_bpm" not in df.columns:
        raise ValueError(f"Missing 'gt_hr_bpm' in {matched_csv.name}")
    gt_hr = df["gt_hr_bpm"].to_numpy()

    rows = []
    for col in df.columns:
        parsed = parse_column_key(col)
        if not parsed:
            continue
        tech_raw, method_raw, roi_raw = parsed
        y = df[col].to_numpy()
        mask = np.isfinite(gt_hr) & np.isfinite(y)
        n_valid = int(mask.sum())
        if n_valid < 5:
            continue
        mae = mean_abs_error(y[mask], gt_hr[mask])
        r_val, p_val, _ = safe_pearson(y, gt_hr, min_valid=5)

        tech_disp = tech_raw.upper()
        method_key = base_method_name(method_raw).lower()
        roi_key = roi_raw

        method_disp = METHOD_LABELS.get(method_key, method_key.upper())
        roi_disp = ROI_LABELS.get(roi_key, roi_key.replace("_", " ").title())

        rows.append({
            "_tech": tech_disp,
            "_roi_key": roi_key,
            "_method_key": method_key,
            "Technique": tech_disp,
            "Method": method_disp,
            "ROI": roi_disp,
            "MAE (bpm)": mae,
            "r": r_val,
            "p": p_val,
            "n": n_valid,
        })

    if not rows:
        raise RuntimeError(f"No valid technique×method×ROI rows found for {matched_csv.name}")

    spread = pd.DataFrame(rows)

    comp_methods = {"pca", "zca", "ica"}
    is_comp = spread["_method_key"].isin(comp_methods)
    if is_comp.any():
        best_idx = spread[is_comp].groupby(["_tech", "_roi_key", "_method_key"])["MAE (bpm)"].idxmin()
        spread = pd.concat([spread.loc[best_idx], spread[~is_comp]], ignore_index=True)

    spread["_tech"] = pd.Categorical(spread["_tech"], categories=TECH_ORDER, ordered=True)
    spread["_roi_key"] = pd.Categorical(spread["_roi_key"], categories=ROI_ORDER, ordered=True)
    spread["_method_key"] = pd.Categorical(spread["_method_key"], categories=METHOD_ORDER, ordered=True)

    spread = spread.sort_values(by=["_tech", "_roi_key", "_method_key", "MAE (bpm)"], ascending=True)

    spread["MAE (bpm)"] = spread["MAE (bpm)"].round(2)
    spread["r"] = spread["r"].round(2)
    spread["p_display"] = spread["p"].apply(format_pvalue)

    display_cols = ["Technique", "Method", "ROI", "MAE (bpm)", "r", "p_display", "n"]
    spread_out = spread[display_cols].rename(columns={"p_display": "p"})

    stem = re.sub(r"\s+", "_", label.strip().lower())
    csv_out = out_dir / f"appendix_spread_{stem}.csv"
    spread_out.to_csv(csv_out, index=False)

    latex_body = spread_out.to_latex(index=False, escape=False, column_format="l l l r r l r")
    caption = f"All methods for {label}: mean absolute error and correlation with ECG."
    tag = f"tab:spread_{stem}"
    table_tex = (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{tag}}}\n"
        f"{latex_body}\n"
        "\\end{table}\n"
    )
    tex_out = out_dir / f"appendix_spread_{stem}.tex"
    tex_out.write_text(table_tex, encoding="utf-8")

    print(f"\nSaved appendix spread CSV  → {csv_out}")
    print(f"Saved appendix spread LaTeX → {tex_out}")
    return tex_out

if __name__ == "__main__":
    CSV_DIR = Path("outputs/dset1_timeseries")
    OUT_DIR = Path("outputs/plots")
    build_spread_table_for_video_label("S28 T1", CSV_DIR, OUT_DIR)
    build_spread_table_for_video_label("S28 T2", CSV_DIR, OUT_DIR)
