from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from ds2_full_analysis import load_ds2, collapse_components_min_mae


def fisher_ci(r: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if not np.isfinite(r) or n < 4:
        return (np.nan, np.nan)
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(max(1, n - 3))
    zcrit = 1.959963984540054
    lo = np.tanh(z - zcrit * se)
    hi = np.tanh(z + zcrit * se)
    return float(lo), float(hi)


def extract_pca_poly_data(csv_path: str, fail_thresh: float = 10.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_df = load_ds2(csv_path)
    collapsed_df = collapse_components_min_mae(raw_df)
    pca_poly = collapsed_df[
        (collapsed_df['method'].astype(str).str.upper() == 'PCA') &
        (collapsed_df['roi'].astype(str).str.lower() == 'poly')
    ].copy()
    if pca_poly.empty:
        raise SystemExit("No rows found for PCA @ ROI=poly. Check your CSV and ROI naming.")
    pca_poly['video'] = pca_poly['subject'].astype(str) + '|' + pca_poly['recording'].astype(str)
    if pca_poly.duplicated('video').any():
        best_idx = pca_poly.groupby('video')['mae'].idxmin()
        pca_poly = pca_poly.loc[best_idx].copy()
    pca_poly['mean_pair'] = (
        pd.to_numeric(pca_poly['mean_gt'], errors='coerce') +
        pd.to_numeric(pca_poly['mean_rppg'], errors='coerce')
    ) / 2.0
    pca_poly['diff'] = (
        pd.to_numeric(pca_poly['mean_gt'], errors='coerce') -
        pd.to_numeric(pca_poly['mean_rppg'], errors='coerce')
    )
    pca_poly['failure'] = (pd.to_numeric(pca_poly['mae'], errors='coerce') > float(fail_thresh)).astype(int)
    gt_values = pd.to_numeric(pca_poly['mean_gt'], errors='coerce').to_numpy(dtype=float)
    rppg_values = pd.to_numeric(pca_poly['mean_rppg'], errors='coerce').to_numpy(dtype=float)
    valid_mask = np.isfinite(gt_values) & np.isfinite(rppg_values)
    n_agree = int(valid_mask.sum())
    agreement_r = float(np.corrcoef(gt_values[valid_mask], rppg_values[valid_mask])[0, 1]) if n_agree >= 2 else float('nan')
    agree_lo, agree_hi = fisher_ci(agreement_r, n_agree)
    diff_vals = pca_poly['diff'].to_numpy(dtype=float)
    ba_bias = float(np.nanmean(diff_vals))
    ba_sd = float(np.nanstd(diff_vals, ddof=1))
    ba_loa_lo = float(ba_bias - 2.0 * ba_sd)
    ba_loa_hi = float(ba_bias + 2.0 * ba_sd)
    columns_to_keep = [
        'subject', 'recording', 'video', 'roi', 'mae', 'rmse', 'r', 'bias', 'sd',
        'mean_gt', 'mean_rppg', 'mean_pair', 'diff', 'failure'
    ]
    ba_points_df = pca_poly[columns_to_keep].copy()
    summary_df = pd.DataFrame([{
        'method': 'PCA',
        'roi': 'poly',
        'n_subjects': int(pd.unique(pca_poly['subject']).size),
        'n_videos': int(pd.unique(pca_poly['video']).size),
        'mae_mean': float(np.nanmean(pca_poly['mae'])),
        'mae_median': float(np.nanmedian(pca_poly['mae'])),
        'rmse_mean': float(np.nanmean(pca_poly['rmse'])),
        'rmse_median': float(np.nanmedian(pca_poly['rmse'])),
        'r_mean': float(np.nanmean(pca_poly['r'])),
        'r_median': float(np.nanmedian(pca_poly['r'])),
        'agreement_r': agreement_r,
        'agreement_r_lo95': agree_lo,
        'agreement_r_hi95': agree_hi,
        'ba_bias': ba_bias,
        'ba_sd': ba_sd,
        'ba_loa_lo': ba_loa_lo,
        'ba_loa_hi': ba_loa_hi,
        'failure_thresh': float(fail_thresh),
        'failure_rate_pct': float(100.0 * np.nanmean(pd.to_numeric(pca_poly['mae'], errors='coerce') > float(fail_thresh))),
    }])
    return ba_points_df, summary_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default='outputs/Dataset2_hilbert_results.csv')
    parser.add_argument('--fail', type=float, default=10.0)
    args = parser.parse_args()
    output_dir = Path('outputs')
    output_dir.mkdir(parents=True, exist_ok=True)
    ba_points_df, summary_df = extract_pca_poly_data(args.csv, args.fail)
    ba_points_df.to_csv(output_dir / 'pca_poly_ba_points.csv', index=False)
    summary_df.round(4).to_csv(output_dir / 'pca_poly_summary.csv', index=False)
    print('\n=== PCA @ ROI=poly — per-video BA points ===')
    print(ba_points_df.head(10).round(3).to_string(index=False))
    print('\n=== PCA @ ROI=poly — summary (MAE/RMSE/r + BA + agreement r) ===')
    print(summary_df.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
