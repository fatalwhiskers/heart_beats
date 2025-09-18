# ds2_full_analysis.py
import re
from pathlib import Path
from contextlib import contextmanager

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


@contextmanager
def apa7_style(figsize=(6.5, 4.5), dpi=300):
    """
    Use inside a `with` block to apply APA 7–ish defaults:
    - Sans-serif font (Arial/Helvetica/DejaVu Sans), 12 pt text
    - Clean axes (no top/right spines), minimal grid
    - 300 dpi, tight bounding box
    """
    original_rc = mpl.rcParams.copy()
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
        mpl.rcParams.update(original_rc)


CANON_RECORDINGS = [
    'Baseline', 'Breathing', 'Counting1', 'Counting2', 'Counting3',
    'Math', 'Reading', 'Relax', 'Speaking', 'Stroop', 'Video1', 'Video2'
]

REC_PATTERNS = {
    'Baseline': r'baseline',
    'Breathing': r'breath',
    'Counting1': r'count(?:ing)?[_\s-]*1',
    'Counting2': r'count(?:ing)?[_\s-]*2',
    'Counting3': r'count(?:ing)?[_\s-]*3',
    'Math': r'math',
    'Reading': r'read',
    'Relax': r'relax',
    'Speaking': r'speak',
    'Stroop': r'stroop',
    'Video1': r'video[_\s-]*1',
    'Video2': r'video[_\s-]*2',
}


def _strip_subject_prefix(recording: str, subject: str) -> str:
    recording_str = str(recording)
    prefix = f"{subject}_"
    return recording_str[len(prefix):] if recording_str.startswith(prefix) else recording_str


def _to_canonical_recording(rec_name_raw: str) -> str | None:
    raw = str(rec_name_raw).lower()
    for canonical, pattern in REC_PATTERNS.items():
        if re.search(pattern, raw, flags=re.IGNORECASE):
            return canonical
    return None


def _pca_poly_filtered(df_collapsed: pd.DataFrame) -> pd.DataFrame:
    filtered = df_collapsed.copy()
    filtered = filtered[
        (filtered['method'].astype(str).str.upper() == 'PCA') &
        (filtered['roi'].astype(str).str.lower() == 'poly')
    ].copy()
    if filtered.empty:
        raise ValueError("No rows for method='PCA' with roi='poly'.")
    filtered['video'] = filtered['subject'].astype(str) + '|' + filtered['recording'].astype(str)
    if filtered.duplicated('video').any():
        best_idx = filtered.groupby('video')['mae'].idxmin()
        filtered = filtered.loc[best_idx].copy()
    filtered['record_raw'] = [
        _strip_subject_prefix(rec, subj) for subj, rec in zip(filtered['subject'], filtered['recording'])
    ]
    filtered['recording_canon'] = filtered['record_raw'].map(_to_canonical_recording)
    filtered = filtered[filtered['recording_canon'].isin(CANON_RECORDINGS)].copy()
    filtered['video_id'] = filtered['subject'].astype(str) + '_' + filtered['recording_canon'].astype(str)
    return filtered


def table_recordings_best_to_worst(df_collapsed: pd.DataFrame, failure_thresh: float = 10.0) -> pd.DataFrame:
    """
    Rank the 12 recording types by mean MAE (ascending).
    Columns: recording, n_videos, mean_mae, median_mae, min_mae, max_mae, sd_mae, failure_rate_%.
    """
    filtered = _pca_poly_filtered(df_collapsed)
    grouped = filtered.groupby('recording_canon')['mae']
    ranking = pd.DataFrame({
        'recording': grouped.count().index,
        'n_videos': grouped.count().values,
        'mean_mae': grouped.mean().values,
        'median_mae': grouped.median().values,
        'min_mae': grouped.min().values,
        'max_mae': grouped.max().values,
        'sd_mae': grouped.std(ddof=1).values,
    })
    failure_mask = (filtered.assign(fail=filtered['mae'] > failure_thresh)
                    .groupby('recording_canon')['fail'].mean()
                    .reindex(ranking['recording']).to_numpy())
    ranking['failure_rate_%'] = np.round(100.0 * failure_mask, 2)
    ranking = ranking.sort_values('mean_mae', ascending=True).reset_index(drop=True)
    ranking.insert(0, 'rank', np.arange(1, len(ranking) + 1))
    numeric_cols = ['mean_mae', 'median_mae', 'min_mae', 'max_mae', 'sd_mae']
    ranking[numeric_cols] = ranking[numeric_cols].round(2)
    return ranking


def table_videos_best_to_worst(df_collapsed: pd.DataFrame) -> pd.DataFrame:
    """
    Rank individual videos by MAE (ascending). One row per subject_recording.
    Columns: rank, video_id, subject, recording, mae, rmse, r, bias, sd.
    """
    filtered = _pca_poly_filtered(df_collapsed)
    table = filtered.loc[:, ['video_id', 'subject', 'recording', 'mae', 'rmse', 'r', 'bias', 'sd']].copy()
    table = table.sort_values('mae', ascending=True).reset_index(drop=True)
    table.insert(0, 'rank', np.arange(1, len(table) + 1))
    for col in ['mae', 'rmse', 'r', 'bias', 'sd']:
        if col in table.columns:
            table[col] = pd.to_numeric(table[col], errors='coerce').round(2)
    return table


NUM_COLS = [
    'Number of Windows', 'MAE (bpm)', 'RMSE (bpm)', 'Pearson r', 'Pearson p',
    'Bias (bpm)', 'SD (bpm)', 'LoA Lower (bpm)', 'LoA Upper (bpm)',
    'Mean rPPG (bpm)', 'Median rPPG (bpm)',
    'Mean Ground Truth (bpm)', 'Median Ground Truth (bpm)',
    'Mean Error (bpm)', 'Median Error (bpm)', 'Median Absolute Error (bpm)'
]
RENAME = {
    'Subject ID': 'subject',
    'Recording ID': 'recording',
    'ROI': 'roi',
    'Extraction Method': 'method',
    'Number of Windows': 'n_windows',
    'MAE (bpm)': 'mae',
    'RMSE (bpm)': 'rmse',
    'Pearson r': 'r',
    'Pearson p': 'pval',
    'Bias (bpm)': 'bias',
    'SD (bpm)': 'sd',
    'LoA Lower (bpm)': 'loa_lower',
    'LoA Upper (bpm)': 'loa_upper',
    'Mean rPPG (bpm)': 'mean_rppg',
    'Median rPPG (bpm)': 'median_rppg',
    'Mean Ground Truth (bpm)': 'mean_gt',
    'Median Ground Truth (bpm)': 'median_gt',
    'Mean Error (bpm)': 'mean_err',
    'Median Error (bpm)': 'median_err',
    'Median Absolute Error (bpm)': 'med_abs_err',
}


def _to_float(value):
    if isinstance(value, str) and value.strip().startswith('<'):
        try:
            return float(value.strip()[1:])
        except Exception:
            return np.nan
    try:
        return float(value)
    except Exception:
        return np.nan


def load_ds2(csv_path: str) -> pd.DataFrame:
    dataframe = pd.read_csv(csv_path)
    for col in NUM_COLS:
        if col in dataframe.columns:
            dataframe[col] = dataframe[col].apply(_to_float)
    dataframe = dataframe.rename(columns=RENAME)
    dataframe['video'] = dataframe['subject'].astype(str) + '|' + dataframe['recording'].astype(str)
    return dataframe


def add_base_method(df: pd.DataFrame, method_col: str = 'method') -> pd.DataFrame:
    updated = df.copy()
    base_methods, components = [], []
    for method_str in updated[method_col].astype(str):
        method_str = method_str.strip()
        parsed = re.match(r'^(PCA|ZCA|ICA)_(\d+)$', method_str, flags=re.IGNORECASE)
        if parsed:
            base_methods.append(parsed.group(1).upper())
            components.append(int(parsed.group(2)))
        else:
            base_methods.append(method_str)
            components.append(np.nan)
    updated['base_method'] = base_methods
    updated['component'] = components
    return updated


def collapse_components_min_mae(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pick the lowest-MAE component within each (video, roi, base_method).
    Returns one row per (video, roi, base_method) with columns:
      - method (base method), best_component, original_method (e.g., PCA_2)
    """
    source = df
    if 'base_method' not in source.columns or 'component' not in source.columns:
        source = add_base_method(source)
    best_idx = source.groupby(['video', 'roi', 'base_method'])['mae'].idxmin()
    collapsed = source.loc[best_idx].copy()
    collapsed = collapsed.rename(columns={'method': 'original_method', 'component': 'best_component'})
    collapsed['method'] = collapsed['base_method']
    return collapsed


def best_roi_per_method(
    df: pd.DataFrame,
    score_col='mae',
    recordings_filter_contains: list[str] | None = None
) -> pd.DataFrame:
    data = df.copy()
    if recordings_filter_contains:
        include_mask = np.zeros(len(data), dtype=bool)
        for pattern in recordings_filter_contains:
            include_mask |= data['recording'].astype(str).str.contains(pattern)
        data = data[include_mask]
    summary = (
        data.groupby(['method', 'roi'])
        .agg(
            n_videos=('video', 'nunique'),
            n_rows=('video', 'size'),
            mean_score=(score_col, 'mean'),
            median_score=(score_col, 'median'),
        )
        .reset_index()
    )
    best_indices = summary.groupby('method')['mean_score'].idxmin()
    best = summary.loc[best_indices].sort_values('mean_score').reset_index(drop=True)
    best = best.rename(columns={
        'roi': 'best_roi',
        'mean_score': f'mean_{score_col}',
        'median_score': f'median_{score_col}'
    })
    return best


def per_video_bestROI(df: pd.DataFrame, best_table: pd.DataFrame, method_subset=None) -> pd.DataFrame:
    method_to_roi = dict(zip(best_table['method'], best_table['best_roi']))
    subset = df[df['method'].isin(method_to_roi.keys())].copy()
    subset = subset[subset.apply(lambda row: row['roi'] == method_to_roi[row['method']], axis=1)]
    if method_subset:
        subset = subset[subset['method'].isin(method_subset)]
    best_idx = subset.groupby(['video', 'method'])['mae'].idxmin()
    per_video = subset.loc[
        best_idx, ['subject', 'recording', 'video', 'method', 'roi',
                   'mean_gt', 'mean_rppg', 'mae', 'rmse', 'r', 'bias', 'sd']
    ].copy()
    return per_video.sort_values(['method', 'subject', 'recording']).reset_index(drop=True)


def plot_timeseries(window_csv: str, methods_to_plot: list[str], title='Timeseries: GT vs rPPG'):
    timeseries = pd.read_csv(window_csv)
    if not {'time', 'gt_hr'}.issubset(timeseries.columns):
        raise ValueError("CSV must contain columns 'time' and 'gt_hr' plus method columns.")
    plt.figure()
    plt.plot(timeseries['time'], timeseries['gt_hr'], label='Ground Truth')
    for method_name in methods_to_plot:
        if method_name not in timeseries.columns:
            print(f"Warning: method '{method_name}' not found in window CSV; skipping.")
            continue
        plt.plot(timeseries['time'], timeseries[method_name], label=method_name)
    plt.xlabel('Time')
    plt.ylabel('Heart Rate (bpm)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_scatter_across_videos(per_video_tbl: pd.DataFrame, title='GT vs rPPG (means across videos)'):
    plt.figure()
    for method_name, subset in per_video_tbl.groupby('method'):
        plt.scatter(subset['mean_gt'], subset['mean_rppg'], label=method_name, alpha=0.8)
    lower = float(np.nanmin([per_video_tbl['mean_gt'].min(), per_video_tbl['mean_rppg'].min()]))
    upper = float(np.nanmax([per_video_tbl['mean_gt'].max(), per_video_tbl['mean_rppg'].max()]))
    plt.plot([lower, upper], [lower, upper])
    plt.xlabel('Mean HR (GT) [bpm]')
    plt.ylabel('Mean HR (rPPG) [bpm]')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_bland_altman_camera_by_method_markers_from_recording(
    df: pd.DataFrame,
    method: str,
    gt_col: str = 'mean_gt',
    cam_col: str = 'mean_rppg',
    subj_col: str = 'subject',
    rec_col: str = 'recording',
    roi_col: str = 'roi',
    meth_col: str = 'method',
    mae_col: str = 'mae',
    recordings_filter_contains: list[str] | None = None,
    best_roi: bool = True,
    k_sd: float = 2.0,
    show_tolerance10: bool = False,
    tol_bpm: float = 10.0,
    color_by_subject: bool = True,
    save_path: str | None = None,
    figure_title_prefix: str = 'ECG–Camera',
    title: bool = True,
    speaking_label: str = "Speaking"
):
    import re as _re

    data = df.copy()
    for col in [gt_col, cam_col, mae_col]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')

    def _parse(method_str):
        method_str = str(method_str).strip()
        match = _re.match(r'^(PCA|ZCA|ICA)_(\d+)$', method_str, flags=_re.I)
        if match:
            return match.group(1).upper(), int(match.group(2))
        return method_str, np.nan

    base_method_vals, component_vals = zip(*data[meth_col].map(_parse))
    data['base_method'] = base_method_vals
    data['component'] = component_vals

    if _re.fullmatch(r'(PCA|ZCA|ICA)', method, flags=_re.I):
        data = data[data['base_method'].str.upper() == method.upper()].copy()
        data['video'] = data[subj_col].astype(str) + '|' + data[rec_col].astype(str)
        best_idx = data.groupby(['video', roi_col])[mae_col].idxmin()
        data = data.loc[best_idx].copy()
        method_display = method.upper()
    else:
        data = data[data[meth_col].astype(str).str.upper() == method.upper()].copy()
        method_display = method

    if recordings_filter_contains:
        mask = np.zeros(len(data), dtype=bool)
        rec_names = data[rec_col].astype(str)
        for pattern in recordings_filter_contains:
            mask |= rec_names.str.contains(pattern, na=False)
        data = data[mask]

    if data.empty:
        raise ValueError(f"No rows found for method='{method}' after filtering.")

    best_roi_name = None
    if best_roi and roi_col in data.columns:
        roi_rank = data.groupby(roi_col)[mae_col].mean().sort_values()
        best_roi_name = roi_rank.index[0]
        data = data[data[roi_col] == best_roi_name].copy()

    data['video'] = data[subj_col].astype(str) + '|' + data[rec_col].astype(str)
    if data.duplicated('video').any():
        best_idx2 = data.groupby('video')[mae_col].idxmin()
        data = data.loc[best_idx2].copy()

    rec_low = data[rec_col].astype(str).str.lower()
    condition = np.where(
        rec_low.str.contains("speaking", na=False), "speaking",
        np.where(rec_low.str.contains("baseline", na=False), "baseline", "ignore")
    )
    data['__condition__'] = condition
    data = data[data['__condition__'] != "ignore"].copy()

    if data.empty:
        raise ValueError(f"No baseline/speaking recordings found for method='{method}'.")

    mean_pair = (data[gt_col].values + data[cam_col].values) / 2.0
    diff = (data[gt_col].values - data[cam_col].values)

    bias = float(np.nanmean(diff))
    sd = float(np.nanstd(diff, ddof=1))
    loa_lo, loa_hi = bias - k_sd * sd, bias + k_sd * sd

    if color_by_subject:
        subjects = pd.unique(data[subj_col])
        cmap = plt.get_cmap('tab20' if len(subjects) > 10 else 'tab10')
        color_map = {s: cmap(i / max(1, len(subjects) - 1)) for i, s in enumerate(subjects)}
        point_colors = data[subj_col].map(color_map).tolist()
    else:
        point_colors = ['0.35'] * len(data)

    with apa7_style():
        fig, ax = plt.subplots()

        for condition_value in ["baseline", "Counting1"]:
            mask = data['__condition__'] == condition_value
            if not np.any(mask):
                continue
            marker_style = "o" if condition_value == "baseline" else "^"
            mp_subset = np.asarray(mean_pair)[mask.values]
            df_subset = np.asarray(diff)[mask.values]
            color_subset = [c for c, keep in zip(point_colors, mask.values) if keep]
            ax.scatter(mp_subset, df_subset, s=36, alpha=0.9, marker=marker_style,
                       c=color_subset, edgecolors='none')

        ax.axhline(bias, linestyle='--', linewidth=1.6, color='0.15')
        ax.axhline(loa_hi, linestyle='--', linewidth=1.6, color='0.25')
        ax.axhline(loa_lo, linestyle='--', linewidth=1.6, color='0.25')

        roi_note = f" — best ROI: {best_roi_name}" if best_roi_name else ""
        ax.set_xlabel('Mean of HR estimates (BPM)')
        ax.set_ylabel('Difference (ECG − Camera) (BPM)')

        if title:
            ax.set_title(f'Bland–Altman: {figure_title_prefix} (method={method_display}){roi_note}')

        x0, x1 = ax.get_xlim()
        xr = (x1 - x0)
        y0, y1 = ax.get_ylim()
        yr = (y1 - y0)
        offset = 0.04 * yr

        ax.text(x0 + 0.98 * xr, loa_hi + offset,
                f"+{k_sd:g}SD", ha='right', va='bottom', fontsize=9, color='0.25')
        ax.text(x0 + 0.98 * xr, loa_lo - offset,
                f"−{k_sd:g}SD", ha='right', va='top', fontsize=9, color='0.25')

        if show_tolerance10:
            ax.axhline(+tol_bpm, linestyle=':', linewidth=1.2, color='0.35')
            ax.axhline(-tol_bpm, linestyle=':', linewidth=1.2, color='0.35')

        ax.text(x0 + 0.98 * xr, y0 + 0.90 * yr, 'ECG > Camera', ha='right', va='center')
        ax.text(x0 + 0.98 * xr, y0 + 0.10 * yr, f'{speaking_label} > ECG', ha='right', va='center')

        ax.grid(False)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path)

    return {
        'bias': bias,
        'sd': sd,
        'loa_lo': loa_lo,
        'loa_hi': loa_hi,
        'best_roi': best_roi_name
    }


def plot_bland_altman_camera_by_method(
    df: pd.DataFrame,
    method: str,
    gt_col: str = 'mean_gt',
    cam_col: str = 'mean_rppg',
    subj_col: str = 'subject',
    rec_col: str = 'recording',
    roi_col: str = 'roi',
    meth_col: str = 'method',
    mae_col: str = 'mae',
    recordings_filter_contains: list[str] | None = None,
    best_roi: bool = True,
    k_sd: float = 2.0,
    show_tolerance10: bool = False,
    tol_bpm: float = 10.0,
    color_by_subject: bool = True,
    save_path: str | None = None,
    figure_title_prefix: str = 'ECG–Camera',
    title: bool = False,
):
    import re as _re

    data = df.copy()
    for col in [gt_col, cam_col, mae_col]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')

    def _parse(method_str):
        method_str = str(method_str).strip()
        match = _re.match(r'^(PCA|ZCA|ICA)_(\d+)$', method_str, flags=_re.I)
        if match:
            return match.group(1).upper(), int(match.group(2))
        return method_str, np.nan

    base_method_vals, component_vals = zip(*data[meth_col].map(_parse))
    data['base_method'] = base_method_vals
    data['component'] = component_vals

    if _re.fullmatch(r'(PCA|ZCA|ICA)', method, flags=_re.I):
        data = data[data['base_method'].str.upper() == method.upper()].copy()
        data['video'] = data[subj_col].astype(str) + '|' + data[rec_col].astype(str)
        best_idx = data.groupby(['video', roi_col])[mae_col].idxmin()
        data = data.loc[best_idx].copy()
        method_display = method.upper()
    else:
        data = data[data[meth_col].astype(str).str.upper() == method.upper()].copy()
        method_display = method

    if recordings_filter_contains:
        mask = np.zeros(len(data), dtype=bool)
        rec_names = data[rec_col].astype(str)
        for pattern in recordings_filter_contains:
            mask |= rec_names.str.contains(pattern, na=False)
        data = data[mask]

    if data.empty:
        raise ValueError(f"No rows found for method='{method}' after filtering.")

    best_roi_name = None
    if best_roi and roi_col in data.columns:
        roi_rank = data.groupby(roi_col)[mae_col].mean().sort_values()
        best_roi_name = roi_rank.index[0]
        data = data[data[roi_col] == best_roi_name].copy()

    data['video'] = data[subj_col].astype(str) + '|' + data[rec_col].astype(str)
    if data.duplicated('video').any():
        best_idx2 = data.groupby('video')[mae_col].idxmin()
        data = data.loc[best_idx2].copy()

    mean_pair = (data[gt_col].values + data[cam_col].values) / 2.0
    diff = (data[gt_col].values - data[cam_col].values)

    bias = float(np.nanmean(diff))
    sd = float(np.nanstd(diff, ddof=1))
    loa_lo, loa_hi = bias - k_sd * sd, bias + k_sd * sd

    with apa7_style():
        fig, ax = plt.subplots()
        if color_by_subject:
            subjects = pd.unique(data[subj_col])
            cmap = plt.get_cmap('tab20' if len(subjects) > 10 else 'tab10')
            color_map = {s: cmap(i / max(1, len(subjects) - 1)) for i, s in enumerate(subjects)}
            colors = [color_map[s] for s in data[subj_col]]
        else:
            colors = '0.35'
        ax.scatter(mean_pair, diff, s=36, alpha=0.9, marker='o', c=colors, edgecolors='none')

        ax.axhline(bias, linestyle='--', linewidth=1.6, color='0.15')
        ax.axhline(loa_hi, linestyle='--', linewidth=1.6, color='0.25')
        ax.axhline(loa_lo, linestyle='--', linewidth=1.6, color='0.25')

        roi_note = f" — best ROI: {best_roi_name}" if best_roi_name else ""
        ax.set_xlabel('Mean of HR Estimates (BPM)')
        ax.set_ylabel('Difference (BPM)')

        if title:
            ax.set_title(f'Bland–Altman: {figure_title_prefix} (method={method_display}){roi_note}')

        x0, x1 = ax.get_xlim()
        xr = (x1 - x0)
        y0, y1 = ax.get_ylim()
        yr = (y1 - y0)
        offset = 0.02 * yr

        ax.text(x0 + 0.98 * xr, loa_hi + offset,
                f"+{k_sd:g}SD", ha='right', va='bottom', fontsize=9, color='0.25')
        ax.text(x0 + 0.98 * xr, loa_lo - offset,
                f"−{k_sd:g}SD", ha='right', va='top', fontsize=9, color='0.25')

        if show_tolerance10:
            ax.axhline(+tol_bpm, linestyle=':', linewidth=1.2, color='0.35')
            ax.axhline(-tol_bpm, linestyle=':', linewidth=1.2, color='0.35')

        ax.text(x0 + 0.98 * xr, y0 + 0.90 * yr, 'ECG > Camera', ha='right', va='center')
        ax.text(x0 + 0.98 * xr, y0 + 0.10 * yr, 'Camera > ECG', ha='right', va='center')

        ax.grid(False)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path)

    return {'bias': bias, 'sd': sd, 'loa_lo': loa_lo, 'loa_hi': loa_hi, 'best_roi': best_roi_name}


def plot_boxplot_top_methods(per_video_tbl: pd.DataFrame, top_k=3):
    method_ranking = (
        per_video_tbl.groupby('method')['mae'].mean()
        .sort_values()
        .head(top_k)
        .index.tolist()
    )
    data = [per_video_tbl[per_video_tbl['method'] == m]['mae'].values for m in method_ranking]
    plt.figure()
    plt.boxplot(data, labels=method_ranking, showfliers=True)
    plt.ylabel('MAE (bpm)')
    plt.title(f'Per-video MAE — Top {top_k} methods (best ROI)')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()


def _pivot_mae(df, rec_substr):
    subset = df[df['recording'].astype(str).str.contains(rec_substr)]
    if subset.empty:
        return pd.DataFrame()
    pivot = pd.pivot_table(subset, index='method', columns='roi', values='mae', aggfunc='mean')
    return pivot


def plot_heatmaps_baseline_breathing(df: pd.DataFrame, baseline_key='Baseline', breathing_key='Breathing'):
    pv_baseline = _pivot_mae(df, baseline_key)
    pv_breathing = _pivot_mae(df, breathing_key)

    if pv_baseline.empty and pv_breathing.empty:
        print("No rows matched 'Baseline' or 'Breathing' in 'recording'; heatmaps skipped.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, pivot, title in [(axes[0], pv_baseline, 'Baseline'), (axes[1], pv_breathing, 'Breathing')]:
        if pivot.empty:
            ax.axis('off')
            ax.set_title(f'Mean MAE — {title} (no data)')
            continue
        im = ax.imshow(pivot.values, aspect='auto')
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha='right')
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_title(f'Mean MAE — {title}')
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                value = pivot.values[i, j]
                if np.isfinite(value):
                    ax.text(j, i, f'{value:.1f}', ha='center', va='center', fontsize=8)
    fig.colorbar(im, ax=axes.ravel().tolist(), label='MAE (bpm)')
    plt.tight_layout()
    plt.show()


def failure_table(df: pd.DataFrame, threshold=10.0, by=('method',)) -> pd.DataFrame:
    stats = (
        df.groupby(list(by))
        .agg(
            n_videos=('video', 'nunique'),
            n_rows=('video', 'size'),
            n_failures=('mae', lambda x: (x > threshold).sum())
        )
        .reset_index()
    )
    stats['failure_rate_%'] = 100.0 * stats['n_failures'] / stats['n_videos']
    return stats.sort_values('failure_rate_%')


def failure_rates_both(df: pd.DataFrame, best: pd.DataFrame, threshold=10.0):
    all_fail = failure_table(df, threshold=threshold, by=('method', 'roi'))
    method_to_roi = dict(zip(best['method'], best['best_roi']))
    best_only = df[df['method'].isin(method_to_roi.keys()) & df.apply(
        lambda row: row['roi'] == method_to_roi[row['method']], axis=1)]
    best_fail = failure_table(best_only, threshold=threshold, by=('method',))
    return all_fail, best_fail


def plot_failure_bars(fail_df: pd.DataFrame, x='method', y='failure_rate_%', title='Failure rate (MAE > 10 bpm)'):
    plt.figure(figsize=(10, 5))
    plt.bar(fail_df[x].astype(str), fail_df[y])
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Failure rate (%)')
    plt.title(title)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    df_raw = load_ds2('outputs\Dataset2_hilbert_results.csv')
    df_collapsed = collapse_components_min_mae(df_raw)

    best = best_roi_per_method(df_collapsed, score_col='mae')
    best_display = best[['method', 'best_roi', 'n_videos', 'n_rows', 'mean_mae', 'median_mae']].copy()
    best_display[['mean_mae', 'median_mae']] = best_display[['mean_mae', 'median_mae']].round(2)
    print("\nBest ROI per method (components collapsed):\n", best_display.to_string(index=False))

    method_subset = None
    per_video = per_video_bestROI(df_collapsed, best, method_subset=method_subset)

    chosen_method = 'PCA'

    stats = plot_bland_altman_camera_by_method(
        df=df_collapsed,
        method='PCA',
        recordings_filter_contains=None,
        best_roi=True,
        k_sd=2.0,
        show_tolerance10=True,
        color_by_subject=True,
        save_path='figure_ba_pca.png'
    )

    results = plot_bland_altman_camera_by_method_markers_from_recording(
        df=df_collapsed,
        method="POS",
        speaking_label="Counting1",
        title=False,
        show_tolerance10=True
    )

    plot_boxplot_top_methods(per_video, top_k=3)
    plot_heatmaps_baseline_breathing(df_collapsed, baseline_key='Baseline', breathing_key='Breathing')

    fail_all, fail_best = failure_rates_both(df_collapsed, best, threshold=10.0)
    print("\nFailure (all ROI × method):\n", fail_all.to_string(index=False))
    print("\nFailure (best ROI per method):\n", fail_best.to_string(index=False))
    plot_failure_bars(fail_best, x='method', title='Failure rate (MAE > 10) — best ROI per method')
