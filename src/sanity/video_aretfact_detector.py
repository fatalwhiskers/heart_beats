#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:
    plt = None


def _variance_of_laplacian(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var()) if gray.size else 0.0


def _tenengrad(gray: np.ndarray, ksize: int = 3) -> float:
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    return float(np.mean(gx * gx + gy * gy))


def _hf_ratio(gray: np.ndarray, hp_cutoff: float = 0.15) -> float:
    fft2 = np.fft.rfft2(gray.astype(np.float32))
    mag2 = (fft2.real ** 2 + fft2.imag ** 2)
    h, w = gray.shape
    yy, xx = np.ogrid[:h, :w]
    cy, cx = h // 2, w // 2
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_norm = r / r.max()
    hf_mask = r_norm >= hp_cutoff
    total = float(mag2.sum() + 1e-9)
    return float(mag2[hf_mask].sum() / total)


def compute_metrics_from_video(
    video_path: Path,
    fps_override: Optional[float] = None,
    resize_max: Optional[int] = 480,
    compute_hf: bool = True,
    start_s: Optional[float] = None,
    end_s: Optional[float] = None,
) -> pd.DataFrame:
    if cv2 is None:
        raise RuntimeError("OpenCV not available. Install opencv-python to process video files.")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if not np.isfinite(fps) or fps <= 0:
        fps = fps_override or 30.0

    if start_s is not None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(round(start_s * fps))))

    rows = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        t_s = frame_idx / fps
        if end_s is not None and t_s > end_s:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if resize_max and max(gray.shape[:2]) > resize_max:
            scale = resize_max / max(gray.shape[:2])
            gray = cv2.resize(gray, (int(gray.shape[1] * scale), int(gray.shape[0] * scale)), interpolation=cv2.INTER_AREA)

        row = {
            "frame": frame_idx,
            "time_sec": t_s,
            "vol": _variance_of_laplacian(gray),
            "tenengrad": _tenengrad(gray),
            "hf_ratio": _hf_ratio(gray) if compute_hf else np.nan,
        }
        rows.append(row)

    cap.release()
    return pd.DataFrame(rows)


def _pick_time_col(df: pd.DataFrame) -> Optional[str]:
    candidates = [c for c in df.columns if any(k in c.lower() for k in ["time_sec", "time_s", "seconds", "sec", "timestamp", "time"])]
    for preferred in ["time_sec", "time_s", "seconds", "sec"]:
        if preferred in df.columns:
            return preferred
    return candidates[0] if candidates else None


def _pick_metric_cols(df: pd.DataFrame) -> List[str]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    preferred: List[str] = []
    for key in ["vol", "variance_of_laplacian", "laplacian_var", "tenengrad", "tenen", "hf_ratio", "hf", "sharpness"]:
        preferred += [c for c in numeric_cols if key == c.lower()]
    for c in numeric_cols:
        cl = c.lower()
        if any(k in cl for k in ["vol", "laplac", "tenen", "tenengrad", "hf", "sharp"]):
            if c not in preferred:
                preferred.append(c)
    if not preferred:
        exclude = {"frame", "index", _pick_time_col(df) or ""}
        preferred = [c for c in numeric_cols if c not in exclude][:3]
    seen, out = set(), []
    for c in preferred:
        if c not in seen and c in df.columns:
            seen.add(c)
            out.append(c)
    return out


def detect_autofocus_hunts(
    df: pd.DataFrame,
    fps: Optional[float] = None,
    smooth_s: float = 0.2,
    thresh_z: float = 5.0,
    sign_change_window_s: float = 0.7,
    cluster_gap_s: float = 1.0,
    sharpness_preference: Optional[List[str]] = None,
) -> Dict[str, object]:
    time_col = _pick_time_col(df)
    metric_cols = sharpness_preference or _pick_metric_cols(df)
    if not metric_cols:
        raise ValueError("No numeric metric columns found.")
    priority = ["tenengrad", "tenen", "vol", "variance_of_laplacian", "laplacian_var"]
    sharp_col = next((c for p in priority for c in metric_cols if p == c.lower()), metric_cols[0])

    if time_col and pd.api.types.is_numeric_dtype(df[time_col]):
        t_s = df[time_col].astype(float).values
        if fps is None and len(t_s) > 1:
            dt = np.diff(t_s)
            dt = dt[dt > 0]
            fps = 1.0 / np.median(dt) if dt.size else 30.0
    else:
        n = len(df)
        fps = fps or 30.0
        t_s = np.arange(n, dtype=float) / fps
        time_col = "time_sec"

    sharp = df[sharp_col].astype(float).values
    win = max(3, int(round(fps * smooth_s)))
    sharp_smooth = pd.Series(sharp).rolling(win, center=True, min_periods=1).median().values

    d_sharp = np.diff(sharp_smooth)
    zscore = (d_sharp - np.nanmedian(d_sharp)) / (np.nanstd(d_sharp) + 1e-9)

    spike_idx = np.where(np.abs(zscore) > thresh_z)[0]
    window = max(5, int(round(fps * sign_change_window_s)))
    keep_idx = []
    for p in spike_idx:
        q0 = max(0, p - window)
        q1 = min(len(zscore) - 1, p + window)
        if np.any(zscore[q0:q1] * zscore[p] < 0):
            keep_idx.append(p)

    hunt_frames = np.array(sorted(set(keep_idx)), dtype=int)
    hunt_times = t_s[hunt_frames] if hunt_frames.size else np.array([], dtype=float)

    hunts_df = pd.DataFrame(
        {"hunt_frame": hunt_frames, time_col: hunt_times, "z_change": zscore[hunt_frames], "ds": d_sharp[hunt_frames]}
    )

    bursts = []
    if hunt_times.size:
        starts = [0]
        for i in range(1, len(hunt_times)):
            if (hunt_times[i] - hunt_times[i - 1]) > cluster_gap_s:
                starts.append(i)
        starts.append(len(hunt_times))
        for a, b in zip(starts[:-1], starts[1:]):
            sub_f = hunt_frames[a:b]
            sub_t = hunt_times[a:b]
            bursts.append(
                {
                    "hunts_in_burst": int(len(sub_f)),
                    "start_frame": int(sub_f[0]),
                    "end_frame": int(sub_f[-1] + 1),
                    "t_start_s": float(sub_t[0]),
                    "t_end_s": float(sub_t[-1]),
                    "duration_s": float(sub_t[-1] - sub_t[0]) if len(sub_t) > 1 else 0.0,
                }
            )
    return {"sharp_col": sharp_col, "time_col": time_col, "hunts": hunts_df, "bursts": pd.DataFrame(bursts)}


def merge_intervals(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        cur_start, cur_end = merged[-1]
        if start <= cur_end:
            merged[-1][1] = max(cur_end, end)
        else:
            merged.append([start, end])
    return [(float(a), float(b)) for a, b in merged]


def complement_intervals(intervals: List[Tuple[float, float]], start: float, end: float) -> List[Tuple[float, float]]:
    if not intervals:
        return [(start, end)]
    out: List[Tuple[float, float]] = []
    cursor = start
    for a, b in intervals:
        if a > cursor:
            out.append((cursor, a))
        cursor = max(cursor, b)
    if cursor < end:
        out.append((cursor, end))
    return out


def find_clean_windows(hunt_times: np.ndarray, t_start: float, t_end: float, window_s: float) -> List[Tuple[float, float]]:
    if t_end - t_start < window_s:
        return []
    if hunt_times.size == 0:
        return [(t_start, t_end)]
    step = 1.0
    cursor = t_start
    good_start: Optional[float] = None
    valid_ranges: List[Tuple[float, float]] = []
    while cursor + window_s <= t_end:
        has_hunt = np.any((hunt_times >= cursor) & (hunt_times <= cursor + window_s))
        if not has_hunt and good_start is None:
            good_start = cursor
        if has_hunt and good_start is not None:
            valid_ranges.append((good_start, cursor + window_s))
            good_start = None
        cursor += step
    if good_start is not None:
        valid_ranges.append((good_start, min(t_end, cursor + window_s)))
    return merge_intervals(valid_ranges)


def load_metrics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().replace(" ", "_").lower() for c in df.columns]
    if "frame" not in df.columns:
        df.insert(0, "frame", np.arange(len(df)))
    time_col = _pick_time_col(df)
    if time_col and not pd.api.types.is_numeric_dtype(df[time_col]):
        t = pd.to_datetime(df[time_col], errors="coerce")
        if not t.isna().all():
            t0 = t.dropna().iloc[0]
            df["time_sec"] = (t - t0).dt.total_seconds()
    return df


def save_df(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("input", type=str, help="Path to metrics CSV or a video file")
    parser.add_argument("--outdir", type=str, default=None, help="Output directory")
    parser.add_argument("--fps", type=float, default=None, help="FPS override")
    parser.add_argument("--smooth-s", type=float, default=0.2, help="Median smoothing window (seconds)")
    parser.add_argument("--thresh-z", type=float, default=5.0, help="Z-threshold on sharpness change")
    parser.add_argument("--hunt-window-s", type=float, default=0.7, help="Opposite-signed change window (seconds)")
    parser.add_argument("--cluster-gap-s", type=float, default=1.0, help="Max gap between hunts to cluster (seconds)")
    parser.add_argument("--pad-s", type=float, default=0.5, help="Pad seconds around each burst when excluding")
    parser.add_argument("--compute-hf", action="store_true", help="(Video) also compute hf_ratio (slower)")
    parser.add_argument("--resize-max", type=int, default=480, help="Resize long side to speed up (video)")
    parser.add_argument("--start-s", type=float, default=None, help="(Video) start time")
    parser.add_argument("--end-s", type=float, default=None, help="(Video) end time")
    parser.add_argument("--save-plot", action="store_true", help="Save a PNG plot of sharpness and AF hunts")
    parser.add_argument("--quiet", action="store_true", help="Reduce console output")
    args = parser.parse_args()

    inp_path = Path(args.input)
    out_dir_base = Path(args.outdir) if args.outdir else inp_path.with_suffix("")
    out_dir = out_dir_base.parent / (out_dir_base.name + "_qa")
    out_dir.mkdir(parents=True, exist_ok=True)

    if inp_path.suffix.lower() == ".csv":
        metrics_df = load_metrics(inp_path)
        src_kind = "csv"
    else:
        metrics_df = compute_metrics_from_video(
            inp_path,
            fps_override=args.fps,
            resize_max=args.resize_max,
            compute_hf=args.compute_hf,
            start_s=args.start_s,
            end_s=args.end_s,
        )
        save_df(metrics_df, out_dir / f"{inp_path.stem}_metrics.csv")
        src_kind = "video"

    detection = detect_autofocus_hunts(
        metrics_df,
        fps=args.fps,
        smooth_s=args.smooth_s,
        thresh_z=args.thresh_z,
        sign_change_window_s=args.hunt_window_s,
        cluster_gap_s=args.cluster_gap_s,
    )
    hunts_df: pd.DataFrame = detection["hunts"]  # type: ignore
    bursts_df: pd.DataFrame = detection["bursts"]  # type: ignore
    time_col: str = detection["time_col"]  # type: ignore
    sharp_col: str = detection["sharp_col"]  # type: ignore

    exclude_ranges: List[Tuple[float, float]] = []
    for _, row in bursts_df.iterrows():
        start = max(0.0, float(row["t_start_s"]) - args.pad_s)
        end = float(row["t_end_s"]) + args.pad_s
        exclude_ranges.append((start, end))
    exclude_merged = merge_intervals(exclude_ranges)

    t_start = float(metrics_df[time_col].iloc[0]) if time_col in metrics_df.columns else 0.0
    t_end = float(metrics_df[time_col].iloc[-1]) if time_col in metrics_df.columns else (len(metrics_df) / (args.fps or 30.0))
    keep_ranges = complement_intervals(exclude_merged, t_start, t_end)

    hunt_times = hunts_df[time_col].values if len(hunts_df) else np.array([], dtype=float)
    clean_30 = find_clean_windows(hunt_times, t_start, t_end, 30.0)
    clean_60 = find_clean_windows(hunt_times, t_start, t_end, 60.0)

    save_df(hunts_df, out_dir / f"{inp_path.stem}_hunts.csv")
    save_df(bursts_df, out_dir / f"{inp_path.stem}_hunt_bursts.csv")
    save_df(pd.DataFrame(exclude_merged, columns=["start_s", "end_s"]), out_dir / f"{inp_path.stem}_exclude_intervals.csv")
    save_df(pd.DataFrame(keep_ranges, columns=["start_s", "end_s"]), out_dir / f"{inp_path.stem}_keep_intervals.csv")
    save_df(
        pd.DataFrame(
            [{"window": "30s", "start_s": a, "end_s": b, "duration_s": b - a} for a, b in clean_30]
            + [{"window": "60s", "start_s": a, "end_s": b, "duration_s": b - a} for a, b in clean_60]
        ),
        out_dir / f"{inp_path.stem}_clean_windows.csv",
    )

    fps_est = None
    if time_col in metrics_df.columns and len(metrics_df) > 1:
        dt = np.diff(metrics_df[time_col].astype(float).values)
        dt = dt[dt > 0]
        if dt.size:
            fps_est = float(1.0 / np.median(dt))

    summary = {
        "input": str(inp_path),
        "source": src_kind,
        "rows": int(len(metrics_df)),
        "time_col": time_col,
        "sharpness_col": sharp_col,
        "fps_est": fps_est,
        "hunts_count": int(len(hunts_df)),
        "hunt_bursts": int(len(bursts_df)),
        "exclude_intervals_count": int(len(exclude_merged)),
        "keep_intervals_count": int(len(keep_ranges)),
        "params": {
            "smooth_s": args.smooth_s,
            "thresh_z": args.thresh_z,
            "sign_change_window_s": args.hunt_window_s,
            "cluster_gap_s": args.cluster_gap_s,
            "pad_s": args.pad_s,
        },
    }
    (out_dir / f"{inp_path.stem}_summary.json").write_text(json.dumps(summary, indent=2))

    if args.save_plot and plt is not None:
        t_vals = metrics_df[time_col].astype(float).values
        sharp_vals = metrics_df[sharp_col].astype(float).values
        win = max(3, int(round((fps_est or 30.0) * args.smooth_s)))
        sharp_smooth = pd.Series(sharp_vals).rolling(win, center=True, min_periods=1).median().values
        plt.figure()
        plt.plot(t_vals, sharp_smooth, label=f"{sharp_col} (smooth)")
        if len(hunts_df):
            plt.scatter(
                hunts_df[time_col].values,
                np.interp(hunts_df[time_col].values, t_vals, sharp_smooth),
                s=10,
                label="AF hunts",
            )
        plt.xlabel("Time (s)")
        plt.ylabel(sharp_col)
        plt.title("Sharpness & detected autofocus hunts")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{inp_path.stem}_hunts_plot.png", dpi=150)

    if not args.quiet:
        print(json.dumps(summary, indent=2))
        print(f"\nSaved outputs to: {out_dir.resolve()}")
        if args.save_plot and plt is None:
            print("(Install matplotlib to enable --save-plot)")
        if src_kind == "video" and cv2 is None:
            print("(Install opencv-python to process video files)")


if __name__ == "__main__":
    main()
