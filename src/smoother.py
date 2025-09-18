import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from collections import deque

#Casiez, Roussel, Vogel, 2012

def cutoff_to_alpha(cutoff_hz, dt: float):
    """
    Convert a cutoff frequency (Hz) to an EMA alpha for a sample interval dt (s).
    Accepts float or numpy array for cutoff_hz.
    """
    cutoff_hz = np.maximum(cutoff_hz, 1e-6)  # avoid div-by-zero
    dt = max(dt, 1e-6)
    tau = 1.0 / (2.0 * np.pi * cutoff_hz)
    return 1.0 / (1.0 + tau / dt)

@dataclass
class OneEuroFilter:
    """
    One-Euro filter for smoothing a 4D bbox state: (cx, cy, w, h).
    - base_cutoff_hz: smoothing when motion is small (lower = smoother)
    - responsiveness: how much to relax smoothing when motion increases
    - deriv_cutoff_hz: smoothing for the derivative estimate
    """
    base_cutoff_hz: float = 1.4
    responsiveness: float = 0.3     # AKA beta
    deriv_cutoff_hz: float = 1.0

    prev_value: Optional[np.ndarray] = None  # last smoothed (cx,cy,w,h)
    prev_derivative: Optional[np.ndarray] = None
    prev_time_s: Optional[float] = None

    def reset(self) -> None:
        self.prev_value = None
        self.prev_derivative = None
        self.prev_time_s = None

    def filter(self, value: np.ndarray, time_s: float) -> np.ndarray:
        """
        Smooth a new bbox vector at timestamp time_s (seconds).
        value shape: (4,) = (cx, cy, w, h)
        """
        if self.prev_value is None:
            self.prev_value = value.astype(np.float32)
            self.prev_derivative = np.zeros_like(self.prev_value)
            self.prev_time_s = time_s
            return self.prev_value

        dt = max(time_s - self.prev_time_s, 1e-3)

        # 1) Derivative (velocity of bbox state)
        derivative = (value - self.prev_value) / dt
        alpha_deriv = cutoff_to_alpha(self.deriv_cutoff_hz, dt)
        smoothed_derivative = alpha_deriv * derivative + (1 - alpha_deriv) * self.prev_derivative

        # 2) Adaptive smoothing based on motion magnitude
        adaptive_cutoff = self.base_cutoff_hz + self.responsiveness * np.abs(smoothed_derivative)
        alpha_value = cutoff_to_alpha(adaptive_cutoff, dt)
        smoothed_value = alpha_value * value + (1 - alpha_value) * self.prev_value

        # Update state
        self.prev_value = smoothed_value
        self.prev_derivative = smoothed_derivative
        self.prev_time_s = time_s
        return smoothed_value

def box_xyxy_to_cxcywh(x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
    """Convert [x1,y1,x2,y2] to [cx,cy,w,h]."""
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    return np.array([cx, cy, w, h], dtype=np.float32)

def box_cxcywh_to_xyxy(cx: float, cy: float, w: float, h: float) -> Tuple[float, float, float, float]:
    """Convert [cx,cy,w,h] to [x1,y1,x2,y2]."""
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return x1, y1, x2, y2

import cv2
import numpy as np
from src.config import *
class ROIKalman:
    def __init__(self):
        dt = 1.0 / float(Video.FPS)

        self.kf = cv2.KalmanFilter(8, 4, type=cv2.CV_32F)

        # State: [cx, cy, s, r, vx, vy, vs, vr]
        F = np.block([
            [np.eye(4), dt*np.eye(4)],
            [np.zeros((4,4)), np.eye(4)]
        ]).astype(np.float32)
        H = np.block([
            [np.eye(4), np.zeros((4,4))]
        ]).astype(np.float32)

        self.kf.transitionMatrix = F
        self.kf.measurementMatrix = H

        # Noise tuning (start here; then tweak)
        self.kf.processNoiseCov = np.diag([
            1e-4, 1e-4,   1e-5, 1e-5,   # cx,cy,s,r (positions & size change very slowly)
            1e-3, 1e-3,   1e-3, 1e-3    # velocities (small but nonzero so it can follow motion)
        ]).astype(np.float32)

        self.kf.measurementNoiseCov = np.diag([
             10.0, 10.0, 5.0, 5.0           # pixels for cx,cy and px for s; r is unitless
        ]).astype(np.float32)

        self.kf.errorCovPost = np.eye(8, dtype=np.float32)
        self.initialized = False

    def _box_to_state(self, box):
        x1,y1,x2,y2 = box
        w  = max(1.0, float(x2 - x1))
        h  = max(1.0, float(y2 - y1))
        cx = x1 + 0.5*w
        cy = y1 + 0.5*h
        s  = w                       # use width as scale
        r  = h / w                   # aspect
        return np.array([cx, cy, s, r], dtype=np.float32)

    def _state_to_box(self, x):
        cx, cy, s, r = x[:4]
        w = max(1.0, float(s))
        h = max(1.0, float(s * r))
        x1 = int(cx - 0.5*w); y1 = int(cy - 0.5*h)
        x2 = int(cx + 0.5*w); y2 = int(cy + 0.5*h)
        return (x1, y1, x2, y2)

    def update(self, measured_box, conf: float = 1.0):

        if not self.initialized:
            if measured_box is None:
                # no measurement yet → nothing sensible to return
                return None
            z4 = self._box_to_state(measured_box).reshape(4,1)
            # Seed the filter state with the first box, zero velocities
            self.kf.statePost = np.zeros((8,1), dtype=np.float32)
            self.kf.statePost[:4, 0] = z4[:, 0]
            self.kf.statePre  = self.kf.statePost.copy()
            self.initialized = True
            # Return the first box directly (no predict/correct on first frame)
            return measured_box


        # Predict first (good for dropouts)
        pred = self.kf.predict()
        pred_box = self._state_to_box(pred[:,0])

        if measured_box is None or conf <= 0.0:
            return pred_box  # pure prediction during dropout

        z = self._box_to_state(measured_box).reshape(4,1)

        # Higher conf -> lower noise; conf in [0,1]
        if conf < 1.0:
            R_base = np.diag([2.0, 2.0, 1.0, 1.0]).astype(np.float32)
            scale = 1.0 + 4.0*(1.0 - conf)  # up to 5x noisier at low conf
            self.kf.measurementNoiseCov = (R_base * scale).astype(np.float32)

        if not self.initialized:
            # Initialize state with first measurement
            self.kf.statePost[:4,0] = z[:,0]
            self.kf.statePost[4:,0] = 0.0
            self.initialized = True

        self.kf.correct(z)
        est = self.kf.statePost[:,0]
        return self._state_to_box(est)
    

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional


class JitterMeter:
    """
    Tracks ROI boxes over time and computes jitter metrics.
    Jitter is reported in px/frame (lower is better).
    """

    def __init__(self, fps: float, win_sec: float = 1.0) -> None:
        self.frames_per_second: float = float(fps)
        # smoothing window for drift removal
        self.window_length: int = int(max(3, round(win_sec * self.frames_per_second)))

        # time series of box centers and sizes
        self.center_x_values: List[float] = []
        self.center_y_values: List[float] = []
        self.width_values: List[float] = []
        self.height_values: List[float] = []

        # store raw boxes as (x1, y1, x2, y2)
        self.boxes_xyxy: List[Tuple[int, int, int, int]] = []

    @staticmethod
    def _box_to_center_and_size(box_xyxy: Tuple[int, int, int, int]) -> Tuple[float, float, float, float]:
        x1, y1, x2, y2 = box_xyxy
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        center_x = x1 + width * 0.5
        center_y = y1 + height * 0.5
        return center_x, center_y, width, height

    @staticmethod
    def _moving_average(x: np.ndarray, k: int) -> np.ndarray:
        if len(x) < 3 or k <= 1:
            return np.zeros_like(x, dtype=float)
        k = int(k)
        kernel = np.ones(k, dtype=float) / k
        smoothed = np.convolve(x, kernel, mode="same")
        return smoothed

    @staticmethod
    def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b

        inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
        inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
        inter_w, inter_h = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(1, (bx2 - bx1) * (by2 - by1))
        union = area_a + area_b - inter_area
        return inter_area / union if union > 0 else 0.0

    def add(self, box_xyxy: Tuple[int, int, int, int]) -> None:
        """Append a new ROI box (x1, y1, x2, y2)."""
        box_int = tuple(map(int, box_xyxy))
        self.boxes_xyxy.append(box_int)

        center_x, center_y, width, height = self._box_to_center_and_size(box_int)
        self.center_x_values.append(center_x)
        self.center_y_values.append(center_y)
        self.width_values.append(width)
        self.height_values.append(height)

    def metrics(self) -> Dict[str, float]:
        import numpy as _np  # keep original local import pattern

        if len(self.center_x_values) < 3:
            return dict(
                jitter_px=_np.nan, jitter_x=_np.nan, jitter_y=_np.nan,
                size_jitter=_np.nan, mean_iou=_np.nan
            )

        centers = _np.column_stack([self.center_x_values, self.center_y_values])
        center_deltas = _np.diff(centers, axis=0)  # (N-1, 2)
        delta_x, delta_y = center_deltas[:, 0], center_deltas[:, 1]

        # moving-average trend (same length as dx/dy)
        ma_dx = self._moving_average(delta_x, self.window_length)
        ma_dy = self._moving_average(delta_y, self.window_length)

        # ensure equal lengths (just in case)
        L = min(len(delta_x), len(ma_dx), len(delta_y), len(ma_dy))
        dx_high_freq = delta_x[-L:] - ma_dx[-L:]
        dy_high_freq = delta_y[-L:] - ma_dy[-L:]

        jitter_x = float(_np.nanstd(dx_high_freq))
        jitter_y = float(_np.nanstd(dy_high_freq))
        jitter_px = float(_np.nanmean(_np.sqrt(dx_high_freq ** 2 + dy_high_freq ** 2)))

        # size jitter
        widths = _np.asarray(self.width_values)
        heights = _np.asarray(self.height_values)
        dW = _np.diff(widths)
        dH = _np.diff(heights)

        ma_dW = self._moving_average(dW, self.window_length)
        ma_dH = self._moving_average(dH, self.window_length)

        Ls = min(len(dW), len(ma_dW), len(dH), len(ma_dH))
        dW_high_freq = dW[-Ls:] - ma_dW[-Ls:]
        dH_high_freq = dH[-Ls:] - ma_dH[-Ls:]
        size_jitter = float(0.5 * (_np.nanstd(dW_high_freq) + _np.nanstd(dH_high_freq)))

        # IoU stability
        ious = [self._iou(self.boxes_xyxy[i - 1], self.boxes_xyxy[i]) for i in range(1, len(self.boxes_xyxy))]
        mean_iou = float(_np.mean(ious)) if len(ious) else _np.nan

        return dict(
            jitter_px=jitter_px,
            jitter_x=jitter_x,
            jitter_y=jitter_y,
            size_jitter=size_jitter,
            mean_iou=mean_iou
        )


def summarize_all(jm_raw: JitterMeter, jm_kf: JitterMeter, jm_eur: JitterMeter, out_csv_path: Optional[str] = None) -> None:
    import csv

    raw_metrics = jm_raw.metrics()
    kf_metrics = jm_kf.metrics()
    eur_metrics = jm_eur.metrics()

    def rel_improve(baseline: Optional[float], other: Optional[float]) -> float:
        # percent decrease (positive = better)
        if baseline is None or other is None:
            return float('nan')
        return 100.0 * (baseline - other) / max(1e-9, baseline)

    # Compare jitter (primary) and IoU (secondary)
    raw_jitter = raw_metrics['jitter_px']; kf_jitter = kf_metrics['jitter_px']; eur_jitter = eur_metrics['jitter_px']
    raw_iou = raw_metrics['mean_iou'];  kf_iou = kf_metrics['mean_iou'];  eur_iou = eur_metrics['mean_iou']

    kf_impr_jitter = rel_improve(raw_jitter, kf_jitter)
    eur_impr_jitter = rel_improve(raw_jitter, eur_jitter)
    kf_impr_iou = rel_improve(1.0 - raw_iou, 1.0 - kf_iou)   # higher IoU = better
    eur_impr_iou = rel_improve(1.0 - raw_iou, 1.0 - eur_iou)

    # Choose winner by jitter first, break tie with IoU
    contenders = [
        ("RAW", raw_jitter, raw_iou),
        ("KF",  kf_jitter,  kf_iou),
        ("EUR", eur_jitter, eur_iou),
    ]
    winner = sorted(contenders, key=lambda t: (t[1], -t[2]))[0][0]

    print("\n=== ROI Stability Summary (lower jitter is better) ===")
    print(f"RAW : jitter={raw_jitter:.3f} px/frame | IoU={raw_iou:.3f}")
    print(f"KF  : jitter={kf_jitter:.3f} px/frame | IoU={kf_iou:.3f} | Δjitter vs RAW={kf_impr_jitter:+.1f}% | ΔIoU={kf_impr_iou:+.1f}%")
    print(f"EUR : jitter={eur_jitter:.3f} px/frame | IoU={eur_iou:.3f} | Δjitter vs RAW={eur_impr_jitter:+.1f}% | ΔIoU={eur_impr_iou:+.1f}%")
    print(f"\nWinner by jitter → {winner}")

    # Optional CSV
    if out_csv_path:
        rows = [
            ["method", "jitter_px_per_frame", "mean_iou", "delta_jitter_vs_raw_pct", "delta_iou_vs_raw_pct"],
            ["RAW", raw_jitter, raw_iou, 0.0, 0.0],
            ["KF",  kf_jitter,  kf_iou, kf_impr_jitter, kf_impr_iou],
            ["EUR", eur_jitter, eur_iou, eur_impr_jitter, eur_impr_iou],
        ]
        with open(out_csv_path, "w", newline="") as f:
            csv.writer(f).writerows(rows)
        print(f"Saved comparison CSV → {out_csv_path}")


def plot_traces(t: np.ndarray, raw: np.ndarray, med: np.ndarray, skin: np.ndarray, label: str = "G") -> None:
    plt.figure(figsize=(12, 4))
    plt.plot(t, raw,  label=f"{label} raw", alpha=0.5)
    plt.plot(t, med,  label=f"{label} median", linewidth=2)
    plt.plot(t, skin, label=f"{label} skin+eq", linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Mean intensity")
    plt.legend()
    plt.title(f"{label} channel traces")
    plt.show()
