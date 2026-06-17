"""
debug_visualiser.py
-------------------
Visualisation helpers for development and diagnostics.

All methods are intentionally stateless — pass in the data, get a result
back.  Nothing is written to disk here; call sites decide where to save.

Typical usage
-------------
>>> vis = DebugVisualiser()
>>> annotated = vis.draw_overlay(frame, crop_mode, crop_coords)
>>> vis.plot_rgb(R, G, B, timestamps)
"""

from __future__ import annotations

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


# Colour per crop mode (BGR, for OpenCV)
_MODE_COLOURS: dict[str, tuple[int, int, int]] = {
    "manual":         (0,   255,   0),   # green
    "face_track":     (255, 165,   0),   # orange
    "bbox_forehead":  (255,   0,   0),   # blue
    "mesh_forehead":  (0,     0, 255),   # red
    "poly":           (0,   255, 255),   # yellow
}


class DebugVisualiser:
    """
    Collection of debug / diagnostic helpers.

    Parameters
    ----------
    default_colour : tuple[int, int, int]
        BGR fallback colour when the crop mode is not in ``_MODE_COLOURS``.
    """

    def __init__(
        self,
        default_colour: tuple[int, int, int] = (255, 255, 255),
    ) -> None:
        self._default_colour = default_colour

    # ------------------------------------------------------------------
    # Frame-level helpers
    # ------------------------------------------------------------------

    def draw_overlay(
        self,
        frame: np.ndarray,
        crop_mode: str,
        crop_coords: tuple[int, int, int, int] | None = None,
        landmarks: list[tuple[int, int]] | None = None,
    ) -> np.ndarray:
        """
        Return a copy of *frame* (BGR) annotated with the ROI rectangle
        and optional landmark dots.

        Parameters
        ----------
        frame : np.ndarray
            Original BGR frame from OpenCV.
        crop_mode : str
            Used to pick the rectangle colour.
        crop_coords : (x1, y1, x2, y2) | None
        landmarks : list[(px, py)] | None

        Returns
        -------
        np.ndarray
            Annotated BGR frame (copy, not in-place).
        """
        out = frame.copy()

        if crop_coords is not None:
            x1, y1, x2, y2 = (int(v) for v in crop_coords)
            colour = _MODE_COLOURS.get(crop_mode, self._default_colour)
            cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)

        if landmarks:
            for px, py in landmarks:
                cv2.circle(out, (int(px), int(py)), 1, (0, 255, 255), -1)

        return out

    # ------------------------------------------------------------------
    # Signal diagnostics
    # ------------------------------------------------------------------

    def plot_rgb(
        self,
        R: np.ndarray,
        G: np.ndarray,
        B: np.ndarray,
        timestamps: np.ndarray,
        title: str = "RGB traces",
    ) -> None:
        """
        Plot all three channels against time and block until the window
        is closed.
        """
        fig, axs = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
        axs[0].plot(timestamps, R, color="red");   axs[0].set_ylabel("Red")
        axs[1].plot(timestamps, G, color="green"); axs[1].set_ylabel("Green")
        axs[2].plot(timestamps, B, color="blue");  axs[2].set_ylabel("Blue")
        axs[2].set_xlabel("Time (s)")
        fig.suptitle(title)
        plt.tight_layout()
        plt.show()

    def plot_rgb_with_peaks(
        self,
        R: np.ndarray,
        G: np.ndarray,
        B: np.ndarray,
        prominence: float = 5.0,
        distance: int = 30,
    ) -> None:
        """
        Plot R, G, B with auto-detected peaks marked.

        Parameters
        ----------
        prominence, distance
            Forwarded to :func:`scipy.signal.find_peaks`.
        """
        peaks_R, _ = find_peaks(R, prominence=prominence, distance=distance)
        peaks_G, _ = find_peaks(G, prominence=prominence, distance=distance)
        peaks_B, _ = find_peaks(B, prominence=prominence, distance=distance)

        fig, axes = plt.subplots(3, 1, figsize=(15, 6), sharex=True)
        axes[0].plot(R, "r"); axes[0].plot(peaks_R, R[peaks_R], "ko"); axes[0].set_ylabel("Red")
        axes[1].plot(G, "g"); axes[1].plot(peaks_G, G[peaks_G], "ko"); axes[1].set_ylabel("Green")
        axes[2].plot(B, "b"); axes[2].plot(peaks_B, B[peaks_B], "ko"); axes[2].set_ylabel("Blue")
        axes[2].set_xlabel("Frame")
        plt.tight_layout()
        plt.show()

    def snr_report(
        self,
        G_raw: np.ndarray,
        G_processed: np.ndarray,
        prominence: float = 5.0,
    ) -> dict[str, float]:
        """
        Compute and print SNR for raw vs. processed green channel.

        Returns
        -------
        dict with keys ``"raw"`` and ``"processed"``.
        """
        def _snr(trace: np.ndarray, peaks: np.ndarray) -> float:
            baseline = np.delete(trace, peaks)
            return float(np.mean(trace[peaks]) / np.std(baseline))

        peaks_raw,  _ = find_peaks(G_raw,       prominence=prominence)
        peaks_proc, _ = find_peaks(G_processed,  prominence=prominence)

        result = {
            "raw":       _snr(G_raw,       peaks_raw),
            "processed": _snr(G_processed,  peaks_proc),
        }
        print(f"SNR raw:       {result['raw']:.3f}")
        print(f"SNR processed: {result['processed']:.3f}")
        return result
