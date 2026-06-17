"""
signal_cleaner.py
-----------------
Post-processing for raw R/G/B time-series extracted from video.

Pipeline (applied in :meth:`SignalCleaner.clean`)
--------------------------------------------------
1. PCHIP resample onto a uniform time grid  (optional)
2. Fill short NaN gaps; drop long ones       (via extract_wave helpers)
3. Hampel impulse filter                     (optional)

Typical usage
-------------
>>> cleaner = SignalCleaner()
>>> R, G, B, t = cleaner.clean(Rm_raw, Gm_raw, Bm_raw, timestamps)
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter

import src.extract_wave as ext


class SignalCleaner:
    """
    Cleans raw per-frame R, G, B mean traces.

    Parameters
    ----------
    interpolate : bool
        Resample onto a uniform grid using PCHIP before gap-filling.
        Set ``False`` to keep the raw (possibly irregular) timestamps.
    hampel_win_sec : float
        Half-window size (in seconds) for the Hampel outlier filter.
        Pass 0 to skip the Hampel step entirely.
    hampel_sigma : float
        Number of scaled-MAD units that define an outlier.
    fps : float
        Frame rate, used only when ``interpolate=False`` and Hampel is active.
    """

    def __init__(
        self,
        interpolate: bool = True,
        hampel_win_sec: float = 0.5,
        hampel_sigma: float = 5.0,
        fps: float = 30.0,
    ) -> None:
        self.interpolate    = interpolate
        self.hampel_win_sec = hampel_win_sec
        self.hampel_sigma   = hampel_sigma
        self.fps            = fps

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def clean(
        self,
        R: list | np.ndarray,
        G: list | np.ndarray,
        B: list | np.ndarray,
        timestamps: list | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Full cleaning pipeline.

        Parameters
        ----------
        R, G, B : array-like
            Raw per-frame mean channel values (may contain NaNs).
        timestamps : array-like
            Per-frame timestamps in seconds.

        Returns
        -------
        R, G, B, t : np.ndarray
            Cleaned signals on a uniform time grid.
        """
        # Step 1: resample
        if self.interpolate:
            R, G, B, t = ext.resample_rgb_pchip(R, G, B, timestamps)
        else:
            t = np.asarray(timestamps, dtype=float)
            R = np.asarray(R, dtype=float)
            G = np.asarray(G, dtype=float)
            B = np.asarray(B, dtype=float)

        # Step 2: fill short gaps / drop long gaps
        R, t, *_ = ext.fill_short_gaps_then_drop(R, t)
        G, *_    = ext.fill_short_gaps_then_drop(G, t)
        B, *_    = ext.fill_short_gaps_then_drop(B, t)

        # Step 3: Hampel impulse removal
        if self.hampel_win_sec > 0:
            R, G, B, _ = self.repair_impulses(R, G, B)

        return np.asarray(R), np.asarray(G), np.asarray(B), np.asarray(t)

    def hampel(
        self,
        x: np.ndarray,
        win_sec: float | None = None,
        n_sigma: float | None = None,
        replace: str = "interp",
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Hampel identifier: replace impulse outliers with local median or
        linear interpolation.

        Parameters
        ----------
        x : np.ndarray
            1-D signal.
        win_sec : float | None
            Half-window in seconds.  Defaults to ``self.hampel_win_sec``.
        n_sigma : float | None
            Outlier threshold in scaled-MAD units. Defaults to
            ``self.hampel_sigma``.
        replace : {'interp', 'median', 'none'}
            Replacement strategy.

        Returns
        -------
        x_fixed : np.ndarray
        outlier_mask : np.ndarray[bool]
        """
        win_sec = win_sec if win_sec is not None else self.hampel_win_sec
        n_sigma = n_sigma if n_sigma is not None else self.hampel_sigma

        x = np.asarray(x, dtype=float)
        k    = max(1, int(round(win_sec * self.fps)))
        size = 2 * k + 1

        med     = median_filter(x, size=size, mode="reflect")
        abs_dev = np.abs(x - med)
        mad     = 1.4826 * median_filter(abs_dev, size=size, mode="reflect") + 1e-12
        outlier = abs_dev > n_sigma * mad

        if replace == "none":
            return x.copy(), outlier

        x_fixed = x.copy()
        if replace == "median":
            x_fixed[outlier] = med[outlier]
            return x_fixed, outlier

        # replace == 'interp'
        if outlier.any():
            good = ~outlier
            idx  = np.arange(len(x))
            x_fixed[outlier] = np.interp(idx[outlier], idx[good], x_fixed[good])
        return x_fixed, outlier

    def repair_impulses(
        self,
        R: np.ndarray,
        G: np.ndarray,
        B: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply Hampel filter to all three channels; pixels flagged as outliers
        in *any* channel are re-interpolated across all three for consistency.

        Returns
        -------
        Rf, Gf, Bf : np.ndarray
            Cleaned channels.
        bad : np.ndarray[bool]
            Union outlier mask.
        """
        Rf, mR = self.hampel(R)
        Gf, mG = self.hampel(G)
        Bf, mB = self.hampel(B)

        bad = mR | mG | mB
        if bad.any():
            idx  = np.arange(len(R))
            good = ~bad
            for arr in (Rf, Gf, Bf):
                arr[bad] = np.interp(idx[bad], idx[good], arr[good])
        return Rf, Gf, Bf, bad
