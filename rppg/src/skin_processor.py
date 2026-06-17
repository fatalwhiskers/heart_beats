"""
skin_processor.py
-----------------
Skin segmentation and masked mean-RGB extraction.

All functions operate on HxWx3 uint8 RGB images.

Typical usage
-------------
>>> proc = SkinMaskProcessor()
>>> R, G, B, mask = proc.mean_rgb(crop_rgb)
"""

from __future__ import annotations

import cv2
import numpy as np


class SkinMaskProcessor:
    """
    Extracts mean R/G/B values from a crop using a YCrCb skin mask.

    Parameters
    ----------
    equalize_y : bool
        If True, histogram-equalise the Y (luma) channel before sampling
        to reduce illumination variance.
    min_coverage : float
        Minimum fraction of pixels that must pass the skin test.
        Falls back to the full crop if coverage is below this threshold.
    morph : bool
        Apply morphological open/close to the skin mask to remove noise.
    """

    # YCrCb skin-colour bounds (Cr 133–173, Cb 77–127)
    _LOWER = (0,   133,  77)
    _UPPER = (255, 173, 127)

    def __init__(
        self,
        equalize_y: bool = True,
        min_coverage: float = 0.08,
        morph: bool = True,
    ) -> None:
        self.equalize_y   = equalize_y
        self.min_coverage = min_coverage
        self.morph        = morph

        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def mean_rgb(
        self,
        rgb: np.ndarray,
    ) -> tuple[float, float, float, np.ndarray]:
        """
        Compute mean R, G, B inside the skin mask for *rgb*.

        Parameters
        ----------
        rgb : np.ndarray
            HxWx3 uint8 RGB crop.

        Returns
        -------
        R_mean, G_mean, B_mean : float
        mask : np.ndarray
            The uint8 (0/255) skin mask that was applied.
        """
        rgb = self._ensure_uint8(rgb)
        mask = self._build_mask(rgb)

        # Illumination normalisation (equalise Y in YCrCb)
        img = self._equalize(rgb) if self.equalize_y else rgb

        # Fallback to full ROI if skin coverage is too sparse
        if mask.sum() < self.min_coverage * mask.size * 255:
            mask = np.full_like(mask, 255)

        sel = mask > 0
        R = float(img[..., 0][sel].mean())
        G = float(img[..., 1][sel].mean())
        B = float(img[..., 2][sel].mean())
        return R, G, B, mask

    def masked_mean_from_mask(
        self,
        rgb: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[float, float, float] | None:
        """
        Compute mean R, G, B from a pre-built *mask* (e.g. polygon mask).

        Drops pixels that are very dark (max channel < 10) or near-saturated
        (max channel > 250) for robustness to shadows and highlights.

        Returns ``None`` if no valid pixels remain.
        """
        sel = (mask > 0) if mask.dtype != bool else mask
        if not np.any(sel):
            return None

        roi = rgb[sel]
        v   = roi.max(axis=1)
        keep = (v > 10) & (v < 250)
        if not np.any(keep):
            return None

        roi = roi[keep]
        return float(roi[:, 0].mean()), float(roi[:, 1].mean()), float(roi[:, 2].mean())

    def combine_masks(self, *masks: np.ndarray | None) -> np.ndarray | None:
        """
        Bitwise-AND all non-None masks together.

        Returns ``None`` if every mask is ``None``.
        """
        valid = [m for m in masks if m is not None]
        if not valid:
            return None
        out = valid[0].copy()
        for m in valid[1:]:
            out = cv2.bitwise_and(out, m)
        return out

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_mask(self, rgb: np.ndarray) -> np.ndarray:
        ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
        mask  = cv2.inRange(ycrcb, self._LOWER, self._UPPER)
        if self.morph:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self._kernel, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel, iterations=2)
        return mask

    @staticmethod
    def _equalize(rgb: np.ndarray) -> np.ndarray:
        ycrcb      = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
        Y, Cr, Cb  = cv2.split(ycrcb)
        return cv2.cvtColor(cv2.merge([Y, Cr, Cb]), cv2.COLOR_YCrCb2RGB)

    @staticmethod
    def _ensure_uint8(rgb: np.ndarray) -> np.ndarray:
        if rgb.dtype != np.uint8:
            return np.clip(rgb, 0, 255).astype(np.uint8)
        return rgb
