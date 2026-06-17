import cv2
import numpy as np
from typing import Optional, Tuple, Dict

# \cite{Kakumanu2007} https://doi.org/10.1016/j.patcog.2006.06.010
# De Haan, G., & Jeanne, V. (2013). "Robust pulse rate from chrominance-based rPPG." IEEE Trans. Biomedical Engineering, 60(10), 2878–2886.
# McDuff, D., Estepp, J. R., Piasecki, A. M., & Blackford, E. B. (2015). "A survey of remote optical photoplethysmographic imaging methods." Proc. 37th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC).


def get_skin_means(frame: np.ndarray, cfg) -> Optional[Tuple[float, float, float]]:
    """
    Try to compute mean R,G,B only over skin pixels (HSV mask).
    Returns (R,G,B) if mask is valid, else None.

    Parameters
    ----------
    frame : np.ndarray (H,W,3) BGR image
    cfg   : config.Signal with thresholds (ROI_MIN_FRAC, ROI_MAX_FRAC, etc.)
    """
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Two hue ranges because skin hue wraps around red (0 and 180 in OpenCV HSV)
    lower_red_range_1 = np.array([0,   30,  40], dtype=np.uint8)
    upper_red_range_1 = np.array([25, 180, 255], dtype=np.uint8)
    lower_red_range_2 = np.array([160, 30,  40], dtype=np.uint8)
    upper_red_range_2 = np.array([179, 180, 255], dtype=np.uint8)

    mask_range_1 = cv2.inRange(hsv_image, lower_red_range_1, upper_red_range_1)
    mask_range_2 = cv2.inRange(hsv_image, lower_red_range_2, upper_red_range_2)
    skin_mask = cv2.bitwise_or(mask_range_1, mask_range_2)

    # Morphological cleanup
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, morph_kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, morph_kernel)

    # Coverage check
    skin_pixel_count = int(np.count_nonzero(skin_mask))
    coverage_fraction = skin_pixel_count / float(frame.shape[0] * frame.shape[1])
    if coverage_fraction < cfg.ROI_MIN_FRAC or coverage_fraction > cfg.ROI_MAX_FRAC:
        return None

    # Compute means over mask (frame is BGR; return R,G,B in that order to match docstring)
    mean_B = cv2.mean(frame[:, :, 0], mask=skin_mask)[0]
    mean_G = cv2.mean(frame[:, :, 1], mask=skin_mask)[0]
    mean_R = cv2.mean(frame[:, :, 2], mask=skin_mask)[0]
    return mean_R, mean_G, mean_B


# ----------------------- Agreement measures -----------------------

def _kendalls_tau_from_counts(
    count_00: np.ndarray,
    count_01: np.ndarray,
    count_10: np.ndarray,
    count_11: np.ndarray
) -> np.ndarray:
    """Kendall's tau-b for 2x2 counts."""
    total_x0 = (count_00 + count_01)  # X=0 total
    total_x1 = (count_10 + count_11)  # X=1 total
    total_y0 = (count_00 + count_10)  # Y*=0 total
    total_y1 = (count_01 + count_11)  # Y*=1 total

    denom = (total_x0 * total_x1 * total_y0 * total_y1).astype(np.float64)
    numer = (count_00 * count_11 - count_01 * count_10).astype(np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        tau = np.where(denom > 0, numer / np.sqrt(denom), -np.inf)
    return tau


# ----------------------- Core optimizer -----------------------

def _maximize_bounds_for_signal(
    reference_binary: np.ndarray,
    signal_u8: np.ndarray
) -> Tuple[Tuple[int, int], float]:
    """
    Given binary reference X (0/1) and signal Y (uint8 0..255),
    find [i, j] that maximize agreement (Kendall's tau).
    """
    assert reference_binary.shape == signal_u8.shape
    x_ref = reference_binary.astype(np.uint8).ravel()
    y_signal = signal_u8.ravel()

    # Histograms of signal values conditioned on X
    hist_y_given_x1 = np.bincount(y_signal[x_ref == 1], minlength=256).astype(np.int64)
    hist_y_given_x0 = np.bincount(y_signal[x_ref == 0], minlength=256).astype(np.int64)

    cumsum_x1 = np.cumsum(hist_y_given_x1)
    cumsum_x0 = np.cumsum(hist_y_given_x0)
    total_x1 = cumsum_x1[-1]
    total_x0 = cumsum_x0[-1]

    best_tau = -np.inf
    best_bounds = (0, 255)

    # Evaluate all i..j using prefix sums (vectorized over j for each i)
    for lower_bound in range(256):
        base_x1 = cumsum_x1[lower_bound - 1] if lower_bound > 0 else 0
        base_x0 = cumsum_x0[lower_bound - 1] if lower_bound > 0 else 0

        c11 = cumsum_x1[lower_bound:] - base_x1  # X=1 and Y in [lower_bound..j]
        c01 = cumsum_x0[lower_bound:] - base_x0  # X=0 and Y in [lower_bound..j]
        c10 = total_x1 - c11
        c00 = total_x0 - c01

        tau_vals = _kendalls_tau_from_counts(c00, c01, c10, c11)
        best_offset = int(np.argmax(tau_vals))
        candidate_tau = float(tau_vals[best_offset])

        if candidate_tau > best_tau:
            best_tau = candidate_tau
            best_bounds = (lower_bound, lower_bound + best_offset)

    return best_bounds, best_tau


# ----------------------- Utils -----------------------

def _shift_hue_center_red(hue_u8: np.ndarray) -> np.ndarray:
    """
    OpenCV hue is 0..179. Map to 0..255 and rotate so red is centered.
    """
    hue_255 = (hue_u8.astype(np.uint16) * 255 // 179).astype(np.uint8)
    return ((hue_255.astype(np.int16) + 128) % 256).astype(np.uint8)


def _in_bounds(values_u8: np.ndarray, lower: int, upper: int) -> np.ndarray:
    return (values_u8 >= lower) & (values_u8 <= upper)


# ----------------------- Public: detect_skin_conaire -----------------------

def detect_skin_conaire(
    bgr: np.ndarray,
    max_iters: int = 5,
    init: str = "color"
) -> Tuple[np.ndarray, Dict[str, Tuple[int, int]]]:
    """
    Conaire et al. agreement-maximization skin detector.
    Args:
      bgr: uint8 HxWx3 (OpenCV BGR).
      max_iters: outer iterations of alternating optimization.
      init: "color" (broad HSV init). (Kept for API similarity; no IR path here.)
    Returns:
      mask_bool: HxW boolean skin mask
      params:    dict of learned bounds: {"H":(l1,l2), "S":(l3,l4), "V":(l5,l6)}
    """
    if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.dtype != np.uint8:
        raise ValueError("bgr must be a uint8 HxWx3 OpenCV image")

    hsv_image = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue_centered = _shift_hue_center_red(hsv_image[..., 0])  # 0..255, red-centered
    sat = hsv_image[..., 1]
    val = hsv_image[..., 2]

    # Initialize bounds (broad)
    initial_margin = 255 // 5
    hue_lower, hue_upper = 128 - initial_margin, 128 + initial_margin
    sat_lower, sat_upper = initial_margin, 255
    val_lower, val_upper = initial_margin, 255

    def _visual_mask() -> np.ndarray:
        return (
            _in_bounds(hue_centered, hue_lower, hue_upper)
            & _in_bounds(sat, sat_lower, sat_upper)
            & _in_bounds(val, val_lower, val_upper)
        )

    for _iter in range(max_iters):
        changed = False

        # Optimize H given S,V
        reference_mask = _in_bounds(sat, sat_lower, sat_upper) & _in_bounds(val, val_lower, val_upper)
        (opt_h_lo, opt_h_hi), _ = _maximize_bounds_for_signal(reference_mask.astype(np.uint8), hue_centered)
        if (opt_h_lo, opt_h_hi) != (hue_lower, hue_upper):
            hue_lower, hue_upper = opt_h_lo, opt_h_hi
            changed = True

        # Optimize S given H,V
        reference_mask = _in_bounds(hue_centered, hue_lower, hue_upper) & _in_bounds(val, val_lower, val_upper)
        (opt_s_lo, opt_s_hi), _ = _maximize_bounds_for_signal(reference_mask.astype(np.uint8), sat)
        if (opt_s_lo, opt_s_hi) != (sat_lower, sat_upper):
            sat_lower, sat_upper = opt_s_lo, opt_s_hi
            changed = True

        # Optimize V given H,S
        reference_mask = _in_bounds(hue_centered, hue_lower, hue_upper) & _in_bounds(sat, sat_lower, sat_upper)
        (opt_v_lo, opt_v_hi), _ = _maximize_bounds_for_signal(reference_mask.astype(np.uint8), val)
        if (opt_v_lo, opt_v_hi) != (val_lower, val_upper):
            val_lower, val_upper = opt_v_lo, opt_v_hi
            changed = True

        # (Optional early stop): if not changed, you could break; left as-is to match original behavior.
        # if not changed:
        #     break

    visual_mask = _visual_mask()
    final_mask = visual_mask

    params = {
        "H": (int(hue_lower), int(hue_upper)),
        "S": (int(sat_lower), int(sat_upper)),
        "V": (int(val_lower), int(val_upper)),
    }

    return final_mask.astype(bool), params


# ----------------------- Apply cached params (no re-learning) -----------------------

def mask_bgr_with_params(bgr: np.ndarray, params: Dict[str, Tuple[int, int]]) -> np.ndarray:
    """Apply learned HSV bounds to a new BGR image/crop."""
    hsv_image = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue_centered = _shift_hue_center_red(hsv_image[..., 0])
    sat = hsv_image[..., 1]
    val = hsv_image[..., 2]

    hue_lower, hue_upper = params["H"]
    sat_lower, sat_upper = params["S"]
    val_lower, val_upper = params["V"]

    return (
        _in_bounds(hue_centered, hue_lower, hue_upper)
        & _in_bounds(sat, sat_lower, sat_upper)
        & _in_bounds(val, val_lower, val_upper)
    )


# ----------------------- rPPG-friendly wrapper -----------------------

class ConaireSkin:
    """
    Learn Conaire HSV bounds on an ROI every `update_every_frames` frames,
    otherwise reuse cached bounds for stability and speed.
    """
    def __init__(self, update_every_frames: int = 60, max_iters: int = 4, min_skin_px: int = 200):
        self.update_every = int(max(1, update_every_frames))
        self.max_iters = int(max_iters)
        self.min_skin_px = int(min_skin_px)
        self.params: Optional[Dict[str, Tuple[int, int]]] = None
        self._frame_counter = 0
        self._last_mask: Optional[np.ndarray] = None

    def mask(self, bgr_crop: np.ndarray) -> np.ndarray:
        """
        Returns a boolean skin mask for the given crop.
        Relearns bounds periodically; otherwise applies cached params.
        """
        self._frame_counter += 1
        should_learn_now = (self.params is None) or (self._frame_counter % self.update_every == 0)

        if should_learn_now:
            learned_mask, learned_params = detect_skin_conaire(bgr_crop, max_iters=self.max_iters, init="color")
            # Avoid degenerate updates (no skin found)
            if learned_mask.sum() >= self.min_skin_px:
                self.params = learned_params
                self._last_mask = learned_mask
                return learned_mask
            # If degenerate, fall through and reuse last good params (if any)

        if self.params is not None:
            applied_mask = mask_bgr_with_params(bgr_crop, self.params)
            self._last_mask = applied_mask
            return applied_mask

        # No params yet and learning failed: fallback to simple HSV gate
        hsv_image = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
        hue, sat, val = hsv_image[..., 0], hsv_image[..., 1], hsv_image[..., 2]
        fallback_mask = (((hue < 25) | (hue > 160)) & (sat > 40) & (val > 50))
        self._last_mask = fallback_mask
        return fallback_mask
