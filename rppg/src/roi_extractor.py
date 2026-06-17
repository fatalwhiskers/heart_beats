"""
roi_extractor.py
----------------
Converts raw face detection / landmark results into a smoothed pixel-domain
bounding box (or polygon mask) for a given crop mode.

Supported crop modes
--------------------
``manual``            Fixed (x1,y1,x2,y2) supplied at construction time.
``none``              Full frame — no cropping.
``face_track``        Full face bbox, Kalman-smoothed.
``bbox_forehead``     Forehead band from detector bbox, Kalman-smoothed.
``mesh_forehead``     Forehead band from 478-pt landmark mesh, Kalman-smoothed.
``poly``              Polygon union (forehead + cheeks) — returns mask, not bbox.

Typical usage
-------------
>>> extractor = ROIExtractor(crop_mode="bbox_forehead")
>>> result    = extractor.extract(frame_rgb, ts_ms,
...                               detector_result=det_result)
>>> if result.detected:
...     crop = frame_rgb[result.y1:result.y2, result.x1:result.x2]
"""

from __future__ import annotations

import numpy as np
import cv2
from dataclasses import dataclass, field

import src.smoother as smo

# ---------------------------------------------------------------------------
# Landmark index sets
# ---------------------------------------------------------------------------
FOREHEAD_IDX = [107, 66, 69, 109, 338, 299, 296, 336]
L_CHEEK_IDX  = [118, 119, 100, 126, 209, 49, 129, 203, 205, 50]
R_CHEEK_IDX  = [347, 348, 329, 355, 429, 279, 358, 423, 425, 280]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class ROIResult:
    """Return value from :meth:`ROIExtractor.extract`."""
    detected: bool = False
    x1: int = 0
    y1: int = 0
    x2: int = 0
    y2: int = 0
    mask: np.ndarray | None = None          # only set for crop_mode='poly'
    landmarks: list[tuple[int, int]] = field(default_factory=list)

    @property
    def crop(self):
        """Convenience: return (x1, y1, x2, y2) as a tuple."""
        return self.x1, self.y1, self.x2, self.y2


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------
class ROIExtractor:
    """
    Converts detector / landmarker results into a smoothed ROI for one frame.

    Parameters
    ----------
    crop_mode : str
        One of the supported mode strings (see module docstring).
    x1, y1, x2, y2 : int
        Used only when ``crop_mode='manual'``.
    """

    SUPPORTED_MODES = frozenset({
        "manual", "none",
        "face_track", "bbox_forehead", "mesh_forehead", "poly",
    })

    def __init__(
        self,
        crop_mode: str = "bbox_forehead",
        x1: int = 0, y1: int = 0, x2: int = 0, y2: int = 0,
    ) -> None:
        if crop_mode not in self.SUPPORTED_MODES:
            raise ValueError(
                f"Unknown crop_mode '{crop_mode}'. "
                f"Choose from {sorted(self.SUPPORTED_MODES)}"
            )
        self.crop_mode = crop_mode
        self._manual_box = (x1, y1, x2, y2)
        self._kf = smo.ROIKalman()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        rgb: np.ndarray,
        ts_ms: float,
        detector_result=None,
        landmarker_result=None,
    ) -> ROIResult:
        """
        Compute the ROI for a single frame.

        Parameters
        ----------
        rgb : np.ndarray
            HxWx3 uint8 RGB image.
        ts_ms : float
            Frame timestamp in milliseconds.
        detector_result :
            Output of :meth:`FaceDetectorV2.process`, required for
            ``face_track`` and ``bbox_forehead`` modes.
        landmarker_result :
            Output of :meth:`FaceLandmarkerV2.process`, required for
            ``mesh_forehead`` and ``poly`` modes.

        Returns
        -------
        ROIResult
        """
        mode = self.crop_mode

        if mode == "manual":
            return self._manual(rgb)
        if mode == "none":
            return ROIResult(detected=True, x1=0, y1=0,
                             x2=rgb.shape[1], y2=rgb.shape[0])
        if mode == "face_track":
            return self._face_track(rgb, detector_result)
        if mode == "bbox_forehead":
            return self._bbox_forehead(rgb, detector_result)
        if mode == "mesh_forehead":
            return self._mesh_forehead(rgb, landmarker_result)
        if mode == "poly":
            return self._poly(rgb, landmarker_result)

        # Should never reach here given __init__ validation
        raise ValueError(f"Unhandled crop_mode: {mode}")

    def kalman_fallback(self) -> ROIResult | None:
        """
        Ask the Kalman filter for its last known position (used when detection
        fails for a frame).

        Returns ``None`` if there is no prior estimate.
        """
        box = self._kf.update(None, conf=0.0)
        if box is None:
            return None
        x1, y1, x2, y2 = box
        return ROIResult(detected=True, x1=int(x1), y1=int(y1),
                         x2=int(x2), y2=int(y2))

    # ------------------------------------------------------------------
    # Private per-mode helpers
    # ------------------------------------------------------------------

    def _manual(self, rgb: np.ndarray) -> ROIResult:
        x1, y1, x2, y2 = self._manual_box
        return ROIResult(detected=True, x1=x1, y1=y1, x2=x2, y2=y2)

    def _face_track(self, rgb: np.ndarray, det_result) -> ROIResult:
        if not (det_result and det_result.detections):
            return ROIResult(detected=False)

        det  = det_result.detections[0]
        bbox = det.bounding_box
        h, w = rgb.shape[:2]

        raw = (int(bbox.origin_x), int(bbox.origin_y),
               int(bbox.origin_x + bbox.width),
               int(bbox.origin_y + bbox.height))

        conf = _det_confidence(det)
        kx1, ky1, kx2, ky2 = self._kf.update(raw, conf=conf)

        margin_x = int(bbox.width  * 0.1)
        margin_y = int(bbox.height * 0.2)
        x1 = max(0, kx1 - margin_x);  x2 = min(w, kx2 + margin_x)
        y1 = max(0, ky1 - margin_y);  y2 = min(h, ky2 + margin_y)
        return ROIResult(detected=True, x1=x1, y1=y1, x2=x2, y2=y2)

    def _bbox_forehead(self, rgb: np.ndarray, det_result) -> ROIResult:
        if not (det_result and det_result.detections):
            return ROIResult(detected=False)

        det  = det_result.detections[0]
        bbox = det.bounding_box
        h, w = rgb.shape[:2]

        raw = _forehead_box_from_bbox(rgb, bbox)
        conf = _det_confidence(det)
        kx1, ky1, kx2, ky2 = self._kf.update(raw, conf=conf)

        x1 = int(np.clip(kx1, 0, w - 1));  x2 = int(np.clip(kx2, 0, w - 1))
        y1 = int(np.clip(ky1, 0, h - 1));  y2 = int(np.clip(ky2, 0, h - 1))

        if x2 <= x1 or y2 <= y1:
            return ROIResult(detected=False)
        return ROIResult(detected=True, x1=x1, y1=y1, x2=x2, y2=y2)

    def _mesh_forehead(self, rgb: np.ndarray, lm_result) -> ROIResult:
        if not (lm_result and lm_result.face_landmarks):
            return ROIResult(detected=False)

        lms  = lm_result.face_landmarks[0]
        h, w = rgb.shape[:2]

        raw = _forehead_box_from_landmarks(rgb, lms)
        conf = 1.0
        kx1, ky1, kx2, ky2 = self._kf.update(raw, conf=conf)

        landmarks = [(int(pt.x * w), int(pt.y * h)) for pt in lms]

        x1 = int(np.clip(kx1, 0, w - 1));  x2 = int(np.clip(kx2, 0, w - 1))
        y1 = int(np.clip(ky1, 0, h - 1));  y2 = int(np.clip(ky2, 0, h - 1))

        if x2 <= x1 or y2 <= y1:
            return ROIResult(detected=False, landmarks=landmarks)
        return ROIResult(detected=True, x1=x1, y1=y1, x2=x2, y2=y2,
                         landmarks=landmarks)

    def _poly(self, rgb: np.ndarray, lm_result) -> ROIResult:
        if not (lm_result and lm_result.face_landmarks):
            return ROIResult(detected=False)

        lms = lm_result.face_landmarks[0]
        masks = [
            _polygon_mask(rgb.shape, lms, FOREHEAD_IDX),
            _polygon_mask(rgb.shape, lms, L_CHEEK_IDX),
            _polygon_mask(rgb.shape, lms, R_CHEEK_IDX),
        ]

        combined = None
        for m in masks:
            if m is not None:
                combined = m if combined is None else cv2.bitwise_or(combined, m)

        if combined is None or not np.any(combined):
            return ROIResult(detected=False)

        return ROIResult(detected=True, mask=combined)


# ---------------------------------------------------------------------------
# Module-level geometry helpers (pure functions)
# ---------------------------------------------------------------------------

def _det_confidence(detection) -> float:
    return float(getattr(detection, "score", [1.0])[0]) \
        if hasattr(detection, "score") else 1.0


def _forehead_box_from_bbox(frame, bbox) -> tuple[int, int, int, int]:
    """Estimate forehead band from a Tasks pixel bounding box."""
    h, w = frame.shape[:2]
    x      = int(bbox.origin_x)
    y      = int(bbox.origin_y)
    width  = int(bbox.width)
    height = int(bbox.height)

    band_h = int(0.14 * height)
    top    = y - int(0.10 * height)

    fh_y1 = max(0, top)
    fh_y2 = min(h, top + band_h)
    fh_x1 = max(0, x + int(0.10 * width))
    fh_x2 = min(w, x + int(0.90 * width))
    return fh_x1, fh_y1, fh_x2, fh_y2


def _forehead_box_from_landmarks(frame, face_landmarks) -> tuple[int, int, int, int]:
    """Tight bounding box around FOREHEAD_IDX landmarks."""
    h, w = frame.shape[:2]
    points = [
        (int(face_landmarks[i].x * w), int(face_landmarks[i].y * h))
        for i in FOREHEAD_IDX if i < len(face_landmarks)
    ]
    xs, ys = zip(*points)
    pad = 10
    return (max(0, min(xs) - pad), max(0, min(ys) - pad),
            min(w, max(xs) + pad), min(h, max(ys) + pad))


def _polygon_mask(frame_shape, face_landmarks, idx_list) -> np.ndarray | None:
    """Binary uint8 mask (0/255) from a set of landmark indices."""
    h, w = frame_shape[:2]
    pts = []
    for i in idx_list:
        if i < len(face_landmarks):
            x = int(np.clip(face_landmarks[i].x * w, 0, w - 1))
            y = int(np.clip(face_landmarks[i].y * h, 0, h - 1))
            pts.append((x, y))
    if len(pts) < 3:
        return None
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 255)
    return mask
