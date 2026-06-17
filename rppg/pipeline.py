"""
pipeline.py
-----------
Main extraction pipeline.  Owns the OpenCV capture loop and delegates
every specialist concern to the appropriate module.

Typical usage
-------------
>>> extractor = VideoRGBExtractor(crop_mode="bbox_forehead")
>>> R, G, B, t = extractor.extract("path/to/video.mp4")
"""

from __future__ import annotations

import os
import cv2
import numpy as np

from rppg.src.face_detection   import FaceDetectorV2, FaceLandmarkerV2
from rppg.src.roi_extractor    import ROIExtractor, ROIResult
from rppg.src.skin_processor   import SkinMaskProcessor
from rppg.src.signal_cleaner   import SignalCleaner
from rppg.src.debug_visualiser import DebugVisualiser
from src.config                import Video


# Modes that need a face detector
_DETECTOR_MODES   = {"face_track", "bbox_forehead"}
# Modes that need a landmarker
_LANDMARKER_MODES = {"mesh_forehead", "poly"}


class VideoRGBExtractor:
    """
    Extract mean R, G, B traces from a video file.

    Parameters
    ----------
    crop_mode : str
        One of the modes supported by :class:`~src.roi_extractor.ROIExtractor`.
    x1, y1, x2, y2 : int
        Manual crop box (only used when ``crop_mode='manual'``).
    display : bool
        Show live OpenCV windows while processing.
    testing : bool
        Cap at 30 s of video; save debug frames.
    test_output_dir : str
        Directory for test debug frames.
    interpolate : bool
        Passed to :class:`~src.signal_cleaner.SignalCleaner`.
    apply_hampel : bool
        Run Hampel outlier removal after gap-filling.
    """

    def __init__(
        self,
        crop_mode: str = "bbox_forehead",
        x1: int = 0, y1: int = 0, x2: int = 0, y2: int = 0,
        display: bool = False,
        testing: bool = False,
        test_output_dir: str = r"outputs\test_frames",
        interpolate: bool = True,
        apply_hampel: bool = True,
    ) -> None:
        self.crop_mode       = crop_mode
        self.display         = display
        self.testing         = testing
        self.test_output_dir = test_output_dir

        self._roi     = ROIExtractor(crop_mode, x1, y1, x2, y2)
        self._skin    = SkinMaskProcessor()
        self._cleaner = SignalCleaner(
            interpolate=interpolate,
            hampel_win_sec=0.5 if apply_hampel else 0.0,
        )
        self._vis = DebugVisualiser()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        video_path: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Process *video_path* and return cleaned RGB traces.

        Parameters
        ----------
        video_path : str
            Path to the video file.

        Returns
        -------
        R, G, B : np.ndarray
            Cleaned mean channel traces on a uniform time grid.
        t : np.ndarray
            Corresponding timestamps in seconds.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video: {video_path}")

        Rm, Gm, Bm, timestamps = [], [], [], []

        if self.display:
            cv2.namedWindow("full",  cv2.WINDOW_NORMAL)
            cv2.namedWindow("crop",  cv2.WINDOW_NORMAL)

        if self.testing:
            os.makedirs(self.test_output_dir, exist_ok=True)

        detector   = FaceDetectorV2()   if self.crop_mode in _DETECTOR_MODES   else None
        landmarker = FaceLandmarkerV2() if self.crop_mode in _LANDMARKER_MODES else None

        try:
            frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                ts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                h, w  = rgb.shape[:2]

                # --- run the appropriate detector ---
                det_result = lm_result = None
                if detector   is not None:
                    det_result = detector.process(rgb, ts_ms)
                if landmarker is not None:
                    lm_result  = landmarker.process(rgb, ts_ms)

                # --- get ROI ---
                result = self._roi.extract(
                    rgb, ts_ms,
                    detector_result=det_result,
                    landmarker_result=lm_result,
                )

                # --- sample colour ---
                R_val, G_val, B_val = self._sample_colour(rgb, result)
                Rm.append(R_val); Gm.append(G_val); Bm.append(B_val)
                timestamps.append(ts_ms / 1000.0)

                # --- optional display ---
                if self.display:
                    self._show(frame, rgb, result)

                # --- optional debug frame save ---
                if self.testing and frame_count == 0:
                    self._save_debug_frame(frame, video_path, result)

                frame_count += 1
                if self.testing and frame_count >= Video.FPS * 30:
                    break

        finally:
            if detector   is not None: detector.close()
            if landmarker is not None: landmarker.close()
            cap.release()
            if self.display:
                cv2.destroyAllWindows()

        return self._cleaner.clean(Rm, Gm, Bm, timestamps)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _sample_colour(
        self,
        rgb: np.ndarray,
        result: ROIResult,
    ) -> tuple[float, float, float]:
        """Return (R, G, B) for this frame; NaN if detection failed."""
        nan3 = (np.nan, np.nan, np.nan)

        # poly mode: mask already computed in ROIResult
        if self.crop_mode == "poly":
            if not result.detected or result.mask is None:
                return nan3
            skin_mask = self._skin.combine_masks(result.mask,
                                                  self._skin._build_mask(rgb))
            if skin_mask is None or not np.any(skin_mask):
                return nan3
            vals = self._skin.masked_mean_from_mask(rgb, skin_mask)
            return vals if vals is not None else nan3

        # bbox / mesh / manual modes: crop then skin-mask
        if not result.detected:
            # ask Kalman for fallback position
            fb = self._roi.kalman_fallback()
            if fb is None:
                return nan3
            result = fb

        crop = rgb[result.y1:result.y2, result.x1:result.x2]
        if crop.size == 0:
            return nan3

        R, G, B, _ = self._skin.mean_rgb(crop)
        return R, G, B

    def _show(
        self,
        bgr_frame: np.ndarray,
        rgb: np.ndarray,
        result: ROIResult,
    ) -> None:
        box = result.crop if result.detected else None
        annotated = self._vis.draw_overlay(
            bgr_frame, self.crop_mode, box, result.landmarks or None
        )
        cv2.imshow("full", annotated)

        if result.detected:
            crop_bgr = cv2.cvtColor(
                rgb[result.y1:result.y2, result.x1:result.x2],
                cv2.COLOR_RGB2BGR,
            )
            if crop_bgr.size:
                cv2.imshow("crop", crop_bgr)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            raise KeyboardInterrupt("User quit")

    def _save_debug_frame(
        self,
        bgr_frame: np.ndarray,
        video_path: str,
        result: ROIResult,
    ) -> None:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        mode_dir   = os.path.join(self.test_output_dir, video_name, self.crop_mode)
        os.makedirs(mode_dir, exist_ok=True)

        box       = result.crop if result.detected else None
        annotated = self._vis.draw_overlay(
            bgr_frame, self.crop_mode, box, result.landmarks or None
        )
        save_path = os.path.join(mode_dir, f"{self.crop_mode}_frame_000_{video_name}.jpg")
        cv2.imwrite(save_path, annotated)
