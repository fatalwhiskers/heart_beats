"""
face_detection.py
-----------------
Thin wrappers around MediaPipe Tasks-based face detection and landmarking.
Both classes operate in VIDEO running mode (stateful, timestamp-aware).

Typical usage
-------------
>>> detector   = FaceDetectorV2()
>>> landmarker = FaceLandmarkerV2()
>>> results    = detector.process(rgb_frame, timestamp_ms=ts)
>>> lm_results = landmarker.process(rgb_frame, timestamp_ms=ts)
>>> detector.close(); landmarker.close()
"""

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from src.model_manager import ModelManager

_manager = ModelManager()


class FaceDetectorV2:
    """
    BlazeFace short-range face detector using the MediaPipe Tasks API.

    Parameters
    ----------
    model_path : str | None
        Path to ``blaze_face_short_range.tflite``.
        If *None*, the file is fetched/cached automatically via
        :class:`~src.model_manager.ModelManager`.
    min_detection_confidence : float
        Minimum confidence threshold for detections.
    """

    def __init__(
        self,
        model_path: str | None = None,
        min_detection_confidence: float = 0.7,
    ) -> None:
        if model_path is None:
            model_path = _manager.get("blaze_face_short_range.tflite")

        base = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.FaceDetectorOptions(
            base_options=base,
            running_mode=mp_vision.RunningMode.VIDEO,
            min_detection_confidence=min_detection_confidence,
        )
        self._detector = mp_vision.FaceDetector.create_from_options(options)

    def process(self, rgb_frame, timestamp_ms: float):
        """
        Run detection on a single RGB frame.

        Parameters
        ----------
        rgb_frame : np.ndarray
            HxWx3 uint8 RGB image.
        timestamp_ms : float
            Video timestamp in milliseconds (must be monotonically increasing).

        Returns
        -------
        mediapipe FaceDetectorResult
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        return self._detector.detect_for_video(mp_image, int(timestamp_ms))

    def close(self) -> None:
        """Release MediaPipe resources."""
        try:
            self._detector.close()
        except AttributeError:
            pass


class FaceLandmarkerV2:
    """
    478-point face mesh landmarker using the MediaPipe Tasks API.

    Parameters
    ----------
    model_path : str | None
        Path to ``face_landmarker.task``.
        Auto-downloaded if *None*.
    min_detection_confidence : float
    min_presence_confidence : float
    min_tracking_confidence : float
    num_faces : int
        Maximum number of faces to detect (default 1).
    """

    def __init__(
        self,
        model_path: str | None = None,
        min_detection_confidence: float = 0.7,
        min_presence_confidence: float = 0.7,
        min_tracking_confidence: float = 0.7,
        num_faces: int = 1,
    ) -> None:
        if model_path is None:
            model_path = _manager.get("face_landmarker.task")

        base = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=num_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    def process(self, rgb_frame, timestamp_ms: float):
        """
        Run landmarking on a single RGB frame.

        Parameters
        ----------
        rgb_frame : np.ndarray
            HxWx3 uint8 RGB image.
        timestamp_ms : float
            Monotonically increasing video timestamp in milliseconds.

        Returns
        -------
        mediapipe FaceLandmarkerResult
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        return self._landmarker.detect_for_video(mp_image, int(timestamp_ms))

    def close(self) -> None:
        """Release MediaPipe resources."""
        try:
            self._landmarker.close()
        except AttributeError:
            pass
