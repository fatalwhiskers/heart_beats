"""
model_manager.py
----------------
Responsible for ensuring MediaPipe model files are available locally.
Downloads from Google Storage on first use and caches under ./models/.
"""

import os
import urllib.request


MODEL_DIR = "models"

_MODEL_URLS: dict[str, str] = {
    "blaze_face_short_range.tflite": (
        "https://storage.googleapis.com/mediapipe-models/face_detector"
        "/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
    ),
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker"
        "/face_landmarker/float16/1/face_landmarker.task"
    ),
}


class ModelManager:
    """
    Ensures MediaPipe model files exist locally, downloading them if needed.

    Parameters
    ----------
    model_dir : str
        Directory where model files are stored / downloaded to.
        Defaults to ``./models``.

    Example
    -------
    >>> manager = ModelManager()
    >>> path = manager.get("face_landmarker.task")
    """

    def __init__(self, model_dir: str = MODEL_DIR) -> None:
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

    def get(self, filename: str) -> str:
        """
        Return the local path for *filename*, downloading it first if absent.

        Parameters
        ----------
        filename : str
            One of the known model filenames (see ``_MODEL_URLS``).

        Returns
        -------
        str
            Absolute-ish local path ready to pass to MediaPipe options.

        Raises
        ------
        ValueError
            If *filename* is not a recognised model name.
        """
        path = os.path.join(self.model_dir, filename)
        if not os.path.exists(path):
            self._download(filename, path)
        return path

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _download(self, filename: str, dest: str) -> None:
        url = _MODEL_URLS.get(filename)
        if url is None:
            raise ValueError(
                f"No download URL known for '{filename}'. "
                f"Known models: {list(_MODEL_URLS)}"
            )
        print(f"[ModelManager] Downloading {filename} ...")
        urllib.request.urlretrieve(url, dest)
        print(f"[ModelManager] Saved to {dest}")
