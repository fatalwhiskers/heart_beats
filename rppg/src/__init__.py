"""
rppg/src
--------
Public re-exports for the rPPG extraction library.
"""

from src.model_manager    import ModelManager
from src.face_detection   import FaceDetectorV2, FaceLandmarkerV2
from src.roi_extractor    import ROIExtractor, ROIResult
from src.skin_processor   import SkinMaskProcessor
from src.signal_cleaner   import SignalCleaner
from src.dataset_loader   import DatasetLoader, Dataset1Row, Dataset2Row
from src.debug_visualiser import DebugVisualiser

__all__ = [
    "ModelManager",
    "FaceDetectorV2",
    "FaceLandmarkerV2",
    "ROIExtractor",
    "ROIResult",
    "SkinMaskProcessor",
    "SignalCleaner",
    "DatasetLoader",
    "Dataset1Row",
    "Dataset2Row",
    "DebugVisualiser",
]
