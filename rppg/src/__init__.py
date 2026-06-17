"""
rppg/src
--------
Public re-exports for the rPPG extraction library.
"""

from rppg.src.model_manager    import ModelManager
from rppg.src.face_detection   import FaceDetectorV2, FaceLandmarkerV2
from rppg.src.roi_extractor    import ROIExtractor, ROIResult
from rppg.src.skin_processor   import SkinMaskProcessor
from rppg.src.signal_cleaner   import SignalCleaner
from rppg.src.dataset_loader   import DatasetLoader, Dataset1Row, Dataset2Row
from rppg.src.debug_visualiser import DebugVisualiser

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
