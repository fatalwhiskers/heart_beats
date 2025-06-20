from dataclasses import dataclass

@dataclass(frozen=True)
class Paths:
    DATA_RAW: str = "data/raw"
    DATA_PROCESSED: str = "data/processed"
    MODEL_DIR: str = "models"
    OUTPUT_PLOTS: str = "outputs/plots"

@dataclass(frozen=True)
class Video:
    FPS: int = 30
    ROI_WIDTH: int = 200      # px
    ROI_HEIGHT: int = 200     # px

@dataclass(frozen=True)
class Signal:
    WINDOW_SECONDS: int = 10
    HR_LOW: float = 0.75      # Hz   (45 bpm)
    HR_HIGH: float = 3.0      # Hz  (180 bpm)