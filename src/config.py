from dataclasses import dataclass

@dataclass(frozen=False)
class Video:
    FPS: int = 35
    target_FPS: int = 35

@dataclass(frozen=False)
class rppg:
    window_size: int = 15
    step_size: int = 5

@dataclass(frozen=True)
class Signal:
    WINDOW_SECONDS: int = 10
    HR_LOW: float = 0.75      # Hz   (45 bpm)
    HR_HIGH: float = 3.0      # Hz  (180 bpm)
    HR_LOW_BPM: float = 45
    HR_HIGH_BPM: float = 180
    HR_ORDER = 7


@dataclass(frozen=True)
class filePaths:
    folder_path = r"data\Dataset1"
    csv_path = r"data\CSVFiles\Settings.csv"
    output_path = r"outputs"  

@dataclass(frozen=True)
class BVP:
    BVP_RATE: int = 64

@dataclass(frozen=True)
class PRV:
    PROM_THRESHOLD: float = 0.3
    MIN_PEAK_WIDTH: float = 0.12
    FPS_RESAMPLE_RATE: float = 128.0
    PROMINENCE_LOWER_BOUND: float = 5
    PROMINENCE_UPPER_BOUND: float = 95
    KUBIOS_L: float = 51
    KUBIOS_THRESHOLD: float = 0.15