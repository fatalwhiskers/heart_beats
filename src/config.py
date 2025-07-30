from dataclasses import dataclass

@dataclass(frozen=False)
class Video:
    FPS: int = 35
    target_FPS: int = 35

@dataclass(frozen=True)
class Signal:
    WINDOW_SECONDS: int = 10
    HR_LOW: float = 0.75      # Hz   (45 bpm)
    HR_HIGH: float = 3.0      # Hz  (180 bpm)
    HR_ORDER = 3