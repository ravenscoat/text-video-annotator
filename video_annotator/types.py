from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Detection:
    label: str
    box_xyxy: tuple[float, float, float, float]
    score: float
    mask: np.ndarray | None = None
    track_id: int | None = None


@dataclass
class FrameResult:
    frame_index: int
    timestamp_seconds: float
    objects: list[Detection] = field(default_factory=list)


@dataclass
class MediaInfo:
    width: int
    height: int
    fps: float
    frame_count: int
    is_video: bool


@dataclass
class AnnotationResult:
    output_path: str
    media_type: str
    frame_count: int
    objects_found: int
    metadata_path: str | None = None
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
