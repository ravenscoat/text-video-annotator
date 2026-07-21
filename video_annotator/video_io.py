from pathlib import Path
import tempfile

import cv2
import numpy as np

from .types import MediaInfo

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def is_video(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTENSIONS


def probe(path: Path) -> MediaInfo:
    if not is_video(path):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot read image: {path}")
        height, width = image.shape[:2]
        return MediaInfo(width, height, 0.0, 1, False)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()
    if width <= 0 or height <= 0:
        raise ValueError(f"Video has invalid dimensions: {path}")
    if count <= 0:
        raise ValueError(f"Video has no decodable frames: {path}")
    return MediaInfo(width, height, fps if fps > 0 else 30.0, count, True)


def resize_for_inference(frame: np.ndarray, long_side: int) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = min(1.0, long_side / max(height, width))
    if scale == 1.0:
        return frame.copy()
    return cv2.resize(frame, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)


def iter_video_chunks(path: Path, chunk_frames: int, long_side: int):
    """Yield (start_index, [(original_bgr, resized_rgb), ...]) without loading the video."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    start = 0
    try:
        while True:
            chunk = []
            for _ in range(chunk_frames):
                ok, frame = cap.read()
                if not ok:
                    break
                small = resize_for_inference(frame, long_side)
                chunk.append((frame, cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
            if not chunk:
                break
            yield start, chunk
            start += len(chunk)
    finally:
        cap.release()


def make_writer(path: Path, info: MediaInfo):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), info.fps, (info.width, info.height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create output video: {path}")
    return writer


def temporary_chunk_dir():
    return tempfile.TemporaryDirectory(prefix="video_annotator_")
