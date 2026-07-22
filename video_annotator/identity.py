from dataclasses import dataclass, field
import math

import numpy as np

from .types import Detection


@dataclass
class TrackState:
    track_id: int
    label: str
    score: float
    last_box_xyxy: tuple[float, float, float, float]
    last_mask: np.ndarray | None
    last_seen_frame: int
    missed_frames: int = 0
    centers: list[tuple[int, float, float]] = field(default_factory=list)
    areas: list[tuple[int, float]] = field(default_factory=list)


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (area_a + area_b - inter) if area_a + area_b - inter else 0.0


def associate(previous: list[Detection], current: list[Detection], next_id: int, threshold: float = 0.3):
    used = set()
    for detection in current:
        candidates = [(iou(previous[index].box_xyxy, detection.box_xyxy), index) for index in range(len(previous)) if index not in used and previous[index].label == detection.label]
        if candidates:
            score, index = max(candidates)
            if score >= threshold:
                detection.track_id = previous[index].track_id
                used.add(index)
        if detection.track_id is None:
            detection.track_id, next_id = next_id, next_id + 1
    return current, next_id


def _label(value: str) -> str:
    value = " ".join(str(value).lower().strip(" .,;:!?\t\r\n").split())
    return value[:-1] if value.endswith("s") and not value.endswith("ss") else value


def _mask_iou(first: np.ndarray | None, second: np.ndarray | None) -> float | None:
    if first is None or second is None:
        return None
    if first.shape != second.shape:
        return None
    intersection = np.logical_and(first, second).sum()
    union = np.logical_or(first, second).sum()
    return float(intersection / union) if union else 0.0


def _center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _motion_score(track: TrackState, box: tuple[float, float, float, float]) -> float:
    current = _center(box)
    if len(track.centers) < 2:
        return 0.5
    _, previous_x, previous_y = track.centers[-2]
    _, last_x, last_y = track.centers[-1]
    predicted_x = last_x + (last_x - previous_x)
    predicted_y = last_y + (last_y - previous_y)
    distance = math.hypot(current[0] - predicted_x, current[1] - predicted_y)
    diagonal = max(math.hypot(track.last_box_xyxy[2] - track.last_box_xyxy[0], track.last_box_xyxy[3] - track.last_box_xyxy[1]), 1.0)
    return float(math.exp(-distance / diagonal))


class TrackManager:
    """Deterministic multi-object association with persistent global IDs."""

    def __init__(self, max_missed_redetections: int = 2, association_threshold: float = 0.25):
        self.max_missed_redetections = max_missed_redetections
        self.association_threshold = association_threshold
        self._next_id = 1
        self._tracks: dict[int, TrackState] = {}

    @property
    def tracks(self) -> dict[int, TrackState]:
        return dict(self._tracks)

    def _score(self, track: TrackState, detection: Detection) -> float:
        box_score = iou(track.last_box_xyxy, detection.box_xyxy)
        motion_score = _motion_score(track, detection.box_xyxy)
        mask_score = _mask_iou(track.last_mask, detection.mask)
        if mask_score is not None:
            return 0.45 * box_score + 0.35 * mask_score + 0.20 * motion_score
        # Renormalize the box/motion weights when no masks are available.
        return (0.45 * box_score + 0.20 * motion_score) / 0.65

    def update(self, detections: list[Detection], frame_index: int) -> list[Detection]:
        candidates = []
        for track_id, track in self._tracks.items():
            for index, detection in enumerate(detections):
                if _label(track.label) == _label(detection.label):
                    candidates.append((self._score(track, detection), track_id, index))
        used_tracks: set[int] = set()
        used_detections: set[int] = set()
        for score, track_id, index in sorted(candidates, key=lambda item: (-item[0], item[1], item[2])):
            if score < self.association_threshold or track_id in used_tracks or index in used_detections:
                continue
            self._update_track(self._tracks[track_id], detections[index], frame_index)
            detections[index].track_id = track_id
            used_tracks.add(track_id); used_detections.add(index)
        for index, detection in enumerate(detections):
            if index in used_detections:
                continue
            track_id = self._next_id; self._next_id += 1
            state = TrackState(track_id, detection.label, detection.score, detection.box_xyxy, detection.mask, frame_index)
            self._record(state, detection, frame_index)
            self._tracks[track_id] = state
            detection.track_id = track_id
        for track_id, track in list(self._tracks.items()):
            if track_id not in used_tracks and not any(detection.track_id == track_id for detection in detections):
                track.missed_frames += 1
                if track.missed_frames > self.max_missed_redetections:
                    del self._tracks[track_id]
        return detections

    @staticmethod
    def _record(track: TrackState, detection: Detection, frame_index: int) -> None:
        x, y = _center(detection.box_xyxy)
        area = max(0.0, detection.box_xyxy[2] - detection.box_xyxy[0]) * max(0.0, detection.box_xyxy[3] - detection.box_xyxy[1])
        track.centers.append((frame_index, x, y)); track.areas.append((frame_index, area))

    def _update_track(self, track: TrackState, detection: Detection, frame_index: int) -> None:
        track.label = detection.label; track.score = detection.score; track.last_box_xyxy = detection.box_xyxy; track.last_mask = detection.mask; track.last_seen_frame = frame_index; track.missed_frames = 0
        self._record(track, detection, frame_index)
