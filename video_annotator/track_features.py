"""CPU-side temporal, relationship, and lightweight appearance features."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from .track_cache import AnchorToken, TrackCache, CachedTrack, TrackFrame
from .motion import sample_anchor_indices


def _visible(track: CachedTrack) -> list[TrackFrame]:
    return [frame for frame in track.frames if frame.visible and frame.center_xy is not None]


def _box_iou(first, second) -> float:
    if first is None or second is None:
        return 0.0
    ax1, ay1, ax2, ay2 = first; bx1, by1, bx2, by2 = second
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(area_a + area_b - inter, 1e-9)


def choose_anchor_indices(track: CachedTrack, frame_count: int, uniform_count: int = 5, adaptive_count: int = 2) -> list[tuple[int, str]]:
    """Choose bounded chronological anchors: uniform coverage plus adaptive gaps."""
    visible = [frame for frame in _visible(track)]
    if not visible:
        return []
    uniform_positions = sample_anchor_indices(len(visible), min(uniform_count, 6))
    selected: dict[int, str] = {visible[position].frame_index: "uniform" for position in uniform_positions}
    remaining = [frame for frame in visible if frame.frame_index not in selected]
    # Confidence and temporal novelty are available before semantic encoders.
    remaining.sort(key=lambda frame: (-(frame.detector_score or 0.0), -min(frame.frame_index, frame_count - frame.frame_index), frame.frame_index))
    for frame in remaining[:max(0, min(adaptive_count, 2))]:
        selected[frame.frame_index] = "adaptive"
    return sorted(selected.items())


def populate_temporal_features(cache: TrackCache) -> None:
    """Add normalized trajectory and visibility features in-place."""
    width = max(float(cache.source.get("width", 1)), 1.0)
    height = max(float(cache.source.get("height", 1)), 1.0)
    frame_count = max(int(cache.source.get("frame_count", 1)), 1)
    for track in cache.tracks:
        frames = _visible(track)
        if not frames:
            track.temporal_features = {"visible_fraction": 0.0, "first_frame": -1, "last_frame": -1}
            continue
        centers = [(frame.frame_index, frame.center_xy[0] / width, frame.center_xy[1] / height) for frame in frames]
        speeds = [float(np.hypot(x2 - x1, y2 - y1) / max(f2 - f1, 1)) for (f1, x1, y1), (f2, x2, y2) in zip(centers, centers[1:])]
        first_frame, first_x, first_y = centers[0]; last_frame, last_x, last_y = centers[-1]
        area_values = [float(frame.area or 0.0) / (width * height) for frame in frames]
        dx, dy = last_x - first_x, last_y - first_y
        direction = "stationary" if float(np.hypot(dx, dy)) < 0.02 else ("right" if abs(dx) >= abs(dy) and dx > 0 else "left" if abs(dx) >= abs(dy) else "down" if dy > 0 else "up")
        track.temporal_features = {
            "displacement_x": float(dx), "displacement_y": float(dy), "mean_speed": float(np.mean(speeds)) if speeds else 0.0,
            "max_speed": float(max(speeds, default=0.0)), "visible_fraction": float(len(frames) / frame_count),
            "first_frame": int(first_frame), "last_frame": int(last_frame), "direction": direction,
            "mean_area": float(np.mean(area_values)), "area_change": float((area_values[-1] - area_values[0]) / max(area_values[0], 1e-9)),
        }
        anchors = choose_anchor_indices(track, frame_count)
        anchor_frames = {frame.frame_index: frame for frame in frames}
        tokens: list[AnchorToken] = []
        previous_velocity = (0.0, 0.0)
        previous_center = None
        previous_index = None
        for frame_index, provenance in anchors:
            frame = anchor_frames[frame_index]
            center = (frame.center_xy[0] / width, frame.center_xy[1] / height)
            box = frame.box_xyxy or (0.0, 0.0, 0.0, 0.0)
            size = ((box[2] - box[0]) / width, (box[3] - box[1]) / height)
            if previous_center is None:
                velocity = (0.0, 0.0)
            else:
                delta = max(frame_index - (previous_index or frame_index), 1)
                velocity = ((center[0] - previous_center[0]) / delta, (center[1] - previous_center[1]) / delta)
            acceleration = (velocity[0] - previous_velocity[0], velocity[1] - previous_velocity[1])
            tokens.append(AnchorToken(frame_index, frame_index / max(frame_count - 1, 1), center, size, velocity, acceleration, 1.0, float(frame.detector_score or 0.0), provenance))
            previous_center, previous_index, previous_velocity = center, frame_index, velocity
        track.anchor_tokens = tokens


def populate_relationship_features(cache: TrackCache, near_distance: float = 0.12) -> None:
    """Add pairwise near/left/right/overlap summaries using normalized centers."""
    width = max(float(cache.source.get("width", 1)), 1.0)
    height = max(float(cache.source.get("height", 1)), 1.0)
    frame_maps = {track.track_id: {frame.frame_index: frame for frame in _visible(track)} for track in cache.tracks}
    for track in cache.tracks:
        relations: dict[str, dict[str, float]] = {}
        for other in cache.tracks:
            if other.track_id == track.track_id:
                continue
            comparisons = []
            for index, frame in frame_maps[track.track_id].items():
                peer = frame_maps[other.track_id].get(index)
                if peer is None or frame.center_xy is None or peer.center_xy is None:
                    continue
                dx = (peer.center_xy[0] - frame.center_xy[0]) / width
                dy = (peer.center_xy[1] - frame.center_xy[1]) / height
                comparisons.append((dx, dy, _box_iou(frame.box_xyxy, peer.box_xyxy)))
            if comparisons:
                values = np.asarray(comparisons, dtype=float)
                relations[str(other.track_id)] = {"near_fraction": float(np.mean(np.hypot(values[:, 0], values[:, 1]) <= near_distance)), "left_of_fraction": float(np.mean(values[:, 0] < 0)), "right_of_fraction": float(np.mean(values[:, 0] > 0)), "mean_distance": float(np.mean(np.hypot(values[:, 0], values[:, 1]))), "mean_overlap": float(np.mean(values[:, 2]))}
        track.relation_features = relations


def _appearance_vector(frame_bgr: np.ndarray, mask: np.ndarray) -> list[float]:
    if mask.shape[:2] != frame_bgr.shape[:2]:
        mask = cv2.resize(mask.astype(np.uint8), (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    else:
        mask = mask.astype(bool)
    pixels = frame_bgr[mask]
    if len(pixels) == 0:
        return [0.0] * 9
    mean = pixels.mean(axis=0) / 255.0
    std = pixels.std(axis=0) / 255.0
    hist = []
    for channel in range(3):
        values, _ = np.histogram(pixels[:, channel], bins=3, range=(0, 255), density=False)
        hist.extend((values / max(len(pixels), 1)).tolist())
    return [float(value) for value in np.concatenate([mean, std, np.asarray(hist)])]


def populate_appearance_features(cache: TrackCache, video_path: Path, anchor_count: int = 6, encoder=None) -> None:
    """Sample masked crop color statistics without retaining video frames."""
    selected_by_track: dict[int, set[int]] = {}
    for track in cache.tracks:
        token_frames = [token.frame_index for token in track.anchor_tokens]
        positions = sample_anchor_indices(len(token_frames), anchor_count)
        selected_by_track[track.track_id] = {token_frames[position] for position in positions}
    needed = set().union(*selected_by_track.values()) if selected_by_track else set()
    if not needed:
        return
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video for appearance features: {video_path}")
    vectors: dict[int, list[list[float]]] = {track.track_id: [] for track in cache.tracks}
    frames_by_index = {track.track_id: {frame.frame_index: frame for frame in _visible(track)} for track in cache.tracks}
    try:
        for index in range(max(needed) + 1):
            ok, image = cap.read()
            if not ok:
                break
            if index not in needed:
                continue
            for track in cache.tracks:
                if index not in selected_by_track[track.track_id]:
                    continue
                cached = frames_by_index[track.track_id].get(index)
                if cached is None or not cached.mask_path:
                    continue
                mask = cv2.imread(str(cached.mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    crop = image.copy()
                    crop[~(mask > 0)] = 0
                    vector = encoder.encode_images([crop])[0].tolist() if encoder is not None else _appearance_vector(image, mask)
                    vectors[track.track_id].append([float(value) for value in vector])
                    for token in track.anchor_tokens:
                        if token.frame_index == index:
                            token_index = track.anchor_tokens.index(token)
                            track.anchor_tokens[token_index] = AnchorToken(token.frame_index, token.timestamp, token.position_xy, token.size_wh, token.velocity_xy, token.acceleration_xy, token.visibility, token.confidence, token.provenance, token.relation_features, [float(value) for value in vector])
    finally:
        cap.release()
    for track in cache.tracks:
        track.appearance_features = vectors[track.track_id]


def populate_cached_features(cache: TrackCache, video_path: Path | None = None, anchor_count: int = 6, encoder=None) -> TrackCache:
    """Populate all CPU-side features and return the same cache object."""
    populate_temporal_features(cache)
    populate_relationship_features(cache)
    if video_path is not None:
        populate_appearance_features(cache, video_path, anchor_count, encoder)
    return cache
