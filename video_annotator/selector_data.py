"""Build video-disjoint selector-training samples from candidate caches."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .track_cache import TrackCache, CachedTrack, load_cache


TEMPORAL_KEYS = ("displacement_x", "displacement_y", "mean_speed", "max_speed", "visible_fraction", "area_change")


@dataclass(frozen=True)
class SelectorSample:
    video_id: str
    expression_id: str
    prompt: str
    track_ids: tuple[int, ...]
    features: tuple[tuple[float, ...], ...]
    labels: tuple[int, ...]
    feature_names: tuple[str, ...]
    token_sequences: tuple[tuple[tuple[float, ...], ...], ...] = ()
    text_embedding: tuple[float, ...] = ()


def _mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    if first.shape != second.shape:
        import cv2
        second = cv2.resize(second.astype(np.uint8), (first.shape[1], first.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    union = np.logical_or(first, second).sum()
    return float(np.logical_and(first, second).sum() / union) if union else 0.0


def _track_vector(track: CachedTrack) -> tuple[tuple[float, ...], tuple[str, ...]]:
    temporal = track.temporal_features
    values = [float(temporal.get(key, 0.0)) for key in TEMPORAL_KEYS]
    names = list(TEMPORAL_KEYS)
    appearance = track.appearance_features
    if appearance:
        values.extend(np.asarray(appearance, dtype=np.float32).mean(axis=0).tolist())
        names.extend(f"appearance_{index}" for index in range(len(values) - len(names)))
    else:
        values.extend([0.0] * 15)
        names.extend(f"appearance_{index}" for index in range(15))
    relations = list(track.relation_features.values())
    for key in ("near_fraction", "left_of_fraction", "right_of_fraction", "mean_distance", "mean_overlap"):
        values.append(float(np.mean([float(item.get(key, 0.0)) for item in relations])) if relations else 0.0)
        names.append(f"relation_{key}")
    return tuple(float(value) for value in values), tuple(names)


def _token_vector(token) -> tuple[float, ...]:
    return tuple(float(value) for value in (*token.position_xy, *token.size_wh, *token.velocity_xy, *token.acceleration_xy, token.timestamp, token.visibility, token.confidence, *token.image_embedding))


def build_labels(cache: TrackCache, target_masks: dict[int, np.ndarray], iou_threshold: float = 0.10) -> dict[int, int]:
    """Label a track positive if any visible cached mask overlaps target evidence."""
    labels: dict[int, int] = {}
    for track in cache.tracks:
        positive = False
        for frame in track.frames:
            if not frame.visible or not frame.mask_path or frame.frame_index not in target_masks:
                continue
            predicted = __import__("cv2").imread(str(frame.mask_path), __import__("cv2").IMREAD_GRAYSCALE)
            if predicted is not None and _mask_iou(predicted > 0, target_masks[frame.frame_index]) >= iou_threshold:
                positive = True
                break
        labels[track.track_id] = int(positive)
    return labels


def build_sample(case: dict[str, Any], cache: TrackCache, target_masks: dict[int, np.ndarray], iou_threshold: float = 0.10, text_encoder=None) -> SelectorSample:
    vectors = [_track_vector(track) for track in cache.tracks]
    if vectors:
        feature_names = vectors[0][1]
        features = tuple(item[0] for item in vectors)
    else:
        feature_names, features = (), ()
    labels = build_labels(cache, target_masks, iou_threshold)
    prompt = str(case["prompt"])
    text_embedding = tuple(float(value) for value in text_encoder.encode_text([prompt])[0]) if text_encoder is not None else ()
    token_sequences = tuple(tuple(_token_vector(token) for token in track.anchor_tokens) for track in cache.tracks)
    return SelectorSample(str(case["video_id"]), str(case.get("expression_id", "")), prompt, tuple(track.track_id for track in cache.tracks), features, tuple(labels[track.track_id] for track in cache.tracks), feature_names, token_sequences, text_embedding)


def split_by_video(samples: list[SelectorSample], validation_fraction: float = 0.2) -> tuple[list[SelectorSample], list[SelectorSample]]:
    """Deterministically split whole videos, never individual expressions."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    videos = sorted({sample.video_id for sample in samples})
    validation_count = max(1, round(len(videos) * validation_fraction)) if videos else 0
    validation_videos = set(videos[-validation_count:])
    train = [sample for sample in samples if sample.video_id not in validation_videos]
    validation = [sample for sample in samples if sample.video_id in validation_videos]
    return train, validation


def write_selector_dataset(samples: list[SelectorSample], output: Path, validation_fraction: float = 0.2) -> dict[str, Any]:
    train, validation = split_by_video(samples, validation_fraction)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for split, rows in (("train", train), ("validation", validation)):
            for sample in rows:
                handle.write(json.dumps({"split": split, **sample.__dict__}, separators=(",", ":")) + "\n")
    return {"samples": len(samples), "train_samples": len(train), "validation_samples": len(validation), "train_videos": sorted({item.video_id for item in train}), "validation_videos": sorted({item.video_id for item in validation})}
