"""Lightweight temporal features and rule-based motion intent parsing.

This module deliberately provides an explainable baseline. It does not claim
to understand unrestricted action language; it turns common motion words into
testable constraints over tracked trajectories.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from .identity import TrackState


@dataclass(frozen=True)
class MotionIntent:
    text: str
    directions: frozenset[str] = frozenset()
    moving: bool = False
    stationary: bool = False
    rank: str | None = None
    visibility: str | None = None
    relation: str | None = None
    actions: frozenset[str] = frozenset()
    object_ids: frozenset[int] = frozenset()


@dataclass(frozen=True)
class TrackFeatures:
    track_id: int
    label: str
    displacement_x: float
    displacement_y: float
    mean_speed: float
    visible_fraction: float
    first_frame: int
    last_frame: int
    direction: str | None
    area_change: float


def sample_anchor_indices(frame_count: int, count: int = 6) -> list[int]:
    """Uniformly sample frame indices without loading frames into memory."""
    if frame_count <= 0:
        return []
    count = max(1, min(8, count))
    if count == 1:
        return [0]
    last = frame_count - 1
    return sorted({round(index * last / (count - 1)) for index in range(count)})


def _direction(dx: float, dy: float, threshold: float = 0.02) -> str | None:
    if abs(dx) < threshold and abs(dy) < threshold:
        return None
    if abs(dx) >= abs(dy):
        return "right" if dx > 0 else "left"
    return "down" if dy > 0 else "up"


def track_features(track: TrackState, frame_count: int | None = None) -> TrackFeatures:
    """Summarize one TrackManager state using CPU-side trajectory history."""
    centers = track.centers
    if not centers:
        return TrackFeatures(track.track_id, track.label, 0.0, 0.0, 0.0, 0.0, 0, 0, None, 0.0)
    first_frame, first_x, first_y = centers[0]
    last_frame, last_x, last_y = centers[-1]
    span = max(last_frame - first_frame, 1)
    dx, dy = (last_x - first_x) / span, (last_y - first_y) / span
    speed = sum(
        ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5 / max(f2 - f1, 1)
        for (f1, x1, y1), (f2, x2, y2) in zip(centers, centers[1:])
    ) / max(len(centers) - 1, 1)
    visible = len(centers) / max(frame_count or (last_frame + 1), 1)
    area_change = 0.0
    if track.areas:
        initial = track.areas[0][1]
        area_change = (track.areas[-1][1] - initial) / max(initial, 1.0)
    return TrackFeatures(
        track.track_id, track.label, dx, dy, speed, min(1.0, visible),
        first_frame, last_frame, _direction(last_x - first_x, last_y - first_y), area_change,
    )


def parse_motion_intent(text: str) -> MotionIntent:
    """Parse common motion/action words into an explainable intent."""
    lowered = " ".join(text.lower().split())
    directions = frozenset(word for word in ("left", "right", "up", "down") if re.search(rf"\b{word}\b", lowered))
    actions = frozenset(word for word in ("running", "moving", "walking", "flying", "sitting", "standing", "stationary", "still", "approaching", "separating", "touching") if re.search(rf"\b{word}\b", lowered))
    moving = bool(actions & {"running", "moving", "walking", "flying", "approaching", "separating"})
    stationary = bool(actions & {"sitting", "standing", "stationary", "still"})
    rank = next((name for name in ("first", "last", "front", "ahead", "behind", "back") if re.search(rf"\b{name}\b", lowered)), None)
    visibility = "least" if re.search(r"\b(least-visible|least visible|hardest to see)\b", lowered) else None
    relation = next((name for name in ("touching", "near", "approaching", "separating") if re.search(rf"\b{name}\b", lowered)), None)
    object_ids = frozenset(int(value) for value in re.findall(r"\bobject\s*(\d+)\b", lowered))
    return MotionIntent(lowered, directions, moving, stationary, rank, visibility, relation, actions, object_ids)


def intent_for_clauses(text: str) -> list[MotionIntent]:
    """Split only obvious selection clauses, preserving prose such as 'black and white'."""
    clauses = re.split(r"\s+and\s+(?=(?:the|a|an|object)\b)", text, flags=re.IGNORECASE)
    return [parse_motion_intent(clause) for clause in clauses if clause.strip()]
