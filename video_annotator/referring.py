"""Candidate-track selection for natural-language referring expressions."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from .identity import TrackState
from .motion import MotionIntent, TrackFeatures, intent_for_clauses, track_features


@dataclass(frozen=True)
class Selection:
    track_id: int
    score: float
    reasons: tuple[str, ...] = field(default_factory=tuple)


def _label_matches(text: str, label: str) -> bool:
    normalized = re.sub(r"[^a-z0-9 ]", " ", text.lower())
    label = label.lower().strip()
    return bool(label) and re.search(rf"\b{re.escape(label)}s?\b", normalized) is not None


def _score(features: TrackFeatures, track: TrackState, intent: MotionIntent, text: str) -> Selection | None:
    if intent.object_ids and track.track_id not in intent.object_ids:
        return None
    score = 0.5
    reasons: list[str] = []
    if intent.object_ids and track.track_id in intent.object_ids:
        score += 0.5
        reasons.append(f"object_{track.track_id}")
    if _label_matches(text, track.label):
        score += 0.25; reasons.append(f"label_{track.label}")
    elif re.search(r"\b(the|a|an)\s+[a-z]", text.lower()) and not intent.object_ids:
        # A concrete noun was requested but this candidate's label is absent.
        score -= 0.15
    if intent.moving:
        if features.mean_speed > 0.5:
            score += 0.2; reasons.append("moving")
        else:
            score -= 0.2
    if intent.stationary:
        if features.mean_speed <= 0.5:
            score += 0.2; reasons.append("stationary")
        else:
            score -= 0.2
    if intent.directions:
        if features.direction in intent.directions:
            score += 0.2; reasons.append(f"moves_{features.direction}")
        else:
            score -= 0.15
    if intent.visibility == "least":
        score += max(0.0, 0.15 * (1.0 - features.visible_fraction)); reasons.append("least_visible")
    if intent.rank in {"first", "last"}:
        if intent.rank == "first" and features.first_frame <= min(2, features.last_frame):
            score += 0.15; reasons.append("visible_first")
        if intent.rank == "last" and features.last_frame >= features.first_frame:
            score += 0.15; reasons.append("visible_last")
    if intent.relation in {"approaching", "separating"}:
        # Relation-specific pair scoring is added when neighboring tracks are available.
        if intent.relation == "approaching" and features.area_change > 0:
            score += 0.1; reasons.append("approaching")
        if intent.relation == "separating" and features.area_change < 0:
            score += 0.1; reasons.append("separating")
    return Selection(track.track_id, max(0.0, min(1.0, score)), tuple(reasons))


def select_tracks(
    tracks: Iterable[TrackState],
    expression: str,
    *,
    frame_count: int | None = None,
    threshold: float = 0.55,
) -> list[Selection]:
    """Score all candidates, unioning explicit selection clauses.

    Tracks are never discarded before scoring. A low score produces no target,
    which is preferable to silently annotating an unrelated object.
    """
    candidates = list(tracks)
    selections: dict[int, Selection] = {}
    for intent in intent_for_clauses(expression):
        for track in candidates:
            result = _score(track_features(track, frame_count), track, intent, intent.text)
            if result is not None and result.score >= threshold:
                previous = selections.get(result.track_id)
                if previous is None or result.score > previous.score:
                    selections[result.track_id] = result
    return sorted(selections.values(), key=lambda item: (-item.score, item.track_id))
