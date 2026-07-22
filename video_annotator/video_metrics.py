"""Small, dataset-agnostic metrics for video mask tracking diagnostics."""
from __future__ import annotations

import numpy as np


def mask_iou(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    union = np.logical_or(prediction, target).sum()
    return float(np.logical_and(prediction, target).sum() / union) if union else 0.0


def frame_match(predictions: list[np.ndarray], targets: list[np.ndarray], threshold: float = 0.5) -> dict:
    candidates = sorted(
        ((mask_iou(prediction, target), p, t)
         for p, prediction in enumerate(predictions)
         for t, target in enumerate(targets)),
        reverse=True,
    )
    used_predictions, used_targets, matches = set(), set(), []
    for score, prediction_index, target_index in candidates:
        if prediction_index in used_predictions or target_index in used_targets:
            continue
        used_predictions.add(prediction_index)
        used_targets.add(target_index)
        matches.append({"prediction": prediction_index, "target": target_index, "iou": score, "hit": score >= threshold})
    true_positives = sum(match["hit"] for match in matches)
    return {
        "matches": matches,
        "true_positives": true_positives,
        "false_positives": len(predictions) - true_positives,
        "false_negatives": len(targets) - true_positives,
        "mean_matched_iou": float(np.mean([match["iou"] for match in matches])) if matches else 0.0,
    }


def track_fragmentation(assignments: list[int | None]) -> int:
    """Count reappearing predicted track segments for one ground-truth object."""
    segments = 0
    previous = None
    for track_id in assignments:
        if track_id is None:
            previous = None
            continue
        if track_id != previous:
            segments += 1
        previous = track_id
    return max(0, segments - 1)
