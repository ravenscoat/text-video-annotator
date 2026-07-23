"""Measure whether cached tracks contain each MeViS target object."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def iou(first: np.ndarray, second: np.ndarray) -> float:
    if first.shape != second.shape:
        second = cv2.resize(second.astype(np.uint8), (first.shape[1], first.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    union = np.logical_or(first, second).sum()
    return float(np.logical_and(first, second).sum() / union) if union else 0.0


def evaluate(manifest: Path, generation_report: Path, masks_path: Path, thresholds: tuple[float, ...] = (0.10, 0.30, 0.50)) -> dict:
    try:
        from pycocotools import mask as mask_utils
    except ImportError as exc:
        raise RuntimeError("pycocotools is required") from exc
    payload = json.loads(manifest.read_text(encoding="utf-8")); generated = json.loads(generation_report.read_text(encoding="utf-8")); mask_dict = json.loads(masks_path.read_text(encoding="utf-8"))
    cases = []
    for case, generated_case in zip(payload.get("cases", []), generated.get("cases", [])):
        cache = json.loads(Path(generated_case["cache"]).read_text(encoding="utf-8"))
        predicted_by_frame: dict[int, list[np.ndarray]] = {}
        for track in cache.get("tracks", []):
            for frame in track.get("frames", []):
                if frame.get("mask_path"):
                    mask = cv2.imread(frame["mask_path"], cv2.IMREAD_GRAYSCALE)
                    if mask is not None:
                        predicted_by_frame.setdefault(int(frame["frame_index"]), []).append(mask > 0)
        target_ids = [str(value) for value in case.get("annotation_ids", [])]
        per_target = []
        for annotation_id in target_ids:
            best = 0.0
            frame_rles = mask_dict.get(annotation_id, [])
            for frame_index, rle in enumerate(frame_rles):
                if rle is None:
                    continue
                target = mask_utils.decode(rle).astype(bool)
                best = max(best, max((iou(prediction, target) for prediction in predicted_by_frame.get(frame_index, [])), default=0.0))
            per_target.append(best)
        cases.append({"video_id": case["video_id"], "expression_id": case["expression_id"], "prompt": case["prompt"], "candidate_track_count": len(cache.get("tracks", [])), "target_count": len(per_target), "best_iou_per_target": per_target, "candidate_recall": {str(threshold): sum(score >= threshold for score in per_target) / max(len(per_target), 1) for threshold in thresholds}})
    report = {"case_count": len(cases), "target_count": sum(case["target_count"] for case in cases), "candidate_recall": {str(threshold): sum(score >= threshold for case in cases for score in case["best_iou_per_target"]) / max(sum(case["target_count"] for case in cases), 1) for threshold in thresholds}, "cases": cases, "definition": "A target is candidate-present when any cached predicted mask overlaps any ground-truth target frame at or above the stated IoU threshold. This is a proposal-coverage diagnostic, not official MeViS performance."}
    return report


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--manifest", type=Path, required=True); parser.add_argument("--generation-report", type=Path, required=True); parser.add_argument("--masks", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); report = evaluate(args.manifest, args.generation_report, args.masks); args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(report, indent=2), encoding="utf-8"); print(json.dumps({key: value for key, value in report.items() if key != "cases"}, indent=2))


if __name__ == "__main__": main()
