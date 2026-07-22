"""Merge split first-50 MeViS A/B outputs without rerunning model inference."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_mevis_subset import decode_target
from scripts.evaluate_refdavis_subset import boundary_f, read_prediction_masks
from video_annotator.video_metrics import frame_match, track_fragmentation


def evaluate_saved_case(case: dict, root: Path, threshold: float) -> dict:
    frames = case["frame_names"]
    stem = f"{case['video_id']}_{case['expression_id']}"
    masks, ids = read_prediction_masks(root / f"masks_{stem}", len(frames))
    first = cv2.imread(str(Path(case["video_dir"]) / f"{frames[0]}.jpg"), cv2.IMREAD_COLOR)
    if first is None:
        raise FileNotFoundError(case["video_dir"])
    height, width = first.shape[:2]
    mask_dict = json.loads(Path(case["mask_dict"]).read_text(encoding="utf-8"))
    ious, boundaries, assignments = [], [], []
    tp = fp = fn = 0
    for index in range(len(frames)):
        target = decode_target(mask_dict, case["annotation_ids"], index, height, width)
        if masks[index] and masks[index][0].shape != target.shape:
            target = cv2.resize(target.astype(np.uint8), (masks[index][0].shape[1], masks[index][0].shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        matching = frame_match(masks[index], [target] if target.any() else [], threshold)
        hit = matching["matches"][0] if matching["matches"] else None
        if hit and hit["hit"]:
            prediction = masks[index][hit["prediction"]]
            ious.append(hit["iou"]); boundaries.append(boundary_f(prediction, target))
            assignments.append(ids[index][hit["prediction"]])
        else:
            assignments.append(None)
        tp += matching["true_positives"]; fp += matching["false_positives"]; fn += matching["false_negatives"]
    return {
        "video_id": case["video_id"], "expression_id": case["expression_id"], "prompt": case["prompt"],
        "frame_count": len(frames), "annotated_frames": sum(bool(items) for items in masks),
        "mean_region_jaccard": float(np.mean(ious)) if ious else 0.0,
        "mean_boundary_f": float(np.mean(boundaries)) if boundaries else 0.0,
        "recall_at_iou_0_50": tp / max(tp + fn, 1),
        "false_positive_masks": fp, "false_negative_masks": fn,
        "track_fragmentation": track_fragmentation(assignments),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--first-root", type=Path, required=True)
    parser.add_argument("--resume-root", type=Path, required=True)
    parser.add_argument("--split-index", type=int, default=21)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = json.loads(args.manifest.read_text(encoding="utf-8"))["cases"][:50]
    results = []
    for index, case in enumerate(cases):
        root = args.first_root if index < args.split_index else args.resume_root
        results.append(evaluate_saved_case(case, root, 0.5))
    report = {
        "dataset": "MeViS Val-u first-50 A/B diagnostic subset",
        "case_count": len(results), "video_count": len({item["video_id"] for item in results}),
        "mean_region_jaccard": float(np.mean([item["mean_region_jaccard"] for item in results])),
        "mean_boundary_f": float(np.mean([item["mean_boundary_f"] for item in results])),
        "mean_recall_at_iou_0_50": float(np.mean([item["recall_at_iou_0_50"] for item in results])),
        "false_positive_masks": sum(item["false_positive_masks"] for item in results),
        "false_negative_masks": sum(item["false_negative_masks"] for item in results),
        "track_fragmentation": sum(item["track_fragmentation"] for item in results),
        "definition": "Diagnostic region Jaccard and boundary F over exactly the first 50 expressions; not official MeViS leaderboard metrics.",
        "cases": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "cases"}, indent=2))


if __name__ == "__main__":
    main()
