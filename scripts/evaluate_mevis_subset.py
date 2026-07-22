"""Evaluate MeViS Val-u expression cases with the local annotation pipeline."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from video_annotator.config import AnnotationConfig
from video_annotator.detector import GroundingDinoDetector
from video_annotator.pipeline import annotate_media
from video_annotator.tracker import Sam2VideoTracker
from video_annotator.video_metrics import frame_match, track_fragmentation
from scripts.evaluate_refdavis_subset import boundary_f, read_prediction_masks


def decode_target(mask_dict: dict, annotation_ids: list[str], frame_index: int, height: int, width: int) -> np.ndarray:
    try:
        from pycocotools import mask as mask_utils
    except ImportError as exc:
        raise RuntimeError("Install pycocotools before MeViS evaluation") from exc
    target = np.zeros((height, width), dtype=bool)
    for annotation_id in annotation_ids:
        frames = mask_dict.get(str(annotation_id), mask_dict.get(annotation_id, []))
        rle = frames[frame_index] if frame_index < len(frames) else None
        if rle is not None:
            target |= mask_utils.decode(rle).astype(bool)
    return target


def evaluate_case(case: dict, mask_dict: dict, detector, tracker, output_root: Path, long_side: int, chunk_frames: int, threshold: float, device: str) -> dict:
    video_dir = Path(case["video_dir"]); frames = case["frame_names"]
    first = cv2.imread(str(video_dir / f"{frames[0]}.jpg"), cv2.IMREAD_COLOR)
    if first is None: raise FileNotFoundError(f"Cannot decode MeViS frame: {video_dir / frames[0]}.jpg")
    height, width = first.shape[:2]; stem = f"{case['video_id']}_{case['expression_id']}"
    input_video = output_root / f"input_{stem}.mp4"
    writer = cv2.VideoWriter(str(input_video), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (width, height))
    if not writer.isOpened(): raise RuntimeError(f"Cannot create temporary video: {input_video}")
    try:
        for frame_name in frames:
            image = cv2.imread(str(video_dir / f"{frame_name}.jpg"), cv2.IMREAD_COLOR)
            if image is None: raise FileNotFoundError(f"Cannot decode MeViS frame: {video_dir / frame_name}.jpg")
            writer.write(image)
    finally: writer.release()
    mask_dir = output_root / f"masks_{stem}"
    config = AnnotationConfig(input_video, case["prompt"], output_root / f"annotated_{stem}.mp4", export_json=output_root / f"predictions_{stem}.json", export_masks=mask_dir, long_side=long_side, chunk_frames=chunk_frames, device=device)
    result = annotate_media(config, detector=detector, video_tracker=tracker)
    predictions, prediction_ids = read_prediction_masks(mask_dir, len(frames))
    frame_results = []; assignments = []; ious = []; boundaries = []; tp = fp = fn = 0
    for index in range(len(frames)):
        target = decode_target(mask_dict, case["annotation_ids"], index, height, width)
        if predictions[index] and predictions[index][0].shape != target.shape:
            target = cv2.resize(target.astype(np.uint8), (predictions[index][0].shape[1], predictions[index][0].shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        matching = frame_match(predictions[index], [target] if target.any() else [], threshold)
        hit = matching["matches"][0] if matching["matches"] else None
        if hit and hit["hit"]:
            prediction = predictions[index][hit["prediction"]]; ious.append(hit["iou"]); boundaries.append(boundary_f(prediction, target)); assignments.append(prediction_ids[index][hit["prediction"]])
        else: assignments.append(None)
        frame_results.append({"frame_index": index, **matching}); tp += matching["true_positives"]; fp += matching["false_positives"]; fn += matching["false_negatives"]
    return {"video_id": case["video_id"], "expression_id": case["expression_id"], "prompt": case["prompt"], "frame_count": len(frames), "annotated_frames": result.frame_count, "mean_region_jaccard": float(np.mean(ious)) if ious else 0.0, "mean_boundary_f": float(np.mean(boundaries)) if boundaries else 0.0, "recall_at_iou_0_50": tp / max(tp + fn, 1), "false_positive_masks": fp, "false_negative_masks": fn, "track_fragmentation": track_fragmentation(assignments), "frames": frame_results}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--manifest", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--long-side", type=int, default=512); parser.add_argument("--chunk-frames", type=int, default=30); parser.add_argument("--iou-threshold", type=float, default=0.5); parser.add_argument("--device", default="cuda"); parser.add_argument("--start-case", type=int, default=0); parser.add_argument("--max-cases", type=int)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    payload = json.loads(args.manifest.read_text(encoding="utf-8")); all_cases = payload.get("cases", []); cases = all_cases[args.start_case:]; cases = cases[:args.max_cases] if args.max_cases else cases
    if not cases: raise ValueError("Manifest has no cases")
    mask_path = Path(cases[0]["mask_dict"]); mask_dict = json.loads(mask_path.read_text(encoding="utf-8"))
    detector, tracker = GroundingDinoDetector(device=args.device), Sam2VideoTracker(device=args.device); results = []
    for index, case in enumerate(cases, 1):
        print(f"[{index}/{len(cases)}] {case['prompt']} ({case['video_id']})")
        results.append(evaluate_case(case, mask_dict, detector, tracker, args.output, args.long_side, args.chunk_frames, args.iou_threshold, args.device))
    report = {"dataset": "MeViS Val-u diagnostic subset", "case_count": len(results), "video_count": len({r["video_id"] for r in results}), "mean_region_jaccard": float(np.mean([r["mean_region_jaccard"] for r in results])), "mean_boundary_f": float(np.mean([r["mean_boundary_f"] for r in results])), "mean_recall_at_iou_0_50": float(np.mean([r["recall_at_iou_0_50"] for r in results])), "false_positive_masks": sum(r["false_positive_masks"] for r in results), "false_negative_masks": sum(r["false_negative_masks"] for r in results), "track_fragmentation": sum(r["track_fragmentation"] for r in results), "definition": "Diagnostic subset region Jaccard and boundary F over selected expressions; not official MeViS leaderboard J&F.", "cases": results}
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8"); print(json.dumps({k: v for k, v in report.items() if k != "cases"}, indent=2))


if __name__ == "__main__": main()
