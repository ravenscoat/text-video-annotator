"""Evaluate referring-expression cases from a local Ref-DAVIS17 manifest."""
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


def read_prediction_masks(mask_dir: Path, frame_count: int):
    masks = {i: [] for i in range(frame_count)}
    ids = {i: [] for i in range(frame_count)}
    for path in sorted(mask_dir.glob("frame_*_object_*.png")):
        try:
            frame, track = path.stem.split("_object_")
            frame = int(frame.removeprefix("frame_"))
            track = int(track)
        except (ValueError, IndexError):
            continue
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is not None and frame in masks:
            masks[frame].append(image > 0)
            ids[frame].append(track)
    return masks, ids


def indexed_mask(path: Path, object_id: str) -> np.ndarray:
    # DAVIS PNGs are palette-mode images. OpenCV expands the palette to RGB,
    # losing the instance indices, so decode with Pillow instead.
    try:
        from PIL import Image
        image = np.asarray(Image.open(path).convert("P"))
    except (ImportError, OSError) as exc:
        raise FileNotFoundError(f"Cannot decode Ref-DAVIS annotation: {path}") from exc
    return image == int(object_id)


def boundary_f(prediction: np.ndarray, target: np.ndarray, tolerance: int = 2) -> float:
    if not prediction.any() and not target.any():
        return 1.0
    if not prediction.any() or not target.any():
        return 0.0
    def edge(mask):
        kernel = np.ones((3, 3), np.uint8)
        return mask & ~cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    pred_edge, target_edge = edge(prediction), edge(target)
    kernel = np.ones((2 * tolerance + 1, 2 * tolerance + 1), np.uint8)
    target_near = cv2.dilate(target_edge.astype(np.uint8), kernel) > 0
    pred_near = cv2.dilate(pred_edge.astype(np.uint8), kernel) > 0
    precision = float((pred_edge & target_near).sum()) / max(float(pred_edge.sum()), 1.0)
    recall = float((target_edge & pred_near).sum()) / max(float(target_edge.sum()), 1.0)
    return 2 * precision * recall / max(precision + recall, 1e-8)


def evaluate_case(case: dict, detector, tracker, output_root: Path, long_side: int, chunk_frames: int, threshold: float, device: str) -> dict:
    video_dir = Path(case["video_dir"])
    annotation_dir = Path(case["annotation_dir"])
    frame_names = case["frame_names"]
    first = cv2.imread(str(video_dir / frame_names[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise FileNotFoundError(f"Cannot decode first frame: {video_dir / frame_names[0]}")
    height, width = first.shape[:2]
    stem = f"{case['video_id']}_{case['expression_id']}"
    input_video = output_root / f"input_{stem}.mp4"
    writer = cv2.VideoWriter(str(input_video), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create temporary video: {input_video}")
    try:
        for name in frame_names:
            frame = cv2.imread(str(video_dir / name), cv2.IMREAD_COLOR)
            if frame is None:
                raise FileNotFoundError(f"Cannot decode frame: {video_dir / name}")
            writer.write(frame)
    finally:
        writer.release()
    mask_dir = output_root / f"masks_{stem}"
    config = AnnotationConfig(input_video, case["prompt"], output_root / f"annotated_{stem}.mp4", export_json=output_root / f"predictions_{stem}.json", export_masks=mask_dir, long_side=long_side, chunk_frames=chunk_frames, device=device)
    result = annotate_media(config, detector=detector, video_tracker=tracker)
    predictions, prediction_ids = read_prediction_masks(mask_dir, len(frame_names))
    frame_results, assignments = [], []
    ious, boundaries = [], []
    tp = fp = fn = 0
    for index, name in enumerate(frame_names):
        target = indexed_mask(annotation_dir / Path(name).with_suffix(".png").name, case["object_id"])
        matching = frame_match(predictions[index], [target] if target.any() else [], threshold)
        matched = matching["matches"][0] if matching["matches"] else None
        if matched and matched["hit"]:
            pred = predictions[index][matched["prediction"]]
            ious.append(matched["iou"])
            boundaries.append(boundary_f(pred, target))
            assignments.append(prediction_ids[index][matched["prediction"]])
        else:
            assignments.append(None)
        frame_results.append({"frame_index": index, **matching})
        tp += matching["true_positives"]; fp += matching["false_positives"]; fn += matching["false_negatives"]
    return {"video_id": case["video_id"], "expression_id": case["expression_id"], "object_id": case["object_id"], "prompt": case["prompt"], "frame_count": len(frame_names), "annotated_frames": result.frame_count, "mean_region_jaccard": float(np.mean(ious)) if ious else 0.0, "mean_boundary_f": float(np.mean(boundaries)) if boundaries else 0.0, "recall_at_iou_0_50": tp / max(tp + fn, 1), "false_positive_masks": fp, "false_negative_masks": fn, "track_fragmentation": track_fragmentation(assignments), "frames": frame_results}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--long-side", type=int, default=512)
    parser.add_argument("--chunk-frames", type=int, default=30)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    cases = json.loads(args.manifest.read_text(encoding="utf-8")).get("cases", [])
    if not cases: raise ValueError("Manifest has no cases")
    detector, tracker = GroundingDinoDetector(device=args.device), Sam2VideoTracker(device=args.device)
    results = []
    for index, case in enumerate(cases, 1):
        print(f"[{index}/{len(cases)}] {case['prompt']} ({case['video_id']})")
        results.append(evaluate_case(case, detector, tracker, args.output, args.long_side, args.chunk_frames, args.iou_threshold, args.device))
    report = {"dataset": "Ref-DAVIS17 validation diagnostic subset", "case_count": len(results), "mean_region_jaccard": float(np.mean([r["mean_region_jaccard"] for r in results])), "mean_boundary_f": float(np.mean([r["mean_boundary_f"] for r in results])), "mean_recall_at_iou_0_50": float(np.mean([r["recall_at_iou_0_50"] for r in results])), "false_positive_masks": sum(r["false_positive_masks"] for r in results), "false_negative_masks": sum(r["false_negative_masks"] for r in results), "track_fragmentation": sum(r["track_fragmentation"] for r in results), "definition": "Diagnostic subset region Jaccard (mask IoU) and boundary F; not official Ref-DAVIS17 leaderboard J&F.", "cases": results}
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}, indent=2))


if __name__ == "__main__": main()
