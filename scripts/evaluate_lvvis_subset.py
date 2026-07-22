"""Run the offline annotator on a prepared LV-VIS validation manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from video_annotator.config import AnnotationConfig
from video_annotator.detector import GroundingDinoDetector
from video_annotator.pipeline import annotate_media
from video_annotator.tracker import Sam2VideoTracker
from video_annotator.video_metrics import frame_match, mask_iou, track_fragmentation


def decode_segmentation(segmentation, height: int, width: int) -> np.ndarray:
    try:
        from pycocotools import mask as mask_utils
    except ImportError as exc:
        raise RuntimeError("Install pycocotools before LV-VIS evaluation") from exc
    if not segmentation:
        return np.zeros((height, width), dtype=bool)
    if isinstance(segmentation, dict):
        rle = segmentation
        if isinstance(rle.get("counts"), list):
            rle = mask_utils.frPyObjects(rle, height, width)
        decoded = mask_utils.decode(rle)
    else:
        decoded = mask_utils.decode(mask_utils.frPyObjects(segmentation, height, width))
    if decoded.ndim == 3:
        decoded = np.any(decoded, axis=2)
    return decoded.astype(bool)


def read_prediction_masks(mask_dir: Path, frame_count: int) -> tuple[dict[int, list[np.ndarray]], dict[int, list[int]]]:
    masks: dict[int, list[np.ndarray]] = {frame: [] for frame in range(frame_count)}
    ids: dict[int, list[int]] = {frame: [] for frame in range(frame_count)}
    for path in sorted(mask_dir.glob("frame_*_object_*.png")):
        stem = path.stem
        try:
            frame = int(stem.split("_object_")[0].removeprefix("frame_"))
            track_id = int(stem.split("_object_")[1])
        except (IndexError, ValueError):
            continue
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is not None and frame in masks:
            masks[frame].append(mask > 0)
            ids[frame].append(track_id)
    return masks, ids


def evaluate_case(case: dict, detector, tracker, output_root: Path, chunk_frames: int, long_side: int, threshold: float, device: str) -> dict:
    video_dir = Path(case["video_dir"])
    frame_names = case.get("frame_names", [])
    if not frame_names:
        frame_names = sorted(path.name for path in video_dir.glob("*.jpg"))
    first = cv2.imread(str(video_dir / frame_names[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise FileNotFoundError(f"Cannot decode first LV-VIS frame in {video_dir}")
    height, width = first.shape[:2]
    input_video = output_root / f"inputs_{case['video_id']}.mp4"
    # The production pipeline accepts a video file. Assemble only this case's
    # frames, one at a time, so no full source video is held in memory.
    writer = cv2.VideoWriter(str(input_video), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create temporary evaluator video: {input_video}")
    try:
        for name in frame_names:
            frame = cv2.imread(str(video_dir / name), cv2.IMREAD_COLOR)
            if frame is None:
                raise FileNotFoundError(f"Cannot decode LV-VIS frame: {video_dir / name}")
            writer.write(frame)
    finally:
        writer.release()
    output_video = output_root / f"annotated_{case['video_id']}.mp4"
    export_json = output_root / f"predictions_{case['video_id']}.json"
    mask_dir = output_root / f"masks_{case['video_id']}"
    config = AnnotationConfig(input_video, case["prompt"], output_video, export_json=export_json, export_masks=mask_dir, long_side=long_side, chunk_frames=chunk_frames, device=device)
    result = annotate_media(config, detector=detector, video_tracker=tracker)
    prediction_masks, prediction_ids = read_prediction_masks(mask_dir, len(frame_names))
    frame_results = []
    assignments = []
    total_targets = total_predictions = total_tp = total_fp = total_fn = 0
    for frame_index in range(len(frame_names)):
        source = case.get("segmentations", [])
        segmentation = source[frame_index] if frame_index < len(source) else None
        target = decode_segmentation(segmentation, height, width)
        targets = [target] if target.any() else []
        predictions = prediction_masks.get(frame_index, [])
        matching = frame_match(predictions, targets, threshold)
        matched = matching["matches"][0] if matching["matches"] else None
        assignments.append(prediction_ids[frame_index][matched["prediction"]] if matched and matched["hit"] else None)
        frame_results.append({"frame_index": frame_index, **matching})
        total_targets += len(targets)
        total_predictions += len(predictions)
        total_tp += matching["true_positives"]
        total_fp += matching["false_positives"]
        total_fn += matching["false_negatives"]
    ious = [match["iou"] for frame in frame_results for match in frame["matches"]]
    return {
        "video_id": case["video_id"],
        "category_name": case["category_name"],
        "prompt": case["prompt"],
        "frame_count": len(frame_names),
        "annotated_frames": result.frame_count,
        "mean_frame_mask_iou": float(np.mean(ious)) if ious else 0.0,
        "recall_at_iou_0_50": total_tp / total_targets if total_targets else 0.0,
        "false_positive_masks": total_fp,
        "false_negative_masks": total_fn,
        "track_fragmentation": track_fragmentation(assignments),
        "frames": frame_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--long-side", type=int, default=768)
    parser.add_argument("--chunk-frames", type=int, default=30)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    cases = payload.get("cases", [])
    if not cases:
        raise ValueError("Manifest has no cases; run prepare_video_manifest.py first")
    args.output.mkdir(parents=True, exist_ok=True)
    detector = GroundingDinoDetector(device=args.device)
    tracker = Sam2VideoTracker(device=args.device)
    results = []
    for index, case in enumerate(cases, 1):
        print(f"[{index}/{len(cases)}] {case['prompt']} ({case['video_id']})")
        results.append(evaluate_case(case, detector, tracker, args.output, args.chunk_frames, args.long_side, args.iou_threshold, args.device))
    total_frames = sum(item["frame_count"] for item in results)
    report = {
        "dataset": "LV-VIS validation diagnostic subset",
        "case_count": len(results),
        "frame_count": total_frames,
        "mean_frame_mask_iou": float(np.mean([item["mean_frame_mask_iou"] for item in results])) if results else 0.0,
        "mean_recall_at_iou_0_50": float(np.mean([item["recall_at_iou_0_50"] for item in results])) if results else 0.0,
        "false_positive_masks": sum(item["false_positive_masks"] for item in results),
        "false_negative_masks": sum(item["false_negative_masks"] for item in results),
        "track_fragmentation": sum(item["track_fragmentation"] for item in results),
        "definition": "Metrics are computed on the locally selected diagnostic subset and are not official LV-VIS AP.",
        "cases": results,
    }
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "cases"}, indent=2))


if __name__ == "__main__":
    main()
