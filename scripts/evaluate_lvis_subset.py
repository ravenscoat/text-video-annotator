"""Evaluate Grounding DINO + SAM 2 on a small LVIS image/category manifest."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from video_annotator.detector import GroundingDinoDetector
from video_annotator.render import render
from video_annotator.tracker import Sam2ImageSegmenter
from video_annotator.types import Detection


def load_coco_masks(annotation: dict, height: int, width: int) -> np.ndarray:
    try:
        from pycocotools import mask as mask_utils
    except ImportError as exc:
        raise RuntimeError("Install pycocotools before running LVIS evaluation") from exc
    segmentation = annotation.get("segmentation")
    if segmentation is None:
        return np.zeros((height, width), dtype=bool)
    if isinstance(segmentation, dict):
        rle = segmentation
        if isinstance(rle.get("counts"), list):
            rle = mask_utils.frPyObjects(rle, height, width)
        decoded = mask_utils.decode(rle)
    else:
        rles = mask_utils.frPyObjects(segmentation, height, width)
        decoded = mask_utils.decode(rles)
    if decoded.ndim == 3:
        decoded = np.any(decoded, axis=2)
    return decoded.astype(bool)


def mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    intersection = np.logical_and(left, right).sum()
    union = np.logical_or(left, right).sum()
    return float(intersection / union) if union else 0.0


def greedy_match(predictions: list[np.ndarray], targets: list[np.ndarray], threshold: float = 0.5) -> dict:
    candidates = sorted(
        ((mask_iou(prediction, target), pred_index, target_index)
         for pred_index, prediction in enumerate(predictions)
         for target_index, target in enumerate(targets)),
        reverse=True,
    )
    used_predictions, used_targets, matches = set(), set(), []
    for score, pred_index, target_index in candidates:
        if pred_index in used_predictions or target_index in used_targets:
            continue
        used_predictions.add(pred_index)
        used_targets.add(target_index)
        matches.append({"prediction": pred_index, "target": target_index, "iou": score, "hit": score >= threshold})
    true_positives = sum(match["hit"] for match in matches)
    return {
        "matches": matches,
        "true_positives": true_positives,
        "false_positives": len(predictions) - true_positives,
        "false_negatives": len(targets) - true_positives,
        "mean_matched_iou": float(np.mean([match["iou"] for match in matches])) if matches else 0.0,
    }


def resize_for_inference(image: np.ndarray, long_side: int) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    scale = min(1.0, long_side / max(height, width))
    if scale == 1.0:
        return image, scale
    resized = cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    return resized, scale


def evaluate_case(case: dict, annotations_by_id: dict, detector, segmenter, long_side: int, box_threshold: float, text_threshold: float, max_objects: int, device: str) -> tuple[dict, np.ndarray | None]:
    image_path = Path(case["image_path"])
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not decode image: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    inference_rgb, scale = resize_for_inference(image_rgb, long_side)
    detections = detector.detect(inference_rgb, case["prompt"], box_threshold, text_threshold, max_objects)
    detections = segmenter.segment(inference_rgb, detections)
    predictions = []
    for detection in detections:
        if detection.mask is None:
            continue
        mask = cv2.resize(detection.mask.astype(np.uint8), (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        detection.mask = mask
        predictions.append(mask)
    targets = [load_coco_masks(annotations_by_id[annotation_id], image_bgr.shape[0], image_bgr.shape[1]) for annotation_id in case["annotation_ids"]]
    result = greedy_match(predictions, targets)
    result.update({
        "image_id": case["image_id"],
        "image_path": str(image_path),
        "category_id": case["category_id"],
        "category_name": case["category_name"],
        "prompt": case["prompt"],
        "target_count": len(targets),
        "prediction_count": len(predictions),
        "inference_long_side": long_side,
        "scale": scale,
        "no_detection": not bool(predictions),
        "detections": [{"label": detection.label, "score": detection.score, "box_xyxy": detection.box_xyxy} for detection in detections],
    })
    preview = render(image_bgr, detections)
    return result, preview


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--long-side", type=int, default=768)
    parser.add_argument("--box-threshold", type=float, default=0.30)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--max-objects", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    annotation_payload = json.loads(args.annotations.read_text(encoding="utf-8"))
    annotations_by_id = {annotation["id"]: annotation for annotation in annotation_payload.get("annotations", []) if "id" in annotation}
    cases = payload.get("cases", [])
    if not cases:
        raise ValueError("Manifest has no LVIS cases; regenerate it with prepare_dataset.py")
    args.output.mkdir(parents=True, exist_ok=True)
    preview_dir = args.output / "previews"
    preview_dir.mkdir(exist_ok=True)
    detector = GroundingDinoDetector(device=args.device)
    segmenter = Sam2ImageSegmenter(device=args.device)
    per_case = []
    resolutions = [args.long_side] + [value for value in (640, 512) if value < args.long_side]
    for index, case in enumerate(cases, 1):
        last_error = None
        for resolution in resolutions:
            try:
                result, preview = evaluate_case(case, annotations_by_id, detector, segmenter, resolution, args.box_threshold, args.text_threshold, args.max_objects, args.device)
                result["case_index"] = index
                if last_error:
                    result["warning"] = f"Retried after CUDA OOM: {last_error}"
                if preview is not None:
                    cv2.imwrite(str(preview_dir / f"{index:03d}_{case['category_id']}.jpg"), preview)
                per_case.append(result)
                print(f"[{index}/{len(cases)}] {case['prompt']}: predictions={result['prediction_count']} mean_iou={result['mean_matched_iou']:.3f}")
                break
            except RuntimeError as exc:
                try:
                    import torch
                    is_oom = "out of memory" in str(exc).lower() and torch.cuda.is_available()
                    if is_oom:
                        torch.cuda.empty_cache()
                except ImportError:
                    is_oom = False
                if not is_oom or resolution == resolutions[-1]:
                    raise
                last_error = str(exc)
        else:
            raise RuntimeError(f"Could not evaluate case {index}")

    total_targets = sum(item["target_count"] for item in per_case)
    total_predictions = sum(item["prediction_count"] for item in per_case)
    total_true_positives = sum(item["true_positives"] for item in per_case)
    matched_ious = [match["iou"] for item in per_case for match in item["matches"]]
    metrics = {
        "dataset": "LVIS v1 validation diagnostic subset",
        "case_count": len(per_case),
        "category_count": len({item["category_id"] for item in per_case}),
        "mean_matched_mask_iou": float(np.mean(matched_ious)) if matched_ious else 0.0,
        "recall_at_iou_0_50": total_true_positives / total_targets if total_targets else 0.0,
        "precision_at_iou_0_50": total_true_positives / total_predictions if total_predictions else 0.0,
        "false_positive_count": total_predictions - total_true_positives,
        "false_negative_count": total_targets - total_true_positives,
        "no_detection_case_count": sum(1 for item in per_case if item["no_detection"]),
        "definition": "A true positive is a one-to-one greedy prediction/ground-truth mask match with IoU >= 0.50. This is a small diagnostic subset, not official LVIS AP.",
        "per_category": {},
    }
    by_category = defaultdict(list)
    for item in per_case:
        by_category[item["category_name"]].append(item)
    for category, items in sorted(by_category.items()):
        category_ious = [match["iou"] for item in items for match in item["matches"]]
        category_tp = sum(item["true_positives"] for item in items)
        category_targets = sum(item["target_count"] for item in items)
        category_predictions = sum(item["prediction_count"] for item in items)
        metrics["per_category"][category] = {
            "cases": len(items),
            "mean_matched_mask_iou": float(np.mean(category_ious)) if category_ious else 0.0,
            "recall_at_iou_0_50": category_tp / category_targets if category_targets else 0.0,
            "precision_at_iou_0_50": category_tp / category_predictions if category_predictions else 0.0,
        }
    if args.device.startswith("cuda"):
        try:
            import torch
            metrics["peak_cuda_memory_mb"] = round(torch.cuda.max_memory_allocated() / 1024**2, 2)
        except Exception:
            metrics["peak_cuda_memory_mb"] = None
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (args.output / "cases.jsonl").open("w", encoding="utf-8") as handle:
        for item in per_case:
            handle.write(json.dumps(item) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
