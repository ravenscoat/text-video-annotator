from __future__ import annotations

from typing import Any
import numpy as np

from .types import Detection


def _normalize_label(value: str) -> str:
    return " ".join(str(value).lower().strip(" .,;:!?\t\r\n").split())


def _singular(value: str) -> str:
    return value[:-1] if value.endswith("s") and not value.endswith("ss") else value


def _label_matches(label: str, target: str) -> bool:
    label, target = _normalize_label(label), _normalize_label(target)
    if label == target or _singular(label) == _singular(target):
        return True
    label_tokens, target_tokens = set(label.split()), set(target.split())
    return bool(label_tokens and (label_tokens <= target_tokens or target_tokens <= label_tokens))


def _box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (area_a + area_b - inter) if area_a + area_b - inter else 0.0


def _class_nms(detections: list[Detection], threshold: float = 0.50) -> list[Detection]:
    kept: list[Detection] = []
    for detection in sorted(detections, key=lambda item: item.score, reverse=True):
        if all(_box_iou(detection.box_xyxy, previous.box_xyxy) < threshold for previous in kept):
            kept.append(detection)
    return kept


def postprocess_detections(
    boxes,
    scores,
    labels,
    targets: tuple[str, ...] = (),
    max_objects: int = 10,
    max_objects_per_target: int = 5,
    nms_threshold: float = 0.50,
) -> list[Detection]:
    """Map model labels to requested classes and apply class-aware limits."""
    candidates: list[Detection] = []
    for box, score, label in zip(boxes, scores, labels):
        label = str(label)
        target = next((requested for requested in targets if _label_matches(label, requested)), None) if targets else label
        if targets and target is None:
            continue
        candidates.append(Detection(target or label, tuple(float(x) for x in box), float(score)))
    if not targets:
        return _class_nms(candidates, nms_threshold)[:max_objects]
    grouped: dict[str, list[Detection]] = {target: [] for target in targets}
    for candidate in candidates:
        grouped.setdefault(candidate.label, []).append(candidate)
    groups = []
    for target in targets:
        group = _class_nms(grouped.get(target, []), nms_threshold)[:max_objects_per_target]
        if group:
            groups.append(group)
    # Round-robin selection prevents a high-confidence class from consuming
    # the global cap and starving another requested class.
    selected: list[Detection] = []
    index = 0
    while len(selected) < max_objects and any(index < len(group) for group in groups):
        for group in groups:
            if index < len(group) and len(selected) < max_objects:
                selected.append(group[index])
        index += 1
    return selected


class GroundingDinoDetector:
    """Lazy Hugging Face Grounding DINO Tiny adapter."""

    def __init__(self, model_id: str = "IDEA-Research/grounding-dino-tiny", device: str = "cuda"):
        self.model_id = model_id
        self.device = device
        self.processor = None
        self.model = None

    def load(self) -> None:
        try:
            import torch
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ImportError as exc:
            raise RuntimeError("Install video-annotator[models] to use Grounding DINO") from exc
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id)
        self.model.to(self.device)
        self.model.eval()

    def detect(self, image_rgb: np.ndarray, prompt: str, box_threshold: float, text_threshold: float, max_objects: int, targets: tuple[str, ...] = (), max_objects_per_target: int = 5) -> list[Detection]:
        if self.model is None:
            self.load()
        import torch
        text = prompt.strip()
        if not text.endswith("."):
            text += "."
        inputs = self.processor(images=image_rgb, text=text, return_tensors="pt")
        inputs = {key: value.to(self.device) if hasattr(value, "to") else value for key, value in inputs.items()}
        with torch.inference_mode():
            outputs = self.model(**inputs)
        target_sizes = torch.tensor([image_rgb.shape[:2]], device=self.device)
        result = self.processor.post_process_grounded_object_detection(
            outputs, inputs["input_ids"], threshold=box_threshold, text_threshold=text_threshold, target_sizes=target_sizes
        )[0]
        boxes = result.get("boxes", []).detach().cpu().numpy()
        scores = result.get("scores", []).detach().cpu().numpy()
        labels = result.get("text_labels", result.get("labels", []))
        return postprocess_detections(boxes, scores, labels, targets, max_objects, max_objects_per_target)
