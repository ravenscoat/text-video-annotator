from __future__ import annotations

from typing import Any
import numpy as np

from .types import Detection


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

    def detect(self, image_rgb: np.ndarray, prompt: str, box_threshold: float, text_threshold: float, max_objects: int) -> list[Detection]:
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
        detections: list[Detection] = []
        boxes = result.get("boxes", []).detach().cpu().numpy()
        scores = result.get("scores", []).detach().cpu().numpy()
        labels = result.get("text_labels", result.get("labels", []))
        for box, score, label in sorted(zip(boxes, scores, labels), key=lambda item: float(item[1]), reverse=True)[:max_objects]:
            detections.append(Detection(str(label), tuple(float(x) for x in box), float(score)))
        return detections
