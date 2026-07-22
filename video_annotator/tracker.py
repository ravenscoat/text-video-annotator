from __future__ import annotations

from pathlib import Path
import sys
import numpy as np

from .types import Detection


class Sam2ImageSegmenter:
    def __init__(self, model_id: str = "facebook/sam2.1-hiera-tiny", device: str = "cuda"):
        self.model_id, self.device = model_id, device
        self.predictor = None

    def load(self):
        try:
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError:
            # Editable installs can retain the old absolute path after the
            # repository is moved. Prefer the vendored source in this checkout.
            local_vendor = Path(__file__).resolve().parents[1] / "vendor" / "sam2"
            if local_vendor.is_dir():
                sys.path.insert(0, str(local_vendor))
                from sam2.sam2_image_predictor import SAM2ImagePredictor
            else:
                raise RuntimeError("Install the official SAM 2 package to use segmentation")
        self.predictor = SAM2ImagePredictor.from_pretrained(self.model_id)
        self.predictor.model.to(self.device)

    def segment(self, image_rgb: np.ndarray, detections: list[Detection]) -> list[Detection]:
        if self.predictor is None:
            self.load()
        import torch
        self.predictor.set_image(image_rgb)
        for detection in detections:
            masks, _, _ = self.predictor.predict(box=np.asarray(detection.box_xyxy, dtype=np.float32), multimask_output=False)
            detection.mask = np.asarray(masks[0], dtype=bool)
        return detections


class Sam2VideoTracker:
    def __init__(self, model_id: str = "facebook/sam2.1-hiera-tiny", device: str = "cuda"):
        self.model_id, self.device = model_id, device
        self.predictor = None

    def load(self):
        try:
            from sam2.sam2_video_predictor import SAM2VideoPredictor
        except ImportError:
            local_vendor = Path(__file__).resolve().parents[1] / "vendor" / "sam2"
            if local_vendor.is_dir():
                sys.path.insert(0, str(local_vendor))
                from sam2.sam2_video_predictor import SAM2VideoPredictor
            else:
                raise RuntimeError("Install the official SAM 2 package to use video tracking")
        self.predictor = SAM2VideoPredictor.from_pretrained(self.model_id)
        self.predictor.to(self.device)

    def propagate(self, frame_dir: Path, detections: list[Detection]):
        if self.predictor is None:
            self.load()
        state = self.predictor.init_state(str(frame_dir), offload_video_to_cpu=True, offload_state_to_cpu=True, async_loading_frames=False)
        try:
            for obj_id, detection in enumerate(detections, start=1):
                self.predictor.add_new_points_or_box(state, frame_idx=0, obj_id=obj_id, box=np.asarray(detection.box_xyxy, dtype=np.float32))
            for frame_idx, object_ids, mask_logits in self.predictor.propagate_in_video(state):
                result = []
                for index, object_id in enumerate(object_ids):
                    if index >= len(mask_logits):
                        continue
                    mask = (mask_logits[index] > 0).detach().cpu().numpy().squeeze().astype(bool)
                    source = detections[int(object_id) - 1] if int(object_id) - 1 < len(detections) else Detection("object", (0, 0, 0, 0), 0)
                    result.append(Detection(source.label, source.box_xyxy, source.score, mask=mask, track_id=int(object_id)))
                yield int(frame_idx), result
        finally:
            self.predictor.reset_state(state)
