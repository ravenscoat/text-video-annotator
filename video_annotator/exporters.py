import json
from pathlib import Path
import cv2
import numpy as np


def _json_object(detection):
    item = {"track_id": detection.track_id, "label": detection.label, "score": detection.score, "bbox_xyxy": list(detection.box_xyxy)}
    if detection.mask is not None:
        ys, xs = np.where(detection.mask)
        item["mask_area"] = int(len(xs))
        item["mask_bbox"] = [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)] if len(xs) else None
    return item


class JsonExporter:
    def __init__(self, path: Path, prompt: str, media_info):
        self.path, self.prompt, self.media_info = path, prompt, media_info
        self.frames = []

    def add(self, frame_index, timestamp, detections):
        self.frames.append({"frame_index": frame_index, "timestamp_seconds": timestamp, "objects": [_json_object(x) for x in detections]})

    def close(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"prompt": self.prompt, "source": self.media_info.__dict__, "frames": self.frames}
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_masks(directory: Path, frame_index: int, detections) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for detection in detections:
        if detection.mask is None:
            continue
        object_id = detection.track_id if detection.track_id is not None else 0
        cv2.imwrite(str(directory / f"frame_{frame_index:06d}_object_{object_id:04d}.png"), (detection.mask.astype(np.uint8) * 255))
