import cv2
import numpy as np


def color_for(track_id: int | None):
    value = (track_id or 0) * 2654435761 % 0xFFFFFF
    return int(value & 255), int((value >> 8) & 255), int((value >> 16) & 255)


def render(frame_bgr: np.ndarray, detections, alpha: float = 0.4) -> np.ndarray:
    output = frame_bgr.copy()
    for detection in detections:
        if detection.mask is None:
            continue
        mask = detection.mask.astype(np.uint8)
        if mask.shape[:2] != output.shape[:2]:
            mask = cv2.resize(mask, (output.shape[1], output.shape[0]), interpolation=cv2.INTER_NEAREST)
        color = np.array(color_for(detection.track_id), dtype=np.uint8)
        overlay = np.empty_like(output)
        overlay[:] = color
        selector = mask > 0
        output[selector] = (output[selector] * (1 - alpha) + overlay[selector] * alpha).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(output, contours, -1, tuple(int(x) for x in color), 2)
        x1, y1, _, _ = (int(v) for v in detection.box_xyxy)
        label = detection.label if detection.track_id is None else f"{detection.label} #{detection.track_id}"
        cv2.putText(output, label, (max(0, x1), max(16, y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, tuple(int(x) for x in color), 2, cv2.LINE_AA)
    return output
