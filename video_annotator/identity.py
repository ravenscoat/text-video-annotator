from .types import Detection


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (area_a + area_b - inter) if area_a + area_b - inter else 0.0


def associate(previous: list[Detection], current: list[Detection], next_id: int, threshold: float = 0.3):
    used = set()
    for detection in current:
        candidates = [(iou(previous[index].box_xyxy, detection.box_xyxy), index) for index in range(len(previous)) if index not in used and previous[index].label == detection.label]
        if candidates:
            score, index = max(candidates)
            if score >= threshold:
                detection.track_id = previous[index].track_id
                used.add(index)
        if detection.track_id is None:
            detection.track_id, next_id = next_id, next_id + 1
    return current, next_id
