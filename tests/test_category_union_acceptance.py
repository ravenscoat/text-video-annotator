import numpy as np

from video_annotator.detector import postprocess_detections
from video_annotator.identity import TrackManager
from video_annotator.types import Detection


def test_category_union_filters_cat_and_keeps_late_requested_object_ids():
    boxes = np.asarray(
        [
            [0, 0, 20, 20],   # dog
            [30, 0, 50, 20],  # horse
            [60, 0, 80, 20],  # cat (must be removed)
        ],
        dtype=np.float32,
    )
    scores = np.asarray([0.9, 0.8, 0.99], dtype=np.float32)
    labels = ["dog", "horse", "cat"]
    detections = postprocess_detections(
        boxes, scores, labels, targets=("dog", "horse"), max_objects=10
    )
    assert {item.label for item in detections} == {"dog", "horse"}

    manager = TrackManager()
    first = manager.update(detections, frame_index=0)
    ids = {item.label: item.track_id for item in first}

    # The horse appears after the initial detector refresh; it receives a
    # persistent ID while the dog keeps its original ID.
    second = manager.update(
        [Detection("dog", (1, 0, 21, 20), 0.88), Detection("horse", (31, 0, 51, 20), 0.82)],
        frame_index=15,
    )
    assert next(item.track_id for item in second if item.label == "dog") == ids["dog"]
    assert next(item.track_id for item in second if item.label == "horse") == ids["horse"]

    third = manager.update(
        [Detection("dog", (2, 0, 22, 20), 0.87), Detection("horse", (32, 0, 52, 20), 0.81)],
        frame_index=30,
    )
    assert {item.track_id for item in third} == set(ids.values())
    assert all(item.label in {"dog", "horse"} for item in third)
