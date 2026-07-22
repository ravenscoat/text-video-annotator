import numpy as np

from video_annotator.detector import postprocess_detections


def test_unrequested_label_is_removed_and_requested_labels_are_mapped():
    detections = postprocess_detections(
        np.array([[0, 0, 10, 10], [20, 0, 30, 10], [40, 0, 50, 10]], dtype=float),
        np.array([0.9, 0.8, 0.99]),
        ["dog", "horse", "cat"],
        targets=("dog", "horse"),
    )
    assert [item.label for item in detections] == ["dog", "horse"]


def test_class_aware_nms_keeps_separate_classes():
    detections = postprocess_detections(
        np.array([[0, 0, 10, 10], [0, 0, 10, 10], [20, 0, 30, 10]], dtype=float),
        np.array([0.9, 0.8, 0.7]),
        ["dog", "dog", "horse"],
        targets=("dog", "horse"),
        max_objects=10,
    )
    assert [(item.label, item.score) for item in detections] == [("dog", 0.9), ("horse", 0.7)]


def test_round_robin_global_cap_does_not_starve_second_class():
    boxes = np.array([[i * 20, 0, i * 20 + 10, 10] for i in range(4)], dtype=float)
    detections = postprocess_detections(boxes, np.array([.99, .98, .97, .96]), ["dog"] * 3 + ["horse"], targets=("dog", "horse"), max_objects=2, max_objects_per_target=5)
    assert {item.label for item in detections} == {"dog", "horse"}


def test_plural_label_matches_requested_singular_target():
    detections = postprocess_detections(np.array([[0, 0, 10, 10]], dtype=float), np.array([.9]), ["dogs"], targets=("dog",))
    assert detections[0].label == "dog"
