import numpy as np

from video_annotator.video_metrics import frame_match, mask_iou, track_fragmentation


def test_video_mask_iou_and_frame_matching():
    target = np.zeros((4, 4), dtype=bool)
    target[:2, :2] = True
    prediction = target.copy()
    assert mask_iou(prediction, target) == 1.0
    result = frame_match([prediction, np.ones((4, 4), dtype=bool)], [target], threshold=0.5)
    assert result["true_positives"] == 1
    assert result["false_positives"] == 1


def test_track_fragmentation_counts_reappearing_segments():
    assert track_fragmentation([1, 1, None, 1, 2, 2]) == 2
    assert track_fragmentation([None, None]) == 0
