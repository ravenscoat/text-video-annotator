from video_annotator.pipeline import redetection_indices


def test_redetection_indices_are_global_and_periodic():
    assert redetection_indices(61, 15) == [0, 15, 30, 45, 60]


def test_redetection_empty_video_is_safe():
    assert redetection_indices(0, 15) == []
