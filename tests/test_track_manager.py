import numpy as np

from video_annotator.identity import TrackManager
from video_annotator.types import Detection


def test_tracks_keep_ids_for_two_same_class_objects():
    manager = TrackManager()
    first = manager.update([Detection("dog", (0, 0, 10, 10), .9), Detection("dog", (30, 0, 40, 10), .8)], 0)
    ids = [item.track_id for item in first]
    second = manager.update([Detection("dog", (3, 0, 13, 10), .9), Detection("dog", (27, 0, 37, 10), .8)], 1)
    assert [item.track_id for item in second] == ids


def test_unmatched_track_survives_brief_miss_and_reclaims_id():
    manager = TrackManager(max_missed_redetections=2)
    first = manager.update([Detection("horse", (0, 0, 10, 10), .9)], 0)
    track_id = first[0].track_id
    manager.update([], 1)
    recovered = manager.update([Detection("horse", (1, 0, 11, 10), .9)], 2)
    assert recovered[0].track_id == track_id


def test_late_object_gets_new_id_and_labels_cannot_cross_match():
    manager = TrackManager()
    first = manager.update([Detection("dog", (0, 0, 10, 10), .9)], 0)
    dog_id = first[0].track_id
    second = manager.update([Detection("cat", (0, 0, 10, 10), .9), Detection("horse", (20, 0, 30, 10), .8)], 3)
    assert {item.label for item in second} == {"cat", "horse"}
    assert all(item.track_id != dog_id for item in second)
    assert len({item.track_id for item in second}) == 2


def test_mask_overlap_contributes_to_matching():
    manager = TrackManager()
    mask = np.zeros((8, 8), dtype=bool); mask[2:6, 2:6] = True
    first = manager.update([Detection("dog", (0, 0, 8, 8), .9, mask=mask)], 0)
    moved = manager.update([Detection("dog", (1, 0, 9, 8), .9, mask=mask)], 1)
    assert moved[0].track_id == first[0].track_id
