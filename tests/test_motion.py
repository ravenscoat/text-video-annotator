from video_annotator.identity import TrackState
from video_annotator.motion import parse_motion_intent, sample_anchor_indices, track_features
from video_annotator.referring import select_tracks


def _track(track_id, label, centers):
    state = TrackState(track_id, label, 0.9, (0, 0, 10, 10), None, centers[-1][0])
    state.centers.extend(centers)
    state.areas.extend((frame, 100.0) for frame, _, _ in centers)
    return state


def test_anchor_sampling_is_uniform_and_bounded():
    assert sample_anchor_indices(100, 6) == [0, 20, 40, 59, 79, 99]
    assert sample_anchor_indices(0, 6) == []
    assert len(sample_anchor_indices(4, 20)) <= 8


def test_motion_intent_parses_actions_directions_and_ids():
    intent = parse_motion_intent("object 1 and object 3 running left")
    assert intent.object_ids == {1, 3}
    assert intent.moving and "running" in intent.actions
    assert intent.directions == {"left"}


def test_selects_only_matching_moving_and_stationary_clauses():
    running = _track(1, "person", [(0, 0, 0), (5, 8, 0), (10, 18, 0)])
    sitting = _track(2, "person", [(0, 40, 0), (5, 40, 0), (10, 40, 0)])
    selected = select_tracks([running, sitting], "the person running and the person sitting", frame_count=11)
    assert {item.track_id for item in selected} == {1, 2}
    assert all(item.reasons for item in selected)


def test_object_id_selection_excludes_other_tracks():
    first = _track(1, "bird", [(0, 0, 0), (5, 1, 0)])
    third = _track(3, "bird", [(0, 20, 0), (5, 21, 0)])
    selected = select_tracks([first, third], "object 1 and object 3", frame_count=6)
    assert {item.track_id for item in selected} == {1, 3}


def test_features_report_direction_and_visibility():
    features = track_features(_track(1, "car", [(0, 0, 0), (4, 8, 0)]), frame_count=10)
    assert features.direction == "right"
    assert features.visible_fraction == 0.2
