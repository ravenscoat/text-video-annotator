import json
from pathlib import Path

import pytest

from video_annotator.track_cache import CachedTrack, TrackCache, TrackFrame, load_cache, make_proposal_scope, save_cache, scope_fingerprint


def _cache() -> TrackCache:
    return TrackCache(
        source={"source_id": "video-a", "frame_count": 4, "width": 32, "height": 24, "fps": 10.0},
        config={"long_side": 512, "redetect_every": 15},
        proposal_scope=make_proposal_scope(["person", "car"], ["person . car ."]),
        tracks=[CachedTrack(1, "person", 0.9, [TrackFrame(0, True, (1, 2, 8, 10), "masks/1_000.png", 0.9, center_xy=(4.5, 6.0), area=56.0)], appearance_features=[[0.1, 0.2]], temporal_features={"mean_speed": 1.2})],
    )


def test_cache_round_trip_preserves_ids_masks_and_dimensions(tmp_path: Path):
    path = tmp_path / "tracks.json"
    save_cache(path, _cache())
    loaded = load_cache(path, expected_source_id="video-a", expected_scope_fingerprint=_cache().scope_fingerprint)
    assert loaded.source["width"] == 32 and loaded.source["height"] == 24
    assert loaded.tracks[0].track_id == 1
    assert loaded.tracks[0].frames[0].mask_path == "masks/1_000.png"


def test_cache_write_is_valid_json_and_scope_is_deterministic(tmp_path: Path):
    path = tmp_path / "tracks.json"
    save_cache(path, _cache())
    assert json.loads(path.read_text(encoding="utf-8"))["schema_version"] == 2
    assert scope_fingerprint(["car", "person"], ["person . car ."]) == _cache().scope_fingerprint


def test_cache_rejects_corrupt_and_incompatible_files(tmp_path: Path):
    path = tmp_path / "tracks.json"
    path.write_text("not json", encoding="utf-8")
    with pytest.raises(ValueError, match="Cannot read track cache"):
        load_cache(path)
    save_cache(path, _cache())
    with pytest.raises(ValueError, match="source fingerprint"):
        load_cache(path, expected_source_id="other-video")
    with pytest.raises(ValueError, match="proposal scope"):
        load_cache(path, expected_scope_fingerprint="wrong")


def test_cache_rejects_duplicate_ids_and_bad_frame_dimensions(tmp_path: Path):
    cache = _cache()
    cache.tracks.append(CachedTrack(1, "car", 0.8))
    with pytest.raises(ValueError, match="duplicate"):
        save_cache(tmp_path / "bad.json", cache)
    cache = _cache()
    cache.tracks[0].frames.append(TrackFrame(99, True))
    with pytest.raises(ValueError, match="out-of-range"):
        save_cache(tmp_path / "bad-frame.json", cache)
