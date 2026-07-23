from pathlib import Path

import cv2
import numpy as np

from video_annotator.track_cache import CachedTrack, TrackCache, TrackFrame, make_proposal_scope
from video_annotator.track_features import choose_anchor_indices, populate_cached_features


def _cache(tmp_path: Path) -> TrackCache:
    frames_a = [TrackFrame(i, True, (2 + i, 2, 8 + i, 8), None, 0.9, center_xy=(5 + i, 5), area=36) for i in range(4)]
    frames_b = [TrackFrame(i, True, (20, 2, 26, 8), None, 0.8, center_xy=(23, 5), area=36) for i in range(4)]
    return TrackCache({"source_id": "fixture", "frame_count": 4, "width": 32, "height": 24, "fps": 5.0}, {}, make_proposal_scope(["dog", "cat"], ["dog . cat ."]), [CachedTrack(1, "dog", 0.9, frames_a), CachedTrack(2, "cat", 0.8, frames_b)])


def test_temporal_and_relationship_features_are_normalized(tmp_path: Path):
    cache = _cache(tmp_path)
    populate_cached_features(cache)
    first = cache.tracks[0]
    assert first.temporal_features["direction"] == "right"
    assert 0.0 < first.temporal_features["visible_fraction"] <= 1.0
    assert "2" in first.relation_features
    assert first.relation_features["2"]["right_of_fraction"] == 1.0


def test_appearance_features_sample_cached_masks_and_frames(tmp_path: Path):
    video = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (32, 24))
    cache = _cache(tmp_path)
    for index, color in enumerate(((0, 0, 255), (0, 0, 255), (0, 0, 255), (0, 0, 255))):
        image = np.zeros((24, 32, 3), dtype=np.uint8); image[:] = color
        mask = np.zeros((24, 32), dtype=np.uint8); mask[2:8, 2 + index:8 + index] = 255
        mask_path = tmp_path / f"mask_{index}.png"; cv2.imwrite(str(mask_path), mask)
        cache.tracks[0].frames[index] = TrackFrame(index, True, cache.tracks[0].frames[index].box_xyxy, str(mask_path), 0.9, center_xy=(5 + index, 5), area=36)
        writer.write(image)
    writer.release()
    populate_cached_features(cache, video, anchor_count=4)
    assert len(cache.tracks[0].appearance_features) == 4
    assert len(cache.tracks[0].appearance_features[0]) == 15


def test_short_late_track_gets_its_own_visible_anchors(tmp_path: Path):
    video = tmp_path / "late.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (32, 24))
    for _ in range(8):
        writer.write(np.full((24, 32, 3), 100, dtype=np.uint8))
    writer.release()
    cache = TrackCache({"source_id": "late", "frame_count": 8, "width": 32, "height": 24}, {}, make_proposal_scope(["bird"], ["bird ."]), [CachedTrack(1, "bird", 0.9)])
    for index in (6, 7):
        mask = np.zeros((24, 32), dtype=np.uint8); mask[5:10, 5:10] = 255
        path = tmp_path / f"late_{index}.png"; cv2.imwrite(str(path), mask)
        cache.tracks[0].frames.append(TrackFrame(index, True, (5, 5, 10, 10), str(path), 0.9, center_xy=(7.5, 7.5), area=25))
    populate_cached_features(cache, video, anchor_count=6)
    assert len(cache.tracks[0].appearance_features) == 2


def test_hybrid_anchor_budget_preserves_uniform_coverage():
    track = CachedTrack(1, "object", 0.9, [TrackFrame(index, True, (index, 1, index + 3, 4), None, 0.9, center_xy=(index + 1.5, 2.5), area=9) for index in range(30)])
    anchors = choose_anchor_indices(track, 30, uniform_count=5, adaptive_count=2)
    assert len(anchors) <= 7
    assert anchors[0][0] == 0 and anchors[-1][0] == 29
    assert sum(reason == "uniform" for _, reason in anchors) == 5


def test_mock_semantic_encoder_is_written_to_ordered_tokens(tmp_path: Path):
    video = tmp_path / "semantic.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (32, 24))
    cache = TrackCache({"source_id": "semantic", "frame_count": 2, "width": 32, "height": 24}, {}, make_proposal_scope(["object"], ["object ."]), [CachedTrack(1, "object", 0.9)])
    for index in range(2):
        image = np.full((24, 32, 3), 100 + index, dtype=np.uint8); writer.write(image)
        mask = np.zeros((24, 32), dtype=np.uint8); mask[3:8, 4:9] = 255
        path = tmp_path / f"semantic_{index}.png"; cv2.imwrite(str(path), mask)
        cache.tracks[0].frames.append(TrackFrame(index, True, (4, 3, 9, 8), str(path), 0.9, center_xy=(6.5, 5.5), area=25))
    writer.release()

    class Encoder:
        def encode_images(self, images):
            return np.asarray([[0.1, 0.2, 0.3] for _ in images], dtype=np.float32)

    populate_cached_features(cache, video, encoder=Encoder())
    assert cache.tracks[0].anchor_tokens
    assert all(np.allclose(token.image_embedding, [0.1, 0.2, 0.3]) for token in cache.tracks[0].anchor_tokens)
