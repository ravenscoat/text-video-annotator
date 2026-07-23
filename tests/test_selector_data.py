from pathlib import Path

import cv2
import numpy as np

from video_annotator.selector_data import build_sample, split_by_video, write_selector_dataset
from video_annotator.track_cache import CachedTrack, TrackCache, TrackFrame, make_proposal_scope


def _cache(tmp_path: Path) -> TrackCache:
    tracks = []
    for track_id, x in ((1, 2), (2, 20)):
        mask = np.zeros((16, 24), dtype=np.uint8); mask[2:8, x:x + 6] = 255
        path = tmp_path / f"mask_{track_id}.png"; cv2.imwrite(str(path), mask)
        track = CachedTrack(track_id, "person", 0.9, [TrackFrame(0, True, (x, 2, x + 6, 8), str(path), 0.9, center_xy=(x + 3, 5), area=36)])
        track.temporal_features = {"mean_speed": float(track_id)}
        tracks.append(track)
    return TrackCache({"source_id": "fixture", "frame_count": 1, "width": 24, "height": 16}, {}, make_proposal_scope(["person"], ["person ."]), tracks)


def test_build_sample_supports_multiple_positive_tracks(tmp_path: Path):
    cache = _cache(tmp_path)
    target = np.zeros((16, 24), dtype=bool); target[2:8, 2:8] = True; target[2:8, 20:26] = True
    sample = build_sample({"video_id": "v1", "expression_id": "e1", "prompt": "the two people"}, cache, {0: target})
    assert sample.labels == (1, 1)
    assert len(sample.features[0]) == len(sample.feature_names)
    assert len(sample.token_sequences) == 2


def test_video_split_prevents_expression_leakage(tmp_path: Path):
    cache = _cache(tmp_path)
    target = np.zeros((16, 24), dtype=bool)
    samples = [build_sample({"video_id": video, "expression_id": str(i), "prompt": "person"}, cache, {0: target}) for i, video in enumerate(("a", "a", "b", "c", "c"))]
    train, validation = split_by_video(samples, 0.4)
    assert not ({item.video_id for item in train} & {item.video_id for item in validation})
    assert len(train) + len(validation) == len(samples)


def test_dataset_writer_records_splits(tmp_path: Path):
    cache = _cache(tmp_path)
    target = np.zeros((16, 24), dtype=bool)
    sample = build_sample({"video_id": "v", "expression_id": "e", "prompt": "person"}, cache, {0: target})
    output = tmp_path / "selector.jsonl"
    report = write_selector_dataset([sample], output, 0.5)
    assert report["samples"] == 1
    assert output.read_text(encoding="utf-8").count("\n") == 1


def test_sample_can_store_frozen_text_embedding(tmp_path: Path):
    class Encoder:
        def encode_text(self, texts):
            return np.asarray([[0.25, 0.75] for _ in texts], dtype=np.float32)
    cache = _cache(tmp_path)
    sample = build_sample({"video_id": "v", "expression_id": "e", "prompt": "person"}, cache, {0: np.zeros((16, 24), dtype=bool)}, text_encoder=Encoder())
    assert sample.text_embedding == (0.25, 0.75)
