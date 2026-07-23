from pathlib import Path

import cv2
import numpy as np

from video_annotator.pipeline import annotate_media, load_compatible_track_cache
from video_annotator.prompts import parse_prompt
from video_annotator.track_cache import CachedTrack, TrackCache, TrackFrame, make_proposal_scope, save_cache
from video_annotator.types import MediaInfo
from video_annotator.config import AnnotationConfig
from video_annotator.types import Detection
from video_annotator.track_cache import load_cache


def test_pipeline_cache_loader_checks_source_and_prompt_scope(tmp_path: Path):
    input_path = tmp_path / "clip.mp4"
    input_path.write_bytes(b"fixture")
    info = MediaInfo(32, 24, 10.0, 4, True)
    spec = parse_prompt("dog and horse")
    source_id = f"{input_path.resolve()}:{info.width}x{info.height}:{info.frame_count}:{info.fps:.6f}"
    cache = TrackCache(
        source={"source_id": source_id, "frame_count": info.frame_count},
        config={},
        proposal_scope=make_proposal_scope(list(spec.targets), [spec.detector_prompt]),
        tracks=[CachedTrack(1, "dog", 0.8)],
    )
    path = tmp_path / "clip.track-cache.json"
    save_cache(path, cache)
    loaded = load_compatible_track_cache(path, input_path, spec, info)
    assert loaded.tracks[0].label == "dog"


def test_cache_reuse_renders_without_detector_or_tracker(tmp_path: Path):
    source = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (32, 24))
    for _ in range(2):
        writer.write(np.zeros((24, 32, 3), dtype=np.uint8))
    writer.release()
    info = MediaInfo(32, 24, 5.0, 2, True)
    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()
    frames = []
    for index in range(2):
        mask = np.zeros((24, 32), dtype=np.uint8)
        mask[4:12, 5:13] = 255
        mask_path = mask_dir / f"m{index}.png"
        cv2.imwrite(str(mask_path), mask)
        frames.append(TrackFrame(index, True, (5, 4, 13, 12), str(mask_path), 0.9, center_xy=(9.0, 8.0), area=64.0))
    spec = parse_prompt("dog")
    source_id = f"{source.resolve()}:{info.width}x{info.height}:{info.frame_count}:{info.fps:.6f}"
    cache = TrackCache(source={"source_id": source_id, "frame_count": 2}, config={}, proposal_scope=make_proposal_scope(["dog"], ["dog ."]), tracks=[CachedTrack(1, "dog", 0.9, frames)])
    cache_path = tmp_path / "cache.json"
    save_cache(cache_path, cache)

    class MustNotRun:
        def detect(self, *args, **kwargs):
            raise AssertionError("detector should not run during cache reuse")

    result = annotate_media(AnnotationConfig(source, "dog", tmp_path / "out.mp4", track_cache_path=cache_path, reuse_track_cache=True), detector=MustNotRun())
    assert result.frame_count == 2
    assert result.objects_found == 2


def test_category_prompt_matches_compound_detector_label(tmp_path: Path):
    source = tmp_path / "turtles.avi"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (32, 24))
    writer.write(np.zeros((24, 32, 3), dtype=np.uint8)); writer.release()
    mask_dir = tmp_path / "masks"; mask_dir.mkdir(); mask_path = mask_dir / "m.png"
    mask = np.zeros((24, 32), dtype=np.uint8); mask[2:10, 2:10] = 255; cv2.imwrite(str(mask_path), mask)
    info = MediaInfo(32, 24, 5.0, 1, True); source_id = f"{source.resolve()}:{info.width}x{info.height}:{info.frame_count}:{info.fps:.6f}"
    frame = TrackFrame(0, True, (2, 2, 10, 10), str(mask_path), 0.9, center_xy=(6.0, 6.0), area=64.0)
    cache = TrackCache(source={"source_id": source_id, "frame_count": 1}, config={}, proposal_scope=make_proposal_scope(["turtles"], ["turtles ."]), tracks=[CachedTrack(1, "three turtles", 0.9, [frame])])
    cache_path = tmp_path / "cache.json"; save_cache(cache_path, cache)
    result = annotate_media(AnnotationConfig(source, "turtles", tmp_path / "out.mp4", track_cache_path=cache_path, reuse_track_cache=True))
    assert result.objects_found == 1


def test_generation_caches_rejected_candidates_and_second_pass_skips_models(tmp_path: Path):
    source = tmp_path / "source.avi"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (32, 24))
    for _ in range(2):
        writer.write(np.zeros((24, 32, 3), dtype=np.uint8))
    writer.release()

    class Detector:
        calls = 0
        def detect(self, *args, **kwargs):
            self.calls += 1
            return [Detection("dog", (2, 2, 10, 10), 0.9), Detection("cat", (18, 2, 26, 10), 0.8)]

    class Tracker:
        calls = 0
        def propagate(self, frame_dir, detections):
            self.calls += 1
            for index in range(2):
                rows = []
                for object_id, source_detection in enumerate(detections, 1):
                    mask = np.zeros((24, 32), dtype=bool)
                    x1, y1, x2, y2 = map(int, source_detection.box_xyxy)
                    mask[y1:y2, x1:x2] = True
                    rows.append(Detection(source_detection.label, source_detection.box_xyxy, source_detection.score, mask=mask, track_id=object_id))
                yield index, rows

    detector, tracker = Detector(), Tracker()
    cache_path = tmp_path / "tracks.json"
    annotate_media(AnnotationConfig(source, "dog running", tmp_path / "first.mp4", prompt_mode="referring", chunk_frames=2, redetect_every=2, track_cache_path=cache_path), detector=detector, video_tracker=tracker)
    cache = load_cache(cache_path)
    assert {track.label for track in cache.tracks} == {"dog", "cat"}
    assert all(track.temporal_features for track in cache.tracks)
    first_calls = (detector.calls, tracker.calls)
    annotate_media(AnnotationConfig(source, "dog sitting", tmp_path / "second.mp4", prompt_mode="referring", track_cache_path=cache_path, reuse_track_cache=True), detector=detector, video_tracker=tracker)
    assert (detector.calls, tracker.calls) == first_calls
