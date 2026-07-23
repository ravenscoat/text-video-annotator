from __future__ import annotations

from pathlib import Path
import tempfile
import cv2
import numpy as np

from .config import AnnotationConfig
from .detector import GroundingDinoDetector
from .exporters import JsonExporter, save_masks
from .identity import TrackManager, TrackState
from .prompts import parse_prompt
from .referring import select_tracks
from .render import render
from .tracker import Sam2ImageSegmenter, Sam2VideoTracker
from .track_cache import CachedTrack, TrackCache, TrackFrame, load_cache, make_proposal_scope, save_cache
from .track_features import populate_cached_features
from .semantic_encoder import FrozenClipEncoder
from .learned_selector import score_cached_tracks
from .types import AnnotationResult, Detection
from .video_io import is_video, iter_video_chunks, make_writer, probe


def load_compatible_track_cache(path: Path, input_path: Path, prompt_spec, info):
    """Load a source-matched cache whose candidates can cover the request."""
    source_id = f"{input_path.resolve()}:{info.width}x{info.height}:{info.frame_count}:{info.fps:.6f}"
    cache = load_cache(path, expected_source_id=source_id)
    normalize = lambda value: " ".join(str(value).lower().strip(" .,;:!?").split()).rstrip("s")
    available = {normalize(value) for value in cache.proposal_scope.get("targets", [])}
    available.update(normalize(track.label) for track in cache.tracks)
    if prompt_spec.mode == "category_union":
        missing = {normalize(value) for value in prompt_spec.targets} - available
        if missing:
            raise ValueError(f"Track cache proposal scope is missing requested targets: {', '.join(sorted(missing))}")
    elif not any(label and label in normalize(prompt_spec.raw) for label in available) and "object " not in prompt_spec.raw.lower():
        raise ValueError("Track cache has no candidate label compatible with this referring expression")
    return cache


def redetection_indices(frame_count: int, redetect_every: int) -> list[int]:
    """Return global frame indices at which detector refreshes are required."""
    if frame_count < 1 or redetect_every < 1:
        return []
    return list(range(0, frame_count, redetect_every))


def _cached_track_states(cache: TrackCache) -> list[TrackState]:
    states = []
    for track in cache.tracks:
        frames = [frame for frame in track.frames if frame.visible]
        if not frames:
            continue
        last = frames[-1]
        state = TrackState(track.track_id, track.label, track.detector_score, last.box_xyxy or (0.0, 0.0, 0.0, 0.0), None, last.frame_index)
        state.centers.extend((frame.frame_index, frame.center_xy[0], frame.center_xy[1]) for frame in frames if frame.center_xy is not None)
        state.areas.extend((frame.frame_index, frame.area or 0.0) for frame in frames)
        states.append(state)
    return states


def _mask_overlap(first: np.ndarray, second: np.ndarray) -> float:
    union = np.logical_or(first, second).sum()
    return float(np.logical_and(first, second).sum() / union) if union else 0.0


def annotate_from_track_cache(config: AnnotationConfig) -> AnnotationResult:
    """Render a compatible candidate cache without loading detector or SAM 2."""
    config.validate()
    if config.track_cache_path is None:
        raise ValueError("track_cache_path is required when reusing a candidate cache")
    prompt_spec = parse_prompt(config.prompt, config.prompt_mode, config.targets)
    info = probe(config.input_path)
    if not info.is_video:
        raise ValueError("Track caches currently support videos only")
    cache = load_compatible_track_cache(config.track_cache_path, config.input_path, prompt_spec, info)
    tracks = _cached_track_states(cache)
    learned_scores = {}
    if config.selector_checkpoint:
        if not config.semantic_model_id:
            raise ValueError("semantic_model_id is required with selector_checkpoint")
        encoder = FrozenClipEncoder(config.semantic_model_id, config.semantic_device)
        learned_scores = score_cached_tracks(cache, config.prompt, config.selector_checkpoint, encoder, config.semantic_device)
    if prompt_spec.mode == "category_union":
        requested = {item.lower().rstrip("s") for item in prompt_spec.targets}
        selected = {track.track_id: None for track in tracks if any(target in track.label.lower().rstrip("s") for target in requested)}
    else:
        selected = {item.track_id: item for item in select_tracks(tracks, prompt_spec.motion_text or prompt_spec.raw, frame_count=info.frame_count)}
    if learned_scores and prompt_spec.mode != "category_union":
        selected = {track.track_id: selected.get(track.track_id) for track in tracks if track.track_id in selected and learned_scores.get(track.track_id, 0.0) >= config.selector_threshold}
    by_frame: dict[int, list[tuple[CachedTrack, TrackFrame]]] = {}
    for track in cache.tracks:
        if track.track_id not in selected:
            continue
        for frame in track.frames:
            if frame.visible:
                by_frame.setdefault(frame.frame_index, []).append((track, frame))
    writer = make_writer(config.output_path, info)
    exporter = JsonExporter(config.export_json, config.prompt, info) if config.export_json else None
    count = found = 0
    try:
        cap = cv2.VideoCapture(str(config.input_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {config.input_path}")
        try:
            while count < info.frame_count:
                ok, original_bgr = cap.read()
                if not ok:
                    break
                detections = []
                frame_candidates = []
                for track, cached_frame in by_frame.get(count, []):
                    mask = cv2.imread(str(cached_frame.mask_path), cv2.IMREAD_GRAYSCALE) if cached_frame.mask_path else None
                    if mask is None:
                        continue
                    frame_candidates.append((track, cached_frame, mask > 0))
                kept_masks = []
                for track, cached_frame, mask_bool in sorted(frame_candidates, key=lambda item: item[0].detector_score, reverse=True):
                    if any(_mask_overlap(mask_bool, prior) >= 0.90 for prior in kept_masks):
                        continue
                    kept_masks.append(mask_bool)
                    detection = Detection(track.label, cached_frame.box_xyxy or (0.0, 0.0, float(info.width), float(info.height)), track.detector_score, mask=mask_bool, track_id=track.track_id)
                    selection = selected.get(track.track_id)
                    if selection is not None:
                        detection.selection_score = selection.score
                        detection.selection_reasons = selection.reasons
                    detections.append(detection)
                found += len(detections)
                writer.write(render(original_bgr, detections))
                if config.export_masks:
                    save_masks(config.export_masks, count, detections)
                if exporter:
                    exporter.add(count, count / info.fps, detections)
                count += 1
        finally:
            cap.release()
    finally:
        writer.release()
        if exporter:
            exporter.close()
    return AnnotationResult(str(config.output_path), "video", count, found, str(config.export_json) if config.export_json else None)


def annotate_media(config: AnnotationConfig, detector=None, image_segmenter=None, video_tracker=None) -> AnnotationResult:
    config.validate()
    if config.reuse_track_cache:
        return annotate_from_track_cache(config)
    prompt_spec = parse_prompt(config.prompt, config.prompt_mode, config.targets)
    detector_prompt = prompt_spec.detector_prompt if prompt_spec.mode == "category_union" else prompt_spec.raw
    info = probe(config.input_path)
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    detector = detector or GroundingDinoDetector(device=config.device)
    exporter = JsonExporter(config.export_json, config.prompt or detector_prompt, info) if config.export_json else None
    if not info.is_video:
        image = cv2.imread(str(config.input_path), cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detections = detector.detect(image_rgb, detector_prompt, config.box_threshold, config.text_threshold, config.max_objects, prompt_spec.targets, config.max_objects_per_target)
        segmenter = image_segmenter or Sam2ImageSegmenter(device=config.device)
        detections = segmenter.segment(image_rgb, detections)
        output = render(image, detections)
        config.output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(config.output_path), output)
        if config.export_masks:
            save_masks(config.export_masks, 0, detections)
        if exporter:
            exporter.add(0, 0.0, detections)
            exporter.close()
        return AnnotationResult(str(config.output_path), "image", 1, len(detections), str(config.export_json) if config.export_json else None)

    writer = make_writer(config.output_path, info)
    tracker = video_tracker or Sam2VideoTracker(device=config.device)
    track_manager = TrackManager()
    cached_tracks: dict[int, CachedTrack] = {}
    cache_targets = list(prompt_spec.targets) or [prompt_spec.raw]
    cache_scope = make_proposal_scope(cache_targets, [detector_prompt or prompt_spec.raw])
    source_id = f"{config.input_path.resolve()}:{info.width}x{info.height}:{info.frame_count}:{info.fps:.6f}"
    cache_complete = False
    count, found = 0, 0
    next_detection_frame = 0
    try:
        for start, chunk in iter_video_chunks(config.input_path, config.chunk_frames, config.long_side):
            offset = 0
            while offset < len(chunk):
                frame_index = start + offset
                window_end = min(offset + config.redetect_every, len(chunk))
                window = chunk[offset:window_end]
                first_rgb = window[0][1]
                # Each window begins with a detector refresh. With normal
                # settings (redetect_every <= chunk_frames), these are exact
                # global indices 0, N, 2N, ...; TrackManager preserves IDs.
                try:
                    detections = detector.detect(first_rgb, detector_prompt, config.box_threshold, config.text_threshold, config.max_objects, prompt_spec.targets, config.max_objects_per_target)
                except TypeError:
                    # Compatibility for user-provided detectors using the old
                    # five-argument adapter signature.
                    detections = detector.detect(first_rgb, detector_prompt, config.box_threshold, config.text_threshold, config.max_objects)
                detections = track_manager.update(detections, frame_index)
                selected = {}
                if prompt_spec.mode == "referring":
                    selected = {
                        item.track_id: item
                        for item in select_tracks(
                            track_manager.tracks.values(),
                            prompt_spec.motion_text or prompt_spec.raw,
                            frame_count=info.frame_count,
                        )
                    }
                next_detection_frame = frame_index + config.redetect_every
                if not detections:
                    for local_offset, (original_bgr, _) in enumerate(window):
                        current_index = frame_index + local_offset
                        writer.write(original_bgr)
                        if exporter:
                            exporter.add(current_index, current_index / info.fps, [])
                        count += 1
                    offset = window_end
                    continue
                found += len(detections)
                with tempfile.TemporaryDirectory(prefix="sam2_window_") as temp_dir:
                    frame_dir = Path(temp_dir)
                    for local_offset, (_, rgb) in enumerate(window):
                        cv2.imwrite(str(frame_dir / f"{local_offset:06d}.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                    propagated = {frame_idx: frame_detections for frame_idx, frame_detections in tracker.propagate(frame_dir, detections)}
                for local_offset, (original_bgr, _) in enumerate(window):
                    current_index = frame_index + local_offset
                    output_detections = propagated.get(local_offset, [])
                    local_to_global = {index + 1: detections[index].track_id for index in range(len(detections))}
                    for output_detection in output_detections:
                        output_detection.track_id = local_to_global.get(output_detection.track_id, output_detection.track_id)
                        if output_detection.mask is not None:
                            output_detection.mask = cv2.resize(output_detection.mask.astype(np.uint8), (original_bgr.shape[1], original_bgr.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
                            ys, xs = np.where(output_detection.mask)
                            if len(xs):
                                output_detection.box_xyxy = (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))
                        if config.track_cache_path is not None and output_detection.track_id is not None:
                            track = cached_tracks.setdefault(output_detection.track_id, CachedTrack(output_detection.track_id, output_detection.label, float(output_detection.score)))
                            mask_path = None
                            if output_detection.mask is not None:
                                mask_root = config.track_cache_path.with_name(config.track_cache_path.stem + "_masks")
                                mask_root.mkdir(parents=True, exist_ok=True)
                                mask_file = mask_root / f"frame_{current_index:06d}_object_{output_detection.track_id:04d}.png"
                                cv2.imwrite(str(mask_file), output_detection.mask.astype(np.uint8) * 255)
                                mask_path = str(mask_file)
                            box = tuple(float(value) for value in output_detection.box_xyxy)
                            center = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
                            area = float(output_detection.mask.sum()) if output_detection.mask is not None else max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
                            track.frames.append(TrackFrame(current_index, True, box, mask_path, float(output_detection.score), center_xy=center, area=area))
                        if prompt_spec.mode == "referring":
                            selection = selected.get(output_detection.track_id)
                            if selection is None:
                                output_detection.track_id = None
                            else:
                                output_detection.selection_score = selection.score
                                output_detection.selection_reasons = selection.reasons
                    output_detections = [item for item in output_detections if item.track_id is not None]
                    writer.write(render(original_bgr, output_detections))
                    if config.export_masks:
                        save_masks(config.export_masks, current_index, output_detections)
                    if exporter:
                        exporter.add(current_index, current_index / info.fps, output_detections)
                    count += 1
                offset = window_end
        cache_complete = True
    finally:
        writer.release()
        if config.track_cache_path is not None and cache_complete:
            cache = TrackCache(
                source={"source_id": source_id, "path": str(config.input_path.resolve()), "width": info.width, "height": info.height, "fps": info.fps, "frame_count": info.frame_count},
                config={"long_side": config.long_side, "chunk_frames": config.chunk_frames, "redetect_every": config.redetect_every, "box_threshold": config.box_threshold, "text_threshold": config.text_threshold, "model_detector": "IDEA-Research/grounding-dino-tiny", "model_tracker": "facebook/sam2.1-hiera-tiny"},
                proposal_scope=cache_scope,
                tracks=list(cached_tracks.values()),
            )
            semantic_encoder = FrozenClipEncoder(config.semantic_model_id, config.semantic_device) if config.semantic_model_id else None
            populate_cached_features(cache, config.input_path, encoder=semantic_encoder)
            if semantic_encoder is not None:
                embedding_dim = next((len(token.image_embedding) for track in cache.tracks for token in track.anchor_tokens if token.image_embedding), 0)
                cache.config["semantic_model_id"] = config.semantic_model_id
                cache.config["semantic_device"] = config.semantic_device
                cache.config["semantic_embedding_dim"] = embedding_dim
            save_cache(config.track_cache_path, cache)
    if exporter:
        exporter.close()
    return AnnotationResult(str(config.output_path), "video", count, found, str(config.export_json) if config.export_json else None)
