from __future__ import annotations

from pathlib import Path
import tempfile
import cv2
import numpy as np

from .config import AnnotationConfig
from .detector import GroundingDinoDetector
from .exporters import JsonExporter, save_masks
from .identity import TrackManager
from .prompts import parse_prompt
from .render import render
from .tracker import Sam2ImageSegmenter, Sam2VideoTracker
from .types import AnnotationResult
from .video_io import is_video, iter_video_chunks, make_writer, probe


def redetection_indices(frame_count: int, redetect_every: int) -> list[int]:
    """Return global frame indices at which detector refreshes are required."""
    if frame_count < 1 or redetect_every < 1:
        return []
    return list(range(0, frame_count, redetect_every))


def annotate_media(config: AnnotationConfig, detector=None, image_segmenter=None, video_tracker=None) -> AnnotationResult:
    config.validate()
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
                    writer.write(render(original_bgr, output_detections))
                    if config.export_masks:
                        save_masks(config.export_masks, current_index, output_detections)
                    if exporter:
                        exporter.add(current_index, current_index / info.fps, output_detections)
                    count += 1
                offset = window_end
    finally:
        writer.release()
    if exporter:
        exporter.close()
    return AnnotationResult(str(config.output_path), "video", count, found, str(config.export_json) if config.export_json else None)
