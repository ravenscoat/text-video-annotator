from __future__ import annotations

from pathlib import Path
import tempfile
import cv2
import numpy as np

from .config import AnnotationConfig
from .detector import GroundingDinoDetector
from .exporters import JsonExporter, save_masks
from .identity import associate
from .render import render
from .tracker import Sam2ImageSegmenter, Sam2VideoTracker
from .types import AnnotationResult
from .video_io import is_video, iter_video_chunks, make_writer, probe


def annotate_media(config: AnnotationConfig, detector=None, image_segmenter=None, video_tracker=None) -> AnnotationResult:
    config.validate()
    info = probe(config.input_path)
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    detector = detector or GroundingDinoDetector(device=config.device)
    exporter = JsonExporter(config.export_json, config.prompt, info) if config.export_json else None
    if not info.is_video:
        image = cv2.imread(str(config.input_path), cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detections = detector.detect(image_rgb, config.prompt, config.box_threshold, config.text_threshold, config.max_objects)
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
    previous, next_id, count, found = [], 1, 0, 0
    try:
        for start, chunk in iter_video_chunks(config.input_path, config.chunk_frames, config.long_side):
            first_rgb = chunk[0][1]
            detections = detector.detect(first_rgb, config.prompt, config.box_threshold, config.text_threshold, config.max_objects)
            detections, next_id = associate(previous, detections, next_id)
            previous = detections
            if not detections:
                for offset, (original_bgr, _) in enumerate(chunk):
                    frame_index = start + offset
                    writer.write(original_bgr)
                    if exporter:
                        exporter.add(frame_index, frame_index / info.fps, [])
                    count += 1
                continue
            found += len(detections)
            with tempfile.TemporaryDirectory(prefix="sam2_chunk_") as temp_dir:
                frame_dir = Path(temp_dir)
                for offset, (_, rgb) in enumerate(chunk):
                    # SAM 2's video loader expects numbered image files.
                    cv2.imwrite(str(frame_dir / f"{offset:06d}.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                propagated = {frame_idx: frame_detections for frame_idx, frame_detections in tracker.propagate(frame_dir, detections)}
            for offset, (original_bgr, rgb) in enumerate(chunk):
                frame_index = start + offset
                output_detections = propagated.get(offset, [])
                local_to_global = {index + 1: detections[index].track_id for index in range(len(detections))}
                for output_detection in output_detections:
                    output_detection.track_id = local_to_global.get(output_detection.track_id, output_detection.track_id)
                    if output_detection.mask is not None:
                        # SAM returns a mask in inference resolution; export at source resolution.
                        output_detection.mask = cv2.resize(output_detection.mask.astype(np.uint8), (original_bgr.shape[1], original_bgr.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
                        ys, xs = np.where(output_detection.mask)
                        if len(xs):
                            output_detection.box_xyxy = (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))
                writer.write(render(original_bgr, output_detections))
                if config.export_masks:
                    save_masks(config.export_masks, frame_index, output_detections)
                if exporter:
                    exporter.add(frame_index, frame_index / info.fps, output_detections)
                count += 1
    finally:
        writer.release()
    if exporter:
        exporter.close()
    return AnnotationResult(str(config.output_path), "video", count, found, str(config.export_json) if config.export_json else None)
