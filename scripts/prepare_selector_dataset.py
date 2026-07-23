"""Build selector JSONL samples from a manifest, cache-generation report and MeViS masks."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from video_annotator.selector_data import build_sample, write_selector_dataset
from video_annotator.track_cache import load_cache
from video_annotator.semantic_encoder import FrozenClipEncoder


def target_masks(mask_dict: dict, annotation_ids: list[str], frame_count: int) -> dict[int, np.ndarray]:
    from pycocotools import mask as mask_utils
    result: dict[int, np.ndarray] = {}
    for annotation_id in annotation_ids:
        frames = mask_dict.get(str(annotation_id), mask_dict.get(annotation_id, []))
        for index in range(min(frame_count, len(frames))):
            rle = frames[index]
            if rle is not None:
                decoded = mask_utils.decode(rle).astype(bool)
                result[index] = np.logical_or(result.get(index, np.zeros_like(decoded)), decoded)
    return result


def prepare(manifest: Path, generation_report: Path, masks: Path, output: Path, semantic_model: Path | None = None, semantic_device: str = "cpu", validation_fraction: float = 0.2) -> dict:
    manifest_data = json.loads(manifest.read_text(encoding="utf-8")); generation = json.loads(generation_report.read_text(encoding="utf-8")); mask_dict = json.loads(masks.read_text(encoding="utf-8"))
    encoder = FrozenClipEncoder(str(semantic_model), semantic_device) if semantic_model else None
    samples = []
    for case, generated in zip(manifest_data.get("cases", []), generation.get("cases", [])):
        cache = load_cache(Path(generated["cache"]))
        sample = build_sample(case, cache, target_masks(mask_dict, [str(value) for value in case.get("annotation_ids", [])], int(cache.source["frame_count"])), text_encoder=encoder)
        samples.append(sample)
    report = write_selector_dataset(samples, output, validation_fraction)
    report.update({"manifest": str(manifest.resolve()), "candidate_cache_report": str(generation_report.resolve()), "sample_count": len(samples), "positive_track_count": sum(sum(sample.labels) for sample in samples), "zero_positive_samples": sum(not any(sample.labels) for sample in samples)})
    report_path = output.with_suffix(".report.json"); report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--manifest", type=Path, required=True); parser.add_argument("--generation-report", type=Path, required=True); parser.add_argument("--masks", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--semantic-model", type=Path); parser.add_argument("--semantic-device", default="cpu"); parser.add_argument("--validation-fraction", type=float, default=.2)
    args = parser.parse_args(); print(json.dumps(prepare(args.manifest, args.generation_report, args.masks, args.output, args.semantic_model, args.semantic_device, args.validation_fraction), indent=2))


if __name__ == "__main__": main()
