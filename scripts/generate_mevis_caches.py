"""Generate candidate caches for a frozen MeViS expression manifest."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from video_annotator.config import AnnotationConfig
from video_annotator.detector import GroundingDinoDetector
from video_annotator.pipeline import annotate_media
from video_annotator.tracker import Sam2VideoTracker
from video_annotator.track_cache import load_cache


def make_video(case: dict, path: Path, fps: float = 10.0) -> None:
    frame_dir = Path(case["video_dir"])
    frames = [str(name) for name in case["frame_names"]]
    first = cv2.imread(str(frame_dir / f"{frames[0]}.jpg"), cv2.IMREAD_COLOR)
    if first is None:
        raise FileNotFoundError(f"Cannot decode MeViS frame {frame_dir / frames[0]}.jpg")
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create temporary video {path}")
    try:
        for frame_name in frames:
            image = cv2.imread(str(frame_dir / f"{frame_name}.jpg"), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f"Cannot decode MeViS frame {frame_dir / frame_name}.jpg")
            writer.write(image)
    finally:
        writer.release()


def generate(manifest: Path, output: Path, semantic_model: Path, long_side: int = 512, device: str = "cuda") -> dict:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    inputs = output / "inputs"; inputs.mkdir(exist_ok=True)
    caches = output / "caches"; caches.mkdir(exist_ok=True)
    renders = output / "renders"; renders.mkdir(exist_ok=True)
    detector = GroundingDinoDetector(device=device)
    tracker = Sam2VideoTracker(device=device)
    results = []
    for index, case in enumerate(payload.get("cases", []), 1):
        stem = f"{index:02d}_{case['video_id']}_{case['expression_id']}"
        input_video = inputs / f"{stem}.mp4"; cache_path = caches / f"{stem}.json"; render_path = renders / f"{stem}.mp4"
        if cache_path.exists() and render_path.exists():
            try:
                cached = load_cache(cache_path)
                results.append({"index": index, "video_id": case["video_id"], "expression_id": case["expression_id"], "prompt": case["prompt"], "cache": str(cache_path), "frame_count": int(cached.source["frame_count"]), "objects_found": len(cached.tracks), "status": "reused"})
                print(json.dumps(results[-1]), flush=True)
                (output / "generation_report.json").write_text(json.dumps({"manifest": str(manifest.resolve()), "case_count": len(results), "cases": results}, indent=2), encoding="utf-8")
                continue
            except Exception:
                pass
        make_video(case, input_video)
        result = annotate_media(AnnotationConfig(input_video, case["prompt"], render_path, prompt_mode="referring", long_side=long_side, chunk_frames=30, redetect_every=15, device=device, track_cache_path=cache_path, semantic_model_id=str(semantic_model), semantic_device="cpu"), detector=detector, video_tracker=tracker)
        results.append({"index": index, "video_id": case["video_id"], "expression_id": case["expression_id"], "prompt": case["prompt"], "cache": str(cache_path), "frame_count": result.frame_count, "objects_found": result.objects_found, "status": "generated"})
        print(json.dumps(results[-1]), flush=True)
        (output / "generation_report.json").write_text(json.dumps({"manifest": str(manifest.resolve()), "case_count": len(results), "cases": results}, indent=2), encoding="utf-8")
    report = {"manifest": str(manifest.resolve()), "case_count": len(results), "cases": results}
    (output / "generation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--semantic-model", type=Path, required=True)
    parser.add_argument("--long-side", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(json.dumps(generate(args.manifest, args.output, args.semantic_model, args.long_side, args.device), indent=2))


if __name__ == "__main__":
    main()
