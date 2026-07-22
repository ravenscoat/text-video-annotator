"""Prepare a deterministic MeViS Val-u expression manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def prepare(meta_path: Path, split_root: Path, masks_path: Path, output: Path, max_videos: int = 20, max_items: int | None = None) -> dict:
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    videos = payload.get("videos", payload)
    candidates = []
    for video_id, video in videos.items():
        if not isinstance(video, dict):
            continue
        frame_dir = split_root / "JPEGImages" / str(video_id)
        frames = [str(name) for name in (video.get("frames") or [])]
        if not frames and frame_dir.is_dir():
            frames = sorted(path.stem for path in frame_dir.glob("*.jpg"))
        if not frames or not frame_dir.is_dir() or not (frame_dir / f"{frames[0]}.jpg").is_file():
            continue
        for expression_id, expression in (video.get("expressions") or {}).items():
            if not isinstance(expression, dict) or not str(expression.get("exp", "")).strip():
                continue
            anno_ids = [str(value) for value in expression.get("anno_id", [])]
            object_ids = [str(value) for value in expression.get("obj_id", [])]
            if anno_ids:
                candidates.append((str(video_id), frame_dir, frames, str(expression_id), str(expression["exp"]).strip(), anno_ids, object_ids))
    candidates.sort(key=lambda item: (item[0], item[3]))
    selected_videos = sorted({item[0] for item in candidates})[:max_videos]
    selected = [item for item in candidates if item[0] in selected_videos]
    if max_items is not None:
        selected = selected[:max_items]
    cases = [{"video_id": vid, "video_dir": str(frame_dir), "frame_names": frames, "expression_id": exp_id, "prompt": prompt, "annotation_ids": anno_ids, "object_ids": object_ids, "mask_dict": str(masks_path), "mask_encoding": "coco_rle"} for vid, frame_dir, frames, exp_id, prompt, anno_ids, object_ids in selected]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"dataset": "mevis-valid-u", "meta_file": str(meta_path), "cases": cases}, indent=2), encoding="utf-8")
    return {"cases": len(cases), "videos": len(selected_videos), "output": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", type=Path, required=True)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-videos", type=int, default=20)
    parser.add_argument("--max-items", type=int)
    args = parser.parse_args()
    print(json.dumps(prepare(args.meta, args.split_root, args.masks, args.output, args.max_videos, args.max_items), indent=2))


if __name__ == "__main__":
    main()
