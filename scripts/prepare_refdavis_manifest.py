"""Prepare Ref-DAVIS17 referring-expression cases from a local split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _find_dir(root: Path, name: str, video_id: str) -> Path:
    candidates = (
        root / name / video_id,
        root / "valid" / name / video_id,
        root / "train" / name / video_id,
        root / video_id,
    )
    return next((path for path in candidates if path.is_dir()), candidates[0])


def _expressions(video: dict):
    raw = video.get("expressions", {})
    if isinstance(raw, list):
        raw = {str(i): value for i, value in enumerate(raw)}
    for expression_id, entry in raw.items():
        if isinstance(entry, str):
            entry = {"exp": entry}
        if not isinstance(entry, dict):
            continue
        prompt = str(entry.get("exp", entry.get("expression", ""))).strip()
        if prompt:
            yield str(expression_id), entry, prompt


def prepare(meta_path: Path, split_root: Path, output: Path, max_items: int = 10, allow_missing_media: bool = False) -> dict:
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    videos = payload.get("videos", payload if isinstance(payload, dict) else {})
    candidates = []
    for video_id, video in videos.items():
        if not isinstance(video, dict):
            continue
        video_id = str(video_id)
        frame_dir = _find_dir(split_root, "JPEGImages", video_id)
        annotation_dir = _find_dir(split_root, "Annotations", video_id)
        frame_names = [Path(name).name for name in (video.get("frames") or video.get("file_names") or [])]
        if frame_names and not Path(frame_names[0]).suffix:
            frame_names = [name + ".jpg" for name in frame_names]
        if not frame_names and frame_dir.is_dir():
            frame_names = sorted(path.name for path in frame_dir.glob("*.jpg"))
        if not allow_missing_media and (not frame_dir.is_dir() or not frame_names or not (frame_dir / frame_names[0]).is_file()):
            continue
        for expression_id, expression, prompt in _expressions(video):
            object_id = str(expression.get("obj_id", expression.get("object_id", expression_id)))
            candidates.append((video_id, frame_dir, annotation_dir, frame_names, expression_id, object_id, prompt))
    candidates.sort(key=lambda item: (item[0], item[6], item[4]))
    selected = candidates[:max_items]
    cases = [
        {
            "video_id": video_id,
            "video_dir": str(frame_dir),
            "annotation_dir": str(annotation_dir),
            "frame_names": frame_names,
            "prompt": prompt,
            "expression_id": expression_id,
            "object_id": object_id,
            "mask_encoding": "indexed_png",
        }
        for video_id, frame_dir, annotation_dir, frame_names, expression_id, object_id, prompt in selected
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"dataset": "ref-davis17", "meta_file": str(meta_path), "cases": cases}, indent=2), encoding="utf-8")
    return {"cases": len(cases), "videos": len({case["video_id"] for case in cases}), "meta_file": str(meta_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", type=Path, required=True)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-items", type=int, default=10)
    parser.add_argument("--allow-missing-media", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare(args.meta, args.split_root, args.output, args.max_items, args.allow_missing_media), indent=2))


if __name__ == "__main__":
    main()
