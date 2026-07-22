"""Prepare phrase-prompt cases from a local Refer-YouTube-VOS split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _video_dir(root: Path, video_id: str) -> Path:
    for candidate in (root / "JPEGImages" / video_id, root / "JPEGImages" / "train" / video_id, root / "JPEGImages" / "valid" / video_id, root / video_id):
        if candidate.is_dir():
            return candidate
    return root / "JPEGImages" / video_id


def _expressions(video: dict):
    raw = video.get("expressions", {})
    if isinstance(raw, list):
        return raw
    result = []
    for object_id, entries in raw.items():
        if isinstance(entries, dict):
            entries = [entries]
        for index, entry in enumerate(entries or []):
            if isinstance(entry, str):
                entry = {"exp": entry}
            result.append({**entry, "obj_id": str(entry.get("obj_id", object_id)), "exp_id": str(entry.get("exp_id", f"{object_id}_{index}"))})
    return result


def prepare(meta_path: Path, split_root: Path, output: Path, max_items: int = 10, allow_missing_media: bool = False) -> dict:
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    videos = payload.get("videos", payload if isinstance(payload, dict) else {})
    candidates = []
    for video_id, video in videos.items():
        if not isinstance(video, dict):
            continue
        video_dir = _video_dir(split_root, str(video_id))
        frame_names = [Path(name).name for name in (video.get("frames") or video.get("file_names") or [])]
        if not frame_names and video_dir.is_dir():
            frame_names = sorted(path.name for path in video_dir.glob("*.jpg"))
        if not allow_missing_media and (not video_dir.is_dir() or not frame_names or not (video_dir / frame_names[0]).is_file()):
            continue
        for expression in _expressions(video):
            prompt = str(expression.get("exp", expression.get("expression", ""))).strip()
            if prompt:
                candidates.append((str(video_id), video_dir, frame_names, expression, prompt))
    candidates.sort(key=lambda item: (item[0], item[4], item[3].get("exp_id", "")))
    selected = candidates[:max_items]
    cases = [{
        "video_id": video_id,
        "video_dir": str(video_dir),
        "frame_names": frame_names,
        "prompt": prompt,
        "expression_id": str(expression.get("exp_id", "")),
        "object_id": str(expression.get("obj_id", "")),
    } for video_id, video_dir, frame_names, expression, prompt in selected]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"dataset": "refer-youtube-vos", "meta_file": str(meta_path), "cases": cases}, indent=2), encoding="utf-8")
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
