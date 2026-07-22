"""Prepare a deterministic LV-VIS video/category manifest from local files.

This script never downloads or rewrites LV-VIS media or annotations. The
dataset is licensed for non-commercial research; keep it outside Git.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def prompt_for_category(category: dict) -> str:
    synonyms = category.get("synonyms") or []
    prompt = str(synonyms[0] if synonyms else category.get("name", "object"))
    prompt = re.sub(r"\s*\([^)]*\)", "", prompt).replace("_", " ")
    return re.sub(r"\s+", " ", prompt).strip(" _") or "object"


def _video_dir(dataset_root: Path, video: dict) -> Path:
    video_id = str(video.get("id", video.get("video_id")))
    candidates = [
        dataset_root / "JPEGImages" / "val" / video_id,
        dataset_root / "JPEGImages" / "val" / video_id.zfill(5),
        dataset_root / "val" / "JPEGImages" / video_id,
        dataset_root / "val" / "JPEGImages" / video_id.zfill(5),
        dataset_root / "val" / video_id,
        dataset_root / video_id,
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]


def _frame_names(video: dict) -> list[str]:
    raw = video.get("file_names") or video.get("frames") or []
    names = []
    for item in raw:
        if isinstance(item, (list, tuple)):
            names.extend(str(value) for value in item)
        else:
            names.append(str(item))
    return names


def prepare(annotation_path: Path, dataset_root: Path, output: Path, max_items: int = 10, require_media: bool = True) -> dict:
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    categories = {item["id"]: item for item in payload.get("categories", []) if "id" in item}
    videos = {str(item.get("id", item.get("video_id"))): item for item in payload.get("videos", [])}
    annotations = [item for item in payload.get("annotations", []) if item.get("category_id") in categories]
    candidates = []
    for annotation in annotations:
        video_id = str(annotation.get("video_id"))
        video = videos.get(video_id, {})
        video_dir = _video_dir(dataset_root, video)
        frame_names = _frame_names(video)
        if require_media and (not video_dir.is_dir() or not frame_names or not (video_dir / Path(frame_names[0]).name).is_file()):
            continue
        candidates.append((str(categories[annotation["category_id"]].get("name", "")), video_id, annotation, video_dir, frame_names))
    candidates.sort(key=lambda item: (item[0], item[1], int(item[2].get("id", 0))))
    selected = []
    seen_categories = set()
    for candidate in candidates:
        if candidate[2]["category_id"] in seen_categories:
            continue
        selected.append(candidate)
        seen_categories.add(candidate[2]["category_id"])
        if len(selected) >= max_items:
            break
    if len(selected) < max_items:
        selected.extend(candidate for candidate in candidates if candidate not in selected)
    selected = selected[:max_items]
    cases = []
    for category_name, video_id, annotation, video_dir, frame_names in selected:
        category = categories[annotation["category_id"]]
        cases.append({
            "video_id": video_id,
            "video_dir": str(video_dir),
            "frame_names": [Path(name).name for name in frame_names],
            "category_id": annotation["category_id"],
            "category_name": category_name,
            "prompt": prompt_for_category(category),
            "annotation_id": annotation.get("id"),
            "segmentations": annotation.get("segmentations", []),
            "bboxes": annotation.get("bboxes", []),
            "length": video.get("length", len(frame_names)),
            "width": video.get("width"),
            "height": video.get("height"),
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {"dataset": "lv-vis", "annotation_file": str(annotation_path), "cases": cases}
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return {"cases": len(cases), "categories": sorted({case["category_name"] for case in cases}), "annotation_file": str(annotation_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-items", type=int, default=10)
    parser.add_argument("--allow-missing-media", action="store_true")
    args = parser.parse_args()
    if args.max_items < 1:
        parser.error("--max-items must be positive")
    print(json.dumps(prepare(args.annotations, args.dataset_root, args.output, args.max_items, not args.allow_missing_media), indent=2))


if __name__ == "__main__":
    main()
