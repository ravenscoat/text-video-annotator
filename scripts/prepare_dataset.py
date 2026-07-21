"""Create small deterministic manifests from locally downloaded benchmark data.

This script intentionally never downloads or rewrites source annotations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _annotation_file(root: Path, dataset: str) -> Path:
    candidates = {
        "lv-vis": ["annotations.json", "instances.json", "train.json", "valid.json"],
        "youtube-vis": ["instances_train.json", "instances_valid.json", "train.json", "valid.json"],
        "refer-youtube-vos": ["meta_expressions.json"],
    }[dataset]
    for name in candidates:
        matches = list(root.rglob(name))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Could not find an annotation JSON below {root}")


def _records(payload: dict):
    if isinstance(payload.get("videos"), list):
        videos = {item.get("id", item.get("video_id")): item for item in payload["videos"]}
        annotations = payload.get("annotations", [])
        for annotation in annotations:
            video_id = annotation.get("video_id")
            yield {"video_id": video_id, "category_id": annotation.get("category_id"), "annotation": annotation, "video": videos.get(video_id, {})}
    elif isinstance(payload.get("videos"), dict):
        for video_id, video in payload["videos"].items():
            yield {"video_id": video_id, "category_id": None, "annotation": video, "video": video}
    elif isinstance(payload.get("meta_expressions"), dict):
        for video_id, video in payload["meta_expressions"].items():
            yield {"video_id": video_id, "category_id": None, "annotation": video, "video": video}
    else:
        raise ValueError("Unsupported annotation schema; add an adapter for this dataset version")


def prepare(dataset: str, root: Path, output: Path, max_videos: int) -> dict:
    annotation_path = _annotation_file(root, dataset)
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    categories = {item["id"]: item["name"] for item in payload.get("categories", []) if "id" in item and "name" in item}
    selected = []
    seen = set()
    for record in _records(payload):
        video_id = record["video_id"]
        if video_id in seen:
            continue
        category = categories.get(record["category_id"], "unknown")
        selected.append({"video_id": video_id, "category": category, "annotation_file": str(annotation_path)})
        seen.add(video_id)
        if len(selected) >= max_videos:
            break
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"dataset": dataset, "videos": selected}, indent=2), encoding="utf-8")
    return {"annotation_file": str(annotation_path), "videos": len(selected), "categories": sorted({x["category"] for x in selected})}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["lv-vis", "refer-youtube-vos", "youtube-vis"], required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-videos", type=int, default=60)
    parser.add_argument("--strategy", default="balanced")
    args = parser.parse_args()
    print(json.dumps(prepare(args.dataset, args.dataset_root, args.output, args.max_videos), indent=2))


if __name__ == "__main__":
    main()
