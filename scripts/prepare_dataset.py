"""Create small deterministic manifests from locally downloaded benchmark data.

This script intentionally never downloads or rewrites source annotations.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _annotation_file(root: Path, dataset: str) -> Path:
    candidates = {
        "lvis": ["lvis_v1_val_subset.json", "lvis_v1_val.json", "lvis_v1_train.json"],
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
    if isinstance(payload.get("images"), list):
        annotations_by_image = {}
        for annotation in payload.get("annotations", []):
            annotations_by_image.setdefault(annotation.get("image_id"), []).append(annotation)
        for image in payload["images"]:
            annotations = annotations_by_image.get(image.get("id"), [])
            category_id = annotations[0].get("category_id") if annotations else None
            yield {"video_id": image.get("id"), "category_id": category_id, "annotation": annotations, "video": image}
    elif isinstance(payload.get("videos"), list):
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


def _lvis_prompt(category: dict) -> str:
    synonyms = category.get("synonyms") or []
    if synonyms:
        prompt = str(synonyms[0]).strip()
    else:
        prompt = str(category.get("name", "object")).replace("_", " ").strip()
    # Parenthetical taxonomy notes are usually not useful as visual prompts.
    prompt = re.sub(r"\s*\([^)]*\)", "", prompt).replace("_", " ").strip()
    prompt = re.sub(r"\s+", " ", prompt).strip(" _")
    return prompt or "object"


def prepare_lvis(root: Path, output: Path, max_items: int) -> dict:
    annotation_path = _annotation_file(root, "lvis")
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    categories = {item["id"]: item for item in payload.get("categories", []) if "id" in item}
    images = {item["id"]: item for item in payload.get("images", []) if "id" in item}
    grouped: dict[tuple[int, int], list[dict]] = {}
    for annotation in payload.get("annotations", []):
        key = (annotation.get("image_id"), annotation.get("category_id"))
        if key[0] in images and key[1] in categories:
            grouped.setdefault(key, []).append(annotation)

    # Deterministically spread the first pass over distinct categories. The
    # remaining sorted pairs fill the requested count if necessary.
    pairs = sorted(
        grouped,
        key=lambda pair: (str(categories[pair[1]].get("name", "")), pair[0], pair[1]),
    )
    selected = []
    seen_categories = set()
    for pair in pairs:
        if pair[1] in seen_categories:
            continue
        selected.append(pair)
        seen_categories.add(pair[1])
        if len(selected) >= max_items:
            break
    if len(selected) < max_items:
        selected.extend(pair for pair in pairs if pair not in selected[:max_items])
    selected = selected[:max_items]

    cases = []
    for image_id, category_id in selected:
        image = images[image_id]
        file_name = image.get("file_name") or Path(image["coco_url"].split("?", 1)[0]).name
        annotations = grouped[(image_id, category_id)]
        cases.append(
            {
                "image_id": image_id,
                "image_path": str(root / "images" / file_name),
                "category_id": category_id,
                "category_name": categories[category_id].get("name", "unknown"),
                "prompt": _lvis_prompt(categories[category_id]),
                "annotation_ids": [annotation["id"] for annotation in annotations if "id" in annotation],
                "annotation_file": str(annotation_path),
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {"dataset": "lvis", "cases": cases, "annotation_file": str(annotation_path)}
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return {
        "annotation_file": str(annotation_path),
        "cases": len(cases),
        "categories": sorted({case["category_name"] for case in cases}),
    }


def prepare(dataset: str, root: Path, output: Path, max_videos: int) -> dict:
    if dataset == "lvis":
        return prepare_lvis(root, output, max_videos)
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
    parser.add_argument("--dataset", choices=["lvis", "lv-vis", "refer-youtube-vos", "youtube-vis"], required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--max-videos", type=int, default=None, help="Deprecated alias for --max-items")
    parser.add_argument("--strategy", default="balanced")
    args = parser.parse_args()
    max_items = args.max_items if args.max_items is not None else (args.max_videos if args.max_videos is not None else 60)
    if max_items < 1:
        parser.error("--max-items must be positive")
    print(json.dumps(prepare(args.dataset, args.dataset_root, args.output, max_items), indent=2))


if __name__ == "__main__":
    main()
