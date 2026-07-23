"""Write a reproducible identity report for the exact MeViS files in use."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_identity(manifest: Path, meta: Path, masks: Path, output: Path) -> dict:
    manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
    meta_data = json.loads(meta.read_text(encoding="utf-8"))
    videos = meta_data.get("videos", meta_data)
    cases = manifest_data.get("cases", [])
    expressions = sum(len(video.get("expressions", {})) for video in videos.values() if isinstance(video, dict))
    identity = {
        "dataset": "MeViS",
        "release_label": "local metadata file; verify against the upstream release before leaderboard claims",
        "split": manifest_data.get("dataset", "mevis-valid-u"),
        "manifest": {"path": str(manifest.resolve()), "sha256": sha256(manifest), "case_count": len(cases)},
        "metadata": {"path": str(meta.resolve()), "sha256": sha256(meta), "video_count": len(videos), "expression_count": expressions},
        "masks": {"path": str(masks.resolve()), "sha256": sha256(masks), "size_bytes": masks.stat().st_size},
        "case_video_count": len({str(case.get("video_id")) for case in cases}),
        "audio_used": False,
        "no_target_examples": any(not case.get("annotation_ids") for case in cases),
        "metrics": "Diagnostic region Jaccard and boundary F over selected expressions; not official MeViS leaderboard J&F.",
        "case_ids": [{"video_id": str(case.get("video_id")), "expression_id": str(case.get("expression_id")), "prompt": str(case.get("prompt", ""))} for case in cases],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(identity, indent=2), encoding="utf-8")
    return identity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--meta", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = write_identity(args.manifest, args.meta, args.masks, args.output)
    print(json.dumps({"manifest_sha256": result["manifest"]["sha256"], "metadata_sha256": result["metadata"]["sha256"], "mask_sha256": result["masks"]["sha256"], "cases": result["manifest"]["case_count"]}, indent=2))


if __name__ == "__main__":
    main()
