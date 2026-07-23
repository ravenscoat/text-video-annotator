import json
from pathlib import Path

from scripts.write_mevis_identity import write_identity


def test_identity_report_hashes_files_and_freezes_cases(tmp_path: Path):
    manifest = tmp_path / "manifest.json"; manifest.write_text(json.dumps({"dataset": "fixture", "cases": [{"video_id": "v1", "expression_id": "e1", "prompt": "object", "annotation_ids": ["1"]}]}), encoding="utf-8")
    meta = tmp_path / "meta.json"; meta.write_text(json.dumps({"videos": {"v1": {"expressions": {"e1": {"exp": "object"}}}}}), encoding="utf-8")
    masks = tmp_path / "masks.json"; masks.write_text("{}", encoding="utf-8")
    report = write_identity(manifest, meta, masks, tmp_path / "identity.json")
    assert report["manifest"]["sha256"] and report["metadata"]["sha256"] and report["masks"]["sha256"]
    assert report["manifest"]["case_count"] == 1
    assert report["case_ids"][0]["video_id"] == "v1"
