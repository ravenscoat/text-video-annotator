import json
from pathlib import Path

from scripts.prepare_mevis_diagnostic import prepare


def test_diagnostic_manifest_is_deterministic_and_covers_buckets(tmp_path: Path):
    source = tmp_path / "manifest.json"
    source.write_text(json.dumps({"dataset": "fixture", "cases": [
        {"video_id": "a", "prompt": "the person running left"},
        {"video_id": "b", "prompt": "the dog and horse"},
        {"video_id": "c", "prompt": "the bird near the car"},
        {"video_id": "d", "prompt": "the person who enters later"},
        {"video_id": "e", "prompt": "the red bicycle"},
    ]}), encoding="utf-8")
    output = tmp_path / "diagnostic.json"
    result = prepare(source, output, 5)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert result["cases"] == 5
    assert {case["diagnostic_bucket"] for case in payload["cases"]} == {"multi_target", "motion", "relation", "temporal", "single_target"}
    assert [case["source_index"] for case in payload["cases"]] == [1, 0, 2, 3, 4]
    assert "candidate_missing" in payload["candidate_recall_definition"]
