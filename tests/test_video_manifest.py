import json
from pathlib import Path

import cv2
import numpy as np

from scripts.prepare_video_manifest import prepare


def test_lvvis_manifest_selects_distinct_categories(tmp_path: Path):
    root = tmp_path / "dataset"
    frame_dir = root / "JPEGImages" / "val" / "00001"
    frame_dir.mkdir(parents=True)
    cv2.imwrite(str(frame_dir / "00000.jpg"), np.zeros((8, 8, 3), dtype=np.uint8))
    annotations = {
        "categories": [
            {"id": 1, "name": "cat", "synonyms": ["cat"]},
            {"id": 2, "name": "traffic_light", "synonyms": ["traffic light"]},
        ],
        "videos": [{"id": "00001", "file_names": [["00000.jpg"]], "length": 1, "width": 8, "height": 8}],
        "annotations": [
            {"id": 10, "video_id": "00001", "category_id": 1, "segmentations": [[[0, 0, 4, 0, 4, 4, 0, 4]]], "bboxes": [[0, 0, 4, 4]]},
            {"id": 11, "video_id": "00001", "category_id": 2, "segmentations": [[[2, 2, 6, 2, 6, 6, 2, 6]]], "bboxes": [[2, 2, 4, 4]]},
        ],
    }
    annotation_path = tmp_path / "val_instances.json"
    annotation_path.write_text(json.dumps(annotations), encoding="utf-8")
    output = tmp_path / "manifest.json"
    result = prepare(annotation_path, root, output, max_items=2)
    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert result["cases"] == 2
    assert {case["prompt"] for case in manifest["cases"]} == {"cat", "traffic light"}
    assert all(case["video_dir"] == str(frame_dir) for case in manifest["cases"])


def test_lvvis_manifest_can_be_prepared_before_media_download(tmp_path: Path):
    annotations = {"categories": [{"id": 1, "name": "cup"}], "videos": [{"id": 3, "file_names": ["000.jpg"]}], "annotations": [{"id": 1, "video_id": 3, "category_id": 1, "segmentations": [[]]}]}
    annotation_path = tmp_path / "annotations.json"
    annotation_path.write_text(json.dumps(annotations), encoding="utf-8")
    result = prepare(annotation_path, tmp_path / "missing", tmp_path / "manifest.json", max_items=1, require_media=False)
    assert result["cases"] == 1
