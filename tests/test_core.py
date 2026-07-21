import json
from pathlib import Path

from video_annotator.identity import associate, iou
from video_annotator.types import Detection


def test_iou_and_identity_association():
    assert iou((0, 0, 10, 10), (5, 5, 15, 15)) > 0.1
    previous = [Detection("cup", (0, 0, 10, 10), 0.9, track_id=7)]
    current, next_id = associate(previous, [Detection("cup", (1, 1, 11, 11), 0.8)], 8)
    assert current[0].track_id == 7
    assert next_id == 8


def test_manifest_script_with_coco_fixture(tmp_path: Path):
    root = tmp_path / "raw"
    root.mkdir()
    payload = {"categories": [{"id": 1, "name": "cup"}], "videos": [{"id": 4}], "annotations": [{"video_id": 4, "category_id": 1}]}
    (root / "instances.json").write_text(json.dumps(payload), encoding="utf-8")
    from scripts.prepare_dataset import prepare
    result = prepare("lv-vis", root, tmp_path / "manifest.json", 10)
    assert result["videos"] == 1
    assert json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))["videos"][0]["category"] == "cup"
