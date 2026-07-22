import json
from pathlib import Path

import cv2
import numpy as np

from scripts.prepare_mevis_manifest import prepare


def test_mevis_manifest_keeps_rle_annotation_ids(tmp_path: Path):
    split = tmp_path / "valid_u"
    frame_dir = split / "JPEGImages" / "v1"
    frame_dir.mkdir(parents=True)
    cv2.imwrite(str(frame_dir / "00000.jpg"), np.zeros((8, 8, 3), dtype=np.uint8))
    meta = {"videos": {"v1": {"frames": ["00000"], "expressions": {"0": {"exp": "the object moving left", "obj_id": [1], "anno_id": ["a1"]}}}}}
    meta_path = split / "meta_expressions.json"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    masks = split / "mask_dict.json"
    masks.write_text(json.dumps({"a1": [None]}), encoding="utf-8")
    output = tmp_path / "manifest.json"
    result = prepare(meta_path, split, masks, output, max_videos=20)
    case = json.loads(output.read_text(encoding="utf-8"))["cases"][0]
    assert result["videos"] == 1
    assert case["annotation_ids"] == ["a1"]
    assert case["mask_encoding"] == "coco_rle"
