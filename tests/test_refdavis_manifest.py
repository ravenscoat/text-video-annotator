import json
from pathlib import Path

import cv2
import numpy as np

from scripts.prepare_refdavis_manifest import prepare


def test_refdavis_manifest_records_expression_and_indexed_masks(tmp_path: Path):
    split = tmp_path / "valid"
    frames = split / "JPEGImages" / "bear"
    masks = split / "Annotations" / "bear"
    frames.mkdir(parents=True)
    masks.mkdir(parents=True)
    cv2.imwrite(str(frames / "00000.jpg"), np.zeros((8, 8, 3), dtype=np.uint8))
    cv2.imwrite(str(masks / "00000.png"), np.ones((8, 8), dtype=np.uint8))
    meta_path = split / "meta_expressions.json"
    meta_path.write_text(json.dumps({"videos": {"bear": {"frames": ["00000"], "expressions": {"4": {"exp": "the bear on the left", "obj_id": "2"}}}}}), encoding="utf-8")
    output = tmp_path / "manifest.json"
    result = prepare(meta_path, split, output, max_items=1)
    case = json.loads(output.read_text(encoding="utf-8"))["cases"][0]
    assert result["cases"] == 1
    assert case["prompt"] == "the bear on the left"
    assert case["object_id"] == "2"
    assert case["mask_encoding"] == "indexed_png"
