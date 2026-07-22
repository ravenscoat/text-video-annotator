import json
from pathlib import Path

import cv2
import numpy as np

from scripts.prepare_refytvos_manifest import prepare


def test_refytvos_manifest_preserves_phrase_and_object_id(tmp_path: Path):
    split = tmp_path / "valid"
    frames = split / "JPEGImages" / "video_a"
    frames.mkdir(parents=True)
    cv2.imwrite(str(frames / "00000.jpg"), np.zeros((8, 8, 3), dtype=np.uint8))
    meta = {"videos": {"video_a": {"frames": ["00000.jpg"], "expressions": {"3": [{"exp": "the person in red", "exp_id": "x", "obj_id": "3"}]}}}}
    meta_path = tmp_path / "meta_expressions.json"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    output = tmp_path / "manifest.json"
    result = prepare(meta_path, split, output, max_items=1)
    case = json.loads(output.read_text(encoding="utf-8"))["cases"][0]
    assert result["cases"] == 1
    assert case["prompt"] == "the person in red"
    assert case["object_id"] == "3"
