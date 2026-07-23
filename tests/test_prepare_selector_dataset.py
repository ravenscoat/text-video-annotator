from pathlib import Path

import json

from scripts.prepare_selector_dataset import target_masks


def test_target_masks_union_is_frame_indexed():
    assert target_masks({"1": [None, None]}, ["1"], 2) == {}
