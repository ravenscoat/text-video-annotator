import numpy as np

from scripts.evaluate_lvis_subset import greedy_match, load_coco_masks, mask_iou
from scripts.prepare_dataset import _lvis_prompt


def test_mask_iou_examples():
    left = np.zeros((4, 4), dtype=bool)
    left[:2, :2] = True
    identical = left.copy()
    disjoint = np.zeros((4, 4), dtype=bool)
    disjoint[2:, 2:] = True
    partial = np.zeros((4, 4), dtype=bool)
    partial[1:3, 1:3] = True
    assert mask_iou(left, identical) == 1.0
    assert mask_iou(left, disjoint) == 0.0
    assert 0.0 < mask_iou(left, partial) < 1.0


def test_greedy_matching_is_one_to_one():
    target = np.zeros((3, 3), dtype=bool)
    target[:2, :2] = True
    prediction = target.copy()
    result = greedy_match([prediction, prediction], [target], threshold=0.5)
    assert result["true_positives"] == 1
    assert result["false_positives"] == 1
    assert result["false_negatives"] == 0


def test_lvis_polygon_and_rle_decode():
    from pycocotools import mask as mask_utils

    polygon = {"segmentation": [[0, 0, 3, 0, 3, 3, 0, 3]]}
    decoded_polygon = load_coco_masks(polygon, 4, 4)
    assert decoded_polygon.shape == (4, 4)
    assert decoded_polygon.sum() > 0

    source = np.zeros((4, 4), dtype=np.uint8)
    source[1:3, 1:3] = 1
    encoded = mask_utils.encode(np.asfortranarray(source))
    decoded_rle = load_coco_masks({"segmentation": encoded}, 4, 4)
    assert np.array_equal(decoded_rle, source.astype(bool))


def test_lvis_prompt_normalization():
    assert _lvis_prompt({"name": "car_(automobile)", "synonyms": ["car_(automobile)"]}) == "car"
    assert _lvis_prompt({"name": "traffic_light"}) == "traffic light"
