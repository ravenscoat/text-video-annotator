from pathlib import Path

import cv2
import numpy as np

from scripts.evaluate_candidate_recall import iou


def test_candidate_recall_iou_resizes_masks():
    first = np.zeros((8, 8), dtype=bool); first[2:6, 2:6] = True
    second = np.zeros((4, 4), dtype=bool); second[1:3, 1:3] = True
    assert iou(first, second) > 0.2
