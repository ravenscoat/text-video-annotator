import json
from pathlib import Path

import cv2
import numpy as np

from scripts.generate_mevis_caches import make_video


def test_make_video_converts_case_frame_directory(tmp_path: Path):
    frame_dir = tmp_path / "frames"; frame_dir.mkdir()
    for index in range(2):
        cv2.imwrite(str(frame_dir / f"{index:06d}.jpg"), np.full((8, 10, 3), index * 50, dtype=np.uint8))
    output = tmp_path / "case.mp4"
    make_video({"video_dir": str(frame_dir), "frame_names": ["000000", "000001"]}, output, fps=5.0)
    capture = cv2.VideoCapture(str(output))
    assert capture.isOpened() and int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 2
    capture.release()
