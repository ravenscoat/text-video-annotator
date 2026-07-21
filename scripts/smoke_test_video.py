from pathlib import Path

import cv2
import numpy as np

from video_annotator import AnnotationConfig, annotate_media


def main():
    root = Path("outputs")
    root.mkdir(exist_ok=True)
    source = root / "smoke_video_input.avi"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (512, 384))
    if not writer.isOpened():
        raise RuntimeError("Could not create the synthetic AVI fixture")
    for index in range(8):
        frame = np.zeros((384, 512, 3), dtype=np.uint8)
        frame[100 + index * 3:220 + index * 3, 170 + index * 4:330 + index * 4] = 180
        writer.write(frame)
    writer.release()
    result = annotate_media(
        AnnotationConfig(
            source,
            "object",
            root / "smoke_video_annotated.mp4",
            root / "smoke_video.json",
            root / "smoke_video_masks",
            chunk_frames=8,
            redetect_every=8,
            device="cuda",
        )
    )
    print(result)


if __name__ == "__main__":
    main()
