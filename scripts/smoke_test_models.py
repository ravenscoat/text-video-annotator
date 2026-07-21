"""Download and exercise the tiny Grounding DINO + SAM 2 models once."""
from __future__ import annotations

import gc
import sys

import numpy as np


def main() -> int:
    try:
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError as exc:
        print(f"Missing dependency: {exc}", file=sys.stderr)
        return 2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    print(f"device={device}")
    if device == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}")
        print(f"torch={torch.__version__}")

    detector_id = "IDEA-Research/grounding-dino-tiny"
    tracker_id = "facebook/sam2.1-hiera-tiny"
    print(f"loading detector={detector_id}")
    processor = AutoProcessor.from_pretrained(detector_id)
    detector = AutoModelForZeroShotObjectDetection.from_pretrained(detector_id).to(device).eval()

    # A small, deterministic image is enough to validate preprocessing and output shapes.
    image = np.zeros((384, 512, 3), dtype=np.uint8)
    image[100:280, 170:350] = (180, 180, 180)
    inputs = processor(images=image, text="object.", return_tensors="pt")
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
    with torch.inference_mode(), torch.autocast(device, dtype=dtype, enabled=device == "cuda"):
        outputs = detector(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        threshold=0.20,
        text_threshold=0.20,
        target_sizes=[image.shape[:2]],
    )[0]
    print(f"grounding_dino_boxes={len(results.get('boxes', []))}")

    del detector, processor, inputs, outputs
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    print(f"loading segmenter={tracker_id}")
    segmenter = SAM2ImagePredictor.from_pretrained(tracker_id)
    segmenter.model.to(device).eval()
    with torch.inference_mode(), torch.autocast(device, dtype=dtype, enabled=device == "cuda"):
        segmenter.set_image(image)
        masks, scores, _ = segmenter.predict(
            box=np.array([170, 100, 350, 280], dtype=np.float32),
            multimask_output=False,
        )
    print(f"sam2_mask_shape={masks.shape} sam2_score_shape={scores.shape}")
    if device == "cuda":
        print(f"cuda_peak_allocated_mb={torch.cuda.max_memory_allocated() / 1024**2:.1f}")
    print("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
