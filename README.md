# Text Video Annotator

Offline open-vocabulary image/video annotation using Grounding DINO Tiny and SAM 2.1 Hiera Tiny.

The implementation plan is in [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md). The current code provides the package structure, lazy model adapters, image path, chunked video orchestration, CLI, rendering/export interfaces, and dataset manifest preparation.

## Install

Create the environment described in the plan first. Then install this package:

```bash
pip install -e ".[models,dev]"
```

For a model-free smoke test, install only the base dependencies and run:

```bash
python -m compileall video_annotator scripts
pytest -q
```

## CLI

```bash
video-annotator annotate --input input.mp4 --prompt "red backpack" --output output.mp4 --export-json output.json
video-annotator annotate --input image.jpg --prompt "power drill" --output image_annotated.jpg
```

Models are loaded lazily on first inference. Dataset archives are intentionally not included in this repository; use `scripts/prepare_dataset.py` after obtaining them under their official terms.
