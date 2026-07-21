# Text Video Annotator

Offline open-vocabulary image/video annotation using Grounding DINO Tiny and SAM 2.1 Hiera Tiny.

The implementation plan is in [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md). Coding agents must read [AGENTS.md](AGENTS.md), which points to the exact handoff and dataset-evaluation checklist in [NEXT_STEPS.md](NEXT_STEPS.md). The current code provides the package structure, lazy model adapters, image path, chunked video orchestration, CLI, rendering/export interfaces, and dataset manifest preparation.

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

## LVIS diagnostic evaluation

The first local general-purpose evaluation uses 12 LVIS v1 validation image/category cases spanning 12 categories. The lightweight Grounding DINO Tiny and SAM 2.1 Hiera Tiny models completed the batch on an RTX 5060 Laptop GPU with 2,215 MB peak allocated CUDA memory.

Run it offline after downloading the subset:

```powershell
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
.\.venv\Scripts\python.exe scripts\evaluate_lvis_subset.py --manifest data\lvis\manifests\general_eval.json --annotations data\lvis\raw\lvis_v1_val_subset.json --output outputs\lvis_eval --long-side 768
```

The diagnostic result was mean mask IoU `0.5694`, recall at IoU 0.50 `0.4839`, and precision at IoU 0.50 `0.4688`. These numbers are from a 12-case diagnostic subset and are not official LVIS AP. Per-case results and previews are written to `outputs/lvis_eval/`.
