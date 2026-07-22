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

## LV-VIS video evaluation

The next-stage evaluator is implemented in [evaluate_lvvis_subset.py](scripts/evaluate_lvvis_subset.py). Download the validation video archive and `val_instances.json` from the [official LV-VIS repository](https://github.com/haochenheheda/LVVIS), extract them locally under `data/lv_vis/raw`, and do not add the media or annotations to Git. LV-VIS is CC BY-NC-SA 4.0 and released for non-commercial research.

Prepare a small, category-balanced manifest after the files are present:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_video_manifest.py --annotations data\lv_vis\raw\val_instances.json --dataset-root data\lv_vis\raw --output data\lv_vis\manifests\general_eval.json --max-items 10
```

Run the chunked video evaluator offline:

```powershell
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
.\.venv\Scripts\python.exe scripts\evaluate_lvvis_subset.py --manifest data\lv_vis\manifests\general_eval.json --output outputs\lvvis_eval --long-side 768 --chunk-frames 30
```

It reports per-frame mask IoU, recall at IoU 0.50, false-positive/negative masks, and track fragmentation. These are diagnostic subset metrics, not official LV-VIS AP.

The completed 10-case run covered 228 frames and 10 categories at long side 512 on the RTX 5060. Results: mean frame mask IoU `0.8717`, recall@IoU 0.50 `1.0000`, 232 false-positive masks, 0 false-negative masks, and 0 track fragmentation. This small diagnostic result shows strong propagation on the selected cases but also shows that open-vocabulary detection can return extra same-prompt objects; it is not official LV-VIS AP.

## Refer-YouTube-VOS phrase evaluation

The next benchmark tests language expressions rather than category names. Download the validation videos and `meta_expressions.json` from the [official Refer-YouTube-VOS page](https://youtube-vos.org/dataset/rvos/). Keep the media outside Git. The official dataset has 3,978 videos and 15k language expressions; its validation split has 202 videos.

Expected local layout:

```text
data\refer_youtube_vos\valid\
├── JPEGImages\<video_id>\*.jpg
└── meta_expressions.json
```

Prepare phrase cases:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_refytvos_manifest.py --meta data\refer_youtube_vos\valid\meta_expressions.json --split-root data\refer_youtube_vos\valid --output data\refer_youtube_vos\manifests\general_eval.json --max-items 10
```

This manifest is the input for the upcoming phrase-aware video evaluator, which will compare the referred-object masks using region Jaccard (J) and boundary F metrics.

### Referring-video benchmark order

We will use **Ref-DAVIS17 first** for a fast local phrase-segmentation test, then use **MeViS** for a larger language-guided video benchmark with motion expressions. Refer-YouTube-VOS is not required because its validation data currently depends on the inaccessible legacy CodaLab download workflow.
