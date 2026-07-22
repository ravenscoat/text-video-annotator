# Next AI Handoff: LVIS Evaluation, Then Video Benchmarks

This file is the source of truth for the next implementation steps. Read it completely before changing the project. Do not restart setup or replace the working model stack.

## Goal

Build and verify a general-purpose, offline text-prompted annotation pipeline:

1. Grounding DINO Tiny finds only objects matching a natural-language prompt.
2. SAM 2.1 Hiera Tiny converts boxes to masks and tracks masks through video.
3. The tool writes annotated image/video output and optional mask/JSON data.
4. Evaluation covers many object categories, not only cats and dogs.

The immediate task is **LVIS image-subset evaluation**. After that passes, add a small video benchmark using LV-VIS or Refer-YouTube-VOS.

## Known Working State — Do Not Repeat Setup

- Workspace: `D:\work\segment anything`
- GitHub repository: `https://github.com/ravenscoat/text-video-annotator`
- Use native Windows PowerShell commands. Do not use Bash line continuation syntax.
- Virtual environment: `.venv`
- Python in that environment: Python 3.14.6
- PyTorch: `2.11.0+cu128`
- CUDA is working on `NVIDIA GeForce RTX 5060 Laptop GPU`.
- Grounding DINO model is cached: `IDEA-Research/grounding-dino-tiny`.
- SAM 2 model is cached: `facebook/sam2.1-hiera-tiny`.
- Official SAM 2 source is already present under `vendor/sam2` and imports successfully.
- `SAM2_BUILD_CUDA=0` was used. A missing optional SAM 2 `_C` extension warning is expected and is not a blocker for this project.
- Model smoke test already passed: one Grounding DINO detection, one SAM 2 mask, peak allocated VRAM about 1.7 GB.
- Eight-frame video smoke test already passed and produced `outputs/smoke_video_annotated.mp4`, JSON for all eight frames, and a preview PNG.

Do not reinstall Python, PyTorch, CUDA, Transformers, or SAM 2. Do not download the model checkpoints again. Only rerun smoke tests if inference code is changed.

## Hardware Rules

- Assume only 8 GB VRAM and 16 GB system RAM.
- Use Grounding DINO Tiny and SAM 2.1 Hiera Tiny only.
- Default inference long side is 768 pixels. If CUDA runs out of memory, retry at 640, then 512.
- Use `torch.inference_mode()` and mixed precision where supported.
- Never load an entire real video into RAM or VRAM. Continue using frame chunks and SAM 2 CPU offloading.
- Instantiate each model once per evaluation process. Do not reload models for every image.
- Clear CUDA cache only after an out-of-memory error or between model stages when necessary; do not use it as a substitute for fixing retained tensors.

## Current Dataset State and Exact Blocker

The official LVIS v1 validation annotations were downloaded successfully. A 12-image subset annotation and initial manifest exist:

- `data/lvis/raw/lvis_v1_val_subset.json`
- `data/lvis/raw/subset_manifest.json`
- `data/lvis/manifests/general_eval.json`

The source images initially failed to download. The downloader was repaired and the subset is now complete. The original failure was:

```text
SSL: CERTIFICATE_VERIFY_FAILED: Hostname mismatch, certificate is not valid for 'images.cocodataset.org'
```

Cause: `scripts/download_lvis_subset.py` changed each official `http://` COCO URL to `https://`:

```python
download_bytes(image["coco_url"].replace("http://", "https://"))
```

That rewrite must be removed. First try the original `image["coco_url"]` unchanged. Do not globally disable TLS certificate validation.

### Verified progress

- Twelve images now exist under `data/lvis/raw/images` and all decode successfully.
- A second downloader run validated and skipped the good files.
- `data/lvis/manifests/general_eval.json` now contains 12 image/category cases across 12 categories.
- `scripts/evaluate_lvis_subset.py` completed all 12 cases offline.
- Results are in `outputs/lvis_eval/metrics.json`, `outputs/lvis_eval/cases.jsonl`, and `outputs/lvis_eval/previews/`.
- Diagnostic aggregate metrics: mean mask IoU `0.5694`, recall@IoU 0.50 `0.4839`, precision@IoU 0.50 `0.4688`, 17 false positives, 16 false negatives, and peak CUDA allocation `2215.34 MB`.
- Preview masks for air conditioner, banana, and bucket were visually inspected and aligned with the visible objects.
- Fast compile and existing tests pass: `2 passed`.

Do not repeat Steps 1–3 unless their implementation changes. LV-VIS video preparation and evaluation code is now present:

- `scripts/prepare_video_manifest.py` selects deterministic category-balanced cases from local LV-VIS annotations.
- `scripts/evaluate_lvvis_subset.py` runs the existing chunked pipeline and reports temporal mask IoU, recall, false positives/negatives, and track fragmentation.
- `video_annotator/video_metrics.py` contains the tested matching and fragmentation metrics.
- The full fast suite passes: `10 passed`.

The remaining video milestone is to obtain the authorized LV-VIS validation archive locally, generate a manifest, run the real evaluator, inspect previews, and record the results. Do not download or commit the dataset automatically.

### Verified LV-VIS result

- The local archive is valid: 837 videos, 3,719 annotations, 1,196 categories, and 19,139 JPEG frames.
- The official layout is `data/lv_vis/raw/val/JPEGImages/<video_id>/...`; the downloaded annotation filename is `val_instances_.json`.
- The 10-case manifest and full evaluator completed successfully on 228 frames.
- Aggregate metrics: mean frame mask IoU `0.8717`, recall@IoU 0.50 `1.0000`, 232 false-positive masks, 0 false-negative masks, and 0 track fragmentation.
- The moved checkout now imports SAM 2 from the local `vendor/sam2` fallback if its old editable-install path is stale.

Do not rerun the full 10-case benchmark unless evaluator code or thresholds change. The next product milestone is a referring-expression benchmark (Refer-YouTube-VOS), followed by the web API/UI layer.

### Refer-YouTube-VOS progress

- Official dataset structure and statistics were verified from the project page.
- `scripts/prepare_refytvos_manifest.py` now creates phrase/object cases from local `meta_expressions.json` and `JPEGImages` files.
- The full fast suite passes: `11 passed`.
- Remaining work: download the authorized validation split, generate a 10-expression manifest, add the phrase evaluator with J/F metrics, inspect outputs, and push the results.

## Step 1 — Repair and Complete the LVIS Subset Download

Update `scripts/download_lvis_subset.py` with these behaviors:

1. Use the official `coco_url` unchanged instead of replacing `http://` with `https://`.
2. Keep the existing user-agent and timeout.
3. Skip an image only when the existing destination is a valid, non-empty image. Validate it with Pillow or OpenCV; a partial file must be downloaded again.
4. Download to a temporary file beside the destination and rename it only after validation, so failed downloads cannot leave corrupt JPEGs.
5. Retry each image up to three times with a short increasing delay.
6. Reuse `data/lvis/raw/lvis_v1_val_subset.json` when it already exists and contains at least the requested number of images. Do not redownload the full annotation archive on every retry.
7. Report failed URLs clearly and exit nonzero if any requested image is missing or unreadable.
8. Preserve the official annotations. Never rewrite polygons, RLE masks, category IDs, or image dimensions.

Run from the workspace root:

```powershell
.\.venv\Scripts\python.exe scripts\download_lvis_subset.py --output data\lvis\raw --max-images 12
```

If the execution environment blocks network access, request network approval and rerun the same command. Do not work around it by disabling SSL verification.

Acceptance checks:

- Exactly 12 requested images exist under `data/lvis/raw/images`.
- Every file can be decoded with OpenCV or Pillow.
- Each downloaded image ID and file name matches `lvis_v1_val_subset.json`.
- The command can be run twice; the second run validates and skips good files.

## Step 2 — Fix the Evaluation Manifest Semantics

`scripts/prepare_dataset.py` currently records only the first annotation category for each image. That does not produce a genuinely balanced general-purpose test.

Change LVIS preparation so each evaluation case represents an **image/category pair**, not merely one image:

```json
{
  "image_id": 123,
  "image_path": "data/lvis/raw/images/000000000123.jpg",
  "category_id": 17,
  "category_name": "cat",
  "prompt": "cat",
  "annotation_ids": [10, 11]
}
```

Requirements:

1. Include all ground-truth instances of the selected category in that image.
2. Convert LVIS category names into natural prompts: prefer the first entry in `synonyms`; otherwise replace underscores with spaces. Do not send parenthetical metadata as part of the prompt when a clean synonym exists.
3. Select categories deterministically and spread cases across as many distinct categories as possible before adding repeated categories.
4. Store paths relative to the workspace or dataset root so the manifest remains usable after cloning.
5. Keep adapters for LV-VIS, YouTube-VIS, and Refer-YouTube-VOS intact.
6. Rename `--max-videos` to a neutral `--max-items`, but retain `--max-videos` as a deprecated alias if practical.

Generate the manifest:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_dataset.py --dataset lvis --dataset-root data\lvis\raw --output data\lvis\manifests\general_eval.json --max-items 12
```

Acceptance checks:

- The manifest contains 12 usable image/category evaluation cases.
- Paths exist and decode successfully.
- It contains multiple unrelated categories, not a cat/dog-only test.
- Every `annotation_id` belongs to the specified image and category.

## Step 3 — Add a Real LVIS Batch Evaluator

Create `scripts/evaluate_lvis_subset.py`. It must use the real Grounding DINO Tiny and SAM 2.1 Tiny adapters already in `video_annotator`; do not create duplicate model implementations.

Evaluation procedure for each image/category case:

1. Load the image at original resolution.
2. Downsample for inference when the long side exceeds the configured limit.
3. Run Grounding DINO once using the case prompt.
4. Run SAM 2 image segmentation on every returned box.
5. Resize predicted masks to the original image size using nearest-neighbor interpolation.
6. Decode all ground-truth LVIS masks for the case category with `pycocotools.mask`.
7. Match predictions to ground-truth instances one-to-one, greedily by highest mask IoU or with Hungarian matching. A prediction and ground truth cannot be used twice.
8. Save per-case metrics, aggregate metrics, and optional annotated previews.

Required metrics:

- Mean matched mask IoU.
- Recall at mask IoU >= 0.50.
- Precision at mask IoU >= 0.50.
- False-positive count.
- False-negative count.
- Number of cases where Grounding DINO returned no matching box.
- Per-category results.

Metric definitions must be written into the JSON report. Do not call this official LVIS AP; it is a small diagnostic subset, not the full LVIS benchmark.

Suggested command:

```powershell
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
.\.venv\Scripts\python.exe scripts\evaluate_lvis_subset.py --manifest data\lvis\manifests\general_eval.json --annotations data\lvis\raw\lvis_v1_val_subset.json --output outputs\lvis_eval --long-side 768
```

The evaluator must expose thresholds as arguments, defaulting to the existing pipeline values:

- `--box-threshold 0.30`
- `--text-threshold 0.25`
- `--max-objects 10`

If CUDA OOM occurs:

1. Catch only the CUDA out-of-memory exception.
2. Release references to failed outputs and call `torch.cuda.empty_cache()`.
3. Retry that case once at the next smaller long-side value: 768 -> 640 -> 512.
4. Record the fallback resolution and warning in the report.
5. Do not silently switch the entire evaluation to CPU.

Required outputs:

- `outputs/lvis_eval/metrics.json`
- `outputs/lvis_eval/cases.jsonl`
- `outputs/lvis_eval/previews/*.jpg`

Acceptance checks:

- The evaluator completes all manifest cases without loading models repeatedly.
- It records zero-detection cases instead of crashing.
- Metrics are finite and bounded where applicable.
- Preview masks align with original-resolution objects.
- Peak CUDA memory is recorded with `torch.cuda.max_memory_allocated()`.

## Step 4 — Tests Before Real Inference

Add fast tests that do not load model weights:

1. LVIS polygon mask decoding.
2. LVIS RLE mask decoding if present in the subset.
3. Mask IoU for identical, disjoint, and partially overlapping masks.
4. One-to-one matching with more predictions than ground truths and vice versa.
5. Category prompt normalization.
6. Manifest image/category grouping.
7. Corrupt/empty downloaded-image detection.

Run:

```powershell
.\.venv\Scripts\python.exe -m compileall video_annotator scripts tests
.\.venv\Scripts\python.exe -m pytest -q
```

Then run the real 12-case evaluator. Unit-test success alone is not completion.

## Step 5 — Report Results Honestly

After evaluation, update `README.md` with:

- The exact command used.
- Dataset name and subset size.
- Hardware and model variants.
- Aggregate metrics from `metrics.json`.
- A statement that this is a small diagnostic subset, not official LVIS benchmark performance.
- A preview image if it is appropriate to commit; otherwise keep generated outputs ignored.

Do not claim the system can identify every imaginable object. Grounding DINO is open-vocabulary, but success depends on its learned visual concepts, prompt wording, image quality, and thresholds.

## Step 6 — Only After LVIS Image Evaluation Passes

Move to video evaluation:

1. Prefer a small LV-VIS subset for category-prompted general-purpose video segmentation.
2. Add Refer-YouTube-VOS later for phrase prompts such as `the person in the red shirt`.
3. Do not use ordinary YouTube-VIS alone as proof of text grounding; it has segmentation/tracking labels but is not a referring-expression benchmark.
4. Stream or chunk frames. Never preload the full dataset video.
5. Add video metrics:
   - Per-frame mask IoU.
   - Track recall.
   - False-positive tracks.
   - Track fragmentation: number of predicted track-ID segments assigned to one ground-truth trajectory minus one.
6. Test periodic redetection separately from pure SAM 2 propagation.

## Code Quality and Safety Rules

- Preserve working user changes and inspect `git status` before editing.
- Use `pathlib.Path`; support Windows paths containing spaces.
- Use explicit errors with actionable messages.
- Keep generated data, checkpoints, videos, masks, and evaluation outputs out of Git.
- Do not commit `python-3.11.9-amd64.exe`, `.venv`, `*.egg-info`, model caches, `data/`, or `outputs/`.
- Add `*.egg-info/` and `python-*.exe` to `.gitignore` if they are not already covered.
- Never commit tokens, GitHub credentials, or Hugging Face cache contents.
- Do not alter the existing Git remote.
- Before committing, review the diff and run tests.

## Git Completion

The working tree already contains local changes and untracked smoke/dataset scripts. Review them instead of deleting them. When the requested dataset/evaluator milestone passes:

```powershell
git status --short
git diff --check
git add .gitignore README.md NEXT_STEPS.md scripts tests video_annotator pyproject.toml
git status --short
git commit -m "Add LVIS subset evaluation workflow"
git push origin main
```

Do not use `git add -A` because large local installers and generated artifacts are present. Confirm no dataset images, model files, `.egg-info`, or executable installers are staged before committing.

## Definition of Done for the Next Milestone

The next milestone is complete only when all of the following are true:

- Twelve LVIS images are valid and locally available.
- A deterministic multi-category image/category manifest exists.
- Fast tests pass.
- The real Grounding DINO + SAM 2 evaluator completes all 12 cases on the RTX 5060.
- Metrics and per-case results are saved.
- At least several preview masks have been visually inspected for alignment.
- README contains reproducible evaluation instructions and honest limitations.
- Intended source changes are committed and pushed without data, weights, caches, or installers.
