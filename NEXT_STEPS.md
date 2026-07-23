# Next AI Handoff: LVIS Evaluation, Then Video Benchmarks

## Current Priority Override

**Post-gate override (active):** Read `LEARNED_SELECTOR_PLAN.md` completely and implement it from the first incomplete acceptance criterion. The rule-based first-50 MeViS gate failed, so do not continue tuning hand-written motion thresholds and do not run cases 50+. Phase 0, the versioned cache foundation, cache-only rendering, lightweight features, and selector-sample contracts are implemented and verified. The active assignment is ordered semantic anchor tokens plus hybrid uniform/adaptive anchor selection. Only after candidate-recall validation should the small order-aware selector be trained, while Grounding DINO Tiny, SAM 2.1 Hiera Tiny, and semantic encoders remain frozen.

Phase 0/1 source implementation is now present: `scripts/prepare_mevis_diagnostic.py` creates a deterministic 10-case diagnostic manifest with candidate-recall reporting semantics, and `video_annotator/track_cache.py` provides schema-versioned, proposal-scope-fingerprinted, atomic cache save/load/validation. The pipeline records all propagated candidate tracks (including tracks later rejected by referring selection), writes original-resolution mask references, and exposes compatible-cache loading through `load_compatible_track_cache`; CLI users can request cache output with `--track-cache`. The cache-backed second pass is implemented with `--reuse-track-cache`: it validates source/scope, reads cached masks, applies category or referring selection, and renders/exports without detector or SAM 2 calls. Cache compatibility now permits a new referring expression when its concrete label is covered by cached candidates, while rejecting unrelated requests. `video_annotator/track_features.py` adds normalized temporal trajectories, visibility/area/direction, pairwise near/left/right/overlap relations, and per-track 6-8-anchor masked color-statistics appearance features, including late/short tracks. `video_annotator/selector_data.py` builds fixed per-track feature vectors and multi-positive labels from cached masks, and splits samples by whole video to prevent expression leakage. Verified evidence: full suite `48 passed` (two existing pycocotools deprecation warnings), compilation passed, and `git diff --check` passed. A real eight-frame Grounding DINO Tiny + SAM 2.1 Tiny smoke run generated a cache with one stable track, eight masks, six appearance anchors, and finite temporal features; the cache-only second pass produced a byte-identical MP4 without model inference. Generated verification files remain in the system temp directory. A generated diagnostic manifest was also created from the exact first-50 source order with bucket counts multi-target `2`, motion `1`, relation `1`, temporal `1`, single-target `5`. The current appearance vectors are lightweight color/statistics descriptors, not semantic frozen vision embeddings, and no frozen language embedding exists yet. Therefore selector training is not ready. The next incomplete criterion is adding frozen semantic image/text embeddings (or safely exposed SAM 2 object tokens plus a frozen text encoder), then generating real MeViS selector samples and measuring candidate-recall ceilings. Do not train the selector or rerun MeViS yet.

`TEMPORAL_MULTI_OBJECT_PLAN.md` records the completed rule-based upgrade and its failed A/B gate. Keep it as historical requirements and regression context; `LEARNED_SELECTOR_PLAN.md` now supersedes it for new implementation work.

**Ordered-token implementation progress:** Cache schema is now version `2` and supports chronologically ordered `AnchorToken` records with normalized timestamp, position/size, velocity/acceleration, confidence, visibility, provenance, relationship data, and optional semantic image embeddings. `semantic_encoder.py` provides an opt-in, mockable frozen CLIP image/text interface; it does not download weights automatically. Hybrid anchor selection preserves 4-6 uniform coverage anchors and adds at most 2 adaptive anchors. Selector samples now preserve per-track token sequences and optional expression text embeddings. Verified evidence: full model-free suite `51 passed` (two existing pycocotools deprecation warnings), compilation passed, and `git diff --check` passed. No semantic model weights were downloaded and no MeViS batch was rerun. Remaining Phase 3 work is to connect an explicitly approved/cached semantic encoder to real cache generation, validate token embedding dimensions and padding, pin the MeViS dataset identity, then generate diagnostic caches and measure candidate recall.

The semantic encoder is now connected to real cache generation behind explicit CLI options: `--semantic-model MODEL_ID` and `--semantic-device cpu|cuda`. Without `--semantic-model`, no semantic weights are loaded or downloaded. When enabled, the pipeline writes the model ID, device, and embedding dimension into cache metadata and validates dimensions on load. Verified evidence: full model-free suite `52 passed` (two existing pycocotools deprecation warnings), compilation passed, and `git diff --check` passed. The local Hugging Face cache does not currently contain `openai/clip-vit-base-patch32`, so no real semantic-model smoke run has been claimed. The next step is to obtain/authorize that lightweight checkpoint, run one 8-frame smoke cache at CPU or measured GPU settings, and then pin the MeViS dataset identity before real diagnostic cache generation.

The user-supplied local CLIP checkpoint is now available at `D:\work\segment anything\clip-model\clip-vit-base-patch32` and is ignored by Git. The Transformers compatibility path was verified against the installed version. A real 8-frame smoke cache was generated with `--semantic-model D:\work\segment anything\clip-model\clip-vit-base-patch32 --semantic-device cpu`: schema `2`, one track, eight masks, seven ordered anchor tokens, and 512-dimensional image embeddings. The annotated MP4 was written successfully. The next step is pinning the exact MeViS dataset identity and generating the fixed 10-case candidate caches; do not train the selector or run cases 0-49 yet.

**Architecture revision:** The cached color/statistics vectors and aggregate motion features are diagnostic fallbacks, not the learned selector input. Before training, upgrade the cache to chronological per-anchor semantic tokens and add a frozen image/text embedding adapter. Each token must retain normalized timestamp, position/size, velocity/acceleration, visibility, confidence, relationships, and anchor provenance. Candidate discovery must use 4-6 uniform coverage anchors plus at most 2 adaptive anchors, while preserving periodic redetection. Pin the exact MeViS release and split using filenames, SHA-256 hashes, counts, no-target/audio flags, and metric definitions before sample generation. Measure candidate recall before training. The learned selector must be order-aware; reversed/shuffled token discrimination may be used only as an auxiliary loss. This revision supersedes the older aggregate-feature next-step wording above.

Tasks 1-4 are complete: prompts, class-aware detector filtering, persistent multi-object IDs, and real redetection windows are implemented. The motion/referring baseline is now integrated into video inference/export: referring prompts score all current candidate tracks and only selected IDs are rendered and written to JSON/masks, with selection scores and reasons preserved. The full model-free suite passes (`34 passed`). A real 10-case Ref-DAVIS17 run completed at long side 512: mean region Jaccard `0.6723`, boundary F `0.5861`, recall@0.50 `0.8000`, 276 false-positive masks, 138 false-negative masks, and 0 track fragmentation. The exact first-50 MeViS A/B comparison also completed. Baseline -> upgraded: Jaccard `0.3460 -> 0.3436`, boundary F `0.2457 -> 0.2452`, recall `0.3934 -> 0.3840`, false positives `3209 -> 2758` (-14.1%), false negatives `1002 -> 1016`, fragmentation `61 -> 53` (-13.1%). The documented gate fails because Jaccard did not improve and false positives did not drop by 25%; do not run later MeViS batches yet. Inspect candidate-selection diagnostics and improve the baseline first.

This file is the source of truth for the next implementation steps. Read it completely before changing the project. Do not restart setup or replace the working model stack.

## Goal

Build and verify a general-purpose, offline text-prompted annotation pipeline:

1. Grounding DINO Tiny finds only objects matching a natural-language prompt.
2. SAM 2.1 Hiera Tiny converts boxes to masks and tracks masks through video.
3. The tool writes annotated image/video output and optional mask/JSON data.
4. Evaluation covers many object categories, not only cats and dogs.

The intended interaction is not tied to any particular classes. A video may contain many objects, and a prompt may request a subset such as "the person running and the person sitting," "the red car moving left," or previously displayed object IDs such as "object 1 and object 3." The system must discover candidate objects, track them through time, evaluate the prompt against their visual and temporal properties, and export only the matching tracks. Example nouns and actions in documentation are illustrative test cases, not required user-provided videos.

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

Do not rerun the full 10-case benchmark unless evaluator code or thresholds change. The next product milestone is a referring-expression benchmark, followed by the web API/UI layer.

### Updated referring-video benchmark order

1. **Ref-DAVIS17 first** — use it as the fast local phrase-segmentation test because it is small and practical to validate locally.
2. **MeViS afterward** — use it as the larger language-guided video benchmark, including motion expressions.
3. **Do not depend on Refer-YouTube-VOS** — its current validation-download workflow depends on the legacy CodaLab competition server, so it is removed from the required project roadmap.

### Ref-DAVIS17 download and preparation

Use the parsed Ref-DAVIS17 archive from the [SgMg Google Drive release](https://drive.google.com/file/d/1W0RsdxMK3VkNL80H1OWNmia-2asdCyYF/view?usp=sharing) when possible. The official preparation instructions are in [SgMg docs/data.md](https://github.com/bo-miao/SgMg/blob/main/docs/data.md). The alternative route is DAVIS 2017's two 480p zips plus the Ref-DAVIS text-annotation zip, followed by the SgMg conversion script.

Expected validation layout:

```text
data\\ref_davis17\\valid\\JPEGImages\\<video_id>\\*.jpg
data\\ref_davis17\\valid\\Annotations\\<video_id>\\*.png
data\\ref_davis17\\valid\\meta_expressions.json
```

After downloading, create the ten-expression manifest with `scripts\\prepare_refdavis_manifest.py`. The next implementation task is the phrase evaluator using Grounding DINO + SAM 2 propagation and region Jaccard/boundary-F metrics.

The phrase evaluator is now implemented in `scripts/evaluate_refdavis_subset.py`. It decodes DAVIS palette PNGs with Pillow, streams frames through the existing chunked pipeline, and reports region Jaccard, boundary F, recall, false positives/negatives, and track fragmentation. The ten-case run passed at long side 512. The MeViS follow-up evaluation target is now **20 videos**.

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

## Current MeViS Diagnostic Progress (2026-07-23)

- Generated semantic candidate caches for the fixed 10-video MeViS diagnostic using the local CLIP checkpoint at `clip-model/clip-vit-base-patch32`; generation completed for all 10 cases.
- Candidate coverage is `0.9565` at IoU thresholds 0.10, 0.30, and 0.50 (22/23 ground-truth objects covered). The sole miss is one horseback target with best IoU about 0.05; this is proposal coverage only, not official MeViS performance.
- Prepared video-disjoint selector JSONL: 10 samples (8 train / 2 validation), 38 positive tracks, zero empty-label samples. Samples include ordered temporal anchor tokens and 512-dimensional frozen CLIP text embeddings.
- Fixed the frame-size mismatch introduced by temporary MP4 conversion by nearest-neighbor resizing target masks during IoU labeling.
- Regression suite: 56 passed, 2 expected pycocotools deprecation warnings.

Next implementation task: train/evaluate a lightweight selector on these cached candidates, preserving ordered token sequences (with an optional reversed/shuffled temporal-order auxiliary loss), then run the 10-case diagnostic selector evaluation. Do not expand to the first-50/full MeViS runs until the selector passes the diagnostic gate.

### Selector iteration 1 result

Implemented `video_annotator/learned_selector.py` and `scripts/train_selector.py` with a compact GRU over ordered anchor tokens plus track features and frozen CLIP text embeddings. The first binary-loss-only run reached validation precision `0.50` and recall `0.33`. A second run adds pairwise positive-vs-negative ranking loss and train-only threshold calibration; it reached precision `0.75` and recall `1.00` on two held-out videos. Expanding to a 6-train / 4-validation split reached precision `0.833` and recall `1.00` (one false positive). The calibrated threshold remained `0.1`, so this is still diagnostic evidence rather than a production gate pass; require more expressions/videos and threshold stability before claiming generalization.

## Active Recommended Sequence (Audit 2026-07-23)

The 10-case diagnostic is complete and the codebase passes 56 tests. The attempted 20-case expansion stopped after six completed candidate caches and produced no final generation report. No generation process is currently running. Complete the following in order:

1. Make `scripts/generate_mevis_caches.py` resumable. It must validate and skip completed cache/render pairs, continue from the first incomplete case, and write an incremental report after every completed case so an interruption does not lose progress.
2. Move or copy durable evaluation reports and caches out of Windows temporary storage into an ignored project output directory. Keep datasets, masks, videos, checkpoints, and generated caches out of Git.
3. Resume cases 7-20 in explicit Hugging Face offline mode using the existing Grounding DINO, SAM 2, and local CLIP caches. Do not redo the six valid completed cases.
4. Calculate candidate recall for the completed 20-case run at IoU thresholds 0.10, 0.30, and 0.50. Report missing targets separately from selector mistakes. The candidate-generation gate remains at least 80% recall at IoU 0.10.
5. Generate the expanded selector JSONL with video-disjoint train/validation splits, ordered temporal tokens, and 512-dimensional frozen CLIP text embeddings. Confirm zero accidental empty-label samples.
6. Make selector training reproducible by fixing Python, NumPy, and PyTorch seeds and recording the seed, split identities, threshold, and model dimensions in the checkpoint/report.
7. Add dedicated tests for `TemporalTrackSelector`, variable-length/padded token sequences, checkpoint loading, deterministic splitting/training behavior, and cache-resume logic.
8. Train and evaluate the selector on the expanded data. Require stable validation precision/recall across more than one video split; do not accept a result only because a threshold of 0.1 performs well on one small split.
9. Integrate learned-selector inference into the main annotation pipeline behind an explicit option. Preserve the existing selector as a fallback and make no-target output possible when every candidate score is below threshold.
10. Render and visually inspect representative single-target, multi-target, motion, relation, temporal-order, and no-match videos. Confirm that only prompt-selected objects are shown and object identities remain stable.
11. Run compilation, the full test suite, and `git diff --check`. Review the staged file list to ensure no datasets, generated videos, masks, model weights, caches, installers, credentials, or temporary artifacts are included.
12. Commit and push the verified source changes to `origin/main`. The current remote still points to `https://github.com/ravenscoat/text-video-annotator.git`, and all work after commit `872fbc8` is currently uncommitted.

Do not expand to first-50 or full MeViS evaluation until steps 1-10 pass. Do not describe the current four-video selector result as general-purpose benchmark performance.

### 20-case expansion update

- Resumable generation completed all 20 cases (20 caches and 20 renders) with incremental reporting.
- Candidate coverage across 45 target objects: recall `0.9556` at IoU 0.10, `0.8889` at IoU 0.30, and `0.8667` at IoU 0.50. This clears the 0.10 candidate gate, but candidate misses remain distinct from selector misses.
- Expanded selector data contains 20 samples (12 train / 8 validation), 72 positive tracks, and zero empty-label samples.
- Selector iteration on the expanded split reached precision `0.9524` and recall `0.8333` at the train-calibrated threshold `0.1`. This is materially more informative than the two-video split, but threshold calibration and more expression diversity are still required before pipeline integration.
- Added selector checkpoint/threshold configuration and CLI options as integration scaffolding; actual cache-track scoring and render filtering remain the next implementation task. The legacy selector remains active by default.
- Connected learned checkpoint scoring to cache-only rendering. Supplying `--selector-checkpoint` plus `--semantic-model` now scores cached tracks and renders only tracks at or above `--selector-threshold`; omitting the checkpoint preserves the legacy selector. A real MeViS cache smoke render completed successfully (430 selected track-frame annotations).
- Added learned-selector forward/checkpoint round-trip regression tests. The suite now passes 58 tests (two expected pycocotools warnings).
- Added a regression test for compound detector labels matching category prompts (`three turtles` → `turtles`). The suite now passes 59 tests.
- Multi-object smoke testing initially exposed an intersection bug where learned scores could override category filtering. Fixed selection to intersect learned scores with the existing prompt selection. A corrected `turtles` render completed with 15 selected track-frame annotations; the test suite remains 58 passed.
- Visual QA of the corrected `turtles` frame confirms prompt isolation, but only one of several visible turtles is highlighted. Multi-object category filtering is therefore not ready for a quality gate: improve per-object recall/calibration (especially multi-target expressions) before treating learned-selector output as production quality.
- Fixed category matching to accept detector labels containing the requested category (for example, `three turtles` for `turtles`) and disabled learned-score filtering for category-union prompts. Visual QA now shows all three visible turtles selected; mask overlap/extra small regions still require quality tuning.
- Added per-frame 0.90 mask-IoU suppression for duplicate cached proposals during rendering. The 20-case smoke output still retains 290 selected track-frame annotations, so this removes exact duplicates but does not solve all small stray regions; stricter quality filtering remains a later tuning task.
