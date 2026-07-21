# Text-Prompted Video Segmentation and Tracking

## 1. Goal

Build an offline, open-vocabulary command-line pipeline that accepts:

```text
input image or video + text prompt
```

and produces:

- an annotated image or MP4 containing only the matching objects with translucent masks, outlines, labels, and, for video, stable IDs;
- optional per-frame binary PNG masks;
- a JSON manifest with object IDs, boxes, scores, and mask references or COCO RLE.

The target machine is a Windows laptop with an RTX 5060 (8 GB VRAM) and 16 GB system RAM. The first implementation should prefer reliability and bounded memory over speed.

The tool is **general purpose**, not cat/dog-specific. Examples include `cat`, `forklift`, `red backpack`, `the cup on the left`, or `person wearing a yellow helmet`. Cat-and-dog footage is only a simple smoke test. Because Grounding DINO is open-vocabulary rather than omniscient, arbitrary prompts are best-effort: uncommon objects, relationships, counting, motion descriptions, and fine-grained attributes may fail and must be covered by evaluation and honest confidence/error reporting.

## 2. Recommended technical choices

### Models

- Detector: `IDEA-Research/grounding-dino-tiny` loaded with Hugging Face Transformers.
- Segmenter/tracker: `facebook/sam2.1-hiera-tiny` using Meta's official `SAM2VideoPredictor`.
- Default inference resolution: preserve aspect ratio and limit the long side to 768 pixels. Fall back to 640, then 512 after CUDA out-of-memory.
- Default precision: BF16 autocast on the RTX 5060. If a kernel rejects BF16, retry with FP16.

Use Transformers for Grounding DINO rather than the original GroundingDINO package. It avoids compiling the detector's custom extension and is substantially easier to install on Windows. Use the official Meta package for SAM 2 video tracking because its video state and propagation API are required.

For a still image, use Grounding DINO to generate text-matched boxes and SAM 2's image predictor to create masks. For a video, use the same grounding step followed by SAM 2 video propagation. Do not make an image pass through the video/chunk implementation.

### Operating environment

Use WSL2 with Ubuntu 24.04, Python 3.11, a current NVIDIA Windows driver, and a PyTorch CUDA wheel. Meta officially recommends WSL for Windows. Start with the official CUDA 12.8 wheel because RTX 50-series/Blackwell support requires a recent CUDA build:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
pip install transformers huggingface-hub safetensors accelerate opencv-python-headless pillow numpy tqdm typer pydantic pycocotools
git clone https://github.com/facebookresearch/sam2.git vendor/sam2
SAM2_BUILD_CUDA=0 pip install --no-build-isolation -e vendor/sam2
```

`SAM2_BUILD_CUDA=0` skips optional mask cleanup kernels, not SAM 2 inference. It avoids a common Windows/WSL build failure. Pin every resolved package version with `pip freeze > requirements-lock.txt` after the smoke test passes.

Verify the GPU before model work:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected: CUDA is `True` and the device name contains `RTX 5060`.

## 3. Architecture

```text
CLI / future API
      |
      v
Media router
   |-- image -> Grounding DINO -> SAM 2 image predictor -> image/export renderer
   |
   +-- video -> probe -> chunk extractor -> Grounding DINO -> SAM 2 propagation
      |              |                  |                  |
      |              |                  |                  +-> masks/boxes/IDs
      |              |                  +-> prompt boxes
      |              +-> bounded temporary JPEG directory
      +-> fps, dimensions, frame count
                                                |
                                                v
                                  renderer + mask/JSON exporters
                                                |
                                                v
                                  annotated MP4 + optional data
```

Important: the stock SAM 2 video predictor maintains per-video inference state and can cache all input frames. CPU offloading reduces VRAM but does not make memory constant. Therefore the production path must process a bounded chunk, destroy its inference state, clear cached tensors, and continue with the next chunk.

## 4. Proposed repository structure

```text
video_annotator/
  __init__.py
  cli.py                 # Typer CLI and argument validation
  config.py              # Pydantic configuration and defaults
  pipeline.py            # orchestration only
  video_io.py            # probe, chunk extraction, writer, audio remux
  detector.py            # Grounding DINO adapter
  tracker.py             # SAM 2 adapter and chunk propagation
  identity.py            # cross-chunk object ID association
  render.py              # overlay, contours, labels
  exporters.py           # PNG and COCO/RLE/JSON
  memory.py              # CUDA cleanup and OOM retry policy
  types.py               # Detection, Track, FrameResult dataclasses
tests/
  test_detector_postprocess.py
  test_coordinate_scaling.py
  test_identity_matching.py
  test_export_schema.py
  test_tiny_video_e2e.py
scripts/
  smoke_test_gpu.py
  download_models.py
pyproject.toml
README.md
```

Keep model-specific code behind `Detector` and `VideoTracker` interfaces so the future FastAPI layer calls a single `annotate_video(config)` service without importing model internals.

## 5. CLI contract

```bash
python -m video_annotator.cli annotate \
  --input videos/cat_and_dog.mp4 \
  --prompt "cat" \
  --output outputs/cats.mp4 \
  --export-json outputs/cats.json \
  --export-masks outputs/masks \
  --long-side 768 \
  --chunk-frames 120 \
  --redetect-every 60
```

The same command accepts an image:

```bash
python -m video_annotator.cli annotate \
  --input images/workbench.jpg \
  --prompt "red power drill" \
  --output outputs/drill.png \
  --export-json outputs/drill.json
```

Core options:

- `--box-threshold 0.30`
- `--text-threshold 0.25`
- `--mask-threshold 0.0` (SAM logits)
- `--max-objects 10`
- `--chunk-frames 120`
- `--redetect-every 60`
- `--long-side 768`
- `--device cuda|cpu`
- `--export-format none|png|coco-rle`
- `--keep-audio/--drop-audio`

Support two explicit prompt modes:

- `category`: `cat`, `forklift`, or `traffic cone`; annotate every matching instance.
- `referring`: `the red cup on the left` or `the worker wearing a yellow helmet`; annotate the best matching instance unless `--all-matches` is supplied.

Normalize a simple category prompt to the punctuation expected by Grounding DINO. Do not automatically turn singular into plural. For multiple categories, prefer repeatable `--class` arguments over guessing how to split natural language. Retain the phrase label returned by the detector for each detection.

Grounding DINO can often ground adjectives and short noun phrases, but it is not a full language-reasoning or motion-reasoning system. Prompts such as `the person who enters last` cannot reliably be solved by detecting the first frame. Reject explicitly motion-dependent prompt modes in version 1 or return a clear `unsupported_prompt_semantics` warning rather than pretending the result is reliable.

## 6. Processing algorithm

### Phase A: validate and probe

1. Check that the input exists, is readable, and is not the output path.
2. Open it with OpenCV or PyAV and read FPS, width, height, frame count, duration, and codec.
3. Reject zero-frame videos and invalid dimensions with a useful message.
4. Calculate the inference size once, preserving aspect ratio. Keep original dimensions for the output writer and coordinate transforms.
5. Estimate temporary disk usage and ensure sufficient free space. JPEG chunks are intentionally on disk so RAM stays bounded.

### Phase B: process bounded chunks

For every chunk of approximately 60-120 frames:

1. Decode only that chunk and save sequential JPEGs (`000000.jpg`, etc.) at the inference resolution.
2. Select the chunk's first frame as the conditioning frame.
3. Run Grounding DINO Tiny on that frame with the user prompt.
4. Post-process boxes directly into the resized frame's pixel coordinates. Apply thresholding, clipping, invalid-box removal, and class-aware NMS.
5. Associate detections with tracks from the previous chunk using mask IoU or box IoU, label equality, and optionally center distance. Reuse an ID only above a conservative threshold; otherwise allocate a new stable ID.
6. Initialize SAM 2 with:

   ```python
   predictor.init_state(
       video_path=chunk_directory,
       offload_video_to_cpu=True,
       offload_state_to_cpu=True,
       async_loading_frames=False,
   )
   ```

7. Add one box prompt per detected object at local frame index 0 using `add_new_points_or_box`.
8. Call `propagate_in_video`. For each yielded frame, immediately:
   - threshold mask logits;
   - move the boolean mask to CPU;
   - compute a bounding box and mask area;
   - resize the mask to original resolution with nearest-neighbor interpolation;
   - draw the overlay and write that output frame;
   - write optional mask/JSON data;
   - discard frame-level arrays.
9. Keep only the last valid mask/box/ID summary for cross-chunk association.
10. Delete the predictor state, run `gc.collect()` and `torch.cuda.empty_cache()`, then remove the temporary chunk directory.

### Phase C: periodic re-detection inside a chunk

Version 1 may use chunk boundaries as the periodic re-detection points. That is simpler and naturally bounded. A later version can re-detect inside a chunk to discover objects entering midway.

If `--redetect-every` is smaller than `--chunk-frames`, split the effective chunk at that interval. Each interval becomes a new SAM 2 session. This avoids complicated mutation of an active propagation state.

### Phase D: output finalization

1. Close the video writer even if processing fails.
2. If audio preservation is requested and FFmpeg is available, remux the original audio into the annotated video without re-encoding audio.
3. Atomically rename a `.partial.mp4` only after successful completion.
4. Write a final metadata record containing model IDs, thresholds, inference size, source properties, elapsed time, and any fallback used.

## 7. Identity across chunks

SAM 2 object IDs are stable only within one predictor session. Preserve user-facing IDs across chunks with a small association layer:

1. Compare each new Grounding DINO box with the previous chunk's last-frame masks/boxes.
2. Construct a cost matrix from `1 - IoU`, with a large penalty for different text labels.
3. Use Hungarian matching if SciPy is installed; otherwise greedy matching is sufficient for the first version.
4. Accept matches only when IoU is at least 0.3 (configurable).
5. Allocate a new monotonically increasing ID for unmatched detections.
6. Retire tracks after two missed re-detection intervals.

This is an approximation. If a cat is fully absent at a boundary and later reappears, it may receive a new ID. The masks remain correct even if identity continuity is imperfect.

## 8. Empty detections and new objects

- If the first frame has no match, do not immediately declare failure. Scan keyframes every `redetect_every` frames until a match is found.
- Frames before the first match are written unchanged with empty annotations.
- If no frame contains a match, finish successfully with an unchanged video and JSON status `no_objects_found`; also print a clear warning.
- Re-detection at each effective chunk boundary finds matching objects that enter after frame zero.
- Cap detections with `max_objects`, sorted by confidence, to protect VRAM.

## 9. Memory and OOM policy

Use all of these from the start:

- one Grounding DINO frame at a time;
- `torch.inference_mode()` around every model call;
- BF16 autocast on CUDA, FP16 fallback;
- SAM 2.1 Hiera Tiny by default;
- `offload_video_to_cpu=True` and `offload_state_to_cpu=True`;
- short JPEG chunks on disk;
- immediate CPU transfer and serialization of masks;
- no dictionary containing masks for every frame;
- explicit destruction of chunk state between sessions;
- detection count cap.

On `torch.cuda.OutOfMemoryError`:

1. close and discard only the current SAM session;
2. clear Python and CUDA caches;
3. retry the same chunk at the next resolution: 768 -> 640 -> 512;
4. if still failing, halve `chunk_frames`: 120 -> 60 -> 30;
5. if still failing, reduce `max_objects` and report which objects were omitted;
6. fail with a diagnostic that includes peak allocated/reserved VRAM and the exact retry settings.

Do not silently switch the whole pipeline to CPU; it may take many hours. Offer CPU as an explicit CLI option.

System RAM note: CPU offloading can still be large because SAM 2 keeps resized tensors and tracking state for the current chunk. With 16 GB RAM, begin at 60 frames if the source is long or other applications are open.

## 10. Rendering and exports

### Annotated video

- Use a deterministic color based on object ID.
- Alpha-blend the mask at 0.35-0.5.
- Draw the outer contour at 2 pixels after resizing to original resolution.
- Draw `cat #3 0.81` near the current mask bounding box.
- Keep original FPS and dimensions.

OpenCV often drops audio. Write video first, then use FFmpeg to copy the source audio into the final MP4. Document that `mp4v` is the portable fallback; use H.264 when an encoder is available.

### JSON schema

```json
{
  "source": {"path": "input.mp4", "width": 1920, "height": 1080, "fps": 30.0},
  "prompt": "cat",
  "models": {
    "detector": "IDEA-Research/grounding-dino-tiny",
    "tracker": "facebook/sam2.1-hiera-tiny"
  },
  "frames": [
    {
      "frame_index": 0,
      "timestamp_seconds": 0.0,
      "objects": [
        {
          "track_id": 1,
          "label": "cat",
          "detector_score": 0.83,
          "bbox_xyxy": [120, 95, 440, 510],
          "segmentation": {"format": "coco_rle", "size": [1080, 1920], "counts": "..."}
        }
      ]
    }
  ]
}
```

Detector confidence is only measured on re-detection frames. On propagated frames, either carry it as `source_detection_score` or use `null`; do not invent a SAM tracking confidence.

For PNG export use one lossless single-channel mask per object per frame, not a combined palette mask, unless the user selects a combined mode. Suggested name: `frame_000123_object_0003.png`.

## 11. Error handling

Handle and test:

- missing/unreadable input;
- unsupported or corrupt video;
- invalid/zero FPS metadata (use a documented fallback only if decoding succeeds);
- empty prompt;
- no detections anywhere;
- degenerate/out-of-bounds boxes;
- failed model download in offline mode;
- CUDA unavailable or incompatible wheel;
- CUDA OOM with bounded retries;
- full disk while extracting frames or exporting masks;
- interrupted processing, leaving only a clearly named partial file;
- output encoder unavailable;
- FFmpeg unavailable (produce silent output and warn);
- exceptions during processing while still releasing video handles and temp files.

“Offline” means inference makes no network calls after model files are downloaded. Add a download command and then support `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`.

## 12. General-purpose evaluation datasets

No single dataset tests category prompts, unusual objects, natural-language references, images, and tracking equally well. Use the following small evaluation suite. These datasets are for benchmarking the pretrained pipeline, not for training version 1.

### Primary video benchmark: LV-VIS

Use **LV-VIS (Large-Vocabulary Video Instance Segmentation)** as the main general-purpose video benchmark. It provides instance masks and identities across video for 1,196 object categories and was designed for open-vocabulary video segmentation.

Create a deterministic laptop-sized manifest containing:

- 20 common-category videos;
- 20 rare-category videos;
- 10 crowded or multi-instance videos;
- 10 negative-prompt videos in which the requested item is verified absent.

Include household objects, tools, food, clothing, furniture, vehicles, animals, and small objects. Do not let cat/dog examples dominate the benchmark. LV-VIS uses a CC BY-NC-SA 4.0 license, so keep its media outside Git and use it only where those terms permit.

### Natural-language benchmark: Refer-YouTube-VOS

Use **Refer-YouTube-VOS** for prompts that refer to one particular object rather than an entire category. It contains 3,978 videos, about 131,000 masks, and 15,000 human language expressions. Its prompts test cases such as:

- `the red car`;
- `the person on the left`;
- one object among several of the same category;
- longer descriptions with appearance or spatial attributes.

Report these results separately. Grounding DINO plus SAM 2 is a modular open-vocabulary baseline, not a dedicated referring-video model, so relational expressions will generally be harder than category names.

### Lightweight smoke benchmark: YouTube-VIS 2021

Use a small, balanced **YouTube-VIS 2021** subset for quick regression runs. The full release has 2,985 training videos, 421 validation videos, 40 categories, 8,171 instances, and about 232,000 masks. Cat/dog can remain two smoke-test cases, but the subset must cover many categories.

```text
data/
  lv_vis/                       # ignored by Git
    raw/
    manifests/
      general_common.json
      general_rare.json
      negative_prompts.json
  refer_youtube_vos/            # ignored by Git
    raw/
    manifests/
      referring_validation.json
  youtube_vis_2021/             # ignored by Git
    raw/
    manifests/
      smoke_balanced.json
  fixtures/                     # only generated or redistributable tiny data
```

Do not hard-code category IDs. Resolve names through each official annotation file's category table.

### Image evaluation

Initially, evaluate the image route on annotated frames sampled from the video datasets. Later, add a balanced LVIS image-validation subset; LVIS contains more than 1,200 categories and over two million instance masks. This tests the image code without narrowing the product to a fixed category list.

### Evaluation protocol

1. **Category prompts:** derive a simple noun prompt from the ground-truth category and annotate all matching instances.
2. **Referring prompts:** use the original human expression unchanged and select the referenced instance.
3. **Negative prompts:** request a verified-absent item and count false positives.
4. **Common versus rare:** report results separately to expose open-vocabulary failures.
5. **Image versus video:** report single-image mask quality separately from tracking quality.
6. **Regression subset:** freeze a diverse manifest of 5-10 small inputs while leaving the actual media outside source control.

Report mask IoU, region similarity `J`, boundary accuracy `F`, video AP at IoU 0.50, macro-average category recall, false-positive masks per 1,000 negative frames, track fragmentation, peak VRAM/RAM, and seconds per frame. Evaluate sparse annotations only on annotated frames.

### Licensing and download rules

Download datasets only from their official project pages after accepting their terms. YouTube-VIS and Refer-YouTube-VOS state that their annotations are CC BY 4.0 while the data is restricted to non-commercial research. LV-VIS is CC BY-NC-SA 4.0. Cite every dataset used, do not commit or redistribute its media/annotations, and do not treat research-only data as commercial product data.

Ignore all downloaded dataset roots under `data/`. Store only preparation code, schemas, authorized checksums, and instructions.

### Dataset preparation task

Implement one adapter-based command:

```bash
python scripts/prepare_dataset.py \
  --dataset lv-vis \
  --dataset-root data/lv_vis/raw \
  --strategy balanced \
  --max-videos 60 \
  --output data/lv_vis/manifests/general_eval.json
```

It must support `lv-vis`, `refer-youtube-vos`, and `youtube-vis`; validate the selected schema; verify referenced media; create deterministic manifests; print category/video/instance counts; and never download or rewrite official data. Test it with synthetic annotation JSON rather than copyrighted files.

## 13. Development milestones for small coding agents

Each milestone should be a separate change with tests. Later agents should not rewrite working earlier modules.

### Milestone 1: environment and smoke test

- Add `pyproject.toml`, installation instructions, `.gitignore`, and `scripts/smoke_test_gpu.py`.
- Load both tiny models and run one dummy/fixture image.
- Print device, dtype, peak VRAM, and model IDs.
- Acceptance: both models load on CUDA and the process exits successfully.

### Milestone 2: video I/O

- Implement probe, aspect-ratio resize, bounded chunk extraction, output writer, and cleanup.
- Test a generated 10-frame video without models.
- Acceptance: frame count/FPS/dimensions are preserved and memory does not grow with repeated chunks.

### Milestone 2A: dataset preparation

- Add all dataset exclusions and adapter-based `scripts/prepare_dataset.py`.
- Support balanced LV-VIS, Refer-YouTube-VOS, and YouTube-VIS manifests without assumed category IDs.
- Generate common, rare, referring-expression, and verified-negative manifests.
- Acceptance: a synthetic annotation fixture produces deterministic manifests, and no dataset media is staged for source control.

### Milestone 3: detector adapter

- Implement Grounding DINO Tiny loading and single-frame text detection.
- Add category/referring modes, box conversion, clipping, NMS, and threshold tests.
- Acceptance: several unrelated prompts (`cat`, `power drill`, `red cup`) select the intended objects on known fixtures, and an absent prompt returns no detection.

### Milestone 3A: still-image pipeline

- Connect Grounding DINO boxes to the SAM 2 image predictor.
- Render and export masks using the same schema as a one-frame video result.
- Acceptance: image input creates a correctly sized annotated image and JSON without initializing the video predictor.

### Milestone 4: SAM 2 chunk tracker

- Initialize the official video predictor on a frame directory, add box prompts, and yield masks frame by frame.
- Never collect the whole result in a dictionary.
- Acceptance: one prompted object yields one correctly shaped boolean mask per decoded frame.

### Milestone 5: end-to-end renderer

- Connect media routing, image/video I/O, detector, tracker, and renderer.
- Add the CLI and an unchanged-frame path for no detections.
- Acceptance: diverse image and video fixtures annotate only the requested objects; the cat-and-dog case remains one smoke test, not the full acceptance suite.

### Milestone 6: exports and stable IDs

- Add per-object PNG and COCO-RLE JSON exports.
- Add cross-chunk IoU association and tests for match/new/retired tracks.
- Acceptance: IDs remain stable across a simple two-chunk fixture.

### Milestone 7: resilience

- Add resolution/chunk-size OOM retries, partial-file behavior, disk checks, and useful logging.
- Add mocked OOM tests.
- Acceptance: a forced first-attempt OOM retries at lower resolution without duplicating output frames.

### Milestone 8: future API boundary

- Expose `annotate_media(config, progress_callback=None) -> AnnotationResult`, with image and video implementations behind it.
- Keep FastAPI out of the GPU worker initially; an API should enqueue one GPU job at a time.
- Acceptance: CLI uses the same service function that a future web route will call.

## 14. Minimum end-to-end acceptance suite

The cat-and-dog video remains a useful smoke test:

Run:

```bash
python -m video_annotator.cli annotate --input cat_dog.mp4 --prompt cat --output cats.mp4 --export-json cats.json --chunk-frames 30 --redetect-every 30
```

Pass criteria:

- output frame count equals input frame count;
- output dimensions and FPS match the source within container precision;
- every visible overlay corresponds to a cat;
- the dog has no mask;
- the later cat is discovered at the next re-detection boundary;
- JSON frame indices and mask dimensions match the source;
- peak VRAM stays below 8 GB;
- process RAM stays bounded by the configured chunk size;
- a second run with cached weights performs no network access.

Also require at least:

- one still image with a common household-object prompt;
- one rare LV-VIS category video;
- one multi-instance category prompt that returns all matching objects;
- one Refer-YouTube-VOS expression that selects only the described instance;
- one verified-absent prompt that produces no masks and does not fail the job.

Release reporting must include results across the balanced general-purpose manifest. Passing the cat-and-dog smoke test alone is not sufficient.

## 15. Known limitations to state honestly

- Grounding DINO can miss small, blurred, occluded, rare, fine-grained, or visually ambiguous objects and can confuse related categories.
- First-frame-only detection cannot discover later arrivals; periodic chunk-boundary detection is required.
- Chunking bounds memory but weakens identity continuity at boundaries.
- SAM 2 can drift after long occlusion. Re-detection limits but does not eliminate drift.
- Downsampling saves memory but loses detail around thin structures and small objects.
- Category prompts and referring prompts are different tasks. `cup` means all cups; `the red cup on the left` asks for a specific instance and may require spatial reasoning Grounding DINO cannot always perform.
- Motion-dependent expressions such as `the car that turns first` are outside version 1 because a still-frame grounding model cannot reliably understand them.
- “General purpose” means open-vocabulary best effort, not guaranteed segmentation of every imaginable item.

## 16. Definition of done

The project is complete when a fresh WSL environment can follow the README, cache both models, prepare authorized balanced manifests for the evaluation suite, annotate both images and videos under the stated hardware limits, create playable/rendered outputs and optional PNG/COCO-RLE masks, report common/rare/referring/negative benchmark results, handle no-match and simulated OOM cases, and expose a model-independent Python service suitable for a later FastAPI wrapper.
