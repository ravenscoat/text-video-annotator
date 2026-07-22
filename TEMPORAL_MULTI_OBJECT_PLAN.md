# Temporal Multi-Object Upgrade Plan

This document is the implementation handoff for improving the text-video annotator after the first 50 MeViS expression cases. Coding agents must read `AGENTS.md`, `NEXT_STEPS.md`, and this file completely before editing code.

## Decision

Pause the remaining MeViS batches. The first 50 expression cases are enough to establish the baseline:

- Mean of the five equal-sized batch means: region Jaccard `0.3460`, boundary F `0.2457`, recall@IoU 0.50 `0.3934`.
- Total false-positive masks: `3,209`.
- Total false-negative masks: `1,002`.
- Total track fragmentation: `61`.

Do not spend more GPU time on the remaining expressions until the upgraded pipeline passes an A/B test on these same 50 cases.

## Product Requirements

The upgraded tool must support two distinct prompt modes.

### 1. Category union

Example input:

```text
dog and horse
```

Expected behavior:

- Detect and segment every visible dog and every visible horse.
- Assign a stable, unique track ID to every instance.
- Do not annotate the cat or any other unrequested class.
- Preserve the requested class label on every mask and JSON record.
- Detect requested objects that first appear after frame zero.

The canonical detector prompt for this example is:

```text
dog . horse .
```

Grounding DINO officially supports period-separated class prompts. SAM 2 officially supports multiple object IDs in one video inference state.

### 2. Referring/motion expression

Example inputs:

```text
the dog running to the left
the horse walking ahead
the two elephants touching trunks
```

Expected behavior:

- Generate candidate tracks for the concrete object class or classes.
- Use evidence from multiple frames to select the track or tracks matching the motion expression.
- Support expressions referring to one object, multiple objects, or no object.
- Do not assume the target is identifiable in the first frame.

Category union is the first milestone. Motion-expression selection is the second milestone. Do not mix the two implementations into one untestable function.

## Research Basis

- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) demonstrates multi-class prompts such as `chair . person . dog .`.
- [Hugging Face Grounding DINO documentation](https://huggingface.co/docs/transformers/model_doc/grounding-dino) explicitly documents multi-class prompts such as `a cat. a dog.`.
- [Meta SAM 2](https://github.com/facebookresearch/sam2) documents adding prompts with unique object IDs and tracking multiple objects in one video inference state.
- [IDEA Research Grounded SAM 2](https://github.com/IDEA-Research/Grounded-SAM-2) provides video grounding/tracking and continuous-ID reference implementations. Treat these as design references; do not replace the working local model installation.
- [MeViS](https://github.com/henghuiding/MeViS) contains multi-target and motion-focused expressions that often cannot be resolved from a single frame.
- The [MeViS ICCV paper](https://openaccess.thecvf.com/content/ICCV2023/html/Ding_MeViS_A_Large-scale_Benchmark_for_Video_Segmentation_with_Motion_Expressions_ICCV_2023_paper.html) uses global temporal context and motion-language matching.
- The [2025 MeViS challenge winner](https://arxiv.org/abs/2504.05178) uniformly samples video frames for whole-video understanding and combines expert outputs. This supports anchor-frame sampling rather than first-frame-only grounding.
- [SAMURAI](https://github.com/yangchris11/samurai) adds motion-aware memory and a Kalman-filter motion prior to SAM 2 tracking. Borrow the lightweight motion-prior idea; do not adopt its entire stack in the first implementation.

## Verified Current-Code Problems

1. `AnnotationConfig.redetect_every` is exposed by the CLI but is never used by `video_annotator/pipeline.py`.
2. Detection occurs on the first frame of every chunk, not at the requested redetection interval.
3. SAM 2 state is reset for every chunk, so memory does not persist across chunk boundaries.
4. `identity.associate` uses only same-label box IoU. It has no center-motion prediction, mask IoU, missed-frame tolerance, or label normalization.
5. `max_objects` is a global cap. One common class can consume the limit and prevent another requested class from appearing.
6. Free-form `dog and horse` is passed directly to the detector. It is not normalized into Grounding DINO's canonical period-separated class format.
7. The pipeline does not explicitly filter returned labels against the requested target set.
8. MeViS evaluation treats each expression independently and repeats detection/tracking for expressions that share a video. This is correct for the baseline but inefficient for temporal candidate selection.

## Target Architecture

```text
raw prompt
   |
   v
PromptSpec parser ----> requested classes: [dog, horse]
   |                    mode: category_union | referring
   v
anchor-frame detector: frame 0, N, 2N, ...
   |
   v
class-filtered detections + per-class NMS/caps
   |
   v
TrackManager: stable global IDs, late appearances, missed-frame tolerance
   |
   v
SAM 2 multi-object propagation/correction
   |
   +---- category_union: export every requested-class track
   |
   +---- referring: temporal feature extraction and track selection
```

## Data Contracts

Add these types to `video_annotator/types.py` or a dedicated `prompts.py` module:

```python
@dataclass(frozen=True)
class PromptSpec:
    raw: str
    mode: str                    # category_union | referring
    targets: tuple[str, ...]     # normalized concrete classes
    detector_prompt: str         # e.g. "dog . horse ."
    motion_text: str | None = None

@dataclass
class Track:
    track_id: int
    label: str
    score: float
    last_box_xyxy: tuple[float, float, float, float]
    last_mask: np.ndarray | None
    last_seen_frame: int
    missed_frames: int = 0
    centers: list[tuple[int, float, float]] = field(default_factory=list)
    areas: list[tuple[int, float]] = field(default_factory=list)
```

Extend exported JSON objects with:

```json
{
  "track_id": 7,
  "label": "horse",
  "requested_target": "horse",
  "detector_score": 0.81,
  "selection_score": null
}
```

Do not silently rename track IDs between chunks.

## Implementation Tasks

Complete these tasks sequentially. Each task must leave tests passing before the next task begins.

### Task 1: Prompt parsing and explicit multi-target API

Files:

- Add `video_annotator/prompts.py`.
- Update `video_annotator/config.py`.
- Update `video_annotator/cli.py`.
- Add `tests/test_prompts.py`.

Requirements:

1. Preserve `--prompt` for backward compatibility.
2. Add repeatable `--target` options as the unambiguous API:

   ```powershell
   video-annotator annotate --input input.mp4 --target dog --target horse --output output.mp4
   ```

3. When only `--prompt` is supplied in category mode, support conservative parsing of commas, periods, and the word `and`:

   - `dog and horse` -> `("dog", "horse")`
   - `dog, horse` -> `("dog", "horse")`
   - `dog . horse .` -> `("dog", "horse")`
   - remove standalone `only`, articles, surrounding whitespace, and duplicates.

4. Do not aggressively split referring expressions. `the dog running left` remains one referring expression with a concrete target hint if available.
5. Produce `detector_prompt = "dog . horse ."`.
6. Reject an empty target list with an actionable error.

Acceptance tests:

- The three equivalent category inputs above produce the same `PromptSpec`.
- `dog and dog` contains one target.
- `annotate cats only` normalizes to `cat` only if an explicit singularization rule is tested; otherwise preserve `cats`.
- Referring mode does not split `the man in black and white clothes` into multiple classes.

### Task 2: Multi-class Grounding DINO adapter

Files:

- Update `video_annotator/detector.py`.
- Update `video_annotator/types.py` if needed.
- Add `tests/test_detector_postprocess.py` using mocked model outputs.

Requirements:

1. Accept a `PromptSpec` or explicit target list while retaining the old string method as a compatibility wrapper.
2. Run one Grounding DINO inference for all requested classes using the period-separated prompt.
3. Normalize returned text labels and map each detection to exactly one requested target.
4. Drop labels that cannot be mapped to a requested target. This is the primary defense against annotating the cat when only dog and horse were requested.
5. Apply class-aware NMS, default IoU `0.50`.
6. Add `max_objects_per_target`, default `5`; do not let dog detections consume the horse allocation.
7. Keep detector scores and mapped target labels.

Acceptance tests:

- Mock results containing dog, horse, and cat retain dog and horse only.
- Multiple dogs and horses survive when below the per-target cap.
- Duplicate dog boxes are removed by class-aware NMS.
- A low-scoring requested class does not disappear because another class has many high-scoring boxes.

### Task 3: Stable multi-object TrackManager

Files:

- Replace or extend `video_annotator/identity.py`.
- Add `tests/test_track_manager.py`.

Requirements:

1. Match only tracks with the same normalized requested label.
2. Association score should combine:

   ```text
   0.45 * box IoU
   0.35 * mask IoU when available
   0.20 * center-motion consistency
   ```

   Renormalize weights when a component is unavailable.

3. Use one-to-one assignment. Hungarian assignment is preferred; deterministic greedy matching is acceptable if tested.
4. Keep unmatched tracks alive for `max_missed_redetections=2`.
5. Create a new global ID for an unmatched requested-class detection, including objects appearing after frame zero.
6. Never recycle a global ID within one video.
7. Record center and mask-area histories for motion scoring.

Acceptance tests:

- Two crossing dogs retain IDs better than box-IoU-only association.
- A dog disappearing for one redetection interval can reclaim its ID.
- A newly appearing horse receives a new ID.
- A cat detection never associates with a dog or horse track.

### Task 4: Real periodic redetection and SAM 2 correction

Files:

- Refactor `video_annotator/pipeline.py`.
- Extend `video_annotator/tracker.py`.
- Add model-free orchestration tests in `tests/test_redetection.py`.

Requirements:

1. Make `redetect_every` functional and independent from `chunk_frames`.
2. Default values for the 8 GB laptop:

   ```text
   long_side = 512
   chunk_frames = 30
   redetect_every = 15
   max_objects_per_target = 5
   max_active_tracks = 10
   ```

3. Detect at frame 0 and every `redetect_every` frames.
4. Associate detections with active global tracks.
5. Use each global track ID as the SAM 2 object ID where the API permits.
6. Add requested objects that appear later. The official SAM 2 tooling explicitly supports datasets where objects appear later; use that behavior as the reference.
7. At a redetection frame, add a correction box or mask for an existing track rather than creating a duplicate track.
8. If the current SAM 2 wrapper cannot safely add prompts during propagation, process overlapping windows and preserve global IDs through `TrackManager`. Document this limitation in code.
9. Continue frame streaming and CPU offloading. Never hold the full video in RAM or VRAM.
10. Catch CUDA OOM only, retry the current window at 448 and then 384, and record the fallback.

Acceptance tests with fake detector/tracker:

- Detector calls occur at exact configured frame indices.
- `chunk_frames=30` and `redetect_every=15` cause detections at 0, 15, 30, 45, not merely 0 and 30.
- An object appearing at frame 30 is added and exported afterward.
- Existing IDs survive window boundaries.
- No requested detections produces a valid unannotated output rather than a crash.

### Task 5: Category-union end-to-end acceptance

Add a small fixture video or use a locally created fixture whose objects are known. Do not commit large media.

Command:

```powershell
video-annotator annotate --input INPUT.mp4 --target dog --target horse --output outputs\\dog_horse.mp4 --export-json outputs\\dog_horse.json --export-masks outputs\\dog_horse_masks --long-side 512 --chunk-frames 30 --redetect-every 15
```

Acceptance criteria:

- At least one dog and one horse are annotated when both are visibly present and detected.
- A visible cat is not exported.
- Every exported object label belongs to `{dog, horse}`.
- Multiple dogs or multiple horses have distinct stable IDs.
- Annotated frame count equals source frame count.
- JSON and PNG masks agree on object IDs.
- Peak allocated CUDA memory remains below 7.5 GB.

Do not proceed to motion-expression work until this passes.

### Task 6: Motion-expression candidate generation

Files:

- Add `video_annotator/motion.py`.
- Add `video_annotator/referring.py`.
- Add `tests/test_motion.py`.

Requirements:

1. Uniformly sample `5` to `8` anchor frames across the full video, capped so only one anchor frame is resident on GPU at a time.
2. Detect the concrete noun targets on every anchor frame.
3. Build candidate tracks before choosing a referred target.
4. Compute lightweight per-track temporal features on CPU:

   - normalized center positions;
   - displacement and mean velocity;
   - direction histogram;
   - visible-frame fraction;
   - first/last visible frame;
   - relative left/right/front rank among same-class tracks;
   - mask-area change;
   - contact/proximity to other tracks.

5. Implement a tested baseline `MotionIntent` parser for common MeViS concepts:

   - left, right, up, down;
   - moving, stationary;
   - ahead/front, behind/back;
   - first, last, least-visible;
   - approaching, separating;
   - touching/near.

6. Score every candidate track against the expression. Do not discard candidates before temporal features are available.
7. Allow multiple selected tracks for plural/multi-target expressions.
8. Support a no-target result when every score is below a configurable threshold.
9. Preserve a diagnostic explanation in JSON:

   ```json
   {
     "selection_score": 0.72,
     "selection_reasons": ["moves_left", "visible_0.94"]
   }
   ```

This rule-based scorer is a baseline. Do not claim it solves unrestricted natural-language motion reasoning.

### Task 7: Optional motion-aware tracking experiment

Only after Tasks 1-6 pass, evaluate whether a Kalman box prior inspired by SAMURAI improves fragmentation. Do not replace the working SAM 2 package or install a second incompatible model stack.

Experiment:

- Predict the next center/scale from track history.
- Prefer SAM 2 masks whose boxes are consistent with the predicted state.
- Compare with and without the prior on the same first 50 MeViS cases.

Accept only if it reduces fragmentation without materially reducing Jaccard.

### Task 8: A/B evaluation gate

Rerun exactly MeViS cases 0-49 using the upgraded pipeline. Save results under a new output directory; never overwrite the baseline.

Required comparison report:

| Metric | Baseline | Upgraded | Delta |
|---|---:|---:|---:|
| Region Jaccard | 0.3460 | | |
| Boundary F | 0.2457 | | |
| Recall@0.50 | 0.3934 | | |
| False-positive masks | 3,209 | | |
| False-negative masks | 1,002 | | |
| Fragmentation | 61 | | |

Gate for continuing the remaining MeViS evaluation:

- Region Jaccard improves by at least `+0.05` absolute, or false positives drop by at least `25%` without reducing Jaccard.
- Category-union dog/horse/cat acceptance passes.
- No regression on the existing LVIS, LV-VIS, and Ref-DAVIS smoke tests.

If the gate fails, report the failure and inspect candidate-selection diagnostics. Do not hide poor results by changing the evaluation subset.

## Small-Model Work Instructions

Send one task at a time to a smaller coding model. Include this exact preamble:

```text
Work in D:\work\segment anything.
Read AGENTS.md, NEXT_STEPS.md, and TEMPORAL_MULTI_OBJECT_PLAN.md completely.
Continue only the assigned task from TEMPORAL_MULTI_OBJECT_PLAN.md.
Do not reinstall Python, PyTorch, CUDA, Grounding DINO, Transformers, or SAM 2.
Do not run additional MeViS batches unless the Task 8 gate is reached.
Preserve existing changes and generated outputs. Do not commit data, videos, masks, model files, caches, or installers.
Use apply_patch for edits and native Windows PowerShell commands.
Run the task's model-free tests and report exact evidence. Do not claim later tasks are complete.
```

Recommended task sequence for smaller models:

1. `Implement Task 1 only: PromptSpec and explicit multi-target CLI.`
2. `Implement Task 2 only: multi-class detector postprocessing.`
3. `Implement Task 3 only: stable TrackManager.`
4. `Implement Task 4 only: periodic redetection orchestration.`
5. `Execute Task 5 category-union acceptance and fix only defects within Tasks 1-4.`
6. `Implement Task 6 only: temporal features and MotionIntent baseline.`
7. `Run Task 8 A/B evaluation; do not tune on different cases.`

Each model must update `NEXT_STEPS.md` with verified progress and the next incomplete acceptance criterion.

## Out of Scope for This Upgrade

- Training or fine-tuning a new detector.
- Replacing Grounding DINO Tiny or SAM 2.1 Hiera Tiny.
- Installing a large video-language model on the 8 GB laptop.
- Cloud-only Grounding DINO 1.5, DINO-X, or proprietary APIs.
- Real-time processing.
- Continuing all 184 MeViS cases before the A/B gate passes.

## Definition of Done

This upgrade is complete only when:

- `dog and horse` is represented as two explicit requested targets.
- All detected dog and horse instances are tracked with stable IDs.
- Unrequested cat detections are absent from masks and JSON.
- Late-appearing requested objects can be added.
- `redetect_every` controls actual detector calls.
- Motion expressions use multi-frame candidate evidence.
- Existing fast tests pass.
- The same first 50 MeViS cases show a measurable improvement according to the Task 8 gate.
- Source changes and documentation are committed and pushed without generated data.
