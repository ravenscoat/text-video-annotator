# Learned Whole-Video Selector: Agent Handoff

This is the active implementation handoff after the rule-based MeViS A/B gate failed. Read `AGENTS.md`, `NEXT_STEPS.md`, `TEMPORAL_MULTI_OBJECT_PLAN.md`, and this file completely before editing code.

## Objective

Keep Grounding DINO Tiny and SAM 2.1 Hiera Tiny, but replace online rule-only referring selection with a two-pass, whole-video architecture inspired by SOLA:

1. Generate a bounded set of candidate object tracks across the video.
2. Cache track evidence on CPU/disk.
3. Score complete tracks against the full expression.
4. Export every matching track and no non-matching track.

The system is general purpose. Class names such as dog, horse, person, or bird are examples only. Expressions may identify targets by category, appearance, motion, position, quantity, identity, temporal event, or relationship. They may select one object, several objects, all matching objects, or no object.

## Temporal-Order and Adaptive-Anchor Revision

The selector must preserve the order of events. Aggregate speed, displacement, and pooled appearance alone cannot distinguish expressions such as `jumps high then jumps far` from the same actions in reverse order. Every selected anchor therefore produces an ordered token containing semantic appearance, normalized timestamp, position, size, velocity, acceleration, visibility, confidence, and relationship evidence. The learned selector consumes the token sequence in chronological order.

Candidate discovery uses a hybrid anchor policy, not a single best frame and not only a fixed uniform sample:

- retain 4-6 uniform coverage anchors across the complete video;
- add at most 2 adaptive anchors using detector confidence, vision-text alignment, novelty from existing tracks, and distance from already selected anchors;
- retain periodic redetection for late entries;
- never let adaptive scoring remove uniform temporal coverage;
- stop adding adaptive anchors when candidate coverage stabilizes.

The training objective may add an auxiliary temporal-order loss using reversed or shuffled track-token sequences as negatives. Multi-label track selection remains the primary loss.

## Verified Starting State

- Work only in `D:\work\segment anything`.
- Do not use or recreate the former OneDrive checkout.
- Python, CUDA, PyTorch, Grounding DINO Tiny, SAM 2.1 Hiera Tiny, model caches, and datasets are already installed.
- The model-free suite last passed with `34 passed`.
- Ref-DAVIS17 10-case result: J `0.6723`, F `0.5861`, recall `0.8000`, 276 false-positive masks, 138 false-negative masks, fragmentation `0`.
- Exact first-50 MeViS baseline: J `0.3460`, F `0.2457`, recall `0.3934`, FP `3209`, FN `1002`, fragmentation `61`.
- Rule-based upgrade: J `0.3436`, F `0.2452`, recall `0.3840`, FP `2758`, FN `1016`, fragmentation `53`.
- The gate failed. Do not run cases 50+ and do not claim the rule-based selector succeeded.

## Non-Negotiable Constraints

- 8 GB VRAM and 16 GB system RAM.
- Keep the existing Grounding DINO Tiny and SAM 2.1 Hiera Tiny adapters.
- Freeze both foundation models. Do not fine-tune either model.
- Stream frames. At most one anchor frame or one small window may be on the GPU at a time.
- Cap active candidate tracks at 16 by default and make the cap configurable.
- Store caches under `outputs/` or another ignored generated directory; never commit them.
- Use mixed precision where already supported and keep peak allocated CUDA memory below 7.5 GB.
- Preserve category-union behavior for explicit prompts such as `--target dog --target horse`.
- Do not reinstall or replace the working environment.
- Do not integrate a large video-language model, proprietary API, or cloud-only dependency.

## Architectural Boundary

Do not send a full motion sentence to Grounding DINO and expect it to resolve the referred track. Grounding DINO proposes concrete candidate objects. The selector resolves the full expression after track evidence exists.

```text
prompt + streamed video
        |
        v
concrete target hints + periodic/anchor proposals
        |
        v
Grounding DINO Tiny + SAM 2.1 Tiny candidate tracks
        |
        v
versioned disk cache: masks/boxes/visibility/features
        |
        v
frozen text/visual encoders + small trainable temporal selector
        |
        v
independent score per track -> zero/one/many selected tracks
        |
        v
render video and export masks/JSON from cached tracks
```

## Work Order

Complete one phase at a time. Keep tests passing after every phase. Do not start selector training before cache validation succeeds.

### Phase 0: Preserve and Diagnose the Failed Baseline

1. Do not overwrite existing first-50 MeViS outputs.
2. Add a small diagnostic manifest containing 10 expressions from the same evaluated cases. It must cover, where present:
   - a single target;
   - multiple targets;
   - moving versus stationary targets;
   - a spatial or interaction relation;
   - late appearance or a temporal event;
   - a no-target/ambiguous negative if the dataset provides one.
3. Record the exact case IDs. Selection must be deterministic; do not cherry-pick new cases after seeing new-model results.
4. Add a diagnostic report showing candidate recall separately from selector accuracy. A selector cannot recover a ground-truth object that candidate generation never produced.

Acceptance:

- The manifest is deterministic and contains no media files.
- Existing baseline metrics remain unchanged.
- Reports distinguish `candidate_missing` from `candidate_present_but_not_selected`.

### Phase 1: Versioned Candidate-Track Cache

Add a focused cache module, preferably `video_annotator/track_cache.py`, with typed records and tests.

Each video cache must include:

- schema version and configuration fingerprint;
- source video/frame identity, dimensions, FPS, and frame count;
- stable track ID, normalized detector label, and candidate-generation scope;
- per-frame visibility, box, mask reference, detector confidence, and propagation confidence when available;
- normalized center, area, velocity, acceleration, direction, first/last visibility, and same-class rank;
- chronologically ordered per-track anchor tokens; do not replace them with only one pooled appearance vector;
- pairwise features needed for near/touching/left/right/front/behind relations;
- exact model names, proposal prompts/target vocabulary, thresholds, long side, redetection interval, and candidate cap.

Masks may be stored as PNGs or compact RLE. Do not put a full uncompressed mask tensor for every frame into one JSON file. Write atomically: temporary path first, then rename after validation.

Expose separate operations for:

1. generating a cache;
2. validating/loading a compatible cache;
3. scoring expressions from an existing cache;
4. rendering/exporting selected cached tracks.

Acceptance tests:

- round-trip a synthetic two-track cache;
- reject a corrupt or incompatible schema;
- reject a cache whose source/config fingerprint differs;
- confirm repeated expressions with a compatible candidate scope reuse the cache without invoking detector/tracker mocks;
- confirm mask dimensions and track IDs survive round-trip.

### Phase 2: Improve Candidate Recall Without Selecting Targets Early

Refactor candidate generation so compatible expressions can share tracks. For a benchmark video with multiple expressions, build the proposal scope from the union of concrete target hints across those expressions plus the fallback proposals, then generate once. For an arbitrary product request, a request-scoped cache is acceptable and its proposal vocabulary must be fingerprinted. Never reuse a cache whose scope cannot contain the new expression's concrete targets.

Requirements:

1. Start with 4-6 uniform coverage anchors, add at most 2 adaptive anchors, and retain periodic redetection for late entries. Adaptive ranking combines detector confidence, semantic vision-text alignment, track novelty, and temporal distance from existing anchors.
2. Use concrete noun/category hints when available, but never use action, relation, or rule scores to discard a candidate before full-video selection.
3. Add a configurable fallback proposal route for expressions whose noun hint yields no candidates. Prefer an existing lightweight/category-agnostic SAM 2 prompt strategy or a small fixed grid of point prompts; do not add a heavy model.
4. Merge duplicate proposals using label compatibility, mask/box overlap, and trajectory consistency.
5. Retain at most 16 active tracks with deterministic pruning based on coverage and detection confidence, not expression-specific motion rules.
6. Record why every proposal was kept, merged, or dropped.
7. Store both the uniform/adaptive reason and the chronological frame index for every selected anchor.

Acceptance:

- objects first appearing after frame zero can become tracks;
- duplicate tracks are merged without recycling IDs;
- compatible expressions reuse one scope-complete candidate cache, while an incompatible target vocabulary invalidates it;
- candidate recall is reported on the fixed 10-case diagnostic set;
- peak CUDA allocation stays below 7.5 GB.

### Phase 3: Feature Extraction and Selector Data

Implement feature extraction behind small interfaces so encoders can be changed without rewriting tracking.

Required feature groups:

- appearance: frozen semantic embeddings of masked crops at ordered anchors;
- language: frozen token-level or sentence embeddings of the complete expression;
- motion: per-anchor normalized position, size, velocity, and acceleration plus aggregate diagnostics;
- visibility/time: normalized timestamp, first, last, duration, intermittency, and anchor-presence mask;
- relationships: per-anchor relative positions, distances, overlap/contact, and their temporal changes;
- provenance: whether each anchor was uniform, adaptive, or periodic-redetection evidence.

Prefer features already available from SAM 2 object pointers/tokens if the installed API exposes them safely. If it does not, use a small frozen image/text encoder in an isolated dependency addition. Do not silently download a model during tests; document and cache it explicitly.

CLIP-style similarity is useful for appearance alignment but is not sufficient evidence for motion order. Preserve the ordered token sequence for the learned temporal module. Bump the cache schema when semantic embeddings and ordered token metadata are introduced; reject old incompatible caches with an actionable message.

Build MeViS selector samples as `(video cache, expression, per-track binary labels)`. One expression can have several positive tracks or none. Split by video, never by expression, to avoid leakage between expressions describing the same video.

Acceptance tests:

- feature shapes and padding masks are deterministic;
- chronological token order and normalized timestamps survive cache round-trip;
- a late/short track receives its own visible anchors;
- multi-positive and zero-positive labels are preserved;
- no video ID appears in more than one train/validation split;
- features contain no NaN/Inf values;
- feature extraction streams anchors rather than retaining the video on GPU.

### Phase 4: Small Learned Whole-Video Selector

Add a small selector module, preferably `video_annotator/selector.py`. Freeze every encoder and train only a lightweight fusion/scoring head.

Minimum behavior:

- consume the full expression plus the chronologically ordered token sequence for every candidate track;
- use an order-aware temporal Transformer, GRU, or equivalently tested sequence model;
- output one independent logit per track, not a single softmax winner;
- support multiple positives with class-balanced BCE or focal loss;
- support no-target results with a calibrated threshold;
- expose per-track score and feature-level diagnostics in JSON;
- run inference with bounded memory and without loading Grounding DINO and selector training tensors simultaneously when avoidable.

The existing rule-based `MotionIntent` score may remain as a diagnostic feature or fallback. It must not be the final authority in learned mode.

Acceptance tests:

- a synthetic training set can be overfit, proving the training loop and checkpoint round-trip;
- multi-target examples can select two or more tracks;
- a negative example selects no tracks;
- permuting non-temporal track order does not corrupt IDs;
- a checkpoint/config mismatch fails with an actionable error;
- reversing or shuffling temporal tokens changes the order-sensitive representation while leaving track IDs intact;
- an auxiliary order-discrimination loss can distinguish a real sequence from reversed/shuffled negatives on a synthetic fixture.

### Phase 5: Train and Calibrate on MeViS

1. Train only the selector head using MeViS training data.
2. Use a video-disjoint validation split or the provided held-out split. Do not tune on the fixed first-50 A/B cases.
3. Handle class imbalance and include hard negatives consisting of visually similar non-target tracks.
4. Save checkpoints and training logs under ignored output directories.
5. Calibrate the selection threshold on validation data, including no-target and multi-target cases.
6. Record reproducible seed, split, epochs, batch/accumulation settings, peak VRAM, and best validation metric.
7. Keep the primary class-balanced multi-label selection loss. If used, add reversed/shuffled sequence discrimination only as an auxiliary loss with separately reported weight and validation effect.

Before preparing samples, create a dataset identity report containing the MeViS release/version, split, metadata and mask filenames with SHA-256 hashes, video/expression counts, no-target availability, audio usage, and metric definitions. Do not mix a newer MeViS release with the existing first-50 baseline. Treat another release as a separate benchmark.

Do not claim completion if training examples lack candidate tracks for their ground truth. Report the candidate-recall ceiling first.

### Phase 6: Evaluation Gates

Run gates in this order:

1. Model-free unit suite.
2. Fixed 10-case diagnostic evaluation with saved videos/previews.
3. Existing Ref-DAVIS17 10-case smoke regression.
4. Exact MeViS cases 0-49, saved to a new output directory.
5. Only after the gate passes, continue cases 50+.

Required MeViS comparison:

| Metric | Original baseline | Rule baseline | Learned selector |
|---|---:|---:|---:|
| Region Jaccard | 0.3460 | 0.3436 | |
| Boundary F | 0.2457 | 0.2452 | |
| Recall@0.50 | 0.3934 | 0.3840 | |
| False-positive masks | 3209 | 2758 | |
| False-negative masks | 1002 | 1016 | |
| Fragmentation | 61 | 53 | |

Pass only if one condition holds:

- J improves by at least `+0.05` absolute over `0.3460`; or
- false-positive masks drop by at least `25%` from `3209` while J is not below `0.3460`.

Additionally:

- candidate recall must be reported;
- multi-target and no-target diagnostic cases must behave correctly;
- no regression may be hidden by changing the case list or thresholds per case;
- Ref-DAVIS J must not drop by more than `0.03` absolute;
- peak allocated CUDA memory must remain below 7.5 GB.

If the gate fails, stop. Inspect whether the cause is candidate recall, feature quality, or selector error. Do not run more MeViS batches and do not tune on a different subset.

## SAMWISE Benchmark Track

SAMWISE is an optional external benchmark, not the main implementation dependency.

Rules:

1. Create a separate Python 3.10 environment outside the existing `.venv` only after explicit approval.
2. Do not modify or downgrade the working project environment.
3. Use official pretrained weights and official inference code.
4. Start with one short clip at long side 512 and measure peak VRAM.
5. Stop if projected or measured allocation exceeds 7.5 GB or if setup requires replacing the current CUDA/PyTorch stack.
6. Record accuracy/runtime/memory as comparison evidence; do not copy its code into this project without license and compatibility review.

This benchmark can proceed independently after Phase 0, but failure to run SAMWISE does not block the SOLA-style local implementation.

## Commands and Safety

Use native Windows PowerShell. A backtick is a line-continuation character only when it is the final character of a continued PowerShell line; never pass it as a pip requirement.

Before editing:

```powershell
Set-Location 'D:\work\segment anything'
git status --short
```

Fast verification after each phase:

```powershell
.\.venv\Scripts\python.exe -m compileall video_annotator scripts tests
.\.venv\Scripts\python.exe -m pytest -q
git diff --check
```

Do not use `git add -A`. Never stage `data/`, `outputs/`, `.venv/`, caches, model weights, videos, masks, `.egg-info`, or installers. Do not push until tests and the phase-specific acceptance checks pass.

## Required Agent Reporting

At the end of each assigned phase, report:

- files changed;
- exact commands run and their results;
- test count;
- real inference cases run, if any;
- peak VRAM, if GPU inference ran;
- acceptance criteria passed/failed;
- known limitations;
- exact next incomplete criterion.

Update `NEXT_STEPS.md` only with verified evidence. Never mark a phase complete based only on code presence.

## Copy-Paste Prompt for a Coding Agent

```text
Work only in D:\work\segment anything.

Read AGENTS.md, NEXT_STEPS.md, TEMPORAL_MULTI_OBJECT_PLAN.md, and LEARNED_SELECTOR_PLAN.md completely before changing anything. LEARNED_SELECTOR_PLAN.md is the active post-gate plan.

Continue from the first incomplete acceptance criterion in LEARNED_SELECTOR_PLAN.md. Complete only one phase at a time. Do not redo environment installation, model downloads, completed datasets, smoke tests, or completed benchmark batches.

Keep Grounding DINO Tiny and SAM 2.1 Hiera Tiny frozen. The machine has an RTX 5060 Laptop GPU with 8 GB VRAM and 16 GB RAM. Stream frames, cache track evidence on CPU/disk, cap candidate tracks at 16, and keep peak allocated CUDA memory below 7.5 GB.

The product is general-purpose text-guided video object selection. Example classes are illustrative only. A prompt may select zero, one, or multiple tracks by category, appearance, motion, position, quantity, identity, time, or relationship. Explicit category-union prompts must keep working.

Do not continue MeViS cases 50+ until the learned-selector A/B gate passes. Do not tune on a different case subset, overwrite baseline outputs, hide missing candidates as selector failures, or claim unrestricted language understanding.

Preserve existing changes. Use apply_patch for edits and native Windows PowerShell commands. Do not commit data, outputs, videos, masks, weights, caches, .egg-info, virtual environments, or installers. Run the full model-free tests and the assigned phase's acceptance checks. Update NEXT_STEPS.md with verified evidence and state the exact next incomplete criterion.
```

## Recommended Next Assignment

Give the next coding agent this bounded task from the current verified state:

```text
Implement the remaining Phase 3 semantic-sequence work only. Add a frozen lightweight image/text embedding adapter behind a mockable interface. Upgrade the cache schema to store chronologically ordered per-track anchor tokens containing semantic crop embeddings, normalized timestamps, position/size, velocity/acceleration, visibility, confidence, relationship features, and anchor provenance. Implement hybrid anchor selection with 4-6 uniform coverage anchors plus at most 2 adaptive anchors, while preserving periodic redetection. Add model-free tests with mocked encoder outputs for chronological order, late/short tracks, schema rejection, deterministic padding, and adaptive-anchor budget. Do not download a model, train the selector, generate full MeViS caches, or rerun the first-50 evaluation yet.
```
