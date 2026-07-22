# Instructions for Coding Agents

Canonical project root: `D:\work\segment anything`. Work from this directory. The former OneDrive project copy has been moved; do not create new edits in the old path.

Before taking any action in this repository, read `NEXT_STEPS.md` completely and treat it as the current source of truth.

Important constraints:

- Continue from the documented working state; do not reinstall Python, CUDA, PyTorch, Grounding DINO, Transformers, or SAM 2.
- The immediate milestone is the LVIS 12-image evaluation described in `NEXT_STEPS.md`.
- Fix the official COCO image URL handling without disabling TLS certificate verification.
- Use the existing lightweight model adapters and keep inference within the documented 8 GB VRAM limits.
- Run the specified acceptance checks and real evaluation before claiming completion.
- Preserve existing local changes. Do not stage generated data, outputs, model files, caches, `.egg-info`, or executable installers.
- Use native Windows PowerShell syntax. A backtick copied as a standalone pip argument is invalid.

If actual project state differs from `NEXT_STEPS.md`, inspect the files and test evidence, update the handoff with the verified state, and continue from the first incomplete acceptance criterion.
