# CLAUDE.md

Project: ChemVision-Tuned — fine-tune a small VLM for molecule image → SMILES, benchmark on Fireworks.

## Collaboration: push to GitHub often

Two people (and their agents) work on this repo at the same time. Every agent, including subagents:

- Commit and push small, working increments to `origin` frequently — at least after every self-contained step (a script works, a test passes, a doc section is done). Don't sit on unpushed work.
- `git pull --rebase` before starting work and before every push. Resolve conflicts rather than force-pushing; never `git push --force` to `main`.
- Keep commits focused with clear messages so the other person can follow along.
- Never commit secrets (`.env`) or large generated data (`data/raw/`, `data/images/`, `results/` are gitignored — keep it that way). Commit the scripts that regenerate them instead.

## Compute: prefer the local Mac when it's cheap and safe

Budget matters. Machine: Apple M5, 10 cores, 24 GB unified memory. Use local compute instead of Fireworks when it is possible, efficient, and won't overload the laptop.

Run locally:
- Data prep: RDKit parsing/canonicalization, rendering molecule images, building JSONL datasets.
- Scoring/eval math: SMILES validation, exact-match/Tanimoto metrics, plots.
- Quick smoke tests of a VLM on a handful of images, via MLX (`mlx-vlm`) with a 4-bit quantized model ≤ ~8B params.
- Unit tests.

Use Fireworks for:
- Fine-tuning (LoRA/SFT) — too slow and hot to do on the laptop.
- Large models (e.g. the 235B comparison model) and anything that doesn't fit in ~16 GB.
- The official benchmark runs, so base/large/tuned numbers come from the same platform.

Keep the laptop healthy:
- Cap parallel workers at ~6 (`multiprocessing`/`concurrent.futures`), leaving cores for the OS and the other agent.
- Process big datasets in chunks/streams; don't load all images into RAM.
- Don't run more than one local model at a time; run long jobs in the background with progress logging and check memory pressure if things slow down.
- Before a Fireworks run, test the pipeline end-to-end locally on a tiny subset (e.g. 10–50 samples) so paid runs don't fail on bugs.
