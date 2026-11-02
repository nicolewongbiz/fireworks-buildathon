# CLAUDE.md

Project: ChemVision-Tuned — fine-tune a small VLM for molecule image → SMILES, benchmark on Fireworks.

Base model: **Qwen3.5-9B** (Fireworks `accounts/fireworks/models/qwen3p5-9b`, HF `Qwen/Qwen3.5-9B`). Not Qwen3-VL-8B: Fireworks lists it "Training: Not supported" (`useTrainingV2: false`) and rejects SFT jobs. Qwen3.5 thinks by default — send `chat_template_kwargs: {"enable_thinking": false}` (vLLM) to get a bare answer.

As of 2026-11-01, Fireworks dataset uploads are disabled on the account (`FailedPrecondition`); waiting on Fireworks support. Until then no managed SFT job can start.

## Collaboration: push to GitHub often

Two people (and their agents) work on this repo at the same time. Every agent, including subagents:

- Commit and push small, working increments to `origin` frequently — at least after every self-contained step (a script works, a test passes, a doc section is done). Don't sit on unpushed work.
- `git pull --rebase` before starting work and before every push. Resolve conflicts rather than force-pushing; never `git push --force` to `main`.
- Keep commits focused with clear messages so the other person can follow along.
- Never commit secrets (`.env`) or large generated data (`data/raw/`, `data/images/`, `results/` are gitignored — keep it that way). Commit the scripts that regenerate them instead.

## Ownership (two-person plan)

- **Person A — platform and training (nicolewongbiz):** Fireworks account, tuning jobs, deployments, Modal serving, cost log. Owns `src/predict.py`, `src/run_system.py`, `src/modal_serve.py`, `smoke/`, `results/`.
- **Person B — data and eval:** owns `data/`, `src/render.py`, `src/build_dataset.py`, `src/validate.py`, `src/evaluate.py`, `app.py`.
- Message the owner before changing their files. Only Person A creates or deletes Fireworks deployments or Modal GPU apps.
- Shared contracts (don't change without telling the other person): `data/train.jsonl` (Fireworks VLM SFT format), `data/test_*.json` (`[{"image_path", "smiles"}]`), `predict(image_path) -> {smiles, latency_s, tokens_in, tokens_out, attempts, escalated}`, and `results/<system>.json` (`{"summary": {...}, "items": [...]}`).

## Compute: prefer the local Mac when it's cheap and safe

Budget matters. Machine: Apple M5, 10 cores, 24 GB unified memory. Use local compute instead of Fireworks when it is possible, efficient, and won't overload the laptop.

Run locally:
- Data prep: RDKit parsing/canonicalization, rendering molecule images, building JSONL datasets.
- Scoring/eval math: SMILES validation, exact-match/Tanimoto metrics, plots.
- Quick smoke tests of a VLM on a handful of images, via MLX (`mlx-vlm`) with a 4-bit quantized model ≤ ~8B params.
- Unit tests.

Use Fireworks for:
- Fine-tuning (LoRA/SFT) — too slow and hot to do on the laptop. Muse Glimmer 30B LoRA SFT is free via UI / Serverless Training API until 2026-11-30; always pass an explicit LoRA rank (omitted rank = full-parameter = paid).
- The official benchmark runs, if hackathon rules require Fireworks serving.

Use Modal (workspace `nicolewongbiz`, ~$200 credit) for GPU inference that Fireworks would only offer as a dedicated deployment (e.g. Muse Glimmer 30B, which is not on Fireworks serverless):
- Modal bills per second and costs nothing idle; a Fireworks H200 deployment is $8/hr and bills while it is up.
- Download model weights into a Modal Volume from a CPU-only function, never inside a GPU container.
- Always set a container `timeout` and `scaledown_window` so nothing idles on a GPU, and confirm `modal app list` shows nothing running when done.

Any paid run (Fireworks deployment, non-free tuning job, Modal GPU job) needs the user's approval with an estimated cost first.

`firectl` blocks mutating commands (dataset/job/deployment create, delete, etc.) when run by an AI agent on this account. Don't work around it via the REST API: hand the exact command to the human to run. Read-only `firectl ... get/list` is fine. The block message echoes `--api-key` in plain text, so prefer commands that don't pass the key inline.

Keep the laptop healthy:
- Cap parallel workers at ~6 (`multiprocessing`/`concurrent.futures`), leaving cores for the OS and the other agent.
- Process big datasets in chunks/streams; don't load all images into RAM.
- Don't run more than one local model at a time; run long jobs in the background with progress logging and check memory pressure if things slow down.
- Before a Fireworks run, test the pipeline end-to-end locally on a tiny subset (e.g. 10–50 samples) so paid runs don't fail on bugs.
