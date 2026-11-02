# ChemVision-Tuned

Fine-tune a small open vision-language model to read molecule drawings into SMILES, and measure what the
tuning buys in accuracy against cost per correct answer. Built for the Fireworks Deploy Buildathon
(San Francisco, 2 Nov 2026).

**Task:** molecule image in, SMILES string out. Scored by exact match on RDKit-canonical SMILES
(stereo stripped, because a flat drawing without wedge bonds cannot convey it). Syntactic validity and
Morgan-fingerprint Tanimoto are secondary metrics for error analysis only.

**Systems compared on the same test set:**

| System | What it is |
|---|---|
| `untuned` | Qwen3.5-9B as released, thinking disabled, temperature 0 |
| `tuned` | Qwen3.5-9B + LoRA trained on synthetic RDKit renderings |
| `large` | A large hosted VLM baseline (model to be fixed on build day) |
| `routed` | tuned model first, escalate to the large model when the answer fails RDKit |

Current status and numbers: [results/RESULTS.md](results/RESULTS.md). Spend: [results/cost_log.md](results/cost_log.md).
Team rules, ownership, and compute policy: [CLAUDE.md](CLAUDE.md).

## Why Qwen3.5-9B and why Modal

The spec named Qwen3-VL-8B-Instruct. On Fireworks that model is listed "Training: Not supported" and the
SFT API rejects it, so the base model is **Qwen3.5-9B** (Fireworks `accounts/fireworks/models/qwen3p5-9b`,
Hugging Face `Qwen/Qwen3.5-9B`), which is vision-capable and tunable. Qwen3.5 thinks by default; every
request sends `chat_template_kwargs: {"enable_thinking": false}` so the model answers with a bare SMILES.

As of 1 Nov 2026 the Fireworks account cannot upload datasets (`FailedPrecondition`), so no managed SFT
job can start. A support request is open. Until it clears, training and serving run on
[Modal](https://modal.com) (L40S, $1.95/h, billed per second, scales to zero) using the same data format
and the same `predict()` contract, so the swap back to Fireworks is a model-ID change.

## Layout

```
smoke/make_smoke_data.py   50-molecule throwaway set (40 train / 10 test) for the platform smoke test
src/make_dataset.py        real dataset: ZINC + PubChem drugs, scaffold split, leakage checks, 384/640 px renders
src/make_real_test.py      real-world test sets: 200 USPTO patent scans + 200 DECIMER hand-drawn, from data/raw
src/validate.py            RDKit parse, canonicalize, extract SMILES from free text, score one prediction
src/predict.py             one predict(image_path) per system; OpenAI-compatible chat with image data URI
src/run_system.py          run a predictor over a test set -> results/<system>.json (metrics + per-item)
src/modal_serve.py         vLLM OpenAI-compatible server for the base model + any LoRA in a Modal volume
src/modal_train.py         LoRA SFT fallback on Modal (PEFT, language-model layers only, vLLM-loadable)
data/                      manifests {val,test_synthetic,test_drugs,test_real_uspto,test_handdrawn}_{384,640}.json are
                           committed with image sha256 lists; train manifests, images and raw sources are regenerated
results/                   results/<system>.json and cost_log.md are committed; results/raw/ logs are not
```

Still to come from the baselines/comparison owner: closed-model and MolScribe predictors, `src/evaluate.py`
(cross-system metrics and the exact-match vs cost chart), and `app.py` (demo UI).

## Shared contracts

- `data/train.jsonl`: one JSON object per line, `{"messages": [user, assistant]}`. The user turn has a text
  part with the fixed prompt and an `image_url` part whose URL is a `data:image/png;base64,...` URI. The
  assistant turn is the canonical SMILES only. This is the Fireworks VLM SFT format; `modal_train.py`
  reads the same file.
- `data/test_*.json`: `[{"image_path": "...", "smiles": "<canonical isomeric SMILES>", ...}]`; items may carry
  difficulty `tags`.
- `predict(image_path) -> {"smiles", "latency_s", "tokens_in", "tokens_out", "attempts", "escalated"}`.
- `results/<system>.json`: `{"summary": {...}, "items": [...]}`, written by `src/run_system.py`. Summary
  includes stereo-aware `exact_match`, `exact_match_nostereo`, `valid_rate`, `mean_tanimoto`, `loop_rate`,
  `median_latency_s`, `p95_latency_s`, `max_tokens` (108), and GPU cost fields.

## Setup

```bash
uv sync                                   # Python 3.12 venv with RDKit, openai, modal, etc.
cp .env.example .env                      # fill in keys; .env is gitignored
uv run python smoke/make_smoke_data.py    # needs data/raw/zinc250k/250k_rndm_zinc_drugs_clean_3.csv
```

## Run the smoke pipeline (Modal)

```bash
uv run modal run src/modal_serve.py::download_model      # one time, CPU only, caches weights in a volume
uv run modal deploy src/modal_serve.py                   # OpenAI-compatible endpoint, scales to zero
uv run python -m src.run_system untuned data/smoke_test.json \
    --out results/smoke_untuned.json --gpu-usd-per-hour 1.95
uv run modal run -m src.modal_train --data data/smoke_train.jsonl --name smoke --epochs 3
uv run python -m src.run_system tuned data/smoke_test.json \
    --out results/smoke_tuned.json --gpu-usd-per-hour 1.95   # serves LoRA "smoke" after a cold start
uv run modal app stop chemvision-vllm                    # hard stop when done
```

`src/run_system.py` records GPU wall-clock for the run times the hourly rate, and records time to the
first answer separately so cold start can be included or excluded explicitly.

## Fireworks notes

- `firectl` refuses mutating commands (dataset, job, deployment create/delete) when run by an AI agent on
  this account. Agents print the exact command for a human to run; read-only `get`/`list` is fine.
- Any paid run (Fireworks deployment, paid tuning job, Modal GPU job) needs explicit approval with a
  cost estimate first. Costs are logged in `results/cost_log.md` from billing reports, not estimates.
