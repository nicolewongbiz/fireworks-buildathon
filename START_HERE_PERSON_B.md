# Start here: Person B (baselines and comparison)

Read this after `git pull`. Your Claude agents get the same rules from `CLAUDE.md`.

## Your job
You build every baseline and the comparison. Person A (Nicole) builds the data and the tuned model.

| You own (Person B) | Person A owns, so message before touching it |
|---|---|
| Closed-model API keys, spend caps, and API spend | Modal GPUs, any Fireworks deployment |
| `src/predict_baselines.py` (create it): closed-model and MolScribe predictors | `src/predict.py`, `src/run_system.py`, `src/modal_*.py`, `src/make_dataset.py` |
| `src/evaluate.py`, charts, `app.py` | `data/` (train, val, test, drugs sets) |
| `data/test_real.json` (30 to 50 hand-verified real images) | learning curve, ablations, repetition-loop fix |
| `results/FINAL_COMPARISON.md`, results chart, honest-limits slide | training story and cost-of-serving slide |

## 1. Setup (about 10 minutes)
```bash
git pull
uv sync                          # Python 3.12 env from uv.lock
cp .env.example .env             # then add YOUR keys: GEMINI_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY
```
- Never commit `.env`, paste a key into chat, or put one in a screenshot.
- You don't need Nicole's Fireworks or Modal keys.
- Set a spend cap with each closed-model provider before your first call.

## 2. Get the frozen test images and check them
The test, validation, and drugs **labels** are committed. The **images** are not (too big). Regenerate them, then check they match Nicole's byte for byte:
```bash
uv run python -m src.make_dataset               # ~3 min on 6 cores; downloads ZINC250k (22 MB) automatically
shasum -a 256 -c data/frozen_images.sha256 --quiet && echo "frozen images OK"
```
If `shasum` prints any `FAILED` line, stop and tell Nicole. Don't score anything until it passes. (The build also renders 20,000 training images you won't need. Ignore them.)

## 3. Frozen files: never edit these
If one must change, both of you agree first, and every system that used the old version gets rerun.

| File | Rows | What |
|---|---|---|
| `data/test_synthetic_384.json` / `_640.json` | 500 | **Headline test set.** ZINC molecules, Bemis-Murcko scaffold split (no scaffold shared with train) |
| `data/test_drugs_384.json` / `_640.json` | 100 | Recognizable drugs from PubChem (incl. 6 with phosphorus); removed from train |
| `data/val_384.json` / `_640.json` | 200 | Validation loss during training only. Never score systems on it |
| `data/dataset_report.json` | | Counts, feature mix, leakage checks, label token lengths |
| `data/frozen_images.sha256` | 1,600 | Checksums of every val/test/drugs image |

**Which resolution:** score every system on the **384 px** files, the same images the tuned model trains on. If Nicole's resolution ablation shows 640 px wins, she'll announce a switch and everything gets rerun on 640.

Each row looks like this:
```json
{"image_path": "data/images/384/test/00000.png",
 "smiles": "<RDKit canonical isomeric SMILES, stereo kept>",
 "smiles_nostereo": "<same, stereo removed>",
 "tags": {"heavy_atoms": 23, "rings": 3, "stereocenters": 1, "ez_bonds": 0, "charged_atoms": 0,
          "elements": ["C", "N", "O"], "fragments": 1, "bridged": false, "fused": true}}
```
Drugs rows also have `name` and `pubchem_cid`. Use `tags` for the error breakdown by difficulty.

## 4. Add your systems
Create `src/predict_baselines.py` with a `SYSTEMS` dict. The runner picks it up automatically. Every function follows the shared contract:
```python
from src.predict import MAX_TOKENS, PROMPT   # use the same prompt and token cap as the open model

def predict_gemini(image_path: str) -> dict:
    ...
    return {"smiles": raw_text_or_None,      # raw output; the scorer canonicalizes
            "latency_s": float, "tokens_in": int,
            "tokens_out": int,               # include reasoning/thinking tokens; they bill as output
            "attempts": 1, "escalated": False}

SYSTEMS = {"gemini_argon": predict_gemini}   # plus oracle / garbage / partial fakes to test the scorer
```
Rules for a fair comparison:
- Use the same prompt and the same `MAX_TOKENS` (108).
- Temperature 0.
- Reasoning at the lowest setting, or off.
- Look up the current model IDs and prices from each vendor's docs, and write them with the date to `results/closed_models.md`.

## 5. Run and score
```bash
# fakes first (free): oracle should give exact_match 1.0
uv run python -m src.run_system oracle data/test_synthetic_384.json --out results/oracle.json --gpu-usd-per-hour 0

# a closed model: pass its per-1M-token prices so the dollars are computed for you
uv run python -m src.run_system gemini_argon data/test_synthetic_384.json --out results/gemini_argon_pilot50.json \
  --gpu-usd-per-hour 0 --usd-per-mtok-in <price> --usd-per-mtok-out <price> --workers 8
```
For the 50-image pilot, slice a fixed subset first: the first 50 rows of `test_synthetic_384.json`, saved to `data/test_synthetic_384_first50.json`. Every model uses that same file.

**Results file** `results/<system>.json` = `{"summary": {...}, "items": [...]}`.
- **Summary keys:**
  - `exact_match`: stereo-aware; this is the headline
  - `exact_match_nostereo`: stereo-agnostic
  - `valid_rate`, `mean_tanimoto`
  - `loop_rate`: outputs that hit `max_tokens`
  - `mean/median/p95_latency_s`, `mean_tokens_in/out`, `usd`, `gpu_hours`, `wall_s`
- **Each item:** `predict` output, `match` (incl. `exact`, `exact_stereo`, `valid`, `tanimoto`, `hit_cap`) and `tags`.
- **Scorer:** `src/validate.py`, which does the RDKit canonicalization.
- **Noise:** with 500 test images, differences under about **4 points** are noise. Say so.
- **Valid ≠ correct:** a valid SMILES can still be the wrong molecule, so lead with exact match.

## 6. Money and logging
- You own closed-model API spend. Nicole owns Modal GPUs, so please don't start Modal apps.
- After every paid run, append a row to `results/cost_log.md`: platform, model or GPU, images, dollars. Use actual numbers, not estimates from memory. Pull first, because both of you edit it.
- Post "up/down" in chat when you start or stop anything that bills by the hour.

## 7. Git
- `git pull --rebase` before you start and before every push.
- Commit small and often to `main`, and never force-push.
- Message the owner before editing their files.

## 8. Where things stand (2026-11-01 night)
- **Pipeline works end to end on Modal:** vLLM serving Qwen3.5-9B with LoRA adapters, plus LoRA training. Smoke results are in `results/RESULTS.md`, costs in `results/cost_log.md`.
- **Fireworks dataset uploads are blocked** on Nicole's account (support contacted), so training and serving run on Modal for now. Ask an organizer early whether Modal-served models count.
- **Large serverless vision models on Fireworks,** if you want a Fireworks-hosted reference: `glm-5p3-flash`, `qwen3p8-max`, `kimi-k3`. See `docs/fireworks-notes.md`.
- **Real-world image sources** already downloaded under `data/raw/` (USPTO, UOB, DECIMER hand-drawn): see `docs/data-sources.md` for licenses. The ACS images are copyrighted, so internal evaluation only.
