# Closed-model baseline: runbook for build day

Written for: the teammate running the closed frontier baseline on 2 Nov. Everything below is prepared; the
only missing piece is an API key. Budget: a few dollars.

## Why this matters

The brief is "beat a closed frontier model with open models". Our open stack is the tuned Qwen3.5-9B,
its harness (RDKit filter, self-consistency), and the router that escalates to Kimi K3. The closed model
is the opponent on the headline chart, so it has to be scored on the same frozen test sets, once, with
the settings written down before the run.

## Decision rules (write-once, before any number is seen)

1. **Headline metric:** stereo-aware exact match (`summary["exact_match"]`), always shown next to the
   stereo-agnostic number.
2. **Which closed model is the headline opponent:** run a 50-image pilot on `data/val_384.json` for
   each candidate. 50 images gives about ±12 points, so candidates will usually tie. Pick the headline
   by **cost per image** unless one candidate leads another by **more than 20 points**; then take the
   leader. Write the chosen model and the pilot numbers into `results/DECISION_RULES.md` before the test
   run.
3. **Wording:** call it the "closed frontier baseline" if the chosen model is the vendor's top tier
   (Claude Opus 5.5, the top GPT, the top Gemini). If cost picks a mid-tier model, say "closed mid-tier
   baseline" and run the top-tier model on the 100-image drugs set only, as a reference point.
4. **One test run per selected model.** Score `data/test_synthetic_384.json` (500), `data/test_drugs_384.json`
   (100), and `data/test_real_uspto_384.json` (200). Repeat the synthetic run once to measure run-to-run
   noise (these models are not deterministic).
5. **Comparisons are paired.** Every system scores the same images, so differences are reported with a
   paired bootstrap interval and McNemar p-value (`src/evaluate.py` does this), not by eyeballing two CIs.

## Reasoning budgets

- Open models and Kimi K3 run with thinking off and a 108-token answer cap.
- Models whose reasoning cannot be turned off get a larger budget so the answer is not truncated.
  Anthropic `max_tokens` and OpenAI `max_completion_tokens` both include reasoning tokens. Gemini's
  `thinkingBudget` is reported to be separate from `maxOutputTokens`; verify in the Gemini docs before
  relying on it.
- Report reasoning tokens and their cost on a separate line from answer tokens. The Claude predictor
  already records `answer_tokens`, `reasoning_tokens`, and `reasoning_usd` per item.

## Claude Opus 5.5 (ready)

1. Put `ANTHROPIC_API_KEY=...` in `.env` (never commit it).
2. Pilot, 50 images from the validation set:

   ```bash
   .venv/bin/python - <<'PY'
   import json; items = json.load(open("data/val_384.json"))[:50]
   json.dump(items, open("data/val_384_pilot50.json", "w"))
   PY
   .venv/bin/python -m src.run_system closed_opus data/val_384_pilot50.json \
       --out results/closed_opus__val_384_pilot50.json --gpu-usd-per-hour 0 --workers 4
   ```

3. Test runs (after the decision is written down):

   ```bash
   for s in test_synthetic_384 test_drugs_384 test_real_uspto_384; do
     .venv/bin/python -m src.run_system closed_opus data/$s.json \
         --out results/closed_opus__$s.json --gpu-usd-per-hour 0 --workers 4
   done
   .venv/bin/python -m src.evaluate && .venv/bin/python src/analyze_errors.py results/closed_opus__test_synthetic_384.json
   ```

   Settings baked into `src/predict_closed.py`: model `claude-opus-5-5`, effort `low` (closest analogue to
   thinking off), `max_tokens` 2048, a system line asking for SMILES only, server-side refusal fallback on.
   Override with `CLOSED_MODEL=claude-sonnet-5-5` or `CLOSED_EFFORT=medium` in the environment for ablations.

   Expected spend: about 1,200 input tokens per image at $4/M plus roughly 100 to 300 output tokens at
   $20/M, so around $0.01 per image, $5 to $10 for everything above including the repeat.

## Adding a GPT or Gemini predictor

Create `src/predict_openai.py` or `src/predict_gemini.py` exposing `SYSTEMS = {"closed_gpt": predict}` with
the same `predict(image_path) -> dict` contract as `src/predict_closed.py` (include `usd`, `answer_tokens`,
`reasoning_tokens`, `reasoning_usd`), then add the module name to the loop in `src/predict_baselines.py`.
The runner picks it up automatically. Use the vendor's own SDK, set the key in `.env`, and pin the model ID
in the module docstring so the results file is reproducible.

## Checklist before you say the number out loud

- [ ] Decision rules written in `results/DECISION_RULES.md` before the test run
- [ ] Same frozen manifests as every other system (`data/test_*_384.json`, committed)
- [ ] `results/<system>__<testset>.json` naming, so `src/evaluate.py` pairs it correctly
- [ ] Spend logged in `results/cost_log.md` from the vendor's usage page, not estimates
- [ ] Reasoning tokens and cost on their own line in the write-up
