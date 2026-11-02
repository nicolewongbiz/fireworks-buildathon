# Results so far

Last updated 2026-11-01 22:15 PT. Smoke-test scale only: nothing here is a headline number. The point of
tonight was to retire platform risk by running upload, train, serve, and query end to end once.

## Smoke test: untuned Qwen3.5-9B on 10 synthetic images

Source: `results/smoke_untuned.json`. Data: `smoke/make_smoke_data.py` (ZINC250k molecules, RDKit
Cairo renderings at 384 px, random bond width, font size, rotation, and palette). Served by vLLM on a
Modal L40S, temperature 0, max 150 output tokens, thinking disabled.

| Metric | Value |
|---|---|
| Exact match (canonical SMILES) | 0 / 10 |
| Valid SMILES | 7 / 10 |
| Mean Tanimoto (valid outputs) | 0.17 |
| Mean latency per image (warm) | 3.7 s |
| Cold start to first answer | 250 s |
| Mean tokens in / out | 175 / 81 |
| GPU time for the run (incl. cold start) | 4.7 min, $0.15 at $1.95/h |

Per image:

| Image | Valid | Exact | Tanimoto | Latency | Out tokens | Failure mode |
|---|---|---|---|---|---|---|
| test_000 | yes | no | 0.21 | 7.6 s | 65 | plausible but wrong molecule |
| test_001 | no | no | – | 5.7 s | 150 | degenerate repeat `C[C@H](C)...` until token cap |
| test_002 | no | no | – | 5.8 s | 150 | degenerate repeat `[C@H](O)...` until token cap |
| test_003 | yes | no | 0.07 | 2.6 s | 64 | wrong molecule |
| test_004 | yes | no | 0.25 | 1.8 s | 42 | wrong molecule, right size |
| test_005 | yes | no | 0.22 | 1.9 s | 47 | wrong molecule |
| test_006 | no | no | – | 5.8 s | 150 | degenerate repeat until token cap |
| test_007 | yes | no | 0.08 | 3.2 s | 80 | wrong molecule |
| test_008 | yes | no | 0.26 | 1.4 s | 33 | wrong molecule |
| test_009 | yes | no | 0.13 | 1.3 s | 32 | wrong molecule |

What this tells us:

- The untuned model is nowhere near the task. Even its valid outputs are unrelated molecules
  (Tanimoto around 0.2 is what two random drug-like molecules score). There is a lot of headroom for tuning,
  which is what the hypothesis needs.
- Three of ten outputs are degenerate stereo-tagged repeats that run into the 150-token cap. Those cost
  about 3x the latency of a normal answer. The RDKit retry loop in the spec would catch these, but a tuned
  model is expected to stop producing them, so measure the invalid rate after tuning before building it.
- The untuned model emits stereo tags (`@`, `@@`) that the flat drawings cannot justify. Stripping stereo
  before comparison is the right call; otherwise even a correct skeleton would be scored wrong.
- At 3.7 s per image warm and 250 s cold start, the 500-image synthetic test set takes about 30 min of
  L40S time serially (about $1). Running requests concurrently against vLLM should cut that several-fold.

## Smoke test: LoRA SFT on Modal

Fireworks SFT could not be used tonight: the account's dataset uploads are disabled
(`FailedPrecondition`), and Qwen3-VL-8B-Instruct, the model named in the spec, is not tunable there.
`src/modal_train.py` is the fallback: PEFT LoRA (rank 8, alpha 16, LR 2e-4, 3 epochs, grad-accum 8)
on the language-model attention and MLP projections only, so vLLM can load the adapter next to the base.

Run on 40 training images (`data/smoke_train.jsonl`), one L40S:

| Metric | Value |
|---|---|
| Trainable params | 14.5 M of 9.42 B (0.15 %) |
| Trained tokens per example | 21 (the SMILES only; image and prompt are masked) |
| Epoch mean loss | 1.035 -> 0.617 -> 0.358 |
| Training time (3 epochs, 40 images) | 99 s, about 0.8 s per image-step |
| Container time (image pull, weight load, train, save) | 2.4 min, about $0.08 at $1.95/h |
| Output | `chemvision-loras` volume, `/smoke` (`adapter_config.json`, `adapter_model.safetensors`) |

The loss drop on 40 images says the labels are wired correctly (only the assistant turn is trained on,
confirmed by decoding example 0), not that the model learned the task. Scaling linearly, 5,000 images for
3 epochs is about 3.3 GPU-hours on one L40S (about $7); batching more than one example per step should
cut that. The next step is `run_system tuned` on the 10 smoke images to confirm vLLM loads the adapter
and that outputs change; that has not been run yet.

## Blockers and open items

1. **Fireworks dataset upload disabled** on the account. Support request open. Until it clears, no
   Fireworks SFT job and no Fireworks-served tuned model. Modal is the fallback for both.
2. **Base model changed** from Qwen3-VL-8B-Instruct (not tunable on Fireworks) to Qwen3.5-9B. The spec
   and pitch should say Qwen3.5-9B.
3. **Large baseline model not chosen yet.** Needs to be a vision model on Fireworks serverless; pick on
   build day from the live model list.
4. **Real dataset not built yet.** Tonight used a 50-molecule throwaway set. `src/render.py` and
   `src/build_dataset.py` (scaffold split, leakage check) are still owed by the data/eval owner.
5. **Pre-work rules.** Confirm with organizers whether code written tonight is allowed.

## Spend

See `cost_log.md`. Total Modal spend tonight is under $1.
