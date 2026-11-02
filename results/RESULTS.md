# Results so far

Last updated 2026-11-02 00:45 PT. Smoke-test scale only: nothing here is a headline number. The point of
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
cut that. The adapter was then served and evaluated; see the next section.

## Smoke test: tuned adapter served by vLLM on the same 10 images

Source: `results/smoke_tuned.json`. vLLM loaded the adapter from the volume at startup
(`Loaded new LoRA adapter: name 'smoke'` in the server log) and served it as model `smoke` next to `base`.
The container cold-started in about 217 s (Modal logs; image already cached); the runner's 35 s wall
time covers only the request phase, so the two GPU-time rows below are not like for like.

| Metric | Untuned | Tuned (40 images, 3 epochs) |
|---|---|---|
| Exact match | 0 / 10 | 0 / 10 |
| Valid SMILES | 7 / 10 | 9 / 10 |
| Mean Tanimoto over valid outputs | 0.17 (max 0.26) | 0.19 (max 0.34) |
| Outputs with stereo tags | 7 / 10 | 0 / 10 |
| Outputs hitting the 150-token cap | 3 / 10 | 2 / 10 |
| Mean output tokens | 81 | 55 |
| Mean latency per image (warm) | 3.7 s | 3.5 s |
| GPU time billed for the run | 4.7 min incl. 250 s cold start, $0.15 | 4.2 min incl. 217 s cold start, $0.14 |

What this tells us:

- **The full loop works end to end on Modal:** JSONL in the Fireworks VLM SFT format, LoRA training,
  adapter in a volume, vLLM serving base and adapter from one GPU, OpenAI-compatible client, RDKit scoring.
  That was tonight's goal. The swap to Fireworks, if uploads are re-enabled, is a model-ID change.
- **40 training images are enough to see the adapter take effect, not to learn the task.** The tuned model
  stopped emitting stereo tags (its labels have none), answers got shorter and more often valid, and
  Tanimoto nudged up. Exact match stayed at zero, as expected at this data size.
- **Two of ten outputs still loop until the token cap**, now as valid periodic SMILES rather than invalid
  stereo spam. A repetition penalty or a lower cap would bound the cost; more training data should remove
  the behaviour. Track the cap-hit rate as a side metric.
- **Error analysis** (`src/analyze_errors.py`, outputs in `results/analysis/`): the tuned model's
  dominant mistake class is wrong heavy-atom count, usually too many (extra carbons or an extra ring),
  on 6/10 images; its two cap-hits are valid but runaway chains. Tanimoto improved on 7/10 images and
  regressed on 3 after tuning. A repetition penalty or tighter cap converts the runaways into ordinary
  wrong answers; only more training data fixes the counting.
- **Cost accounting must separate cold start from steady state.** Both runs paid 3.5 to 4 min of cold
  start for 35 s of inference. Keep the server warm across the whole eval window tomorrow and report
  steady-state cost per image separately from the fixed start-up cost.

## Smoke test: large serverless baselines on the same 10 images

Source: `results/smoke_large_kimi.json`, `results/smoke_large_glm.json`, run with `src/predict_fireworks.py`
against Fireworks serverless (temperature 0, max_tokens 108, same prompt and image format as the 9B runs).

| Metric | Kimi K3 | Qwen3.8-Max | GLM-5.3-Flash |
|---|---|---|---|
| Exact match (stereo-aware = agnostic here) | 8 / 10 | 2 / 10 | 0 / 10 |
| Valid SMILES | 10 / 10 | 8 / 10 (inflated, see below) | 0 / 10 |
| Mean Tanimoto | 0.97 | 0.39 | – |
| Hit the 108-token cap | 0 / 10 | 8 / 10 | 10 / 10 |
| Median / p95 latency | 1.2 s / 3.5 s | 1.3 s / 1.7 s | 1.0 s / 1.4 s |
| Prompt tokens per 384 px image | 241 | 174 | 226 |
| Spend for 10 images | $0.0128 | $0.0091 | $0.0009 |
| Thinking-off flag that worked | `reasoning_effort: "none"` | `reasoning_effort: "none"` | none accepted |

What this tells us:

- **Kimi K3 is the large baseline to beat.** 8/10 on clean renders with bare-SMILES output and no loops,
  at about $0.0013 per image, so roughly $0.0016 per correct answer on this sample. Both misses were
  plausible regiochemistry errors, not garbage. This is the bar the tuned 9B has to approach.
- **GLM-5.3-Flash cannot be used at this token cap.** It is a thinking-only model: every thinking-off
  flag is rejected, and with an image attached Fireworks reports the rejection as a misleading
  `400 Invalid image or video input`. With thinking on, all 108 tokens go to reasoning and no answer is
  produced. It would need a much larger cap and pays for reasoning tokens, so it is dropped.
- **Qwen3.8-Max is not usable under the shared prompt either.** With thinking off it still writes
  chain-of-thought prose into the answer ("Analyzing the image... wait, let me check...") and 8/10 answers
  are cut off by the cap. Its 8/10 "valid" is `extract_smiles` fishing a draft SMILES out of prose. A
  system prompt forcing bare output or a larger cap would be needed; not worth it when Kimi K3 behaves.
- **Serverless spend is negligible next to GPU time.** 500 images on Kimi K3 is about $0.65; on the whole
  synthetic plus real-world sets (about 1,300 images at two sizes) well under $5.

## Reference: MolScribe, a purpose-built OCSR model, on the same 10 images

Source: `results/smoke_molscribe.json`, `src/predict_molscribe.py` (MIT-licensed MolScribe, Swin-base
checkpoint, runs on the laptop CPU in its own `.venv-molscribe`; install recipe in the module docstring).

| Metric | MolScribe (CPU, 4 threads) |
|---|---|
| Exact match | 10 / 10 |
| Valid | 10 / 10 |
| Latency per image | 0.9 s median, 1.1 s p95 (laptop CPU, not comparable to GPU or API latency) |
| Cost | $0 (local) |

Clean RDKit renders are MolScribe's home turf, so 100% is the expected ceiling and the number the tuned
VLM should be measured against on synthetic data. The interesting comparison is on the patent-scan and
hand-drawn sets, where MolScribe's published accuracy is around 90% on patents and much lower on hand
drawings. Run it with `.venv-molscribe/bin/python -m src.predict_molscribe <manifest> --out results/<name>.json`;
the shared runner lists `molscribe` but can only call it from that interpreter.

## First full-size number: untuned Qwen3.5-9B on the 500-image synthetic test set

Source: `results/untuned__test_synthetic_384.json` (run by the other session on Modal, L40S, max_tokens 108).

| Metric | Untuned 9B, 500 images at 384 px |
|---|---|
| Exact match, stereo-aware and agnostic | 0.0% [0.0, 0.0] |
| Valid SMILES | 67.0% [63.0, 71.4] |
| Mean Tanimoto | 0.11 |
| Loop rate (hit the 108 cap) | 19.2% |
| Latency p50 / p95 | 2.2 s / 4.0 s |
| GPU spend for the run | $0.19 |

The "before" number is settled: zero. Everything the tuned model gets right is gain. For the pitch, the
untuned model's cost per correct answer is undefined, which is itself the point.

## Comparison tooling

`src/evaluate.py` recomputes every metric from per-item records, adds bootstrap 95% intervals, computes
cost per correct answer, and writes `results/comparison.md`, `results/comparison.json`, and the headline
chart `results/chart_exact_vs_cost.png` (plus a per-test-set bar chart when more than one set is present).
`src/analyze_errors.py` produces mistake classes, per-tag accuracy, and contact sheets. Run both after
every new results file.

## First tuned-vs-untuned number on the 500-image synthetic test set

Sources: `results/untuned__test_synthetic_384.json`, `results/n500_r384_lang_e3__test_synthetic_384.json`
(LoRA on 500 training images, 3 epochs, language-model layers only, rank 8; run by the other session on
Modal), and `results/molscribe__test_synthetic_384.json` (specialist reference, laptop CPU). Full table with
confidence intervals in `results/comparison.md`.

| System | Exact (stereo-aware) | Exact (no stereo) | Valid | Mean Tanimoto | Loop rate | Latency p50 / p95 | Run spend |
|---|---|---|---|---|---|---|---|
| Untuned Qwen3.5-9B | 0.0% | 0.0% | 67.0% | 0.11 | 19.2% | 2.2 s / 4.0 s | $0.19 |
| Tuned on 500 (n500_r384_lang_e3) | 6.8% | 11.4% | 72.0% | 0.42 | 0.0% | 3.4 s / 4.9 s | $0.26 |
| MolScribe (specialist, CPU) | 93.4% | 95.4% | 99.2% | 0.98 | 0% | 0.9 s / 1.1 s | $0 |

What this tells us:

- **500 training images already move the model off zero.** Exact match went from 0 to 6.8%
  stereo-aware (11.4% ignoring stereo), looping dropped from 19.2% to 0.0%, and mean
  Tanimoto rose from 0.11 to 0.42. The validation loss curve on this run bottomed at epoch 2 and rose
  slightly at epoch 3, so 500 examples are data-limited, not epoch-limited: the next lever is 10k images.
- **Stereo costs about 5 points at this stage.** The gap between the two exact-match columns is the share
  of molecules the model gets right except for wedge or double-bond geometry.
- **The specialist ceiling on these renders is 93.4%.** MolScribe at 95.4% stereo-agnostic shows the renders
  are readable; the remaining distance is model capacity and data, not label noise.

## MolScribe on the real-world patent set

Source: `results/molscribe__test_real_uspto_384.json` (200 USPTO patent scans, resized to 384 px).
MolScribe scores 89.0% stereo-aware, 92.0% stereo-agnostic, 96.5% valid, mean Tanimoto 0.95.
Its published USPTO accuracy is higher at native resolution, so some of the drop is the 384 px downscale
of scans that are often 800 px wide or more; the 640 px manifests exist for that comparison. The
hand-drawn result is pending.

## Changes after the smoke runs that affect comparability

Commit `57f43b5` and the dataset build changed the scoring and the generation settings. Numbers above were
produced before these changes; do not compare them directly with later runs.

- **Token cap is now 108, down from 150.** That is 1.5x the longest label in the real dataset (72 Qwen3.5
  tokens, from `data/dataset_report.json`). Both smoke runs used 150.
- **Exact match is now stereo-aware.** Labels in the real dataset are canonical isomeric SMILES, so a
  prediction must get wedge-bond and double-bond geometry right to count. `exact_match_nostereo` is
  reported alongside as the stereo-agnostic number. For the smoke files the two are identical because the
  smoke labels had no stereo.
- **New summary keys** in `results/<system>.json`: `exact_match_nostereo`, `mean_tanimoto`, `loop_rate`
  (share of outputs that hit the cap), `median_latency_s`, `p95_latency_s`, `max_tokens`. Each item gains
  `match.exact_stereo`, `match.hit_cap`, and difficulty `tags`.
- **Real dataset is built** (`src/make_dataset.py`, seed 20261102): 10,000 train / 200 val / 500 synthetic
  test / 100 PubChem drugs, rendered at 384 and 640 px, Bemis-Murcko scaffold split. Leakage checks
  passed: zero shared canonical SMILES (with and without stereo) and zero shared scaffolds across splits.
  Manifests: `data/{train,val,test_synthetic,test_drugs}_{384,640}.json`.
- **Real-world test sets built** (`src/make_real_test.py`, seed 20261102): 200 USPTO patent scans
  (`data/test_real_uspto_{384,640}.json`, 61 with stereo, 10 to 51 heavy atoms) and 200 DECIMER hand-drawn
  photos (`data/test_handdrawn_{384,640}.json`, 57 with stereo, small odd molecules). Both have zero
  overlap with the training set on stereo-aware and stereo-stripped SMILES. These replace the
  hand-labelled set the spec planned.
- **Roles revised:** Person A (nicolewongbiz) owns data and training; Person B owns baselines, closed-model
  and MolScribe predictors, `src/evaluate.py`, charts, the app, and the final comparison. See `CLAUDE.md`.

## Blockers and open items

1. **Fireworks dataset upload disabled** on the account. Exact error, from the dataset upload-endpoint
   call: gRPC `FailedPrecondition: dataset uploads are temporarily disabled, please contact Fireworks AI
   to get access`. It is not account gating: a payment method is on file ($5 paid, upload still failed
   afterwards at 22:04 PT), training quota is 32 H200/B200/B300, account state is READY and unsuspended,
   and the file was 750 KB. The one flag that stands out is `accountVerification: NOT_VERIFIED` on an
   account created at 20:57 PT tonight. Support request open; also ask the Fireworks staff on site first
   thing. A fresh dataset ID is a cheap retry but the error fires before the ID or file matter. Agents
   cannot run mutating `firectl` commands on this account, so the user runs the retry. Until it clears,
   Modal is the fallback for training and serving.
2. **Base model changed** from Qwen3-VL-8B-Instruct (not tunable on Fireworks) to Qwen3.5-9B. The spec
   and pitch should say Qwen3.5-9B.
3. **Large baseline: Kimi K3 (8/10 on the smoke set).** GLM-5.3-Flash (thinking-only) and Qwen3.8-Max
   (writes reasoning into the answer) are out under the shared prompt and cap. Qwen3-VL-235B and Llama 4 Maverick are on-demand only. Note
   Qwen3.5-9B itself is on-demand only on Fireworks, so the untuned 9B baseline needs a dedicated GPU
   there too (or Modal).
4. **Real dataset built** (see above). Still owed: `src/evaluate.py` and the exact-match vs cost chart,
   closed-model and MolScribe baseline predictors, and `app.py` (Person B).
5. **Pre-work rules.** Confirm with organizers whether code written tonight is allowed.

## Spend

See `cost_log.md`. Total Modal spend tonight is under $1.
