# Pitch: ChemVision-Tuned (3-minute, Fireworks Deploy Buildathon, 2 Nov 2026)

For the two presenters. Written 02:00 PT with the numbers measured so far; everything else is a `{{PLACEHOLDER}}`.
Rule for the stage: say nothing that is not in a results file or on this page. If a placeholder is still empty at
16:00, cut the sentence, do not guess.

## Placeholders (fill from these files; keep `src/evaluate.py` output as the source of truth)

| Placeholder | Source |
|---|---|
| `{{BEST}}` | results/selection.json: the checkpoint chosen on validation for the largest finished train size (e.g. `n10000_r384_lang_e3_ep2`) |
| `{{TUNED_10K_EXACT}}` / `{{TUNED_10K_NOSTEREO}}` | results/n10000_r384_lang_e3*__test_synthetic_384.json summary.exact_match / exact_match_nostereo (use the selected `_epK`) |
| `{{TUNED_20K_EXACT}}` / `{{TUNED_20K_NOSTEREO}}` | results/n20000_r384_lang_e3*__test_synthetic_384.json (needs a 20k train manifest; data/dataset_report.json has 10,000 today) |
| `{{BEST_DRUGS_EXACT}}` | results/{{BEST}}__test_drugs_384.json summary.exact_match |
| `{{BEST_USPTO_EXACT}}` / `{{BEST_USPTO_NOSTEREO}}` | results/{{BEST}}__test_real_uspto_384.json summary.exact_match / exact_match_nostereo |
| `{{BEST_HANDDRAWN_EXACT}}` | results/{{BEST}}__test_handdrawn_384.json summary.exact_match |
| `{{STEREO_TAX}}` | {{BEST}} exact_match_nostereo minus exact_match, in points (derived) |
| `{{HARNESS_EXACT}}` / `{{HARNESS_COST_PER_CORRECT}}` | results/<harness system>__test_synthetic_384.json (RDKit filter + self-consistency; system name to be set when `src/predict.py` gains it) |
| `{{ROUTED_EXACT}}` / `{{ROUTED_ESC_RATE}}` / `{{ROUTED_COST_PER_CORRECT}}` | results/routed__test_synthetic_384.json summary.exact_match; share of items with predict.escalated true; comparison.json cost_per_correct |
| `{{CLOSED_MODEL}}` | results/DECISION_RULES.md dated note after the 50-image val pilot (runbook rule 2) |
| `{{CLOSED_SYN_EXACT}}` / `{{CLOSED_DRUGS_EXACT}}` / `{{CLOSED_USPTO_EXACT}}` | results/closed_opus__test_{synthetic,drugs,real_uspto}_384.json summary.exact_match (rename if the pilot picks another model) |
| `{{CLOSED_COST_PER_IMAGE}}` / `{{CLOSED_COST_PER_CORRECT}}` / `{{CLOSED_REASONING_USD}}` | comparison.md Cost / image and Cost / correct; sum of items[].predict.reasoning_usd |
| `{{KIMI_SYN_EXACT}}` / `{{KIMI_USPTO_EXACT}}` / `{{KIMI_COST_PER_CORRECT}}` | results/large_kimi__test_synthetic_384.json, results/large_kimi__test_real_uspto_384.json; comparison.md |
| `{{UNTUNED_USPTO_SAID}}` / `{{UNTUNED_USPTO_EXACT}}` | results/untuned__test_real_uspto_384.json items[image_path = demo image].match.pred_raw; summary.exact_match |
| `{{TUNED_STEADY_1K_W8}}` / `{{TUNED_STEADY_1K_W32}}` | results/{{BEST}}__test_synthetic_384.json and results/{{BEST}}_w32__test_synthetic_384.json summary.steady_usd_per_1k_images |
| `{{TUNED_COLD_START_S}}` | results/{{BEST}}__test_synthetic_384.json summary.cold_start_s |
| `{{TUNED_STEADY_COST_PER_CORRECT}}` | results/comparison.json steady_cost_per_correct for {{BEST}} |
| `{{BEST_VS_CLOSED_DIFF}}` / `{{BEST_VS_CLOSED_P}}` | results/paired.md row {{BEST}} vs closed model, stereo-aware, test_synthetic_384 |
| `{{ROUTED_VS_CLOSED_DIFF}}` / `{{ROUTED_VS_CLOSED_P}}` | results/paired.md row routed vs closed model |
| `{{TRAIN_USD_10K}}` / `{{TRAIN_MIN_10K}}` | results/cost_log.md row for the 10k LoRA SFT (billed actual, or "est" if the hour has not closed) |
| `{{TOTAL_SPEND}}` | results/cost_log.md running total plus closed-model vendor usage page |
| `{{TRAIN_SIZE_BEST}}` | training images behind {{BEST}} (the N in its run name, e.g. 10,000) |

Measured tonight and safe to say (file in brackets): untuned Qwen3.5-9B 0.0% on 500 synthetic, 67.0% valid, 19.2% loop rate
[results/untuned__test_synthetic_384.json]; tuned on 500 images 6.8% [4.6, 9.0] stereo-aware, 11.4% agnostic, 72.0% valid, 0.0% loops,
whole-run cost per correct $0.0076 [results/n500_r384_lang_e3__test_synthetic_384.json, results/comparison.md]; paired +6.8 pts
[+4.6, +9.0], McNemar p < 0.001, 34 discordant images all in the tuned model's favour [results/paired.md]; Kimi K3 8/10 on the 10-image
smoke set at $0.0013/image, $0.0016/correct [results/smoke_large_kimi.json]; MolScribe 93.4% synthetic, 89.0% USPTO, 11.5% hand-drawn
[results/molscribe__*.json]; Qwen3-VL-8B "Training: Not supported" on Fireworks and Fireworks dataset uploads blocked (`FailedPrecondition`)
[docs/fireworks-notes.md, results/RESULTS.md]; Modal L40S $1.95/h vs Fireworks H100/H200 $8/h [docs/spec-review.md]; 175 prompt tokens per
request of which 144 are image tokens at 384 px [summary.mean_tokens_in; docs/fireworks-notes.md sec. 7]. Derived from the n500 results file
(wall_s 479.6 − first_answer_s 259.1 = 220.5 s for 500 images, 8 workers, $1.95/h): about $0.24 per 1k images steady state, cold start 259 s.

## 1. Slide list (7 slides, 3:00 total; the 45 s demo is a separate segment, see section 3)

**Slide 1, 0:00-0:20: "Chemists retype molecules."** Claim: structure images in papers and patents are read by humans or by
general models that get them wrong. On it: the USPTO scan `data/images/real_uspto/384/US07314693-20080101-C00493.png` (22 heavy atoms,
two morpholine rings) next to its true SMILES `Oc1cc(CN2CCOCC2)c(O)cc1CN1CCOCC1`, and below it what the untuned 9B returned:
`{{UNTUNED_USPTO_SAID}}`. Fallback if that run is missing: synthetic test image `data/images/384/test/00003.png`, truth
`O=C1CCCN1C1CCN(S(=O)(=O)CC2CCC2)CC1`, untuned said `O=S1(=O)CCN(CC1)C1CCC(CC1)N(C)C(=O)C1CC1` (valid SMILES, wrong molecule).
Spoken: "Every patent and paper draws molecules as pictures. Getting them back into a database means a chemist retyping them."
"A 9-billion-parameter open vision model, out of the box, gets zero of 500 right, and one in five answers loops until the token cap."
"Our question: can a cheap open model, tuned on synthetic data, match a closed frontier model on this, per dollar?"

**Slide 2, 0:20-0:45: "Synthetic data, a small LoRA, a chemistry check, and a router."** Claim: the whole stack is open and cheap.
On it: a four-box flow: RDKit renders ZINC/PubChem molecules with random fonts, rotations, bond widths → LoRA (rank 8, language
layers only, 0.15% of weights) on Qwen3.5-9B → RDKit parses the answer and self-consistency votes → disagreements escalate to Kimi K3
on Fireworks serverless. Spoken: "Labels are free: RDKit draws a molecule and we know the answer, so we rendered 10,000 training
images and scaffold-split them so no test molecule shares a skeleton with training." "We tuned Qwen3.5-9B, not Qwen3-VL-8B, because
Fireworks lists the 8B as not trainable; and when Fireworks uploads were blocked at midnight we moved training to Modal with the same
JSONL." "The router only pays for the big model when the small one cannot agree with itself."

**Slide 3, 0:45-1:05: "Frozen tests, pre-registered rules."** Claim: the numbers were decided before the runs. On it: a table of the
four frozen sets (500 synthetic scaffold-split, 100 PubChem drugs, 200 USPTO patent scans, 200 DECIMER hand-drawn) and the three rules
from `results/DECISION_RULES.md`: stereo-aware exact match is the headline, stereo-agnostic always next to it; checkpoints chosen on a
200-image validation set only; two systems compared by paired McNemar on the same images. Spoken: "Exact match is stereo-aware: a wedge
bond drawn wrong is a wrong molecule." "We never picked anything on the test set, and every comparison is paired, so a 2-point gap has a
p-value, not an opinion." "At n=500 a single number is about plus or minus 2 to 4 points; the paired tests are sharper."

**Slide 4, 1:05-1:40: "Exact match against cost per correct answer."** Claim: the tuned open stack reaches `{{ROUTED_EXACT}}` at
`{{ROUTED_COST_PER_CORRECT}}` per correct answer versus the closed model's `{{CLOSED_SYN_EXACT}}` at `{{CLOSED_COST_PER_CORRECT}}`.
On it: `results/chart_exact_vs_cost.png` (x = cost per correct, log scale; y = stereo-aware exact match on test_synthetic_384) with
seven points: untuned 0.0% (cost undefined, plotted as a marker at the top of the x axis), tuned-500 6.8% at $0.0076, tuned-{{BEST}}
`{{TUNED_10K_EXACT}}` at `{{TUNED_STEADY_COST_PER_CORRECT}}`, tuned+harness `{{HARNESS_EXACT}}` at `{{HARNESS_COST_PER_CORRECT}}`,
router `{{ROUTED_EXACT}}` at `{{ROUTED_COST_PER_CORRECT}}`, `{{CLOSED_MODEL}}` `{{CLOSED_SYN_EXACT}}` at `{{CLOSED_COST_PER_CORRECT}}`,
Kimi K3 `{{KIMI_SYN_EXACT}}` at `{{KIMI_COST_PER_CORRECT}}`, and MolScribe 93.4% as a dashed "specialist ceiling" line. Spoken: "Untuned
is zero, so its cost per correct answer is infinite; 500 training images already moved it to 6.8 percent, paired p under 0.001." "The
tuned model with the chemistry check and router scores `{{ROUTED_EXACT}}`; the closed frontier model scores `{{CLOSED_SYN_EXACT}}`; the
paired difference is `{{ROUTED_VS_CLOSED_DIFF}}` points, p = `{{ROUTED_VS_CLOSED_P}}`." "Per correct answer that is
`{{ROUTED_COST_PER_CORRECT}}` against `{{CLOSED_COST_PER_CORRECT}}`, with the router escalating `{{ROUTED_ESC_RATE}}` of images."

**Slide 5, 1:40-2:00: "More synthetic data, more accuracy."** Claim: accuracy is data-limited, not epoch-limited. On it:
`results/learning_curve.png`, x = training images (0, 500, 10k, 20k), y = stereo-aware exact match with Wilson intervals, stereo-agnostic
as a lighter series. Points: 0.0%, 6.8% (11.4% agnostic), `{{TUNED_10K_EXACT}}` (`{{TUNED_10K_NOSTEREO}}`), `{{TUNED_20K_EXACT}}`
(`{{TUNED_20K_NOSTEREO}}`). Spoken: "The 500-image run's validation loss bottomed at epoch 2, so we scaled data, not epochs."
"Ten thousand images cost `{{TRAIN_USD_10K}}` and `{{TRAIN_MIN_10K}}` minutes on one L40S." "Each step is a paired test against the
previous size; the McNemar column in results/LEARNING_CURVE.md says which steps are real."

**Slide 6, 2:00-2:30: "What we did not beat."** Claim: we report the gaps. On it: a four-row table. Synthetic vs real: tuned
`{{TUNED_10K_EXACT}}` synthetic vs `{{BEST_USPTO_EXACT}}` USPTO vs `{{BEST_HANDDRAWN_EXACT}}` hand-drawn. Stereo tax: `{{STEREO_TAX}}`
points (5 points on the 500-image run: 6.8 vs 11.4; on test molecules with a stereocentre it scored 1.3% vs 15.9% without). Specialist:
MolScribe 93.4% synthetic, 89.0% USPTO, 11.5% hand-drawn, $0 on a laptop CPU. GPU reality: cold start `{{TUNED_COLD_START_S}}` s (259 s on
the 500 run), L40S $1.95/h on Modal vs $8/h for a Fireworks H100; steady-state `{{TUNED_STEADY_1K_W8}}` per 1k images at 8 workers,
`{{TUNED_STEADY_1K_W32}}` at 32. Spoken: "A purpose-built OCSR model still beats every VLM here by a wide margin on clean drawings; our
contribution is the measurement, not a new state of the art." "Stereo is where the small model loses most: on molecules with a
stereocentre it is near zero." "Cost per correct answer on a dedicated GPU is a utilisation number; we show the measured concurrency and
the cold start separately, never a blended figure."

**Slide 7, 2:30-3:00: "Cost per correct answer at volume."** Claim: at steady state the open stack costs `{{TUNED_STEADY_COST_PER_CORRECT}}`
per correct molecule versus `{{CLOSED_COST_PER_CORRECT}}` for the closed model (both from results/comparison.json). On it: a two-column
table for 1,000,000 images: tuned steady `{{TUNED_STEADY_1K_W8}}` x 1,000 plus one cold start; router: that plus `{{ROUTED_ESC_RATE}}` x
Kimi K3 at $0.0013/image (smoke-set rate); closed model `{{CLOSED_COST_PER_IMAGE}}` x 1,000,000; each divided by its exact match to give
cost per correct. Spoken: "Pharma patent pipelines and chemistry publishers already pay for this by the structure, in curator hours; the
buyer is whoever runs that pipeline." "Every number on this slide is from a results file in the repo, with its concurrency and cold start
written next to it." "Total spend to produce everything you saw: `{{TOTAL_SPEND}}`."

## 2. Numbers (see the table and the measured list at the top; nothing on the slides is outside them)

## 3. Demo script (45 s, separate segment at the judging table; if only the 3 minutes exist, drop slide 5 to 0:10 and slide 6 to 0:20 and run it after slide 4)

`app.py` (Gradio, `uv run python app.py`, http://127.0.0.1:7860) has an image upload, an "Example images" dropdown (first 8 items of
each committed test manifest), a ground-truth box, one checkbox per system (`untuned, tuned, large_kimi, large_qwen, large_glm,
closed_opus, molscribe`; `tuned`, `large_kimi`, `closed_opus` are pre-ticked), Run, a table and a gallery. There is no `routed` checkbox
yet. Two traps to clear before 16:00: `src/predict.py` `predict_tuned` defaults to `lora="smoke"` (the 40-image adapter), so point it at
{{BEST}} and make sure that adapter is in `SERVE_LORAS` on the Modal server; and keep the Modal vLLM app warm (`uv run modal deploy
src/modal_serve.py` at least 5 minutes before; cold start was 201 to 259 s). `FIREWORKS_API_KEY` must be in `.env`.

1. Pick `real_uspto / US07314693-20080101-C00493` from the dropdown (or upload `data/images/real_uspto/384/US07314693-20080101-C00493.png`;
   22 heavy atoms, no stereo, so the stereo tax does not confuse the demo). The truth box fills itself. Say: "A scan from a 2008 US
   patent, 384 pixels wide, as a chemist would get it."
2. Tick `untuned`, `tuned` ({{BEST}}) and `large_kimi`; untick `large_qwen` and `large_glm`. Leave `closed_opus` ticked only if the key
   is loaded and the spend is approved. `molscribe` shows "not configured" in this venv; show its answer from the results file instead.
3. While it runs (3 to 5 s per system at the measured p50 of 2.2 to 3.4 s): "The truth is `Oc1cc(CN2CCOCC2)c(O)cc1CN1CCOCC1`. RDKit
   canonicalises each answer; a green row is an exact stereo-aware match, amber is a valid SMILES of the wrong molecule, red is unparseable."
   Point at the untuned row: "valid-looking, wrong molecule, which is why validity is never our headline."
4. Point at the `large_kimi` row and its cost column: "this is what the router escalates to when the small model cannot agree with
   itself, about a tenth of a cent per image." (The router itself is scored in the results files, not run live, unless a `routed`
   checkbox lands in `app.py` before 16:00.)
5. Second image if time allows: `data/images/384/test/00003.png` (clean render) to show the synthetic-to-real gap in one glance.

Network-dead fallback (have these open in a second window before you start): `results/analysis/smoke_large_kimi/contact_sheet.png`,
`results/analysis/smoke_untuned/contact_sheet.png`, `results/analysis/smoke_tuned/contact_sheet.png`, the {{BEST}} contact sheet once
`src/analyze_errors.py` has run on it, and `results/chart_exact_vs_cost.png`. Say the same lines over the contact sheets; the per-image
truth and prediction are printed on them.

## 4. Judge Q&A (two sentences each; say the number only if it is filled in)

1. Why not just use the closed model? It scores `{{CLOSED_SYN_EXACT}}` at `{{CLOSED_COST_PER_IMAGE}}` per image, and its reasoning tokens
   alone cost `{{CLOSED_REASONING_USD}}` over our test sets. Our router reaches `{{ROUTED_EXACT}}` at `{{ROUTED_COST_PER_CORRECT}}` per
   correct answer and the weights are ours, so a patent pipeline can run it on-prem.
2. Why not MolScribe? On clean drawings you should: 93.4% synthetic and 89.0% USPTO at $0 on a CPU, and it is on our chart as the
   ceiling. It drops to 11.5% on hand-drawn photos, cannot be prompted or re-tuned on a new drawing style in an afternoon, and the point of
   the project is to measure what general open models buy for a fixed training budget.
3. Is 500 (or 1,000) test images enough? For a single system the 95% half-width at n=500 is about 2.2 points at 7% and 4.4 at 50%,
   which is why we never rank two systems by comparing their intervals. Every ranking is a paired McNemar test on the same images, which
   is sharper: the 500-image LoRA beat untuned on 34 images and lost on none, p < 0.001.
4. How does cost per correct change with GPU utilisation? Linearly with idle time: the GPU bills $1.95 an hour whether it answers one
   image or eight per second, so we report steady-state dollars per 1k images at the measured concurrency (`{{TUNED_STEADY_1K_W8}}` at 8
   workers, `{{TUNED_STEADY_1K_W32}}` at 32) and the cold start as a separate one-off. The serverless and closed-model costs are per token,
   so they do not move with load; the crossover is a volume question we can read off the chart.
5. What happens on hand-drawn images? The tuned model scores `{{BEST_HANDDRAWN_EXACT}}` and MolScribe 11.5%, because the set is photos of
   small, odd molecules and we trained on none of that style. It is reported as a stretch set; the fix is rendering training data in a
   hand-drawn style, which we did not do today.
6. Why stereo-aware exact match? A wedge bond drawn the wrong way is a different drug, and our labels and USPTO drawings carry stereo, so
   accepting the stereo-stripped answer would score a wrong molecule as right. We decided this before any run and always show the
   stereo-agnostic number next to it; the gap, 5 points on the 500-image run, is the honest size of the problem.
7. Did you train on Fireworks? No: the account's dataset uploads returned `FailedPrecondition` all night and the spec's Qwen3-VL-8B is
   listed as not trainable there, so LoRA training and 9B serving ran on Modal L40S with the same Fireworks VLM-SFT JSONL. Fireworks
   serverless served Kimi K3 as the open large model and the router's escalation target.
8. Why is the synthetic score higher than USPTO? Training renders are RDKit's font and line style; patent scans are 800-pixel-wide
   bitmaps downscaled to 384 with other conventions, and the gap is `{{TUNED_10K_EXACT}}` vs `{{BEST_USPTO_EXACT}}`. MolScribe also drops,
   from 93.4% to 89.0%, so part of it is the downscale; the 640 px manifests exist to measure that.
9. Why Kimi K3 as the open large model? It was the only serverless vision model on Fireworks that answered with a bare SMILES under a
   108-token cap, 8/10 on the smoke set at $0.0013 per image. GLM-5.3-Flash is thinking-only and produced no answer at that cap;
   Qwen3.8-Max wrote reasoning prose into the answer and was cut off on 8 of 10.
10. What would you do with a week? Render 200k to 1M training images with hand-drawn and scanned-style augmentation, since OCSR
    specialists train on that scale and our curve is still data-limited. Then tune the vision projector too, run the 640 px ablation
    properly, and let the router use self-consistency agreement as its confidence signal across all three real sets.

## 5. Submission blurb (120 words)

ChemVision-Tuned reads molecule drawings into SMILES with an open stack: Qwen3.5-9B plus a rank-8 LoRA trained on RDKit-rendered
synthetic molecules, an RDKit validity and self-consistency check, and a router that escalates only disagreements to Kimi K3 on Fireworks
serverless. We froze four test sets (500 scaffold-split synthetic, 100 drugs, 200 USPTO patent scans, 200 hand-drawn), pre-registered
stereo-aware exact match as the headline, and compared systems with paired McNemar tests. Untuned: 0.0%. Tuned on 500 images: 6.8%,
p < 0.001. Tuned on `{{TRAIN_SIZE_BEST}}`: `{{TUNED_10K_EXACT}}`; routed: `{{ROUTED_EXACT}}` at `{{ROUTED_COST_PER_CORRECT}}` per correct
answer versus `{{CLOSED_MODEL}}` at `{{CLOSED_SYN_EXACT}}` and `{{CLOSED_COST_PER_CORRECT}}`. MolScribe (93.4%) is shown as the specialist
ceiling. Cold start, GPU utilisation and the synthetic-to-real gap are reported, not hidden. Total spend: `{{TOTAL_SPEND}}`.
