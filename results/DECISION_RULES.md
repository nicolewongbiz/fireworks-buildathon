# Decision rules (pre-registered)

Written 2026-11-01, before the 2,000 / 5,000 / ablation results existed. Agreed after an independent review of the
experiment plan. Change these only by adding a dated note below, never by editing them after seeing results.

## Metrics
- **Headline metric: stereo-aware exact match** (`summary["exact_match"]`: RDKit canonical isomeric SMILES equal to
  the label). Always show stereo-agnostic exact match (`exact_match_nostereo`) next to it.
- Valid SMILES is never the headline: a valid string can be the wrong molecule.
- **Comparisons between two systems use a paired test** on the same images (exact McNemar on discordant pairs, or a
  paired bootstrap of the difference). A difference counts as real only if p < 0.05. We do not use a fixed
  "under 4 points is noise" rule. For reference, single-system 95% half-widths at n = 500 are about ±2.2 points at
  7%, ±3.1 at 15%, and ±4.4 at 50%.
- Difficulty breakdowns are reported only for bins with more than 100 test molecules (heavy-atom bands, stereo yes/no),
  with counts. Smaller bins (bridged 19, multi-fragment 27, charged 61) are listed as counts only, without claims.

## Selection: the test set is never used to choose anything
- Every training run saves one checkpoint per epoch. **The reported checkpoint is the one with the highest stereo-aware
  exact match on the 200-image validation set** (ties: lower validation loss, then earlier epoch).
- Ablation winners and the final "best config" are chosen the same way, on validation.
- The 500-image test set and the drugs set are scored once per chosen checkpoint, for reporting only.

## Scaling
- **10,000 examples run only if** the chosen 5,000 checkpoint beats the chosen 2,000 checkpoint on validation, and
  the test-set difference between them is paired-significant (McNemar p < 0.05), **and** the run can start by
  12:00 on build day. Otherwise the curve stops at 5,000.

## Repetition loops
- Phase 5 (decoding comparison) is dropped: the 500-example LoRA already has 0% outputs hitting the token cap
  (vs 19% untuned). A repetition penalty is not used at all, because valid SMILES legitimately repeat `C`, `c`,
  parentheses and ring digits. Keep the loop-rate column; if loops reappear, add a cap plus n-gram stop only.

## Cost and latency
- Self-hosted cost is reported as **steady-state $ per 1,000 images at a stated concurrency**
  (`steady_usd_per_1k_images`, `workers`), with **cold start as a separate one-off line** (`cold_start_s`,
  `cold_start_usd`). One extra high-concurrency run (32 workers) gives the best-case throughput cost.
- Latency is quoted at the measured concurrency and never taken from the high-concurrency run.
- All checkpoint evaluations in a comparison run back to back in one warm server session with the same settings.

## Baselines (Person B)
- **Closed-model pilot (50 validation images, per `docs/closed-model-runbook.md`; choosing a model is selection, so
  it stays off the test set):** at about ±12 points the
  mid-tier models will usually tie, so the headline closed model is chosen by cost unless one leads another by
  more than 20 points.
- A larger output budget goes only to closed models whose reasoning cannot be turned off (OpenAI and Anthropic count
  reasoning toward the output cap); their reasoning tokens and cost are reported separately. Everything else keeps
  the 108-token answer cap.
- Untuned strawman check (reporting only, not selection): the untuned 9B with thinking on and a 1,024-token budget on
  the fixed `data/test_synthetic_384_first50.json`, next to the plain untuned model on the same 50 images; footnote
  (`untuned_thinking`).
