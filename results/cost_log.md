# Cost log (Person A)

Actual costs from `modal billing report --show-resources` and Fireworks billing, not estimates, unless marked "est".

| When (PT) | Run | Platform / GPU | GPU time | Cost (USD) | Notes |
|---|---|---|---|---|---|
| 2026-11-01 21:55 | Qwen3-VL-8B weight download (abandoned model) | Modal CPU | — | 0.016 | 71 s download |
| 2026-11-01 22:00 | Qwen3.5-9B weight download | Modal CPU | — | 0.010 | |
| 2026-11-01 22:01–22:06 | `smoke_untuned` baseline, 10 images | Modal 1× L40S ($1.95/hr) | ~4.4 min billed so far | 0.144 GPU + 0.006 CPU/mem (partial hour) | cold start 250 s (image build + load + profiling); plus up to 5 min idle before scale-down, est +0.16 |
| 2026-11-01 22:09–22:12 | `smoke` LoRA SFT, 40 images, 3 epochs | Modal 1× L40S ($1.95/hr) | 2.4 min | est 0.08 GPU | train 99 s; adapter saved to `chemvision-loras:/smoke`; container exited, no idle |
| 2026-11-01 22:12–22:17 | `smoke_tuned` eval, 10 images, LoRA `smoke` | Modal 1× L40S ($1.95/hr) | ~4.2 min | est 0.14 GPU | cold start 217 s (image cached); app stopped right after, no idle tail |
| 2026-11-01 23:20–23:50 | Serverless smoke: Kimi K3, Qwen3.8-Max, GLM-5.3-Flash, 10 images each + ~9 probe calls | Fireworks serverless | — | 0.023 (token-priced, computed from usage) | Kimi 8/10 exact; Qwen writes reasoning into answer; GLM thinking-only |
| 2026-11-01 | Fireworks dataset upload / SFT | Fireworks | — | 0.00 | blocked: dataset uploads disabled on account |

Running total: $0.58 billed through 23:00 PT (table below); add the rows after 23:00 PT as their hours close. `chemvision-vllm` stopped at 22:18 PT; redeploy with `uv run modal deploy src/modal_serve.py`.

Billing actuals for the 21:00 and 22:00 PT hours (closed), from
`modal billing report --start 2026-11-01 --resolution h --show-resources --tz local` at 23:04 PT:

| App | L40S | CPU + memory + egress | Total |
|---|---|---|---|
| chemvision-vllm (untuned eval, idle tail, tuned eval; one deployed app) | 0.464 | 0.015 | 0.478 |
| chemvision-train (`smoke` LoRA SFT) | 0.074 | 0.004 | 0.078 |
| chemvision-vllm download runs and other CPU-only apps | — | 0.027 | 0.027 |
| **Through 23:00 PT** | **0.537** | **0.046** | **0.583** |

Estimates in the rows above were within a few cents of these actuals. Runs after 23:00 PT (the n=500
cycle) are logged by their owner below and will show in the next closed hour.

### Phase 3 (Person A), 2026-11-01 from 23:00 PT; estimates from container time, replace with billed actuals when the hour closes

| When (PT) | Run | Platform / GPU | Container time | Cost (USD) | Notes |
|---|---|---|---|---|---|
| 22:59–23:06 | upload 384 px images (164 MB) to volume `chemvision-data` | Modal volume | — | ~0.00 | |
| 23:08–23:09 | `dry_384` training check (64 ex, 5 steps) | Modal 1× L40S ($1.95/hr) | ~1.2 min | est 0.04 | 1.65 s/step at batch 8 |
| 23:09–23:17 | `n500_r384_lang_e3` LoRA SFT (500 ex × 3 epochs) | Modal 1× L40S | 6.7 min | est 0.22 | 189 steps, ~5.5 ex/s |
| 23:09–23:17 | `untuned` eval, 500 test images, 8 workers | Modal 1× L40S (vLLM) | ~7.5 min incl. 201 s cold start + ~1.5 min idle | est 0.24 | app stopped manually |
| 23:17–23:26 | `n500_r384_lang_e3` eval, 500 test images, 8 workers | Modal 1× L40S (vLLM + LoRA) | ~8.5 min incl. 259 s cold start | est 0.28 | app stopped right after |
| 23:27–23:29 | first `n2000` / `n5000` launches, stopped ~10% in to add per-epoch checkpoints (local client killed; remote apps had to be stopped explicitly) | Modal 2× L40S | ~2 min each | est 0.13 | wasted; lesson: always confirm with `modal app list` |
| 23:30–23:52 | `n2000_r384_lang_e3` LoRA SFT (2,000 × 3 epochs, per-epoch ckpts, val on 200) | Modal 1× L40S | 21.6 min | est 0.70 | 750 steps; val loss 0.224 → 0.169 → 0.164 |
| 23:32–23:58 | `n2000_r384_merger_e3` ablation (a), LoRA + vision merger | Modal 1× L40S | 23.6 min | est 0.77 | val loss 0.210 → 0.160 → 0.156 |
| 23:53–00:11 | Session A (`src/eval_session.py`): n2000 ep1-3 on val, untuned + n500 rescored on test, untuned / untuned_thinking on first50, n2000_ep3 on test at 8 and 32 workers | Modal 1× L40S (vLLM, CUDA graphs) | 17.3 min incl. ~6 min cold start + 78 s one-off compile | est 0.56 | app stopped by the script |
