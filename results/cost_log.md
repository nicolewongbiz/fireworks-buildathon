# Cost log (Person A)

Actual costs from `modal billing report --show-resources` and Fireworks billing, not estimates, unless marked "est".

| When (PT) | Run | Platform / GPU | GPU time | Cost (USD) | Notes |
|---|---|---|---|---|---|
| 2026-11-01 21:55 | Qwen3-VL-8B weight download (abandoned model) | Modal CPU | — | 0.016 | 71 s download |
| 2026-11-01 22:00 | Qwen3.5-9B weight download | Modal CPU | — | 0.010 | |
| 2026-11-01 22:01–22:06 | `smoke_untuned` baseline, 10 images | Modal 1× L40S ($1.95/hr) | ~4.4 min billed so far | 0.144 GPU + 0.006 CPU/mem (partial hour) | cold start 250 s (image build + load + profiling); plus up to 5 min idle before scale-down, est +0.16 |
| 2026-11-01 22:09–22:12 | `smoke` LoRA SFT, 40 images, 3 epochs | Modal 1× L40S ($1.95/hr) | 2.4 min | est 0.08 GPU | train 99 s; adapter saved to `chemvision-loras:/smoke`; container exited, no idle |
| 2026-11-01 22:12–22:17 | `smoke_tuned` eval, 10 images, LoRA `smoke` | Modal 1× L40S ($1.95/hr) | ~4.2 min | est 0.14 GPU | cold start 217 s (image cached); app stopped right after, no idle tail |
| 2026-11-01 | Fireworks dataset upload / SFT | Fireworks | — | 0.00 | blocked: dataset uploads disabled on account |

Running total: est ~$0.40 incl. the baseline's idle tail (replace with billed numbers after 23:00 PT). `chemvision-vllm` stopped at 22:18 PT; redeploy with `uv run modal deploy src/modal_serve.py`.

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
