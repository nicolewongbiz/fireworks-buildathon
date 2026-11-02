# Cost log (Person A)

Actual costs from `modal billing report --show-resources` and Fireworks billing, not estimates, unless marked "est".

| When (PT) | Run | Platform / GPU | GPU time | Cost (USD) | Notes |
|---|---|---|---|---|---|
| 2026-11-01 21:55 | Qwen3-VL-8B weight download (abandoned model) | Modal CPU | — | 0.016 | 71 s download |
| 2026-11-01 22:00 | Qwen3.5-9B weight download | Modal CPU | — | 0.010 | |
| 2026-11-01 22:01–22:06 | `smoke_untuned` baseline, 10 images | Modal 1× L40S ($1.95/hr) | ~4.4 min billed so far | 0.144 GPU + 0.006 CPU/mem (partial hour) | cold start 250 s (image build + load + profiling); plus up to 5 min idle before scale-down, est +0.16 |
| 2026-11-01 | Fireworks dataset upload / SFT | Fireworks | — | 0.00 | blocked: dataset uploads disabled on account |

Running total: ~$0.18 billed so far (est ≤ $0.35 once idle scale-down is billed).
