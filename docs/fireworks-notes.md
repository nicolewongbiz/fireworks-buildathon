# Fireworks AI research notes (VLM SFT: molecule image -> SMILES)

Researched 1 Nov 2026 from official pages only (docs.fireworks.ai, fireworks.ai/pricing, fireworks.ai/models). Pages are undated unless noted; "unconfirmed" = not stated on any page I could reach. No API key was used.

## 0. Corrections to team assumptions (read first)

| Assumption | Verdict | Source |
|---|---|---|
| Qwen3-VL-8B-Instruct is tunable with vision | **False.** Model page: Fine-tuning "Not Supported"; also Serverless "Not Supported" (on-demand only). | https://fireworks.ai/models/fireworks/qwen3-vl-8b-instruct |
| "Qwen3.5 9B" exists on Fireworks | **True.** `accounts/fireworks/models/qwen3p5-9b`, ~9.4B params, image input Supported, fine-tuning Supported (LoRA), 262k ctx, HF Qwen/Qwen3.5-9B, created 1 Apr 2026. Serverless **Not supported** (on-demand only, no serverless price). Vision *training* for this specific model is not stated on the page (docs say "vision support is model- and surface-specific") -- unconfirmed from docs; team reports the SFT API accepted it. | https://fireworks.ai/models/fireworks/qwen3p5-9b ; https://docs.fireworks.ai/fine-tuning/models.md |
| Dedicated GPU ~ $8/hour | **True for H100/H200** ($8.00/hr each). B200 $13, B300 $15, GB300 $20. A100 price not listed and default A100 quota is 0. | https://fireworks.ai/pricing |
| Training ~6M tokens costs $1-10 | **True.** LoRA SFT, base <=16B: $0.50 / 1M training tokens -> 6M tokens x 1 epoch = ~$3. Tokens = dataset tokens x epochs; image tokens are billed ("VLM SFT is also billed per 1M tokens"). | https://fireworks.ai/pricing |
| Tuned model needs a dedicated deployment | **True.** "Serverless deployment is not supported for LoRA models"; on-demand is "the only supported method for serving trained models." | https://docs.fireworks.ai/fine-tuning/deploying-loras.md ; https://docs.fireworks.ai/fine-tuning/fine-tuning-models.md |
| Untuned Qwen3-VL-8B baseline can run on serverless | **False.** Not on serverless; needs its own on-demand deployment. Same for qwen3p5-9b. | model pages above |

## 1. Which vision models are tunable (SFT) right now

Tunability check: `firectl model get -a fireworks <MODEL-ID>` -> `Tunable: true`, or the "Tunable" tag in the model library (https://docs.fireworks.ai/fine-tuning/fine-tuning-models.md). The docs' tunable-models catalog is an interactive picker, so I could only verify per model page:

| Model ID | Image input | Fine-tune | Serverless | Source |
|---|---|---|---|---|
| accounts/fireworks/models/qwen3p5-9b | Yes | Yes (LoRA) | No | https://fireworks.ai/models/fireworks/qwen3p5-9b |
| accounts/fireworks/models/qwen3p5-27b | Yes | Yes (LoRA) | No | https://fireworks.ai/models/fireworks/qwen3p5-27b |
| accounts/fireworks/models/qwen3p5-35b-a3b | **No** | Yes | No | https://fireworks.ai/models/fireworks/qwen3p5-35b-a3b |
| accounts/fireworks/models/qwen3p5-122b-a10b | Yes | No | No | https://fireworks.ai/models/fireworks/qwen3p5-122b-a10b |
| accounts/fireworks/models/qwen3p5-397b-a17b | Yes | Yes (LoRA), beta | No | https://fireworks.ai/models/fireworks/qwen3p5-397b-a17b |
| accounts/fireworks/models/qwen3-vl-8b-instruct | Yes | **No** | No | https://fireworks.ai/models/fireworks/qwen3-vl-8b-instruct |
| accounts/fireworks/models/qwen3-vl-32b-instruct | Yes | No | No | https://fireworks.ai/models/fireworks/qwen3-vl-32b-instruct |
| accounts/fireworks/models/qwen3-vl-235b-a22b-instruct | Yes | No | No | https://fireworks.ai/models/fireworks/qwen3-vl-235b-a22b-instruct |
| accounts/fireworks/models/qwen2p5-vl-7b-instruct | Yes | No | No | https://fireworks.ai/models/fireworks/qwen2p5-vl-7b-instruct |
| accounts/fireworks/models/qwen2p5-vl-32b-instruct | Yes | No | No | https://fireworks.ai/models/fireworks/qwen2p5-vl-32b-instruct |
| accounts/fireworks/models/kimi-k2p5 (1.02T MoE) | Yes | Yes (LoRA) | No | https://fireworks.ai/models/fireworks/kimi-k2p5 |
| accounts/fireworks/models/kimi-k3 (2.81T MoE) | Yes | Yes (LoRA) | Yes | https://fireworks.ai/models/fireworks/kimi-k3 |
| accounts/fireworks/models/glm-5p3-flash (320B/18B active) | Yes | Yes (LoRA) | Yes | https://fireworks.ai/models/fireworks/glm-5p3-flash |

Note: `glm-4p6v` model page returned 404 (not checked further). A model page saying "Fine-tuning: Supported" does not by itself prove *vision* SFT; the VLM SFT doc only says to "confirm the base model in Models" (https://docs.fireworks.ai/fine-tuning/fine-tuning-vlm).

## 2. JSONL dataset format for VLM SFT

Source: https://docs.fireworks.ai/fine-tuning/fine-tuning-vlm (same as https://docs.fireworks.ai/fine-tuning/fine-tuning-vlm.md).

- Same `messages` JSONL as text SFT (OpenAI chat format works without conversion). Optional `tools`, per-message `weight` (0 = exclude from loss) or `trainable`.
- Images are **embedded as base64 data URIs** inside the JSONL; no separate image upload and no URL-hosted images are described. "Encode each image as a data URI before upload: `data:image/<type>;base64,...`"; keep MIME type accurate (`image/jpeg`, `image/png`).
- Image parts go on **`user`** messages (some renderers also allow `tool`); `system` and `assistant` messages **cannot** contain images.
- Three accepted part shapes: `{"type":"image_url","image_url":{"url":"data:..."}}`, `{"type":"image_url","image_url":"data:..."}`, `{"type":"image","image":"data:..."}`.
- The page gives the part shapes but no complete vision line. Composed from the documented shapes (illustrative, not a verbatim quote):

```json
{"messages":[{"role":"user","content":[{"type":"text","text":"Give the SMILES for this molecule."},{"type":"image_url","image_url":{"url":"data:image/png;base64,iVBORw0KGgo..."}}]},{"role":"assistant","content":"CC(=O)Oc1ccccc1C(=O)O"}]}
```

- Dataset limits: **3 to 3,000,000 lines**; max 3M examples x epochs; examples truncated at max context length (>=32768 for most models). UI recommended for files <500 MB, firectl for larger. Single-request REST upload handles files up to **150 MB**; larger files use the signed-URL flow (https://docs.fireworks.ai/api-reference/upload-dataset-files.md).
- **Unconfirmed:** max image resolution/pixels, max image bytes, max images per example, or whether the 120-image / 10 MB inference limits apply to training.

## 3. Creating a dataset and launching SFT

**firectl install (macOS):** `brew tap fw-ai/firectl && brew install firectl` (run `brew update` on SHA256 failure); or download `darwin-arm64.gz` / `darwin-amd64.gz` from `storage.googleapis.com/fireworks-public/firectl/`. Sign in: `firectl signin`, verify `firectl whoami`. Shapes need firectl >= 1.8.8. https://docs.fireworks.ai/tools-sdks/firectl/firectl ; https://docs.fireworks.ai/getting-started/ondemand-quickstart.md

**firectl commands** (https://docs.fireworks.ai/fine-tuning/fine-tuning-models.md ; https://docs.fireworks.ai/tools-sdks/firectl/commands/supervised-fine-tuning-job-create.md):

```bash
firectl dataset create <DATASET_ID> /path/to/train.jsonl
firectl sftj create --base-model qwen3p5-9b --dataset <DATASET_ID> --output-model <OUT_ID> \
  --epochs 1 --lora-rank 8 --batch-size-samples 32 [--learning-rate 1e-4] \
  [--learning-rate-scheduler cosine --learning-rate-warmup-steps 10] [--evaluation-dataset <EVAL_ID>]
firectl sftj get <JOB_ID>
firectl model list
```

**REST:** `POST https://api.fireworks.ai/v1/accounts/{account_id}/datasets` body `{"datasetId":"...","dataset":{"format":"CHAT","exampleCount":"N"}}` -> `POST .../datasets/{id}:upload` (multipart `file`, <=150 MB) -> poll `GET .../datasets/{id}` until `state == READY`. Large files: `POST .../datasets/{id}:getUploadEndpoint` with `filenameToSize`, PUT to signed URLs, then `POST .../datasets/{id}:validateUpload`. SFT: `POST https://api.fireworks.ai/v1/accounts/{account_id}/supervisedFineTuningJobs` with `baseModel`, `dataset`, `outputModel`, `epochs`, `learningRate`, `loraRank`, `batchSizeSamples`, `maxContextLength`, `lrScheduler`, `learningRateWarmupSteps`, `evaluationDataset`, `isTurbo`. Error `FAILED_PRECONDITION` maps to HTTP 400 in the gateway enum. https://docs.fireworks.ai/api-reference/create-dataset.md ; https://docs.fireworks.ai/api-reference/upload-dataset-files.md ; https://docs.fireworks.ai/api-reference/get-dataset-upload-endpoint.md ; https://docs.fireworks.ai/api-reference/validate-dataset-upload.md ; https://docs.fireworks.ai/api-reference/create-supervised-fine-tuning-job.md

**Python SDK:** `pip install fireworks-ai` (training extras: Python 3.11+, `pip install --upgrade 'fireworks-ai[training]>=1.2.11,<2'`). Documented: `client.datasets.create(dataset_id=..., dataset={"exampleCount": ...})`, `client.datasets.upload(dataset_id=..., file=Path(...))`, poll `client.datasets.get(...)` for `state == "READY"`, `client.supervised_fine_tuning_jobs.cancel(...)`. A full SDK SFT-create example was **not** in the README or docs (unconfirmed param names; mirror the REST fields). https://github.com/fw-ai-external/python-sdk ; https://docs.fireworks.ai/tools-sdks/python-sdk.md

**Hyperparameters and defaults** (https://docs.fireworks.ai/fine-tuning/fine-tuning-models.md): epochs default 1 (integers); learning rate set per model automatically (docs advise not changing); LoRA rank default 8 via firectl, max 128 (shape-dependent); **via REST/SDK an omitted rank = 0 = full-parameter training, so set `loraRank` explicitly**; `batch_size_samples` default 32 (SFT); max context >=32768 default; scheduler `constant` (legacy) / `linear` / `cosine`; weight decay default 0.01; eval auto-carved if no eval dataset.

**"V2":** no separate endpoint. "Training V2" is a per-model routing path with different accepted params: `batch_size` and `gradient_accumulation_steps` rejected (use `batch_size_samples`), `jinja_template` rejected (base model renderer used), `early_stop`/MTP/`extra_values` unsupported; `linear`/`cosine` schedulers only on V2 models. https://docs.fireworks.ai/fine-tuning/fine-tuning-models.md ; https://docs.fireworks.ai/api-reference/create-supervised-fine-tuning-job.md

## 4. Deploying the fine-tuned model

(a) **Serverless LoRA: not available** ("LoRA requires On-demand"; "Serverless deployment is not supported for LoRA models"). https://docs.fireworks.ai/serverless/overview.md ; https://docs.fireworks.ai/fine-tuning/deploying-loras.md

(b) **On-demand with addons** (https://docs.fireworks.ai/fine-tuning/deploying-loras.md ; https://docs.fireworks.ai/tools-sdks/firectl/commands/deployment-create.md ; https://docs.fireworks.ai/deployments/autoscaling.md):

```bash
firectl deployment-shape-version list --base-model accounts/fireworks/models/qwen3p5-9b   # pick a BF16 shape (FP8/FP4 shapes do not support addons)
firectl deployment create accounts/fireworks/models/qwen3p5-9b --deployment-shape <bf16-shape> --enable-addons \
  --min-replica-count 0 --max-replica-count 1 --scale-to-zero-window 5m --scale-down-window 5m --wait
firectl load-lora <FINE_TUNED_MODEL_ID> --deployment <DEPLOYMENT_ID>
```

- Request routing: LoRA -> `model: "accounts/<ACCT>/models/<FT_ID>#accounts/<ACCT>/deployments/<DEP_ID>"`; base model on a deployment -> `model: "accounts/fireworks/models/<BASE>#accounts/<ACCT>/deployments/<DEP_ID>"` (Fireworks SDK) or `model: "accounts/<ACCT>/deployments/<DEP_ID>"` (OpenAI SDK/curl). The `#` syntax implies one addons-enabled deployment can serve the base and LoRAs, but no page states this explicitly -- **unconfirmed**; test it. Alternative: "live merge" (single adapter merged at deploy time, no `#` routing). https://docs.fireworks.ai/getting-started/ondemand-quickstart.md
- Accelerators: NVIDIA_A100_80GB, H100_80GB, H200_141GB, B200_180GB, B300_288GB, AMD_MI325X_256GB, MI350X_288GB (region-dependent). Pass `--accelerator-type`; always pass `--deployment-shape` (`default` lets Fireworks pick). https://docs.fireworks.ai/guides/ondemand-deployments
- Autoscale defaults: min replicas 0, max 1, `--scale-to-zero-window` 1h (min 5m), `--scale-down-window` 10m, `--scale-up-window` 30s, `--load-targets default=0.8`. Scaled-to-zero deployments return **503 `DEPLOYMENT_SCALING_UP`** (not queued) -> clients must retry. Deployments at min 0 are auto-deleted after 7 days idle. https://docs.fireworks.ai/deployments/autoscaling.md
- Spin-up: "a few minutes" (no figure); billed per GPU-second, "no extra charges for start-up times". https://docs.fireworks.ai/getting-started/ondemand-quickstart.md ; https://fireworks.ai/pricing
- Max LoRAs per deployment: not stated; account quota is 100 on-demand LoRAs. https://docs.fireworks.ai/guides/quotas_usage/account-quotas.md

## 5. Pricing (https://fireworks.ai/pricing, fetched 1 Nov 2026)

- **Fine-tuning ($/1M training tokens):** <=16B: LoRA SFT $0.50, full-param $1.00; 16.1-80B: $3 / $6; 80-300B: $6 / $12; >300B: $10 / $20. Tokens = dataset tokens x epochs; VLM SFT "also billed per 1M tokens" (image tokens count; see sec. 7). Muse Glimmer 30B SFT free until 30 Nov 2026.
- **On-demand GPUs ($/hr):** H100 80GB $8.00, H200 141GB $8.00, B200 $13.00, B300 $15.00, GB300 $20.00. Per GPU-second; region-restricted deployments 1.5x. A100 / AMD prices: not listed.
- **Serverless per-token (docs):** size tiers for unlisted models: <4B $0.10, 4-16B $0.20, >16B $0.90, MoE <=56B $0.50, MoE 56-176B $1.20 (input = output). Named: Kimi K3 $3.00 in / $0.30 cached / $15.00 out; GLM 5.3 $1.40/$0.26/$4.40; GLM 5.3 Flash $0.15/$0.03/$0.50; Qwen 3.8 Max $2.00/$0.25/$6.00. Priority +25%, Fast +50%, US-only +10%. Batch = 50% of serverless. https://docs.fireworks.ai/serverless/pricing.md
- Example: 1 H100 deployment for 3 hours = $24. ~1000 images x ~1.3k tokens on Kimi K3 serverless ~= $4 input.

## 6. Large vision models on serverless (for the "big baseline")

Confirmed serverless + image input, with price (model pages, 1 Nov 2026):

| Model ID | Params | $/1M in / out | Source |
|---|---|---|---|
| accounts/fireworks/models/kimi-k3 | 2.81T MoE | $3.00 / $15.00 | https://fireworks.ai/models/fireworks/kimi-k3 |
| accounts/fireworks/models/qwen3p8-max | ~2.4T MoE, 95B active | $2.00 / $6.00 | https://fireworks.ai/models/fireworks/qwen3p8-max |
| accounts/fireworks/models/glm-5p3-flash | 320B MoE, 18B active | $0.15 / $0.50 | https://fireworks.ai/models/fireworks/glm-5p3-flash |

**Not** on serverless (on-demand only): qwen3-vl-235b-a22b-instruct (recommends 8x H200), qwen3-vl-32b/8b, qwen2p5-vl-7b/32b, llama4-maverick-instruct-basic, kimi-k2p5, qwen3p5-397b-a17b/122b-a10b/27b/9b. glm-5p3 (non-flash) has **no** image input. GLM-4.6V, Gemma 3, Mistral Small 3.x: not checked (unconfirmed). Full live list: `GET https://api.fireworks.ai/v1/accounts/fireworks/models?filter=supports_serverless%3Dtrue` or https://app.fireworks.ai/models?filter=LLM&serverless=true (https://docs.fireworks.ai/faq-new/models-inference/how-to-check-if-a-model-is-available-on-serverless.md). **Qwen3-VL-8B-Instruct is not on serverless** -> untuned baseline needs an on-demand deployment.

## 7. Image token counting

Only Qwen2.5-VL is documented (https://docs.fireworks.ai/faq-new/billing-pricing/how-many-tokens-per-image.md): 336x336 -> 144 tokens, 672x672 -> 576, 1024x1024 -> 1,369, 1280x720 -> 1,196, 1920x1080 -> 2,769. Interpolating: ~448x448 -> ~256, ~512x512 -> ~330 (my estimate, not from the page). Method: count image token ID 151655 after running the HF AutoProcessor; a `token_calculator.py` script is on the page. Qwen3-VL / Qwen3.5 counts: **unconfirmed** (not documented; run the HF processor for `Qwen/Qwen3.5-9B` locally). Inference FAQ elsewhere says images typically cost 1,000-2,500 tokens, resolution-dependent.

## 8. Quotas, credits, and the dataset-upload FailedPrecondition

> **Correction (22:55 PT, from live API reads on the team account):** the gating hypothesis below does
> not fit. Payment method on file and $5 paid; `GET /v1/accounts/{id}/quotas` shows training-h200/b200/b300
> = 32, job-submission-count 8, monthly-spend-usd 25; `GET /v1/accounts/{id}` shows state READY,
> suspendState UNSUSPENDED, ratelimitTier 1, `accountVerification: NOT_VERIFIED`, created 2026-11-01
> 20:57 PT. The error is gRPC `FailedPrecondition: "dataset uploads are temporarily disabled, please
> contact Fireworks AI to get access"`, returned by the get-upload-endpoint step on a 750 KB file. Best
> remaining lead: NOT_VERIFIED on an hours-old account; resolution is through Fireworks support or staff.

- Default on-demand GPU quota (GLOBAL multi-region, new account): H100 16, H200 16, B200 16, B300 16, GB300 0, A100 0, AMD 0; on-demand LoRAs 100. All single regions/US/EU start at 0. **Training GPU quota: 0 without a payment method; 32 each of H200/B200/B300 with one.** Without a payment method training jobs can be rejected with 403 `quota_exceeded` / `tier_required`. Check with `firectl quota list`. https://docs.fireworks.ai/guides/quotas_usage/account-quotas.md ; https://docs.fireworks.ai/deployments/regions.md
- Credits: $1 starter credit; prepaid; without a payment method the account is **suspended** once the $1 is used; usage pauses at zero balance unless Auto Reload. https://docs.fireworks.ai/faq-new/billing-pricing/what-happens-when-i-finish-my-1-dollar-credit.md ; https://docs.fireworks.ai/faq-new/billing-pricing/how-does-billing-and-credit-usage-work.md
- **FailedPrecondition on dataset upload:** no doc mentions it. Closest: HTTP 412 Precondition Failed = "Account is suspended or there's an issue with account status" (https://docs.fireworks.ai/guides/inference-error-codes.md), and the payment-method gating above. Most plausible causes (unconfirmed): no payment method / suspended after the $1 credit; dataset not in `UPLOADING`/expected state (e.g. re-uploading to an existing dataset); file >150 MB on the single-request path. Try: add a payment method, `firectl quota list`, create a fresh dataset ID, keep file <150 MB or use `:getUploadEndpoint` + `:validateUpload`, or `--external-url gs://...`.
- Hackathon credits: no official policy page. Example sponsored event gave "$50 in Fireworks credit for all participants" (https://lu.ma/MongoDB-GenAI-Mountain-View); Fireworks community page says they sometimes provide credits for hackathons via Discord / devrel@fireworks.ai (https://fireworks.ai/community). Amount for this event: unconfirmed.

## 9. Serverless rate limits (for ~1000 image requests)

- Account-wide RPM cap: **10 RPM** with no payment method / no credits; **6,000 RPM** with payment method + active credits. https://docs.fireworks.ai/guides/quotas_usage/account-quotas.md
- TPM is adaptive, per account per model: Tier 3+ ceilings for models <800B: 216M prompt TPM / 54M uncached / 2.16M generated; 800B-1.6T: 43.2M / 10.8M / 432k; >=1.6T (Kimi K3, Qwen 3.8 Max): 21.6M / 5.4M / 216k. Lower tiers have lower (unlisted) ceilings; limits start at a default and rise 25% per 15 min when usage >50% of limit. 429 -> honor `Retry-After` / exponential backoff; 503 load-shedding possible. No concurrency limit documented. https://docs.fireworks.ai/serverless/rate-limits.md
- On-demand deployments: no account rate limits; 429 means the deployment is saturated. https://docs.fireworks.ai/guides/inference-error-codes.md
- Inference image limits: <=120 images per request, base64 images <10 MB total per request, URL images <5 MB each; formats png/jpg/jpeg/gif/bmp/tiff/ppm. https://docs.fireworks.ai/guides/querying-vision-language-models

## 10. Endpoint, image input, disabling thinking

- Endpoint: `POST https://api.fireworks.ai/inference/v1/chat/completions` (OpenAI-compatible; base URL `https://api.fireworks.ai/inference/v1`), `Authorization: Bearer $FIREWORKS_API_KEY`. https://docs.fireworks.ai/guides/querying-vision-language-models
- Image part: `{"type":"image_url","image_url":{"url":"data:image/png;base64,<B64>"}}` alongside `{"type":"text","text":"..."}`; `image_url` also accepts `detail` (low/default/high) and `max_long_side_pixel`. https://docs.fireworks.ai/api-reference/post-chatcompletions
- Disable thinking: `"reasoning_effort": "none"` (or `false`); Anthropic-style `"thinking": {"type":"disabled"}` is equivalent. `chat_template_kwargs: {"enable_thinking": false}` is auto-mapped to `reasoning_effort: "none"`. Models that reject `none` return 400. Documented for Qwen3.8/DeepSeek/GLM 5.2/Kimi K3; Qwen3.5-9B specifically: unconfirmed (test one request). https://docs.fireworks.ai/api-reference/post-chatcompletions ; https://docs.fireworks.ai/tools-sdks/nim-compatibility.md
