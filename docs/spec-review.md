# Spec review: where to push back

Written 2026-11-01 against the "Tuned Small VLM for Molecule-to-SMILES" spec, after tonight's smoke test.
Items are ordered by how much they change the plan. Anything verified against Fireworks docs is in
`docs/fireworks-notes.md`; this file is about the design.

## 1. Base model: the spec's model cannot be tuned on Fireworks

Qwen3-VL-8B-Instruct is listed "Training: Not supported" and the SFT API rejects it. The alternate the
spec names, "Qwen3.5 9B", does exist as Qwen3.5-9B and is both vision-capable and tunable. Everything
tonight ran on Qwen3.5-9B. Update the spec, pitch, and the "8B" wording to "9B".

Qwen3.5 is a thinking model by default. Every call must disable thinking or the latency and token numbers
are meaningless. The predictors already do this.

## 2. The RDKit retry loop targets the wrong failure mode

The retry-with-error-message idea only helps when the model emits an unparseable string. Tonight the
untuned model produced 3/10 invalid outputs, all degenerate repeats, and 7/10 valid outputs that were
simply the wrong molecule. A tuned model will produce even fewer invalid strings. Valid-but-wrong is the
dominant failure, and no RDKit message fixes it.

Better use of the same hour: **self-consistency as a confidence signal for routing.** Sample n=3 at a
modest temperature, canonicalize, and if all three agree accept the answer; otherwise escalate to the
large model. Agreement is the only confidence signal an open model gives us for free, and it catches
wrong-but-valid answers. Keep the retry loop as a one-line fallback for parse failures only.

## 3. Stereochemistry: decided, keep it but report both numbers

Decision taken on 1 Nov (commit `57f43b5`): labels keep stereo (canonical isomeric SMILES), RDKit draws
wedge bonds for the chiral centres, and the headline exact match is stereo-aware. The stereo-agnostic
number (`exact_match_nostereo`) is reported next to it. Original concern, still valid as a risk: 63% of
ZINC molecules carry stereo marks, and small VLMs read wedge direction poorly, so expect the stereo-aware
number to sit well below the stereo-agnostic one. If the gap is large, say so in the pitch rather than
switching metrics after the fact.

## 4. Do not hand-label the real-world set

Thirty to fifty hand-verified images is two or three hours of build-day time and gives a set too small to
say anything. Public OCSR benchmarks exist with labels: the USPTO patent set, UOB, CLEF-2012, JPO, and
DECIMER's hand-drawn set. Download one tonight, sample 200 to 500 images, and report on that. It is more
credible to judges than our own labels. Details in `docs/data-sources.md`.

## 5. Make the test set bigger; it is free

The spec's 500 synthetic images give a 95% interval of about plus or minus 4.4 points at 50% accuracy, so
a 3-point gap is noise. Rendering is free and inference on the small model is about $1 per 500 images on
an L40S. Use 1,000 synthetic test images, and run the large baseline on at least 500 of them. Say the
interval width in the pitch.

## 6. Train on more than 5,000 images

Only the SMILES tokens are trained on (21 tokens per example tonight); the image and prompt are masked.
Training cost is dominated by the forward pass over roughly 200 image tokens, and the smoke run measured
about 0.8 s per image-step on one L40S with batch size 1. That makes 5,000 images x 3 epochs about 3.3
GPU-hours, and 20,000 images x 2 epochs about 9 GPU-hours. On Fireworks (if uploads are re-enabled) the
token-priced cost is small either way. OCSR models that work are trained on hundreds of thousands to
millions of images; 5,000 is a lower bound for showing any signal. Plan for 20,000 and let time decide.

Also batch the training: the smoke trainer uses batch size 1 with gradient accumulation. Padding to batch
size 8 should cut wall-clock several-fold.

## 7. One deployment can serve both small systems, and neither small system is serverless

Qwen3.5-9B is on-demand only on Fireworks (no serverless), so the untuned baseline cannot be run "for
free" on serverless either; it needs the same dedicated GPU as the tuned model. On Modal the base model and
its LoRA adapters are served from one GPU (confirmed tonight). On Fireworks an addons-enabled BF16
deployment with `#`-routing should do the same, but the docs do not state it outright; test it before
relying on it. Either way the GPU cost line is paid once, and the untuned-vs-tuned comparison is clean.

Cost reality: Fireworks H100/H200 is $8.00/h; Modal L40S is $1.95/h and bills per second with scale to
zero. If the rules allow it, serve the 9B systems from Modal and use Fireworks for the large serverless
baseline and (once uploads work) for training. If judges expect Fireworks serving, budget about 4 to 6
H100-hours ($32 to $48) and keep the deployment warm only during eval windows.

## 7b. Large baseline: pick from what is actually on serverless

Confirmed serverless with image input as of tonight: Kimi K3 ($3 in / $15 out per 1M tokens),
Qwen3.8-Max ($2 / $6), GLM-5.3-Flash ($0.15 / $0.50). Qwen3-VL-235B, Llama 4 Maverick and Kimi K2.5
are on-demand only, so the spec's implied "big Qwen VL" baseline would cost another $8/h GPU. Use
GLM-5.3-Flash as the cheap large baseline and Kimi K3 as the frontier reference on a 200-image subset.
Details and URLs in `docs/fireworks-notes.md`.

## 8. Run the eval concurrently

Serial inference ran at 3.7 s per image warm. The runner now takes `--workers`; 8 concurrent requests
against vLLM should bring 1,000 images to well under 15 minutes. Report latency per request, not wall
time divided by n.

## 9. Pitch the honest version of cost per correct answer

Dedicated-GPU cost per correct answer depends entirely on utilisation: a GPU that answers one image a
minute looks terrible, one that answers eight per second looks great. Report two numbers: measured cost at
our eval concurrency, and cost at the deployment's measured peak throughput. Say which is which.

## 10. Smaller items

- Cold start was 250 s for inference and 30 s for the trainer. Keep the deployment warm during the eval
  window; scale to zero between.
- Cap `max_tokens` at 150. A 40-heavy-atom SMILES is under 100 tokens; the cap also bounds the degenerate
  repeat case.
- Via the REST API or SDK an omitted LoRA rank means full-parameter training at double the token price.
  Always pass `loraRank` explicitly.
- A scaled-to-zero Fireworks deployment answers 503 `DEPLOYMENT_SCALING_UP` until warm; the predictor's
  503 retry loop already handles this.
- Ask the organizers tonight whether pre-written code is allowed. If not, tonight's work is the plan and
  the platform checks, which is what the spec says anyway.
