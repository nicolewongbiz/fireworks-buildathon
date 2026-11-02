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

## 3. Stereochemistry has to be stripped on both sides

ZINC SMILES carry `@`, `/`, `\` marks. A flat 2D rendering without wedge bonds cannot show them, so a
model that reads the picture correctly still gets scored wrong if the label keeps stereo. The validator
strips stereo from both prediction and truth. The renderer should also strip stereo from the molecule
before drawing so no wedges appear. If we want stereo as a stretch, that is a separate experiment with
wedged renderings and stereo-preserving labels.

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

## 7. One deployment can serve both small systems

On Modal, and on a Fireworks on-demand deployment, the base model and its LoRA adapters are served from
the same GPU. The untuned baseline and the tuned model share one deployment, so the "dedicated GPU" cost
line is paid once, not twice, and the cost-per-correct-answer comparison between them is clean.

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
- Ask the organizers tonight whether pre-written code is allowed. If not, tonight's work is the plan and
  the platform checks, which is what the spec says anyway.
