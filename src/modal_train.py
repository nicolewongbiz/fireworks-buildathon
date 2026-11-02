"""LoRA SFT for Qwen3.5-9B (molecule image -> SMILES) on Modal, used while Fireworks dataset uploads are disabled.

Person A owns this. Data contract (built by Person B):
  data/train_{res}.json, data/val_{res}.json  -> [{"image_path": "data/images/<res>/<split>/<id>.png", "smiles": ...}, ...]
  Images live in the Modal volume `chemvision-data` at the same repo-relative path minus the leading "data/"
  (uploaded separately with `modal volume put chemvision-data data/images /images`). Only paths and SMILES go
  over the wire; the container reads the PNGs from the volume.

Usage (every GPU run needs the user's approval and a cost estimate first, see CLAUDE.md):
  # quick dry check: 5 optimizer steps, tiny val set, throwaway run name
  uv run modal run -m src.modal_train --train data/train_384.json --val data/val_384.json \
      --n-train 64 --n-val 16 --max-steps 5 --run-name dry_384 --overwrite
  # real run, language-only LoRA (what vLLM serves today with no extra flags)
  uv run modal run -m src.modal_train --train data/train_384.json --val data/val_384.json --n-train 500 --epochs 3
  # also adapt the projector, or projector + vision encoder (vision LR = lr * vision-lr-mult)
  uv run modal run -m src.modal_train ... --lora-scope language+merger
  uv run modal run -m src.modal_train ... --lora-scope language+merger+vision --vision-lr-mult 0.1 --gpu H100
  # also write a merged bf16 model (fallback serving path, ~19 GB in the volume)
  uv run modal run -m src.modal_train ... --merge-full

Flags (defaults): --n-train 0 (=all; takes the FIRST N rows so subsets are nested), --n-val 128 (0=all), --epochs 3,
  --lr 2e-4, --rank 8 (alpha = 2*rank), --lora-dropout 0.05, --batch-size 8, --grad-accum 1,
  --warmup-frac 0.05 (cosine decay to 0), --max-grad-norm 1.0, --eval-every 0 (=auto, ~10 evals per run plus
  one before training and one at the end), --max-steps 0 (=no cap), --seed 0, --gpu L40S|H100,
  --timeout-min 120, --lora-scope language|language+merger|language+merger+vision, --vision-lr-mult 0.1,
  --run-name "" (=auto), --overwrite, --merge-full, --num-workers 4.

Outputs in the `chemvision-loras` volume:
  /<run_name>/adapter_model.safetensors, adapter_config.json   PEFT adapter; src/modal_serve.py serves every
                                                               dir with an adapter_config.json as model "<run_name>"
  /<run_name>/train_config.json, history.json                  args, data sizes, LoRA module report, losses, timings
  /merged/<run_name>/                                          only with --merge-full: merged bf16 model + processor
Run name defaults to n{n_train}_r{res}_{lang|merger|vision}_e{epochs} (+ _s{max_steps} for capped runs). An
existing run dir is never overwritten without --overwrite.

Serving adapters that touch the vision side (vLLM 0.21): Qwen3_5ForConditionalGeneration inherits
get_mm_mapping()/get_num_mm_encoder_tokens()/get_num_mm_connector_tokens() from Qwen3-VL, so tower (visual.blocks)
and connector (visual.merger) LoRA are supported, but only with `--enable-tower-connector-lora` (experimental).
Without that flag vLLM loads the adapter and silently drops the visual.* LoRA weights ("no matching PunicaWrapper
... will be ignored"), so the served model would not be the trained one. modal_serve.py must add the flag when
any adapter targets visual.* modules. --merge-full is the fallback if that path misbehaves.

Cost model (fill in the TODOs after the first real run; the script prints s/step and examples/s at every eval):
  steps        = ceil(n_train / (batch_size * grad_accum)) * epochs          (or --max-steps if smaller)
  train_s      = steps * S_PER_STEP[gpu, res, batch_size]                     TODO: measure (smoke: ~0.8 s/example at bs 1)
  eval_s       = (n_evals + 2) * ceil(n_val / (2 * batch_size)) * S_PER_EVAL_BATCH   TODO: measure
  total_s      = ~150 s container start + weight load  + train_s + eval_s  (+ ~3-5 min with --merge-full)
  cost         = total_s / 3600 * rate   (L40S $1.95/h as in results/cost_log.md; H100: check modal.com/pricing)
"""
import modal

from src.modal_serve import MODEL_NAME, MODEL_REVISION, hf_cache_vol, lora_vol
from src.predict import PROMPT  # train and inference prompts must be identical

MINUTES = 60
GPUS = ("L40S", "H100")
DATA_MOUNT = "/data"
LORA_MOUNT = "/loras"

# PEFT matches these with re.fullmatch against module names such as
# "model.language_model.layers.3.self_attn.q_proj" or "model.visual.blocks.0.attn.qkv".
# Language: MLP in all 32 layers + q/k/v/o in the 8 full-attention layers. The 24 Gated DeltaNet `linear_attn`
# layers and the `mtp.*` (multi-token prediction) head are deliberately excluded.
_LANG = r".*language_model\.layers\.\d+\.(?:mlp\.(?:gate|up|down)_proj|self_attn\.(?:q|k|v|o)_proj)"
_MERGER = r".*visual\.merger\.linear_fc[12]"
_VISION = r".*visual\.blocks\.\d+\.(?:attn\.(?:qkv|proj)|mlp\.linear_fc[12])"
SCOPES = {
    "language": (_LANG,),
    "language+merger": (_LANG, _MERGER),
    "language+merger+vision": (_LANG, _MERGER, _VISION),
}
SCOPE_SHORT = {"language": "lang", "language+merger": "merger", "language+merger+vision": "vision"}
# From the checkpoint's safetensors shapes, LoRA params = rank * (in + out) per wrapped Linear:
#   language 128 modules -> 1,818,624 * rank; merger 2 -> 17,920 * rank; vision blocks 108 -> 481,248 * rank.
EXPECTED = {  # scope -> (wrapped modules, trainable params per unit rank)
    "language": (128, 1_818_624),
    "language+merger": (130, 1_836_544),
    "language+merger+vision": (238, 2_317_792),
}


def lora_target_regex(scope: str) -> str:
    return "(?:" + "|".join(SCOPES[scope]) + ")"


def default_run_name(n_train: int, res: str, scope: str, epochs: int, max_steps: int) -> str:
    name = f"n{n_train}_r{res}_{SCOPE_SHORT[scope]}_e{epochs}"
    return name + (f"_s{max_steps}" if max_steps else "")


def volume_path(image_path: str) -> str:
    """Repo-relative manifest path (data/images/...) -> path inside the container (/data/images/...)."""
    if not image_path.startswith("data/"):
        raise ValueError(f"image_path must start with 'data/': {image_path!r}")
    return f"{DATA_MOUNT}/{image_path.removeprefix('data/')}"


def encode_example(processor, image, smiles: str) -> dict:
    """One example -> tensors with a leading batch dim of 1. Same rendering vLLM uses at inference:
    user turn (text then image, as in src/predict.py), generation prompt, thinking disabled; the target is
    the SMILES plus end-of-turn, and only the target is trained on (prompt labels = -100)."""
    import torch

    content = [{"type": "text", "text": PROMPT}, {"type": "image"}]
    prompt = processor.apply_chat_template(
        [{"role": "user", "content": content}], add_generation_prompt=True, tokenize=False, enable_thinking=False
    )
    enc = dict(processor(text=[prompt], images=[image], return_tensors="pt"))
    target = processor.tokenizer(smiles + "<|im_end|>\n", add_special_tokens=False, return_tensors="pt").input_ids
    n_prompt = enc["input_ids"].shape[1]
    out = {}
    for k, v in enc.items():
        if k == "input_ids":
            out[k] = torch.cat([v, target], dim=1)
        elif k == "attention_mask":
            out[k] = torch.cat([v, torch.ones_like(target)], dim=1)
        elif torch.is_tensor(v) and v.dim() == 2 and v.shape == (1, n_prompt):
            out[k] = torch.cat([v, torch.zeros_like(target, dtype=v.dtype)], dim=1)  # per-token ids: 0 = text
        else:
            out[k] = v  # pixel_values (n_patches, patch_dim), image_grid_thw (1, 3)
    out["labels"] = torch.cat([torch.full_like(enc["input_ids"], -100), target], dim=1)
    return out


# Qwen-VL processors flatten all images of a batch into one patch matrix and keep one (t, h, w) row per image;
# the model splits pixel_values back per image using image_grid_thw, in the same order as the image tokens.
_CONCAT_KEYS = ("pixel_values", "image_grid_thw")


def collate(examples: list[dict], pad_id: int) -> dict:
    """Right-pad per-token tensors to the longest sequence; concatenate image tensors along dim 0.
    Right padding keeps the Gated DeltaNet (recurrent) layers' state for real tokens untouched."""
    import torch

    max_len = max(ex["input_ids"].shape[1] for ex in examples)
    pad_value = {"input_ids": pad_id, "labels": -100}  # everything else per-token (attention_mask, mm ids) -> 0
    batch = {}
    for k in examples[0]:
        vals = [ex[k] for ex in examples]
        if k in _CONCAT_KEYS:
            batch[k] = torch.cat(vals, dim=0)
        elif all(v.dim() == 2 and v.shape[0] == 1 and v.shape[1] == ex["input_ids"].shape[1] for v, ex in zip(vals, examples)):
            fill = pad_value.get(k, 0)
            batch[k] = torch.cat(
                [torch.nn.functional.pad(v, (0, max_len - v.shape[1]), value=fill) for v in vals], dim=0
            )
        else:
            raise ValueError(f"collate: don't know how to batch {k!r} with shapes {[tuple(v.shape) for v in vals]}")
    return batch


class ManifestDataset:
    """Map-style dataset over (image_path, smiles) pairs; images are read and processed lazily (in DataLoader
    workers), since pre-encoding 10k images would need tens of GB of RAM for pixel_values."""

    def __init__(self, processor, rows: list[list[str]]):
        self.processor = processor
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        from PIL import Image

        path, smiles = self.rows[i]
        with Image.open(volume_path(path)) as im:
            image = im.convert("RGB")
        return encode_example(self.processor, image, smiles)


def lora_module_report(model, scope: str, rank: int) -> dict:
    """Which modules PEFT actually wrapped, as a deduplicated pattern summary, plus vision/merger flags."""
    import re
    from collections import Counter

    from peft.tuners.lora import LoraLayer

    names = [n for n, m in model.named_modules() if isinstance(m, LoraLayer)]
    patterns = Counter(re.sub(r"\.\d+\.", ".N.", n.removeprefix("base_model.model.")) for n in names)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    report = {
        "scope": scope,
        "target_regex": lora_target_regex(scope),
        "n_modules": len(names),
        "patterns": dict(sorted(patterns.items())),
        "trainable_params": trainable,
        "total_params": total,
        "includes_vision_blocks": any("visual.blocks" in n for n in names),
        "includes_merger": any("visual.merger" in n for n in names),
        "includes_mtp": any("mtp" in n.split(".") for n in names),
        "expected_modules": EXPECTED[scope][0],
        "expected_trainable_params": EXPECTED[scope][1] * rank,
    }
    print(f"LoRA scope {scope!r}: wraps {len(names)} modules (expected {report['expected_modules']})")
    for pat, n in report["patterns"].items():
        print(f"  {n:4d} x {pat}")
    print(f"trainable params: {trainable:,} of {total:,} ({100 * trainable / total:.3f} %), "
          f"expected {report['expected_trainable_params']:,}")
    print(f"vision encoder (model.visual.blocks) included: {report['includes_vision_blocks']}")
    print(f"projector (model.visual.merger) included: {report['includes_merger']}")
    if report["includes_mtp"]:
        raise RuntimeError("LoRA wrapped mtp.* modules; refusing to train")
    if report["includes_merger"] != ("merger" in scope) or report["includes_vision_blocks"] != ("vision" in scope):
        raise RuntimeError(f"LoRA modules don't match scope {scope!r}; check module names above")
    if (len(names), trainable) != (report["expected_modules"], report["expected_trainable_params"]):
        print("WARNING: module or parameter count differs from the checkpoint-shape estimate")
    return report


train_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch>=2.13", "torchvision", "transformers==5.10.4", "peft", "accelerate", "pillow")
    .uv_pip_install("openai", "python-dotenv")  # only so `from src.predict import PROMPT` imports in the container
    .env({"TOKENIZERS_PARALLELISM": "false"})
    .add_local_python_source("src")
)
data_vol = modal.Volume.from_name("chemvision-data")

app = modal.App("chemvision-train")


@app.function(
    image=train_image,
    gpu="L40S",  # default; main() overrides per run with .with_options(gpu=..., timeout=...)
    cpu=4.0,  # DataLoader workers decode + preprocess images
    timeout=120 * MINUTES,
    volumes={"/root/.cache/huggingface": hf_cache_vol, LORA_MOUNT: lora_vol, DATA_MOUNT: data_vol},
)
def train(train_rows: list[list[str]], val_rows: list[list[str]], cfg: dict) -> dict:
    import json
    import math
    import os
    import random
    import shutil
    import time
    from pathlib import Path

    import torch
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import DataLoader
    from transformers import AutoModelForImageTextToText, AutoProcessor, get_cosine_schedule_with_warmup

    t_start = time.time()
    run_dir = Path(LORA_MOUNT) / cfg["run_name"]
    merged_dir = Path(LORA_MOUNT) / "merged" / cfg["run_name"]
    if not cfg["overwrite"] and (run_dir.exists() or (cfg["merge_full"] and merged_dir.exists())):
        raise FileExistsError(f"{run_dir} (or {merged_dir}) exists; pass --overwrite to replace it")

    # Fail fast on missing images, before paying for the weight load.
    by_dir: dict[str, set[str]] = {}
    for path, _ in train_rows + val_rows:
        d, f = os.path.split(volume_path(path))
        by_dir.setdefault(d, set()).add(f)
    missing = [f"{d}/{f}" for d, fs in by_dir.items() for f in fs - set(os.listdir(d) if os.path.isdir(d) else [])]
    if missing:
        raise FileNotFoundError(f"{len(missing)} images missing from volume chemvision-data, e.g. {missing[:3]}")

    seed = cfg["seed"]
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    processor = AutoProcessor.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    pad_id = processor.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = processor.tokenizer.eos_token_id
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_NAME, revision=MODEL_REVISION, dtype=torch.bfloat16, device_map="cuda"
    )
    # Non-reentrant checkpointing: frozen vision inputs never require grad, and reentrant checkpointing would
    # then silently give vision-block LoRA weights no gradient.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    rank = cfg["rank"]
    model = get_peft_model(
        model,
        LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=cfg["lora_dropout"],
                   target_modules=lora_target_regex(cfg["lora_scope"])),
    )
    module_report = lora_module_report(model, cfg["lora_scope"], rank)
    load_s = time.time() - t_start

    # Param groups: vision-encoder LoRA gets a lower LR; language and merger use the base LR.
    groups = {"language": [], "merger": [], "vision": []}
    for n, p in model.named_parameters():
        if p.requires_grad:
            groups["vision" if "visual.blocks" in n else "merger" if "visual.merger" in n else "language"].append(p)
    lr = cfg["lr"]
    param_groups = [
        {"name": g, "params": ps, "lr": lr * (cfg["vision_lr_mult"] if g == "vision" else 1.0)}
        for g, ps in groups.items() if ps
    ]
    print("param groups:", {g["name"]: (sum(p.numel() for p in g["params"]), g["lr"]) for g in param_groups})
    opt = torch.optim.AdamW(param_groups, weight_decay=0.0)

    bs, accum = cfg["batch_size"], cfg["grad_accum"]
    workers = cfg["num_workers"]

    def worker_init(_):
        torch.set_num_threads(1)

    loader_kw = dict(collate_fn=lambda exs: collate(exs, pad_id), num_workers=workers, pin_memory=True,
                     worker_init_fn=worker_init, persistent_workers=workers > 0)
    train_loader = DataLoader(ManifestDataset(processor, train_rows), batch_size=bs, shuffle=True,
                              generator=torch.Generator().manual_seed(seed), **loader_kw)
    val_loader = DataLoader(ManifestDataset(processor, val_rows), batch_size=2 * bs, shuffle=False, **loader_kw)

    steps_per_epoch = math.ceil(len(train_loader) / accum)
    total_steps = steps_per_epoch * cfg["epochs"]
    if cfg["max_steps"]:
        total_steps = min(total_steps, cfg["max_steps"])
    eval_every = cfg["eval_every"] or max(1, total_steps // 10)
    warmup = max(1, round(cfg["warmup_frac"] * total_steps))
    sched = get_cosine_schedule_with_warmup(opt, warmup, total_steps)
    print(f"train {len(train_rows)} / val {len(val_rows)} examples; batch {bs} x accum {accum}; "
          f"{steps_per_epoch} steps/epoch, {total_steps} steps total, warmup {warmup}, eval every {eval_every}")

    def to_cuda(batch: dict) -> dict:
        return {k: v.to("cuda", non_blocking=True) for k, v in batch.items()}

    def loss_terms(batch: dict) -> tuple:
        """(summed token NLL, number of target tokens). Only the logits for the target span are computed
        (logits_to_keep), which avoids a (batch x seq x 248k-vocab) fp32 logits tensor."""
        labels = batch["labels"]
        inputs = {k: v for k, v in batch.items() if k != "labels"}
        start = int((labels != -100).int().argmax(dim=1).min())  # first target position in the batch
        keep = labels.shape[1] - start + 1  # logits at positions start-1 .. L-1
        logits = model(**inputs, use_cache=False, logits_to_keep=keep).logits
        tgt = labels[:, start:]
        nll = torch.nn.functional.cross_entropy(
            logits[:, :-1].float().reshape(-1, logits.shape[-1]), tgt.reshape(-1), ignore_index=-100, reduction="sum"
        )
        return nll, int((tgt != -100).sum())

    @torch.no_grad()
    def evaluate() -> float:
        model.eval()
        tot, n = 0.0, 0
        for batch in val_loader:
            nll, k = loss_terms(to_cuda(batch))
            tot, n = tot + nll.item(), n + k
        model.train()
        return tot / max(n, 1)

    @torch.no_grad()
    def check_batching() -> dict:
        """Padded-batch loss must match per-example losses; catches collation/padding/image-order bugs."""
        model.eval()
        exs = [val_loader.dataset[i] for i in range(min(4, len(val_rows)))]
        nll_b, n_b = loss_terms(to_cuda(collate(exs, pad_id)))
        singles = [loss_terms(to_cuda(collate([ex], pad_id))) for ex in exs]
        model.train()
        batched = nll_b.item() / n_b
        single = sum(s.item() for s, _ in singles) / sum(k for _, k in singles)
        rel = abs(batched - single) / max(single, 1e-6)
        print(f"batching check: batched loss {batched:.4f} vs per-example {single:.4f} (rel diff {rel:.2%})")
        if rel > 0.10:
            raise RuntimeError("batched and per-example losses disagree by >10%; collation is broken")
        if rel > 0.02:
            print("WARNING: batched vs per-example loss differ by >2% (bf16 noise is usually <1%)")
        return {"batched": batched, "per_example": single, "rel_diff": rel}

    ex0 = val_loader.dataset[0] if val_rows else train_loader.dataset[0]
    print("example 0 tail:", processor.tokenizer.decode(ex0["input_ids"][0][-40:]))
    print("example 0 shapes:", {k: tuple(v.shape) for k, v in ex0.items()},
          "trained tokens:", int((ex0["labels"] != -100).sum()))
    batching = check_batching() if val_rows else None

    history = []
    t_eval = time.time()
    val0 = evaluate() if val_rows else None
    eval_s = time.time() - t_eval
    history.append({"step": 0, "epoch": 0, "examples": 0, "train_loss": None, "val_loss": val0, "lr": 0.0,
                    "elapsed_s": round(time.time() - t_start, 1)})
    print(f"step 0: val loss {val0}")

    model.train()
    step, examples_seen, micro = 0, 0, 0
    run_nll, run_tok = 0.0, 0
    train_s, t_window, steps_window = 0.0, time.time(), 0
    grad_check = None
    done = False
    for epoch in range(cfg["epochs"]):
        for i, batch in enumerate(train_loader):
            batch = to_cuda(batch)
            nll, k = loss_terms(batch)
            (nll / max(k, 1) / accum).backward()
            run_nll, run_tok = run_nll + nll.item(), run_tok + k
            examples_seen += batch["input_ids"].shape[0]
            micro += 1
            if micro % accum and i + 1 < len(train_loader):  # also step on a partial accumulation at epoch end
                continue
            micro = 0
            if grad_check is None:  # first step: every param group must actually receive gradient
                grad_check = {
                    g["name"]: math.sqrt(sum(float(p.grad.float().pow(2).sum()) for p in g["params"] if p.grad is not None))
                    for g in param_groups
                }
                print("grad norms at step 1:", {g: f"{v:.3e}" for g, v in grad_check.items()})
                dead = [g for g, v in grad_check.items() if not v > 0]
                if dead:
                    raise RuntimeError(f"no gradient reaches LoRA params in groups {dead}")
            torch.nn.utils.clip_grad_norm_([p for g in param_groups for p in g["params"]], cfg["max_grad_norm"])
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            steps_window += 1
            done = step >= total_steps
            if step % eval_every == 0 or done:
                window_s = time.time() - t_window
                train_s += window_s
                t_eval = time.time()
                val = evaluate() if val_rows else None
                eval_s += time.time() - t_eval
                rec = {
                    "step": step, "epoch": round(examples_seen / len(train_rows), 3), "examples": examples_seen,
                    "train_loss": run_nll / max(run_tok, 1), "val_loss": val, "lr": sched.get_last_lr()[0],
                    "s_per_step": window_s / steps_window, "examples_per_s": steps_window * bs * accum / window_s,
                    "elapsed_s": round(time.time() - t_start, 1),
                }
                history.append(rec)
                print(f"step {step}/{total_steps} ep {rec['epoch']:.2f}: train {rec['train_loss']:.4f} "
                      f"val {val if val is None else round(val, 4)} lr {rec['lr']:.2e} | "
                      f"{rec['s_per_step']:.2f} s/step, {rec['examples_per_s']:.1f} ex/s", flush=True)
                run_nll, run_tok, steps_window, t_window = 0.0, 0, 0, time.time()
            if done:
                break
        if done:
            break
    if history[-1]["step"] != step:  # loop ended without a final eval (should not happen)
        history.append({"step": step, "examples": examples_seen, "train_loss": run_nll / max(run_tok, 1),
                        "val_loss": evaluate() if val_rows else None, "elapsed_s": round(time.time() - t_start, 1)})

    t_save = time.time()
    if run_dir.exists():
        shutil.rmtree(run_dir)
    model.save_pretrained(run_dir)
    summary = {
        "run_name": cfg["run_name"], "optimizer_steps": step, "examples_seen": examples_seen,
        "train_size": len(train_rows), "val_size": len(val_rows), "seed": seed,
        "load_s": round(load_s, 1), "train_s": round(train_s, 1), "eval_s": round(eval_s, 1),
        "final_val_loss": history[-1]["val_loss"], "batching_check": batching, "grad_norms_step1": grad_check,
    }
    (run_dir / "train_config.json").write_text(json.dumps({**cfg, "lora": module_report}, indent=1))
    (run_dir / "history.json").write_text(json.dumps({"summary": summary, "history": history}, indent=1))
    lora_vol.commit()
    files = sorted(p.name for p in run_dir.iterdir())
    print("saved", run_dir, files)

    if cfg["merge_full"]:
        merged = model.merge_and_unload()
        if merged_dir.exists():
            shutil.rmtree(merged_dir)
        merged.save_pretrained(merged_dir, safe_serialization=True, max_shard_size="5GB")
        processor.save_pretrained(merged_dir)
        lora_vol.commit()
        print("saved merged model", merged_dir, sorted(p.name for p in merged_dir.iterdir()))
    summary["save_s"] = round(time.time() - t_save, 1)
    summary["total_s"] = round(time.time() - t_start, 1)
    (run_dir / "history.json").write_text(json.dumps({"summary": summary, "history": history}, indent=1))
    lora_vol.commit()
    return {**summary, "files": files, "history": history, "lora": module_report,
            "merged_dir": str(merged_dir) if cfg["merge_full"] else None}


_train_fn = train  # main()'s `--train` argument shadows the name inside main


def _load_manifest(path: str, n: int, strict: bool = True) -> list[list[str]]:
    import json
    from pathlib import Path

    rows = json.loads(Path(path).read_text())
    if n:
        if n > len(rows) and strict:
            raise SystemExit(f"asked for {n} rows but {path} has {len(rows)}")
        rows = rows[:n]  # first N, so smaller subsets are nested in larger ones
    for r in rows:
        volume_path(r["image_path"])  # validates the data/ prefix
    return [[r["image_path"], r["smiles"]] for r in rows]


def _infer_res(train_path: str, rows: list[list[str]]) -> str:
    import re

    m = re.search(r"data/images/(\d+)/", rows[0][0]) or re.search(r"_(\d+)\.json$", train_path)
    return m.group(1) if m else "NA"


@app.local_entrypoint()
def main(
    train: str,
    val: str,
    n_train: int = 0,
    n_val: int = 128,
    epochs: int = 3,
    lr: float = 2e-4,
    rank: int = 8,
    lora_dropout: float = 0.05,
    batch_size: int = 8,
    grad_accum: int = 1,
    warmup_frac: float = 0.05,
    max_grad_norm: float = 1.0,
    eval_every: int = 0,
    max_steps: int = 0,
    seed: int = 0,
    gpu: str = "L40S",
    timeout_min: int = 120,
    lora_scope: str = "language",
    vision_lr_mult: float = 0.1,
    run_name: str = "",
    overwrite: bool = False,
    merge_full: bool = False,
    num_workers: int = 4,
):
    import json
    import math
    import subprocess

    if lora_scope not in SCOPES:
        raise SystemExit(f"--lora-scope must be one of {list(SCOPES)}")
    if gpu not in GPUS:
        raise SystemExit(f"--gpu must be one of {GPUS}")
    train_rows = _load_manifest(train, n_train)
    val_rows = _load_manifest(val, n_val, strict=False)
    res = _infer_res(train, train_rows)
    run_name = run_name or default_run_name(len(train_rows), res, lora_scope, epochs, max_steps)
    try:
        existing = {e.path.strip("/") for e in lora_vol.listdir("/")}
    except Exception as e:  # the remote function re-checks anyway
        print("could not list chemvision-loras:", e)
        existing = set()
    if run_name in existing and not overwrite:
        raise SystemExit(f"run dir /{run_name} already exists in chemvision-loras; pass --overwrite or --run-name")
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = ""
    cfg = dict(
        run_name=run_name, train_manifest=train, val_manifest=val, n_train=len(train_rows), n_val=len(val_rows),
        res=res, epochs=epochs, lr=lr, rank=rank, lora_alpha=2 * rank, lora_dropout=lora_dropout,
        batch_size=batch_size, grad_accum=grad_accum, warmup_frac=warmup_frac, max_grad_norm=max_grad_norm,
        eval_every=eval_every, max_steps=max_steps, seed=seed, gpu=gpu, lora_scope=lora_scope,
        vision_lr_mult=vision_lr_mult, overwrite=overwrite, merge_full=merge_full, num_workers=num_workers,
        model=MODEL_NAME, revision=MODEL_REVISION, prompt=PROMPT, git_commit=commit,
    )
    steps = math.ceil(len(train_rows) / (batch_size * grad_accum)) * epochs
    if max_steps:
        steps = min(steps, max_steps)
    print(f"run {run_name!r}: {len(train_rows)} train / {len(val_rows)} val, scope {lora_scope}, gpu {gpu}, "
          f"~{steps} optimizer steps (see module docstring for the cost formula)")
    fn = _train_fn.with_options(gpu=gpu, timeout=timeout_min * MINUTES)
    result = fn.remote(train_rows, val_rows, cfg)
    print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=1))
    for h in result["history"]:
        print(h)
