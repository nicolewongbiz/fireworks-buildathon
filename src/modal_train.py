"""Fallback LoRA SFT for Qwen3.5-9B on Modal, used while Fireworks dataset uploads are disabled.

Person A owns this. Usage:
  uv run modal run -m src.modal_train --data data/smoke_train.jsonl --name smoke --epochs 3

Writes a PEFT adapter to the `chemvision-loras` volume at /<name>, which src/modal_serve.py serves
as model "<name>" on its next cold start. LoRA covers only language-model MLP and full-attention
projections (no vision tower, no linear-attention layers) so vLLM can load it.
"""
import modal

from src.modal_serve import MODEL_NAME, MODEL_REVISION, hf_cache_vol, lora_vol

GPU = "L40S"
MINUTES = 60
LORA_TARGETS = r".*language_model\.layers\.\d+\.(mlp\.(gate|up|down)_proj|self_attn\.(q|k|v|o)_proj)"

train_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch>=2.13", "torchvision", "transformers==5.10.4", "peft", "accelerate", "pillow")
    .add_local_python_source("src")
)

app = modal.App("chemvision-train")


@app.function(
    image=train_image,
    gpu=GPU,
    timeout=45 * MINUTES,
    volumes={"/root/.cache/huggingface": hf_cache_vol, "/loras": lora_vol},
)
def train(rows: list[dict], name: str, epochs: int, lr: float, rank: int, grad_accum: int) -> dict:
    import base64
    import io
    import json
    import random
    import time
    from pathlib import Path

    import torch
    from peft import LoraConfig, get_peft_model
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    t0 = time.time()
    processor = AutoProcessor.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    model = AutoModelForImageTextToText.from_pretrained(MODEL_NAME, revision=MODEL_REVISION, dtype=torch.bfloat16, device_map="cuda")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.05, target_modules=LORA_TARGETS))
    model.print_trainable_parameters()
    load_s = time.time() - t0

    def encode(row: dict) -> dict:
        user, assistant = row["messages"]
        image = None
        content = []
        for part in user["content"]:
            if part["type"] == "image_url":
                b64 = part["image_url"]["url"].split(",", 1)[1]
                image = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
                content.append({"type": "image"})
            else:
                content.append(part)
        # Same rendering vLLM uses at inference: generation prompt with thinking disabled.
        prompt = processor.apply_chat_template(
            [{"role": "user", "content": content}], add_generation_prompt=True, tokenize=False, enable_thinking=False
        )
        enc = dict(processor(text=[prompt], images=[image], return_tensors="pt"))
        target = processor.tokenizer(assistant["content"] + "<|im_end|>\n", add_special_tokens=False, return_tensors="pt").input_ids
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
                out[k] = v  # pixel_values, image_grid_thw
        out["labels"] = torch.cat([torch.full_like(enc["input_ids"], -100), target], dim=1)
        return out

    examples = [encode(r) for r in rows]
    print("example 0 decoded:", processor.tokenizer.decode(examples[0]["input_ids"][0][-40:]))
    print("trained tokens in example 0:", int((examples[0]["labels"] != -100).sum()))

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
    rng = random.Random(0)
    model.train()
    losses, step = [], 0
    t1 = time.time()
    for epoch in range(epochs):
        order = list(range(len(examples)))
        rng.shuffle(order)
        epoch_loss = 0.0
        for i, idx in enumerate(order):
            batch = {k: v.to("cuda") for k, v in examples[idx].items()}
            loss = model(**batch).loss
            (loss / grad_accum).backward()
            epoch_loss += loss.item()
            if (i + 1) % grad_accum == 0 or i + 1 == len(order):
                opt.step()
                opt.zero_grad()
                step += 1
        losses.append(epoch_loss / len(order))
        print(f"epoch {epoch + 1}: mean loss {losses[-1]:.4f}")
    train_s = time.time() - t1

    out = Path("/loras") / name
    model.save_pretrained(out)
    lora_vol.commit()
    files = sorted(p.name for p in out.iterdir())
    print("saved", out, files)
    print(json.dumps(json.loads((out / "adapter_config.json").read_text()), indent=1)[:800])
    return {"name": name, "epoch_losses": losses, "optimizer_steps": step, "load_s": load_s, "train_s": train_s, "files": files}


@app.local_entrypoint()
def main(data: str, name: str, epochs: int = 3, lr: float = 2e-4, rank: int = 8, grad_accum: int = 8):
    import json
    from pathlib import Path

    rows = [json.loads(line) for line in Path(data).read_text().splitlines() if line.strip()]
    print(f"training {name!r} on {len(rows)} rows from {data}")
    print(json.dumps(train.remote(rows, name, epochs, lr, rank, grad_accum), indent=2))
