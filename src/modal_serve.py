"""OpenAI-compatible vLLM server on Modal for Qwen3.5-9B, plus any LoRAs in the `chemvision-loras` volume.

Person A owns this. Usage:
  uv run modal run src/modal_serve.py::download_model   # one-time, CPU only, caches weights
  modal volume put chemvision-loras <local_lora_dir> /<lora_name>   # optional, adds a LoRA
  uv run modal deploy src/modal_serve.py                # endpoint; scales to zero when idle
  uv run modal app stop chemvision-vllm                 # hard stop when done

Requests need `Authorization: Bearer $MODAL_VLLM_API_KEY` (Modal secret `chemvision-vllm-key`).
Model names: "base" for the untuned model, and each LoRA's directory name in the volume.
"""
import modal

MODEL_NAME = "Qwen/Qwen3.5-9B"  # Fireworks `qwen3p5-9b`; Qwen3-VL-8B is not trainable on Fireworks
MODEL_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
GPU = "L40S"
VLLM_PORT = 8000
MINUTES = 60

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.21.0")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)
download_image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("huggingface_hub[hf_xet]")

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)
lora_vol = modal.Volume.from_name("chemvision-loras", create_if_missing=True)

app = modal.App("chemvision-vllm")


@app.function(image=download_image, volumes={"/root/.cache/huggingface": hf_cache_vol}, timeout=30 * MINUTES)
def download_model():
    """CPU-only: pull weights into the volume so GPU containers never pay for the download."""
    from huggingface_hub import snapshot_download

    path = snapshot_download(MODEL_NAME, revision=MODEL_REVISION)
    hf_cache_vol.commit()
    print("cached at", path)


@app.server(
    image=vllm_image,
    gpu=GPU,
    scaledown_window=5 * MINUTES,
    startup_timeout=15 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
        "/loras": lora_vol,
    },
    secrets=[modal.Secret.from_name("chemvision-vllm-key")],
    port=VLLM_PORT,
    max_containers=1,
    unauthenticated=True,  # Modal proxy is open; vLLM itself enforces --api-key
)
class Server:
    @modal.enter()
    def start(self):
        import os
        import subprocess
        from pathlib import Path

        loras = sorted(p for p in Path("/loras").iterdir() if (p / "adapter_config.json").exists())
        cmd = [
            "vllm", "serve", MODEL_NAME,
            "--revision", MODEL_REVISION,
            "--served-model-name", "base",
            "--host", "0.0.0.0",
            "--port", str(VLLM_PORT),
            "--api-key", os.environ["MODAL_VLLM_API_KEY"],
            "--max-model-len", "4096",
            "--limit-mm-per-prompt", '{"image": 1, "video": 0}',
            # CUDA graphs on (no --enforce-eager): eager mode cost ~1 s/request with a LoRA. Compile
            # artifacts are cached in the vllm-cache volume, so only the first cold start pays for it.
        ]
        if loras:
            import json

            cfgs = [json.loads((p / "adapter_config.json").read_text()) for p in loras]
            # Kernels are sized by --max-lora-rank, so use the smallest vLLM-allowed rank that fits.
            max_r = max(c["r"] for c in cfgs)
            rank = next(r for r in (8, 16, 32, 64, 128, 256) if r >= max_r)
            cmd += ["--enable-lora", "--max-lora-rank", str(rank), "--lora-modules"]
            cmd += [f"{p.name}={p}" for p in loras]
            if any("visual" in str(c.get("target_modules")) for c in cfgs):
                cmd += ["--enable-tower-connector-lora"]  # otherwise vLLM silently drops visual.* LoRA weights
        print("LoRAs:", [p.name for p in loras])
        print(*cmd[:8], "...")
        self.process = subprocess.Popen(cmd)

    @modal.exit()
    def stop(self):
        self.process.terminate()
