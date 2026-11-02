"""OpenAI-compatible vLLM server on Modal for Qwen3-VL-8B, plus any LoRAs in the `chemvision-loras` volume.

Person A owns this. Usage:
  uv run modal run src/modal_serve.py::download_model   # one-time, CPU only, caches weights
  modal volume put chemvision-loras <local_lora_dir> /<lora_name>   # optional, adds a LoRA
  uv run modal deploy src/modal_serve.py                # endpoint; scales to zero when idle
  uv run modal app stop chemvision-vllm                 # hard stop when done

Requests need `Authorization: Bearer $MODAL_VLLM_API_KEY` (Modal secret `chemvision-vllm-key`).
Model names: "base" for the untuned model, and each LoRA's directory name in the volume.
"""
import modal

MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
MODEL_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
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
            "--enforce-eager",  # faster cold start; throughput doesn't matter for small eval runs
        ]
        if loras:
            cmd += ["--enable-lora", "--max-lora-rank", "64", "--lora-modules"]
            cmd += [f"{p.name}={p}" for p in loras]
        print("LoRAs:", [p.name for p in loras])
        print(*cmd[:8], "...")
        self.process = subprocess.Popen(cmd)

    @modal.exit()
    def stop(self):
        self.process.terminate()
