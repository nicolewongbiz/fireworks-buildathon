"""MolScribe (Qian et al. 2023, MIT) as a specialist OCSR reference baseline, run locally.

Uses its own environment because MolScribe pins old deps that conflict with the main .venv:

  uv venv --python 3.11 .venv-molscribe
  uv pip install --python .venv-molscribe/bin/python torch torchvision rdkit huggingface_hub pillow \
      "numpy<2" pandas matplotlib opencv-python-headless scipy "albumentations==1.1.0" "timm==0.4.12" \
      configargparse pyyaml six requests tqdm sentencepiece
  uv pip install --python .venv-molscribe/bin/python --no-deps "SmilesPE==0.0.3" "OpenNMT-py==2.2.0" "torchtext==0.5.0"
  uv pip install --python .venv-molscribe/bin/python --no-deps git+https://github.com/thomas0809/MolScribe
  # checkpoint (untrusted data, never executed): yujieq/MolScribe -> data/raw/molscribe_ckpt/swin_base_char_aux_1m.pth
  .venv-molscribe/bin/python -I -c "from huggingface_hub import hf_hub_download as d; \
      d('yujieq/MolScribe', 'swin_base_char_aux_1m.pth', local_dir='data/raw/molscribe_ckpt')"

Score the smoke set:  .venv-molscribe/bin/python -m src.predict_molscribe [test_set] [--out results/...]

Same `predict(image_path)` contract as src/predict.py and the same `{"summary", "items"}` output shape as
src/run_system.py. Token fields are 0 (not an LLM); gpu_hours/usd are 0 (runs on the laptop).
Device: CPU by default (~0.9 s/image on an M5, 4 threads); `MOLSCRIBE_DEVICE=mps` is no faster because the
autoregressive decoder is latency-bound, and it pays a ~5 s first-call warm-up.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
import warnings
from dataclasses import asdict
from pathlib import Path

CKPT = Path(os.environ.get("MOLSCRIBE_CKPT", "data/raw/molscribe_ckpt/swin_base_char_aux_1m.pth"))
DEVICE = os.environ.get("MOLSCRIBE_DEVICE", "cpu")  # "cpu" | "mps"
NUM_THREADS = int(os.environ.get("MOLSCRIBE_THREADS", "4"))  # CLAUDE.md: keep the laptop healthy

_model = None


def _load():
    global _model
    if _model is None:
        warnings.filterwarnings("ignore")  # old timm/onmt emit deprecation noise under torch 2.x
        import torch
        from molscribe import MolScribe

        torch.set_num_threads(NUM_THREADS)
        if DEVICE == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MOLSCRIBE_DEVICE=mps but torch.backends.mps is unavailable")
        if not CKPT.exists():
            raise FileNotFoundError(f"MolScribe checkpoint missing: {CKPT} (see module docstring for the download)")
        _model = MolScribe(str(CKPT), device=torch.device(DEVICE))
    return _model


def predict(image_path: str) -> dict:
    """MolScribe prediction in the shared predict() contract. Latency covers the full image -> SMILES call."""
    model = _load()
    t0 = time.perf_counter()
    try:
        out = model.predict_image_file(image_path, return_atoms_bonds=False, return_confidence=True)
        smiles = out.get("smiles") or ""
        confidence = out.get("confidence")
    except Exception as e:  # noqa: BLE001 - a failed image is a wrong answer, not a crashed run
        smiles, confidence = "", None
        print(f"MolScribe failed on {image_path}: {type(e).__name__}: {e}")
    return {"smiles": smiles, "latency_s": time.perf_counter() - t0, "tokens_in": 0, "tokens_out": 0,
            "attempts": 1, "escalated": False, "confidence": confidence}


SYSTEMS = {"molscribe": predict}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("test_set", nargs="?", default="data/smoke_test.json")
    ap.add_argument("--out", default=None, help="default results/<test_set stem with 'test'->'molscribe'>.json")
    args = ap.parse_args()
    out_path = Path(args.out or f"results/{Path(args.test_set).stem.replace('test', 'molscribe')}.json")

    from src.validate import canonicalize, score

    items = json.loads(Path(args.test_set).read_text())
    t_start = time.perf_counter()
    _load()
    load_s = time.perf_counter() - t_start
    first_answer_s = None
    rows = []
    for it in items:
        out = predict(it["image_path"])
        if first_answer_s is None:
            first_answer_s = time.perf_counter() - t_start
        m = score(out["smiles"] or "", it["smiles"])
        exact_stereo = m.valid and canonicalize(m.pred_smiles, strip_stereo=False) == canonicalize(it["smiles"], strip_stereo=False)
        print(f"{'OK ' if m.exact else 'BAD'} valid={m.valid} {out['latency_s']:.2f}s {out['smiles']!r}")
        rows.append({"image_path": it["image_path"], "truth": it["smiles"], "tags": it.get("tags"), "predict": out,
                     "match": asdict(m) | {"exact_stereo": exact_stereo, "hit_cap": False}})
    wall_s = time.perf_counter() - t_start

    n = len(rows)
    lat = sorted(r["predict"]["latency_s"] for r in rows)
    summary = {
        "system": "molscribe",
        "n": n,
        "exact_match": sum(r["match"]["exact_stereo"] for r in rows) / n,
        "exact_match_nostereo": sum(r["match"]["exact"] for r in rows) / n,
        "valid_rate": sum(r["match"]["valid"] for r in rows) / n,
        "mean_tanimoto": sum(r["match"]["tanimoto"] for r in rows) / n,
        "loop_rate": 0.0,  # no token cap: MolScribe decodes a fixed-length atom/bond graph, not free text
        "max_tokens": None,
        "mean_latency_s": sum(lat) / n,
        "median_latency_s": statistics.median(lat),
        "p95_latency_s": lat[min(n - 1, math.ceil(0.95 * n) - 1)],
        "mean_tokens_in": 0.0,
        "mean_tokens_out": 0.0,
        "first_answer_s": first_answer_s,
        "model_load_s": load_s,
        "wall_s": wall_s,
        "gpu_hours": 0.0,
        "usd": 0.0,
        "gpu_usd_per_hour": 0.0,
        "workers": 1,
        "device": DEVICE,
        "threads": NUM_THREADS,
        "checkpoint": str(CKPT),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"summary": summary, "items": rows}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
