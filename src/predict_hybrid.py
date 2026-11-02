"""Hybrid system: Kimi K3 reads the skeleton, a tuned 9B "stereo fixer" adds stereochemistry, RDKit guards.

  1. Kimi K3 answer for the image (from a cached results/large_kimi__<testset>.json if the image is in one,
     otherwise a live Fireworks call). Strip stereo -> skeleton.
  2. Ask the fixer LoRA (env FIXER_LORA, served by src/modal_serve.py) for the full SMILES given image + skeleton
     (prompt from src.make_stereo_fixer.FIX_PROMPT, the same text it was trained on).
  3. Guard: the fixer's answer stripped of stereo must equal the skeleton, else keep Kimi's answer. The
     hybrid can therefore never lose the skeleton; it can only change stereo assignments.

Registered via src/predict_baselines.py as `hybrid`. Ceiling = Kimi's stereo-agnostic accuracy
(76.8% synthetic, 69% USPTO as of 2 Nov); Kimi alone is 42.6% / 56% stereo-aware.
Per-item extras: kimi_smiles, skeleton, fixer_raw, guard_passed, kimi_cached, usd (Kimi spend if live).
"""
from __future__ import annotations

import json
import os
import time
from functools import cache
from pathlib import Path

from openai import APIStatusError

from src.make_stereo_fixer import fixer_prompt
from src.predict import MAX_TOKENS, _image_message, _modal_client
from src.validate import canonicalize, extract_smiles

ROOT = Path(__file__).resolve().parents[1]


@cache
def _kimi_cache() -> dict[str, dict]:
    """image_path -> Kimi item from every results/large_kimi__*.json (so test images are not re-bought)."""
    out: dict[str, dict] = {}
    for p in sorted(ROOT.glob("results/large_kimi__*.json")):
        for it in json.loads(p.read_text())["items"]:
            out.setdefault(it["image_path"], it)
    return out


def _kimi(image_path: str) -> tuple[str, float, bool, int, int, float]:
    rel = os.path.relpath(image_path, ROOT) if os.path.isabs(image_path) else image_path
    hit = _kimi_cache().get(rel)
    if hit:
        return hit["predict"]["smiles"], 0.0, True, 0, 0, hit["predict"].get("usd", 0.0)
    from src.predict_fireworks import SYSTEMS as FW

    r = FW["large_kimi"](image_path)
    return r["smiles"], r["latency_s"], False, r["tokens_in"], r["tokens_out"], r.get("usd", 0.0)


def _fixer(image_path: str, skeleton: str) -> tuple[str, float, int, int]:
    lora = os.environ.get("FIXER_LORA")
    if not lora:
        raise RuntimeError("set FIXER_LORA to the stereo-fixer adapter name served by src/modal_serve.py")
    msgs = _image_message(image_path)
    msgs[0]["content"][0]["text"] = fixer_prompt(skeleton)
    client = _modal_client()
    for _ in range(90):
        try:
            t0 = time.perf_counter()
            resp = client.chat.completions.create(
                model=lora, messages=msgs, temperature=0, max_tokens=MAX_TOKENS,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            return resp.choices[0].message.content or "", time.perf_counter() - t0, resp.usage.prompt_tokens, resp.usage.completion_tokens
        except APIStatusError as e:
            if e.status_code != 503:
                raise
            time.sleep(10)
    raise TimeoutError(f"{lora} endpoint stayed unavailable")


def predict_hybrid(image_path: str) -> dict:
    kimi_raw, lat_k, cached, tin_k, tout_k, usd_k = _kimi(image_path)
    skeleton = canonicalize(extract_smiles(kimi_raw), strip_stereo=True)
    base = {
        "kimi_smiles": kimi_raw, "skeleton": skeleton, "kimi_cached": cached, "usd": usd_k if not cached else 0.0,
        "escalated": False, "attempts": 2,
    }
    if skeleton is None:  # Kimi gave nothing parseable; nothing to fix
        return dict(base, smiles=kimi_raw, latency_s=lat_k, tokens_in=tin_k, tokens_out=tout_k,
                    fixer_raw=None, guard_passed=False)
    fixed_raw, lat_f, tin_f, tout_f = _fixer(image_path, skeleton)
    fixed = canonicalize(extract_smiles(fixed_raw), strip_stereo=False)
    guard = fixed is not None and canonicalize(fixed, strip_stereo=True) == skeleton
    return dict(base, smiles=fixed if guard else kimi_raw, latency_s=lat_k + lat_f,
                tokens_in=tin_k + tin_f, tokens_out=tout_k + tout_f, fixer_raw=fixed_raw, guard_passed=guard)


SYSTEMS = {"hybrid": predict_hybrid}
