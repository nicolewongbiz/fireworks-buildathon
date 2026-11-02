"""Harness and router systems on top of the tuned open model (the "smarter harness" and "route across models" lanes).

  harness  tuned model, 1 greedy + N sampled answers (one vLLM request with n=N), RDKit validity filter,
           majority vote on canonical SMILES; greedy breaks ties. Fully open, no escalation.
  routed   same candidates; if the vote winner's agreement is below ROUTE_MIN_AGREE (default 0.8) or no
           candidate is valid, escalate the image to Kimi K3 on Fireworks serverless (open weights, so the
           whole stack stays open). Agreement is the only confidence signal a small model gives for free.

Registered via src/predict_baselines.py; run with the shared runner, e.g.
  .venv/bin/python -m src.run_system routed data/test_synthetic_384.json \
      --out results/routed__test_synthetic_384.json --gpu-usd-per-hour 1.95 --workers 8
Env: ROUTE_SAMPLES (default 4), ROUTE_TEMPERATURE (0.7), ROUTE_MIN_AGREE (0.8), TUNED_LORA (see src.predict).
Per-item extras: candidates, n_valid, agreement, vote, escalated, usd (Kimi spend when escalated).
Cost note: run_system prices the tuned calls by GPU wall-clock; `usd` here carries only the escalation spend.
"""
from __future__ import annotations

import os
import time
from collections import Counter

from openai import APIStatusError

from src.predict import MAX_TOKENS, _image_message, _modal_client, best_tuned_checkpoint
from src.validate import canonicalize, extract_smiles

SAMPLES = int(os.environ.get("ROUTE_SAMPLES", "4"))
TEMPERATURE = float(os.environ.get("ROUTE_TEMPERATURE", "0.7"))
MIN_AGREE = float(os.environ.get("ROUTE_MIN_AGREE", "0.8"))


def _completions(model: str, image_path: str, n: int, temperature: float) -> tuple[list[str], int, int, float]:
    """One chat request returning n sampled answers. Waits through Modal cold-start 503s like src.predict."""
    client = _modal_client()
    for _ in range(90):
        try:
            t0 = time.perf_counter()
            resp = client.chat.completions.create(
                model=model,
                messages=_image_message(image_path),
                temperature=temperature,
                n=n,
                max_tokens=MAX_TOKENS,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            latency = time.perf_counter() - t0
            break
        except APIStatusError as e:
            if e.status_code != 503:
                raise
            time.sleep(10)
    else:
        raise TimeoutError(f"{model} endpoint stayed unavailable")
    texts = [c.message.content or "" for c in resp.choices]
    return texts, resp.usage.prompt_tokens, resp.usage.completion_tokens, latency


def vote(candidates: list[str]) -> tuple[str | None, float, int]:
    """Majority vote over canonical SMILES of the valid candidates. candidates[0] is the greedy answer and
    breaks ties. Returns (winner_canonical, agreement among valid, n_valid)."""
    canon = [canonicalize(extract_smiles(c), strip_stereo=False) for c in candidates]
    valid = [c for c in canon if c]
    if not valid:
        return None, 0.0, 0
    counts = Counter(valid)
    top = max(counts.values())
    leaders = [c for c, k in counts.items() if k == top]
    winner = canon[0] if canon[0] in leaders else leaders[0]
    return winner, top / len(valid), len(valid)


def _candidates(image_path: str, lora: str | None = None) -> dict:
    model = lora or best_tuned_checkpoint()
    greedy, tin1, tout1, lat1 = _completions(model, image_path, 1, 0.0)
    sampled, tin2, tout2, lat2 = _completions(model, image_path, SAMPLES, TEMPERATURE)
    cands = greedy + sampled
    winner, agreement, n_valid = vote(cands)
    return {
        "candidates": cands, "vote": winner, "agreement": agreement, "n_valid": n_valid,
        "tokens_in": tin1 + tin2, "tokens_out": tout1 + tout2, "latency_s": lat1 + lat2, "model": model,
    }


def _as_result(c: dict) -> dict:
    return {
        "smiles": c["vote"] or c["candidates"][0],
        "latency_s": c["latency_s"], "tokens_in": c["tokens_in"], "tokens_out": c["tokens_out"],
        "attempts": 1 + SAMPLES, "escalated": False, "usd": 0.0,
        "candidates": c["candidates"], "vote": c["vote"], "agreement": c["agreement"], "n_valid": c["n_valid"],
    }


def predict_harness(image_path: str) -> dict:
    return _as_result(_candidates(image_path))


def predict_routed(image_path: str) -> dict:
    c = _candidates(image_path)
    out = _as_result(c)
    if c["vote"] is not None and c["agreement"] >= MIN_AGREE:
        return out
    from src.predict_fireworks import SYSTEMS as FW  # lazy: needs FIREWORKS_API_KEY only when escalating

    big = FW["large_kimi"](image_path)
    out.update({
        "smiles": big["smiles"], "escalated": True, "escalation_model": big.get("model", "kimi-k3"),
        "latency_s": out["latency_s"] + big["latency_s"],
        "tokens_in": out["tokens_in"] + big["tokens_in"], "tokens_out": out["tokens_out"] + big["tokens_out"],
        "attempts": out["attempts"] + big.get("attempts", 1), "usd": big.get("usd", 0.0),
        "small_answer": c["vote"],
    })
    return out


SYSTEMS = {"harness": predict_harness, "routed": predict_routed}
