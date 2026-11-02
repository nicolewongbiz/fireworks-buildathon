"""Closed frontier baseline: Claude via the Anthropic API, same predict() contract as src/predict.py.

The buildathon brief is "beat a closed frontier model with open models", so this is the opponent.
Registered through src/predict_baselines.py as `closed_opus`; run with the shared runner:

  .venv/bin/python -m src.run_system closed_opus data/smoke_test.json \
      --out results/closed_opus__smoke.json --gpu-usd-per-hour 0

Credentials: ANTHROPIC_API_KEY in .env (or an `ant auth login` profile). Never printed.
Overrides: CLOSED_MODEL (default claude-opus-5-5), CLOSED_EFFORT (default "low").

Comparability notes, recorded per item:
- Claude Opus 5.5 always thinks; effort is the only depth control. We use effort "low" as the closest
  analogue to the open models' thinking-off setting, and max_tokens 2048 so thinking is not truncated
  (the shared 108-token cap cannot be applied; `tokens_out` includes thinking tokens, so it is not
  directly comparable to the vLLM runs' output counts).
- Sampling params are not accepted on this model, so there is no temperature=0; runs are not
  bit-reproducible. Report run-to-run noise on a repeat.
- Server-side refusal fallback ("default") is enabled so a safety decline on a benign image still
  yields an answer from the fallback model; `fallback_ran` is recorded when that happens.
"""
from __future__ import annotations

import base64
import mimetypes
import os
import time
from functools import cache
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from src.predict import PROMPT

load_dotenv()

DEFAULT_MODEL = "claude-opus-5-5"
MAX_TOKENS = 2048
# USD per 1M tokens (Anthropic first-party rates, 2026-10): model -> (input, output)
PRICES = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}


@cache
def _client() -> anthropic.Anthropic:
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        # A zero-arg client also works after `ant auth login`; let the SDK try before failing loudly.
        pass
    return anthropic.Anthropic(max_retries=3)


def _image_block(image_path: str) -> dict:
    mime = mimetypes.guess_type(image_path)[0] or "image/png"
    data = base64.standard_b64encode(Path(image_path).read_bytes()).decode("utf-8")
    return {"type": "image", "source": {"type": "base64", "media_type": mime, "data": data}}


def _cost(model: str, usage) -> float:
    pin, pout = PRICES.get(model, PRICES[DEFAULT_MODEL])
    uncached = usage.input_tokens + (usage.cache_creation_input_tokens or 0)
    cached = usage.cache_read_input_tokens or 0
    return (uncached * pin + cached * pin * 0.05 + usage.output_tokens * pout) / 1e6


def predict(image_path: str) -> dict:
    model = os.environ.get("CLOSED_MODEL", DEFAULT_MODEL)
    effort = os.environ.get("CLOSED_EFFORT", "low")
    client = _client()
    t0 = time.perf_counter()
    try:
        resp = client.beta.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            output_config={"effort": effort},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            system="You are an expert chemist. Answer with the SMILES string only: no prose, no markdown.",
            messages=[{"role": "user", "content": [_image_block(image_path), {"type": "text", "text": PROMPT}]}],
        )
    except anthropic.AuthenticationError as e:
        raise SystemExit("Anthropic auth failed: set ANTHROPIC_API_KEY in .env or run `ant auth login`") from e
    except anthropic.RateLimitError:
        raise  # SDK already retried with backoff; let the runner see it
    latency = time.perf_counter() - t0

    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    fallback_ran = any(getattr(e, "type", None) == "fallback_message" for e in (resp.usage.iterations or []))
    refused = resp.stop_reason == "refusal"
    return {
        "smiles": "" if refused else text,
        "latency_s": latency,
        "tokens_in": resp.usage.input_tokens,
        "tokens_out": resp.usage.output_tokens,  # includes thinking tokens
        "attempts": 1,
        "escalated": False,
        "usd": _cost(resp.model, resp.usage),
        "model": resp.model,
        "effort": effort,
        "stop_reason": resp.stop_reason,
        "refusal_category": (resp.stop_details.category if refused and resp.stop_details else None),
        "fallback_ran": fallback_ran,
    }


SYSTEMS = {"closed_opus": predict}
