"""Large hosted-VLM baselines on Fireworks serverless, same predict() contract as src/predict.py.

  uv run python -m src.predict_fireworks large_glm  [data/smoke_test.json] [--out results/smoke_large_glm.json] [--limit 10]
  uv run python -m src.predict_fireworks large_kimi [data/smoke_test.json] [--out results/smoke_large_kimi.json] [--limit 10]
  uv run python -m src.predict_fireworks large_qwen [data/smoke_test.json] [--out results/smoke_large_qwen.json] [--limit 10]

Systems: `large_kimi` (kimi-k3) and `large_qwen` (qwen3p8-max) are the usable large baselines.
`large_glm` (glm-5p3-flash) is kept for the record but is UNUSABLE at max_tokens 108: GLM-5.3 is a
thinking-only model (every thinking-off flag is rejected with HTTP 400; with an image attached the server
reports it as the misleading "Invalid image or video input"), and even reasoning_effort=low/minimal spends
the whole 108-token budget on reasoning, so content is always None (smoke, 1 Nov 2026: 0/10 valid).

Each predict(image_path) returns {smiles, latency_s, tokens_in, tokens_out, attempts, escalated} plus
`usd` (token spend at the model's serverless list price), `model`, and `thinking_flag` (which request
parameter actually disabled thinking for that model). The runner mirrors src/run_system.py's output
`{"summary": {...}, "items": [...]}` and adds `usd` / `usd_per_image` from tokens.

Thinking is disabled with `reasoning_effort: "none"` (docs/fireworks-notes.md sec. 10). If a model rejects
that with HTTP 400, the module falls back to `chat_template_kwargs: {"enable_thinking": false}` and
remembers which one worked per model. One retry on 429/503 with backoff (honours Retry-After).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from functools import cache
from pathlib import Path

from dotenv import load_dotenv
from openai import APIStatusError, OpenAI

from src.predict import MAX_TOKENS, _image_message  # same prompt and image data-URI message format
from src.validate import canonicalize, score

load_dotenv()

BASE_URL = "https://api.fireworks.ai/inference/v1"
GLM_FLASH = "accounts/fireworks/models/glm-5p3-flash"
KIMI_K3 = "accounts/fireworks/models/kimi-k3"
QWEN_MAX = "accounts/fireworks/models/qwen3p8-max"

# Serverless list prices, USD per 1M tokens (input, output). docs/fireworks-notes.md sec. 5/6, 1 Nov 2026.
PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    GLM_FLASH: (0.15, 0.50),
    KIMI_K3: (3.00, 15.00),
    QWEN_MAX: (2.00, 6.00),
}

# Ordered candidates for turning thinking off; the first not rejected with HTTP 400 is kept per model.
# The last entry sends no flag at all (thinking stays ON) so a thinking-only model still produces a
# measurable result instead of crashing the run; `thinking_flag` in the output records which one won.
# Note: when an image is attached, Fireworks reports a rejected reasoning flag as the misleading
# "Invalid image or video input" (seen on glm-5p3-flash, 1 Nov 2026), so any 400 advances to the next flag.
THINKING_ON = "none (thinking ON; model rejected every thinking-off flag)"
_THINKING_OFF_FLAGS: list[tuple[str, dict]] = [
    ("reasoning_effort=none", {"reasoning_effort": "none"}),
    ("chat_template_kwargs.enable_thinking=false", {"chat_template_kwargs": {"enable_thinking": False}}),
    (THINKING_ON, {}),
]
THINKING_FLAG_USED: dict[str, str] = {}  # model id -> flag label that worked
_RETRY_STATUSES = {429, 503}
_RETRY_DEFAULT_WAIT_S = 5.0


@cache
def _client() -> OpenAI:
    key = os.environ.get("FIREWORKS_API_KEY", "").strip()
    if not key:
        raise RuntimeError("FIREWORKS_API_KEY is empty or missing in .env; refusing to run.")
    return OpenAI(base_url=BASE_URL, api_key=key)


def usd_for(model: str, tokens_in: int, tokens_out: int) -> float:
    p_in, p_out = PRICES_PER_MTOK[model]
    return (tokens_in * p_in + tokens_out * p_out) / 1e6


def _retry_wait(e: APIStatusError) -> float:
    try:
        ra = e.response.headers.get("retry-after")
        if ra:
            return min(float(ra), 60.0)
    except Exception:  # noqa: BLE001
        pass
    return _RETRY_DEFAULT_WAIT_S


def _error_text(e: APIStatusError) -> str:
    try:
        return (e.response.text or "")[:200]
    except Exception:  # noqa: BLE001
        return str(e)[:200]


def _chat(model: str, image_path: str, escalated: bool = False) -> dict:
    client = _client()
    messages = _image_message(image_path)
    # Candidate thinking-off flags: if one already worked for this model, only use that one.
    if model in THINKING_FLAG_USED:
        flags = [f for f in _THINKING_OFF_FLAGS if f[0] == THINKING_FLAG_USED[model]]
    else:
        flags = list(_THINKING_OFF_FLAGS)

    attempts = 0
    resp = None
    flag_label = None
    last_400 = None
    for flag_label, extra_body in flags:
        retried = False
        while True:
            attempts += 1
            t0 = time.perf_counter()
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0,
                    max_tokens=MAX_TOKENS,
                    extra_body=extra_body,
                )
                latency = time.perf_counter() - t0
                break
            except APIStatusError as e:
                if e.status_code == 400 and model not in THINKING_FLAG_USED and flag_label != THINKING_ON:
                    # This model rejects this thinking-off flag; try the next candidate.
                    last_400 = _error_text(e)
                    print(f"  [{model.rsplit('/', 1)[-1]}] 400 with {flag_label}: {last_400!r} -> trying next flag")
                    resp = None
                    break
                if e.status_code in _RETRY_STATUSES and not retried:
                    wait = _retry_wait(e)
                    print(f"  [{model.rsplit('/', 1)[-1]}] HTTP {e.status_code}, retrying once in {wait:.0f}s")
                    time.sleep(wait)
                    retried = True
                    continue
                raise
        if resp is not None:
            THINKING_FLAG_USED.setdefault(model, flag_label)
            break
    if resp is None:
        raise RuntimeError(f"{model}: request rejected with HTTP 400 even without a thinking flag: {last_400!r}")

    msg = resp.choices[0].message
    reasoning = getattr(msg, "reasoning_content", None) or (msg.model_extra or {}).get("reasoning_content")
    usage = resp.usage
    tokens_in = usage.prompt_tokens if usage else 0
    tokens_out = usage.completion_tokens if usage else 0
    return {
        "smiles": msg.content,
        "latency_s": latency,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "attempts": attempts,
        "escalated": escalated,
        "usd": usd_for(model, tokens_in, tokens_out),
        "model": model,
        "thinking_flag": THINKING_FLAG_USED[model],
        "finish_reason": resp.choices[0].finish_reason,
        "reasoning_chars": len(reasoning) if reasoning else 0,  # >0 means thinking was NOT fully disabled
    }


def predict_large_glm(image_path: str) -> dict:
    return _chat(GLM_FLASH, image_path)


def predict_large_kimi(image_path: str) -> dict:
    return _chat(KIMI_K3, image_path)


def predict_large_qwen(image_path: str) -> dict:
    return _chat(QWEN_MAX, image_path)


SYSTEMS = {"large_glm": predict_large_glm, "large_kimi": predict_large_kimi, "large_qwen": predict_large_qwen}
_MODEL_FOR = {"large_glm": GLM_FLASH, "large_kimi": KIMI_K3, "large_qwen": QWEN_MAX}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("system", choices=SYSTEMS)
    ap.add_argument("test_set", nargs="?", default="data/smoke_test.json")
    ap.add_argument("--out", default=None, help="default results/smoke_<system>.json")
    ap.add_argument("--limit", type=int, default=10, help="max images (smoke cap; default 10)")
    args = ap.parse_args()
    out_path = Path(args.out or f"results/smoke_{args.system}.json")

    _client()  # fail fast on a missing key before touching any image
    items = json.loads(Path(args.test_set).read_text())[: args.limit]
    predict = SYSTEMS[args.system]
    model = _MODEL_FOR[args.system]
    print(f"{args.system}: {model} on {len(items)} items from {args.test_set} -> {out_path}")

    t_start = time.perf_counter()
    first_answer_s = None
    rows = []
    for i, it in enumerate(items):
        out = predict(it["image_path"])
        if first_answer_s is None:
            first_answer_s = time.perf_counter() - t_start
        m = score(out["smiles"] or "", it["smiles"])
        exact_stereo = m.valid and canonicalize(m.pred_smiles, strip_stereo=False) == canonicalize(it["smiles"], strip_stereo=False)
        hit_cap = out["tokens_out"] >= MAX_TOKENS
        print(
            f"[{i:02d}] {'OK ' if m.exact else 'BAD'} valid={m.valid!s:5} exact={m.exact!s:5} "
            f"{out['latency_s']:.2f}s tok={out['tokens_in']}+{out['tokens_out']} "
            f"usd={out['usd']:.5f} attempts={out['attempts']} cap={hit_cap} {out['smiles']!r}"
        )
        rows.append({
            "image_path": it["image_path"], "truth": it["smiles"], "tags": it.get("tags"), "predict": out,
            "match": {
                "pred_raw": m.pred_raw, "pred_smiles": m.pred_smiles, "pred_canonical": m.pred_canonical,
                "truth_canonical": m.truth_canonical, "valid": m.valid, "exact": m.exact, "tanimoto": m.tanimoto,
                "exact_stereo": exact_stereo, "hit_cap": hit_cap,
            },
        })
    wall_s = time.perf_counter() - t_start

    n = len(rows)
    lat = sorted(r["predict"]["latency_s"] for r in rows)
    tok_in = sum(r["predict"]["tokens_in"] for r in rows)
    tok_out = sum(r["predict"]["tokens_out"] for r in rows)
    usd = usd_for(model, tok_in, tok_out)
    summary = {
        "system": args.system,
        "model": model,
        "thinking_flag": THINKING_FLAG_USED.get(model),
        "n": n,
        "exact_match": sum(r["match"]["exact_stereo"] for r in rows) / n,  # stereo-aware (headline)
        "exact_match_nostereo": sum(r["match"]["exact"] for r in rows) / n,  # stereo-agnostic
        "valid_rate": sum(r["match"]["valid"] for r in rows) / n,
        "mean_tanimoto": sum(r["match"]["tanimoto"] for r in rows) / n,
        "loop_rate": sum(r["match"]["hit_cap"] for r in rows) / n,  # outputs that hit max_tokens
        "max_tokens": MAX_TOKENS,
        "mean_latency_s": sum(lat) / n,
        "median_latency_s": statistics.median(lat),
        "p95_latency_s": lat[min(n - 1, math.ceil(0.95 * n) - 1)],
        "mean_tokens_in": tok_in / n,
        "mean_tokens_out": tok_out / n,
        "total_tokens_in": tok_in,
        "total_tokens_out": tok_out,
        "first_answer_s": first_answer_s,
        "wall_s": wall_s,
        "gpu_hours": 0.0,  # serverless: no GPU wall-clock cost
        "gpu_usd_per_hour": 0.0,
        "usd_per_mtok_in": PRICES_PER_MTOK[model][0],
        "usd_per_mtok_out": PRICES_PER_MTOK[model][1],
        "usd": usd,
        "usd_per_image": usd / n,
        "retries": sum(r["predict"]["attempts"] - 1 for r in rows),
        "reasoning_leaked": sum(r["predict"]["reasoning_chars"] > 0 for r in rows),
        "workers": 1,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"summary": summary, "items": rows}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
