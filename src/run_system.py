"""Run one predictor over a test set and write results/<system>.json.

  uv run python -m src.run_system untuned data/smoke_test.json --out results/smoke_untuned.json \
      --gpu-usd-per-hour 1.95 [--workers 8]

`--workers` runs requests concurrently (default 1). Per-item latency is still measured per request.

Systems come from src/predict.py (open models, Person A) plus the optional src/predict_baselines.py
(closed models / MolScribe, Person B). Cost is GPU wall-clock times the hourly rate plus API tokens times
--usd-per-mtok-in/out; cold start is recorded separately
(time to the first answer) so it can be added or excluded explicitly.
"""
from __future__ import annotations

import argparse
import functools
import json
import math
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from src.predict import MAX_TOKENS, SYSTEMS

try:  # Person B's closed-model / MolScribe predictors, same predict() contract
    from src.predict_baselines import SYSTEMS as BASELINE_SYSTEMS

    SYSTEMS = SYSTEMS | BASELINE_SYSTEMS
except ModuleNotFoundError as e:
    if e.name != "src.predict_baselines":
        raise  # a missing package inside predict_baselines.py should fail loudly, not hide its systems
from src.validate import canonicalize, score


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("system", help=f"one of {sorted(SYSTEMS)}, or tuned:<lora_run_name> for any adapter on the server")
    ap.add_argument("test_set")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu-usd-per-hour", type=float, required=True, help="0 for API models")
    ap.add_argument("--usd-per-mtok-in", type=float, default=0.0, help="API input price per 1M tokens")
    ap.add_argument("--usd-per-mtok-out", type=float, default=0.0, help="API output price per 1M tokens (incl. reasoning)")
    ap.add_argument("--workers", type=int, default=1, help="concurrent requests (default 1)")
    args = ap.parse_args()

    items = json.loads(Path(args.test_set).read_text())
    if args.system.startswith("tuned:"):
        predict = functools.partial(SYSTEMS["tuned"], lora=args.system.split(":", 1)[1])
    elif args.system in SYSTEMS:
        predict = SYSTEMS[args.system]
    else:
        ap.error(f"unknown system {args.system!r}")
    t_start = time.perf_counter()
    first_answer_s = None
    rows = []

    def run_one(it: dict) -> dict:
        nonlocal first_answer_s
        out = predict(it["image_path"])
        if first_answer_s is None:
            first_answer_s = time.perf_counter() - t_start
        m = score(out["smiles"] or "", it["smiles"])
        # score() compares stereo-stripped forms; labels now carry stereo, so also check the isomeric form.
        exact_stereo = m.valid and canonicalize(m.pred_smiles, strip_stereo=False) == canonicalize(it["smiles"], strip_stereo=False)
        print(f"{'OK ' if m.exact else 'BAD'} valid={m.valid} {out['latency_s']:.2f}s {out['smiles']!r}")
        return {"image_path": it["image_path"], "truth": it["smiles"], "tags": it.get("tags"), "predict": out,
                "match": asdict(m) | {"exact_stereo": exact_stereo, "hit_cap": out["tokens_out"] >= MAX_TOKENS}}

    if args.workers > 1:
        # Warm the server with one request first so the cold start is not counted against every worker.
        rows.append(run_one(items[0]))
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            rows.extend(ex.map(run_one, items[1:]))
    else:
        rows.extend(run_one(it) for it in items)
    wall_s = time.perf_counter() - t_start

    n = len(rows)
    lat = sorted(r["predict"]["latency_s"] for r in rows)
    gpu_hours = wall_s / 3600
    summary = {
        "system": args.system,
        "n": n,
        "exact_match": sum(r["match"]["exact_stereo"] for r in rows) / n,  # stereo-aware (headline)
        "exact_match_nostereo": sum(r["match"]["exact"] for r in rows) / n,  # stereo-agnostic
        "valid_rate": sum(r["match"]["valid"] for r in rows) / n,
        "mean_tanimoto": sum(r["match"]["tanimoto"] for r in rows) / n,
        "loop_rate": sum(r["match"]["hit_cap"] for r in rows) / n,  # outputs that hit max_tokens
        "max_tokens": MAX_TOKENS,
        "mean_latency_s": sum(r["predict"]["latency_s"] for r in rows) / n,
        "median_latency_s": statistics.median(lat),
        "p95_latency_s": lat[min(n - 1, math.ceil(0.95 * n) - 1)],
        "mean_tokens_in": sum(r["predict"]["tokens_in"] for r in rows) / n,
        "mean_tokens_out": sum(r["predict"]["tokens_out"] for r in rows) / n,
        "first_answer_s": first_answer_s,
        "wall_s": wall_s,
        "gpu_hours": gpu_hours,
        "usd": gpu_hours * args.gpu_usd_per_hour
        + (sum(r["predict"]["tokens_in"] for r in rows) * args.usd_per_mtok_in
           + sum(r["predict"]["tokens_out"] for r in rows) * args.usd_per_mtok_out) / 1e6,
        "gpu_usd_per_hour": args.gpu_usd_per_hour,
        "workers": args.workers,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"summary": summary, "items": rows}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
