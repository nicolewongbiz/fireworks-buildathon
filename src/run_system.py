"""Run one predictor over a test set and write results/<system>.json.

  uv run python -m src.run_system untuned data/smoke_test.json --out results/smoke_untuned.json \
      --gpu-usd-per-hour 1.95

Cost is GPU wall-clock for the run times the hourly rate; cold start is recorded separately
(time to the first answer) so it can be added or excluded explicitly.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from src.predict import SYSTEMS
from src.validate import score


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("system", choices=SYSTEMS)
    ap.add_argument("test_set")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu-usd-per-hour", type=float, required=True)
    args = ap.parse_args()

    items = json.loads(Path(args.test_set).read_text())
    predict = SYSTEMS[args.system]
    t_start = time.perf_counter()
    first_answer_s = None
    rows = []
    for it in items:
        out = predict(it["image_path"])
        if first_answer_s is None:
            first_answer_s = time.perf_counter() - t_start
        m = score(out["smiles"] or "", it["smiles"])
        rows.append({"image_path": it["image_path"], "truth": it["smiles"], "predict": out, "match": asdict(m)})
        print(f"{'OK ' if m.exact else 'BAD'} valid={m.valid} {out['latency_s']:.2f}s {out['smiles']!r}")
    wall_s = time.perf_counter() - t_start

    n = len(rows)
    gpu_hours = wall_s / 3600
    summary = {
        "system": args.system,
        "n": n,
        "exact_match": sum(r["match"]["exact"] for r in rows) / n,
        "valid_rate": sum(r["match"]["valid"] for r in rows) / n,
        "mean_latency_s": sum(r["predict"]["latency_s"] for r in rows) / n,
        "mean_tokens_in": sum(r["predict"]["tokens_in"] for r in rows) / n,
        "mean_tokens_out": sum(r["predict"]["tokens_out"] for r in rows) / n,
        "first_answer_s": first_answer_s,
        "wall_s": wall_s,
        "gpu_hours": gpu_hours,
        "usd": gpu_hours * args.gpu_usd_per_hour,
        "gpu_usd_per_hour": args.gpu_usd_per_hour,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"summary": summary, "items": rows}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
