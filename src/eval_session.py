"""One scoring session: deploy the vLLM server with exactly the named adapters, run every job back to back on the
warm server, then always stop the app and confirm nothing is left running.

  uv run python -m src.eval_session --serve n2000_r384_lang_e3_ep2,n2000_r384_lang_e3_ep3 \
      --job tuned:n2000_r384_lang_e3_ep2,data/val_384.json \
      --job untuned,data/test_synthetic_384.json [--workers 8] [--gpu-usd-per-hour 1.95]

Each job writes results/<system>__<testset>.json (system "tuned:X" is saved as "X"). A job may add a third field
to override its worker count, e.g. `--job tuned:X,data/test_synthetic_384.json,32`.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

APP = "chemvision-vllm"


def sh(*cmd: str, env: dict | None = None, check: bool = True) -> str:
    out = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if check and out.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed:\n{out.stdout[-2000:]}\n{out.stderr[-2000:]}")
    return out.stdout


def running_apps() -> list[tuple[str, str, str]]:
    apps = json.loads(sh("uv", "run", "--quiet", "modal", "app", "list", "--json"))
    return [(a["app_id"], a["description"], a["state"]) for a in apps if a["state"] != "stopped"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", default="", help="comma-separated adapter names (SERVE_LORAS); empty = every adapter in the volume")
    ap.add_argument("--job", action="append", required=True, help="system,test_manifest[,workers]")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--gpu-usd-per-hour", type=float, default=1.95)
    args = ap.parse_args()

    t0 = time.time()
    env = os.environ | {"SERVE_LORAS": args.serve}
    print(f"[{time.strftime('%H:%M:%S')}] deploying {APP} with SERVE_LORAS={args.serve or '(all)'}", flush=True)
    sh("uv", "run", "--quiet", "modal", "deploy", "src/modal_serve.py", env=env)
    done = []
    try:
        for job in args.job:
            parts = job.split(",")
            system, manifest = parts[0], parts[1]
            workers = int(parts[2]) if len(parts) > 2 else args.workers
            name = system.split(":", 1)[1] if system.startswith("tuned:") else system
            suffix = f"_w{workers}" if len(parts) > 2 else ""
            out = f"results/{name}{suffix}__{Path(manifest).stem}.json"
            print(f"[{time.strftime('%H:%M:%S')}] {system} on {manifest} ({workers} workers) -> {out}", flush=True)
            log = Path("results/raw") / f"eval_{name}{suffix}__{Path(manifest).stem}.log"
            with log.open("w") as f:
                rc = subprocess.run([sys.executable, "-m", "src.run_system", system, manifest, "--out", out,
                                     "--gpu-usd-per-hour", str(args.gpu_usd_per_hour), "--workers", str(workers)],
                                    stdout=f, stderr=subprocess.STDOUT).returncode
            if rc != 0:
                raise RuntimeError(f"job {job} failed; see {log}")
            s = json.loads(Path(out).read_text())["summary"]
            done.append(out)
            print(f"    exact {s['exact_match']:.3f} | nostereo {s['exact_match_nostereo']:.3f} | valid "
                  f"{s['valid_rate']:.3f} | loops {s['loop_rate']:.3f} | median {s['median_latency_s']:.2f}s | "
                  f"cold {s['cold_start_s']:.0f}s | steady ${s['steady_usd_per_1k_images']:.3f}/1k", flush=True)
    finally:
        sh("uv", "run", "--quiet", "modal", "app", "stop", APP, "--yes", check=False)
        for _ in range(30):
            live = [a for a in running_apps() if a[2] != "stopping..."]
            if not any(desc == APP for _, desc, _ in live):
                break
            time.sleep(10)
        print(f"[{time.strftime('%H:%M:%S')}] stopped {APP}; session wall {(time.time() - t0) / 60:.1f} min; "
              f"apps not fully stopped: {running_apps() or 'none'}", flush=True)
    print("results:", *done, sep="\n  ")


if __name__ == "__main__":
    main()
