"""Learning curve: exact match vs training-set size on the frozen 500-image test set.

  uv run python -m src.learning_curve

Reads results/untuned__test_synthetic_384.json and results/n*_r384_lang_e3__test_synthetic_384.json,
plus training logs results/raw/train_n*_r384_lang_e3.log for train/val loss. Writes
results/learning_curve.png and results/LEARNING_CURVE.md. 95% intervals are Wilson score intervals.
"""
from __future__ import annotations

import ast
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
TEST = "test_synthetic_384"
BLUE, ORANGE = "#2a78d6", "#eb6834"  # dataviz reference palette slots 1-2 (validated, light mode)
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def train_history(n: int) -> dict:
    log = RES / f"raw/train_n{n}_r384_lang_e3.log"
    if not log.exists():
        return {}
    steps = [ast.literal_eval(line) for line in log.read_text().splitlines() if line.startswith("{'step'")]
    if not steps:
        return {}
    vals = [s for s in steps if s.get("val_loss") is not None]
    best = min(vals, key=lambda s: s["val_loss"])
    total = re.search(r'"total_s": ([\d.]+)', log.read_text())
    return {
        "final_train_loss": steps[-1]["train_loss"],
        "final_val_loss": steps[-1]["val_loss"],
        "best_val_loss": best["val_loss"],
        "best_val_epoch": best["epoch"],
        "train_s": float(total.group(1)) if total else None,
    }


def main() -> None:
    rows = []
    files = [RES / f"untuned__{TEST}.json"] + sorted(
        RES.glob(f"n*_r384_lang_e3__{TEST}.json"), key=lambda p: int(re.match(r"n(\d+)", p.name).group(1))
    )
    for f in files:
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        s, items = d["summary"], d["items"]
        n_train = 0 if f.name.startswith("untuned") else int(re.match(r"n(\d+)", f.name).group(1))
        k = sum(r["match"]["exact_stereo"] for r in items)
        k_ns = sum(r["match"]["exact"] for r in items)
        n = len(items)
        rows.append({
            "n_train": n_train, "n_test": n, "k": k, "k_ns": k_ns,
            "exact": k / n, "exact_ci": wilson(k, n), "exact_ns": k_ns / n, "exact_ns_ci": wilson(k_ns, n),
            "valid": s["valid_rate"], "tanimoto": s["mean_tanimoto"], "loop": s["loop_rate"],
            "tok_in": s["mean_tokens_in"], "tok_out": s["mean_tokens_out"],
            "lat_med": s["median_latency_s"], "lat_p95": s["p95_latency_s"], "eval_usd": s["usd"],
            **train_history(n_train),
        })

    # Chart: one axis, two series (stereo-aware headline, stereo-agnostic), Wilson 95% CI bars.
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    xs = [r["n_train"] for r in rows]
    for key, color, marker, label in (
        ("exact_ns", ORANGE, "s", "Exact match, stereo ignored"),
        ("exact", BLUE, "o", "Exact match (stereo-aware, headline)"),
    ):
        ys = [100 * r[key] for r in rows]
        lo = [100 * (r[key] - r[key + "_ci"][0]) for r in rows]
        hi = [100 * (r[key + "_ci"][1] - r[key]) for r in rows]
        ax.errorbar(xs, ys, yerr=[lo, hi], color=color, marker=marker, markersize=7, linewidth=2,
                    capsize=3, elinewidth=1, markeredgecolor=SURFACE, markeredgewidth=2, label=label, zorder=3)
        ax.annotate(f"{ys[-1]:.1f}%", (xs[-1], ys[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK)
    ax.set_xlabel(f"LoRA training examples (Qwen3.5-9B, 3 epochs, 384 px)", color=MUTED, fontsize=9)
    ax.set_ylabel(f"Exact match on {rows[0]['n_test']} held-out test images (%)", color=MUTED, fontsize=9)
    ax.set_title("Learning curve: exact match vs training-set size", color=INK, fontsize=11, loc="left")
    ax.set_xticks(xs)
    ax.set_xticklabels(["untuned" if x == 0 else f"{x:,}" for x in xs], fontsize=8, color=MUTED)
    ax.tick_params(axis="y", labelsize=8, colors=MUTED)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK)
    fig.text(0.01, 0.01, "Bars: Wilson 95% intervals. With 500 test images, differences under ~4 points are noise.",
             fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(RES / "learning_curve.png", facecolor=SURFACE)

    def pct(x):
        return f"{100 * x:.1f}%"

    lines = [
        "# Learning curve (Person A)",
        "",
        f"Qwen3.5-9B + LoRA (rank 8, language-model MLP + attention only), 3 epochs, batch 8, LR 2e-4 cosine, "
        f"384 px renders, thinking off, temperature 0, max_tokens {json.loads(files[0].read_text())['summary']['max_tokens']}. "
        f"Scored on the frozen `data/{TEST}.json` ({rows[0]['n_test']} images, scaffold split). Served by vLLM on one "
        "Modal L40S. Training subsets are nested (first N rows of the shuffled train manifest).",
        "",
        "| Train size | Exact (stereo-aware) [95% CI] | Exact, stereo ignored [95% CI] | Valid | Mean Tanimoto | Loop rate | Tokens in/out | Latency median / p95 | Final train / val loss | Best val loss (epoch) | Train time |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        tl = f"{r['final_train_loss']:.3f} / {r['final_val_loss']:.3f}" if r.get("final_train_loss") else "–"
        bv = f"{r['best_val_loss']:.3f} ({r['best_val_epoch']:.1f})" if r.get("best_val_loss") else "–"
        tt = f"{r['train_s'] / 60:.1f} min" if r.get("train_s") else "–"
        lines.append(
            f"| {'untuned' if r['n_train'] == 0 else f'{r['n_train']:,}'} "
            f"| {pct(r['exact'])} ({r['k']}/{r['n_test']}) [{pct(r['exact_ci'][0])}–{pct(r['exact_ci'][1])}] "
            f"| {pct(r['exact_ns'])} ({r['k_ns']}/{r['n_test']}) [{pct(r['exact_ns_ci'][0])}–{pct(r['exact_ns_ci'][1])}] "
            f"| {pct(r['valid'])} | {r['tanimoto']:.2f} | {pct(r['loop'])} | {r['tok_in']:.0f} / {r['tok_out']:.0f} "
            f"| {r['lat_med']:.2f} s / {r['lat_p95']:.2f} s | {tl} | {bv} | {tt} |"
        )
    lines += [
        "",
        "![learning curve](learning_curve.png)",
        "",
        "Notes: latency is per request with 8 concurrent requests and depends on server settings; compare rows "
        "measured in the same server session. Costs per run are in `cost_log.md`.",
    ]
    (RES / "LEARNING_CURVE.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
