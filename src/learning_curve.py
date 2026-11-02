"""Checkpoint selection (validation only), learning curve, and ablations on the frozen test set.

  uv run python -m src.learning_curve

1. Selection: for each training run, every epoch checkpoint <run>_epK is scored on the 200-image validation
   set (results/<run>_epK__val_<res>.json). The checkpoint with the highest stereo-aware validation exact
   match is selected (ties: lower validation loss from the training log, then the earlier epoch). The test
   set is never used to choose anything. Writes results/selection.json.
2. Learning curve: untuned + selected checkpoint per training size, on results/<ckpt>__test_synthetic_384.json.
3. Ablations at 2,000 examples: each single change vs the 2,000 baseline, with an exact McNemar test on the
   paired per-image outcomes (both systems see the same 500 molecules).
Writes results/learning_curve.png and results/LEARNING_CURVE.md. Intervals are Wilson 95%.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
CURVE_SIZES = (500, 2000, 5000, 10000)
ABLATIONS = {  # run -> (label, test set it is scored on)
    "n2000_r384_merger_e3": ("LoRA also on vision projector/merger", "test_synthetic_384"),
    "n2000_r640_lang_e3": ("640 px images (vs 384 px)", "test_synthetic_640"),
}
BASELINE_2K = "n2000_r384_lang_e3"
BLUE, ORANGE = "#2a78d6", "#eb6834"  # dataviz reference palette slots 1-2 (validated, light mode)
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def mcnemar(a: list[bool], b: list[bool]) -> dict:
    """Exact two-sided McNemar test on paired outcomes."""
    only_a = sum(x and not y for x, y in zip(a, b))
    only_b = sum(y and not x for x, y in zip(a, b))
    n = only_a + only_b
    k = min(only_a, only_b)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n) if n else 1.0
    return {"only_a": only_a, "only_b": only_b, "p": p}


def load(name: str) -> dict | None:
    f = RES / f"{name}.json"
    return json.loads(f.read_text()) if f.exists() else None


def outcomes(d: dict) -> list[bool]:
    # Pair by ground-truth SMILES so 384 px and 640 px files line up molecule by molecule.
    return [r["match"]["exact_stereo"] for r in sorted(d["items"], key=lambda r: r["truth"])]


def epoch_val_losses(run: str) -> dict[int, float]:
    log = RES / f"raw/train_{run}.log"
    if not log.exists():
        return {}
    return {int(m.group(1)): float(m.group(2))
            for m in re.finditer(r"^epoch (\d+) checkpoint \S+: val loss ([\d.]+)", log.read_text(), re.M)}


def select(run: str) -> dict | None:
    res = "640" if "_r640_" in run else "384"
    losses = epoch_val_losses(run)
    cands = []
    for ep in sorted(losses):
        d = load(f"{run}_ep{ep}__val_{res}")
        if d:
            cands.append({"ckpt": f"{run}_ep{ep}", "epoch": ep, "val_exact": d["summary"]["exact_match"],
                          "val_exact_nostereo": d["summary"]["exact_match_nostereo"], "val_loss": losses[ep]})
    if not cands:
        return None
    best = max(cands, key=lambda c: (c["val_exact"], -c["val_loss"], -c["epoch"]))
    return {"selected": best["ckpt"], "rule": "max val exact (stereo-aware); ties: min val loss, then earlier epoch",
            "candidates": cands}


def summarize(d: dict) -> dict:
    s, items = d["summary"], d["items"]
    n = len(items)
    k = sum(r["match"]["exact_stereo"] for r in items)
    k_ns = sum(r["match"]["exact"] for r in items)
    return {"n_test": n, "k": k, "k_ns": k_ns, "exact": k / n, "exact_ci": wilson(k, n), "exact_ns": k_ns / n,
            "exact_ns_ci": wilson(k_ns, n), "valid": s["valid_rate"], "tanimoto": s["mean_tanimoto"],
            "loop": s["loop_rate"], "tok_in": s["mean_tokens_in"], "tok_out": s["mean_tokens_out"],
            "lat_med": s["median_latency_s"], "lat_p95": s["p95_latency_s"],
            "per_1k": s.get("steady_usd_per_1k_images"), "workers": s.get("workers")}


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def p_str(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def main() -> None:
    selection = {}
    for run in [f"n{n}_r384_lang_e3" for n in CURVE_SIZES] + list(ABLATIONS):
        sel = select(run)
        if sel:
            selection[run] = sel
    (RES / "selection.json").write_text(json.dumps(selection, indent=2))

    def chosen(run: str) -> str:
        # n500 predates per-epoch checkpoints: only its final (epoch 3) adapter exists.
        return selection[run]["selected"] if run in selection else run

    curve = []
    d0 = load("untuned__test_synthetic_384")
    if d0:
        curve.append({"n_train": 0, "ckpt": "untuned", "data": d0, **summarize(d0)})
    for n in CURVE_SIZES:
        run = f"n{n}_r384_lang_e3"
        d = load(f"{chosen(run)}__test_synthetic_384")
        if d:
            curve.append({"n_train": n, "ckpt": chosen(run), "data": d, **summarize(d)})

    # Chart: one axis, two series, Wilson 95% bars, direct labels on the last point.
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    xs = [r["n_train"] for r in curve]
    for key, color, marker, label in (("exact_ns", ORANGE, "s", "Exact match, stereo ignored"),
                                      ("exact", BLUE, "o", "Exact match (stereo-aware, headline)")):
        ys = [100 * r[key] for r in curve]
        lo = [100 * (r[key] - r[key + "_ci"][0]) for r in curve]
        hi = [100 * (r[key + "_ci"][1] - r[key]) for r in curve]
        ax.errorbar(xs, ys, yerr=[lo, hi], color=color, marker=marker, markersize=7, linewidth=2, capsize=3,
                    elinewidth=1, markeredgecolor=SURFACE, markeredgewidth=2, label=label, zorder=3)
        ax.annotate(f"{ys[-1]:.1f}%", (xs[-1], ys[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK)
    ax.set_xlabel("LoRA training examples (Qwen3.5-9B, 384 px, checkpoint chosen on validation)", color=MUTED, fontsize=9)
    ax.set_ylabel(f"Exact match on {curve[0]['n_test']} held-out test images (%)", color=MUTED, fontsize=9)
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
    fig.text(0.01, 0.01, "Bars: Wilson 95% intervals. Adjacent points are compared with a paired McNemar test "
             "in LEARNING_CURVE.md.", fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(RES / "learning_curve.png", facecolor=SURFACE)

    lines = [
        "# Learning curve and ablations (Person A)",
        "",
        "Qwen3.5-9B + LoRA (rank 8; language-model MLP + full attention unless noted), 3 epochs, batch 8, LR 2e-4 "
        "cosine, thinking off, temperature 0, max_tokens 108. Trained and served on Modal (1x L40S, vLLM). "
        "Nested training subsets (first N rows of the shuffled train manifest). Every run saves one checkpoint per "
        "epoch; the reported checkpoint is chosen on the 200-image **validation** set (highest stereo-aware exact "
        "match; ties: lower validation loss, then earlier epoch). The 500-image test set is scored once per chosen "
        "checkpoint and never used for selection. Headline metric (pre-registered): stereo-aware exact match.",
        "",
        "## Learning curve (test_synthetic_384, 500 images)",
        "",
        "| Train size | Checkpoint | Exact [95% CI] | Exact, stereo ignored [95% CI] | Valid | Tanimoto | Loops | Tokens in/out | Latency med / p95 | Steady $/1k images | vs previous size (McNemar) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    prev = None
    for r in curve:
        cmp = "–"
        if prev is not None:
            m = mcnemar(outcomes(r["data"]), outcomes(prev["data"]))
            cmp = f"+{m['only_a']} / -{m['only_b']} images, p={p_str(m['p'])}"
        per_1k = f"${r['per_1k']:.3f} @ {r['workers']} conc." if r.get("per_1k") is not None else "–"
        lines.append(
            f"| {'untuned' if r['n_train'] == 0 else f'{r['n_train']:,}'} | `{r['ckpt']}` "
            f"| {pct(r['exact'])} ({r['k']}/{r['n_test']}) [{pct(r['exact_ci'][0])}–{pct(r['exact_ci'][1])}] "
            f"| {pct(r['exact_ns'])} [{pct(r['exact_ns_ci'][0])}–{pct(r['exact_ns_ci'][1])}] "
            f"| {pct(r['valid'])} | {r['tanimoto']:.2f} | {pct(r['loop'])} | {r['tok_in']:.0f} / {r['tok_out']:.0f} "
            f"| {r['lat_med']:.2f} s / {r['lat_p95']:.2f} s | {per_1k} | {cmp} |")
        prev = r
    if any(r["n_train"] == 500 for r in curve):
        lines += ["", "The 500 run predates per-epoch checkpoints, so its final (epoch-3) adapter is reported; its "
                  "validation loss was 0.353 at epoch 2 vs 0.363 at epoch 3."]

    base = load(f"{chosen(BASELINE_2K)}__test_synthetic_384")
    abl_rows = []
    for run, (label, test) in ABLATIONS.items():
        d = load(f"{chosen(run)}__{test}")
        if d and base:
            m = mcnemar(outcomes(d), outcomes(base))
            s, sb = summarize(d), summarize(base)
            abl_rows.append(
                f"| {label} | `{chosen(run)}` | {pct(s['exact'])} vs {pct(sb['exact'])} "
                f"| {100 * (s['exact'] - sb['exact']):+.1f} pts | +{m['only_a']} / -{m['only_b']} | {p_str(m['p'])} "
                f"| {pct(s['exact_ns'])} vs {pct(sb['exact_ns'])} | {s['tok_in']:.0f} vs {sb['tok_in']:.0f} |")
    if abl_rows:
        lines += ["", f"## Ablations at 2,000 examples (each vs `{chosen(BASELINE_2K)}`)", "",
                  "| Change | Checkpoint | Exact (change vs base) | Difference | Images gained / lost | McNemar p "
                  "| Stereo ignored | Tokens in |", "|---|---|---|---|---|---|---|---|", *abl_rows,
                  "", "A difference is reported as real only if McNemar p < 0.05; otherwise it is within noise."]
    lines += ["", "![learning curve](learning_curve.png)", "",
              "Latency is per request at the stated concurrency; compare rows measured in the same server session. "
              "Steady $/1k excludes the one-off cold start (see each results file's `cold_start_s`). "
              "Checkpoint choices: `selection.json`. Costs per run: `cost_log.md`."]
    (RES / "LEARNING_CURVE.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
