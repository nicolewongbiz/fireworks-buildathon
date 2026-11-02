"""Cross-system evaluation: recompute metrics from results/*.json, bootstrap CIs, cost per correct, charts.

Usage
    .venv/bin/python -m src.evaluate                                  # every results/*.json
    .venv/bin/python -m src.evaluate results/tuned.json results/large.json
    .venv/bin/python -m src.evaluate --cost large=4.20 --price-in large=0.15 --price-out large=0.60
    .venv/bin/python -m src.evaluate --out-dir /tmp/cmp --metric exact_nostereo --no-charts
    .venv/bin/python -m src.evaluate --help

Inputs are `results/<name>.json` files shaped `{"summary": {...}, "items": [...]}` as written by
src/run_system.py. Nothing in `summary` is trusted: every metric is recomputed from `items`, and the
summary is only used for cost fields (`usd`, `gpu_hours`, `gpu_usd_per_hour`, `max_tokens`) and to
flag disagreements with the recomputed numbers in the notes.

File-name convention (system and test set)
    results/<system>.json             -> system = <system>, test set inferred from the items
    results/<system>__<testset>.json  -> system = <system>, test set = <testset>   (double underscore)
    Anything else in the stem is kept verbatim as the system label (e.g. `smoke_untuned`), so old
    files stay distinguishable. When the stem has no `__`, the test set is the common parent directory
    of the items' `image_path` relative to `data/images/` (e.g. `smoke`, `384/test`, `384/drugs`), or
    `default` if that cannot be determined. `comparison.json` in the results dir is skipped.

Metrics per system (all from items; missing keys become None, never a crash)
    exact        stereo-aware exact match   = match.exact_stereo (headline, as in run_system.py)
                 If a file predates exact_stereo it is derived from predict.smiles vs truth with
                 src.validate.canonicalize when RDKit is importable, otherwise it is None.
    exact_nostereo  stereo-agnostic exact match = match.exact
    valid        RDKit-parseable output      = match.valid
    tanimoto     mean Morgan Tanimoto        = match.tanimoto
    loop_rate    outputs that hit max_tokens = match.hit_cap, or predict.tokens_out >= summary.max_tokens
    latency      p50 / p95 of predict.latency_s (p95 uses the same index rule as run_system.py)
    Bootstrap 95% CIs (percentile, --bootstrap resamples, fixed --seed) for exact, exact_nostereo, valid.

Cost
    usd            = --cost <system>=<usd> if given, else summary.usd (GPU wall-clock x hourly rate as
                     recorded by run_system.py), plus token_usd when --price-in/--price-out are given.
    token_usd      = sum(tokens_in) * price_in / 1e6 + sum(tokens_out) * price_out / 1e6, only for systems
                     named in --price-in/--price-out. Do not pass prices for a run whose usd already
                     includes tokens (run_system.py --usd-per-mtok-*), or they are counted twice.
    cost_per_correct = usd / (number of headline exact matches); inf when there are zero matches.
    cost_per_image   = usd / n.

Outputs (in --out-dir, default results/)
    comparison.md, comparison.json
    chart_exact_vs_cost.{png,svg} and chart_exact_vs_cost_dark.{png,svg}   headline chart
    chart_exact_by_testset.{png,svg} and _dark variants                     only if >1 test set seen
Only the standard library plus numpy, pandas and matplotlib are required; RDKit is optional.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HEADLINE_LABEL = {"exact": "stereo-aware", "exact_nostereo": "stereo-agnostic"}
KNOWN_ORDER = ["untuned", "tuned", "large", "routed"]  # fixed hue slots so colors follow the entity
N_COLOR_SLOTS = 3  # validated all-pairs cap for a scatter (dataviz palette.md); systems beyond use neutral gray

# ---------------------------------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------------------------------


def _get(d: Any, *keys: str, default: Any = None) -> Any:
    """Nested dict lookup that returns `default` for any missing level or non-dict."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _try_canonicalize():
    """Return src.validate.canonicalize if RDKit is importable, else None (stereo derivation is optional)."""
    try:
        here = Path(__file__).resolve().parent.parent
        if str(here) not in sys.path:
            sys.path.insert(0, str(here))
        from src.validate import canonicalize  # noqa: WPS433

        return canonicalize
    except Exception:  # noqa: BLE001  (rdkit missing, import error, ...)
        return None


def split_name(stem: str) -> tuple[str, str | None]:
    """`<system>__<testset>` -> (system, testset); otherwise (stem, None)."""
    if "__" in stem:
        system, testset = stem.split("__", 1)
        return system or stem, testset or None
    return stem, None


def infer_testset(items: list[dict]) -> str:
    """Common parent dir of image paths, relative to data/images/, e.g. `smoke` or `384/test`."""
    paths = [str(_get(it, "image_path") or "") for it in items]
    paths = [p for p in paths if p]
    if not paths:
        return "default"
    parents = {str(Path(p).parent).replace("\\", "/") for p in paths}
    common = os.path.commonpath(list(parents)) if len(parents) > 1 else parents.pop()
    common = common.replace("\\", "/")
    marker = "data/images/"
    if marker in common:
        common = common.split(marker, 1)[1]
    return common.strip("/") or "default"


def load_results(paths: list[Path]) -> list[dict]:
    """Read each results file into {"path", "system", "testset", "summary", "items"}; skips malformed files."""
    runs = []
    for p in paths:
        try:
            data = json.loads(p.read_text())
        except Exception as e:  # noqa: BLE001
            print(f"skip {p}: cannot parse JSON ({e})", file=sys.stderr)
            continue
        if not isinstance(data, dict) or not isinstance(data.get("items"), list):
            print(f"skip {p}: not a results file (expected {{'summary', 'items'}})", file=sys.stderr)
            continue
        summary = data.get("summary") if isinstance(data.get("summary"), dict) else {}
        system, testset = split_name(p.stem)
        runs.append({
            "path": str(p),
            "system": system,
            "testset": testset or infer_testset(data["items"]),
            "summary": summary,
            "items": data["items"],
        })
    return runs


# ---------------------------------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------------------------------


def bootstrap_ci(values: list[float], n_boot: int, seed: int) -> tuple[float | None, float | None]:
    """Percentile bootstrap 95% CI of the mean. Fresh RNG per call so results do not depend on order."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or n_boot <= 0:
        return None, None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _mean(values: list) -> float | None:
    vals = [float(v) for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def _p95(sorted_vals: list[float]) -> float | None:
    n = len(sorted_vals)
    if n == 0:
        return None
    return sorted_vals[min(n - 1, math.ceil(0.95 * n) - 1)]


def _bools(items: list[dict], *keys: str) -> list[bool | None]:
    out = []
    for it in items:
        v = _get(it, *keys)
        out.append(None if v is None else bool(v))
    return out


def evaluate_run(run: dict, args: argparse.Namespace, canonicalize) -> dict:
    """Recompute every metric for one results file. Returns a flat dict used for tables, JSON and charts."""
    items, summary, system = run["items"], run["summary"], run["system"]
    n = len(items)
    notes: list[str] = []

    exact_ns = _bools(items, "match", "exact")
    valid = _bools(items, "match", "valid")
    exact_st = _bools(items, "match", "exact_stereo")
    if any(v is None for v in exact_st) and n:
        if canonicalize is not None:
            derived = []
            for it, v, ok, ex in zip(items, exact_st, valid, exact_ns):
                if v is not None:
                    derived.append(v)
                    continue
                if ex is False or not ok:  # a stereo match implies a skeleton match, so skip RDKit when exact is False
                    derived.append(False)
                    continue
                pred = _get(it, "match", "pred_smiles") or _get(it, "predict", "smiles") or ""
                truth = _get(it, "truth") or ""
                pc, tc = canonicalize(pred, strip_stereo=False), canonicalize(truth, strip_stereo=False)
                derived.append(pc is not None and pc == tc)
            exact_st = derived
            notes.append("match.exact_stereo missing; derived with RDKit from pred_smiles vs truth")
        else:
            exact_st = [None] * n
            notes.append("match.exact_stereo missing and RDKit unavailable; stereo-aware exact is n/a")

    hit_cap = _bools(items, "match", "hit_cap")
    max_tokens = _get(summary, "max_tokens")
    if any(v is None for v in hit_cap) and n:
        if isinstance(max_tokens, (int, float)):
            hit_cap = [bool(v) if v is not None else (_get(it, "predict", "tokens_out") or 0) >= max_tokens
                       for it, v in zip(items, hit_cap)]
            notes.append("match.hit_cap missing; derived from tokens_out >= summary.max_tokens")
        else:
            hit_cap = [None] * n
            notes.append("match.hit_cap and summary.max_tokens missing; loop rate is n/a")

    tanimoto = [_get(it, "match", "tanimoto") for it in items]
    lat = sorted(float(v) for v in (_get(it, "predict", "latency_s") for it in items) if v is not None)
    tok_in = [_get(it, "predict", "tokens_in") for it in items]
    tok_out = [_get(it, "predict", "tokens_out") for it in items]
    if any(v is None for v in tok_in + tok_out):
        notes.append("some items lack predict.tokens_in/tokens_out")
    missing_tags = sum(1 for it in items if it.get("tags") is None)
    if missing_tags == n and n:
        notes.append("items carry no tags")

    def rate(vals):
        v = [x for x in vals if x is not None]
        return (sum(v) / len(v)) if v else None

    def ci(vals):
        v = [float(x) for x in vals if x is not None]
        return bootstrap_ci(v, args.bootstrap, args.seed) if v else (None, None)

    metrics = {
        "exact": rate(exact_st), "exact_ci": ci(exact_st), "exact_n": sum(1 for v in exact_st if v),
        "exact_nostereo": rate(exact_ns), "exact_nostereo_ci": ci(exact_ns),
        "exact_nostereo_n": sum(1 for v in exact_ns if v),
        "valid": rate(valid), "valid_ci": ci(valid),
        "mean_tanimoto": _mean(tanimoto),
        "loop_rate": rate(hit_cap),
        "median_latency_s": float(np.median(lat)) if lat else None,
        "p95_latency_s": _p95(lat),
        "mean_latency_s": _mean(lat),
        "tokens_in_total": sum(int(v) for v in tok_in if v is not None),
        "tokens_out_total": sum(int(v) for v in tok_out if v is not None),
        "escalation_rate": rate(_bools(items, "predict", "escalated")) if n else None,
    }

    # ---- cost -------------------------------------------------------------------------------------
    summary_usd = _get(summary, "usd")
    price_in, price_out = args.price_in.get(system), args.price_out.get(system)
    token_usd = None
    if price_in is not None or price_out is not None:
        token_usd = (metrics["tokens_in_total"] * (price_in or 0.0) + metrics["tokens_out_total"] * (price_out or 0.0)) / 1e6
    if system in args.cost:
        usd, cost_source = args.cost[system], "--cost override"
    elif summary_usd is not None:
        usd, cost_source = float(summary_usd), "summary.usd"
    else:
        usd, cost_source = None, "none"
        notes.append("summary.usd missing; pass --cost <system>=<usd>")
    if token_usd is not None and (_get(summary, "usd_per_mtok_in") or _get(summary, "usd_per_mtok_out")):
        notes.append("--price-* given but summary.usd already includes token cost (usd_per_mtok_*); tokens counted twice")
    if token_usd is not None and usd is not None:
        usd = usd + token_usd
        cost_source += " + tokens"
    elif token_usd is not None:
        usd, cost_source = token_usd, "tokens"

    headline_n = metrics["exact_n"] if args.metric == "exact" else metrics["exact_nostereo_n"]
    headline_rate = metrics[args.metric]
    headline_used = args.metric
    if headline_rate is None:  # fall back to the other variant for this system and say so
        headline_used = "exact_nostereo" if args.metric == "exact" else "exact"
        headline_rate = metrics[headline_used]
        headline_n = metrics[f"{headline_used}_n"]
        notes.append(f"{args.metric} unavailable; chart/cost use {headline_used}")
    cost_per_correct = None if usd is None else (usd / headline_n if headline_n else math.inf)
    cost_per_image = None if usd is None or n == 0 else usd / n

    # ---- compare with the stored summary ----------------------------------------------------------
    for key, mine in (("exact_match", metrics["exact"]), ("exact_match_nostereo", metrics["exact_nostereo"]),
                      ("valid_rate", metrics["valid"]), ("mean_tanimoto", metrics["mean_tanimoto"]),
                      ("loop_rate", metrics["loop_rate"]), ("n", n)):
        theirs = _get(summary, key)
        if theirs is not None and mine is not None and abs(float(theirs) - float(mine)) > 1e-6:
            notes.append(f"summary.{key}={float(theirs):.4g} but recomputed {float(mine):.4g}")
    if _get(summary, "exact_match") is not None and _get(summary, "exact_match_nostereo") is None:
        notes.append("pre-57f43b5 summary: exact_match there is stereo-agnostic")

    return {
        "system": system, "testset": run["testset"], "path": run["path"], "n": n,
        **metrics,
        "usd": usd, "cost_source": cost_source, "token_usd": token_usd,
        "gpu_hours": _get(summary, "gpu_hours"), "gpu_usd_per_hour": _get(summary, "gpu_usd_per_hour"),
        "first_answer_s": _get(summary, "first_answer_s"), "wall_s": _get(summary, "wall_s"),
        "max_tokens": max_tokens,
        "headline_metric": headline_used, "headline_rate": headline_rate, "headline_n": headline_n,
        "cost_per_correct": cost_per_correct, "cost_per_image": cost_per_image,
        "notes": notes,
    }


# ---------------------------------------------------------------------------------------------------
# Tags
# ---------------------------------------------------------------------------------------------------


def _bin_heavy(v) -> str:
    v = int(v)
    return "heavy_atoms<=20" if v <= 20 else "heavy_atoms 21-30" if v <= 30 else "heavy_atoms 31-40" if v <= 40 else "heavy_atoms>40"


def tag_labels(tags: Any) -> list[str]:
    """Normalize an item's tags to a list of labels.

    List of strings -> as is. Dict (make_dataset.py difficulty attributes) -> bucketed labels, e.g.
    `heavy_atoms 21-30`, `rings=3`, `stereocenters>=3`, `charged`, `bridged`, `fused`, `multi-fragment`,
    `elements: beyond CNO`.
    """
    if tags is None:
        return []
    if isinstance(tags, (list, tuple, set)):
        return [str(t) for t in tags]
    if not isinstance(tags, dict):
        return [str(tags)]
    out: list[str] = []
    for key, v in tags.items():
        if v is None:
            continue
        if key == "heavy_atoms":
            out.append(_bin_heavy(v))
        elif key == "rings":
            out.append(f"rings={int(v)}" if int(v) < 4 else "rings>=4")
        elif key == "stereocenters":
            out.append("stereocenters=0" if int(v) == 0 else "stereocenters 1-2" if int(v) <= 2 else "stereocenters>=3")
        elif key == "ez_bonds":
            out.append("E/Z bonds" if int(v) > 0 else "no E/Z bonds")
        elif key == "charged_atoms":
            out.append("charged" if int(v) > 0 else "neutral")
        elif key == "fragments":
            out.append("multi-fragment" if int(v) > 1 else "single fragment")
        elif key == "elements":
            els = set(v) if isinstance(v, (list, tuple, set)) else {str(v)}
            out.append("elements: beyond CNO" if els - {"C", "N", "O", "H"} else "elements: CNO only")
        elif isinstance(v, bool):
            out.append(key if v else f"not {key}")
        elif isinstance(v, (int, float)):
            out.append(f"{key}={v}")
        else:
            out.append(f"{key}={v}")
    return out


_TAG_FAMILIES = ["heavy_atoms", "rings", "stereocenters", "E/Z", "charged", "neutral", "fragment", "elements", "bridged", "fused"]
_HEAVY_ORDER = ["heavy_atoms<=20", "heavy_atoms 21-30", "heavy_atoms 31-40", "heavy_atoms>40"]


def _tag_sort_key(label: str) -> tuple:
    """Group bucketed tags by attribute family, then by their numeric bin; free-form list tags go last."""
    fam = next((i for i, f in enumerate(_TAG_FAMILIES) if f in label), len(_TAG_FAMILIES))
    if label in _HEAVY_ORDER:
        return (fam, _HEAVY_ORDER.index(label), label)
    m = re.search(r"(\d+)", label)
    negative = label.startswith(("no ", "not ")) or label == "neutral" or label == "single fragment"
    return (fam, int(m.group(1)) if m else (-1 if negative else 0), label)


def per_tag_table(runs: list[dict], results: list[dict]) -> pd.DataFrame | None:
    """Rows = tag labels, columns = `system [testset]`, cells = (hits, n) on the headline metric."""
    cells: dict[str, dict[str, tuple[int, int]]] = {}
    cols: list[str] = []
    for run, res in zip(runs, results):
        key = res["metric_key"] if "metric_key" in res else "exact_stereo"
        col = res["label"]
        has_tags = False
        for it in run["items"]:
            labels = tag_labels(it.get("tags"))
            if not labels:
                continue
            has_tags = True
            hit = _get(it, "match", key)
            if hit is None and key == "exact_stereo":
                hit = _get(it, "match", "exact")
            for lab in labels:
                h, c = cells.setdefault(lab, {}).get(col, (0, 0))
                cells[lab][col] = (h + int(bool(hit)), c + 1)
        if has_tags:
            cols.append(col)
    if not cells:
        return None
    df = pd.DataFrame(index=sorted(cells, key=_tag_sort_key), columns=cols, dtype=object)
    for lab, d in cells.items():
        for col, hc in d.items():
            df.at[lab, col] = hc
    return df


# ---------------------------------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------------------------------


def pct(v: float | None) -> str:
    return "n/a" if v is None else f"{100 * v:.1f}%"


def pct_ci(v: float | None, ci: tuple) -> str:
    if v is None:
        return "n/a"
    lo, hi = ci
    if lo is None:
        return pct(v)
    return f"{100 * v:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]"


def usd(v: float | None) -> str:
    if v is None:
        return "n/a"
    if math.isinf(v):
        return "inf (0 correct)"
    return f"${v:,.4f}" if v < 0.1 else f"${v:,.2f}"


def secs(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.2f}s"


def num(v: float | None, nd: int = 3) -> str:
    return "n/a" if v is None else f"{v:.{nd}f}"


def write_markdown(results: list[dict], tags: pd.DataFrame | None, out: Path, args: argparse.Namespace,
                   chart_files: list[Path], multi_testset: bool) -> None:
    hl = HEADLINE_LABEL[args.metric]
    lines = [
        "# System comparison",
        "",
        f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} by `src/evaluate.py` from "
        f"{len(results)} results file(s). All metrics recomputed from per-item records; "
        f"brackets are bootstrap 95% CIs ({args.bootstrap} resamples, seed {args.seed}). "
        f"Headline exact match is {hl}; cost per correct uses it.",
        "",
        "| System | Test set | n | Exact (stereo) | Exact (no stereo) | Valid | Mean Tanimoto | Loop rate | Latency p50 / p95 | USD | Cost / correct | Cost / image |",
        "|---|---|---:|---|---|---|---:|---:|---|---:|---:|---:|",
    ]
    for r in results:
        flag = "" if not r["notes"] else " *"
        lines.append(
            f"| {r['system']}{flag} | {r['testset']} | {r['n']} | {pct_ci(r['exact'], r['exact_ci'])} | "
            f"{pct_ci(r['exact_nostereo'], r['exact_nostereo_ci'])} | {pct_ci(r['valid'], r['valid_ci'])} | "
            f"{num(r['mean_tanimoto'])} | {pct(r['loop_rate'])} | {secs(r['median_latency_s'])} / {secs(r['p95_latency_s'])} | "
            f"{usd(r['usd'])} | {usd(r['cost_per_correct'])} | {usd(r['cost_per_image'])} |"
        )
    lines += ["", "Cost source per system: " + "; ".join(f"{r['label']}: {r['cost_source']}" for r in results) + ".", ""]
    noted = [r for r in results if r["notes"]]
    if noted:
        lines += ["## Notes (* in the table)", ""]
        for r in noted:
            for note in r["notes"]:
                lines.append(f"- `{r['system']}` ({r['testset']}): {note}")
        lines.append("")
    if chart_files:
        lines += ["## Charts", ""]
        for f in chart_files:
            if f.suffix == ".png" and "dark" not in f.name:
                lines.append(f"![{f.stem}]({f.name})")
        lines += ["", "Dark-surface variants: " + ", ".join(f.name for f in chart_files if "dark" in f.name and f.suffix == ".png"), ""]
    if not multi_testset:
        lines += ["Per-test-set chart skipped: all runs are on one test set.", ""]
    if tags is not None:
        lines += [f"## Exact match ({hl}) by tag", "",
                  "Tags from the test manifest (`make_dataset.py` difficulty attributes are bucketed; list tags used as is). Cell = exact % (hits/n).", "",
                  "| Tag | " + " | ".join(tags.columns) + " |", "|---|" + "---|" * len(tags.columns)]
        for lab, row in tags.iterrows():
            cells = []
            for col in tags.columns:
                hc = row[col]
                cells.append("–" if not isinstance(hc, tuple) else f"{100 * hc[0] / hc[1]:.1f}% ({hc[0]}/{hc[1]})")
            lines.append(f"| {lab} | " + " | ".join(cells) + " |")
        lines.append("")
    out.write_text("\n".join(lines))


def _jsonable(v: Any) -> Any:
    if isinstance(v, float) and (math.isinf(v) or math.isnan(v)):
        return None if math.isnan(v) else "inf"
    if isinstance(v, tuple):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_jsonable(x) for x in v]
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    return v


# ---------------------------------------------------------------------------------------------------
# Charts (follows the dataviz skill: validated palette, thin marks, hairline grid, direct labels, light + dark)
# ---------------------------------------------------------------------------------------------------

THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781", "grid": "#e1e0d9",
              "axis": "#c3c2b7", "series": ["#2a78d6", "#eb6834", "#1baf7a"]},
    "dark": {"surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781", "grid": "#2c2c2a",
             "axis": "#383835", "series": ["#3987e5", "#d95926", "#199e70"]},
}
MARKERS = ["o", "s", "D", "^", "v", "P", "X", "*", "h", "<", ">"]


def assign_slots(systems: list[str]) -> dict[str, int]:
    """Fixed slot per system so the color follows the entity across runs and filters.

    Known systems take fixed slots in KNOWN_ORDER; the rest take the next free slot in order of
    appearance. Slots >= N_COLOR_SLOTS render in neutral gray (the palette only validates three hues
    for all-pairs scatter use) and rely on the marker shape and the direct label for identity.
    """
    slots: dict[str, int] = {}
    used: set[int] = set()
    for s in systems:
        tokens = set(re.split(r"[_\-\s]+", s.lower()))
        for i, k in enumerate(KNOWN_ORDER):
            if k in tokens and i not in used:
                slots[s] = i
                used.add(i)
                break
    nxt = 0
    for s in systems:
        if s in slots:
            continue
        while nxt in used:
            nxt += 1
        slots[s] = nxt
        used.add(nxt)
    return slots


def _style(theme: dict):
    import matplotlib as mpl

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "svg.fonttype": "none",
        "axes.facecolor": theme["surface"], "figure.facecolor": theme["surface"], "savefig.facecolor": theme["surface"],
        "axes.edgecolor": theme["axis"], "axes.labelcolor": theme["ink2"], "xtick.color": theme["muted"],
        "ytick.color": theme["muted"], "text.color": theme["ink"], "axes.linewidth": 1.0,
        "xtick.major.size": 0, "ytick.major.size": 0, "xtick.minor.size": 0, "ytick.minor.size": 0,
        "axes.titlesize": 18, "axes.labelsize": 14, "xtick.labelsize": 12, "ytick.labelsize": 12,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def _series_color(theme: dict, slot: int) -> str:
    return theme["series"][slot] if slot < N_COLOR_SLOTS else theme["muted"]


def _hatch(slot: int) -> str | None:
    """Texture (45/135 degree lines only, per the dataviz skill) tells the gray slots apart in bar charts."""
    if slot < N_COLOR_SLOTS:
        return None
    return ["//", "\\\\", "///", "\\\\\\"][(slot - N_COLOR_SLOTS) % 4]


def _save(fig, out_dir: Path, stem: str, theme_name: str) -> list[Path]:
    suffix = "" if theme_name == "light" else "_dark"
    files = []
    for ext in ("png", "svg"):
        f = out_dir / f"{stem}{suffix}.{ext}"
        fig.savefig(f, dpi=200, bbox_inches="tight", pad_inches=0.25)
        files.append(f)
    return files


def chart_exact_vs_cost(results: list[dict], out_dir: Path, args: argparse.Namespace, multi_testset: bool) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    pts = [r for r in results if r["headline_rate"] is not None]
    if not pts:
        print("chart_exact_vs_cost: no system has an exact-match value; skipped", file=sys.stderr)
        return []
    slots = assign_slots(list(OrderedDict.fromkeys(r["system"] for r in results)))
    finite = [r["cost_per_correct"] for r in pts if r["cost_per_correct"] is not None and math.isfinite(r["cost_per_correct"]) and r["cost_per_correct"] > 0]
    has_zero = any(r["cost_per_correct"] == 0 for r in pts)
    unknown = [r for r in pts if r["cost_per_correct"] is None]
    log = bool(finite) and (max(finite) / min(finite) > 10)
    if finite:
        lo, hi = min(finite), max(finite)
        x_inf = hi * (6 if log else 1.0) + (0 if log else 0.35 * max(hi - lo, hi, 1e-9))
        xlim = (lo / 3, x_inf * 1.8) if log else (-0.05 * x_inf * 1.12, x_inf * 1.12)
    else:
        x_inf, xlim = 1.0, (-0.065, 1.3)
    has_inf = any(r["cost_per_correct"] is None or math.isinf(r["cost_per_correct"]) for r in pts)

    files: list[Path] = []
    for theme_name, theme in THEMES.items():
        _style(theme)
        fig, ax = plt.subplots(figsize=(10, 6.2))
        ax.grid(True, axis="y", color=theme["grid"], linewidth=1.0, linestyle="-")
        ax.set_axisbelow(True)
        if log:
            ax.set_xscale("log")
            ax.xaxis.set_major_locator(LogLocator(base=10, numticks=12))
            ax.xaxis.set_minor_formatter(NullFormatter())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "" if v <= 0 else (f"${v:,.0f}" if v >= 1 and float(v).is_integer() else f"${v:g}")))
        if not finite:  # nothing to put on the x axis: every system is at infinity or unknown
            ax.set_xticks([])
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
        ax.set_xlim(*xlim)
        ymax = max(100 * (r["headline_ci"][1] if r["headline_ci"][1] is not None else r["headline_rate"]) for r in pts)
        ax.set_ylim(0, max(10, min(100, ymax * 1.18 + 4)))

        ordered = sorted(pts, key=lambda r: -(r["headline_rate"] or 0))
        inf_names: list[str] = []
        placed: list[tuple[float, float, int]] = []  # (x key, y, stack depth) of labels already placed
        for r in ordered:
            slot = slots[r["system"]]
            color = _series_color(theme, slot)
            cpc = r["cost_per_correct"]
            at_inf = cpc is None or math.isinf(cpc)
            at_zero = cpc == 0
            x = x_inf if at_inf else (xlim[0] * 1.6 if at_zero and log else cpc)  # $0 cannot sit on a log axis
            y = 100 * r["headline_rate"]
            lo_ci, hi_ci = r["headline_ci"]
            yerr = None if lo_ci is None else [[y - 100 * lo_ci], [100 * hi_ci - y]]
            ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor=color, elinewidth=1.5, capsize=4, capthick=1.5, zorder=3, clip_on=False)
            ax.plot(x, y, marker=MARKERS[slot % len(MARKERS)], markersize=12, color=color,
                    markeredgecolor=color if at_inf else theme["surface"], markeredgewidth=2, linestyle="none", zorder=4,
                    clip_on=False, markerfacecolor=theme["surface"] if at_inf else color)
            label = r["label"] if multi_testset else r["system"]
            if r["headline_metric"] != args.metric:
                label += " \u2020"
            if at_inf:
                inf_names.append(label + (" (cost unknown)" if cpc is None else ""))
                continue
            if at_zero:
                label += " (cost $0)"
            # stack labels of points that sit (almost) on top of each other instead of overprinting them
            xk = math.log10(max(x, 1e-12)) if log else x / max(xlim[1], 1e-12)
            tol_x, tol_y = (0.12 if log else 0.04), 6.0
            depth = sum(1 for (px, py, _) in placed if abs(px - xk) < tol_x and abs(py - y) < tol_y)
            placed.append((xk, y, depth))
            right_edge = (xk > math.log10(x_inf) - 0.45) if log else (xk > 0.8)
            near_top = y > 0.9 * ax.get_ylim()[1]  # keep the label inside the axes, below the mark
            dy = -(10 + 17 * depth) if near_top else (8 + 17 * depth)
            ax.annotate(label, (x, y), xytext=(-12 if right_edge else 12, dy), textcoords="offset points",
                        ha="right" if right_edge else "left", va="top" if near_top else "bottom", fontsize=13,
                        color=theme["ink"], zorder=5)
        if inf_names:
            ax.axvline(x_inf, color=theme["axis"], linewidth=1.0, zorder=1)
            top = ax.get_ylim()[1]
            ax.text(x_inf, top, "\u221e  no correct answers", ha="right", va="top", fontsize=12, color=theme["muted"])
            for k, name in enumerate(inf_names):
                ax.annotate(name, (x_inf, top), xytext=(-6, -22 - 17 * k), textcoords="offset points", ha="right", va="top",
                            fontsize=13, color=theme["ink"], zorder=5)

        hl = HEADLINE_LABEL[args.metric]
        ax.set_xlabel("Cost per correct answer (USD" + (", log scale)" if log else ")") + ("" if finite else " \u2014 undefined, no system answered correctly"))
        ax.set_ylabel(f"Exact match ({hl})")
        ax.set_title(args.title or "Accuracy vs cost per correct answer", loc="left", pad=28, fontweight="bold")
        n_desc = ", ".join(f"{r['system']} n={r['n']}" for r in pts) if len(pts) <= 4 else f"{len(pts)} systems"
        sub = f"Error bars: bootstrap 95% CI ({args.bootstrap} resamples). {n_desc}."
        if has_zero and log:
            sub += " Zero-cost systems sit at the left edge."
        if has_inf:
            sub += " Open markers on the \u221e line: no correct answers."
        if any(r["headline_metric"] != args.metric for r in pts):
            sub += f" † {HEADLINE_LABEL['exact_nostereo' if args.metric == 'exact' else 'exact']} (stereo variant unavailable)."
        ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=11.5, color=theme["ink2"], va="bottom")
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(theme["axis"])
        fig.tight_layout()
        files += _save(fig, out_dir, "chart_exact_vs_cost", theme_name)
        plt.close(fig)
    return files


def chart_by_testset(results: list[dict], out_dir: Path, args: argparse.Namespace) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    systems = list(OrderedDict.fromkeys(r["system"] for r in results))
    testsets = list(OrderedDict.fromkeys(r["testset"] for r in results))
    slots = assign_slots(systems)
    table = {(r["system"], r["testset"]): r for r in results}
    files: list[Path] = []
    for theme_name, theme in THEMES.items():
        _style(theme)
        fig, ax = plt.subplots(figsize=(max(7.0, 2.2 * len(testsets) + 3.0), 5.6))
        import matplotlib as mpl
        mpl.rcParams["hatch.linewidth"] = 1.2
        ax.grid(True, axis="y", color=theme["grid"], linewidth=1.0)
        ax.set_axisbelow(True)
        group_w = 0.7
        bar_w = min(0.11, group_w / max(1, len(systems)) * 0.8)
        gap = 0.025
        for gi, ts in enumerate(testsets):
            present = [s for s in systems if (s, ts) in table and table[(s, ts)]["headline_rate"] is not None]
            total = len(present) * bar_w + max(0, len(present) - 1) * gap
            start = gi - total / 2 + bar_w / 2
            best = max((table[(s, ts)]["headline_rate"] for s in present), default=None)
            for si, s in enumerate(present):
                r = table[(s, ts)]
                x = start + si * (bar_w + gap)
                y = 100 * r["headline_rate"]
                color = _series_color(theme, slots[s])
                ax.bar(x, y, width=bar_w, color=color, linewidth=0, zorder=3, hatch=_hatch(slots[s]), edgecolor=theme["surface"])
                lo_ci, hi_ci = r["headline_ci"]
                if lo_ci is not None:
                    ax.errorbar(x, y, yerr=[[y - 100 * lo_ci], [100 * hi_ci - y]], fmt="none", ecolor=theme["ink2"],
                                elinewidth=1.2, capsize=3, zorder=4)
                if r["headline_rate"] == best:  # label only the extreme in each group
                    ax.annotate(f"{y:.0f}%", (x, 100 * (hi_ci if hi_ci is not None else r["headline_rate"])),
                                xytext=(0, 5), textcoords="offset points", ha="center", va="bottom", fontsize=12, color=theme["ink"])
        ax.set_xticks(range(len(testsets)))
        ax.set_xticklabels(testsets, fontsize=13, color=theme["ink2"])
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
        ax.set_ylim(0, 112)  # headroom for the cap label above a 100% bar
        ax.set_yticks(range(0, 101, 20))
        ax.set_ylabel(f"Exact match ({HEADLINE_LABEL[args.metric]})")
        ax.set_title("Exact match by test set", loc="left", pad=28, fontweight="bold")
        ax.text(0, 1.02, f"Bars: one per system; error bars are bootstrap 95% CIs ({args.bootstrap} resamples).",
                transform=ax.transAxes, fontsize=11.5, color=theme["ink2"], va="bottom")
        from matplotlib.patches import Patch
        handles = [Patch(facecolor=_series_color(theme, slots[s]), edgecolor=theme["surface"], hatch=_hatch(slots[s]), linewidth=0, label=s) for s in systems]
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=12, labelcolor=theme["ink"])
        fig.tight_layout()
        files += _save(fig, out_dir, "chart_exact_by_testset", theme_name)
        plt.close(fig)
    return files


# ---------------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------------


def _kv_float(pairs: list[str] | None, flag: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for p in pairs or []:
        if "=" not in p:
            raise SystemExit(f"{flag} expects <system>=<number>, got {p!r}")
        k, v = p.split("=", 1)
        try:
            out[k.strip()] = float(v)
        except ValueError:
            raise SystemExit(f"{flag} value for {k!r} is not a number: {v!r}") from None
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n", 1)[1])
    ap.add_argument("files", nargs="*", help="results files; default: every <results-dir>/*.json except comparison.json")
    ap.add_argument("--results-dir", default="results", help="where to glob results (default results/)")
    ap.add_argument("--out-dir", default=None, help="where to write comparison.* and charts (default: --results-dir)")
    ap.add_argument("--cost", action="append", metavar="SYSTEM=USD", help="total spend override for a system (repeatable)")
    ap.add_argument("--price-in", action="append", metavar="SYSTEM=USD_PER_MTOK", help="input token price (repeatable)")
    ap.add_argument("--price-out", action="append", metavar="SYSTEM=USD_PER_MTOK", help="output token price (repeatable)")
    ap.add_argument("--metric", choices=["exact", "exact_nostereo"], default="exact",
                    help="headline exact-match variant for chart and cost per correct (default exact = stereo-aware)")
    ap.add_argument("--bootstrap", type=int, default=1000, help="bootstrap resamples (default 1000)")
    ap.add_argument("--seed", type=int, default=0, help="bootstrap RNG seed (default 0)")
    ap.add_argument("--no-charts", action="store_true", help="write tables only")
    ap.add_argument("--title", default=None, help="headline chart title")
    args = ap.parse_args(argv)
    args.cost = _kv_float(args.cost, "--cost")
    args.price_in = _kv_float(args.price_in, "--price-in")
    args.price_out = _kv_float(args.price_out, "--price-out")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir) if args.out_dir else results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.files:
        paths = [Path(f) for f in args.files]
    else:
        paths = sorted(p for p in results_dir.glob("*.json") if p.name != "comparison.json")
    if not paths:
        print(f"no results files found in {results_dir}/", file=sys.stderr)
        return 1
    runs = load_results(paths)
    if not runs:
        print("no usable results files", file=sys.stderr)
        return 1

    canonicalize = _try_canonicalize()
    results = [evaluate_run(run, args, canonicalize) for run in runs]
    for r in results:
        r["headline_ci"] = r["exact_ci"] if r["headline_metric"] == "exact" else r["exact_nostereo_ci"]
        r["metric_key"] = "exact_stereo" if r["headline_metric"] == "exact" else "exact"
    testsets = {r["testset"] for r in results}
    multi_testset = len(testsets) > 1
    for r in results:
        r["label"] = f"{r['system']} · {r['testset']}" if multi_testset else r["system"]
    dup = pd.Series([r["label"] for r in results]).duplicated()
    if dup.any():
        print("warning: duplicate system/test-set labels: " + ", ".join(r["label"] for r, d in zip(results, dup) if d), file=sys.stderr)

    tags = per_tag_table(runs, results)

    chart_files: list[Path] = []
    if not args.no_charts:
        chart_files += chart_exact_vs_cost(results, out_dir, args, multi_testset)
        if multi_testset:
            chart_files += chart_by_testset(results, out_dir, args)

    write_markdown(results, tags, out_dir / "comparison.md", args, chart_files, multi_testset)
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "headline_metric": args.metric,
        "bootstrap": {"resamples": args.bootstrap, "seed": args.seed, "ci": 0.95},
        "cost_overrides": args.cost, "price_in": args.price_in, "price_out": args.price_out,
        "systems": [_jsonable({k: v for k, v in r.items()}) for r in results],
        "by_tag": None if tags is None else {lab: {col: (list(hc) if isinstance(hc, tuple) else None) for col, hc in row.items()}
                                              for lab, row in tags.iterrows()},
        "charts": [str(f) for f in chart_files],
    }
    (out_dir / "comparison.json").write_text(json.dumps(payload, indent=2))

    # console summary
    cols = ["system", "testset", "n", "exact", "exact_nostereo", "valid", "mean_tanimoto", "loop_rate", "median_latency_s", "usd", "cost_per_correct"]
    df = pd.DataFrame([{c: r[c] for c in cols} for r in results])
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.float_format", lambda v: f"{v:.4g}"):
        print(df.to_string(index=False))
    print(f"\nwrote {out_dir / 'comparison.md'}, {out_dir / 'comparison.json'}" + (f", {len(chart_files)} chart files" if chart_files else ""))
    for r in results:
        for note in r["notes"]:
            print(f"note [{r['system']}]: {note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
