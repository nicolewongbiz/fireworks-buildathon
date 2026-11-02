"""Error analysis for results/<system>.json files (read-only on inputs).

  .venv/bin/python src/analyze_errors.py results/smoke_untuned.json results/smoke_tuned.json [--n-worst 12]

Per file -> results/analysis/<name>/errors.md, contact_sheet.png (N worst failures), successes.png (exact matches).
With two or more files -> results/analysis/paired.md compares the first two systems on their shared images.
Each wrong prediction gets one primary mistake class (first match in CLASSES order) plus every flag that applies.
Older results files lack match.hit_cap / match.exact_stereo / tags / summary.max_tokens; all are recomputed or skipped.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw, rdMolDescriptors

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.validate import canonicalize, parse_error  # noqa: E402

RDLogger.DisableLog("rdApp.*")
# dataviz reference palette: surface, primary/secondary/muted ink, hairline, status critical, success text
SURFACE, INK, INK2, MUTED, HAIRLINE, CRITICAL, GOOD = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#d03b3b", "#006300"
CLASSES = ["repeat_loop", "invalid", "stereo_only", "same_formula_diff_connectivity", "wrong_heavy_atoms",
           "wrong_ring_count", "wrong_aromatic_rings", "wrong_elements", "other_connectivity"]


def is_repeat(raw: str) -> bool:
    """A substring of length >= 6 occurs >= 4 times (non-overlapping) -> degenerate generation loop.

    Bracketed stereo atoms are collapsed first ([C@@H] -> C) so ordinary chiral SMILES do not trip the test."""
    raw = re.sub(r"\[([A-Za-z]{1,2})@+H?\]", r"\1", raw or "")
    return any(raw.count(raw[i:i + 6]) >= 4 for i in range(max(0, len(raw) - 5)))


def describe(mol: Chem.Mol) -> dict:
    ri = mol.GetRingInfo()
    arom = sum(all(mol.GetAtomWithIdx(a).GetIsAromatic() for a in r) for r in ri.AtomRings())
    return {"heavy": mol.GetNumHeavyAtoms(), "rings": ri.NumRings(), "arom": arom,
            "elems": Counter(a.GetSymbol() for a in mol.GetAtoms()), "formula": rdMolDescriptors.CalcMolFormula(mol)}


def classify(it: dict, max_tokens: int) -> dict:
    m, p = it["match"], it.get("predict", {})
    raw = m.get("pred_raw") or p.get("smiles") or ""
    hit_cap = m.get("hit_cap")
    if hit_cap is None:
        hit_cap = p.get("tokens_out", 0) >= max_tokens
    exact_stereo = m.get("exact_stereo")
    if exact_stereo is None:
        exact_stereo = bool(m["valid"]) and canonicalize(m["pred_smiles"], False) == canonicalize(it["truth"], False)
    info = {"hit_cap": hit_cap, "repeat": is_repeat(raw), "exact_stereo": exact_stereo, "flags": [], "detail": ""}
    if exact_stereo:
        info["cls"] = "exact"
        return info
    flags, details = info["flags"], []
    if hit_cap or info["repeat"]:
        flags.append("repeat_loop")
        details.append("hit token cap" if hit_cap else "repeated substring")
    if not m["valid"]:
        flags.append("invalid")
        details.append(parse_error(m.get("pred_smiles") or raw) or "RDKit returned None")
    elif m.get("exact"):
        flags.append("stereo_only")
    else:
        t, q = describe(Chem.MolFromSmiles(m["truth_canonical"])), describe(Chem.MolFromSmiles(m["pred_canonical"]))
        if t["formula"] == q["formula"]:
            flags.append("same_formula_diff_connectivity")
        for key, name in (("heavy", "wrong_heavy_atoms"), ("rings", "wrong_ring_count"), ("arom", "wrong_aromatic_rings")):
            if q[key] != t[key]:
                flags.append(name)
                details.append(f"{key} {q[key] - t[key]:+d} (pred {q[key]} vs truth {t[key]})")
        diff = {e: q["elems"][e] - t["elems"][e] for e in q["elems"].keys() | t["elems"].keys() if q["elems"][e] != t["elems"][e]}
        if diff:
            flags.append("wrong_elements")
            details.append("elements " + " ".join(f"{e}{d:+d}" for e, d in sorted(diff.items())))
        if not flags:
            flags.append("other_connectivity")
    info["cls"] = min(flags, key=CLASSES.index)
    info["detail"] = "; ".join(details)
    return info


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def trunc(s: str, n: int) -> str:
    s = (s or "").replace("\n", "\\n")
    return s if len(s) <= n else s[: n - 1] + "…"


def write_errors(name: str, d: dict, infos: list[dict], out: Path) -> None:
    items, n = d["items"], len(d["items"])
    tan = [it["match"]["tanimoto"] for it in items]
    L = [f"# Error analysis: {name}", "", f"Source `{d['summary'].get('system', name)}`, n={n}. "
         f"Exact (stereo-aware) {sum(i['exact_stereo'] for i in infos)}/{n}, exact (stereo-agnostic) "
         f"{sum(bool(it['match'].get('exact')) for it in items)}/{n}, valid {sum(bool(it['match']['valid']) for it in items)}/{n}, "
         f"mean Tanimoto {mean(tan):.3f}, hit token cap {sum(i['hit_cap'] for i in infos)}/{n}.", "", "## Accuracy per tag", ""]
    tags = sorted({t for it in items for t in (it.get("tags") or [])})
    if tags:
        L += ["| tag | n | exact (stereo) | exact (no stereo) | valid | mean Tanimoto |", "|---|---|---|---|---|---|"]
        for t in tags:
            sub = [(it, i) for it, i in zip(items, infos) if t in (it.get("tags") or [])]
            L.append(f"| {t} | {len(sub)} | {sum(i['exact_stereo'] for _, i in sub)} | {sum(bool(it['match'].get('exact')) for it, _ in sub)} "
                     f"| {sum(bool(it['match']['valid']) for it, _ in sub)} | {mean([it['match']['tanimoto'] for it, _ in sub]):.3f} |")
    else:
        L.append("No `tags` field in this results file.")
    wrong = [(it, i) for it, i in zip(items, infos) if i["cls"] != "exact"]
    L += ["", "## Mistake classes (primary class per wrong prediction)", "",
          "Priority order when several apply: " + " > ".join(CLASSES) + ". `repeat_loop` = hit the token cap or a substring "
          "of length >= 6 repeats >= 4 times. Tanimoto is 0 for invalid SMILES.", "",
          "| class | n | share of wrong | mean Tanimoto | n carrying this flag (any position) |", "|---|---|---|---|---|"]
    for c in CLASSES:
        prim = [it for it, i in wrong if i["cls"] == c]
        anyf = sum(c in i["flags"] for _, i in wrong)
        if prim or anyf:
            mt = f"{mean([it['match']['tanimoto'] for it in prim]):.3f}" if prim else "–"
            L.append(f"| {c} | {len(prim)} | {len(prim) / max(1, len(wrong)):.0%} | {mt} | {anyf} |")
    L += ["", "## Per-item classification", "", "| image | class | Tanimoto | tokens_out | detail | predicted (truncated) |", "|---|---|---|---|---|---|"]
    for it, i in sorted(wrong, key=lambda x: (x[0]["match"]["valid"], x[0]["match"]["tanimoto"])):
        L.append(f"| {Path(it['image_path']).name} | {i['cls']} | {it['match']['tanimoto']:.2f} | {it.get('predict', {}).get('tokens_out', '')} "
                 f"| {trunc(i['detail'], 110)} | `{trunc(it['match'].get('pred_raw') or '', 60)}` |")
    (out / "errors.md").write_text("\n".join(L) + "\n")


def font(size: int) -> ImageFont.FreeTypeFont:
    from matplotlib import font_manager

    return ImageFont.truetype(font_manager.findfont("DejaVu Sans"), size)


def placeholder(tile: int, text: str, color: str) -> Image.Image:
    im = Image.new("RGB", (tile, tile), SURFACE)
    dr = ImageDraw.Draw(im)
    dr.rectangle([4, 4, tile - 5, tile - 5], outline=color, width=4)
    f = font(28)
    w = dr.textlength(text, font=f)
    dr.text(((tile - w) / 2, tile / 2 - 16), text, fill=color, font=f)
    return im


def fit_text(dr: ImageDraw.ImageDraw, text: str, f: ImageFont.FreeTypeFont, max_w: int) -> str:
    text = (text or "").replace("\n", "\\n")
    while text and dr.textlength(text, font=f) > max_w:
        text = text[:-2].rstrip("…") + "…"
    return text


def fit(im: Image.Image, tile: int) -> Image.Image:
    im = im.convert("RGB")
    im.thumbnail((tile, tile))
    canvas = Image.new("RGB", (tile, tile), "white")
    canvas.paste(im, ((tile - im.width) // 2, (tile - im.height) // 2))
    return canvas


def contact_sheet(rows: list[tuple[dict, dict]], title: str, out: Path, image_root: Path, cols: int = 3, tile: int = 320) -> None:
    pad, label_h, cap_h, head_h = 16, 26, 128, 64
    cols = max(1, min(cols, len(rows)))
    cell_w, cell_h = 2 * tile + 3 * pad, label_h + tile + cap_h + pad
    nrows = (len(rows) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell_w, head_h + nrows * cell_h), SURFACE)
    dr = ImageDraw.Draw(sheet)
    dr.text((pad, 18), title, fill=INK, font=font(30))
    f_cap, f_lab = font(19), font(16)
    max_w = cell_w - 2 * pad
    for k, (it, info) in enumerate(rows):
        x0, y0 = (k % cols) * cell_w, head_h + (k // cols) * cell_h
        m = it["match"]
        img_path = image_root / it["image_path"]
        left = fit(Image.open(img_path), tile) if img_path.exists() else placeholder(tile, "image missing", MUTED)
        mol = Chem.MolFromSmiles(m.get("pred_smiles") or "") or (Chem.MolFromSmiles(m["pred_canonical"]) if m.get("pred_canonical") else None)
        right = fit(Draw.MolToImage(mol, size=(tile, tile)), tile) if mol is not None else placeholder(tile, "invalid SMILES", CRITICAL)
        for j, (im, lab) in enumerate(((left, "input image"), (right, "predicted molecule (RDKit)"))):
            tx, ty = x0 + pad + j * (tile + pad), y0 + label_h
            dr.text((tx, y0 + 4), lab, fill=MUTED, font=f_lab)
            sheet.paste(im, (tx, ty))
            dr.rectangle([tx - 1, ty - 1, tx + tile, ty + tile], outline=HAIRLINE, width=1)
        cls_color = GOOD if info["cls"] == "exact" else (CRITICAL if info["cls"] in ("invalid", "repeat_loop") else INK)
        cy = y0 + label_h + tile + 8
        dr.text((x0 + pad, cy), fit_text(dr, "truth  " + it["truth"], f_cap, max_w), fill=INK, font=f_cap)
        dr.text((x0 + pad, cy + 26), fit_text(dr, "pred   " + (m.get("pred_raw") or ""), f_cap, max_w), fill=INK2, font=f_cap)
        dr.text((x0 + pad, cy + 52), f"Tanimoto {m['tanimoto']:.2f}  ·  {info['cls']}", fill=cls_color, font=f_cap)
        dr.text((x0 + pad, cy + 78), fit_text(dr, info["detail"] or Path(it["image_path"]).name, f_lab, max_w), fill=INK2, font=f_lab)
    sheet.save(out)


def write_paired(names: list[str], ds: list[dict], infos: list[list[dict]], out: Path) -> None:
    (na, nb), (da, db), (ia, ib) = names, ds, infos
    A = {it["image_path"]: (it, i) for it, i in zip(da["items"], ia)}
    B = {it["image_path"]: (it, i) for it, i in zip(db["items"], ib)}
    shared = [p for p in A if p in B]
    both = sum(A[p][1]["exact_stereo"] and B[p][1]["exact_stereo"] for p in shared)
    only_a = sum(A[p][1]["exact_stereo"] and not B[p][1]["exact_stereo"] for p in shared)
    only_b = sum(B[p][1]["exact_stereo"] and not A[p][1]["exact_stereo"] for p in shared)
    L = [f"# Paired comparison: A = {na} vs B = {nb}", "", f"{len(shared)} shared images (A has {len(A)}, B has {len(B)}). "
         "Correct = stereo-aware exact match.", "", "| outcome | n |", "|---|---|", f"| both right | {both} |", f"| only A right | {only_a} |",
         f"| only B right | {only_b} |", f"| both wrong | {len(shared) - both - only_a - only_b} |", "",
         f"Mean Tanimoto on shared images: A {mean([A[p][0]['match']['tanimoto'] for p in shared]):.3f}, "
         f"B {mean([B[p][0]['match']['tanimoto'] for p in shared]):.3f}; valid A {sum(bool(A[p][0]['match']['valid']) for p in shared)}, "
         f"B {sum(bool(B[p][0]['match']['valid']) for p in shared)}; hit cap A {sum(A[p][1]['hit_cap'] for p in shared)}, B {sum(B[p][1]['hit_cap'] for p in shared)}.",
         "", "## Images where the systems disagree", ""]
    dis = [p for p in shared if A[p][1]["exact_stereo"] != B[p][1]["exact_stereo"]]
    L.append("None: both systems are right or wrong on exactly the same images." if not dis else "")
    L += ["", "## Per-image detail (all shared images, sorted by Tanimoto change B − A)", "",
          "| image | A class | A Tanimoto | B class | B Tanimoto | Δ | truth |", "|---|---|---|---|---|---|---|"]
    for p in sorted(shared, key=lambda p: B[p][0]["match"]["tanimoto"] - A[p][0]["match"]["tanimoto"]):
        ta, tb = A[p][0]["match"]["tanimoto"], B[p][0]["match"]["tanimoto"]
        mark = " **(disagree)**" if p in dis else ""
        L.append(f"| {Path(p).name}{mark} | {A[p][1]['cls']} | {ta:.2f} | {B[p][1]['cls']} | {tb:.2f} | {tb - ta:+.2f} | `{trunc(A[p][0]['truth'], 50)}` |")
    out.write_text("\n".join(L) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+", help="results/<system>.json files written by src.run_system")
    ap.add_argument("--out-dir", default="results/analysis")
    ap.add_argument("--n-worst", type=int, default=12, help="failures on contact_sheet.png (invalid first, then lowest Tanimoto)")
    ap.add_argument("--n-success", type=int, default=6, help="exact matches on successes.png")
    ap.add_argument("--max-tokens", type=int, default=150, help="token cap when match.hit_cap and summary.max_tokens are both absent (smoke runs used 150)")
    ap.add_argument("--image-root", default=".", help="directory that item image_path values are relative to")
    args = ap.parse_args()
    names, ds, infos = [], [], []
    for f in args.results:
        d = json.loads(Path(f).read_text())
        name = Path(f).stem
        cap = d.get("summary", {}).get("max_tokens") or args.max_tokens
        inf = [classify(it, cap) for it in d["items"]]
        out = Path(args.out_dir) / name
        out.mkdir(parents=True, exist_ok=True)
        write_errors(name, d, inf, out)
        pairs = list(zip(d["items"], inf))
        worst = sorted([x for x in pairs if x[1]["cls"] != "exact"], key=lambda x: (bool(x[0]["match"]["valid"]), x[0]["match"]["tanimoto"]))[: args.n_worst]
        if worst:
            contact_sheet(worst, f"{name}: {len(worst)} worst failures (invalid first, then lowest Tanimoto)", out / "contact_sheet.png", Path(args.image_root))
        good = [x for x in pairs if x[1]["cls"] == "exact"][: args.n_success]
        if good:
            contact_sheet(good, f"{name}: exact matches", out / "successes.png", Path(args.image_root))
        counts = Counter(i["cls"] for i in inf)
        print(f"{name}: wrote {out}/ (errors.md, contact_sheet.png{', successes.png' if good else ''}); classes: {dict(counts)}")
        names.append(name), ds.append(d), infos.append(inf)
    if len(ds) >= 2:
        write_paired(names[:2], ds[:2], infos[:2], Path(args.out_dir) / "paired.md")
        print(f"wrote {Path(args.out_dir) / 'paired.md'}")


if __name__ == "__main__":
    main()
