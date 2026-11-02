"""Build real-world test manifests from downloaded OCSR benchmarks (patent scans + hand-drawn).

  .venv/bin/python -m src.make_real_test

Sources (data/raw/<source>/SOURCE.md has URL, license, sha256; treat the files as untrusted data)
  uspto_osra          5,719 USPTO patent TIFs + MOL ground truth (OSRA validation set, public-domain images)
  decimer_handdrawn   5,088 hand-drawn PNGs + SMILES TSV (DECIMER HDM v1.2, CC BY 4.0)

For each source: parse every label with RDKit, drop unparsable / Markush (dummy or query atoms) /
>60 heavy atoms / duplicate molecules, then sample N (fixed seed). Images are converted to PNG,
resized so the long side is 384 or 640 px (aspect preserved) and padded to a square on white.

Outputs (same schema as src/make_dataset.py, so src/run_system.py scores them identically)
  data/images/{real_uspto,handdrawn}/{res}/<source id>.png
  data/{test_real_uspto,test_handdrawn}_{res}.json
      [{"image_path", "smiles", "smiles_nostereo", "tags", "source", "source_id"}]
"""
from __future__ import annotations

import csv
import json
import random
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

from PIL import Image
from rdkit import Chem

from src.make_dataset import ROOT, can, tags  # same canonicalization and tag thresholds as the synthetic sets

USPTO = ROOT / "data/raw/uspto_osra/uspto-validation-updated"
HDM = ROOT / "data/raw/decimer_handdrawn"
RESOLUTIONS = (384, 640)
SEED = 20261102
N_PER_SET = 200
MAX_HEAVY = 60
WORKERS = 4

SETS = {  # name -> (manifest stem, image dir, source tag)
    "uspto": ("test_real_uspto", "real_uspto", "uspto_osra"),
    "handdrawn": ("test_handdrawn", "handdrawn", "decimer_handdrawn"),
}


# ---------- labels ----------

def label(m: Chem.Mol | None, skips: Counter) -> dict | None:
    """Canonical labels for one parsed molecule, or None (with the reason counted) if it is excluded."""
    if m is None:
        skips["rdkit_parse_failed"] += 1
        return None
    if any(a.GetAtomicNum() == 0 or a.HasQuery() for a in m.GetAtoms()):
        skips["markush_or_dummy_atom"] += 1
        return None
    if m.GetNumHeavyAtoms() > MAX_HEAVY:
        skips[f"heavy_atoms_gt_{MAX_HEAVY}"] += 1
        return None
    smi = can(m)
    if Chem.MolFromSmiles(smi) is None:  # label must round-trip, since scoring re-parses it
        skips["canonical_smiles_not_reparsable"] += 1
        return None
    return {"smiles": smi, "smiles_nostereo": can(m, stereo=False)}


def load_uspto(skips: Counter) -> list[dict]:
    items = []
    for mol_path in sorted((USPTO / "molfiles").glob("*.MOL")):
        img = USPTO / "images" / (mol_path.stem + ".TIF")
        if not img.exists():
            skips["image_missing"] += 1
            continue
        lab = label(Chem.MolFromMolFile(str(mol_path)), skips)
        if lab:
            items.append(lab | {"source_id": mol_path.stem, "src_image": img})
    return items


def load_handdrawn(skips: Counter) -> list[dict]:
    items = []
    with (HDM / "DECIMER_HDM_Dataset_SMILES.tsv").open(newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            img = HDM / "DECIMER_HDM_Dataset_Images" / f"{row['IDs']}.png"
            if not img.exists():
                skips["image_missing"] += 1
                continue
            lab = label(Chem.MolFromSmiles(row["SMILES"].strip()), skips)
            if lab:
                items.append(lab | {"source_id": row["IDs"], "src_image": img})
    return items


def dedupe_and_sample(items: list[dict], skips: Counter, rng: random.Random) -> list[dict]:
    seen, uniq = set(), []
    for it in items:
        if it["smiles_nostereo"] in seen:
            skips["duplicate_molecule"] += 1
        else:
            seen.add(it["smiles_nostereo"])
            uniq.append(it)
    rng.shuffle(uniq)
    return sorted(uniq[:N_PER_SET], key=lambda it: it["source_id"])


# ---------- images ----------

def convert(job: tuple[Path, Path, int]) -> None:
    src, dst, res = job
    im = Image.open(src)
    if im.mode in ("RGBA", "LA") or "transparency" in im.info:  # flatten alpha onto white
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    im = im.convert("L") if im.mode == "1" else im  # bilevel -> gray so the downscale is anti-aliased
    im = im.convert("RGB")
    scale = res / max(im.size)
    w, h = max(1, round(im.width * scale)), max(1, round(im.height * scale))
    im = im.resize((w, h), Image.LANCZOS)
    canvas = Image.new("RGB", (res, res), (255, 255, 255))
    canvas.paste(im, ((res - w) // 2, (res - h) // 2))
    canvas.save(dst, format="PNG", optimize=True)


# ---------- main ----------

def main() -> None:
    rng = random.Random(SEED)
    loaders = {"uspto": load_uspto, "handdrawn": load_handdrawn}
    chosen, jobs, manifests = {}, [], {}
    for name, (stem, img_dir, source) in SETS.items():
        skips: Counter = Counter()
        pool = loaders[name](skips)
        n_pool = len(pool) + sum(skips.values())
        picked = dedupe_and_sample(pool, skips, rng)
        assert len(picked) == N_PER_SET, f"{name}: only {len(picked)} eligible items"
        print(f"{name}: {n_pool} source items -> {len(pool) - skips['duplicate_molecule']} eligible, "
              f"kept {len(picked)}; skipped {dict(sorted(skips.items()))}")
        chosen[name] = picked
        for res in RESOLUTIONS:
            (ROOT / f"data/images/{img_dir}/{res}").mkdir(parents=True, exist_ok=True)
            manifest = []
            for it in picked:
                rel = f"data/images/{img_dir}/{res}/{it['source_id']}.png"
                jobs.append((it["src_image"], ROOT / rel, res))
                manifest.append({
                    "image_path": rel,
                    "smiles": it["smiles"],
                    "smiles_nostereo": it["smiles_nostereo"],
                    "tags": tags(Chem.MolFromSmiles(it["smiles"])),
                    "source": source,
                    "source_id": it["source_id"],
                })
            manifests[f"data/{stem}_{res}.json"] = manifest

    print(f"converting {len(jobs)} images with {WORKERS} workers ...")
    with Pool(WORKERS) as p:
        p.map(convert, jobs, chunksize=10)
    for rel, manifest in manifests.items():
        (ROOT / rel).write_text(json.dumps(manifest, indent=1))
        print(f"wrote {rel} ({len(manifest)} items)")

    # Leakage check against the training set, with and without stereo. train_*.json is gitignored and
    # regenerated per machine, so a missing file is an error, not a reason to skip the check.
    train_path = ROOT / "data/train_384.json"
    if not train_path.exists():
        sys.exit(f"ERROR: {train_path} not found; build it first with `uv run python -m src.make_dataset` "
                 "so the leakage check can run (it is gitignored, not committed)")
    train = json.loads(train_path.read_text())
    for field in ("smiles", "smiles_nostereo"):
        train_keys = {e[field] for e in train}
        for name, picked in chosen.items():
            overlap = {it[field] for it in picked} & train_keys
            print(f"leakage {name} vs train ({field}): {len(overlap)} shared of {len(picked)}")
            assert not overlap, f"LEAKAGE: {name} shares {len(overlap)} {field} with train, e.g. {sorted(overlap)[:3]}"
    both = {it["smiles_nostereo"] for it in chosen["uspto"]} & {it["smiles_nostereo"] for it in chosen["handdrawn"]}
    print(f"uspto vs handdrawn shared molecules (nostereo): {len(both)}")

    # Spot check: open 3 images per manifest.
    for rel, manifest in manifests.items():
        print(f"\n{rel}")
        for it in manifest[:3]:
            im = Image.open(ROOT / it["image_path"])
            print(f"  {it['image_path']}  {im.size[0]}x{im.size[1]} {im.mode}  heavy={it['tags']['heavy_atoms']}  {it['smiles']}")


if __name__ == "__main__":
    sys.exit(main())
