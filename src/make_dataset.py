"""Build the scaled molecule-image -> SMILES dataset (train / val / test / drugs) at 384 and 640 px.

DRAFT by Person A, handed to Person B (data owner), who may adopt, change, or replace it. It ran once
successfully on 2026-11-01 (190 s on 6 cores); see data/dataset_report.json for the numbers.

  uv run python -m src.make_dataset                  # full build (10k train, 200 val, 500 test, drugs)
  uv run python -m src.make_dataset --export-jsonl   # also write Fireworks VLM SFT JSONL for train/val

Sources
  ZINC250k (data/raw/zinc250k, Apache-2.0 repo; ZINC is free to use, cite Sterling & Irwin 2015).
  Drugs: SMILES fetched from PubChem by name (public domain), cached in data/raw/pubchem_drugs/.

Labels are RDKit canonical *isomeric* SMILES (stereo kept and drawn with wedges); scoring reports both
stereo-aware and stereo-agnostic exact match. ~70% of charged ZINC molecules are neutralised (real
documents mostly draw neutral forms); the rest keep their charges, and some charged ones become salts
with a counter-ion fragment, since ZINC has no multi-fragment entries.

Split: Bemis-Murcko scaffold groups; test and val take whole small scaffold groups, train gets the
rest, so no scaffold (and no molecule) is shared. Train order is shuffled once, so the first N rows
are a random nested subset (500 < 2,000 < 5,000 < 10,000).

Outputs (data/ is gitignored except scripts' docs)
  data/images/{res}/{split}/{id}.png
  data/{train,val,test_synthetic,test_drugs}_{res}.json   [{"image_path", "smiles", "smiles_nostereo", "tags"}]
  data/dataset_report.json                                 counts, overlap checks, label token stats
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import rdDepictor, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")
rdDepictor.SetPreferCoordGen(True)

ROOT = Path(__file__).resolve().parents[1]
ZINC = ROOT / "data/raw/zinc250k/250k_rndm_zinc_drugs_clean_3.csv"
DRUG_DIR = ROOT / "data/raw/pubchem_drugs"
PROMPT = "Extract the chemical structure from this image and output ONLY the canonical SMILES string."  # = src.predict.PROMPT
RESOLUTIONS = (384, 640)
SEED = 20261102
N_TRAIN, N_VAL, N_TEST = 10_000, 200, 500
MIN_HEAVY, MAX_HEAVY = 10, 40
MAX_TEST_GROUP = 8  # val/test only take scaffold groups at most this big, so they stay diverse
WORKERS = 6
TOKENIZER_URL = "https://huggingface.co/Qwen/Qwen3.5-9B/resolve/c202236235762e1c871ad0ccb60c8ee5ba337b9a/tokenizer.json"

# Well-known small-molecule drugs; resolved to SMILES via PubChem. Includes P-containing drugs.
DRUG_NAMES = """aspirin ibuprofen acetaminophen naproxen diclofenac celecoxib atorvastatin simvastatin rosuvastatin
lisinopril enalapril losartan valsartan amlodipine nifedipine metoprolol propranolol atenolol carvedilol warfarin
clopidogrel apixaban rivaroxaban omeprazole lansoprazole famotidine loratadine cetirizine fexofenadine
diphenhydramine sertraline fluoxetine paroxetine citalopram venlafaxine duloxetine bupropion mirtazapine trazodone
amitriptyline diazepam alprazolam lorazepam zolpidem quetiapine olanzapine risperidone aripiprazole haloperidol
clozapine lamotrigine carbamazepine levetiracetam gabapentin pregabalin topiramate phenytoin donepezil memantine
levodopa sumatriptan morphine codeine oxycodone tramadol fentanyl naloxone buprenorphine methadone caffeine
nicotine sildenafil tadalafil finasteride tamoxifen letrozole anastrozole imatinib gefitinib erlotinib methotrexate
ciprofloxacin levofloxacin amoxicillin ampicillin benzylpenicillin cephalexin doxycycline tetracycline
sulfamethoxazole trimethoprim metronidazole fluconazole acyclovir oseltamivir zidovudine prednisone dexamethasone
hydrocortisone testosterone estradiol progesterone levothyroxine furosemide hydrochlorothiazide spironolactone
montelukast salbutamol theophylline allopurinol colchicine hydroxychloroquine chloroquine sitagliptin glipizide
pioglitazone ondansetron metoclopramide loperamide sofosbuvir cyclophosphamide alendronic_acid tenofovir_disoproxil
ifosfamide fosinopril""".split()

STATE: dict = {}  # per-worker globals (set in worker init)


# ---------- chemistry ----------

def heavy_atoms(m: Chem.Mol) -> int:
    return m.GetNumHeavyAtoms()


def can(m: Chem.Mol, stereo: bool = True) -> str:
    if not stereo:
        m = Chem.Mol(m)
        Chem.RemoveStereochemistry(m)
    return Chem.MolToSmiles(m, isomericSmiles=stereo)


def scaffold_of(m: Chem.Mol) -> str:
    largest = max(Chem.GetMolFrags(m, asMols=True), key=lambda f: f.GetNumHeavyAtoms())
    return MurckoScaffold.MurckoScaffoldSmiles(mol=largest)


def tags(m: Chem.Mol) -> dict:
    ri = m.GetRingInfo()
    rings = [set(r) for r in ri.AtomRings()]
    return {
        "heavy_atoms": heavy_atoms(m),
        "rings": rdMolDescriptors.CalcNumRings(m),
        "stereocenters": len(Chem.FindMolChiralCenters(m, includeUnassigned=False, useLegacyImplementation=False)),
        "ez_bonds": sum(b.GetStereo() != Chem.BondStereo.STEREONONE for b in m.GetBonds()),
        "charged_atoms": sum(a.GetFormalCharge() != 0 for a in m.GetAtoms()),
        "elements": sorted({a.GetSymbol() for a in m.GetAtoms()}),
        "fragments": len(Chem.GetMolFrags(m)),
        "bridged": rdMolDescriptors.CalcNumBridgeheadAtoms(m) > 0,
        "fused": any(len(a & b) >= 2 for i, a in enumerate(rings) for b in rings[i + 1:]),
    }


def prepare_zinc(args: tuple[str, int]) -> dict | None:
    """Parse, vary charge state / salt form deterministically per row, return label + scaffold."""
    smi, idx = args
    m = Chem.MolFromSmiles(smi.strip())
    if m is None or not (MIN_HEAVY <= heavy_atoms(m) <= MAX_HEAVY):
        return None
    rng = random.Random(SEED * 1_000_003 + idx)
    charge = Chem.GetFormalCharge(m)
    if any(a.GetFormalCharge() for a in m.GetAtoms()):
        roll = rng.random()
        if roll < 0.70:
            m = rdMolStandardize.Uncharger().uncharge(m)
        elif roll < 0.85 and charge in (1, -1):
            ion = rng.choice(["[Cl-]", "[Br-]"]) if charge == 1 else rng.choice(["[Na+]", "[K+]"])
            m = Chem.MolFromSmiles(Chem.MolToSmiles(m) + "." + ion)
    if m is None:
        return None
    return {"smiles": can(m), "smiles_nostereo": can(m, stereo=False), "scaffold": scaffold_of(m)}


# ---------- rendering ----------

def style_for(key: str) -> dict:
    rng = random.Random(f"{SEED}:{key}")
    return {
        "line": rng.uniform(1.0, 3.0),  # at 384 px; scaled with resolution
        "font": rng.uniform(0.8, 1.3),  # relative font size
        "rotate": rng.uniform(0, 360),
        "padding": rng.uniform(0.03, 0.18),  # varies apparent molecule size in the frame
        "bw": rng.random() < 0.35,
        "explicit_methyl": rng.random() < 0.10,
    }


def render(smiles: str, path: Path, res: int, style: dict) -> None:
    m = Chem.MolFromSmiles(smiles)
    rdDepictor.Compute2DCoords(m)
    d = rdMolDraw2D.MolDraw2DCairo(res, res)
    o = d.drawOptions()
    scale = res / 384
    o.bondLineWidth = max(1, round(style["line"] * scale))
    o.minFontSize = max(6, round(10 * style["font"] * scale))
    o.maxFontSize = max(o.minFontSize + 2, round(26 * style["font"] * scale))
    o.rotate = style["rotate"]
    o.padding = style["padding"]
    o.explicitMethyl = style["explicit_methyl"]
    if style["bw"]:
        o.useBWAtomPalette()
    rdMolDraw2D.PrepareAndDrawMolecule(d, m)
    d.FinishDrawing()
    path.write_bytes(d.GetDrawingText())


def render_job(job: tuple[str, str, int, str]) -> None:
    smiles, rel_path, res, style_key = job
    p = ROOT / rel_path
    if not p.exists():
        render(smiles, p, res, style_for(style_key))


# ---------- drugs ----------

def fetch_drugs() -> list[dict]:
    import requests

    DRUG_DIR.mkdir(parents=True, exist_ok=True)
    cache = DRUG_DIR / "drugs.csv"
    if cache.exists():
        return list(csv.DictReader(cache.open()))
    rows = []
    for name in DRUG_NAMES:
        q = name.replace("_", " ")
        r = requests.get(f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{q}/property/SMILES/JSON", timeout=30)
        if r.ok:
            p = r.json()["PropertyTable"]["Properties"][0]
            rows.append({"name": q, "cid": p["CID"], "smiles": p["SMILES"]})
        else:
            print(f"  PubChem miss: {q} ({r.status_code})")
        time.sleep(0.25)  # PubChem asks for <= 5 requests/s
    with cache.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "cid", "smiles"])
        w.writeheader()
        w.writerows(rows)
    (DRUG_DIR / "SOURCE.md").write_text(
        "# PubChem drug SMILES\n\nFetched by `src/make_dataset.py` via PUG REST "
        "`compound/name/<name>/property/SMILES/JSON` on "
        f"{time.strftime('%Y-%m-%d')}. PubChem data is public domain (NCBI data usage policy); "
        "cite Kim et al., PubChem 2023 update, NAR 51:D1373.\n"
    )
    return rows


# ---------- main ----------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-jsonl", action="store_true", help="also write Fireworks SFT JSONL for train/val")
    ap.add_argument("--pool", type=int, default=40_000, help="random ZINC rows to consider before splitting")
    args = ap.parse_args()
    t0 = time.time()
    rng = random.Random(SEED)

    # 1. Pool: random ZINC rows plus every P-containing and bridged molecule (rare features).
    zinc = [r["smiles"].strip() for r in csv.DictReader(ZINC.open())]
    order = list(range(len(zinc)))
    rng.shuffle(order)
    rare = [i for i, s in enumerate(zinc) if "P" in s or "p" in s]  # phosphorus (ZINC has no Pd/Pt/Pb)
    pool_idx = list(dict.fromkeys(rare + order[: args.pool]))
    with Pool(WORKERS) as p:
        prepared = p.map(prepare_zinc, [(zinc[i], i) for i in pool_idx], chunksize=500)
    # Bridged molecules are ~1.4% of ZINC; pull more from the rest of the file to reach ~3% of train.
    with Pool(WORKERS) as p:
        extra = p.map(prepare_zinc, [(zinc[i], i) for i in order[args.pool: args.pool + 60_000]], chunksize=500)
    extra = [e for e in extra if e and rdMolDescriptors.CalcNumBridgeheadAtoms(Chem.MolFromSmiles(e["smiles"])) > 0][:400]

    # 2. Drugs: parse, size filter, single fragment; their molecules are excluded from every other split.
    drugs = []
    for row in fetch_drugs():
        m = Chem.MolFromSmiles(row["smiles"])
        if m is None or len(Chem.GetMolFrags(m)) != 1 or not (MIN_HEAVY <= heavy_atoms(m) <= 50):
            continue
        drugs.append({"name": row["name"], "cid": row["cid"], "smiles": can(m), "smiles_nostereo": can(m, False)})
    drugs = drugs[:100]
    drug_keys = {d["smiles_nostereo"] for d in drugs}

    seen: set[str] = set()
    mols = []
    for e in prepared + extra:
        if e and e["smiles_nostereo"] not in seen and e["smiles_nostereo"] not in drug_keys:
            seen.add(e["smiles_nostereo"])  # dedupe on the stereo-free form so stereoisomers can't straddle splits
            mols.append(e)

    # 3. Scaffold split.
    groups: dict[str, list[dict]] = defaultdict(list)
    for e in mols:
        groups[e["scaffold"]].append(e)
    keys = sorted(groups)
    rng.shuffle(keys)
    test, val, train = [], [], []
    for k in keys:
        g = groups[k]
        if k and len(g) <= MAX_TEST_GROUP and len(test) + len(g) <= N_TEST:
            test += g
        elif k and len(g) <= MAX_TEST_GROUP and len(val) + len(g) <= N_VAL:
            val += g
        else:
            train += g
    rng.shuffle(train)
    train = train[:N_TRAIN]
    assert len(test) == N_TEST and len(val) == N_VAL, f"split sizes off: test={len(test)} val={len(val)}"

    # 4. Leakage checks (fail loudly).
    sets = {"train": train, "val": val, "test": test, "drugs": drugs}
    for field in ("smiles", "smiles_nostereo"):
        for a in sets:
            for b in sets:
                if a < b:
                    overlap = {e[field] for e in sets[a]} & {e[field] for e in sets[b]}
                    if overlap:
                        sys.exit(f"LEAKAGE: {len(overlap)} {field} shared between {a} and {b}, e.g. {sorted(overlap)[:3]}")
    scaf = {s: {e["scaffold"] for e in sets[s]} for s in ("train", "val", "test")}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if scaf[a] & scaf[b]:
            sys.exit(f"LEAKAGE: scaffolds shared between {a} and {b}: {sorted(scaf[a] & scaf[b])[:3]}")
    for e in drugs + train + val + test:
        assert Chem.MolFromSmiles(e["smiles"]) is not None, e

    # 5. Render every split at both resolutions (same style per molecule across resolutions).
    split_name = {"train": "train", "val": "val", "test": "test_synthetic", "drugs": "test_drugs"}
    jobs, manifests = [], defaultdict(list)
    for split, items in sets.items():
        for res in RESOLUTIONS:
            (ROOT / f"data/images/{res}/{split}").mkdir(parents=True, exist_ok=True)
        for i, e in enumerate(items):
            m = Chem.MolFromSmiles(e["smiles"])
            t = tags(m)
            for res in RESOLUTIONS:
                rel = f"data/images/{res}/{split}/{i:05d}.png"
                jobs.append((e["smiles"], rel, res, f"{split}:{i}"))
                item = {"image_path": rel, "smiles": e["smiles"], "smiles_nostereo": e["smiles_nostereo"], "tags": t}
                if split == "drugs":
                    item |= {"name": e["name"], "pubchem_cid": e["cid"]}
                manifests[(split, res)].append(item)
    print(f"rendering {len(jobs)} images with {WORKERS} workers ...")
    with Pool(WORKERS) as p:
        p.map(render_job, jobs, chunksize=50)
    for (split, res), items in manifests.items():
        (ROOT / f"data/{split_name[split]}_{res}.json").write_text(json.dumps(items, indent=1))

    if args.export_jsonl:
        for split in ("train", "val"):
            for res in RESOLUTIONS:
                with (ROOT / f"data/{split}_{res}.jsonl").open("w") as f:
                    for it in manifests[(split, res)]:
                        b64 = base64.b64encode((ROOT / it["image_path"]).read_bytes()).decode()
                        f.write(json.dumps({"messages": [
                            {"role": "user", "content": [
                                {"type": "text", "text": PROMPT},
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]},
                            {"role": "assistant", "content": it["smiles"]}]}) + "\n")

    # 6. Label length in Qwen3.5 tokens -> max_tokens cap.
    import requests
    from tokenizers import Tokenizer

    tok = Tokenizer.from_str(requests.get(TOKENIZER_URL, timeout=60).text)
    lengths = {s: [len(tok.encode(e["smiles"], add_special_tokens=False).ids) for e in sets[s]] for s in sets}
    longest = max(max(v) for v in lengths.values())
    longest_item = max((e for s in sets.values() for e in s), key=lambda e: len(tok.encode(e["smiles"], add_special_tokens=False).ids))

    def feature_counts(items):
        c = Counter()
        for e in items:
            t = tags(Chem.MolFromSmiles(e["smiles"]))
            c["stereo"] += t["stereocenters"] > 0 or t["ez_bonds"] > 0
            c["charged"] += t["charged_atoms"] > 0
            c["multi_fragment"] += t["fragments"] > 1
            c["bridged"] += t["bridged"]
            c["fused"] += t["fused"]
            c["P"] += "P" in t["elements"]
            c["S"] += "S" in t["elements"]
            c["halogen"] += bool(set(t["elements"]) & {"F", "Cl", "Br", "I"})
        return dict(c)

    report = {
        "seed": SEED,
        "sizes": {s: len(v) for s, v in sets.items()},
        "scaffolds": {s: len({e["scaffold"] for e in sets[s]}) for s in ("train", "val", "test")},
        "features": {s: feature_counts(v) for s, v in sets.items()},
        "leakage_checks": "passed: zero shared canonical SMILES (with and without stereo) across train/val/test/drugs; zero shared scaffolds across train/val/test",
        "label_tokens": {s: {"max": max(v), "mean": round(sum(v) / len(v), 1)} for s, v in lengths.items()},
        "longest_label": {"tokens": longest, "smiles": longest_item["smiles"]},
        "max_tokens_cap": math.ceil(1.5 * longest),
        "drugs_skipped_or_missing": len(DRUG_NAMES) - len(drugs),
        "build_seconds": round(time.time() - t0),
    }
    (ROOT / "data/dataset_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
