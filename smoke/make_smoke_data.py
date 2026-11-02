"""Throwaway 50-molecule dataset for Person A's platform smoke test.

Not the real dataset (that is Person B's src/render.py + src/build_dataset.py). Uses the same
file contracts so the pipeline can be exercised end to end:
  data/smoke_train.jsonl   Fireworks VLM SFT format
  data/smoke_test.json     [{"image_path": ..., "smiles": <canonical>}]
"""
from __future__ import annotations

import base64
import csv
import json
import random
import sys
from pathlib import Path

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.validate import canonicalize  # noqa: E402

PROMPT = "Extract the chemical structure from this image and output ONLY the canonical SMILES string."
SEED = 1101
N_TRAIN, N_TEST = 40, 10
SIZE = 384
SRC = ROOT / "data/raw/zinc250k/250k_rndm_zinc_drugs_clean_3.csv"
IMG_DIR = ROOT / "data/images/smoke"


def render(smiles: str, path: Path, rng: random.Random) -> None:
    mol = Chem.MolFromSmiles(smiles)
    d = rdMolDraw2D.MolDraw2DCairo(SIZE, SIZE)
    opts = d.drawOptions()
    opts.bondLineWidth = rng.choice([1, 2, 3])
    opts.minFontSize = rng.choice([10, 12, 14])
    opts.rotate = rng.uniform(0, 360)
    if rng.random() < 0.3:
        opts.useBWAtomPalette()
    d.DrawMolecule(mol)
    d.FinishDrawing()
    path.write_bytes(d.GetDrawingText())


def sft_row(img_path: Path, smiles: str) -> dict:
    b64 = base64.b64encode(img_path.read_bytes()).decode()
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            },
            {"role": "assistant", "content": smiles},
        ]
    }


def main() -> None:
    rng = random.Random(SEED)
    with SRC.open() as f:
        rows = [r["smiles"].strip() for r in csv.DictReader(f)]
    rng.shuffle(rows)

    picked: list[str] = []
    seen: set[str] = set()
    for s in rows:
        can = canonicalize(s)
        if can and can not in seen:
            seen.add(can)
            picked.append(can)
        if len(picked) == N_TRAIN + N_TEST:
            break
    train, test = picked[:N_TRAIN], picked[N_TRAIN:]
    assert not set(train) & set(test)

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    with (ROOT / "data/smoke_train.jsonl").open("w") as f:
        for i, s in enumerate(train):
            p = IMG_DIR / f"train_{i:03d}.png"
            render(s, p, rng)
            f.write(json.dumps(sft_row(p, s)) + "\n")
    test_items = []
    for i, s in enumerate(test):
        p = IMG_DIR / f"test_{i:03d}.png"
        render(s, p, rng)
        test_items.append({"image_path": str(p.relative_to(ROOT)), "smiles": s})
    (ROOT / "data/smoke_test.json").write_text(json.dumps(test_items, indent=2))

    # Validate what we wrote
    lines = (ROOT / "data/smoke_train.jsonl").read_text().splitlines()
    for line in lines:
        row = json.loads(line)
        assert [m["role"] for m in row["messages"]] == ["user", "assistant"]
        assert canonicalize(row["messages"][1]["content"]) == row["messages"][1]["content"]
    size_kb = (ROOT / "data/smoke_train.jsonl").stat().st_size / 1024
    print(f"train rows: {len(lines)}, test items: {len(test_items)}, train.jsonl: {size_kb:.0f} KB")


if __name__ == "__main__":
    main()
