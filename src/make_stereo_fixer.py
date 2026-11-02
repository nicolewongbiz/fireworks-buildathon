"""Build the "stereo fixer" training manifests: image + connectivity-only SMILES in the prompt, full SMILES as label.

  .venv/bin/python -m src.make_stereo_fixer            # data/{train,val}_fix_{384,640}.json from the frozen manifests
  .venv/bin/python -m src.make_stereo_fixer --jsonl    # also data/train_fix_384.jsonl in Fireworks VLM SFT format

The hybrid system (src/predict_hybrid.py) lets Kimi K3 read the skeleton (right 77% of the time on synthetic,
69% on patent scans) and asks a tuned 9B only to add wedge / double-bond stereo given that skeleton. This is
a far easier task than full structure recognition. Rows keep every molecule, including the ~37% with no
stereo, so the model also learns to return the skeleton unchanged.

Row schema: the frozen manifest item plus `prompt` (per-row prompt the trainer should use instead of the
fixed PROMPT) and `skeleton` (= smiles_nostereo). Labels are unchanged (`smiles`).
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIX_PROMPT = (
    "The molecule in this image has the connectivity (SMILES without stereochemistry): {skeleton}\n"
    "Read the wedge bonds and double-bond geometry from the drawing and output ONLY the full canonical "
    "isomeric SMILES with stereochemistry. If the drawing shows no stereochemistry, output the given SMILES."
)


def fixer_prompt(skeleton: str) -> str:
    return FIX_PROMPT.format(skeleton=skeleton)


def build(split: str, res: int) -> list[dict]:
    src = ROOT / f"data/{split}_{res}.json"
    if not src.exists():
        raise SystemExit(f"{src} missing; run `uv run python -m src.make_dataset` first")
    items = json.loads(src.read_text())
    out = [dict(it, prompt=fixer_prompt(it["smiles_nostereo"]), skeleton=it["smiles_nostereo"]) for it in items]
    (ROOT / f"data/{split}_fix_{res}.json").write_text(json.dumps(out, indent=1))
    return out


def export_jsonl(rows: list[dict], path: Path) -> None:
    with path.open("w") as f:
        for it in rows:
            b64 = base64.b64encode((ROOT / it["image_path"]).read_bytes()).decode()
            f.write(json.dumps({"messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": it["prompt"]},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]},
                {"role": "assistant", "content": it["smiles"]}]}) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", action="store_true")
    ap.add_argument("--res", nargs="*", type=int, default=[384, 640])
    args = ap.parse_args()
    for res in args.res:
        for split in ("train", "val"):
            rows = build(split, res)
            with_stereo = sum(r["smiles"] != r["skeleton"] for r in rows)
            print(f"{split}_fix_{res}: {len(rows)} rows, {with_stereo} with stereo ({100*with_stereo/len(rows):.0f}%)")
            if args.jsonl and split == "train":
                export_jsonl(rows, ROOT / f"data/train_fix_{res}.jsonl")


if __name__ == "__main__":
    main()
