"""RDKit parse + canonicalize helpers. Single source of truth for "is this the same molecule".

Ground truth and predictions must both go through `canonicalize` before comparison.
Stereo is stripped by default because a 2D drawing without wedge bonds cannot convey it;
the renderer strips stereo from the source molecule too, so labels match what is drawn.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

_SMILES_RE = re.compile(r"[A-Za-z0-9@+\-\[\]\(\)\\/%=#$.:*]+")


def extract_smiles(text: str) -> str:
    """Pull the most SMILES-looking token out of free-form model output.

    Handles outputs like 'The SMILES is: CCO', '`CCO`', 'SMILES: CCO\n'. Falls back to the
    stripped text. Chooses the longest candidate token that RDKit can parse; if none parse,
    returns the longest token so the failure is still visible.
    """
    if text is None:
        return ""
    text = text.strip().strip("`").strip()
    # Common prefixes
    text = re.sub(r"^(smiles\s*[:=]\s*)", "", text, flags=re.I)
    candidates = _SMILES_RE.findall(text)
    if not candidates:
        return text
    parseable = [c for c in candidates if Chem.MolFromSmiles(c) is not None]
    pool = parseable or candidates
    return max(pool, key=len)


def canonicalize(smiles: str, strip_stereo: bool = True) -> str | None:
    """Return RDKit canonical SMILES, or None if unparseable."""
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if strip_stereo:
        Chem.RemoveStereochemistry(mol)
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def parse_error(smiles: str) -> str | None:
    """Return a short RDKit error message for an unparseable SMILES, or None if it parses.

    Used by the retry loop to tell the model what went wrong.
    """
    if not smiles:
        return "Empty output"
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return "SMILES syntax error: RDKit could not parse the string (check ring closures, brackets, parentheses)."
    try:
        Chem.SanitizeMol(mol)
    except Exception as e:  # noqa: BLE001
        return f"Chemistry error: {type(e).__name__}: {e}"
    return None


@dataclass
class Match:
    pred_raw: str
    pred_smiles: str
    pred_canonical: str | None
    truth_canonical: str
    valid: bool
    exact: bool
    tanimoto: float


def score(pred_text: str, truth_smiles: str) -> Match:
    """Score one prediction against ground truth. Tanimoto is a secondary, graded signal."""
    from rdkit import DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    pred_smiles = extract_smiles(pred_text)
    pred_can = canonicalize(pred_smiles)
    truth_can = canonicalize(truth_smiles)
    if truth_can is None:
        raise ValueError(f"Ground truth does not parse: {truth_smiles!r}")
    valid = pred_can is not None
    exact = valid and pred_can == truth_can
    tanimoto = 0.0
    if valid:
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        fp_p = gen.GetFingerprint(Chem.MolFromSmiles(pred_can))
        fp_t = gen.GetFingerprint(Chem.MolFromSmiles(truth_can))
        tanimoto = DataStructs.TanimotoSimilarity(fp_p, fp_t)
    return Match(pred_text, pred_smiles, pred_can, truth_can, valid, exact, tanimoto)


if __name__ == "__main__":
    import sys

    for s in sys.argv[1:] or ["CC(=O)Oc1ccccc1C(=O)O", "The SMILES is: C1CC1", "not a molecule"]:
        print(repr(s), "->", canonicalize(extract_smiles(s)), "|", parse_error(extract_smiles(s)))
