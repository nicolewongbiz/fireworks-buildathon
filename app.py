"""ChemVision-Tuned demo: upload a molecule drawing, see what each system reads.

    .venv/bin/python app.py          # http://127.0.0.1:7860

Systems come from src.predict (open models on Modal vLLM) and src.predict_baselines (Fireworks serverless,
Claude, MolScribe). A system is offered only when its credentials are present in the environment / .env:
  untuned, tuned      MODAL_VLLM_URL + MODAL_VLLM_API_KEY
  large_*             FIREWORKS_API_KEY
  closed_opus         ANTHROPIC_API_KEY
  molscribe           `import molscribe` must succeed (needs .venv-molscribe, not this venv)
"""
from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv
from PIL import Image, ImageDraw
from rdkit import Chem
from rdkit.Chem import Draw

load_dotenv()

from src import predict as open_models  # noqa: E402
from src import predict_baselines as baselines  # noqa: E402
from src.validate import canonicalize, extract_smiles, score  # noqa: E402

ROOT = Path(__file__).resolve().parent
MANIFESTS = ["test_synthetic_384", "test_drugs_384", "test_real_uspto_384", "test_handdrawn_384"]
PER_MANIFEST = 8
DEFAULT_CHECKED = {"tuned", "large_kimi", "closed_opus"}
SYSTEM_ORDER = ["untuned", "tuned", "large_kimi", "large_qwen", "large_glm", "closed_opus", "molscribe"]
MAX_WORKERS = 4
TABLE_HEADERS = ["System", "SMILES", "Valid", "Exact", "Exact (no stereo)", "Tanimoto", "Latency (s)", "USD", "Note"]

# ---------------------------------------------------------------- registry + availability

PREDICTORS: dict = {**open_models.SYSTEMS, **baselines.SYSTEMS}
SYSTEMS = [s for s in SYSTEM_ORDER if s in PREDICTORS]  # untuned_thinking (strawman) is left out of the demo


def _env(*names: str) -> bool:
    return all(os.environ.get(n, "").strip() for n in names)


def availability(name: str) -> str | None:
    """None if the system can run here, else a short reason it cannot."""
    if name.startswith("large_"):
        return None if _env("FIREWORKS_API_KEY") else "FIREWORKS_API_KEY"
    if name.startswith("closed_"):
        return None if _env("ANTHROPIC_API_KEY") or _env("ANTHROPIC_AUTH_TOKEN") else "ANTHROPIC_API_KEY"
    if name == "molscribe":
        return None if importlib.util.find_spec("molscribe") else "molscribe package"
    return None if _env("MODAL_VLLM_URL", "MODAL_VLLM_API_KEY") else "MODAL_VLLM_URL + MODAL_VLLM_API_KEY"


AVAILABLE = {s: availability(s) is None for s in SYSTEMS}

# ---------------------------------------------------------------- examples


def load_examples() -> list[tuple[str, str, str]]:
    """(label, image_path, truth_smiles) for the first PER_MANIFEST items of each manifest present on disk."""
    out = []
    for name in MANIFESTS:
        path = ROOT / "data" / f"{name}.json"
        if not path.exists():
            continue
        kept = 0
        for item in json.loads(path.read_text()):
            img = ROOT / item["image_path"]
            if not img.exists():
                continue
            short = name.removeprefix("test_").removesuffix("_384")
            tag = item.get("name") or Path(item["image_path"]).stem
            out.append((f"{short} / {tag}", str(img), item["smiles"]))
            kept += 1
            if kept >= PER_MANIFEST:
                break
    return out


EXAMPLES = load_examples()
EXAMPLE_BY_PATH = {p: (p, s) for _, p, s in EXAMPLES}


def pick_example(path: str | None):
    if not path or path not in EXAMPLE_BY_PATH:
        return gr.update(), gr.update()
    img, truth = EXAMPLE_BY_PATH[path]
    return Image.open(img).convert("RGB"), truth


# ---------------------------------------------------------------- inference + scoring


def placeholder(text: str, size: int = 300) -> Image.Image:
    im = Image.new("RGB", (size, size), (255, 235, 235))
    d = ImageDraw.Draw(im)
    d.rectangle([4, 4, size - 5, size - 5], outline=(200, 30, 30), width=4)
    d.text((size // 2, size // 2), text, fill=(200, 30, 30), anchor="mm")
    return im


def run_one(name: str, image_path: str, truth: str) -> tuple[list, Image.Image]:
    """Call one predictor and score it. Never raises: errors become the row's Note."""
    t0 = time.perf_counter()
    try:
        out = PREDICTORS[name](image_path)
    except BaseException as e:  # noqa: BLE001  (predict_closed raises SystemExit on auth failure)
        err = f"{type(e).__name__}: {e}"[:200]
        return [name, "", "", "", "", "", f"{time.perf_counter() - t0:.2f}", "", f"ERROR {err}"], placeholder("error")
    raw = (out.get("smiles") or "").strip()
    latency = out.get("latency_s", time.perf_counter() - t0)
    usd = out.get("usd")
    note = "" if raw else "empty output"
    if truth:
        m = score(raw, truth)
        pred_smiles, valid = m.pred_smiles, m.valid
        exact_stereo = valid and canonicalize(pred_smiles, strip_stereo=False) == canonicalize(truth, strip_stereo=False)
        exact_cells = ["yes" if exact_stereo else "no", "yes" if m.exact else "no", f"{m.tanimoto:.2f}"]
    else:
        pred_smiles = extract_smiles(raw)
        valid = canonicalize(pred_smiles) is not None
        exact_cells = ["-", "-", "-"]
    mol = Chem.MolFromSmiles(pred_smiles) if valid else None
    img = Draw.MolToImage(mol, size=(300, 300)) if mol is not None else placeholder("invalid SMILES")
    row = [name, raw[:120], "yes" if valid else "no", *exact_cells, f"{latency:.2f}",
           f"{usd:.5f}" if isinstance(usd, (int, float)) else "", note]
    return row, img


def run(image: Image.Image | None, truth: str, *checks: bool):
    truth = (truth or "").strip()
    chosen = [s for s, c in zip(SYSTEMS, checks) if c and AVAILABLE[s]]
    gallery: list = []
    if image is None:
        return "Upload an image or pick an example first.", [], gallery
    if not chosen:
        missing = [f"{s} ({availability(s)})" for s in SYSTEMS if not AVAILABLE[s]]
        return ("No runnable system selected. Tick at least one configured system. Not configured: "
                + (", ".join(missing) if missing else "none") + ". See .env.example."), [], gallery
    if truth and canonicalize(truth) is None:
        return f"Ground-truth SMILES does not parse: {truth!r}. Fix it or clear the box.", [], gallery
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        image.convert("RGB").save(tmp.name)
    gallery.append((image, "input"))
    try:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            results = list(pool.map(lambda s: run_one(s, tmp.name, truth), chosen))
    finally:
        os.unlink(tmp.name)
    rows = []
    for name, (row, img) in zip(chosen, results):
        rows.append(row)
        gallery.append((img, name))
    truth_note = "" if truth else " (no ground truth given, so no exact-match column)"
    return f"Ran {len(chosen)} system(s){truth_note}.", rows, gallery


# ---------------------------------------------------------------- footer from results/comparison.json


def footer_markdown() -> str:
    path = ROOT / "results" / "comparison.json"
    if not path.exists():
        return "_No results/comparison.json yet; run src/evaluate.py to populate the measured table._"
    try:
        rows = json.loads(path.read_text())["systems"]
    except (json.JSONDecodeError, KeyError) as e:
        return f"_results/comparison.json unreadable: {e}_"
    full = [r for r in rows if r.get("testset") != "smoke"] or rows
    lines = ["**Measured so far** (`results/comparison.json`, exact match on canonical SMILES, stereo-aware):", "",
             "| System | Test set | n | Exact |", "|---|---|---|---|"]
    for r in sorted(full, key=lambda r: (r.get("testset", ""), -(r.get("exact") or 0))):
        lo, hi = r.get("exact_ci") or (None, None)
        ci = f" [{lo:.1%}, {hi:.1%}]" if lo is not None else ""
        lines.append(f"| {r['system']} | {r.get('testset', '')} | {r.get('n', '')} | {r.get('exact', 0):.1%}{ci} |")
    return "\n".join(lines)


# ---------------------------------------------------------------- UI

with gr.Blocks(title="ChemVision-Tuned") as demo:
    gr.Markdown(
        "## ChemVision-Tuned\n"
        "A small open VLM (Qwen3.5-9B + LoRA) fine-tuned to read molecule drawings into SMILES, measured on "
        "cost per correct answer against large hosted models, a closed frontier model, and a specialist OCSR tool."
    )
    with gr.Row():
        with gr.Column(scale=1):
            image_in = gr.Image(type="pil", label="Molecule drawing", height=320)
            example_dd = gr.Dropdown(
                choices=[(label, path) for label, path, _ in EXAMPLES], value=None,
                label=f"Example images ({len(EXAMPLES)} from the committed test manifests)", interactive=True,
            )
            truth_in = gr.Textbox(label="Ground-truth SMILES (optional)", placeholder="CC(=O)Oc1ccccc1C(=O)O")
        with gr.Column(scale=1):
            gr.Markdown("**Systems to run**")
            checks = []
            for s in SYSTEMS:
                ok = AVAILABLE[s]
                checks.append(gr.Checkbox(
                    value=ok and s in DEFAULT_CHECKED, interactive=ok,
                    label=s if ok else f"{s} (not configured: {availability(s)})",
                ))
            run_btn = gr.Button("Run", variant="primary")
            status = gr.Markdown()
    table = gr.Dataframe(headers=TABLE_HEADERS, value=[], interactive=False, label="What each system read", wrap=True)
    gallery = gr.Gallery(label="Input and each system's molecule", columns=4, height=340, object_fit="contain")
    gr.Markdown(footer_markdown())

    example_dd.change(pick_example, inputs=example_dd, outputs=[image_in, truth_in])
    run_btn.click(run, inputs=[image_in, truth_in, *checks], outputs=[status, table, gallery])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, theme=gr.themes.Soft())
