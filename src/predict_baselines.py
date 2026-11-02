"""Registry of baseline predictors for src/run_system.py (auto-loaded if present).

Each module below exposes `SYSTEMS = {name: predict}` with the shared predict() contract. Imports are
guarded per module so a missing optional dependency (e.g. MolScribe's torch env) only drops that module's
systems; a genuine bug inside a module still raises.

  large_kimi, large_qwen, large_glm   src/predict_fireworks.py  (Fireworks serverless; needs FIREWORKS_API_KEY)
  molscribe                            src/predict_molscribe.py  (needs .venv-molscribe; see that module)
  closed_opus                          src/predict_closed.py     (Claude Opus 5.5 via Anthropic API; needs ANTHROPIC_API_KEY)
  harness, routed                      src/predict_routed.py     (tuned + vote; routed escalates to Kimi K3 on disagreement)
"""
from __future__ import annotations

import importlib

SYSTEMS: dict = {}
UNAVAILABLE: dict[str, str] = {}

for _mod in ("src.predict_fireworks", "src.predict_molscribe", "src.predict_closed", "src.predict_routed"):
    try:
        SYSTEMS |= importlib.import_module(_mod).SYSTEMS
    except ModuleNotFoundError as e:  # the module itself or one of its optional deps is absent
        UNAVAILABLE[_mod] = f"missing module: {e.name}"
