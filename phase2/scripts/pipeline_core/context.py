from __future__ import annotations

import os
import sys
from pathlib import Path


PHASE2_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PHASE2_ROOT.parent


def _load_workspace_env() -> None:
    env_path = WORKSPACE_ROOT / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_workspace_env()
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from wara_core.llm.profiles import DEFAULT_MODEL_PROFILE, MODEL_PROFILES, normalize_model_profile  # noqa: E402

PHASE1_ROOT = WORKSPACE_ROOT / "phase1"
PHASE3_ROOT = WORKSPACE_ROOT / "phase3"
ROOT = PHASE2_ROOT
OUTPUTS_ROOT = WORKSPACE_ROOT / "outputs"
PAPER_RUNS_ROOT = OUTPUTS_ROOT / "paper_runs"
SHARED_RUNS_DIR = PAPER_RUNS_ROOT / "shared"
RUNS_DIR = PAPER_RUNS_ROOT / "phase2"
DOCS_DIR = ROOT / "docs"
ENGINE_ROOT = PHASE1_ROOT / "engine"
WARA_LLM_CONFIG_TEMPLATE = PHASE2_ROOT / "config" / "llm.wara.yaml"
PHASE1_RUNS_DIR = PAPER_RUNS_ROOT / "phase1"
PHASE3_RUNS_DIR = PAPER_RUNS_ROOT / "phase3"
FINAL_PAPERS_DIR = PAPER_RUNS_ROOT / "final_papers"
WARA_RUNS_DIR = SHARED_RUNS_DIR
WARA_PHASE1_TAIL_RUNS_DIR = WARA_RUNS_DIR / "phase1_tail"
WARA_HANDOFFS_DIR = WARA_RUNS_DIR / "handoffs"
