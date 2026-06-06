from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PHASE1_ROOT = ROOT / "phase1"
ENGINE_ROOT = PHASE1_ROOT / "engine"


def ensure_vendor_engine_path() -> None:
    """Expose vendored implementation modules behind WARA-owned interfaces."""
    engine = str(ENGINE_ROOT)
    if engine not in sys.path:
        sys.path.insert(0, engine)

