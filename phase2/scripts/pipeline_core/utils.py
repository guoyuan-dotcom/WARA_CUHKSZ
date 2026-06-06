from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .context import PHASE1_RUNS_DIR, WARA_HANDOFFS_DIR, WARA_PHASE1_TAIL_RUNS_DIR


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def make_run_id(topic: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    suffix = hex(abs(hash(topic)) % 0xFFFFFF)[2:].zfill(6)
    return f"phase2-{stamp}-{suffix}"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore").lstrip("\ufeff")


def looks_like_phase1_run(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "phase3-3" / "hypotheses.md").exists()
        and (path / "phase3-3" / "topic_score.json").exists()
    )


def find_default_phase1_run() -> Path | None:
    if not PHASE1_RUNS_DIR.exists():
        return None
    candidates = [path for path in PHASE1_RUNS_DIR.iterdir() if looks_like_phase1_run(path)]
    if not candidates:
        return None
    candidates.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    return candidates[0]


def looks_like_phase1_handoff(path: Path) -> bool:
    if path.is_dir():
        path = path / "phase1_handoff.json"
    if not (path.is_file() and path.name == "phase1_handoff.json"):
        return False
    payload = read_json(path)
    if not isinstance(payload, dict):
        return False
    return isinstance(payload.get("selected_candidate"), dict) and isinstance(
        payload.get("problem_contract_seed"), dict
    )


def find_default_phase1_handoff() -> Path | None:
    candidates: list[Path] = []
    for root in (WARA_PHASE1_TAIL_RUNS_DIR, WARA_HANDOFFS_DIR):
        if not root.exists():
            continue
        candidates.extend(path for path in root.iterdir() if path.is_dir() and looks_like_phase1_handoff(path))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    return candidates[0]


def resolve_phase1_run_path(raw_path: str | None) -> Path | None:
    if not raw_path:
        return None
    candidate = Path(str(raw_path))
    if candidate.exists():
        return candidate
    fallback = PHASE1_RUNS_DIR / candidate.name
    if fallback.exists():
        return fallback
    return None


def resolve_phase1_handoff_path(raw_path: str | None) -> Path | None:
    if not raw_path:
        return None
    candidate = Path(str(raw_path)).expanduser()
    if looks_like_phase1_handoff(candidate):
        return candidate
    for root in (WARA_PHASE1_TAIL_RUNS_DIR, WARA_HANDOFFS_DIR):
        fallback = root / candidate.name
        if looks_like_phase1_handoff(fallback):
            return fallback
    return None


def read_json(path: Path) -> dict[str, Any] | list[Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(read_text(path).lstrip("\ufeff"))
    except json.JSONDecodeError:
        return None


def compact_text(text: str, limit: int = 2200) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    head = text[: limit - 160].rstrip()
    tail = text[-120:].lstrip()
    return f"{head}\n...\n{tail}"


def extract_python_source(text: str) -> str:
    text = (text or "").strip()
    fenced = re.search(r"```(?:python|py)?\s*(.*?)```", text, flags=re.S | re.I)
    if fenced:
        return fenced.group(1).strip()
    if text.startswith("```"):
        text = text.removeprefix("```python").removeprefix("```py").removeprefix("```").removesuffix("```")
    first_python_line = re.search(
        r"(?m)^(?:from\s+__future__\s+import|import\s+[A-Za-z_]|[A-Za-z_][A-Za-z0-9_]*\s*=|def\s+[A-Za-z_]|class\s+[A-Za-z_]|\"\"\"|''')",
        text,
    )
    if first_python_line and first_python_line.start() > 0:
        text = text[first_python_line.start() :]
    return text.strip()
