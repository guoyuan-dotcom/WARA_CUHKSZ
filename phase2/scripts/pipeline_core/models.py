from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .contracts import PHASE_FLOW


@dataclass
class Phase2RunSummary:
    run_id: str
    topic: str
    created_at: str
    root: str
    phase1_run: str | None
    model_profile: str
    phases: list[dict[str, Any]]
    phase1_handoff: str | None = None
    selected_title: str | None = None


def make_phase2_phase_flow() -> list[dict[str, Any]]:
    phases: list[dict[str, Any]] = []
    for item in PHASE_FLOW:
        index = item["phase_step"]
        phases.append(
            {
                "phase_step": index,
                "phase_id": item["phase_id"],
                "phase": item["phase"],
                "phase_name": item["phase_name"],
                "name": item["name"],
                "status": "running" if index == 1 else "ready",
            }
        )
    return phases
