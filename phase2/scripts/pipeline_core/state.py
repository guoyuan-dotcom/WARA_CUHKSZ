from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .executor import block_phase, complete_phase, fail_phase, finish_phase_flow, skip_phase
from .models import Phase2RunSummary
from .utils import write_text


class Phase2RunState:
    def __init__(self, run_dir: Path, summary: Phase2RunSummary) -> None:
        self.run_dir = Path(run_dir)
        self.summary = summary

    @property
    def phases(self) -> list[dict]:
        return self.summary.phases

    def persist(self) -> None:
        write_text(
            self.run_dir / "phase2_summary.json",
            json.dumps(asdict(self.summary), ensure_ascii=False, indent=2),
        )

    def complete_phase(self, completed_index: int, next_index: int | None = None) -> None:
        complete_phase(self.phases, completed_index, next_index)
        self.persist()

    def block_phase(self, blocked_index: int) -> None:
        block_phase(self.phases, blocked_index)
        self.persist()

    def fail_phase(self, failed_index: int, blocked_next_index: int | None = None) -> None:
        fail_phase(self.phases, failed_index, blocked_next_index)
        self.persist()

    def skip_phase(self, skipped_index: int) -> None:
        skip_phase(self.phases, skipped_index)
        self.persist()

    def finish(self, final_index: int) -> None:
        finish_phase_flow(self.phases, final_index)
        self.persist()
