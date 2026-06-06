from __future__ import annotations


def complete_phase(phases: list[dict], completed_index: int, next_index: int | None = None) -> None:
    phases[completed_index]["status"] = "done"
    if next_index is not None:
        phases[next_index]["status"] = "running"


def block_phase(phases: list[dict], blocked_index: int) -> None:
    phases[blocked_index]["status"] = "blocked"


def skip_phase(phases: list[dict], skipped_index: int) -> None:
    phases[skipped_index]["status"] = "skipped"


def fail_phase(phases: list[dict], failed_index: int, blocked_next_index: int | None = None) -> None:
    phases[failed_index]["status"] = "failed"
    if blocked_next_index is not None:
        phases[blocked_next_index]["status"] = "blocked"


def finish_phase_flow(phases: list[dict], final_index: int) -> None:
    phases[final_index]["status"] = "done"
