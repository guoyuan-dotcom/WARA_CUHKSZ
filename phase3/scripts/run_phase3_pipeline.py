from __future__ import annotations

import argparse
import json
import shutil
import sys
import traceback
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PHASE2_SCRIPTS = ROOT / "phase2" / "scripts"
if str(PHASE2_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(PHASE2_SCRIPTS))

import phase_runtime_impl as impl  # noqa: E402
from pipeline_core import (  # noqa: E402
    DEFAULT_MODEL_PROFILE,
    PHASE3_RUNS_DIR,
    RUNS_DIR,
    Phase2RunState,
    Phase2RunSummary,
    execute_phase3_flow,
    make_phase2_flow_callbacks,
    normalize_model_profile,
    read_json,
    utcnow_iso,
)
from pipeline_core.context import FINAL_PAPERS_DIR  # noqa: E402


PHASE3_OUTPUT_DIR_NAMES = (
    "phase3-1",
    "phase3-2",
    "phase3-3",
    "phase3-4",
    "phase3-5",
    "phase3-6",
    "phase3-figure",
)


def _normalize_run_id(raw_run_id: str | None) -> str | None:
    value = str(raw_run_id or "").strip()
    return value or None


def _latest_phase2_run() -> Path | None:
    if not RUNS_DIR.exists():
        return None
    runs = [
        path
        for path in RUNS_DIR.iterdir()
        if path.is_dir() and (path / "phase2-5" / "phase2_to_phase3_handoff.json").exists()
    ]
    runs.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    return runs[0] if runs else None


def _resolve_phase2_run(value: str | None, run_id: str | None) -> Path:
    if value:
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = (Path.cwd() / candidate).resolve()
        if not candidate.exists():
            by_id = RUNS_DIR / value
            if by_id.exists():
                candidate = by_id
        if candidate.exists():
            return candidate
        raise SystemExit(f"Invalid Phase 2 run directory: {value}")
    if run_id:
        candidate = RUNS_DIR / run_id
        if candidate.exists():
            return candidate
        raise SystemExit(f"No Phase 2 run found for --run-id {run_id}")
    latest = _latest_phase2_run()
    if latest is None:
        raise SystemExit(f"No Phase 2 run with phase2_to_phase3_handoff.json found under {RUNS_DIR}")
    return latest


def _load_state(run_dir: Path) -> Phase2RunState:
    payload = read_json(run_dir / "phase2_summary.json") or {}
    if not isinstance(payload, dict) or not payload:
        raise SystemExit(f"Missing Phase 2 summary: {run_dir / 'phase2_summary.json'}")
    summary = Phase2RunSummary(
        run_id=str(payload.get("run_id") or run_dir.name),
        topic=str(payload.get("topic") or run_dir.name),
        created_at=str(payload.get("created_at") or utcnow_iso()),
        root=str(payload.get("root") or run_dir),
        phase1_run=str(payload.get("phase1_run")) if payload.get("phase1_run") else None,
        model_profile=str(payload.get("model_profile") or DEFAULT_MODEL_PROFILE),
        phases=list(payload.get("phases") or []),
        phase1_handoff=str(payload.get("phase1_handoff")) if payload.get("phase1_handoff") else None,
        selected_title=str(payload.get("selected_title")) if payload.get("selected_title") else None,
    )
    if not summary.phases:
        raise SystemExit(f"Phase 2 summary has no phase flow: {run_dir / 'phase2_summary.json'}")
    return Phase2RunState(run_dir, summary)


def _path_or_none(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.exists() else None


def _remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def _reset_phase3_outputs(state: Phase2RunState) -> None:
    """Start a Phase 3 rerun from a clean Phase-3 controller state."""

    run_dir = state.run_dir
    for dirname in PHASE3_OUTPUT_DIR_NAMES:
        _remove_path(run_dir / dirname)
    for filename in ("phase3_controller_manifest.json", "phase3_error.json"):
        _remove_path(run_dir / filename)
    _remove_path(PHASE3_RUNS_DIR / run_dir.name)
    _remove_path(FINAL_PAPERS_DIR / run_dir.name)
    for phase in state.phases:
        if str(phase.get("phase") or "") == "phase3" or str(phase.get("phase_id") or "").startswith("phase3."):
            phase["status"] = "ready"
    state.persist()


def run_phase3(
    *,
    phase2_run: Path,
    topic: str | None,
    model_profile: str | None,
    stop_phase: str | None,
) -> Phase2RunSummary:
    state = _load_state(phase2_run)
    _reset_phase3_outputs(state)
    chosen_topic = str(topic or state.summary.selected_title or state.summary.topic or phase2_run.name)
    chosen_model = normalize_model_profile(str(model_profile or state.summary.model_profile or DEFAULT_MODEL_PROFILE))
    phase2_to_phase3_handoff = read_json(phase2_run / "phase2-5" / "phase2_to_phase3_handoff.json") or {}
    if not phase2_to_phase3_handoff:
        raise SystemExit(f"Missing Phase 2 to Phase 3 handoff: {phase2_run / 'phase2-5' / 'phase2_to_phase3_handoff.json'}")
    try:
        execute_phase3_flow(
            run_dir=phase2_run,
            state=state,
            topic=chosen_topic,
            model_profile=chosen_model,
            phase1_run=_path_or_none(state.summary.phase1_run),
            callbacks=make_phase2_flow_callbacks(impl),
            phase1_handoff=_path_or_none(state.summary.phase1_handoff),
            phase2_to_phase3_handoff=phase2_to_phase3_handoff,
            stop_after_phase=stop_phase,
        )
    except Exception as exc:
        running_index = next((idx for idx, item in enumerate(state.phases) if item.get("status") == "running"), None)
        if running_index is not None:
            state.phases[running_index]["status"] = "failed"
            state.persist()
        (phase2_run / "phase3_error.json").write_text(
            json.dumps(
                {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "failed_phase": state.phases[running_index] if running_index is not None else None,
                    "traceback": traceback.format_exc(),
                    "updated_at": utcnow_iso(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        raise
    return state.summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run WARA Phase 3 from a frozen Phase 2 handoff")
    parser.add_argument("--phase2-run", required=False, help="Path to a Phase 2 run directory")
    parser.add_argument("--run-id", required=False, help="Phase 2 run id under outputs/paper_runs/phase2")
    parser.add_argument("--topic", required=False, help="Override topic/title for Phase 3")
    parser.add_argument("--model-profile", required=False, help="Override model profile for Phase 3")
    parser.add_argument("--stop-phase", required=False, help="Stop after a Phase 3 step, e.g. 3.1 or 3.4")
    args = parser.parse_args()

    phase2_run = _resolve_phase2_run(args.phase2_run, _normalize_run_id(args.run_id))
    summary = run_phase3(
        phase2_run=phase2_run,
        topic=args.topic.strip() if args.topic else None,
        model_profile=args.model_profile.strip() if args.model_profile else None,
        stop_phase=args.stop_phase.strip() if args.stop_phase else None,
    )
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
