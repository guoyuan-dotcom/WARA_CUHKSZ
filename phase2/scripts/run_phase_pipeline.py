from __future__ import annotations

import argparse
import json
import os
import re
import traceback
from dataclasses import asdict
from pathlib import Path

import phase_runtime_impl as impl
from pipeline_core import (
    DEFAULT_MODEL_PROFILE,
    DOCS_DIR,
    RUNS_DIR,
    PHASE1_RUNS_DIR,
    Phase2FlowCallbacks,
    Phase2RunState,
    Phase2RunSummary,
    execute_phase2_flow,
    find_default_phase1_handoff,
    find_default_phase1_run,
    looks_like_phase1_handoff,
    looks_like_phase1_run,
    make_phase2_flow_callbacks,
    make_run_id,
    make_phase2_phase_flow,
    read_json,
    read_text,
    normalize_model_profile,
    resolve_phase1_handoff_path,
    resolve_phase1_run_path,
    utcnow_iso,
)

_PHASE2_FLOW_CALLBACKS_TYPE = Phase2FlowCallbacks


PHASE1_BLOCKING_RECOMMENDATIONS = {"revise", "reject", "failed", "fail", "not_ready"}


def normalize_run_id(raw_run_id: str | None) -> str | None:
    value = str(raw_run_id or "").strip()
    if not value:
        return None
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", value):
        raise SystemExit("--run-id must use only letters, numbers, '.', '_' or '-' and be at most 80 characters.")
    return value


def _normalize_gate_value(value: object) -> str:
    return str(value or "").strip().lower().replace(" ", "_").replace("-", "_")


def validate_phase1_quality_gate(phase1_run: Path | None) -> None:
    if phase1_run is None or not phase1_run.exists():
        return
    if os.environ.get("WARA_ALLOW_WEAK_PHASE1", "").strip() == "1":
        return

    topic_score = read_json(phase1_run / "phase3-3" / "topic_score.json") or {}
    review_report = read_json(phase1_run / "phase3-3" / "review_report.json") or {}
    min_score = float(os.environ.get("WARA_PHASE1_MIN_SCORE", "7.0"))
    score_raw = topic_score.get("overall_score")
    try:
        score = float(score_raw)
    except (TypeError, ValueError):
        score = 0.0

    verdict = _normalize_gate_value(topic_score.get("verdict"))
    recommendation = _normalize_gate_value(review_report.get("overall_recommendation"))
    reasons: list[str] = []
    if score < min_score:
        reasons.append(f"overall_score {score:g} is below required {min_score:g}")
    if verdict in PHASE1_BLOCKING_RECOMMENDATIONS:
        reasons.append(f"topic_score verdict is {verdict}")
    if recommendation in PHASE1_BLOCKING_RECOMMENDATIONS:
        reasons.append(f"review recommendation is {recommendation}")
    if reasons:
        raise SystemExit(
            "Phase1 quality gate blocked Phase2 handoff for "
            f"{phase1_run}: " + "; ".join(reasons) + ". "
            "Rerun Phase1 with a stronger topic, or set WARA_ALLOW_WEAK_PHASE1=1 for debugging only."
        )


def validate_wara_phase1_handoff_gate(phase1_handoff: Path | None) -> None:
    if phase1_handoff is None or not phase1_handoff.exists():
        return
    if os.environ.get("WARA_ALLOW_WEAK_PHASE1", "").strip() == "1":
        return
    handoff_file = phase1_handoff / "phase1_handoff.json" if phase1_handoff.is_dir() else phase1_handoff
    payload = read_json(handoff_file) or {}
    selected = payload.get("selected_candidate") if isinstance(payload, dict) else {}
    required = {
        "selected_candidate": selected,
        "problem_contract_seed": payload.get("problem_contract_seed") if isinstance(payload, dict) else None,
        "novelty_contract": payload.get("novelty_contract") if isinstance(payload, dict) else None,
        "proof_contract": payload.get("proof_contract") if isinstance(payload, dict) else None,
        "validation_contract": payload.get("validation_contract") if isinstance(payload, dict) else None,
        "kill_criteria": payload.get("kill_criteria") if isinstance(payload, dict) else None,
    }
    missing = [key for key, value in required.items() if not value]
    if missing:
        raise SystemExit(
            "WARA Phase1 handoff gate blocked Phase2: missing "
            + ", ".join(missing)
            + f" in {handoff_file}."
        )
    if not isinstance(selected, dict) or not str(selected.get("title") or "").strip():
        raise SystemExit(f"WARA Phase1 handoff gate blocked Phase2: selected_candidate.title is missing in {handoff_file}.")
    source_dir = handoff_file.parent
    topic_focused_literature = read_json(source_dir / "topic_focused_literature.json") or {}
    topic_focused_references = read_text(source_dir / "topic_focused_references.bib")
    minimum_reference_target = int(os.environ.get("WARA_PHASE1_REFERENCE_MIN", "12") or 12)
    reference_count = max(
        len(topic_focused_literature.get("references", [])) if isinstance(topic_focused_literature, dict) and isinstance(topic_focused_literature.get("references"), list) else 0,
        len(re.findall(r"@\w+\s*\{", topic_focused_references or "")),
    )
    if reference_count < minimum_reference_target:
        raise SystemExit(
            "WARA Phase1 handoff gate blocked Phase2: "
            f"{reference_count} topic-focused references < hard target {minimum_reference_target} in {source_dir}. "
            "This is a Phase1 LiteratureAgent/handoff error; rerun Phase1 instead of letting Phase2 add references."
        )


def topic_from_wara_phase1_handoff(phase1_handoff: Path) -> str:
    handoff_file = phase1_handoff / "phase1_handoff.json" if phase1_handoff.is_dir() else phase1_handoff
    payload = read_json(handoff_file) or {}
    if isinstance(payload, dict):
        selected = payload.get("selected_candidate")
        if isinstance(selected, dict):
            title = str(selected.get("title") or "").strip()
            if title:
                return title
    summary = read_json(handoff_file.parent / "phase1_tail_summary.json") or {}
    if isinstance(summary, dict):
        title = str(summary.get("selected_title") or summary.get("topic") or "").strip()
        if title:
            return title
    return handoff_file.parent.name


def bootstrap_run(
    topic: str,
    model_profile: str,
    phase1_run: Path | None = None,
    phase1_handoff: Path | None = None,
    stop_phase: str | None = None,
    requested_run_id: str | None = None,
) -> Phase2RunSummary:
    model_profile = normalize_model_profile(model_profile)
    if phase1_handoff is not None:
        validate_wara_phase1_handoff_gate(phase1_handoff)
    else:
        validate_phase1_quality_gate(phase1_run)
    run_id = normalize_run_id(requested_run_id) or make_run_id(topic)
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    summary = Phase2RunSummary(
        run_id=run_id,
        topic=topic,
        created_at=utcnow_iso(),
        root=str(run_dir),
        phase1_run=str(phase1_run) if phase1_run else None,
        model_profile=model_profile,
        phases=make_phase2_phase_flow(),
        phase1_handoff=str(phase1_handoff) if phase1_handoff else None,
    )
    state = Phase2RunState(run_dir, summary)
    state.persist()

    try:
        execute_phase2_flow(
            run_dir=run_dir,
            state=state,
            topic=topic,
            model_profile=model_profile,
            phase1_run=phase1_run,
            docs_dir=DOCS_DIR,
            callbacks=make_phase2_flow_callbacks(impl),
            phase1_handoff=phase1_handoff,
            stop_after_phase=stop_phase,
        )
        blocking_phases = [
            item
            for item in state.phases
            if str(item.get("status") or "").strip().lower() in {"failed", "blocked", "running"}
        ]
        if blocking_phases:
            error_payload = {
                "error_type": "Phase2Blocked",
                "error": "Phase 2 flow stopped before all phases reached done status.",
                "blocking_phases": blocking_phases,
            }
            error_path = run_dir / "phase2_error.json"
            if not error_path.exists():
                error_path.write_text(json.dumps(error_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            raise RuntimeError(
                "Phase 2 flow stopped before completion: "
                + "; ".join(
                    f"{item.get('phase_step')} {item.get('name')}={item.get('status')}" for item in blocking_phases
                )
            )
    except Exception as exc:
        running_index = next((idx for idx, item in enumerate(state.phases) if item.get("status") == "running"), None)
        if running_index is not None:
            state.phases[running_index]["status"] = "failed"
            state.persist()
        (run_dir / "phase2_error.json").write_text(
            json.dumps(
                {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "failed_phase": state.phases[running_index] if running_index is not None else None,
                    "traceback": traceback.format_exc(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        raise
    return summary


def list_phase1_run_dirs() -> list[Path]:
    if not PHASE1_RUNS_DIR.exists():
        return []
    runs = [path for path in PHASE1_RUNS_DIR.iterdir() if path.is_dir()]
    runs.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap Phase 2 wireless theory workspace")
    parser.add_argument("--topic", required=False, help="Research topic for Phase 2")
    parser.add_argument("--phase1-run", required=False, help="Path to the Phase 1 run directory")
    parser.add_argument("--phase1-handoff", required=False, help="Path to a WARA phase1_handoff.json file or its directory")
    parser.add_argument("--model-profile", required=False, default=DEFAULT_MODEL_PROFILE)
    parser.add_argument("--stop-phase", required=False, help="Stop after this phase; run_phase2_pipeline.py defaults to 2.5.")
    parser.add_argument("--run-id", required=False, help="Optional stable WARA run id, e.g. wara001")
    args = parser.parse_args()

    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    if args.phase1_run and args.phase1_handoff:
        raise SystemExit("Use either --phase1-run or --phase1-handoff, not both.")

    phase1_handoff = None
    phase1_run = None
    if args.phase1_handoff:
        phase1_handoff = resolve_phase1_handoff_path(args.phase1_handoff)
        if phase1_handoff is None:
            raise SystemExit(f"Invalid WARA Phase1 handoff: {args.phase1_handoff}")
    elif args.phase1_run:
        phase1_run = Path(args.phase1_run)
    else:
        phase1_handoff = find_default_phase1_handoff()
        if phase1_handoff is None:
            phase1_run = find_default_phase1_run()

    if args.topic:
        topic = args.topic.strip()
    elif phase1_handoff is not None and looks_like_phase1_handoff(phase1_handoff):
        topic = topic_from_wara_phase1_handoff(phase1_handoff)
    elif phase1_run is not None and phase1_run.exists():
        topic_score = read_json(phase1_run / "phase3-3" / "topic_score.json") or {}
        hypotheses_md = read_text(phase1_run / "phase3-3" / "hypotheses.md")
        topic = str(
            topic_score.get("recommended_title")
            or impl.extract_first_candidate_title(hypotheses_md)
            or phase1_run.name
        )
    else:
        raise SystemExit("Either --topic or a valid --phase1-run/default phase1 run must be provided.")

    summary = bootstrap_run(
        topic,
        normalize_model_profile(args.model_profile.strip()),
        phase1_run if phase1_run is not None and phase1_run.exists() else None,
        phase1_handoff if phase1_handoff is not None and looks_like_phase1_handoff(phase1_handoff) else None,
        args.stop_phase.strip() if args.stop_phase else "2.5",
        normalize_run_id(args.run_id),
    )
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
