from __future__ import annotations

import csv
import json
import os
import re
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import yaml

from .controller import AgentSpec, ArtifactRef, ControllerDecision, GateResult, WaraController
from .controller import _review_repair_round_limit
from .context import DEFAULT_MODEL_PROFILE, FINAL_PAPERS_DIR, PHASE3_RUNS_DIR, WORKSPACE_ROOT
from .handoff import build_wara_phase1_handoff
from .state import Phase2RunState
from .subagents import (
    audit_implementation_contract,
    audit_model_contract,
    audit_phase25_evidence,
    audit_theory_contract,
    build_algorithm_contract,
    build_claim_map,
    build_experiment_design_contract,
    build_problem_contract,
    build_phase24_execution_contract,
    build_tractability_route_policy,
    contract_prompt_block,
    select_wireless_benchmark_plan,
    write_json_artifact,
)
from .utils import read_json, read_text, utcnow_iso, write_text

if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from wara_core.agents import ExperimentAgent, RoleAgent  # noqa: E402


PAPER_READY_PHASE25_STATUSES = {
    "paper_minimum_ready",
    "paper_preferred_ready",
    "high_confidence_ready",
}

DEFAULT_PHASE2_CONTRACT_REPAIR_ROUNDS = 3
DEFAULT_PHASE3_1_WRITING_REPAIR_ROUNDS = 3
DEFAULT_PHASE3_TO_PHASE2_ROUTE_ROUNDS = 0
PHASE3_PHASE_DIR_NAMES = (
    "phase3-1",
    "phase3-2",
    "phase3-3",
    "phase3-4",
    "phase3-5",
    "phase3-6",
    "phase3-figure",
)
FINAL_PAPER_ARTIFACTS = (
    ("phase3-6/revised_full_paper_preview.pdf", "paper.pdf"),
    ("phase3-6/revised_full_paper.tex", "paper.tex"),
    ("phase3-6/revised_full_paper_expanded_for_review.tex", "paper_expanded_for_review.tex"),
    ("phase3-6/revised_full_paper_preview.log", "paper_compile.log"),
    ("phase3-6/abstract.tex", "abstract.tex"),
    ("phase3-6/introduction.tex", "introduction.tex"),
    ("phase3-6/system_model_problem_formulation_section.tex", "system_model_problem_formulation_section.tex"),
    ("phase3-6/proposed_solution_section.tex", "proposed_solution_section.tex"),
    ("phase3-6/numerical_results_section.tex", "numerical_results_section.tex"),
    ("phase3-6/conclusion.tex", "conclusion.tex"),
    ("phase3-6/conceptual_diagram.tex", "conceptual_diagram.tex"),
    ("phase3-6/phase3_6_manifest.json", "phase3_6_manifest.json"),
    ("phase3-6/phase3_6_quality_gate.json", "phase3_6_quality_gate.json"),
    ("phase3-6/post_revision_full_paper_quality_gate.json", "post_revision_full_paper_quality_gate.json"),
    ("phase3-6/full_paper_abbreviation_report.json", "full_paper_abbreviation_report.json"),
    ("phase3-6/contract_scope_report.json", "contract_scope_report.json"),
    ("phase3-4/references.bib", "references.bib"),
    ("phase3-4/verified_references.bib", "verified_references.bib"),
    ("phase3-4/reference_quality_report.json", "reference_quality_report.json"),
    ("phase2_summary.json", "phase2_summary.json"),
)

CONTROLLER_INDEX_MANIFEST = "controller_manifest.json"
PHASE2_CONTROLLER_MANIFEST = "phase2_controller_manifest.json"
PHASE3_CONTROLLER_MANIFEST = "phase3_controller_manifest.json"
PHASE2_CONTROLLER_AGENT_IDS = {
    "formulation_agent",
    "theory_agent",
    "experiment_agent",
    "validation_agent",
}
PHASE3_CONTROLLER_AGENT_IDS = {
    "analysis_agent",
    "writing_agent",
    "literature_agent",
    "review_agent",
    "repair_agent",
}


REVIEW_OWNER_PHASE_INDEX = {
    "formulation_agent": 0,
    "theory_agent": 2,
    "experiment_agent": 3,
    "validation_agent": 4,
    "literature_agent": 8,
}


def _phase24_repair_round_limit() -> int:
    for env_name in ("WARA_PHASE24_REPAIR_ROUNDS", "WCL_PHASE24_REPAIR_ROUNDS"):
        raw_value = os.environ.get(env_name)
        if raw_value is None:
            continue
        try:
            return max(0, int(raw_value))
        except ValueError:
            continue
    return 10


def _phase2_contract_repair_round_limit() -> int:
    for env_name in ("WARA_PHASE2_CONTRACT_REPAIR_ROUNDS", "WCL_PHASE2_CONTRACT_REPAIR_ROUNDS"):
        raw_value = os.environ.get(env_name)
        if raw_value is None:
            continue
        try:
            return max(0, int(raw_value))
        except ValueError:
            continue
    return DEFAULT_PHASE2_CONTRACT_REPAIR_ROUNDS


def _phase3_1_writing_repair_round_limit() -> int:
    for env_name in ("WARA_PHASE3_1_WRITING_REPAIR_ROUNDS", "WCL_PHASE3_1_WRITING_REPAIR_ROUNDS"):
        raw_value = os.environ.get(env_name)
        if raw_value is None:
            continue
        try:
            return max(0, int(raw_value))
        except ValueError:
            continue
    return DEFAULT_PHASE3_1_WRITING_REPAIR_ROUNDS


def _phase3_to_phase2_route_round_limit() -> int:
    for env_name in ("WARA_PHASE3_TO_PHASE2_ROUTE_ROUNDS", "WARA_PHASE3_UPSTREAM_ROUTE_ROUNDS"):
        raw_value = os.environ.get(env_name)
        if raw_value is None:
            continue
        try:
            return max(0, int(raw_value))
        except ValueError:
            continue
    return DEFAULT_PHASE3_TO_PHASE2_ROUTE_ROUNDS


def _copytree_fresh(source: Path, target: Path) -> None:
    if target.is_symlink() or target.exists():
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        else:
            target.unlink()
    shutil.copytree(
        source,
        target,
        ignore=shutil.ignore_patterns(
            "__pycache__",
            "*.aux",
            "*.bbl",
            "*.blg",
            "*.fdb_latexmk",
            "*.fls",
            "*.synctex.gz",
        ),
    )


def _sync_phase3_public_run(run_dir: Path) -> Path:
    """Expose canonical Phase 3 artifacts under outputs/paper_runs/phase3/<run_id>."""

    public_run_dir = PHASE3_RUNS_DIR / run_dir.name
    public_run_dir.mkdir(parents=True, exist_ok=True)
    source_marker = public_run_dir / "source_phase2_run.txt"
    if not source_marker.exists() or source_marker.read_text(encoding="utf-8").strip() != str(run_dir):
        source_marker.write_text(str(run_dir) + "\n", encoding="utf-8")
    synced_phase_dirs: list[str] = []
    for phase_dir_name in PHASE3_PHASE_DIR_NAMES:
        source_phase_dir = run_dir / phase_dir_name
        public_phase_dir = public_run_dir / phase_dir_name
        if not source_phase_dir.exists():
            continue
        _copytree_fresh(source_phase_dir, public_phase_dir)
        synced_phase_dirs.append(phase_dir_name)
    manifest = {
        "run_id": run_dir.name,
        "source_phase2_run": str(run_dir),
        "phase3_run_dir": str(public_run_dir),
        "controller_index": str(run_dir / CONTROLLER_INDEX_MANIFEST),
        "phase2_controller_manifest": str(run_dir / PHASE2_CONTROLLER_MANIFEST),
        "phase3_controller_manifest": str(run_dir / PHASE3_CONTROLLER_MANIFEST),
        "phase2_to_phase3_handoff": str(run_dir / "phase2-5" / "phase2_to_phase3_handoff.json"),
        "synced_phase_dirs": synced_phase_dirs,
        "updated_at": utcnow_iso(),
        "layout": {
            "phase3_artifacts": "outputs/paper_runs/phase3/<run_id>/phase3-*",
            "final_package": "outputs/paper_runs/final_papers/<run_id>",
        },
    }
    write_text(public_run_dir / "phase3_public_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return public_run_dir


def _publish_final_paper_package(run_dir: Path) -> Path | None:
    """Publish a clean final paper package for humans and batch manifests."""

    phase3_run_dir = _sync_phase3_public_run(run_dir)
    review_routing_decision = read_json(run_dir / "phase3-5" / "review_routing_decision.json") or {}
    controller_review_decision = read_json(run_dir / "phase3-5" / "controller_review_decision.json") or {}
    phase3_6_quality_gate = read_json(run_dir / "phase3-6" / "phase3_6_quality_gate.json") or {}
    phase3_6_manifest = read_json(run_dir / "phase3-6" / "phase3_6_manifest.json") or {}
    review_routes = review_routing_decision.get("routes") if isinstance(review_routing_decision, dict) else []
    if not isinstance(review_routes, list):
        review_routes = []
    route_budget = read_json(run_dir / "phase3-5" / "phase3_to_phase2_route_budget.json") or {}
    route_attempts = route_budget.get("attempts") if isinstance(route_budget, dict) else []
    if not isinstance(route_attempts, list):
        route_attempts = []
    active_phase2_route_consumed = bool(route_budget.get("active_phase2_route_consumed", False)) if isinstance(route_budget, dict) else False
    phase3_to_phase2_route_limit = (
        int(route_budget.get("limit", _phase3_to_phase2_route_round_limit()) or 0)
        if isinstance(route_budget, dict)
        else _phase3_to_phase2_route_round_limit()
    )
    phase3_to_phase2_route_attempts_used = len(route_attempts)
    upstream_routes = [
        route
        for route in review_routes
        if isinstance(route, dict)
        and str(route.get("target_agent", "")).strip() not in {"writing_agent", "repair_agent", ""}
    ]
    upstream_repair_recommended = bool(upstream_routes) or (
        isinstance(controller_review_decision, dict)
        and str(controller_review_decision.get("owner_agent", "")).strip() not in {"", "writing_agent", "repair_agent"}
        and str(controller_review_decision.get("action", "")).strip() == "repair"
    )
    phase3_6_gate_passed = (
        isinstance(phase3_6_quality_gate, dict)
        and str(phase3_6_quality_gate.get("status", "")).strip() == "passed"
    )
    ready_to_submit_estimate = bool(phase3_6_manifest.get("ready_to_submit_estimate", False)) if isinstance(phase3_6_manifest, dict) else False
    submission_ready = bool(phase3_6_gate_passed and ready_to_submit_estimate and not upstream_repair_recommended)
    paper_package_status = (
        "submission_ready"
        if submission_ready
        else "exported_with_phase2_route_consumed"
        if active_phase2_route_consumed
        else "exported_with_upstream_repair_recommended"
        if upstream_repair_recommended
        else "exported_with_review_findings"
    )
    phase3_6_revision_source = run_dir / "phase3-6" / "revised_full_paper_preview.pdf"
    phase3_6_source = run_dir / "phase3-5" / "full_paper_preview.pdf"
    phase3_5_source = run_dir / "phase3-4" / "full_paper_preview.pdf"
    if phase3_6_revision_source.exists():
        final_source = phase3_6_revision_source
    elif phase3_6_source.exists():
        final_source = phase3_6_source
    elif phase3_5_source.exists():
        final_source = phase3_5_source
    else:
        final_source = phase3_6_revision_source
    if not final_source.exists():
        return None
    package_dir = FINAL_PAPERS_DIR / run_dir.name
    package_dir.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    for relative_source, target_name in FINAL_PAPER_ARTIFACTS:
        source = run_dir / relative_source
        if not source.exists():
            continue
        target = package_dir / target_name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied[target_name] = str(target)
    phase_preview_artifacts = (
        ("phase3-5/full_paper_preview.pdf", "paper.pdf"),
        ("phase3-5/full_paper.tex", "paper.tex"),
        ("phase3-5/full_paper_expanded_for_review.tex", "paper_expanded_for_review.tex"),
        ("phase3-5/full_paper_preview.log", "paper_compile.log"),
        ("phase3-5/abstract.tex", "abstract.tex"),
        ("phase3-5/introduction.tex", "introduction.tex"),
        ("phase3-5/system_model_problem_formulation_section.tex", "system_model_problem_formulation_section.tex"),
        ("phase3-5/proposed_solution_section.tex", "proposed_solution_section.tex"),
        ("phase3-5/numerical_results_section.tex", "numerical_results_section.tex"),
        ("phase3-5/conclusion.tex", "conclusion.tex"),
        ("phase3-5/conceptual_diagram.tex", "conceptual_diagram.tex"),
        ("phase3-5/full_paper_abbreviation_report.json", "full_paper_abbreviation_report.json"),
        ("phase3-5/full_paper_abbreviation_repair_report.json", "full_paper_abbreviation_repair_report.json"),
        ("phase3-4/abstract.tex", "abstract.tex"),
        ("phase3-4/introduction.tex", "introduction.tex"),
        ("phase3-4/system_model_problem_formulation_section.tex", "system_model_problem_formulation_section.tex"),
        ("phase3-4/proposed_solution_section.tex", "proposed_solution_section.tex"),
        ("phase3-4/numerical_results_section.tex", "numerical_results_section.tex"),
        ("phase3-4/conclusion.tex", "conclusion.tex"),
        ("phase3-4/conceptual_diagram.tex", "conceptual_diagram.tex"),
        ("phase3-4/full_paper_preview.pdf", "paper.pdf"),
        ("phase3-4/full_paper_preview.tex", "paper.tex"),
        ("phase3-4/full_paper_preview.log", "paper_compile.log"),
    )
    for relative_source, target_name in phase_preview_artifacts:
        if target_name in copied:
            continue
        source = run_dir / relative_source
        if not source.exists():
            continue
        target = package_dir / target_name
        shutil.copy2(source, target)
        copied[target_name] = str(target)
    figures_source = run_dir / "phase2-5" / "figures"
    if figures_source.exists():
        _copytree_fresh(figures_source, package_dir / "figures")
        copied["figures"] = str(package_dir / "figures")
    manifest = {
        "run_id": run_dir.name,
        "source_phase2_run": str(run_dir),
        "phase3_run": str(phase3_run_dir),
        "controller_index": str(run_dir / CONTROLLER_INDEX_MANIFEST),
        "phase2_controller_manifest": str(run_dir / PHASE2_CONTROLLER_MANIFEST),
        "phase3_controller_manifest": str(run_dir / PHASE3_CONTROLLER_MANIFEST),
        "final_package": str(package_dir),
        "final_pdf": copied.get("paper.pdf"),
        "final_tex": copied.get("paper.tex"),
        "expanded_review_tex": copied.get("paper_expanded_for_review.tex"),
        "references_bib": copied.get("references.bib"),
        "figures": copied.get("figures"),
        "paper_package_status": paper_package_status,
        "submission_ready": submission_ready,
        "upstream_repair_recommended": upstream_repair_recommended,
        "upstream_repair_routes": upstream_routes,
        "phase3_to_phase2_route_limit": phase3_to_phase2_route_limit,
        "phase3_to_phase2_route_attempts_used": phase3_to_phase2_route_attempts_used,
        "phase3_to_phase2_route_active": active_phase2_route_consumed,
        "phase3_to_phase2_route_budget": str(run_dir / "phase3-5" / "phase3_to_phase2_route_budget.json")
        if (run_dir / "phase3-5" / "phase3_to_phase2_route_budget.json").exists()
        else None,
        "review_routing_decision": str(run_dir / "phase3-5" / "review_routing_decision.json")
        if (run_dir / "phase3-5" / "review_routing_decision.json").exists()
        else None,
        "controller_review_decision": str(run_dir / "phase3-5" / "controller_review_decision.json")
        if (run_dir / "phase3-5" / "controller_review_decision.json").exists()
        else None,
        "paper_export_policy": (
            "After Phase 2 hands off to Phase 3, WARA exports a paper package. "
            "The review report may still recommend scoped upstream repair, and that recommendation "
            "is recorded separately from paper-package export."
        ),
        "copied_artifacts": copied,
        "updated_at": utcnow_iso(),
    }
    write_text(package_dir / "final_package_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return package_dir


def _write_controller_index(
    run_dir: Path,
    *,
    topic: str,
    model_profile: str,
    active_phase: str,
    phase1_handoff: Path | None = None,
    phase1_run: Path | None = None,
) -> None:
    """Write a lightweight pointer manifest for tools that still expect controller_manifest.json."""

    index_path = run_dir / CONTROLLER_INDEX_MANIFEST
    existing = read_json(index_path) or {}
    payload = {
        **(existing if isinstance(existing, dict) else {}),
        "controller_version": "wara_split_controller_index_v1",
        "phase": "phase2_phase3_split",
        "active_phase": active_phase,
        "topic": topic,
        "model_profile": model_profile,
        "run_dir": str(run_dir),
        "phase1_handoff": str(phase1_handoff) if phase1_handoff else None,
        "phase1_run": str(phase1_run) if phase1_run else None,
        "phase2_controller_manifest": str(run_dir / PHASE2_CONTROLLER_MANIFEST),
        "phase3_controller_manifest": str(run_dir / PHASE3_CONTROLLER_MANIFEST),
        "phase2_to_phase3_handoff": str(run_dir / "phase2-5" / "phase2_to_phase3_handoff.json"),
        "updated_at": utcnow_iso(),
    }
    write_text(index_path, json.dumps(payload, ensure_ascii=False, indent=2))


def _filter_controller_agents(controller: WaraController, allowed_ids: set[str]) -> None:
    agents = controller.manifest.get("agents")
    if not isinstance(agents, dict):
        return
    for agent_id in list(agents):
        if agent_id not in allowed_ids:
            agents.pop(agent_id, None)
    controller.persist()


def _write_phase2_to_phase3_handoff(
    run_dir: Path,
    *,
    phase25_gate_ok: bool,
    phase25_status: str,
    evidence_audit: dict[str, Any],
    phase25_auto_expansion: dict[str, Any] | None,
) -> dict[str, Any]:
    handoff = {
        "handoff_version": "wara_phase2_to_phase3_v1",
        "run_id": run_dir.name,
        "created_at": utcnow_iso(),
        "phase2_controller_manifest": str(run_dir / PHASE2_CONTROLLER_MANIFEST),
        "frozen_inputs": {
            "phase1_handoff_manifest": "phase1_handoff_manifest.json",
            "mathematical_contract": "phase2-1/mathematical_contract.json",
            "frozen_math_interface": "phase2-1/frozen_math_interface.md",
            "problem_contract": "phase2-1/problem_contract.json",
            "algorithm_contract": "phase2-2/algorithm_contract.json",
            "algorithm_description": "phase2-3/algorithm.md",
            "claim_map": "phase2-3/claim_map.json",
            "evidence_package": "phase2-5/phase25_experiment_summary.json",
            "evidence_audit": "phase2-5/evidence_audit.json",
        },
        "phase2_gate_summary": {
            "phase25_gate_ok": bool(phase25_gate_ok),
            "phase25_status": phase25_status,
            "evidence_audit_ok": bool(evidence_audit.get("ok", False)) if isinstance(evidence_audit, dict) else False,
            "evidence_audit_errors": evidence_audit.get("errors", []) if isinstance(evidence_audit, dict) else [],
            "evidence_audit_warnings": evidence_audit.get("warnings", []) if isinstance(evidence_audit, dict) else [],
            "auto_expansion": phase25_auto_expansion or {},
        },
        "phase3_policy": {
            "consume_frozen_interfaces_only": True,
            "no_new_experiments": True,
            "no_new_metrics_or_baselines": True,
            "scope_claims_to_phase2_evidence": True,
        },
    }
    write_text(run_dir / "phase2-5" / "phase2_to_phase3_handoff.json", json.dumps(handoff, ensure_ascii=False, indent=2))
    return handoff


def _init_phase2_controller(
    *,
    run_dir: Path,
    topic: str,
    model_profile: str,
    phase1_handoff: Path | None,
    phase1_run: Path | None,
) -> WaraController:
    controller = WaraController(run_dir, manifest_name=PHASE2_CONTROLLER_MANIFEST)
    controller.manifest["phase"] = "phase2"
    controller.manifest["phase_architecture"] = {
        "phase1": "Research direction and handoff",
        "phase2": "Modeling, solution, implementation, and experimental evidence",
        "phase3": "Handled by a separate Phase 3 controller manifest.",
        "phase2_to_phase3_handoff": str(run_dir / "phase2-5" / "phase2_to_phase3_handoff.json"),
        "separate_phase3_controller_manifest": str(run_dir / PHASE3_CONTROLLER_MANIFEST),
    }
    controller.manifest["topic"] = topic
    controller.manifest["model_profile"] = model_profile
    controller.manifest["run_dir"] = str(run_dir)
    controller.manifest["phase1_handoff"] = str(phase1_handoff) if phase1_handoff else None
    controller.manifest["phase1_run"] = str(phase1_run) if phase1_run else None
    controller.manifest["bounded_repair_policy"] = {
        "phase2_contract_repair_rounds": _phase2_contract_repair_round_limit(),
        "phase3_1_writing_repair_rounds": _phase3_1_writing_repair_round_limit(),
        "phase24_implementation_repair_rounds": _phase24_repair_round_limit(),
        "phase25_auto_paper_sweep_rounds": _phase25_auto_paper_run_limit(),
        "phase25_auto_paper_sweep_mode": _phase25_auto_paper_sweep_mode(),
        "phase3_5_review_repair_rounds": _review_repair_round_limit(),
        "policy": (
            "repair reported local issues up to the round limit; when the same gate still fails, "
            "record the blocker and promote the selected artifact so the run can still export a PDF"
        ),
        "env_overrides": [
            "WARA_PHASE2_CONTRACT_REPAIR_ROUNDS",
            "WARA_PHASE3_1_WRITING_REPAIR_ROUNDS",
            "WARA_PHASE24_REPAIR_ROUNDS",
            "WARA_PHASE25_AUTO_PAPER_RUNS",
            "WARA_PHASE25_AUTO_PAPER_MODE",
            "WARA_MAX_REVIEW_REPAIR_ROUNDS",
        ],
    }
    controller.manifest["updated_at"] = utcnow_iso()
    controller.persist()
    _register_phase2_controller_agents(controller)
    _filter_controller_agents(controller, PHASE2_CONTROLLER_AGENT_IDS)
    _write_controller_index(
        run_dir,
        topic=topic,
        model_profile=model_profile,
        active_phase="phase2",
        phase1_handoff=phase1_handoff,
        phase1_run=phase1_run,
    )
    return controller


def _init_phase3_controller(
    *,
    run_dir: Path,
    topic: str,
    model_profile: str,
    phase1_handoff: Path | None,
    phase1_run: Path | None,
    phase2_to_phase3_handoff: dict[str, Any],
) -> WaraController:
    controller = WaraController(run_dir, manifest_name=PHASE3_CONTROLLER_MANIFEST)
    controller.manifest["phase"] = "phase3"
    controller.manifest["phase_architecture"] = {
        "phase1": "Research direction and handoff",
        "phase2": "Completed by the separate Phase 2 controller.",
        "phase3": "Research synthesis, reference integration, review, repair, and export.",
        "phase2_controller_manifest": str(run_dir / PHASE2_CONTROLLER_MANIFEST),
        "phase2_to_phase3_handoff": str(run_dir / "phase2-5" / "phase2_to_phase3_handoff.json"),
        "phase3_public_run_dir": str(_sync_phase3_public_run(run_dir)),
        "final_paper_package_dir": str(FINAL_PAPERS_DIR / run_dir.name),
        "handoff_summary": phase2_to_phase3_handoff.get("phase2_gate_summary", {}),
    }
    controller.manifest["topic"] = topic
    controller.manifest["model_profile"] = model_profile
    controller.manifest["run_dir"] = str(run_dir)
    controller.manifest["phase1_handoff"] = str(phase1_handoff) if phase1_handoff else None
    controller.manifest["phase1_run"] = str(phase1_run) if phase1_run else None
    controller.manifest["bounded_repair_policy"] = {
        "phase3_1_writing_repair_rounds": _phase3_1_writing_repair_round_limit(),
        "phase3_5_review_repair_rounds": _review_repair_round_limit(),
        "phase3_to_phase2_route_rounds": _phase3_to_phase2_route_round_limit(),
        "policy": (
            "consume frozen Phase 2 interfaces; repair local synthesis issues up to the round limit; "
            "export a paper package after Phase 2 handoff; record upstream technical or evidence defects "
            "as review recommendations by default; only open an active Phase 2 route when the route-round budget is positive"
        ),
        "env_overrides": [
            "WARA_PHASE3_1_WRITING_REPAIR_ROUNDS",
            "WARA_MAX_REVIEW_REPAIR_ROUNDS",
            "WARA_PHASE3_TO_PHASE2_ROUTE_ROUNDS",
            "WARA_PHASE3_UPSTREAM_ROUTE_ROUNDS",
        ],
    }
    controller.manifest["updated_at"] = utcnow_iso()
    controller.persist()
    _register_phase2_controller_agents(controller)
    _filter_controller_agents(controller, PHASE3_CONTROLLER_AGENT_IDS)
    _register_phase1_handoff_artifact(controller)
    _register_phase21_controller_artifacts(controller, freeze_contracts=True)
    _register_phase22_controller_artifacts(controller, freeze_contracts=True)
    _register_phase23_controller_artifacts(controller, freeze_claim_map=True)
    _register_phase24_design_artifacts(controller, freeze_contracts=True)
    _register_phase24_implementation_artifacts(controller)
    _register_phase25_controller_artifacts(controller)
    _register_controller_artifact(
        controller,
        "phase2_to_phase3_handoff",
        "phase2-5/phase2_to_phase3_handoff.json",
        kind="json",
        producer="phase2_controller",
        frozen=True,
        reason="Phase 3 consumes this frozen handoff from the Phase 2 controller.",
    )
    _record_controller_gate(
        controller,
        "phase2_to_phase3_handoff_gate",
        ok=True,
        artifact_ids=["phase2_to_phase3_handoff", "mathematical_contract", "algorithm_contract", "phase25_experiment_summary"],
        warnings=phase2_to_phase3_handoff.get("phase2_gate_summary", {}).get("evidence_audit_warnings", []),
    )
    _write_controller_index(
        run_dir,
        topic=topic,
        model_profile=model_profile,
        active_phase="phase3",
        phase1_handoff=phase1_handoff,
        phase1_run=phase1_run,
    )
    return controller


def _register_phase2_controller_agents(controller: WaraController) -> None:
    specs = [
        AgentSpec(
            id="formulation_agent",
            role="Construct the original wireless system model, optimization problem, and frozen mathematical contract.",
            input_artifacts=["phase1_handoff_manifest"],
            output_artifacts=[
                "system_model",
                "problem_formulation",
                "core_theory_package",
                "mathematical_contract",
                "problem_contract",
                "model_audit",
            ],
            tools=["llm", "deterministic_contract_gate"],
            allowed_actions=[
                "define controls, parameters, derived quantities, objective, and constraints",
                "freeze the original mathematical interface after the formulation gate passes",
            ],
            forbidden_actions=[
                "introduce reformulation-only variables into the original problem",
                "change the selected Phase 1 research direction",
                "invent unsupported wireless mechanisms",
            ],
            acceptance_criteria=[
                "mathematical contract is valid JSON",
                "optimizer variables are controls",
                "objective and constraints are physically meaningful",
            ],
        ),
        AgentSpec(
            id="theory_agent",
            role="Audit tractability, define reformulation route, algorithm contract, and scoped proof claims.",
            input_artifacts=["mathematical_contract", "problem_contract"],
            output_artifacts=[
                "tractability_route_policy",
                "convexity_audit",
                "reformulation_path",
                "algorithm_contract",
                "algorithm_description",
                "claim_map",
            ],
            tools=["llm", "math_consistency_checker"],
            frozen_contracts=["mathematical_contract"],
            allowed_actions=[
                "introduce reformulation-only objects with scope",
                "define algorithm execution requirements",
                "state assumptions for convergence or optimality claims",
            ],
            forbidden_actions=[
                "change the frozen original objective",
                "move surrogate variables into the system model",
                "claim global optimality without scoped assumptions",
            ],
            acceptance_criteria=[
                "algorithm contract has an execution contract",
                "reformulation does not rewrite the original problem",
                "claim map is scoped to generated theory artifacts",
            ],
        ),
        AgentSpec(
            id="experiment_agent",
            role="Generate executable experiments from frozen contracts, including benchmarks, sweeps, validation, and figures.",
            input_artifacts=["mathematical_contract", "algorithm_contract", "claim_map"],
            output_artifacts=[
                "wireless_benchmark_plan",
                "experiment_design_contract",
                "validation_plan",
                "phase24_execution_contract",
                "generated_plugin",
                "phase24_validation_manifest",
                "implementation_audit",
            ],
            tools=["llm", "python", "validation_harness", "plotting"],
            frozen_contracts=["mathematical_contract", "algorithm_contract"],
            allowed_actions=[
                "choose paper-level KPIs from the frozen problem semantics",
                "run scout validation before downstream evidence packaging",
                "record invalid benchmark removal reasons",
            ],
            forbidden_actions=[
                "change the mathematical objective",
                "reuse topic-specific metrics from unrelated runs",
                "promote feasibility-only diagnostics as primary system-performance evidence",
            ],
            acceptance_criteria=[
                "validation plan is parseable",
                "generated plugin imports and satisfies the harness",
                "finite metrics and evidence metadata are produced",
            ],
        ),
        AgentSpec(
            id="validation_agent",
            role="Package Phase 2.4 experiment outputs into Phase 3.1-ready evidence without regenerating experiments.",
            input_artifacts=["phase24_validation_manifest", "implementation_audit"],
            output_artifacts=["phase25_experiment_summary", "evidence_audit"],
            tools=["json_reader", "csv_reader", "deterministic_evidence_gate"],
            frozen_contracts=["mathematical_contract", "algorithm_contract"],
            allowed_actions=[
                "read Phase 2.4 outputs",
                "package figure evidence and benchmark definitions",
                "block downstream writing if evidence is not paper-ready",
            ],
            forbidden_actions=[
                "generate new experiment code",
                "change benchmark identities",
                "claim gains not supported by Phase 2.4 data",
            ],
            acceptance_criteria=[
                "Phase 2.5 status is paper-ready",
                "evidence audit is OK",
            ],
        ),
        AgentSpec(
            id="analysis_agent",
            role="Interpret frozen numerical evidence and map verified trends to scoped research claims.",
            input_artifacts=[
                "phase25_experiment_summary",
                "evidence_audit",
                "phase2_to_phase3_handoff",
            ],
            output_artifacts=[
                "phase3_2_numerical_results",
                "claim_evidence_map",
                "figure_observations_summary",
            ],
            tools=["llm", "json_reader", "csv_reader", "claim_evidence_gate"],
            frozen_contracts=["phase25_experiment_summary"],
            allowed_actions=[
                "write numerical-results interpretation from verified figures only",
                "scope empirical claims to the frozen evidence package",
                "record trend, regime, limitation, and claim-evidence links",
            ],
            forbidden_actions=[
                "invent new numerical values",
                "add new metrics, baselines, sweeps, or experiments",
                "strengthen claims beyond Phase 2.5 evidence",
            ],
            acceptance_criteria=[
                "numerical prose references only verified figures and plotted methods",
                "claim statements are traceable to Phase 2.5 records",
                "limitations remain inside the frozen evidence scope",
            ],
        ),
        AgentSpec(
            id="writing_agent",
            role="Draft paper sections from frozen technical contracts and verified experimental evidence.",
            input_artifacts=[
                "mathematical_contract",
                "algorithm_contract",
                "phase25_experiment_summary",
                "phase3_2_numerical_results",
            ],
            output_artifacts=[
                "phase3_2_numerical_results",
                "phase3_3_technical_sections",
                "phase3_4_full_paper",
            ],
            tools=["llm", "latex_preview", "deterministic_writing_gates"],
            frozen_contracts=["mathematical_contract", "algorithm_contract", "phase25_experiment_summary"],
            allowed_actions=[
                "write concise IEEE-style prose from verified artifacts",
                "use only plotted methods and verified figure evidence",
                "preserve notation, method names, references, and claim scope",
            ],
            forbidden_actions=[
                "invent new benchmarks or experimental claims",
                "introduce notation/acronyms not defined in the paper",
                "rewrite technical contracts while drafting prose",
            ],
            acceptance_criteria=[
                "numerical prose uses only paper-ready figures",
                "abstract/conclusion align with plotted evidence",
                "technical sections compile and avoid internal pipeline language",
            ],
        ),
        AgentSpec(
            id="literature_agent",
            role="Verify and place references across introduction and technical sections.",
            input_artifacts=["phase1_handoff_manifest", "phase3_3_technical_sections"],
            output_artifacts=["verified_reference_bank", "references_bib", "citation_claim_map"],
            tools=["reference_verifier", "llm_reference_checker", "bibtex_renderer"],
            allowed_actions=[
                "select verified references from the run's literature source",
                "place citations in both introduction and technical sections",
                "build a BibTeX-backed final reference list",
            ],
            forbidden_actions=[
                "lower the hard minimum reference target",
                "insert unverifiable references",
                "use topic-specific seed references outside the supplied literature source",
            ],
            acceptance_criteria=[
                "final paper cites at least the hard minimum number of verified references",
                "technical sections contain relevant citations where needed",
                "BibTeX and citation keys are consistent",
            ],
        ),
        AgentSpec(
            id="review_agent",
            role="Review the full paper and route failures to the owning role agent.",
            input_artifacts=["phase3_4_full_paper", "references_bib", "phase25_experiment_summary"],
            output_artifacts=["phase3_5_review", "review_routing_decision"],
            tools=["llm_reviewer", "deterministic_full_paper_gate"],
            allowed_actions=[
                "identify technical, evidence, writing, reference, and compile issues",
                "route each blocking issue to the owning agent",
            ],
            forbidden_actions=[
                "silently rewrite content",
                "route upstream technical failures to writing polish",
            ],
            acceptance_criteria=[
                "review report separates critical, major, and minor issues",
                "routing decision names an owning agent and repair scope",
            ],
        ),
        AgentSpec(
            id="repair_agent",
            role="Apply bounded repairs to the artifact selected by the controller.",
            input_artifacts=["phase3_5_review", "review_routing_decision"],
            output_artifacts=["phase3_6_revision", "phase3_6_quality_gate"],
            tools=["llm_repair", "latex_preview", "post_revision_gate"],
            frozen_contracts=["mathematical_contract", "algorithm_contract", "phase25_experiment_summary"],
            allowed_actions=[
                "repair only the controller-selected writing/reference artifact",
                "preserve frozen technical and evidence contracts",
                "rerun quality checks after revision",
            ],
            forbidden_actions=[
                "change the experiment story to hide weak evidence",
                "modify frozen math or algorithm contracts",
                "add new unverified references",
            ],
            acceptance_criteria=[
                "unresolved issue count is zero",
                "compile, reference, abbreviation, and evidence statuses are OK",
            ],
        ),
    ]
    for spec in specs:
        controller.register_agent(spec)


def _register_controller_artifact(
    controller: WaraController,
    artifact_id: str,
    relative_path: str,
    *,
    kind: str,
    producer: str,
    frozen: bool = False,
    reason: str = "",
) -> None:
    path = controller.run_dir / relative_path
    if not path.exists():
        return
    controller.register_artifact(
        ArtifactRef(
            id=artifact_id,
            path=relative_path,
            kind=kind,
            producer=producer,
            frozen=frozen,
            reason=reason,
        )
    )
    if frozen:
        controller.freeze_artifact(artifact_id, reason=reason or f"{artifact_id} frozen by controller.")


def _record_controller_gate(
    controller: WaraController,
    gate_id: str,
    *,
    ok: bool,
    artifact_ids: list[str],
    errors: list[Any] | None = None,
    warnings: list[Any] | None = None,
) -> ControllerDecision:
    return controller.record_gate(
        GateResult(
            gate_id=gate_id,
            ok=ok,
            artifact_ids=artifact_ids,
            errors=[str(item) for item in (errors or [])],
            warnings=[str(item) for item in (warnings or [])],
        )
    )


def _register_phase1_handoff_artifact(controller: WaraController) -> None:
    _register_controller_artifact(
        controller,
        "phase1_handoff_manifest",
        "phase1_handoff_manifest.json",
        kind="json",
        producer="scout_agent",
        frozen=True,
        reason="Phase 1 handoff is the frozen research-direction input for Phase 2.",
    )


def _register_phase21_controller_artifacts(controller: WaraController, *, freeze_contracts: bool) -> None:
    _register_controller_artifact(controller, "system_model", "phase2-1/system_model.md", kind="markdown", producer="formulation_agent")
    _register_controller_artifact(
        controller,
        "problem_formulation",
        "phase2-1/problem_formulation.md",
        kind="markdown",
        producer="formulation_agent",
    )
    _register_controller_artifact(
        controller,
        "core_theory_package",
        "phase2-1/core_theory_package.md",
        kind="markdown",
        producer="formulation_agent",
    )
    _register_controller_artifact(
        controller,
        "mathematical_contract",
        "phase2-1/mathematical_contract.json",
        kind="json",
        producer="formulation_agent",
        frozen=freeze_contracts,
        reason="Formulation gate passed; downstream phases must not rewrite the original problem.",
    )
    _register_controller_artifact(
        controller,
        "frozen_math_interface",
        "phase2-1/frozen_math_interface.md",
        kind="markdown",
        producer="formulation_agent",
        frozen=freeze_contracts,
        reason="Human-readable frozen mathematical interface derived from mathematical_contract.",
    )
    _register_controller_artifact(
        controller,
        "problem_contract",
        "phase2-1/problem_contract.json",
        kind="json",
        producer="formulation_agent",
        frozen=freeze_contracts,
        reason="Problem-contract summary passed the formulation audit.",
    )
    _register_controller_artifact(controller, "model_audit", "phase2-1/model_audit.json", kind="json", producer="formulation_gate")


def _register_phase22_controller_artifacts(controller: WaraController, *, freeze_contracts: bool) -> None:
    _register_controller_artifact(
        controller,
        "tractability_route_policy",
        "phase2-2/tractability_route_policy.json",
        kind="json",
        producer="theory_agent",
    )
    _register_controller_artifact(controller, "convexity_audit", "phase2-2/convexity_audit.md", kind="markdown", producer="theory_agent")
    _register_controller_artifact(
        controller,
        "reformulation_path",
        "phase2-2/reformulation_path.md",
        kind="markdown",
        producer="theory_agent",
    )
    _register_controller_artifact(
        controller,
        "algorithm_contract",
        "phase2-2/algorithm_contract.json",
        kind="json",
        producer="theory_agent",
        frozen=freeze_contracts,
        reason="Algorithm execution contract passed the theory-interface gate.",
    )


def _register_phase23_controller_artifacts(controller: WaraController, *, freeze_claim_map: bool) -> None:
    _register_controller_artifact(controller, "algorithm_description", "phase2-3/algorithm.md", kind="markdown", producer="theory_agent")
    _register_controller_artifact(
        controller,
        "convergence_or_complexity",
        "phase2-3/convergence_or_complexity.md",
        kind="markdown",
        producer="theory_agent",
    )
    _register_controller_artifact(controller, "theory_audit", "phase2-3/theory_audit.json", kind="json", producer="theory_gate")
    _register_controller_artifact(
        controller,
        "claim_map",
        "phase2-3/claim_map.json",
        kind="json",
        producer="theory_agent",
        frozen=freeze_claim_map,
        reason="Theory gate passed; downstream analysis must use this claim map as claim scope.",
    )


def _register_phase24_design_artifacts(controller: WaraController, *, freeze_contracts: bool) -> None:
    for artifact_id, relative_path in (
        ("wireless_benchmark_plan", "phase2-4/wireless_benchmark_plan.json"),
        ("experiment_design_contract", "phase2-4/experiment_design_contract.json"),
        ("phase24_execution_contract", "phase2-4/phase24_execution_contract.json"),
        ("phase24_validation_source_contracts", "phase2-4/phase24_validation_source_contracts.json"),
    ):
        _register_controller_artifact(
            controller,
            artifact_id,
            relative_path,
            kind="json",
            producer="experiment_agent",
            frozen=freeze_contracts,
            reason="Phase 2.4 preflight design contract passed and is fixed for implementation.",
        )
    _register_controller_artifact(
        controller,
        "validation_plan",
        "phase2-4/validation_plan.yaml",
        kind="yaml",
        producer="experiment_agent",
        frozen=freeze_contracts,
        reason="Validation plan passed Phase 2.4 design gate and must be consumed by the implementation.",
    )


def _register_phase24_implementation_artifacts(controller: WaraController) -> None:
    _register_controller_artifact(
        controller,
        "generated_plugin",
        "phase2-4/solver/generated_plugin.py",
        kind="code",
        producer="experiment_agent",
    )
    _register_controller_artifact(
        controller,
        "phase24_validation_manifest",
        "phase2-4/phase24_validation_manifest.json",
        kind="json",
        producer="validation_harness",
    )
    _register_controller_artifact(
        controller,
        "implementation_audit",
        "phase2-4/implementation_audit.json",
        kind="json",
        producer="implementation_gate",
    )


def _register_phase25_controller_artifacts(controller: WaraController) -> None:
    _register_controller_artifact(
        controller,
        "phase25_experiment_summary",
        "phase2-5/phase25_experiment_summary.json",
        kind="json",
        producer="validation_agent",
        frozen=True,
        reason="Phase25 evidence gate passed or blocked based on this summary.",
    )
    _register_controller_artifact(
        controller,
        "evidence_audit",
        "phase2-5/evidence_audit.json",
        kind="json",
        producer="validation_gate",
    )


def _register_phase3_1_controller_artifacts(controller: WaraController) -> None:
    _register_controller_artifact(
        controller,
        "phase3_1_system_model_problem_formulation",
        "phase3-1/system_model_problem_formulation_ieee_wcl.tex",
        kind="latex",
        producer="writing_agent",
    )
    _register_controller_artifact(
        controller,
        "phase3_1_proposed_solution",
        "phase3-1/proposed_solution_ieee_wcl.tex",
        kind="latex",
        producer="writing_agent",
    )
    _register_controller_artifact(
        controller,
        "phase3_1_preview_manifest",
        "phase3-1/phase3_1_preview_manifest.json",
        kind="json",
        producer="latex_preview_gate",
    )


def _register_phase3_2_controller_artifacts(controller: WaraController) -> None:
    _register_controller_artifact(
        controller,
        "phase3_2_numerical_results",
        "phase3-2/numerical_results_section.tex",
        kind="latex",
        producer="analysis_agent",
    )
    _register_controller_artifact(
        controller,
        "phase3_2_manifest",
        "phase3-2/phase3_2_manifest.json",
        kind="json",
        producer="analysis_agent",
    )
    _register_controller_artifact(
        controller,
        "phase3_2_evidence_gate",
        "phase3-2/phase3_2_evidence_gate.json",
        kind="json",
        producer="analysis_gate",
    )


def _phase3_2_has_conservative_llm_candidate(run_dir: Path) -> bool:
    phase3_2_dir = Path(run_dir) / "phase3-2"
    manifest = read_json(phase3_2_dir / "phase3_2_manifest.json") or {}
    section_path = phase3_2_dir / "numerical_results_section.tex"
    if not isinstance(manifest, dict) or not manifest or not section_path.exists():
        return False
    text = section_path.read_text(encoding="utf-8", errors="ignore")
    if len(text.strip()) < 80:
        return False
    if (phase3_2_dir / "phase3_2_llm_generation_error.txt").exists():
        return False
    return True


def _phase3_2_evidence_scope_warning_exists(run_dir: Path) -> bool:
    phase3_2_dir = Path(run_dir) / "phase3-2"
    return (
        (phase3_2_dir / "phase3_2_evidence_scope_warning.md").exists()
        or (phase3_2_dir / "phase3_2_bounded_repair_evidence_warning.md").exists()
    )


def _register_phase3_3_controller_artifacts(controller: WaraController) -> None:
    for artifact_id, relative_path in (
        ("phase3_3_abstract", "phase3-3/abstract.tex"),
        ("phase3_3_keywords", "phase3-3/keywords.tex"),
        ("phase3_3_conclusion", "phase3-3/conclusion.tex"),
        ("phase3_3_manifest", "phase3-3/phase3_3_manifest.json"),
        ("phase3_3_technical_sections", "phase3-3/phase3_3_technical_sections_preview.tex"),
    ):
        _register_controller_artifact(
            controller,
            artifact_id,
            relative_path,
            kind="latex" if relative_path.endswith(".tex") else "json",
            producer="writing_agent",
        )


def _register_phase3_4_controller_artifacts(controller: WaraController) -> None:
    for artifact_id, relative_path, kind, producer in (
        ("phase3_4_introduction", "phase3-4/introduction.tex", "latex", "writing_agent"),
        ("phase3_4_full_paper", "phase3-4/full_paper_preview.tex", "latex", "writing_agent"),
        ("phase3_4_preview_pdf", "phase3-4/full_paper_preview.pdf", "pdf", "latex_preview_gate"),
        ("references_bib", "phase3-4/references.bib", "bibtex", "literature_agent"),
        ("verified_reference_bank", "phase3-4/verified_reference_bank.json", "json", "literature_agent"),
        ("citation_claim_map", "phase3-4/citation_claim_map.json", "json", "literature_agent"),
        ("reference_quality_report", "phase3-4/reference_quality_report.json", "json", "reference_gate"),
        (
            "phase3_4_reference_count_contract",
            "phase3-4/phase3_4_reference_count_contract_report.json",
            "json",
            "reference_gate",
        ),
        ("phase3_4_manifest", "phase3-4/phase3_4_manifest.json", "json", "writing_agent"),
    ):
        _register_controller_artifact(controller, artifact_id, relative_path, kind=kind, producer=producer)


def _register_phase3_6_controller_artifacts(controller: WaraController) -> None:
    for artifact_id, relative_path, kind in (
        ("phase3_6_revision_manifest", "phase3-6/phase3_6_manifest.json", "json"),
        ("phase3_6_quality_gate", "phase3-6/phase3_6_quality_gate.json", "json"),
        ("phase3_6_revised_full_paper", "phase3-6/revised_full_paper.tex", "latex"),
        ("phase3_6_revised_pdf", "phase3-6/revised_full_paper_preview.pdf", "pdf"),
    ):
        _register_controller_artifact(controller, artifact_id, relative_path, kind=kind, producer="repair_agent")


def _phase3_3_manifest_gate(manifest: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(manifest, dict) or not manifest:
        return {"ok": True, "errors": [], "warnings": ["phase3_3_manifest was unavailable; controller recorded a compatibility-pass gate."]}
    errors: list[str] = []
    warnings: list[str] = []
    check_keys = [
        "forbidden_terms_check",
        "claim_strength_check",
        "abstract_structure_check",
        "conclusion_structure_check",
        "paper_objective_alignment_check",
    ]
    for key in check_keys:
        check = manifest.get(key)
        if not isinstance(check, dict):
            warnings.append(f"{key} missing from phase3_3_manifest.")
            continue
        passed = check.get("passed")
        if passed is None:
            passed = check.get("ok")
        if passed is False:
            errors.append(f"{key} failed: {json.dumps(check, ensure_ascii=False)}")
    return {"ok": not errors, "errors": errors, "warnings": warnings}


def _phase3_4_manifest_gate(run_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    phase3_4_dir = Path(run_dir) / "phase3-4"
    errors: list[str] = []
    warnings: list[str] = []

    def _safe_int(value: Any) -> int:
        try:
            return int(value or 0)
        except Exception:
            return 0

    if not isinstance(manifest, dict) or not manifest:
        manifest = read_json(phase3_4_dir / "phase3_4_manifest.json") or {}
    if not isinstance(manifest, dict) or not manifest:
        return {"ok": True, "errors": [], "warnings": ["phase3_4_manifest was unavailable; controller recorded a compatibility-pass gate."]}

    forbidden = manifest.get("forbidden_terms_check")
    if isinstance(forbidden, dict) and forbidden.get("ok") is False:
        errors.append(f"phase3_4 forbidden_terms_check failed: {json.dumps(forbidden, ensure_ascii=False)}")
    intro = manifest.get("introduction_structure_check")
    if isinstance(intro, dict):
        if intro.get("has_section") is False:
            errors.append("phase3_4 introduction_structure_check failed: missing Introduction section.")
        if _safe_int(intro.get("citation_count", 0)) <= 0:
            errors.append("phase3_4 introduction_structure_check failed: no introduction citations.")
        if _safe_int(intro.get("paragraph_count", 0)) < 3:
            warnings.append("phase3_4 introduction has fewer than three paragraphs.")

    reference_count = read_json(phase3_4_dir / "phase3_4_reference_count_contract_report.json") or {}
    if isinstance(reference_count, dict) and reference_count:
        if reference_count.get("ok") is False:
            errors.extend(str(item) for item in reference_count.get("errors", []))
    else:
        warnings.append("phase3_4 reference count contract report missing.")

    abbreviation = manifest.get("full_paper_abbreviation_check")
    if isinstance(abbreviation, dict):
        unresolved = abbreviation.get("undefined_abbreviations") or abbreviation.get("unresolved_abbreviations") or []
        if unresolved:
            warnings.append(
                "full-paper abbreviation check has unresolved abbreviations; "
                f"route to Phase 3.5/3.6 review repair: {unresolved}"
            )

    missing_references = manifest.get("missing_reference_keys")
    if missing_references:
        errors.append(f"phase3_4 has missing reference keys: {missing_references}")

    return {"ok": not errors, "errors": errors, "warnings": warnings}


def _phase3_1_compile_issues_should_block(issue_summary: str) -> bool:
    """Phase 3.1 is a writing preview; polish-only issues should not block experiments."""

    text = str(issue_summary or "").strip().lower()
    if not text:
        return False
    blocking_markers = [
        "fatal error",
        "emergency stop",
        "latex error",
        "missing $",
        "runaway argument",
        "undefined control sequence",
        "no pages of output",
        "output pdf was not created",
    ]
    return any(marker in text for marker in blocking_markers)


def _phase3_1_has_concrete_uncertainty_model(text: str) -> bool:
    return bool(
        re.search(
            r"ellipsoidal|norm[- ]bounded|box uncertainty|polyhedral|wasserstein|cantelli|chebyshev|bernstein|gaussian error|sub-gaussian|scenario|sample approximation|sample-average|moment[- ]based|mean and covariance|support set",
            str(text or ""),
            flags=re.I,
        )
    )


def _phase3_1_ambiguous_multi_equation_leadins(tex: str) -> list[str]:
    """Find multi-line displays introduced by a single underspecified lead-in."""

    issues: list[str] = []
    pattern = re.compile(
        r"(?P<lead>[^.\n]{0,220}(?:define|denote|defined as|denoted by|given by|updates?|updated|written as)[^.\n]{0,120})\n\\begin\{align\}(?P<body>.*?)\\end\{align\}(?P<after>[^A-Za-z0-9]{0,20}[A-Za-z]{0,40})",
        flags=re.I | re.S,
    )
    for match in pattern.finditer(str(tex or "")):
        lead = re.sub(r"\s+", " ", match.group("lead")).strip()
        body = match.group("body")
        after = re.sub(r"\s+", " ", match.group("after")).strip()
        numbered_lines = [
            line
            for line in body.splitlines()
            if "&" in line and not re.search(r"\\nonumber\b|^\s*&", line)
        ]
        label_count = len(re.findall(r"\\label\{", body))
        if max(len(numbered_lines), label_count) < 2:
            continue
        if re.search(
            r"\b(respectively|as follows|so that|where the first|where the second|and write|and obtain|and the corresponding)\b",
            lead,
            flags=re.I,
        ):
            continue
        if re.search(r"\\text\{and\}", body) and re.search(r"^respectively\b", after, flags=re.I):
            continue
        issues.append(lead[:240])
    return issues


def _phase3_1_technical_writing_contract(
    *,
    run_dir: Path,
    system_tex: str,
    proposed_tex: str,
) -> dict[str, Any]:
    """Record advisory signals for technical prose; do not keyword-block writing."""

    technical_text = f"{system_tex}\n\n{proposed_tex}"
    lowered = technical_text.lower()
    errors: list[str] = []
    warnings: list[str] = []
    system_equation_count = len(
        re.findall(r"\\begin\{(?:equation|align|subequations)\}", system_tex)
    )
    system_leadin_count = len(
        re.findall(
            r"\b(?:is given by|can be written as|is expressed as|we obtain|we define|we denote|is modeled as|is denoted by|denotes|follows as|this yields|accordingly)\b",
            system_tex,
            flags=re.I,
        )
    )
    bare_definition_count = len(
        re.findall(
            r"\bThe\s+(?:signal|observation|metric|utility|SINR|rate|channel|distance|objective|constraint)\s+is\s*(?:\\begin|\$|\\\(|\n)",
            system_tex,
            flags=re.I,
        )
    )
    if system_equation_count >= 3:
        min_leadins = min(4, max(2, system_equation_count // 3))
        if system_leadin_count < min_leadins:
            errors.append(
                "System Model prose is too equation-list-like: displayed relations need IEEE-style lead-ins such as "
                "'is given by', 'can be written as', 'is expressed as', 'we denote', or 'where ... denotes ...'."
            )
    if bare_definition_count:
        errors.append(
            "System Model uses bare-definition openings such as 'The signal is' or 'The metric is' before displayed formulas; "
            "rewrite them with paper-facing mathematical transitions."
        )
    ambiguous_multi_equation_leadins = _phase3_1_ambiguous_multi_equation_leadins(technical_text)
    if ambiguous_multi_equation_leadins:
        errors.append(
            "Technical prose introduces multiple displayed equations with an ambiguous single lead-in. "
            "Use 'respectively', 'as follows', 'define ..., and write ...', or split the lead-in into separate sentences. "
            f"Examples: {'; '.join(ambiguous_multi_equation_leadins[:3])}"
        )
    has_phi_placeholder = bool(re.search(r"\\Phi|Phi_m|\bphi_m\b|\\mathcal\s*C_\{?\\Phi\}?", technical_text))
    phi_placeholder_language = bool(
        re.search(
            r"selected ambiguity model|chosen ambiguity model|standard safe counterpart|safe conic counterpart|\\mathcal\s*C_\{?\\Phi\}?\s+denotes|when the selected\s+\\Phi|denoted by\s+\$?\\Phi",
            technical_text,
            flags=re.I,
        )
    )
    phi_has_reproducible_definition = bool(
        re.search(r"\\Phi[^=\n]{0,120}(?:=|\\triangleq|:=)", technical_text)
        and _phase3_1_has_concrete_uncertainty_model(technical_text)
    )
    if has_phi_placeholder and (phi_placeholder_language or not phi_has_reproducible_definition):
        warnings.append(
            "Phase 3.1 uses a Phi-style safe-counterpart placeholder in the paper-facing method. "
            "Ask the theory/writing agent to verify whether the current artifacts actually define the active uncertainty/reliability mechanism; "
            "if not, scope the claim as empirical/conservative instead of theorem-level."
        )
    if re.search(r"(t[_\{]?[^\n]{0,40}\\rho|\\rho[^\n]{0,40}t[_\{]?)", technical_text, flags=re.I) and re.search(
        r"\\ge\s*1|\\geq\s*1|>=\s*1",
        technical_text,
    ):
        if not re.search(r"rotated second-order cone|rotated SOC|RSOC|perspective|reciprocal epigraph", technical_text, flags=re.I):
            warnings.append(
                "Phase 3.1 displays reciprocal/product epigraph constraints without an explicit RSOC, perspective, or reciprocal-epigraph representation."
            )
    if re.search(r"rank recovery is unnecessary|no rank recovery|without rank recovery", technical_text, flags=re.I) and not re.search(
        r"gaussian signaling|multi-stream|multistream|covariance-domain|high-rank covariance|rank-one transmission|rank-one recovery",
        technical_text,
        flags=re.I,
    ):
        warnings.append(
            "Phase 3.1 dismisses rank recovery without explaining physical covariance-domain signaling, rank-one transmission, or recovery scope."
        )
    if re.search(
        r"(shared|sensing|energy|common|auxiliary|non-information|service|artificial|jamming|pilot)[^.\n]{0,60}(covariance|waveform|signal|resource)[^.\n]{0,120}(interference|denominator|SINR)",
        technical_text,
        flags=re.I,
    ) and not re.search(
        r"not decoded|not canceled|not cancelled|treat(?:ed)? as noise|unknown to the receiver|known and cancel|known and cancell|jointly decoded|joint decoding|successive interference cancellation|SIC",
        technical_text,
        flags=re.I,
    ):
        warnings.append(
            "Phase 3.1 places a shared/auxiliary signal in a receiver metric without stating the decoding or cancellation assumption."
        )
    if re.search(r"epsilon[^.\n]{0,90}(outage|chance)", lowered) and re.search(
        r"epsilon[^.\n]{0,90}(uncertainty|radius|csi|channel)",
        lowered,
    ):
        warnings.append(
            "Phase 3.1 uses epsilon for both outage/chance tolerance and uncertainty radius. Use distinct public symbols, e.g. epsilon and delta_h."
        )
    algorithm_block = re.search(r"\\begin\{algorithm\}.*?\\end\{algorithm\}", proposed_tex, flags=re.S)
    has_loop_or_iteration = False
    has_model_specific_construction = False
    has_postsolve_interpretation = False
    if algorithm_block:
        algorithm_text = algorithm_block.group(0)
        state_count = len(re.findall(r"\\State\b", algorithm_text))
        if state_count < 4:
            warnings.append("Algorithm block has fewer than four paper-facing method steps.")
        has_loop_or_iteration = bool(re.search(r"\\While|\\For|repeat|until|iterat", algorithm_text, flags=re.I))
        has_model_specific_construction = bool(
            re.search(
                r"coefficient|margin|surrogate|counterpart|cover|certificate|lmi|cone|epigraph|projection|randomiz|candidate|accept",
                algorithm_text,
                flags=re.I,
            )
        )
        has_postsolve_interpretation = bool(
            re.search(r"recover|evaluate|verify|check|feasib|constraint|kpi|return best|select best", algorithm_text, flags=re.I)
        )
        if re.search(r"solve problem.*return", algorithm_text, flags=re.I | re.S) and not re.search(
            r"coefficient|margin|surrogate|counterpart|recover|evaluate|cover|certificate|projection|randomiz|candidate|accept",
            algorithm_text,
            flags=re.I,
        ):
            warnings.append("Algorithm block may read like a generic solver wrapper rather than a method skeleton.")
        elif not has_loop_or_iteration and re.search(r"\bsolve\b", algorithm_text, flags=re.I) and not (
            has_model_specific_construction and has_postsolve_interpretation
        ):
            warnings.append(
                "Algorithm block may still lack route-specific construction or post-solve verification, so it may not read like a real method procedure."
            )
    report = {
        "ok": True,
        "errors": errors,
        "warnings": warnings,
        "advisory_only": True,
        "checks": {
            "system_equation_count": system_equation_count,
            "system_ieee_leadin_count": system_leadin_count,
            "system_bare_definition_count": bare_definition_count,
            "ambiguous_multi_equation_leadin_count": len(ambiguous_multi_equation_leadins),
            "has_phi_placeholder": has_phi_placeholder,
            "phi_has_reproducible_definition": phi_has_reproducible_definition,
            "phi_placeholder_language": phi_placeholder_language,
            "has_concrete_uncertainty_model": _phase3_1_has_concrete_uncertainty_model(technical_text),
            "algorithm_block_has_route_specific_steps": bool(
                not algorithm_block
                or (
                    has_loop_or_iteration
                    or (has_model_specific_construction and has_postsolve_interpretation)
                )
            ),
        },
    }
    report["ok"] = not errors
    report["advisory_only"] = False if errors else True
    phase3_1_dir = Path(run_dir) / "phase3-1"
    write_text(phase3_1_dir / "phase3_1_technical_writing_contract_report.json", json.dumps(report, ensure_ascii=False, indent=2))
    return report


def _phase3_1_llm_candidate_score(compile_issue_summary: str, writing_contract: dict[str, Any]) -> float:
    """Score Phase 3.1 candidates without generating replacement technical text."""

    score = 100.0
    errors = writing_contract.get("errors", []) if isinstance(writing_contract, dict) else []
    warnings = writing_contract.get("warnings", []) if isinstance(writing_contract, dict) else []
    if compile_issue_summary:
        score -= 35.0 if _phase3_1_compile_issues_should_block(compile_issue_summary) else 8.0
    score -= 18.0 * len(errors or [])
    score -= 4.0 * len(warnings or [])
    if writing_contract.get("ok") if isinstance(writing_contract, dict) else False:
        score += 15.0
    return max(0.0, score)


def _run_phase3_1_technical_writing(
    *,
    run_dir: Path,
    topic: str,
    model_profile: str,
    callbacks: "Phase2FlowCallbacks",
    controller: WaraController,
    mathematical_contract_json: str,
    phase1_outputs: dict[str, Any],
    phase2_outputs: dict[str, Any],
    phase3_outputs: dict[str, Any],
) -> None:
    """Generate paper-facing technical sections after Phase 2 evidence is ready."""

    phase3_1_dir = run_dir / "phase3-1"
    phase3_1_outputs = callbacks.run_phase3_1_writing_llm(
        run_dir=run_dir,
        topic=topic,
        mathematical_contract_json=mathematical_contract_json,
        system_model_md=phase1_outputs["system_model_md"],
        problem_formulation_md=phase1_outputs["problem_formulation_md"],
        core_theory_package_md=phase1_outputs["core_theory_package_md"],
        convexity_audit_md=phase2_outputs["convexity_audit_md"],
        reformulation_path_md=phase2_outputs["reformulation_path_md"],
        algorithm_md=phase3_outputs["algorithm_md"],
        convergence_or_complexity_md=phase3_outputs["convergence_or_complexity_md"],
        benchmark_definition_md=phase3_outputs["benchmark_definition_md"],
        model_profile=model_profile,
    )
    phase3_1_system_tex = phase3_1_outputs["system_model_problem_formulation_tex"]
    phase3_1_proposed_tex = phase3_1_outputs["proposed_solution_tex"]
    phase3_1_section_title = phase3_1_outputs.get("proposed_section_title", "Proposed Method")
    write_text(phase3_1_dir / "system_model_problem_formulation_ieee_wcl.tex", phase3_1_system_tex)
    write_text(phase3_1_dir / "proposed_solution_ieee_wcl.tex", phase3_1_proposed_tex)
    write_text(phase3_1_dir / "section_title.txt", phase3_1_section_title)
    phase3_1_candidate_records: list[dict[str, Any]] = []

    def _record_phase3_1_candidate(label: str, compile_issue_summary: str, writing_contract: dict[str, Any]) -> None:
        phase3_1_candidate_records.append(
            {
                "label": label,
                "score": _phase3_1_llm_candidate_score(compile_issue_summary, writing_contract),
                "compile_issue_summary": compile_issue_summary,
                "writing_contract": writing_contract,
                "system_model_problem_formulation_tex": phase3_1_system_tex,
                "proposed_solution_tex": phase3_1_proposed_tex,
                "proposed_section_title": phase3_1_section_title,
            }
        )

    phase3_1_preview_outputs = callbacks.render_phase3_1_technical_preview_pdf(phase3_1_dir)
    phase3_1_repair_round_limit = _phase3_1_writing_repair_round_limit()
    for repair_attempt in range(1, phase3_1_repair_round_limit + 1):
        compile_log_tail = read_text(phase3_1_dir / "phase3_1_technical_preview.log")[-12000:]
        combined_latex = phase3_1_system_tex + "\n\n" + phase3_1_proposed_tex
        issue_summary = callbacks.extract_latex_issue_summary(compile_log_tail, combined_latex)
        if not issue_summary:
            break
        try:
            repaired_phase3_1 = callbacks.repair_phase3_1_latex_llm(
                run_dir=run_dir,
                topic=topic,
                mathematical_contract_json=mathematical_contract_json,
                current_system_model_problem_formulation_tex=phase3_1_system_tex,
                current_proposed_solution_tex=phase3_1_proposed_tex,
                issue_summary=issue_summary,
                compile_log_tail=compile_log_tail,
                model_profile=model_profile,
            )
        except Exception as exc:  # noqa: BLE001 - keep the rendered draft and let downstream gates judge it.
            write_json_artifact(
                phase3_1_dir / f"phase3_1_latex_repair_exception_round{repair_attempt}.json",
                {
                    "repair_round": repair_attempt,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "decision": "continue_with_rendered_technical_sections",
                },
            )
            break
        if not repaired_phase3_1.get("system_model_problem_formulation_tex", "").strip():
            break
        if not repaired_phase3_1.get("proposed_solution_tex", "").strip():
            break
        phase3_1_system_tex = repaired_phase3_1["system_model_problem_formulation_tex"]
        phase3_1_proposed_tex = repaired_phase3_1["proposed_solution_tex"]
        write_text(phase3_1_dir / "system_model_problem_formulation_ieee_wcl.tex", phase3_1_system_tex)
        write_text(phase3_1_dir / "proposed_solution_ieee_wcl.tex", phase3_1_proposed_tex)
        phase3_1_section_title = repaired_phase3_1.get("proposed_section_title", "Proposed Method")
        write_text(phase3_1_dir / "section_title.txt", phase3_1_section_title)
        phase3_1_preview_outputs = callbacks.render_phase3_1_technical_preview_pdf(phase3_1_dir)
        _write_compile_repair_artifacts(
            phase_dir=phase3_1_dir,
            prefix="phase3_1_latex",
            attempt=repair_attempt,
            issue_summary=issue_summary,
            repaired_latex=phase3_1_system_tex + "\n\n" + phase3_1_proposed_tex,
        )
    final_phase3_1_issue_summary = _phase_preview_compile_issue_summary(
        callbacks=callbacks,
        log_path=phase3_1_dir / "phase3_1_technical_preview.log",
        latex_text=phase3_1_system_tex + "\n\n" + phase3_1_proposed_tex,
    )
    write_text(phase3_1_dir / "phase3_1_latex_compile_issue_report.txt", final_phase3_1_issue_summary or "ok")
    write_text(
        phase3_1_dir / "phase3_1_preview_manifest.json",
        json.dumps(phase3_1_preview_outputs, ensure_ascii=False, indent=2),
    )
    phase3_1_compile_blocks = bool(final_phase3_1_issue_summary and _phase3_1_compile_issues_should_block(final_phase3_1_issue_summary))
    phase3_1_writing_contract = _phase3_1_technical_writing_contract(
        run_dir=run_dir,
        system_tex=phase3_1_system_tex,
        proposed_tex=phase3_1_proposed_tex,
    )
    _record_phase3_1_candidate("initial_after_latex_repair", final_phase3_1_issue_summary or "", phase3_1_writing_contract)
    content_repair_rounds = 0
    while (
        not bool(phase3_1_writing_contract.get("ok", False))
        and content_repair_rounds < phase3_1_repair_round_limit
    ):
        content_repair_rounds += 1
        issue_summary = "\n".join(str(item) for item in phase3_1_writing_contract.get("errors", []))
        write_text(
            phase3_1_dir / f"phase3_1_technical_writing_contract_repair_reason_round{content_repair_rounds}.txt",
            issue_summary,
        )
        try:
            repaired_phase3_1 = callbacks.repair_phase3_1_latex_llm(
                run_dir=run_dir,
                topic=topic,
                mathematical_contract_json=mathematical_contract_json,
                current_system_model_problem_formulation_tex=phase3_1_system_tex,
                current_proposed_solution_tex=phase3_1_proposed_tex,
                issue_summary=(
                    issue_summary
                    + "\n\nThis is a technical-writing/content contract failure, not just a LaTeX compile failure. "
                    + "Repair only if the current upstream artifacts contain enough information. Do not invent missing theory; "
                    + "if a counterpart/proof cannot be made concrete, scope the claim as conservative/empirical."
                ),
                compile_log_tail=read_text(phase3_1_dir / "phase3_1_technical_preview.log")[-12000:],
                model_profile=model_profile,
            )
        except Exception as exc:  # noqa: BLE001 - keep the rendered draft and let downstream gates judge it.
            write_json_artifact(
                phase3_1_dir / f"phase3_1_content_repair_exception_round{content_repair_rounds}.json",
                {
                    "repair_round": content_repair_rounds,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "decision": "continue_with_rendered_technical_sections",
                },
            )
            break
        if not repaired_phase3_1.get("system_model_problem_formulation_tex", "").strip():
            break
        if not repaired_phase3_1.get("proposed_solution_tex", "").strip():
            break
        phase3_1_system_tex = repaired_phase3_1["system_model_problem_formulation_tex"]
        phase3_1_proposed_tex = repaired_phase3_1["proposed_solution_tex"]
        write_text(phase3_1_dir / "system_model_problem_formulation_ieee_wcl.tex", phase3_1_system_tex)
        write_text(phase3_1_dir / "proposed_solution_ieee_wcl.tex", phase3_1_proposed_tex)
        phase3_1_section_title = repaired_phase3_1.get("proposed_section_title", "Proposed Method")
        write_text(phase3_1_dir / "section_title.txt", phase3_1_section_title)
        phase3_1_preview_outputs = callbacks.render_phase3_1_technical_preview_pdf(phase3_1_dir)
        _write_compile_repair_artifacts(
            phase_dir=phase3_1_dir,
            prefix="phase3_1_content_contract",
            attempt=content_repair_rounds,
            issue_summary=issue_summary,
            repaired_latex=phase3_1_system_tex + "\n\n" + phase3_1_proposed_tex,
        )
        phase3_1_writing_contract = _phase3_1_technical_writing_contract(
            run_dir=run_dir,
            system_tex=phase3_1_system_tex,
            proposed_tex=phase3_1_proposed_tex,
        )
        current_compile_issue_summary = _phase_preview_compile_issue_summary(
            callbacks=callbacks,
            log_path=phase3_1_dir / "phase3_1_technical_preview.log",
            latex_text=phase3_1_system_tex + "\n\n" + phase3_1_proposed_tex,
        )
        _record_phase3_1_candidate(
            f"content_repair_round_{content_repair_rounds}",
            current_compile_issue_summary or "",
            phase3_1_writing_contract,
        )
    if content_repair_rounds:
        final_phase3_1_issue_summary = _phase_preview_compile_issue_summary(
            callbacks=callbacks,
            log_path=phase3_1_dir / "phase3_1_technical_preview.log",
            latex_text=phase3_1_system_tex + "\n\n" + phase3_1_proposed_tex,
        )
        write_text(phase3_1_dir / "phase3_1_latex_compile_issue_report.txt", final_phase3_1_issue_summary or "ok")
        write_text(
            phase3_1_dir / "phase3_1_preview_manifest.json",
            json.dumps(phase3_1_preview_outputs, ensure_ascii=False, indent=2),
        )
        phase3_1_compile_blocks = bool(final_phase3_1_issue_summary and _phase3_1_compile_issues_should_block(final_phase3_1_issue_summary))
    phase3_1_content_blocks = not bool(phase3_1_writing_contract.get("ok", False))
    if (phase3_1_compile_blocks or phase3_1_content_blocks) and phase3_1_candidate_records:
        selected_candidate = max(phase3_1_candidate_records, key=lambda item: float(item.get("score") or 0.0))
        phase3_1_system_tex = str(selected_candidate.get("system_model_problem_formulation_tex") or "")
        phase3_1_proposed_tex = str(selected_candidate.get("proposed_solution_tex") or "")
        phase3_1_section_title = str(selected_candidate.get("proposed_section_title") or "Proposed Method")
        write_text(phase3_1_dir / "system_model_problem_formulation_ieee_wcl.tex", phase3_1_system_tex)
        write_text(phase3_1_dir / "proposed_solution_ieee_wcl.tex", phase3_1_proposed_tex)
        write_text(phase3_1_dir / "section_title.txt", phase3_1_section_title)
        phase3_1_preview_outputs = callbacks.render_phase3_1_technical_preview_pdf(phase3_1_dir)
        final_phase3_1_issue_summary = _phase_preview_compile_issue_summary(
            callbacks=callbacks,
            log_path=phase3_1_dir / "phase3_1_technical_preview.log",
            latex_text=phase3_1_system_tex + "\n\n" + phase3_1_proposed_tex,
        )
        phase3_1_writing_contract = _phase3_1_technical_writing_contract(
            run_dir=run_dir,
            system_tex=phase3_1_system_tex,
            proposed_tex=phase3_1_proposed_tex,
        )
        phase3_1_compile_blocks = bool(final_phase3_1_issue_summary and _phase3_1_compile_issues_should_block(final_phase3_1_issue_summary))
        phase3_1_content_blocks = not bool(phase3_1_writing_contract.get("ok", False))
        write_json_artifact(
            phase3_1_dir / "phase3_1_selected_llm_candidate_after_repair_budget.json",
            {
                "selection_policy": "highest_gate_score_among_llm_writing_candidates",
                "selected_label": selected_candidate.get("label"),
                "selected_score": selected_candidate.get("score"),
                "candidate_count": len(phase3_1_candidate_records),
                "final_compile_blocks": phase3_1_compile_blocks,
                "final_content_blocks": phase3_1_content_blocks,
                "candidates": [
                    {
                        "label": item.get("label"),
                        "score": item.get("score"),
                        "compile_issue_summary": item.get("compile_issue_summary"),
                        "writing_errors": (item.get("writing_contract") or {}).get("errors", []),
                        "writing_warnings": (item.get("writing_contract") or {}).get("warnings", []),
                    }
                    for item in phase3_1_candidate_records
                ],
            },
        )
    _register_phase3_1_controller_artifacts(controller)
    _record_controller_gate(
        controller,
        "phase3_1_technical_writing_gate",
        ok=not phase3_1_compile_blocks and not phase3_1_content_blocks,
        artifact_ids=[
            "phase3_1_system_model_problem_formulation",
            "phase3_1_proposed_solution",
            "phase3_1_preview_manifest",
        ],
        errors=(
            ([final_phase3_1_issue_summary] if phase3_1_compile_blocks else [])
            + [str(item) for item in phase3_1_writing_contract.get("errors", [])]
        ),
        warnings=(
            ([final_phase3_1_issue_summary] if final_phase3_1_issue_summary and not phase3_1_compile_blocks else [])
            + [str(item) for item in phase3_1_writing_contract.get("warnings", [])]
        ),
    )
    if phase3_1_compile_blocks:
        write_text(
            phase3_1_dir / "phase3_1_selected_candidate_compile_warning.txt",
            "Phase 3.1 technical preview still has compile-quality issues after LLM repair budget was exhausted. "
            "The controller selected the highest-scoring LLM writing candidate for downstream synthesis.\n"
            + final_phase3_1_issue_summary,
        )
    if phase3_1_content_blocks:
        write_text(
            phase3_1_dir / "phase3_1_selected_candidate_content_warning.txt",
            "Phase 3.1 technical writing contract did not fully pass after LLM repair budget was exhausted. "
            "The controller selected the highest-scoring LLM writing candidate for downstream synthesis.\n"
            + "\n".join(str(item) for item in phase3_1_writing_contract.get("errors", [])),
        )


def _parse_phase24_validation_plan_text(plan_text: str) -> dict[str, Any]:
    """Parse the executable Phase 2.4 plan returned by the ValidationPlanAgent."""

    text = (plan_text or "").strip()
    if not text:
        raise ValueError("Phase 2.4 ValidationPlanAgent returned an empty validation plan")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise ValueError("Phase 2.4 validation plan must parse to a mapping")
    return payload


def _build_frozen_math_interface_markdown(mathematical_contract_json: str) -> str:
    """Render the Phase 2.1 contract as a human-readable immutable interface."""

    try:
        contract = json.loads(mathematical_contract_json or "{}")
    except json.JSONDecodeError:
        contract = {}
    if not isinstance(contract, dict):
        contract = {}

    def _symbols(items: Any, key: str = "symbol") -> str:
        if not isinstance(items, list):
            return "not specified"
        values = []
        for item in items:
            if isinstance(item, dict):
                value = str(item.get(key) or item.get("canonical_symbol") or item.get("id") or "").strip()
            else:
                value = str(item).strip()
            if value:
                values.append(value)
        return ", ".join(values) if values else "not specified"

    objective = contract.get("objective") if isinstance(contract.get("objective"), dict) else {}
    objective_text = "not specified"
    if objective:
        sense = str(objective.get("sense") or "").strip()
        expression = str(objective.get("expression") or "").strip()
        objective_text = " ".join(part for part in [sense, expression] if part) or "not specified"

    constraints = contract.get("constraints")
    constraint_lines = []
    if isinstance(constraints, list):
        for item in constraints:
            if isinstance(item, dict):
                cid = str(item.get("id") or "constraint").strip()
                relation = str(item.get("relation") or item.get("meaning") or "").strip()
                constraint_lines.append(f"- {cid}: {relation}" if relation else f"- {cid}")
    if not constraint_lines:
        constraint_lines = ["- not specified"]

    return "\n".join(
        [
            "# Frozen Mathematical Interface",
            "",
            "This artifact is produced by Phase 2.1 and is read-only for Phase 2.2 onward.",
            "Later phases may introduce reformulation-only or implementation-only objects, but they must not redefine the original problem.",
            "",
            f"- Controls allowed in the original optimizer: {_symbols(contract.get('controls'))}",
            f"- Parameters: {_symbols(contract.get('parameters'))}",
            f"- Random quantities: {_symbols(contract.get('random_quantities'))}",
            f"- Derived quantities: {_symbols(contract.get('derived_quantities'))}",
            f"- Objective: {objective_text}",
            "- Original constraints:",
            *constraint_lines,
            f"- Reformulation-only objects already declared: {_symbols(contract.get('reformulation_only'))}",
            f"- Canonical notation to preserve: {_symbols(contract.get('notation_to_preserve'))}",
        ]
    ).strip()


@dataclass
class Phase2FlowCallbacks:
    build_pipeline_experiment_design_notes: Callable[[], str]
    build_phase1_handoff: Callable[[Path, Path], dict[str, Any]]
    build_phase3_design_notes: Callable[[], str]
    build_phase24_design_notes: Callable[[], str]
    extract_latex_issue_summary: Callable[[str, str], str]
    render_phase1_ieee_preview_pdf: Callable[[Path], dict[str, str]]
    render_phase3_ieee_preview_pdf: Callable[[Path], dict[str, str]]
    render_phase3_1_technical_preview_pdf: Callable[[Path], dict[str, str]]
    repair_phase2_phase1_latex_llm: Callable[..., str]
    repair_phase2_phase3_latex_llm: Callable[..., str]
    repair_phase3_1_latex_llm: Callable[..., dict[str, str]]
    repair_phase2_phase24_plugin_llm: Callable[..., str]
    run_phase3_6_apply_review_fixes_package: Callable[[Path], dict[str, Any]]
    run_phase2_phase1_latex_llm: Callable[..., str]
    run_phase2_phase1_llm: Callable[..., dict[str, Any]]
    run_phase2_phase2_llm: Callable[..., dict[str, Any]]
    run_phase2_phase3_latex_llm: Callable[..., str]
    run_phase2_phase3_llm: Callable[..., dict[str, Any]]
    run_phase3_1_writing_llm: Callable[..., dict[str, str]]
    run_phase2_phase24_benchmark_llm: Callable[..., dict[str, Any]]
    run_phase2_phase24_plugin_llm: Callable[..., str]
    run_phase2_phase24_validation_llm: Callable[..., str]
    run_phase24_paper_sweep_from_plan: Callable[[Path, bool], dict[str, Any]]
    run_phase25_wcl_package: Callable[[Path], dict[str, Any]]
    run_phase3_2_numerical_results_package: Callable[[Path], dict[str, Any]]
    run_phase3_3_technical_sections_package: Callable[[Path], dict[str, Any]]
    run_phase3_4_introduction_references_package: Callable[[Path], dict[str, Any]]
    run_phase3_5_paper_review_package: Callable[[Path], dict[str, Any]]
    phase24_validation_allows_repair: Callable[[dict[str, Any]], bool]
    phase24_validation_error_text: Callable[[Path, dict[str, Any]], str]
    validate_phase24_evidence_contract_design: Callable[[Path], dict[str, Any]]
    validate_phase2_phase24_plugin_bundle: Callable[[Path], dict[str, Any]]
    write_phase2_phase24_fixed_harness: Callable[[Path], None]


def _phase_preview_compile_issue_summary(
    *,
    callbacks: Phase2FlowCallbacks,
    log_path: Path,
    latex_text: str,
) -> str:
    return callbacks.extract_latex_issue_summary(read_text(log_path)[-12000:], latex_text)


def _write_compile_repair_artifacts(
    *,
    phase_dir: Path,
    prefix: str,
    attempt: int,
    issue_summary: str,
    repaired_latex: str,
) -> None:
    suffix = "" if attempt == 1 else f"_{attempt}"
    write_text(phase_dir / f"{prefix}_compile_issue_summary{suffix}.txt", issue_summary)
    write_text(phase_dir / f"{prefix}_compile_repaired{suffix}.tex", repaired_latex)


def _phase25_status_from_manifest(run_dir: Path, result: dict[str, Any] | None) -> str:
    if isinstance(result, dict):
        for key in ("phase25_status", "status", "overall_status"):
            value = str(result.get(key) or "").strip()
            if value:
                return value
        nested = result.get("experiment_summary")
        if isinstance(nested, dict):
            value = str(nested.get("phase25_status") or nested.get("overall_status") or "").strip()
            if value:
                return value

    for relative in (
        "phase2-5/phase25_experiment_summary.json",
        "phase2-5/phase25_manifest.json",
        "phase2-5/plot_quality_report.json",
    ):
        payload = read_json(run_dir / relative) or {}
        if isinstance(payload, dict):
            for key in ("phase25_status", "status", "overall_status"):
                value = str(payload.get(key) or "").strip()
                if value:
                    return value
    return "unknown"


def _phase25_is_paper_ready(status: str) -> bool:
    return str(status or "").strip() in PAPER_READY_PHASE25_STATUSES


def _phase25_auto_paper_run_limit() -> int:
    for env_name in ("WARA_PHASE25_AUTO_PAPER_RUNS", "WCL_PHASE25_AUTO_PAPER_RUNS"):
        raw_value = os.environ.get(env_name)
        if raw_value is None:
            continue
        try:
            return max(0, int(raw_value))
        except ValueError:
            continue
    return 10


def _phase25_bounded_budget_continuation_enabled() -> bool:
    raw_value = (
        os.environ.get("WARA_PHASE25_ALLOW_NONPAPER_CONTINUATION")
        or os.environ.get("WCL_ALLOW_DRAFT_PHASE25_CONTINUE")
        or ""
    )
    return str(raw_value).strip().lower() in {"1", "true", "yes"}


def _phase24_validation_candidate_score(run_dir: Path, validation_status: dict[str, Any]) -> float:
    """Score the current Phase 2.4 candidate before deciding whether to keep it.

    The score is intentionally based on executable artifacts, not on prose:
    compiling, running, producing finite physical KPIs, covering the proposed
    and at least one benchmark method, and responding to a sweep all increase
    the chance that the candidate is usable by Phase 2.5/3.
    """

    phase24_dir = Path(run_dir) / "phase2-4"
    status = str(validation_status.get("status") or "").strip().lower()
    score = 0.0
    if status in {"ok", "passed"}:
        score += 1000.0
    hard_failure_statuses = {
        "codegen_package_failed",
        "compile_failed",
        "compile_timeout",
        "interface_failed",
        "missing_outputs",
        "schema_alignment_failed",
        "smoke_failed",
        "smoke_timeout",
        "validation_failed",
        "validation_timeout",
    }
    if status in hard_failure_statuses:
        score -= 200.0

    codegen = read_json(phase24_dir / "phase24_codegen_package_check.json") or {}
    schema = read_json(phase24_dir / "phase24_schema_alignment.json") or {}
    evidence_contract = read_json(phase24_dir / "phase24_evidence_contract_check.json") or {}
    method_semantics = read_json(phase24_dir / "phase24_method_semantics_check.json") or {}
    pilot_gain = read_json(phase24_dir / "phase24_pilot_gain_check.json") or {}
    if codegen.get("ok") is True:
        score += 80.0
    if schema.get("ok") is True:
        score += 80.0
    if evidence_contract.get("ok") is True:
        score += 100.0
    if method_semantics.get("ok") is True:
        score += 100.0
    if evidence_contract.get("ok") is False or method_semantics.get("ok") is False:
        score -= 120.0
    if pilot_gain.get("ok") is True:
        score += 160.0
        score += 80.0 * max(0.0, min(1.0, float(pilot_gain.get("pilot_win_rate") or 0.0)))
        score += 80.0 * max(0.0, min(1.0, float(pilot_gain.get("pilot_median_relative_gain") or 0.0)))
    elif pilot_gain.get("ok") is False:
        score -= 180.0

    results_path = phase24_dir / "solver" / "outputs" / "validation_results.csv"
    try:
        with results_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except Exception:  # noqa: BLE001
        rows = []
    if not rows:
        return score

    methods = {str(row.get("method") or row.get("method_id") or "").strip().lower() for row in rows}
    figures = {str(row.get("figure_id") or "").strip() for row in rows}
    finite_objective_rows = 0
    finite_physical_rows = 0
    ok_rows = 0
    feasible_rows = 0
    for row in rows:
        if str(row.get("status") or "").strip().lower() == "ok":
            ok_rows += 1
        if str(row.get("feasible") or "").strip().lower() in {"true", "1", "yes"}:
            feasible_rows += 1
        for key in ("objective", "objective_value"):
            try:
                value = float(row.get(key, ""))
            except (TypeError, ValueError):
                continue
            if value == value and value not in {float("inf"), float("-inf")}:
                finite_objective_rows += 1
                break
        for key in ("sum_rate_bpsHz", "sum_rate_bps_hz", "min_user_rate_bpsHz", "rate_bpsHz", "sum_power_W"):
            try:
                value = float(row.get(key, ""))
            except (TypeError, ValueError):
                continue
            if value == value and value not in {float("inf"), float("-inf")}:
                finite_physical_rows += 1
                break

    score += min(len(rows), 100) * 2.0
    score += min(finite_objective_rows, 100) * 3.0
    score += min(finite_physical_rows, 100) * 4.0
    score += min(ok_rows, 100) * 2.0
    score += min(feasible_rows, 100) * 3.0
    if "proposed" in methods:
        score += 80.0
    if len({method for method in methods if method}) >= 2:
        score += 80.0
    if len({figure for figure in figures if figure}) >= 1:
        score += 50.0

    responsiveness = read_json(phase24_dir / "phase24_experiment_responsiveness_check.json") or {}
    responsive_checks = [
        check
        for check in responsiveness.get("checks", [])
        if isinstance(check, dict)
        and float(check.get("relative_metric_span") or 0.0) > 0.01
        and int(check.get("num_x_values") or 0) >= 3
    ]
    if responsive_checks or responsiveness.get("ok") is True:
        score += 100.0
    return score


def _phase24_selected_candidate_can_continue(run_dir: Path, validation_status: dict[str, Any]) -> bool:
    status = str(validation_status.get("status") or "").strip().lower()
    if status == "ok":
        return True
    if status == "pilot_gain_failed":
        return False
    hard_failure_statuses = {
        "codegen_package_failed",
        "compile_failed",
        "compile_timeout",
        "interface_failed",
        "missing_outputs",
        "schema_alignment_failed",
        "smoke_failed",
        "smoke_timeout",
        "validation_failed",
        "validation_timeout",
    }
    if status in hard_failure_statuses:
        return False
    phase24_dir = Path(run_dir) / "phase2-4"
    results_path = phase24_dir / "solver" / "outputs" / "validation_results.csv"
    if not results_path.exists():
        return False
    try:
        with results_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except Exception:  # noqa: BLE001
        return False
    if not rows:
        return False
    methods = {str(row.get("method") or row.get("method_id") or "").strip().lower() for row in rows}
    if "proposed" not in methods or len({method for method in methods if method}) < 2:
        return False

    sweep_keys: set[tuple[str, str]] = set()
    for row in rows:
        figure_id = str(row.get("figure_id") or "").strip()
        sweep_id = str(row.get("sweep_id") or row.get("sweep_name") or "").strip()
        swept_param = str(row.get("swept_param") or row.get("swept_parameter") or row.get("sweep_parameter") or "").strip()
        swept_value = str(row.get("swept_value") or row.get("sweep_value") or "").strip()
        if figure_id:
            sweep_keys.add(("figure", figure_id))
        if sweep_id and sweep_id.lower() not in {"canonical", "baseline", "default"}:
            sweep_keys.add((sweep_id, swept_value))
        if swept_param and swept_param.lower() not in {"canonical", "baseline", "default"}:
            sweep_keys.add((swept_param, swept_value))
    if len(sweep_keys) < 2:
        return False

    seeds = {str(row.get("seed") or "").strip() for row in rows if str(row.get("seed") or "").strip()}
    required_seed_floor = max(2, min(10, int(os.environ.get("WARA_PHASE24_PILOT_MIN_PAIRED_SEEDS", "20") or 20)))
    if len(seeds) < required_seed_floor:
        return False
    pilot_gain = read_json(phase24_dir / "phase24_pilot_gain_check.json") or {}
    if pilot_gain.get("ok") is not True:
        return False
    finite_objective_rows = 0
    finite_physical_rows = 0
    for row in rows:
        for key in ("objective", "objective_value"):
            try:
                value = float(row.get(key, ""))
            except (TypeError, ValueError):
                continue
            if value == value and abs(value) < float("inf"):
                finite_objective_rows += 1
                break
        for key in ("sum_rate_bpsHz", "sum_rate_bps_hz", "min_user_rate_bpsHz", "rate_bpsHz", "sum_power_W"):
            try:
                value = float(row.get(key, ""))
            except (TypeError, ValueError):
                continue
            if value == value and value not in {float("inf"), float("-inf")}:
                finite_physical_rows += 1
                break
    return finite_objective_rows > 0 and finite_physical_rows > 0


def _phase24_mark_selected_candidate_validation(run_dir: Path, validation_status: dict[str, Any]) -> dict[str, Any]:
    phase24_dir = Path(run_dir) / "phase2-4"
    selected_status = dict(validation_status)
    selected_status["original_status"] = validation_status.get("status")
    selected_status["status"] = "selected_after_bounded_repairs"
    selected_status["selected_after_bounded_repairs"] = True
    selected_status["selection_score"] = _phase24_validation_candidate_score(run_dir, validation_status)
    selected_status["reason"] = (
        "Phase 2.4 reached the repair limit, so the controller selected the strongest "
        "executable candidate and forwarded it to Phase 2.5 for evidence packaging. "
        "The final PDF must still be produced by the normal Phase 3 writing and review chain."
    )
    write_text(phase24_dir / "phase24_selected_candidate_note.md", selected_status["reason"])
    write_text(phase24_dir / "phase24_validation_manifest.json", json.dumps(selected_status, ensure_ascii=False, indent=2))
    return selected_status


def _phase24_prepare_clean_generation_workspace(phase24_dir: Path, solver_dir: Path) -> None:
    """Remove stale generated code, outputs, and validation residue before codegen.

    Phase 2.4 can be resumed from the same run directory. Without an explicit
    cleanup, an interrupted run may leave a manifest/error file that no longer
    matches the current generated code, or an old CSV that later phases mistake
    for fresh evidence.
    """

    solver_dir.mkdir(parents=True, exist_ok=True)
    for stale_py in solver_dir.glob("*.py"):
        stale_py.unlink()
    for stale_cache in solver_dir.glob("__pycache__"):
        if stale_cache.is_dir():
            shutil.rmtree(stale_cache, ignore_errors=True)
    outputs_dir = solver_dir / "outputs"
    if outputs_dir.exists():
        shutil.rmtree(outputs_dir, ignore_errors=True)

    residue_names = {
        "phase24_split_code_manifest.json",
        "phase24_codegen_package_check.json",
        "phase24_validation_manifest.json",
        "phase24_validation_error.txt",
        "phase24_interface_errors.txt",
        "phase24_py_compile_stdout.txt",
        "phase24_py_compile_stderr.txt",
        "phase24_smoke_stdout.txt",
        "phase24_smoke_stderr.txt",
        "phase24_validation_stdout.txt",
        "phase24_validation_stderr.txt",
        "phase24_numerical_runtime_warning_check.json",
        "phase24_runtime_budget_precheck.json",
        "phase24_runtime_budget_check.json",
        "phase24_schema_alignment.json",
        "phase24_evidence_contract_check.json",
        "phase24_basic_evidence_quality_check.json",
        "phase24_method_semantics_check.json",
        "phase24_experiment_responsiveness_check.json",
        "implementation_audit.json",
        "implementation_audit_blocking_errors.txt",
    }
    for name in residue_names:
        path = phase24_dir / name
        if path.exists():
            path.unlink()
    for pattern in (
        "phase24_generated_plugin_repaired_round*.py",
        "phase24_generated_plugin_raw_response*.txt",
        "phase24_generated_plugin_repair_raw_response*.txt",
        "phase24_generated_experiment_core*.py",
    ):
        for path in phase24_dir.glob(pattern):
            if path.is_file():
                path.unlink()


def _phase25_auto_paper_sweep_mode() -> str:
    raw_value = os.environ.get("WARA_PHASE25_AUTO_PAPER_MODE") or os.environ.get("WCL_PHASE25_AUTO_PAPER_MODE")
    value = str(raw_value or "auto").strip().lower()
    if value in {"auto", "progressive", "tiered"}:
        return "auto"
    if value in {"medium", "mid", "moderate"}:
        return "medium"
    if value in {"paper", "full", "preferred", "0", "false", "no"}:
        return "paper"
    return "scout"


def _phase25_sweep_mode_for_round(round_index: int) -> str:
    mode = _phase25_auto_paper_sweep_mode()
    if mode != "auto":
        return mode
    if round_index <= 1:
        return "scout"
    if round_index == 2:
        return "medium"
    return "paper"


def _phase25_validation_prefix_for_mode(mode: str) -> str:
    return {
        "scout": "scout_validation",
        "medium": "medium_validation",
        "paper": "paper_validation",
    }.get(mode, "scout_validation")


def _run_phase25_sweep_callback_with_tier(callbacks: "Phase2FlowCallbacks", run_dir: Path, mode: str) -> dict[str, Any]:
    quick_sweep = mode in {"scout", "medium"}
    previous_tier = os.environ.get("WARA_PHASE25_SWEEP_TIER")
    os.environ["WARA_PHASE25_SWEEP_TIER"] = mode
    try:
        return callbacks.run_phase24_paper_sweep_from_plan(run_dir, quick_sweep)
    finally:
        if previous_tier is None:
            os.environ.pop("WARA_PHASE25_SWEEP_TIER", None)
        else:
            os.environ["WARA_PHASE25_SWEEP_TIER"] = previous_tier


def _phase25_sweep_runtime_abort_reason(sweep_result: dict[str, Any] | None) -> str:
    if not isinstance(sweep_result, dict):
        return ""
    direct_reason = str(sweep_result.get("runtime_abort_reason") or "").strip()
    if direct_reason:
        return direct_reason
    summary_path_raw = str(sweep_result.get("summary_json") or "").strip()
    if not summary_path_raw:
        return ""
    summary = read_json(Path(summary_path_raw)) or {}
    if not isinstance(summary, dict):
        return ""
    return str(summary.get("runtime_abort_reason") or "").strip()


def _phase25_should_auto_expand(run_dir: Path, status: str) -> bool:
    return (not _phase25_is_paper_ready(status)) and (run_dir / "phase2-5" / "paper_sweep_plan.json").exists()


def _phase25_float(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if numeric == numeric else default


def _phase25_claim_snapshot(run_dir: Path) -> dict[str, Any]:
    summary = read_json(run_dir / "phase2-5" / "phase25_experiment_summary.json") or {}
    if not isinstance(summary, dict):
        summary = {}
    primary = summary.get("primary_claim_check") if isinstance(summary.get("primary_claim_check"), dict) else {}
    strongest = summary.get("strongest_practical_baseline_audit") if isinstance(summary.get("strongest_practical_baseline_audit"), dict) else {}
    win_rate = max(
        _phase25_float(summary.get("proposed_win_rate")),
        _phase25_float(primary.get("proposed_win_rate")),
        _phase25_float(strongest.get("proposed_win_rate")),
    )
    median_gain = max(
        _phase25_float(summary.get("proposed_median_relative_gain"), -1.0),
        _phase25_float(primary.get("proposed_median_relative_gain"), -1.0),
        _phase25_float(strongest.get("proposed_median_relative_gain"), -1.0),
    )
    mean_gain = max(
        _phase25_float(summary.get("proposed_mean_relative_gain"), -1.0),
        _phase25_float(primary.get("proposed_mean_relative_gain"), -1.0),
        _phase25_float(strongest.get("proposed_mean_relative_gain"), -1.0),
    )
    primary_pass = bool(primary.get("passes", False))
    strongest_pass = bool(strongest.get("passes", False)) if strongest else primary_pass
    return {
        "phase25_status": str(summary.get("phase25_status") or ""),
        "num_comparable_cases": int(_phase25_float(summary.get("num_comparable_cases"), 0.0)),
        "primary_passes": primary_pass,
        "strongest_practical_passes": strongest_pass,
        "proposed_win_rate": win_rate,
        "proposed_median_relative_gain": median_gain,
        "proposed_mean_relative_gain": mean_gain,
        "primary_claim_check": primary,
        "strongest_practical_baseline_audit": strongest,
    }


def _phase25_claim_promising(snapshot: dict[str, Any]) -> bool:
    if _phase25_is_paper_ready(str(snapshot.get("phase25_status") or "")):
        return True
    min_win_rate = _phase25_float(os.environ.get("WARA_PHASE25_PROMOTION_MIN_WIN_RATE"), 0.55)
    min_median_gain = _phase25_float(os.environ.get("WARA_PHASE25_PROMOTION_MIN_MEDIAN_GAIN"), 0.0)
    has_comparable = int(snapshot.get("num_comparable_cases") or 0) > 0
    passes = bool(snapshot.get("primary_passes")) and bool(snapshot.get("strongest_practical_passes", True))
    gain_ok = (
        has_comparable
        and _phase25_float(snapshot.get("proposed_win_rate")) >= min_win_rate
        and _phase25_float(snapshot.get("proposed_median_relative_gain"), -1.0) > min_median_gain
    )
    return passes or gain_ok


def _phase25_can_skip_auto_expansion(status: str, snapshot: dict[str, Any]) -> bool:
    """Phase 2.5 should not skip paper-evidence expansion from quick evidence alone."""

    return False


PHASE25_COVERAGE_ONLY_BLOCKERS = {
    "quick_mode_only",
    "medium_mode_only",
    "too_few_x_points",
    "too_few_categories",
    "too_few_samples_for_box",
    "too_few_iterations",
    "insufficient_heatmap_coverage",
    "too_few_seeds_per_point",
    "too_few_effective_x_points_after_feasibility_filter",
    "too_few_medium_x_points_after_filter",
    "planned_x_points_missing_after_feasibility_filter",
    "low_paired_success_rate_per_x",
}


def _phase25_paper_quality_requires_phase24_design_revision(run_dir: Path) -> dict[str, Any]:
    report = read_json(Path(run_dir) / "phase2-5" / "plot_quality_report.json") or {}
    figures = report.get("figures") if isinstance(report, dict) else []
    if not isinstance(figures, list):
        figures = []
    design_blockers: list[dict[str, Any]] = []
    coverage_blockers: list[dict[str, Any]] = []
    ready_figures: list[dict[str, Any]] = []
    story_keys: list[tuple[str, str, str]] = []
    for item in figures:
        if not isinstance(item, dict) or not item.get("counts_toward_paper_minimum", True):
            continue
        issues = [str(issue) for issue in item.get("blocking_issues", []) if str(issue)]
        noncoverage = sorted({issue for issue in issues if issue not in PHASE25_COVERAGE_ONLY_BLOCKERS})
        figure_summary = {
            "figure_id": item.get("figure_id", ""),
            "x_axis_param": item.get("x_axis_param", ""),
            "required_sweep": item.get("required_sweep", ""),
            "y_metric": item.get("y_metric", ""),
            "num_x_points": item.get("num_x_points"),
            "min_seeds_per_x": item.get("min_seeds_per_x"),
        }
        if noncoverage:
            design_blockers.append(
                {
                    **figure_summary,
                    "blocking_issues": noncoverage,
                }
            )
        elif issues:
            coverage_blockers.append({**figure_summary, "blocking_issues": sorted(set(issues))})
        else:
            ready_figures.append(figure_summary)
        story_keys.append(
            (
                str(item.get("required_sweep") or ""),
                str(item.get("x_axis_param") or ""),
                str(item.get("y_metric") or ""),
            )
        )
    nonempty_story_keys = [key for key in story_keys if any(key)]
    if len(nonempty_story_keys) >= 2 and len(set(nonempty_story_keys)) < len(nonempty_story_keys):
        design_blockers.append(
            {
                "figure_id": "global",
                "blocking_issues": ["duplicate_final_figure_story"],
                "story_keys": [list(key) for key in nonempty_story_keys],
            }
        )
    if not design_blockers:
        return {
            "requires_phase24_design_revision": False,
            "ready_figures": ready_figures,
            "coverage_blockers": coverage_blockers,
        }
    return {
        "requires_phase24_design_revision": True,
        "reason": "paper_quality_has_noncoverage_blockers",
        "design_blockers": design_blockers,
        "ready_figures": ready_figures,
        "coverage_blockers": coverage_blockers,
        "repair_scope": {
            "preserve_figures": ready_figures,
            "redesign_figures": [
                item
                for item in design_blockers
                if str(item.get("figure_id") or "") and str(item.get("figure_id") or "") != "global"
            ],
        },
        "advice": (
            "Do not add more seeds or densify the same x-grid for noncoverage blockers. Preserve paper-ready "
            "figures unless they conflict with frozen contracts, and redesign only failed figures so each final "
            "figure has a responsive mechanism axis, a distinct story, and a metric that varies with the selected "
            "operating-regime parameter."
        ),
    }


def _phase25_experiment_redesign_limit() -> int:
    for env_name in ("WARA_PHASE25_REDESIGN_ROUNDS", "WCL_PHASE25_REDESIGN_ROUNDS"):
        raw_value = os.environ.get(env_name)
        if raw_value is None:
            continue
        try:
            return max(0, int(raw_value))
        except ValueError:
            continue
    return 10


def _phase25_coverage_extension_limit() -> int:
    for env_name in ("WARA_PHASE25_COVERAGE_EXTENSION_ROUNDS", "WCL_PHASE25_COVERAGE_EXTENSION_ROUNDS"):
        raw_value = os.environ.get(env_name)
        if raw_value is None:
            continue
        try:
            return max(0, int(raw_value))
        except ValueError:
            continue
    return 10


def _phase25_refine_sweep_plan(
    *,
    run_dir: Path,
    topic: str,
    model_profile: str,
    reason: str,
    round_index: int,
) -> dict[str, Any]:
    from phase_runtime.phase25_planning import call_llm_phase25_sweep_refiner  # noqa: PLC0415

    phase25_dir = run_dir / "phase2-5"
    phase25_dir.mkdir(parents=True, exist_ok=True)
    marker = {
        "reason": reason,
        "round": round_index,
        "policy": "Refine operating regime, x-axis values, and plotted benchmark set before expensive paper-scale expansion.",
    }
    write_json_artifact(phase25_dir / f"paper_sweep_refiner_request_round{round_index}.json", marker)
    result = call_llm_phase25_sweep_refiner(
        run_dir=run_dir,
        topic=topic,
        algorithm_md=read_text(run_dir / "phase2-3" / "algorithm.md"),
        benchmark_definition_md=read_text(run_dir / "phase2-4" / "benchmark_plan.md")
        or read_text(run_dir / "phase2-3" / "benchmark_definition.md"),
        model_profile=model_profile,
    )
    write_json_artifact(phase25_dir / f"paper_sweep_refiner_result_round{round_index}.json", result if isinstance(result, dict) else {"result": result})
    return result if isinstance(result, dict) else {}


def _phase25_refiner_inputs_ready(run_dir: Path) -> bool:
    phase25_dir = Path(run_dir) / "phase2-5"
    required_files = (
        "experiment_plan.json",
        "available_data_summary.json",
        "phase25_experiment_summary.json",
        "paper_sweep_plan.json",
        "missing_experiments.md",
    )
    return all((phase25_dir / name).exists() for name in required_files)


def _phase25_refiner_requests_phase24_design_revision(refined: dict[str, Any] | None) -> bool:
    if not isinstance(refined, dict):
        return False
    status = str(refined.get("status") or "").strip().lower()
    if status == "requires_phase24_design_revision" or bool(refined.get("requires_phase24_design_revision")):
        return True
    notes = refined.get("notes", [])
    notes_text = " ".join(str(item or "") for item in notes) if isinstance(notes, list) else str(notes or "")
    notes_text = notes_text.lower()
    return (
        "requires_phase24_design_revision" in notes_text
        or "phase 2.4 experiment-design repair" in notes_text
        or "phase 2.4 design revision" in notes_text
    )


def _phase25_defer_initial_refiner_redesign_until_sweep(
    run_dir: Path,
    current_status: str,
    refined: dict[str, Any] | None,
) -> bool:
    if str(current_status or "").strip() not in {"quick_mode_only", "needs_more_phase24_runs"}:
        return False
    if not _phase25_refiner_requests_phase24_design_revision(refined):
        return False
    if not (Path(run_dir) / "phase2-5" / "paper_sweep_plan.json").exists():
        return False
    design_check = read_json(Path(run_dir) / "phase2-4" / "phase24_evidence_contract_design_check.json") or {}
    if not design_check or not bool(design_check.get("ok")):
        return False
    if bool(_phase25_objective_metric_alignment_precheck(Path(run_dir)).get("requires_phase24_design_revision")):
        return False
    return True


def _phase25_semantic_family(text: str) -> str:
    lowered = str(text or "").lower()
    families = {
        "utility": (
            "service level",
            "service-level",
            "minimum normalized",
            "eta",
            "eta_service",
            "eta service",
            "normalized service",
            "service margin",
            "service-margin",
            "normalized surplus",
            "deterministic surplus",
            "worst normalized",
            "utility",
            "tau",
            "\\tau",
        ),
        "energy": ("harvest", "harvested", "energy harvesting", "rf-to-dc", "rectifier", "powering", "dc power"),
        "rate": ("sum-rate", "sum rate", "throughput", "spectral efficiency", "bps/hz", "sinr rate"),
        "sensing": ("sensing", "radar", "crb", "beampattern", "illumination"),
        "secrecy": ("secrecy", "confidential", "eavesdropper"),
        "power": ("transmit power", "sum power", "total power", "power consumption", "resource minimization"),
        "efficiency": ("energy efficiency", "bit/j", "bit per joule"),
        "reliability": ("outage", "reliability", "robust", "chance constraint"),
    }
    scores = {
        family: sum(lowered.count(token) for token in tokens)
        for family, tokens in families.items()
    }
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    if not ordered or ordered[0][1] <= 0:
        return ""
    if len(lowered) < 80:
        return ordered[0][0]
    if len(ordered) > 1 and ordered[0][1] < ordered[1][1] + 2:
        return ""
    return ordered[0][0]


def _phase25_objective_metric_alignment_precheck(run_dir: Path) -> dict[str, Any]:
    phase25_summary = read_json(run_dir / "phase2-5" / "phase25_experiment_summary.json") or {}
    primary_metric = ""
    if isinstance(phase25_summary.get("primary_metric"), dict):
        primary_metric = str(phase25_summary.get("primary_metric", {}).get("name") or "").strip()
    if not primary_metric:
        return {"requires_phase24_design_revision": False}
    math_contract = read_json(run_dir / "phase2-1" / "mathematical_contract.frozen.json") or read_json(run_dir / "phase2-1" / "mathematical_contract.json") or {}
    objective = math_contract.get("objective") if isinstance(math_contract, dict) else {}
    if isinstance(objective, dict):
        objective_bits = [
            str(objective.get("sense") or ""),
            str(objective.get("expression") or ""),
            str(objective.get("meaning") or ""),
            json.dumps(objective.get("terms") or [], ensure_ascii=False),
        ]
    else:
        objective_bits = [str(objective or "")]
    objective_text = "\n".join(objective_bits)
    objective_family = _phase25_semantic_family(objective_text)
    if not objective_family:
        objective_text = "\n".join(
            [
                json.dumps(objective if isinstance(objective, dict) else {}, ensure_ascii=False),
                read_text(run_dir / "phase2-2" / "reformulation_path.md")[:4000],
                read_text(run_dir / "phase2-3" / "algorithm.md")[:4000],
            ]
        )
        objective_family = _phase25_semantic_family(objective_text)
    metric_family = _phase25_semantic_family(primary_metric.replace("_", " "))
    if not objective_family or not metric_family or objective_family == metric_family:
        return {"requires_phase24_design_revision": False}
    if any(token in primary_metric.lower() for token in ("objective", "utility", "weighted", "eta_service", "service_level", "service_margin", "normalized_service", "tau")):
        return {"requires_phase24_design_revision": False}
    return {
        "requires_phase24_design_revision": True,
        "reason": "primary_metric_family_mismatch_with_frozen_objective",
        "objective_family": objective_family,
        "primary_metric": primary_metric,
        "primary_metric_family": metric_family,
        "advice": (
            "Phase 2.4 experiment-design repair must regenerate the experiment contract so the main evidence KPI "
            "matches the frozen objective-equivalent physical metric before any scout/medium/paper sweep."
        ),
    }


def _run_phase25_auto_paper_expansion(
    *,
    run_dir: Path,
    callbacks: Phase2FlowCallbacks,
    topic: str = "",
    model_profile: str = DEFAULT_MODEL_PROFILE,
    initial_result: dict[str, Any],
    initial_status: str,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    """Close the Phase 2.4 -> Phase 2.5 loop by executing the paper sweep plan."""

    limit = _phase25_auto_paper_run_limit()
    manifest: dict[str, Any] = {
        "enabled": limit > 0,
        "round_limit": limit,
        "sweep_mode": _phase25_auto_paper_sweep_mode(),
        "redesign_round_limit": _phase25_experiment_redesign_limit(),
        "coverage_extension_round_limit": _phase25_coverage_extension_limit(),
        "initial_phase25_status": initial_status,
        "rounds": [],
        "redesigns": [],
        "coverage_extensions": [],
        "final_phase25_status": initial_status,
        "reason": "",
    }
    phase24_dir = run_dir / "phase2-4"
    phase24_dir.mkdir(parents=True, exist_ok=True)
    current_result = initial_result
    current_status = initial_status

    if limit <= 0:
        manifest["reason"] = "disabled_by_WARA_PHASE25_AUTO_PAPER_RUNS"
    elif not _phase25_should_auto_expand(run_dir, current_status):
        manifest["reason"] = "paper_sweep_plan_not_needed_or_missing"
    else:
        alignment_precheck = _phase25_objective_metric_alignment_precheck(run_dir)
        if bool(alignment_precheck.get("requires_phase24_design_revision")):
            current_status = "requires_phase24_design_revision"
            manifest["reason"] = str(alignment_precheck.get("reason") or "primary_metric_family_mismatch_with_frozen_objective")
            manifest["redesigns"].append(
                {
                    "round": 0,
                    "trigger": "deterministic_objective_metric_alignment_precheck",
                    "counts_against_redesign_limit": False,
                    "ok": False,
                    "status": current_status,
                    "precheck": alignment_precheck,
                }
            )
            manifest["final_phase25_status"] = current_status
            write_json_artifact(run_dir / "phase2-5" / "phase25_auto_expansion_manifest.json", manifest)
            return current_result, current_status, manifest
        initial_snapshot = _phase25_claim_snapshot(run_dir)
        manifest["initial_claim_snapshot"] = initial_snapshot
        if _phase25_can_skip_auto_expansion(current_status, initial_snapshot):
            manifest["reason"] = "auto_expansion_skipped_by_policy"
            manifest["final_phase25_status"] = current_status
            write_json_artifact(run_dir / "phase2-5" / "phase25_auto_expansion_manifest.json", manifest)
            return current_result, current_status, manifest
        redesign_round_limit = _phase25_experiment_redesign_limit()
        coverage_extension_limit = _phase25_coverage_extension_limit()
        effective_limit = limit + redesign_round_limit + coverage_extension_limit
        manifest["effective_round_limit"] = effective_limit
        redesign_rounds = 0
        coverage_extension_rounds = 0
        phase24_design_revision_required = False
        if not _phase25_refiner_inputs_ready(run_dir):
            manifest["redesigns"].append(
                {
                    "round": 0,
                    "trigger": "initial_quick_results_before_auto_expansion",
                    "counts_against_redesign_limit": False,
                    "ok": False,
                    "status": "skipped_missing_phase25_refiner_inputs",
                }
            )
        else:
            try:
                refined = _phase25_refine_sweep_plan(
                    run_dir=run_dir,
                    topic=topic,
                    model_profile=model_profile,
                    reason="initial_quick_results_before_auto_expansion",
                    round_index=0,
                )
                manifest["redesigns"].append(
                    {
                        "round": 0,
                        "trigger": "initial_quick_results_before_auto_expansion",
                        "counts_against_redesign_limit": False,
                        "ok": bool(refined.get("figures")),
                        "status": refined.get("status", ""),
                    }
                )
                if _phase25_refiner_requests_phase24_design_revision(refined):
                    if _phase25_defer_initial_refiner_redesign_until_sweep(run_dir, current_status, refined):
                        manifest["redesigns"][-1]["deferred_until_after_tiered_sweep"] = True
                        manifest["redesigns"][-1]["defer_reason"] = (
                            "initial quick data are too sparse; deterministic Phase 2.4 design checks passed, "
                            "so scout/medium evidence should run before redesign routing"
                        )
                    else:
                        phase24_design_revision_required = True
                        current_status = "requires_phase24_design_revision"
                        effective_limit = 0
                        manifest["reason"] = "requires_phase24_design_revision_before_paper_sweep"
                        _sync_experiment_agent(
                            run_dir,
                            event="phase25_requires_phase24_design_revision",
                            extra_metadata={
                                "trigger": "initial_quick_results_before_auto_expansion",
                                "refiner_result": refined,
                            },
                        )
            except Exception as exc:  # noqa: BLE001 - refiner is helpful but not mandatory
                manifest["redesigns"].append(
                    {
                        "round": 0,
                        "trigger": "initial_quick_results_before_auto_expansion",
                        "counts_against_redesign_limit": False,
                        "ok": False,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
        next_mode_override = ""
        last_nonpaper_promising = True
        for round_index in range(1, effective_limit + 1):
            if phase24_design_revision_required:
                break
            plan_path = run_dir / "phase2-5" / "paper_sweep_plan.json"
            if not _phase25_should_auto_expand(run_dir, current_status):
                break
            sweep_mode = next_mode_override or _phase25_sweep_mode_for_round(round_index)
            next_mode_override = ""
            if sweep_mode == "paper" and not last_nonpaper_promising:
                manifest["reason"] = "paper_sweep_skipped_after_unpromising_scout_or_medium"
                break
            _sync_experiment_agent(
                run_dir,
                event="phase25_auto_paper_sweep_requested",
                extra_metadata={
                    "round": round_index,
                    "mode": sweep_mode,
                    "phase25_status_before": current_status,
                    "paper_sweep_plan": str(plan_path),
                },
            )
            try:
                sweep_result = _run_phase25_sweep_callback_with_tier(callbacks, run_dir, sweep_mode)
            except Exception as exc:  # noqa: BLE001 - controller must persist recoverable run state
                sweep_result = {
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "validation_output_prefix": _phase25_validation_prefix_for_mode(sweep_mode),
                }
                write_json_artifact(
                    phase24_dir / f"phase25_auto_paper_sweep_round{round_index}_error.json",
                    sweep_result,
                )
                manifest["rounds"].append(
                    {
                        "round": round_index,
                        "mode": sweep_mode,
                        "phase25_status_after": current_status,
                        "paper_ready_after": False,
                        "sweep_result": sweep_result,
                    }
                )
                manifest["reason"] = f"auto_expansion_failed:{type(exc).__name__}"
                _sync_experiment_agent(
                    run_dir,
                    event="phase25_auto_paper_sweep_failed",
                    extra_metadata={
                        "round": round_index,
                        "mode": sweep_mode,
                        "phase25_status_before": current_status,
                        "sweep_result": sweep_result,
                    },
                )
                break
            write_json_artifact(
                phase24_dir / f"phase25_auto_paper_sweep_round{round_index}.json",
                sweep_result,
            )
            try:
                current_result = callbacks.run_phase25_wcl_package(run_dir)
                current_status = _phase25_status_from_manifest(run_dir, current_result)
            except Exception as exc:  # noqa: BLE001 - preserve a structured stop reason
                reanalysis_error = {
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "phase": "phase25_reanalysis_after_paper_sweep",
                }
                write_json_artifact(
                    phase24_dir / f"phase25_auto_paper_reanalysis_round{round_index}_error.json",
                    reanalysis_error,
                )
                manifest["rounds"].append(
                    {
                        "round": round_index,
                        "mode": sweep_mode,
                        "phase25_status_after": current_status,
                        "paper_ready_after": False,
                        "sweep_result": sweep_result,
                        "reanalysis_error": reanalysis_error,
                    }
                )
                manifest["reason"] = f"auto_reanalysis_failed:{type(exc).__name__}"
                _sync_experiment_agent(
                    run_dir,
                    event="phase25_auto_reanalysis_failed",
                    extra_metadata={
                        "round": round_index,
                        "phase25_status_before": current_status,
                        "sweep_result": sweep_result,
                        "reanalysis_error": reanalysis_error,
                    },
                )
                break
            manifest["rounds"].append(
                {
                    "round": round_index,
                    "mode": sweep_mode,
                    "phase25_status_after": current_status,
                    "paper_ready_after": _phase25_is_paper_ready(current_status),
                    "sweep_result": sweep_result,
                    "claim_snapshot": _phase25_claim_snapshot(run_dir),
                }
            )
            runtime_abort_reason = _phase25_sweep_runtime_abort_reason(sweep_result)
            if runtime_abort_reason:
                manifest["reason"] = f"auto_expansion_runtime_blocked:{runtime_abort_reason}"
                _sync_experiment_agent(
                    run_dir,
                    event="phase25_auto_paper_sweep_runtime_blocked",
                    extra_metadata={
                        "round": round_index,
                        "mode": sweep_mode,
                        "phase25_status_after": current_status,
                        "runtime_abort_reason": runtime_abort_reason,
                        "sweep_result": sweep_result,
                    },
                )
                break
            if _phase25_is_paper_ready(current_status):
                break
            snapshot = _phase25_claim_snapshot(run_dir)
            promising = _phase25_claim_promising(snapshot)
            if (
                sweep_mode == "paper"
                and promising
                and str(current_status or "") == "needs_more_phase24_runs"
                and _phase25_should_auto_expand(run_dir, current_status)
            ):
                design_revision = _phase25_paper_quality_requires_phase24_design_revision(run_dir)
                if bool(design_revision.get("requires_phase24_design_revision")):
                    current_status = "requires_phase24_design_revision"
                    manifest["reason"] = str(design_revision.get("reason") or "paper_quality_requires_phase24_design_revision")
                    manifest["redesigns"].append(
                        {
                            "round": round_index,
                            "trigger": "paper_quality_noncoverage_blockers",
                            "counts_against_redesign_limit": False,
                            "ok": False,
                            "status": current_status,
                            "claim_snapshot": snapshot,
                            "quality_revision": design_revision,
                        }
                    )
                    _sync_experiment_agent(
                        run_dir,
                        event="phase25_requires_phase24_design_revision",
                        extra_metadata={
                            "round": round_index,
                            "trigger": "paper_quality_noncoverage_blockers",
                            "claim_snapshot": snapshot,
                            "quality_revision": design_revision,
                        },
                    )
                    break
                if coverage_extension_rounds < coverage_extension_limit:
                    coverage_extension_rounds += 1
                    manifest["coverage_extensions"].append(
                        {
                            "round": round_index,
                            "trigger": "paper_claim_promising_but_more_coverage_needed",
                            "counts_against_coverage_extension_limit": True,
                            "claim_snapshot": snapshot,
                            "next_mode": "paper",
                        }
                    )
                    next_mode_override = "paper"
                    continue
                manifest["reason"] = "coverage_extension_limit_reached_after_promising_paper_sweep"
                break
            if sweep_mode in {"scout", "medium"}:
                last_nonpaper_promising = promising
            if sweep_mode in {"scout", "medium"} and not promising:
                if redesign_rounds < redesign_round_limit:
                    try:
                        refined = _phase25_refine_sweep_plan(
                            run_dir=run_dir,
                            topic=topic,
                            model_profile=model_profile,
                            reason=f"{sweep_mode}_claim_not_promising",
                            round_index=round_index,
                        )
                        redesign_rounds += 1
                        manifest["redesigns"].append(
                            {
                                "round": round_index,
                                "trigger": f"{sweep_mode}_claim_not_promising",
                                "counts_against_redesign_limit": True,
                                "ok": bool(refined.get("figures")),
                                "status": refined.get("status", ""),
                                "claim_snapshot": snapshot,
                            }
                        )
                        if str(refined.get("status") or "") == "requires_phase24_design_revision" or bool(refined.get("requires_phase24_design_revision")):
                            phase24_design_revision_required = True
                            current_status = "requires_phase24_design_revision"
                            manifest["reason"] = f"requires_phase24_design_revision_after_{sweep_mode}"
                            _sync_experiment_agent(
                                run_dir,
                                event="phase25_requires_phase24_design_revision",
                                extra_metadata={
                                    "round": round_index,
                                    "trigger": f"{sweep_mode}_claim_not_promising",
                                    "claim_snapshot": snapshot,
                                    "refiner_result": refined,
                                },
                            )
                            break
                        next_mode_override = "scout" if sweep_mode == "scout" else "medium"
                        continue
                    except Exception as exc:  # noqa: BLE001
                        manifest["redesigns"].append(
                            {
                                "round": round_index,
                                "trigger": f"{sweep_mode}_claim_not_promising",
                                "counts_against_redesign_limit": True,
                                "ok": False,
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                                "claim_snapshot": snapshot,
                            }
                        )
                current_status = "requires_phase24_design_revision"
                manifest["reason"] = f"requires_phase24_design_revision_after_unpromising_{sweep_mode}"
                break
        if not str(manifest.get("reason") or "").startswith(("auto_expansion_failed", "auto_reanalysis_failed")):
            manifest["reason"] = (
                "paper_ready_after_auto_expansion"
                if _phase25_is_paper_ready(current_status)
                else (manifest.get("reason") or "auto_expansion_limit_reached_or_no_paper_ready_plan")
            )

    manifest["final_phase25_status"] = current_status
    phase25_dir = run_dir / "phase2-5"
    phase25_dir.mkdir(parents=True, exist_ok=True)
    write_json_artifact(phase25_dir / "phase25_auto_expansion_manifest.json", manifest)
    return current_result, current_status, manifest


def _sync_experiment_agent(
    run_dir: Path,
    *,
    event: str,
    write_request: bool = True,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Synchronize the WARA ExperimentAgent workspace with the current run layout."""

    agent = ExperimentAgent(run_dir)
    snapshot = agent.bootstrap(
        event=event,
        message="Synchronized ExperimentAgent with Phase 2.4 execution.",
        metadata=extra_metadata or {},
    )
    request_path: Path | None = None
    if write_request:
        request_path = agent.write_request_payload()
    payload: dict[str, Any] = {
        "event": event,
        "agent_id": agent.id,
        "snapshot": snapshot.to_dict(),
        "request_path": str(request_path) if request_path is not None else "",
        "workspace_manifest_path": str(Path(run_dir) / "agent_workspace_manifest.json"),
    }
    if extra_metadata:
        payload["metadata"] = extra_metadata
    target_dir = Path(run_dir) / "phase2-4"
    write_text(
        target_dir / f"experiment_agent_{event}.json",
        json.dumps(payload, ensure_ascii=False, indent=2),
    )
    return payload


def _sync_role_agent(
    run_dir: Path,
    agent_id: str,
    *,
    event: str,
    write_request: bool = True,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Synchronize a WARA role agent workspace with the current run layout."""

    agent = RoleAgent(run_dir, agent_id)
    snapshot = agent.bootstrap(
        event=event,
        message="Synchronized role agent with phase execution.",
        metadata=extra_metadata or {},
    )
    request_path: Path | None = None
    if write_request:
        request_path = agent.write_request_payload()
    payload: dict[str, Any] = {
        "event": event,
        "agent_id": agent.id,
        "snapshot": snapshot.to_dict(),
        "request_path": str(request_path) if request_path is not None else "",
        "workspace_manifest_path": str(Path(run_dir) / "agent_workspace_manifest.json"),
    }
    if extra_metadata:
        payload["metadata"] = extra_metadata

    target_dir = Path(run_dir) / "agent-sync"
    write_text(
        target_dir / f"{agent.id}_{event}.json",
        json.dumps(payload, ensure_ascii=False, indent=2),
    )
    return payload


def _load_phase3_5_review_routing(run_dir: Path, phase3_5_result: dict[str, Any] | None) -> dict[str, Any]:
    if isinstance(phase3_5_result, dict):
        routing = phase3_5_result.get("review_routing_decision")
        if isinstance(routing, dict):
            return routing
        nested = phase3_5_result.get("routing_decision")
        if isinstance(nested, dict):
            return nested
    routing = read_json(Path(run_dir) / "phase3-5" / "review_routing_decision.json")
    return routing if isinstance(routing, dict) else {}


def _record_phase3_5_controller_decision(
    run_dir: Path,
    phase3_5_result: dict[str, Any] | None,
    *,
    controller: WaraController,
) -> tuple[ControllerDecision, dict[str, Any]]:
    routing = _load_phase3_5_review_routing(run_dir, phase3_5_result)
    decision = controller.record_review_routing(
        routing,
        source_path="phase3-5/review_routing_decision.json",
    )
    phase3_5_dir = Path(run_dir) / "phase3-5"
    write_text(phase3_5_dir / "controller_review_decision.json", json.dumps(asdict(decision), ensure_ascii=False, indent=2))
    write_text(
        phase3_5_dir / "controller_review_route_manifest.json",
        json.dumps(
            {
                "routing_decision": routing,
                "controller_decision": asdict(decision),
                "controller_manifest": str(controller.manifest_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
    )
    return decision, routing


def _review_route_text(route: dict[str, Any]) -> str:
    raw_issue = route.get("raw_issue") if isinstance(route.get("raw_issue"), dict) else {}
    fields = [
        route.get("issue_id"),
        route.get("title"),
        route.get("routing_reason"),
        route.get("source"),
        raw_issue.get("issue_id"),
        raw_issue.get("title"),
        raw_issue.get("category"),
        raw_issue.get("issue"),
        raw_issue.get("why_it_matters"),
        raw_issue.get("exact_location"),
        raw_issue.get("suggested_action"),
        raw_issue.get("responsible_phase"),
    ]
    return "\n".join(str(item or "") for item in fields).lower()


def _review_route_targets_agent(route: dict[str, Any], owner_agent: str) -> bool:
    target_agent = str(route.get("target_agent", "")).strip()
    if target_agent == "implementation_agent":
        target_agent = "experiment_agent"
    normalized_owner = "experiment_agent" if owner_agent == "implementation_agent" else owner_agent
    return target_agent == normalized_owner


def _review_route_is_phase25_evidence_expansion(route: dict[str, Any]) -> bool:
    text = _review_route_text(route)
    phase25_terms = (
        "phase 2.5",
        "phase2.5",
        "phase25",
        "paper-ready",
        "paper ready",
        "quick-mode",
        "quick_mode",
        "draft figure",
        "draft figures",
        "final figure",
        "final figures",
        "figure registry",
        "evidence package",
        "experiment summary",
        "verified final figures",
        "paper-ready simulations",
        "numerical validation",
        "submission-grade support",
    )
    return any(term in text for term in phase25_terms)


def _phase_index_for_review_owner(owner_agent: str, routing: dict[str, Any] | None = None) -> int | None:
    owner = str(owner_agent or "").strip()
    if owner == "implementation_agent":
        owner = "experiment_agent"
    if owner == "experiment_agent" and isinstance(routing, dict):
        routes = routing.get("routes") if isinstance(routing.get("routes"), list) else []
        candidate_routes = [
            route
            for route in routes
            if isinstance(route, dict)
            and _review_route_targets_agent(route, owner)
            and str(route.get("priority", "")).strip() in {"P0", "P1"}
        ]
        if not candidate_routes and str(routing.get("target_agent") or "").strip() in {"experiment_agent", "implementation_agent"}:
            candidate_routes = [routing]
        if any(_review_route_is_phase25_evidence_expansion(route) for route in candidate_routes):
            return 4
    return REVIEW_OWNER_PHASE_INDEX.get(owner)


def _phase3_5_review_blocks_downstream(routing: dict[str, Any]) -> bool:
    """Return true only when review routes the primary repair outside Phase 3."""
    if not isinstance(routing, dict):
        return True
    primary_target = str(routing.get("target_agent") or routing.get("next_agent") or "").strip()
    if primary_target == "implementation_agent":
        primary_target = "experiment_agent"
    primary_reason = str(routing.get("primary_reason") or "").strip()
    if primary_target in {"writing_agent", "repair_agent"} or primary_reason in {"paper_writing_or_latex", "paper_polish_default"}:
        return False
    routes = routing.get("routes") if isinstance(routing.get("routes"), list) else []
    for route in routes:
        if not isinstance(route, dict):
            continue
        priority = str(route.get("priority", "")).strip()
        target_agent = str(route.get("target_agent", "")).strip()
        if target_agent == "implementation_agent":
            target_agent = "experiment_agent"
        if priority == "P0" and target_agent not in {"writing_agent", "repair_agent"}:
            return True
        if priority == "P1" and target_agent in {"experiment_agent", "literature_agent", "theory_agent"}:
            return True
    return False


def _phase3_to_phase2_route_budget_status(
    run_dir: Path,
    *,
    owner_agent: str,
    review_routing: dict[str, Any],
) -> dict[str, Any]:
    phase3_5_dir = Path(run_dir) / "phase3-5"
    route_budget_path = phase3_5_dir / "phase3_to_phase2_route_budget.json"
    existing = read_json(route_budget_path) or {}
    attempts = existing.get("attempts") if isinstance(existing, dict) else []
    if not isinstance(attempts, list):
        attempts = []
    limit = _phase3_to_phase2_route_round_limit()
    repair_phase_index = _phase_index_for_review_owner(owner_agent, review_routing)
    used = len(attempts)
    return {
        "status": "route_available" if limit > used and repair_phase_index is not None else "recommendation_only",
        "limit": limit,
        "attempts_used": used,
        "attempts_remaining": max(0, limit - used),
        "can_route_to_phase2": bool(limit > used and repair_phase_index is not None),
        "owner_agent": owner_agent,
        "recommended_phase_index": repair_phase_index,
        "attempts": attempts,
        "active_phase2_route_consumed": bool(existing.get("active_phase2_route_consumed", False)) if isinstance(existing, dict) else False,
        "path": str(route_budget_path),
    }


def _record_phase3_to_phase2_route_budget(
    run_dir: Path,
    *,
    owner_agent: str,
    review_routing: dict[str, Any],
    consume_attempt: bool,
) -> dict[str, Any]:
    phase3_5_dir = Path(run_dir) / "phase3-5"
    phase3_5_dir.mkdir(parents=True, exist_ok=True)
    route_budget_path = phase3_5_dir / "phase3_to_phase2_route_budget.json"
    status = _phase3_to_phase2_route_budget_status(
        run_dir,
        owner_agent=owner_agent,
        review_routing=review_routing,
    )
    attempts = list(status.get("attempts") or [])
    route_consumed = False
    if consume_attempt and status.get("can_route_to_phase2"):
        route_consumed = True
        attempts.append(
            {
                "attempt_index": len(attempts) + 1,
                "owner_agent": owner_agent,
                "recommended_phase_index": status.get("recommended_phase_index"),
                "primary_issue_id": review_routing.get("primary_issue_id") if isinstance(review_routing, dict) else None,
                "primary_reason": review_routing.get("primary_reason") if isinstance(review_routing, dict) else None,
                "consumed_at": utcnow_iso(),
            }
        )
    payload = {
        **status,
        "status": "route_consumed" if route_consumed else status.get("status"),
        "attempts": attempts,
        "attempts_used": len(attempts),
        "attempts_remaining": max(0, int(status.get("limit", 0) or 0) - len(attempts)),
        "active_phase2_route_consumed": bool(route_consumed or status.get("active_phase2_route_consumed", False)),
        "consume_attempt_requested": bool(consume_attempt),
        "route_consumed_this_call": route_consumed,
        "updated_at": utcnow_iso(),
        "policy": (
            "Phase 3 always exports a paper package after handoff. This budget controls whether a "
            "Phase 3 review finding also opens an active route back to Phase 2. Default limit is 0, "
            "so upstream findings are recorded as review recommendations only."
        ),
        "env_override": "WARA_PHASE3_TO_PHASE2_ROUTE_ROUNDS",
    }
    write_text(route_budget_path, json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def _phase3_6_final_revision_ready(phase3_6_result: dict[str, Any] | None) -> tuple[bool, list[str]]:
    result = phase3_6_result if isinstance(phase3_6_result, dict) else {}
    blockers: list[str] = []
    if not bool(result.get("ready_to_submit_estimate", False)):
        blockers.append("ready_to_submit_estimate is false")
    try:
        unresolved_count = int(result.get("unresolved_issue_count", 0) or 0)
    except Exception:
        unresolved_count = 1
    if unresolved_count:
        blockers.append(f"unresolved_issue_count={unresolved_count}")
    missing_reference_keys = result.get("missing_reference_keys", [])
    if missing_reference_keys:
        blockers.append(f"missing_reference_keys={missing_reference_keys}")
    if str(result.get("compile_status", "")).strip() != "ok":
        blockers.append(f"compile_status={result.get('compile_status')}")
    if str(result.get("reference_status", "")).strip() != "ok":
        blockers.append(f"reference_status={result.get('reference_status')}")
    experiment_status = str(result.get("experiment_status", "")).strip()
    blocked_experiment_statuses = {
        "",
        "quick_mode_only",
        "medium_mode_only",
        "needs_more_phase24_runs",
        "claim_failure_needs_redesign",
        "additional_experiment_needed",
        "no_new_experiment_flag",
    }
    if experiment_status in blocked_experiment_statuses:
        blockers.append(f"experiment_status={experiment_status or 'missing'}")
    ready_blockers = result.get("ready_to_submit_blockers", [])
    if ready_blockers:
        blockers.append(f"ready_to_submit_blockers={ready_blockers}")
    return not blockers, blockers


def _write_phase3_6_quality_gate(run_dir: Path, *, ok: bool, blockers: list[str], phase3_6_result: dict[str, Any] | None) -> None:
    phase3_6_dir = Path(run_dir) / "phase3-6"
    phase3_6_dir.mkdir(parents=True, exist_ok=True)
    write_text(
        phase3_6_dir / "phase3_6_quality_gate.json",
        json.dumps(
            {
                "status": "passed" if ok else "blocked",
                "blockers": blockers,
                "phase3_6_result": phase3_6_result if isinstance(phase3_6_result, dict) else {},
                "legacy_phase3_6_result": phase3_6_result if isinstance(phase3_6_result, dict) else {},
                "reason": (
                    "Phase 3.5 final revision is ready."
                    if ok
                    else "Phase 3.5 final revision still has unresolved review, reference, experiment, or compile blockers."
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
    )


def _load_phase3_frozen_inputs(run_dir: Path) -> tuple[str, dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load the frozen Phase-2 artifacts consumed by an independent Phase-3 runner."""

    mathematical_contract_json = read_text(run_dir / "phase2-1" / "mathematical_contract.json") or "{}"
    phase1_outputs: dict[str, Any] = {
        "mathematical_contract_json": mathematical_contract_json,
        "system_model_md": read_text(run_dir / "phase2-1" / "system_model.md"),
        "problem_formulation_md": read_text(run_dir / "phase2-1" / "problem_formulation.md"),
        "core_theory_package_md": read_text(run_dir / "phase2-1" / "core_theory_package.md"),
    }
    phase2_outputs: dict[str, Any] = {
        "convexity_audit_md": read_text(run_dir / "phase2-2" / "convexity_audit.md"),
        "reformulation_path_md": read_text(run_dir / "phase2-2" / "reformulation_path.md"),
    }
    phase3_outputs: dict[str, Any] = {
        "algorithm_md": read_text(run_dir / "phase2-3" / "algorithm.md"),
        "convergence_or_complexity_md": read_text(run_dir / "phase2-3" / "convergence_or_complexity.md"),
        "benchmark_definition_md": read_text(run_dir / "phase2-3" / "benchmark_definition.md"),
        "validation_principles_md": read_text(run_dir / "phase2-3" / "validation_principles.md"),
        "experiment_blueprint_md": read_text(run_dir / "phase2-3" / "experiment_blueprint.md"),
    }
    missing = [
        label
        for label, value in (
            ("phase2-1/mathematical_contract.json", mathematical_contract_json),
            ("phase2-1/system_model.md", phase1_outputs["system_model_md"]),
            ("phase2-1/problem_formulation.md", phase1_outputs["problem_formulation_md"]),
            ("phase2-2/convexity_audit.md", phase2_outputs["convexity_audit_md"]),
            ("phase2-2/reformulation_path.md", phase2_outputs["reformulation_path_md"]),
            ("phase2-3/algorithm.md", phase3_outputs["algorithm_md"]),
            ("phase2-3/convergence_or_complexity.md", phase3_outputs["convergence_or_complexity_md"]),
        )
        if not str(value or "").strip()
    ]
    if missing:
        raise ValueError("Phase 3 runner is missing frozen Phase-2 inputs: " + ", ".join(missing))
    return mathematical_contract_json, phase1_outputs, phase2_outputs, phase3_outputs


def execute_phase3_flow(
    *,
    run_dir: Path,
    state: Phase2RunState,
    topic: str,
    model_profile: str,
    phase1_run: Path | None,
    callbacks: Phase2FlowCallbacks,
    phase1_handoff: Path | None = None,
    phase2_to_phase3_handoff: dict[str, Any] | None = None,
    stop_after_phase: str | None = None,
) -> None:
    """Run Phase 3 as an independent controller over frozen Phase-2 artifacts."""

    if phase2_to_phase3_handoff is None:
        phase2_to_phase3_handoff = read_json(run_dir / "phase2-5" / "phase2_to_phase3_handoff.json") or {}
    if not isinstance(phase2_to_phase3_handoff, dict) or not phase2_to_phase3_handoff:
        raise ValueError("Phase 3 requires phase2-5/phase2_to_phase3_handoff.json from the Phase 2 runner.")

    mathematical_contract_json, phase1_outputs, phase2_outputs, phase3_outputs = _load_phase3_frozen_inputs(run_dir)

    if len(state.phases) > 5 and str(state.phases[5].get("status") or "") not in {"running", "done"}:
        state.complete_phase(4, 5)

    controller = _init_phase3_controller(
        run_dir=run_dir,
        topic=topic,
        model_profile=model_profile,
        phase1_handoff=phase1_handoff,
        phase1_run=phase1_run,
        phase2_to_phase3_handoff=phase2_to_phase3_handoff,
    )
    try:
        _run_phase3_1_technical_writing(
            run_dir=run_dir,
            topic=topic,
            model_profile=model_profile,
            callbacks=callbacks,
            controller=controller,
            mathematical_contract_json=mathematical_contract_json,
            phase1_outputs=phase1_outputs,
            phase2_outputs=phase2_outputs,
            phase3_outputs=phase3_outputs,
        )
        _sync_phase3_public_run(run_dir)
    except Exception:
        state.fail_phase(5, 6)
        raise
    if stop_after_phase in {"6", "phase3_1", "phase3.1", "3.1", "phase3.1_technical"}:
        state.complete_phase(5)
        return
    state.complete_phase(5, 6)
    try:
        callbacks.run_phase3_2_numerical_results_package(run_dir)
        _sync_phase3_public_run(run_dir)
    except Exception as exc:
        phase3_2_gate = read_json(run_dir / "phase3-2" / "phase3_2_evidence_gate.json") or {}
        _register_phase3_2_controller_artifacts(controller)
        _record_controller_gate(
            controller,
            "phase3_2_numerical_results_gate",
            ok=False,
            artifact_ids=["phase3_2_evidence_gate", "phase3_2_manifest", "phase3_2_numerical_results"],
            errors=phase3_2_gate.get("errors", [str(exc)]) if isinstance(phase3_2_gate, dict) else [str(exc)],
            warnings=phase3_2_gate.get("warnings", []) if isinstance(phase3_2_gate, dict) else [],
        )
        state.fail_phase(6, 7)
        raise
    phase3_2_gate = read_json(run_dir / "phase3-2" / "phase3_2_evidence_gate.json") or {"ok": True, "errors": [], "warnings": []}
    phase3_2_gate_ok = bool(phase3_2_gate.get("ok", True)) if isinstance(phase3_2_gate, dict) else True
    phase3_2_can_continue = phase3_2_gate_ok or (
        _phase3_2_has_conservative_llm_candidate(run_dir)
        and _phase3_2_evidence_scope_warning_exists(run_dir)
    )
    phase3_2_errors = phase3_2_gate.get("errors", []) if isinstance(phase3_2_gate, dict) else []
    phase3_2_warnings = phase3_2_gate.get("warnings", []) if isinstance(phase3_2_gate, dict) else []
    if phase3_2_can_continue and not phase3_2_gate_ok:
        phase3_2_warnings = [
            *phase3_2_warnings,
            "Phase 3.2 continued with an AnalysisAgent-generated conservative numerical-results candidate because Phase 2.5 evidence was finite but not paper-ready.",
            *[f"Evidence-scope limitation: {error}" for error in phase3_2_errors],
        ]
        phase3_2_errors = []
    _register_phase3_2_controller_artifacts(controller)
    _record_controller_gate(
        controller,
        "phase3_2_numerical_results_gate",
        ok=phase3_2_can_continue,
        artifact_ids=["phase3_2_evidence_gate", "phase3_2_manifest", "phase3_2_numerical_results"],
        errors=phase3_2_errors,
        warnings=phase3_2_warnings,
    )
    if not phase3_2_can_continue:
        state.block_phase(6)
        return
    state.complete_phase(6, 7)
    _sync_role_agent(
        run_dir,
        "analysis_agent",
        event="phase3_numerical_results_complete",
        extra_metadata={"phase": "phase3.2"},
    )
    phase3_3_result = callbacks.run_phase3_3_technical_sections_package(run_dir)
    _sync_phase3_public_run(run_dir)
    phase3_3_manifest = phase3_3_result if isinstance(phase3_3_result, dict) and phase3_3_result else read_json(run_dir / "phase3-3" / "phase3_3_manifest.json") or {}
    phase3_3_gate = _phase3_3_manifest_gate(phase3_3_manifest)
    _register_phase3_3_controller_artifacts(controller)
    _record_controller_gate(
        controller,
        "phase3_3_abstract_conclusion_gate",
        ok=bool(phase3_3_gate.get("ok", False)),
        artifact_ids=[
            "phase3_3_abstract",
            "phase3_3_keywords",
            "phase3_3_conclusion",
            "phase3_3_manifest",
            "phase3_3_technical_sections",
        ],
        errors=phase3_3_gate.get("errors", []),
        warnings=phase3_3_gate.get("warnings", []),
    )
    if not phase3_3_gate.get("ok", False):
        state.block_phase(7)
        return
    state.complete_phase(7, 8)
    _sync_role_agent(
        run_dir,
        "writing_agent",
        event="phase3_technical_sections_complete",
        extra_metadata={"phase": "phase3.3"},
    )
    phase3_4_result = callbacks.run_phase3_4_introduction_references_package(run_dir)
    _sync_phase3_public_run(run_dir)
    phase3_4_manifest = phase3_4_result if isinstance(phase3_4_result, dict) and phase3_4_result else read_json(run_dir / "phase3-4" / "phase3_4_manifest.json") or {}
    phase3_4_gate = _phase3_4_manifest_gate(run_dir, phase3_4_manifest)
    _register_phase3_4_controller_artifacts(controller)
    _record_controller_gate(
        controller,
        "phase3_4_reference_writing_gate",
        ok=bool(phase3_4_gate.get("ok", False)),
        artifact_ids=[
            "phase3_4_introduction",
            "phase3_4_full_paper",
            "references_bib",
            "verified_reference_bank",
            "citation_claim_map",
            "reference_quality_report",
            "phase3_4_reference_count_contract",
            "phase3_4_manifest",
        ],
        errors=phase3_4_gate.get("errors", []),
        warnings=phase3_4_gate.get("warnings", []),
    )
    if not phase3_4_gate.get("ok", False):
        state.block_phase(8)
        return
    _publish_final_paper_package(run_dir)
    _sync_role_agent(
        run_dir,
        "literature_agent",
        event="phase3_references_complete",
        extra_metadata={"phase": "phase3.4"},
    )
    _sync_role_agent(
        run_dir,
        "writing_agent",
        event="phase3_full_paper_complete",
        extra_metadata={"phase": "phase3.4"},
    )
    if stop_after_phase in {"9", "phase3_4", "phase3.4", "3.4", "phase3.4_references"}:
        state.complete_phase(8)
        return
    state.complete_phase(8, 9)
    phase3_5_result = callbacks.run_phase3_5_paper_review_package(run_dir)
    _sync_phase3_public_run(run_dir)
    controller_decision, review_routing = _record_phase3_5_controller_decision(
        run_dir,
        phase3_5_result,
        controller=controller,
    )
    _sync_role_agent(
        run_dir,
        "review_agent",
        event="phase3_review_complete",
        extra_metadata={
            "phase": "phase3.5",
            "review_routing_decision": review_routing,
            "controller_decision": asdict(controller_decision),
        },
    )
    if controller_decision.action == "final_ready":
        state.complete_phase(9)
        state.skip_phase(10)
        _publish_final_paper_package(run_dir)
        return

    if controller_decision.action == "repair":
        _sync_role_agent(
            run_dir,
            "repair_agent",
            event="phase3_repair_routed",
            write_request=True,
            extra_metadata={
                "phase": "phase3.5",
                "review_routing_decision": review_routing,
                "controller_decision": asdict(controller_decision),
            },
        )
        owner_agent = controller_decision.owner_agent or controller_decision.repair_scope
        if owner_agent not in {"writing_agent", "repair_agent"} and _phase3_5_review_blocks_downstream(review_routing):
            route_budget_status = _record_phase3_to_phase2_route_budget(
                run_dir,
                owner_agent=owner_agent,
                review_routing=review_routing,
                consume_attempt=False,
            )
            write_text(
                Path(run_dir) / "phase3-5" / "upstream_repair_recommendation.json",
                json.dumps(
                    {
                        "status": "upstream_repair_recommended",
                        "owner_agent": owner_agent,
                        "recommended_phase_index": route_budget_status.get("recommended_phase_index"),
                        "phase3_to_phase2_route_budget": route_budget_status,
                        "review_routing_decision": review_routing,
                        "policy": (
                            "Phase 3 continues to export a paper package after Phase 2 handoff. "
                            "The upstream route is recorded as a review recommendation rather than a Phase 3 export blocker."
                        ),
                        "created_at": utcnow_iso(),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            )
            state.complete_phase(9, 10)
            phase3_6_result = callbacks.run_phase3_6_apply_review_fixes_package(run_dir)
            _sync_phase3_public_run(run_dir)
            _sync_role_agent(
                run_dir,
                "repair_agent",
                event="phase3_repair_complete_with_upstream_recommendation",
                extra_metadata={
                    "phase": "phase3.6",
                    "owner_agent": owner_agent,
                    "review_routing_decision": review_routing,
                    "phase3_6_result": phase3_6_result if isinstance(phase3_6_result, dict) else {},
                },
            )
            _sync_role_agent(
                run_dir,
                "writing_agent",
                event="phase3_final_package_exported_with_review_findings",
                extra_metadata={
                    "phase": "phase3.6",
                    "controller_decision": asdict(controller_decision),
                    "upstream_repair_recommended": True,
                },
            )
            phase3_6_ready, phase3_6_blockers = _phase3_6_final_revision_ready(phase3_6_result)
            upstream_blocker = f"upstream_repair_recommended={owner_agent}"
            if upstream_blocker not in phase3_6_blockers:
                phase3_6_blockers.append(upstream_blocker)
            consumed_route_budget = _record_phase3_to_phase2_route_budget(
                run_dir,
                owner_agent=owner_agent,
                review_routing=review_routing,
                consume_attempt=True,
            )
            if consumed_route_budget.get("route_consumed_this_call"):
                phase3_6_blockers.append(
                    f"phase3_to_phase2_route_consumed={consumed_route_budget.get('recommended_phase_index')}"
                )
            _write_phase3_6_quality_gate(
                run_dir,
                ok=False,
                blockers=phase3_6_blockers,
                phase3_6_result=phase3_6_result if isinstance(phase3_6_result, dict) else {},
            )
            _register_phase3_6_controller_artifacts(controller)
            _record_controller_gate(
                controller,
                "phase3_6_final_revision_gate",
                ok=False,
                artifact_ids=[
                    "phase3_6_revision_manifest",
                    "phase3_6_quality_gate",
                    "phase3_6_revised_full_paper",
                    "phase3_6_revised_pdf",
                ],
                errors=phase3_6_blockers,
                warnings=[
                    "Paper package exported after Phase 3 handoff, but ReviewAgent recommends upstream repair.",
                ],
            )
            _publish_final_paper_package(run_dir)
            state.finish(10)
            if consumed_route_budget.get("route_consumed_this_call") and consumed_route_budget.get("recommended_phase_index") is not None:
                state.block_phase(int(consumed_route_budget["recommended_phase_index"]))
            return
        if owner_agent in {"writing_agent", "repair_agent"}:
            state.complete_phase(9, 10)
            phase3_6_result = callbacks.run_phase3_6_apply_review_fixes_package(run_dir)
            _sync_phase3_public_run(run_dir)
            _sync_role_agent(
                run_dir,
                "repair_agent",
                event="phase3_repair_complete",
                extra_metadata={
                    "phase": "phase3.6",
                    "owner_agent": owner_agent,
                    "phase3_6_result": phase3_6_result if isinstance(phase3_6_result, dict) else {},
                },
            )
            _sync_role_agent(
                run_dir,
                "writing_agent",
                event="phase3_final_revision_complete",
                extra_metadata={
                    "phase": "phase3.6",
                    "controller_decision": asdict(controller_decision),
                },
            )
            phase3_6_ready, phase3_6_blockers = _phase3_6_final_revision_ready(phase3_6_result)
            _write_phase3_6_quality_gate(
                run_dir,
                ok=phase3_6_ready,
                blockers=phase3_6_blockers,
                phase3_6_result=phase3_6_result if isinstance(phase3_6_result, dict) else {},
            )
            _register_phase3_6_controller_artifacts(controller)
            _record_controller_gate(
                controller,
                "phase3_6_final_revision_gate",
                ok=phase3_6_ready,
                artifact_ids=[
                    "phase3_6_revision_manifest",
                    "phase3_6_quality_gate",
                    "phase3_6_revised_full_paper",
                    "phase3_6_revised_pdf",
                ],
                errors=phase3_6_blockers,
                warnings=[],
            )
            _publish_final_paper_package(run_dir)
            if phase3_6_ready:
                state.finish(10)
            else:
                state.block_phase(10)
            return

        state.complete_phase(9)
        repair_phase_index = _phase_index_for_review_owner(owner_agent, review_routing)
        if repair_phase_index is not None:
            state.block_phase(repair_phase_index)
        else:
            state.block_phase(10)
        _publish_final_paper_package(run_dir)
        return

    _publish_final_paper_package(run_dir)
    state.complete_phase(9)
    state.block_phase(10)


def execute_phase2_flow(
    *,
    run_dir: Path,
    state: Phase2RunState,
    topic: str,
    model_profile: str,
    phase1_run: Path | None,
    docs_dir: Path,
    callbacks: Phase2FlowCallbacks,
    phase1_handoff: Path | None = None,
    stop_after_phase: str | None = None,
) -> None:
    handoff = None
    if phase1_handoff is not None and phase1_handoff.exists():
        handoff = build_wara_phase1_handoff(phase1_handoff, run_dir)
    elif phase1_run is not None and phase1_run.exists():
        handoff = callbacks.build_phase1_handoff(phase1_run, run_dir)
    controller = _init_phase2_controller(
        run_dir=run_dir,
        topic=topic,
        model_profile=model_profile,
        phase1_handoff=phase1_handoff,
        phase1_run=phase1_run,
    )

    topic_taxonomy = read_json(Path(handoff["handoff_dir"]) / "topic_taxonomy.json") if handoff else {}
    synthesis_md = read_text(Path(handoff["handoff_dir"]) / "synthesis.md") if handoff else ""

    write_text(docs_dir / "pipeline_experiment_design.md", callbacks.build_pipeline_experiment_design_notes())
    if handoff is not None:
        write_text(run_dir / "phase1_handoff_manifest.json", json.dumps(handoff, ensure_ascii=False, indent=2))
        selected_title = str(handoff.get("final_title") or "").strip()
        if selected_title:
            state.summary.selected_title = selected_title
            state.persist()
            controller.manifest["paper_title"] = selected_title
            controller.persist()
        _register_phase1_handoff_artifact(controller)

    phase1_outputs = callbacks.run_phase2_phase1_llm(
        run_dir=run_dir,
        topic=topic,
        handoff=handoff or {},
        topic_taxonomy=topic_taxonomy or {},
        synthesis_md=synthesis_md,
        model_profile=model_profile,
    )

    mathematical_contract_json = str(phase1_outputs.get("mathematical_contract_json") or "{}")
    write_text(run_dir / "phase2-1" / "mathematical_contract.json", mathematical_contract_json)
    write_text(run_dir / "phase2-1" / "mathematical_contract.frozen.json", mathematical_contract_json)
    write_text(
        run_dir / "phase2-1" / "frozen_math_interface.md",
        _build_frozen_math_interface_markdown(mathematical_contract_json),
    )
    write_text(run_dir / "phase2-1" / "system_model.md", phase1_outputs["system_model_md"])
    write_text(run_dir / "phase2-1" / "problem_formulation.md", phase1_outputs["problem_formulation_md"])
    write_text(run_dir / "phase2-1" / "core_theory_package.md", phase1_outputs["core_theory_package_md"])
    problem_contract = build_problem_contract(
        topic=topic,
        handoff=handoff or {},
        system_model_md=phase1_outputs["system_model_md"],
        problem_formulation_md=phase1_outputs["problem_formulation_md"],
        core_theory_package_md=phase1_outputs["core_theory_package_md"],
        mathematical_contract_json=mathematical_contract_json,
    )
    model_audit = audit_model_contract(
        problem_contract=problem_contract,
        system_model_md=phase1_outputs["system_model_md"],
        problem_formulation_md=phase1_outputs["problem_formulation_md"],
        core_theory_package_md=phase1_outputs["core_theory_package_md"],
    )
    write_json_artifact(run_dir / "phase2-1" / "problem_contract.json", problem_contract)
    write_json_artifact(run_dir / "phase2-1" / "model_audit.json", model_audit)
    model_audit_ok = bool(model_audit.get("ok", False))
    phase21_selected_after_budget = bool(phase1_outputs.get("selected_after_repair_budget"))
    phase21_promoted = model_audit_ok or phase21_selected_after_budget
    _register_phase21_controller_artifacts(controller, freeze_contracts=phase21_promoted)
    _record_controller_gate(
        controller,
        "formulation_gate",
        ok=model_audit_ok,
        artifact_ids=["mathematical_contract", "problem_contract", "model_audit"],
        errors=model_audit.get("errors", []),
        warnings=(
            list(model_audit.get("warnings", []))
            + (
                [
                    "Phase 2.1 repair budget was exhausted; the controller promoted the highest-scoring LLM FormulationAgent candidate."
                ]
                if phase21_selected_after_budget and not model_audit_ok
                else []
            )
        ),
    )
    if not model_audit_ok:
        write_text(
            run_dir / "phase2-1" / "model_audit_blocking_errors.txt",
            "\n".join(str(item) for item in model_audit.get("errors", [])),
        )
        if not phase21_selected_after_budget:
            state.fail_phase(0, 1)
            return
        write_text(
            run_dir / "phase2-1" / "selected_after_repair_budget.md",
            "# Phase 2.1 Candidate Promotion\n\n"
            "The formulation gate did not fully pass after the configured LLM repair budget. "
            "The controller is continuing with the highest-scoring FormulationAgent candidate. "
            "No deterministic formulation text was generated.\n",
        )
    write_text(
        run_dir / "phase2-1" / "writing_deferred_to_phase3.md",
        "# Writing Deferred\n\n"
        "Phase 2.1 now produces the frozen mathematical interface, system-model facts, "
        "original-problem facts, and convexity/tractability notes. The final IEEE LaTeX prose for "
        "System Model and Problem Formulation is written together with the proposed-method "
        "section in Phase 3.1 so that notation and transitions are continuous.\n",
    )

    state.complete_phase(0, 1)
    _sync_role_agent(
        run_dir,
        "formulation_agent",
        event="phase2_formulation_complete",
        extra_metadata={"phase": "phase2.1"},
    )

    tractability_route_policy = build_tractability_route_policy(
        topic=topic,
        handoff=handoff or {},
        mathematical_contract_json=mathematical_contract_json,
        system_model_md=phase1_outputs["system_model_md"],
        problem_formulation_md=phase1_outputs["problem_formulation_md"],
        core_theory_package_md=phase1_outputs["core_theory_package_md"],
        problem_contract=problem_contract,
    )
    write_json_artifact(run_dir / "phase2-2" / "tractability_route_policy.json", tractability_route_policy)

    phase2_outputs = callbacks.run_phase2_phase2_llm(
        run_dir=run_dir,
        topic=topic,
        handoff=handoff or {},
        mathematical_contract_json=mathematical_contract_json,
        system_model_md=phase1_outputs["system_model_md"],
        problem_formulation_md=phase1_outputs["problem_formulation_md"],
        core_theory_package_md=phase1_outputs["core_theory_package_md"],
        tractability_route_policy=tractability_route_policy,
        model_profile=model_profile,
    )
    write_text(run_dir / "phase2-2" / "convexity_audit.md", phase2_outputs["convexity_audit_md"])
    write_text(run_dir / "phase2-2" / "reformulation_path.md", phase2_outputs["reformulation_path_md"])
    algorithm_contract = build_algorithm_contract(
        topic=topic,
        problem_contract=problem_contract,
        convexity_audit_md=phase2_outputs["convexity_audit_md"],
        reformulation_path_md=phase2_outputs["reformulation_path_md"],
    )
    write_json_artifact(run_dir / "phase2-2" / "algorithm_contract.json", algorithm_contract)
    algorithm_execution_contract = algorithm_contract.get("algorithm_execution_contract") if isinstance(algorithm_contract, dict) else None
    algorithm_gate_errors = [] if isinstance(algorithm_execution_contract, dict) and algorithm_execution_contract else [
        "algorithm_contract is missing a non-empty algorithm_execution_contract"
    ]
    phase22_selected_after_budget = bool(phase2_outputs.get("selected_after_repair_budget"))
    algorithm_gate_promoted = not algorithm_gate_errors or phase22_selected_after_budget
    algorithm_gate_warnings: list[str] = []
    if phase22_selected_after_budget and algorithm_gate_errors:
        algorithm_gate_warnings.append(
            "Phase 2.2 algorithm contract gate did not fully pass after the repair budget; "
            "the controller is advancing the highest-scoring TheoryAgent tractability candidate without generating replacement technical content."
        )
    _register_phase22_controller_artifacts(controller, freeze_contracts=algorithm_gate_promoted)
    _record_controller_gate(
        controller,
        "algorithm_contract_gate",
        ok=algorithm_gate_promoted,
        artifact_ids=["tractability_route_policy", "convexity_audit", "reformulation_path", "algorithm_contract"],
        errors=algorithm_gate_errors,
        warnings=algorithm_gate_warnings,
    )
    if algorithm_gate_errors and not algorithm_gate_promoted:
        write_text(run_dir / "phase2-2" / "algorithm_contract_blocking_errors.txt", "\n".join(algorithm_gate_errors))
        state.fail_phase(1, 2)
        return
    if phase22_selected_after_budget:
        write_text(
            run_dir / "phase2-2" / "selected_after_repair_budget.md",
            "# Selected Tractability Candidate After Repair Budget\n\n"
            "The Phase 2.2 gate promoted the highest-scoring LLM-generated candidate after the repair budget. "
            "No deterministic tractability or reformulation text was generated.\n",
        )

    state.complete_phase(1, 2)
    _sync_role_agent(
        run_dir,
        "theory_agent",
        event="phase2_reformulation_complete",
        extra_metadata={"phase": "phase2.2", "algorithm_description_pending": True},
    )

    phase3_outputs = callbacks.run_phase2_phase3_llm(
        run_dir=run_dir,
        topic=topic,
        handoff=handoff or {},
        mathematical_contract_json=mathematical_contract_json,
        system_model_md=phase1_outputs["system_model_md"],
        problem_formulation_md=phase1_outputs["problem_formulation_md"],
        core_theory_package_md=phase1_outputs["core_theory_package_md"],
        convexity_audit_md=phase2_outputs["convexity_audit_md"],
        reformulation_path_md=phase2_outputs["reformulation_path_md"],
        tractability_route_policy=tractability_route_policy,
        model_profile=model_profile,
    )
    write_text(run_dir / "phase2-3" / "algorithm.md", phase3_outputs["algorithm_md"])
    write_text(run_dir / "phase2-3" / "convergence_or_complexity.md", phase3_outputs["convergence_or_complexity_md"])
    write_text(run_dir / "phase2-3" / "proof_skeleton.md", phase3_outputs["convergence_or_complexity_md"])
    write_text(run_dir / "phase2-3" / "benchmark_definition.md", phase3_outputs["benchmark_definition_md"])
    write_text(run_dir / "phase2-3" / "validation_principles.md", phase3_outputs["validation_principles_md"])
    write_text(run_dir / "phase2-3" / "experiment_blueprint.md", phase3_outputs.get("experiment_blueprint_md", phase3_outputs["validation_principles_md"]))
    write_text(run_dir / "phase2-3" / "phase3_design_notes.md", callbacks.build_phase3_design_notes())
    theory_audit = audit_theory_contract(
        algorithm_contract=algorithm_contract,
        algorithm_md=phase3_outputs["algorithm_md"],
        convergence_or_complexity_md=phase3_outputs["convergence_or_complexity_md"],
    )
    claim_map = build_claim_map(
        topic=topic,
        problem_contract=problem_contract,
        algorithm_contract=algorithm_contract,
        algorithm_md=phase3_outputs["algorithm_md"],
        convergence_or_complexity_md=phase3_outputs["convergence_or_complexity_md"],
    )
    write_json_artifact(run_dir / "phase2-3" / "theory_audit.json", theory_audit)
    write_json_artifact(run_dir / "phase2-3" / "claim_map.json", claim_map)
    theory_audit_ok = bool(theory_audit.get("ok", False))
    phase23_selected_after_budget = bool(phase3_outputs.get("selected_after_repair_budget"))
    theory_gate_promoted = theory_audit_ok or phase23_selected_after_budget
    theory_gate_warnings = list(theory_audit.get("warnings", []))
    if phase23_selected_after_budget and not theory_audit_ok:
        theory_gate_warnings.append(
            "Phase 2.3 theory audit did not fully pass after the repair budget; "
            "the controller is advancing the highest-scoring TheoryAgent candidate without generating replacement technical content."
        )
    _register_phase23_controller_artifacts(controller, freeze_claim_map=theory_gate_promoted)
    _record_controller_gate(
        controller,
        "theory_gate",
        ok=theory_gate_promoted,
        artifact_ids=["algorithm_description", "theory_audit", "claim_map"],
        errors=theory_audit.get("errors", []),
        warnings=theory_gate_warnings,
    )
    if not theory_gate_promoted:
        write_text(
            run_dir / "phase2-3" / "theory_audit_blocking_errors.txt",
            "\n".join(str(item) for item in theory_audit.get("errors", [])),
        )
        state.fail_phase(2, 3)
        return
    if phase23_selected_after_budget and not theory_audit_ok:
        write_text(
            run_dir / "phase2-3" / "selected_after_repair_budget.md",
            "# Selected TheoryAgent Candidate After Repair Budget\n\n"
            "The Phase 2.3 gate promoted the highest-scoring LLM-generated candidate after the repair budget. "
            "No deterministic algorithm or theory text was generated.\n",
        )

    phase3_dir = run_dir / "phase2-3"
    write_text(
        phase3_dir / "writing_deferred_to_phase3.md",
        "# Writing Deferred\n\n"
        "Phase 2.3 now produces algorithm-design facts, auxiliary-variable roles, "
        "algorithm flow, and convergence/complexity notes. The final IEEE LaTeX prose "
        "for the proposed-method section is written in Phase 3.1 together with the "
        "System Model and Problem Formulation section.\n",
    )

    state.complete_phase(2, 3)
    _sync_role_agent(
        run_dir,
        "theory_agent",
        event="phase2_algorithm_complete",
        extra_metadata={"phase": "phase2.3"},
    )

    phase24_dir = run_dir / "phase2-4"
    solver_dir = phase24_dir / "solver"
    wireless_benchmark_plan = select_wireless_benchmark_plan(
        topic=topic,
        problem_contract=problem_contract,
        algorithm_contract=algorithm_contract,
        claim_map=claim_map,
    )
    experiment_design_contract = build_experiment_design_contract(
        problem_contract=problem_contract,
        benchmark_plan=wireless_benchmark_plan,
        claim_map=claim_map,
    )
    write_json_artifact(phase24_dir / "wireless_benchmark_plan.json", wireless_benchmark_plan)
    write_json_artifact(phase24_dir / "experiment_design_contract.json", experiment_design_contract)
    phase24_evidence_contract_summary = (
        "[Phase 2.4-owned evidence contract]\n"
        "Use the structured Phase 2.4 contracts below as the "
        "source of truth for claims, compared methods, sweeps, metrics, figure targets, and table targets.\n"
        + contract_prompt_block(
            benchmark_plan=wireless_benchmark_plan,
            experiment_design_contract=experiment_design_contract,
        )
    )
    benchmark_definition_for_phase24 = (
        "[Phase 2.4 WirelessBenchmarkAgent benchmark contract]\n"
        + json.dumps(wireless_benchmark_plan, ensure_ascii=False, indent=2)
    )
    design_repair_round_limit = max(0, int(os.environ.get("WARA_PHASE24_DESIGN_REPAIR_ROUNDS", "10") or 10))
    phase24_validation_yaml = ""
    phase24_execution_contract: dict[str, Any] = {}
    evidence_design_status: dict[str, Any] = {}
    evidence_design_ok = False
    for design_round in range(0, design_repair_round_limit + 1):
        design_feedback = ""
        if design_round > 0:
            previous_errors = "\n".join(str(item) for item in evidence_design_status.get("errors", []))
            previous_warnings = "\n".join(str(item) for item in evidence_design_status.get("warnings", []))
            design_feedback = (
                "\n\n[Phase 2.4 design-gate retry feedback]\n"
                f"Retry round {design_round}/{design_repair_round_limit}. The previous validation plan failed the experiment-design gate.\n"
                f"Errors:\n{previous_errors or '(none)'}\n"
                f"Warnings:\n{previous_warnings or '(none)'}\n\n"
                "Revise the validation plan instead of bypassing the gate. If the paper objective is scalarized or weighted "
                "(for example utility_U_alpha), do not make every final figure use that objective-like y_metric. Use decomposed "
                "paper-facing physical KPIs tied to the claim, such as sum_rate_bpsHz, min_user_rate_bpsHz, sensing_snr_dB, "
                "sensing_snr_linear, CRB/MSE, harvested power, reliability, or energy efficiency when those columns are available. "
                "If the paper objective is resource minimization, do not plot resource usage against a resource upper bound as "
                "the main evidence. Sweep an active service demand, load, channel-severity, uncertainty, reliability target, "
                "mobility/deployment, or mechanism parameter. Preserve frozen method ids, objective semantics, physical constraints, "
                "and the same plotted benchmark set across final figures."
            )
            write_text(phase24_dir / f"phase24_design_retry_feedback_round{design_round}.txt", design_feedback)
        phase24_validation_yaml = callbacks.run_phase2_phase24_validation_llm(
            run_dir=run_dir,
            topic=topic,
            handoff=handoff or {},
            mathematical_contract_json=mathematical_contract_json,
            system_model_md=phase1_outputs["system_model_md"],
            problem_formulation_md=phase1_outputs["problem_formulation_md"],
            convexity_audit_md=phase2_outputs["convexity_audit_md"],
            reformulation_path_md=phase2_outputs["reformulation_path_md"],
            experiment_blueprint_md=phase24_evidence_contract_summary + design_feedback,
            model_profile=model_profile,
        )
        if design_round > 0:
            write_text(phase24_dir / f"validation_plan_design_retry_round{design_round}.yaml", phase24_validation_yaml)
        phase24_validation_plan = _parse_phase24_validation_plan_text(phase24_validation_yaml)
        phase24_execution_contract = build_phase24_execution_contract(
            validation_plan=phase24_validation_plan,
            problem_contract=problem_contract,
            algorithm_contract=algorithm_contract,
            benchmark_plan=wireless_benchmark_plan,
        )
        write_json_artifact(phase24_dir / "phase24_execution_contract.json", phase24_execution_contract)
        write_json_artifact(
            phase24_dir / "phase24_validation_source_contracts.json",
            {
                "mathematical_contract": json.loads(mathematical_contract_json or "{}"),
                "phase24_execution_contract": phase24_execution_contract,
                "wireless_benchmark_plan": wireless_benchmark_plan,
                "experiment_design_contract": experiment_design_contract,
                "claim_map": claim_map,
            },
        )
        write_text(phase24_dir / "validation_plan.yaml", phase24_validation_yaml)
        write_text(solver_dir / "validation_plan.yaml", phase24_validation_yaml)
        _sync_experiment_agent(
            run_dir,
            event="preflight",
            extra_metadata={
                "legacy_executor": "phase2.4",
                "executor": "phase2.4",
                "validation_plan_written": True,
                "phase24_execution_contract_written": True,
                "design_retry_round": design_round,
            },
        )

        evidence_design_status = callbacks.validate_phase24_evidence_contract_design(run_dir)
        evidence_design_status["design_retry_round"] = design_round
        evidence_design_status["design_repair_round_limit"] = design_repair_round_limit
        write_text(
            phase24_dir / "phase24_evidence_contract_design_check.json",
            json.dumps(evidence_design_status, ensure_ascii=False, indent=2),
        )
        if design_round > 0:
            write_text(
                phase24_dir / f"phase24_evidence_contract_design_check_round{design_round}.json",
                json.dumps(evidence_design_status, ensure_ascii=False, indent=2),
            )
        evidence_design_ok = bool(evidence_design_status.get("ok"))
        if evidence_design_ok:
            break
    _register_phase24_design_artifacts(controller, freeze_contracts=evidence_design_ok)
    _record_controller_gate(
        controller,
        "phase24_experiment_design_gate",
        ok=evidence_design_ok,
        artifact_ids=[
            "wireless_benchmark_plan",
            "experiment_design_contract",
            "validation_plan",
            "phase24_execution_contract",
        ],
        errors=evidence_design_status.get("errors", []),
        warnings=evidence_design_status.get("warnings", []),
    )
    if not evidence_design_ok:
        error_path = phase24_dir / "phase24_validation_error.txt"
        write_text(error_path, "[research_evidence_contract_design]\n" + "\n".join(evidence_design_status.get("errors", [])))
        write_text(
            phase24_dir / "phase24_validation_manifest.json",
            json.dumps(
                {
                    "status": "evidence_contract_design_failed",
                    "returncode": 1,
                    "error_path": str(error_path),
                    "repair_attempted": design_repair_round_limit > 0,
                    "design_repair_rounds": int(evidence_design_status.get("design_retry_round", 0) or 0),
                    "design_repair_round_limit": design_repair_round_limit,
                },
                ensure_ascii=False,
                indent=2,
            ),
        )
        _sync_experiment_agent(
            run_dir,
            event="blocked_evidence_design",
            extra_metadata={"evidence_design_status": evidence_design_status},
        )
        state.fail_phase(3, 4)
        return

    phase24_benchmark = callbacks.run_phase2_phase24_benchmark_llm(
        run_dir=run_dir,
        topic=topic,
        handoff=handoff or {},
        mathematical_contract_json=mathematical_contract_json,
        system_model_md=phase1_outputs["system_model_md"],
        problem_formulation_md=phase1_outputs["problem_formulation_md"],
        benchmark_definition_md=benchmark_definition_for_phase24,
        model_profile=model_profile,
    )
    write_text(phase24_dir / "benchmark_plan.md", phase24_benchmark["benchmark_plan_md"])
    write_text(solver_dir / "README.md", phase24_benchmark["solver_readme_md"])
    phase24_harness_manifest = {
        "mode": "fixed_harness_split_core",
        "fixed_files": [
            "problem_data.py",
            "validation_cases.py",
            "run_validation.py",
            "generated_plugin.py",
        ],
        "generated_file": "generated_experiment_core.py",
        "adapter_file": "generated_plugin.py",
        "plugin_exports": [
            "build_model",
            "initial_state",
            "proposed_step",
            "baseline_solution",
            "evaluate_state",
        ],
    }
    write_text(phase24_dir / "phase24_harness_manifest.json", json.dumps(phase24_harness_manifest, ensure_ascii=False, indent=2))
    write_text(phase24_dir / "phase24_design_notes.md", callbacks.build_phase24_design_notes())

    _phase24_prepare_clean_generation_workspace(phase24_dir, solver_dir)
    callbacks.write_phase2_phase24_fixed_harness(run_dir)
    generated_plugin = callbacks.run_phase2_phase24_plugin_llm(
        run_dir=run_dir,
        topic=topic,
        mathematical_contract_json=mathematical_contract_json,
        system_model_md=phase1_outputs["system_model_md"],
        problem_formulation_md=phase1_outputs["problem_formulation_md"],
        reformulation_path_md=phase2_outputs["reformulation_path_md"],
        algorithm_md=phase3_outputs["algorithm_md"],
        benchmark_definition_md=benchmark_definition_for_phase24,
        experiment_blueprint_md=phase24_evidence_contract_summary,
        model_profile=model_profile,
    )
    write_text(solver_dir / "generated_plugin.py", generated_plugin)

    validation_status = callbacks.validate_phase2_phase24_plugin_bundle(run_dir)
    validation_status["repair_attempted"] = False
    repair_rounds = 0
    repair_round_limit = _phase24_repair_round_limit()
    original_prompt = read_text(phase24_dir / "phase24_generated_plugin_prompt.txt")
    current_plugin_code = generated_plugin

    phase24_candidates_dir = phase24_dir / "bounded_repair_candidates"
    phase24_candidates_dir.mkdir(parents=True, exist_ok=True)
    phase24_candidate_records: list[dict[str, Any]] = []

    def _record_phase24_candidate(label: str, plugin_code: str, status: dict[str, Any]) -> None:
        score = _phase24_validation_candidate_score(run_dir, status)
        plugin_path = phase24_candidates_dir / f"{label}.py"
        status_path = phase24_candidates_dir / f"{label}.json"
        record = {
            "label": label,
            "score": score,
            "status": status.get("status"),
            "plugin_path": str(plugin_path),
            "status_path": str(status_path),
        }
        write_text(plugin_path, plugin_code)
        write_text(status_path, json.dumps({**status, "selection_score": score}, ensure_ascii=False, indent=2))
        phase24_candidate_records.append(record)

    _record_phase24_candidate("initial", current_plugin_code, validation_status)
    while callbacks.phase24_validation_allows_repair(validation_status) and repair_rounds < repair_round_limit:
        validation_error_text = callbacks.phase24_validation_error_text(run_dir, validation_status)
        try:
            repaired_plugin = callbacks.repair_phase2_phase24_plugin_llm(
                run_dir=run_dir,
                topic=topic,
                original_prompt=original_prompt,
                current_plugin_code=current_plugin_code,
                validation_status=validation_status,
                validation_error_text=validation_error_text,
                model_profile=model_profile,
            )
        except Exception as exc:  # noqa: BLE001 - keep the strongest candidate and continue policy decisions below.
            validation_status["repair_exception_type"] = type(exc).__name__
            validation_status["repair_exception"] = str(exc)
            break
        repair_rounds += 1
        current_plugin_code = repaired_plugin
        write_text(solver_dir / "generated_plugin.py", repaired_plugin)
        write_text(phase24_dir / f"phase24_generated_plugin_repaired_round{repair_rounds}.py", repaired_plugin)
        validation_status = callbacks.validate_phase2_phase24_plugin_bundle(run_dir)
        validation_status["repair_attempted"] = True
        validation_status["repair_rounds"] = repair_rounds
        validation_status["repair_round_limit"] = repair_round_limit
        _record_phase24_candidate(f"repair_round_{repair_rounds}", current_plugin_code, validation_status)
    validation_status.setdefault("repair_rounds", repair_rounds)
    validation_status.setdefault("repair_round_limit", repair_round_limit)
    write_text(
        phase24_dir / "phase24_validation_manifest.json",
        json.dumps(validation_status, ensure_ascii=False, indent=2),
    )
    if validation_status.get("status") != "ok" and phase24_candidate_records:
        selected_record = max(phase24_candidate_records, key=lambda item: float(item.get("score") or 0.0))
        selected_plugin = read_text(Path(str(selected_record.get("plugin_path"))))
        if selected_plugin and selected_plugin != current_plugin_code:
            current_plugin_code = selected_plugin
            write_text(solver_dir / "generated_plugin.py", current_plugin_code)
            validation_status = callbacks.validate_phase2_phase24_plugin_bundle(run_dir)
            validation_status["repair_attempted"] = repair_rounds > 0
            validation_status["repair_rounds"] = repair_rounds
            validation_status["repair_round_limit"] = repair_round_limit
        validation_status["selected_candidate"] = selected_record
        write_text(
            phase24_dir / "phase24_selected_candidate.json",
            json.dumps({"selected": selected_record, "candidates": phase24_candidate_records}, ensure_ascii=False, indent=2),
        )
    if validation_status.get("status") != "ok" and _phase24_selected_candidate_can_continue(run_dir, validation_status):
        validation_status = _phase24_mark_selected_candidate_validation(run_dir, validation_status)
    implementation_audit = audit_implementation_contract(
        run_dir=run_dir,
        generated_plugin=current_plugin_code,
        validation_status=validation_status,
    )
    write_json_artifact(phase24_dir / "implementation_audit.json", implementation_audit)
    implementation_audit_ok = bool(implementation_audit.get("ok", False))
    implementation_gate_errors = list(implementation_audit.get("errors", []))
    if not implementation_audit_ok and str(validation_status.get("status") or "") != "ok":
        validation_error_text = callbacks.phase24_validation_error_text(run_dir, validation_status)
        if validation_error_text.strip():
            implementation_gate_errors.append(validation_error_text)
    _register_phase24_implementation_artifacts(controller)
    _record_controller_gate(
        controller,
        "implementation_gate",
        ok=implementation_audit_ok,
        artifact_ids=["generated_plugin", "phase24_validation_manifest", "implementation_audit"],
        errors=implementation_gate_errors,
        warnings=implementation_audit.get("warnings", []),
    )
    if not implementation_audit_ok:
        write_text(
            phase24_dir / "implementation_audit_blocking_errors.txt",
            "\n".join(str(item) for item in implementation_audit.get("errors", [])),
        )
        _sync_experiment_agent(
            run_dir,
            event="blocked_implementation_audit",
            extra_metadata={
                "validation_status": validation_status,
                "implementation_audit": implementation_audit,
            },
        )
        state.fail_phase(3, 4)
        return

    _sync_experiment_agent(
        run_dir,
        event="phase24_complete",
        extra_metadata={
            "validation_status": validation_status,
            "implementation_audit": implementation_audit,
        },
    )
    state.complete_phase(3, 4)

    if validation_status.get("status") in {"ok", "selected_after_bounded_repairs"}:
        phase25_result = callbacks.run_phase25_wcl_package(run_dir)
        phase25_status = _phase25_status_from_manifest(run_dir, phase25_result)
        phase25_result, phase25_status, phase25_auto_expansion = _run_phase25_auto_paper_expansion(
            run_dir=run_dir,
            callbacks=callbacks,
            topic=topic,
            model_profile=model_profile,
            initial_result=phase25_result,
            initial_status=phase25_status,
        )
        evidence_audit = audit_phase25_evidence(
            run_dir=run_dir,
            phase25_result=phase25_result,
            phase25_status=phase25_status,
        )
        write_json_artifact(run_dir / "phase2-5" / "evidence_audit.json", evidence_audit)
        bounded_budget_continuation = _phase25_bounded_budget_continuation_enabled()
        phase25_gate_ok = (
            _phase25_is_paper_ready(phase25_status)
            and bool(evidence_audit.get("ok", False))
        ) or bounded_budget_continuation
        phase25_gate_errors = list(evidence_audit.get("errors", []))
        if not _phase25_is_paper_ready(phase25_status):
            phase25_gate_errors.append(f"phase25_status is not paper-ready: {phase25_status}")
        write_text(
            run_dir / "phase2-5" / "phase25_quality_gate.json",
            json.dumps(
                {
                    "status": "passed" if phase25_gate_ok else "blocked",
                    "phase25_status": phase25_status,
                    "evidence_audit_ok": evidence_audit.get("ok", False),
                    "evidence_audit_errors": evidence_audit.get("errors", []),
                    "auto_expansion": phase25_auto_expansion,
                    "allowed_statuses": sorted(PAPER_READY_PHASE25_STATUSES),
                    "paper_ready": _phase25_is_paper_ready(phase25_status),
                    "bounded_budget_continuation": bounded_budget_continuation
                    and not (_phase25_is_paper_ready(phase25_status) and bool(evidence_audit.get("ok", False))),
                    "reason": (
                        "Phase 2.5 produced paper-ready experimental evidence."
                        if _phase25_is_paper_ready(phase25_status) and bool(evidence_audit.get("ok", False))
                        else (
                            "Phase 2.5 completed its bounded expansion budget; the controller is continuing with the selected experimental evidence package and preserving evidence-scope metadata for Phase 3."
                            if bounded_budget_continuation
                            else "Phase 2.5 evidence is not paper-ready; stop before drafting final paper sections."
                        )
                    ),
                },
                ensure_ascii=False,
                indent=2,
            ),
        )
        _register_phase25_controller_artifacts(controller)
        _record_controller_gate(
            controller,
            "phase25_evidence_gate",
            ok=phase25_gate_ok,
            artifact_ids=["phase25_experiment_summary", "evidence_audit"],
            errors=phase25_gate_errors,
            warnings=evidence_audit.get("warnings", []),
        )
        _sync_experiment_agent(
            run_dir,
            event="phase25_evidence_gate",
            extra_metadata={
                "phase25_status": phase25_status,
                "evidence_audit": evidence_audit,
                "auto_expansion": phase25_auto_expansion,
            },
        )
        if not phase25_gate_ok:
            state.block_phase(4)
            return

        phase2_to_phase3_handoff = _write_phase2_to_phase3_handoff(
            run_dir,
            phase25_gate_ok=phase25_gate_ok,
            phase25_status=phase25_status,
            evidence_audit=evidence_audit,
            phase25_auto_expansion=phase25_auto_expansion,
        )
        # Phase 2 owns only modeling, algorithm construction, experiments, and
        # the frozen evidence handoff. Paper-facing writing must be launched by
        # the independent Phase 3 runner over phase2-5/phase2_to_phase3_handoff.json.
        state.complete_phase(4)
        return

    _sync_experiment_agent(
        run_dir,
        event="blocked_validation_status",
        extra_metadata={"validation_status": validation_status},
    )
    state.fail_phase(3, 4)
