from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from .utils import read_json, read_text, write_text


def looks_like_phase1_handoff(path: Path) -> bool:
    handoff_file = _handoff_file(path)
    if handoff_file is None or not handoff_file.exists():
        return False
    payload = read_json(handoff_file)
    return isinstance(payload, dict) and isinstance(payload.get("selected_candidate"), dict)


def build_wara_phase1_handoff(phase1_handoff: Path, run_dir: Path) -> dict[str, Any]:
    """Normalize a WARA-native Phase 1 handoff into Phase 2's input directory.

    WARA has a clean Phase 2/3 flow (`phase2.1-phase2.5` and
    `phase3.2-phase3.6`), but its
    downstream prompts still expect a small `input_from_phase1` directory with a
    few legacy filenames. This adapter keeps the Phase 2/3 flow stable while making
    WARA's `phase1_handoff.json` the source of truth.
    """

    handoff_file = _handoff_file(phase1_handoff)
    if handoff_file is None:
        raise ValueError(f"not a WARA phase1 handoff path: {phase1_handoff}")
    source_dir = handoff_file.parent
    payload = read_json(handoff_file) or {}
    if not isinstance(payload, dict) or not isinstance(payload.get("selected_candidate"), dict):
        raise ValueError(f"invalid WARA phase1 handoff file: {handoff_file}")

    evidence_pack = _read_optional_json(source_dir / "evidence_pack.json")
    candidates = _read_optional_json(source_dir / "candidates.json")
    candidate_review = _read_optional_json(source_dir / "candidate_review.json")
    source_context = _read_optional_json(source_dir / "source_context.json")

    selected = _dict(payload.get("selected_candidate"))
    problem_seed = _dict(payload.get("problem_contract_seed"))
    novelty_contract = _dict(payload.get("novelty_contract"))
    proof_contract = _dict(payload.get("proof_contract"))
    validation_contract = _dict(payload.get("validation_contract"))

    final_title = _first_text(
        selected.get("title"),
        _dict(candidate_review.get("selection_decision")).get("selected_title"),
        source_dir.name,
    )
    paper_title = _first_text(
        selected.get("paper_title"),
        _dict(candidate_review.get("selection_decision")).get("paper_title"),
    )
    objective = _first_text(selected.get("objective"), problem_seed.get("objective"))
    variables = _coerce_lines(
        problem_seed.get("variables")
        or problem_seed.get("controls")
        or selected.get("variables")
        or selected.get("controls")
    )
    constraints = _coerce_lines(problem_seed.get("constraints") or selected.get("core_constraints"))
    theorem_target = _coerce_lines(
        proof_contract.get("target_claims")
        or proof_contract.get("theorem_targets")
        or selected.get("theorem_or_algorithmic_claim")
    )
    validation_plan = _coerce_lines(
        validation_contract.get("figures")
        or validation_contract.get("validation_targets")
        or validation_contract.get("metrics")
        or validation_contract.get("expected_trends")
    )
    novelty_delta = _first_text(
        novelty_contract.get("claim_boundary"),
        selected.get("novelty_bet"),
        selected.get("novelty_delta"),
    )
    claimed_contribution = _first_text(
        selected.get("claimed_contribution"),
        selected.get("analytical_hook"),
        novelty_contract.get("claim_boundary"),
    )
    reformulation_path = _first_text(
        proof_contract.get("route"),
        proof_contract.get("algorithmic_route"),
        selected.get("tractability_path"),
        selected.get("convexification_path"),
        selected.get("analytical_hook"),
    )
    nonconvexity = _first_text(
        selected.get("source_of_difficulty"),
        selected.get("source_of_nonconvexity"),
        problem_seed.get("source_of_difficulty"),
        problem_seed.get("nonconvexity"),
        novelty_contract.get("main_risk"),
    )
    topic_taxonomy = (
        _dict(source_context.get("gap_matrix")).get("taxonomy")
        or _dict(evidence_pack.get("topic_taxonomy"))
        or {}
    )
    topic_score = _dict(evidence_pack.get("topic_score"))
    synthesis_md = _first_text(evidence_pack.get("synthesis_excerpt"), _dict(source_context.get("gap_matrix")).get("synthesis_excerpt"))
    references = _first_text(evidence_pack.get("references_excerpt"), evidence_pack.get("references_bib"))
    topic_focused_literature = _read_optional_json(source_dir / "topic_focused_literature.json")
    topic_focused_references = read_text(source_dir / "topic_focused_references.bib").strip()
    topic_focused_markdown = read_text(source_dir / "topic_focused_literature.md").strip()
    references_for_phase2 = topic_focused_references or references
    minimum_reference_target = int(os.environ.get("WARA_PHASE1_REFERENCE_MIN", "12") or 12)
    topic_focused_reference_count = max(
        len(topic_focused_literature.get("references", [])) if isinstance(topic_focused_literature.get("references"), list) else 0,
        _count_bibtex_entries(topic_focused_references),
    )
    if topic_focused_reference_count < minimum_reference_target:
        raise ValueError(
            "WARA Phase 1 handoff reference contract failed: "
            f"{topic_focused_reference_count} references < hard target {minimum_reference_target}. "
            "Phase 2/3 must not repair this by adding references; rerun Phase 1 LiteratureAgent."
        )

    handoff_dir = run_dir / "input_from_phase1"
    handoff_dir.mkdir(parents=True, exist_ok=True)
    final_topic_md = (
        "# Final Topic\n\n"
        f"## Recommended Title\n{final_title or 'TBD'}\n\n"
        f"## Paper-Facing Title\n{paper_title or final_title or 'TBD'}\n\n"
        f"## Source Handoff\n{handoff_file}\n\n"
        f"## WARA Native Handoff\ntrue\n"
    )
    problem_statement_md = (
        "# Problem Statement\n\n"
        f"## Wireless Scenario\n{_first_text(selected.get('wireless_scenario'), 'TBD')}\n\n"
        f"## Problem Statement\n{_first_text(selected.get('problem_statement'), 'TBD')}\n\n"
        f"## Objective\n{objective or 'TBD'}\n\n"
        f"## Declared Variables\n{variables or 'TBD'}\n\n"
        f"## Core Constraints\n{constraints or 'TBD'}\n"
    )
    algorithm_sketch_md = (
        "# Algorithm Sketch\n\n"
        f"## Reformulation Path\n{reformulation_path or 'TBD'}\n\n"
        f"## Proposed Route\n{claimed_contribution or 'TBD'}\n"
    )
    theorem_targets_md = (
        "# Theorem Targets\n\n"
        f"## Target\n{theorem_target or 'TBD'}\n\n"
        f"## Novelty Delta\n{novelty_delta or 'TBD'}\n"
    )
    validation_targets_md = f"# Validation Targets\n\n{validation_plan or 'TBD'}\n"
    hypotheses_md = (
        f"# Selected WARA Candidate\n\n"
        f"## Title\n{final_title}\n\n"
        f"## Problem statement\n{_first_text(selected.get('problem_statement'), '')}\n\n"
        f"## Objective\n{objective}\n\n"
        f"## Declared variables\n{variables}\n\n"
        f"## Core constraints\n{constraints}\n\n"
        f"## Theorem / proof target\n{theorem_target}\n\n"
        f"## Claimed contribution\n{claimed_contribution}\n\n"
        f"## Novelty delta vs prior art\n{novelty_delta}\n"
    )

    write_text(handoff_dir / "final_topic.md", final_topic_md)
    write_text(handoff_dir / "problem_statement.md", problem_statement_md)
    write_text(handoff_dir / "algorithm_sketch.md", algorithm_sketch_md)
    write_text(handoff_dir / "theorem_targets.md", theorem_targets_md)
    write_text(handoff_dir / "validation_targets.md", validation_targets_md)
    write_text(handoff_dir / "hypotheses.md", hypotheses_md)
    write_text(handoff_dir / "synthesis.md", synthesis_md)
    write_text(handoff_dir / "broad_scout_references.bib", references)
    write_text(handoff_dir / "references.bib", references_for_phase2)
    write_text(handoff_dir / "topic_focused_references.bib", topic_focused_references)
    write_text(handoff_dir / "topic_focused_literature.md", topic_focused_markdown)
    write_text(handoff_dir / "topic_focused_literature.json", json.dumps(topic_focused_literature, ensure_ascii=False, indent=2))
    write_text(handoff_dir / "topic_taxonomy.json", json.dumps(topic_taxonomy, ensure_ascii=False, indent=2))
    write_text(handoff_dir / "topic_score.json", json.dumps(topic_score, ensure_ascii=False, indent=2))
    write_text(handoff_dir / "review_report.json", json.dumps(candidate_review, ensure_ascii=False, indent=2))
    write_text(handoff_dir / "phase1_handoff.json", json.dumps(payload, ensure_ascii=False, indent=2))
    write_text(handoff_dir / "evidence_pack.json", json.dumps(evidence_pack, ensure_ascii=False, indent=2))
    write_text(handoff_dir / "candidates.json", json.dumps(candidates, ensure_ascii=False, indent=2))
    write_text(handoff_dir / "candidate_review.json", json.dumps(candidate_review, ensure_ascii=False, indent=2))

    source_phase1_run = _first_text(
        payload.get("source_phase1_run"),
        evidence_pack.get("source_phase1_run"),
        evidence_pack.get("source_run"),
        source_context.get("source_phase1_run"),
    )
    return {
        "source_kind": "wara_phase1_handoff",
        "phase1_handoff": str(handoff_file),
        "phase1_run": source_phase1_run,
        "final_title": final_title,
        "paper_title": paper_title,
        "problem_statement": _first_text(selected.get("problem_statement")),
        "wireless_scenario": _first_text(selected.get("wireless_scenario")),
        "objective": objective,
        "variables": variables,
        "core_constraints": constraints,
        "theorem_target": theorem_target,
        "reformulation_path": reformulation_path,
        "validation_plan": validation_plan,
        "claimed_contribution": claimed_contribution,
        "novelty_delta": novelty_delta,
        "nonconvexity": nonconvexity,
        "handoff_dir": str(handoff_dir),
        "topic_focused_reference_count": topic_focused_reference_count,
    }


def _handoff_file(path: Path) -> Path | None:
    candidate = path.expanduser()
    if candidate.is_dir():
        direct = candidate / "phase1_handoff.json"
        return direct if direct.exists() else None
    if candidate.name == "phase1_handoff.json" and candidate.exists():
        return candidate
    return None


def _read_optional_json(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    return payload if isinstance(payload, dict) else {}


def _count_bibtex_entries(text: str) -> int:
    return len(re.findall(r"@\w+\s*\{", str(text or "")))


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _first_text(*values: Any) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, list) and value:
            return _coerce_lines(value)
        if isinstance(value, dict) and value:
            return json.dumps(value, ensure_ascii=False)
    return ""


def _coerce_lines(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "\n".join(f"- {item}" for item in value if str(item).strip())
    if isinstance(value, dict):
        return "\n".join(f"- {key}: {val}" for key, val in value.items())
    return ""
