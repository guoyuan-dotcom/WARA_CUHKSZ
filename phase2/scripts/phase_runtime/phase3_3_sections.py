from __future__ import annotations

import importlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from pipeline_core import DEFAULT_MODEL_PROFILE, DOCS_DIR, compact_text, read_json, read_text, write_text
from pipeline_core.json_utils import _safe_json_loads
from phase_runtime.agent_context import build_role_agent_request_json
from phase_runtime.llm import create_llm_client
from phase_runtime.paper_mode import _paper_writing_mode_snapshot
from phase_runtime.prompt_templates import render_prompt_template
from phase_runtime.phase3_figure import (
    PHASE2_FIGURE_ENVIRONMENT,
    PHASE2_FIGURE_FLOAT_PLACEMENT,
    PHASE2_FIGURE_LATEX_WIDTH,
    find_phase3_figure_asset_for_phase,
)
from phase_runtime.phase3_4_references import resolve_paper_title


def _impl() -> Any:
    return importlib.import_module("phase_runtime_impl")


def build_pipeline_experiment_design_notes() -> str:
    return _impl().build_pipeline_experiment_design_notes()


def load_phase3_proposed_solution_snippet(run_dir: Path) -> str:
    return _impl().load_phase3_proposed_solution_snippet(run_dir)


def load_phase3_1_system_model_problem_snippet(run_dir: Path) -> str:
    return _impl().load_phase3_1_system_model_problem_snippet(run_dir)


def load_phase3_1_proposed_solution_snippet(run_dir: Path) -> str:
    return _impl().load_phase3_1_proposed_solution_snippet(run_dir)


def sanitize_phase3_2_numerical_results_tex(tex: str) -> str:
    return _impl().sanitize_phase3_2_numerical_results_tex(tex)


def sanitize_phase3_3_embedded_section_tex(tex: str) -> str:
    """Remove subsection labels that would duplicate wrapper section labels."""
    text = re.sub(r"(\\subsection\*?\{[^{}]+\})\s*\\label\{[^}]+\}", r"\1", str(tex or ""))
    text = re.sub(r"\\label\{eq:p0_([^{}]+)\}", r"\\label{eq:\1}", text)
    text = re.sub(r"\bPhase\s*~?\d(?:\.\d+)?\b", "the preceding construction", text, flags=re.I)
    text = re.sub(r"\bLLM\b", "automated", text)
    text = re.sub(r"\bpipeline\b", "workflow", text, flags=re.I)
    text = re.sub(r"\bgenerated_plugin\b", "experiment implementation", text)
    text = re.sub(r"\bdraft\b", "section", text, flags=re.I)
    text = re.sub(r"\bpreliminary\b", "current", text, flags=re.I)
    return text


def _phase3_3_split_independent_aligned_equations(tex: str) -> str:
    """Split ordinary equation/aligned blocks that define multiple quantities."""

    relation_re = re.compile(r"(?<![<>=])(?:\\triangleq|\\leq|\\geq|=|<|>)(?![<>=])")

    def replace_block(match: re.Match[str]) -> str:
        body = match.group("body").strip()
        if "\\label{" in body:
            return match.group(0)
        raw_lines = [line.strip() for line in re.split(r"\\\\", body) if line.strip()]
        relation_lines = [line for line in raw_lines if relation_re.search(line)]
        if len(relation_lines) <= 1:
            return match.group(0)
        pieces = []
        for line in raw_lines:
            cleaned = line.strip().rstrip(",").rstrip()
            if not cleaned:
                continue
            pieces.append("\\begin{equation}\n" + cleaned + "\n\\end{equation}")
        return "\n".join(pieces) if pieces else match.group(0)

    pattern = re.compile(
        r"\\begin\{equation\}\s*\\begin\{aligned\}(?P<body>.*?)\\end\{aligned\}\s*\\end\{equation\}",
        flags=re.DOTALL,
    )
    return pattern.sub(replace_block, tex)


def _phase3_3_assert_equation_display_format(phase_dir: Path, artifact_stem: str, tex: str) -> None:
    """Block technical-section assembly if upstream LaTeX still has dense equations."""
    report = _impl().analyze_latex_equation_line_format(tex)
    write_text(
        phase_dir / f"{artifact_stem}_equation_format_report.json",
        json.dumps(report, ensure_ascii=False, indent=2),
    )
    if not report.get("ok", False):
        summary = _impl().latex_equation_format_issue_summary(tex)
        write_text(phase_dir / f"{artifact_stem}_equation_format_blocking_errors.txt", summary)
        raise ValueError(f"Phase 3.3 blocked invalid equation display formatting in {artifact_stem}: {summary}")


def _phase3_3_clear_technical_gate_residue(phase_dir: Path) -> None:
    for path in phase_dir.glob("*_blocking_errors.txt"):
        path.unlink()


def _phase3_3_validate_technical_sections(
    phase3_3_dir: Path,
    *,
    system_model_problem_formulation_section: str,
    proposed_solution_section: str,
    numerical_results_section: str,
) -> dict[str, Any]:
    _phase3_3_clear_technical_gate_residue(phase3_3_dir)
    snippets = {
        "system_model_problem_formulation_section": system_model_problem_formulation_section,
        "proposed_solution_section": proposed_solution_section,
        "numerical_results_section": numerical_results_section,
    }
    errors: list[str] = []
    checks = [
        lambda: _phase3_3_assert_required_section_not_empty(
            phase3_3_dir,
            "system_model_problem_formulation_section",
            system_model_problem_formulation_section,
            min_words=120,
        ),
        lambda: _phase3_3_assert_required_section_not_empty(
            phase3_3_dir,
            "proposed_solution_section",
            proposed_solution_section,
            min_words=120,
        ),
        lambda: _phase3_3_assert_required_section_not_empty(
            phase3_3_dir,
            "numerical_results_section",
            numerical_results_section,
            min_words=120,
        ),
        lambda: _phase3_3_assert_equation_display_format(
            phase3_3_dir,
            "system_model_problem_formulation_section",
            system_model_problem_formulation_section,
        ),
        lambda: _phase3_3_assert_equation_display_format(
            phase3_3_dir,
            "proposed_solution_section",
            proposed_solution_section,
        ),
        lambda: _phase3_3_assert_equation_display_format(
            phase3_3_dir,
            "numerical_results_section",
            numerical_results_section,
        ),
        lambda: _phase3_3_assert_no_internal_paper_terms(
            phase3_3_dir,
            "system_model_problem_formulation_section",
            system_model_problem_formulation_section,
        ),
        lambda: _phase3_3_assert_no_internal_paper_terms(
            phase3_3_dir,
            "proposed_solution_section",
            proposed_solution_section,
        ),
        lambda: _phase3_3_assert_no_internal_paper_terms(
            phase3_3_dir,
            "numerical_results_section",
            numerical_results_section,
        ),
        lambda: _phase3_3_assert_unique_latex_labels(phase3_3_dir, snippets),
        lambda: _phase3_3_assert_technical_closure_quality(
            phase3_3_dir,
            {
                "system_model_problem_formulation_section": system_model_problem_formulation_section,
                "proposed_solution_section": proposed_solution_section,
            },
        ),
    ]
    for check in checks:
        try:
            check()
        except Exception as exc:  # noqa: BLE001 - collect all gate failures for one repair prompt
            errors.append(str(exc))
    report = {"ok": not errors, "errors": errors}
    write_text(phase3_3_dir / "phase3_3_technical_sections_gate_report.json", json.dumps(report, ensure_ascii=False, indent=2))
    return report


def _phase3_3_repair_technical_sections_llm(
    *,
    run_dir: Path,
    phase3_3_dir: Path,
    topic: str,
    model_profile: str,
    repair_round: int,
    gate_report: dict[str, Any],
    system_model_problem_formulation_section: str,
    proposed_solution_section: str,
    numerical_results_section: str,
) -> dict[str, str]:
    prompt = (
        "You are the WARA WritingAgent/RepairAgent for Phase 3.3 technical manuscript assembly.\n"
        "Repair ONLY LaTeX manuscript-structure issues reported by the gate. Preserve the scientific model, "
        "variables, objective, constraints, algorithm meaning, numerical values, figure references, and claims.\n\n"
        "Required repairs:\n"
        "- Split dense displayed equations so each displayed equation or align line has at most one primary equality/inequality.\n"
        "- Do not put independent SINR and rate definitions under one equation number.\n"
        "- If one sentence introduces two align lines, use IEEE-style mapping: put \\text{and}\\quad at the beginning of the second align line, end the second relation with a comma, and write 'respectively.' immediately after the display.\n"
        "- Formal optimization problems must use subequations with an inner numbered align environment.\n"
        "- Label the objective with an obj:p0_* label and every constraint line with con:p0_* labels.\n"
        "- Avoid eq:p0_* labels and do not reference optimization problems with eqref.\n"
        "- Keep section text self-contained and IEEE WCL compatible.\n"
        "- Do not add citations, new experiments, new baselines, new claims, or new notation.\n\n"
        "Return JSON only with exactly these string fields:\n"
        "system_model_problem_formulation_section_tex, proposed_solution_section_tex, numerical_results_section_tex.\n\n"
        f"Topic:\n{topic}\n\n"
        "Gate report:\n"
        f"{json.dumps(gate_report, ensure_ascii=False, indent=2)}\n\n"
        "Current system/model/problem section:\n"
        f"{system_model_problem_formulation_section}\n\n"
        "Current proposed-solution section:\n"
        f"{proposed_solution_section}\n\n"
        "Current numerical-results section:\n"
        f"{numerical_results_section}\n"
    )
    write_text(phase3_3_dir / f"phase3_3_technical_repair_prompt_round{repair_round}.txt", prompt)
    llm = create_llm_client(model_profile)
    thinking = {"type": "enabled"} if model_profile == "kimi-k2.6-thinking" else None
    response = llm.chat(
        [{"role": "user", "content": prompt}],
        json_mode=True,
        strip_thinking=True,
        thinking=thinking,
        max_tokens=int(os.environ.get("WARA_PHASE32_REPAIR_MAX_TOKENS", "12000") or 12000),
    )
    write_text(phase3_3_dir / f"phase3_3_technical_repair_raw_response_round{repair_round}.txt", response.content)
    payload = _safe_json_loads(response.content, {})
    if not isinstance(payload, dict):
        raise ValueError("Phase 3.3 technical-section repair LLM did not return a JSON object")
    repaired = {
        "system_model_problem_formulation_section": sanitize_phase3_3_embedded_section_tex(
            str(payload.get("system_model_problem_formulation_section_tex") or "").strip()
        ),
        "proposed_solution_section": sanitize_phase3_3_embedded_section_tex(
            str(payload.get("proposed_solution_section_tex") or "").strip()
        ),
        "numerical_results_section": sanitize_phase3_2_numerical_results_tex(
            str(payload.get("numerical_results_section_tex") or "").strip()
        ),
    }
    if not all(value.strip() for value in repaired.values()):
        raise ValueError("Phase 3.3 technical-section repair LLM returned an empty section")
    write_text(
        phase3_3_dir / f"phase3_3_technical_repair_sections_round{repair_round}.json",
        json.dumps(repaired, ensure_ascii=False, indent=2),
    )
    return repaired


def _phase3_3_assert_unique_latex_labels(phase_dir: Path, snippets: dict[str, str]) -> None:
    seen: dict[str, str] = {}
    duplicates: list[dict[str, str]] = []
    for artifact_stem, tex in snippets.items():
        for label in re.findall(r"\\label\{([^{}]+)\}", str(tex or "")):
            if label in seen:
                duplicates.append({"label": label, "first": seen[label], "duplicate": artifact_stem})
            else:
                seen[label] = artifact_stem
    report = {"ok": not duplicates, "duplicates": duplicates}
    write_text(phase_dir / "technical_label_uniqueness_report.json", json.dumps(report, ensure_ascii=False, indent=2))
    if duplicates:
        summary = "; ".join(
            f"{item['label']} first in {item['first']} and duplicated in {item['duplicate']}"
            for item in duplicates[:12]
        )
        write_text(phase_dir / "technical_label_uniqueness_blocking_errors.txt", summary)
        raise ValueError(f"Phase 3.3 blocked duplicate LaTeX labels: {summary}")


def _phase3_3_assert_required_section_not_empty(
    phase_dir: Path,
    artifact_stem: str,
    tex: str,
    *,
    min_words: int = 80,
) -> None:
    word_count = len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", str(tex or "")))
    report = {"ok": word_count >= min_words, "word_count": word_count, "minimum_word_count": min_words}
    write_text(phase_dir / f"{artifact_stem}_nonempty_report.json", json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ok"]:
        raise ValueError(
            f"Phase 3.3 blocked empty or incomplete {artifact_stem}: "
            f"{word_count} words < {min_words}. Run Phase 3.1 technical drafting before paper assembly."
        )


def _phase3_3_assert_no_internal_paper_terms(phase_dir: Path, artifact_stem: str, tex: str) -> None:
    stripped = re.sub(r"%.*", "", str(tex or ""))
    stripped = re.sub(
        r"\\includegraphics(?:\[[^\]]*\])?\{[^}]*\}",
        r"\\includegraphics{}",
        stripped,
    )
    patterns = [
        r"\bPhase\s*~?\d(?:\.\d+)?\b",
        r"\bpipeline\b",
        r"\bLLM\b",
        r"\bCodex\b",
        r"\bgenerated_plugin\b",
        r"\bdraft\b",
        r"\bpreliminary\b",
    ]
    hits: list[dict[str, str]] = []
    for pattern in patterns:
        for match in re.finditer(pattern, stripped, flags=re.IGNORECASE):
            line = stripped[: match.start()].count("\n") + 1
            hits.append({"line": str(line), "text": match.group(0)})
    report = {"ok": not hits, "hits": hits}
    write_text(phase_dir / f"{artifact_stem}_internal_terms_report.json", json.dumps(report, ensure_ascii=False, indent=2))
    if hits:
        summary = "; ".join(f"line {item['line']}: {item['text']}" for item in hits[:12])
        write_text(phase_dir / f"{artifact_stem}_internal_terms_blocking_errors.txt", summary)
        raise ValueError(f"Phase 3.3 blocked internal workflow terms in {artifact_stem}: {summary}")


def _phase3_3_has_concrete_uncertainty_model(tex: str) -> bool:
    return bool(
        re.search(
            r"ellipsoidal|norm[- ]bounded|box uncertainty|polyhedral|wasserstein|cantelli|chebyshev|bernstein|gaussian error|sub-gaussian|scenario|sample approximation|sample-average|moment[- ]based|mean and covariance|support set",
            str(tex or ""),
            flags=re.I,
        )
    )


def _phase3_3_assert_technical_closure_quality(phase_dir: Path, snippets: dict[str, str]) -> None:
    """Record advisory technical-closure signals without keyword-blocking the draft."""
    technical_text = "\n\n".join(str(value or "") for value in snippets.values())
    errors: list[str] = []
    warnings: list[str] = []
    has_phi = bool(re.search(r"\\Phi|Phi_m|\bphi_m\b|\\mathcal\s*C_\{?\\Phi\}?", technical_text))
    phi_placeholder_language = bool(
        re.search(
            r"selected ambiguity model|chosen ambiguity model|standard safe counterpart|safe conic counterpart|\\mathcal\s*C_\{?\\Phi\}?\s+denotes|when the selected\s+\\Phi|denoted by\s+\$?\\Phi",
            technical_text,
            flags=re.I,
        )
    )
    phi_defined = bool(
        re.search(r"\\Phi[^=\n]{0,120}(?:=|\\triangleq|:=)", technical_text)
        and _phase3_3_has_concrete_uncertainty_model(technical_text)
    )
    if has_phi and (phi_placeholder_language or not phi_defined):
        warnings.append(
            "Technical draft still contains a Phi-style safe-counterpart placeholder instead of a reproducible uncertainty model and deterministic expression."
        )
    if re.search(r"epsilon[^.\n]{0,90}(outage|chance)", technical_text, flags=re.I) and re.search(
        r"epsilon[^.\n]{0,90}(uncertainty|radius|csi|channel)",
        technical_text,
        flags=re.I,
    ):
        warnings.append("Technical draft may overload epsilon for both outage tolerance and uncertainty radius.")
    if re.search(r"rank recovery is unnecessary|no rank recovery|without rank recovery", technical_text, flags=re.I) and not re.search(
        r"gaussian signaling|multi-stream|multistream|covariance-domain|high-rank covariance|rank-one transmission|rank-one recovery",
        technical_text,
        flags=re.I,
    ):
        warnings.append("Technical draft may dismiss rank recovery without physical signaling or recovery scope.")
    report = {
        "ok": True,
        "errors": errors,
        "warnings": warnings,
        "advisory_only": True,
        "checks": {
            "has_phi_placeholder": has_phi,
            "phi_placeholder_language": phi_placeholder_language,
            "phi_has_reproducible_definition": phi_defined,
        },
    }
    write_text(phase_dir / "technical_closure_quality_report.json", json.dumps(report, ensure_ascii=False, indent=2))


def build_phase3_3_design_notes() -> str:
    return """
# Phase 3.3 Design Notes

## Phase 3.3 mission

Phase 3.3 is the technical-sections assembly phase.

It:
- reuses the Phase 2.1 system-model/problem-formulation section
- reuses the Phase 2.3 proposed-solution section
- reuses the Phase 3.2 numerical-results section
- generates a concise paper-facing abstract
- generates a concise paper-facing conclusion
- assembles these pieces into a single IEEE-style technical preview PDF

Phase 3.3 does not:
- write the full introduction
- produce the final reference list
- redesign algorithms, figures, or experiments
- change any numerical results

## Default assembled structure

The Phase 3.3 preview assembles:
- Abstract
- System Model and Problem Formulation
- Proposed Solution
- Numerical Results
- Conclusion

This phase is intended to produce a technical preview PDF, not the full final paper package.

Phase 3.3 extends this package with:
- the introduction
- curated references
- a bibliography-aware preview
""".strip()


def _phase3_3_load_method_naming(method_naming_summary_json: str, experiment_plan_json: str) -> dict[str, Any]:
    default_name_used = False
    methods: list[dict[str, str]] = []
    payload = _safe_json_loads(method_naming_summary_json, {})
    if isinstance(payload, dict) and isinstance(payload.get("methods"), list):
        for item in payload["methods"]:
            if not isinstance(item, dict):
                continue
            methods.append(
                {
                    "internal_name": str(item.get("internal_name") or item.get("name") or "").strip(),
                    "role": str(item.get("role") or "").strip(),
                    "display_name_short": str(item.get("display_name_short") or "").strip(),
                    "display_name_long": str(item.get("display_name_long") or "").strip(),
                }
            )
    elif isinstance(payload, dict) and isinstance(payload.get("methods"), dict):
        for internal_name, item in payload["methods"].items():
            if not isinstance(item, dict):
                continue
            methods.append(
                {
                    "internal_name": str(item.get("internal_name") or item.get("name") or internal_name).strip(),
                    "role": str(item.get("role") or "").strip(),
                    "display_name_short": str(item.get("display_name_short") or "").strip(),
                    "display_name_long": str(item.get("display_name_long") or "").strip(),
                }
            )
    if not methods:
        plan = _safe_json_loads(experiment_plan_json, {})
        compared = plan.get("compared_methods", []) if isinstance(plan, dict) else []
        for item in compared:
            if not isinstance(item, dict):
                continue
            methods.append(
                {
                    "internal_name": str(item.get("internal_name") or item.get("name") or "").strip(),
                    "role": str(item.get("role") or "").strip(),
                    "display_name_short": str(item.get("display_name_short") or "").strip(),
                    "display_name_long": str(item.get("display_name_long") or "").strip(),
                }
            )
    cleaned: list[dict[str, str]] = []
    for item in methods:
        short_name = item.get("display_name_short", "").strip()
        long_name = item.get("display_name_long", "").strip()
        role = item.get("role", "").strip()
        internal = item.get("internal_name", "").strip()
        if not short_name:
            default_name_used = True
            short_name = "Proposed method" if role == "proposed" else "Main benchmark" if "baseline" in role else internal or "Method"
        if not long_name:
            default_name_used = True
            long_name = short_name
        cleaned.append(
            {
                "internal_name": internal or role or "method",
                "role": role or "method",
                "display_name_short": short_name,
                "display_name_long": long_name,
            }
        )
    return {"methods": cleaned, "default_name_used": default_name_used}


def _phase3_3_select_methods(methods_payload: dict[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    methods = methods_payload.get("methods", []) if isinstance(methods_payload, dict) else []
    proposed = next((m for m in methods if str(m.get("role")) == "proposed"), None)
    baseline = next((m for m in methods if "baseline" in str(m.get("role")) or "benchmark" in str(m.get("role"))), None)
    if proposed is None:
        proposed = {
            "internal_name": "proposed",
            "role": "proposed",
            "display_name_short": "Proposed method",
            "display_name_long": "Proposed method",
        }
    if baseline is None:
        baseline = {
            "internal_name": "baseline",
            "role": "main_benchmark",
            "display_name_short": "Main benchmark",
            "display_name_long": "Main benchmark",
        }
    return proposed, baseline


def _phase3_3_clean_latex_prose(tex: str) -> str:
    """Extract prose from a LaTeX snippet for downstream writing prompts."""
    cleaned = str(tex or "")
    cleaned = re.sub(r"\\begin\{figure\}.*?\\end\{figure\}", " ", cleaned, flags=re.S)
    cleaned = re.sub(r"\\begin\{table\*?\}.*?\\end\{table\*?\}", " ", cleaned, flags=re.S)
    cleaned = re.sub(r"\\begin\{tabular\}.*?\\end\{tabular\}", " ", cleaned, flags=re.S)
    cleaned = re.sub(r"\\includegraphics(?:\[[^\]]*\])?\{[^{}]*\}", " ", cleaned)
    cleaned = re.sub(r"\\label\{[^{}]*\}", " ", cleaned)
    cleaned = re.sub(r"\\caption\{([^{}]*)\}", r" \1 ", cleaned)
    cleaned = re.sub(r"\\(?:section|subsection|subsubsection)\*?\{([^{}]*)\}", r"\1. ", cleaned)
    cleaned = re.sub(r"\\(?:begin|end)\{(?:itemize|enumerate|abstract|IEEEkeywords)\}", " ", cleaned)
    cleaned = re.sub(r"\\item(?:\[[^\]]*\])?", " ", cleaned)
    cleaned = re.sub(r"\\textbf\{([^{}]*)\}", r"\1", cleaned)
    cleaned = re.sub(r"\\emph\{([^{}]*)\}", r"\1", cleaned)
    cleaned = re.sub(r"\\ref\{([^{}]*)\}", r"\1", cleaned)
    cleaned = re.sub(r"\\[A-Za-z]+", " ", cleaned)
    cleaned = cleaned.replace("~", " ")
    return re.sub(r"\s+", " ", cleaned).strip()


def _phase3_3_method_label_map(method_naming_payload: dict[str, Any]) -> dict[str, str]:
    labels: dict[str, str] = {}
    methods = method_naming_payload.get("methods", []) if isinstance(method_naming_payload, dict) else []
    for item in methods:
        if not isinstance(item, dict):
            continue
        internal = str(item.get("internal_name") or item.get("name") or "").strip()
        label = str(item.get("display_name_short") or item.get("display_name_long") or internal).strip()
        if internal and label:
            labels[internal] = label
    return labels


def _phase3_3_build_figure_evidence(
    *,
    phase25_summary: dict[str, Any],
    experiment_plan: dict[str, Any],
    verified_registry: dict[str, Any],
    method_naming_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    labels = _phase3_3_method_label_map(method_naming_payload)
    plan_figures = {
        str(item.get("figure_id") or "").strip(): item
        for item in (experiment_plan.get("figure_specs", []) if isinstance(experiment_plan, dict) else [])
        if isinstance(item, dict) and str(item.get("figure_id") or "").strip()
    }
    registry_figures = {
        str(item.get("figure_id") or "").strip(): item
        for item in (verified_registry.get("figures", []) if isinstance(verified_registry, dict) else [])
        if isinstance(item, dict) and str(item.get("figure_id") or "").strip()
    }
    summary_figures = [
        item
        for item in (phase25_summary.get("figures", []) if isinstance(phase25_summary, dict) else [])
        if isinstance(item, dict)
    ]
    if not summary_figures:
        summary_figures = list(registry_figures.values()) or list(plan_figures.values())

    evidence: list[dict[str, Any]] = []
    for item in summary_figures[:4]:
        fig_id = str(item.get("figure_id") or "").strip()
        plan = plan_figures.get(fig_id, {})
        registry = registry_figures.get(fig_id, {})
        metric = item.get("y_axis_label") or item.get("y_metric") or (plan.get("metric") or {}).get("display_name") or (plan.get("metric") or {}).get("name")
        x_axis = item.get("x_axis_label") or item.get("x_axis_param") or ((plan.get("encoding") or {}).get("x") or {}).get("display_name") or ((plan.get("encoding") or {}).get("x") or {}).get("sweep_param")
        methods = item.get("methods") or plan.get("methods") or registry.get("methods") or []
        method_labels = [labels.get(str(method), str(method)) for method in methods if str(method)]
        evidence.append(
            {
                "figure_id": fig_id or "figure",
                "chart_type": item.get("chart_type") or plan.get("chart_type") or registry.get("chart_type") or "not specified",
                "metric": str(metric or "not specified"),
                "x_axis": str(x_axis or "not specified"),
                "purpose": str(item.get("purpose") or plan.get("purpose") or registry.get("purpose") or item.get("figure_intent") or plan.get("chart_intent") or "not specified"),
                "methods": method_labels or ["not specified"],
                "paper_ready": bool(item.get("paper_ready") or registry.get("paper_ready")),
                "evidence_scope": "final" if bool(item.get("paper_ready") or registry.get("paper_ready")) else "limited_nonfinal",
                "source_files": [
                    f"phase2-5/figures/{item.get('filename_pdf') or registry.get('filename_pdf') or fig_id + '.pdf'}",
                    "phase2-5/phase25_experiment_summary.json",
                    "phase3-2/numerical_results_section.tex",
                ],
            }
        )
    return evidence


def _phase3_3_compact_verified_registry(verified_registry: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(verified_registry, dict):
        return {}
    raw_status = str(verified_registry.get("phase25_status") or verified_registry.get("status") or "")
    return {
        "status": verified_registry.get("status", ""),
        "evidence_scope": "final" if raw_status in {"paper_ready", "paper_preferred_ready", "high_confidence_ready"} else "limited_nonfinal",
        "summary_numbers": verified_registry.get("summary_numbers", {}),
        "primary_metric": verified_registry.get("primary_metric", {}),
        "paper_claims_to_test": verified_registry.get("paper_claims_to_test", []),
        "methods": verified_registry.get("methods", {}),
        "figures": [
            {
                key: figure.get(key)
                for key in [
                    "figure_id",
                    "chart_type",
                    "x_axis_label",
                    "x_axis_param",
                    "y_axis_label",
                    "y_metric",
                    "methods",
                    "paper_ready",
                    "draft_or_final",
                    "num_x_points",
                    "required_sweep",
                ]
                if key in figure
            }
            for figure in verified_registry.get("figures", [])
            if isinstance(figure, dict)
        ],
    }


def build_phase3_3_paper_facts(
    *,
    topic: str,
    system_model_md: str,
    problem_formulation_md: str,
    reformulation_path_md: str,
    algorithm_md: str,
    benchmark_definition_md: str,
    convergence_or_complexity_md: str,
    phase25_summary: dict[str, Any],
    experiment_plan: dict[str, Any],
    method_naming_payload: dict[str, Any],
    phase3_2_manifest: dict[str, Any],
    phase3_2_numerical_results_tex: str,
    verified_registry: dict[str, Any],
) -> dict[str, Any]:
    proposed, baseline = _phase3_3_select_methods(method_naming_payload)
    primary_metric = (phase25_summary.get("primary_metric") or experiment_plan.get("primary_metric") or {}) if isinstance(phase25_summary, dict) else {}
    numbers_used = phase3_2_manifest.get("numbers_used", []) if isinstance(phase3_2_manifest, dict) else []
    paper_claims = experiment_plan.get("paper_claims_to_test", []) if isinstance(experiment_plan, dict) else []
    limitations = phase25_summary.get("limitations", []) if isinstance(phase25_summary, dict) else []
    final_evidence_ready = bool(phase25_summary.get("paper_minimum_ready")) if isinstance(phase25_summary, dict) else False
    figure_evidence = _phase3_3_build_figure_evidence(
        phase25_summary=phase25_summary if isinstance(phase25_summary, dict) else {},
        experiment_plan=experiment_plan if isinstance(experiment_plan, dict) else {},
        verified_registry=verified_registry if isinstance(verified_registry, dict) else {},
        method_naming_payload=method_naming_payload if isinstance(method_naming_payload, dict) else {},
    )
    comparison_summary = {
        "proposed_win_rate": (phase25_summary.get("overall") or {}).get("proposed_win_rate", "not specified"),
        "proposed_mean_relative_gain": (phase25_summary.get("overall") or {}).get("proposed_mean_relative_gain", "not specified"),
        "proposed_median_relative_gain": (phase25_summary.get("overall") or {}).get("proposed_median_relative_gain", "not specified"),
        "final_evidence_ready": final_evidence_ready,
        "evidence_scope": "final" if final_evidence_ready else "limited_nonfinal",
        "numeric_summary_policy": (
            "These aggregate numbers are development diagnostics only because the supplied numerical evidence is not final; do not use them in final abstract/conclusion prose."
            if not final_evidence_ready
            else "These aggregate numbers may be used only if they appear in phase3_2_numbers_available or the verified registry."
        ),
    }
    phase3_2_prose = compact_text(_phase3_3_clean_latex_prose(phase3_2_numerical_results_tex), 1800)

    return {
        "topic": topic or "not specified",
        "target_venue": "IEEE WCL",
        "problem_context": compact_text(system_model_md, 900) or "not specified",
        "practical_motivation": compact_text(problem_formulation_md, 900) or "not specified",
        "limitation_of_existing_methods": compact_text(benchmark_definition_md, 700) or "not specified",
        "proposed_method": {
            "display_name_long": proposed["display_name_long"] or "not specified",
            "display_name_short": proposed["display_name_short"] or "not specified",
            "core_mechanism": compact_text(algorithm_md, 900) or "not specified",
            "optimization_variables_or_design_degrees": compact_text(problem_formulation_md, 700) or "not specified",
            "key_reformulation_or_algorithmic_tools": compact_text(reformulation_path_md or algorithm_md, 800) or "not specified",
        },
        "main_benchmark": {
            "display_name_long": baseline["display_name_long"] or "not specified",
            "display_name_short": baseline["display_name_short"] or "not specified",
            "limitation_relative_to_proposed": compact_text(benchmark_definition_md, 700) or "not specified",
        },
        "theoretical_contribution": compact_text(reformulation_path_md, 900) or "not specified",
        "algorithmic_contribution": compact_text(algorithm_md, 900) or "not specified",
        "convergence_or_complexity": compact_text(convergence_or_complexity_md, 700) or "not specified",
        "main_metric": {
            "name": str(primary_metric.get("name", "not specified")),
            "display_name": str(primary_metric.get("display_name", "not specified")),
            "higher_is_better": primary_metric.get("higher_is_better", "not specified"),
        },
        "numerical_evidence": {
            "evidence_mode": "figure_and_phase3_2_text",
            "table_evidence": "not used; the supplied numerical evidence is a figure-driven, table-free numerical-results section",
            "figures": figure_evidence or [{"figure_id": "not specified", "source_files": ["not specified"]}],
            "phase3_2_numerical_results_summary": phase3_2_prose or "not specified",
            "comparison_summary": comparison_summary,
        },
        "paper_claims_to_test": paper_claims if isinstance(paper_claims, list) and paper_claims else ["not specified"],
        "allowed_claims": [
            "improves the reported metric under the considered settings when supported by the figure evidence",
            "supports the benefit of the proposed design",
            "maintains feasibility if the evidence supports it",
            "identifies the operating regimes where the proposed design is most beneficial",
        ],
        "forbidden_claims": [
            "proves",
            "guarantees",
            "globally optimal",
            "universally optimal",
            "always superior",
            "statistically significant",
        ],
        "limitations": limitations if isinstance(limitations, list) and limitations else ["not specified"],
        "method_name_default_used": bool(method_naming_payload.get("default_name_used")),
        "phase3_2_numbers_available": numbers_used,
        "verified_evidence_registry": _phase3_3_compact_verified_registry(verified_registry),
    }


def build_phase3_3_abstract_conclusion_prompt(
    *,
    topic: str,
    paper_facts_json: str,
    writing_agent_request_json: str = "",
) -> str:
    return render_prompt_template(
        "phase3_3/abstract_conclusion.prompt.yaml",
        paper_facts_json=paper_facts_json,
        writing_agent_request_json=compact_text(writing_agent_request_json, 7000),
    )


def _phase3_3_extract_text_body(text: str, begin_env: str, end_env: str) -> str:
    pattern = re.compile(re.escape(begin_env) + r"(.*?)" + re.escape(end_env), flags=re.S | re.I)
    match = pattern.search(text)
    return match.group(1).strip() if match else text.strip()


def _phase3_3_defined_abstract_acronyms(text: str) -> set[str]:
    defined: set[str] = set()
    for match in re.finditer(r"\(([^()]+)\)", text):
        content = match.group(1).strip()
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]{1,}", content):
            defined.add(content)
        for token in re.findall(r"\b(?:[A-Z0-9]+(?:-[A-Z0-9]+)+|[A-Z][A-Za-z0-9]*-[A-Z0-9][A-Za-z0-9-]*|[A-Z]{2,}[A-Z0-9]*)\b", content):
            defined.add(token)
    for pattern in [
        r"\btermed\s+([A-Za-z0-9]+(?:-[A-Za-z0-9]+)+)\b",
        r"\bdenoted\s+as\s+([A-Za-z0-9]+(?:-[A-Za-z0-9]+)+)\b",
        r"\breferred\s+to\s+as\s+([A-Za-z0-9]+(?:-[A-Za-z0-9]+)+)\b",
        r"\bcalled\s+([A-Za-z0-9]+(?:-[A-Za-z0-9]+)+)\b",
    ]:
        for match in re.finditer(pattern, text, flags=re.I):
            defined.add(match.group(1))
    return defined


def find_undefined_abstract_abbreviations(text: str, paper_facts: dict[str, Any] | None = None) -> list[str]:
    body = _phase3_3_extract_text_body(text, r"\begin{abstract}", r"\end{abstract}")
    candidates = re.findall(
        r"\b(?:[A-Z0-9]+(?:-[A-Z0-9]+)+|[A-Z]{2,}[A-Z0-9]*|[0-9]+[A-Z]|[A-Z][a-z]+[A-Z][A-Za-z0-9]*)\b",
        body,
    )
    defined = _phase3_3_defined_abstract_acronyms(body)
    allowed = {
        "IEEE",
        "WCL",
        "LaTeX",
    }
    for term in list(defined):
        if "-" in term:
            allowed.update(part for part in term.split("-") if part)
    if isinstance(paper_facts, dict):
        for method_key in ["proposed_method", "main_benchmark"]:
            method = paper_facts.get(method_key) or {}
            for field in ["display_name_short", "display_name_long"]:
                value = str(method.get(field, "")).strip()
                if not value:
                    continue
                for token in re.findall(r"\b[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*\b", value):
                    if token in defined:
                        allowed.add(token)
    unresolved: list[str] = []
    seen: set[str] = set()
    for token in candidates:
        if len(token) <= 1:
            continue
        if token in allowed or token in defined:
            continue
        if token.startswith("Fig") or token.startswith("Table"):
            continue
        if token not in seen:
            unresolved.append(token)
            seen.add(token)
    return unresolved


def _find_phase3_3_public_summary_notation_issues(text: str, section_name: str) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    for match in re.finditer(r"\\\((.*?)\\\)|\$([^$]+)\$", text, flags=re.S):
        token = (match.group(1) if match.group(1) is not None else match.group(2) or "").strip()
        if not token:
            continue
        if re.search(r"\\mathbf|\\mathcal|\\mathrm|\\rm|[_^]|\\sum|\\operatorname|\\mathrm", token):
            issues.append(
                {
                    "token": token,
                    "message": f"{section_name} uses mathematical notation; describe the object in words instead.",
                }
            )
    return issues


def find_phase3_3_abstract_notation_issues(text: str) -> list[dict[str, str]]:
    """Find paper-unfriendly optimizer notation in the abstract body."""
    body = _phase3_3_extract_text_body(text, r"\begin{abstract}", r"\end{abstract}") or text
    return _find_phase3_3_public_summary_notation_issues(body, "Abstract")


def find_phase3_3_conclusion_notation_issues(text: str) -> list[dict[str, str]]:
    """Find paper-unfriendly optimizer notation in the conclusion body."""
    body = re.sub(r"\\section\*?\{Conclusion\}", " ", text, flags=re.I)
    body = re.sub(r"\\label\{sec:conclusion\}", " ", body, flags=re.I)
    return _find_phase3_3_public_summary_notation_issues(body, "Conclusion")


def _soften_phase3_3_public_summary_notation_body(body: str) -> str:
    body = re.sub(
        r"access points\s*\(APs?\)",
        "access points",
        body,
        flags=re.I,
    )
    body = re.sub(
        r"per-access-point\s*\(AP\)",
        "per-access-point",
        body,
        flags=re.I,
    )
    body = re.sub(r"\bper-AP\b", "per-access-point", body)
    body = re.sub(r"\bAPs\b", "access points", body)
    body = re.sub(r"\bAP\b", "access point", body)
    body = re.sub(
        r"signal-to-interference-plus-noise-ratio\s*\(SINR\)",
        "signal-to-interference-plus-noise-ratio",
        body,
        flags=re.I,
    )
    body = re.sub(
        r"\bSINR\b",
        "signal-to-interference-plus-noise-ratio",
        body,
    )
    body = re.sub(
        r"semidefinite programming\s*\(SDP\)",
        "semidefinite programming",
        body,
        flags=re.I,
    )
    body = re.sub(
        r"\bSDP\b",
        "semidefinite programming",
        body,
    )
    body = re.sub(
        r"positive semidefinite\s*\(PSD\)",
        "positive semidefinite",
        body,
        flags=re.I,
    )
    body = re.sub(
        r"\bPSD\b",
        "positive semidefinite",
        body,
    )
    body = re.sub(
        r"energy harvesting\s*\(EH\)",
        "energy harvesting",
        body,
        flags=re.I,
    )
    body = re.sub(r"\bEH\b", "energy harvesting", body)
    body = re.sub(
        r"maximum-ratio-transmission\s*\(MRT\)",
        "maximum-ratio-transmission",
        body,
        flags=re.I,
    )
    body = re.sub(r"\bMRT\b", "maximum-ratio-transmission", body)
    body = re.sub(
        r"(?:the\s+)?(?:user|communication)\s+covariances\s*\\\(\s*\\\{\\mathbf\{W\}_k\\\}\s*\\\)\s+and\s+the\s+shared\s+covariance\s*\\\(\s*\\mathbf\{V\}\s*\\\)",
        "the optimized transmit-design variables",
        body,
        flags=re.I,
    )
    body = re.sub(
        r"(?:the\s+)?(?:communication|user)\s+covariance\s+matrices\s*\\\(\s*\\\{\\mathbf\{W\}_k\\\}\s*\\\)\s+and\s+the\s+shared\s+covariance\s+matrix\s*\\\(\s*\\mathbf\{V\}\s*\\\)",
        "the optimized transmit-design variables",
        body,
        flags=re.I,
    )
    replacements = [
        (r"\\\(\s*\\\{\\mathbf\{W\}_k\\\}\s*\\\)", "the optimized transmit-design variables"),
        (r"\\\(\s*\\mathbf\{W\}_k\s*\\\)", "the optimized transmit-design variable"),
        (r"\\\(\s*\\mathbf\{V\}\s*\\\)", "the auxiliary transmit-design variable"),
        (r"\\\(\s*P_\{\\rm\s+tx\}\s*\\\)", "transmit power"),
        (r"\\\(\s*P_\{\\mathrm\{tot\}\}\s*\\\)", "total transmit power"),
        (r"\\\(\s*\\gamma_k\s*\\\)", "the service threshold"),
        (r"\\\(\s*E_\{\\min\}\s*\\\)", "the energy requirement"),
        (r"\\\(\s*S_\{\\min\}\s*\\\)", "the service requirement"),
    ]
    for pattern, replacement in replacements:
        body = re.sub(pattern, replacement, body)
    body = re.sub(
        r"\b(?:the\s+)?(?:user|communication)\s+covariances\s+the optimized transmit-design variables",
        "the optimized transmit-design variables",
        body,
        flags=re.I,
    )
    body = re.sub(
        r"\b(?:the\s+)?shared\s+covariance\s+the auxiliary transmit-design variable",
        "the auxiliary transmit-design variable",
        body,
        flags=re.I,
    )
    body = re.sub(
        r"\b(?:required|optimized|total)\s+transmit\s+power\s+transmit\s+power\b",
        lambda m: m.group(0).replace(" transmit power transmit power", " transmit power"),
        body,
        flags=re.I,
    )
    body = re.sub(r"\s{2,}", " ", body)
    return body


def soften_phase3_3_abstract_notation(text: str) -> str:
    """Rewrite detailed optimizer symbols in the abstract into public-facing prose."""
    cleaned = text

    def _rewrite_body(match: re.Match[str]) -> str:
        body = _soften_phase3_3_public_summary_notation_body(match.group(1))
        return "\\begin{abstract}\n" + body.strip() + "\n\\end{abstract}"

    cleaned = re.sub(
        r"\\begin\{abstract\}\s*(.*?)\s*\\end\{abstract\}",
        _rewrite_body,
        cleaned,
        count=1,
        flags=re.I | re.S,
    )
    return cleaned


def soften_phase3_3_conclusion_notation(text: str) -> str:
    """Rewrite detailed optimizer symbols in the conclusion into public-facing prose."""
    match = re.match(
        r"(?P<header>\s*\\section\*?\{Conclusion\}\s*(?:\\label\{sec:conclusion\}\s*)?)(?P<body>.*)",
        text,
        flags=re.I | re.S,
    )
    if not match:
        return _soften_phase3_3_public_summary_notation_body(text)
    header = match.group("header").rstrip()
    body = _soften_phase3_3_public_summary_notation_body(match.group("body")).strip()
    return header + "\n" + body


def sanitize_phase3_3_abstract_tex(text: str) -> str:
    cleaned = text.replace("\r\n", "\n").strip()
    cleaned = re.sub(r"^\s*Abstract\s*:?\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"^\s*\\section\*?\{Conclusion\}\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\\end\{document\}\s*$", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\n{2,}", "\n", cleaned)
    if r"\begin{abstract}" not in cleaned:
        cleaned = "\\begin{abstract}\n" + cleaned
    if r"\end{abstract}" not in cleaned:
        cleaned = cleaned.rstrip() + "\n\\end{abstract}"
    cleaned = soften_phase3_3_abstract_notation(cleaned)
    return cleaned.strip() + "\n"


def sanitize_phase3_3_keywords_tex(text: str, paper_facts: dict[str, Any] | None = None) -> str:
    cleaned = text.replace("\r\n", "\n").strip()
    cleaned = re.sub(r"^\s*Keywords\s*:?\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"^\s*Index Terms\s*[-:]\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\\begin\{abstract\}.*?\\end\{abstract\}", "", cleaned, flags=re.I | re.S)
    cleaned = re.sub(r"\\section\*?\{Conclusion\}.*$", "", cleaned, flags=re.I | re.S)
    if r"\begin{IEEEkeywords}" not in cleaned:
        fallback_terms: list[str] = []
        if isinstance(paper_facts, dict):
            for key in ["topic", "main_metric_name", "proposed_display_name_short"]:
                value = str(paper_facts.get(key, "")).strip()
                if value and value.lower() != "not specified" and value not in fallback_terms:
                    fallback_terms.append(value)
            tools = paper_facts.get("key_tools", [])
            if isinstance(tools, list):
                for item in tools:
                    value = str(item).strip()
                    if value and value.lower() != "not specified" and value not in fallback_terms:
                        fallback_terms.append(value)
                    if len(fallback_terms) >= 5:
                        break
        if not cleaned:
            cleaned = ", ".join(fallback_terms[:5]) if fallback_terms else "wireless communications, optimization"
        cleaned = "\\begin{IEEEkeywords}\n" + cleaned
    if r"\end{IEEEkeywords}" not in cleaned:
        cleaned = cleaned.rstrip() + "\n\\end{IEEEkeywords}"
    return cleaned.strip() + "\n"


def sanitize_phase3_3_conclusion_tex(text: str) -> str:
    cleaned = text.replace("\r\n", "\n").strip()
    for _ in range(3):
        cleaned = re.sub(r"\\n(?=(?:\\n|\s|[}\]]|$))", "\n", cleaned)
        cleaned = re.sub(r"(?<=[.!?;:,}\n])\\n", "\n", cleaned)
    cleaned = re.sub(r"^\s*Conclusion\s*:?\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\\begin\{conclusion\}\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*\\end\{conclusion\}", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\\begin\{section\}\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*\\end\{section\}", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\\begin\{abstract\}\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*\\end\{abstract\}", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\\begin\{IEEEkeywords\}.*?\\end\{IEEEkeywords\}", "", cleaned, flags=re.I | re.S)
    cleaned = re.sub(r"\\end\{IEEEkeywords\}", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\\end\{document\}\s*$", "", cleaned, flags=re.I)
    while cleaned.rstrip().endswith("}") and cleaned.count("}") > cleaned.count("{"):
        cleaned = cleaned.rstrip()[:-1].rstrip()
    cleaned = re.sub(r"\n{2,}", "\n", cleaned)
    if not re.search(r"^\s*\\section\*?\{Conclusion\}", cleaned, flags=re.I):
        cleaned = "\\section{Conclusion}\n" + cleaned
    cleaned = re.sub(
        r"^\s*\\section\{Conclusion\}(?!\s*\\label\{sec:conclusion\})",
        r"\\section{Conclusion}\n\\label{sec:conclusion}",
        cleaned,
        count=1,
        flags=re.I,
    )
    cleaned = soften_phase3_3_conclusion_notation(cleaned)
    return cleaned.strip() + "\n"


def _rewrite_phase25_figure_paths_for_preview(text: str, run_dir: Path) -> str:
    figures_dir = run_dir / "phase2-5" / "figures"
    summary = read_json(run_dir / "phase2-5" / "phase25_experiment_summary.json") or {}
    figure_items = summary.get("figures", []) if isinstance(summary, dict) else []
    figure_aliases: dict[str, str] = {}
    if isinstance(figure_items, list):
        for idx, item in enumerate([entry for entry in figure_items if isinstance(entry, dict)], start=1):
            filename_pdf = str(item.get("filename_pdf") or "").strip()
            filename_png = str(item.get("filename_png") or "").strip()
            chosen = filename_pdf if filename_pdf and (figures_dir / filename_pdf).exists() else filename_png
            if not chosen or not (figures_dir / chosen).exists():
                continue
            figure_id = str(item.get("figure_id") or "").strip()
            for alias in [f"figure_{idx}.pdf", f"figure_{idx}.png", f"{figure_id}.pdf", f"{figure_id}.png"]:
                if alias and alias != ".pdf" and alias != ".png":
                    figure_aliases[alias.lower()] = chosen
    for idx, path in enumerate(sorted(figures_dir.glob("*.pdf")), start=1):
        figure_aliases.setdefault(f"figure_{idx}.pdf", path.name)
    for idx, path in enumerate(sorted(figures_dir.glob("*.png")), start=1):
        figure_aliases.setdefault(f"figure_{idx}.png", path.name)

    def _replace(match: re.Match[str]) -> str:
        raw_path = match.group(0).replace("\\", "/")
        filename = match.group(1)
        requested = figures_dir / filename
        if requested.exists():
            return raw_path

        requested_path = Path(filename)
        alias = figure_aliases.get(filename.lower())
        if alias and (figures_dir / alias).exists():
            return f"../phase2-5/figures/{alias}"
        stem = requested_path.stem
        suffix = requested_path.suffix or ".pdf"
        candidates = [
            f"{stem}_draft{suffix}",
            f"{stem}_draft.png",
            f"{stem}.png",
        ]
        for candidate in candidates:
            if (figures_dir / candidate).exists():
                return f"../phase2-5/figures/{candidate}"
        return raw_path

    return re.sub(r"\.\./phase2-5/figures/([A-Za-z0-9_.-]+)", _replace, text)


def sanitize_latex_alignment_label_breaks(text: str) -> str:
    cleaned = str(text or "")
    cleaned = re.sub(
        r"(\\label\{[^{}]+\})\\(?=\s*\n\s*\\text\{s\.t\.\})",
        r"\1\\\\",
        cleaned,
    )
    cleaned = re.sub(
        r"(\\label\{[^{}]+\})\\(?=\s*\n\s*&)",
        r"\1\\\\",
        cleaned,
    )
    return cleaned


def _phase3_figure_latex_settings(asset: dict[str, Any] | None = None) -> tuple[str, str, str]:
    asset = asset or {}
    environment = str(asset.get("figure_environment") or PHASE2_FIGURE_ENVIRONMENT).strip()
    if environment not in {"figure", "figure*"}:
        environment = PHASE2_FIGURE_ENVIRONMENT
    placement = str(asset.get("float_placement") or PHASE2_FIGURE_FLOAT_PLACEMENT).strip()
    if not re.fullmatch(r"[!htbpH]+", placement):
        placement = PHASE2_FIGURE_FLOAT_PLACEMENT
    width = str(asset.get("width") or PHASE2_FIGURE_LATEX_WIDTH).strip()
    if width != PHASE2_FIGURE_LATEX_WIDTH:
        width = PHASE2_FIGURE_LATEX_WIDTH
    return environment, placement, width


def _prepare_conceptual_diagram_input(phase_dir: Path, build_dir: Path) -> None:
    manifest_asset = find_phase3_figure_asset_for_phase(phase_dir)
    if manifest_asset is not None:
        source = Path(manifest_asset["source_path"])
        figure_dir = build_dir / "figures"
        figure_dir.mkdir(parents=True, exist_ok=True)
        target = figure_dir / source.name
        shutil.copyfile(source, target)
        caption = str(manifest_asset.get("caption") or "").strip()
        if not caption:
            caption = "Conceptual diagram of the considered system and proposed processing flow."
        label = str(manifest_asset.get("label") or "").strip() or "fig:conceptual_diagram"
        environment, placement, width = _phase3_figure_latex_settings(manifest_asset)
        figure_tex = rf"""
\begin{{{environment}}}[{placement}]
\centering
\includegraphics[width={width}]{{figures/{target.name}}}
\caption{{{caption}}}
\label{{{label}}}
\end{{{environment}}}
""".strip()
        write_text(build_dir / "conceptual_diagram.tex", figure_tex + "\n")
        return

    candidates = [
        phase_dir / "figures" / "conceptual_diagram.png",
        phase_dir / "figures" / "conceptual_diagram.pdf",
        phase_dir / "conceptual_diagram.png",
        phase_dir / "conceptual_diagram.pdf",
    ]
    source = next((path for path in candidates if path.exists()), None)
    if source is None:
        write_text(build_dir / "conceptual_diagram.tex", "")
        return

    figure_dir = build_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    target = figure_dir / source.name
    shutil.copyfile(source, target)

    caption = read_text(phase_dir / "conceptual_diagram_caption.txt").strip()
    if not caption:
        caption = "Conceptual diagram of the considered system and proposed processing flow."
    label = read_text(phase_dir / "conceptual_diagram_label.txt").strip() or "fig:conceptual_diagram"
    environment, placement, width = _phase3_figure_latex_settings()
    figure_tex = rf"""
\begin{{{environment}}}[{placement}]
\centering
\includegraphics[width={width}]{{figures/{target.name}}}
\caption{{{caption}}}
\label{{{label}}}
\end{{{environment}}}
""".strip()
    write_text(build_dir / "conceptual_diagram.tex", figure_tex + "\n")


def _replace_unresolved_phase25_figure_refs(tex: str) -> str:
    """Avoid unresolved figure references when bounded-budget Phase2.5 has no figure floats."""

    labels = set(re.findall(r"\\label\{([^{}]+)\}", tex or ""))

    def replace_ref(match: re.Match[str]) -> str:
        label = match.group(1)
        if label in labels:
            return match.group(0)
        number_match = re.search(r"phase25_figure_([0-9]+)", label)
        if number_match:
            return number_match.group(1)
        return match.group(0)

    return re.sub(r"\\ref\{(fig:phase25_figure_[0-9]+)\}", replace_ref, tex or "")


def _prepare_full_paper_preview_inputs(phase_dir: Path, build_dir: Path) -> None:
    run_dir = phase_dir.parent
    for file_name in [
        "abstract.tex",
        "introduction.tex",
        "system_model_problem_formulation_section.tex",
        "proposed_solution_section.tex",
        "numerical_results_section.tex",
        "conclusion.tex",
    ]:
        text = read_text(phase_dir / file_name)
        if file_name == "numerical_results_section.tex":
            text = sanitize_phase3_2_numerical_results_tex(text)
        elif file_name == "conclusion.tex":
            text = sanitize_phase3_3_conclusion_tex(text)
        text = sanitize_latex_alignment_label_breaks(text)
        text = _rewrite_phase25_figure_paths_for_preview(text, run_dir)
        if file_name == "numerical_results_section.tex":
            text = _replace_unresolved_phase25_figure_refs(text)
        write_text(build_dir / file_name, text)
    _prepare_conceptual_diagram_input(phase_dir, build_dir)


def call_llm_phase3_3_abstract_conclusion_writer(
    *,
    run_dir: Path,
    topic: str,
    system_model_md: str,
    problem_formulation_md: str,
    reformulation_path_md: str,
    algorithm_md: str,
    benchmark_definition_md: str,
    convergence_or_complexity_md: str,
    model_profile: str,
) -> tuple[dict[str, str], dict[str, Any]]:
    llm = create_llm_client(model_profile)
    phase3_3_dir = run_dir / "phase3-3"
    phase3_2_dir = run_dir / "phase3-2"
    phase25_dir = run_dir / "phase2-5"
    method_naming_summary_json = read_text(phase25_dir / "method_naming_summary.json")
    experiment_plan_json = read_text(phase25_dir / "experiment_plan.json")
    phase25_summary = read_json(phase25_dir / "phase25_experiment_summary.json") or {}
    experiment_plan = read_json(phase25_dir / "experiment_plan.json") or {}
    phase3_2_manifest = read_json(phase3_2_dir / "phase3_2_manifest.json") or {}
    phase3_2_numerical_results_tex = read_text(phase3_2_dir / "numerical_results_section.tex")
    verified_registry = read_json(phase25_dir / "phase25_verified_registry.json") or {}
    method_naming_payload = _phase3_3_load_method_naming(method_naming_summary_json, experiment_plan_json)
    paper_facts = build_phase3_3_paper_facts(
        topic=topic,
        system_model_md=system_model_md,
        problem_formulation_md=problem_formulation_md,
        reformulation_path_md=reformulation_path_md,
        algorithm_md=algorithm_md,
        benchmark_definition_md=benchmark_definition_md,
        convergence_or_complexity_md=convergence_or_complexity_md,
        phase25_summary=phase25_summary if isinstance(phase25_summary, dict) else {},
        experiment_plan=experiment_plan if isinstance(experiment_plan, dict) else {},
        method_naming_payload=method_naming_payload,
        phase3_2_manifest=phase3_2_manifest if isinstance(phase3_2_manifest, dict) else {},
        phase3_2_numerical_results_tex=phase3_2_numerical_results_tex,
        verified_registry=verified_registry if isinstance(verified_registry, dict) else {},
    )
    writing_agent_request_json = build_role_agent_request_json(
        run_dir,
        "writing_agent",
        event="phase3_3_prompt",
        max_chars=7000,
    )
    write_text(phase3_3_dir / "writing_agent_request_excerpt.json", writing_agent_request_json)
    prompt = build_phase3_3_abstract_conclusion_prompt(
        topic=topic,
        paper_facts_json=compact_text(json.dumps(paper_facts, ensure_ascii=False, indent=2), 10000),
        writing_agent_request_json=writing_agent_request_json,
    )
    write_text(phase3_3_dir / "phase3_3_prompt.txt", prompt)
    thinking = {"type": "enabled"} if model_profile == "kimi-k2.6-thinking" else None
    response = llm.chat(
        [{"role": "user", "content": prompt}],
        json_mode=True,
        thinking=thinking,
        max_tokens=4000,
    )
    write_text(phase3_3_dir / "phase3_3_raw_response.txt", response.content)
    payload = _safe_json_loads(response.content, {})
    if not isinstance(payload, dict):
        raise ValueError("Phase 3.3 call did not return a valid structured object")
    for retry_idx in range(2):
        abstract_candidate = sanitize_phase3_3_abstract_tex(str(payload.get("abstract_tex") or "").strip())
        undefined_abbrs = find_undefined_abstract_abbreviations(abstract_candidate, paper_facts)
        if not undefined_abbrs:
            break
        retry_prompt = prompt + (
            "\n\nRevision required:\n"
            "The previous abstract still contains undefined acronyms/abbreviations. "
            "Rewrite the abstract_tex and keywords_tex so that every technical acronym in the abstract is expanded on first appearance. "
            f"The unresolved items were: {', '.join(undefined_abbrs)}.\n"
            "If you use a method short name or algorithm acronym, define it immediately after the full method phrase.\n"
            "Return the full JSON again with abstract_tex, keywords_tex, and conclusion_tex."
        )
        suffix = "" if retry_idx == 0 else f"_{retry_idx + 1}"
        write_text(phase3_3_dir / f"phase3_3_prompt_retry{suffix}.txt", retry_prompt)
        retry_response = llm.chat(
            [{"role": "user", "content": retry_prompt}],
            json_mode=True,
            thinking=thinking,
            max_tokens=4000,
        )
        write_text(phase3_3_dir / f"phase3_3_raw_response_retry{suffix}.txt", retry_response.content)
        retry_payload = _safe_json_loads(retry_response.content, {})
        if isinstance(retry_payload, dict):
            payload = retry_payload
    return (
        {
            "abstract_tex": str(payload.get("abstract_tex") or "").strip(),
            "keywords_tex": str(payload.get("keywords_tex") or "").strip(),
            "conclusion_tex": str(payload.get("conclusion_tex") or "").strip(),
        },
        paper_facts,
    )


def _phase3_3_public_summary_gate(
    *,
    abstract_tex: str,
    keywords_tex: str,
    conclusion_tex: str,
    paper_facts: dict[str, Any],
) -> dict[str, Any]:
    abstract_with_keywords = abstract_tex.rstrip() + "\n\n" + keywords_tex
    combined_text = abstract_with_keywords + "\n" + conclusion_tex
    checks = {
        "forbidden_terms_check": analyze_phase3_3_forbidden_terms(combined_text),
        "claim_strength_check": analyze_phase3_3_claim_strength(combined_text),
        "abstract_structure_check": analyze_phase3_3_abstract_structure(abstract_with_keywords),
        "conclusion_structure_check": analyze_phase3_3_conclusion_structure(conclusion_tex),
        "paper_objective_alignment_check": analyze_phase3_3_paper_objective_alignment(combined_text, paper_facts),
    }
    errors: list[str] = []
    for key, check in checks.items():
        if not isinstance(check, dict):
            continue
        passed = check.get("passed")
        if passed is None:
            passed = check.get("ok")
        if passed is False:
            errors.append(f"{key} failed: {json.dumps(check, ensure_ascii=False)}")
    return {"ok": not errors, "errors": errors, "checks": checks}


def _phase3_3_repair_public_summary_llm(
    *,
    run_dir: Path,
    phase3_3_dir: Path,
    model_profile: str,
    repair_round: int,
    gate_report: dict[str, Any],
    paper_facts: dict[str, Any],
    abstract_tex: str,
    keywords_tex: str,
    conclusion_tex: str,
) -> dict[str, str]:
    writing_agent_request_json = read_text(phase3_3_dir / "writing_agent_request_excerpt.json")
    if not writing_agent_request_json.strip():
        writing_agent_request_json = build_role_agent_request_json(
            run_dir,
            "writing_agent",
            event=f"phase3_3_public_summary_repair_round{repair_round}",
            max_chars=7000,
        )
    prompt = (
        "You are the WARA WritingAgent/RepairAgent for Phase 3.3 public summary repair.\n"
        "Repair ONLY the abstract, keywords, and conclusion so that the gate failures below are resolved.\n"
        "Preserve the frozen model, formulation, algorithm, method names, benchmark names, metrics, numerical values, "
        "and evidence scope. Do not invent new experiments, baselines, citations, or claims.\n\n"
        "Return valid JSON only with exactly these string fields: abstract_tex, keywords_tex, conclusion_tex.\n\n"
        "Mandatory repair rules:\n"
        "- conclusion_tex must contain \\section{Conclusion} and must summarize the main figure-supported findings.\n"
        "- If the gate says findings are missing, add a scoped result sentence using words such as numerical results, improve, gain, outperform, or feasibility, but only using evidence present in paper_facts.\n"
        "- The conclusion should state the design implication under the considered settings, not a generic future-promise sentence.\n"
        "- Keep the abstract and conclusion free of optimizer variables and unsupported numerical claims.\n"
        "- Keep every acronym in the abstract expanded on first use or avoid the acronym.\n\n"
        "Gate report:\n"
        f"{json.dumps(gate_report, ensure_ascii=False, indent=2)}\n\n"
        "Paper facts JSON:\n"
        f"{json.dumps(paper_facts, ensure_ascii=False, indent=2)}\n\n"
        "WritingAgent request JSON:\n"
        f"{writing_agent_request_json}\n\n"
        "Current abstract_tex:\n"
        f"{abstract_tex}\n\n"
        "Current keywords_tex:\n"
        f"{keywords_tex}\n\n"
        "Current conclusion_tex:\n"
        f"{conclusion_tex}\n"
    )
    write_text(phase3_3_dir / f"phase3_3_public_summary_repair_prompt_round{repair_round}.txt", prompt)
    llm = create_llm_client(model_profile)
    thinking = {"type": "enabled"} if model_profile == "kimi-k2.6-thinking" else None
    response = llm.chat(
        [{"role": "user", "content": prompt}],
        json_mode=True,
        thinking=thinking,
        strip_thinking=True,
        max_tokens=int(os.environ.get("WARA_PHASE33_PUBLIC_SUMMARY_REPAIR_MAX_TOKENS", "5000") or 5000),
    )
    write_text(phase3_3_dir / f"phase3_3_public_summary_repair_raw_response_round{repair_round}.txt", response.content)
    payload = _safe_json_loads(response.content, {})
    if not isinstance(payload, dict):
        raise ValueError("Phase 3.3 public-summary repair LLM did not return a JSON object")
    repaired = {
        "abstract_tex": str(payload.get("abstract_tex") or "").strip(),
        "keywords_tex": str(payload.get("keywords_tex") or "").strip(),
        "conclusion_tex": str(payload.get("conclusion_tex") or "").strip(),
    }
    if not all(value.strip() for value in repaired.values()):
        raise ValueError("Phase 3.3 public-summary repair LLM returned an empty field")
    write_text(
        phase3_3_dir / f"phase3_3_public_summary_repair_round{repair_round}.json",
        json.dumps(repaired, ensure_ascii=False, indent=2),
    )
    return repaired


def render_phase3_3_technical_sections_preview_pdf(phase_dir: Path, title: str) -> dict[str, str]:
    wrapper_tex = phase_dir / "phase3_3_technical_sections_preview.tex"
    safe_title = resolve_paper_title(phase_dir, title).replace("\\", " ").replace("{", "(").replace("}", ")")
    proposed_section_title = _impl().load_phase3_section_title(phase_dir.parent)
    wrapper_tex_content = f"""\\documentclass[journal]{{IEEEtran}}
\\usepackage{{amsmath,amssymb,amsfonts,bm,mathtools}}
\\usepackage{{graphicx}}
\\usepackage{{booktabs}}
\\usepackage{{algorithm}}
\\usepackage{{algpseudocode}}
\\usepackage[hidelinks]{{hyperref}}

\\begin{{document}}
\\title{{{safe_title}}}
\\author{{WARA CUHKSZ}}
\\maketitle

\\input{{abstract.tex}}

\\section{{System Model and Problem Formulation}}\\label{{sec:system_model}}
\\input{{system_model_problem_formulation_section.tex}}

\\section{{{proposed_section_title}}}\\label{{sec:proposed_solution}}
\\input{{proposed_solution_section.tex}}

\\input{{numerical_results_section.tex}}

\\input{{conclusion.tex}}

\\end{{document}}
""".strip()
    write_text(wrapper_tex, wrapper_tex_content + "\n")

    last_result = None
    for _ in range(2):
        last_result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", wrapper_tex.name],
            cwd=phase_dir,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )

    pdf_path = phase_dir / "phase3_3_technical_sections_preview.pdf"
    log_path = phase_dir / "phase3_3_technical_sections_preview.log"
    write_text(log_path, (last_result.stdout if last_result else "") + "\n" + (last_result.stderr if last_result else ""))
    return {
        "tex_path": str(wrapper_tex),
        "pdf_path": str(pdf_path),
        "log_path": str(log_path),
        "status": "ok" if pdf_path.exists() else "missing_pdf",
    }


def analyze_phase3_3_forbidden_terms(text: str) -> dict[str, Any]:
    forbidden_terms = [
        "pipeline",
        "phase 2.4",
        "phase 2.5",
        "llm",
        "codex",
        "generated_plugin",
        "draft",
        "preliminary",
        "proves",
        "guarantees",
        "statistically significant",
    ]
    hits = [
        term
        for term in forbidden_terms
        if re.search(rf"(?<![A-Za-z0-9-]){re.escape(term)}(?![A-Za-z0-9-])", text, flags=re.IGNORECASE)
    ]
    return {"passed": not hits, "hits": hits}


def analyze_phase3_3_claim_strength(text: str) -> dict[str, Any]:
    warnings = [
        term
        for term in [
            "proves",
            "guarantees",
            "universally optimal",
            "globally optimal",
            "always superior",
            "statistically significant",
        ]
        if re.search(rf"(?<![A-Za-z0-9-]){re.escape(term)}(?![A-Za-z0-9-])", text, flags=re.IGNORECASE)
    ]
    lowered = text.lower()
    qualifiers_present = any(
        phrase in lowered
        for phrase in ["under the considered settings", "in the evaluated scenarios", "over the tested configurations"]
    )
    return {"passed": not warnings, "warnings": warnings, "uses_scope_qualifier": qualifiers_present}


def analyze_phase3_3_abstract_structure(text: str) -> dict[str, Any]:
    body = re.sub(r"\\begin\{abstract\}|\s*\\end\{abstract\}", " ", text, flags=re.I).strip()
    lowered = body.lower()
    undefined_abbreviations = find_undefined_abstract_abbreviations(text)
    notation_issues = find_phase3_3_abstract_notation_issues(text)
    sentences = [
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+", re.sub(r"\\begin\{IEEEkeywords\}.*", "", body, flags=re.I | re.S))
        if part.strip()
    ]
    checklist_like_sentence_count = sum(
        1
        for sentence in sentences
        if re.match(r"^(?:This letter|This paper|We|The proposed|Numerical results|Simulations)\b", sentence, flags=re.I)
    )
    weak_gap_template_count = len(
        re.findall(r"\bExisting\b[^.]{0,160}\bdo not jointly\b", body, flags=re.I)
    )
    checks = {
        "has_environment": r"\begin{abstract}" in text and r"\end{abstract}" in text,
        "has_keywords": r"\begin{IEEEkeywords}" in text and r"\end{IEEEkeywords}" in text,
        "mentions_problem_or_context": any(term in lowered for term in ["system", "network", "communication", "optimization", "resource", "design", "model", "problem"]),
        "mentions_gap_or_limitation": any(term in lowered for term in ["however", "conventional", "existing", "fixed", "limited", "cannot", "challenge", "challenging", "requires", "rather than"]),
        "mentions_method": any(term in lowered for term in ["propose", "develop", "present", "design"]),
        "mentions_result": any(term in lowered for term in ["numerical results", "simulations", "show", "shows", "improve", "gain", "gains", "outperform", "higher", "maintains", "achieves", "yields", "supports", "reduces", "relative to"]),
        "undefined_abbreviations": undefined_abbreviations,
        "abbreviation_discipline_ok": not undefined_abbreviations,
        "notation_issues": notation_issues,
        "notation_discipline_ok": not notation_issues,
        "checklist_like_sentence_count": checklist_like_sentence_count,
        "weak_gap_template_count": weak_gap_template_count,
        "narrative_warning": checklist_like_sentence_count >= 3 or weak_gap_template_count > 0,
    }
    checks["passed"] = all(
        value
        for key, value in checks.items()
        if key
        not in {
            "undefined_abbreviations",
            "notation_issues",
            "checklist_like_sentence_count",
            "weak_gap_template_count",
            "narrative_warning",
        }
    )
    return checks


def analyze_phase3_3_conclusion_structure(text: str) -> dict[str, Any]:
    body = re.sub(r"\\section\*?\{Conclusion\}", " ", text, flags=re.I).strip()
    lowered = body.lower()
    contribution_patterns = [
        r"\bwe\s+(?:proposed|developed|studied|addressed|introduced|designed|investigated)\b",
        r"\bthis\s+(?:letter|paper|work)\s+(?:proposed|developed|studied|addressed|introduced|designed|investigated)\b",
        r"\bthe\s+proposed\b.{0,160}\b(?:method|scheme|algorithm|design|framework|approach)\b",
        r"\b(?:our|this)\s+(?:method|scheme|algorithm|design|framework|approach)\b",
        r"\bthe\s+key\s+mechanism\b",
        r"\bthe\s+main\s+contribution\b",
    ]
    checks = {
        "has_section": bool(re.search(r"\\section\*?\{Conclusion\}", text, flags=re.I)),
        "mentions_contribution": any(re.search(pattern, lowered, flags=re.S) for pattern in contribution_patterns),
        "mentions_findings": any(term in lowered for term in ["results", "numerical", "improve", "gain", "outperform", "feasibility"]),
        "mentions_implication": any(
            term in lowered
            for term in ["indicates", "indicate", "implies", "imply", "supports", "suggests", "suggest", "future", "value of"]
        ),
    }
    checks["passed"] = all(checks.values())
    return checks


def analyze_phase3_3_paper_objective_alignment(text: str, paper_facts: dict[str, Any]) -> dict[str, Any]:
    lowered = text.lower()
    topic = str(paper_facts.get("topic", "")).lower()
    metric = paper_facts.get("main_metric") or {}
    metric_name = " ".join([str(metric.get("display_name", "")), str(metric.get("name", ""))]).lower().replace("_", " ")
    proposed_short = str(((paper_facts.get("proposed_method") or {}).get("display_name_short", ""))).lower()
    proposed_long = str(((paper_facts.get("proposed_method") or {}).get("display_name_long", ""))).lower()
    baseline_short = str(((paper_facts.get("main_benchmark") or {}).get("display_name_short", ""))).lower()
    baseline_long = str(((paper_facts.get("main_benchmark") or {}).get("display_name_long", ""))).lower()
    proposed_candidates = [re.sub(r"\s*\(.*?\)\s*", "", proposed_short).strip(), proposed_short, proposed_long]
    baseline_candidates = [re.sub(r"\s*\(.*?\)\s*", "", baseline_short).strip(), baseline_short, baseline_long]
    checks = {
        "mentions_topic_or_context": bool(topic and any(token for token in topic.split()[:4] if token and token in lowered)),
        "mentions_metric_or_goal": bool(metric_name and any(token for token in metric_name.split() if len(token) > 3 and token in lowered)),
        "mentions_proposed_method": any(candidate and candidate in lowered for candidate in proposed_candidates),
        "mentions_benchmark_or_limitation": any(candidate and candidate in lowered for candidate in baseline_candidates) or "benchmark" in lowered,
    }
    checks["passed"] = sum(1 for value in checks.values() if value) >= 3
    return checks


def extract_phase3_3_numbers_used(text_parts: dict[str, str], phase3_2_numbers_used: list[dict[str, Any]]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for part_name, text in text_parts.items():
        for item in phase3_2_numbers_used:
            number = str(item.get("number", "")).strip()
            if not number:
                continue
            candidates = [number]
            try:
                number_float = float(number)
                candidates.extend([f"{number_float:.1f}", f"{number_float:.2f}"])
                if abs(number_float) < 1.0:
                    percent_value = number_float * 100.0
                    candidates.extend([f"{percent_value:.1f}", f"{percent_value:.2f}"])
            except ValueError:
                pass
            if not any(candidate and candidate in text for candidate in candidates):
                continue
            key = (part_name, str(item.get("source_field_or_row", "")))
            if key in seen:
                continue
            seen.add(key)
            records.append(
                {
                    "number": number,
                    "source_file": str(item.get("source_file", "")),
                    "source_field_or_row": str(item.get("source_field_or_row", "")),
                    "where_used": part_name,
                }
            )
    return records


def run_phase3_3_technical_sections_package(run_dir: Path, paper_target: str = "IEEE WCL") -> dict[str, Any]:
    _ = paper_target
    run_dir = Path(run_dir)
    summary_payload = read_json(run_dir / "phase2_summary.json") or {}
    topic = str(summary_payload.get("topic", run_dir.name))
    model_profile = str(summary_payload.get("model_profile") or DEFAULT_MODEL_PROFILE)
    phase3_3_dir = run_dir / "phase3-3"
    phase3_3_dir.mkdir(parents=True, exist_ok=True)
    write_text(phase3_3_dir / "phase3_3_design_notes.md", build_phase3_3_design_notes())
    write_text(DOCS_DIR / "pipeline_experiment_design.md", build_pipeline_experiment_design_notes())

    system_model_md = read_text(run_dir / "phase2-1" / "system_model.md")
    problem_formulation_md = read_text(run_dir / "phase2-1" / "problem_formulation.md")
    reformulation_path_md = read_text(run_dir / "phase2-2" / "reformulation_path.md")
    algorithm_md = read_text(run_dir / "phase2-3" / "algorithm.md")
    benchmark_definition_md = read_text(run_dir / "phase2-4" / "benchmark_plan.md")
    if not benchmark_definition_md.strip():
        benchmark_definition_md = read_text(run_dir / "phase2-3" / "benchmark_definition.md")
    convergence_or_complexity_md = read_text(run_dir / "phase2-3" / "convergence_or_complexity.md")

    generated, paper_facts = call_llm_phase3_3_abstract_conclusion_writer(
        run_dir=run_dir,
        topic=topic,
        system_model_md=system_model_md,
        problem_formulation_md=problem_formulation_md,
        reformulation_path_md=reformulation_path_md,
        algorithm_md=algorithm_md,
        benchmark_definition_md=benchmark_definition_md,
        convergence_or_complexity_md=convergence_or_complexity_md,
        model_profile=model_profile,
    )
    abstract_tex = sanitize_phase3_3_abstract_tex(generated["abstract_tex"])
    keywords_tex = sanitize_phase3_3_keywords_tex(generated.get("keywords_tex", ""), paper_facts)
    conclusion_tex = sanitize_phase3_3_conclusion_tex(generated["conclusion_tex"])
    write_text(phase3_3_dir / "abstract.tex", abstract_tex.rstrip() + "\n\n" + keywords_tex)
    write_text(phase3_3_dir / "keywords.tex", keywords_tex)
    write_text(phase3_3_dir / "conclusion.tex", conclusion_tex)

    phase1_snippet = sanitize_phase3_3_embedded_section_tex(load_phase3_1_system_model_problem_snippet(run_dir).strip())
    phase3_snippet = sanitize_phase3_3_embedded_section_tex(load_phase3_1_proposed_solution_snippet(run_dir))
    phase3_2_snippet = sanitize_phase3_2_numerical_results_tex(read_text(run_dir / "phase3-2" / "numerical_results_section.tex"))

    repair_round_limit = max(0, int(os.environ.get("WARA_PHASE32_TECHNICAL_REPAIR_ROUNDS", "3") or 3))
    technical_gate = _phase3_3_validate_technical_sections(
        phase3_3_dir,
        system_model_problem_formulation_section=phase1_snippet,
        proposed_solution_section=phase3_snippet,
        numerical_results_section=phase3_2_snippet,
    )
    repair_history: list[dict[str, Any]] = []
    for repair_round in range(1, repair_round_limit + 1):
        if technical_gate.get("ok", False):
            break
        repaired = _phase3_3_repair_technical_sections_llm(
            run_dir=run_dir,
            phase3_3_dir=phase3_3_dir,
            topic=topic,
            model_profile=model_profile,
            repair_round=repair_round,
            gate_report=technical_gate,
            system_model_problem_formulation_section=phase1_snippet,
            proposed_solution_section=phase3_snippet,
            numerical_results_section=phase3_2_snippet,
        )
        phase1_snippet = repaired["system_model_problem_formulation_section"]
        phase3_snippet = repaired["proposed_solution_section"]
        phase3_2_snippet = repaired["numerical_results_section"]
        technical_gate = _phase3_3_validate_technical_sections(
            phase3_3_dir,
            system_model_problem_formulation_section=phase1_snippet,
            proposed_solution_section=phase3_snippet,
            numerical_results_section=phase3_2_snippet,
        )
        repair_history.append({"round": repair_round, "gate": technical_gate})
    write_text(phase3_3_dir / "phase3_3_technical_repair_history.json", json.dumps(repair_history, ensure_ascii=False, indent=2))
    if not technical_gate.get("ok", False):
        raise ValueError(
            "Phase 3.3 technical-section gate failed after LLM repair: "
            + "; ".join(str(item) for item in technical_gate.get("errors", [])[:8])
        )
    write_text(phase3_3_dir / "system_model_problem_formulation_section.tex", phase1_snippet + "\n")
    write_text(phase3_3_dir / "proposed_solution_section.tex", phase3_snippet)
    write_text(phase3_3_dir / "numerical_results_section.tex", phase3_2_snippet)

    public_summary_repair_history: list[dict[str, Any]] = []
    public_summary_gate = _phase3_3_public_summary_gate(
        abstract_tex=abstract_tex,
        keywords_tex=keywords_tex,
        conclusion_tex=conclusion_tex,
        paper_facts=paper_facts,
    )
    public_summary_repair_limit = max(0, int(os.environ.get("WARA_PHASE33_PUBLIC_SUMMARY_REPAIR_ROUNDS", "3") or 3))
    for repair_round in range(1, public_summary_repair_limit + 1):
        if public_summary_gate.get("ok", False):
            break
        repaired_summary = _phase3_3_repair_public_summary_llm(
            run_dir=run_dir,
            phase3_3_dir=phase3_3_dir,
            model_profile=model_profile,
            repair_round=repair_round,
            gate_report=public_summary_gate,
            paper_facts=paper_facts,
            abstract_tex=abstract_tex,
            keywords_tex=keywords_tex,
            conclusion_tex=conclusion_tex,
        )
        abstract_tex = sanitize_phase3_3_abstract_tex(repaired_summary["abstract_tex"])
        keywords_tex = sanitize_phase3_3_keywords_tex(repaired_summary["keywords_tex"], paper_facts)
        conclusion_tex = sanitize_phase3_3_conclusion_tex(repaired_summary["conclusion_tex"])
        write_text(phase3_3_dir / "abstract.tex", abstract_tex.rstrip() + "\n\n" + keywords_tex)
        write_text(phase3_3_dir / "keywords.tex", keywords_tex)
        write_text(phase3_3_dir / "conclusion.tex", conclusion_tex)
        public_summary_gate = _phase3_3_public_summary_gate(
            abstract_tex=abstract_tex,
            keywords_tex=keywords_tex,
            conclusion_tex=conclusion_tex,
            paper_facts=paper_facts,
        )
        public_summary_repair_history.append({"round": repair_round, "gate": public_summary_gate})
    write_text(
        phase3_3_dir / "phase3_3_public_summary_repair_history.json",
        json.dumps(public_summary_repair_history, ensure_ascii=False, indent=2),
    )

    preview = render_phase3_3_technical_sections_preview_pdf(phase3_3_dir, topic)
    abstract_word_count = len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", abstract_tex))
    keywords_word_count = len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", keywords_tex))
    conclusion_word_count = len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", conclusion_tex))
    method_names_used = {
        "proposed_display_name_long": str(((paper_facts.get("proposed_method") or {}).get("display_name_long", ""))),
        "proposed_display_name_short": str(((paper_facts.get("proposed_method") or {}).get("display_name_short", ""))),
        "benchmark_display_name_long": str(((paper_facts.get("main_benchmark") or {}).get("display_name_long", ""))),
        "benchmark_display_name_short": str(((paper_facts.get("main_benchmark") or {}).get("display_name_short", ""))),
    }
    phase3_2_manifest = read_json(run_dir / "phase3-2" / "phase3_2_manifest.json") or {}
    abstract_with_keywords = abstract_tex.rstrip() + "\n\n" + keywords_tex
    text_parts = {"abstract": abstract_tex, "conclusion": conclusion_tex}
    combined_text = abstract_with_keywords + "\n" + conclusion_tex
    manifest = {
        "paper_target": paper_target,
        "paper_writing_mode": _paper_writing_mode_snapshot(),
        "title": topic,
        "input_files_used": [
            str(run_dir / "phase2-1" / "system_model.md"),
            str(run_dir / "phase2-1" / "problem_formulation.md"),
            str(run_dir / "phase2-2" / "reformulation_path.md"),
            str(run_dir / "phase2-3" / "algorithm.md"),
            str(run_dir / "phase2-4" / "benchmark_plan.md"),
            str(run_dir / "phase2-3" / "convergence_or_complexity.md"),
            str(run_dir / "phase2-5" / "phase25_experiment_summary.json"),
            str(run_dir / "phase2-5" / "method_naming_summary.json"),
            str(run_dir / "phase2-5" / "experiment_plan.json"),
            str(run_dir / "phase3-2" / "phase3_2_manifest.json"),
            str(run_dir / "phase3-2" / "numerical_results_section.tex"),
        ],
        "abstract_path": str(phase3_3_dir / "abstract.tex"),
        "keywords_path": str(phase3_3_dir / "keywords.tex"),
        "conclusion_path": str(phase3_3_dir / "conclusion.tex"),
        "preview_pdf_path": str(phase3_3_dir / "phase3_3_technical_sections_preview.pdf"),
        "prompt_path": str(phase3_3_dir / "phase3_3_prompt.txt"),
        "raw_response_path": str(phase3_3_dir / "phase3_3_raw_response.txt"),
        "word_count_abstract": abstract_word_count,
        "word_count_keywords": keywords_word_count,
        "word_count_conclusion": conclusion_word_count,
        "method_names_used": method_names_used,
        "method_name_default_used": bool(paper_facts.get("method_name_default_used")),
        "numbers_used": extract_phase3_3_numbers_used(text_parts, phase3_2_manifest.get("numbers_used", []) if isinstance(phase3_2_manifest, dict) else []),
        "forbidden_terms_check": analyze_phase3_3_forbidden_terms(combined_text),
        "claim_strength_check": analyze_phase3_3_claim_strength(combined_text),
        "abstract_structure_check": analyze_phase3_3_abstract_structure(abstract_with_keywords),
        "conclusion_structure_check": analyze_phase3_3_conclusion_structure(conclusion_tex),
        "paper_objective_alignment_check": analyze_phase3_3_paper_objective_alignment(combined_text, paper_facts),
        "public_summary_repair_gate": public_summary_gate,
        "public_summary_repair_rounds": len(public_summary_repair_history),
        "paper_facts": paper_facts,
        "preview": preview,
    }
    write_text(phase3_3_dir / "phase3_3_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest
