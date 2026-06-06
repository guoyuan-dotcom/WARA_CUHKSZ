from __future__ import annotations

import importlib
import json
import os
import re
from pathlib import Path
from typing import Any

from pipeline_core import DEFAULT_MODEL_PROFILE


def _impl() -> Any:
    return importlib.import_module("phase_runtime_impl")


def _replace_phase3_2_figure_aliases(numerical_results_tex: str, phase25_summary: dict[str, Any]) -> str:
    figures = [item for item in phase25_summary.get("figures", []) if isinstance(item, dict)]
    replacements: dict[str, str] = {}
    if len(figures) >= 1:
        filename = str(figures[0].get("filename_pdf") or "").strip()
        if filename:
            replacements["../phase2-5/figures/figure_1.pdf"] = f"../phase2-5/figures/{filename}"
    if len(figures) >= 2:
        filename = str(figures[1].get("filename_pdf") or "").strip()
        if filename:
            replacements["../phase2-5/figures/figure_2.pdf"] = f"../phase2-5/figures/{filename}"
    for old, new in replacements.items():
        numerical_results_tex = numerical_results_tex.replace(old, new)
    return numerical_results_tex


def _enforce_phase3_2_registered_figures_only(numerical_results_tex: str, phase25_summary: dict[str, Any]) -> str:
    """Remove figure prose that references files not registered by Phase 2.5."""

    figures = [item for item in phase25_summary.get("figures", []) if isinstance(item, dict)]
    allowed_filenames = {
        str(figure.get("filename_pdf") or "").strip()
        for figure in figures
        if str(figure.get("filename_pdf") or "").strip()
    }
    allowed_labels = {
        f"fig:phase25_{str(figure.get('figure_id') or '').strip()}"
        for figure in figures
        if str(figure.get("figure_id") or "").strip()
    }
    missing_labels: set[str] = set()

    def keep_or_drop_figure(match: re.Match[str]) -> str:
        block = match.group(0)
        include_match = re.search(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", block)
        label_match = re.search(r"\\label\{([^}]+)\}", block)
        include_path = include_match.group(1).strip() if include_match else ""
        filename = Path(include_path).name
        label = label_match.group(1).strip() if label_match else ""
        if filename in allowed_filenames:
            return block
        if label:
            missing_labels.add(label)
        return "\n"

    cleaned = re.sub(
        r"\n?\\begin\{figure\}\[[^\]]*\].*?\\end\{figure\}\s*",
        keep_or_drop_figure,
        numerical_results_tex,
        flags=re.S,
    )
    for label in sorted(missing_labels):
        if label in allowed_labels:
            continue
        cleaned = re.sub(
            rf"\n?Fig\.~\\ref\{{{re.escape(label)}\}}.*?(?=\n\s*(?:\\begin\{{figure\}}|\\section|\\subsection|$))",
            "\n",
            cleaned,
            flags=re.S,
        )
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def _phase3_2_conservative_continuation_allowed(run_dir: Path) -> bool:
    raw_value = (
        os.environ.get("WARA_PHASE3_ALLOW_NONPAPER_EVIDENCE")
        or os.environ.get("WARA_PHASE25_ALLOW_NONPAPER_CONTINUATION")
        or ""
    )
    return str(raw_value).strip().lower() in {"1", "true", "yes"}


def run_phase3_2_numerical_results_package(run_dir: Path, paper_target: str = "IEEE WCL") -> dict[str, Any]:
    impl = _impl()
    run_dir = Path(run_dir)
    summary_payload = impl.read_json(run_dir / "phase2_summary.json") or {}
    topic = str(summary_payload.get("topic", run_dir.name))
    model_profile = str(summary_payload.get("model_profile") or DEFAULT_MODEL_PROFILE)
    phase3_2_dir = run_dir / "phase3-2"
    phase3_2_dir.mkdir(parents=True, exist_ok=True)
    impl.write_text(phase3_2_dir / "phase3_2_design_notes.md", impl.build_phase3_2_design_notes())
    evidence_gate = impl.validate_phase3_2_paper_evidence_gate(run_dir)
    impl.write_text(phase3_2_dir / "phase3_2_evidence_gate.json", json.dumps(evidence_gate, ensure_ascii=False, indent=2))
    if not evidence_gate.get("ok"):
        if _phase3_2_conservative_continuation_allowed(run_dir):
            impl.write_text(
                phase3_2_dir / "phase3_2_evidence_scope_warning.md",
                "# Phase 3.2 Evidence Scope Warning\n\n"
                "Phase 2.5 did not reach paper-ready status after bounded repair/expansion, "
                "but it produced finite usable figures/results. The AnalysisAgent must write "
                "a conservative numerical-results section from the selected evidence package; "
                "final review must keep this limitation visible.\n\n"
                + "\n".join(f"- {error}" for error in evidence_gate.get("errors", []))
                + "\n",
            )
        else:
            impl.write_text(
                phase3_2_dir / "phase3_2_evidence_not_ready.md",
                "# Phase 3.2 Evidence Not Ready\n\n"
                + "\n".join(f"- {error}" for error in evidence_gate.get("errors", []))
                + "\n",
            )
            raise RuntimeError("Phase 3.2 requires review because Phase 2.5 evidence is not paper-ready: " + "; ".join(evidence_gate.get("errors", [])))

    system_model_md = impl.read_text(run_dir / "phase2-1" / "system_model.md")
    problem_formulation_md = impl.read_text(run_dir / "phase2-1" / "problem_formulation.md")
    algorithm_md = impl.read_text(run_dir / "phase2-3" / "algorithm.md")
    benchmark_definition_md = impl.read_text(run_dir / "phase2-4" / "benchmark_plan.md")
    if not benchmark_definition_md.strip():
        benchmark_definition_md = impl.read_text(run_dir / "phase2-3" / "benchmark_definition.md")

    phase25_summary = impl.read_json(run_dir / "phase2-5" / "phase25_experiment_summary.json") or {}
    try:
        numerical_results_tex = impl.call_llm_phase3_2_numerical_results_writer(
            run_dir=run_dir,
            topic=topic,
            system_model_md=system_model_md,
            problem_formulation_md=problem_formulation_md,
            algorithm_md=algorithm_md,
            benchmark_definition_md=benchmark_definition_md,
            model_profile=model_profile,
        )
    except Exception as exc:  # noqa: BLE001
        impl.write_text(
            phase3_2_dir / "phase3_2_llm_generation_error.txt",
            f"{type(exc).__name__}: {exc}\n",
        )
        raise
    table_csv_text = ""
    numerical_results_tex = impl.sanitize_phase3_2_numerical_results_tex(numerical_results_tex)
    numerical_results_tex = _replace_phase3_2_figure_aliases(numerical_results_tex, phase25_summary)
    numerical_results_tex = _enforce_phase3_2_registered_figures_only(numerical_results_tex, phase25_summary)
    method_naming_summary_json = impl.read_text(run_dir / "phase2-5" / "method_naming_summary.json")
    numerical_results_tex = impl.enforce_phase3_2_plotted_method_definitions(
        numerical_results_tex,
        phase25_summary if isinstance(phase25_summary, dict) else {},
        method_naming_summary_json,
    )
    simulation_setup_facts = impl.read_json(phase3_2_dir / "simulation_setup_facts.json") or impl.build_phase3_2_simulation_setup_facts(run_dir)
    numerical_results_tex = impl.enforce_phase3_2_setup_paragraph(numerical_results_tex, simulation_setup_facts)
    numerical_results_tex = impl.enforce_phase3_2_axis_value_consistency(numerical_results_tex, simulation_setup_facts)
    numerical_results_tex = impl.sanitize_phase3_2_numerical_results_tex(numerical_results_tex)
    contamination_check = impl.analyze_phase3_2_cross_topic_contamination(numerical_results_tex, topic)
    if not contamination_check["passed"]:
        impl.write_text(phase3_2_dir / "phase3_2_cross_topic_contamination.json", json.dumps(contamination_check, ensure_ascii=False, indent=2))
        raise ValueError(f"Phase 3.2 numerical results contain cross-topic leftovers: {contamination_check['hits']}")
    impl.write_text(phase3_2_dir / "numerical_results_section.tex", numerical_results_tex + "\n")
    preview = impl.render_phase3_2_numerical_results_preview_pdf(phase3_2_dir)
    prose_only = re.sub(r"\\begin\{.*?\}.*?\\end\{.*?\}", "", numerical_results_tex, flags=re.S)
    word_count = len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", prose_only))
    table_format_check = {"table_enabled": False, "table_present": r"\begin{table" in numerical_results_tex}
    manifest = {
        "paper_target": paper_target,
        "paper_writing_mode": impl._paper_writing_mode_snapshot(),
        "section_tex_path": str(phase3_2_dir / "numerical_results_section.tex"),
        "preview": preview,
        "word_count": word_count,
        "numbers_used": impl._extract_phase3_2_numbers_used(phase25_summary, table_csv_text, numerical_results_tex),
        "simulation_setup_facts_path": str(phase3_2_dir / "simulation_setup_facts.json"),
        "plotted_method_naming_summary_path": str(phase3_2_dir / "plotted_method_naming_summary.json"),
        "verified_registry_path": str(run_dir / "phase2-5" / "phase25_verified_registry.json"),
        "table_format_check": table_format_check,
        "cross_topic_contamination_check": contamination_check,
    }
    impl.write_text(phase3_2_dir / "phase3_2_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest
