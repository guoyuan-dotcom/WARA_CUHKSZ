from __future__ import annotations

import ast
import copy
import csv
import json
import math
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import yaml

from pipeline_core import compact_text, read_json, read_text, write_text
from pipeline_core.json_utils import _safe_json_loads
from phase_runtime.agent_context import build_role_agent_request_json
from phase_runtime.llm import create_llm_client
from phase_runtime.prompt_templates import render_prompt_template


PHASE25_PAPER_READY_STATUSES = {
    "paper_minimum_ready",
    "paper_preferred_ready",
    "high_confidence_ready",
}


def build_phase3_2_design_notes() -> str:
    return """
# Phase 3.2 Design Notes

## Phase 3.2 mission

Phase 3.2 is the paper-facing numerical-results writing step.

It:
- reads the finalized Phase 2.5 figures and experiment summary
- writes a concise IEEE WCL-style Numerical Results section
- keeps the discussion faithful to existing Phase 2.5 results
- produces a LaTeX section snippet and a preview PDF

Phase 3.2 does not:
- rerun experiments
- redesign figures
- change metrics, baselines, or algorithms
- invent numerical claims that are absent from Phase 2.5 outputs

## Writing structure

The default Numerical Results section follows a compact WCL structure:
- paragraph 1: simulation parameters, reported metric, and compared methods
- figure paragraphs: each starts by saying what the figure plots, then gives observation, reason, and insight

The figures are imported from Phase 2.5 artifacts. Tables are intentionally not used in the current WCL-style section.
""".strip()


def render_phase3_2_numerical_results_preview_pdf(phase_dir: Path) -> dict[str, str]:
    wrapper_tex = phase_dir / "phase3_2_numerical_results_preview.tex"
    wrapper_tex_content = r"""\documentclass[journal]{IEEEtran}
\usepackage{amsmath,amssymb,amsfonts,bm,mathtools}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage[hidelinks]{hyperref}

\begin{document}
\input{numerical_results_section.tex}
\end{document}
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

    pdf_path = phase_dir / "phase3_2_numerical_results_preview.pdf"
    log_path = phase_dir / "phase3_2_numerical_results_preview.log"
    write_text(log_path, (last_result.stdout if last_result else "") + "\n" + (last_result.stderr if last_result else ""))
    return {
        "tex_path": str(wrapper_tex),
        "pdf_path": str(pdf_path),
        "log_path": str(log_path),
        "status": "ok" if pdf_path.exists() else "missing_pdf",
    }


def build_phase3_2_numerical_results_prompt(
    *,
    topic: str,
    system_model_md: str,
    problem_formulation_md: str,
    algorithm_md: str,
    benchmark_definition_md: str,
    method_mechanism_summary: str,
    paper_objective_summary: str,
    figure_to_claim_summary: str,
    figure_observations_summary: str,
    figure_evidence_summary_json: str,
    claim_constraints_summary: str,
    simulation_setup_facts_json: str,
    phase25_summary_json: str,
    verified_registry_json: str,
    figure_captions_md: str,
    writing_agent_request_json: str = "",
    analysis_agent_request_json: str = "",
) -> str:
    role_agent_request_json = analysis_agent_request_json or writing_agent_request_json
    return render_prompt_template(
        "phase3_2/numerical_results.prompt.yaml",
        topic=topic,
        system_model_summary=compact_text(system_model_md, 2200),
        problem_formulation_summary=compact_text(problem_formulation_md, 2200),
        algorithm_summary=compact_text(algorithm_md, 2600),
        benchmark_definition_summary=compact_text(benchmark_definition_md, 2200),
        method_mechanism_summary=compact_text(method_mechanism_summary, 1800),
        paper_objective_summary=compact_text(paper_objective_summary, 1600),
        figure_to_claim_summary=compact_text(figure_to_claim_summary, 1800),
        figure_observations_summary=compact_text(figure_observations_summary, 2200),
        figure_evidence_summary_json=compact_text(figure_evidence_summary_json, 4200),
        claim_constraints_summary=compact_text(claim_constraints_summary, 1400),
        simulation_setup_facts_json=compact_text(simulation_setup_facts_json, 3200),
        phase25_summary_json=compact_text(phase25_summary_json, 5000),
        verified_registry_json=compact_text(verified_registry_json, 7000),
        figure_captions_md=compact_text(figure_captions_md, 2200),
        analysis_agent_request_json=compact_text(role_agent_request_json, 7000),
    )


def _extract_phase3_2_numbers_used(phase25_summary: dict[str, Any], table_csv_text: str, tex_text: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    reader = csv.DictReader(table_csv_text.splitlines())
    for raw_row in reader:
        if not raw_row:
            continue
        row = {str(key).replace("\ufeff", "").strip(): str(value).strip() for key, value in raw_row.items()}
        rows.append(row)
    records: list[dict[str, str]] = []
    prose_only = re.sub(r"\\begin\{.*?\}.*?\\end\{.*?\}", " ", tex_text, flags=re.S)
    prose_sentences = re.split(r"(?<=[.!?])\s+", prose_only.replace("\n", " "))
    all_sentences = re.split(r"(?<=[.!?])\s+", tex_text.replace("\n", " "))
    for row in rows:
        scenario = str(row.get("Scenario", "")).strip()
        mappings: list[tuple[str, str]] = []
        for column_name, raw_value in row.items():
            column_clean = str(column_name).strip()
            if column_clean.lower() in {"scenario", "case", "regime"}:
                continue
            if not str(raw_value).strip():
                continue
            mappings.append((column_clean, column_clean.replace("_", " ")))
        for column_name, label in mappings:
            value = str(row.get(column_name, "")).strip()
            if not value:
                continue
            value_float = None
            try:
                value_float = float(value)
            except ValueError:
                value_float = None
            display_candidates = [value]
            if value_float is not None:
                display_candidates.append(f"{value_float:.2f}")
            matching_sentence = ""
            for sentence in prose_sentences:
                if any(candidate and candidate in sentence for candidate in display_candidates):
                    matching_sentence = sentence.strip()
                    break
            if not matching_sentence and "feasibility" in label.lower():
                for sentence in prose_sentences:
                    if "feasibility" in sentence.lower() or "full feasibility" in sentence.lower():
                        matching_sentence = sentence.strip()
                        break
            if not matching_sentence:
                for sentence in all_sentences:
                    if any(candidate and candidate in sentence for candidate in display_candidates):
                        matching_sentence = sentence.strip()
                        break
            records.append(
                {
                    "number": value,
                    "source_file": "phase2-5/tables/table_1.csv",
                    "source_field_or_row": f"{scenario}::{column_name}",
                    "sentence_where_used": matching_sentence,
                }
            )
    overall = phase25_summary.get("overall", {}) if isinstance(phase25_summary, dict) else {}
    for field in ["proposed_win_rate", "proposed_mean_relative_gain", "proposed_median_relative_gain"]:
        if field in overall:
            value = overall[field]
            value_str = str(value)
            matching_sentence = ""
            for sentence in prose_sentences:
                if value_str in sentence:
                    matching_sentence = sentence.strip()
                    break
            if not matching_sentence:
                for sentence in all_sentences:
                    if value_str in sentence:
                        matching_sentence = sentence.strip()
                        break
            records.append(
                {
                    "number": value_str,
                    "source_file": "phase2-5/phase25_experiment_summary.json",
                    "source_field_or_row": f"overall.{field}",
                    "sentence_where_used": matching_sentence,
                }
            )
    return records


def build_phase3_2_method_mechanism_summary(
    *,
    algorithm_md: str,
    benchmark_definition_md: str,
    method_naming_summary_json: str,
) -> str:
    method_names = ""
    try:
        payload = json.loads(method_naming_summary_json) if method_naming_summary_json.strip() else {}
        methods = payload.get("methods", []) if isinstance(payload, dict) else []
        lines = []
        for item in methods:
            if not isinstance(item, dict):
                continue
            short_name = str(item.get("display_name_short", "")).strip()
            long_name = str(item.get("display_name_long", "")).strip()
            role = str(item.get("role", "")).strip()
            if short_name or long_name:
                lines.append(f"- {role}: {short_name} :: {long_name}")
        method_names = "\n".join(lines)
    except Exception:
        method_names = ""
    return render_prompt_template(
        "phase3_2/method_mechanism_summary.prompt.yaml",
        method_names=method_names,
        algorithm_summary=compact_text(algorithm_md, 1400),
        benchmark_summary=compact_text(benchmark_definition_md, 1200),
    )


def _phase3_2_final_plotted_method_ids(phase25_summary: dict[str, Any]) -> list[str]:
    """Return methods that actually appear in final Phase 2.5 paper figures."""
    if not isinstance(phase25_summary, dict):
        return []
    figures = phase25_summary.get("figures", [])
    if not isinstance(figures, list):
        return []
    ordered: list[str] = []
    for figure in figures:
        if not isinstance(figure, dict):
            continue
        if figure.get("paper_ready") is False or str(figure.get("draft_or_final") or "").lower() == "draft":
            continue
        methods = figure.get("methods", [])
        if not isinstance(methods, list):
            continue
        for method in methods:
            method_id = str(method or "").strip()
            if method_id and method_id not in ordered:
                ordered.append(method_id)
    return ordered


def _phase3_2_method_aliases(method_naming_summary_json: str) -> dict[str, dict[str, str]]:
    try:
        payload = json.loads(method_naming_summary_json) if str(method_naming_summary_json or "").strip() else {}
    except Exception:
        payload = {}
    methods = payload.get("methods", []) if isinstance(payload, dict) else []
    aliases: dict[str, dict[str, str]] = {}
    if not isinstance(methods, list):
        return aliases
    for item in methods:
        if not isinstance(item, dict):
            continue
        method_id = str(item.get("internal_name") or item.get("id") or item.get("name") or "").strip()
        if not method_id:
            continue
        tokens = {
            method_id,
            str(item.get("display_name_short") or "").strip(),
            str(item.get("display_name_long") or "").strip(),
        }
        aliases[method_id] = {
            "role": str(item.get("role") or "").strip(),
            "display_name_short": str(item.get("display_name_short") or "").strip(),
            "display_name_long": str(item.get("display_name_long") or "").strip(),
            "tokens": "\n".join(sorted(token for token in tokens if token)),
        }
    return aliases


def filter_phase3_2_method_naming_summary_for_plotted_methods(
    method_naming_summary_json: str,
    phase25_summary: dict[str, Any],
) -> str:
    """Expose only final plotted methods to the numerical-results writer."""
    plotted = set(_phase3_2_final_plotted_method_ids(phase25_summary))
    if not plotted:
        return method_naming_summary_json
    try:
        payload = json.loads(method_naming_summary_json) if str(method_naming_summary_json or "").strip() else {}
    except Exception:
        return method_naming_summary_json
    if not isinstance(payload, dict):
        return method_naming_summary_json
    methods = payload.get("methods", [])
    if not isinstance(methods, list):
        return method_naming_summary_json
    payload = dict(payload)
    payload["methods"] = [
        item
        for item in methods
        if isinstance(item, dict)
        and str(item.get("internal_name") or item.get("id") or item.get("name") or "").strip() in plotted
    ]
    payload["plotted_method_filter"] = {
        "source": "phase25_summary.figures[].methods",
        "plotted_methods": _phase3_2_final_plotted_method_ids(phase25_summary),
        "policy": "Only define methods that are actually plotted in final paper figures; keep executed-but-not-plotted methods out of paper prose.",
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def filter_phase3_2_benchmark_definition_for_plotted_methods(
    benchmark_definition_md: str,
    method_naming_summary_json: str,
    phase25_summary: dict[str, Any],
) -> str:
    plotted = set(_phase3_2_final_plotted_method_ids(phase25_summary))
    if not plotted:
        return benchmark_definition_md
    aliases = _phase3_2_method_aliases(method_naming_summary_json)
    unplotted = set(aliases) - plotted
    if not unplotted:
        return benchmark_definition_md
    blocked_tokens: list[str] = []
    for method_id in sorted(unplotted):
        blocked_tokens.append(method_id)
        blocked_tokens.extend(token for token in aliases.get(method_id, {}).get("tokens", "").splitlines() if token)
    filtered_lines: list[str] = []
    for line in str(benchmark_definition_md or "").splitlines():
        stripped = line.strip()
        is_method_table_row = stripped.startswith("|") and any(f"`{method_id}`" in stripped for method_id in unplotted)
        mentions_blocked_method = any(token and token in stripped for token in blocked_tokens)
        if is_method_table_row or (mentions_blocked_method and stripped.startswith("|")):
            continue
        filtered_lines.append(line)
    filtered = "\n".join(filtered_lines).strip()
    note = (
        "\n\n### Final plotted-method filter\n"
        "Only the methods that appear in every final Phase 2.5 figure should be defined in the paper text. "
        "Other executed methods are audit-only diagnostics and must not appear in the benchmark-definition itemize block.\n"
        "Final plotted methods: "
        + ", ".join(_phase3_2_final_plotted_method_ids(phase25_summary))
        + ".\n"
    )
    return (filtered + note).strip()


def enforce_phase3_2_plotted_method_definitions(
    tex_text: str,
    phase25_summary: dict[str, Any],
    method_naming_summary_json: str,
) -> str:
    """Remove benchmark-definition items for methods not shown in final figures."""
    plotted = set(_phase3_2_final_plotted_method_ids(phase25_summary))
    if not plotted:
        return tex_text
    aliases = _phase3_2_method_aliases(method_naming_summary_json)
    unplotted = set(aliases) - plotted
    if not unplotted:
        return tex_text
    blocked_tokens: list[str] = []
    for method_id in sorted(unplotted):
        blocked_tokens.append(method_id)
        blocked_tokens.extend(token for token in aliases.get(method_id, {}).get("tokens", "").splitlines() if token)
    blocked_tokens = [token for token in dict.fromkeys(blocked_tokens) if token]

    def item_mentions_blocked_method(item: str) -> bool:
        return any(token in item for token in blocked_tokens)

    def replacement_items_for_plotted_methods() -> str:
        try:
            payload = json.loads(method_naming_summary_json or "{}")
        except json.JSONDecodeError:
            payload = {}
        methods = payload.get("methods") if isinstance(payload, dict) else []
        items: list[str] = []
        if isinstance(methods, list):
            for method in methods:
                if not isinstance(method, dict):
                    continue
                method_id = str(method.get("internal_name") or "").strip()
                if method_id not in plotted or method_id == "proposed":
                    continue
                short = str(method.get("display_name_short") or method_id).strip()
                long = str(method.get("display_name_long") or short).strip().rstrip(".")
                role = str(method.get("role") or "benchmark").replace("_", " ")
                items.append(rf"\item \textbf{{{short}:}} {long}, used as the {role} reference.")
        if not items:
            return ""
        return "\\begin{itemize}\n" + "\n".join(items) + "\n\\end{itemize}"

    def filter_itemize(match: re.Match[str]) -> str:
        block = match.group(0)
        pieces = re.split(r"(\\item\b)", block)
        if len(pieces) < 3:
            return block
        prefix = pieces[0]
        kept = [prefix]
        changed = False
        for idx in range(1, len(pieces), 2):
            marker = pieces[idx]
            body = pieces[idx + 1] if idx + 1 < len(pieces) else ""
            if item_mentions_blocked_method(body):
                changed = True
                continue
            kept.extend([marker, body])
        if not changed:
            return block
        filtered = "".join(kept)
        if r"\item" not in filtered:
            return replacement_items_for_plotted_methods()
        return filtered

    return re.sub(r"\\begin\{itemize\}.*?\\end\{itemize\}", filter_itemize, tex_text, flags=re.S)


def build_phase3_2_paper_objective_summary(
    *,
    problem_formulation_md: str,
    reformulation_path_md: str,
    algorithm_md: str,
    benchmark_definition_md: str,
    experiment_plan_json: str,
) -> str:
    return render_prompt_template(
        "phase3_2/paper_objective_summary.prompt.yaml",
        problem_formulation_summary=compact_text(problem_formulation_md, 1200),
        reformulation_summary=compact_text(reformulation_path_md, 900),
        algorithm_summary=compact_text(algorithm_md, 1200),
        benchmark_summary=compact_text(benchmark_definition_md, 1000),
        experiment_plan_summary=compact_text(experiment_plan_json, 1000),
    )


def build_phase3_2_figure_to_claim_summary(
    *,
    experiment_plan_json: str,
    plot_quality_report_json: str,
) -> str:
    return render_prompt_template(
        "phase3_2/figure_to_claim_summary.prompt.yaml",
        experiment_plan_summary=compact_text(experiment_plan_json, 1800),
        plot_quality_report_summary=compact_text(plot_quality_report_json, 1600),
    )


def build_phase3_2_figure_observations_summary(phase25_summary: dict[str, Any]) -> str:
    figures = phase25_summary.get("figures", []) if isinstance(phase25_summary, dict) else []
    lines: list[str] = []
    for fig in figures:
        if not isinstance(fig, dict):
            continue
        fig_id = str(fig.get("figure_id", ""))
        fig_short = "Figure 1" if fig_id == "figure_1" else "Figure 2" if fig_id == "figure_2" else fig_id
        x_param = str(fig.get("x_axis_param", "")).strip()
        chart_type = str(fig.get("chart_type", "")).strip()
        methods = ", ".join(str(x) for x in fig.get("methods", []))
        y_metric = str(fig.get("y_metric", "")).strip()
        filename_pdf = str(fig.get("filename_pdf", "")).strip()
        paper_ready = bool(fig.get("paper_ready", False))
        blocking = fig.get("blocking_issues", [])
        error_label = str(fig.get("error_display_label", "")).strip()
        num_points = fig.get("num_x_points", "")
        lines.append(
            f"- {fig_short}: chart_type={chart_type or 'unknown'}, y_metric={y_metric or 'primary metric'}, "
            f"x_axis_param='{x_param or 'not specified'}', methods=[{methods}], aggregated_points={num_points}, paper_ready={paper_ready}."
        )
        if filename_pdf:
            lines.append(f"- {fig_short}: use only existing file ../phase2-5/figures/{filename_pdf}.")
        if blocking:
            lines.append(f"- {fig_short}: quality limitations={', '.join(str(item) for item in blocking)}; keep claims scoped to the supplied Phase 2.5 evidence.")
        lines.append(
            f"- {fig_short} discussion should start by stating what the figure plots, then describe the observed trend, explain the reason, and state the paper-level insight supported by this figure."
        )
        if error_label:
            lines.append(f"- {fig_short} uses error bars corresponding to {error_label}.")
    overall = phase25_summary.get("overall", {}) if isinstance(phase25_summary, dict) else {}
    if overall:
        lines.append(f"- Overall proposed win rate: {overall.get('proposed_win_rate')}.")
        lines.append(f"- Overall mean relative gain: {overall.get('proposed_mean_relative_gain')}.")
        lines.append(f"- Overall median relative gain: {overall.get('proposed_median_relative_gain')}.")
    return "\n".join(lines).strip()


def build_phase3_2_table_observations_summary(table_csv_text: str) -> str:
    rows: list[dict[str, str]] = []
    reader = csv.DictReader(table_csv_text.splitlines())
    for raw_row in reader:
        if not raw_row:
            continue
        row = {str(key).replace("\ufeff", "").strip(): str(value).strip() for key, value in raw_row.items()}
        rows.append(row)
    lines: list[str] = []
    if rows:
        lines.append(f"- Table columns available: {', '.join(rows[0].keys())}.")
    for row in rows:
        scenario = row.get("Scenario", "")
        proposed_col = next((key for key in row if "proposed" in key.lower()), "")
        baseline_col = next((key for key in row if "baseline" in key.lower() or "benchmark" in key.lower()), "")
        gain_col = next((key for key in row if "gain" in key.lower() or "improvement" in key.lower()), "")
        proposed = row.get(proposed_col, "")
        baseline = row.get(baseline_col, "")
        gain = row.get(gain_col, "")
        feas_cols = [key for key in row if "feas" in key.lower()]
        feas_text = "; ".join(f"{key}={row.get(key, '')}" for key in feas_cols if row.get(key, ""))
        lines.append(
            f"- Scenario '{scenario}': proposed={proposed}, baseline={baseline}, relative_gain_percent={gain}, feasibility={feas_text or 'not specified'}."
        )
    lines.append("- The table paragraph should identify which scenario group yields the larger average gain and explain that identical feasibility means the gain is not achieved by relaxing constraints.")
    lines.append("- The LaTeX table should be compact and single-column friendly; representative rows are preferred over dumping every raw result row.")
    lines.append("- Use short row labels derived from the current scenario names; do not introduce variables or labels that are absent from the current table.")
    return "\n".join(lines).strip()


def build_phase3_2_claim_constraints_summary(phase25_summary: dict[str, Any]) -> str:
    figure_constraints: list[str] = []
    figures = phase25_summary.get("figures", []) if isinstance(phase25_summary, dict) else []
    for fig in figures:
        if not isinstance(fig, dict):
            continue
        fig_id = str(fig.get("figure_id", ""))
        label = "Figure 1" if fig_id == "figure_1" else "Figure 2" if fig_id == "figure_2" else fig_id
        error_label = str(fig.get("error_display_label", "")).strip()
        if error_label:
            figure_constraints.append(f"- {label}: plotted uncertainty graphics = {error_label}; do not claim statistical significance unless a statistical test is explicitly provided.")
        else:
            figure_constraints.append(f"- {label}: no plotted uncertainty graphics; do not discuss error bars, confidence intervals, seed means, or standard errors.")
    return """- Do not claim strict monotonicity unless directly supported by the aggregated data.
- Do not claim statistical significance unless a statistical test is available.
- If the average gain varies across the x-axis, explain it as regime-dependent behavior tied to the current system model, objective components, and constraints.
- If proposed and baseline have identical feasibility, state that the gain is obtained without violating the constraints.
- Use paper-level implication statements such as "supports the benefit of joint adaptation" rather than overclaims such as "proves universal superiority".
""" + "\n" + "\n".join(figure_constraints)


def sanitize_phase3_2_numerical_results_tex(tex: str) -> str:
    cleaned = tex.replace("\r\n", "\n").strip()
    control_latex_replacements = {
        "\a": "\\",
        "\b": "\\",
        "\f": "\\",
        "\v": "\\",
    }
    for control_char, replacement in control_latex_replacements.items():
        cleaned = cleaned.replace(control_char, replacement)
    cleaned = re.sub(r"[\x00-\x06\x08-\x09\x0b-\x1f\x7f]", "", cleaned)
    if re.search(r"\bOFDM\b|subcarrier", cleaned, flags=re.IGNORECASE):
        cleaned = re.sub(r"(\$M\s*=\s*[^$]+\$)\s*EH users", r"\1 transmit antennas", cleaned)
        cleaned = re.sub(r"(\$N\s*=\s*[^$]+\$)\s*RIS elements", r"\1 OFDM subcarriers", cleaned)
    for command in ("lambda", "rho", "Phi", "Psi", "mathbf", "boldsymbol", "mathrm", "rm"):
        cleaned = cleaned.replace("\\\\" + command, "\\" + command)
    if re.search(r"^\s*\\section\{Numerical Results\}(?!\s*\\label\{sec:numerical_results\})", cleaned):
        cleaned = re.sub(
            r"^\s*\\section\{Numerical Results\}",
            r"\\section{Numerical Results}\n\\label{sec:numerical_results}",
            cleaned,
            count=1,
        )
    cleaned = re.sub(r"\\begin\{figure\}\[[^\]]*\]", lambda _m: r"\begin{figure}[t]", cleaned)
    cleaned = re.sub(r"\\begin\{figure\}(?!\[)", r"\begin{figure}[t]", cleaned)
    cleaned = re.sub(r"\\begin\{table\*\}\[[^\]]*\]", lambda _m: r"\begin{table*}[!t]", cleaned)
    cleaned = re.sub(r"\\begin\{table\*\}(?!\[)", r"\\begin{table*}[!t]", cleaned)
    cleaned = re.sub(r"\\begin\{table\}\[[^\]]*\]", lambda _m: r"\begin{table}[!t]", cleaned)
    cleaned = re.sub(r"\\begin\{table\}(?!\[)", r"\\begin{table}[!t]", cleaned)
    cleaned = cleaned.replace(r"\includegraphics[width=0.9\columnwidth]", r"\includegraphics[width=\linewidth]")
    cleaned = cleaned.replace(r"\includegraphics[width=\columnwidth]", r"\includegraphics[width=\linewidth]")
    cleaned = cleaned.replace(r"\includegraphics[width=0.9\linewidth]", r"\includegraphics[width=\linewidth]")
    cleaned = re.sub(
        r"\\textbf\{MRT direction plus ([^{}]+?) \(MRT\):\}",
        r"\\textbf{Maximum-ratio transmission (MRT) direction plus \1:}",
        cleaned,
    )
    cleaned = re.sub(
        r"\s*The figure should be interpreted as an overall performance comparison across the tested range, rather than as evidence of a strictly monotonic trend\.\s*",
        " ",
        cleaned,
        flags=re.I,
    )
    cleaned = re.sub(
        r"\s*The results should be interpreted as an overall performance comparison across the tested range rather than evidence of a strictly monotonic trend\.\s*",
        " ",
        cleaned,
        flags=re.I,
    )
    cleaned = re.sub(
        r"\s*Both methods maintain full feasibility across all tested scenarios\.\s*",
        r" Both methods achieve 100\\% feasibility across all tested scenarios. ",
        cleaned,
        flags=re.I,
    )
    workflow_replacements = [
        (r"\bUnless otherwise swept\b", "Unless otherwise stated"),
        (r"\bunless otherwise swept\b", "unless otherwise stated"),
        (r"\bplotted sweep\b", "plotted operating range"),
        (r"\btested sweep\b", "evaluated operating range"),
        (r"\bsimulated sweep\b", "simulated operating range"),
        (r"\bacross the sweep\b", "across the evaluated operating range"),
        (r"\bin the sweep\b", "in the evaluated operating range"),
        (r"\bswept value grid\b", "x-axis value grid"),
        (r"\bnon-swept parameter\b", "fixed parameter"),
    ]
    for pattern, replacement in workflow_replacements:
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.I)
    cleaned = re.sub(
        r"\bThe reported metric is\s+eta_service_level\s*\.",
        lambda _m: r"The reported metric is the service-level utility $\eta$.",
        cleaned,
    )
    cleaned = cleaned.replace(r"\begin{figure}[H]", r"\begin{figure}[t]")
    cleaned = cleaned.replace(r"\begin{table}[H]", r"\begin{table}[!t]")
    cleaned = cleaned.replace(r"\begin{figure}[h]", r"\begin{figure}[t]")
    cleaned = cleaned.replace(r"\begin{table}[h]", r"\begin{table}[!t]")
    cleaned = re.sub(r"\n?\\begin\{table\*?\}\[[^\]]*\].*?\\end\{table\*?\}\n?", "\n", cleaned, flags=re.S)
    cleaned = re.sub(r"\n?\\begin\{tabular\}\{[^}]*\}.*?\\end\{tabular\}\n?", "\n", cleaned, flags=re.S)
    cleaned = re.sub(r"\s*Table~\\ref\{[^}]+\}[^.\n]*(?:\.[^\n]*)?\n?", "\n", cleaned, flags=re.S)

    latex_aliases = {
        "eta_service_level": r"\eta",
        "min_harvested_dc_mW": r"P_{\mathrm{DC,min}}",
        "resources.P_max_W": r"P_{\max}",
        "uncertainty.r_h_common": r"r_h",
        "Pmax_dBm": r"P_{\max}",
        "P_max_dBm": r"P_{\max}",
        "Emin_e_mW": r"E_{\min}^{(e)}",
        "Emin_c_mW": r"E_{\min}^{(c)}",
        "E_min_mW": r"E_{\min}",
        "Emin_mW": r"E_{\min}",
        "P_m_dBm": r"P_{m,\mathrm{dBm}}",
        "P_m_dB": r"P_{m,\mathrm{dB}}",
        "Rmin": r"R_{\min}",
        "lambda_s": r"\lambda_s",
        "lambda_p": r"\lambda_p",
        "lambda_sl": r"\lambda_{\rm sl}",
    }

    def replace_aliases(segment: str, in_math: bool) -> str:
        updated = segment
        for raw, latex in latex_aliases.items():
            replacement = latex if in_math else f"${latex}$"
            updated = re.sub(
                rf"(?<![A-Za-z0-9_\\]){re.escape(raw)}(?![A-Za-z0-9_])",
                lambda _m, replacement=replacement: replacement,
                updated,
            )
        return updated

    parts = re.split(r"(\$[^$]*\$)", cleaned)
    cleaned = "".join(replace_aliases(part, part.startswith("$") and part.endswith("$")) for part in parts)

    def replace_tablenotes(match: re.Match[str]) -> str:
        body = match.group(1)
        body = re.sub(r"^\s*\\item\s*", "", body.strip())
        body = re.sub(r"\s+", " ", body).strip()
        return "\n\\vspace{0.25em}\n{\\footnotesize " + body + "}\n"

    cleaned = re.sub(
        r"\\begin\{tablenotes\}(.*?)\\end\{tablenotes\}",
        replace_tablenotes,
        cleaned,
        flags=re.S,
    )
    cleaned = re.sub(r"\n?\\begin\{itemize\}\s*\\end\{itemize\}\n?", "\n", cleaned, flags=re.S)

    table_match = re.search(r"\\begin\{table\}\[ht\](.*?)\\end\{table\}", cleaned, flags=re.S)
    if table_match:
        table_block = table_match.group(0)
        tabular_match = re.search(r"\\begin\{tabular\}\{([^}]*)\}", table_block)
        num_columns = 0
        if tabular_match:
            spec = tabular_match.group(1)
            num_columns = sum(1 for ch in spec if ch in "lcrpmbX")
        if num_columns >= 6:
            table_block_new = table_block.replace(r"\begin{table}[!t]", r"\begin{table*}[!t]", 1)
            table_block_new = table_block_new.replace(r"\end{table}", r"\end{table*}", 1)
            cleaned = cleaned.replace(table_block, table_block_new, 1)

    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip() + "\n"


def _latex_escape_table_text(value: str) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def _shorten_phase3_2_scenario_label(label: str) -> str:
    text = str(label).strip()
    text = re.sub(r"\$[^$]*\$", "", text).strip()
    text = re.sub(r"\s*\([^)]*\)", "", text).strip()
    text = re.sub(r"\s+", " ", text)
    if len(text) > 28:
        return text[:25].rstrip() + "..."
    return text or "Overall"


def _format_phase3_2_table_number(value: str) -> str:
    text = str(value).strip()
    if not text:
        return "--"
    text = text.rstrip("%").strip()
    try:
        numeric = float(text)
    except ValueError:
        return _latex_escape_table_text(text)
    if not math.isfinite(numeric):
        return "--"
    abs_value = abs(numeric)
    if abs_value != 0 and (abs_value >= 1.0e4 or abs_value < 1.0e-3):
        return f"{numeric:.2e}"
    if abs_value >= 100:
        return f"{numeric:.2f}"
    if abs_value >= 10:
        return f"{numeric:.2f}"
    return f"{numeric:.3f}".rstrip("0").rstrip(".")


def _find_phase3_2_table_column(headers: list[str], *, include: list[str], exclude: list[str] | None = None) -> str:
    exclude = exclude or []
    for header in headers:
        lowered = header.lower()
        if all(token.lower() in lowered for token in include) and not any(token.lower() in lowered for token in exclude):
            return header
    return ""


def _phase3_2_metric_key_from_header(header: str, method_token: str) -> str:
    text = str(header).strip()
    text = re.sub(r"\([^)]*\)", "", text)
    text = re.sub(r"\bproposed\b", "", text, flags=re.I)
    text = re.sub(r"\bbaseline\b", "", text, flags=re.I)
    text = re.sub(r"\bbenchmark\b", "", text, flags=re.I)
    text = re.sub(re.escape(method_token), "", text, flags=re.I)
    text = re.sub(r"[^A-Za-z0-9]+", " ", text).strip().lower()
    return re.sub(r"\s+", " ", text)


def _phase3_2_metric_label_from_key(metric_key: str) -> str:
    words = [word for word in str(metric_key).split() if word not in {"mean", "average", "metric"}]
    if not words:
        return "Metric"
    label = " ".join(words[:3]).title()
    short_map = {
        "Objective": "Obj.",
        "Feasible": "Feas.",
        "Feasibility": "Feas.",
        "Constraint Violation": "Viol.",
        "Violation": "Viol.",
        "Solve Time": "Time",
        "Iterations": "Iter.",
    }
    return short_map.get(label, label[:18])


def _phase3_2_paired_metric_columns(headers: list[str], max_pairs: int = 3) -> list[tuple[str, str, str]]:
    skip_tokens = {"feasibility", "feasible", "success"}
    proposed_headers = [
        header for header in headers
        if "proposed" in header.lower()
        and not any(token in _phase3_2_metric_key_from_header(header, "proposed").split() for token in skip_tokens)
    ]
    baseline_headers = [
        header for header in headers
        if ("baseline" in header.lower() or "benchmark" in header.lower())
        and not any(token in _phase3_2_metric_key_from_header(header, "baseline").split() for token in skip_tokens)
    ]
    baseline_by_key = {
        _phase3_2_metric_key_from_header(header, "baseline"): header
        for header in baseline_headers
    }
    baseline_by_key.update(
        {
            _phase3_2_metric_key_from_header(header, "benchmark"): header
            for header in baseline_headers
        }
    )
    pairs: list[tuple[str, str, str]] = []
    used_baseline: set[str] = set()
    for idx, proposed_col in enumerate(proposed_headers):
        metric_key = _phase3_2_metric_key_from_header(proposed_col, "proposed")
        if not metric_key:
            continue
        baseline_col = baseline_by_key.get(metric_key) or (baseline_headers[idx] if idx < len(baseline_headers) else "")
        if not baseline_col:
            continue
        if baseline_col in used_baseline:
            continue
        used_baseline.add(baseline_col)
        pairs.append((_phase3_2_metric_label_from_key(metric_key), proposed_col, baseline_col))
        if len(pairs) >= max_pairs:
            break
    return pairs


def build_phase3_2_dynamic_latex_table(table_csv_text: str) -> str:
    reader = csv.DictReader(table_csv_text.splitlines())
    rows = [
        {str(key).replace("\ufeff", "").strip(): str(value).strip() for key, value in raw.items()}
        for raw in reader
        if raw
    ]
    if not rows:
        return ""
    headers = list(rows[0].keys())
    improvement_col = _find_phase3_2_table_column(headers, include=["improvement"])
    gap_col = _find_phase3_2_table_column(headers, include=["gap"])
    comparison_col = improvement_col or gap_col
    proposed_feas_col = _find_phase3_2_table_column(headers, include=["proposed", "feasibility"], exclude=["baseline"])
    baseline_feas_col = _find_phase3_2_table_column(headers, include=["baseline", "feasibility"])
    metric_pairs = _phase3_2_paired_metric_columns(headers, max_pairs=3)
    has_comparison = bool(comparison_col)
    column_count = 1 + (1 if has_comparison else 0) + len(metric_pairs) + (1 if proposed_feas_col and baseline_feas_col else 0)
    spec = "l" + "c" * (column_count - 1)
    header_cells = ["Regime"]
    if has_comparison:
        header_cells.append("Gap[\\%]" if gap_col and not improvement_col else "Impr.[\\%]")
    header_cells.extend(label for label, _, _ in metric_pairs)
    if proposed_feas_col and baseline_feas_col:
        header_cells.append("Feas.")
    body_lines: list[str] = []
    for row in rows[:4]:
        cells = [
            _latex_escape_table_text(_shorten_phase3_2_scenario_label(row.get("Scenario", ""))),
        ]
        if has_comparison:
            cells.append(_format_phase3_2_table_number(row.get(comparison_col, "")))
        for _, proposed_col, baseline_col in metric_pairs:
            cells.append(f"{_format_phase3_2_table_number(row.get(proposed_col, ''))}/{_format_phase3_2_table_number(row.get(baseline_col, ''))}")
        if proposed_feas_col and baseline_feas_col:
            cells.append(f"{_format_phase3_2_table_number(row.get(proposed_feas_col, ''))}/{_format_phase3_2_table_number(row.get(baseline_feas_col, ''))}")
        body_lines.append(" & ".join(cells) + r" \\")
    return "\n".join(
        [
            r"\begin{table}[!t]",
            r"\footnotesize",
            r"\setlength{\tabcolsep}{2.2pt}",
            r"\renewcommand{\arraystretch}{1.05}",
            r"\centering",
            r"\caption{Claim-focused performance summary.}",
            r"\label{tab:overall_perf}",
            rf"\begin{{tabular}}{{{spec}}}",
            r"\hline",
            " & ".join(header_cells) + r" \\",
            r"\hline",
            *body_lines,
            r"\hline",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )


def replace_phase3_2_table_from_csv(tex: str, table_csv_text: str) -> str:
    table_block = build_phase3_2_dynamic_latex_table(table_csv_text)
    if not table_block:
        return tex
    pattern = r"\\begin\{table\*?\}\[[^\]]*\].*?\\end\{table\*?\}"
    if re.search(pattern, tex, flags=re.S):
        return re.sub(pattern, lambda _match: table_block, tex, count=1, flags=re.S)
    return tex.rstrip() + "\n\n" + table_block + "\n"


def analyze_phase3_2_cross_topic_contamination(tex: str, topic: str) -> dict[str, Any]:
    lowered_topic = str(topic or "").lower()
    forbidden_terms = [
        "movable antenna",
        "movable-antenna",
        "ma-ao",
        "fixed-pos wmmse",
        "fixed-position wmmse",
        "antenna movable range",
        "fig:wsr_delta",
        "fig:wsr_users",
    ]
    if "movable" in lowered_topic:
        forbidden_terms = [term for term in forbidden_terms if "movable" not in term and "ma-ao" not in term and "fixed-pos" not in term]
    lowered = tex.lower()
    hits = [term for term in forbidden_terms if term in lowered]
    return {"passed": not hits, "hits": hits}


def analyze_phase3_2_table_format(tex: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "num_columns": None,
        "uses_compact_font": False,
        "uses_tabcolsep": False,
        "resizebox_used": False,
        "long_header_warnings": [],
        "table_after_section": False,
        "table_fits_single_column_assumed": False,
        "table_environment": "",
        "table_placement": "",
        "table_span": "",
        "table_star_used": False,
        "uses_H_float": False,
        "figure_placement": "[!t]",
        "preview_documentclass": "IEEEtran",
    }
    section_pos = tex.find(r"\section{Numerical Results}")
    table_pos = tex.find(r"\begin{table")
    result["table_after_section"] = section_pos != -1 and table_pos != -1 and table_pos > section_pos
    result["uses_compact_font"] = (r"\footnotesize" in tex) or (r"\scriptsize" in tex)
    result["uses_tabcolsep"] = r"\setlength{\tabcolsep}" in tex
    result["resizebox_used"] = r"\resizebox" in tex
    result["uses_H_float"] = any(token in tex for token in [r"\begin{figure}[H]", r"\begin{table}[H]", r"\begin{figure}[h]", r"\begin{table}[h]"])

    fig_match = re.search(r"\\begin\{figure\}\[([^\]]*)\]", tex)
    if fig_match:
        result["figure_placement"] = f"[{fig_match.group(1)}]"

    table_match = re.search(r"\\begin\{(table\*?)\}\[([^\]]*)\](.*?)\\end\{\1\}", tex, flags=re.S)
    if not table_match:
        return result
    env_name = table_match.group(1)
    placement = table_match.group(2)
    table_tex = table_match.group(0)
    result["table_environment"] = env_name
    result["table_placement"] = f"[{placement}]"
    result["table_star_used"] = env_name == "table*"
    result["table_span"] = "double_column" if env_name == "table*" else "single_column"
    tabular_match = re.search(r"\\begin\{tabular\}\{([^}]*)\}", table_tex)
    if tabular_match:
        spec = tabular_match.group(1)
        result["num_columns"] = sum(1 for ch in spec if ch in "lcrpmbX")
    header_rows = []
    for line in table_tex.splitlines():
        striped = line.strip()
        if "&" in striped and striped.endswith(r"\\"):
            if "hline" not in striped.lower():
                header_rows.append(striped)
            if len(header_rows) >= 1:
                break
    long_header_warnings: list[str] = []
    for row in header_rows:
        row_clean = row.replace(r"\\", "")
        cells = [cell.strip() for cell in row_clean.split("&")]
        for cell in cells:
            plain = re.sub(r"\\[A-Za-z]+", "", cell)
            plain = plain.replace("{", "").replace("}", "").strip()
            if len(plain) > 18:
                long_header_warnings.append(plain)
    result["long_header_warnings"] = long_header_warnings
    result["table_fits_single_column_assumed"] = bool(
        result["num_columns"] is not None
        and result["num_columns"] <= 6
        and result["uses_compact_font"]
        and result["uses_tabcolsep"]
        and not result["long_header_warnings"]
        and env_name != "table*"
    )
    return result


def _phase3_2_format_number(value: Any) -> str:
    if isinstance(value, bool):
        return str(value)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(numeric):
        return str(value)
    if abs(numeric) >= 1000 or (0 < abs(numeric) < 1e-3):
        return f"{numeric:.2e}".replace("e-0", "e-").replace("e+0", "e")
    if abs(numeric - round(numeric)) < 1e-9:
        return str(int(round(numeric)))
    return f"{numeric:.4g}"


def _phase3_2_format_sequence(values: Any) -> str:
    if not isinstance(values, list):
        return ""
    return ",".join(_phase3_2_format_number(value) for value in values)


def _phase3_2_latex_param_name(param_path: str) -> str:
    mapping = {
        "optimization.lambda1": r"\lambda_1",
        "optimization.lambda2": r"\lambda_2",
        "optimization.lambda3": r"\lambda_3",
        "system.M": r"M",
        "system.Nt": r"N_t",
        "system.Nr": r"N_r",
        "system.Ne": r"N_e",
        "system.Mr": r"M_r",
        "system.Me": r"M_e",
        "system.Psi_sat": r"\Psi_{\rm sat}",
        "system.Pmax": r"P_{\max}",
        "system.Pmax_dBm": r"P_{\max}",
        "system.P_max_dBm": r"P_{\max}",
        "constraints.Rmin": r"R_{\min}",
        "constraints.Emin_e_mW": r"E_{\min}^{(e)}",
        "constraints.Emin_c_mW": r"E_{\min}^{(c)}",
        "optimization.lambda_sl": r"\lambda_{\rm sl}",
        "optimization.R_min": r"R_{\min}",
        "optimization.E_min": r"E_{\min}",
        "constraints.gamma_target": r"\gamma",
        "constraints.gamma_target_dB": r"\gamma",
        "constraints.sinr_target_dB": r"\gamma",
        "constraints.uncertainty_radius": r"\epsilon",
        "system.gamma_target": r"\gamma",
        "system.gamma_target_dB": r"\gamma",
        "crb_trace": r"\operatorname{tr}(\mathbf{C}_{\rm CRB})",
        "rate_Mbps": r"R_c",
        "eh_total_mW": r"P_{\rm EH}",
    }
    return mapping.get(param_path, param_path.split(".")[-1])


def _phase3_2_safe_method_text(value: Any, default: str) -> str:
    text = str(value or "").strip()
    text = text.encode("ascii", errors="ignore").decode("ascii")
    text = re.sub(r"\s+", " ", text).strip()
    return text or default


def build_phase3_2_simulation_setup_facts(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    summary_payload = read_json(run_dir / "phase2_summary.json") or {}
    if not isinstance(summary_payload, dict):
        summary_payload = {}
    plan_payload: dict[str, Any] = {}
    try:
        loaded = yaml.safe_load(read_text(run_dir / "phase2-4" / "solver" / "validation_plan.yaml")) or {}
        plan_payload = loaded if isinstance(loaded, dict) else {}
    except Exception:
        plan_payload = {}

    phase25_summary = read_json(run_dir / "phase2-5" / "phase25_experiment_summary.json") or {}
    if not isinstance(phase25_summary, dict):
        phase25_summary = {}
    registry_payload = read_json(run_dir / "phase2-5" / "phase25_verified_registry.json") or {}
    if not isinstance(registry_payload, dict):
        registry_payload = {}
    canonical = plan_payload.get("canonical_config", {}) if isinstance(plan_payload, dict) else {}
    if not isinstance(canonical, dict):
        canonical = {}
    topic = str(summary_payload.get("topic") or "")
    model_text = " ".join(
        [
            topic,
            str(plan_payload.get("problem_family") or ""),
            str(plan_payload.get("objective_sense") or ""),
            read_text(run_dir / "phase2-1" / "system_model.md")[:4000],
            read_text(run_dir / "phase2-1" / "problem_formulation.md")[:4000],
        ]
    ).lower()

    def has_any(*terms: str) -> bool:
        return any(term in model_text for term in terms)

    feature_flags = {
        "ofdm": has_any("ofdm", "subcarrier", "sub-carrier", "multicarrier", "multi-carrier"),
        "ris": has_any("ris", "reconfigurable intelligent surface", "reflecting surface"),
        "energy_harvesting": has_any(
            "energy harvesting",
            "harvested energy",
            "eh user",
            "eh receiver",
            "swipt",
            "rectenna",
            "wireless power transfer",
        ),
        "sensing": has_any("isac", "sensing", "radar", "crb", "beampattern", "target"),
    }

    def normalize_metric_label(label: Any, metric: Any = "") -> str:
        text = str(label or "").strip()
        metric_name = str(metric or "").strip()
        compact = re.sub(r"\s+", "", text)
        wc_alias = "U_" + r"{\mathrm{wc}}"
        wc_alias_short = "U_" + r"\mathrm{wc}"
        if metric_name == "worst_case_utility" and (
            compact in {
                f"${wc_alias}$",
                wc_alias,
                f"${wc_alias_short}$",
                wc_alias_short,
            }
            or ("U_" in compact and "wc" in compact)
        ):
            return r"$\psi$"
        return text

    figures: list[dict[str, Any]] = []
    plot_quality = phase25_summary.get("plot_quality_report", {})
    plot_figures = plot_quality.get("figures", []) if isinstance(plot_quality, dict) else []
    for figure in plot_figures:
        if not isinstance(figure, dict):
            continue
        x_axis_param = str(figure.get("x_axis_param", ""))
        figures.append(
            {
                "figure_id": figure.get("figure_id", ""),
                "chart_type": figure.get("chart_type", ""),
                "purpose": figure.get("purpose", ""),
                "x_axis_param": x_axis_param,
                "x_axis_latex": figure.get("x_axis_label") or _phase3_2_latex_param_name(x_axis_param),
                "x_values": figure.get("x_values", []),
                "y_metric": figure.get("y_metric", ""),
                "y_axis_label": normalize_metric_label(figure.get("y_axis_label", ""), figure.get("y_metric", "")),
                "num_x_points": figure.get("num_x_points", ""),
                "seeds_per_point_summary": figure.get("seeds_per_point_summary", {}),
                "has_error_bars": figure.get("has_error_bars", False),
                "error_display": "95% confidence interval" if figure.get("has_error_bars") else "",
            }
        )

    method_names: dict[str, Any] = {}
    summary_figures = phase25_summary.get("figures", [])
    if isinstance(summary_figures, list):
        first_figure = next((item for item in summary_figures if isinstance(item, dict)), {})
        if isinstance(first_figure, dict):
            method_names = {
                "short": first_figure.get("method_display_names_short", {}),
                "long": first_figure.get("method_display_names_long", {}),
            }

    return {
        "source_files": [
            "phase2-4/solver/validation_plan.yaml",
            "phase2-5/phase25_experiment_summary.json",
            "phase2-5/plot_quality_report.json",
        ],
        "produced_by_phase": phase25_summary.get("produced_by_phase") or phase25_summary.get("produced_by_" + "phase", ""),
        "phase25_status": phase25_summary.get("phase25_status", ""),
        "paper_minimum_ready": phase25_summary.get("paper_minimum_ready", ""),
        "phase24_setup": phase25_summary.get("simulation_setup") or registry_payload.get("simulation_setup") or {},
        "topic": topic,
        "problem_family": plan_payload.get("problem_family", ""),
        "objective_sense": plan_payload.get("objective_sense", ""),
        "feature_flags": feature_flags,
        "canonical_config": copy.deepcopy(canonical),
        "methods": method_names,
        "monte_carlo": {
            "quick_mode": phase25_summary.get("quick_mode", ""),
            "num_cases": phase25_summary.get("num_cases", ""),
            "num_results": phase25_summary.get("num_results", ""),
            "num_comparable_cases": phase25_summary.get("num_comparable_cases", ""),
            "seeds_per_point": [
                figure.get("seeds_per_point_summary", {}) for figure in figures if figure.get("seeds_per_point_summary")
            ],
        },
        "figures": figures,
        "primary_metric": phase25_summary.get("primary_metric", {}),
        "overall": phase25_summary.get("overall", {}),
    }


def render_phase3_2_setup_paragraph(facts: dict[str, Any]) -> str:
    if isinstance(facts, dict) and str(facts.get("produced_by_phase") or facts.get("produced_by_" + "phase") or "") in {"phase2.4", "phase2.4"}:
        phase24_setup = facts.get("phase24_setup", {}) if isinstance(facts.get("phase24_setup"), dict) else {}
        methods = facts.get("methods", {}) if isinstance(facts.get("methods"), dict) else {}
        short_methods = methods.get("short", {}) if isinstance(methods.get("short"), dict) else {}
        method_values = [
            _phase3_2_safe_method_text(value, "")
            for value in short_methods.values()
            if _phase3_2_safe_method_text(value, "")
        ]
        figures = [figure for figure in facts.get("figures", []) if isinstance(figure, dict)] if isinstance(facts.get("figures"), list) else []
        y_axis = ""

        def setup_math(symbol: str, key: str) -> str:
            value = phase24_setup.get(key, "")
            if value in ("", None, []):
                return ""
            if isinstance(value, list):
                shown = _phase3_2_format_sequence(value)
                return f"${symbol}\\in\\{{{shown}\\}}$" if shown else ""
            return f"${symbol}={_phase3_2_format_number(value)}$"

        for figure in figures[:2]:
            y_axis = y_axis or str(figure.get("y_axis_label") or "")
        k_term = setup_math("K", "K")
        nt_term = setup_math("N_t", "N_t")
        system_terms = [
            f"{k_term} users" if k_term else "",
            f"{nt_term} BS antennas" if nt_term else "",
        ]
        model_terms = [
            setup_math("\\beta/\\alpha", "beta_over_alpha"),
            setup_math("\\gamma_{\\min}", "gamma_min"),
            setup_math("\\epsilon", "epsilon_scale"),
            setup_math("\\rho_{\\rm fix}", "fixed_rho"),
        ]
        active_term = setup_math("|\\mathcal A|", "active_scenario_count_design")
        eval_term = setup_math("N_{\\rm eval}", "uncertainty_scenario_count_eval")
        system_terms = [term for term in system_terms if term]
        system_clause = " and ".join(system_terms) if len(system_terms) <= 2 else ", ".join(system_terms[:-1]) + ", and " + system_terms[-1]
        model_clause = ", ".join(term for term in model_terms if term)
        scenario_clause = ""
        if active_term and eval_term:
            scenario_clause = f"Each robust update uses {active_term} active error scenarios and is checked with {eval_term} uncertainty samples"
        elif active_term:
            scenario_clause = f"Each robust update uses {active_term} active error scenarios"
        elif eval_term:
            scenario_clause = f"The final robust check uses {eval_term} uncertainty samples"
        metric_label = y_axis or "the paper-defined objective"
        topic_text = _phase3_2_safe_method_text(facts.get("topic"), "the considered wireless system")
        proposed_name = _phase3_2_safe_method_text(short_methods.get("M0_proposed") or short_methods.get("proposed"), "the proposed scheme")
        return (
            f"In this section, we evaluate {proposed_name} for {topic_text}"
            f"{' with ' + system_clause if system_clause else ''}. "
            f"The reported metric is {metric_label}. "
            f"The main simulation parameters are {model_clause if model_clause else 'set according to the Phase 2.4 experiment design'}"
            f". {' ' + scenario_clause + '.' if scenario_clause else ''}"
        ).replace("  ", " ").strip()

    canonical = facts.get("canonical_config", {}) if isinstance(facts, dict) else {}
    system = canonical.get("system", {}) if isinstance(canonical, dict) and isinstance(canonical.get("system", {}), dict) else {}
    optimization = canonical.get("optimization", {}) if isinstance(canonical, dict) and isinstance(canonical.get("optimization", {}), dict) else {}
    ris = canonical.get("RIS", {}) if isinstance(canonical, dict) and isinstance(canonical.get("RIS", {}), dict) else {}
    geometry = canonical.get("geometry", {}) if isinstance(canonical, dict) and isinstance(canonical.get("geometry", {}), dict) else {}
    sensing = canonical.get("sensing", {}) if isinstance(canonical, dict) and isinstance(canonical.get("sensing", {}), dict) else {}
    eh_model = {}
    for key in ("eh_model", "EH", "eh"):
        if isinstance(canonical, dict) and isinstance(canonical.get(key), dict):
            eh_model = canonical[key]
            break
    constraints = canonical.get("constraints", {}) if isinstance(canonical, dict) and isinstance(canonical.get("constraints", {}), dict) else {}
    methods = facts.get("methods", {}) if isinstance(facts, dict) and isinstance(facts.get("methods", {}), dict) else {}
    short_methods = methods.get("short", {}) if isinstance(methods.get("short", {}), dict) else {}
    long_methods = methods.get("long", {}) if isinstance(methods.get("long", {}), dict) else {}
    proposed_name = _phase3_2_safe_method_text(short_methods.get("proposed") or long_methods.get("proposed"), "the proposed method")
    baseline_key = "baseline"
    if baseline_key not in short_methods and baseline_key not in long_methods:
        candidates = [
            str(candidate).strip()
            for candidate in list(short_methods.keys()) + list(long_methods.keys())
            if str(candidate).strip() and str(candidate).strip() != "proposed"
        ]
        practical_candidates = [
            candidate
            for candidate in candidates
            if not any(token in candidate.lower() for token in ("central", "oracle", "optimal", "upper_bound", "lower_bound", "lp"))
        ]
        for candidate in practical_candidates or candidates:
            if candidate:
                baseline_key = str(candidate).strip()
                break
    baseline_name = _phase3_2_safe_method_text(short_methods.get(baseline_key) or long_methods.get(baseline_key), "the baseline")
    feature_flags = facts.get("feature_flags", {}) if isinstance(facts, dict) and isinstance(facts.get("feature_flags", {}), dict) else {}
    is_ofdm = bool(feature_flags.get("ofdm"))
    has_ris = bool(feature_flags.get("ris"))
    has_eh = bool(feature_flags.get("energy_harvesting"))
    has_sensing = bool(feature_flags.get("sensing"))

    def first_value(*values: Any) -> Any:
        for value in values:
            if value not in (None, "", "None"):
                return value
        return "not specified"

    seed_count = ""
    has_ci = False
    metric_label = ""
    figures = facts.get("figures", []) if isinstance(facts, dict) else []
    for figure in figures if isinstance(figures, list) else []:
        if not isinstance(figure, dict):
            continue
        metric_label = metric_label or str(figure.get("y_axis_label") or figure.get("y_metric") or "").strip()
        seeds = figure.get("seeds_per_point_summary", {})
        if isinstance(seeds, dict) and seeds.get("min") == seeds.get("max") and seeds.get("min"):
            seed_count = _phase3_2_format_number(seeds.get("min"))
        has_ci = has_ci or bool(figure.get("has_error_bars"))

    scenario_geometry = canonical.get("scenario_geometry", {}) if isinstance(canonical, dict) and isinstance(canonical.get("scenario_geometry", {}), dict) else {}
    p0 = first_value(geometry.get("p0"), geometry.get("tx_pos_m"), scenario_geometry.get("tx_pos_m"))
    pr = first_value(geometry.get("pr"), geometry.get("rx_pos_m"), scenario_geometry.get("rx_pos_m"))
    pt = first_value(geometry.get("pt"), geometry.get("target_pos_m"), scenario_geometry.get("target_pos_m"))
    pc = first_value(geometry.get("pc"), geometry.get("ue_pos_m"), scenario_geometry.get("ue_pos_m"))
    geometry_clause = ""
    if all(isinstance(item, list) for item in [p0, pr, pt, pc]) and (has_ris or has_sensing):
        if has_ris and has_sensing:
            geometry_clause = (
                "The BS, receiver-side surface, target, and CU are located at "
                f"$({_phase3_2_format_sequence(p0)})$, $({_phase3_2_format_sequence(pr)})$, "
                f"$({_phase3_2_format_sequence(pt)})$, and $({_phase3_2_format_sequence(pc)})$ m, respectively."
            )
        elif has_ris:
            geometry_clause = (
                "The BS, surface, and CU reference locations are "
                f"$({_phase3_2_format_sequence(p0)})$, $({_phase3_2_format_sequence(pr)})$, "
                f"and $({_phase3_2_format_sequence(pc)})$ m, respectively."
            )
        else:
            geometry_clause = (
                "The BS, sensing target, and CU reference locations are "
                f"$({_phase3_2_format_sequence(p0)})$, $({_phase3_2_format_sequence(pt)})$, "
                f"and $({_phase3_2_format_sequence(pc)})$ m, respectively."
            )

    nr_total = first_value(system.get("Nr_total"), ris.get("M"), ris.get("N"))
    n_ris = first_value(ris.get("N"), system.get("N_RIS"), ris.get("M"), system.get("N") if has_ris and not is_ofdm else None)
    n_id_users = first_value(system.get("K"), system.get("K_users"), system.get("num_id_users"))
    n_tx_antennas = first_value(
        system.get("Nt"),
        system.get("N_t"),
        system.get("num_tx_antennas"),
        system.get("transmit_antennas"),
        system.get("M") if is_ofdm and not has_eh else None,
    )
    n_subcarriers = first_value(
        system.get("N_sc"),
        system.get("num_subcarriers"),
        system.get("subcarriers"),
        system.get("N") if is_ofdm and not has_ris else None,
    )
    n_eh_users = first_value(system.get("M_EH_users"), system.get("num_eh_users"), system.get("M") if has_eh and not is_ofdm else None)
    n_rx = first_value(system.get("Nr"), system.get("N_rx"), system.get("receive_antennas"))
    n_eh = first_value(system.get("Ne"), system.get("N_EH"), ris.get("N_EH"), system.get("Me"))
    n_eh_fraction = first_value(ris.get("N_EH_fraction"), system.get("N_EH_fraction"))
    n_ref = first_value(ris.get("N_ref"), system.get("N_ref"))
    try:
        nr_float = float(nr_total)
        frac_float = float(n_eh_fraction)
        n_ref = int(round(nr_float * (1.0 - frac_float)))
        n_eh = int(round(nr_float * frac_float))
    except Exception:
        pass

    def is_specified(value: Any) -> bool:
        return value not in (None, "", "None", "not specified")

    def math_assignment(symbol: str, value: Any, suffix: str = "") -> str:
        if not is_specified(value):
            return ""
        return f"${symbol}={_phase3_2_format_number(value)}${suffix}"

    def text_value(value: Any, suffix: str = "") -> str:
        if not is_specified(value):
            return ""
        return f"{_phase3_2_format_number(value)}{suffix}"

    bandwidth = first_value(system.get("bandwidth_MHz"), system.get("bandwidth_Hz"), system.get("B_MHz"), system.get("B_Hz"))
    bandwidth_suffix = " MHz" if is_specified(system.get("bandwidth_MHz")) or is_specified(system.get("B_MHz")) else " Hz"
    carrier = first_value(system.get("carrier_GHz"), system.get("carrier_freq_GHz"), system.get("fc_GHz"))
    noise_c = first_value(system.get("noise_power_dBm_c"), system.get("noise_power_c_dBm"), system.get("sigma2_c"), system.get("sigma_c2_dBm"))
    noise_r = first_value(system.get("noise_power_dBm_r"), system.get("noise_power_r_dBm"), system.get("sigma2_r"), system.get("sigma_r2_dBm"))
    noise_linear = first_value(system.get("noise_power"), system.get("sigma2"), system.get("sigma_k2"), system.get("noise_variance"))
    pmax = first_value(system.get("Pmax"), system.get("P_max_dBm"), system.get("Pmax_dBm"))

    setup_items = [
        math_assignment("N_t", n_tx_antennas, " BS antennas"),
        math_assignment("K", n_id_users, " users"),
        math_assignment("N_{\\rm sc}", n_subcarriers, " OFDM subcarriers") if is_ofdm else "",
        math_assignment("M", n_eh_users, " EH users") if has_eh else "",
        math_assignment("N", n_ris, " RIS elements") if has_ris else "",
        math_assignment("N_r", n_rx, " receive antennas") if has_sensing or has_ris else "",
        math_assignment("N_e", n_eh, " EH rectenna elements") if has_eh else "",
        math_assignment("N_{\\rm ref}", n_ref, " reflecting RIS elements") if has_ris else "",
        math_assignment("P_{\\max}", pmax, " dBm"),
    ]
    text_items = [
        f"bandwidth {text_value(bandwidth, bandwidth_suffix)}" if text_value(bandwidth) else "",
        f"carrier frequency {text_value(carrier, ' GHz')}" if text_value(carrier) else "",
        math_assignment("\\sigma^2", noise_linear),
        math_assignment("\\sigma_c^2", noise_c, " dBm"),
        math_assignment("\\sigma_r^2", noise_r, " dBm"),
    ]
    setup_values = [item for item in setup_items + text_items if item]
    topic_text = _phase3_2_safe_method_text(facts.get("topic"), "the considered wireless system")
    intro_sentence = f"In this section, we evaluate {proposed_name} against {baseline_name} for {topic_text}."
    if metric_label:
        intro_sentence += f" The reported metric is {metric_label}."
    setup_sentence = "Unless otherwise stated, the simulations use " + ", ".join(setup_values) + "." if setup_values else ""

    harvester_items = [
        math_assignment("\\Psi_{\\rm sat}", first_value(eh_model.get("saturation_mW"), eh_model.get("saturation_level_mW"), eh_model.get("Psi_sat_mW"), eh_model.get("Ms"), system.get("Psi_sat")), " mW"),
        math_assignment("a", first_value(eh_model.get("sensitivity_a"), eh_model.get("a_steepness"), eh_model.get("a"), system.get("a_EH"))),
        math_assignment("b", first_value(eh_model.get("inflection_b"), eh_model.get("b_offset_mW"), eh_model.get("b_W"), eh_model.get("b_offset"), eh_model.get("b"), system.get("b_EH"))),
        math_assignment("P_{\\rm sen}", first_value(eh_model.get("sensitivity_mW"), eh_model.get("sensitivity_level_mW")), " mW"),
    ]
    constraint_items = [
        math_assignment("R_{\\min}", first_value(optimization.get("R_min_bpsHz"), optimization.get("R_min_Mbps"), optimization.get("R_min"), constraints.get("Rmin"), constraints.get("R_min")), " b/s/Hz"),
        math_assignment("E_{\\min}^{(e)}", first_value(constraints.get("Emin_e_mW"), constraints.get("E_min_e_mW"), optimization.get("Emin_e_mW")), " mW") if has_eh else "",
        math_assignment("E_{\\min}^{(c)}", first_value(constraints.get("Emin_c_mW"), constraints.get("E_min_c_mW"), optimization.get("Emin_c_mW")), " mW") if has_eh else "",
        math_assignment("E_{\\min}", first_value(optimization.get("E_min_dBm"), optimization.get("E_min_mW"), optimization.get("E_min"), constraints.get("E_min")), " mW") if has_eh else "",
    ]
    weight_items = [
        math_assignment("\\alpha_c", first_value(optimization.get("alpha_comm"), optimization.get("lambda_1"), optimization.get("lambda1"))),
        math_assignment("\\alpha_s", first_value(optimization.get("alpha_sensing"), optimization.get("lambda_2"), optimization.get("lambda2"))) if has_sensing else "",
        math_assignment("\\alpha_e", first_value(optimization.get("alpha_EH"), optimization.get("lambda_3"), optimization.get("lambda3"))) if has_eh else "",
        math_assignment("\\lambda_{\\rm sl}", optimization.get("lambda_sl")),
    ]
    harvester_values = [item for item in harvester_items if item]
    constraint_values = [item for item in constraint_items if item]
    weight_values = [item for item in weight_items if item]
    qos_parts: list[str] = []
    if has_eh and harvester_values:
        qos_parts.append("the nonlinear harvester uses " + ", ".join(harvester_values))
    if constraint_values:
        qos_parts.append("the QoS constraints set " + ", ".join(constraint_values))
    if weight_values:
        qos_parts.append("the configured optimization weights include " + ", ".join(weight_values))
    qos_sentence = ". ".join(part[:1].upper() + part[1:] for part in qos_parts) + "." if qos_parts else ""
    sensing_sentence = ""
    if has_sensing and (sensing.get("N") or sensing.get("eta")):
        eta = sensing.get("eta")
        eta_text = _phase3_2_format_sequence(eta) if isinstance(eta, list) else ""
        sensing_sentence = (
            f"The sensing block uses $N={_phase3_2_format_number(sensing.get('N'))}$ samples"
            + (f" and target parameters $[{eta_text}]$." if eta_text else ".")
        )
    ci_clause = " with 95\\% confidence intervals" if has_ci else ""
    mc_sentence = (
        "The comparison uses "
        f"{seed_count or 'the configured Monte Carlo'} independent channel realizations for each plotted setting"
        f"{ci_clause}."
    )
    return " ".join(
        sentence for sentence in [intro_sentence, setup_sentence, geometry_clause, qos_sentence, sensing_sentence, mc_sentence] if sentence
    ).replace("  ", " ").strip()


def enforce_phase3_2_setup_paragraph(tex_text: str, facts: dict[str, Any]) -> str:
    setup_paragraph = render_phase3_2_setup_paragraph(facts)
    if not setup_paragraph:
        return tex_text
    section_match = re.search(r"(\\section\{Numerical Results\}(?:\s*\\label\{[^}]+\})?\s*)", tex_text)
    if not section_match:
        return tex_text
    start = section_match.end()
    remainder = tex_text[start:]
    first_para_match = re.search(r"\S.*?(?=\n\s*\n)", remainder, flags=re.S)
    if not first_para_match:
        return tex_text[:start] + setup_paragraph + "\n\n" + remainder.lstrip()
    para_start = start + first_para_match.start()
    para_end = start + first_para_match.end()
    first_paragraph = remainder[first_para_match.start() : first_para_match.end()].strip()
    if re.match(r"(?is)^In this section,\s+we\b", first_paragraph):
        return tex_text
    return tex_text[:para_start] + setup_paragraph + tex_text[para_end:]


def enforce_phase3_2_axis_value_consistency(tex_text: str, facts: dict[str, Any]) -> str:
    figures = facts.get("figures", []) if isinstance(facts, dict) else []
    if not isinstance(figures, list):
        return tex_text
    updated = tex_text
    values_by_param: dict[str, list[list[Any]]] = {}
    for figure in figures:
        if not isinstance(figure, dict):
            continue
        param = str(figure.get("x_axis_latex") or _phase3_2_latex_param_name(str(figure.get("x_axis_param", ""))))
        if not param:
            continue
        values = figure.get("x_values", [])
        if isinstance(values, list):
            values_by_param.setdefault(param, []).append(values)
    for param, value_lists in values_by_param.items():
        try:
            normalized_sets = {
                tuple(sorted({float(value) for value in raw_values}))
                for raw_values in value_lists
            }
        except (TypeError, ValueError):
            normalized_sets = {tuple(str(value) for value in raw_values) for raw_values in value_lists}
        if len(normalized_sets) != 1:
            continue
        merged_values = list(next(iter(normalized_sets)))
        values = _phase3_2_format_sequence(merged_values)
        if not values:
            continue
        pattern = rf"({re.escape(param)}\s*\\in\s*\\\{{)[^}}]*(\\\}})"
        updated = re.sub(pattern, rf"\g<1>{values}\g<2>", updated)
    return updated


def validate_phase3_2_paper_evidence_gate(run_dir: Path) -> dict[str, Any]:
    """Require conservative numerical prose unless Phase 2.5 evidence is explicitly paper-ready."""
    run_dir = Path(run_dir)
    phase25_dir = run_dir / "phase2-5"
    errors: list[str] = []
    warnings: list[str] = []
    phase25_summary = read_json(phase25_dir / "phase25_experiment_summary.json") or {}
    if not isinstance(phase25_summary, dict) or not phase25_summary:
        return {
            "ok": False,
            "errors": ["phase2-5/phase25_experiment_summary.json is missing or invalid."],
            "warnings": [],
            "allowed_statuses": sorted(PHASE25_PAPER_READY_STATUSES),
        }

    phase25_status = str(phase25_summary.get("phase25_status") or "").strip()
    if phase25_status not in PHASE25_PAPER_READY_STATUSES:
        errors.append(
            f"Phase 2.5 status `{phase25_status or 'unknown'}` is not paper-ready; rerun/redesign experiments before Phase 3.2."
        )
    if bool(phase25_summary.get("generated_figures_are_draft_only")):
        errors.append("Phase 2.5 marks generated_figures_are_draft_only=true; final numerical-results prose requires review.")
    if not bool(phase25_summary.get("paper_minimum_ready")):
        errors.append("Phase 2.5 paper_minimum_ready is false; final paper sections must not be drafted.")

    figures = phase25_summary.get("figures", [])
    if not isinstance(figures, list) or not figures:
        errors.append("Phase 2.5 did not register any figures for the paper.")
    else:
        draft_figures = [
            str(figure.get("figure_id") or figure.get("id") or index)
            for index, figure in enumerate(figures, start=1)
            if isinstance(figure, dict)
            and (not bool(figure.get("paper_ready")) or str(figure.get("draft_or_final") or "").lower() == "draft")
        ]
        if draft_figures:
            errors.append("Phase 2.5 contains draft/non-paper-ready figures: " + ", ".join(draft_figures))

    registry = read_json(phase25_dir / "phase25_verified_registry.json") or {}
    if not isinstance(registry, dict) or registry.get("status") != "verified_experiment_registry":
        errors.append("phase2-5/phase25_verified_registry.json is missing or not a verified_experiment_registry.")
    elif str(registry.get("phase25_status") or "").strip() != phase25_status:
        warnings.append("Verified registry phase25_status differs from phase25_experiment_summary.json.")

    plot_quality = read_json(phase25_dir / "plot_quality_report.json") or {}
    if isinstance(plot_quality, dict):
        plot_status = str(plot_quality.get("overall_status") or "").strip()
        if plot_status and plot_status not in PHASE25_PAPER_READY_STATUSES:
            errors.append(f"plot_quality_report overall_status `{plot_status}` is not paper-ready.")

    allow_draft_continue = str(os.environ.get("WCL_ALLOW_DRAFT_PHASE25_CONTINUE", "")).strip().lower() in {
        "1",
        "true",
        "yes",
    }
    if allow_draft_continue and errors:
        warnings.extend(
            [
                "WCL_ALLOW_DRAFT_PHASE25_CONTINUE=1 is set; Phase 3.2 is allowed to draft from quick/draft Phase 2.5 evidence for development only.",
                *errors,
            ]
        )
        errors = []

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "phase25_status": phase25_status,
        "allowed_statuses": sorted(PHASE25_PAPER_READY_STATUSES),
        "generated_figures_are_draft_only": bool(phase25_summary.get("generated_figures_are_draft_only")),
        "paper_ready_figures": phase25_summary.get("paper_ready_figures", []),
    }


def _read_phase3_2_curve_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for raw in csv.DictReader(read_text(path).splitlines()):
        row: dict[str, Any] = {}
        for key, value in raw.items():
            key_text = str(key or "").strip()
            value_text = str(value or "").strip()
            try:
                numeric = float(value_text)
                row[key_text] = numeric if math.isfinite(numeric) else value_text
            except ValueError:
                row[key_text] = value_text
        rows.append(row)
    return rows


def _phase3_2_curve_endpoint(rows: list[dict[str, Any]], method: str) -> tuple[dict[str, Any], dict[str, Any]]:
    method_rows = [row for row in rows if str(row.get("method", "")).strip() == method]
    method_rows.sort(key=lambda row: float(row.get("x_value", row.get("swept_value", 0.0)) or 0.0))
    if not method_rows:
        return {}, {}
    return method_rows[0], method_rows[-1]


def _phase3_2_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _phase3_2_relative_gain(proposed_value: Any, baseline_value: Any) -> float | None:
    proposed_numeric = _phase3_2_float(proposed_value)
    baseline_numeric = _phase3_2_float(baseline_value)
    if proposed_numeric is None or baseline_numeric is None:
        return None
    denominator = abs(baseline_numeric) if abs(baseline_numeric) > 1.0e-12 else 1.0
    return (proposed_numeric - baseline_numeric) / denominator


def _build_phase3_2_curve_evidence_summary(
    run_dir: Path,
    phase25_summary: dict[str, Any],
    experiment_plan: dict[str, Any],
) -> dict[str, Any]:
    phase25_dir = run_dir / "phase2-5"
    figures = [item for item in phase25_summary.get("figures", []) if isinstance(item, dict)]
    method_names: dict[str, str] = {}
    for item in experiment_plan.get("compared_methods", []) if isinstance(experiment_plan, dict) else []:
        if not isinstance(item, dict):
            continue
        method_id = str(item.get("internal_name") or item.get("id") or item.get("name") or "").strip()
        if method_id:
            method_names[method_id] = _phase3_2_safe_method_text(item.get("display_name_short"), method_id)
    figure_methods = [
        str(method)
        for figure in figures
        for method in (figure.get("methods", []) if isinstance(figure.get("methods", []), list) else [])
        if str(method).strip()
    ]
    proposed = str((phase25_summary.get("primary_claim_check") or {}).get("proposed_method") or "").strip()
    if not proposed or proposed not in set(figure_methods):
        proposed = "proposed" if "proposed" in set(figure_methods) else (figure_methods[0] if figure_methods else "proposed")
    baseline = str((phase25_summary.get("primary_claim_check") or {}).get("baseline_method") or "").strip()
    if not baseline or baseline not in set(figure_methods):
        baseline = next((method for method in figure_methods if method != proposed), "")
    evidence_figures: list[dict[str, Any]] = []

    def method_label(method_id: str) -> str:
        return method_names.get(method_id, method_id).replace("_", "-")

    for figure in figures[:2]:
        fig_id = str(figure.get("figure_id") or "")
        if not fig_id:
            continue
        rows = _read_phase3_2_curve_rows(phase25_dir / "figures" / f"{fig_id}_curve_data.csv")
        methods = [str(method) for method in figure.get("methods", []) if str(method)]
        ablations = [method for method in methods if method not in {proposed, baseline}]
        x_values = sorted({_phase3_2_float(row.get("x_value", row.get("swept_value"))) for row in rows if _phase3_2_float(row.get("x_value", row.get("swept_value"))) is not None})

        def value_at(method_id: str, x_value: float) -> float | None:
            for row in rows:
                row_x = _phase3_2_float(row.get("x_value", row.get("swept_value")))
                if row_x is not None and abs(row_x - x_value) < 1.0e-9 and str(row.get("method", "")) == method_id:
                    return _phase3_2_float(row.get("mean_metric"))
            return None

        def pair_summary(target_method: str) -> dict[str, Any]:
            points: list[dict[str, Any]] = []
            gains: list[float] = []
            win_x: list[float] = []
            lose_x: list[float] = []
            for x_value in x_values:
                proposed_value = value_at(proposed, x_value)
                target_value = value_at(target_method, x_value)
                rel_gain = _phase3_2_relative_gain(proposed_value, target_value)
                if proposed_value is None or target_value is None or rel_gain is None:
                    continue
                gains.append(rel_gain)
                if proposed_value >= target_value:
                    win_x.append(x_value)
                else:
                    lose_x.append(x_value)
                points.append(
                    {
                        "x": x_value,
                        "proposed": proposed_value,
                        "target": target_value,
                        "relative_gain_percent": 100.0 * rel_gain,
                    }
                )
            best = max(points, key=lambda item: item["relative_gain_percent"], default={})
            last = points[-1] if points else {}
            first_positive = next((item["x"] for item in points if item["relative_gain_percent"] >= 0), "")
            return {
                "target_method": target_method,
                "target_label": method_label(target_method),
                "win_x_values": win_x,
                "lose_x_values": lose_x,
                "wins": len(win_x),
                "total_points": len(points),
                "first_nonnegative_gain_x": first_positive,
                "best_point": best,
                "last_point": last,
                "pointwise": points,
            }

        proposed_series = [
            {"x": x_value, "value": value_at(proposed, x_value)}
            for x_value in x_values
            if value_at(proposed, x_value) is not None
        ]
        trend = ""
        if len(proposed_series) >= 2:
            first_value = _phase3_2_float(proposed_series[0].get("value"))
            last_value = _phase3_2_float(proposed_series[-1].get("value"))
            if first_value is not None and last_value is not None:
                trend = "increases" if last_value > first_value else "decreases" if last_value < first_value else "is approximately flat"
        evidence_figures.append(
            {
                "figure_id": fig_id,
                "label": "Fig. 1" if fig_id == "figure_1" else "Fig. 2" if fig_id == "figure_2" else fig_id,
                "x_axis_label": figure.get("x_axis_label", ""),
                "y_axis_label": figure.get("y_axis_label", ""),
                "x_values": x_values,
                "proposed_method": proposed,
                "proposed_label": method_label(proposed),
                "primary_benchmark": baseline,
                "primary_benchmark_label": method_label(baseline),
                "ablation_methods": [{"method": method, "label": method_label(method)} for method in ablations],
                "proposed_trend": trend,
                "proposed_series": proposed_series,
                "comparison_to_primary_benchmark": pair_summary(baseline) if baseline else {},
                "comparison_to_first_ablation": pair_summary(ablations[0]) if ablations else {},
            }
        )

    return {
        "metric": phase25_summary.get("primary_metric", {}),
        "claim_check": phase25_summary.get("primary_claim_check", {}),
        "figures": evidence_figures,
    }


def build_phase3_2_figure_evidence_summary_json(run_dir: Path) -> str:
    phase25_dir = Path(run_dir) / "phase2-5"
    phase25_summary = read_json(phase25_dir / "phase25_experiment_summary.json") or {}
    experiment_plan = read_json(phase25_dir / "experiment_plan.json") or {}
    if not isinstance(phase25_summary, dict):
        phase25_summary = {}
    if not isinstance(experiment_plan, dict):
        experiment_plan = {}
    return json.dumps(_build_phase3_2_curve_evidence_summary(Path(run_dir), phase25_summary, experiment_plan), ensure_ascii=False, indent=2)


def call_llm_phase3_2_numerical_results_writer(
    *,
    run_dir: Path,
    topic: str,
    system_model_md: str,
    problem_formulation_md: str,
    algorithm_md: str,
    benchmark_definition_md: str,
    model_profile: str,
) -> str:
    llm = create_llm_client(model_profile)
    phase3_2_dir = run_dir / "phase3-2"
    phase25_dir = run_dir / "phase2-5"
    phase25_summary_json = read_text(phase25_dir / "phase25_experiment_summary.json")
    verified_registry_json = read_text(phase25_dir / "phase25_verified_registry.json")
    figure_captions_md = read_text(phase25_dir / "figure_captions.md")
    method_naming_summary_json = read_text(phase25_dir / "method_naming_summary.json")
    experiment_plan_json = read_text(phase25_dir / "experiment_plan.json")
    plot_quality_report_json = read_text(phase25_dir / "plot_quality_report.json")
    reformulation_path_md = read_text(run_dir / "phase2-2" / "reformulation_path.md")
    phase25_summary = _safe_json_loads(phase25_summary_json, {}) if phase25_summary_json.strip() else {}
    filtered_method_naming_summary_json = filter_phase3_2_method_naming_summary_for_plotted_methods(
        method_naming_summary_json,
        phase25_summary if isinstance(phase25_summary, dict) else {},
    )
    filtered_benchmark_definition_md = filter_phase3_2_benchmark_definition_for_plotted_methods(
        benchmark_definition_md,
        method_naming_summary_json,
        phase25_summary if isinstance(phase25_summary, dict) else {},
    )
    simulation_setup_facts = build_phase3_2_simulation_setup_facts(run_dir)
    simulation_setup_facts_json = json.dumps(simulation_setup_facts, ensure_ascii=False, indent=2)
    figure_evidence_summary_json = build_phase3_2_figure_evidence_summary_json(run_dir)
    analysis_agent_request_json = build_role_agent_request_json(
        run_dir,
        "analysis_agent",
        event="phase3_2_prompt",
        max_chars=7000,
    )
    write_text(phase3_2_dir / "simulation_setup_facts.json", simulation_setup_facts_json)
    write_text(phase3_2_dir / "figure_evidence_summary.json", figure_evidence_summary_json)
    write_text(phase3_2_dir / "analysis_agent_request_excerpt.json", analysis_agent_request_json)
    write_text(phase3_2_dir / "plotted_method_naming_summary.json", filtered_method_naming_summary_json)
    prompt = build_phase3_2_numerical_results_prompt(
        topic=topic,
        system_model_md=system_model_md,
        problem_formulation_md=problem_formulation_md,
        algorithm_md=algorithm_md,
        benchmark_definition_md=filtered_benchmark_definition_md,
        method_mechanism_summary=build_phase3_2_method_mechanism_summary(
            algorithm_md=algorithm_md,
            benchmark_definition_md=filtered_benchmark_definition_md,
            method_naming_summary_json=filtered_method_naming_summary_json,
        ),
        paper_objective_summary=build_phase3_2_paper_objective_summary(
            problem_formulation_md=problem_formulation_md,
            reformulation_path_md=reformulation_path_md,
            algorithm_md=algorithm_md,
            benchmark_definition_md=benchmark_definition_md,
            experiment_plan_json=experiment_plan_json,
        ),
        figure_to_claim_summary=build_phase3_2_figure_to_claim_summary(
            experiment_plan_json=experiment_plan_json,
            plot_quality_report_json=plot_quality_report_json,
        ),
        figure_observations_summary=build_phase3_2_figure_observations_summary(phase25_summary if isinstance(phase25_summary, dict) else {}),
        figure_evidence_summary_json=figure_evidence_summary_json,
        claim_constraints_summary=build_phase3_2_claim_constraints_summary(phase25_summary if isinstance(phase25_summary, dict) else {}),
        simulation_setup_facts_json=simulation_setup_facts_json,
        phase25_summary_json=phase25_summary_json,
        verified_registry_json=verified_registry_json,
        figure_captions_md=figure_captions_md,
        analysis_agent_request_json=analysis_agent_request_json,
    )
    write_text(phase3_2_dir / "phase3_2_prompt.txt", prompt)
    thinking = {"type": "enabled"} if model_profile == "kimi-k2.6-thinking" else None
    response = llm.chat(
        [{"role": "user", "content": prompt}],
        json_mode=True,
        thinking=thinking,
        max_tokens=int(os.environ.get("WARA_PHASE31_MAX_TOKENS", "9000")),
    )
    write_text(phase3_2_dir / "phase3_2_raw_response.txt", response.content)
    if not str(response.content or "").strip():
        raise ValueError("Phase 3.2 LLM returned an empty response")
    payload = _safe_json_loads(response.content, {})
    if not isinstance(payload, dict):
        raise ValueError("Phase 3.2 call did not return a valid structured object")
    numerical_results_tex = str(payload.get("numerical_results_tex") or "").strip()
    if len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", numerical_results_tex)) < 120:
        raise ValueError("Phase 3.2 LLM returned an empty or incomplete numerical_results_tex")
    return numerical_results_tex
