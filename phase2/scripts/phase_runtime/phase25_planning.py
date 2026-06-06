from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path
from typing import Any

import yaml

from pipeline_core import compact_text, read_json, read_text, write_text
from pipeline_core.json_utils import _safe_json_loads
from phase_runtime.llm import create_llm_client
from phase_runtime.prompt_templates import render_prompt_template
from phase_runtime.phase24_plan import (
    _phase24_normalize_method_entry,
    _phase24_publication_metric_for_target,
    _phase24_select_evidence_targets,
    summarize_validation_plan,
)


def build_phase25_design_notes() -> str:
    return """
# Phase 2.5 Design Notes

## Phase 2.5 mission

Phase 2.5 is the paper-facing experiment packaging step.

It:
- reads Phase 2.3 theory outputs for the proposed algorithm
- reads the Phase 2.4 evidence contract and quick validation outputs
- chooses renderings for the predeclared evidence needs
- checks whether the current data are sufficient for paper-ready claims
- produces draft figures or paper-ready figures depending on the quality gate
- writes missing-experiment plans when the data are not yet sufficient

## Correct boundary with Phase 2.3

Phase 2.3 freezes the algorithm route, benchmark requirements, validation principles, and claim map. It does not choose concrete sweep grids, Monte Carlo settings, final plotting details, tables, or paper-ready sample sizes.

Phase 2.5 is responsible for:
- `experiment_plan.json`
- figure rendering and layout
- contract-preserving chart normalization
- paper-ready quality gates
- Monte Carlo sufficiency checks
- `paper_sweep_plan.json`

Phase 2.5 must not:
- invent a new claim after seeing weak data
- choose a new primary KPI, benchmark family, or x-axis family after code generation
- swap to easier benchmarks after seeing results
- convert a failed experiment into a different paper story

## Correct boundary with Phase 2.4

Phase 2.4 owns the experiment contract and quick validation. Its `validation_plan.yaml` declares the compared methods, required metrics, sweeps, and figure evidence candidates before Phase 2.5 sees the data.

Phase 2.5 decides whether those quick results are enough for a paper figure. If not, Phase 2.5 produces a paper sweep plan for denser values/seeds within the same predeclared claim family, or routes back to Phase 2.4 experiment-design repair when the KPI/benchmark/sweep family itself is wrong.

## Division of labor

LLM responsibilities in Phase 2.5:
- package and interpret the frozen Phase 2.4 evidence contract
- assess whether predeclared figures have enough x-axis coverage, seeds, paired successful rows, and stable trends
- request denser values/seeds within the same Phase 2.4 claim family when needed
- route back to Phase 2.4 experiment-design repair when KPI, benchmark family, or sweep family is wrong
- write WCL-style result interpretation

Deterministic Python responsibilities in Phase 2.5:
- load and normalize validation data
- aggregate Monte Carlo statistics
- check data sufficiency
- generate draft/final figures
- keep tables optional and disabled for the current WARA route
- compute win rate, relative gain, feasibility, and finite-value checks
- write `paper_sweep_plan.json` when more runs are needed

## Default publication target

Phase 2.5 treats paper preferred as the default target.

Typical defaults:
- quick/scout mode: 20 seeds per point on a small x-axis grid, draft only
- medium mode: 50 seeds per point for curve-shape confirmation
- paper minimum: 80 seeds per point
- paper preferred: 100 seeds per point
- high confidence: 150 seeds per point

For line plots, paper preferred typically means about 12 x-axis points. For grouped bars, paper preferred usually means 5 to 6 representative categories with sufficient averaging.

## Status logic

Phase 2.5 may end in:
- `quick_mode_only`
- `needs_more_phase24_runs`
- `paper_minimum_ready`
- `paper_preferred_ready`
- `high_confidence_ready`

Quick mode never produces paper-ready figures.
""".strip()


def build_phase25_experiment_planner_prompt(
    *,
    topic: str,
    validation_principles_summary: str,
    system_model_md: str,
    problem_formulation_md: str,
    reformulation_path_md: str,
    algorithm_md: str,
    benchmark_definition_md: str,
    experiment_blueprint_md: str,
    validation_plan_summary: str,
    available_data_summary_json: str,
    phase24_summary_excerpt_json: str,
) -> str:
    return render_prompt_template(
        "phase2_5/experiment_planner.prompt.yaml",
        topic=topic,
        validation_principles_summary=validation_principles_summary,
        system_model_md=system_model_md,
        problem_formulation_md=problem_formulation_md,
        reformulation_path_md=reformulation_path_md,
        algorithm_md=algorithm_md,
        benchmark_definition_md=benchmark_definition_md,
        phase24_evidence_contract_md=experiment_blueprint_md,
        validation_plan_summary=validation_plan_summary,
        available_data_summary_json=available_data_summary_json,
        phase24_summary_excerpt_json=phase24_summary_excerpt_json,
    )


def _extract_first_heading_after(prefix: str, text: str) -> str:
    pattern = re.compile(rf"{re.escape(prefix)}\s*(.+)")
    match = pattern.search(text)
    return match.group(1).strip() if match else ""


def _infer_proposed_acronym(algorithm_md: str) -> str:
    text = algorithm_md.lower()
    tags: list[str] = []
    if "wmmse" in text:
        tags.append("WMMSE")
    if "sca" in text:
        tags.append("SCA")
    if "ao" in text or "alternating optimization" in text:
        tags.append("AO")
    if "bcd" in text:
        tags.append("BCD")
    if "sdr" in text:
        tags.append("SDR")
    deduped: list[str] = []
    for tag in tags:
        if tag not in deduped:
            deduped.append(tag)
    return "-".join(deduped[:2]) if deduped else "Proposed"


def _parse_benchmark_headings(benchmark_definition_md: str) -> list[str]:
    headings: list[str] = []
    for line in benchmark_definition_md.splitlines():
        stripped = line.strip()
        match = re.match(r"^#{2,6}\s*(?:B\d+:\s*)?(.+)$", stripped)
        if match and "benchmark definitions" not in stripped.lower() and "comparison metrics" not in stripped.lower():
            headings.append(match.group(1).strip())
    return headings


def enrich_phase25_method_naming(
    plan: dict[str, Any],
    *,
    algorithm_md: str,
    benchmark_definition_md: str,
) -> dict[str, Any]:
    payload = dict(plan)
    compared_methods = list(payload.get("compared_methods", []))
    benchmark_headings = _parse_benchmark_headings(benchmark_definition_md)
    proposed_heading = _extract_first_heading_after("## Proposed Algorithm:", algorithm_md) or _extract_first_heading_after("# Proposed Algorithm:", algorithm_md)
    proposed_acronym = _infer_proposed_acronym(algorithm_md)
    baseline_heading = benchmark_headings[0] if benchmark_headings else ""
    enriched_methods: list[dict[str, Any]] = []
    for item in compared_methods:
        method = dict(item)
        internal_name = str(method.get("name", "method"))
        role = str(method.get("role", ""))
        short_name = str(method.get("display_name_short", "")).strip()
        long_name = str(method.get("display_name_long", "")).strip()
        source_of_name = str(method.get("source_of_name", "")).strip()
        if not short_name or not long_name:
            if role == "proposed" or internal_name == "proposed":
                long_name = long_name or (proposed_heading if proposed_heading else "Proposed optimization method")
                short_name = short_name or (f"Proposed ({proposed_acronym})" if proposed_acronym != "Proposed" else "Proposed method")
                source_of_name = source_of_name or "algorithm_md"
            elif role == "main_baseline" or internal_name == "baseline":
                long_name = long_name or (baseline_heading if baseline_heading else "Main benchmark method")
                if not short_name:
                    if baseline_heading:
                        short_name = re.sub(r"\s*\(.*?\)\s*", "", baseline_heading).strip()
                    else:
                        short_name = "Main benchmark"
                source_of_name = source_of_name or "benchmark_definition_md"
            elif "ablation" in role or "ablation" in internal_name:
                long_name = long_name or f"Ablation variant: {internal_name.replace('_', ' ')}"
                short_name = short_name or internal_name.replace("_", "-")
                source_of_name = source_of_name or "auto_generated"
            else:
                long_name = long_name or internal_name.replace("_", " ").strip().title()
                short_name = short_name or long_name
                source_of_name = source_of_name or "auto_generated"
        method["internal_name"] = internal_name
        method["display_name_short"] = short_name
        method["display_name_long"] = long_name
        method["source_of_name"] = source_of_name or "auto_generated"
        enriched_methods.append(method)
    payload["compared_methods"] = enriched_methods
    return payload


def enforce_phase25_actual_method_names(plan: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    payload = copy.deepcopy(plan)
    methods = payload.get("compared_methods", [])
    if not isinstance(methods, list):
        return payload
    for method in methods:
        if not isinstance(method, dict):
            continue
        internal_name = str(method.get("internal_name") or method.get("name", "")).strip().lower()
        role = str(method.get("role", "")).strip().lower()
        if internal_name == "proposed" or role == "proposed":
            if "display_name_short" not in method or not str(method.get("display_name_short", "")).strip():
                method["display_name_short"] = "Proposed"
            if "display_name_long" not in method or not str(method.get("display_name_long", "")).strip():
                method["display_name_long"] = "Proposed optimization method"
        if internal_name == "baseline" or role == "main_baseline":
            if "display_name_short" not in method or not str(method.get("display_name_short", "")).strip():
                method["display_name_short"] = "Main benchmark"
            if "display_name_long" not in method or not str(method.get("display_name_long", "")).strip():
                method["display_name_long"] = "Main fair benchmark implemented by Phase 2.4 baseline_solution"
            method.setdefault("source_of_name", "phase24_contract")
    payload["compared_methods"] = methods
    return payload


def validate_phase25_experiment_plan(plan: dict[str, Any]) -> None:
    if not isinstance(plan, dict):
        raise ValueError("experiment_plan is not a dict")
    required_top = ["paper_target", "primary_metric", "compared_methods", "figure_specs", "table_specs", "data_sufficiency_rules"]
    for key in required_top:
        if key not in plan:
            raise ValueError(f"experiment_plan missing required key: {key}")
    if not isinstance(plan.get("figure_specs"), list) or len(plan["figure_specs"]) < 2:
        raise ValueError("experiment_plan must contain at least two figure_specs")
    if len(plan["figure_specs"]) > 3:
        raise ValueError("experiment_plan must contain at most three figure_specs")
    if not isinstance(plan.get("table_specs"), list):
        raise ValueError("experiment_plan table_specs must be a list; use [] when no table is needed")
    for item in plan.get("compared_methods", []):
        if "name" not in item or "role" not in item:
            raise ValueError("compared_methods entries must contain name and role")
        if "internal_name" not in item or "display_name_short" not in item or "display_name_long" not in item:
            raise ValueError("compared_methods entries must contain internal_name, display_name_short, and display_name_long")
    allowed_chart_types = {"line", "grouped_bar", "bar", "box", "convergence", "heatmap", "scatter", "categorical_summary", "ablation_bar"}
    for fig in plan.get("figure_specs", []):
        chart_type = str(fig.get("chart_type", "")).strip()
        if chart_type not in allowed_chart_types:
            raise ValueError(f"unsupported chart_type in figure_spec: {chart_type}")
        if "metric" not in fig or "encoding" not in fig:
            raise ValueError("figure_specs must contain metric and encoding")
        if not str(fig.get("chart_choice_rationale", "")).strip():
            raise ValueError("figure_specs must contain chart_choice_rationale")
    for table in plan.get("table_specs", []):
        if not str(table.get("table_design_rationale", "")).strip():
            raise ValueError("table_specs must contain table_design_rationale")


def de_template_phase25_experiment_plan(plan: dict[str, Any], available_data: dict[str, Any]) -> dict[str, Any]:
    """Repair common prompt-copying patterns while staying schema/data driven."""
    if not isinstance(plan, dict):
        return plan
    payload = copy.deepcopy(plan)
    figures = payload.get("figure_specs", [])
    if not isinstance(figures, list):
        return payload
    seed_column = str(available_data.get("seed_column", "")).strip()
    numeric_metrics = set(str(x) for x in available_data.get("numeric_metrics", []) if str(x).strip())
    grouping_keys = set(str(x) for x in available_data.get("possible_grouping_keys", []) if str(x).strip())

    for fig in figures:
        if not isinstance(fig, dict):
            continue
        chart_type = str(fig.get("chart_type", "")).strip()
        purpose = str(fig.get("purpose", "")).lower()
        message = str(fig.get("primary_message", "")).lower()
        x_info = fig.get("encoding", {}).get("x", {}) if isinstance(fig.get("encoding", {}), dict) else {}
        sweep_param = str(x_info.get("sweep_param", ""))
        if not str(fig.get("chart_choice_rationale", "")).strip():
            fig["chart_choice_rationale"] = (
                "Chosen from the current claim, sweep semantics, available metrics, and Monte Carlo coverage rather "
                "than copied from a fixed plotting template."
            )
        if chart_type == "heatmap":
            facet = fig.get("encoding", {}).get("facet", {}) if isinstance(fig.get("encoding", {}), dict) else {}
            facet_field = str(facet.get("field", "") if isinstance(facet, dict) else "").strip()
            if not facet_field or facet_field.lower() == "none" or facet_field not in grouping_keys:
                fig["chart_type"] = "box" if seed_column else "scatter"
                fig["error_display"] = "none"
                fig["chart_choice_rationale"] = (
                    "The requested heatmap was downgraded because the available data do not contain a genuine second "
                    "grid/facet variable. The replacement chart uses the actual one-dimensional sweep and Monte Carlo "
                    "coverage without inventing a two-dimensional surface."
                )
                chart_type = str(fig.get("chart_type", "")).strip()
        if chart_type == "line" and any(token in purpose + " " + message for token in ["tradeoff", "pareto", "regime"]):
            fig["chart_type"] = "scatter"
            fig["chart_choice_rationale"] = (
                "A scatter view is used because the claim concerns a tradeoff or operating regime rather "
                "than a guaranteed monotone scaling law; connecting points as a curve would overstate continuity."
            )
        if chart_type in {"grouped_bar", "bar", "categorical_summary"} and seed_column and sweep_param:
            metric_name = str((fig.get("metric") or {}).get("name", ""))
            if metric_name in numeric_metrics or metric_name == str((payload.get("primary_metric") or {}).get("name", "")):
                fig["chart_type"] = "box"
                fig["error_display"] = "none"
                fig["chart_choice_rationale"] = (
                    "A box plot is used because repeated Monte Carlo realizations are available; it shows distribution "
                    "and stability across the selected regimes better than a plain grouped bar."
                )
                req = fig.get("data_requirements", {})
                if isinstance(req, dict):
                    req["min_samples_per_group"] = max(int(req.get("min_samples_per_group", 30) or 30), 30)
                    req["preferred_samples_per_group"] = max(int(req.get("preferred_samples_per_group", 50) or 50), 50)
                    fig["data_requirements"] = req

    tables = payload.get("table_specs", [])
    if isinstance(tables, list):
        for table in tables:
            if not isinstance(table, dict):
                continue
            if not str(table.get("table_design_rationale", "")).strip():
                table["table_design_rationale"] = (
                    "The table is claim-focused: it keeps only the primary improvement, feasibility, and topic-specific "
                    "secondary metrics that explain the mechanism."
                )
            if str(table.get("purpose", "")).strip().lower() in {"overall performance summary", "domain-specific performance summary with eh and sensing metrics"}:
                table["purpose"] = "claim-focused numerical evidence summary"
            if str(table.get("row_selection", "")).strip() == "representative_scenarios_or_sweep_groups":
                table["row_selection"] = "claim_relevant_regimes_or_sweep_groups"
            group_by = str(table.get("group_by", "")).strip()
            if group_by and group_by not in grouping_keys:
                if "swept_param" in grouping_keys:
                    table["group_by"] = "swept_param"
                elif "scenario_name" in grouping_keys:
                    table["group_by"] = "scenario_name"
                else:
                    table["group_by"] = ""
    return payload


def align_phase25_plan_with_phase24_contract(plan: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """Keep Phase 2.5 figures tied to the executable Phase 2.4 contract."""
    if not isinstance(plan, dict):
        return plan
    payload = copy.deepcopy(plan)
    try:
        validation_payload = yaml.safe_load(read_text(Path(run_dir) / "phase2-4" / "validation_plan.yaml")) or {}
    except Exception:
        validation_payload = {}
    if not isinstance(validation_payload, dict):
        return payload
    evidence = validation_payload.get("research_evidence_contract", {})
    if not isinstance(evidence, dict) or not evidence:
        evidence = validation_payload.get("paper_evidence_contract", {})
    if not isinstance(evidence, dict):
        return payload
    contract_figures = evidence.get("figures", [])
    raw_figure_targets = validation_payload.get("figure_targets", [])
    raw_targets = [item for item in raw_figure_targets if isinstance(item, dict)] if isinstance(raw_figure_targets, list) else []
    if isinstance(contract_figures, list) and len([item for item in contract_figures if isinstance(item, dict)]) >= 2:
        contract_figures = [item for item in contract_figures if isinstance(item, dict)][:3]
    else:
        candidate_figures: list[Any] = []
        if isinstance(raw_figure_targets, list):
            candidate_figures.extend(raw_figure_targets)
        if isinstance(contract_figures, list):
            candidate_figures.extend(contract_figures)
        contract_figures = _phase24_select_evidence_targets(candidate_figures, limit=3)
    if not isinstance(contract_figures, list) or len(contract_figures) < 2:
        return payload

    methods: list[dict[str, Any]] = []
    seen_methods: set[str] = set()

    def add_phase25_method(item: Any, fallback_id: str, role: str, source: str) -> None:
        method = _phase24_normalize_method_entry(item, fallback_id, role)
        method_id = str(method.get("id") or method.get("internal_name") or method.get("name") or "").strip()
        if not method_id or method_id in seen_methods:
            return
        method["internal_name"] = method_id
        method["name"] = method_id
        method["source_of_name"] = source
        methods.append(method)
        seen_methods.add(method_id)

    for idx, item in enumerate(evidence.get("compared_methods", []) if isinstance(evidence.get("compared_methods"), list) else []):
        add_phase25_method(item, f"contract_method_{idx + 1}", "comparison", "phase24_contract")
    for idx, item in enumerate(payload.get("compared_methods", []) if isinstance(payload.get("compared_methods"), list) else []):
        add_phase25_method(item, f"planner_method_{idx + 1}", "comparison", "phase25_plan")
    if "proposed" not in seen_methods:
        add_phase25_method(
            {"id": "proposed", "display_name_short": "Proposed method", "display_name_long": "Proposed optimization method"},
            "proposed",
            "proposed",
            "phase24_contract_default",
        )
    has_non_proposed_method = any(method_id != "proposed" for method_id in seen_methods)
    if "baseline" not in seen_methods and not has_non_proposed_method:
        add_phase25_method(
            {"id": "baseline", "display_name_short": "Main benchmark", "display_name_long": "Implemented Phase 2.4 benchmark"},
            "baseline",
            "main_baseline",
            "phase24_contract_default",
        )
    payload["compared_methods"] = methods
    method_role_by_id = {
        str(item.get("internal_name") or item.get("name") or item.get("id")): str(item.get("role", "")).lower()
        for item in methods
        if isinstance(item, dict)
    }

    scalar_metrics = []
    required_outputs = validation_payload.get("required_outputs", {})
    if isinstance(required_outputs, dict) and isinstance(required_outputs.get("scalar_metrics"), list):
        scalar_metrics = [str(item) for item in required_outputs["scalar_metrics"] if str(item).strip()]

    objective_sense = str(validation_payload.get("objective_sense") or "maximize").strip().lower()

    def metric_display(metric: str) -> str:
        metric_lower = str(metric or "").strip().lower()
        labels = {
            "objective": "Weighted objective",
            "objective_value": "Weighted objective",
            "crb_trace": "CRB trace",
            "rate_Mbps": "Rate (Mbps)",
            "sum_rate": "Sum rate (bps/Hz)",
            "sum_rate_bpsHz": "Sum rate (bps/Hz)",
            "sum_rate_bps_hz": "Sum rate (bps/Hz)",
            "radar_snr_dB": "Radar SNR (dB)",
            "radar_SNR_dB": "Radar SNR (dB)",
            "eh_total_mW": "Harvested power (mW)",
            "harvested_energy": "Harvested energy (mW)",
            "harvested_energy_mW": "Harvested energy (mW)",
            "true_harvested_energy_mW": "True harvested energy (mW)",
            "feasible": "Feasibility rate",
            "feasibility_rate": "Feasibility rate",
            "max_constraint_violation": "Max constraint violation",
            "constraint_violation_max": "Max constraint violation",
            "optimal_rho": "Structural separation ratio",
            "rho": "Structural separation ratio",
        }
        return labels.get(metric, labels.get(metric_lower, metric.replace("_", " ")))

    def metric_higher_is_better(metric: str) -> bool:
        lowered = str(metric or "").strip().lower()
        if any(token in lowered for token in ("power", "violation", "outage", "gap", "error", "mse", "crb", "runtime", "time", "iteration")):
            return False
        if lowered in {"objective", "objective_value", "weighted_objective", "utility", "weighted_utility"}:
            return objective_sense != "minimize"
        return True

    def metric_from_planner_figure(fig: dict[str, Any]) -> str:
        metric_payload = fig.get("metric") if isinstance(fig, dict) else None
        if isinstance(metric_payload, dict):
            return str(metric_payload.get("name") or "").strip()
        return str(fig.get("y_metric") or fig.get("metric") or "").strip() if isinstance(fig, dict) else ""

    def diagnostic_metric(metric_name: str) -> bool:
        lowered = str(metric_name or "").strip().lower()
        return any(token in lowered for token in ("violation", "outage", "feasible", "feasibility", "constraint", "shortfall", "residual"))

    def prefer_planner_metric(contract_metric: str, planner_metric: str, contract: dict[str, Any]) -> bool:
        if not planner_metric or planner_metric == contract_metric:
            return False
        if scalar_metrics and planner_metric not in scalar_metrics:
            return False
        if diagnostic_metric(contract_metric) and not diagnostic_metric(planner_metric):
            return True
        evidence_text = " ".join(
            [
                str(contract.get("chart_intent") or contract.get("intent") or ""),
                str(contract.get("claim") or ""),
                str(contract.get("evidence_rationale") or ""),
                str(contract.get("why_alternatives_are_weaker") or ""),
                planner_metric,
            ]
        ).lower()
        mechanism_tokens = ("mechanism", "ablation", "sensitivity", "rho", "split", "harvest", "rate", "power", "resource")
        if any(token in evidence_text for token in mechanism_tokens) and not diagnostic_metric(planner_metric):
            return True
        return False

    existing = {
        str(fig.get("figure_id") or fig.get("id") or ""): fig
        for fig in payload.get("figure_specs", [])
        if isinstance(fig, dict)
    }
    sweep_defs = [
        sweep
        for sweep in (validation_payload.get("sweep_definitions", []) if isinstance(validation_payload.get("sweep_definitions"), list) else [])
        if isinstance(sweep, dict)
    ]
    used_sweep_ids: set[str] = set()

    def choose_contract_sweep(contract: dict[str, Any]) -> dict[str, Any]:
        required_sweep = str(contract.get("required_sweep") or "")
        x_field = str(contract.get("x_field") or contract.get("required_sweep_param") or "").split(".")[-1].lower()
        requested = [
            sweep
            for sweep in sweep_defs
            if str(sweep.get("id") or sweep.get("name") or "") in required_sweep
            or (x_field and str(sweep.get("variable") or sweep.get("target") or "").split(".")[-1].lower() == x_field)
        ]
        for sweep in requested:
            sweep_id = str(sweep.get("id") or sweep.get("name") or "")
            if sweep_id not in used_sweep_ids:
                used_sweep_ids.add(sweep_id)
                return sweep
        for sweep in sweep_defs:
            sweep_id = str(sweep.get("id") or sweep.get("name") or "")
            if sweep_id not in used_sweep_ids:
                used_sweep_ids.add(sweep_id)
                return sweep
        return {}

    def find_matching_sweep(contract: dict[str, Any], required_sweep_param: str) -> dict[str, Any]:
        required_sweep = str(contract.get("required_sweep") or "")
        param_norm = str(required_sweep_param or contract.get("x_field") or "").split(".")[-1].lower()
        for sweep in sweep_defs:
            sweep_id = str(sweep.get("id") or sweep.get("name") or "")
            variable = str(sweep.get("variable") or sweep.get("target") or "")
            if sweep_id and sweep_id in required_sweep:
                return sweep
            if param_norm and variable.split(".")[-1].lower() == param_norm:
                return sweep
        return {}

    def phase25_contract_chart_type(contract: dict[str, Any], raw_chart_type: str, chosen_sweep: dict[str, Any], metric: str) -> str:
        chart_type = str(raw_chart_type or "").strip().lower() or "line"
        if chart_type == "heatmap":
            chart_type = "box"
        raw_values = chosen_sweep.get("paper_mode_values", chosen_sweep.get("values", [])) if isinstance(chosen_sweep, dict) else []
        numeric_values = 0
        categorical_values = 0
        if isinstance(raw_values, list):
            for value in raw_values:
                try:
                    float(value)
                    numeric_values += 1
                except Exception:
                    categorical_values += 1
        intent = str(contract.get("chart_intent") or contract.get("intent") or "").lower()
        evidence_text = " ".join(
            [
                intent,
                str(contract.get("claim") or ""),
                str(contract.get("evidence_rationale") or ""),
                str(contract.get("chart_choice_rationale") or ""),
                str(metric or ""),
                str(chosen_sweep.get("variable") or "") if isinstance(chosen_sweep, dict) else "",
            ]
        ).lower()
        if categorical_values and not numeric_values:
            if "distribution" in evidence_text or "robustness" in evidence_text:
                return "box"
            return "grouped_bar"
        if numeric_values >= 4:
            if any(token in evidence_text for token in ("runtime", "scaling", "complexity", "snr", "power", "sensitivity", "tradeoff", "sweep")):
                return "line"
            if intent in {"main_comparison", "mechanism_ablation", "sensitivity", "scalability"}:
                return "line"
        if chart_type in {"line", "scatter", "grouped_bar", "bar", "box", "categorical_summary", "ablation_bar"}:
            return chart_type
        return "line" if numeric_values >= 4 else "grouped_bar"

    def concrete_methods_from_raw_targets(contract: dict[str, Any], current_methods: list[str]) -> list[str]:
        generic = {"proposed", "baseline"}
        if not current_methods or not any(method in generic for method in current_methods):
            return current_methods
        claim = str(contract.get("claim") or "").strip().lower()
        required_sweep = str(contract.get("required_sweep") or "").strip().lower()
        required_param = str(contract.get("required_sweep_param") or "").split(".")[-1].lower()
        best_methods: list[str] = []
        best_score = -1
        for raw in raw_targets:
            methods = raw.get("methods_to_run") or raw.get("methods") or raw.get("curves") or []
            if not isinstance(methods, list):
                continue
            concrete = [
                str(method.get("id") or method.get("internal_name") or method.get("name") if isinstance(method, dict) else method).strip()
                for method in methods
                if str(method.get("id") or method.get("internal_name") or method.get("name") if isinstance(method, dict) else method).strip()
            ]
            if not concrete or set(concrete).issubset(generic):
                continue
            raw_claim = str(raw.get("claim") or "").strip().lower()
            raw_sweep = str(raw.get("required_sweep") or raw.get("required_sweep_param") or raw.get("x_field") or "").strip().lower()
            score = 0
            if claim and raw_claim and (claim == raw_claim or claim in raw_claim or raw_claim in claim):
                score += 4
            if required_sweep and required_sweep in raw_sweep:
                score += 3
            if required_param and required_param in raw_sweep:
                score += 2
            if str(contract.get("chart_intent") or "").lower() == str(raw.get("chart_intent") or raw.get("intent") or "").lower():
                score += 1
            if score > best_score:
                best_score = score
                best_methods = concrete
        return best_methods if best_score > 0 else current_methods

    repaired_figures: list[dict[str, Any]] = []
    for idx, contract in enumerate(contract_figures[:3]):
        if not isinstance(contract, dict):
            continue
        figure_id = str(contract.get("figure_id") or contract.get("id") or f"figure_{idx + 1}")
        if not figure_id.startswith("figure_"):
            figure_id = f"figure_{idx + 1}"
        old = existing.get(figure_id, {})
        metric = str(contract.get("y_metric") or contract.get("metric") or "").strip()
        if not metric or (scalar_metrics and metric not in scalar_metrics):
            metric = next((item for item in ["objective_value", "objective", "rate_Mbps", "crb_trace", "eh_total_mW"] if item in scalar_metrics), "objective")
        metric = _phase24_publication_metric_for_target(contract, scalar_metrics, idx, {})
        planner_metric = metric_from_planner_figure(old)
        if prefer_planner_metric(metric, planner_metric, contract):
            metric = planner_metric
        chart_intent_lower = str(contract.get("chart_intent") or contract.get("intent") or "").lower()
        original_chart_type = str(contract.get("chart_type") or old.get("chart_type") or "line").strip()
        requested_methods = contract.get("methods_to_run", [])
        if isinstance(requested_methods, list) and requested_methods:
            figure_methods = [
                str(method.get("id") or method.get("internal_name") or method.get("name") if isinstance(method, dict) else method).strip()
                for method in requested_methods
                if str(method.get("id") or method.get("internal_name") or method.get("name") if isinstance(method, dict) else method).strip()
            ]
        else:
            figure_methods = [
                str(item.get("internal_name") or item.get("name"))
                for item in methods
                if isinstance(item, dict) and str(item.get("internal_name") or item.get("name", "")).strip()
            ]
        figure_methods = concrete_methods_from_raw_targets(contract, figure_methods)
        if metric.lower() in {"spectral_radius_f", "rho_f", "rho"}:
            figure_methods = ["proposed"]
        if any(token in metric.lower() for token in ("violation", "feasible", "outage")):
            filtered_methods = [
                method
                for method in figure_methods
                if not any(token in method_role_by_id.get(method, "") for token in ("upper", "oracle", "relax"))
            ]
            figure_methods = filtered_methods or figure_methods
        required_sweep_param = str(contract.get("required_sweep_param") or "").strip()
        chosen_sweep = find_matching_sweep(contract, required_sweep_param)
        if not required_sweep_param:
            chosen_sweep = choose_contract_sweep(contract)
            required_sweep_param = str(chosen_sweep.get("variable") or chosen_sweep.get("target") or "")
        required_sweep_id = str(
            contract.get("required_sweep")
            or chosen_sweep.get("id")
            or chosen_sweep.get("name")
            or ""
        ).strip()
        chart_type = phase25_contract_chart_type(contract, original_chart_type, chosen_sweep, metric)
        rationale_parts = [
            str(contract.get("chart_choice_rationale") or "").strip(),
            str(contract.get("evidence_rationale") or "").strip(),
            str(contract.get("why_alternatives_are_weaker") or "").strip(),
        ]
        chart_choice_rationale = " ".join(part for part in rationale_parts if part).strip()
        if not chart_choice_rationale:
            chart_choice_rationale = "Chosen from the executable Phase 2.4 evidence contract."
        repaired_figures.append(
            {
                "figure_id": figure_id,
                "purpose": str(contract.get("claim") or old.get("purpose") or "Claim-focused experiment figure"),
                "chart_intent": chart_intent_lower or str(old.get("chart_intent") or old.get("intent") or ""),
                "chart_type": chart_type,
                "required_sweep": required_sweep_id,
                "chart_choice_rationale": chart_choice_rationale,
                "primary_message": str(contract.get("claim") or old.get("primary_message") or ""),
                "methods": figure_methods,
                "metric": {
                    "name": metric,
                    "display_name": metric_display(metric),
                    "higher_is_better": metric_higher_is_better(metric),
                    "aggregation": "mean",
                },
                "encoding": {
                    "x": {
                        "type": "numeric",
                        "field": "swept_value",
                        "sweep_param": required_sweep_param,
                        "sweep_id": required_sweep_id,
                        "display_name": required_sweep_param.split(".")[-1] if required_sweep_param else "Sweep value",
                    },
                    "group": {"type": "method", "field": "method", "display_name": "Method"},
                    "facet": {"type": "none", "field": None},
                },
                "error_display": "none",
                "data_requirements": {
                    "min_points": max(
                        4 if chart_type == "box" else 10,
                        min(int(contract.get("minimum_paper_points") or (4 if chart_type == "box" else 10)), 8 if chart_type == "box" else 10_000),
                    ),
                    "preferred_points": max(
                        min(int(contract.get("minimum_paper_points") or (6 if chart_type == "box" else 14)), 8 if chart_type == "box" else 10_000),
                        6 if chart_type == "box" else 14,
                    ),
                    "min_samples_per_group": max(int(contract.get("minimum_paper_seeds") or 50), 50),
                    "preferred_samples_per_group": max(int(contract.get("minimum_paper_seeds") or 100), 100),
                },
            }
        )
    if len(repaired_figures) >= 2:
        payload["figure_specs"] = repaired_figures
        def is_diagnostic_metric(metric_name: str) -> bool:
            lowered = str(metric_name or "").strip().lower()
            return any(token in lowered for token in ("violation", "outage", "feasible", "feasibility", "constraint", "shortfall"))

        primary_figure = next(
            (
                fig
                for fig in repaired_figures
                if str(fig.get("chart_intent") or "").strip().lower() in {"main_comparison", "overall_utility", "utility_comparison"}
                and not is_diagnostic_metric(str((fig.get("metric") or {}).get("name") or ""))
            ),
            None,
        )
        if primary_figure is None:
            primary_figure = next(
                (
                    fig
                    for fig in repaired_figures
                    if not is_diagnostic_metric(str((fig.get("metric") or {}).get("name") or ""))
                ),
                repaired_figures[0],
            )
        primary_metric = copy.deepcopy(primary_figure.get("metric", {})) if isinstance(primary_figure, dict) else {}
        if isinstance(primary_metric, dict) and str(primary_metric.get("name") or "").strip():
            payload["primary_metric"] = primary_metric

    if str(evidence.get("tables_optional", "")).strip().lower() in {"1", "true", "yes"}:
        payload["table_specs"] = []
        payload.setdefault("paper_claims_to_test", [])
        payload.setdefault("missing_experiment_recommendations", [])
        return payload

    contract_tables = evidence.get("tables", [])
    raw_table_target = validation_payload.get("table_target", [])
    raw_tables: list[Any] = []
    if isinstance(raw_table_target, dict):
        raw_tables.append(raw_table_target)
    elif isinstance(raw_table_target, list):
        raw_tables.extend(raw_table_target)
    if isinstance(contract_tables, list):
        raw_tables.extend(contract_tables)

    def is_generic_table(table_payload: dict[str, Any]) -> bool:
        cols = [str(item).lower() for item in table_payload.get("columns", []) if str(item).strip()]
        if not cols:
            return True
        generic_tokens = ("objective", "relative_gain", "feasibility", "baseline_", "proposed_")
        return all(any(token in col for token in generic_tokens) or col in {"scenario", "swept_value"} for col in cols)

    def phase25_table_columns_from_contract(table_payload: dict[str, Any]) -> list[str]:
        raw_cols = [str(item).strip() for item in table_payload.get("columns", []) if str(item).strip()]
        metadata_cols = {
            "method",
            "scenario",
            "scenario_name",
            "swept_value",
            "swept_param",
            "lambda_s_ratio",
            "P_max_dBm",
            "Pmax_dBm",
            "M_ris",
            "E_min_mW",
            "K_users",
        }
        preferred_metrics = [
            *raw_cols,
            "weighted_sum_rate_bpsHz",
            "sum_rate_bpsHz",
            "min_user_rate_bpsHz",
            "total_runtime_ms",
            "w_update_time_ms",
            "fp_fixed_point_gap",
            "per_antenna_violation_max_dB",
            "per_antenna_violation_linear_max",
            "objective",
        ]
        cols = ["scenario", "relative_gain_percent", "proposed_feasibility_rate", "baseline_feasibility_rate", "baseline_method"]
        for metric in preferred_metrics:
            metric = metric.strip()
            metric_base = re.sub(r"_(mean|median|std|rate)$", "", metric)
            if not metric_base or metric_base in metadata_cols or metric_base == "feasible":
                continue
            if metric_base not in scalar_metrics and metric not in scalar_metrics:
                continue
            metric_name = metric_base if metric_base in scalar_metrics else metric
            for col in (f"proposed_{metric_name}_mean", f"baseline_{metric_name}_mean"):
                if col not in cols:
                    cols.append(col)
            if len(cols) >= 10:
                break
        return cols[:10]

    table_candidates = [item for item in raw_tables if isinstance(item, dict)]
    if table_candidates:
        table = next((item for item in table_candidates if not is_generic_table(item)), table_candidates[0])
        table_columns = phase25_table_columns_from_contract(table)
        payload["table_specs"] = [
            {
                "table_id": str(table.get("id") or "table_1"),
                "purpose": str(table.get("claim") or "Claim-focused numerical summary"),
                "table_design_rationale": str(table.get("table_design_rationale") or "Rows follow the executed sweep regimes and columns report feasibility plus decomposed physical metrics, rather than a generic objective-only average."),
                "row_selection": str(table.get("row_granularity") or "executed_sweep_regimes"),
                "group_by": "swept_param",
                "columns": table_columns,
            }
        ]
    payload.setdefault("paper_claims_to_test", [])
    payload.setdefault("missing_experiment_recommendations", [])
    return payload


def call_llm_phase25_experiment_planner(
    *,
    run_dir: Path,
    topic: str,
    validation_principles_summary: str,
    system_model_md: str,
    problem_formulation_md: str,
    reformulation_path_md: str,
    algorithm_md: str,
    benchmark_definition_md: str,
    model_profile: str,
) -> dict[str, Any]:
    from phase25_analysis import load_phase24_results, normalize_results_dataframe, summarize_available_results

    phase25_dir = run_dir / "phase2-5"
    phase25_dir.mkdir(parents=True, exist_ok=True)
    phase24_summary, df_raw = load_phase24_results(run_dir)
    df = normalize_results_dataframe(df_raw)
    available = summarize_available_results(phase24_summary, df)
    try:
        validation_payload = yaml.safe_load(read_text(run_dir / "phase2-4" / "validation_plan.yaml")) or {}
        if isinstance(validation_payload, dict):
            evidence_contract = validation_payload.get("research_evidence_contract")
            if not isinstance(evidence_contract, dict) or not evidence_contract:
                evidence_contract = validation_payload.get("paper_evidence_contract")
            if isinstance(evidence_contract, dict):
                available["phase24_research_evidence_contract"] = evidence_contract
    except Exception:
        pass
    write_text(phase25_dir / "available_data_summary.json", json.dumps(available, ensure_ascii=False, indent=2))
    prompt = build_phase25_experiment_planner_prompt(
        topic=topic,
        validation_principles_summary=compact_text(validation_principles_summary, 2200),
        system_model_md=compact_text(system_model_md, 1800),
        problem_formulation_md=compact_text(problem_formulation_md, 1800),
        reformulation_path_md=compact_text(reformulation_path_md, 1800),
        algorithm_md=compact_text(algorithm_md, 2200),
        benchmark_definition_md=compact_text(benchmark_definition_md, 1400),
        experiment_blueprint_md=compact_text(
            read_text(run_dir / "phase2-4" / "phase24_validation_source_contracts.json")
            or summarize_validation_plan(read_text(run_dir / "phase2-4" / "validation_plan.yaml"))
            or validation_principles_summary,
            3000,
        ),
        validation_plan_summary=summarize_validation_plan(read_text(run_dir / "phase2-4" / "validation_plan.yaml")),
        available_data_summary_json=compact_text(json.dumps(available, ensure_ascii=False, indent=2), 4500),
        phase24_summary_excerpt_json=compact_text(json.dumps({k: phase24_summary.get(k) for k in ("num_cases", "num_results", "num_comparable_cases", "proposed_win_rate", "all_finite", "rejection_reason_counts", "infeasible_reason_counts")}, ensure_ascii=False, indent=2), 2000),
    )
    write_text(phase25_dir / "experiment_plan_prompt.txt", prompt)
    disable_llm_planner = str(os.environ.get("WCL_PHASE25_DISABLE_LLM_PLANNER", "")).strip().lower() in {"1", "true", "yes"}
    if disable_llm_planner:
        raise RuntimeError(
            "Phase25 LLM planner is disabled. Deterministic experiment planning has been removed; "
            "enable the LLM planner to create or repair the paper experiment plan."
        )
    llm = create_llm_client(model_profile)
    thinking = {"type": "enabled"} if model_profile == "kimi-k2.6-thinking" else None
    try:
        response = llm.chat(
            [{"role": "user", "content": prompt}],
            json_mode=True,
            strip_thinking=True,
            thinking=thinking,
            max_tokens=9000,
        )
        write_text(phase25_dir / "experiment_plan_raw_response.txt", response.content)
        payload = _safe_json_loads(response.content, {})
        payload = enrich_phase25_method_naming(
            payload,
            algorithm_md=algorithm_md,
            benchmark_definition_md=benchmark_definition_md,
        )
        payload = enforce_phase25_actual_method_names(payload, run_dir)
        payload = de_template_phase25_experiment_plan(payload, available)
        payload = align_phase25_plan_with_phase24_contract(payload, run_dir)
        validate_phase25_experiment_plan(payload)
        write_text(phase25_dir / "experiment_plan.json", json.dumps(payload, ensure_ascii=False, indent=2))
        write_text(
            phase25_dir / "experiment_plan_planner_gate.json",
            json.dumps(
                {
                    "planner": "llm_proposal",
                    "ok": True,
                    "contract_alignment_applied": True,
                },
                ensure_ascii=False,
                indent=2,
            ),
        )
        return payload
    except Exception as exc:
        write_text(
            phase25_dir / "experiment_plan_planner_gate.json",
            json.dumps(
                {
                    "planner": "llm_proposal",
                    "ok": False,
                    "error": str(exc),
                    "contract_alignment_applied": False,
                },
                ensure_ascii=False,
                indent=2,
            ),
        )
        raise


def build_phase25_result_writer_prompt(
    *,
    topic: str,
    algorithm_md: str,
    benchmark_definition_md: str,
    experiment_plan_json: str,
    phase25_experiment_summary_json: str,
    table_md: str,
    figure_metadata_json: str,
    missing_experiments_md: str,
) -> str:
    return render_prompt_template(
        "phase2_5/result_writer.prompt.yaml",
        topic=topic,
        algorithm_md=algorithm_md,
        benchmark_definition_md=benchmark_definition_md,
        experiment_plan_json=experiment_plan_json,
        phase25_experiment_summary_json=phase25_experiment_summary_json,
        table_md=table_md,
        figure_metadata_json=figure_metadata_json,
        missing_experiments_md=missing_experiments_md,
    )


def build_phase25_sweep_refiner_prompt(
    *,
    topic: str,
    algorithm_md: str,
    benchmark_definition_md: str,
    experiment_plan_json: str,
    available_data_summary_json: str,
    phase25_experiment_summary_json: str = "{}",
    deterministic_paper_sweep_plan_json: str,
    missing_experiments_md: str,
) -> str:
    return render_prompt_template(
        "phase2_5/sweep_refiner.prompt.yaml",
        topic=topic,
        algorithm_md=algorithm_md,
        benchmark_definition_md=benchmark_definition_md,
        experiment_plan_json=experiment_plan_json,
        available_data_summary_json=available_data_summary_json,
        phase25_experiment_summary_json=phase25_experiment_summary_json,
        deterministic_paper_sweep_plan_json=deterministic_paper_sweep_plan_json,
        missing_experiments_md=missing_experiments_md,
    )


def call_llm_phase25_result_writer(
    *,
    run_dir: Path,
    topic: str,
    algorithm_md: str,
    benchmark_definition_md: str,
    model_profile: str,
) -> str:
    phase25_dir = run_dir / "phase2-5"
    plan_text = read_text(phase25_dir / "experiment_plan.json")
    summary_text = read_text(phase25_dir / "phase25_experiment_summary.json")
    table_text = read_text(phase25_dir / "tables" / "table_1.md")
    if not table_text.strip():
        first_table = next(iter(sorted((phase25_dir / "tables").glob("*.md"))), None)
        table_text = read_text(first_table) if first_table else ""
    missing_text = read_text(phase25_dir / "missing_experiments.md")
    phase25_summary = read_json(phase25_dir / "phase25_experiment_summary.json") or {}
    if str(os.environ.get("WCL_PHASE25_SKIP_LLM_SUMMARY", "")).strip().lower() in {"1", "true", "yes"}:
        plan = read_json(phase25_dir / "experiment_plan.json") or {}
        status = str(phase25_summary.get("phase25_status") or phase25_summary.get("data_sufficiency_status") or "unknown")
        data_source = str(phase25_summary.get("data_source") or "phase25")
        figures = phase25_summary.get("figure_metadata")
        if not isinstance(figures, list):
            figures = plan.get("figure_specs", []) if isinstance(plan, dict) else []
        figure_bits: list[str] = []
        for figure in figures[:3]:
            if not isinstance(figure, dict):
                continue
            figure_id = str(figure.get("figure_id") or figure.get("id") or "figure")
            metric = figure.get("metric")
            if isinstance(metric, dict):
                metric_name = str(metric.get("name") or "selected metric")
            else:
                metric_name = str(figure.get("metric") or "selected metric")
            x_axis = figure.get("x_field") or figure.get("sweep_param") or ""
            encoding = figure.get("encoding")
            if isinstance(encoding, dict):
                x_info = encoding.get("x")
                if isinstance(x_info, dict):
                    x_axis = x_axis or str(x_info.get("sweep_param") or x_info.get("field") or "")
            figure_bits.append(f"{figure_id} reports {metric_name}" + (f" versus {x_axis}" if x_axis else ""))
        if not figure_bits:
            figure_bits.append("the generated figures follow the Phase 2.4 evidence contract")
        table_line = "No table is generated in the current Phase 2.5 route; the experiment package is figure-first."
        if table_text.strip():
            table_line = "An optional table artifact exists, but the current paper-facing evidence is driven by the generated figures."
        missing_clean = missing_text.strip()
        if not missing_clean:
            missing_clean = "No additional missing-experiment note was emitted by Phase 2.5."
        text = "\n\n".join(
            [
                (
                    f"The experiment package is built from `{data_source}` with Phase 2.5 status `{status}`. "
                    "The compared methods, sweeps, and metrics are inherited from the frozen Phase 2.4 evidence contract, so the figures summarize executed rows rather than invented paper claims."
                ),
                "The generated figures are: " + "; ".join(figure_bits) + ". These plots should be interpreted together with feasibility and constraint-violation diagnostics.",
                table_line,
                (
                    "Limitations and follow-up runs are governed by the Phase 2.5 missing-experiment report. "
                    + missing_clean.splitlines()[0]
                ),
            ]
        )
        write_text(phase25_dir / "result_writer_raw_response.txt", "Phase25 LLM summary explicitly skipped")
        write_text(phase25_dir / "phase25_wcl_experiment_summary.md", text.strip())
        return text.strip()
    prompt = build_phase25_result_writer_prompt(
        topic=topic,
        algorithm_md=compact_text(algorithm_md, 1800),
        benchmark_definition_md=compact_text(benchmark_definition_md, 1200),
        experiment_plan_json=compact_text(plan_text, 4000),
        phase25_experiment_summary_json=compact_text(summary_text, 5000),
        table_md=compact_text(table_text, 2500),
        figure_metadata_json=compact_text(json.dumps(phase25_summary.get("figure_metadata", []), ensure_ascii=False, indent=2), 2000),
        missing_experiments_md=compact_text(missing_text, 2500),
    )
    llm = create_llm_client(model_profile)
    thinking = {"type": "enabled"} if model_profile == "kimi-k2.6-thinking" else None
    response = llm.chat(
        [{"role": "user", "content": prompt}],
        json_mode=False,
        strip_thinking=True,
        thinking=thinking,
        max_tokens=4000,
    )
    write_text(phase25_dir / "result_writer_raw_response.txt", response.content)
    text = response.content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:markdown)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    write_text(phase25_dir / "phase25_wcl_experiment_summary.md", text.strip())
    return text.strip()


def call_llm_phase25_sweep_refiner(
    *,
    run_dir: Path,
    topic: str,
    algorithm_md: str,
    benchmark_definition_md: str,
    model_profile: str,
) -> dict[str, Any]:
    phase25_dir = run_dir / "phase2-5"
    experiment_plan_json = read_text(phase25_dir / "experiment_plan.json")
    available_data_summary_json = read_text(phase25_dir / "available_data_summary.json")
    phase25_experiment_summary_json = read_text(phase25_dir / "phase25_experiment_summary.json")
    deterministic_paper_sweep_plan_json = read_text(phase25_dir / "paper_sweep_plan.json")
    missing_experiments_md = read_text(phase25_dir / "missing_experiments.md")
    disable_llm_refiner = str(os.environ.get("WCL_PHASE25_DISABLE_LLM_REFINER", "")).strip().lower() in {"1", "true", "yes"}
    if str(os.environ.get("WCL_PHASE25_SKIP_REFINER", "")).strip().lower() in {"1", "true", "yes"} or disable_llm_refiner:
        payload = read_json(phase25_dir / "paper_sweep_plan.json") or {}
        if isinstance(payload, dict):
            payload["refined_by_llm"] = False
            payload["refiner_skipped"] = "explicit_skip"
            write_text(phase25_dir / "paper_sweep_plan_refined.json", json.dumps(payload, ensure_ascii=False, indent=2))
            return payload
        return {}
    prompt = build_phase25_sweep_refiner_prompt(
        topic=topic,
        algorithm_md=compact_text(algorithm_md, 1800),
        benchmark_definition_md=compact_text(benchmark_definition_md, 1200),
        experiment_plan_json=compact_text(experiment_plan_json, 4000),
        available_data_summary_json=compact_text(available_data_summary_json, 3500),
        phase25_experiment_summary_json=compact_text(phase25_experiment_summary_json, 5000),
        deterministic_paper_sweep_plan_json=compact_text(deterministic_paper_sweep_plan_json, 3500),
        missing_experiments_md=compact_text(missing_experiments_md, 2500),
    )
    llm = create_llm_client(model_profile)
    thinking = {"type": "enabled"} if model_profile == "kimi-k2.6-thinking" else None
    try:
        response = llm.chat(
            [{"role": "user", "content": prompt}],
            json_mode=True,
            strip_thinking=True,
            thinking=thinking,
            max_tokens=5000,
        )
        write_text(phase25_dir / "paper_sweep_plan_raw_response.txt", response.content)
        payload = _safe_json_loads(response.content, {})
        if payload:
            write_text(phase25_dir / "paper_sweep_plan_refined.json", json.dumps(payload, ensure_ascii=False, indent=2))
            normalized = _normalize_phase25_refined_sweep_plan(payload, phase25_dir)
            if normalized.get("figures"):
                normalized["refined_by_llm"] = True
                write_text(phase25_dir / "paper_sweep_plan.json", json.dumps(normalized, ensure_ascii=False, indent=2))
                return normalized
        return payload
    except Exception as exc:
        write_text(
            phase25_dir / "paper_sweep_refiner_gate.json",
            json.dumps(
                {
                    "refiner": "llm_proposal",
                    "ok": False,
                    "error": str(exc),
                },
                ensure_ascii=False,
                indent=2,
            ),
        )
        raise


def _normalize_phase25_refined_sweep_plan(payload: dict[str, Any], phase25_dir: Path) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    deterministic = read_json(phase25_dir / "paper_sweep_plan.json") or {}
    status = str(payload.get("status") or "").strip()
    if status == "requires_phase24_design_revision":
        result = dict(deterministic) if isinstance(deterministic, dict) else {}
        result["status"] = status
        result["refined_by_llm"] = False
        result["requires_phase24_design_revision"] = True
        result["figures"] = []
        result["notes"] = payload.get("notes", [])
        return result
    default_methods = [
        str(method.get("internal_name") or method.get("name") or method.get("id") or "").strip()
        for method in deterministic.get("compared_methods", [])
        if isinstance(method, dict) and str(method.get("internal_name") or method.get("name") or method.get("id") or "").strip()
    ] if isinstance(deterministic, dict) else []
    default_methods = default_methods or ["proposed"]
    items = payload.get("figures")
    if not isinstance(items, list):
        items = payload.get("missing_for_figures")
    if not isinstance(items, list):
        return {}
    existing_by_id = {
        str(item.get("figure_id") or ""): item
        for item in deterministic.get("figures", [])
        if isinstance(item, dict)
    } if isinstance(deterministic, dict) else {}
    figures: list[dict[str, Any]] = []

    def _numeric_list(raw: Any) -> list[float]:
        if not isinstance(raw, list):
            return []
        numeric_values: list[float] = []
        for value in raw:
            try:
                numeric_values.append(float(value))
            except (TypeError, ValueError):
                continue
        return sorted(dict.fromkeys(numeric_values))

    def _scout_subset(values: list[float]) -> list[float]:
        if len(values) <= 4:
            return list(values)
        indexes = sorted({0, len(values) // 3, (2 * len(values)) // 3, len(values) - 1})
        return [values[index] for index in indexes]

    def _int_with_floor(*raw_values: Any, floor: int) -> int:
        candidates: list[int] = [floor]
        for raw_value in raw_values:
            try:
                candidates.append(int(raw_value))
            except (TypeError, ValueError):
                continue
        return max(candidates)

    def _resolve_required_sweep(param: str, requested_sweep: str = "") -> str:
        param = str(param or "").strip()
        requested_sweep_norm = str(requested_sweep or "").strip().lower()

        def norm(value: str) -> str:
            return str(value or "").strip().replace("/", ".").lower()

        target_norm = norm(param)
        for plan_path in [
            phase25_dir.parent / "phase2-4" / "validation_plan.yaml",
            phase25_dir.parent / "phase2-4" / "solver" / "validation_plan.yaml",
        ]:
            if not plan_path.exists():
                continue
            try:
                plan = yaml.safe_load(plan_path.read_text(encoding="utf-8")) or {}
            except Exception:
                continue
            sweeps = plan.get("sweep_definitions", []) if isinstance(plan, dict) else []
            if not isinstance(sweeps, list):
                continue
            parsed: list[tuple[str, dict[str, Any]]] = []
            for idx, sweep in enumerate(sweeps):
                if not isinstance(sweep, dict):
                    continue
                sweep_id = str(sweep.get("id") or sweep.get("name") or f"sweep_{idx + 1}").strip()
                parsed.append((sweep_id, sweep))
            for sweep_id, sweep in parsed:
                sweep_names = {
                    sweep_id.lower(),
                    str(sweep.get("name") or "").strip().lower(),
                }
                sweep_param_names = {
                    norm(sweep.get("canonical_path") or ""),
                    norm(sweep.get("variable") or ""),
                    norm(sweep.get("target") or ""),
                }
                if requested_sweep_norm and requested_sweep_norm in sweep_names and target_norm in sweep_param_names:
                    return sweep_id
            for sweep_id, sweep in parsed:
                sweep_param_names = {
                    norm(sweep.get("canonical_path") or ""),
                    norm(sweep.get("variable") or ""),
                    norm(sweep.get("target") or ""),
                }
                if target_norm in sweep_param_names:
                    return sweep_id
        return str(requested_sweep or "").strip()

    def _validation_plan_values_for_param(param: str, requested_sweep: str = "") -> list[float]:
        param = str(param or "").strip()
        requested_sweep_norm = str(requested_sweep or "").strip().lower()

        def norm(value: str) -> str:
            return str(value or "").strip().replace("/", ".").lower()

        def extract_values(sweep: dict[str, Any]) -> list[float]:
            raw_values: Any = []
            for key in ("paper_values", "paper_mode_values", "suggested_values", "values"):
                raw_values = sweep.get(key, [])
                if raw_values:
                    break
            if isinstance(raw_values, dict):
                raw_values = raw_values.get("values", [])
            if not isinstance(raw_values, list):
                return []
            return _numeric_list(raw_values)

        target_norm = norm(param)
        for plan_path in [
            phase25_dir.parent / "phase2-4" / "validation_plan.yaml",
            phase25_dir.parent / "phase2-4" / "solver" / "validation_plan.yaml",
        ]:
            try:
                plan = yaml.safe_load(read_text(plan_path)) or {}
            except Exception:
                continue
            sweeps = plan.get("sweep_definitions", []) if isinstance(plan, dict) else []
            if not isinstance(sweeps, list):
                continue
            parsed = [sweep for sweep in sweeps if isinstance(sweep, dict)]
            for sweep in parsed:
                sweep_id = str(sweep.get("id") or sweep.get("name") or "").strip().lower()
                candidate_names = {
                    norm(sweep.get("canonical_path") or ""),
                    norm(sweep.get("variable") or ""),
                    norm(sweep.get("target") or ""),
                }
                if requested_sweep_norm and requested_sweep_norm != sweep_id:
                    continue
                if target_norm and target_norm not in candidate_names:
                    continue
                values = extract_values(sweep)
                if values:
                    return values
            for sweep in parsed:
                candidate_names = {
                    norm(sweep.get("canonical_path") or ""),
                    norm(sweep.get("variable") or ""),
                    norm(sweep.get("target") or ""),
                }
                if target_norm and target_norm in candidate_names:
                    values = extract_values(sweep)
                    if values:
                        return values
        return []

    for item in items:
        if not isinstance(item, dict):
            continue
        figure_id = str(item.get("figure_id") or item.get("id") or "").strip()
        if not figure_id:
            continue
        base = dict(existing_by_id.get(figure_id, {}))
        base_required_param = str(base.get("required_sweep_param") or base.get("sweep_param") or "").strip()
        required_param = str(item.get("required_sweep_param") or item.get("sweep_param") or base_required_param or "").strip()
        param_changed = bool(required_param and base_required_param and required_param != base_required_param)
        base_values = base.get("suggested_values", base.get("suggested_categories_or_values", []))
        if not isinstance(base_values, list):
            base_values = []
        values = item.get("suggested_values", item.get("suggested_categories_or_values", []))
        if not isinstance(values, list):
            values = []
        if param_changed and len(_numeric_list(values)) < 4:
            continue
        replace_values = bool(item.get("replace_existing_values", False))
        if values:
            value_source = values
        elif replace_values:
            value_source = []
        else:
            value_source = base_values
        numeric_values = _numeric_list(value_source)
        base_methods = [
            str(method).strip()
            for method in base.get("methods_to_run", [])
            if str(method).strip()
        ]
        methods: list[str] = []
        method_source = base_methods if base_methods else default_methods
        for method in method_source:
            if method not in methods:
                methods.append(method)
        if not methods:
            methods = list(default_methods)
        suggested_num_seeds = _int_with_floor(base.get("suggested_num_seeds"), item.get("suggested_num_seeds"), floor=100)
        quick_num_seeds = _int_with_floor(base.get("quick_num_seeds"), item.get("quick_num_seeds"), item.get("scout_num_seeds"), floor=20)
        medium_num_seeds = _int_with_floor(base.get("medium_num_seeds"), item.get("medium_num_seeds"), floor=50)
        scout_values = _numeric_list(item.get("scout_values"))
        medium_values = _numeric_list(item.get("medium_values"))
        if not scout_values and (values or replace_values or param_changed):
            scout_values = _scout_subset(numeric_values)
        requested_sweep = str(
            item.get("required_sweep")
            or item.get("required_sweep_id")
            or (base.get("required_sweep") if not param_changed else "")
            or ""
        ).strip()
        resolved_sweep = _resolve_required_sweep(required_param, requested_sweep)
        validation_values = _validation_plan_values_for_param(required_param, resolved_sweep)
        if validation_values:
            if not numeric_values:
                numeric_values = validation_values
            else:
                lower = min(validation_values)
                upper = max(validation_values)
                tolerance = max(abs(upper - lower) * 1e-6, 1e-9)
                if any(value < lower - tolerance or value > upper + tolerance for value in numeric_values):
                    numeric_values = validation_values
                    scout_values = _scout_subset(numeric_values)
                    medium_values = _scout_subset(numeric_values)
        base.update(
            {
                "figure_id": figure_id,
                "chart_type": str(item.get("chart_type") or base.get("chart_type") or "line"),
                "required_sweep": resolved_sweep,
                "required_sweep_param": required_param,
                "suggested_values": sorted(dict.fromkeys(numeric_values)),
                "suggested_categories_or_values": sorted(dict.fromkeys(numeric_values)),
                "suggested_num_seeds": suggested_num_seeds,
                "quick_num_seeds": quick_num_seeds,
                "scout_num_seeds": quick_num_seeds,
                "medium_num_seeds": medium_num_seeds,
                "methods_to_run": methods,
                "claim_tested": str(item.get("claim_tested") or item.get("claim") or base.get("claim_tested") or ""),
                "reason": str(item.get("reason") or base.get("reason") or ""),
            }
        )
        if scout_values:
            base["scout_values"] = scout_values
        elif values or replace_values or param_changed:
            base.pop("scout_values", None)
            base.pop("quick_values", None)
        if medium_values:
            base["medium_values"] = medium_values
        elif values or replace_values or param_changed:
            base.pop("medium_values", None)
        if base["required_sweep_param"] and base["suggested_values"]:
            figures.append(base)
    result = dict(deterministic) if isinstance(deterministic, dict) else {}
    result["status"] = str(payload.get("status") or result.get("status") or "needs_more_phase24_runs")
    result["figures"] = figures
    result["refined_by_llm"] = True
    if "notes" in payload:
        result["notes"] = payload.get("notes")
    return result
