from __future__ import annotations

import importlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

import yaml


def _impl() -> Any:
    return importlib.import_module("phase_runtime_impl")


def _load_phase24_phase25_package(phase25_dir: Path) -> dict[str, Any]:
    impl = _impl()
    experiment_plan = impl.read_json(phase25_dir / "experiment_plan.json") or {}
    phase25_summary = impl.read_json(phase25_dir / "phase25_experiment_summary.json") or {}
    summary_md = impl.read_text(phase25_dir / "phase25_wcl_experiment_summary.md")
    return {
        "plan": experiment_plan,
        "analysis": {"summary": phase25_summary, "source": "phase2.4_contract_evidence_package"},
        "summary_md": summary_md,
        "phase2_5_mode": "evidence_analysis_only",
    }


def _has_phase24_phase25_package(phase25_dir: Path) -> bool:
    impl = _impl()
    existing_manifest = impl.read_json(phase25_dir / "phase25_manifest.json") or {}
    return (
        isinstance(existing_manifest, dict)
        and str(existing_manifest.get("produced_by_phase") or existing_manifest.get("produced_by_" + "phase") or "") in {"phase2.4", "phase2.4_contract"}
        and str(existing_manifest.get("phase2_5_mode") or "") in {"evidence_analysis_only", "contract_packaging"}
        and (phase25_dir / "phase25_experiment_summary.json").exists()
        and (phase25_dir / "experiment_plan.json").exists()
        and (phase25_dir / "plot_quality_report.json").exists()
        and (phase25_dir / "phase25_verified_registry.json").exists()
    )


def _field(payload: dict[str, Any], *names: str) -> Any:
    for name in names:
        value = payload.get(name)
        if value not in (None, ""):
            return value
    return None


def _clean_label(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    # Do not expose internal schema paths as paper-facing labels.
    if "." in text and "\\" not in text and "$" not in text and len(text.split()) == 1:
        return ""
    return text


def _default_metric_label(metric: str) -> str:
    metric_key = str(metric or "").strip()
    labels = {
        "weighted_sum_rate_bpsHz": r"weighted sum-rate $U$ (bit/s/Hz)",
        "weighted_sum_rate_bps_hz": r"weighted sum-rate $U$ (bit/s/Hz)",
        "sum_rate_bpsHz": r"sum-rate $R_{\mathrm{sum}}$ (bit/s/Hz)",
        "sum_rate_bps_hz": r"sum-rate $R_{\mathrm{sum}}$ (bit/s/Hz)",
    }
    return labels.get(metric_key, metric_key.replace("_", " "))


def _metric_label_looks_unpublishable(metric: str, label: str) -> bool:
    compact = re.sub(r"\s+", "", str(label or "")).lower()
    if metric in {"weighted_sum_rate_bpsHz", "weighted_sum_rate_bps_hz"}:
        return (
            not label
            or "bpshz" in compact
            or "\\sum" in label
            or "\\mu" in label
            or "mu_k" in compact
        )
    return False


def _diagnostic_metric(metric: str) -> bool:
    lowered = str(metric or "").strip().lower()
    return any(token in lowered for token in ("violation", "feasible", "feasibility", "residual", "runtime", "status", "solve_time", "gap"))


def _bool_value(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value if value is not None else "").strip().lower()
    if text in {"1", "true", "yes", "y", "higher", "higher_is_better"}:
        return True
    if text in {"0", "false", "no", "n", "lower", "lower_is_better"}:
        return False
    return default


def _metric_higher_is_better(metric: str, objective_sense: str) -> bool:
    lowered = str(metric or "").strip().lower()
    if any(token in lowered for token in ("power", "energy_consumption", "cost", "latency", "delay", "error", "mse", "crb", "violation", "runtime", "time")):
        return False
    if any(token in lowered for token in ("rate", "throughput", "utility", "efficiency", "harvest", "reliability", "snr", "sinr")):
        return True
    return str(objective_sense or "").strip().lower() != "minimize"


def _method_id(method: Any) -> str:
    if isinstance(method, dict):
        return str(method.get("id") or method.get("internal_name") or method.get("name") or "").strip()
    return str(method or "").strip()


def _normalize_methods(methods: Any) -> list[dict[str, Any]]:
    if not isinstance(methods, list):
        return []
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(methods, start=1):
        if not isinstance(raw, dict):
            method_id = str(raw or "").strip()
            raw = {"id": method_id}
        method_id = _method_id(raw)
        if not method_id:
            continue
        short_name = str(raw.get("display_name_short") or raw.get("display_name") or raw.get("name") or method_id).strip()
        long_name = str(raw.get("display_name_long") or raw.get("scientific_purpose") or short_name).strip()
        normalized.append(
            {
                **raw,
                "internal_name": method_id,
                "name": method_id,
                "role": str(raw.get("role") or ("proposed" if method_id == "proposed" else "comparison")).strip(),
                "display_name_short": short_name,
                "display_name_long": long_name,
                "source_of_name": str(raw.get("source_of_name") or "phase24_contract").strip(),
                "display_priority": raw.get("display_priority", index),
            }
        )
    return normalized


def _research_evidence(plan: dict[str, Any]) -> dict[str, Any]:
    evidence = plan.get("research_evidence_contract")
    if not isinstance(evidence, dict):
        evidence = plan.get("paper_evidence_contract") if isinstance(plan.get("paper_evidence_contract"), dict) else {}
    return evidence if isinstance(evidence, dict) else {}


def _sweeps_by_id(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    sweeps: dict[str, dict[str, Any]] = {}
    for raw in plan.get("sweep_definitions", []) if isinstance(plan.get("sweep_definitions"), list) else []:
        if not isinstance(raw, dict):
            continue
        for key in (raw.get("id"), raw.get("name"), raw.get("canonical_path"), raw.get("variable")):
            key_text = str(key or "").strip()
            if key_text:
                sweeps[key_text] = raw
    return sweeps


def _metric_display(metric: str, figure: dict[str, Any], evidence: dict[str, Any]) -> str:
    metric_obj = figure.get("metric") if isinstance(figure.get("metric"), dict) else {}
    axis_labels = figure.get("axis_labels") if isinstance(figure.get("axis_labels"), dict) else {}
    primary = evidence.get("primary_metric") if isinstance(evidence.get("primary_metric"), dict) else {}
    candidates = [
        metric_obj.get("display_name"),
        figure.get("y_axis_label"),
        figure.get("y_display_name"),
        axis_labels.get("y"),
    ]
    if str(primary.get("name") or "").strip() == metric:
        candidates.append(primary.get("display_name"))
    for candidate in candidates:
        label = _clean_label(candidate)
        if label and not _metric_label_looks_unpublishable(metric, label):
            return label
    return _default_metric_label(metric or "objective")


def _x_display(figure: dict[str, Any], sweep: dict[str, Any] | None, sweep_id: str) -> str:
    axis_labels = figure.get("axis_labels") if isinstance(figure.get("axis_labels"), dict) else {}
    candidates = [
        figure.get("x_axis_label"),
        figure.get("x_display_name"),
        axis_labels.get("x"),
    ]
    if isinstance(sweep, dict):
        candidates.extend(
            [
                sweep.get("display_name"),
                sweep.get("label"),
                sweep.get("paper_symbol"),
                sweep.get("variable_symbol"),
                sweep.get("notation"),
                sweep.get("variable"),
            ]
        )
    for candidate in candidates:
        label = _clean_label(candidate)
        if label:
            return label
    return str(sweep_id or "swept value").replace("_", " ")


def _paper_policy(evidence: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    policy = evidence.get("paper_sweep_policy") if isinstance(evidence.get("paper_sweep_policy"), dict) else {}
    top_policy = plan.get("paper_sweep_policy") if isinstance(plan.get("paper_sweep_policy"), dict) else {}
    merged = {**top_policy, **policy}
    return merged


def _primary_metric(evidence: dict[str, Any], figures: list[dict[str, Any]], objective_sense: str) -> dict[str, Any]:
    declared = evidence.get("primary_metric") if isinstance(evidence.get("primary_metric"), dict) else {}
    declared_name = str(declared.get("name") or "").strip()
    if declared_name and not _diagnostic_metric(declared_name):
        default_direction = _metric_higher_is_better(declared_name, objective_sense)
        display_name = _clean_label(declared.get("display_name"))
        if _metric_label_looks_unpublishable(declared_name, display_name):
            display_name = _default_metric_label(declared_name)
        return {
            "name": declared_name,
            "display_name": display_name or _default_metric_label(declared_name),
            "higher_is_better": _bool_value(declared.get("higher_is_better"), default_direction),
        }
    candidates = [
        fig
        for fig in figures
        if str(fig.get("chart_intent") or "").strip().lower() in {"main_comparison", "overall_utility", "utility_comparison"}
    ] + figures
    for fig in candidates:
        metric = str(_field(fig, "y_metric", "primary_metric", "metric") or "").strip()
        if isinstance(fig.get("metric"), dict):
            metric = str(fig["metric"].get("name") or metric).strip()
        if metric and not _diagnostic_metric(metric):
            return {
                "name": metric,
                "display_name": _metric_display(metric, fig, evidence),
                "higher_is_better": _metric_higher_is_better(metric, objective_sense),
            }
    return {
        "name": "objective",
        "display_name": "objective",
        "higher_is_better": str(objective_sense or "").strip().lower() != "minimize",
    }


def _phase24_validation_plan_to_phase25_plan(run_dir: Path) -> dict[str, Any] | None:
    phase24_dir = run_dir / "phase2-4"
    plan_path = phase24_dir / "validation_plan.yaml"
    if not plan_path.exists():
        return None
    plan = yaml.safe_load(plan_path.read_text(encoding="utf-8")) or {}
    if not isinstance(plan, dict):
        return None
    evidence = _research_evidence(plan)
    figures = evidence.get("figures") if isinstance(evidence, dict) else None
    if not isinstance(figures, list):
        figures = plan.get("figures") if isinstance(plan.get("figures"), list) else []
    compared_methods = _normalize_methods(evidence.get("compared_methods") if isinstance(evidence, dict) else [])
    objective_sense = str(plan.get("objective_sense") or evidence.get("objective_sense") or "maximize").strip().lower()
    sweep_lookup = _sweeps_by_id(plan)
    paper_policy = _paper_policy(evidence, plan)

    valid_figures = [fig for fig in figures if isinstance(fig, dict)]
    non_diagnostic = [
        fig
        for fig in valid_figures
        if "violation" not in str(fig.get("y_metric") or fig.get("metric") or "").lower()
        and "feasibility" not in str(fig.get("chart_intent") or "").lower()
    ]
    selected_figures = non_diagnostic[:2]
    selected_ids = {id(fig) for fig in selected_figures}
    for fig in valid_figures:
        if len(selected_figures) >= 2:
            break
        if id(fig) not in selected_ids:
            selected_figures.append(fig)
            selected_ids.add(id(fig))
    figure_specs: list[dict[str, Any]] = []
    for idx, fig in enumerate(selected_figures, start=1):
        metric = str(fig.get("y_metric") or fig.get("primary_metric") or "").strip()
        if isinstance(fig.get("metric"), dict):
            metric = str(fig["metric"].get("name") or metric).strip()
        elif not metric:
            metric = str(fig.get("metric") or "").strip()
        if not metric:
            metric = "objective"
        sweep_id = str(fig.get("required_sweep") or fig.get("sweep_id") or fig.get("id") or "").strip()
        required_sweep_param = str(fig.get("required_sweep_param") or fig.get("canonical_path") or fig.get("sweep_param") or "").strip()
        sweep_spec = sweep_lookup.get(sweep_id) or sweep_lookup.get(required_sweep_param)
        if isinstance(sweep_spec, dict):
            required_sweep_param = str(sweep_spec.get("canonical_path") or required_sweep_param or sweep_spec.get("variable") or "").strip()
            sweep_id = str(sweep_spec.get("id") or sweep_id or "").strip()
        methods_to_run = list(fig.get("methods_to_run") or [])
        if not methods_to_run:
            methods_to_run = [method.get("internal_name") for method in compared_methods]
        methods_to_run = [_method_id(method) for method in methods_to_run]
        methods_to_run = [method for method in methods_to_run if method]
        figure_specs.append(
            {
                "figure_id": str(fig.get("id") or fig.get("figure_id") or f"figure_{idx}"),
                "claim": str(fig.get("claim") or ""),
                "chart_intent": str(fig.get("chart_intent") or "main_comparison"),
                "chart_type": "line" if str(fig.get("chart_type") or "line").strip() in {"line_or_bar_selected_by_data", ""} else str(fig.get("chart_type") or "line"),
                "required_sweep": sweep_id,
                "chart_choice_rationale": str(fig.get("chart_choice_rationale") or fig.get("evidence_rule") or fig.get("evidence_rationale") or "Chosen from the Phase 2.4 executable experiment contract.").strip(),
                "primary_message": str(fig.get("primary_message") or fig.get("claim") or ""),
                "metric": {
                    "name": metric,
                    "display_name": _metric_display(metric, fig, evidence),
                    "higher_is_better": _metric_higher_is_better(metric, objective_sense),
                    "aggregation": "mean",
                },
                "encoding": {
                    "x": {
                        "type": "numeric",
                        "field": "swept_value",
                        "sweep_param": required_sweep_param,
                        "sweep_id": sweep_id,
                        "display_name": _x_display(fig, sweep_spec, sweep_id),
                    },
                    "group": {"type": "method", "field": "method", "display_name": "Method"},
                    "facet": {"type": "none", "field": None},
                },
                "methods": methods_to_run or ["proposed"],
                "error_display": "none",
                "data_requirements": {
                    "min_points": max(10, int(fig.get("minimum_paper_points") or paper_policy.get("minimum_points") or paper_policy.get("min_points") or 10)),
                    "preferred_points": max(14, int(fig.get("preferred_paper_points") or fig.get("minimum_paper_points") or paper_policy.get("preferred_points") or 14)),
                    "min_samples_per_group": max(80, int(fig.get("minimum_paper_seeds") or paper_policy.get("minimum_seeds_per_point") or paper_policy.get("min_seeds_per_point") or 80)),
                    "preferred_samples_per_group": max(100, int(fig.get("preferred_paper_seeds") or fig.get("minimum_paper_seeds") or paper_policy.get("preferred_seeds_per_point") or 100)),
                },
            }
        )
    if not figure_specs:
        return None
    primary_metric = _primary_metric(evidence, selected_figures, objective_sense)
    return {
        "plan_schema": "phase2.5_from_phase2.4_experiment_contract.v2",
        "paper_target": "IEEE WCL",
        "primary_metric": primary_metric,
        "compared_methods": compared_methods,
        "figure_specs": figure_specs,
        "table_specs": [],
        "data_sufficiency_rules": {
            "scout_x_points_per_figure": int(paper_policy.get("scout_points") or 4),
            "scout_seeds_per_point": int(paper_policy.get("scout_seeds_per_point") or 20),
            "medium_x_points_per_figure": int(paper_policy.get("medium_points") or 8),
            "medium_seeds_per_point": int(paper_policy.get("medium_seeds_per_point") or 50),
            "min_x_points_per_figure": int(paper_policy.get("minimum_points") or paper_policy.get("min_points") or 10),
            "preferred_x_points_per_figure": int(paper_policy.get("preferred_points") or 14),
            "min_seeds_per_point": int(paper_policy.get("minimum_seeds_per_point") or paper_policy.get("min_seeds_per_point") or 80),
            "preferred_seeds_per_point": int(paper_policy.get("preferred_seeds_per_point") or 100),
            "max_publication_figures": 3,
            "min_publication_figures": 2,
        },
        "paper_claims_to_test": [
            {
                "claim": str(fig.get("claim") or fig.get("claim_id") or ""),
                "required_evidence": str(fig.get("id") or fig.get("figure_id") or ""),
                "failure_mode": str(fig.get("failure_mode") or "No positive, stable, contract-aligned gain over the selected practical benchmark."),
            }
            for fig in selected_figures
        ],
        "missing_experiment_recommendations": [],
        "phase2_5_mode": "evidence_analysis_only",
        "source_validation_plan": str(plan_path),
    }


def _publish_phase24_solver_outputs_for_phase3_2(run_dir: Path) -> dict[str, Any] | None:
    run_dir = Path(run_dir)
    results_path = run_dir / "phase2-4" / "solver" / "outputs" / "validation_results.csv"
    summary_path = run_dir / "phase2-4" / "solver" / "outputs" / "validation_summary.json"
    if not results_path.exists() or not summary_path.exists():
        return None
    phase25 = importlib.import_module("phase25_analysis")
    phase25_dir = run_dir / "phase2-5"
    if phase25_dir.exists():
        for child in phase25_dir.iterdir():
            if child.name.startswith("paper_sweep_refiner_") or child.name in {
                "paper_sweep_plan_raw_response.txt",
                "paper_sweep_plan_refined.json",
            }:
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    phase25_dir.mkdir(parents=True, exist_ok=True)
    experiment_plan = _phase24_validation_plan_to_phase25_plan(run_dir)
    if not experiment_plan or not experiment_plan.get("figure_specs"):
        return None
    experiment_plan_path = phase25_dir / "experiment_plan.json"
    experiment_plan_path.write_text(json.dumps(experiment_plan, ensure_ascii=False, indent=2), encoding="utf-8")
    result = phase25.run_phase25_analysis(run_dir, experiment_plan_path)
    manifest = {
        "produced_by_phase": "phase2.4_contract",
        "phase2_5_mode": "evidence_analysis_only",
        "source": "phase2-4/solver/outputs/validation_results.csv",
        "experiment_plan": str(experiment_plan_path),
    }
    (phase25_dir / "phase25_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def run_phase25_wcl_package(run_dir: Path, paper_target: str = "IEEE WCL") -> dict[str, Any]:
    """Analyze Phase 2.4 experiment outputs for Phase 3.2.

    Phase 2.4 owns experiment design, KPI selection, benchmark selection, code
    generation, and quick/paper sweep execution. Phase 2.5 does not redesign the
    experiment story after code is fixed; it packages the frozen Phase 2.4
    experiment contract, analyzes data sufficiency, and emits rerun needs.
    """

    run_dir = Path(run_dir)
    phase25_dir = run_dir / "phase2-5"
    phase25_dir.mkdir(parents=True, exist_ok=True)

    export_result = _publish_phase24_solver_outputs_for_phase3_2(run_dir)
    if export_result is not None or _has_phase24_phase25_package(phase25_dir):
        return _load_phase24_phase25_package(phase25_dir)
    raise RuntimeError(
        "Phase 2.5 no longer designs or regenerates experiments. Run Phase 2.4 first so "
        "`phase2-4/solver/outputs/validation_results.csv` and `validation_summary.json` exist."
    )
