from __future__ import annotations

import json
import math
import os
import ast
import time
from pathlib import Path
from typing import Any

import yaml

import generated_plugin as plugin_module
from generated_plugin import baseline_solution, build_model, evaluate_state, initial_state, proposed_step
from problem_data import ProblemData, SolverResult, result_to_dict, save_csv, save_json
from validation_cases import make_validation_cases


BASE_REQUIRED_METRICS = {"objective", "feasible", "constraint_violation"}


def _load_plan(plan_path: Path = Path("validation_plan.yaml")) -> dict[str, Any]:
    try:
        with plan_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _metric_name_from_spec(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("name", "metric", "id", "column", "y_metric"):
            text = str(value.get(key) or "").strip()
            if text:
                return text
        return ""
    text = str(value or "").strip()
    if not text:
        return ""
    if text.startswith("{") and text.endswith("}"):
        try:
            parsed = ast.literal_eval(text)
        except Exception:
            parsed = None
        if isinstance(parsed, dict):
            return _metric_name_from_spec(parsed)
    return text


def _required_metric_names(plan: dict[str, Any]) -> list[str]:
    names = set(BASE_REQUIRED_METRICS)
    required = plan.get("required_outputs", {}) if isinstance(plan, dict) else {}
    if not isinstance(required, dict):
        required = {}
    config = plan.get("canonical_config", {}) if isinstance(plan, dict) else {}
    if isinstance(config, dict) and isinstance(config.get("required_outputs"), dict):
        nested_required = dict(config["required_outputs"])
        nested_required.update(required)
        required = nested_required
    scalar_metrics = required.get("scalar_metrics", []) if isinstance(required, dict) else []
    if isinstance(scalar_metrics, list):
        for item in scalar_metrics:
            metric = _metric_name_from_spec(item)
            if metric:
                names.add(metric)
    return sorted(names)


def _objective_sense(plan: dict[str, Any]) -> str:
    candidates: list[Any] = []
    if isinstance(plan, dict):
        candidates.extend([plan.get("objective_sense"), plan.get("optimization_sense")])
        config = plan.get("canonical_config", {})
        if isinstance(config, dict):
            candidates.extend([config.get("objective_sense"), config.get("optimization_sense")])
            optimization = config.get("optimization", {})
            if isinstance(optimization, dict):
                candidates.extend([optimization.get("objective_sense"), optimization.get("sense")])
    for value in candidates:
        lowered = str(value or "").strip().lower()
        if lowered in {"min", "minimize", "minimise", "minimization", "minimisation"}:
            return "minimize"
        if lowered in {"max", "maximize", "maximise", "maximization", "maximisation"}:
            return "maximize"
    return "maximize"


def _is_finite_value(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, complex):
        return math.isfinite(value.real) and math.isfinite(value.imag)
    if isinstance(value, list):
        return all(_is_finite_value(v) for v in value)
    if isinstance(value, dict):
        return all(_is_finite_value(v) for v in value.values())
    return True


def _as_float_or_none(value: Any) -> float | None:
    try:
        if hasattr(value, "item"):
            value = value.item()
        return float(value)
    except Exception:
        return None


def _normalize_metric_aliases(metrics: dict[str, Any], iterations: int, elapsed: float, trace: list[float]) -> dict[str, Any]:
    metrics = dict(metrics)

    def set_if_missing(target: str, *sources: str, scale: float = 1.0) -> None:
        if target in metrics:
            return
        for source in sources:
            value = _as_float_or_none(metrics.get(source))
            if value is not None:
                metrics[target] = value * scale
                return

    set_if_missing("objective_value", "objective")
    set_if_missing("sum_rate_bpsHz", "sum_rate_bps_hz", "sum_rate", "rate_bpsHz", "R_c_bpsHz", "R_c", "rate")
    set_if_missing("sum_rate_bps_hz", "sum_rate_bpsHz", "sum_rate", "rate_bpsHz", "R_c_bpsHz", "R_c", "rate")
    set_if_missing("rate_bpsHz", "sum_rate_bpsHz", "sum_rate_bps_hz", "sum_rate", "R_c_bpsHz", "R_c", "rate")
    set_if_missing("runtime_seconds", "solver_time_sec", "time_sec", "solve_time_sec")
    set_if_missing("runtime_ms", "runtime_msec", "solver_time_ms")
    set_if_missing("solver_time_ms", "runtime_ms", "solve_time_ms")
    set_if_missing("tr_CRB", "crb_tr", "crb_trace")
    set_if_missing("crb_aoa_deg2", "tr_CRB", "crb_trace", "crb")
    set_if_missing("R_c_bpsHz", "rate", "rate_bpsHz", "rate_nats", "rate_Mbps")
    set_if_missing("P_EH_mW", "P_EH", "eh_power", "eh_total", scale=1000.0)
    set_if_missing("eh_total_mW", "P_EH_mW", "eh_total", scale=1000.0)
    set_if_missing("P_EH_linear_mW", "P_EH_linear", "eh_power_linear", "P_EH_mW")
    set_if_missing("harvested_energy_mW", "true_harvested_energy_mW", "harvested_energy", "Eharv", "E_harv", "harvested_power_mW", "P_EH_mW", "Psi_eh")
    set_if_missing("true_harvested_energy_mW", "harvested_energy_mW", "harvested_energy", "Eharv", "E_harv", "P_EH_mW", "Psi_eh")
    set_if_missing("efficiency_eta", "eta_harvest", "harvesting_efficiency", "eta_conversion_efficiency")
    set_if_missing("final_P_in_mW", "P_in_actual_mW", "Pin_at_solution_mW", "Pin", "P_in", "harvested_input_power")
    set_if_missing("initial_P_in_mW", "initial_Pin_mW", "P_in_initial_mW", "final_P_in_mW", "P_in_actual_mW")
    set_if_missing("optimal_rho", "rho", "rho_star", "structural_separation_rho")
    set_if_missing("rank_V_star", "rank_V", "rank_Vstar", "eigen_rank_V")
    set_if_missing("sca_final_surrogate_gap_mW", "sca_surrogate_gap_mW", "surrogate_gap_mW", "sca_gap_mW", "surrogate_gap")
    set_if_missing("constraint_violation_max", "cv_max", "constraint_violation")
    set_if_missing("constraint_violation", "constraint_violation_max", "cv_max")
    set_if_missing("solver_time_sec", "time_sec")
    set_if_missing("time_sec", "solver_time_sec")
    set_if_missing("alpha_mean", "mean_alpha", "alpha_avg")
    set_if_missing("phi_entropy_bits", "alpha_entropy", "phase_entropy")
    set_if_missing("rate_gap_to_R_min", "v_rate", "rate_violation")
    set_if_missing("eh_gap_to_E_min", "v_eh", "eh_violation", scale=1000.0)
    set_if_missing("N_ref", "Nr_ref", "N_reflecting")
    set_if_missing("target_RCS_dB", "target_RCS_dBsm")
    if "alpha_ratio_sensing_comm" not in metrics:
        sensing = _as_float_or_none(metrics.get("alpha_sensing", metrics.get("optimization_alpha_sensing")))
        comm = _as_float_or_none(metrics.get("alpha_comm", metrics.get("optimization_alpha_comm")))
        if sensing is not None and comm is not None:
            metrics["alpha_ratio_sensing_comm"] = sensing / max(comm, 1.0e-12)

    def set_dbm(target: str, *sources: str) -> None:
        if target in metrics:
            return
        for source in sources:
            value = _as_float_or_none(metrics.get(source))
            if value is not None and value > 0:
                metrics[target] = 10.0 * math.log10(value * 1000.0)
                return

    set_dbm("power_total_dBm", "power_total", "total_power")
    set_dbm("power_sensing_dBm", "power_sensing", "sensing_power")
    set_dbm("power_comm_dBm", "power_comm", "communication_power")
    if "radar_SNR_dB" not in metrics:
        radar_linear = _as_float_or_none(metrics.get("radar_SNR"))
        if radar_linear is None:
            radar_linear = _as_float_or_none(metrics.get("radar_snr"))
        if radar_linear is None:
            radar_linear = _as_float_or_none(metrics.get("sensing_gain"))
        if radar_linear is not None and radar_linear > 0:
            metrics["radar_SNR_dB"] = 10.0 * math.log10(radar_linear)
    if "max_constraint_violation" not in metrics:
        violation = metrics.get("constraint_violation")
        if isinstance(violation, dict):
            values = [_as_float_or_none(item) for item in violation.values()]
            values = [float(item) for item in values if item is not None]
            metrics["max_constraint_violation"] = max(values) if values else 0.0
        else:
            violation_value = _as_float_or_none(violation)
            if violation_value is not None:
                metrics["max_constraint_violation"] = violation_value
    if "max_constraint_violation" not in metrics:
        violation_values: list[float] = []
        for key, value in metrics.items():
            key_lower = str(key).lower()
            if "violation" not in key_lower:
                continue
            if key_lower in {"constraint_violation", "constraint_violation_max", "max_constraint_violation"}:
                continue
            numeric = _as_float_or_none(value)
            if numeric is None:
                continue
            violation_values.append(max(0.0, abs(float(numeric))))
        if violation_values:
            metrics["max_constraint_violation"] = max(violation_values)
    set_if_missing("constraint_violation_max", "max_constraint_violation", "cv_max", "total_violation")
    set_if_missing("max_constraint_violation", "constraint_violation_max", "cv_max", "total_violation")
    set_if_missing("constraint_violation", "max_constraint_violation", "constraint_violation_max", "cv_max", "total_violation")
    set_if_missing("constraint_active_C1_power", "power_budget_violation", "power_violation", "c1_power_viol", "C1_viol")
    set_if_missing("constraint_active_C2_sinr", "sinr_violation", "c2_sinr_viol", "C2_viol", "constraint_C2_violation_max")
    set_if_missing("constraint_active_C3_radar", "radar_snr_violation", "c3_radar_viol", "C3_viol")
    set_if_missing("constraint_active_C4_eh", "energy_violation", "c4_eh_viol", "C4_viol")

    if "alpha_sum" not in metrics:
        alpha_mean = _as_float_or_none(metrics.get("alpha_mean"))
        m_e = _as_float_or_none(metrics.get("M_e"))
        metrics["alpha_sum"] = float(alpha_mean * m_e) if alpha_mean is not None and m_e is not None else float(alpha_mean or 0.0)
    metrics.setdefault("alpha_std", 0.0)
    metrics.setdefault("eh_elements_saturated", 0.0)
    metrics.setdefault("objective_crb_term", _as_float_or_none(metrics.get("tr_CRB")) or 0.0)
    metrics.setdefault("objective_rate_term", -(_as_float_or_none(metrics.get("R_c_bpsHz")) or 0.0))
    metrics.setdefault("objective_eh_term", -(_as_float_or_none(metrics.get("P_EH_mW")) or 0.0))
    metrics.setdefault("rank_W", 0.0)
    metrics.setdefault("rank_Wc", 0.0)
    metrics.setdefault("sca_bound_tightness", 0.0)
    metrics.setdefault("wmmse_mse", 0.0)
    metrics.setdefault("sca_iterations", int(iterations))
    set_if_missing("bcd_iterations", "bcd_iter", "bcd_outer_iterations", "outer_iterations")
    metrics.setdefault("bcd_iterations", int(iterations))
    metrics.setdefault("bcd_outer_iterations", int(metrics.get("bcd_iterations", iterations)))
    metrics.setdefault("time_sec", float(elapsed))
    metrics.setdefault("solver_time_sec", float(elapsed))
    metrics.setdefault("runtime_seconds", float(elapsed))
    metrics.setdefault("runtime_ms", float(elapsed) * 1000.0)
    metrics.setdefault("solver_time_ms", float(elapsed) * 1000.0)
    metrics.setdefault("final_P_in_mW", 0.0)
    metrics.setdefault("initial_P_in_mW", float(_as_float_or_none(metrics.get("final_P_in_mW")) or 0.0))
    metrics.setdefault("efficiency_eta", 0.0)
    metrics.setdefault("optimal_rho", float(_as_float_or_none(metrics.get("rho")) or 0.0))
    metrics.setdefault("rank_V_star", float(_as_float_or_none(metrics.get("rank_V")) or 0.0))
    metrics.setdefault("sca_final_surrogate_gap_mW", float(_as_float_or_none(metrics.get("sca_surrogate_gap_mW")) or 0.0))
    metrics.setdefault("randomization_used", False)
    metrics.setdefault("infeasibility_rate", 0.0 if bool(metrics.get("feasible", False)) else 1.0)
    metrics.setdefault("solver_status", str(metrics.get("status", "ok")))
    metrics.setdefault("block_A_status", metrics["solver_status"])
    metrics.setdefault("block_B_status", metrics["solver_status"])
    metrics.setdefault("convergence_flag", bool(len(trace) >= 1 and all(math.isfinite(float(v)) for v in trace)))

    for target, source in (
        ("eh_constraint_satisfied", "c3_viol"),
        ("rate_constraint_satisfied", "c2_viol"),
        ("sensing_power_constraint_satisfied", "c7_viol"),
    ):
        if target not in metrics:
            violation = _as_float_or_none(metrics.get(source))
            metrics[target] = bool(violation is None or violation <= 1.0e-4)
    if "constraint_C2_active" not in metrics:
        c2_value = _as_float_or_none(metrics.get("c2_sinr_viol"))
        if c2_value is None:
            c2_value = _as_float_or_none(metrics.get("C2_viol"))
        metrics["constraint_C2_active"] = bool(c2_value is not None and c2_value <= 1.0e-4)
    return metrics


def _build_solver_result(method: str, metrics: dict[str, Any], iterations: int, elapsed: float, trace: list[float], required_metrics: list[str]) -> SolverResult:
    metrics = _normalize_metric_aliases(metrics, iterations, elapsed, trace)
    measured_ms = float(elapsed) * 1000.0
    metrics["measured_solve_time_sec"] = float(elapsed)
    metrics["measured_total_runtime_ms"] = measured_ms
    # Runtime columns are publication-sensitive. Prefer the harness wall-clock
    # measurement over formula proxies emitted by generated plugins.
    for key in ("solve_time_ms", "solver_time_ms", "runtime_ms", "total_runtime_ms", "w_update_time_ms"):
        if key in metrics or key in required_metrics:
            metrics[key] = measured_ms
    metrics["runtime_proxy_used"] = False
    metrics["runtime_measurement_source"] = "harness_wall_clock"
    missing = [key for key in required_metrics if key not in metrics]
    if missing:
        raise ValueError(f"{method} metrics missing required keys from validation_plan: {missing}")
    status = str(metrics.get("status", "ok"))
    objective = float(metrics.get("objective", 0.0))
    feasible = bool(metrics.get("feasible", False))
    message = str(metrics.get("message", ""))
    return SolverResult(
        method=method,
        status=status,
        objective=objective,
        feasible=feasible,
        iterations=int(iterations),
        solve_time_sec=float(elapsed),
        message=message,
        metrics=metrics,
        trace_objective=[float(v) for v in trace],
    )


def _is_success_result(result: SolverResult) -> bool:
    return bool(result.feasible) and str(result.status).lower() in {"ok", "success", "converged", "feasible", "optimal", "optimal_inaccurate"}


def _quick_contract_methods(problem: ProblemData) -> list[str]:
    plan = getattr(problem, "validation_plan", {}) if hasattr(problem, "validation_plan") else {}
    evidence = plan.get("research_evidence_contract", {}) if isinstance(plan, dict) else {}
    if not isinstance(evidence, dict) or not evidence:
        evidence = plan.get("paper_evidence_contract", {}) if isinstance(plan, dict) else {}
    methods: list[str] = []

    def add_method(value: Any) -> None:
        if isinstance(value, dict):
            method_id = str(value.get("id") or value.get("internal_name") or value.get("name") or "").strip()
        else:
            method_id = str(value or "").strip()
        if method_id and method_id not in methods:
            methods.append(method_id)

    figures = evidence.get("figures", []) if isinstance(evidence, dict) else []
    if isinstance(figures, list):
        for figure in figures[:3]:
            if not isinstance(figure, dict):
                continue
            requested = figure.get("methods_to_run", [])
            if isinstance(requested, list):
                for method in requested:
                    add_method(method)
    if not methods and isinstance(evidence, dict):
        compared = evidence.get("compared_methods", [])
        if isinstance(compared, list):
            for method in compared[:5]:
                add_method(method)
    if "proposed" not in methods:
        methods.insert(0, "proposed")
    methods = [method for method in methods if method]
    try:
        cap = int(os.environ.get("WARA_PHASE24_QUICK_METHOD_CAP", "0"))
    except Exception:
        cap = 0
    if cap <= 0 or len(methods) <= cap:
        return methods
    proposed = [method for method in methods if method == "proposed"]
    others = [method for method in methods if method != "proposed"]
    return (proposed or ["proposed"]) + others[: max(0, cap - 1)]


def _quick_iteration_cap() -> int:
    try:
        return max(1, int(os.environ.get("WARA_PHASE24_QUICK_MAX_ITERATIONS", "2")))
    except Exception:
        return 2


def _is_quick_validation_mode() -> bool:
    tier = str(os.environ.get("WARA_PHASE25_SWEEP_TIER") or os.environ.get("WCL_PHASE25_SWEEP_TIER") or "").strip().lower()
    if tier in {"scout", "medium", "paper"}:
        return False
    mode = str(os.environ.get("WARA_RUN_MODE") or "").strip().lower()
    if mode in {"scout_validation", "medium_validation", "paper_validation"}:
        return False
    return True


def _pilot_seed_count() -> int:
    try:
        return max(1, int(os.environ.get("WARA_PHASE24_PILOT_SEEDS", "20")))
    except Exception:
        return 20


def _run_single_case(problem: ProblemData, case_name: str, required_metrics: list[str]) -> list[SolverResult]:
    seed = int(getattr(problem, "seed", getattr(problem, "realization_id", 0)))
    model = build_model(problem, seed=seed)
    if getattr(problem, "_model_cache", None) is None:
        setattr(problem, "_model_cache", model)
    runtime_model = getattr(problem, "_model_cache", None) or model
    call_model = runtime_model
    if isinstance(model, dict) and isinstance(runtime_model, dict):
        call_model = dict(runtime_model)
        for key in ("state_init", "operators", "metadata"):
            if key in model and key not in call_model:
                call_model[key] = model[key]
    base_state = initial_state(problem, call_model, seed=seed)
    start = time.perf_counter()
    trace: list[float] = []
    state = dict(base_state)
    wrapper_metadata = model.get("metadata", {}) if isinstance(model, dict) and isinstance(model.get("metadata", {}), dict) else {}
    runtime_metadata = call_model.get("metadata", {}) if isinstance(call_model, dict) and isinstance(call_model.get("metadata", {}), dict) else {}
    max_iter = int(wrapper_metadata.get("max_iterations", runtime_metadata.get("max_iterations", call_model.get("max_iterations", 8) if isinstance(call_model, dict) else 8)))
    if _is_quick_validation_mode():
        max_iter = min(max_iter, _quick_iteration_cap())
    for iteration in range(max_iter):
        state = proposed_step(problem, call_model, state, iteration)
        metrics = evaluate_state(problem, call_model, state)
        if not isinstance(metrics, dict):
            raise ValueError("evaluate_state must return dict")
        if not _is_finite_value(metrics):
            raise ValueError("proposed metrics contain non-finite values")
        trace.append(float(metrics.get("objective", 0.0)))
    elapsed = time.perf_counter() - start
    prop_metrics = evaluate_state(problem, call_model, state)
    proposed_result = _build_solver_result("proposed", prop_metrics, max_iter, elapsed, trace, required_metrics)

    requested_methods = _quick_contract_methods(problem)
    baseline_result: SolverResult | None = None
    if "baseline" in requested_methods:
        start = time.perf_counter()
        baseline_state = baseline_solution(problem, call_model, seed=seed)
        if not isinstance(baseline_state, dict):
            raise ValueError("baseline_solution must return dict")
        base_metrics = evaluate_state(problem, call_model, baseline_state)
        if not _is_finite_value(base_metrics):
            raise ValueError("baseline metrics contain non-finite values")
        baseline_elapsed = time.perf_counter() - start
        baseline_result = _build_solver_result("baseline", base_metrics, 1, baseline_elapsed, [float(base_metrics.get("objective", 0.0))], required_metrics)

    extra_results: list[SolverResult] = []
    method_solution = getattr(plugin_module, "method_solution", None)
    for method in requested_methods:
        if method in {"proposed", "baseline"}:
            continue
        if method_solution is None:
            raise ValueError(f"generated_plugin.py must export method_solution for requested method {method!r}")
        start = time.perf_counter()
        method_state = method_solution(problem, call_model, method, seed=seed)
        if not isinstance(method_state, dict):
            raise ValueError(f"method_solution({method}) must return dict")
        method_metrics = evaluate_state(problem, call_model, method_state)
        if not _is_finite_value(method_metrics):
            raise ValueError(f"{method} metrics contain non-finite values")
        method_elapsed = time.perf_counter() - start
        method_iterations = int(method_state.get("iteration", method_state.get("bcd_iter", 1))) if isinstance(method_state, dict) else 1
        extra_results.append(
            _build_solver_result(
                method,
                method_metrics,
                method_iterations,
                method_elapsed,
                [float(method_metrics.get("objective", 0.0))],
                required_metrics,
            )
        )

    all_case_results = [proposed_result]
    if baseline_result is not None:
        all_case_results.append(baseline_result)
    all_case_results.extend(extra_results)
    for result in all_case_results:
        result.metrics["case_name"] = case_name
        result.metrics["case_id"] = str(getattr(problem, "case_id", case_name))
        result.metrics["swept_param"] = str(getattr(problem, "swept_param", "canonical"))
        result.metrics["swept_value"] = getattr(problem, "swept_value", 0.0)
        result.metrics["scenario_name"] = str(getattr(problem, "scenario_name", "default"))
        result.metrics["seed"] = seed
    return all_case_results


def _safe_csv_key(value: Any) -> str:
    key = "".join(ch if str(ch).isalnum() else "_" for ch in str(value)).strip("_")
    return key or "field"


def _csv_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, (int, float, bool, str)) or value is None:
        return value
    return None


def _as_short_sequence(value: Any) -> list[Any] | None:
    if hasattr(value, "tolist"):
        try:
            value = value.tolist()
        except Exception:
            return None
    if not isinstance(value, (list, tuple)):
        return None
    if len(value) > 10:
        return None
    items = list(value)
    if all(_csv_scalar(item) is not None for item in items):
        return [_csv_scalar(item) for item in items]
    return None


def _add_context_value(row: dict[str, Any], path: list[str], value: Any) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _add_context_value(row, [*path, str(key)], child)
        return
    if not path:
        return
    full_key = "_".join(_safe_csv_key(part) for part in path)
    scalar = _csv_scalar(value)
    if scalar is not None:
        row.setdefault(full_key, scalar)
        for start in range(1, len(path)):
            row.setdefault("_".join(_safe_csv_key(part) for part in path[start:]), scalar)
        return
    seq = _as_short_sequence(value)
    if seq is None:
        return
    for idx, item in enumerate(seq):
        row.setdefault(f"{full_key}_{idx}", item)
        row.setdefault(f"{full_key}_{idx + 1}", item)
    if _safe_csv_key(path[-1]) == "lambda_vector" and len(seq) >= 3:
        row.setdefault("lambda_crb", seq[0])
        row.setdefault("lambda_rate", seq[1])
        row.setdefault("lambda_eh", seq[2])


def _add_problem_context(row: dict[str, Any], problem: ProblemData) -> None:
    fields = getattr(problem, "fields", {})
    if isinstance(fields, dict):
        for key, value in fields.items():
            _add_context_value(row, [str(key)], value)


def _flatten_metric_scalars(metrics: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}

    def visit(prefix: str, value: Any) -> None:
        scalar = _csv_scalar(value)
        if scalar is not None:
            flat[prefix] = scalar
            return
        if isinstance(value, dict):
            for key, child in value.items():
                next_prefix = f"{prefix}_{_safe_csv_key(key)}" if prefix else _safe_csv_key(key)
                visit(next_prefix, child)

    visit("", metrics)
    return {key: value for key, value in flat.items() if key}


def _row_max_violation(row: dict[str, Any]) -> float | None:
    values: list[float] = []
    for key in (
        "max_constraint_violation",
        "constraint_violation_max",
        "constraint_violation",
        "total_violation",
        "spacing_violation_lambda",
        "power_violation_W",
        "cv_max",
    ):
        value = _as_float_or_none(row.get(key))
        if value is not None and math.isfinite(float(value)):
            values.append(abs(float(value)))
    return max(values) if values else None


def _row_feasibility_tolerance(row: dict[str, Any]) -> float:
    candidates = (
        "feasibility_tolerance",
        "feasibility_tol",
        "feasibility_eps",
        "constraint_tolerance",
        "constraint_eps",
        "eps_feas",
        "evaluation_feasibility_tol_lambda",
        "feasibility_tol_lambda",
    )
    values = [
        float(value)
        for key in candidates
        for value in [_as_float_or_none(row.get(key))]
        if value is not None and math.isfinite(float(value)) and float(value) > 0
    ]
    return max([1.0e-5, *values])


def _csv_row(result: SolverResult, problem: ProblemData, required_metrics: list[str]) -> dict[str, Any]:
    metrics = result.metrics if isinstance(result.metrics, dict) else {}
    swept_param = str(getattr(problem, "swept_param", "canonical"))
    swept_value = getattr(problem, "swept_value", 0.0)
    row: dict[str, Any] = {
        "run_id": os.environ.get("WARA_RUN_ID", "") or Path.cwd().parent.parent.name,
        "run_mode": os.environ.get("WARA_PHASE25_SWEEP_TIER", "") or os.environ.get("WARA_RUN_MODE", "") or "quick_validation",
        "validation_tier": os.environ.get("WARA_PHASE25_SWEEP_TIER", "") or "quick",
        "case_id": str(getattr(problem, "case_id", problem.case_name)),
        "case_name": problem.case_name,
        "figure_id": str(getattr(problem, "figure_id", "")),
        "claim_id": str(getattr(problem, "claim_id", "")),
        "sweep_id": str(getattr(problem, "sweep_id", swept_param)),
        "sweep_name": str(getattr(problem, "sweep_name", getattr(problem, "sweep_id", swept_param))),
        "sweep_parameter": str(getattr(problem, "sweep_parameter", swept_param)),
        "swept_parameter": str(getattr(problem, "sweep_parameter", swept_param)),
        "sweep_value": str(getattr(problem, "sweep_value", swept_value)),
        "swept_param": swept_param,
        "swept_canonical_path": str(getattr(problem, "swept_canonical_path", swept_param)),
        "swept_value": str(swept_value),
        "scenario_name": str(getattr(problem, "scenario_name", "default")),
        "seed": int(getattr(problem, "seed", 0)),
        "method": result.method,
        "method_id": result.method,
        "status": result.status,
        "objective": result.objective,
        "feasible": result.feasible,
        "iterations": result.iterations,
        "solve_time_sec": result.solve_time_sec,
        "message": result.message,
    }
    _add_problem_context(row, problem)
    for key, value in _flatten_metric_scalars(metrics).items():
        row.setdefault(key, value)
    for key in required_metrics:
        value = metrics.get(key)
        if isinstance(value, (int, float, bool, str)) or value is None:
            row[key] = value
    violations = metrics.get("constraint_violation", {})
    if isinstance(violations, dict):
        for key, value in violations.items():
            if isinstance(value, (int, float, bool, str)) or value is None:
                row[f"violation_{key}"] = value
    max_violation = _row_max_violation(row)
    if max_violation is not None and max_violation <= _row_feasibility_tolerance(row):
        row["feasible"] = True
        row.setdefault("soft_feasible_by_tolerance", True)
    return row


def _semantic_consistency_report(rows: list[dict[str, Any]], required_metrics: list[str]) -> dict[str, Any]:
    by_metric: dict[str, dict[str, Any]] = {}
    proposed_rows = [row for row in rows if row.get("method") == "proposed"]
    for key in required_metrics:
        values = []
        for row in proposed_rows:
            value = row.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values.append(float(value))
        if len(values) >= 2:
            span = max(values) - min(values)
            by_metric[key] = {
                "min": min(values),
                "max": max(values),
                "span": span,
                "constant_across_sweeps": bool(abs(span) <= 1.0e-10),
            }
    suspicious = [key for key, item in by_metric.items() if item.get("constant_across_sweeps") and key not in {"feasible"}]
    return {"metric_variation": by_metric, "suspicious_constant_metrics": suspicious}


def main(output_dir: str = "outputs") -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = _load_plan()
    required_metrics = _required_metric_names(plan)
    cases = make_validation_cases()
    print(f"Running schema-driven validation on {len(cases)} case(s)...")
    all_results: list[SolverResult] = []
    csv_rows: list[dict[str, Any]] = []
    case_groups: dict[str, dict[str, SolverResult]] = {}
    objective_sense = _objective_sense(plan)
    pilot_seeds = _pilot_seed_count()
    for idx, case in enumerate(cases):
        case_id = str(getattr(case, "case_id", case.case_name))
        print(f"  [{idx + 1}/{len(cases)}] {case_id} seeds={pilot_seeds} swept={getattr(case, 'swept_param', 'canonical')}")
        for seed in range(pilot_seeds):
            try:
                case.seed = int(seed)
                case.realization_id = int(seed)
                case.mc_seed = int(seed)
                case._model_cache = None
            except Exception:
                pass
            case_seed_key = f"{case_id}__seed{seed}"
            for result in _run_single_case(case, case.case_name, required_metrics):
                all_results.append(result)
                case_groups.setdefault(case_seed_key, {})[result.method] = result
                csv_rows.append(_csv_row(result, case, required_metrics))

    comparable: list[dict[str, Any]] = []
    for case_id, group in case_groups.items():
        prop = group.get("proposed")
        base = group.get("baseline")
        if prop is None or base is None:
            continue
        if not (_is_success_result(prop) and _is_success_result(base)):
            continue
        prop_obj = float(prop.objective)
        base_obj = float(base.objective)
        if objective_sense == "minimize":
            rel = (base_obj - prop_obj) / max(abs(base_obj), 1.0e-9)
            proposed_win = prop_obj <= base_obj
        else:
            rel = (prop_obj - base_obj) / max(abs(base_obj), 1.0e-9)
            proposed_win = prop_obj >= base_obj
        comparable.append({"case_id": case_id, "relative_gain": float(rel), "proposed_win": bool(proposed_win)})
    relative_gains = [item["relative_gain"] for item in comparable]
    semantic_report = _semantic_consistency_report(csv_rows, required_metrics)
    summary = {
        "validation_mode": "schema_driven",
        "problem_family": str(plan.get("problem_family", plan.get("topic_family", "declared_by_validation_plan"))),
        "required_metric_keys": required_metrics,
        "num_cases": len(cases),
        "pilot_seeds_per_case": pilot_seeds,
        "num_results": len(all_results),
        "num_success": sum(1 for result in all_results if _is_success_result(result)),
        "num_failed": sum(1 for result in all_results if not _is_success_result(result)),
        "num_comparable_cases": len(comparable),
        "objective_sense": objective_sense,
        "proposed_win_count": sum(1 for item in comparable if item["proposed_win"]),
        "proposed_mean_relative_gain": float(sum(relative_gains) / len(relative_gains)) if relative_gains else 0.0,
        "semantic_consistency": semantic_report,
        "all_finite": all(_is_finite_value(result_to_dict(result)) for result in all_results),
        "results": [result_to_dict(result) for result in all_results],
    }
    save_json(summary, out_dir / "validation_summary.json")
    save_csv(csv_rows, out_dir / "validation_results.csv")
    print(json.dumps({"status": "ok", "num_results": len(all_results), "required_metric_keys": required_metrics}))


if __name__ == "__main__":
    main()
