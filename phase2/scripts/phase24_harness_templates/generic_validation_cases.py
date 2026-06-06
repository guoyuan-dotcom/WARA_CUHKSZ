from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from problem_data import ProblemData, make_canonical_problem


def _load_plan(plan_path: Path) -> dict[str, Any]:
    if not plan_path.exists():
        return {}
    try:
        with plan_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _normalize_update_path(variable: str) -> str:
    key = str(variable).strip()
    if "." in key:
        return key
    normalized = (
        key.lower()
        .replace("$", "")
        .replace("\\", "")
        .replace("{", "")
        .replace("}", "")
        .replace(" ", "")
        .replace("-", "_")
    )
    aliases = {
        "pmax": "system.Pmax",
        "p_max": "system.Pmax",
        "pmax_dbm": "system.Pmax_dBm",
        "p_max_dbm": "system.Pmax_dBm",
        "lambda": "optimization.lambda",
        "lambda_weights": "optimization.lambda",
        "m": "system.M",
        "nt": "system.Nt",
        "n_t": "system.Nt",
        "e_m^dc": "requirements.E_dc_mW",
        "e_dc": "requirements.E_dc_mW",
        "edc": "requirements.E_dc_mW",
        "e_min": "requirements.E_dc_mW",
        "energy_threshold": "requirements.E_dc_mW",
        "gamma_k": "requirements.gamma_dB",
        "gamma": "requirements.gamma_dB",
        "sinr_target": "requirements.gamma_dB",
        "sinr_threshold": "requirements.gamma_dB",
        "gamma_s": "requirements.Gamma_s_mW",
        "gammas": "requirements.Gamma_s_mW",
        "sensing_threshold": "requirements.Gamma_s_mW",
        "sensing_requirement": "requirements.Gamma_s_mW",
        "b": "rectifier.steepness_b",
        "steepness_b": "rectifier.steepness_b",
        "rectifier_steepness": "rectifier.steepness_b",
        "p_sat": "rectifier.P_sat_mW",
        "psat": "rectifier.P_sat_mW",
        "turn_on": "rectifier.turn_on_a_mW",
        "turn_on_a": "rectifier.turn_on_a_mW",
    }
    return aliases.get(normalized, key)


def _as_int(value: Any, default: int, *, lo: int | None = None, hi: int | None = None) -> int:
    try:
        parsed = int(round(float(value)))
    except Exception:
        parsed = int(default)
    if lo is not None:
        parsed = max(int(lo), parsed)
    if hi is not None:
        parsed = min(int(hi), parsed)
    return parsed


def _sweep_values(spec: dict[str, Any], mode: str = "quick_mode") -> list[Any]:
    mode_aliases = [mode]
    if mode == "quick_mode":
        mode_aliases.extend(["scout_values", "scout_mode_values", "scout_mode", "quick_mode_values", "quick_values"])
    if mode == "paper_mode":
        mode_aliases.extend(["paper_mode_values", "paper_values"])
    for key in [*mode_aliases, "values", "quick_mode", "paper_mode", "paper_mode_values", "paper_values"]:
        payload = spec.get(key)
        if isinstance(payload, dict):
            values = payload.get("values", [])
        else:
            values = payload
        if isinstance(values, list) and values:
            return values
    return []


def _policy_value(plan: dict[str, Any], *keys: str, default: Any = None) -> Any:
    evidence = plan.get("research_evidence_contract", {}) if isinstance(plan, dict) else {}
    if not isinstance(evidence, dict) or not evidence:
        evidence = plan.get("paper_evidence_contract", {}) if isinstance(plan, dict) else {}
    policy = evidence.get("two_pass_policy", {}) if isinstance(evidence, dict) else {}
    if not isinstance(policy, dict):
        policy = {}
    for key in keys:
        if key in policy and policy[key] is not None:
            return policy[key]
    return default


def _quick_seed_values(plan: dict[str, Any], spec: dict[str, Any]) -> list[int]:
    """Return deterministic quick-validation seeds.

    Phase 2.4 is still a structural validation pass, so the harness caps quick
    seeds for runtime. Phase 2.5/paper sweeps consume the large paper-level seed
    policy. The cap is environment-overridable for long unattended tests.
    """

    override = os.environ.get("WARA_PHASE24_QUICK_SEEDS_PER_POINT", "").strip()
    if override:
        count = _as_int(override, 1, lo=1)
    else:
        count = _as_int(
            spec.get(
                "quick_seeds_per_point",
                spec.get(
                    "scout_seeds_per_point",
                    _policy_value(plan, "quick_seeds_per_point", "scout_seeds_per_point", default=1),
                ),
            ),
            1,
            lo=1,
        )
    cap = _as_int(os.environ.get("WARA_PHASE24_QUICK_SEED_CAP", "1"), 1, lo=1)
    count = min(count, cap)
    base = _as_int(spec.get("seed_base", plan.get("random_seed_base", 0) if isinstance(plan, dict) else 0), 0)
    return [base + idx for idx in range(count)]


def _quick_value_cap() -> int:
    return _as_int(os.environ.get("WARA_PHASE24_QUICK_VALUES_PER_SWEEP_CAP", "3"), 3, lo=1)


def _quick_dimension_updates() -> dict[str, Any]:
    """Keep Phase 2.4 quick validation executable; Phase 2.5 expands scale."""

    return {
        "system.K": _as_int(os.environ.get("WARA_PHASE24_QUICK_MAX_K", "6"), 6, lo=1),
        "system.M": _as_int(os.environ.get("WARA_PHASE24_QUICK_MAX_M", "8"), 8, lo=1),
        "system.N": _as_int(os.environ.get("WARA_PHASE24_QUICK_MAX_N", "1"), 1, lo=1),
    }


def _mapping_from_aliases(spec: dict[str, Any], aliases: list[str]) -> dict[str, Any]:
    for alias in aliases:
        value = spec.get(alias)
        if isinstance(value, dict):
            return value
    return {}


def _sweep_static_updates(spec: dict[str, Any]) -> dict[str, Any]:
    updates: dict[str, Any] = {}
    context = _mapping_from_aliases(
        spec,
        [
            "context_overrides",
            "operating_regime_overrides",
            "fixed_overrides",
            "figure_context_overrides",
        ],
    )
    for path, value in context.items():
        normalized = _normalize_update_path(str(path))
        if normalized:
            updates[normalized] = value
    return updates


def _sweep_linked_updates(spec: dict[str, Any], swept_value: Any) -> dict[str, Any]:
    updates: dict[str, Any] = {}
    linked_raw: Any = {}
    for alias in ["linked_paths", "coupled_paths", "synchronized_paths"]:
        if alias in spec:
            linked_raw = spec.get(alias)
            break
    if isinstance(linked_raw, dict):
        items = list(linked_raw.items())
    elif isinstance(linked_raw, list):
        items = []
        for entry in linked_raw:
            if not isinstance(entry, dict):
                continue
            path = str(entry.get("canonical_path") or entry.get("path") or entry.get("target") or "").strip()
            if not path:
                continue
            if bool(entry.get("same_value")) or str(entry.get("rule") or "").lower() in {"same", "same_value", "copy_swept_value", "equal"}:
                rule: Any = "same_value"
            else:
                rule = entry.get("value", entry.get("fixed_value"))
            items.append((path, rule))
    else:
        items = []
    for path, rule in items:
        normalized = _normalize_update_path(str(path))
        if not normalized:
            continue
        if isinstance(rule, str) and rule.strip().lower() in {"same", "same_value", "copy_swept_value", "equal"}:
            updates[normalized] = swept_value
        else:
            updates[normalized] = rule
    return updates


def _iter_sweep_specs(plan: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    candidates: list[Any] = []
    if isinstance(plan, dict):
        candidates.append(plan.get("sweep_definitions"))
        config = plan.get("canonical_config")
        if isinstance(config, dict):
            candidates.append(config.get("sweep_definitions"))

    specs: list[tuple[str, dict[str, Any]]] = []
    for sweeps in candidates:
        if isinstance(sweeps, dict):
            for name, spec in sweeps.items():
                if isinstance(spec, dict):
                    specs.append((str(name), spec))
        elif isinstance(sweeps, list):
            for idx, spec in enumerate(sweeps):
                if isinstance(spec, dict):
                    name = str(spec.get("name") or spec.get("id") or f"sweep_{idx}")
                    specs.append((name, spec))
    return specs


def load_canonical_case(plan_path: Path = Path("validation_plan.yaml")) -> ProblemData:
    plan = _load_plan(plan_path)
    return make_canonical_problem(plan)


def make_validation_cases(plan_path: Path = Path("validation_plan.yaml")) -> list[ProblemData]:
    plan = _load_plan(plan_path)
    canonical = make_canonical_problem(plan)
    quick_updates = _quick_dimension_updates()
    canonical_quick = canonical.clone_with(
        case_name=canonical.case_name,
        case_id=canonical.case_id,
        swept_param=canonical.swept_param,
        swept_value=canonical.swept_value,
        scenario_name=canonical.scenario_name,
        updates=quick_updates,
    )
    cases: list[ProblemData] = [canonical_quick]
    evidence = plan.get("research_evidence_contract", {}) if isinstance(plan, dict) else {}
    if not isinstance(evidence, dict) or not evidence:
        evidence = plan.get("paper_evidence_contract", {}) if isinstance(plan, dict) else {}
    figure_by_sweep: dict[str, dict[str, Any]] = {}
    for idx, figure in enumerate(evidence.get("figures", []) if isinstance(evidence, dict) else []):
        if not isinstance(figure, dict):
            continue
        sweep_id = str(figure.get("required_sweep") or figure.get("sweep_id") or "").strip()
        if not sweep_id:
            continue
        figure_by_sweep.setdefault(
            sweep_id,
            {
                "figure_id": str(figure.get("id") or figure.get("figure_id") or f"figure_{idx + 1}"),
                "claim_id": str(figure.get("claim") or figure.get("claim_id") or ""),
            },
        )
    for sweep_name, spec in _iter_sweep_specs(plan):
        values = list(_sweep_values(spec, mode="quick_mode"))[: _quick_value_cap()]
        variable = _normalize_update_path(str(spec.get("canonical_path") or spec.get("variable") or spec.get("target", "")))
        if not variable:
            continue
        static_updates = _sweep_static_updates(spec)
        seeds = _quick_seed_values(plan, spec)
        for idx, value in enumerate(values):
            updates = dict(static_updates)
            updates.update(_sweep_linked_updates(spec, value))
            updates.update(quick_updates)
            updates[variable] = value
            for seed_idx, seed in enumerate(seeds):
                case_id = f"{sweep_name}_{idx}_seed{seed_idx}"
                case = canonical.clone_with(
                    case_name=case_id,
                    case_id=case_id,
                    swept_param=variable,
                    swept_value=value,
                    scenario_name=str(spec.get("description", sweep_name)),
                    updates=updates,
                )
                setattr(case, "seed", int(seed))
                setattr(case, "sweep_id", str(sweep_name))
                setattr(case, "sweep_parameter", str(variable))
                setattr(case, "sweep_value", value)
                figure_meta = figure_by_sweep.get(str(sweep_name), {})
                setattr(case, "figure_id", str(figure_meta.get("figure_id") or ""))
                setattr(case, "claim_id", str(figure_meta.get("claim_id") or ""))
                cases.append(case)
    return cases
