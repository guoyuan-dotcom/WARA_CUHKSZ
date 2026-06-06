from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from pipeline_core import (
    PHASE24_BASE_SIGNATURES,
    PHASE24_FIXED_FILE_CONTRACTS,
    PHASE24_ZERO_ARG_CALLABLES,
    read_json,
    read_text,
    write_text,
)
from phase_runtime.phase24_codegen import PHASE24_SPLIT_ADAPTER_VERSION


def _phase2_has_any(text: str, needles: list[str]) -> bool:
    lowered = str(text or "").lower()
    return any(needle.lower() in lowered for needle in needles)


def _phase24_declared_max_iterations(validation_plan_text: str) -> int:
    try:
        payload = yaml.safe_load(validation_plan_text) or {}
    except Exception:
        return 0
    values: list[int] = []

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if str(key) in {"max_iterations", "max_iter"}:
                    try:
                        values.append(int(value))
                    except (TypeError, ValueError):
                        pass
                visit(value)
        elif isinstance(node, list):
            for item in node:
                visit(item)

    visit(payload)
    return max(values) if values else 0


def _phase24_iteration_cap_mismatch_errors(plugin_code: str, validation_plan_text: str) -> list[str]:
    declared_max_iter = _phase24_declared_max_iterations(validation_plan_text)
    if declared_max_iter <= 0:
        return []
    findings: list[str] = []
    patterns = [
        r"min\s*\(\s*max_iter_cfg\s*,\s*(\d+)\s*\)",
        r"min\s*\(\s*(?:int\s*\(\s*)?(?:metadata|algorithm|model|runtime_metadata)[^,\n]{0,120}max_iter(?:ations)?[^,\n]*,\s*(\d+)\s*\)",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, plugin_code, flags=re.IGNORECASE | re.DOTALL):
            try:
                hard_cap = int(match.group(1))
            except (TypeError, ValueError):
                continue
            if hard_cap < declared_max_iter:
                line_no = plugin_code.count("\n", 0, match.start()) + 1
                findings.append(
                    "Generated experiment code hard-caps algorithm iterations below the validation-plan contract: "
                    f"line {line_no} uses cap {hard_cap}, while the declared max_iterations is {declared_max_iter}. "
                    "Paper-level evidence must use the configured algorithm stopping rule; if a smaller iteration budget is "
                    "scientifically intended, revise the validation plan instead of silently clipping it in code."
                )
    return findings


_PHASE24_SERIOUS_RUNTIME_WARNING_NEEDLES = [
    "divide by zero",
    "overflow",
    "invalid value",
    "invalid encountered",
    "encountered in matmul",
    "encountered in multiply",
    "encountered in true_divide",
]


def _phase24_numerical_runtime_warning_report(stderr_streams: dict[str, str]) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    ignored_findings: list[dict[str, Any]] = []
    for stream_name, stderr in stderr_streams.items():
        if not stderr:
            continue
        lines = str(stderr).splitlines()
        for index, line in enumerate(lines):
            lowered = line.lower()
            if "runtimewarning" not in lowered:
                continue
            if not any(needle in lowered for needle in _PHASE24_SERIOUS_RUNTIME_WARNING_NEEDLES):
                continue
            context_start = max(0, index - 2)
            context_end = min(len(lines), index + 2)
            context = "\n".join(lines[context_start:context_end]).strip()
            item = {
                "stream": stream_name,
                "line_number": index + 1,
                "message": line.strip(),
                "context": context,
            }
            if "site-packages" in (lowered + "\n" + context.lower()) and "cvxpy" in (lowered + "\n" + context.lower()):
                ignored_findings.append(item)
                continue
            findings.append(item)

    errors = [
        f"{item['stream']} line {item['line_number']}: {item['message']}"
        for item in findings
    ]
    return {
        "ok": not findings,
        "errors": errors,
        "findings": findings,
        "ignored_external_solver_warnings": ignored_findings,
    }


def _phase24_validation_outputs_are_finite(summary_path: Path) -> bool:
    summary = read_json(summary_path) or {}
    if not isinstance(summary, dict):
        return False
    if summary.get("all_finite") is False:
        return False
    try:
        num_results = int(summary.get("num_results", 0) or 0)
        num_failed = int(summary.get("num_failed", 0) or 0)
    except (TypeError, ValueError):
        return False
    return num_results > 0 and num_failed == 0


def _phase24_research_evidence(plan: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(plan, dict):
        return {}
    evidence = plan.get("research_evidence_contract")
    if isinstance(evidence, dict) and evidence:
        return evidence
    evidence = plan.get("paper_evidence_contract")
    return evidence if isinstance(evidence, dict) else {}


def _phase24_generated_source_items(solver_dir: Path) -> list[tuple[str, str]]:
    solver_dir = Path(solver_dir)
    names = [
        "generated_plugin.py",
        "generated_experiment_core.py",
        "generated_model.py",
        "generated_methods.py",
        "generated_metrics.py",
    ]
    items: list[tuple[str, str]] = []
    for name in names:
        path = solver_dir / name
        if path.exists():
            items.append((name, read_text(path)))
    return items


def _phase24_combined_generated_source(solver_dir: Path) -> str:
    return "\n\n".join(f"# --- {name} ---\n{source}" for name, source in _phase24_generated_source_items(solver_dir))


def _validate_phase24_split_codegen_package(phase24_dir: Path) -> dict[str, Any]:
    """Reject stale split-code artifacts before interface/runtime validation.

    The public Phase 2.4 runtime uses a deterministic adapter plus one
    LLM-authored core file. If a previous interrupted run leaves only a manifest,
    an old adapter, or an old output CSV, downstream stages can accidentally read
    inconsistent evidence. This check is intentionally conditional: legacy unit
    tests and ad hoc harness checks that do not create a split manifest are still
    allowed to exercise the lower-level validators directly.
    """

    solver_dir = phase24_dir / "solver"
    manifest_path = phase24_dir / "phase24_split_code_manifest.json"
    plugin_path = solver_dir / "generated_plugin.py"
    core_path = solver_dir / "generated_experiment_core.py"
    adapter_text = read_text(plugin_path)
    manifest = read_json(manifest_path) or {}
    split_expected = (
        manifest_path.exists()
        or core_path.exists()
        or "import generated_experiment_core as _core" in adapter_text
        or "PHASE24_SPLIT_ADAPTER_VERSION" in adapter_text
    )
    if not split_expected:
        return {"ok": True, "errors": [], "warnings": ["split adapter package check skipped for legacy single-file plugin"]}

    errors: list[str] = []
    warnings: list[str] = []
    if not plugin_path.exists():
        errors.append("missing solver/generated_plugin.py for split Phase 2.4 package")
    if not core_path.exists():
        errors.append("missing solver/generated_experiment_core.py for split Phase 2.4 package")
    if not manifest_path.exists():
        errors.append("missing phase24_split_code_manifest.json for split Phase 2.4 package")
    if errors:
        return {"ok": False, "errors": errors, "warnings": warnings, "codegen_version": PHASE24_SPLIT_ADAPTER_VERSION}

    if f'PHASE24_SPLIT_ADAPTER_VERSION = "{PHASE24_SPLIT_ADAPTER_VERSION}"' not in adapter_text:
        errors.append(
            "generated_plugin.py is not the current deterministic Phase 2.4 adapter; "
            "force a clean Phase 2.4 regeneration instead of validating stale code"
        )
    if "import generated_experiment_core as _core" not in adapter_text:
        errors.append("generated_plugin.py does not delegate to generated_experiment_core.py")

    manifest_version = str(manifest.get("codegen_version") or "")
    if manifest_version and manifest_version != PHASE24_SPLIT_ADAPTER_VERSION:
        errors.append(
            f"phase24_split_code_manifest.json has stale codegen_version `{manifest_version}`; "
            f"expected `{PHASE24_SPLIT_ADAPTER_VERSION}`"
        )
    elif not manifest_version:
        errors.append("phase24_split_code_manifest.json is missing codegen_version")

    core_text = read_text(core_path)
    expected_hash = str(manifest.get("generated_experiment_core_sha256") or "")
    actual_hash = hashlib.sha256(core_text.encode("utf-8")).hexdigest()
    if expected_hash and expected_hash != actual_hash:
        compile_probe = subprocess.run(
            [sys.executable, "-m", "py_compile", str(core_path.resolve())],
            cwd=solver_dir,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=None,
        )
        if compile_probe.returncode == 0:
            manifest["generated_experiment_core_sha256"] = actual_hash
            manifest["previous_generated_experiment_core_sha256"] = expected_hash
            manifest["hash_refreshed_after_repair"] = True
            write_text(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2))
            warnings.append(
                "generated_experiment_core.py hash differed from the manifest; "
                "the core compiles, so the manifest hash was refreshed after repair."
            )
        else:
            errors.append(
                "generated_experiment_core.py hash does not match phase24_split_code_manifest.json "
                "and the current core does not compile"
            )
    elif not expected_hash:
        errors.append("phase24_split_code_manifest.json is missing generated_experiment_core_sha256")

    outputs_dir = solver_dir / "outputs"
    code_mtime = max(plugin_path.stat().st_mtime, core_path.stat().st_mtime)
    stale_outputs = [
        path.name
        for path in (outputs_dir / "validation_results.csv", outputs_dir / "paper_validation_results.csv")
        if path.exists() and path.stat().st_mtime + 1.0 < code_mtime
    ]
    if stale_outputs:
        warnings.append(
            "existing output CSV files predate the current generated code and must be regenerated: "
            + ", ".join(stale_outputs)
        )

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "codegen_version": PHASE24_SPLIT_ADAPTER_VERSION,
        "generated_experiment_core_sha256": actual_hash,
    }


def _phase24_function_source_from_generated_sources(
    solver_dir: Path,
    function_name: str,
    *,
    prefer_core: bool = False,
) -> str:
    items = _phase24_generated_source_items(solver_dir)
    if prefer_core:
        items = sorted(items, key=lambda item: 1 if item[0] == "generated_plugin.py" else 0)
    for _name, source in items:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
                return ast.get_source_segment(source, node) or ""
    return ""


def run_phase24_paper_sweep_from_plan(run_dir: Path, quick: bool = False) -> dict[str, Any]:
    from phase25_analysis import run_phase24_paper_sweep_from_plan as _run_phase24_paper_sweep_from_plan

    return _run_phase24_paper_sweep_from_plan(run_dir, quick=quick)


def write_phase2_phase24_fixed_harness(run_dir: Path) -> None:
    phase24_dir = run_dir / "phase2-4"
    solver_dir = phase24_dir / "solver"
    solver_dir.mkdir(parents=True, exist_ok=True)

    template_dir = Path(__file__).resolve().parent.parent / "phase24_harness_templates"
    template_map = {
        "generic_problem_data.py": "problem_data.py",
        "generic_validation_cases.py": "validation_cases.py",
        "generic_run_validation.py": "run_validation.py",
    }
    if all((template_dir / src).exists() for src in template_map):
        for src, dst in template_map.items():
            shutil.copyfile(template_dir / src, solver_dir / dst)
        return

    problem_data_py = """from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class ProblemData:
    N: int
    K: int
    alpha: np.ndarray
    P_max: float
    sigma2: float
    p_nominal: np.ndarray
    delta: float
    d_min: float
    q_users: np.ndarray
    fc: float
    R_min: np.ndarray
    case_name: str = "canonical"

    @property
    def wavelength(self) -> float:
        return 3.0e8 / float(self.fc)

    def __post_init__(self) -> None:
        self.alpha = np.asarray(self.alpha, dtype=float)
        self.p_nominal = np.asarray(self.p_nominal, dtype=float)
        self.q_users = np.asarray(self.q_users, dtype=float)
        self.R_min = np.asarray(self.R_min, dtype=float)


@dataclass
class SolverResult:
    method: str
    status: str
    objective: float
    feasible: bool
    iterations: int
    solve_time_sec: float
    message: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    trace_objective: list[float] = field(default_factory=list)


def make_canonical_problem(
    N: int = 16,
    K: int = 4,
    alpha: Any = None,
    P_max: float = 1.0,
    sigma2: float = 1.0e-9,
    p_nominal: Any = None,
    delta: float = 0.2,
    d_min: float = 0.05,
    q_users: Any = None,
    fc: float = 28.0e9,
    R_min: Any = None,
    case_name: str = "canonical",
) -> ProblemData:
    if alpha is None:
        alpha = np.ones(K, dtype=float)
    if p_nominal is None:
        p_nominal = np.column_stack((np.linspace(0.0, 0.15 * (N - 1), N), np.zeros(N), np.zeros(N)))
    if q_users is None:
        q_users = np.array([[5.0, 0.5 * k, 0.0] for k in range(K)], dtype=float)
    if R_min is None:
        R_min = np.full(K, 0.1, dtype=float)
    return ProblemData(
        N=N,
        K=K,
        alpha=np.asarray(alpha, dtype=float),
        P_max=float(P_max),
        sigma2=float(sigma2),
        p_nominal=np.asarray(p_nominal, dtype=float),
        delta=float(delta),
        d_min=float(d_min),
        q_users=np.asarray(q_users, dtype=float),
        fc=float(fc),
        R_min=np.asarray(R_min, dtype=float),
        case_name=str(case_name),
    )


def _serialize_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, dict):
        return {str(k): _serialize_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_serialize_value(v) for v in value]
    return value


def result_to_dict(result: SolverResult) -> dict[str, Any]:
    raw = asdict(result)
    return {str(k): _serialize_value(v) for k, v in raw.items()}


def save_json(obj: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_serialize_value(obj), handle, indent=2, ensure_ascii=False)


def save_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _serialize_value(v) for k, v in row.items()})
"""

    validation_cases_py = """from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from problem_data import ProblemData, make_canonical_problem


# Deterministic Phase 2.4 harness:
# - reads validation_plan.yaml
# - adapts schema conservatively
# - constructs canonical and sweep cases
# - never relies on LLM-generated validation code


def _load_plan(plan_path: Path) -> dict[str, Any]:
    if not plan_path.exists():
        return {}
    try:
        with plan_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _set_case_meta(problem: ProblemData, *, case_id: str, swept_param: str, swept_value: float, scenario_name: str) -> ProblemData:
    problem.case_id = str(case_id)
    problem.swept_param = str(swept_param)
    problem.swept_value = float(swept_value)
    problem.scenario_name = str(scenario_name)
    return problem


def _first_present(mapping: dict[str, Any], keys: list[str], default: Any) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def _to_power_watt(value: float) -> float:
    numeric = float(value)
    if numeric > 100.0:
        return 10 ** ((numeric - 30.0) / 10.0)
    return numeric


def _to_noise_watt(value: float) -> float:
    numeric = float(value)
    if numeric > 1.0e-3:
        return 10 ** ((numeric - 30.0) / 10.0)
    return numeric


def _plan_to_problem_data(plan: dict[str, Any]) -> ProblemData:
    canonical = plan.get("canonical_config", {}) if isinstance(plan, dict) else {}
    if not isinstance(canonical, dict):
        canonical = {}
    system = canonical.get("system", {}) if isinstance(canonical.get("system"), dict) else canonical
    geometry = canonical.get("geometry", {}) if isinstance(canonical.get("geometry"), dict) else canonical
    weights = canonical.get("weights", {}) if isinstance(canonical.get("weights"), dict) else canonical

    base = make_canonical_problem()
    N = int(_first_present(system, ["N", "n_antennas", "num_antennas", "array_size"], base.N))
    K = int(_first_present(system, ["K", "num_users", "users"], base.K))
    fc_raw = _first_present(system, ["fc_hz", "fc", "carrier_hz"], None)
    if fc_raw is not None:
        fc = float(fc_raw)
    else:
        fc = float(_first_present(system, ["fc_ghz", "carrier_ghz"], base.fc / 1.0e9)) * 1.0e9
    p_max = _to_power_watt(_first_present(system, ["P_max", "p_max", "Pmax_dBm", "p_max_dbm", "power_budget", "transmit_power"], base.P_max))
    sigma2 = _to_noise_watt(_first_present(system, ["sigma2", "noise_power", "noise_floor_dBm", "noise_power_dBm", "noise_variance"], base.sigma2))
    delta = float(_first_present(geometry, ["delta_local_region", "delta_m", "delta", "local_region_size"], base.delta))
    d_min = float(_first_present(geometry, ["d_min_collision", "dmin_m", "d_min", "min_separation"], base.d_min))
    alpha = np.asarray(_first_present(weights, ["alpha", "alpha_weights", "weights"], np.ones(K)), dtype=float)
    if alpha.size != K:
        alpha = np.ones(K, dtype=float)
    r_min_val = _first_present(weights, ["qos_rate_bps_hz", "Rmin_bpsHz", "R_min", "qos_threshold"], 0.1)
    R_min = np.full(K, float(r_min_val), dtype=float) if np.isscalar(r_min_val) else np.asarray(r_min_val, dtype=float)
    if R_min.size != K:
        R_min = np.full(K, 0.1, dtype=float)

    aperture = float(_first_present(geometry, ["array_aperture_m", "aperture_m", "array_length_m"], 0.15))
    p_nominal = np.column_stack((np.linspace(0.0, aperture, N), np.zeros(N), np.zeros(N)))
    distances = _first_present(geometry, ["user_distances_m", "user_ranges_m", "ranges_m"], [3.0] * K)
    if len(distances) < K:
        distances = list(distances) + [float(distances[-1] if distances else 3.0)] * (K - len(distances))
    close_angles = _first_present(geometry, ["user_angle_deg", "angles_deg", "user_angles_deg"], None)
    if close_angles is None:
        close_angles = np.linspace(-8.0, 8.0, K).tolist()
    if len(close_angles) < K:
        close_angles = list(close_angles) + [float(close_angles[-1] if close_angles else 0.0)] * (K - len(close_angles))
    q_users = []
    for k in range(K):
        radius = float(distances[k])
        theta_deg = float(close_angles[k])
        theta = np.deg2rad(theta_deg)
        q_users.append([radius * np.cos(theta), radius * np.sin(theta), 0.0])
    q_users = np.asarray(q_users, dtype=float)

    problem = make_canonical_problem(
        N=N,
        K=K,
        alpha=alpha,
        P_max=p_max,
        sigma2=sigma2,
        p_nominal=p_nominal,
        delta=delta,
        d_min=d_min,
        q_users=q_users,
        fc=fc,
        R_min=R_min,
        case_name="canonical",
    )
    return _set_case_meta(problem, case_id="canonical", swept_param="canonical", swept_value=0.0, scenario_name=str(_first_present(canonical, ["scenario", "scenario_name"], "canonical")))


def _clone_problem(problem: ProblemData, **updates: Any) -> ProblemData:
    new_problem = make_canonical_problem(
        N=int(updates.get("N", problem.N)),
        K=int(updates.get("K", problem.K)),
        alpha=np.asarray(updates.get("alpha", problem.alpha), dtype=float),
        P_max=float(updates.get("P_max", problem.P_max)),
        sigma2=float(updates.get("sigma2", problem.sigma2)),
        p_nominal=np.asarray(updates.get("p_nominal", problem.p_nominal), dtype=float),
        delta=float(updates.get("delta", problem.delta)),
        d_min=float(updates.get("d_min", problem.d_min)),
        q_users=np.asarray(updates.get("q_users", problem.q_users), dtype=float),
        fc=float(updates.get("fc", problem.fc)),
        R_min=np.asarray(updates.get("R_min", problem.R_min), dtype=float),
        case_name=str(updates.get("case_name", problem.case_name)),
    )
    return _set_case_meta(
        new_problem,
        case_id=str(updates.get("case_id", new_problem.case_name)),
        swept_param=str(updates.get("swept_param", "canonical")),
        swept_value=float(updates.get("swept_value", 0.0)),
        scenario_name=str(updates.get("scenario_name", "default")),
    )


def load_canonical_case(plan_path: Path = Path("validation_plan.yaml")) -> ProblemData:
    plan = _load_plan(plan_path)
    if not plan:
        return _set_case_meta(make_canonical_problem(), case_id="canonical", swept_param="canonical", swept_value=0.0, scenario_name="canonical")
    return _plan_to_problem_data(plan)


def make_validation_cases(plan_path: Path = Path("validation_plan.yaml")) -> list[ProblemData]:
    plan = _load_plan(plan_path)
    canonical = load_canonical_case(plan_path)
    cases = [canonical]
    sweeps = plan.get("sweep_definitions", {}) if isinstance(plan, dict) else {}
    if isinstance(sweeps, dict):
        for sweep_name, spec in sweeps.items():
            if not isinstance(spec, dict):
                continue
            values = list(spec.get("values", []))[:5]
            variable = str(spec.get("variable", spec.get("target", ""))).lower()
            for idx, value in enumerate(values):
                if variable in {"delta_m", "delta_local_region", "delta"}:
                    cases.append(_clone_problem(canonical, delta=float(value), case_name=f"{sweep_name}_{idx}", case_id=f"{sweep_name}_{idx}", swept_param="delta_m", swept_value=float(value), scenario_name="geometry_delta_sweep"))
                elif variable in {"pmax_dbm", "p_max_dbm", "power_budget", "transmit_power"}:
                    cases.append(_clone_problem(canonical, P_max=10 ** ((float(value) - 30.0) / 10.0), case_name=f"{sweep_name}_{idx}", case_id=f"{sweep_name}_{idx}", swept_param="Pmax_dBm", swept_value=float(value), scenario_name="power_sweep"))
                elif variable in {"fc_ghz", "frequency", "fc"}:
                    cases.append(_clone_problem(canonical, fc=float(value) * 1.0e9, case_name=f"{sweep_name}_{idx}", case_id=f"{sweep_name}_{idx}", swept_param="fc_ghz", swept_value=float(value), scenario_name="frequency_sweep"))
                elif variable in {"k", "num_users"}:
                    k_new = int(value)
                    q = np.asarray(canonical.q_users, dtype=float)
                    if k_new <= q.shape[0]:
                        q_new = q[:k_new]
                    else:
                        extra = []
                        for m in range(q.shape[0], k_new):
                            extra.append([canonical.q_users[-1][0], canonical.q_users[-1][1] + 0.1 * (m - q.shape[0] + 1), canonical.q_users[-1][2]])
                        q_new = np.vstack([q, np.asarray(extra, dtype=float)])
                    alpha = np.ones(k_new, dtype=float) / max(k_new, 1)
                    r_min = np.full(k_new, float(np.mean(np.asarray(canonical.R_min, dtype=float))), dtype=float)
                    cases.append(_clone_problem(canonical, K=k_new, alpha=alpha, R_min=r_min, q_users=q_new, case_name=f"{sweep_name}_{idx}", case_id=f"{sweep_name}_{idx}", swept_param="K", swept_value=float(k_new), scenario_name="user_count_sweep"))

    default_cases = [
        ("power_0p8x", {"P_max": max(float(canonical.P_max) * 0.8, 1.0e-9), "swept_param": "P_max", "swept_value": float(canonical.P_max) * 0.8, "scenario_name": "default_power_sweep"}),
        ("power_1p2x", {"P_max": max(float(canonical.P_max) * 1.2, 1.0e-9), "swept_param": "P_max", "swept_value": float(canonical.P_max) * 1.2, "scenario_name": "default_power_sweep"}),
        ("qos_0p8x", {"R_min": np.asarray(canonical.R_min, dtype=float) * 0.8, "swept_param": "R_min", "swept_value": 0.8, "scenario_name": "default_qos_sweep"}),
        ("qos_1p2x", {"R_min": np.asarray(canonical.R_min, dtype=float) * 1.2, "swept_param": "R_min", "swept_value": 1.2, "scenario_name": "default_qos_sweep"}),
        ("delta_0p8x", {"delta": max(float(canonical.delta) * 0.8, 1.0e-9), "swept_param": "delta", "swept_value": float(canonical.delta) * 0.8, "scenario_name": "default_region_sweep"}),
        ("delta_1p2x", {"delta": max(float(canonical.delta) * 1.2, 1.0e-9), "swept_param": "delta", "swept_value": float(canonical.delta) * 1.2, "scenario_name": "default_region_sweep"}),
        ("fc_0p9x", {"fc": float(canonical.fc) * 0.9, "swept_param": "fc", "swept_value": float(canonical.fc) * 0.9, "scenario_name": "default_frequency_sweep"}),
        ("fc_1p1x", {"fc": float(canonical.fc) * 1.1, "swept_param": "fc", "swept_value": float(canonical.fc) * 1.1, "scenario_name": "default_frequency_sweep"}),
    ]
    q = np.asarray(canonical.q_users, dtype=float)
    if q.ndim == 2 and q.shape[0] >= 2:
        q_close = q.copy()
        q_close[:, 1] = np.linspace(-0.05, 0.05, q.shape[0])
        q_very_close = q.copy()
        q_very_close[:, 1] = np.linspace(-0.02, 0.02, q.shape[0])
        default_cases.extend([
            ("geometry_close", {"q_users": q_close, "swept_param": "geometry", "swept_value": 1.0, "scenario_name": "close_user_geometry"}),
            ("geometry_very_close", {"q_users": q_very_close, "swept_param": "geometry", "swept_value": 2.0, "scenario_name": "very_close_user_geometry"}),
        ])

    existing_ids = {getattr(case, "case_id", getattr(case, "case_name", "")) for case in cases}
    for case_id, updates in default_cases:
        if case_id in existing_ids:
            continue
        cases.append(_clone_problem(canonical, case_name=case_id, case_id=case_id, **updates))
    return cases
"""

    run_validation_py = """from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Any

from generated_plugin import baseline_solution, build_model, evaluate_state, initial_state, proposed_step
from problem_data import ProblemData, SolverResult, result_to_dict, save_csv, save_json
from validation_cases import make_validation_cases


# Deterministic Phase 2.4 harness:
# - owns the validation loop
# - is not generated by the LLM
# - converts plugin outputs into stable summary/csv artifacts


def _is_finite_value(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, list):
        return all(_is_finite_value(v) for v in value)
    if isinstance(value, dict):
        return all(_is_finite_value(v) for v in value.values())
    return True


def _build_solver_result(method: str, metrics: dict[str, Any], iterations: int, elapsed: float, trace: list[float]) -> SolverResult:
    status = str(metrics.get("status", "ok"))
    objective = float(metrics.get("objective", metrics.get("objective_value", 0.0)))
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
    success_statuses = {"ok", "success", "converged", "feasible"}
    return result.feasible and str(result.status).lower() in success_statuses


def _compact_diagnostics(metrics: dict[str, Any]) -> dict[str, Any]:
    diagnostics = metrics.get("diagnostics", {}) if isinstance(metrics, dict) else {}
    if not isinstance(diagnostics, dict):
        diagnostics = {}
    summary = {}
    direct_keys = (
        "iteration",
        "objective",
        "objective_delta",
        "feasible",
        "rejection_reason",
        "power_violation",
        "separation_violation",
        "qos_violation",
        "used_position_update",
        "position_step_norm",
        "trust_radius",
    )
    aliases = {
        "used_position_update": (
            "used_position_update",
            "used_p_update",
            "used_antenna_position_update",
            "used_location_update",
            "used_coordinate_update",
        ),
        "position_step_norm": (
            "position_step_norm",
            "p_update_norm",
            "position_update_norm",
            "p_step_norm",
            "coordinate_update_norm",
        ),
        "objective_delta": (
            "objective_delta",
            "delta_objective",
            "wsr_delta",
            "rate_delta",
        ),
    }
    for key in direct_keys:
        if key in diagnostics:
            summary[key] = diagnostics[key]
        elif key in metrics:
            summary[key] = metrics[key]
    for canonical, candidates in aliases.items():
        if canonical in summary and summary[canonical] not in (None, ""):
            continue
        for key in candidates:
            if key in diagnostics and diagnostics[key] not in (None, ""):
                summary[canonical] = diagnostics[key]
                break
            if key in metrics and metrics[key] not in (None, ""):
                summary[canonical] = metrics[key]
                break
    return summary


def _collect_reason_counts(results: list[SolverResult], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        metrics = result.metrics if isinstance(result.metrics, dict) else {}
        diagnostics = metrics.get("diagnostics", {}) if isinstance(metrics.get("diagnostics", {}), dict) else {}
        value = diagnostics.get(field)
        if value in (None, "", []):
            violations = metrics.get("violations", [])
            if field == "infeasible_reason" and isinstance(violations, list) and violations:
                value = ",".join(str(v) for v in violations)
        if value in (None, "", []):
            continue
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _pilot_seed_count() -> int:
    raw_value = os.environ.get("WARA_PHASE24_PILOT_SEEDS", "20")
    try:
        value = int(raw_value)
    except Exception:
        value = 20
    return max(1, value)


def _run_single_case(problem: ProblemData, case_name: str, seed: int) -> list[SolverResult]:
    try:
        problem.seed = int(seed)
        problem.realization_id = int(seed)
        problem.mc_seed = int(seed)
    except Exception:
        pass
    model = build_model(problem, seed=seed)
    base_state = initial_state(problem, model, seed=seed)
    start = time.perf_counter()
    trace: list[float] = []
    state = dict(base_state)
    max_iter = int(model.get("metadata", {}).get("max_iterations", 8))
    for iteration in range(max_iter):
        state = proposed_step(problem, model, state, iteration)
        metrics = evaluate_state(problem, model, state)
        if not isinstance(metrics, dict):
            raise ValueError("evaluate_state must return dict")
        if not _is_finite_value(metrics):
            raise ValueError("proposed metrics contain non-finite values")
        trace.append(float(metrics.get("objective", metrics.get("objective_value", 0.0))))
    elapsed = time.perf_counter() - start
    prop_metrics = evaluate_state(problem, model, state)
    if not _is_finite_value(prop_metrics):
        raise ValueError("final proposed metrics contain non-finite values")
    proposed_result = _build_solver_result("proposed", prop_metrics, max_iter, elapsed, trace)

    start = time.perf_counter()
    baseline_state = baseline_solution(problem, model, seed=seed)
    if not isinstance(baseline_state, dict):
        raise ValueError("baseline_solution must return dict")
    base_metrics = evaluate_state(problem, model, baseline_state)
    if not _is_finite_value(base_metrics):
        raise ValueError("baseline metrics contain non-finite values")
    baseline_elapsed = time.perf_counter() - start
    baseline_result = _build_solver_result("baseline", base_metrics, 1, baseline_elapsed, [float(base_metrics.get("objective", base_metrics.get("objective_value", 0.0)))])

    proposed_result.metrics["case_name"] = case_name
    baseline_result.metrics["case_name"] = case_name
    for result in (proposed_result, baseline_result):
        result.metrics["case_id"] = str(getattr(problem, "case_id", case_name))
        result.metrics["seed"] = int(seed)
        result.metrics["swept_param"] = str(getattr(problem, "swept_param", "canonical"))
        result.metrics["swept_value"] = float(getattr(problem, "swept_value", 0.0))
        result.metrics["scenario_name"] = str(getattr(problem, "scenario_name", "default"))
    return [proposed_result, baseline_result]


def main(output_dir: str = "outputs") -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = make_validation_cases()
    print(f"Running validation on {len(cases)} case(s)...")
    all_results: list[SolverResult] = []
    csv_rows: list[dict[str, Any]] = []
    case_groups: dict[str, dict[str, SolverResult]] = {}
    seed_count = _pilot_seed_count()
    for idx, case in enumerate(cases):
        name = getattr(case, "case_name", f"case_{idx:03d}")
        case_id = str(getattr(case, "case_id", name))
        swept_param = str(getattr(case, "swept_param", "canonical"))
        swept_value = float(getattr(case, "swept_value", 0.0))
        scenario_name = str(getattr(case, "scenario_name", "default"))
        print(f"  [{idx + 1}/{len(cases)}] {case_id}  seeds={seed_count}  N={case.N} K={case.K} fc={case.fc/1e9:.1f}GHz")
        for seed in range(seed_count):
            case_seed_key = f"{case_id}__seed{seed}"
            for result in _run_single_case(case, name, seed):
                all_results.append(result)
                case_groups.setdefault(case_seed_key, {})[result.method] = result
                metrics = result.metrics if isinstance(result.metrics, dict) else {}
                violation = metrics.get("constraint_violation", {}) if isinstance(metrics.get("constraint_violation", {}), dict) else {}
                csv_rows.append(
                    {
                        "case_id": case_id,
                        "case_name": name,
                        "seed": seed,
                        "swept_param": swept_param,
                        "swept_value": swept_value,
                        "scenario_name": scenario_name,
                        "method": result.method,
                        "status": result.status,
                        "objective": result.objective,
                        "feasible": result.feasible,
                        "iterations": result.iterations,
                        "solve_time_sec": result.solve_time_sec,
                        "power": float(metrics.get("total_power", 0.0)),
                        "qos_violation": float(violation.get("qos", 0.0)),
                        "separation_violation": float(violation.get("separation", 0.0)),
                        "rejection_reason": str(_compact_diagnostics(metrics).get("rejection_reason", "")),
                        "used_position_update": _compact_diagnostics(metrics).get("used_position_update", ""),
                        "objective_delta": _compact_diagnostics(metrics).get("objective_delta", ""),
                        "position_step_norm": _compact_diagnostics(metrics).get("position_step_norm", ""),
                        "message": result.message,
                    }
                )

    comparable: list[dict[str, Any]] = []
    for case_id, group in case_groups.items():
        prop = group.get("proposed")
        base = group.get("baseline")
        if prop is None or base is None:
            continue
        if not (_is_success_result(prop) and _is_success_result(base)):
            continue
        rel = (float(prop.objective) - float(base.objective)) / max(abs(float(base.objective)), 1.0e-9)
        comparable.append({"case_id": case_id, "relative_gain": float(rel), "proposed_win": bool(float(prop.objective) >= float(base.objective))})
    relative_gains = [item["relative_gain"] for item in comparable]
    proposed_win_count = sum(1 for item in comparable if item["proposed_win"])
    best_case = max(comparable, key=lambda item: item["relative_gain"], default=None)
    worst_case = min(comparable, key=lambda item: item["relative_gain"], default=None)
    summary = {
        "num_cases": len(cases),
        "pilot_seeds_per_case": seed_count,
        "num_results": len(all_results),
        "num_success": sum(1 for r in all_results if _is_success_result(r)),
        "num_failed": sum(1 for r in all_results if not _is_success_result(r)),
        "num_comparable_cases": len(comparable),
        "proposed_win_count": proposed_win_count,
        "proposed_win_rate": float(proposed_win_count / len(comparable)) if comparable else 0.0,
        "proposed_mean_relative_gain": float(sum(relative_gains) / len(relative_gains)) if relative_gains else 0.0,
        "proposed_median_relative_gain": float(sorted(relative_gains)[len(relative_gains) // 2]) if relative_gains else 0.0,
        "best_gain_case_id": str(best_case["case_id"]) if best_case else "",
        "worst_gain_case_id": str(worst_case["case_id"]) if worst_case else "",
        "rejection_reason_counts": _collect_reason_counts(all_results, "rejection_reason"),
        "infeasible_reason_counts": _collect_reason_counts(all_results, "infeasible_reason"),
        "all_finite": all(_is_finite_value(result_to_dict(r)) for r in all_results),
        "results": [result_to_dict(r) for r in all_results],
    }
    save_json(summary, out_dir / "validation_summary.json")
    save_csv(csv_rows, out_dir / "validation_results.csv")
    print(f"Finished validation: {len(all_results)} results written to {out_dir}")


if __name__ == "__main__":
    main()
"""

    write_text(solver_dir / "problem_data.py", problem_data_py)
    write_text(solver_dir / "validation_cases.py", validation_cases_py)
    write_text(solver_dir / "run_validation.py", run_validation_py)


def _phase24_positional_arg_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    return [arg.arg for arg in list(node.args.posonlyargs) + list(node.args.args)]


def _phase24_signature_matches(actual: list[str], expected: list[str], aliases: dict[str, set[str]] | None = None) -> bool:
    aliases = aliases or {}
    if len(actual) < len(expected):
        return False
    for got, want in zip(actual[: len(expected)], expected):
        if got == want:
            continue
        if got in aliases.get(want, set()):
            continue
        return False
    return True


def _phase24_harness_signature_errors(file_name: str, function_nodes: dict[str, ast.FunctionDef | ast.AsyncFunctionDef]) -> list[str]:
    """Enforce the fixed Phase 2.4 harness API before smoke execution.

    This is an interface contract, not a topic-specific scientific gate. It
    prevents LLM repair from silently changing argument order and sending the
    deterministic harness into the wrong method branch.
    """

    expected_signatures: dict[str, tuple[list[str], dict[str, set[str]]]] = {
        "build_model": (["problem", "seed"], {}),
        "initial_state": (["problem", "model", "seed"], {}),
        "proposed_step": (["problem", "model", "state", "iteration"], {}),
        "baseline_solution": (["problem", "model", "seed"], {}),
        "evaluate_state": (["problem", "model", "state"], {}),
        "method_solution": (["problem", "model", "method", "seed"], {"method": {"method_id", "method_name"}}),
    }
    errors: list[str] = []
    for fn_name, (expected, aliases) in expected_signatures.items():
        node = function_nodes.get(fn_name)
        if node is None:
            continue
        actual = _phase24_positional_arg_names(node)
        if _phase24_signature_matches(actual, expected, aliases):
            continue
        errors.append(
            f"{file_name} function signature mismatch for {fn_name}: "
            f"expected leading args {expected}, got {actual[:len(expected)]}. "
            "Phase 2.4 harness-facing exports have fixed argument order; repair must not reorder or drop them."
        )
    return errors


def validate_phase2_phase24_plugin_interfaces(solver_dir: Path) -> dict[str, Any]:
    solver_dir = Path(solver_dir)
    core_exists = (solver_dir / "generated_experiment_core.py").exists()
    required = {
        "generated_plugin.py": ["build_model", "initial_state", "proposed_step", "baseline_solution", "evaluate_state"],
    }
    if core_exists:
        required["generated_experiment_core.py"] = [
            "build_model",
            "initial_state",
            "proposed_step",
            "baseline_solution",
            "evaluate_state",
        ]
    errors: list[str] = []
    declared_methods: set[str] = set()
    contract_methods: set[str] = set()
    figure_methods: set[str] = set()
    method_contract_texts: dict[str, str] = {}
    try:
        plan_payload = yaml.safe_load(read_text(Path(solver_dir).parent / "validation_plan.yaml")) or {}
    except Exception:
        plan_payload = {}
    if isinstance(plan_payload, dict):
        evidence = _phase24_research_evidence(plan_payload)
        if isinstance(evidence, dict):
            method_entries = evidence.get("compared_methods", [])
            if isinstance(method_entries, list):
                for item in method_entries:
                    if isinstance(item, dict):
                        method_id = str(item.get("id") or item.get("internal_name") or item.get("name") or "").strip()
                        contract_text = " ".join(
                            str(item.get(key) or "")
                            for key in (
                                "id",
                                "internal_name",
                                "name",
                                "role",
                                "display_name_short",
                                "display_name_long",
                                "scientific_purpose",
                                "implementation_hint",
                                "fairness_rule",
                            )
                        )
                    else:
                        method_id = str(item or "").strip()
                        contract_text = method_id
                    if method_id:
                        contract_methods.add(method_id)
                        method_contract_texts[method_id] = contract_text
            figures = evidence.get("figures", [])
            if isinstance(figures, list):
                for figure in figures:
                    if not isinstance(figure, dict):
                        continue
                    for item in figure.get("methods_to_run", []) if isinstance(figure.get("methods_to_run"), list) else []:
                        if isinstance(item, dict):
                            method_id = str(item.get("id") or item.get("internal_name") or item.get("name") or "").strip()
                        else:
                            method_id = str(item or "").strip()
                        if method_id:
                            figure_methods.add(method_id)
            # Only figure/table-active methods are executable obligations for the
            # generated plugin. The broader compared_methods contract may include
            # future or optional ablations, and forcing all of them into one quick
            # Phase 2.4 plugin made repairs oscillate and degrade the experiments.
            declared_methods = set(figure_methods) if figure_methods else set(contract_methods)
    for file_name, functions in required.items():
        path = solver_dir / file_name
        if not path.exists():
            errors.append(f"missing required plugin file: {file_name}")
            continue
        source = read_text(path)
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            errors.append(f"{file_name} has syntax error: {exc.msg} (line {exc.lineno})")
            continue
        function_nodes = {
            node.name: node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        top_level_functions = set(function_nodes)
        for fn_name in functions:
            if fn_name not in top_level_functions:
                errors.append(f"{file_name} missing required function: {fn_name}")
        errors.extend(_phase24_harness_signature_errors(file_name, function_nodes))
        extra_methods = sorted(method for method in declared_methods if method not in {"proposed", "baseline"})
        check_method_branches = file_name != "generated_plugin.py" or not core_exists
        if check_method_branches and extra_methods and "method_solution" not in top_level_functions:
            errors.append(
                f"{file_name} missing required function: method_solution "
                f"for compared methods declared by validation_plan.yaml: {extra_methods}"
            )
        for fn_name, fn_node in function_nodes.items():
            if fn_name == "build_model":
                continue
            if fn_name in {
                "run_method",
                "run_point",
                "evaluate",
                "solve",
                "solve_method",
                "proposed_solution",
                "run_single",
                "run_sweep",
                "run_experiment",
                "execute",
                "main",
            } and fn_name not in set(functions):
                # Some generated cores carry legacy debugging wrappers. The fixed
                # adapter does not expose these names to the harness, so do not
                # block an otherwise valid solver solely because a non-contracted
                # wrapper can rebuild a model when called manually.
                continue
            fn_source = ast.get_source_segment(source, fn_node) or ""
            for call in ast.walk(fn_node):
                if not isinstance(call, ast.Call):
                    continue
                callee = call.func
                callee_name = callee.id if isinstance(callee, ast.Name) else callee.attr if isinstance(callee, ast.Attribute) else ""
                if callee_name != "build_model":
                    continue
                if _phase24_private_cached_model_fallback(fn_name, fn_source):
                    continue
                if _phase24_optional_model_fallback(fn_node, fn_source):
                    continue
                seed_zero = any(
                    kw.arg == "seed" and isinstance(kw.value, ast.Constant) and kw.value.value == 0
                    for kw in call.keywords
                )
                seed_note = " with seed=0" if seed_zero else ""
                errors.append(
                    f"{file_name}:{getattr(call, 'lineno', '?')} {fn_name} calls build_model{seed_note}. "
                    "The harness calls build_model once per case/seed; helpers must use the supplied/cached model."
                )
        aggregate_covariance_sinr_errors = analyze_phase24_aggregate_covariance_sinr_antipattern(source)
        errors.extend(aggregate_covariance_sinr_errors)
        ris_quadratic_errors = analyze_phase24_ris_quadratic_dimension_antipattern(source)
        errors.extend(ris_quadratic_errors)
        no_rho_partition_errors = analyze_phase24_no_rho_partition_antipattern(source, method_contract_texts)
        errors.extend(no_rho_partition_errors)
        if check_method_branches and extra_methods and "method_solution" in function_nodes:
            method_node = function_nodes["method_solution"]
            method_source = ast.get_source_segment(source, method_node) or ""
            missing_ids = [
                method
                for method in extra_methods
                if not _phase24_method_id_is_explicitly_reachable(source, method_source, method)
            ]
            if missing_ids:
                errors.append(
                    "method_solution does not explicitly implement method ids requested by research_evidence_contract figures: "
                    + ", ".join(missing_ids)
                    + ". These are exact executable ids, not prose labels; add literal branches or dispatch keys for the exact strings "
                    + ", ".join(f"`{method}`" for method in missing_ids)
                    + " and return state['method'] with the same exact id."
                )
            lowered_method_source = method_source.lower()
            ast_match_node = getattr(ast, "Match", None)
            branching_nodes = (ast.If,) if ast_match_node is None else (ast.If, ast_match_node)
            has_branching = any(isinstance(node, branching_nodes) for node in ast.walk(method_node))
            if (
                "baseline_solution" in method_source
                and not has_branching
                and ("state[\"method\"] = method" in method_source or "state['method'] = method" in method_source)
            ) or "only baseline and proposed" in lowered_method_source:
                errors.append(
                    "method_solution appears to relabel one baseline/proposed state for all methods. "
                    "Each compared method must change the candidate state or model assumption according to its contract."
                )
            method_semantic_source_errors = analyze_phase24_method_solution_source_semantics(
                method_source,
                {method: method_contract_texts.get(method, method) for method in extra_methods},
            )
            errors.extend(method_semantic_source_errors)
    return {"ok": not errors, "errors": errors}


def _phase24_private_cached_model_fallback(fn_name: str, fn_source: str) -> bool:
    """Allow private helpers to accept either a cached model or a problem object.

    The public harness calls `build_model` once and then passes the model into
    `evaluate_state`. Some high-quality generated cores use a private helper
    such as `_evaluate_state_core(problem_or_model, ...)` to support both direct
    debugging and harness execution. Blocking that helper caused repair loops
    even when the public interface used the cached model correctly. Public
    operators such as `channel_from_state` are still forbidden from rebuilding.
    """

    if not str(fn_name or "").startswith("_"):
        return False
    lowered = str(fn_source or "").lower()
    return (
        "build_model(" in lowered
        and "isinstance(" in lowered
        and "dict" in lowered
        and ("problem_or_model" in lowered or "model_or_problem" in lowered or "model" in lowered)
    )


def _phase24_optional_model_fallback(fn_node: ast.FunctionDef | ast.AsyncFunctionDef, fn_source: str) -> bool:
    """Allow public helpers that only build a model when no cached model is supplied.

    The Phase 2.4 harness passes a cached model into the contracted public
    functions. Generated code may still expose paper-author/debug convenience
    functions with signatures such as ``baseline_solution(problem, model=None)``.
    This is safe when every build_model call is guarded by ``if model is None``;
    blocking that pattern forced unnecessary LLM repairs without improving
    reproducibility.
    """

    has_model_arg = any(arg.arg == "model" for arg in fn_node.args.args)
    if not has_model_arg:
        return False
    lowered = str(fn_source or "").lower()
    if "build_model(" not in lowered or "if model is none" not in lowered:
        return False
    build_calls = lowered.count("build_model(")
    guarded_calls = len(
        re.findall(
            r"if\s+model\s+is\s+none\s*:\s*(?:\n\s+[^#\n]*)*?\n\s*model\s*=\s*build_model\s*\(",
            lowered,
        )
    )
    return build_calls > 0 and build_calls == guarded_calls


def _phase24_method_id_is_explicitly_reachable(source: str, method_source: str, method: str) -> bool:
    literal_re = rf"['\"]{re.escape(method)}['\"]"
    if re.search(literal_re, method_source):
        return True
    if not re.search(literal_re, source):
        return False
    if not re.search(r"\w+\s*\([^)]*\bmethod\b", method_source):
        return False
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not any(arg.arg == "method" for arg in node.args.args):
            continue
        segment = ast.get_source_segment(source, node) or ""
        if node.name == "method_solution":
            continue
        if re.search(literal_re, segment) and re.search(r"\b(if|elif|match)\b", segment):
            return True
    registry_names = ("ACTIVE_METHOD_IDS", "SUPPORTED_METHODS", "METHOD_IDS", "METHOD_REGISTRY")
    if any(name in method_source for name in registry_names):
        return re.search(literal_re, source) is not None
    return False


def analyze_phase24_aggregate_covariance_sinr_antipattern(source: str) -> list[str]:
    """Detect a common invalid multi-user SINR implementation in generated plugins.

    Phase 2.4 often prototypes with one aggregate transmit covariance Rx. That is fine
    for sensing/energy surrogates, but it is not a per-stream downlink model. Counting
    h_j^H Rx h_j for j != k as interference while h_k^H Rx h_k is the desired signal
    double-counts the same covariance and makes every SINR nearly zero.
    """
    if not source.strip():
        return []
    source_lower = source.lower()
    if "sinr" not in source_lower or "rx" not in source_lower:
        return []
    has_proxy_marker = any(
        marker in source_lower
        for marker in [
            "effective_snr_proxy",
            "noise_limited_snr_proxy",
            "aggregate_covariance_snr_proxy",
        ]
    )
    if any(marker in source_lower for marker in ["per_user_covariance", "per-user covariance"]):
        return []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    errors: list[str] = []
    for fn_node in [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        fn_source = ast.get_source_segment(source, fn_node) or ""
        compact = re.sub(r"\s+", "", fn_source.lower())
        if "sinr" not in compact or "rx" not in compact:
            continue
        has_per_stream_state = any(
            token in compact
            for token in [
                "state.get('w",
                'state.get("w',
                "state.get('beams",
                'state.get("beams',
                "state.get('wk",
                'state.get("wk',
                "state.get('user_cov",
                'state.get("user_cov',
                "state.get('per_user",
                'state.get("per_user',
                "w_users",
                "w_stream",
                "w_list",
                "wk_list",
            ]
        )
        if has_proxy_marker and "intf" in compact and re.search(r"\(?k-1\)?|k-1|k\)-1", compact) and not has_per_stream_state:
            if "intf" in compact and re.search(r"\(?k-1\)?|k-1|k\)-1", compact):
                errors.append(
                    f"generated_plugin.py:{getattr(fn_node, 'lineno', '?')} {fn_node.name} is labeled as an effective SNR proxy "
                    "but still introduces synthetic (K-1) interference from the same aggregate covariance/channel power. "
                    "For an aggregate-covariance prototype the proxy must be noise-limited or use explicit per-stream beams; "
                    "otherwise SINR is capped near 1/(K-1) and the QoS gate becomes infeasible by construction."
                )
            continue
        has_user_interference_loop = bool(
            re.search(r"for\w+inrange\(\w+\).*?if\w+==\w+:continue", compact)
            or re.search(r"for\w+inrange\(k\).*?if\w+==\w+:continue", compact)
        )
        aggregate_interference_terms = [
            r"intf\+=.*?@rx@",
            r"interference\+=.*?@rx@",
            r"denom.*?=.*?intf",
            r"sinr=.*?/\(intf\+",
        ]
        has_aggregate_interference = sum(1 for pattern in aggregate_interference_terms if re.search(pattern, compact)) >= 2
        if has_user_interference_loop and has_aggregate_interference and not has_per_stream_state:
            errors.append(
                f"generated_plugin.py:{getattr(fn_node, 'lineno', '?')} {fn_node.name} appears to compute multi-user SINR "
                "by using one aggregate covariance Rx as both desired signal and other-user interference "
                "(e.g., h_j^H Rx h_j inside the interference loop). This is invalid for the Phase 2.4 evidence check: "
                "carry per-user beams/covariances and compute stream interference, or explicitly document/use an "
                "effective_snr_proxy for aggregate-covariance prototypes."
            )
    return errors


def analyze_phase24_ris_quadratic_dimension_antipattern(source: str) -> list[str]:
    """Detect RIS steering-vector quadratic forms with a known double-transpose bug."""
    if not source.strip():
        return []
    if "a_radar" not in source.lower() or "@" not in source:
        return []
    errors: list[str] = []
    for match in re.finditer(r"(?P<var>\w+)\s*=\s*_herm\(\s*A_radar\s*\)\.T\b", source, flags=re.IGNORECASE):
        var_name = match.group("var")
        line_no = source.count("\n", 0, match.start()) + 1
        nearby_source = source[match.end() : match.end() + 700]
        if re.search(rf"\b{re.escape(var_name)}\b\s*@\s*A_radar\b", nearby_source, flags=re.IGNORECASE):
            errors.append(
                f"generated_plugin.py:{line_no} has a double-transpose RIS/radar bug: it assigns "
                f"`{var_name} = _herm(A_radar).T` and later multiplies "
                f"`{var_name} @ A_radar`. For a radar steering row A_radar with shape 1 x M, `_herm(A_radar)` "
                "is already M x 1; the extra `.T` makes a 1 x M row and breaks the V-gradient. "
                "Use `_herm(A_radar) @ A_radar` or `A_radar.conj().T @ A_radar` to obtain an M x M matrix."
            )
    for match in re.finditer(r"_herm\(\s*A_radar\s*\)\.T\s*@\s*A_radar\b", source, flags=re.IGNORECASE):
        line_no = source.count("\n", 0, match.start()) + 1
        errors.append(
            f"generated_plugin.py:{line_no} uses `_herm(A_radar).T @ A_radar`, which is a double-transpose "
            "RIS/radar quadratic-form bug. If A_radar is 1 x M and V is M x M, use `_herm(A_radar) @ A_radar` "
            "for the M x M V-gradient/template."
        )
    return errors


def _phase24_contract_has_no_rho(method_contract_texts: dict[str, str]) -> bool:
    for method_id, contract_text_raw in method_contract_texts.items():
        method_key = str(method_id).strip().lower()
        contract_text = str(contract_text_raw or method_id).lower()
        if (
            method_key in {"no_rho", "no_structural_sep", "no_structural_separation"}
            or "no_rho" in method_key
            or "no structural separation" in contract_text
            or bool(re.search(r"\brho\s*=\s*0(?:\.0+)?(?![\d.])", contract_text))
        ):
            return True
    return False


def analyze_phase24_no_rho_partition_antipattern(source: str, method_contract_texts: dict[str, str]) -> list[str]:
    """Reject helpers that make a no-rho ablation impossible by construction."""
    if not source.strip() or not _phase24_contract_has_no_rho(method_contract_texts):
        return []
    if "m_eh" not in source.lower() and "partition" not in source.lower():
        return []
    errors: list[str] = []
    patterns = [
        r"\bM_eh\s*=\s*max\(\s*1\s*,\s*int\(\s*np\.floor\(\s*rho\s*\*\s*M\s*\)\s*\)\s*\)",
        r"\bM_eh\s*=\s*max\(\s*1\s*,\s*int\(\s*floor\(\s*rho\s*\*\s*M\s*\)\s*\)\s*\)",
        r"\bM_eh\s*=\s*max\(\s*1\s*,\s*int\(\s*rho\s*\*\s*M\s*\)\s*\)",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, source, flags=re.IGNORECASE):
            line_no = source.count("\n", 0, match.start()) + 1
            errors.append(
                f"generated_plugin.py:{line_no} forces `M_eh` to be at least one even when rho=0. "
                "The validation plan contains a no-rho/no-structural-separation ablation, so partition helpers "
                "must return M_eh=0 for rho <= 0 and reserve max(1, ...) only for strictly positive rho when needed."
            )
    return errors


def _phase24_extract_method_branch_source(method_source: str, method_id: str) -> str:
    """Best-effort extraction of an if/elif branch for one method id."""
    lines = method_source.splitlines()
    literal_pattern = re.compile(rf"['\"]{re.escape(method_id)}['\"]", flags=re.IGNORECASE)
    for idx, line in enumerate(lines):
        stripped = line.lstrip()
        if not literal_pattern.search(line):
            continue
        if not (stripped.startswith("if ") or stripped.startswith("elif ") or stripped.startswith("case ")):
            continue
        branch_indent = len(line) - len(stripped)
        branch_lines = [line]
        for next_line in lines[idx + 1 :]:
            next_stripped = next_line.lstrip()
            next_indent = len(next_line) - len(next_stripped)
            if next_stripped and next_indent <= branch_indent and (
                next_stripped.startswith("elif ")
                or next_stripped.startswith("else:")
                or next_stripped.startswith("case ")
            ):
                break
            branch_lines.append(next_line)
        return "\n".join(branch_lines)
    return method_source


def analyze_phase24_method_solution_source_semantics(method_source: str, method_contract_texts: dict[str, str]) -> list[str]:
    """Catch obvious fabricated ablations before spending time on smoke/paper sweeps."""
    if not method_source.strip() or not method_contract_texts:
        return []
    errors: list[str] = []
    nonzero_number = r"(?:0\.(?:0*[1-9]\d*)|[1-9]\d*(?:\.\d*)?)(?:e[-+]?\d+)?"
    for method_id, contract_text_raw in method_contract_texts.items():
        method_key = str(method_id).strip().lower()
        contract_text = str(contract_text_raw or method_id).lower()
        no_rho_contract = (
            method_key in {"no_rho", "no_structural_sep", "no_structural_separation"}
            or "no_rho" in method_key
            or "no structural separation" in contract_text
            or bool(re.search(r"\brho\s*=\s*0(?:\.0+)?(?![\d.])", contract_text))
        )
        if not no_rho_contract:
            continue
        branch_source = _phase24_extract_method_branch_source(method_source, method_id)
        literal_nonzero_rho = re.search(
            rf"(?:state\s*\[\s*['\"](?:rho|optimal_rho)['\"]\s*\]|(?<![\w.])(?:rho|rho_new|fixed_rho))\s*=\s*{nonzero_number}\b",
            branch_source,
            flags=re.IGNORECASE,
        )
        if literal_nonzero_rho:
            errors.append(
                f"method_solution branch for `{method_id}` assigns a nonzero rho literal "
                f"(`{literal_nonzero_rho.group(0).strip()}`). A no-rho/no-structural-separation ablation must set "
                "rho/optimal_rho to 0.0 and expose zero EH partition diagnostics when applicable; rho=0.5 or rho=1.0 "
                "is a different fixed-rho heuristic, not this ablation."
            )
    return errors


def _phase24_safe_field_name(value: str) -> str:
    name = re.sub(r"\W+", "_", str(value)).strip("_")
    if not name:
        return "field"
    if name[0].isdigit():
        name = f"field_{name}"
    return name


def _phase24_flatten_plan_fields(prefix: str, value: Any, out: set[str]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            safe_key = _phase24_safe_field_name(str(key))
            next_prefix = f"{prefix}_{safe_key}" if prefix else safe_key
            out.add(next_prefix)
            _phase24_flatten_plan_fields(next_prefix, child, out)


def _phase24_concept_appears(source_lower: str, concept: str) -> bool:
    term = str(concept or "").strip().lower()
    if not term:
        return False
    if term == "cache":
        topic_cache_patterns = [
            r"\bcache[-\s]?aided\b",
            r"\bcontent\s+cach(?:e|ing)\b",
            r"\bwireless\s+cach(?:e|ing)\b",
            r"\bedge\s+cach(?:e|ing)\b",
        ]
        return any(re.search(pattern, source_lower) for pattern in topic_cache_patterns)
    escaped = re.escape(term).replace(r"\_", r"[_\s-]")
    return bool(re.search(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", source_lower))


def validate_phase2_phase24_schema_alignment(run_dir: Path) -> dict[str, Any]:
    phase24_dir = run_dir / "phase2-4"
    solver_dir = phase24_dir / "solver"
    plan_payload = {}
    try:
        plan_payload = yaml.safe_load(read_text(phase24_dir / "validation_plan.yaml")) or {}
    except Exception:
        plan_payload = {}
    if not isinstance(plan_payload, dict):
        plan_payload = {}
    canonical = plan_payload.get("canonical_config", {})
    if not isinstance(canonical, dict):
        canonical = {}

    known_attrs = {
        "fields",
        "flat_fields",
        "validation_plan",
        "case_name",
        "case_id",
        "swept_param",
        "swept_value",
        "scenario_name",
        "_model_cache",
    }
    known_attrs.update(str(key) for key in canonical.keys())
    _phase24_flatten_plan_fields("", canonical, known_attrs)

    source = _phase24_combined_generated_source(solver_dir) or read_text(solver_dir / "generated_plugin.py")
    try:
        tree = ast.parse(source, filename=str(solver_dir / "generated_plugin.py"))
    except SyntaxError:
        return {"ok": True, "errors": [], "warnings": []}

    used_attrs: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "problem":
            used_attrs.add(node.attr)

    allowed_dynamic_methods = {"get", "clone_with"}
    allowed_solver_attrs = {"solve", "status", "value"}
    unknown_attrs = sorted(attr for attr in used_attrs if attr not in known_attrs and attr not in allowed_dynamic_methods)
    unknown_attrs = [attr for attr in unknown_attrs if attr not in allowed_solver_attrs]

    guardrails = plan_payload.get("semantic_guardrails", {})
    if not isinstance(guardrails, dict):
        guardrails = {}
    forbidden = [str(item).strip().lower() for item in guardrails.get("forbidden_concepts", []) if str(item).strip()]
    source_for_concept_scan = "\n".join(
        line
        for line in source.splitlines()
        if "forbidden_concepts" not in line.lower() and "semantic_guardrails" not in line.lower()
    )
    source_lower = source_for_concept_scan.lower()
    forbidden_hits = sorted(term for term in forbidden if _phase24_concept_appears(source_lower, term))

    warnings: list[str] = []
    required = [str(item).strip().lower() for item in guardrails.get("required_concepts", []) if str(item).strip()]
    missing_required = sorted(term for term in required if not _phase24_concept_appears(source_lower, term))
    if missing_required:
        warnings.append(f"generated Phase24 source does not visibly mention required concepts: {missing_required}")

    errors: list[str] = []
    if unknown_attrs:
        errors.append(
            "generated Phase24 source accesses undeclared ProblemData fields. "
            f"Use problem.get('path.to.field', default) or declare fields in validation_plan.yaml. Unknown: {unknown_attrs}"
        )
    if forbidden_hits:
        errors.append(f"generated Phase24 source contains forbidden concepts from semantic_guardrails: {forbidden_hits}")
    return {"ok": not errors, "errors": errors, "warnings": warnings}


def _phase24_contract_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        return list(value.values())
    return []


def _phase24_contract_method_id(item: Any) -> str:
    if isinstance(item, dict):
        for key in ("id", "internal_name", "name", "method"):
            value = str(item.get(key) or "").strip()
            if value:
                return value
        return ""
    return str(item or "").strip()


def _phase24_contract_metric_name(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("name", "field", "column", "metric", "y_metric"):
            nested = value.get(key)
            if isinstance(nested, dict):
                nested_name = _phase24_contract_metric_name(nested)
                if nested_name:
                    return nested_name
            text = str(nested or "").strip()
            if text:
                return text
        return ""
    return str(value or "").strip()


def _phase24_figure_y_metric(figure: dict[str, Any]) -> str:
    for key in ("y_metric", "metric", "primary_metric"):
        metric = _phase24_contract_metric_name(figure.get(key))
        if metric:
            return metric
    encoding = figure.get("encoding", {})
    if isinstance(encoding, dict):
        y_encoding = encoding.get("y", {})
        metric = _phase24_contract_metric_name(y_encoding)
        if metric:
            return metric
    return ""


def _phase24_figure_required_sweep(figure: dict[str, Any]) -> str:
    for key in ("required_sweep", "sweep_id", "sweep", "sweep_param", "swept_param"):
        value = str(figure.get(key) or "").strip()
        if value:
            return value
    encoding = figure.get("encoding", {})
    if isinstance(encoding, dict):
        x_encoding = encoding.get("x", {})
        if isinstance(x_encoding, dict):
            for key in ("sweep_param", "field", "variable", "canonical_path"):
                value = str(x_encoding.get(key) or "").strip()
                if value:
                    return value
    for key in ("x_field", "x_metric", "x_axis"):
        value = figure.get(key)
        if isinstance(value, dict):
            extracted = _phase24_contract_metric_name(value)
        else:
            extracted = str(value or "").strip()
        if extracted:
            return extracted
    return ""


def _phase24_plan_sweep_names(plan: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    sweeps = plan.get("sweep_definitions", [])
    if isinstance(sweeps, dict):
        for key, value in sweeps.items():
            if str(key).strip():
                names.add(str(key).strip())
            if isinstance(value, dict):
                for field in ("id", "name", "variable", "canonical_path", "raw_variable", "display_name"):
                    text = str(value.get(field) or "").strip()
                    if text:
                        names.add(text)
    elif isinstance(sweeps, list):
        for item in sweeps:
            if isinstance(item, dict):
                for field in ("id", "name", "variable", "canonical_path", "raw_variable", "display_name"):
                    text = str(item.get(field) or "").strip()
                    if text:
                        names.add(text)
            elif str(item).strip():
                names.add(str(item).strip())
    return names


def _phase24_evidence_figures(plan: dict[str, Any], evidence: dict[str, Any]) -> list[Any]:
    figures: list[Any] = []
    for source in (
        evidence.get("figures"),
        evidence.get("figure_targets"),
        plan.get("figure_targets"),
        plan.get("figures"),
        plan.get("figure_specs"),
    ):
        figures.extend(_phase24_contract_list(source))
    deduped: list[Any] = []
    seen: set[str] = set()
    for item in figures:
        if isinstance(item, dict):
            key = str(item.get("id") or item.get("figure_id") or json.dumps(item, sort_keys=True, default=str))
        else:
            key = str(item)
        if key not in seen:
            seen.add(key)
            deduped.append(item)
    return deduped


def _phase24_evidence_tables(plan: dict[str, Any], evidence: dict[str, Any]) -> list[Any]:
    tables: list[Any] = []
    for source in (
        evidence.get("tables"),
        evidence.get("table_targets"),
        plan.get("table_targets"),
        plan.get("table_target"),
        plan.get("tables"),
        plan.get("table_specs"),
    ):
        tables.extend(_phase24_contract_list(source))
    return tables


def _phase24_required_columns_from_evidence(evidence: dict[str, Any]) -> set[str]:
    return {
        str(item).strip()
        for item in evidence.get("required_result_columns", [])
        if str(item).strip()
    }


def _phase24_metric_is_objective_like(metric: str) -> bool:
    text = str(metric or "").strip().lower().replace("-", "_")
    objective_names = {
        "objective",
        "objective_value",
        "weighted_objective",
        "utility",
        "weighted_utility",
        "eta_service_level",
        "service_level",
        "normalized_service_level",
        "achieved_tau",
        "tau_star",
        "service_margin",
        "normalized_service_margin",
        "min_normalized_service_margin",
        "achieved_margin",
    }
    return text in objective_names or (
        any(token in text for token in ("objective", "utility"))
        and not any(token in text for token in ("violation", "gap", "residual"))
    ) or (
        "margin" in text and not any(token in text for token in ("violation", "gap", "residual"))
    ) or (
        "service" in text
        and "level" in text
        and not any(token in text for token in ("sinr", "sensing", "eh", "energy", "harvest", "violation", "gap", "residual"))
    )


def _phase24_metric_is_runtime_like(metric: str) -> bool:
    text = str(metric or "").strip().lower()
    return any(token in text for token in ("runtime", "solve_time", "solver_time", "elapsed", "iteration", "iter_count"))


def _phase24_metric_is_solver_diagnostic_y(metric: str) -> bool:
    text = str(metric or "").strip().lower().replace("-", "_")
    return any(
        token in text
        for token in (
            "violation",
            "feasible",
            "feasibility",
            "residual",
            "solver_status",
            "status",
            "runtime",
            "solve_time",
            "solver_time",
            "elapsed",
            "iteration",
            "iter_count",
            "convergence_gap",
            "duality_gap",
        )
    )


def _phase24_metric_is_power_usage_like(metric: str) -> bool:
    text = str(metric or "").strip().lower().replace("-", "_")
    compact = re.sub(r"[^a-z0-9]+", "", text)
    if any(token in text for token in ("transmit_power", "total_power", "sum_power", "power_used", "tx_power")):
        return True
    if re.search(r"(^|_)p(_)?(tx|tot|total|sum)(_|$)", text):
        return True
    return any(compact.startswith(prefix) for prefix in ("ptx", "ptot", "ptotal", "psum"))


def _phase24_sweep_is_resource_upper_bound(text: str) -> bool:
    lowered = str(text or "").strip().lower().replace("-", "_")
    compact = re.sub(r"[^a-z0-9]+", "", lowered)
    if any(token in lowered for token in ("p_max", "pmax", "power_budget", "power_cap", "power_limit", "ap_budget")):
        return True
    return any(token in compact for token in ("pmax", "powerbudget", "powercap", "powerlimit", "apbudget"))


def _phase24_label_looks_internal(label: str) -> bool:
    text = str(label or "").strip()
    return bool(re.search(r"\b(?:system|constraints|requirements|optimization|ambiguity|channel|rectifier|uncertainty)\.", text))


def _phase24_axis_label_has_descriptive_context(label: str) -> bool:
    """Paper captions/axes should not be bare symbols such as `$N$`."""

    text = str(label or "").strip()
    if not text:
        return False
    without_math = re.sub(r"\$[^$]*\$", " ", text)
    without_math = re.sub(r"\\[A-Za-z]+(?:\{[^{}]*\})?", " ", without_math)
    words = re.findall(r"[A-Za-z]{2,}", without_math)
    stopwords = {"rm", "mathrm", "mathit", "max", "min", "sec", "sum", "tot", "dc", "rf"}
    return any(word.lower() not in stopwords for word in words)


def _phase24_metric_is_physical_kpi(metric: str) -> bool:
    text = str(metric or "").strip().lower().replace("-", "_")
    compact = re.sub(r"[^a-z0-9]+", "", text)
    if not text or _phase24_metric_is_objective_like(text) or _phase24_metric_is_runtime_like(text):
        return False
    if text in {"physical_utility", "system_utility", "physical_kpi", "system_performance"}:
        return True
    if re.search(r"(^|_)p(_)?(tx|tot|total|sum|rx|rf|dc|eh)(_|$)", text):
        return True
    if any(compact.startswith(prefix) for prefix in ("ptx", "ptot", "ptotal", "psum", "prx", "prf", "pdc", "peh")):
        return True
    if any(token in compact for token in ("wsr", "bpshz", "bpsperhz", "bitshz", "spectralefficiency")):
        return True
    if (
        any(token in text for token in ("fraction", "ratio", "allocation", "rho"))
        and any(token in text for token in ("power", "energy", "resource", "time", "bandwidth", "shared", "split", "splitting"))
    ):
        return True
    return any(
        token in text
        for token in (
            "rate",
            "bps_hz",
            "bpshz",
            "sinr",
            "snr",
            "ber",
            "outage",
            "throughput",
            "spectral",
            "energy",
            "power",
            "harvest",
            "crb",
            "mse",
            "margin",
            "latency",
            "illumination",
            "fairness",
            "min_user",
        )
    )


def _phase24_semantic_family(text: str) -> str:
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
        "rate": ("sum-rate", "sum rate", "communication rate", "weighted sum communication rate", "throughput", "spectral efficiency", "bps/hz", "sinr rate", " r_k", "r_k"),
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


def _phase24_objective_metric_alignment(run_dir: Path, primary_metric: str) -> dict[str, Any]:
    metric = str(primary_metric or "").strip()
    if not metric:
        return {"requires_design_revision": False}
    if any(token in metric.lower() for token in ("objective", "utility", "weighted", "eta_service", "service_level", "service_margin", "normalized_service", "tau")):
        return {"requires_design_revision": False}
    math_contract = read_json(Path(run_dir) / "phase2-1" / "mathematical_contract.frozen.json") or read_json(Path(run_dir) / "phase2-1" / "mathematical_contract.json") or {}
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
    objective_family = _phase24_semantic_family(objective_text)
    if not objective_family:
        objective_text = "\n".join(
            [
                json.dumps(objective if isinstance(objective, dict) else {}, ensure_ascii=False),
                read_text(Path(run_dir) / "phase2-2" / "reformulation_path.md")[:4000],
                read_text(Path(run_dir) / "phase2-3" / "algorithm.md")[:4000],
            ]
        )
        objective_family = _phase24_semantic_family(objective_text)
    metric_family = _phase24_semantic_family(metric.replace("_", " "))
    if not objective_family or not metric_family or objective_family == metric_family:
        return {"requires_design_revision": False}
    return {
        "requires_design_revision": True,
        "reason": "primary_metric_family_mismatch_with_frozen_objective",
        "objective_family": objective_family,
        "primary_metric": metric,
        "primary_metric_family": metric_family,
        "objective_text_excerpt": objective_text[:600],
    }


def _phase24_metric_family_advice(objective_family: str) -> str:
    family = str(objective_family or "").strip().lower()
    if family == "rate":
        return "Use a rate/fairness KPI such as min_user_rate, min_user_rate_bpsHz, min_spectral_efficiency_bpsHz, or sum_rate_bpsHz when the frozen objective is a rate objective."
    if family == "energy":
        return "Use an energy-harvesting KPI such as min_harvested_dc_mW or harvested_energy_mW when the frozen objective is an EH objective."
    if family == "sensing":
        return "Use a sensing KPI such as crb, sensing_mse, sensing_snr_dB, or sensing_illumination_mW when the frozen objective is a sensing objective."
    if family == "power":
        return "Use a resource-usage KPI such as P_tx_mW, sum_power_mW, or total_tx_power_mW when the frozen objective is resource minimization."
    if family == "secrecy":
        return "Use a secrecy KPI such as min_secrecy_rate_bpsHz or worst_case_min_secrecy_rate_bpsHz when the frozen objective is secrecy-rate optimization."
    if family == "utility":
        return "Use the paper-defined service utility KPI such as eta_service_level, service_margin_tau, min_normalized_service_margin, or a clearly defined utility alias."
    return "Regenerate the experiment design with a KPI aligned to the frozen mathematical objective."


def _phase24_metric_alignment_error(run_dir: Path, metric: str, *, owner: str) -> str:
    alignment = _phase24_objective_metric_alignment(Path(run_dir), metric)
    if not alignment.get("requires_design_revision"):
        return ""
    return (
        f"{owner} `{metric}` does not match the frozen objective family "
        f"`{alignment.get('objective_family')}`; "
        f"{_phase24_metric_family_advice(str(alignment.get('objective_family') or ''))}"
    )


def _phase24_design_contract_figures(design_contract: dict[str, Any]) -> list[Any]:
    figures: list[Any] = []
    for source in (
        design_contract.get("figure_contracts"),
        design_contract.get("figures"),
        design_contract.get("figure_targets"),
    ):
        figures.extend(_phase24_contract_list(source))
    return figures


def _phase24_validate_experiment_design_contract_alignment(run_dir: Path) -> list[str]:
    phase24_dir = Path(run_dir) / "phase2-4"
    design_contract = read_json(phase24_dir / "experiment_design_contract.json") or {}
    if not isinstance(design_contract, dict) or not design_contract:
        return []
    errors: list[str] = []
    design_kpis = [
        str(item or "").strip()
        for item in design_contract.get("primary_physical_kpis", [])
        if str(item or "").strip()
    ]
    if design_kpis:
        aligned_kpis = [
            metric
            for metric in design_kpis
            if not _phase24_objective_metric_alignment(Path(run_dir), metric).get("requires_design_revision")
        ]
        if not aligned_kpis:
            objective_family = str(
                _phase24_objective_metric_alignment(Path(run_dir), design_kpis[0]).get("objective_family") or ""
            )
            errors.append(
                "experiment_design_contract.primary_physical_kpis contains no KPI aligned with the frozen objective; "
                f"{_phase24_metric_family_advice(objective_family)}"
            )
    for index, raw_figure in enumerate(_phase24_design_contract_figures(design_contract), start=1):
        if not isinstance(raw_figure, dict):
            continue
        figure_id = str(raw_figure.get("id") or raw_figure.get("figure_id") or f"design_figure_{index}").strip()
        y_metric = _phase24_figure_y_metric(raw_figure)
        if not y_metric:
            continue
        error = _phase24_metric_alignment_error(
            Path(run_dir),
            y_metric,
            owner=f"experiment_design_contract.{figure_id}.y_metric",
        )
        if error:
            errors.append(error)
    return errors


def validate_phase24_evidence_contract_design(run_dir: Path) -> dict[str, Any]:
    """Validate that Phase 2.4 has a reusable paper-evidence plan before code runs."""
    phase24_dir = Path(run_dir) / "phase2-4"
    try:
        plan = yaml.safe_load(read_text(phase24_dir / "validation_plan.yaml")) or {}
    except Exception as exc:
        return {"ok": False, "errors": [f"validation_plan.yaml is not valid YAML: {exc}"], "warnings": []}
    if not isinstance(plan, dict):
        return {"ok": False, "errors": ["validation_plan.yaml did not parse to a mapping"], "warnings": []}

    evidence = _phase24_research_evidence(plan)
    if not isinstance(evidence, dict) or not evidence:
        return {
            "ok": False,
            "errors": [
                "research_evidence_contract is missing; paper_evidence_contract is missing legacy alias. Phase 2.4 must declare a frozen evidence contract before generating experiment code."
            ],
            "warnings": [],
        }

    errors: list[str] = []
    warnings: list[str] = []
    primary_payload = evidence.get("primary_metric") if isinstance(evidence.get("primary_metric"), dict) else {}
    primary_metric = str(primary_payload.get("name") or primary_payload.get("metric") or evidence.get("primary_metric") or "").strip()
    alignment = _phase24_objective_metric_alignment(Path(run_dir), primary_metric)
    if alignment.get("requires_design_revision"):
        errors.append(
            "research_evidence_contract.primary_metric "
            f"`{primary_metric}` does not match the frozen objective family "
            f"`{alignment.get('objective_family')}`; regenerate the Phase 2.4 experiment design with an objective-aligned KPI "
            f"before code generation. {_phase24_metric_family_advice(str(alignment.get('objective_family') or ''))}"
        )
    errors.extend(_phase24_validate_experiment_design_contract_alignment(Path(run_dir)))
    method_entries = _phase24_contract_list(evidence.get("compared_methods"))
    method_ids: set[str] = set()
    proposed_ids: set[str] = set()
    comparison_ids: set[str] = set()
    for item in method_entries:
        method_id = _phase24_contract_method_id(item)
        if not method_id:
            continue
        method_ids.add(method_id)
        role_text = ""
        if isinstance(item, dict):
            role_text = " ".join(
                str(item.get(key) or "")
                for key in ("role", "scientific_purpose", "display_name_short", "display_name_long")
            ).lower()
        else:
            role_text = method_id.lower()
        if "proposed" in role_text or method_id.lower() == "proposed":
            proposed_ids.add(method_id)
        elif any(token in role_text for token in ("baseline", "benchmark", "heuristic", "ablation", "upper_bound")) or method_id.lower() == "baseline":
            comparison_ids.add(method_id)
        if isinstance(item, dict):
            missing_method_fields = [
                field
                for field in ("role", "scientific_purpose", "implementation_hint", "fairness_rule")
                if not str(item.get(field) or "").strip()
            ]
            if missing_method_fields:
                errors.append(
                    f"compared_methods `{method_id}` is missing executable contract fields: "
                    + ", ".join(missing_method_fields)
                )
            combined_method_text = " ".join(
                str(item.get(field) or "")
                for field in ("scientific_purpose", "implementation_hint", "fairness_rule", "display_name_short", "display_name_long")
            ).strip().lower()
            placeholder_terms = {"tbd", "placeholder", "generic baseline", "generic method", "same as proposed"}
            if any(term in combined_method_text for term in placeholder_terms):
                errors.append(f"compared_methods `{method_id}` contains placeholder/generic method text.")
            if method_id.lower() in {"baseline", "proposed"} and len(combined_method_text.split()) < 8:
                errors.append(
                    f"compared_methods `{method_id}` is too generic; define what it changes, what it tests, and how it is fair."
                )

    if not method_ids:
        errors.append("research_evidence_contract.compared_methods is empty.")
    if not proposed_ids:
        errors.append("research_evidence_contract.compared_methods must include exactly identifiable proposed method entry.")
    if not comparison_ids:
        errors.append("research_evidence_contract.compared_methods must include at least one benchmark, baseline, heuristic, or ablation.")

    required_columns = _phase24_required_columns_from_evidence(evidence)
    metadata_requirements = {
        "method": {"method"},
        "seed": {"seed", "realization_id", "trial_id", "mc_seed", "sample_id"},
        "scenario_name": {"scenario_name", "case_name"},
        "swept_param": {"swept_param", "sweep_param"},
        "swept_value": {"swept_value"},
    }
    missing_metadata = [
        column
        for column, aliases in metadata_requirements.items()
        if not required_columns.intersection(aliases)
    ]
    if missing_metadata:
        errors.append(
            "research_evidence_contract.required_result_columns must include reusable experiment metadata columns: "
            + ", ".join(missing_metadata)
        )

    figures = _phase24_evidence_figures(plan, evidence)
    if not figures:
        errors.append("No figure evidence targets were found in research_evidence_contract.figures or top-level figure_targets.")
    sweep_names = _phase24_plan_sweep_names(plan)
    physical_kpi_figures = 0
    objective_like_figures: list[str] = []

    for index, raw_figure in enumerate(figures, start=1):
        if not isinstance(raw_figure, dict):
            errors.append(f"figure target #{index} is not a mapping.")
            continue
        figure_id = str(raw_figure.get("id") or raw_figure.get("figure_id") or f"figure_{index}").strip()
        claim_text = " ".join(
            str(raw_figure.get(key) or "")
            for key in ("claim", "claim_id", "paper_claim_id", "purpose", "evidence_rationale")
        ).strip()
        if not claim_text:
            errors.append(f"{figure_id}: missing claim/evidence rationale. Each figure must prove a specific paper claim.")
        intent = str(raw_figure.get("chart_intent") or raw_figure.get("intent") or "").strip().lower()
        if not intent:
            errors.append(f"{figure_id}: missing chart_intent.")
        axis_labels = raw_figure.get("axis_labels") if isinstance(raw_figure.get("axis_labels"), dict) else {}
        x_axis_label = str(axis_labels.get("x") or raw_figure.get("x_axis_label") or raw_figure.get("x_display_name") or "").strip()
        y_axis_label = str(axis_labels.get("y") or raw_figure.get("y_axis_label") or raw_figure.get("y_display_name") or "").strip()
        if not x_axis_label or not y_axis_label:
            warnings.append(
                f"{figure_id}: missing paper-facing axis_labels.x/axis_labels.y. "
                "Phase 2.4 must freeze notation-first figure labels before code generation; Phase 2.5 must not infer them from schema paths."
            )
        elif _phase24_label_looks_internal(x_axis_label) or _phase24_label_looks_internal(y_axis_label):
            errors.append(
                f"{figure_id}: axis labels expose internal schema/path names. Use paper notation or concise public KPI symbols instead."
            )
        elif not _phase24_axis_label_has_descriptive_context(x_axis_label):
            errors.append(
                f"{figure_id}: axis_labels.x is a bare symbol. Use a paper-facing physical phrase plus notation, "
                "for example `transmit power budget $P_{\\max}$` or `number of users $K$`, not only `$P_{\\max}$` or `$K$`."
            )
        if not str(raw_figure.get("chart_choice_rationale") or raw_figure.get("evidence_rationale") or "").strip():
            warnings.append(f"{figure_id}: missing chart_choice_rationale/evidence_rationale.")
        if not str(raw_figure.get("expected_trend") or raw_figure.get("trend_hypothesis") or "").strip():
            warnings.append(f"{figure_id}: missing expected_trend/trend_hypothesis.")
        if not str(raw_figure.get("active_regime_note") or "").strip():
            warnings.append(f"{figure_id}: missing active_regime_note.")
        y_metric = _phase24_figure_y_metric(raw_figure)
        if not y_metric:
            errors.append(f"{figure_id}: missing y_metric/metric field.")
        else:
            if _phase24_metric_is_solver_diagnostic_y(y_metric):
                errors.append(
                    f"{figure_id}: y_metric `{y_metric}` is a solver/feasibility diagnostic. "
                    "Final paper figures must use the objective or a physical system KPI; keep violations, feasibility, residuals, runtime, and status in diagnostic columns only."
                )
            if _phase24_metric_is_physical_kpi(y_metric):
                physical_kpi_figures += 1
            if _phase24_metric_is_objective_like(y_metric):
                objective_like_figures.append(figure_id)
                if intent not in {"overall_utility", "utility_comparison", "main_comparison", "mechanism_sensitivity"}:
                    warnings.append(
                        f"{figure_id}: y_metric `{y_metric}` is objective-like; keep at least one other promoted figure on a physical KPI and explain the objective's paper-defined meaning."
                    )
            if _phase24_metric_is_runtime_like(y_metric) and intent not in {"scalability", "convergence"}:
                errors.append(
                    f"{figure_id}: y_metric `{y_metric}` is a runtime/iteration diagnostic, not evidence for `{intent or 'unspecified'}`."
                )
            alignment_error = _phase24_metric_alignment_error(
                Path(run_dir),
                y_metric,
                owner=f"{figure_id}: y_metric",
            )
            if alignment_error:
                errors.append(alignment_error)
            if required_columns and not any(alias in required_columns for alias in _phase24_metric_aliases(y_metric)):
                errors.append(
                    f"{figure_id}: y_metric `{y_metric}` is not covered by research_evidence_contract.required_result_columns."
                )
        sweep_name = _phase24_figure_required_sweep(raw_figure)
        sweep_descriptor = " ".join(
            str(raw_figure.get(key) or "")
            for key in ("required_sweep", "required_sweep_param", "x_field", "claim", "chart_intent")
        )
        if y_metric and _phase24_metric_is_power_usage_like(y_metric) and _phase24_sweep_is_resource_upper_bound(sweep_descriptor):
            errors.append(
                f"{figure_id}: y_metric `{y_metric}` is a resource-usage KPI but the x-axis sweep appears to be a resource upper bound/budget. "
                "For paper evidence, sweep an active demand, load, channel-severity, uncertainty, or mechanism parameter instead."
            )
        if not sweep_name:
            warnings.append(f"{figure_id}: no required_sweep/x-axis sweep was declared; Phase 2.5 may not know how to test coverage.")
        elif sweep_names and sweep_name not in sweep_names:
            warnings.append(
                f"{figure_id}: required sweep `{sweep_name}` is not an exact sweep_definitions id/variable/canonical_path. "
                "This is allowed only if it is an emitted result column derived from a declared sweep."
            )
        methods_to_run = [_phase24_contract_method_id(item) for item in _phase24_contract_list(raw_figure.get("methods_to_run"))]
        methods_to_run = [method for method in methods_to_run if method]
        if len(methods_to_run) < 2:
            errors.append(f"{figure_id}: methods_to_run must include the proposed method and at least one comparison method.")
        elif proposed_ids and not set(methods_to_run).intersection(proposed_ids | {"proposed"}):
            errors.append(f"{figure_id}: methods_to_run does not include the declared proposed method.")
        elif comparison_ids and not set(methods_to_run).intersection(comparison_ids | {"baseline"}):
            errors.append(f"{figure_id}: methods_to_run does not include any declared comparison method.")
        unknown_methods = sorted(method for method in methods_to_run if method not in method_ids and method not in {"proposed", "baseline"})
        if unknown_methods:
            warnings.append(f"{figure_id}: methods_to_run contains methods not declared in compared_methods: {unknown_methods}")

    non_diagnostic_performance_figures = physical_kpi_figures + len(objective_like_figures)
    if figures and non_diagnostic_performance_figures == 0:
        errors.append(
            "All figure targets use objective/runtime/diagnostic metrics. At least one figure must use a physical KPI tied to the wireless claim."
        )
    if len(figures) >= 2 and non_diagnostic_performance_figures < 2:
        errors.append(
            "At least two final paper figure targets must use non-diagnostic system-performance y_metrics; "
            "keep feasibility, violation, runtime, and convergence as diagnostics."
        )
    if len(objective_like_figures) == len(figures) and figures:
        errors.append(
            "Every figure target is objective-like. Do not build the paper around weighted-objective plots; choose claim-specific physical KPIs."
        )

    tables = _phase24_evidence_tables(plan, evidence)
    if not tables:
        if not bool(evidence.get("tables_optional", False)):
            warnings.append("No table target was declared; downstream table writing should stay disabled or add an explicit table contract.")

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "num_declared_methods": len(method_ids),
        "num_figure_targets": len(figures),
        "num_table_targets": len(tables),
        "required_columns": sorted(required_columns),
    }


def validate_phase24_evidence_contract_outputs(run_dir: Path) -> dict[str, Any]:
    phase24_dir = Path(run_dir) / "phase2-4"
    solver_dir = phase24_dir / "solver"
    try:
        plan = yaml.safe_load(read_text(phase24_dir / "validation_plan.yaml")) or {}
    except Exception as exc:
        return {"ok": False, "errors": [f"validation_plan.yaml is not valid YAML: {exc}"], "warnings": []}
    if not isinstance(plan, dict):
        return {"ok": False, "errors": ["validation_plan.yaml did not parse to a mapping"], "warnings": []}
    evidence = _phase24_research_evidence(plan)
    if not isinstance(evidence, dict) or not evidence:
        return {"ok": False, "errors": ["research_evidence_contract is missing"], "warnings": []}
    required = [
        str(item).strip()
        for item in evidence.get("required_result_columns", [])
        if str(item).strip()
    ]
    if not required:
        return {"ok": False, "errors": ["research_evidence_contract.required_result_columns is empty"], "warnings": []}
    results_path = solver_dir / "outputs" / "validation_results.csv"
    if not results_path.exists():
        return {"ok": False, "errors": [f"missing validation results csv: {results_path}"], "warnings": []}
    try:
        with results_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            headers = [str(item).strip() for item in (reader.fieldnames or [])]
            rows = [dict(row) for row in reader]
    except Exception as exc:
        return {"ok": False, "errors": [f"cannot read validation results csv: {exc}"], "warnings": []}
    metadata_aliases = {
        "method": {"method"},
        "seed": {"seed", "realization_id", "trial_id", "mc_seed", "sample_id"},
        "sweep_param": {"sweep_param", "swept_param"},
        "sweep_canonical_path": {"sweep_canonical_path", "swept_canonical_path"},
        "swept_param": {"swept_param", "sweep_param"},
        "swept_canonical_path": {"swept_canonical_path", "sweep_canonical_path"},
        "swept_value": {"swept_value"},
        "scenario_name": {"scenario_name", "case_name"},
        "lambda_1": {"lambda_1", "optimization_lambda_1", "lambda_crb", "optimization_lambda_vector_0", "optimization_lambda_vector_1"},
        "lambda_2": {"lambda_2", "optimization_lambda_2", "lambda_rate", "optimization_lambda_vector_1", "optimization_lambda_vector_2"},
        "lambda_3": {"lambda_3", "optimization_lambda_3", "lambda_eh", "optimization_lambda_vector_2", "optimization_lambda_vector_3"},
        "lambda_crb": {"lambda_crb", "optimization_lambda_vector_0", "optimization_lambda_vector_1"},
        "lambda_rate": {"lambda_rate", "optimization_lambda_vector_1", "optimization_lambda_vector_2"},
        "lambda_eh": {"lambda_eh", "optimization_lambda_vector_2", "optimization_lambda_vector_3"},
        "lambda_s_ratio": {"lambda_s_ratio", "optimization_lambda_s_ratio", "lambda_s", "optimization_lambda_s", "swept_value"},
        "Pmax_dBm": {"Pmax_dBm", "P_max_dBm", "system_Pmax_dBm", "system_P_max_dBm"},
        "P_max_dBm": {"P_max_dBm", "Pmax_dBm", "system_Pmax_dBm", "system_P_max_dBm"},
        "N_ref": {"N_ref", "system_N_ref", "Nr_ref", "RIS_N_ref", "N_reflecting"},
        "M_total": {"M_total", "ris_M_total"},
        "M": {"M", "system_M"},
        "M_ris": {"M_ris", "M", "system_M", "RIS_M", "ris_M_total"},
        "M_r": {"M_r", "Mr", "system_Mr", "system_M_r"},
        "M_e": {"M_e", "Me", "system_Me", "system_M_e"},
        "Me": {"Me", "M_e", "system_Me"},
        "Me_frac": {"Me_frac", "ris_Me_frac"},
        "alpha_ratio_sensing_comm": {"alpha_ratio_sensing_comm", "optimization_alpha_ratio_sensing_comm"},
        "target_RCS_dB": {"target_RCS_dB", "target_RCS_dBsm", "channels_target_RCS_dB", "channels_target_RCS_dBsm"},
        "Psi_sat_mW": {"Psi_sat_mW", "eh_sigmoid_params_Psi_sat_mW", "sigmoid_params_Psi_sat_mW"},
        "a_steepness": {"a_steepness", "EH_a_steepness", "eh_sigmoid_params_a_steepness", "sigmoid_params_a_steepness"},
        "sigmoid_a": {"sigmoid_a", "a_steepness", "EH_a_steepness", "eh_sigmoid_params_a_steepness", "sigmoid_params_a_steepness"},
        "sigmoid_steepness_a": {"sigmoid_steepness_a", "sigmoid_a", "a_steepness", "EH_a_steepness", "EH_steepness_a", "steepness_a", "actual_used_EH_a_steepness"},
        "b_offset": {"b_offset", "b_offset_W", "eh_sigmoid_params_b_offset_W", "sigmoid_params_b_offset_W"},
        "R_min_Mbps": {"R_min_Mbps", "cu_R_min_Mbps"},
        "K_users": {"K_users", "K", "system_K", "num_users", "users_K"},
        "E_min_mW": {"E_min_mW", "eh_E_min_mW"},
        "E_min_frac": {"E_min_frac", "constraints_E_min_frac", "constraints_E_min", "E_min", "E_min_mW", "constraints_E_min_mW"},
        "objective_value": {"objective_value", "objective"},
        "solver_status": {"solver_status", "status"},
        "solver_time_sec": {"solver_time_sec", "solve_time_sec"},
        "solver_time_ms": {"solver_time_ms", "runtime_ms", "solve_time_ms", "runtime_seconds", "solver_time_sec", "solve_time_sec"},
        "eigenvalue_ratio_V": {"eigenvalue_ratio_V", "eigenvalue_ratio", "rank_V_star", "rank_W", "rank_Wc"},
        "constraint_C1_active": {"constraint_C1_active", "rate_constraint_satisfied", "C1_active", "C1_viol"},
        "block_A_status": {"block_A_status", "solver_status", "status"},
        "block_B_status": {"block_B_status", "solver_status", "status"},
        "actual_used_lambda_s": {"actual_used_lambda_s", "actual_used_optimization_lambda_s", "lambda_s", "optimization_lambda_s"},
        "actual_used_lambda_c": {"actual_used_lambda_c", "actual_used_optimization_lambda_c", "lambda_c", "optimization_lambda_c"},
        "actual_used_lambda_p": {"actual_used_lambda_p", "actual_used_optimization_lambda_p", "lambda_p", "optimization_lambda_p"},
        "constraint_C2_violation_max": {"constraint_C2_violation_max", "C2_viol_sum", "C2_viol", "max_constraint_violation"},
        "constraint_C2_active": {"constraint_C2_active", "c2_sinr_viol", "C2_viol", "rate_constraint_satisfied"},
        "constraint_active_C1_power": {"constraint_active_C1_power", "power_budget_violation", "power_violation", "c1_power_viol", "C1_viol"},
        "constraint_active_C2_sinr": {"constraint_active_C2_sinr", "constraint_C2_active", "sinr_violation", "c2_sinr_viol", "C2_viol", "rate_constraint_satisfied"},
        "constraint_active_C3_radar": {"constraint_active_C3_radar", "radar_snr_violation", "c3_radar_viol", "C3_viol", "sensing_power_constraint_satisfied"},
        "constraint_active_C4_eh": {"constraint_active_C4_eh", "energy_violation", "c4_eh_viol", "C4_viol", "eh_constraint_satisfied"},
        "constraint_C3_active": {"constraint_C3_active", "sensing_power_constraint_satisfied", "C3_active", "C3_viol"},
        "constraint_C4_active": {"constraint_C4_active", "eh_constraint_satisfied", "C4_active", "C4_viol"},
        "constraint_C7_rho_interior": {"constraint_C7_rho_interior", "rho", "optimal_rho"},
        "Pin_at_solution_mW": {"Pin_at_solution_mW", "Pin_eh", "Pin_EH_partition_mW"},
        "initial_P_in_mW": {"initial_P_in_mW", "initial_Pin_mW", "P_in_initial_mW", "final_P_in_mW", "P_in_actual_mW"},
        "radar_snr_dB": {"radar_snr_dB", "radar_SNR_dB", "radar_snr", "radar_SNR", "sensing_snr_dB"},
        "sum_rate_bpsHz": {"sum_rate_bpsHz", "sum_rate", "rate_bpsHz", "R_c_bpsHz", "R_c", "rate"},
        "sum_power_W": {"sum_power_W", "total_power_W", "transmit_power_W", "power_W", "power_total_W"},
        "average_rho": {"average_rho", "rho_mean", "mean_rho", "optimal_rho", "rho_min", "rho_max"},
        "min_rho": {"min_rho", "rho_min"},
        "max_rho": {"max_rho", "rho_max"},
        "sum_rf_input_power_mW": {"sum_rf_input_power_mW", "rf_input_power_mW_sum", "input_power_mW_sum", "sum_q_mW"},
        "Pmax_W": {"Pmax_W", "system_Pmax", "system_Pmax_W", "actual_used_system_Pmax", "actual_used_system_Pmax_W"},
        "true_harvested_energy_mW": {"true_harvested_energy_mW", "harvested_energy_mW", "P_EH_mW", "eh_total_mW", "eh_power"},
        "actual_used_V_mm_diagonal": {"actual_used_V_mm_diagonal", "V_mm_diagonal", "rank_V_star", "M_Rx"},
        "actual_used_rho_partition_elements_EH": {"actual_used_rho_partition_elements_EH", "M_EH", "rho_rounded_elements_EH"},
        "actual_used_rho_partition_elements_Rx": {"actual_used_rho_partition_elements_Rx", "M_Rx", "rho_rounded_elements_Rx"},
    }
    def dynamic_column_aliases(column: str) -> set[str]:
        raw = str(column or "").strip()
        snake = re.sub(r"[^A-Za-z0-9]+", "_", raw).strip("_")
        aliases = {raw}
        if snake:
            aliases.update(
                {
                    snake,
                    f"actual_{snake}",
                    f"actual_used_{snake}",
                    f"diagnostics_actual_used_{snake}",
                }
            )
            if snake.startswith("actual_"):
                base = snake[len("actual_") :]
                aliases.update({base, f"actual_used_{base}", f"diagnostics_actual_used_{base}"})
            if snake.startswith("actual_used_"):
                base = snake[len("actual_used_") :]
                aliases.update({base, f"actual_{base}", f"diagnostics_actual_used_{base}"})
        return {alias for alias in aliases if alias}

    header_set = set(headers)
    missing: list[str] = []
    for column in required:
        aliases = set(metadata_aliases.get(column, {column})) | dynamic_column_aliases(column)
        has_vectorized_alias = any(
            any(header == f"{alias}_{index}" for alias in aliases)
            for header in header_set
            for index in range(20)
        )
        has_indexed_matrix_alias = any(
            any(header.startswith(f"{alias}_") for alias in aliases)
            for header in header_set
        )
        if not header_set.intersection(aliases) and not has_vectorized_alias and not has_indexed_matrix_alias:
            missing.append(column)
    errors = []
    warnings: list[str] = []
    if missing:
        errors.append(
            "validation_results.csv is missing research_evidence_contract required columns: "
            + ", ".join(missing)
        )
    placeholder_columns: list[str] = []
    for column in required:
        if column not in header_set:
            continue
        exact_values = [
            value
            for row in rows
            for value in [_phase24_float_cell(row.get(column))]
            if value is not None
        ]
        if not exact_values or max(abs(value) for value in exact_values) > 1.0e-12:
            continue
        aliases = [
            alias
            for alias in (set(metadata_aliases.get(column, {column})) | dynamic_column_aliases(column))
            if alias != column and alias in header_set
        ]
        nonzero_aliases = []
        for alias in aliases:
            alias_values = [
                value
                for row in rows
                for value in [_phase24_float_cell(row.get(alias))]
                if value is not None
            ]
            if alias_values and max(abs(value) for value in alias_values) > 1.0e-12:
                nonzero_aliases.append(alias)
        if nonzero_aliases:
            placeholder_columns.append(f"{column} (nonzero aliases: {', '.join(sorted(nonzero_aliases))})")
    if placeholder_columns:
        errors.append(
            "validation_results.csv appears to contain placeholder zero values for required evidence columns: "
            + "; ".join(placeholder_columns)
            + ". generated_experiment_core.evaluate_state must emit the exact required metric names directly, "
            "not rely on generic zero defaults while semantically equivalent aliases are nonzero."
        )

    sweep_specs = _phase24_sweep_specs_from_plan(plan)
    figures = _phase24_evidence_figures(plan, evidence)
    for idx, raw_figure in enumerate(figures):
        if not isinstance(raw_figure, dict):
            continue
        figure_id = str(raw_figure.get("id") or raw_figure.get("figure_id") or f"figure_{idx + 1}").strip()
        methods_to_run = [
            _phase24_contract_method_id(item)
            for item in _phase24_contract_list(raw_figure.get("methods_to_run"))
        ]
        methods_to_run = [method for method in methods_to_run if method]
        if not methods_to_run:
            continue
        sweep_name = _phase24_figure_required_sweep(raw_figure)
        figure_rows = [
            row
            for row in rows
            if figure_id and str(row.get("figure_id") or row.get("figure") or "").strip().lower() == figure_id.lower()
        ]
        sweep_rows = (
            _phase24_rows_for_sweep(rows, sweep_id=sweep_name, sweep_spec=sweep_specs.get(sweep_name, {}))
            if sweep_name
            else []
        )
        candidate_rows = figure_rows or sweep_rows
        if not candidate_rows:
            errors.append(
                f"{figure_id}: validation_results.csv has no rows for required_sweep `{sweep_name or '<unspecified>'}`."
            )
            continue
        actual_methods = {
            str(row.get("method") or "").strip().lower()
            for row in candidate_rows
            if str(row.get("method") or "").strip()
        }
        missing_methods = [
            method
            for method in methods_to_run
            if method.lower() not in actual_methods
        ]
        if missing_methods:
            errors.append(
                f"{figure_id}: validation_results.csv is missing rows for declared methods_to_run "
                f"{missing_methods}; actual methods for this figure/sweep are {sorted(actual_methods)}. "
                "Phase 2.4 must either implement every final plotted method or remove it from methods_to_run "
                "and keep it only in the candidate/audit benchmark pool."
            )
        if figure_rows and sweep_rows:
            figure_method_set = {
                str(row.get("method") or "").strip().lower()
                for row in figure_rows
                if str(row.get("method") or "").strip()
            }
            sweep_method_set = {
                str(row.get("method") or "").strip().lower()
                for row in sweep_rows
                if str(row.get("method") or "").strip()
            }
            if figure_method_set != sweep_method_set:
                warnings.append(
                    f"{figure_id}: figure_id rows and required_sweep rows expose different method sets "
                    f"({sorted(figure_method_set)} vs {sorted(sweep_method_set)}); check figure tagging."
                )
    return {"ok": not errors, "errors": errors, "warnings": warnings, "required_columns": required, "actual_columns": headers}


def _phase24_metric_aliases(metric: str) -> list[str]:
    metric = str(metric or "").strip()
    alias_map = {
        "objective_value": ["objective_value", "objective"],
        "sum_rate_bpsHz": ["sum_rate_bpsHz", "sum_rate_bps_hz", "sum_rate", "rate_bpsHz", "R_c_bpsHz", "R_c", "rate"],
        "sum_rate_bps_hz": ["sum_rate_bps_hz", "sum_rate_bpsHz", "sum_rate", "rate_bpsHz", "R_c_bpsHz", "R_c", "rate"],
        "rate_bpsHz": ["rate_bpsHz", "sum_rate_bpsHz", "sum_rate_bps_hz", "sum_rate", "R_c_bpsHz", "R_c", "rate"],
        "radar_SNR_dB": ["radar_SNR_dB", "radar_snr_dB", "radar_SNR", "radar_snr", "sensing_gain", "sensing_metric"],
        "radar_snr_dB": ["radar_snr_dB", "radar_SNR_dB", "radar_SNR", "radar_snr", "sensing_gain", "sensing_metric"],
        "harvested_energy_mW": ["harvested_energy_mW", "true_harvested_energy_mW", "harvested_energy", "Eharv", "P_EH_mW", "harvested_power_mW", "Psi_eh", "eh_total_mW"],
        "true_harvested_energy_mW": ["true_harvested_energy_mW", "harvested_energy_mW", "harvested_energy", "Eharv", "P_EH_mW", "Psi_eh"],
        "max_constraint_violation": ["max_constraint_violation", "constraint_violation_max", "cv_max", "total_violation", "C1_viol", "C2_viol_sum", "C3_viol", "C4_viol"],
        "constraint_violation_max": ["constraint_violation_max", "max_constraint_violation", "cv_max", "total_violation"],
        "optimal_rho": ["optimal_rho", "rho", "rho_star", "rho_opt"],
        "rho": ["rho", "optimal_rho", "rho_star", "rho_opt"],
        "M_eh": ["M_eh", "M_EH", "actual_used_rho_partition_elements_EH", "rho_rounded_elements_EH"],
        "rank_V_star": ["rank_V_star", "rank_V", "rank_v", "rank_W", "rank_Wc"],
        "eigenvalue_ratio_V": ["eigenvalue_ratio_V", "eigenvalue_ratio", "eig_ratio_V", "lambda1_over_lambda2"],
        "sca_iterations": ["sca_iterations", "sca_iter", "num_sca_iter"],
        "bcd_iterations": ["bcd_iterations", "bcd_iter", "bcd_outer_iterations"],
        "sca_final_surrogate_gap_mW": ["sca_final_surrogate_gap_mW", "sca_gap", "surrogate_gap", "sca_surrogate_gap"],
        "runtime_seconds": ["runtime_seconds", "solver_time_sec", "time_sec", "solve_time_sec"],
        "solver_status": ["solver_status", "status"],
        "sigmoid_a": ["sigmoid_a", "a_steepness", "EH_a_steepness"],
        "E_min_mW": ["E_min_mW", "constraints_E_min_mW", "E_min", "constraints_E_min"],
        "E_min_frac": ["E_min_frac", "constraints_E_min_frac", "E_min_mW", "constraints_E_min_mW"],
        "lambda_s_ratio": ["lambda_s_ratio", "optimization_lambda_s", "lambda_s", "swept_value"],
    }
    aliases = [metric]
    aliases.extend(alias_map.get(metric, []))
    aliases.extend(alias_map.get(metric.replace("-", "_"), []))
    deduped: list[str] = []
    for alias in aliases:
        alias = str(alias or "").strip()
        if alias and alias not in deduped:
            deduped.append(alias)
    return deduped


def _phase24_float_cell(value: Any) -> float | None:
    text = str(value if value is not None else "").strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered in {"true", "yes"}:
        return 1.0
    if lowered in {"false", "no"}:
        return 0.0
    try:
        numeric = float(text)
    except Exception:
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _phase24_truthy_cell(value: Any) -> bool:
    text = str(value if value is not None else "").strip().lower()
    if text in {"1", "true", "yes", "ok", "success", "feasible", "optimal", "optimal_inaccurate", "solved"}:
        return True
    numeric = _phase24_float_cell(value)
    return bool(numeric is not None and numeric > 0.5)


def _phase24_strict_research_gate() -> bool:
    raw_value = os.environ.get("WARA_PHASE24_STRICT_RESEARCH_GATE") or os.environ.get("WCL_PHASE24_STRICT_RESEARCH_GATE")
    return str(raw_value or "").strip().lower() in {"1", "true", "yes", "strict", "block"}


def _phase24_timeout_seconds(env_name: str, default: int) -> int:
    raw_value = os.environ.get(env_name, "").strip()
    try:
        value = int(float(raw_value if raw_value else default))
    except (TypeError, ValueError):
        value = int(default)
    return max(0, value)


def _phase24_timeout_stream(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return str(value)


def _phase24_counter(items: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        key = str(item or "").strip() or "<empty>"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda pair: (-pair[1], pair[0])))


def _phase24_validation_results_diagnostics(run_dir: Path) -> dict[str, Any]:
    """Summarize method-level failures so LLM repair sees the real blocker."""

    phase24_dir = Path(run_dir) / "phase2-4"
    results_path = phase24_dir / "solver" / "outputs" / "validation_results.csv"
    diagnostics: dict[str, Any] = {
        "ok": False,
        "results_path": str(results_path),
        "total_rows": 0,
        "method_counts": {},
        "status_counts": {},
        "method_status_counts": {},
        "method_solver_status_counts": {},
        "method_cvxpy_status_counts": {},
        "proposed_all_failed": False,
        "first_failed_proposed_row": {},
    }
    if not results_path.exists():
        diagnostics["error"] = "validation_results.csv is missing"
        return diagnostics
    try:
        with results_path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except Exception as exc:  # noqa: BLE001 - diagnostic path should not hide the original gate failure.
        diagnostics["error"] = f"{type(exc).__name__}: {exc}"
        return diagnostics

    diagnostics["total_rows"] = len(rows)
    methods = [str(row.get("method") or "").strip() for row in rows]
    statuses = [str(row.get("status") or "").strip() for row in rows]
    diagnostics["method_counts"] = _phase24_counter(methods)
    diagnostics["status_counts"] = _phase24_counter(statuses)

    method_status_counts: dict[str, dict[str, int]] = {}
    method_solver_status_counts: dict[str, dict[str, int]] = {}
    method_cvxpy_status_counts: dict[str, dict[str, int]] = {}
    for method in sorted(set(methods)):
        method_rows = [row for row in rows if str(row.get("method") or "").strip() == method]
        method_status_counts[method] = _phase24_counter([str(row.get("status") or "").strip() for row in method_rows])
        method_solver_status_counts[method] = _phase24_counter([str(row.get("solver_status") or "").strip() for row in method_rows])
        method_cvxpy_status_counts[method] = _phase24_counter([str(row.get("cvxpy_status") or "").strip() for row in method_rows])
    diagnostics["method_status_counts"] = method_status_counts
    diagnostics["method_solver_status_counts"] = method_solver_status_counts
    diagnostics["method_cvxpy_status_counts"] = method_cvxpy_status_counts

    proposed_rows = [row for row in rows if str(row.get("method") or "").strip().lower() == "proposed"]
    proposed_ok_rows = [row for row in proposed_rows if str(row.get("status") or "").strip().lower() == "ok"]
    diagnostics["proposed_rows"] = len(proposed_rows)
    diagnostics["proposed_ok_rows"] = len(proposed_ok_rows)
    diagnostics["proposed_all_failed"] = bool(proposed_rows and not proposed_ok_rows)
    for row in proposed_rows:
        if str(row.get("status") or "").strip().lower() != "ok":
            keep_keys = [
                "method",
                "status",
                "feasible",
                "solver_status",
                "cvxpy_success",
                "cvxpy_solver",
                "cvxpy_status",
                "objective",
                "sum_rate_bpsHz",
                "min_user_rate_bpsHz",
                "constraint_violation",
                "scenario_name",
                "swept_param",
                "swept_value",
                "seed",
            ]
            diagnostics["first_failed_proposed_row"] = {key: row.get(key) for key in keep_keys if key in row}
            break
    diagnostics["ok"] = True
    write_text(phase24_dir / "phase24_validation_failure_diagnostics.json", json.dumps(diagnostics, ensure_ascii=False, indent=2))
    return diagnostics


def _phase24_probe_solver_exception(run_dir: Path) -> str:
    """Run one proposed step with CVXPY solve monkeypatched to expose hidden traces."""

    phase24_dir = Path(run_dir) / "phase2-4"
    solver_dir = phase24_dir / "solver"
    core_path = solver_dir / "generated_experiment_core.py"
    cases_path = solver_dir / "validation_cases.py"
    if not core_path.exists() or not cases_path.exists():
        return ""
    probe_script = r'''
import json
import traceback

try:
    import cvxpy as cp
except Exception as exc:
    print("CVXPY_IMPORT_EXCEPTION", type(exc).__name__, repr(str(exc)))
    cp = None

import generated_experiment_core as core
import validation_cases as vc

if cp is not None:
    _orig_solve = cp.Problem.solve
    _seen = {"value": False}

    def _solve_with_trace(self, *args, **kwargs):
        try:
            return _orig_solve(self, *args, **kwargs)
        except Exception as exc:
            if not _seen["value"]:
                _seen["value"] = True
                print("FIRST_CVXPY_SOLVE_EXCEPTION", type(exc).__name__, repr(str(exc)))
                traceback.print_exc()
            raise

    cp.Problem.solve = _solve_with_trace

if hasattr(vc, "make_validation_cases"):
    cases = vc.make_validation_cases()
elif hasattr(vc, "validation_cases"):
    cases = vc.validation_cases()
elif hasattr(vc, "load_canonical_case"):
    cases = [vc.load_canonical_case()]
else:
    raise RuntimeError("validation_cases.py does not expose a known case factory")

case = cases[0]
model = core.build_model(case, seed=0)
state = core.initial_state(case, model, seed=0)
out = core.proposed_step(case, model, state, 0)
print("PROPOSED_STEP_DIAGNOSTICS", json.dumps({
    "status": out.get("status"),
    "solver_status": out.get("solver_status"),
    "cvxpy_success": out.get("cvxpy_success"),
    "cvxpy_solver": out.get("cvxpy_solver"),
    "cvxpy_status": out.get("cvxpy_status"),
}, default=str))
'''
    try:
        result = subprocess.run(
            [sys.executable, "-c", probe_script],
            cwd=solver_dir,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=None,
        )
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        write_text(phase24_dir / "phase24_solver_exception_probe_stdout.txt", stdout)
        write_text(phase24_dir / "phase24_solver_exception_probe_stderr.txt", stderr)
        trace_text = (
            f"returncode={result.returncode}\n\n"
            f"STDOUT:\n{stdout}\n\n"
            f"STDERR:\n{stderr}"
        ).strip()
    except Exception as exc:  # noqa: BLE001 - keep original validation failure primary.
        trace_text = f"solver exception probe failed: {type(exc).__name__}: {exc}"
    write_text(phase24_dir / "phase24_solver_exception_trace.txt", trace_text)
    return trace_text


def _phase24_write_failure_diagnostics(run_dir: Path) -> dict[str, Any]:
    diagnostics = _phase24_validation_results_diagnostics(run_dir)
    if diagnostics.get("proposed_all_failed") or "cvxpy" in json.dumps(diagnostics, ensure_ascii=False).lower():
        trace = _phase24_probe_solver_exception(run_dir)
        diagnostics["solver_exception_trace_path"] = str(Path(run_dir) / "phase2-4" / "phase24_solver_exception_trace.txt")
        diagnostics["solver_exception_trace_excerpt"] = trace[:4000]
        write_text(
            Path(run_dir) / "phase2-4" / "phase24_validation_failure_diagnostics.json",
            json.dumps(diagnostics, ensure_ascii=False, indent=2),
        )
    return diagnostics


def _phase24_series_span(rows: list[dict[str, Any]], metric: str, aliases: list[str] | None = None) -> tuple[float, str, int]:
    best_span = 0.0
    best_alias = ""
    best_count = 0
    for alias in aliases or _phase24_metric_aliases(metric):
        values: list[float] = []
        for row in rows:
            if alias not in row:
                continue
            numeric = _phase24_float_cell(row.get(alias))
            if numeric is None:
                continue
            if str(metric).lower().endswith("_db") and not str(alias).lower().endswith("_db"):
                if numeric <= 0:
                    continue
                numeric = 10.0 * math.log10(numeric)
            values.append(float(numeric))
        if len(values) < 2:
            continue
        span = max(values) - min(values)
        if span > best_span or (best_alias == "" and len(values) > best_count):
            best_span = float(span)
            best_alias = alias
            best_count = len(values)
    return best_span, best_alias, best_count


def _phase24_series_scale(rows: list[dict[str, Any]], alias: str, metric: str) -> float:
    values: list[float] = []
    if not alias:
        return 0.0
    for row in rows:
        if alias not in row:
            continue
        numeric = _phase24_float_cell(row.get(alias))
        if numeric is None:
            continue
        if str(metric).lower().endswith("_db") and not str(alias).lower().endswith("_db"):
            if numeric <= 0:
                continue
            numeric = 10.0 * math.log10(numeric)
        values.append(abs(float(numeric)))
    return max(values) if values else 0.0


def _phase24_path_alias_candidates(path: str) -> list[str]:
    cleaned = re.sub(r"[^0-9A-Za-z]+", "_", str(path or "")).strip("_")
    if not cleaned:
        return []
    parts = [part for part in cleaned.split("_") if part]
    suffixes = [cleaned]
    if len(parts) > 1:
        suffixes.append("_".join(parts[-2:]))
    suffixes.append(parts[-1])
    candidates: list[str] = []
    for suffix in suffixes:
        for prefix in ("actual_used_", ""):
            candidate = f"{prefix}{suffix}"
            if candidate not in candidates:
                candidates.append(candidate)
    return candidates


def _phase24_numeric_rows_for_alias(rows: list[dict[str, Any]], alias: str) -> list[tuple[float, float]]:
    pairs: list[tuple[float, float]] = []
    for row in rows:
        actual = _phase24_float_cell(row.get(alias))
        swept = _phase24_float_cell(row.get("swept_value"))
        if actual is None or swept is None:
            continue
        if not (math.isfinite(float(actual)) and math.isfinite(float(swept))):
            continue
        pairs.append((float(actual), float(swept)))
    return pairs


def _phase24_sweep_specs_from_plan(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    sweep_specs: dict[str, dict[str, Any]] = {}
    raw_sweeps = plan.get("sweep_definitions", [])
    if isinstance(raw_sweeps, dict):
        iterable = raw_sweeps.items()
    elif isinstance(raw_sweeps, list):
        iterable = (
            (str(item.get("id") or item.get("name") or f"sweep_{idx + 1}"), item)
            for idx, item in enumerate(raw_sweeps)
            if isinstance(item, dict)
        )
    else:
        iterable = []
    for sweep_name, spec in iterable:
        if not isinstance(spec, dict):
            continue
        sweep_specs[str(sweep_name)] = {
            "variable": str(spec.get("canonical_path") or spec.get("variable") or spec.get("target") or "").strip(),
            "canonical_path": str(spec.get("canonical_path") or "").strip(),
            "raw_variable": str(spec.get("variable") or spec.get("target") or "").strip(),
        }
    return sweep_specs


def _phase24_rows_for_sweep(
    rows: list[dict[str, Any]],
    *,
    sweep_id: str,
    sweep_spec: dict[str, Any],
) -> list[dict[str, Any]]:
    sweep_id_lower = str(sweep_id or "").strip().lower()
    paths = {
        str(sweep_spec.get("variable") or "").strip(),
        str(sweep_spec.get("canonical_path") or "").strip(),
        str(sweep_spec.get("raw_variable") or "").strip(),
    }
    paths = {path for path in paths if path}
    direct_matches: list[dict[str, Any]] = []
    for row in rows:
        case_id = str(row.get("case_id", "")).lower()
        scenario = str(row.get("scenario_name", "")).lower()
        if sweep_id_lower and (case_id.startswith(f"{sweep_id_lower}_") or sweep_id_lower in scenario):
            direct_matches.append(row)
    if direct_matches:
        return direct_matches
    matched: list[dict[str, Any]] = []
    for row in rows:
        swept_param = str(row.get("swept_param", "")).strip()
        if swept_param in paths:
            matched.append(row)
    return matched


def validate_phase24_experiment_responsiveness(run_dir: Path) -> dict[str, Any]:
    phase24_dir = Path(run_dir) / "phase2-4"
    solver_dir = phase24_dir / "solver"
    try:
        plan = yaml.safe_load(read_text(phase24_dir / "validation_plan.yaml")) or {}
    except Exception as exc:
        return {"ok": False, "errors": [f"validation_plan.yaml is not valid YAML: {exc}"], "warnings": []}
    if not isinstance(plan, dict):
        return {"ok": False, "errors": ["validation_plan.yaml did not parse to a mapping"], "warnings": []}
    quality_tolerance = 1.0e-5
    try:
        canonical_config = plan.get("canonical_config", {}) if isinstance(plan, dict) else {}
        algorithm_config = canonical_config.get("algorithm", {}) if isinstance(canonical_config, dict) else {}
        if isinstance(algorithm_config, dict):
            declared_tolerances = [
                _phase24_float_cell(algorithm_config.get(key))
                for key in ("feasibility_tolerance", "primal_dual_tolerance", "solver_tolerance", "tolerance")
            ]
            declared_tolerances.extend(
                _phase24_float_cell(algorithm_config.get(key))
                for key in ("eps_feas", "feasibility_eps", "constraint_tolerance", "constraint_eps")
            )
            declared_tolerances = [
                float(value)
                for value in declared_tolerances
                if value is not None and math.isfinite(float(value)) and float(value) > 0
            ]
            if declared_tolerances:
                quality_tolerance = max(quality_tolerance, min(1.0e-3, 10.0 * min(declared_tolerances)))
    except Exception:
        quality_tolerance = 1.0e-5

    results_path = solver_dir / "outputs" / "validation_results.csv"
    if not results_path.exists():
        return {"ok": False, "errors": [f"missing validation results csv: {results_path}"], "warnings": []}
    try:
        with results_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = [dict(row) for row in reader]
            headers = list(reader.fieldnames or [])
    except Exception as exc:
        return {"ok": False, "errors": [f"cannot read validation results csv: {exc}"], "warnings": []}
    if not rows:
        return {"ok": False, "errors": ["validation_results.csv has no data rows"], "warnings": []}

    sweep_specs = _phase24_sweep_specs_from_plan(plan)

    evidence = _phase24_research_evidence(plan)
    figures = evidence.get("figures")
    if not isinstance(figures, list) or not figures:
        figures = plan.get("figure_targets")
    if not isinstance(figures, list):
        figures = []

    errors: list[str] = []
    warnings: list[str] = []
    repair_advice: list[str] = []
    checks: list[dict[str, Any]] = []
    design_repair_recommended = False
    strict_research_gate = _phase24_strict_research_gate()
    diagnostic_aliases = [header for header in headers if header.startswith("actual_used") or header.startswith("used_") or header.endswith("_actual_used")]
    consumed_aliases = [header for header in headers if header.lower() in {"sweep_consumed", "diagnostics_sweep_consumed"}]
    for idx, figure in enumerate(figures[:3]):
        if not isinstance(figure, dict):
            continue
        figure_id = str(figure.get("id") or figure.get("figure_id") or f"figure_{idx + 1}")
        sweep_id = str(figure.get("required_sweep") or figure.get("sweep_id") or "").strip()
        metric = str(figure.get("y_metric") or figure.get("metric") or "").strip()
        if not sweep_id or not metric:
            continue
        sweep_rows = _phase24_rows_for_sweep(rows, sweep_id=sweep_id, sweep_spec=sweep_specs.get(sweep_id, {}))
        if not sweep_rows:
            warnings.append(f"{figure_id}: no quick-validation rows found for required_sweep={sweep_id}")
            continue
        x_values = {str(row.get("swept_value", "")).strip() for row in sweep_rows if str(row.get("swept_value", "")).strip()}
        if len(x_values) < 2:
            warnings.append(f"{figure_id}: fewer than two x-values observed for required_sweep={sweep_id}")
            continue
        target_methods = [
            str(method).strip()
            for method in figure.get("methods_to_run", ["proposed"])
            if str(method).strip()
        ]
        candidate_methods = target_methods or ["proposed"]
        method_for_check = next(
            (method for method in candidate_methods if any(str(row.get("method", "")) == method for row in sweep_rows)),
            "proposed",
        )
        method_rows = [row for row in sweep_rows if str(row.get("method", "")) == method_for_check] or sweep_rows
        metric_span, metric_alias, metric_count = _phase24_series_span(method_rows, metric)
        metric_scale = _phase24_series_scale(method_rows, metric_alias, metric)
        relative_metric_span = metric_span / max(metric_scale, 1.0e-12) if metric_scale > 0 else 0.0
        sweep_path = str(sweep_specs.get(sweep_id, {}).get("variable") or sweep_specs.get(sweep_id, {}).get("canonical_path") or "")
        targeted_actual_aliases = [alias for alias in _phase24_path_alias_candidates(sweep_path) if alias in headers]
        actual_used_span, actual_used_alias, _ = _phase24_series_span(
            method_rows,
            metric,
            targeted_actual_aliases or diagnostic_aliases,
        )
        consumed_values = [
            _phase24_float_cell(row.get(alias))
            for alias in consumed_aliases
            for row in method_rows
            if alias in row
        ]
        consumed_false = any(value == 0.0 for value in consumed_values if value is not None)
        checks.append(
            {
                "figure_id": figure_id,
                "required_sweep": sweep_id,
                "method_checked": method_for_check,
                "metric": metric,
                "metric_alias_used": metric_alias,
                "metric_span": metric_span,
                "actual_used_alias_used": actual_used_alias,
                "actual_used_span": actual_used_span,
                "relative_metric_span": relative_metric_span,
                "num_metric_values": metric_count,
                "num_x_values": len(x_values),
                "sweep_consumption_proven": False,
            }
        )
        if consumed_false:
            errors.append(f"{figure_id}: plugin reported sweep_consumed=False for required_sweep={sweep_id}")
            continue
        actual_used_mismatches: list[str] = []
        if targeted_actual_aliases:
            for alias in targeted_actual_aliases:
                pairs = _phase24_numeric_rows_for_alias(method_rows, alias)
                if not pairs:
                    continue
                mismatch_count = sum(
                    1
                    for actual, swept in pairs
                    if abs(actual - swept) > max(1.0e-9, 1.0e-6 * max(abs(swept), 1.0))
                )
                if mismatch_count:
                    actual_used_mismatches.append(
                        f"{alias} disagrees with swept_value in {mismatch_count}/{len(pairs)} rows"
                    )
                break
        elif sweep_path:
            actual_used_mismatches.append(
                f"missing actual-used diagnostic for swept executable path `{sweep_path}`"
            )
        metric_lower = metric.lower()
        metric_is_diagnostic = "violation" in metric_lower or "infeasible" in metric_lower or "outage" in metric_lower
        if actual_used_mismatches and metric_is_diagnostic:
            warnings.append(
                f"{figure_id}: diagnostic metric `{metric}` does not prove required_sweep={sweep_id} consumption through "
                + "; ".join(actual_used_mismatches)
                + "; treating the flat diagnostic as reliability evidence rather than a blocking paper-KPI responsiveness check."
            )
        elif actual_used_mismatches:
            errors.append(
                f"{figure_id}: generated experiment does not prove it consumed required_sweep={sweep_id}: "
                + "; ".join(actual_used_mismatches)
            )
            continue
        if checks:
            checks[-1]["sweep_consumption_proven"] = bool(
                targeted_actual_aliases and actual_used_span > 0.0 and not actual_used_mismatches
            )
        if metric_count >= 2 and abs(metric_span) > 1.0e-10 and relative_metric_span >= 1.0e-3:
            continue
        if metric_is_diagnostic:
            values = []
            for row in method_rows:
                for key in (metric_alias, metric):
                    if not key:
                        continue
                    value = _phase24_float_cell(row.get(key))
                    if value is not None and math.isfinite(float(value)):
                        values.append(abs(float(value)))
                        break
            if values and max(values) <= quality_tolerance:
                warnings.append(
                    f"{figure_id}: diagnostic metric `{metric}` is flat because it remains within tolerance "
                    f"({quality_tolerance:g}) across required_sweep={sweep_id}; treating this as reliability evidence."
                )
                continue
        if targeted_actual_aliases and actual_used_span > 0.0 and not actual_used_mismatches:
            design_repair_recommended = True
            advice = (
                f"{figure_id}: metric `{metric}` is constant, missing, or too weakly responsive across required_sweep={sweep_id}"
                + (f" (executable path `{sweep_path}`) " if sweep_path else " ")
                + f"for method={method_for_check}, even though actual-used diagnostics show the sweep value was consumed. "
                "This points to an experiment-design or operating-regime problem more than a syntax/interface bug: "
                "the validation plan should revise context_overrides, scout_values, y_metric, or the figure sweep so the "
                "paper KPI enters an active regime. Do not fabricate KPI variation in code. "
                f"Observed span={metric_span:.6g}, relative_span={relative_metric_span:.6g}."
            )
        else:
            advice = (
                f"{figure_id}: metric `{metric}` is constant, missing, or too weakly responsive across required_sweep={sweep_id}"
                + (f" (executable path `{sweep_path}`) " if sweep_path else " ")
                + f"for method={method_for_check}. This usually means the solver did not consume the swept path "
                "or emitted placeholder required metrics. build_model must read the exact path with problem.get(...), "
                "the proposed update/evaluate_state equations must use the derived value, and the main y_metric itself "
                "must vary materially across the sweep. actual_used diagnostics are required but cannot substitute for evidence. "
                f"Observed span={metric_span:.6g}, relative_span={relative_metric_span:.6g}."
            )
        if strict_research_gate:
            errors.append(advice)
        else:
            repair_advice.append(advice)
            warnings.append(advice)

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "repair_advice": repair_advice,
        "blocking_errors": errors,
        "design_repair_recommended": design_repair_recommended,
        "strict_research_gate": strict_research_gate,
        "research_readiness": "blocking_failed" if errors else ("advisory_only" if repair_advice else "ready"),
        "checks": checks,
    }


def validate_phase24_basic_evidence_quality(run_dir: Path) -> dict[str, Any]:
    """Check output sanity without treating self-reported feasibility as proof."""
    phase24_dir = Path(run_dir) / "phase2-4"
    quality_tolerance = 1.0e-5
    tolerance_cap = float(os.environ.get("WARA_PHASE24_QUALITY_TOLERANCE_CAP", "1e-2") or 1.0e-2)
    try:
        plan_payload = yaml.safe_load(read_text(phase24_dir / "validation_plan.yaml")) or {}
        canonical_config = plan_payload.get("canonical_config", {}) if isinstance(plan_payload, dict) else {}
        algorithm_config = canonical_config.get("algorithm", {}) if isinstance(canonical_config, dict) else {}
        evaluation_config = canonical_config.get("evaluation", {}) if isinstance(canonical_config, dict) else {}
        if isinstance(algorithm_config, dict):
            declared_tolerances = [
                _phase24_float_cell(algorithm_config.get(key))
                for key in ("feasibility_tolerance", "primal_dual_tolerance", "solver_tolerance", "tolerance")
            ]
            declared_tolerances.extend(
                _phase24_float_cell(algorithm_config.get(key))
                for key in ("eps_feas", "feasibility_eps", "constraint_tolerance", "constraint_eps")
            )
            if isinstance(evaluation_config, dict):
                declared_tolerances.extend(
                    _phase24_float_cell(evaluation_config.get(key))
                    for key in ("feasibility_tolerance", "feasibility_tol", "feasibility_eps", "feasibility_tol_lambda")
                )
            declared_tolerances = [
                float(value)
                for value in declared_tolerances
                if value is not None and math.isfinite(float(value)) and float(value) > 0
            ]
            if declared_tolerances:
                quality_tolerance = max(quality_tolerance, min(tolerance_cap, 10.0 * min(declared_tolerances)))
    except Exception:
        quality_tolerance = 1.0e-5
    results_path = phase24_dir / "solver" / "outputs" / "validation_results.csv"
    if not results_path.exists():
        return {"ok": False, "errors": [f"missing validation results csv: {results_path}"], "warnings": []}
    try:
        with results_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = [dict(row) for row in csv.DictReader(handle)]
    except Exception as exc:
        return {"ok": False, "errors": [f"cannot read validation results csv: {exc}"], "warnings": []}
    proposed_rows = [row for row in rows if str(row.get("method", "")).strip().lower() == "proposed"]
    if not proposed_rows:
        return {"ok": False, "errors": ["validation_results.csv has no proposed-method rows"], "warnings": []}

    finite_objective_rows = [
        row
        for row in proposed_rows
        if _phase24_float_cell(row.get("objective", row.get("objective_value"))) is not None
    ]
    errors: list[str] = []
    warnings: list[str] = []
    if not finite_objective_rows:
        errors.append("proposed method has no finite objective values in quick validation")
    repair_advice: list[str] = []
    strict_research_gate = _phase24_strict_research_gate()

    active_context_text = "\n".join(
        read_text(path)
        for path in [
            phase24_dir / "validation_plan.yaml",
            Path(run_dir) / "phase2-1" / "mathematical_contract.frozen.json",
            Path(run_dir) / "phase2-3" / "algorithm_contract.json",
            Path(run_dir) / "phase2-3" / "algorithm_execution_contract.json",
            Path(run_dir) / "phase2-3" / "algorithm_description.md",
        ]
        if path.exists()
    ).lower()

    active_update_tokens: set[str] = set()
    if any(term in active_context_text for term in ["star-ris", "star ris", "theta_t", "theta_r", "theta", "beta_t", "beta_r", "reconfigurable intelligent", "intelligent reflecting", "ris"]):
        active_update_tokens.update({"star", "ris", "theta", "phase", "reflection", "transmission", "coefficient"})
    if any(term in active_context_text for term in ["precoder", "beamform", "beamforming", "covariance", "\\mathbf{w}", " w_", "sinr"]):
        active_update_tokens.update({"beam", "precoder", "covariance", "wmmse"})
    if any(term in active_context_text for term in ["power", "p_u", "p_{u", "allocation"]):
        active_update_tokens.update({"power", "allocation"})
    if any(term in active_context_text for term in ["combiner", "receive", "\\mathbf{u}", "mmse"]):
        active_update_tokens.update({"combiner", "receiver", "mmse"})
    if any(term in active_context_text for term in ["position", "location", "trajectory", "movable antenna", "fluid antenna", "uav"]):
        active_update_tokens.update({"position", "location", "trajectory", "deployment", "coordinate", "antenna_position"})

    def is_active_update_diagnostic(column: str) -> bool:
        lower = str(column or "").strip().lower()
        if not lower:
            return False
        if any(token in lower for token in active_update_tokens):
            return True
        if lower in {"used_proposed_update", "used_mechanism_update", "proposed_step_norm", "step_norm", "update_norm"}:
            return True
        return False

    proposed_headers = sorted({key for row in proposed_rows for key in row.keys()})
    dormant_update_flags: list[str] = []
    for column in proposed_headers:
        lower = column.lower()
        if not lower.startswith("used_") or ("update" not in lower and "mechanism" not in lower):
            continue
        if not is_active_update_diagnostic(column):
            continue
        values = [str(row.get(column, "")).strip().lower() for row in proposed_rows if str(row.get(column, "")).strip()]
        if values and all(value in {"0", "false", "no", "none"} for value in values):
            dormant_update_flags.append(column)
    dormant_update_norms: list[str] = []
    active_norm_columns: list[str] = []
    for column in proposed_headers:
        lower = column.lower()
        if not (
            lower.endswith("_step_norm")
            or lower.endswith("_update_norm")
            or lower in {"step_norm", "update_norm", "position_step_norm"}
        ):
            continue
        if not is_active_update_diagnostic(column):
            continue
        active_norm_columns.append(column)
        values = [
            abs(float(value))
            for row in proposed_rows
            for value in [_phase24_float_cell(row.get(column))]
            if value is not None and math.isfinite(float(value))
        ]
        if values and max(values) <= 1.0e-10:
            dormant_update_norms.append(column)
    if dormant_update_flags:
        advice = (
            "proposed method reports dormant mechanism/update diagnostics in every quick-validation row: "
            + ", ".join(dormant_update_flags)
            + ". Repair generated_experiment_core.py so the declared proposed algorithm actually executes its core update block; "
            "do not satisfy this by deleting diagnostics or relabeling a baseline as proposed."
        )
        if strict_research_gate:
            errors.append(advice)
        else:
            repair_advice.append(advice)
            warnings.append(advice)
    if dormant_update_norms and (dormant_update_flags or len(dormant_update_norms) == len(active_norm_columns)):
        advice = (
            "proposed method also reports zero update-step norm diagnostics: "
            + ", ".join(dormant_update_norms)
            + ". This indicates the implementation is not exercising the mechanism needed for paper evidence."
        )
        if strict_research_gate:
            errors.append(advice)
        else:
            repair_advice.append(advice)
            warnings.append(advice)

    violation_columns = [
        "max_constraint_violation",
        "constraint_violation_max",
        "constraint_violation",
        "total_violation",
        "cv_max",
        "power_budget_violation",
        "sinr_violation",
        "radar_snr_violation",
        "energy_violation",
        "C1_viol",
        "C2_viol",
        "C2_viol_sum",
        "C3_viol",
        "C4_viol",
        "c1_power_viol",
        "c2_sinr_viol",
        "c3_radar_viol",
        "c4_eh_viol",
        "c5_unitarity_viol",
        "c7_rho_viol",
    ]
    present_violation_columns = [
        column
        for column in violation_columns
        if any(_phase24_float_cell(row.get(column)) is not None for row in proposed_rows)
    ]

    def column_values(column: str) -> list[float]:
        values: list[float] = []
        for row in proposed_rows:
            value = _phase24_float_cell(row.get(column))
            if value is not None and math.isfinite(float(value)):
                values.append(float(value))
        return values

    def column_summary(column: str) -> dict[str, Any]:
        values = column_values(column)
        if not values:
            return {}
        sorted_values = sorted(values)
        return {
            "count": len(values),
            "min": float(sorted_values[0]),
            "median": float(sorted_values[len(sorted_values) // 2]),
            "mean": float(sum(values) / len(values)),
            "max": float(sorted_values[-1]),
        }

    violation_summary = {
        column: summary
        for column in present_violation_columns
        for summary in [column_summary(column)]
        if summary
    }
    dominant_violation_column = ""
    if violation_summary:
        dominant_violation_column = max(
            violation_summary,
            key=lambda column: abs(float(violation_summary[column].get("median", 0.0))),
        )

    gamma_values: list[float] = []
    for row in proposed_rows:
        gamma_linear = _phase24_float_cell(row.get("gamma_min"))
        if gamma_linear is None:
            gamma_db = _phase24_float_cell(row.get("gamma_min_dB", row.get("constraints_gamma_min_dB")))
            if gamma_db is not None:
                gamma_linear = 10.0 ** (float(gamma_db) / 10.0)
        if gamma_linear is not None and math.isfinite(float(gamma_linear)):
            gamma_values.append(float(gamma_linear))
    gamma_reference = sorted(gamma_values)[len(gamma_values) // 2] if gamma_values else None
    tx_power_values: list[float] = []
    pmax_values: list[float] = []
    tx_power_fraction_direct_values: list[float] = []
    for row in proposed_rows:
        tx_power = next(
            (
                value
                for value in [
                    _phase24_float_cell(row.get("trace_Rx")),
                    _phase24_float_cell(row.get("power_trace")),
                    _phase24_float_cell(row.get("tx_power")),
                    _phase24_float_cell(row.get("power_total")),
                    _phase24_float_cell(row.get("total_power")),
                    _phase24_float_cell(row.get("final_tx_power")),
                ]
                if value is not None
            ),
            None,
        )
        if tx_power is not None and math.isfinite(float(tx_power)):
            tx_power_values.append(max(0.0, float(tx_power)))
        tx_fraction_direct = next(
            (
                value
                for value in [
                    _phase24_float_cell(row.get("constraint_active_C1_power")),
                    _phase24_float_cell(row.get("tx_power_fraction")),
                    _phase24_float_cell(row.get("power_fraction")),
                ]
                if value is not None
            ),
            None,
        )
        if tx_fraction_direct is not None and math.isfinite(float(tx_fraction_direct)):
            tx_power_fraction_direct_values.append(max(0.0, float(tx_fraction_direct)))
        pmax_linear = next(
            (
                value
                for value in [
                    _phase24_float_cell(row.get("Pmax")),
                    _phase24_float_cell(row.get("P_max")),
                    _phase24_float_cell(row.get("system_Pmax")),
                    _phase24_float_cell(row.get("system_P_max")),
                ]
                if value is not None
            ),
            None,
        )
        if pmax_linear is None:
            pmax_dbm = next(
                (
                    value
                    for value in [
                        _phase24_float_cell(row.get("Pmax_dBm")),
                        _phase24_float_cell(row.get("P_max_dBm")),
                        _phase24_float_cell(row.get("system_Pmax_dBm")),
                        _phase24_float_cell(row.get("system_P_max_dBm")),
                    ]
                    if value is not None
                ),
                None,
            )
            if pmax_dbm is not None:
                pmax_linear = 10.0 ** ((float(pmax_dbm) - 30.0) / 10.0)
        if pmax_linear is not None and math.isfinite(float(pmax_linear)) and float(pmax_linear) > 0:
            pmax_values.append(float(pmax_linear))
    tx_power_fraction_median = None
    if tx_power_values and pmax_values:
        tx_median = sorted(tx_power_values)[len(tx_power_values) // 2]
        pmax_median = sorted(pmax_values)[len(pmax_values) // 2]
        tx_power_fraction_median = float(tx_median / max(pmax_median, 1.0e-12))
    elif tx_power_fraction_direct_values:
        tx_power_fraction_median = float(sorted(tx_power_fraction_direct_values)[len(tx_power_fraction_direct_values) // 2])

    def row_max_violation(row: dict[str, Any]) -> float | None:
        values = [
            abs(float(value))
            for column in violation_columns
            for value in [_phase24_float_cell(row.get(column))]
            if value is not None
        ]
        if not values:
            return None
        return max(values)

    def row_feasibility_tolerance(row: dict[str, Any]) -> float:
        row_tolerances = [
            _phase24_float_cell(row.get(key))
            for key in (
                "feasibility_tolerance",
                "feasibility_tol",
                "feasibility_eps",
                "constraint_tolerance",
                "constraint_eps",
                "eps_feas",
                "evaluation_feasibility_tol_lambda",
                "feasibility_tol_lambda",
            )
        ]
        numeric = [
            float(value)
            for value in row_tolerances
            if value is not None and math.isfinite(float(value)) and float(value) > 0
        ]
        return max([quality_tolerance, *numeric])

    feasible_rows: list[dict[str, Any]] = []
    inconsistent_rows = 0
    invalid_success_rows = 0
    for row in proposed_rows:
        feasible_text = str(row.get("feasible", "")).strip().lower()
        status_text = str(row.get("status", "")).strip().lower()
        feasible_flag = feasible_text in {"1", "true", "yes", "ok", "success", "feasible"}
        status_ok = status_text in {"", "ok", "success", "converged", "feasible", "optimal", "optimal_inaccurate"}
        max_violation = row_max_violation(row)
        violation_ok = max_violation is None or max_violation <= row_feasibility_tolerance(row)
        if status_ok and violation_ok:
            feasible_rows.append(row)
            if not feasible_flag:
                warnings.append(
                    "proposed row was marked infeasible by the generated plugin but is within the declared validation tolerance; "
                    "recording it as diagnostic soft-feasible without using feasible-rate as a Phase 2.4 gate."
                )
        if feasible_flag and max_violation is not None and max_violation > row_feasibility_tolerance(row):
            inconsistent_rows += 1
        if status_ok and max_violation is not None and max_violation > row_feasibility_tolerance(row):
            invalid_success_rows += 1

    if inconsistent_rows:
        message = (
            "proposed method marks rows feasible while reported constraint violations remain nonzero; "
            "this is an evaluator/code-consistency issue, not a feasible-rate failure"
        )
        if strict_research_gate:
            errors.append(message)
        else:
            warnings.append(message)
            repair_advice.append(message)
    if invalid_success_rows:
        message = (
            "proposed method reports `status: ok` for rows whose physical constraint violation exceeds tolerance. "
            "This is tracked as diagnostic evidence; repair should propagate infeasible/failed solver outcomes as "
            "non-success rows before Phase 2.5 uses performance KPIs."
        )
        if strict_research_gate:
            errors.append(message)
        else:
            warnings.append(message)
            repair_advice.append(message)
    feasible_rate = float(len(feasible_rows) / max(len(proposed_rows), 1))
    if not feasible_rows:
        warnings.append(
            "proposed method has zero rows with both ok status and low reported violation. "
            "Phase 2.4 no longer fails on feasible-rate; use this only as a diagnostic while algorithm-code, "
            "sweep-consumption, KPI-responsiveness, and method-semantics gates decide whether the implementation is credible."
        )
        if dominant_violation_column:
            dominant = violation_summary.get(dominant_violation_column, {})
            repair_advice.append(
                "Dominant proposed-method violation is "
                f"`{dominant_violation_column}` with median={float(dominant.get('median', 0.0)):.4g} "
                f"and max={float(dominant.get('max', 0.0)):.4g}; repair should target the physical metric/constraint model, "
                "not paper rendering."
            )
        sinr_summary = next(
            (
                (column, violation_summary[column])
                for column in violation_summary
                if "sinr" in column.lower() or column.lower().startswith("c2")
            ),
            None,
        )
        if sinr_summary:
            sinr_column, summary = sinr_summary
            median_violation = float(summary.get("median", 0.0))
            gamma_note = ""
            if gamma_reference is not None and gamma_reference > 0:
                ratio = median_violation / gamma_reference
                gamma_note = f" median/gamma_min={ratio:.3g}."
                if 0.5 <= ratio <= 1.5:
                    gamma_note += (
                        " This usually means the computed SINR is near zero, often because an aggregate transmit covariance "
                        "is counted as both desired signal and self/other-user interference."
                    )
            repair_advice.append(
                f"SINR-related violation `{sinr_column}` blocks feasibility;"
                + gamma_note
                + " Use per-user beam/covariance variables for downlink SINR, or explicitly use a documented effective_snr_proxy "
                "when the prototype state only contains one aggregate covariance."
            )
            if tx_power_fraction_median is not None and tx_power_fraction_median < 1.0e-4:
                repair_advice.append(
                    "The proposed transmit covariance appears to use a negligible fraction of the power budget "
                    f"(median trace/power-budget fraction={tx_power_fraction_median:.3g}) while SINR is infeasible. "
                    "Repair the initialization/update/candidate selection so the proposed method includes a full-power "
                    "communication-capable or benchmark-like feasible candidate before optimizing sensing/EH tradeoffs."
                )
            sum_rate_values = _phase24_numeric_values_for_aliases(
                proposed_rows,
                _phase24_metric_aliases("sum_rate_bps_hz") + ["sum_rate", "weighted_sum_rate", "rate"],
            )
            per_user_rate_values = _phase24_numeric_values_for_aliases(
                proposed_rows,
                _phase24_metric_aliases("per_user_min_rate") + ["min_rate", "min_user_rate"],
            )
            sum_rate_median = _phase24_median(sum_rate_values)
            per_user_rate_median = _phase24_median(per_user_rate_values)
            if (
                tx_power_fraction_median is not None
                and tx_power_fraction_median > 0.1
                and (
                    (sum_rate_median is not None and abs(sum_rate_median) < 1.0e-3)
                    or (per_user_rate_median is not None and abs(per_user_rate_median) < 1.0e-3)
                )
            ):
                repair_advice.append(
                    "The proposed transmit covariance uses a substantial power-budget fraction "
                    f"(median trace/power-budget fraction={tx_power_fraction_median:.3g}) but communication KPIs are near zero "
                    f"(sum-rate median={sum_rate_median if sum_rate_median is not None else 'missing'}, "
                    f"minimum-user-rate median={per_user_rate_median if per_user_rate_median is not None else 'missing'}). "
                    "This usually means the update steers Rx away from user channels or overwrites the communication-capable candidate. "
                    "Keep a full-power matched-filter/WMMSE-style candidate and retain the best feasible state by evaluate_state "
                    "before applying sensing/EH tradeoffs."
                )

    objective_span, objective_alias, objective_count = _phase24_series_span(proposed_rows, "objective", ["objective", "objective_value"])
    if objective_count >= 2 and objective_span <= 1.0e-12:
        warnings.append("proposed objective is constant across quick-validation rows")
    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "repair_advice": repair_advice,
        "blocking_errors": errors,
        "strict_research_gate": strict_research_gate,
        "research_readiness": "blocking_failed" if errors else ("advisory_only" if repair_advice else "ready"),
        "num_proposed_rows": len(proposed_rows),
        "num_feasible_proposed_rows": len(feasible_rows),
        "objective_alias_used": objective_alias,
        "objective_span": objective_span,
        "dominant_violation_column": dominant_violation_column,
        "violation_summary": violation_summary,
        "gamma_reference": gamma_reference,
        "tx_power_fraction_median": tx_power_fraction_median,
        "quality_tolerance": quality_tolerance,
        "feasible_rate": feasible_rate,
        "feasible_rate_gate_enabled": False,
    }


def _phase24_numeric_values_for_aliases(rows: list[dict[str, Any]], aliases: list[str]) -> list[float]:
    values: list[float] = []
    for row in rows:
        for alias in aliases:
            if alias not in row:
                continue
            value = _phase24_float_cell(row.get(alias))
            if value is not None and math.isfinite(float(value)):
                values.append(float(value))
                break
    return values


def _phase24_median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return float(ordered[len(ordered) // 2])


def validate_phase24_pilot_gain(run_dir: Path) -> dict[str, Any]:
    """Require Phase 2.4 to show a pilot-level proposed-vs-baseline gain signal."""

    phase24_dir = Path(run_dir) / "phase2-4"
    outputs_dir = phase24_dir / "solver" / "outputs"
    results_path = outputs_dir / "validation_results.csv"
    if not results_path.exists():
        return {"ok": False, "errors": [f"missing validation results csv: {results_path}"], "warnings": []}
    try:
        with results_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = [dict(row) for row in csv.DictReader(handle)]
    except Exception as exc:
        return {"ok": False, "errors": [f"cannot read validation results csv: {exc}"], "warnings": []}
    if not rows:
        return {"ok": False, "errors": ["validation_results.csv has no data rows"], "warnings": []}
    try:
        plan = yaml.safe_load(read_text(phase24_dir / "validation_plan.yaml")) or {}
    except Exception as exc:
        return {"ok": False, "errors": [f"validation_plan.yaml is not valid YAML: {exc}"], "warnings": []}
    if not isinstance(plan, dict):
        plan = {}

    evidence = _phase24_research_evidence(plan)
    primary_payload = evidence.get("primary_metric") if isinstance(evidence.get("primary_metric"), dict) else {}
    primary_metric = str(primary_payload.get("name") or primary_payload.get("metric") or "objective").strip()
    higher_is_better = bool(primary_payload.get("higher_is_better", True))
    metric_aliases = _phase24_metric_aliases(primary_metric or "objective") + ["objective", "objective_value"]
    min_paired_seeds = max(1, int(os.environ.get("WARA_PHASE24_PILOT_MIN_PAIRED_SEEDS", "20") or 20))
    min_x_groups = max(1, int(os.environ.get("WARA_PHASE24_PILOT_MIN_X_GROUPS", "3") or 3))
    min_win_rate = float(os.environ.get("WARA_PHASE24_PILOT_MIN_WIN_RATE", "0.55") or 0.55)
    min_median_gain = float(os.environ.get("WARA_PHASE24_PILOT_MIN_MEDIAN_GAIN", "0.0") or 0.0)

    def method_id(row: dict[str, Any]) -> str:
        return str(row.get("method") or row.get("method_id") or "").strip().lower()

    def method_priority(method: str) -> int:
        method = str(method or "").strip().lower()
        if method == "baseline":
            return 0
        if any(token in method for token in ("rzf", "regularized", "fixed", "practical")):
            return 1
        if any(token in method for token in ("heuristic", "benchmark", "zf")):
            return 2
        if any(token in method for token in ("mrt", "matched")):
            return 3
        return 4

    def row_success(row: dict[str, Any]) -> bool:
        status_text = str(row.get("status") or "").strip().lower()
        status_failed = status_text in {"failed", "infeasible", "error", "exception", "nan", "timeout", "stress_infeasible"}
        return _phase24_truthy_cell(row.get("feasible")) and not status_failed

    def metric_value(row: dict[str, Any]) -> float | None:
        for alias in metric_aliases:
            value = _phase24_float_cell(row.get(alias))
            if value is not None and math.isfinite(float(value)):
                return float(value)
        return None

    def seed_value(row: dict[str, Any]) -> str:
        return str(row.get("seed") or row.get("realization_id") or row.get("mc_seed") or row.get("trial_id") or "").strip()

    def pair_key(row: dict[str, Any]) -> tuple[str, str, str, str, str]:
        return (
            str(row.get("case_id") or row.get("case_name") or ""),
            seed_value(row),
            str(row.get("swept_param") or ""),
            str(row.get("swept_value") or ""),
            str(row.get("scenario_name") or ""),
        )

    declared_methods: set[str] = set()
    compared_methods = evidence.get("compared_methods", [])
    if isinstance(compared_methods, list):
        for item in compared_methods:
            if isinstance(item, dict):
                method = str(item.get("id") or item.get("internal_name") or item.get("name") or "").strip().lower()
            else:
                method = str(item or "").strip().lower()
            if method:
                declared_methods.add(method)
    emitted_methods = {method_id(row) for row in rows if method_id(row)}
    baseline_methods = {
        method
        for method in (declared_methods | emitted_methods | {"baseline"})
        if method and method != "proposed"
    }

    by_key: dict[tuple[str, str, str, str, str], dict[str, dict[str, Any]]] = {}
    for row in rows:
        method = method_id(row)
        if method == "proposed":
            by_key.setdefault(pair_key(row), {})["proposed"] = row
            continue
        if method not in baseline_methods:
            continue
        key = pair_key(row)
        current = by_key.setdefault(key, {}).get("baseline")
        if current is None or method_priority(method) < method_priority(method_id(current)):
            baseline_row = dict(row)
            baseline_row["paired_baseline_method"] = method
            by_key[key]["baseline"] = baseline_row

    errors: list[str] = []
    warnings: list[str] = []
    repair_advice: list[str] = []
    if not any(seed_value(row) for row in rows):
        errors.append(
            "Phase 2.4 pilot validation has no seed column. Run multiple paired pilot seeds before Phase 2.5."
        )

    paired_records: list[dict[str, Any]] = []
    for key, group in by_key.items():
        proposed = group.get("proposed")
        baseline = group.get("baseline")
        if not proposed or not baseline:
            continue
        if not (row_success(proposed) and row_success(baseline)):
            continue
        proposed_value = metric_value(proposed)
        baseline_value = metric_value(baseline)
        if proposed_value is None or baseline_value is None:
            continue
        raw_gain = (proposed_value - baseline_value) / max(abs(baseline_value), 1.0e-9)
        relative_gain = raw_gain if higher_is_better else (-raw_gain)
        paired_records.append(
            {
                "case_id": key[0],
                "seed": key[1],
                "x_key": (key[2], key[3], key[4]),
                "relative_gain": float(relative_gain),
                "proposed_win": bool(relative_gain > 1.0e-6),
            }
        )

    x_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for record in paired_records:
        x_groups.setdefault(record["x_key"], []).append(record)
    qualified_x_groups = {
        key: group
        for key, group in x_groups.items()
        if len({str(item.get("seed") or "") for item in group}) >= min_paired_seeds
    }
    all_gains = [float(item["relative_gain"]) for item in paired_records]
    qualified_gains = [
        float(item["relative_gain"])
        for group in qualified_x_groups.values()
        for item in group
    ]
    gains_for_decision = qualified_gains or all_gains
    win_rate = (
        float(sum(1 for gain in gains_for_decision if gain > 1.0e-6) / len(gains_for_decision))
        if gains_for_decision
        else 0.0
    )
    median_gain = float(statistics.median(gains_for_decision)) if gains_for_decision else 0.0
    mean_gain = float(sum(gains_for_decision) / len(gains_for_decision)) if gains_for_decision else 0.0
    x_group_summaries = []
    for key, group in sorted(x_groups.items(), key=lambda item: str(item[0])):
        gains = [float(item["relative_gain"]) for item in group]
        x_group_summaries.append(
            {
                "swept_param": key[0],
                "swept_value": key[1],
                "scenario_name": key[2],
                "paired_seeds": len({str(item.get("seed") or "") for item in group}),
                "median_relative_gain": float(statistics.median(gains)) if gains else 0.0,
                "win_rate": float(sum(1 for gain in gains if gain > 1.0e-6) / len(gains)) if gains else 0.0,
            }
        )

    if len(qualified_x_groups) < min_x_groups:
        errors.append(
            f"Phase 2.4 pilot gain validation needs at least {min_x_groups} x-groups with "
            f">={min_paired_seeds} paired proposed-baseline seeds; found {len(qualified_x_groups)}."
        )
    if not gains_for_decision:
        errors.append(
            f"Phase 2.4 pilot gain validation found no comparable proposed-baseline rows on metric `{primary_metric}`."
        )
    else:
        if win_rate < min_win_rate:
            errors.append(f"Phase 2.4 pilot win rate is {win_rate:.3f}, below required {min_win_rate:.3f}.")
        if median_gain <= min_median_gain:
            errors.append(
                f"Phase 2.4 pilot median relative gain is {median_gain:.6g}, not above required {min_median_gain:.6g}."
            )

    if errors:
        repair_advice.append(
            "Repair Phase 2.4 with the LLM: keep the frozen formulation and algorithm route, but revise the "
            "experiment design or generated implementation so the proposed method and practical baseline are "
            "evaluated on paired pilot seeds, the declared KPI is responsive, and the proposed method shows a "
            "stable positive pilot gain before Phase 2.5 expansion."
        )

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "repair_advice": repair_advice,
        "blocking_errors": errors,
        "primary_metric": primary_metric,
        "higher_is_better": higher_is_better,
        "min_paired_seeds": min_paired_seeds,
        "min_x_groups": min_x_groups,
        "min_win_rate": min_win_rate,
        "min_median_gain": min_median_gain,
        "num_paired_records": len(paired_records),
        "num_x_groups": len(x_groups),
        "num_qualified_x_groups": len(qualified_x_groups),
        "pilot_win_rate": win_rate,
        "pilot_mean_relative_gain": mean_gain,
        "pilot_median_relative_gain": median_gain,
        "x_group_summaries": x_group_summaries,
    }


def validate_phase24_method_semantics(run_dir: Path) -> dict[str, Any]:
    """Reject baselines/ablations whose emitted diagnostics contradict their contract."""
    phase24_dir = Path(run_dir) / "phase2-4"
    outputs_dir = phase24_dir / "solver" / "outputs"
    results_paths = [
        outputs_dir / "validation_results.csv",
        outputs_dir / "paper_validation_results.csv",
    ]
    try:
        plan = yaml.safe_load(read_text(phase24_dir / "validation_plan.yaml")) or {}
    except Exception as exc:
        return {"ok": False, "errors": [f"validation_plan.yaml is not valid YAML: {exc}"], "warnings": []}
    if not isinstance(plan, dict):
        return {"ok": False, "errors": ["validation_plan.yaml did not parse to a mapping"], "warnings": []}
    rows: list[dict[str, Any]] = []
    read_paths: list[str] = []
    warnings: list[str] = []
    plugin_paths = [
        phase24_dir / "solver" / "generated_plugin.py",
        phase24_dir / "solver" / "generated_experiment_core.py",
    ]
    plugin_mtimes = [path.stat().st_mtime for path in plugin_paths if path.exists()]
    plugin_mtime = max(plugin_mtimes) if plugin_mtimes else None
    plugin_source_text = "\n".join(read_text(path) for path in plugin_paths if path.exists()).lower()
    source_has_cvxpy_solver = (
        ("import cvxpy" in plugin_source_text or "from cvxpy" in plugin_source_text)
        and ("cp.variable" in plugin_source_text or "cvxpy.variable" in plugin_source_text)
        and ("cp.problem" in plugin_source_text or "cvxpy.problem" in plugin_source_text)
    )
    for results_path in results_paths:
        if not results_path.exists():
            continue
        if (
            results_path.name == "paper_validation_results.csv"
            and plugin_mtime is not None
            and results_path.stat().st_mtime + 1.0 < plugin_mtime
        ):
            warnings.append(
                f"ignored stale `{results_path.name}` because it predates the current generated_plugin.py; "
                "paper-mode semantics will be checked after the current plugin regenerates paper validation results"
            )
            continue
        try:
            with results_path.open("r", encoding="utf-8-sig", newline="") as handle:
                rows.extend(dict(row) for row in csv.DictReader(handle))
                read_paths.append(str(results_path))
        except Exception as exc:
            return {"ok": False, "errors": [f"cannot read validation results csv `{results_path}`: {exc}"], "warnings": []}
    if not rows:
        return {"ok": False, "errors": [f"missing validation results csv under {outputs_dir}"], "warnings": []}

    evidence = _phase24_research_evidence(plan)
    method_entries = evidence.get("compared_methods", [])
    if not isinstance(method_entries, list):
        method_entries = []
    method_roles: dict[str, str] = {}
    for entry in method_entries:
        if isinstance(entry, dict):
            method_id = str(entry.get("id") or entry.get("internal_name") or entry.get("name") or "").strip().lower()
            role = str(entry.get("role") or "").strip().lower()
        else:
            method_id = str(entry or "").strip().lower()
            role = ""
        if method_id:
            method_roles[method_id] = role

    rows_by_method: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        method = str(row.get("method", "")).strip().lower()
        if method:
            rows_by_method.setdefault(method, []).append(row)

    errors: list[str] = []
    method_fidelity_contract = read_json(phase24_dir / "phase24_method_fidelity_contract.json") or {}
    if bool(method_fidelity_contract.get("route_requires_cvxpy_solver_path")):
        proposed_rows = rows_by_method.get("proposed", [])
        if not proposed_rows:
            errors.append(
                "method fidelity contract requires a CVXPY-backed proposed solver path, "
                "but validation emitted no `proposed` rows."
            )
        else:
            cvxpy_status_values = {"optimal", "optimal_inaccurate", "user_limit"}
            explicit_cvx_used_rows = [
                row
                for row in proposed_rows
                if (
                    _phase24_truthy_cell(row.get("cvxpy_solver_used"))
                    or _phase24_truthy_cell(row.get("cvxpy_success"))
                    or str(row.get("cvxpy_status") or "").strip().lower() in cvxpy_status_values
                )
            ]
            inferred_cvx_used_rows = [
                row
                for row in proposed_rows
                if source_has_cvxpy_solver
                and (
                    str(row.get("solver_status") or "").strip().lower() in cvxpy_status_values
                    or "after_cvxpy" in str(row.get("solver_status") or "").strip().lower()
                )
            ]
            cvx_used_rows = [
                *explicit_cvx_used_rows,
                *(row for row in inferred_cvx_used_rows if row not in explicit_cvx_used_rows),
            ]
            failed_rows = [
                row
                for row in proposed_rows
                if "cvxpy_failed" in str(row.get("solver_status") or row.get("message") or "").lower()
            ]
            if not cvx_used_rows:
                example_row = proposed_rows[0] if proposed_rows else {}
                example_diag = (
                    "Example proposed-row diagnostics: "
                    f"cvxpy_solver_used={str(example_row.get('cvxpy_solver_used', '')).strip() or '<missing>'}, "
                    f"cvxpy_success={str(example_row.get('cvxpy_success', '')).strip() or '<missing>'}, "
                    f"cvxpy_status={str(example_row.get('cvxpy_status', '')).strip() or '<missing>'}, "
                    f"solver_status={str(example_row.get('solver_status') or example_row.get('message') or '').strip() or '<missing>'}."
                )
                errors.append(
                    "method fidelity contract requires a CVXPY solver path, but proposed-method validation "
                    "does not provide explicit CVXPY diagnostics or source-backed CVXPY solver evidence. "
                    f"{example_diag} "
                    "Do not advance evidence generated by a non-CVXPY substitute; repair the LLM-generated "
                    "experiment core so the declared CVXPY subproblem solves successfully and emits solver "
                    "diagnostics, or route back to Phase 2.3 to change the algorithm contract."
                )
            elif inferred_cvx_used_rows and not explicit_cvx_used_rows:
                warnings.append(
                    "proposed-method rows do not emit explicit `cvxpy_solver_used`, `cvxpy_success`, or "
                    "`cvxpy_status` diagnostics; accepting source-level CVXPY markers plus CVXPY-like "
                    "`solver_status` values for this gate, but future repairs should emit explicit CVXPY "
                    "diagnostic columns."
                )
            if failed_rows:
                example_status = str(failed_rows[0].get("solver_status") or failed_rows[0].get("message") or "")
                errors.append(
                    "proposed-method validation rows contain `cvxpy_failed`, followed by continued evidence emission "
                    f"(example solver_status: `{example_status}`). This is a solver-path substitution and violates the "
                    "method fidelity contract; Phase 2.4 must request LLM repair instead of promoting these results."
                )

    figure_active_methods: set[str] = set()
    figure_metrics: set[str] = set()
    figures = evidence.get("figures", [])
    if isinstance(figures, list):
        for figure in figures:
            if not isinstance(figure, dict):
                continue
            metric = _phase24_figure_y_metric(figure)
            if metric:
                figure_metrics.add(metric)
            methods_to_run = figure.get("methods_to_run")
            if isinstance(methods_to_run, list):
                for item in methods_to_run:
                    if isinstance(item, dict):
                        method_id = str(item.get("id") or item.get("internal_name") or item.get("name") or "").strip()
                    else:
                        method_id = str(item or "").strip()
                    if method_id:
                        figure_active_methods.add(method_id.lower())

    def row_feasible(row: dict[str, Any]) -> bool:
        status_text = str(row.get("status", "")).strip().lower()
        status_failed = status_text in {"failed", "infeasible", "error", "exception", "nan", "timeout", "stress_infeasible"}
        return _phase24_truthy_cell(row.get("feasible")) and not status_failed

    def row_uncertainty(row: dict[str, Any]) -> float | None:
        for key in (
            "actual_used_constraints_uncertainty_radius",
            "uncertainty_radius",
            "constraints_uncertainty_radius",
            "epsilon",
            "epsilon_design",
            "swept_value",
        ):
            value = _phase24_float_cell(row.get(key))
            if value is not None:
                return float(value)
        return None

    def row_sinr_target_db(row: dict[str, Any]) -> float | None:
        for key in (
            "actual_used_constraints_sinr_target_dB",
            "sinr_target_dB",
            "constraints_sinr_target_dB",
            "gamma_min_dB",
        ):
            value = _phase24_float_cell(row.get(key))
            if value is not None:
                return float(value)
        return None

    def is_low_stress_row(row: dict[str, Any]) -> bool:
        eps = row_uncertainty(row)
        gamma_db = row_sinr_target_db(row)
        eps_ok = eps is None or eps <= 0.03
        gamma_ok = gamma_db is None or gamma_db <= 10.0
        return bool(eps_ok and gamma_ok)

    def median_for_metric(method_rows: list[dict[str, Any]], *keys: str) -> float | None:
        values: list[float] = []
        for row in method_rows:
            for key in keys:
                value = _phase24_float_cell(row.get(key))
                if value is not None and math.isfinite(float(value)):
                    values.append(float(value))
                    break
        return _phase24_median(values)

    def row_pair_key(row: dict[str, Any]) -> tuple[str, str, str, str, str]:
        return (
            str(row.get("case_id") or row.get("case_name") or ""),
            str(row.get("seed") or ""),
            str(row.get("swept_param") or ""),
            str(row.get("swept_value") or ""),
            str(row.get("scenario_name") or ""),
        )

    proposed_low_stress_keys = {
        row_pair_key(row)
        for row in rows_by_method.get("proposed", [])
        if is_low_stress_row(row)
    }

    active_methods_with_rows = sorted(method for method in figure_active_methods if method in rows_by_method)
    comparison_metrics = [metric for metric in sorted(figure_metrics | {"objective"}) if metric]
    for left_index, left_method in enumerate(active_methods_with_rows):
        left_rows = {row_pair_key(row): row for row in rows_by_method.get(left_method, [])}
        for right_method in active_methods_with_rows[left_index + 1 :]:
            right_rows = {row_pair_key(row): row for row in rows_by_method.get(right_method, [])}
            common_keys = sorted(set(left_rows) & set(right_rows))
            if len(common_keys) < 3:
                continue
            checked_metrics: list[str] = []
            max_abs_diff = 0.0
            comparable_values = 0
            for metric in comparison_metrics:
                metric_diffs: list[float] = []
                for key in common_keys:
                    left_value = _phase24_float_cell(left_rows[key].get(metric))
                    right_value = _phase24_float_cell(right_rows[key].get(metric))
                    if left_value is None or right_value is None:
                        continue
                    if not (math.isfinite(float(left_value)) and math.isfinite(float(right_value))):
                        continue
                    metric_diffs.append(abs(float(left_value) - float(right_value)))
                if not metric_diffs:
                    continue
                checked_metrics.append(metric)
                comparable_values += len(metric_diffs)
                max_abs_diff = max(max_abs_diff, max(metric_diffs))
            if checked_metrics and comparable_values >= 3 and max_abs_diff <= 1.0e-10:
                message = (
                    f"active plotted methods `{left_method}` and `{right_method}` produce numerically identical "
                    f"{', '.join(checked_metrics)} values across {len(common_keys)} matched case/seed/sweep rows. "
                    "This usually means duplicate method dispatch or a redundant benchmark."
                )
                left_is_proposed = left_method == "proposed" or "proposed" in method_roles.get(left_method, "")
                right_is_proposed = right_method == "proposed" or "proposed" in method_roles.get(right_method, "")
                if left_is_proposed or right_is_proposed:
                    errors.append(
                        message
                        + " The proposed method must not be numerically identical to an active comparator; "
                        "repair the method implementation or remove the invalid comparator before writing."
                    )
                else:
                    errors.append(
                        message
                        + " Active benchmark curves must be semantically distinct; remove or repair the redundant "
                        "comparator before evidence promotion."
                    )

    for item in method_entries:
        if isinstance(item, dict):
            method_id = str(item.get("id") or item.get("internal_name") or item.get("name") or "").strip()
            contract_text = " ".join(
                str(item.get(key) or "")
                for key in (
                    "id",
                    "internal_name",
                    "name",
                    "role",
                    "display_name_short",
                    "display_name_long",
                    "scientific_purpose",
                    "implementation_hint",
                    "fairness_rule",
                )
            ).lower()
        else:
            method_id = str(item or "").strip()
            contract_text = method_id.lower()
        if not method_id:
            continue
        method_key = method_id.lower()
        method_rows = rows_by_method.get(method_key, [])
        if not method_rows and method_key not in {"proposed", "baseline"}:
            warnings.append(f"declared method `{method_id}` has no quick-validation rows")
            continue

        practical_contract = (
            method_key not in {"proposed", "baseline"}
            and (
                "heuristic" in contract_text
                or "benchmark" in contract_text
                or "matched" in contract_text
                or "regularized" in contract_text
            )
        )
        robust_loading_contract = practical_contract and (
            "power-loading" in contract_text
            or "robust loading" in contract_text
            or "robust feasibility test" in contract_text
            or "same robust" in contract_text
            or "same uncertainty" in contract_text
        )
        if robust_loading_contract and method_rows:
            low_stress_rows = [row for row in method_rows if is_low_stress_row(row)]
            feasible_rate = sum(1 for row in method_rows if row_feasible(row)) / max(len(method_rows), 1)
            low_feasible_rate = sum(1 for row in low_stress_rows if row_feasible(row)) / max(len(low_stress_rows), 1)
            paired_low_stress_count = sum(1 for row in low_stress_rows if row_pair_key(row) in proposed_low_stress_keys)
            violation_probability_median = median_for_metric(method_rows, "violation_probability", "outage_probability")
            power_median = median_for_metric(method_rows, "sum_power_W", "power_total", "total_power", "objective")
            shortfall_median = median_for_metric(
                method_rows,
                "sampled_sinr_shortfall_max",
                "safe_soc_violation_max",
                "constraint_violation_max",
                "max_constraint_violation",
            )
            if low_stress_rows and low_feasible_rate == 0.0 and paired_low_stress_count > 0:
                warnings.append(
                    f"method `{method_id}` is declared as a practical robust-loading benchmark, but it has zero feasible "
                    "low-stress rows on matched proposed low-stress cases. Feasible-rate is diagnostic only; inspect "
                    "the benchmark directions/power loading if the plotted comparison looks broken."
                )
            if (
                feasible_rate == 0.0
                and violation_probability_median is not None
                and violation_probability_median >= 0.99
                and power_median is not None
                and power_median > 1.0e3
                and shortfall_median is not None
                and shortfall_median > 1.0e-3
            ):
                warnings.append(
                    f"method `{method_id}` reports infeasible/violating rows for every validation sample despite median "
                    f"power {power_median:.4g}. Feasible-rate is diagnostic only, but this may indicate a numerically "
                    "broken comparator. For MISO downlink ZF/RZF code, check that directions are built for the same "
                    "`h_k^H w_j` convention used by evaluate_state."
                )

        rho_values = _phase24_numeric_values_for_aliases(method_rows, _phase24_metric_aliases("rho"))
        rho_median = _phase24_median(rho_values)
        no_rho_contract = (
            method_key in {"no_rho", "no_structural_sep", "no_structural_separation"}
            or "no_rho" in method_key
            or "no structural separation" in contract_text
            or bool(re.search(r"\brho\s*=\s*0(?:\.0+)?(?![\d.])", contract_text))
        )
        if no_rho_contract:
            if not rho_values:
                warnings.append(
                    f"method `{method_id}` is declared as a no-rho/no-structural-separation ablation but emits no rho diagnostic"
                )
            elif max(abs(value) for value in rho_values) > 1.0e-3:
                errors.append(
                    f"method `{method_id}` is declared as no-rho/no-structural-separation, "
                    f"but validation rows report rho median={rho_median:.4g} and max_abs={max(abs(value) for value in rho_values):.4g}. "
                    "Implement method_solution so this ablation sets rho = 0.0 exactly and removes the structural-separation/EH-partition mechanism "
                    "instead of using an arbitrary fixed rho such as 0.5 or 1.0."
                )
            eh_partition_values = _phase24_numeric_values_for_aliases(method_rows, _phase24_metric_aliases("M_eh"))
            if eh_partition_values and max(abs(value) for value in eh_partition_values) > 1.0e-3:
                errors.append(
                    f"method `{method_id}` is declared as no-rho/no-structural-separation, "
                    f"but validation rows report nonzero EH partition count max={max(eh_partition_values):.4g}. "
                    "For this ablation, the no-rho path must expose diagnostics consistent with no structural separation "
                    "(for example rho=0.0 and M_eh=0 when M_eh is reported)."
                )

        fixed_half_contract = (
            "rho_fixed_half" in method_key
            or "fixed half" in contract_text
            or bool(re.search(r"\brho\s*=\s*0\.5(?:0+)?(?!\d)", contract_text))
        )
        if fixed_half_contract and rho_values:
            max_dev = max(abs(value - 0.5) for value in rho_values)
            if max_dev > 5.0e-2:
                errors.append(
                    f"method `{method_id}` is declared as fixed rho=0.5, but validation rows report "
                    f"rho median={rho_median:.4g} with max deviation {max_dev:.4g}."
                )

    return {"ok": not errors, "errors": errors, "warnings": warnings, "checked_result_files": read_paths}


def _phase24_bool_env(name: str, default: bool = False) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return str(raw_value).strip().lower() in {"1", "true", "yes", "on"}


def _phase24_float_env(name: str, default: float) -> float:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except (TypeError, ValueError):
        return default


def validate_phase24_runtime_budget(run_dir: Path) -> dict[str, Any]:
    """Runtime-duration checks are advisory only; they must not block a run."""
    phase24_dir = Path(run_dir) / "phase2-4"
    results_path = phase24_dir / "solver" / "outputs" / "validation_results.csv"
    if not results_path.exists():
        return {"ok": True, "errors": [], "warnings": ["runtime-duration check skipped: validation_results.csv missing"]}
    return {
        "ok": True,
        "errors": [],
        "warnings": ["runtime-duration check disabled: WARA does not stop faithful experiments by estimated wall-clock limits"],
        "disabled": True,
    }
    # Legacy timing analysis is intentionally unreachable. It is kept below only
    # for historical comparison while the active gate above remains non-blocking.
    plugin_paths = [
        phase24_dir / "solver" / "generated_plugin.py",
        phase24_dir / "solver" / "generated_experiment_core.py",
    ]
    plugin_mtime = max((path.stat().st_mtime for path in plugin_paths if path.exists()), default=0.0)
    if plugin_mtime and results_path.stat().st_mtime + 1.0 < plugin_mtime:
        return {
            "ok": True,
            "errors": [],
            "warnings": ["runtime budget skipped: existing validation_results.csv predates current generated code"],
            "stale_results": True,
        }

    slow_is_failure = _phase24_bool_env("WARA_PHASE25_RUNTIME_SLOW_IS_FAILURE", default=False)
    max_median_proposed_sec = _phase24_float_env(
        "WARA_PHASE25_MAX_MEDIAN_SOLVE_TIME_SEC",
        _phase24_float_env("WARA_PHASE24_MAX_MEDIAN_SOLVE_TIME_SEC", 20.0),
    )
    max_median_case_sec = _phase24_float_env(
        "WARA_PHASE25_MAX_MEDIAN_CASE_TIME_SEC",
        _phase24_float_env("WARA_PHASE24_MAX_MEDIAN_CASE_TIME_SEC", max_median_proposed_sec * 2.0),
    )
    if max_median_proposed_sec <= 0 and max_median_case_sec <= 0:
        return {"ok": True, "errors": [], "warnings": ["runtime budget disabled by non-positive thresholds"]}

    rows: list[dict[str, Any]] = []
    with results_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]

    proposed_times: list[float] = []
    case_times: dict[tuple[str, str], float] = {}
    for row in rows:
        try:
            elapsed = float(row.get("solve_time_sec") or row.get("measured_solve_time_sec") or 0.0)
        except (TypeError, ValueError):
            elapsed = 0.0
        if not math.isfinite(elapsed) or elapsed < 0.0:
            continue
        case_key = (str(row.get("case_id") or row.get("case_name") or ""), str(row.get("seed") or "0"))
        case_times[case_key] = case_times.get(case_key, 0.0) + elapsed
        if str(row.get("method") or "").strip().lower() == "proposed":
            proposed_times.append(elapsed)

    proposed_median = float(statistics.median(proposed_times)) if proposed_times else 0.0
    case_median = float(statistics.median(case_times.values())) if case_times else 0.0
    errors: list[str] = []
    warnings: list[str] = []
    if max_median_proposed_sec > 0 and proposed_median > max_median_proposed_sec:
        message = (
            "Phase 2.4 quick validation is too slow for automatic Phase 2.5 expansion: "
            f"median proposed solve time is {proposed_median:.3f}s, threshold is {max_median_proposed_sec:.3f}s. "
            "Repair generated_experiment_core.py so the proposed update is paper-sweep scalable: reduce nested CVXPY solves, "
            "use compact warm-started subproblems or analytic/vectorized block updates, cap inner iterations from model metadata, "
            "and keep the same objective, constraints, methods, and swept-path responsiveness."
        )
        (errors if slow_is_failure else warnings).append(message)
    if max_median_case_sec > 0 and case_median > max_median_case_sec:
        message = (
            "Phase 2.4 quick validation case runtime is too slow for automatic Phase 2.5 expansion: "
            f"median all-method case time is {case_median:.3f}s, threshold is {max_median_case_sec:.3f}s. "
            "The experiment core must be efficient enough for scout/medium Monte Carlo before the controller can request paper sweeps."
        )
        (errors if slow_is_failure else warnings).append(message)

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "slow_is_failure": slow_is_failure,
        "max_median_proposed_sec": max_median_proposed_sec,
        "max_median_case_sec": max_median_case_sec,
        "median_proposed_solve_time_sec": proposed_median,
        "median_case_time_sec": case_median,
        "num_rows": len(rows),
        "num_cases": len(case_times),
        "num_proposed_rows": len(proposed_times),
    }


def validate_phase24_algorithm_code_contract(run_dir: Path) -> dict[str, Any]:
    """Reject generated plugins that silently implement a different algorithm family."""
    run_dir = Path(run_dir)
    phase24_dir = run_dir / "phase2-4"
    algorithm_md = read_text(run_dir / "phase2-3" / "algorithm.md")
    reformulation_md = read_text(run_dir / "phase2-2" / "reformulation_path.md")
    validation_plan_text = read_text(phase24_dir / "validation_plan.yaml")
    algorithm_contract = read_json(run_dir / "phase2-2" / "algorithm_contract.json") or {}
    algorithm_family = str(algorithm_contract.get("algorithm_family") or "").strip().lower()
    plugin_code = _phase24_combined_generated_source(phase24_dir / "solver") or read_text(phase24_dir / "solver" / "generated_plugin.py")
    algo_lower = f"{algorithm_md}\n{reformulation_md}".lower()
    code_lower = plugin_code.lower()
    errors: list[str] = []
    warnings: list[str] = []
    repair_advice: list[str] = []
    strict_research_gate = _phase24_strict_research_gate()
    strict_fidelity_required_by_contract = algorithm_family in {
        "sdp_or_sdr",
        "sca_or_mm",
        "wmmse_block_coordinate",
        "convex_conic",
        "convex_program",
    } or _phase2_has_any(
        algo_lower,
        [
            "cvxpy",
            "sdp",
            "semidefinite",
            "sdr",
            "linear matrix inequality",
            "conic program",
            "convex subproblem",
            "successive convex",
            "wmmse",
        ],
    )

    def route_issue(message: str, *, hard: bool = False) -> None:
        if hard or strict_research_gate or strict_fidelity_required_by_contract:
            errors.append(message)
            repair_advice.append(message)
        else:
            repair_advice.append(message)
            warnings.append(message)

    fallback_reason_path = phase24_dir / "phase24_generated_plugin_fallback_reason.txt"
    if fallback_reason_path.exists() and read_text(fallback_reason_path).strip():
        errors.append(
            "Phase 2.4 used an experiment fallback/reference plugin. Experiment fallbacks are disabled; regenerate "
            "generated_experiment_core.py from the current Phase 2.3 algorithm before producing Phase 2.5 evidence."
        )

    claims_exact_convex_solver = _phase2_has_any(
        algo_lower,
        [
            "sdp",
            "semidefinite",
            "cvx",
            "cvxpy",
            "interior-point",
            "interior point",
            "conic program",
            "second-order cone",
            "second order cone",
            "socp",
            "solve the exact convex",
            "exact convex",
        ],
    )
    claims_convex_subproblem = _phase2_has_any(algo_lower, ["convex subproblem", "convex quadratic subproblem"])
    allows_first_order_or_closed_form = _phase2_has_any(
        algo_lower,
        [
            "projected gradient",
            "projected-gradient",
            "gradient descent",
            "water-filling",
            "water filling",
            "closed-form",
            "closed form",
            "block-coordinate",
            "block coordinate",
            "manifold descent",
            "riemannian descent",
        ],
    )
    first_order_algorithm_families = {
        "heuristic",
        "heuristic_empirical",
        "mixed_discrete_or_manifold",
        "wmmse_block_coordinate",
    }
    claims_convex_solver = claims_exact_convex_solver or (
        claims_convex_subproblem
        and algorithm_family not in first_order_algorithm_families
        and not allows_first_order_or_closed_form
    )
    claims_sdr = _phase2_has_any(algo_lower, ["sdr", "rank-one recovery", "gaussian randomization"])
    claims_wmmse = "wmmse" in algo_lower
    claims_sca = _phase2_has_any(algo_lower, ["sca", "successive convex", "successive approximation"])
    if re.search(r"\b(?:no|not|without)\b.{0,80}\b(?:sdr|rank-one recovery|gaussian randomization)\b", algo_lower):
        claims_sdr = False
    if algorithm_family == "fixed_point_or_lp_reference":
        # Phase 2.3 often discusses WMMSE/SCA/SDR as methods that are unnecessary
        # for standard uplink power control. The frozen algorithm contract is the
        # implementation source of truth, so comparison-only mentions must not
        # force a wrong code obligation.
        claims_sdr = False
        claims_wmmse = False
        claims_sca = False
    elif algorithm_family in {"sdp_or_sdr", "wmmse_block_coordinate", "sca_or_mm"}:
        claims_sdr = claims_sdr and algorithm_family == "sdp_or_sdr"
        claims_wmmse = claims_wmmse and algorithm_family == "wmmse_block_coordinate"
        claims_sca = claims_sca and algorithm_family == "sca_or_mm"

    iteration_cap_errors = _phase24_iteration_cap_mismatch_errors(plugin_code, validation_plan_text)
    for message in iteration_cap_errors:
        errors.append(message)
        repair_advice.append(message)

    has_cvxpy = "import cvxpy" in code_lower or "from cvxpy" in code_lower or "cp.problem" in code_lower
    has_cvxpy_solver_path = has_cvxpy and _phase2_has_any(
        code_lower,
        ["cp.problem", "cp.minimize", "cp.maximize", ".solve("],
    )
    has_randomization = _phase2_has_any(code_lower, ["gaussian", "randomization", "eigh", "eigval", "eigvec", "np.random"])
    has_wmmse = "wmmse" in code_lower
    has_wmmse_update_logic = has_wmmse and _phase2_has_any(
        code_lower,
        ["eta_k", "eta[", "mse_weight", "mse weights", "receive_filter", "mmse_receiver", "e_k(", "u_k"],
    )
    has_fp_wmmse_equivalent_logic = (
        _phase2_has_any(code_lower, ["_fp_gamma", "fp_gamma", "quadratic transform", "gamma_k", "gamma ="])
        and _phase2_has_any(code_lower, ["qcqp", "bisection", "dual", "power multiplier", "_w_qcqp"])
        and _phase2_has_any(code_lower, ["sinr", "rate", "wsr"])
    )
    has_wmmse_update_logic = has_wmmse_update_logic or has_fp_wmmse_equivalent_logic
    has_iterative_step = "def proposed_step" in code_lower and _phase2_has_any(code_lower, ["iteration", "for _", "for i", "while "])
    has_projected_gradient_sca = (
        _phase2_has_any(code_lower, ["projected gradient", "projected-gradient", "projected_gradient", "_p_pg_update", "pg_update"])
        and _phase2_has_any(code_lower, ["majorizer", "spacing_penalty", "project_to_box", "sca"])
    )
    marks_proxy = _phase2_has_any(
        code_lower,
        [
            "heuristic_proxy",
            "surrogate_proxy",
            "proxy_mode",
            "lightweight_candidate_search",
            "candidate_search",
            "_best_candidate",
            "reference plugin",
        ],
    )
    has_forbidden_fallback_marker = _phase2_has_any(
        code_lower,
        ["reference plugin", "fallback plugin", "deterministic fallback", "fallback_reason"],
    )
    has_exact_linear_power_control = (
        _phase2_has_any(algo_lower, ["linear program", "lp", "spectral radius", "fixed-point", "fixed point"])
        and _phase2_has_any(code_lower, ["np.linalg.solve", "linalg.solve", "spectral_radius", "fixed_point", "proposed_step"])
        and _phase2_has_any(code_lower, ["sum_power", "gamma_target", "sinr", "constraint_residual"])
    )

    if has_forbidden_fallback_marker:
        errors.append(
            "Generated experiment code contains a fallback/reference-plugin marker. Fallback experiment code must not produce paper figures."
        )
    elif marks_proxy and not has_exact_linear_power_control:
        route_issue(
            "Generated experiment code is a proxy/fallback/candidate-search implementation. Proxy experiment code "
            "should not be used as final paper evidence unless the paper scope explicitly presents it as a heuristic/empirical route."
        )
    if claims_convex_solver and marks_proxy and not has_cvxpy_solver_path and not has_exact_linear_power_control:
        route_issue(
            "Phase 2.3 claims a convex/CVX/SDP-style solver, but generated_plugin.py explicitly marks a heuristic/proxy implementation. "
            "Do not let proxy code produce paper figures; regenerate Phase 2.3/2.4 with an honest algorithm route."
        )
    elif claims_convex_solver and marks_proxy:
        warnings.append(
            "Generated code contains an explicit proxy fallback marker, but a CVXPY/exact solver path is present. "
            "Phase 2.5 validation must reject rows whose runtime algorithm_approximation/proxy diagnostics are not `none`."
        )
    elif claims_convex_solver and not has_cvxpy_solver_path and not marks_proxy and not has_exact_linear_power_control:
        route_issue(
            "Phase 2.3 claims a convex/CVX/SDP-style solver, but generated_plugin.py does not use CVXPY and does not explicitly mark a heuristic proxy."
        )
    if claims_sdr and marks_proxy:
        route_issue(
            "Phase 2.3 claims SDR/rank recovery/Gaussian randomization, but generated_plugin.py is only a heuristic/proxy. "
            "Proxy SDR results are not acceptable paper evidence."
        )
    elif claims_sdr and not (has_cvxpy and has_randomization) and not marks_proxy:
        route_issue(
            "Phase 2.3 claims SDR/rank recovery/Gaussian randomization, but the plugin lacks semidefinite solving plus recovery/randomization logic."
        )
    if claims_wmmse and not has_wmmse_update_logic:
        route_issue("Phase 2.3 claims WMMSE, but generated_plugin.py does not implement WMMSE update logic.")
    if claims_sca and not has_iterative_step:
        route_issue("Phase 2.3 claims SCA/successive approximation, but generated_plugin.py lacks an iterative proposed-step implementation.")
    if claims_sca and not has_cvxpy_solver_path and not has_exact_linear_power_control and not has_projected_gradient_sca:
        route_issue(
            "Phase 2.3 claims SCA/successive convex approximation, but generated_plugin.py does not solve a convex subproblem "
            "with CVXPY, an explicit exact linear-power-control route, or a documented projected-gradient SCA update."
        )
    mentions_crb_or_fim = _phase2_has_any(algo_lower, ["crb", "fisher information", "fim"])
    validation_requires_crb_or_fim = _phase2_has_any(
        validation_plan_text.lower(),
        ["crb", "fisher", "fim", "jacobian"],
    )
    code_has_crb_or_fim = _phase2_has_any(code_lower, ["crb", "fisher", "fim", "jacobian"])
    code_has_fim_surrogate = code_has_crb_or_fim or _phase2_has_any(
        code_lower,
        ["sensing_logdet", "logdet", "log_det", "information_matrix", "fisher_logdet"],
    )
    code_has_sensing_metric = _phase2_has_any(
        code_lower,
        ["sensing", "radar", "beampattern", "beam_pattern", "sensing_snr", "radar_snr", "illumination", "logdet"],
    )
    if mentions_crb_or_fim and validation_requires_crb_or_fim and not code_has_fim_surrogate:
        route_issue(
            "Phase 2.3/validation plan explicitly requires CRB/FIM evidence, but the plugin computes no CRB/FIM-related metric."
        )
    elif mentions_crb_or_fim and not code_has_fim_surrogate:
        message = (
            "Phase 2.3 mentions CRB/FIM, but Phase 2.4 implements a different sensing metric. "
            "This is allowed only when the validation plan does not request CRB/FIM figures; the writing stage must avoid CRB/FIM claims."
        )
        if code_has_sensing_metric:
            warnings.append(message)
            repair_advice.append(message)
        else:
            route_issue("Phase 2.3 mentions CRB/FIM or sensing, but the plugin computes no sensing-related metric.")
    if "multiuser" in algo_lower or "multi-user" in algo_lower:
        if "sinr" in code_lower and _phase2_has_any(code_lower, ["r_x", "rx"]) and not re.search(r"\b(Wk|W_k|w_k|beams|beam_vectors|covariances)\b", plugin_code):
            warnings.append(
                "Plugin appears to compute multiuser SINR with aggregate covariance naming. Verify per-user beams/covariances are actually used."
            )

    report = {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "repair_advice": repair_advice,
        "blocking_errors": errors,
        "strict_research_gate": strict_research_gate,
        "strict_fidelity_required_by_contract": strict_fidelity_required_by_contract,
        "research_readiness": "blocking_failed" if errors else ("advisory_only" if repair_advice else "ready"),
    }
    write_text(phase24_dir / "phase24_algorithm_code_contract_check.json", json.dumps(report, ensure_ascii=False, indent=2))
    return report


def validate_phase2_phase24_plugin_bundle(run_dir: Path) -> dict[str, Any]:
    phase24_dir = run_dir / "phase2-4"
    solver_dir = phase24_dir / "solver"
    codegen_status = _validate_phase24_split_codegen_package(phase24_dir)
    write_text(phase24_dir / "phase24_codegen_package_check.json", json.dumps(codegen_status, ensure_ascii=False, indent=2))
    if not codegen_status["ok"]:
        error_path = phase24_dir / "phase24_validation_error.txt"
        write_text(error_path, "[codegen_package]\n" + "\n".join(codegen_status["errors"]))
        return {
            "status": "codegen_package_failed",
            "returncode": 1,
            "error_path": str(error_path),
            "codegen_version": PHASE24_SPLIT_ADAPTER_VERSION,
        }

    plugin_status = validate_phase2_phase24_plugin_interfaces(solver_dir)
    if not plugin_status["ok"]:
        error_path = phase24_dir / "phase24_interface_errors.txt"
        write_text(error_path, "\n".join(plugin_status["errors"]))
        write_text(phase24_dir / "phase24_validation_error.txt", "[interface]\n" + "\n".join(plugin_status["errors"]))
        return {"status": "interface_failed", "returncode": 1, "error_path": str(phase24_dir / "phase24_validation_error.txt")}

    schema_status = validate_phase2_phase24_schema_alignment(run_dir)
    write_text(phase24_dir / "phase24_schema_alignment.json", json.dumps(schema_status, ensure_ascii=False, indent=2))
    if not schema_status["ok"]:
        error_path = phase24_dir / "phase24_validation_error.txt"
        write_text(error_path, "[schema_alignment]\n" + "\n".join(schema_status["errors"]))
        return {"status": "schema_alignment_failed", "returncode": 1, "error_path": str(error_path)}

    evidence_design_status = validate_phase24_evidence_contract_design(run_dir)
    write_text(phase24_dir / "phase24_evidence_contract_design_check.json", json.dumps(evidence_design_status, ensure_ascii=False, indent=2))
    if not evidence_design_status["ok"]:
        error_path = phase24_dir / "phase24_validation_error.txt"
        write_text(error_path, "[research_evidence_contract_design]\n" + "\n".join(evidence_design_status["errors"]))
        return {"status": "evidence_contract_design_failed", "returncode": 0, "error_path": str(error_path)}

    algorithm_code_status = validate_phase24_algorithm_code_contract(run_dir)
    if not algorithm_code_status["ok"]:
        error_path = phase24_dir / "phase24_validation_error.txt"
        write_text(error_path, "[algorithm_code_contract]\n" + "\n".join(algorithm_code_status["errors"]))
        return {"status": "algorithm_code_contract_failed", "returncode": 0, "error_path": str(error_path)}

    py_files = sorted(solver_dir.glob("*.py"))
    compile_timeout = _phase24_timeout_seconds("WARA_PHASE24_COMPILE_TIMEOUT_SEC", 120)
    try:
        compile_result = subprocess.run(
            [sys.executable, "-m", "py_compile", *[str(path.resolve()) for path in py_files]],
            cwd=solver_dir,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=compile_timeout if compile_timeout > 0 else None,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = _phase24_timeout_stream(exc.stdout)
        stderr = _phase24_timeout_stream(exc.stderr)
        error_path = phase24_dir / "phase24_validation_error.txt"
        write_text(phase24_dir / "phase24_py_compile_stdout.txt", stdout)
        write_text(phase24_dir / "phase24_py_compile_stderr.txt", stderr)
        write_text(error_path, f"[py_compile_timeout]\nExceeded {compile_timeout} seconds.\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}")
        return {"status": "compile_timeout", "returncode": 124, "error_path": str(error_path)}
    write_text(phase24_dir / "phase24_py_compile_stdout.txt", compile_result.stdout)
    write_text(phase24_dir / "phase24_py_compile_stderr.txt", compile_result.stderr)
    if compile_result.returncode != 0:
        error_path = phase24_dir / "phase24_validation_error.txt"
        write_text(error_path, f"[py_compile]\nSTDOUT:\n{compile_result.stdout}\nSTDERR:\n{compile_result.stderr}")
        return {"status": "compile_failed", "returncode": compile_result.returncode, "error_path": str(error_path)}

    pre_runtime_status = validate_phase24_runtime_budget(run_dir)
    write_text(phase24_dir / "phase24_runtime_duration_precheck.json", json.dumps(pre_runtime_status, ensure_ascii=False, indent=2))

    smoke_script = "\n".join(
        [
            "import json",
            "import yaml",
            "from validation_cases import load_canonical_case",
            "import generated_plugin as plugin",
            "problem = load_canonical_case()",
            "model = plugin.build_model(problem, seed=0)",
            "if getattr(problem, '_model_cache', None) is None:",
            "    setattr(problem, '_model_cache', model)",
            "runtime_model = getattr(problem, '_model_cache', None) or model",
            "call_model = runtime_model",
            "if isinstance(model, dict) and isinstance(runtime_model, dict):",
            "    call_model = dict(runtime_model)",
            "    for key in ('state_init', 'operators', 'metadata'):",
            "        if key in model and key not in call_model:",
            "            call_model[key] = model[key]",
            "state = plugin.initial_state(problem, call_model, seed=0)",
            "assert isinstance(model, dict), f'build_model must return dict, got {type(model).__name__}'",
            "assert all(k in model for k in ['state_init', 'operators', 'metadata']), 'build_model must return keys state_init, operators, and metadata'",
            "proj = model['operators']['project_state'](problem, state)",
            "resp = model['operators']['channel_from_state'](problem, state)",
            "metrics = model['operators']['evaluate_state'](problem, call_model, state)",
            "baseline = plugin.baseline_solution(problem, call_model, seed=0)",
            "step = plugin.proposed_step(problem, call_model, state, 0)",
            "assert isinstance(proj, dict), f'project_state must return dict, got {type(proj).__name__}'",
            "assert isinstance(resp, dict), f'channel_from_state must return dict, got {type(resp).__name__}'",
            "assert isinstance(metrics, dict), f'evaluate_state must return dict, got {type(metrics).__name__}'",
            "assert isinstance(baseline, dict), f'baseline_solution must return dict, got {type(baseline).__name__}'",
            "assert isinstance(step, dict), f'proposed_step must return dict, got {type(step).__name__}'",
            "plan = yaml.safe_load(open('../validation_plan.yaml', encoding='utf-8')) or {}",
            "evidence = plan.get('research_evidence_contract', {}) if isinstance(plan, dict) else {}",
            "if not isinstance(evidence, dict) or not evidence:",
            "    evidence = plan.get('paper_evidence_contract', {}) if isinstance(plan, dict) else {}",
            "declared = []",
            "for figure in evidence.get('figures', []) if isinstance(evidence, dict) else []:",
            "    if not isinstance(figure, dict):",
            "        continue",
            "    for item in figure.get('methods_to_run', []) if isinstance(figure.get('methods_to_run', []), list) else []:",
            "        method = item.get('id') or item.get('internal_name') or item.get('name') if isinstance(item, dict) else item",
            "        if method and str(method) not in ['proposed', 'baseline'] and str(method) not in declared:",
            "            declared.append(str(method))",
            "if not declared:",
            "    for item in evidence.get('compared_methods', []) if isinstance(evidence, dict) else []:",
            "        method = item.get('id') or item.get('internal_name') or item.get('name') if isinstance(item, dict) else item",
            "        if method and str(method) not in ['proposed', 'baseline'] and str(method) not in declared:",
            "            declared.append(str(method))",
            "if declared:",
            "    assert hasattr(plugin, 'method_solution')",
            "    for method in declared[:3]:",
            "        extra = plugin.method_solution(problem, call_model, method, seed=0)",
            "        assert isinstance(extra, dict)",
            "        extra_metrics = plugin.evaluate_state(problem, call_model, extra)",
            "        assert isinstance(extra_metrics, dict)",
            "print(json.dumps({'status': 'ok'}))",
        ]
    )
    smoke_timeout = _phase24_timeout_seconds("WARA_PHASE24_SMOKE_TIMEOUT_SEC", 120)
    try:
        smoke_result = subprocess.run(
            [sys.executable, "-c", smoke_script],
            cwd=solver_dir,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=smoke_timeout if smoke_timeout > 0 else None,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = _phase24_timeout_stream(exc.stdout)
        stderr = _phase24_timeout_stream(exc.stderr)
        error_path = phase24_dir / "phase24_validation_error.txt"
        write_text(phase24_dir / "phase24_smoke_stdout.txt", stdout)
        write_text(phase24_dir / "phase24_smoke_stderr.txt", stderr)
        write_text(error_path, f"[phase24_smoke_timeout]\nExceeded {smoke_timeout} seconds.\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}")
        return {"status": "smoke_timeout", "returncode": 124, "error_path": str(error_path)}
    write_text(phase24_dir / "phase24_smoke_stdout.txt", smoke_result.stdout)
    write_text(phase24_dir / "phase24_smoke_stderr.txt", smoke_result.stderr)
    if smoke_result.returncode != 0:
        error_path = phase24_dir / "phase24_validation_error.txt"
        write_text(error_path, f"[phase24_smoke]\nSTDOUT:\n{smoke_result.stdout}\nSTDERR:\n{smoke_result.stderr}")
        return {"status": "smoke_failed", "returncode": smoke_result.returncode, "error_path": str(error_path)}
    warning_status = _phase24_numerical_runtime_warning_report({"phase24_smoke_stderr": smoke_result.stderr})
    write_text(phase24_dir / "phase24_numerical_runtime_warning_check.json", json.dumps(warning_status, ensure_ascii=False, indent=2))
    if not warning_status["ok"]:
        warning_status["smoke_warning_advisory"] = True
        warning_status["advisory_reason"] = (
            "Smoke execution emitted NumPy runtime warnings. The controller will run full validation "
            "and block only if final validation outputs are missing, failed, or non-finite."
        )
        write_text(phase24_dir / "phase24_numerical_runtime_warning_check.json", json.dumps(warning_status, ensure_ascii=False, indent=2))

    validation_timeout = _phase24_timeout_seconds("WARA_PHASE24_VALIDATION_TIMEOUT_SEC", 300)
    try:
        validation_result = subprocess.run(
            [sys.executable, "run_validation.py"],
            cwd=solver_dir,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=validation_timeout if validation_timeout > 0 else None,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = _phase24_timeout_stream(exc.stdout)
        stderr = _phase24_timeout_stream(exc.stderr)
        error_path = phase24_dir / "phase24_validation_error.txt"
        write_text(phase24_dir / "phase24_validation_stdout.txt", stdout)
        write_text(phase24_dir / "phase24_validation_stderr.txt", stderr)
        write_text(error_path, f"[run_validation_timeout]\nExceeded {validation_timeout} seconds.\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}")
        return {"status": "validation_timeout", "returncode": 124, "error_path": str(error_path)}
    write_text(phase24_dir / "phase24_validation_stdout.txt", validation_result.stdout)
    write_text(phase24_dir / "phase24_validation_stderr.txt", validation_result.stderr)
    if validation_result.returncode != 0:
        error_path = phase24_dir / "phase24_validation_error.txt"
        write_text(error_path, f"[run_validation]\nSTDOUT:\n{validation_result.stdout}\nSTDERR:\n{validation_result.stderr}")
        return {"status": "validation_failed", "returncode": validation_result.returncode, "error_path": str(error_path)}
    warning_status = _phase24_numerical_runtime_warning_report(
        {
            "phase24_smoke_stderr": smoke_result.stderr,
            "phase24_validation_stderr": validation_result.stderr,
        }
    )
    write_text(phase24_dir / "phase24_numerical_runtime_warning_check.json", json.dumps(warning_status, ensure_ascii=False, indent=2))
    if not warning_status["ok"]:
        summary_path = solver_dir / "outputs" / "validation_summary.json"
        if _phase24_validation_outputs_are_finite(summary_path):
            warning_status["ok"] = True
            warning_status["downgraded_to_advisory_after_finite_validation"] = True
            warning_status["advisory_reason"] = (
                "Runtime warnings were observed, but full validation produced finite outputs with zero failed cases."
            )
            write_text(phase24_dir / "phase24_numerical_runtime_warning_check.json", json.dumps(warning_status, ensure_ascii=False, indent=2))
        else:
            error_path = phase24_dir / "phase24_validation_error.txt"
            write_text(
                error_path,
                "[numerical_runtime_warning]\n"
                + "\n".join(warning_status["errors"])
                + "\n\nSerious NumPy runtime warnings indicate non-finite matrix/state arithmetic. "
                "Repair generated_experiment_core.py by normalizing finite physical arrays, applying safe floors/caps before matrix products, "
                "and removing NaN/Inf propagation rather than hiding it in metrics.",
            )
            return {"status": "numerical_runtime_warning_failed", "returncode": 0, "error_path": str(error_path)}

    summary_path = solver_dir / "outputs" / "validation_summary.json"
    results_path = solver_dir / "outputs" / "validation_results.csv"
    if not summary_path.exists() or not results_path.exists():
        missing = []
        if not summary_path.exists():
            missing.append(str(summary_path))
        if not results_path.exists():
            missing.append(str(results_path))
        error_path = phase24_dir / "phase24_validation_error.txt"
        write_text(error_path, "[run_validation]\nValidation completed with exit code 0 but missing expected output files:\n" + "\n".join(missing))
        return {"status": "missing_outputs", "returncode": 0, "error_path": str(error_path)}
    evidence_status = validate_phase24_evidence_contract_outputs(run_dir)
    write_text(phase24_dir / "phase24_evidence_contract_check.json", json.dumps(evidence_status, ensure_ascii=False, indent=2))
    if not evidence_status["ok"]:
        error_path = phase24_dir / "phase24_validation_error.txt"
        write_text(error_path, "[research_evidence_contract]\n" + "\n".join(evidence_status["errors"]))
        return {"status": "evidence_contract_failed", "returncode": 0, "error_path": str(error_path)}
    quality_status = validate_phase24_basic_evidence_quality(run_dir)
    write_text(phase24_dir / "phase24_basic_evidence_quality_check.json", json.dumps(quality_status, ensure_ascii=False, indent=2))
    if not quality_status["ok"]:
        error_path = phase24_dir / "phase24_validation_error.txt"
        write_text(error_path, "[basic_evidence_quality]\n" + "\n".join(quality_status["errors"]))
        return {"status": "basic_evidence_quality_failed", "returncode": 0, "error_path": str(error_path)}
    method_semantics_status = validate_phase24_method_semantics(run_dir)
    write_text(phase24_dir / "phase24_method_semantics_check.json", json.dumps(method_semantics_status, ensure_ascii=False, indent=2))
    if not method_semantics_status["ok"]:
        error_path = phase24_dir / "phase24_validation_error.txt"
        write_text(error_path, "[method_semantics]\n" + "\n".join(method_semantics_status["errors"]))
        return {"status": "method_semantics_failed", "returncode": 0, "error_path": str(error_path)}
    responsiveness_status = validate_phase24_experiment_responsiveness(run_dir)
    write_text(phase24_dir / "phase24_experiment_responsiveness_check.json", json.dumps(responsiveness_status, ensure_ascii=False, indent=2))
    if not responsiveness_status["ok"]:
        error_path = phase24_dir / "phase24_validation_error.txt"
        write_text(error_path, "[experiment_responsiveness]\n" + "\n".join(responsiveness_status["errors"]))
        return {
            "status": "experiment_responsiveness_failed",
            "returncode": 0,
            "error_path": str(error_path),
            "design_repair_recommended": bool(responsiveness_status.get("design_repair_recommended", False)),
        }
    pilot_gain_status = validate_phase24_pilot_gain(run_dir)
    write_text(phase24_dir / "phase24_pilot_gain_check.json", json.dumps(pilot_gain_status, ensure_ascii=False, indent=2))
    if not pilot_gain_status["ok"]:
        failure_diagnostics = _phase24_write_failure_diagnostics(run_dir)
        error_path = phase24_dir / "phase24_validation_error.txt"
        write_text(
            error_path,
            "[pilot_gain]\n"
            + "\n".join(pilot_gain_status["errors"])
            + "\n\n"
            + "\n".join(pilot_gain_status.get("repair_advice", []))
            + "\n\n[method_failure_diagnostics]\n"
            + json.dumps(failure_diagnostics, ensure_ascii=False, indent=2),
        )
        return {"status": "pilot_gain_failed", "returncode": 0, "error_path": str(error_path)}
    runtime_status = validate_phase24_runtime_budget(run_dir)
    write_text(phase24_dir / "phase24_runtime_duration_check.json", json.dumps(runtime_status, ensure_ascii=False, indent=2))
    validation_status = {
        "status": "ok",
        "returncode": 0,
        "codegen_version": PHASE24_SPLIT_ADAPTER_VERSION,
        "generated_experiment_core_sha256": codegen_status.get("generated_experiment_core_sha256"),
    }
    write_text(phase24_dir / "phase24_validation_manifest.json", json.dumps(validation_status, ensure_ascii=False, indent=2))
    for stale_error_name in [
        "phase24_validation_error.txt",
        "phase24_interface_errors.txt",
        "implementation_audit_blocking_errors.txt",
        "phase24_selected_candidate.json",
        "phase24_selected_candidate_status.json",
        "phase24_selected_candidate_note.md",
    ]:
        stale_error_path = phase24_dir / stale_error_name
        if stale_error_path.exists():
            stale_error_path.unlink()
    return validation_status


def _phase24_validation_error_text(run_dir: Path, validation_status: dict[str, Any]) -> str:
    error_path_raw = validation_status.get("error_path")
    error_path = Path(str(error_path_raw)) if error_path_raw else run_dir / "phase2-4" / "phase24_validation_error.txt"
    sections: list[str] = []
    if error_path.exists():
        text = read_text(error_path).strip()
        if text:
            sections.append("[primary validation error]\n" + text)
    phase24_dir = Path(run_dir) / "phase2-4"
    for name in [
        "phase24_evidence_contract_check.json",
        "phase24_basic_evidence_quality_check.json",
        "phase24_method_semantics_check.json",
        "phase24_experiment_responsiveness_check.json",
        "phase24_pilot_gain_check.json",
        "phase24_validation_failure_diagnostics.json",
        "phase24_algorithm_code_contract_check.json",
        "phase24_numerical_runtime_warning_check.json",
    ]:
        report = read_json(phase24_dir / name) or {}
        if not isinstance(report, dict):
            continue
        errors = [str(item) for item in report.get("errors", []) if str(item)]
        blocking = [str(item) for item in report.get("blocking_errors", []) if str(item)]
        advice = [str(item) for item in report.get("repair_advice", []) if str(item)]
        if errors or blocking or advice:
            sections.append(
                f"[{name}]\n"
                + json.dumps(
                    {
                        "errors": errors,
                        "blocking_errors": blocking,
                        "repair_advice": advice,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
    trace_text = read_text(phase24_dir / "phase24_solver_exception_trace.txt").strip()
    if trace_text:
        sections.append("[phase24_solver_exception_trace]\n" + trace_text[:6000])
    return "\n\n".join(sections)


def _phase24_validation_allows_repair(validation_status: dict[str, Any]) -> bool:
    status = str(validation_status.get("status") or "")
    if status == "experiment_responsiveness_failed" and bool(validation_status.get("design_repair_recommended")):
        return False
    return status in {
        "codegen_package_failed",
        "interface_failed",
        "schema_alignment_failed",
        "algorithm_code_contract_failed",
        "compile_failed",
        "smoke_failed",
        "validation_timeout",
        "validation_failed",
        "missing_outputs",
        "evidence_contract_design_failed",
        "evidence_contract_failed",
        "basic_evidence_quality_failed",
        "method_semantics_failed",
        "experiment_responsiveness_failed",
        "pilot_gain_failed",
        "numerical_runtime_warning_failed",
    }
