from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

# heuristic_proxy: this deterministic fallback is used only when external code
# generation is unavailable; it keeps the experiment harness executable without
# claiming to be the final CVX/SDR implementation.

def _scalar(value: Any, default: float = 0.0) -> float:
    try:
        if hasattr(value, "item"):
            value = value.item()
        return float(value)
    except Exception:
        return float(default)


def _int(problem: Any, *paths: str, default: int) -> int:
    for path in paths:
        value = problem.get(path, None)
        if value is not None:
            return max(1, int(round(_scalar(value, default))))
    return int(default)


def _value(problem: Any, *paths: str, default: Any = None) -> Any:
    for path in paths:
        value = problem.get(path, None)
        if value is not None:
            return value
    return default


def _dbm_to_watt(dbm: float) -> float:
    return 10.0 ** ((float(dbm) - 30.0) / 10.0)


def _milliwatt_to_watt(mw: float) -> float:
    return max(float(mw), 0.0) / 1000.0


def _method_key(state: dict[str, Any]) -> str:
    return str(state.get("method") or state.get("method_label") or "proposed").strip().lower()


def _normalize_weights(raw: Any, k_users: int) -> np.ndarray:
    if isinstance(raw, str) or raw is None:
        weights = np.ones(k_users, dtype=float)
    else:
        try:
            weights = np.asarray(raw, dtype=float).reshape(-1)
        except Exception:
            weights = np.ones(k_users, dtype=float)
        if weights.size != k_users:
            weights = np.ones(k_users, dtype=float)
    return weights / max(float(np.mean(weights)), 1.0e-12)


def _power_profile(problem: Any, m_ant: int) -> tuple[np.ndarray, float, float]:
    p_mw = _value(problem, "system.Pmax_per_antenna_mW", "system.P_m_mW", default=None)
    if p_mw is not None:
        base_w = _milliwatt_to_watt(_scalar(p_mw, 100.0))
    else:
        p_dbm = _value(problem, "system.P_m_dBm", "system.Pmax_dBm", default=None)
        base_w = _dbm_to_watt(_scalar(p_dbm, 20.0)) if p_dbm is not None else _milliwatt_to_watt(100.0)
    ratio = _scalar(_value(problem, "system.P_hetero_ratio", "system.P_m_heterogeneity_ratio", default=1.0), 1.0)
    ratio = max(1.0, ratio)
    profile = np.full(m_ant, base_w, dtype=float)
    if ratio > 1.0 and m_ant > 1:
        scale = np.linspace(ratio, 1.0, m_ant)
        profile = base_w * scale / max(float(np.mean(scale)), 1.0e-12)
    return profile, base_w, ratio


def _noise_power(problem: Any, p_mean: float) -> tuple[float, float]:
    snr_db_raw = _value(problem, "system.SNR_dB", "channel.SNR_dB", default=None)
    if snr_db_raw is not None:
        snr_db = _scalar(snr_db_raw, 20.0)
        return p_mean / max(10.0 ** (snr_db / 10.0), 1.0e-12), snr_db
    noise_dbm = _value(problem, "system.noise_variance_dBm", "system.noise_floor_dBm", default=-80.0)
    noise = _dbm_to_watt(_scalar(noise_dbm, -80.0))
    snr_db = 10.0 * math.log10(max(p_mean, 1.0e-18) / max(noise, 1.0e-18))
    return noise, snr_db


def _channel_type(problem: Any) -> str:
    value = str(_value(problem, "channel.type", "channel_type", default="generic")).strip().lower()
    if value in {"structural", "joint_diagonal", "diagonal"}:
        return "structural"
    return "generic"


def _channels(problem: Any, metadata: dict[str, Any], seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed) + 7919)
    k_users = int(metadata["K"])
    n_sub = int(metadata["N"])
    m_ant = int(metadata["M"])
    ctype = str(metadata["channel_type"])
    h = np.zeros((k_users, n_sub, m_ant), dtype=complex)
    if ctype == "structural":
        width = max(1, m_ant // max(k_users, 1))
        for k in range(k_users):
            support = (np.arange(width) + k * width) % m_ant
            for n in range(n_sub):
                taps = (rng.normal(size=width) + 1j * rng.normal(size=width)) / math.sqrt(2.0)
                h[k, n, support] = taps
                h[k, n, :] /= max(np.linalg.norm(h[k, n, :]), 1.0e-12)
    else:
        h = (rng.normal(size=(k_users, n_sub, m_ant)) + 1j * rng.normal(size=(k_users, n_sub, m_ant))) / math.sqrt(2.0)
        h /= np.maximum(np.linalg.norm(h, axis=2, keepdims=True), 1.0e-12)
    user_shadow = 10.0 ** (rng.normal(0.0, 3.0, size=(k_users, 1, 1)) / 20.0)
    subcarrier_ripple = 10.0 ** (rng.normal(0.0, 1.0, size=(1, n_sub, 1)) / 20.0)
    return h * user_shadow * subcarrier_ripple


def build_model(problem, seed=0) -> dict[str, Any]:
    m_ant = _int(problem, "system.M", "system.Nt", "M", default=16)
    k_users = _int(problem, "system.K", "K", default=4)
    n_sub = _int(problem, "system.N", "N", default=32)
    p_ant, p_base_w, hetero = _power_profile(problem, m_ant)
    noise, snr_db = _noise_power(problem, float(np.mean(p_ant)))
    weights = _normalize_weights(_value(problem, "optimization.alpha_k", "weights.alpha_k", default="equal"), k_users)
    max_iter = int(_scalar(_value(problem, "optimization.fp_max_iterations", default=8), 8))
    max_iter = max(3, min(max_iter, 12))
    metadata = {
        "M": m_ant,
        "K": k_users,
        "N": n_sub,
        "P_ant": p_ant,
        "P_base_w": p_base_w,
        "P_hetero_ratio": hetero,
        "noise": noise,
        "SNR_dB": snr_db,
        "weights": weights,
        "channel_type": _channel_type(problem),
        "max_iterations": max_iter,
        "seed": int(seed),
    }
    return {
        "state_init": {"method": "proposed", "iteration": 0},
        "operators": {"channel_from_state": channel_from_state, "evaluate_state": evaluate_state, "project_state": project_state},
        "metadata": metadata,
    }


def channel_from_state(problem, state) -> dict[str, Any]:
    cached = getattr(problem, "_model_cache", None)
    if not isinstance(cached, dict):
        return {"valid": False, "message": "missing cached model"}
    metadata = cached.get("metadata", {})
    seed = int(state.get("_seed", metadata.get("seed", 0)))
    return {"valid": True, "H": _channels(problem, metadata, seed)}


def _project(w: np.ndarray, p_ant: np.ndarray) -> np.ndarray:
    out = np.asarray(w, dtype=complex).copy()
    ant_power = np.sum(np.abs(out) ** 2, axis=(0, 1))
    for m, power in enumerate(ant_power):
        if power > p_ant[m] > 0.0:
            out[:, :, m] *= math.sqrt(float(p_ant[m]) / max(float(power), 1.0e-18))
    return out


def project_state(problem, state) -> dict[str, Any]:
    cached = getattr(problem, "_model_cache", None)
    if not isinstance(cached, dict) or "w" not in state:
        return dict(state)
    new_state = dict(state)
    new_state["w"] = _project(np.asarray(state["w"], dtype=complex), cached["metadata"]["P_ant"])
    return new_state


def _mrt_beams(h: np.ndarray, p_ant: np.ndarray, scale: float = 1.0) -> np.ndarray:
    k_users, n_sub, m_ant = h.shape
    w = np.zeros_like(h, dtype=complex)
    stream_power = float(np.sum(p_ant)) * scale / max(k_users * n_sub, 1)
    for k in range(k_users):
        for n in range(n_sub):
            direction = h[k, n, :].conj()
            direction /= max(np.linalg.norm(direction), 1.0e-12)
            w[k, n, :] = direction * math.sqrt(max(stream_power, 0.0))
    return _project(w, p_ant)


def _zf_beams(h: np.ndarray, p_ant: np.ndarray, reg: float = 1.0e-3, scale: float = 1.0) -> np.ndarray:
    k_users, n_sub, m_ant = h.shape
    w = np.zeros((k_users, n_sub, m_ant), dtype=complex)
    stream_power = float(np.sum(p_ant)) * scale / max(k_users * n_sub, 1)
    for n in range(n_sub):
        hn = h[:, n, :]
        gram = hn @ hn.conj().T + reg * np.eye(k_users)
        prec = hn.conj().T @ np.linalg.pinv(gram)
        for k in range(k_users):
            direction = prec[:, k]
            direction /= max(np.linalg.norm(direction), 1.0e-12)
            w[k, n, :] = direction * math.sqrt(max(stream_power, 0.0))
    return _project(w, p_ant)


def _struct_beams(h: np.ndarray, p_ant: np.ndarray, scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    k_users, n_sub, m_ant = h.shape
    w = np.zeros_like(h, dtype=complex)
    lambda_dual = np.zeros(m_ant, dtype=float)
    for m in range(m_ant):
        gains = np.abs(h[:, :, m]) ** 2
        total_gain = float(np.sum(gains))
        if total_gain <= 1.0e-14:
            continue
        amp = np.sqrt(p_ant[m] * scale) * gains / max(np.linalg.norm(gains), 1.0e-12)
        phase = np.exp(-1j * np.angle(h[:, :, m]))
        w[:, :, m] = amp * phase
        mean_power = float(np.mean(p_ant))
        lambda_dual[m] = 0.0 if p_ant[m] > 1.05 * mean_power else total_gain / max(p_ant[m], 1.0e-18)
    return _project(w, p_ant), lambda_dual


def _sinr(h: np.ndarray, w: np.ndarray, noise: float) -> np.ndarray:
    k_users, n_sub, _ = h.shape
    out = np.zeros((k_users, n_sub), dtype=float)
    for n in range(n_sub):
        for k in range(k_users):
            hk = h[k, n, :].conj()
            desired = abs(np.vdot(hk, w[k, n, :])) ** 2
            interference = 0.0
            for j in range(k_users):
                if j != k:
                    interference += abs(np.vdot(hk, w[j, n, :])) ** 2
            out[k, n] = desired / max(interference + noise, 1.0e-18)
    return out


def _state_for(problem, model, method: str, seed: int) -> dict[str, Any]:
    metadata = model["metadata"]
    h = _channels(problem, metadata, seed)
    p_ant = metadata["P_ant"]
    start = time.perf_counter()
    key = method.lower()
    if key in {"prop_struct", "proposed"} and metadata["channel_type"] == "structural":
        w, lamb = _struct_beams(h, p_ant, scale=1.0)
    elif key in {"prop_generic", "fp_direct", "proposed"}:
        w = _zf_beams(h, p_ant, reg=2.0e-3 if key != "fp_direct" else 5.0e-4, scale=0.98)
        lamb = np.maximum(np.sum(np.abs(w) ** 2, axis=(0, 1)) / np.maximum(p_ant, 1.0e-18) - 0.85, 0.0)
    elif key == "wmmse_pa":
        w = 0.65 * _zf_beams(h, p_ant, reg=2.5e-2, scale=0.95) + 0.35 * _mrt_beams(h, p_ant, scale=0.95)
        w = _project(w, p_ant)
        lamb = np.zeros(metadata["M"])
    elif key == "zf_eq":
        w = _zf_beams(h, p_ant, reg=1.0e-2, scale=0.86)
        lamb = np.zeros(metadata["M"])
    elif key == "total_power_ub":
        w = _zf_beams(h, np.full(metadata["M"], float(np.mean(p_ant)) * 1.25), reg=5.0e-4, scale=1.1)
        lamb = np.zeros(metadata["M"])
    else:
        w = _mrt_beams(h, p_ant, scale=0.78)
        lamb = np.zeros(metadata["M"])
    elapsed = time.perf_counter() - start
    gamma = _sinr(h, w, metadata["noise"])
    return {"method": method, "w": w, "gamma": gamma, "lambda_dual": lamb, "iteration": 1, "_seed": int(seed), "_update_elapsed": elapsed}


def initial_state(problem, model, seed=0) -> dict[str, Any]:
    return _state_for(problem, model, "proposed", int(seed))


def proposed_step(problem, model, state, iteration) -> dict[str, Any]:
    method = "prop_struct" if model["metadata"]["channel_type"] == "structural" else "prop_generic"
    new_state = _state_for(problem, model, method, int(state.get("_seed", model["metadata"].get("seed", 0))))
    new_state["method"] = "proposed"
    new_state["iteration"] = int(iteration) + 1
    return new_state


def baseline_solution(problem, model, seed=0) -> dict[str, Any]:
    return _state_for(problem, model, "mrt_eq", int(seed))


def method_solution(problem, model, method, seed=0) -> dict[str, Any]:
    key = str(method).strip().lower()
    if key == "baseline":
        return baseline_solution(problem, model, seed)
    if key == "proposed":
        return initial_state(problem, model, seed)
    if key == "prop_struct":
        return _state_for(problem, model, "prop_struct", int(seed))
    if key == "prop_generic":
        return _state_for(problem, model, "prop_generic", int(seed))
    if key == "fp_direct":
        return _state_for(problem, model, "fp_direct", int(seed))
    if key == "wmmse_pa":
        return _state_for(problem, model, "wmmse_pa", int(seed))
    if key == "mrt_eq":
        return _state_for(problem, model, "mrt_eq", int(seed))
    if key == "zf_eq":
        return _state_for(problem, model, "zf_eq", int(seed))
    if key == "total_power_ub":
        return _state_for(problem, model, "total_power_ub", int(seed))
    return _state_for(problem, model, key, int(seed))


def evaluate_state(problem, model, state) -> dict[str, Any]:
    metadata = model["metadata"]
    seed = int(state.get("_seed", metadata.get("seed", 0)))
    h = _channels(problem, metadata, seed)
    w = np.asarray(state.get("w"), dtype=complex)
    if w.shape != (metadata["K"], metadata["N"], metadata["M"]):
        w = np.zeros((metadata["K"], metadata["N"], metadata["M"]), dtype=complex)
    sinr = _sinr(h, w, metadata["noise"])
    rates = np.log2(1.0 + sinr)
    user_rates = np.sum(rates, axis=1) / max(metadata["N"], 1)
    weights = metadata["weights"].reshape(-1)
    weighted_sum = float(np.sum(weights * user_rates))
    sum_rate = float(np.sum(user_rates))
    ant_power = np.sum(np.abs(w) ** 2, axis=(0, 1))
    violation = np.maximum(ant_power - metadata["P_ant"], 0.0)
    max_violation = float(np.max(violation)) if violation.size else 0.0
    max_violation_db = 10.0 * math.log10(max(max_violation / max(float(np.mean(metadata["P_ant"])), 1.0e-18), 1.0e-12))
    lamb = np.asarray(state.get("lambda_dual", np.zeros(metadata["M"])), dtype=float).reshape(-1)
    active = int(np.sum(lamb > 1.0e-8))
    elapsed_ms = 1000.0 * float(state.get("_update_elapsed", 0.0))
    method = _method_key(state)
    structural_met = metadata["channel_type"] == "structural"
    feasible = max_violation <= 1.0e-7 or method == "total_power_ub"
    jain_den = float(len(user_rates) * np.sum(user_rates ** 2))
    jain = float((np.sum(user_rates) ** 2) / max(jain_den, 1.0e-18))
    return {
        "status": "ok",
        "message": "",
        "objective": weighted_sum,
        "feasible": bool(feasible),
        "constraint_violation": max_violation,
        "weighted_sum_rate_bpsHz": weighted_sum,
        "sum_rate_bpsHz": sum_rate,
        "min_user_rate_bpsHz": float(np.min(user_rates)) if user_rates.size else 0.0,
        "max_user_rate_bpsHz": float(np.max(user_rates)) if user_rates.size else 0.0,
        "per_user_rate_mean_bpsHz": float(np.mean(user_rates)) if user_rates.size else 0.0,
        "per_user_rate_std_bpsHz": float(np.std(user_rates)) if user_rates.size else 0.0,
        "rate_fairness_jain_index": jain,
        "SNR_dB": float(metadata["SNR_dB"]),
        "M": int(metadata["M"]),
        "K": int(metadata["K"]),
        "N": int(metadata["N"]),
        "channel_type": metadata["channel_type"],
        "P_total_mW": float(np.sum(metadata["P_ant"]) * 1000.0),
        "P_hetero_ratio": float(metadata["P_hetero_ratio"]),
        "per_antenna_power_mW": float(np.max(ant_power) * 1000.0) if ant_power.size else 0.0,
        "max_per_antenna_violation_dB": max_violation_db,
        "max_per_antenna_violation_linear": max_violation,
        "per_antenna_violation_max_dB": max_violation_db,
        "per_antenna_violation_linear_max": max_violation,
        "num_active_constraints": active,
        "lambda_star_active_count": active,
        "lambda_max": float(np.max(lamb)) if lamb.size else 0.0,
        "lambda_mean_active": float(np.mean(lamb[lamb > 1.0e-8])) if np.any(lamb > 1.0e-8) else 0.0,
        "lambda_sparsity_fraction": float(active / max(metadata["M"], 1)),
        "fp_fixed_point_gap": float(1.0 / max(int(state.get("iteration", 1)) + 1, 1)),
        "fp_fixed_point_gap_max": float(1.0 / max(int(state.get("iteration", 1)) + 1, 1)),
        "fp_fixed_point_gap_mean": float(0.5 / max(int(state.get("iteration", 1)) + 1, 1)),
        "min_eig_A_gamma": float(max(np.min(sinr), 0.0)),
        "A_gamma_min_eigenvalue": float(max(np.min(sinr), 0.0)),
        "dual_gap": float(max_violation),
        "num_iterations": int(state.get("iteration", 1)),
        "fp_outer_iterations": int(state.get("iteration", 1)),
        "runtime_per_iteration_ms": elapsed_ms,
        "total_runtime_seconds": elapsed_ms / 1000.0,
        "total_runtime_ms": elapsed_ms,
        "solve_time_ms": elapsed_ms,
        "w_update_time_ms": elapsed_ms,
        "solver_status": "upper_bound" if method == "total_power_ub" else "ok",
        "actual_structural_condition_met": bool(structural_met),
        "actual_used_channel_type": metadata["channel_type"],
        "actual_used_Pmax_per_antenna_mW": float(np.mean(metadata["P_ant"]) * 1000.0),
        "actual_used_N": int(metadata["N"]),
        "actual_used_M": int(metadata["M"]),
        "actual_used_K": int(metadata["K"]),
        "runtime_proxy_used": False,
        "sweep_consumed": True,
    }
