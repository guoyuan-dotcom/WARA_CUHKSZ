from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import math
import multiprocessing
import os
import re
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


SUCCESS_STATUSES = {"ok", "success", "converged", "feasible", "optimal", "optimal_inaccurate", "solved"}
FAIL_STATUSES = {"failed", "infeasible", "error", "exception", "nan", "timeout"}

QUICK_MODE_SEEDS = 20
SCOUT_MODE_SEEDS = 20
SCOUT_LINE_POINTS = 4
SCOUT_CATEGORY_POINTS = 3
MEDIUM_MODE_SEEDS = 50
MEDIUM_LINE_POINTS = 8
MEDIUM_CATEGORY_POINTS = 4
DRAFT_MODE_SEEDS_MIN = 20
DRAFT_MODE_SEEDS_MAX = 30
PAPER_MINIMUM_SEEDS = 80
PAPER_PREFERRED_SEEDS = 100
HIGH_CONFIDENCE_SEEDS = 150
MIN_PAPER_READY_FIGURES = 2
MIN_PREVIEW_PAIRED_SUCCESS_SEEDS = 3
MIN_PREVIEW_PAIRED_SUCCESS_RATE = 0.75
LINE_PAPER_MIN_X_POINTS = 10
LINE_PAPER_PREFERRED_X_POINTS = 14
LINE_HIGH_CONF_X_POINTS = 16
DISCRETE_INTEGER_MIN_X_POINTS = 8
BAR_PAPER_MIN_CATEGORIES = 4
BAR_PAPER_PREFERRED_CATEGORIES = 6
BOX_PAPER_MIN_SAMPLES = 50
BOX_PAPER_PREFERRED_SAMPLES = 100
CONVERGENCE_MIN_ITERATIONS = 20
SEED_ALIASES = ["seed", "realization_id", "trial_id", "mc_seed", "sample_id"]
NON_METRIC_COLUMNS = {
    "case_id",
    "case_name",
    "scenario_name",
    "swept_param",
    "method",
    "status",
    "message",
    "rejection_reason",
}
TABLE_METADATA_COLUMNS = {
    "seed",
    "swept_value",
    "iterations",
    "iter_count",
    "solve_time_sec",
}
FIGURE_COLORS = ["#0b3c5d", "#c45a2a", "#2d6a4f", "#7b2f5c", "#4d4d4d", "#9a3412"]
FIGURE_MARKERS = ["o", "s", "^", "D", "v", "P"]
IEEE_SINGLE_COLUMN_WIDTH_IN = 3.5
IEEE_COMPACT_HEIGHT_IN = 2.35
IEEE_TALL_HEIGHT_IN = 2.65
OBJECTIVE_METRIC_NAMES = {"objective", "objective_value", "weighted_objective", "utility", "weighted_utility"}
OBJECTIVE_ONLY_ALLOWED_INTENTS = {"main_comparison", "overall_utility", "utility_comparison"}
MECHANISM_INTENTS = {
    "mechanism_ablation",
    "sensitivity",
    "robustness",
    "feasibility_boundary",
    "scalability",
    "convergence",
    "structural_tradeoff",
}


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def _style_axis(ax: Any) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.65)
    ax.spines["bottom"].set_linewidth(0.65)
    ax.tick_params(axis="both", labelsize=7, width=0.65, length=2.5)
    ax.margins(x=0.04)


def _finish_ieee_figure(fig: Any, ax: Any, *, legend: bool = True, ncol: int = 1) -> None:
    if legend:
        leg = ax.legend(
            frameon=True,
            fancybox=False,
            edgecolor="#b8b8b8",
            facecolor="white",
            framealpha=0.92,
            fontsize=7,
            loc="best",
            ncol=ncol,
            handlelength=1.35,
            handletextpad=0.45,
            borderpad=0.25,
            columnspacing=0.85,
        )
        leg.get_frame().set_linewidth(0.45)
    fig.tight_layout(pad=0.35)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (int,)):
        return value
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        if math.isfinite(numeric):
            return numeric
        return None
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _safe_sorted_json_values(values: list[Any]) -> list[Any]:
    cleaned: list[Any] = []
    for value in values:
        safe_value = _json_safe(value)
        if safe_value is None:
            continue
        cleaned.append(safe_value)
    unique_values = list({repr(value): value for value in cleaned}.values())

    def _sort_key(value: Any) -> tuple[int, float | str]:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return (0, float(value))
        return (1, str(value))

    return sorted(unique_values, key=_sort_key)


def _integer_grid_values(values: list[Any]) -> list[int] | None:
    integers: list[int] = []
    for value in values:
        if isinstance(value, bool):
            return None
        try:
            numeric = float(value)
        except Exception:
            return None
        if not math.isfinite(numeric):
            return None
        rounded = round(numeric)
        if abs(numeric - rounded) > 1.0e-8:
            return None
        integers.append(int(rounded))
    return sorted(set(integers))


def _is_contiguous_integer_grid(values: list[int]) -> bool:
    if not values:
        return False
    return values == list(range(values[0], values[-1] + 1))


def _is_count_like_sweep_param(param: str, figure_spec: dict[str, Any] | None = None) -> bool:
    param_tokens = [token for token in re.split(r"[^a-z0-9]+", str(param or "").lower()) if token]
    text_parts = [str(param or "")]
    if figure_spec:
        encoding = figure_spec.get("encoding", {}) if isinstance(figure_spec.get("encoding"), dict) else {}
        x_encoding = encoding.get("x", {}) if isinstance(encoding.get("x"), dict) else {}
        text_parts.extend(
            str(value or "")
            for value in (
                x_encoding.get("display_name", ""),
                figure_spec.get("purpose", ""),
                figure_spec.get("primary_message", ""),
                figure_spec.get("chart_intent", ""),
            )
        )
    text = " ".join(text_parts).lower()
    tokens = [token for token in re.split(r"[^a-z0-9]+", text) if token]
    if not tokens:
        return False
    # Single-letter wireless symbols such as M/N/K can be counts, but the same
    # tokens also appear as physical units (e.g., side_length_m, spacing_m).
    # Treat explicit geometry/unit parameters as continuous before applying
    # count heuristics, otherwise Phase 2.5 collapses smooth meter sweeps to
    # integer grids such as [1, 2].
    geometry_or_unit_tokens = {
        "aperture",
        "length",
        "spacing",
        "distance",
        "radius",
        "position",
        "coordinate",
        "location",
        "trajectory",
        "altitude",
        "height",
        "width",
        "depth",
        "meter",
        "meters",
    }
    physical_meter_context = geometry_or_unit_tokens | {"d", "dmin", "min", "max"}
    if any(token in tokens for token in physical_meter_context) and any(token in tokens for token in {"m", "meter", "meters"}):
        return False
    last_token = tokens[-1]
    compact = "_".join(tokens)
    count_tokens = {
        "k",
        "m",
        "n",
        "nt",
        "nr",
        "me",
        "mr",
        "num",
        "number",
        "count",
        "users",
        "user",
        "antennas",
        "antenna",
        "elements",
        "element",
        "devices",
        "device",
        "nodes",
        "node",
        "aps",
        "ap",
        "stations",
        "station",
        "rrhs",
        "rrh",
        "streams",
        "stream",
        "clusters",
        "cluster",
        "subcarriers",
        "subcarrier",
        "slots",
        "slot",
        "links",
        "link",
        "vehicles",
        "vehicle",
        "uavs",
        "uav",
    }
    if param_tokens:
        if param_tokens[-1] in count_tokens:
            return True
        if any(token in param_tokens for token in {"num", "number", "count"}):
            return True
    if last_token in count_tokens:
        return True
    if any(token in tokens for token in {"num", "number", "count"}):
        return True
    non_letter_count_tokens = {token for token in count_tokens if len(token) > 1}
    if any(token in tokens for token in non_letter_count_tokens):
        return True
    return any(fragment in compact for fragment in ("num_", "_num_", "number_", "_number_", "count_", "_count_"))


def _completed_discrete_integer_grid_reason(
    *,
    figure_spec: dict[str, Any],
    sweep_param: str,
    x_values: list[Any],
    all_x_values: list[Any],
    min_points: int,
) -> str:
    if len(x_values) >= min_points or len(x_values) < DISCRETE_INTEGER_MIN_X_POINTS:
        return ""
    if not _is_count_like_sweep_param(sweep_param, figure_spec):
        return ""
    integer_x_values = _integer_grid_values(x_values)
    if integer_x_values is None or len(integer_x_values) != len(x_values):
        return ""
    if not _is_contiguous_integer_grid(integer_x_values):
        return ""
    if all_x_values:
        integer_all_x_values = _integer_grid_values(all_x_values)
        if integer_all_x_values is None:
            return ""
        if not set(integer_all_x_values).issubset(set(integer_x_values)):
            return ""
    return "capped_to_completed_discrete_integer_grid"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _is_scalar_metric_value(value: Any) -> bool:
    if isinstance(value, (bool, int, float, str, np.bool_, np.integer, np.floating)):
        if isinstance(value, str):
            return bool(value.strip())
        return True
    return False


def _coerce_scalar_metric(value: Any) -> Any:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, str):
        text = value.strip()
        if text.lower() in {"true", "false"}:
            return text.lower() == "true"
        try:
            numeric = float(text)
        except ValueError:
            return text
        return numeric if math.isfinite(numeric) else None
    return None


def _flatten_metric_scalars(metrics: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in metrics.items():
        key_str = str(key)
        if key_str == "diagnostics":
            continue
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if _is_scalar_metric_value(sub_value):
                    flat[f"{key_str}_{sub_key}"] = _coerce_scalar_metric(sub_value)
            continue
        if _is_scalar_metric_value(value):
            flat[key_str] = _coerce_scalar_metric(value)
    return flat


def _clean_axis_math_fragment(text: str) -> str:
    cleaned = re.sub(r"\\mathcal\s+([A-Za-z])", r"\\mathcal{\1}", str(text))
    cleaned = re.sub(r"\\{2,}([A-Za-z]+)", r"\\\1", cleaned)
    cleaned = re.sub(r"\s*/\s*", "/", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _repair_unwrapped_latex_axis_label(text: str) -> str:
    r"""Render validation-plan LaTeX snippets as matplotlib mathtext.

    LLM-generated validation plans sometimes use fragments such as
    ``length of \mathcal R_m / \lambda`` without dollar delimiters. Matplotlib
    then prints the backslash commands literally. Keep prose outside math mode,
    but wrap the symbolic fragment so the final figure label is publishable.
    """
    label = str(text or "").strip()
    if "$" in label or "\\" not in label:
        return label
    label = _clean_axis_math_fragment(label)
    prose_tokens = re.compile(
        r"\b(?:length|power|rate|number|minimum|maximum|budget|weight|requirement|versus|of|the|with)\b",
        re.IGNORECASE,
    )
    if not prose_tokens.search(label) and re.fullmatch(r"[A-Za-z0-9\\{}_^/+\-.,\s]+", label):
        return f"${_clean_axis_math_fragment(label)}$"

    symbolic_expr = re.compile(
        r"(?<!\$)("
        r"[A-Za-z](?:_\{\\?[A-Za-z0-9]+\}|_[A-Za-z0-9]+)(?:/\\[A-Za-z]+)?"
        r"|\\(?:mathcal\{[A-Za-z]\}|[A-Za-z]+)(?:_\{?[A-Za-z0-9]+\}?|_[A-Za-z0-9]+)?(?:/\\[A-Za-z]+)*"
        r")(?!\$)"
    )
    return symbolic_expr.sub(lambda m: f"${_clean_axis_math_fragment(m.group(1))}$", label)


def _normalize_axis_math_fragment(fragment: str) -> str:
    """Repair common LLM shorthand inside `$...$` figure labels."""

    text = _clean_axis_math_fragment(str(fragment or "").strip())
    if not text:
        return text
    greek = {
        "alpha": r"\alpha",
        "beta": r"\beta",
        "delta": r"\delta",
        "epsilon": r"\epsilon",
        "eta": r"\eta",
        "gamma": r"\gamma",
        "lambda": r"\lambda",
        "mu": r"\mu",
        "phi": r"\phi",
        "psi": r"\psi",
        "rho": r"\rho",
        "sigma": r"\sigma",
        "theta": r"\theta",
    }

    def _subscript_suffix(sub: str) -> str:
        if sub in {"max", "min"}:
            return rf"{{\{sub}}}"
        if len(sub) == 1:
            return sub
        return rf"{{\mathrm{{{sub}}}}}"

    for name, latex in greek.items():
        text = re.sub(
            rf"(?<!\\)\b{name}_([A-Za-z][A-Za-z0-9]*)\b",
            lambda m, replacement=latex: rf"{replacement}_{_subscript_suffix(m.group(1))}",
            text,
        )
    for name, latex in greek.items():
        text = re.sub(rf"(?<!\\)\b{name}\b", lambda _m, replacement=latex: replacement, text)
    if re.fullmatch(r"P[_ ]?max", text):
        return r"P_{\max}"
    if re.fullmatch(r"P[_ ]?min", text):
        return r"P_{\min}"
    if re.fullmatch(r"R[_ ]?sum", text):
        return r"R_{\mathrm{sum}}"
    if re.fullmatch(r"R[_ ]?sec\s*,\s*min|R[_ ]?sec[_ ]?min", text):
        return r"R_{\mathrm{sec}}^{\min}"
    if re.fullmatch(r"R[_ ]?sec\s*,\s*sum|R[_ ]?sec[_ ]?sum", text):
        return r"R_{\mathrm{sec}}^{\mathrm{sum}}"

    def _repair_simple_subscript(match: re.Match[str]) -> str:
        base = match.group(1)
        sub = match.group(2)
        if sub in {"max", "min"}:
            return rf"{base}_{{\{sub}}}"
        if len(sub) == 1:
            return rf"{base}_{sub}"
        return rf"{base}_{{\mathrm{{{sub}}}}}"

    if "\\" not in text and "{" not in text:
        text = re.sub(r"\b([A-Za-z])_([A-Za-z][A-Za-z0-9]*)\b", _repair_simple_subscript, text)
    return text


def _repair_plain_subscript_axis_label(text: str) -> str:
    label = str(text or "").strip()
    if not label or "$" in label or "\\" in label or "_" not in label:
        return label

    token_pattern = re.compile(r"\b([A-Za-z]+_[A-Za-z][A-Za-z0-9]*)\b")
    return token_pattern.sub(lambda m: f"${_normalize_axis_math_fragment(m.group(1))}$", label)


def _repair_dollar_math_axis_label(text: str) -> str:
    label = str(text or "").strip()
    if "$" not in label:
        return label
    return re.sub(r"\$([^$]+)\$", lambda m: f"${_normalize_axis_math_fragment(m.group(1))}$", label)


def _safe_display(value: Any) -> str:
    text = str(value).replace("\n", " ").strip()
    exact_replacements = {
        "SNR_dB": "SNR (dB)",
        "system.SNR_dB": "SNR (dB)",
        "P_hetero_ratio": "Power heterogeneity ratio",
        "system.P_hetero_ratio": "Power heterogeneity ratio",
        "weighted sum rate bpsHz": "Weighted sum-rate (bps/Hz)",
        "weighted_sum_rate_bpsHz": "Weighted sum-rate (bps/Hz)",
        "sum_rate_bpsHz": "Sum-rate (bps/Hz)",
        "min_user_rate_bpsHz": "Minimum user rate (bps/Hz)",
        "R_sec_min_bpsHz": r"worst-user secrecy rate $R_{\mathrm{sec}}^{\min}$ (bps/Hz)",
        "R_sec_sum_bpsHz": r"sum secrecy rate $R_{\mathrm{sec}}^{\mathrm{sum}}$ (bps/Hz)",
        "sum_power_W": "Sum transmit power (W)",
        "sum_power_dBm": "Sum transmit power (dBm)",
        "sum power W": "Sum transmit power (W)",
        "sum power dBm": "Sum transmit power (dBm)",
        "spectral_radius_F": r"Spectral radius $\rho(F)$",
        "spectral radius F": r"Spectral radius $\rho(F)$",
        "rho_F": r"Spectral radius $\rho(F)$",
        "gamma_target": r"SINR target $\gamma$ (dB)",
        "gamma target": r"SINR target $\gamma$ (dB)",
        "sinr_target_dB": "SINR target (dB)",
        "sinr target dB": "SINR target (dB)",
        "constraints.sinr_target_dB": "SINR target (dB)",
        "uncertainty_radius": "Uncertainty radius",
        "uncertainty radius": "Uncertainty radius",
        "constraints.uncertainty_radius": "Uncertainty radius",
        "constraints.gamma_target": r"SINR target $\gamma$ (dB)",
        "lambda_s": r"$\lambda_s$",
        "lambda_c": r"$\lambda_c$",
        "lambda_p": r"$\lambda_p$",
        "optimization.lambda_s": r"$\lambda_s$",
        "optimization.lambda_c": r"$\lambda_c$",
        "optimization.lambda_p": r"$\lambda_p$",
        "E_min": r"$E_{\min}$",
        "E_min_mW": r"$E_{\min}$ (mW)",
        "constraints.E_min": r"$E_{\min}$",
        "constraints.E_min_mW": r"$E_{\min}$ (mW)",
        "EH.steepness_a": "EH steepness",
        "system.Pmax": r"$P_{\max}$",
    }
    if text in exact_replacements:
        return exact_replacements[text]
    if "$" in text:
        return " ".join(_repair_dollar_math_axis_label(text).split())
    if "\\" in text:
        return _repair_unwrapped_latex_axis_label(text)
    if "_" in text and "." not in text and any(char.isspace() for char in text):
        return " ".join(_repair_plain_subscript_axis_label(text).split())
    replacements = {
        "\u4f4d\u9227?": r"$\lambda$",
        "\u4f4d\u9227\u4f2e": r"$\lambda_1$",
        "\u4f4d\u9227\u506e": r"$\lambda_2$",
        "\u4f4d\u9227\u515f": r"$\lambda_3$",
        "\u922d?": "-",
        "\u7a5e": "tr",
        "\u754f\u8def": r"$\eta$",
        "\u754f": r"$\eta$",
        "λ": r"$\lambda$",
        "ρ": r"$\rho$",
        "\u87fb": r"$\rho$",
        "¦Ñ": r"$\rho$",
        "¦Ë": r"$\lambda$",
        "\u4f4d": r"$\lambda$",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    if ("_" in text or "." in text) and text not in exact_replacements:
        text = text.replace("_", " ").replace(".", " ")
    return " ".join(text.split())


def _error_display_label(value: str) -> str:
    mapping = {
        "std": "standard deviation",
        "stderr": "standard error",
        "ci95": "95% confidence interval",
        "none": "no error bars",
    }
    return mapping.get(str(value).strip().lower(), str(value).strip())


def _metric_higher_is_better(metric: str, default: bool = True) -> bool:
    lowered = str(metric or "").strip().lower()
    if not lowered:
        return default
    lower_is_better_tokens = (
        "sum_power",
        "total_power",
        "transmit_power",
        "power_consumption",
        "energy_consumption",
        "violation",
        "outage",
        "gap",
        "error",
        "mse",
        "crb",
        "runtime",
        "time",
        "iteration",
        "latency",
    )
    if any(token in lowered for token in lower_is_better_tokens):
        return False
    higher_is_better_tokens = (
        "feasible",
        "feasibility",
        "rate",
        "throughput",
        "snr",
        "sinr",
        "energy",
        "harvest",
        "power",
        "objective",
        "utility",
        "gain",
        "efficiency",
    )
    if any(token in lowered for token in higher_is_better_tokens):
        return True
    return default


def _metric_direction_is_forced(metric: str) -> bool:
    lowered = str(metric or "").strip().lower()
    return any(
        token in lowered
        for token in (
            "violation",
            "outage",
            "gap",
            "error",
            "mse",
            "crb",
            "runtime",
            "time",
            "iteration",
            "sum_power",
            "total_power",
            "transmit_power",
            "power_consumption",
            "feasible",
            "feasibility",
        )
    )


def _is_diagnostic_primary_metric(metric: str) -> bool:
    lowered = str(metric or "").strip().lower()
    return any(
        token in lowered
        for token in (
            "violation",
            "outage",
            "feasible",
            "feasibility",
            "constraint",
            "shortfall",
            "residual",
        )
    )


def _figure_counts_toward_paper_minimum(fig: dict[str, Any], y_metric: str) -> bool:
    """Paper figures should be performance/insight figures, not feasibility-only diagnostics."""
    intent = str(fig.get("chart_intent") or fig.get("intent") or "").strip().lower()
    if _is_diagnostic_primary_metric(y_metric):
        return False
    if any(token in intent for token in ("feasibility", "violation", "diagnostic_only")):
        return False
    return True


def _comparison_metric_from_figure_metric(metric: str, df: pd.DataFrame) -> str:
    metric_name = str(metric or "").strip()
    metric_lower = metric_name.lower()
    power_db_aliases = {
        "sum_power_dbm": "sum_power_W",
        "total_power_dbm": "total_power_W",
    }
    if metric_lower in power_db_aliases and power_db_aliases[metric_lower] in df.columns:
        return power_db_aliases[metric_lower]
    return metric_name


def repair_phase25_primary_metric_from_figures(plan: dict[str, Any], df: pd.DataFrame) -> dict[str, Any]:
    """Use the main evidence figure's physical KPI as the claim metric.

    Plot axes may use dBm for readability, but relative gains and claim gates should
    use the linear physical objective (for example W) when available.
    """
    if not isinstance(plan, dict):
        return plan
    figures = plan.get("figure_specs", [])
    if not isinstance(figures, list):
        return plan
    payload = copy.deepcopy(plan)
    current = payload.get("primary_metric", {}) if isinstance(payload.get("primary_metric"), dict) else {}
    current_name = str(current.get("name") or "").strip()
    current_ok = bool(current_name and current_name in df.columns and not _is_diagnostic_primary_metric(current_name))

    candidate_figures = sorted(
        [fig for fig in figures if isinstance(fig, dict)],
        key=lambda fig: 0
        if str(fig.get("chart_intent") or fig.get("intent") or "").strip().lower()
        in {"main_comparison", "overall_utility", "utility_comparison"}
        else 1,
    )
    for fig in candidate_figures:
        metric_obj = fig.get("metric", {}) if isinstance(fig.get("metric"), dict) else {}
        figure_metric = str(metric_obj.get("name") or fig.get("y_metric") or "").strip()
        candidate = _comparison_metric_from_figure_metric(figure_metric, df)
        if not candidate or candidate not in df.columns or _is_diagnostic_primary_metric(candidate):
            continue
        if current_ok and current_name == candidate:
            current["display_name"] = _metric_name_to_default_label(
                candidate,
                str(current.get("display_name") or metric_obj.get("display_name") or candidate),
            )
            payload["primary_metric"] = current
            return payload
        payload["primary_metric"] = {
            "name": candidate,
            "display_name": _metric_name_to_default_label(candidate, str(metric_obj.get("display_name") or candidate)),
            "higher_is_better": _metric_higher_is_better(candidate, bool(metric_obj.get("higher_is_better", True))),
        }
        payload["_primary_metric_repair"] = {
            "old_primary_metric": current,
            "new_primary_metric": payload["primary_metric"],
            "source_figure_id": str(fig.get("figure_id") or fig.get("id") or ""),
            "reason": "claim_metric_aligned_with_main_physical_kpi",
        }
        return payload
    return payload


def _format_category_label(value: Any) -> str:
    try:
        numeric = float(value)
        if math.isfinite(numeric) and abs(numeric - round(numeric)) <= 1.0e-9:
            return str(int(round(numeric)))
        if math.isfinite(numeric):
            return f"{numeric:g}"
    except Exception:
        pass
    return _safe_display(value)


def _is_finite_value(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float, np.integer, np.floating)):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return all(_is_finite_value(v) for v in value.values())
    if isinstance(value, list):
        return all(_is_finite_value(v) for v in value)
    return True


def _coerce_bool_cell(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric = float(value)
        return math.isfinite(numeric) and bool(numeric)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "ok", "success", "feasible", "optimal", "optimal_inaccurate", "solved"}:
        return True
    if text in {"0", "false", "no", "n", "failed", "infeasible", "error", "exception", "nan", "timeout", ""}:
        return False
    return False


def _phase25_soft_feasibility_tolerance() -> float:
    for env_name in (
        "WARA_PHASE25_SOFT_FEASIBILITY_TOLERANCE",
        "WCL_PHASE25_SOFT_FEASIBILITY_TOLERANCE",
        "WARA_PHASE24_QUALITY_TOLERANCE_CAP",
    ):
        raw_value = os.environ.get(env_name)
        if raw_value is None:
            continue
        try:
            value = float(raw_value)
        except ValueError:
            continue
        if math.isfinite(value) and value >= 0.0:
            return value
    return 1.0e-2


def _phase25_max_violation_series(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float, index=df.index)
    candidates = [
        col
        for col in df.columns
        if "violation" in str(col).lower() or "residual" in str(col).lower()
    ]
    if not candidates:
        return pd.Series(np.nan, index=df.index)
    numeric_cols: list[pd.Series] = []
    for col in candidates:
        series = pd.to_numeric(df[col], errors="coerce").abs()
        if bool(series.notna().any()):
            numeric_cols.append(series)
    if not numeric_cols:
        return pd.Series(np.nan, index=df.index)
    return pd.concat(numeric_cols, axis=1).max(axis=1, skipna=True)


def _detect_seed_column(df: pd.DataFrame) -> str | None:
    for col in SEED_ALIASES:
        if col in df.columns:
            return col
    return None


def _build_method_naming_maps(plan: dict[str, Any]) -> dict[str, Any]:
    methods = plan.get("compared_methods", []) if isinstance(plan, dict) else []
    short_by_internal: dict[str, str] = {}
    long_by_internal: dict[str, str] = {}
    role_by_internal: dict[str, str] = {}
    source_by_internal: dict[str, str] = {}
    records: list[dict[str, Any]] = []
    for item in methods:
        if not isinstance(item, dict):
            continue
        internal_name = str(item.get("internal_name") or item.get("name") or "").strip()
        if not internal_name:
            continue
        short_name = _safe_display(str(item.get("display_name_short") or internal_name.replace("_", " ").title()).strip())
        long_name = _safe_display(str(item.get("display_name_long") or short_name).strip())
        role = str(item.get("role") or "").strip()
        source = str(item.get("source_of_name") or "fallback_generated").strip()
        short_by_internal[internal_name] = short_name
        long_by_internal[internal_name] = long_name
        role_by_internal[internal_name] = role
        source_by_internal[internal_name] = source
        alias = str(item.get("name") or internal_name).strip()
        if alias and alias not in short_by_internal:
            short_by_internal[alias] = short_name
            long_by_internal[alias] = long_name
            role_by_internal[alias] = role
            source_by_internal[alias] = source
        records.append(
            {
                "internal_name": internal_name,
                "role": role,
                "display_name_short": short_name,
                "display_name_long": long_name,
                "source_of_name": source,
            }
        )
    return {
        "short_by_internal": short_by_internal,
        "long_by_internal": long_by_internal,
        "role_by_internal": role_by_internal,
        "source_by_internal": source_by_internal,
        "records": records,
    }


def _method_display_name(plan: dict[str, Any], internal_name: str, *, long: bool = False) -> str:
    maps = _build_method_naming_maps(plan)
    bucket = maps["long_by_internal"] if long else maps["short_by_internal"]
    label = bucket.get(str(internal_name).strip())
    if label:
        return _safe_display(label)
    fallback = str(internal_name).replace("_", " ").strip()
    return _safe_display(fallback.title() if fallback else "Method")


def _method_plan_item(plan: dict[str, Any], method_id: str) -> dict[str, Any]:
    target = str(method_id or "").strip()
    for item in plan.get("compared_methods", []) if isinstance(plan, dict) else []:
        if not isinstance(item, dict):
            continue
        aliases = {
            str(item.get("internal_name") or "").strip(),
            str(item.get("name") or "").strip(),
            str(item.get("id") or "").strip(),
        }
        if target and target in aliases:
            return item
    return {}


def _method_descriptor_text(plan: dict[str, Any], method_id: str) -> str:
    item = _method_plan_item(plan, method_id)
    pieces = [
        method_id,
        item.get("role", ""),
        item.get("name", ""),
        item.get("id", ""),
        item.get("internal_name", ""),
        item.get("display_name_short", ""),
        item.get("display_name_long", ""),
        item.get("scientific_purpose", ""),
        item.get("implementation_hint", ""),
        item.get("fairness_rule", ""),
    ]
    return " ".join(str(piece or "") for piece in pieces).strip().lower()


def _is_optimal_reference_method(plan: dict[str, Any], method_id: str) -> bool:
    """Identify methods that should be matched, not beaten."""
    item = _method_plan_item(plan, method_id)
    role = str(item.get("role", "")).strip().lower()
    if any(token in role for token in ("heuristic", "practical", "ablation", "fixed", "random", "greedy")):
        return False
    text = _method_descriptor_text(plan, method_id)
    if not text:
        return False
    if str(method_id).strip().lower() in {"proposed", "baseline"}:
        return False
    text = text.replace("no oracle", "").replace("non-oracle", "").replace("without oracle", "")
    reference_tokens = (
        "centralized_lp",
        "centralized lp",
        "cent-lp",
        "lp optimal",
        "optimal reference",
        "exact reference",
        "global optimum",
        "globally optimal",
        "theoretical lower bound",
        "theoretical upper bound",
        "lower bound",
        "upper bound",
        "oracle",
        "relaxation bound",
        "closed-form optimum",
        "closed form optimum",
    )
    return any(token in text for token in reference_tokens)


def _method_is_practical_benchmark(plan: dict[str, Any], method_id: str) -> bool:
    if not method_id or method_id == "proposed":
        return False
    if _is_optimal_reference_method(plan, method_id):
        return False
    if _method_is_ineligible_display_benchmark(plan, method_id):
        return False
    text = _method_descriptor_text(plan, method_id)
    return any(token in text for token in ("baseline", "benchmark", "heuristic", "ablation", "diagnostic", "fixed", "random", "greedy"))


def _method_is_ineligible_display_benchmark(plan: dict[str, Any], method_id: str) -> bool:
    """Reject baselines that the contract itself marks as absent or redundant."""
    text = _method_descriptor_text(plan, method_id)
    if not text:
        return False
    absent_or_unavailable = any(
        token in text
        for token in (
            "unavailable",
            "invalid_for_frozen_model",
            "absent from the frozen",
            "absent from frozen",
            "not present in the frozen",
            "mark unavailable",
            "no power-splitting control",
            "no ps variable",
        )
    )
    redundant_or_do_not_plot = any(
        token in text
        for token in (
            "redundant with proposed",
            "same as proposed",
            "identical to proposed",
            "do not plot",
            "not plot as a distinct curve",
        )
    )
    no_declared_control = "no " in text and any(token in text for token in ("control", "variable")) and any(
        token in text for token in ("frozen", "contract", "model")
    )
    return bool(absent_or_unavailable or redundant_or_do_not_plot or no_declared_control)


def _method_display_priority(plan: dict[str, Any], method_id: str) -> int:
    """Lower values mean a method is a better primary paper contrast."""
    if _method_is_ineligible_display_benchmark(plan, method_id):
        return 999
    item = _method_plan_item(plan, method_id)
    role = str(item.get("role", "")).strip().lower()
    if any(token in role for token in ("upper", "oracle", "relax")) or _is_optimal_reference_method(plan, method_id):
        return 90
    if any(token in role for token in ("main_baseline", "primary_baseline", "claim_target")):
        return 0
    try:
        return int(item.get("display_priority"))
    except (TypeError, ValueError):
        pass
    text = _method_descriptor_text(plan, method_id)
    if any(token in role for token in ("mechanism", "ablation")):
        return 1
    if any(token in role for token in ("direction", "heuristic", "benchmark")):
        return 2
    if any(token in role for token in ("diagnostic", "model")) or "linear_eh" in text or "linear-eh" in text:
        return 4
    return 3


def _present_methods(df: pd.DataFrame) -> list[str]:
    if "method" not in df.columns:
        return []
    return sorted(str(item) for item in df["method"].dropna().astype(str).unique().tolist())


def _mean_metric_for_method(df: pd.DataFrame, method_id: str, metric: str) -> float | None:
    if "method" not in df.columns or metric not in df.columns:
        return None
    values = pd.to_numeric(df.loc[df["method"].astype(str) == method_id, metric], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.mean())


def _practical_and_reference_candidates(plan: dict[str, Any]) -> tuple[list[str], list[str]]:
    practical_candidates: list[str] = []
    reference_candidates: list[str] = []

    def add_candidate(method_id: str, role: str = "") -> None:
        method_id = str(method_id or "").strip()
        if not method_id or method_id == "proposed":
            return
        role = str(role or "").lower()
        if _is_optimal_reference_method(plan, method_id) or any(token in role for token in ("upper", "oracle", "relax")):
            if method_id not in reference_candidates:
                reference_candidates.append(method_id)
            return
        if (
            any(token in role for token in ("main_baseline", "benchmark", "heuristic", "ablation", "diagnostic", "direction", "model"))
            or _method_is_practical_benchmark(plan, method_id)
        ):
            if method_id not in practical_candidates:
                practical_candidates.append(method_id)

    for method in plan.get("compared_methods", []):
        if not isinstance(method, dict):
            continue
        add_candidate(_method_id_from_plan_item(method), str(method.get("role", "")))
    for fig in plan.get("figure_specs", []):
        if not isinstance(fig, dict):
            continue
        for method_id in _get_figure_methods(normalize_figure_spec(fig, plan.get("primary_metric", {}))):
            add_candidate(method_id)
    return practical_candidates, reference_candidates


def select_strongest_practical_baseline_method(df: pd.DataFrame, plan: dict[str, Any]) -> str:
    """Choose the strongest non-oracle practical method for transparent audit only."""
    present = _present_methods(df)
    primary_metric = str((plan.get("primary_metric") or {}).get("name") or "objective")
    higher_is_better = bool((plan.get("primary_metric") or {}).get("higher_is_better", True))
    practical_candidates, reference_candidates = _practical_and_reference_candidates(plan)
    valid = [method_id for method_id in practical_candidates if method_id in present and method_id != "proposed"]
    if "baseline" in present and "baseline" not in valid and not _is_optimal_reference_method(plan, "baseline"):
        valid.append("baseline")
    if valid and primary_metric in df.columns:
        scored: list[tuple[float, int, str]] = []
        for order, method_id in enumerate(valid):
            strength = _mean_metric_for_method(df, method_id, primary_metric)
            if strength is not None:
                scored.append((strength, order, method_id))
        if scored:
            scored.sort(key=lambda item: (item[0], -item[1]), reverse=higher_is_better)
            return scored[0][2]
    if valid:
        return valid[0]
    for method_id in reference_candidates:
        if method_id in present:
            return method_id
    return next((method_id for method_id in present if method_id != "proposed"), "baseline")


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _prefer_phase24_output_paths(run_dir: Path) -> tuple[Path, Path, str]:
    outputs_dir = run_dir / "phase2-4" / "solver" / "outputs"
    candidates: list[tuple[float, Path, Path, str]] = []
    for prefix, label in (
        ("paper_validation", "paper_validation"),
        ("medium_validation", "medium_validation"),
        ("scout_validation", "scout_validation"),
    ):
        summary_path = outputs_dir / f"{prefix}_summary.json"
        results_path = outputs_dir / f"{prefix}_results.csv"
        staleness = _validation_output_staleness(run_dir, prefix)
        summary_payload = _read_json(summary_path)
        if (
            summary_path.exists()
            and _csv_has_rows(results_path)
            and _validation_summary_is_complete(summary_payload)
            and not staleness.get("is_stale")
        ):
            try:
                mtime = max(summary_path.stat().st_mtime, results_path.stat().st_mtime)
            except OSError:
                mtime = 0.0
            candidates.append((mtime, summary_path, results_path, label))
    if candidates:
        _mtime, summary_path, results_path, label = max(candidates, key=lambda item: item[0])
        return summary_path, results_path, label
    return outputs_dir / "validation_summary.json", outputs_dir / "validation_results.csv", "quick_validation"


def _csv_has_rows(path: Path) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        return False
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                return False
            return any(True for _ in reader)
    except Exception:
        return False


def _validation_source_paths(run_dir: Path, prefix: str, *, include_phase25_plan: bool = False) -> list[Path]:
    solver_dir = run_dir / "phase2-4" / "solver"
    paths = [
        solver_dir / "generated_plugin.py",
        solver_dir / "generated_experiment_core.py",
        solver_dir / "problem_data.py",
        solver_dir / "validation_cases.py",
        run_dir / "phase2-4" / "validation_plan.yaml",
    ]
    if include_phase25_plan:
        paths.extend(
            [
                run_dir / "phase2-5" / "experiment_plan.json",
                run_dir / "phase2-5" / "paper_sweep_plan.json",
            ]
        )
    return paths


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validation_source_fingerprint_payload(
    run_dir: str | Path,
    prefix: str,
    *,
    include_phase25_plan: bool = False,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    sources: dict[str, Any] = {}
    for path in _validation_source_paths(run_dir, prefix, include_phase25_plan=include_phase25_plan):
        try:
            rel = str(path.relative_to(run_dir))
        except ValueError:
            rel = str(path)
        if not path.exists():
            sources[rel] = {"missing": True}
            continue
        try:
            stat = path.stat()
            sources[rel] = {
                "sha256": _file_sha256(path),
                "size": int(stat.st_size),
            }
        except OSError:
            sources[rel] = {"missing": True}
    return {
        "version": 1,
        "prefix": str(prefix or ""),
        "include_phase25_plan": bool(include_phase25_plan),
        "sources": sources,
    }


def _validation_source_fingerprint_path(run_dir: str | Path, prefix: str) -> Path:
    return Path(run_dir) / "phase2-4" / "solver" / "outputs" / f"{prefix}_source_fingerprint.json"


def _stored_validation_source_fingerprint(run_dir: str | Path, prefix: str) -> dict[str, Any]:
    run_dir = Path(run_dir)
    fingerprint_path = _validation_source_fingerprint_path(run_dir, prefix)
    payload = _read_json(fingerprint_path)
    if isinstance(payload.get("sources"), dict):
        return payload
    summary_payload = _read_json(run_dir / "phase2-4" / "solver" / "outputs" / f"{prefix}_summary.json")
    embedded = summary_payload.get("validation_source_fingerprint")
    if isinstance(embedded, dict) and isinstance(embedded.get("sources"), dict):
        return embedded
    return {}


def _write_validation_source_fingerprint(
    run_dir: str | Path,
    prefix: str,
    *,
    include_phase25_plan: bool = False,
) -> dict[str, Any]:
    payload = _validation_source_fingerprint_payload(run_dir, prefix, include_phase25_plan=include_phase25_plan)
    _write_json(_validation_source_fingerprint_path(run_dir, prefix), payload)
    return payload


def _validation_summary_is_complete(summary: dict[str, Any]) -> bool:
    if not isinstance(summary, dict):
        return False
    if bool(summary.get("partial", False)):
        return False
    planned = summary.get("planned_jobs")
    completed = summary.get("actual_completed_jobs")
    if planned is not None and completed is not None:
        try:
            return int(completed) >= int(planned)
        except (TypeError, ValueError):
            return False
    return True


def _cached_validation_output(
    run_dir: str | Path,
    prefix: str,
    staleness: dict[str, Any],
    *,
    include_phase25_plan: bool = False,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    outputs_dir = run_dir / "phase2-4" / "solver" / "outputs"
    summary_path = outputs_dir / f"{prefix}_summary.json"
    results_path = outputs_dir / f"{prefix}_results.csv"
    cases_path = outputs_dir / f"{prefix}_cases.json"
    if bool(staleness.get("is_stale")):
        return {}
    summary = _read_json(summary_path)
    if not (summary_path.exists() and _csv_has_rows(results_path) and _validation_summary_is_complete(summary)):
        return {}
    if not isinstance(staleness.get("stored_source_fingerprint"), dict) or not staleness.get("stored_source_fingerprint", {}).get("sources"):
        staleness = dict(staleness)
        staleness["sealed_legacy_source_fingerprint"] = _write_validation_source_fingerprint(
            run_dir,
            prefix,
            include_phase25_plan=include_phase25_plan,
        )
    return {
        "cached": True,
        "results_csv": str(results_path),
        "summary_json": str(summary_path),
        "cases_json": str(cases_path),
        "num_results": max(sum(1 for _ in results_path.open("r", encoding="utf-8-sig", errors="ignore")) - 1, 0),
        "quick_mode": bool(summary.get("quick_mode", False)),
        "paper_sweep_run_mode": str(summary.get("paper_sweep_run_mode") or ""),
        "validation_output_prefix": prefix,
        "validation_output_staleness_before_run": staleness,
        "paper_validation_staleness_before_run": staleness if prefix == "paper_validation" else _paper_validation_staleness(run_dir),
        "reused_existing_paper_rows": True,
    }


def _validation_output_staleness(
    run_dir: str | Path,
    prefix: str,
    *,
    include_phase25_plan: bool = False,
) -> dict[str, Any]:
    """Return whether generated validation outputs predate current code or contracts.

    Experiments are expensive.  We use content fingerprints first and only fall
    back to mtimes for old runs that predate the fingerprint manifest.
    """
    run_dir = Path(run_dir)
    solver_dir = run_dir / "phase2-4" / "solver"
    outputs_dir = solver_dir / "outputs"
    prefix = str(prefix or "paper_validation").strip()
    source_paths = _validation_source_paths(run_dir, prefix, include_phase25_plan=include_phase25_plan)
    current_fingerprint = _validation_source_fingerprint_payload(
        run_dir,
        prefix,
        include_phase25_plan=include_phase25_plan,
    )
    stored_fingerprint = _stored_validation_source_fingerprint(run_dir, prefix)
    if isinstance(stored_fingerprint.get("sources"), dict):
        current_sources = current_fingerprint.get("sources", {})
        stored_sources = stored_fingerprint.get("sources", {})
        changed_sources = sorted(
            key
            for key in set(current_sources)
            if current_sources.get(key) != stored_sources.get(key)
        )
        return {
            "is_stale": bool(changed_sources),
            "reason": f"{prefix}_source_content_changed" if changed_sources else "",
            "stale_paths": changed_sources,
            "source_fingerprint_path": str(_validation_source_fingerprint_path(run_dir, prefix)),
            "current_source_fingerprint": current_fingerprint,
            "stored_source_fingerprint": stored_fingerprint,
        }
    source_mtimes: dict[str, float] = {}
    for path in source_paths:
        if not path.exists():
            continue
        try:
            source_mtimes[str(path.relative_to(run_dir))] = path.stat().st_mtime
        except OSError:
            continue
    if not source_mtimes:
        return {"is_stale": False, "reason": "missing_phase24_sources", "stale_paths": []}
    source_mtime = max(source_mtimes.values())
    candidate_paths = [
        outputs_dir / f"{prefix}_results.csv",
        outputs_dir / f"{prefix}_summary.json",
        outputs_dir / f"{prefix}_cases.json",
    ]
    stale_paths: list[str] = []
    mtimes: dict[str, float] = dict(source_mtimes)
    for path in candidate_paths:
        if not path.exists():
            continue
        try:
            path_mtime = path.stat().st_mtime
        except OSError:
            continue
        mtimes[path.name] = path_mtime
        if path_mtime + 1.0 < source_mtime:
            stale_paths.append(str(path))
    stale_reason = ""
    if stale_paths:
        stale_reason = f"{prefix}_outputs_predate_phase24_code_or_contract"
    return {
        "is_stale": bool(stale_paths),
        "reason": stale_reason,
        "generated_plugin_path": str(solver_dir / "generated_plugin.py"),
        "generated_core_path": str(solver_dir / "generated_experiment_core.py"),
        "generated_plugin_mtime": source_mtimes.get("phase2-4/solver/generated_plugin.py"),
        "generated_core_mtime": source_mtimes.get("phase2-4/solver/generated_experiment_core.py"),
        "generated_phase24_code_mtime": max(
            source_mtimes.get("phase2-4/solver/generated_plugin.py", 0.0),
            source_mtimes.get("phase2-4/solver/generated_experiment_core.py", 0.0),
        ),
        "source_contract_mtime": source_mtime,
        "stale_paths": stale_paths,
        "mtimes": mtimes,
        "source_fingerprint_path": str(_validation_source_fingerprint_path(run_dir, prefix)),
        "current_source_fingerprint": current_fingerprint,
        "stored_source_fingerprint": {},
    }


def _paper_validation_staleness(run_dir: str | Path) -> dict[str, Any]:
    """Backward-compatible wrapper for paper-sweep staleness checks."""
    return _validation_output_staleness(run_dir, "paper_validation")


def load_phase24_results(run_dir: str | Path) -> tuple[dict[str, Any], pd.DataFrame]:
    run_dir = Path(run_dir)
    paper_staleness = _paper_validation_staleness(run_dir)
    summary_path, results_path, data_source = _prefer_phase24_output_paths(run_dir)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["_phase25_data_source"] = data_source
    summary["_phase25_paper_validation_staleness"] = paper_staleness
    if paper_staleness.get("is_stale") and data_source != "paper_validation":
        summary["_phase25_ignored_paper_validation"] = paper_staleness
    df = pd.read_csv(results_path, encoding="utf-8-sig")
    return summary, df


def _copy_numeric_alias(df: pd.DataFrame, target: str, sources: list[str], scale: float = 1.0) -> None:
    if target in df.columns:
        return
    for source in sources:
        if source not in df.columns:
            continue
        values = pd.to_numeric(df[source], errors="coerce")
        if values.notna().any():
            df[target] = values * scale
            return


def _copy_db_alias(df: pd.DataFrame, target: str, sources: list[str]) -> None:
    if target in df.columns:
        return
    for source in sources:
        if source not in df.columns:
            continue
        values = pd.to_numeric(df[source], errors="coerce")
        finite_positive = values.where(values > 0.0)
        if finite_positive.notna().any():
            df[target] = 10.0 * np.log10(finite_positive.clip(lower=1.0e-30))
            return


def _ensure_metric_alias_columns(df: pd.DataFrame) -> pd.DataFrame:
    _copy_numeric_alias(df, "objective_value", ["objective"])
    _copy_numeric_alias(df, "rate_bpsHz", ["sum_rate_bpsHz", "R_c_bpsHz", "R_c", "rate", "rate_Mbps"])
    _copy_numeric_alias(df, "sum_rate_bpsHz", ["rate_bpsHz", "sum_rate", "R_c_bpsHz", "R_c", "rate"])
    _copy_numeric_alias(df, "sum_rate_bps_hz", ["sum_rate_bpsHz", "sum_rate", "rate_bpsHz", "R_c_bpsHz", "R_c", "rate"])
    _copy_numeric_alias(df, "spectral_efficiency", ["rate_bpsHz", "R_c_bpsHz", "R_c", "rate"])
    _copy_numeric_alias(df, "sensing_gain", ["sensing_metric", "sensing_beampattern_gain", "f_sen", "radar_gain", "radar_SNR"])
    _copy_db_alias(df, "radar_SNR_dB", ["radar_SNR", "sensing_gain", "radar_snr", "sensing_metric"])
    _copy_numeric_alias(df, "crb_trace", ["tr_CRB", "crb_tr", "crb"])
    _copy_numeric_alias(df, "tr_CRB", ["crb_trace", "crb_tr", "crb"])
    _copy_numeric_alias(df, "R_c_bpsHz", ["rate", "rate_bpsHz", "rate_nats", "rate_Mbps"])
    _copy_numeric_alias(df, "P_EH_mW", ["P_EH", "eh_power", "eh_total"], scale=1000.0)
    _copy_numeric_alias(df, "harvested_power_mW", ["P_EH_mW", "eh_total_mW"])
    _copy_numeric_alias(df, "harvested_power_mW", ["P_EH", "eh_power", "eh_total"], scale=1000.0)
    _copy_numeric_alias(df, "eh_total_mW", ["P_EH_mW"])
    _copy_numeric_alias(df, "eh_total_mW", ["P_EH", "eh_power", "eh_total"], scale=1000.0)
    _copy_numeric_alias(df, "harvested_energy_mW", ["true_harvested_energy_mW", "harvested_power_mW", "P_EH_mW", "Psi_eh", "harvested_energy", "Eharv"])
    _copy_numeric_alias(df, "true_harvested_energy_mW", ["harvested_energy_mW", "harvested_energy", "Eharv", "P_EH_mW"])
    _copy_numeric_alias(df, "constraint_violation_max", ["max_constraint_violation", "cv_max", "constraint_violation", "total_violation"])
    _copy_numeric_alias(df, "max_constraint_violation", ["constraint_violation_max", "cv_max", "constraint_violation", "total_violation"])
    _copy_numeric_alias(df, "optimal_rho", ["rho", "rho_star", "rho_opt"])
    _copy_numeric_alias(df, "rank_V_star", ["rank_V", "rank_v", "rank_W", "rank_Wc"])
    _copy_numeric_alias(df, "eigenvalue_ratio_V", ["eigenvalue_ratio", "eig_ratio_V", "lambda1_over_lambda2"])
    _copy_numeric_alias(df, "sca_iterations", ["sca_iter", "num_sca_iter"])
    _copy_numeric_alias(df, "bcd_iterations", ["bcd_iter", "bcd_outer_iterations"])
    _copy_numeric_alias(df, "sca_final_surrogate_gap_mW", ["sca_gap", "surrogate_gap", "sca_surrogate_gap"])
    _copy_numeric_alias(df, "final_P_in_mW", ["Pin", "P_in", "P_in_actual_mW"])
    _copy_numeric_alias(df, "rate_gap_to_R_min", ["v_rate", "rate_violation"])
    _copy_numeric_alias(df, "eh_gap_to_E_min", ["v_eh", "eh_violation"], scale=1000.0)
    return df


def normalize_results_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    numeric_candidates = {
        "objective",
        "solve_time_sec",
        "power",
        "qos_violation",
        "separation_violation",
        "swept_value",
        "iterations",
        "objective_delta",
        "position_step_norm",
        "seed",
        "total_power",
    }
    for col in df.columns:
        if col in numeric_candidates:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in df.columns:
        if col in NON_METRIC_COLUMNS or col in numeric_candidates:
            continue
        if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):
            continue
        nonempty = df[col].notna() & df[col].astype(str).str.strip().ne("")
        if not bool(nonempty.any()):
            continue
        converted = pd.to_numeric(df[col], errors="coerce")
        if float(converted[nonempty].notna().mean()) >= 0.9:
            df[col] = converted
    df = _ensure_metric_alias_columns(df)
    if "feasible" in df.columns:
        df["feasible"] = df["feasible"].apply(_coerce_bool_cell)
    if "status" in df.columns:
        df["status"] = df["status"].astype(str).str.lower()
    if "method" in df.columns:
        df["method"] = df["method"].astype(str).str.lower()
    for col in ("case_id", "case_name", "swept_param", "scenario_name"):
        if col in df.columns:
            df[col] = df[col].astype(str)
    if "status" in df.columns and "feasible" in df.columns:
        status_ok = ~df["status"].isin(FAIL_STATUSES)
        max_violation = _phase25_max_violation_series(df)
        soft_feasible = (~df["feasible"].astype(bool)) & status_ok & max_violation.notna() & (
            max_violation <= _phase25_soft_feasibility_tolerance()
        )
        if bool(soft_feasible.any()):
            df["soft_feasible_by_tolerance"] = False
            df.loc[soft_feasible, "soft_feasible_by_tolerance"] = True
            df.loc[soft_feasible, "feasible"] = True
        # Paper-facing KPI curves must aggregate only executable, constraint-
        # consistent rows. Feasibility itself is not a plotted performance
        # claim, but infeasible/fallback rows are not valid samples of rate,
        # power, sensing, energy, or objective performance.
        df["success"] = (~df["status"].isin(FAIL_STATUSES)) & df["feasible"].astype(bool)
    else:
        df["success"] = df["feasible"].astype(bool) if "feasible" in df.columns else True
    if "objective" in df.columns:
        df["finite_primary_metric"] = df["objective"].apply(lambda x: bool(pd.notna(x) and math.isfinite(float(x))))
    else:
        df["finite_primary_metric"] = False
    return df


def summarize_available_results(summary: dict[str, Any], df: pd.DataFrame) -> dict[str, Any]:
    diagnostics_fields: set[str] = set()
    for item in summary.get("results", []):
        metrics = item.get("metrics", {}) if isinstance(item, dict) else {}
        diagnostics = metrics.get("diagnostics", {}) if isinstance(metrics, dict) else {}
        if isinstance(diagnostics, dict):
            diagnostics_fields.update(diagnostics.keys())
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    bool_cols = [col for col in df.columns if pd.api.types.is_bool_dtype(df[col])]
    seed_column = _detect_seed_column(df)
    seed_coverage = None
    if seed_column:
        seed_counts = (
            df.groupby(["method", "swept_param", "swept_value"], dropna=False)[seed_column]
            .nunique()
            .reset_index(name="seed_count")
        )
        if not seed_counts.empty:
            vals = seed_counts["seed_count"].tolist()
            seed_coverage = {"min": int(min(vals)), "median": float(statistics.median(vals)), "max": int(max(vals))}
    return {
        "columns": list(df.columns),
        "methods": sorted(df["method"].dropna().astype(str).unique().tolist()) if "method" in df.columns else [],
        "numeric_metrics": numeric_cols,
        "boolean_metrics": bool_cols,
        "available_swept_params": sorted(df["swept_param"].dropna().astype(str).unique().tolist()) if "swept_param" in df.columns else [],
        "available_scenario_names": sorted(df["scenario_name"].dropna().astype(str).unique().tolist()) if "scenario_name" in df.columns else [],
        "case_count": int(df["case_id"].nunique()) if "case_id" in df.columns else 0,
        "result_count": int(len(df)),
        "diagnostics_fields": sorted(diagnostics_fields),
        "possible_primary_metrics": [col for col in ("objective", "power", "qos_violation", "separation_violation", "solve_time_sec") if col in df.columns],
        "possible_grouping_keys": [col for col in ("swept_param", "swept_value", "scenario_name", "case_id", "method", seed_column or "") if col and col in df.columns],
        "seed_column": seed_column,
        "seed_coverage": seed_coverage,
        "phase24_summary_excerpt": {
            "data_source": summary.get("_phase25_data_source", "quick_validation"),
            "quick_mode": bool(summary.get("quick_mode", False)),
            "num_cases": summary.get("num_cases"),
            "num_results": summary.get("num_results"),
            "num_comparable_cases": summary.get("num_comparable_cases"),
            "proposed_win_rate": summary.get("proposed_win_rate"),
            "all_finite": summary.get("all_finite"),
        },
    }


def _method_id_from_plan_item(item: Any) -> str:
    if not isinstance(item, dict):
        return str(item or "").strip()
    return str(item.get("internal_name") or item.get("name") or item.get("id") or "").strip()


def select_comparison_baseline_method(df: pd.DataFrame, plan: dict[str, Any]) -> str:
    """Choose the non-proposed method used for the main paper gain claim.

    When several practical baselines are available, the main claim should not
    select the weakest one merely because it yields the largest relative gain.
    It therefore chooses the strongest practical baseline that the proposed
    method still beats; the separate strongest-baseline audit remains as an
    explicit guard for cases where the strongest practical method is not beaten.
    """
    present = _present_methods(df)
    primary_metric = str((plan.get("primary_metric") or {}).get("name") or "objective")
    higher_is_better = bool((plan.get("primary_metric") or {}).get("higher_is_better", True))

    forced = str(
        plan.get("_forced_comparison_baseline_method")
        or plan.get("primary_comparison_baseline")
        or plan.get("declared_primary_comparison_baseline")
        or ""
    ).strip()
    if forced and forced in present and forced != "proposed":
        return forced

    proposed_method = "proposed" if "proposed" in present else select_comparison_proposed_method(df, plan)
    main_figure_candidates: list[str] = []
    for raw_fig in plan.get("figure_specs", []):
        if not isinstance(raw_fig, dict):
            continue
        fig = normalize_figure_spec(raw_fig, plan.get("primary_metric", {}))
        intent = str(fig.get("chart_intent") or fig.get("intent") or "").strip().lower()
        if intent not in {"main_comparison", "overall_utility", "utility_comparison"}:
            continue
        figure_methods = [method for method in _get_figure_methods(fig, present) if method in present]
        non_proposed = [method for method in figure_methods if method != proposed_method]
        for method_id in non_proposed:
            if method_id not in main_figure_candidates:
                main_figure_candidates.append(method_id)
        if len(non_proposed) == 1:
            return non_proposed[0]

    practical_candidates, reference_candidates = _practical_and_reference_candidates(plan)
    if "baseline" in present and "baseline" not in practical_candidates and not _is_optimal_reference_method(plan, "baseline"):
        practical_candidates.append("baseline")

    def strongest_positive_gain_candidate(candidates: list[str]) -> str:
        valid = [method_id for method_id in candidates if method_id in present and method_id != "proposed"]
        if not valid:
            return ""
        proposed_mean = _mean_metric_for_method(df, proposed_method, primary_metric)
        if proposed_mean is None:
            return ""
        scored: list[tuple[float, int, int, str]] = []
        for order, method_id in enumerate(valid):
            baseline_mean = _mean_metric_for_method(df, method_id, primary_metric)
            if baseline_mean is None:
                continue
            gain = proposed_mean - baseline_mean if higher_is_better else baseline_mean - proposed_mean
            if gain <= 1.0e-9:
                continue
            # Stronger means closer to the proposed method under the paper
            # metric: larger values for maximization metrics and smaller values
            # for minimization metrics.
            strength = baseline_mean if higher_is_better else -baseline_mean
            scored.append((float(strength), -_method_display_priority(plan, method_id), -order, method_id))
        if not scored:
            return ""
        scored.sort(reverse=True)
        return scored[0][3]

    selected = strongest_positive_gain_candidate(main_figure_candidates)
    if selected:
        return selected
    selected = strongest_positive_gain_candidate(practical_candidates)
    if selected:
        return selected
    prioritized = sorted(
        [method_id for method_id in practical_candidates if method_id in present and method_id != "proposed"],
        key=lambda method_id: (_method_display_priority(plan, method_id), practical_candidates.index(method_id)),
    )
    if prioritized:
        return prioritized[0]
    for method_id in reference_candidates:
        if method_id in present:
            return method_id
    for method_id in present:
        if method_id != "proposed":
            return method_id
    return "baseline"


def select_comparison_proposed_method(df: pd.DataFrame, plan: dict[str, Any]) -> str:
    """Choose the actual proposed execution id present in the result table."""
    present: list[str] = []
    if "method" in df.columns:
        present = sorted(str(item) for item in df["method"].dropna().astype(str).unique().tolist())
    if "proposed" in present:
        return "proposed"
    for method in plan.get("compared_methods", []):
        if not isinstance(method, dict):
            continue
        method_id = _method_id_from_plan_item(method)
        role = str(method.get("role", "")).strip().lower()
        if role == "proposed" and method_id in present:
            return method_id
    for fig in plan.get("figure_specs", []):
        if not isinstance(fig, dict):
            continue
        intent = str(fig.get("chart_intent") or fig.get("intent") or "").strip().lower()
        if intent not in {"main_comparison", "overall_utility", "utility_comparison"}:
            continue
        for method_id in _get_figure_methods(normalize_figure_spec(fig, plan.get("primary_metric", {}))):
            if method_id in present:
                return method_id
    return "proposed"


def build_per_case_comparison(df: pd.DataFrame, plan: dict[str, Any]) -> pd.DataFrame:
    primary_metric = plan.get("primary_metric", {}).get("name", "objective")
    higher_is_better = bool(plan.get("primary_metric", {}).get("higher_is_better", True))
    if primary_metric not in df.columns:
        return pd.DataFrame()
    proposed_method = select_comparison_proposed_method(df, plan)
    proposed = df[df["method"] == proposed_method].copy()
    baseline_method = select_comparison_baseline_method(df, plan)
    if baseline_method == proposed_method:
        present = [str(item) for item in df["method"].dropna().astype(str).unique().tolist()] if "method" in df.columns else []
        baseline_method = next((method for method in present if method != proposed_method), baseline_method)
    baseline = df[df["method"] == baseline_method].copy()
    if proposed.empty or baseline.empty:
        return pd.DataFrame()
    merge_keys = ["case_id", "case_name", "swept_param", "swept_value", "scenario_name"]
    seed_column = _detect_seed_column(df)
    if seed_column and seed_column in proposed.columns and seed_column in baseline.columns:
        merge_keys.append(seed_column)
    numeric_metric_cols = [
        col
        for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col])
        and col not in merge_keys
        and col not in NON_METRIC_COLUMNS
    ]
    left_cols = list(dict.fromkeys(merge_keys + ["status", "feasible", primary_metric, "success", "finite_primary_metric"] + numeric_metric_cols))
    for extra in ("rejection_reason", "used_position_update"):
        if extra in proposed.columns and extra not in left_cols:
            left_cols.append(extra)
    prop = proposed[left_cols].rename(columns={c: f"proposed_{c}" for c in left_cols if c not in merge_keys})
    base = baseline[left_cols].rename(columns={c: f"baseline_{c}" for c in left_cols if c not in merge_keys})
    merged = prop.merge(base, on=merge_keys, how="inner")
    merged["proposed_method"] = proposed_method
    merged["baseline_method"] = baseline_method
    merged["both_feasible"] = merged["proposed_feasible"].astype(bool) & merged["baseline_feasible"].astype(bool)
    proposed_status_ok = ~merged["proposed_status"].astype(str).str.lower().isin(FAIL_STATUSES)
    baseline_status_ok = ~merged["baseline_status"].astype(str).str.lower().isin(FAIL_STATUSES)
    if "proposed_success" in merged.columns and "baseline_success" in merged.columns:
        proposed_success = merged["proposed_success"].astype(bool) | proposed_status_ok
        baseline_success = merged["baseline_success"].astype(bool) | baseline_status_ok
        merged["both_success"] = merged["both_feasible"] & proposed_success & baseline_success
    else:
        merged["both_success"] = merged["both_feasible"] & proposed_status_ok & baseline_status_ok
    metric_p = f"proposed_{primary_metric}"
    metric_b = f"baseline_{primary_metric}"
    comparable_mask = merged["both_success"] & merged[metric_p].notna() & merged[metric_b].notna()
    merged["comparable"] = comparable_mask
    denom = merged[metric_b].abs().clip(lower=1e-9)
    raw_gain = (merged[metric_p] - merged[metric_b]) / denom
    merged["relative_gain"] = raw_gain if higher_is_better else (-raw_gain)
    merged["proposed_win"] = merged["relative_gain"] > 1e-6
    return merged


def compute_relative_gain(comparison_df: pd.DataFrame, primary_metric: str, higher_is_better: bool) -> pd.DataFrame:
    if comparison_df.empty:
        return comparison_df
    df = comparison_df.copy()
    metric_p = f"proposed_{primary_metric}"
    metric_b = f"baseline_{primary_metric}"
    denom = df[metric_b].abs().clip(lower=1e-9)
    raw_gain = (df[metric_p] - df[metric_b]) / denom
    df["relative_gain"] = raw_gain if higher_is_better else (-raw_gain)
    df["proposed_win"] = df["relative_gain"] > 1e-6
    return df


def evaluate_primary_claim_check(comparable: pd.DataFrame, plan: dict[str, Any]) -> dict[str, Any]:
    baseline_method = str(plan.get("_active_baseline_method") or "")
    if not baseline_method and not comparable.empty and "baseline_method" in comparable.columns:
        baseline_method = str(comparable["baseline_method"].dropna().astype(str).iloc[0])
    if comparable.empty:
        return {
            "mode": "advantage_over_benchmark",
            "baseline_method": baseline_method,
            "passes": False,
            "reason": "no_comparable_primary_metric_cases",
        }

    gains = pd.to_numeric(comparable.get("relative_gain"), errors="coerce").dropna()
    win_rate = float(comparable["proposed_win"].mean()) if "proposed_win" in comparable.columns else 0.0
    median_gain = float(statistics.median(gains.tolist())) if len(gains) else 0.0
    mean_gain = float(gains.mean()) if len(gains) else 0.0
    primary_metric = str(plan.get("primary_metric", {}).get("name") or "objective")
    higher_is_better = bool(plan.get("primary_metric", {}).get("higher_is_better", True))
    metric_p = f"proposed_{primary_metric}"
    metric_b = f"baseline_{primary_metric}"
    baseline_median_abs = 0.0
    proposed_median_abs = 0.0
    if higher_is_better and not _is_violation_or_feasibility_metric(primary_metric) and metric_p in comparable.columns and metric_b in comparable.columns:
        proposed_values = pd.to_numeric(comparable[metric_p], errors="coerce").abs().dropna()
        baseline_values = pd.to_numeric(comparable[metric_b], errors="coerce").abs().dropna()
        if len(proposed_values) and len(baseline_values):
            proposed_median_abs = float(proposed_values.median())
            baseline_median_abs = float(baseline_values.median())
            baseline_floor = max(1.0e-8, 1.0e-3 * proposed_median_abs)
            if proposed_median_abs > 1.0e-6 and baseline_median_abs <= baseline_floor:
                return {
                    "mode": "advantage_over_benchmark",
                    "baseline_method": baseline_method,
                    "passes": False,
                    "proposed_win_rate": win_rate,
                    "proposed_median_relative_gain": median_gain,
                    "proposed_mean_relative_gain": mean_gain,
                    "baseline_median_abs": baseline_median_abs,
                    "proposed_median_abs": proposed_median_abs,
                    "baseline_degenerate": True,
                    "reason": "practical_benchmark_metric_is_degenerate_near_zero",
                }
    if _is_optimal_reference_method(plan, baseline_method):
        abs_gaps = gains.abs()
        max_gap = float(abs_gaps.max()) if len(abs_gaps) else float("inf")
        median_gap = float(abs_gaps.median()) if len(abs_gaps) else float("inf")
        pass_rate = float((abs_gaps <= 1.0e-4).mean()) if len(abs_gaps) else 0.0
        return {
            "mode": "optimal_reference_equivalence",
            "baseline_method": baseline_method,
            "passes": bool(median_gap <= 1.0e-5 and pass_rate >= 0.95),
            "median_relative_gap": median_gap,
            "max_relative_gap": max_gap,
            "within_tolerance_rate": pass_rate,
            "tolerance": 1.0e-4,
            "reason": "proposed_should_match_not_outperform_optimal_reference",
        }
    return {
        "mode": "advantage_over_benchmark",
        "baseline_method": baseline_method,
        "passes": bool(win_rate >= 0.55 and median_gain > 0.0),
        "proposed_win_rate": win_rate,
        "proposed_median_relative_gain": median_gain,
        "proposed_mean_relative_gain": mean_gain,
        "reason": "proposed_should_improve_over_practical_benchmark",
    }


def evaluate_figure_level_primary_claim_check(
    df: pd.DataFrame,
    plan: dict[str, Any],
    *,
    baseline_method: str = "",
) -> dict[str, Any]:
    """Evaluate the paper claim on aggregated figure curves.

    Monte Carlo wireless figures normally support claims through per-x-point
    averages over random channels, not by requiring the proposed method to win
    every individual realization.  The row-level win rate remains useful as a
    diagnostic, but the primary evidence should match what the paper plots.
    """
    baseline_method = str(baseline_method or plan.get("_active_baseline_method") or "").strip()
    primary_metric = str((plan.get("primary_metric") or {}).get("name") or "objective")
    plan_higher_is_better = bool((plan.get("primary_metric") or {}).get("higher_is_better", True))
    figure_checks: list[dict[str, Any]] = []

    for raw_fig in plan.get("figure_specs", []):
        if not isinstance(raw_fig, dict):
            continue
        fig = normalize_figure_spec(raw_fig, plan.get("primary_metric", {}))
        methods = _get_figure_methods(fig, _present_methods(df))
        local_baseline = baseline_method if baseline_method in methods else ""
        if not local_baseline:
            non_proposed = [method for method in methods if method != "proposed"]
            local_baseline = non_proposed[0] if non_proposed else ""
        if "proposed" not in methods or not local_baseline or local_baseline not in methods:
            continue
        try:
            curve_df = aggregate_for_figure(df, fig)
        except Exception as exc:
            figure_checks.append(
                {
                    "figure_id": str(fig.get("figure_id") or "figure"),
                    "intent": _figure_intent(fig),
                    "baseline_method": local_baseline,
                    "passes": False,
                    "reason": "figure_aggregation_failed",
                    "error": str(exc),
                }
            )
            continue
        if curve_df.empty or "mean_metric" not in curve_df.columns or "method" not in curve_df.columns:
            continue
        x_col = "x_value" if "x_value" in curve_df.columns else ("category" if "category" in curve_df.columns else "")
        if not x_col:
            continue
        proposed_curve = curve_df[curve_df["method"].astype(str) == "proposed"][[x_col, "mean_metric"]].copy()
        baseline_curve = curve_df[curve_df["method"].astype(str) == local_baseline][[x_col, "mean_metric"]].copy()
        if proposed_curve.empty or baseline_curve.empty:
            continue
        merged = proposed_curve.merge(baseline_curve, on=x_col, how="inner", suffixes=("_proposed", "_baseline"))
        if merged.empty:
            continue
        proposed_values = pd.to_numeric(merged["mean_metric_proposed"], errors="coerce")
        baseline_values = pd.to_numeric(merged["mean_metric_baseline"], errors="coerce")
        finite_mask = proposed_values.notna() & baseline_values.notna()
        proposed_values = proposed_values[finite_mask]
        baseline_values = baseline_values[finite_mask]
        if proposed_values.empty or baseline_values.empty:
            continue
        metric_info = fig.get("metric", {}) if isinstance(fig.get("metric"), dict) else {}
        higher_is_better = bool(metric_info.get("higher_is_better", plan_higher_is_better))
        signed_delta = proposed_values - baseline_values if higher_is_better else baseline_values - proposed_values
        proposed_scale = float(proposed_values.abs().median()) if len(proposed_values) else 0.0
        baseline_scale = baseline_values.abs().clip(lower=max(1.0e-8, 1.0e-3 * proposed_scale))
        relative_gain = signed_delta / baseline_scale
        tolerance = max(1.0e-8, 1.0e-4 * max(1.0, proposed_scale))
        win_fraction = float((signed_delta > tolerance).mean()) if len(signed_delta) else 0.0
        mean_delta = float(signed_delta.mean()) if len(signed_delta) else 0.0
        median_delta = float(signed_delta.median()) if len(signed_delta) else 0.0
        mean_relative_gain = float(relative_gain.mean()) if len(relative_gain) else 0.0
        median_relative_gain = float(relative_gain.median()) if len(relative_gain) else 0.0
        passes = bool(len(signed_delta) >= 2 and win_fraction >= 0.75 and mean_delta > tolerance and median_delta > 0.0)
        figure_checks.append(
            {
                "figure_id": str(fig.get("figure_id") or "figure"),
                "intent": _figure_intent(fig),
                "baseline_method": local_baseline,
                "metric": str(_get_figure_metric_name(fig) or primary_metric),
                "higher_is_better": higher_is_better,
                "passes": passes,
                "num_x_points": int(len(signed_delta)),
                "aggregate_win_fraction": win_fraction,
                "mean_signed_delta": mean_delta,
                "median_signed_delta": median_delta,
                "mean_relative_gain": mean_relative_gain,
                "median_relative_gain": median_relative_gain,
                "reason": "aggregated_figure_curve_advantage",
            }
        )

    if not figure_checks:
        return {
            "mode": "figure_level_aggregate_advantage",
            "baseline_method": baseline_method,
            "passes": False,
            "reason": "no_aggregated_figure_checks_available",
            "figure_checks": [],
        }

    primary_intents = {"main_comparison", "overall_utility", "utility_comparison"}
    primary_checks = [item for item in figure_checks if str(item.get("intent") or "").lower() in primary_intents]
    checks_for_decision = primary_checks or figure_checks
    return {
        "mode": "figure_level_aggregate_advantage",
        "baseline_method": baseline_method,
        "passes": bool(checks_for_decision and all(bool(item.get("passes")) for item in checks_for_decision)),
        "reason": "paper_claim_evaluated_on_aggregated_figure_curves",
        "decision_scope": "primary_comparison_figures" if primary_checks else "all_comparative_figures",
        "num_decision_figures": int(len(checks_for_decision)),
        "figure_checks": figure_checks,
    }


def run_monte_carlo_check(df: pd.DataFrame, plan: dict[str, Any], primary_metric: str, phase25_dir: Path) -> dict[str, Any]:
    seed_column = _detect_seed_column(df)
    figures: list[dict[str, Any]] = []
    for fig in plan.get("figure_specs", []):
        fig = normalize_figure_spec(fig, plan.get("primary_metric", {}))
        figure_id = fig.get("figure_id", "figure")
        chart_type = str(fig.get("chart_type", "line"))
        sweep_param = _get_figure_sweep_param(fig)
        methods_required = _get_figure_methods(fig)
        x_field = _get_figure_x_field(fig)
        figure_metric = _get_figure_metric_name(fig)
        checked_metric = figure_metric if figure_metric in df.columns else primary_metric
        subset = _filter_rows_for_figure(df.copy(), str(figure_id))
        subset = _filter_rows_for_required_sweep(subset, _get_figure_required_sweep(fig))
        if sweep_param and "swept_param" in subset.columns:
            subset = subset[subset["swept_param"] == sweep_param]
        if chart_type in {"grouped_bar", "bar", "box"} and x_field and x_field in subset.columns:
            pass
        elif x_field not in subset.columns and "swept_value" in subset.columns:
            x_field = "swept_value"
        rows: list[dict[str, Any]] = []
        warnings: list[str] = []
        if seed_column is None:
            warnings.append("missing_seed_column")
        if subset.empty:
            warnings.append("insufficient_comparable_cases")
        else:
            grouped = subset.groupby(["method", x_field], dropna=False) if x_field in subset.columns else subset.groupby(["method"], dropna=False)
            for keys, group in grouped:
                if isinstance(keys, tuple):
                    method, x_value = keys
                else:
                    method, x_value = keys, "all"
                metric_series = pd.to_numeric(group[checked_metric], errors="coerce") if checked_metric in group.columns else pd.Series(dtype=float)
                finite_mask = metric_series.notna() & metric_series.apply(lambda x: math.isfinite(float(x)))
                finite_values = metric_series[finite_mask].tolist()
                num_records = int(len(group))
                num_unique_seeds = int(group[seed_column].nunique()) if seed_column else 0
                num_unique_metric_values = int(pd.Series(finite_values).nunique()) if finite_values else 0
                metric_mean = float(pd.Series(finite_values).mean()) if finite_values else None
                metric_std = float(pd.Series(finite_values).std(ddof=1)) if len(finite_values) >= 2 else None
                metric_stderr = float((metric_std or 0.0) / math.sqrt(len(finite_values))) if len(finite_values) >= 2 and metric_std is not None else None
                feasible_rate = float(pd.to_numeric(group["feasible"], errors="coerce").fillna(0).astype(bool).mean()) if "feasible" in group.columns else 0.0
                finite_rate = float(len(finite_values) / max(len(group), 1))
                row_warnings: list[str] = []
                if seed_column is None:
                    row_warnings.append("missing_seed_column")
                elif num_unique_seeds < PAPER_MINIMUM_SEEDS:
                    row_warnings.append("too_few_seeds")
                if finite_rate < 1.0:
                    row_warnings.append("non_finite_metric")
                if num_unique_seeds > 1 and num_unique_metric_values == 1:
                    row_warnings.append("repeated_identical_outputs_across_seeds")
                if num_unique_metric_values == 1 and num_records >= 2:
                    row_warnings.append("zero_variance_across_all_seeds")
                rows.append(
                    {
                        "figure_id": figure_id,
                        "chart_type": chart_type,
                        "swept_param": sweep_param,
                        "x_field": x_field,
                        "x_value": _json_safe(x_value),
                        "method": str(method),
                        "num_records": num_records,
                        "num_unique_seeds": num_unique_seeds,
                        "num_unique_metric_values": num_unique_metric_values,
                        "metric_mean": metric_mean,
                        "metric_std": metric_std,
                        "metric_stderr": metric_stderr,
                        "feasible_rate": feasible_rate,
                        "finite_rate": finite_rate,
                        "warnings": row_warnings,
                    }
                )
            methods_present = sorted({str(r["method"]) for r in rows})
            for method in methods_required:
                if method not in methods_present:
                    warnings.append("missing_method")
        figures.append(
            {
                "figure_id": figure_id,
                "chart_type": chart_type,
                "required_sweep": _get_figure_required_sweep(fig),
                "swept_param": sweep_param,
                "primary_metric": primary_metric,
                "checked_metric": checked_metric,
                "seed_column": seed_column,
                "rows": rows,
                "warnings": sorted(set(warnings)),
            }
        )
    report = {"seed_column": seed_column, "unknown_seed_coverage": seed_column is None, "figures": figures}
    _write_json(phase25_dir / "monte_carlo_check.json", report)
    lines = ["# Monte Carlo Check", "", f"- seed_column: {seed_column or 'unknown'}", ""]
    for fig in figures:
        lines.append(f"## {fig['figure_id']}")
        lines.append(f"- swept_param: {fig['swept_param']}")
        lines.append(f"- warnings: {fig['warnings']}")
        lines.append("")
        for row in fig["rows"]:
            lines.append(
                f"- x={row['x_value']} method={row['method']} records={row['num_records']} "
                f"seeds={row['num_unique_seeds']} unique_metric_values={row['num_unique_metric_values']} "
                f"mean={row['metric_mean']} std={row['metric_std']} stderr={row['metric_stderr']} "
                f"feasible_rate={row['feasible_rate']} finite_rate={row['finite_rate']} warnings={row['warnings']}"
            )
        lines.append("")
    (phase25_dir / "monte_carlo_check.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def _suggest_sweep_values(existing_values: list[float], min_points: int, preferred_points: int) -> list[float]:
    existing = sorted({float(v) for v in existing_values if pd.notna(v)})
    target_count = max(min_points, preferred_points)
    if not existing:
        return [0.8, 1.0, 1.2, 1.4, 1.6, 1.8][:target_count]
    if len(existing) >= target_count:
        return existing[:target_count]
    if len(existing) == 1:
        base = existing[0]
        multipliers = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
        return sorted({round(base * m, 6) for m in multipliers})[:target_count]
    diffs = [existing[i + 1] - existing[i] for i in range(len(existing) - 1)]
    positive_diffs = [d for d in diffs if d > 0]
    step = statistics.median(positive_diffs) if positive_diffs else max(abs(existing[-1]) * 0.2, 1.0)
    values = list(existing)
    while len(values) < target_count:
        values.insert(0, round(values[0] - step, 6))
        if len(values) >= target_count:
            break
        values.append(round(values[-1] + step, 6))
    return sorted({float(v) for v in values})[:target_count]

def _densify_sweep_values_inside_span(existing_values: list[float], target_count: int) -> list[float]:
    """Return a dense numeric grid inside the already observed/reliable interval."""

    existing = sorted({float(v) for v in existing_values if pd.notna(v) and math.isfinite(float(v))})
    target_count = max(2, int(target_count))
    if len(existing) < 2:
        return existing
    lower = float(existing[0])
    upper = float(existing[-1])
    if upper <= lower:
        return existing[:target_count]
    return sorted({round(float(v), 6) for v in np.linspace(lower, upper, target_count).tolist()})


def _sanitize_suggested_values(param: str, values: list[float]) -> list[float]:
    param_norm = str(param).lower()
    cleaned = [float(v) for v in values if pd.notna(v)]
    if "dbm" in param_norm:
        return sorted({float(v) for v in cleaned})
    last_token = param_norm.replace("/", ".").split(".")[-1]
    if _is_count_like_sweep_param(param_norm) or param_norm in {"num_users", "num_antennas", "num_ris_elements"}:
        return sorted({max(1, int(round(v))) for v in cleaned})
    nonnegative_markers = (
        "lambda",
        "weight",
        "power",
        "pmax",
        "rate",
        "r_min",
        "rmin",
        "frequency",
        "fc",
        "noise",
        "sigma",
        "psi",
        "saturation",
        "_mw",
        ".mw",
        "energy",
        "illumination",
        "requirement",
        "threshold",
    )
    if "lambda" in param_norm or "weight" in param_norm:
        return sorted({float(v) for v in cleaned if v >= 0.0})
    if param_norm in {"delta", "delta_m", "delta_local_region", "pmax_dbm", "p_max_dbm", "power_budget", "transmit_power", "qos_rate_bps_hz", "rmin_bpshz", "r_min", "fc_ghz", "frequency", "fc"} or any(marker in param_norm for marker in nonnegative_markers):
        return sorted({float(v) for v in cleaned if v > 0.0})
    return sorted({float(v) for v in cleaned})


def _select_scout_values(values: list[float], chart_type: str, *, target_points: int | None = None) -> list[float]:
    numeric_values: list[float] = []
    for value in values:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            numeric_values.append(numeric)
    ordered = sorted(dict.fromkeys(numeric_values))
    if not ordered:
        return []
    chart = str(chart_type or "").strip().lower()
    if target_points is None:
        target_points = SCOUT_CATEGORY_POINTS if chart in {"grouped_bar", "bar", "categorical_summary", "ablation_bar", "box"} else SCOUT_LINE_POINTS
    target_points = max(2, min(int(target_points), len(ordered)))
    if len(ordered) <= target_points:
        return ordered
    if target_points == 2:
        return [ordered[0], ordered[-1]]
    selected_indexes = {
        int(round(idx))
        for idx in np.linspace(0, len(ordered) - 1, target_points).tolist()
    }
    return [ordered[idx] for idx in sorted(selected_indexes)]


def _phase25_quick_sweep_tier(*, quick: bool) -> str:
    if not quick:
        return "paper"
    raw_value = os.environ.get("WARA_PHASE25_SWEEP_TIER") or os.environ.get("WCL_PHASE25_SWEEP_TIER")
    value = str(raw_value or "scout").strip().lower()
    if value in {"medium", "mid", "moderate"}:
        return "medium"
    return "scout"


def _medium_target_points(chart_type: str) -> int:
    chart = str(chart_type or "").strip().lower()
    for env_name in ("WARA_PHASE25_MEDIUM_VALUES", "WCL_PHASE25_MEDIUM_VALUES"):
        raw_value = os.environ.get(env_name)
        if raw_value is None:
            continue
        try:
            return max(2, int(raw_value))
        except ValueError:
            continue
    if chart in {"grouped_bar", "bar", "categorical_summary", "ablation_bar", "box"}:
        return MEDIUM_CATEGORY_POINTS
    return MEDIUM_LINE_POINTS


def _medium_num_seeds(item: dict[str, Any]) -> int:
    for env_name in ("WARA_PHASE25_MEDIUM_SEEDS", "WCL_PHASE25_MEDIUM_SEEDS"):
        raw_value = os.environ.get(env_name)
        if raw_value is None:
            continue
        try:
            return max(1, int(raw_value))
        except ValueError:
            continue
    try:
        value = int(item.get("medium_num_seeds", 0) or 0)
    except (TypeError, ValueError):
        value = 0
    if value > 0:
        return value
    return MEDIUM_MODE_SEEDS


def _phase25_float_env(name: str, default: float) -> float:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except (TypeError, ValueError):
        return default


def _phase25_int_env(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        return default


def _phase25_runtime_observation_from_csv(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = [dict(row) for row in csv.DictReader(handle)]
    except Exception:
        return {}
    case_times: dict[tuple[str, str], float] = {}
    proposed_times: list[float] = []
    row_times: list[float] = []
    for row in rows:
        try:
            elapsed = float(row.get("solve_time_sec") or row.get("measured_solve_time_sec") or 0.0)
        except (TypeError, ValueError):
            elapsed = 0.0
        if not math.isfinite(elapsed) or elapsed < 0.0:
            continue
        row_times.append(elapsed)
        key = (str(row.get("case_id") or row.get("case_name") or ""), str(row.get("seed") or "0"))
        case_times[key] = case_times.get(key, 0.0) + elapsed
        if str(row.get("method") or "").strip().lower() == "proposed":
            proposed_times.append(elapsed)
    return {
        "source": str(path),
        "num_rows": len(rows),
        "num_cases": len(case_times),
        "median_row_time_sec": float(statistics.median(row_times)) if row_times else 0.0,
        "median_case_time_sec": float(statistics.median(case_times.values())) if case_times else 0.0,
        "median_proposed_time_sec": float(statistics.median(proposed_times)) if proposed_times else 0.0,
        "max_row_time_sec": float(max(row_times)) if row_times else 0.0,
        "max_case_time_sec": float(max(case_times.values())) if case_times else 0.0,
        "max_proposed_time_sec": float(max(proposed_times)) if proposed_times else 0.0,
    }


def _validation_plan_sweep_values(phase25_dir: Path, param: str, required_sweep: str = "") -> list[float]:
    """Return scalar paper-mode sweep values declared by Phase 2.4 for a parameter/sweep."""
    run_dir = Path(phase25_dir).parent

    def norm_name(value: str) -> str:
        text = str(value).strip().replace("/", ".")
        return text.split(".")[-1].lower()

    def sweep_names(sweep: dict[str, Any]) -> set[str]:
        return {
            str(sweep.get("id") or "").strip().lower(),
            str(sweep.get("name") or "").strip().lower(),
        } - {""}

    def sweep_matches_param(sweep: dict[str, Any]) -> bool:
        if not str(param or "").strip():
            return True
        candidate_names = [
            str(sweep.get("variable") or "").strip(),
            str(sweep.get("target") or "").strip(),
            str(sweep.get("canonical_path") or "").strip(),
        ]
        return any(candidate == str(param).strip() or norm_name(candidate) == target_norm for candidate in candidate_names)

    def extract_values(sweep: dict[str, Any]) -> list[float]:
        raw_values = []
        for key in ("paper_values", "paper_mode_values", "suggested_values", "paper_mode", "values"):
            raw_values = sweep.get(key, [])
            if raw_values:
                break
        if isinstance(raw_values, dict):
            raw_values = raw_values.get("values", [])
        if raw_values == "all_values":
            raw_values = sweep.get("values", [])
        if not isinstance(raw_values, list):
            return []
        values: list[float] = []
        for value in raw_values:
            if isinstance(value, (list, tuple, dict)):
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(numeric):
                values.append(numeric)
        return values

    target_norm = norm_name(param)
    required_norm = str(required_sweep or "").strip().lower()
    for plan_path in [
        run_dir / "phase2-4" / "validation_plan.yaml",
        run_dir / "phase2-4" / "solver" / "validation_plan.yaml",
    ]:
        if not plan_path.exists():
            continue
        try:
            payload = yaml.safe_load(plan_path.read_text(encoding="utf-8")) or {}
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        sweeps = payload.get("sweep_definitions", [])
        if not isinstance(sweeps, list):
            continue
        sweep_dicts = [sweep for sweep in sweeps if isinstance(sweep, dict)]
        if required_norm:
            for sweep in sweep_dicts:
                if required_norm not in sweep_names(sweep):
                    continue
                if not sweep_matches_param(sweep):
                    continue
                values = extract_values(sweep)
                if values:
                    return _sanitize_suggested_values(param, values)
        for sweep in sweep_dicts:
            if not sweep_matches_param(sweep):
                continue
            values = extract_values(sweep)
            if values:
                return _sanitize_suggested_values(param, values)
    return []


def _validation_plan_sweep_id_for_param(phase25_dir: Path, param: str, requested_sweep: str = "") -> str:
    run_dir = Path(phase25_dir).parent

    def norm_name(value: str) -> str:
        text = str(value).strip().replace("/", ".")
        return text.split(".")[-1].lower()

    target_norm = norm_name(param)
    requested_norm = str(requested_sweep or "").strip().lower()
    matches: list[str] = []
    for plan_path in [
        run_dir / "phase2-4" / "validation_plan.yaml",
        run_dir / "phase2-4" / "solver" / "validation_plan.yaml",
    ]:
        if not plan_path.exists():
            continue
        try:
            payload = yaml.safe_load(plan_path.read_text(encoding="utf-8")) or {}
        except Exception:
            continue
        sweeps = payload.get("sweep_definitions", []) if isinstance(payload, dict) else []
        if not isinstance(sweeps, list):
            continue
        for sweep in sweeps:
            if not isinstance(sweep, dict):
                continue
            candidate_names = [
                str(sweep.get("variable") or "").strip(),
                str(sweep.get("target") or "").strip(),
                str(sweep.get("canonical_path") or "").strip(),
            ]
            if any(candidate == str(param).strip() or norm_name(candidate) == target_norm for candidate in candidate_names):
                sweep_id = str(sweep.get("id") or sweep.get("name") or "").strip()
                if not sweep_id:
                    continue
                if requested_norm and sweep_id.lower() == requested_norm:
                    return sweep_id
                if sweep_id not in matches:
                    matches.append(sweep_id)
    if not matches:
        return ""
    if requested_norm:
        # A requested sweep id was supplied but does not match the supplied
        # parameter.  Only auto-correct when the parameter uniquely identifies
        # a single declared sweep; otherwise the plan is ambiguous and should
        # be blocked by the consistency gate rather than silently rewritten.
        return matches[0] if len(matches) == 1 else ""
    return matches[0] if len(matches) == 1 else ""


def _metric_name_to_default_label(metric_name: str, fallback: str = "Metric") -> str:
    metric = str(metric_name or "").strip()
    display_map = {
        "objective": "Objective value",
        "objective_value": "Objective value",
        "weighted_sum_rate_bpsHz": r"weighted sum-rate $U$ (bit/s/Hz)",
        "weighted_sum_rate_bps_hz": r"weighted sum-rate $U$ (bit/s/Hz)",
        "worst_case_utility": r"$\psi$",
        "rate_bpsHz": "Rate (bps/Hz)",
        "sum_rate_bpsHz": "Sum-rate (bps/Hz)",
        "R_c_bpsHz": "Rate (bps/Hz)",
        "R_sec_min_bpsHz": r"worst-user secrecy rate $R_{\mathrm{sec}}^{\min}$ (bps/Hz)",
        "R_sec_sum_bpsHz": r"sum secrecy rate $R_{\mathrm{sec}}^{\mathrm{sum}}$ (bps/Hz)",
        "spectral_efficiency": "Spectral efficiency (bps/Hz)",
        "sum_power_W": "Sum transmit power (W)",
        "sum_power_dBm": "Sum transmit power (dBm)",
        "sum_power_w": "Sum transmit power (W)",
        "sum_power_dbm": "Sum transmit power (dBm)",
        "total_power_W": "Total transmit power (W)",
        "total_power_dBm": "Total transmit power (dBm)",
        "total_power_dbm": "Total transmit power (dBm)",
        "spectral_radius_F": r"Spectral radius $\rho(F)$",
        "rho_F": r"Spectral radius $\rho(F)$",
        "radar_SNR_dB": "Radar SNR (dB)",
        "radar_snr_dB": "Radar SNR (dB)",
        "radar_snr": "Radar SNR",
        "sensing_gain": "Sensing gain",
        "sensing_metric": "Sensing metric",
        "sensing_beampattern_gain": "Sensing beampattern gain",
        "crb_trace": "tr(CRB)",
        "tr_CRB": "tr(CRB)",
        "harvested_power_mW": "Harvested power (mW)",
        "harvested_energy_mW": "Harvested energy (mW)",
        "true_harvested_energy_mW": "True harvested energy (mW)",
        "P_EH_mW": "Harvested power (mW)",
        "eh_total_mW": "Harvested power (mW)",
        "P_in_actual_mW": "EH input power (mW)",
        "optimal_rho": "Structural separation ratio",
        "rho": "Structural separation ratio",
        "feasible": "Feasibility",
        "feasibility_rate": "Feasibility rate",
        "constraint_violation_max": "Max. constraint violation",
        "max_constraint_violation": "Max. constraint violation",
    }
    if metric in display_map:
        return display_map[metric]
    if metric:
        return metric.replace("_", " ")
    return str(fallback or "Metric")


def _prefer_dbm_power_axis(figure_spec: dict[str, Any], metric_name: str, chart_type: str) -> bool:
    metric = str(metric_name or "").strip().lower()
    if metric not in {"sum_power_w", "total_power_w"}:
        return False
    if str(figure_spec.get("preserve_linear_power_axis", "")).strip().lower() in {"1", "true", "yes"}:
        return False
    intent = str(
        figure_spec.get("chart_intent")
        or figure_spec.get("intent")
        or figure_spec.get("claim_type")
        or ""
    ).strip().lower()
    if intent not in {"main_comparison", "overall_utility", "utility_comparison"}:
        return False
    return str(chart_type or "").strip().lower() in {"line", "scatter", "scatter_trend", "scatter_with_trend"}


def normalize_figure_spec(fig: dict[str, Any], primary_metric: dict[str, Any] | None = None) -> dict[str, Any]:
    spec = dict(fig)
    chart_type = str(spec.get("chart_type", "line")).strip() or "line"
    if "metric" not in spec:
        y_axis = spec.get("y_axis", {}) if isinstance(spec.get("y_axis"), dict) else {}
        top_level_metric = str(spec.get("y_metric") or spec.get("metric_name") or "").strip()
        metric_name = top_level_metric or y_axis.get("metric", (primary_metric or {}).get("name", "objective"))
        spec["metric"] = {
            "name": metric_name,
            "display_name": y_axis.get("display_name", _metric_name_to_default_label(metric_name, (primary_metric or {}).get("display_name", "Metric"))),
            "higher_is_better": _metric_higher_is_better(metric_name, bool((primary_metric or {}).get("higher_is_better", True))),
            "aggregation": y_axis.get("aggregation", "mean"),
        }
    metric_obj = spec.get("metric", {})
    if isinstance(metric_obj, dict):
        metric_name = str(metric_obj.get("name", "")).strip()
        default_metric_label = _metric_name_to_default_label(metric_name, (primary_metric or {}).get("display_name", "Metric"))
        current_metric_label = str(metric_obj.get("display_name", "")).strip()
        raw_metric_label = metric_name.replace("_", " ").strip().lower()
        if metric_name in {"weighted_sum_rate_bpsHz", "weighted_sum_rate_bps_hz"}:
            compact_label = re.sub(r"\s+", "", current_metric_label).lower()
            if (
                not current_metric_label
                or "bpshz" in compact_label
                or "\\sum" in current_metric_label
                or "mu_k" in compact_label
                or "\\mu" in current_metric_label
            ):
                current_metric_label = ""
        if metric_name == "worst_case_utility":
            compact_label = re.sub(r"\s+", "", current_metric_label)
            if "U_" in compact_label and "wc" in compact_label:
                current_metric_label = ""
        if (
            not current_metric_label
            or current_metric_label.strip().lower() in {metric_name.lower(), raw_metric_label}
        ):
            metric_obj["display_name"] = default_metric_label
        else:
            safe_metric_label = _safe_display(current_metric_label)
            if safe_metric_label != current_metric_label:
                metric_obj["display_name"] = safe_metric_label
        if "higher_is_better" not in metric_obj or _metric_direction_is_forced(metric_name):
            metric_obj["higher_is_better"] = _metric_higher_is_better(
                metric_name,
                bool((primary_metric or {}).get("higher_is_better", True)),
            )
        if _prefer_dbm_power_axis(spec, metric_name, chart_type):
            metric_obj["name"] = "sum_power_dBm" if metric_name.lower() == "sum_power_w" else "total_power_dBm"
            metric_obj["display_name"] = _metric_name_to_default_label(
                str(metric_obj["name"]),
                str(metric_obj.get("display_name") or "Power"),
            )
            metric_obj["higher_is_better"] = False
        spec["metric"] = metric_obj
    if "encoding" not in spec:
        x_axis = spec.get("x_axis", {}) if isinstance(spec.get("x_axis"), dict) else {}
        spec["encoding"] = {
            "x": {
                "type": "numeric",
                "field": x_axis.get("source", "swept_value"),
                "sweep_param": x_axis.get("sweep_param", ""),
                "display_name": x_axis.get("display_name", x_axis.get("sweep_param", "")),
            },
            "group": {
                "type": "method",
                "field": "method",
                "display_name": "Method",
            },
            "facet": {"type": "none", "field": None},
        }
    else:
        encoding = spec.get("encoding", {})
        if isinstance(encoding, dict):
            x_axis = encoding.get("x", {})
            if isinstance(x_axis, dict) and str(x_axis.get("display_name") or "").strip():
                x_axis["display_name"] = _safe_display(x_axis.get("display_name"))
                encoding["x"] = x_axis
                spec["encoding"] = encoding
    if "methods" not in spec:
        spec["methods"] = list(spec.get("curves", ["proposed"]))
    if "error_display" not in spec:
        spec["error_display"] = "none"
    if "data_requirements" not in spec:
        spec["data_requirements"] = {
            "min_points": int(spec.get("min_points", 6)),
            "preferred_points": int(spec.get("preferred_points", 8)),
            "min_samples_per_group": int(spec.get("min_seeds_per_point", PAPER_MINIMUM_SEEDS)),
            "preferred_samples_per_group": int(spec.get("preferred_seeds_per_point", PAPER_PREFERRED_SEEDS)),
        }
    x_encoding = spec.get("encoding", {}).get("x", {}) if isinstance(spec.get("encoding"), dict) else {}
    x_type = str(x_encoding.get("type", "")).strip().lower() if isinstance(x_encoding, dict) else ""
    x_field = str(x_encoding.get("field", "")).strip() if isinstance(x_encoding, dict) else ""
    sweep_param = str(x_encoding.get("sweep_param", "")).strip() if isinstance(x_encoding, dict) else ""
    numeric_sweep = bool(sweep_param) and (x_type in {"", "numeric", "continuous", "ordered"} or x_field == "swept_value")
    distribution_intents = {"distribution", "monte_carlo_distribution", "uncertainty_distribution"}
    scatter_intents = distribution_intents | {
        "regime",
        "operating_regime",
        "tradeoff",
        "gain_profile",
        "stochastic_trend",
        "noisy_trend",
    }
    if chart_type == "box" and numeric_sweep and _figure_intent(spec) not in distribution_intents:
        chart_type = "line"
        if str(spec.get("error_display", "")).strip().lower() in {"", "none", "iqr", "box"}:
            spec["error_display"] = "none"
    trend_guide_enabled = bool(spec.get("trend_guide")) or str(spec.get("display_mode", "")).strip().lower() in {
        "scatter_with_trend",
        "scatter_with_trend_guide",
    }
    if chart_type == "scatter" and numeric_sweep and _figure_intent(spec) not in scatter_intents and not trend_guide_enabled:
        chart_type = "line"
    spec["chart_type"] = chart_type
    return spec


def _get_figure_metric_name(figure_spec: dict[str, Any]) -> str:
    return str(figure_spec.get("metric", {}).get("name", "objective"))


def _get_figure_methods(figure_spec: dict[str, Any], fallback: list[str] | None = None) -> list[str]:
    methods = figure_spec.get("methods")
    if isinstance(methods, list) and methods:
        return [str(item) for item in methods]
    curves = figure_spec.get("curves")
    if isinstance(curves, list) and curves:
        return [str(item) for item in curves]
    return list(fallback or ["proposed"])


def _final_plotted_methods_by_figure(plan: dict[str, Any]) -> dict[str, list[str]]:
    """Return the methods that the final Phase-2.5 figures actually plot."""
    mapping: dict[str, list[str]] = {}
    if not isinstance(plan, dict):
        return mapping
    primary_metric = plan.get("primary_metric", {}) if isinstance(plan.get("primary_metric"), dict) else {}
    for raw_fig in plan.get("figure_specs", []):
        if not isinstance(raw_fig, dict):
            continue
        figure_id = str(raw_fig.get("figure_id") or raw_fig.get("id") or "").strip()
        if not figure_id:
            continue
        methods = [method for method in _get_figure_methods(normalize_figure_spec(raw_fig, primary_metric), []) if method]
        if "proposed" not in methods:
            methods.insert(0, "proposed")
        if methods:
            mapping[figure_id] = list(dict.fromkeys(str(method).strip() for method in methods if str(method).strip()))
    return mapping


def _get_figure_x_field(figure_spec: dict[str, Any]) -> str:
    encoding = figure_spec.get("encoding", {})
    x = encoding.get("x", {}) if isinstance(encoding, dict) else {}
    return str(x.get("field", "swept_value"))


def _get_figure_x_display(figure_spec: dict[str, Any]) -> str:
    encoding = figure_spec.get("encoding", {})
    x = encoding.get("x", {}) if isinstance(encoding, dict) else {}
    display_name = _safe_display(x.get("display_name", ""))
    sweep_param = str(x.get("sweep_param", "")).strip()
    param_labels = {
        "optimization.lambda1": r"Weight $\lambda_1$",
        "optimization.lambda2": r"Weight $\lambda_2$",
        "optimization.lambda3": r"EH weight $\lambda_3$",
        "system.M": r"Number of RIS elements $M$",
        "system.Nt": r"Number of BS antennas $N_t$",
        "system.Psi_sat": r"EH saturation power $\Psi_{\rm sat}$",
        "system.Pmax": r"Power budget $P_{\max}$",
        "system.Pmax_dBm": r"Power budget $P_{\max}$ (dBm)",
        "system.P_max_dBm": r"Power budget $P_{\max}$ (dBm)",
        "EH.steepness_a": r"EH sigmoid steepness $a$",
        "EH.a_steepness": r"EH sigmoid steepness $a$",
        "constraints.E_min_mW": r"EH requirement $E_{\min}$ (mW)",
        "optimization.lambda_s": r"Sensing weight $\lambda_s$",
        "optimization.lambda_c": r"Communication weight $\lambda_c$",
        "optimization.lambda_p": r"Powering weight $\lambda_p$",
        "constraints.sinr_target_dB": "SINR target (dB)",
        "constraints.uncertainty_radius": "Uncertainty radius",
    }
    if sweep_param in param_labels:
        return param_labels[sweep_param]
    raw_display = str(x.get("display_name", "")).strip()
    looks_internal = bool(re.search(r"\b(?:system|constraints|requirements|optimization|ambiguity|channel|rectifier)\.", raw_display))
    if display_name and not looks_internal and not any(marker in display_name for marker in ["\u9227", "\u754f", "�"]):
        return display_name
    return param_labels.get(sweep_param, _safe_display(x.get("field", "x")))


def _get_figure_sweep_param(figure_spec: dict[str, Any]) -> str:
    encoding = figure_spec.get("encoding", {})
    x = encoding.get("x", {}) if isinstance(encoding, dict) else {}
    return str(x.get("sweep_param", figure_spec.get("x_axis", {}).get("sweep_param", "")))


def _get_figure_required_sweep(figure_spec: dict[str, Any]) -> str:
    encoding = figure_spec.get("encoding", {})
    x = encoding.get("x", {}) if isinstance(encoding, dict) else {}
    return str(
        figure_spec.get("required_sweep")
        or figure_spec.get("required_sweep_id")
        or x.get("sweep_id")
        or ""
    ).strip()


def _looks_internal_axis_label(label: str) -> bool:
    text = str(label or "").strip()
    return bool(re.search(r"\b(?:system|constraints|requirements|optimization|ambiguity|channel|rectifier|uncertainty)\.", text))


def _phase25_sweep_display_map(run_dir: str | Path) -> dict[str, str]:
    run_dir = Path(run_dir)
    display_map: dict[str, str] = {}
    for plan_path in [
        run_dir / "phase2-4" / "validation_plan.yaml",
        run_dir / "phase2-4" / "solver" / "validation_plan.yaml",
    ]:
        if not plan_path.exists():
            continue
        try:
            payload = yaml.safe_load(plan_path.read_text(encoding="utf-8")) or {}
        except Exception:
            continue
        sweeps = payload.get("sweep_definitions", []) if isinstance(payload, dict) else []
        if not isinstance(sweeps, list):
            continue
        for sweep in sweeps:
            if not isinstance(sweep, dict):
                continue
            variable = str(
                sweep.get("display_name")
                or sweep.get("paper_symbol")
                or sweep.get("variable_symbol")
                or sweep.get("notation")
                or sweep.get("variable")
                or ""
            ).strip()
            if not variable or _looks_internal_axis_label(variable):
                continue
            for key in ("canonical_path", "id", "target", "variable"):
                value = str(sweep.get(key) or "").strip()
                if value:
                    display_map[value] = variable
    return display_map


def apply_phase25_sweep_display_names(plan: dict[str, Any], run_dir: str | Path) -> dict[str, Any]:
    payload = copy.deepcopy(plan)
    sweep_display = _phase25_sweep_display_map(run_dir)
    if not sweep_display:
        return payload
    repaired: list[dict[str, Any]] = []
    for fig in payload.get("figure_specs", []):
        if not isinstance(fig, dict):
            repaired.append(fig)
            continue
        normalized = normalize_figure_spec(fig, payload.get("primary_metric", {}))
        encoding = normalized.get("encoding", {}) if isinstance(normalized.get("encoding"), dict) else {}
        x_axis = encoding.get("x", {}) if isinstance(encoding.get("x"), dict) else {}
        sweep_param = str(x_axis.get("sweep_param") or "").strip()
        current_display = str(x_axis.get("display_name") or "").strip()
        current_compact = re.sub(r"[^a-z0-9]+", "_", current_display.lower()).strip("_")
        sweep_compact = re.sub(r"[^a-z0-9]+", "_", sweep_param.lower()).strip("_")
        current_is_schema_fallback = (
            "_" in current_display
            and "$" not in current_display
            and not any(char.isspace() for char in current_display)
        )
        current_is_schema_words = bool(
            current_compact
            and sweep_compact
            and "$" not in current_display
            and (current_compact == sweep_compact or current_compact in sweep_compact)
        )
        sweep_label = sweep_display.get(sweep_param, "")
        current_lost_subscript = bool(
            sweep_label
            and "_" in sweep_label
            and "$" not in current_display
            and current_display == sweep_label.replace("_", " ")
        )
        if sweep_param in sweep_display and (
            not current_display
            or current_display == sweep_param
            or current_is_schema_fallback
            or current_is_schema_words
            or current_lost_subscript
            or _looks_internal_axis_label(current_display)
            or any(marker in current_display for marker in ["\u9227", "\u754f", "�"])
        ):
            x_axis["display_name"] = _safe_display(sweep_display[sweep_param])
            encoding["x"] = x_axis
            normalized["encoding"] = encoding
        repaired.append(normalized)
    payload["figure_specs"] = repaired
    return payload


def _phase25_sweep_plan_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    items = payload.get("figures") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        items = payload.get("missing_for_figures") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict)]


def _phase25_observed_sweep_params_for_figure(df: pd.DataFrame, figure_id: str) -> set[str]:
    if df.empty or "swept_param" not in df.columns:
        return set()
    subset = _filter_rows_for_figure(df, figure_id)
    if subset.empty:
        return set()
    return {
        str(value).strip()
        for value in subset["swept_param"].dropna().astype(str).tolist()
        if str(value).strip()
    }


def align_phase25_plan_with_observed_sweep_plan(plan: dict[str, Any], phase25_dir: Path, df: pd.DataFrame) -> dict[str, Any]:
    """Keep figure specs aligned with the active/refined paper-sweep rows.

    The LLM sweep refiner may pivot a figure to a better x-axis. The paper
    sweep runner consumes that refined plan immediately, while a later analysis
    pass can otherwise reload stale figure specs from `experiment_plan.json` and
    incorrectly filter out the newly generated rows.
    """

    payload = copy.deepcopy(plan)
    candidate_items: list[dict[str, Any]] = []
    for plan_name in ("paper_sweep_plan_refined.json", "paper_sweep_plan.json"):
        candidate_items.extend(_phase25_sweep_plan_items(_read_json(phase25_dir / plan_name)))
    if not candidate_items:
        return payload
    by_figure: dict[str, list[dict[str, Any]]] = {}
    for item in candidate_items:
        figure_id = str(item.get("figure_id") or item.get("id") or "").strip()
        required_param = str(item.get("required_sweep_param") or item.get("sweep_param") or "").strip()
        if figure_id and required_param:
            by_figure.setdefault(figure_id, []).append(item)

    sweep_display = _phase25_sweep_display_map(phase25_dir.parent)
    aligned: list[dict[str, Any]] = []
    alignment_report: list[dict[str, Any]] = []
    for raw_fig in payload.get("figure_specs", []):
        if not isinstance(raw_fig, dict):
            aligned.append(raw_fig)
            continue
        fig = normalize_figure_spec(raw_fig, payload.get("primary_metric", {}))
        figure_id = str(fig.get("figure_id") or "").strip()
        observed_params = _phase25_observed_sweep_params_for_figure(df, figure_id)
        candidates = by_figure.get(figure_id, [])
        chosen = next(
            (
                item
                for item in candidates
                if str(item.get("required_sweep_param") or item.get("sweep_param") or "").strip() in observed_params
            ),
            None,
        )
        if chosen is None and len(observed_params) == 1:
            chosen = {"required_sweep_param": next(iter(observed_params))}
        if chosen is None:
            aligned.append(fig)
            continue
        required_param = str(chosen.get("required_sweep_param") or chosen.get("sweep_param") or "").strip()
        if not required_param:
            aligned.append(fig)
            continue
        old_param = _get_figure_sweep_param(fig)
        encoding = fig.get("encoding", {}) if isinstance(fig.get("encoding"), dict) else {}
        x_axis = encoding.get("x", {}) if isinstance(encoding.get("x"), dict) else {}
        x_axis["type"] = str(x_axis.get("type") or "numeric")
        x_axis["field"] = "swept_value"
        x_axis["sweep_param"] = required_param
        requested_sweep_id = str(chosen.get("required_sweep") or chosen.get("required_sweep_id") or "").strip()
        resolved_sweep_id = _validation_plan_sweep_id_for_param(phase25_dir, required_param, requested_sweep_id)
        if resolved_sweep_id:
            x_axis["sweep_id"] = resolved_sweep_id
        display = str(
            chosen.get("x_display_name")
            or chosen.get("display_name")
            or chosen.get("variable")
            or sweep_display.get(required_param)
            or required_param.split(".")[-1]
        ).strip()
        if display:
            x_axis["display_name"] = display
        encoding["x"] = x_axis
        fig["encoding"] = encoding
        if resolved_sweep_id:
            fig["required_sweep"] = resolved_sweep_id
        elif requested_sweep_id:
            fig["required_sweep"] = requested_sweep_id
        methods = [str(method).strip() for method in chosen.get("methods_to_run", []) if str(method).strip()]
        if methods:
            fig["methods"] = methods
        requirements = fig.get("data_requirements", {}) if isinstance(fig.get("data_requirements"), dict) else {}
        for source_key, target_key in (
            ("required_min_points", "min_points"),
            ("preferred_points", "preferred_points"),
            ("required_min_seeds_per_point", "min_samples_per_group"),
            ("preferred_seeds_per_point", "preferred_samples_per_group"),
        ):
            try:
                value = int(chosen.get(source_key))
            except (TypeError, ValueError):
                continue
            if value > 0:
                requirements[target_key] = value
        fig["data_requirements"] = requirements
        if chosen.get("claim_tested"):
            fig["claim"] = str(chosen.get("claim_tested"))
        alignment_report.append(
            {
                "figure_id": figure_id,
                "old_sweep_param": old_param,
                "new_sweep_param": required_param,
                "observed_sweep_params": sorted(observed_params),
                "source": "paper_sweep_plan",
            }
        )
        aligned.append(fig)
    payload["figure_specs"] = aligned
    if alignment_report:
        payload["_phase25_sweep_alignment"] = alignment_report
    return payload


def _figure_subset_for_selection(df: pd.DataFrame, figure_spec: dict[str, Any]) -> pd.DataFrame:
    fig = normalize_figure_spec(figure_spec)
    subset = _filter_rows_for_figure(df.copy(), str(fig.get("figure_id", "figure")))
    subset = _filter_rows_for_required_sweep(subset, _get_figure_required_sweep(fig))
    sweep_param = _get_figure_sweep_param(fig)
    if sweep_param and "swept_param" in subset.columns:
        filtered = subset[subset["swept_param"].astype(str) == str(sweep_param)].copy()
        if not filtered.empty:
            subset = filtered
    return subset


def _caption_blocks_from_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return {}
    blocks: dict[str, list[str]] = {}
    current_id = ""
    for line in text.splitlines():
        header = re.match(r"^##\s+(.+?)\s*$", line.strip())
        if header:
            current_id = header.group(1).strip()
            blocks.setdefault(current_id, [])
            continue
        if current_id:
            blocks[current_id].append(line.strip())
    return {key: " ".join(value).strip() for key, value in blocks.items()}


def _math_fragments_for_consistency(label: str) -> list[str]:
    text = str(label or "")
    fragments: list[str] = []
    for fragment in re.findall(r"\$([^$]+)\$", text):
        cleaned = re.sub(r"\s+", "", fragment)
        if cleaned and cleaned not in fragments:
            fragments.append(cleaned)
    return fragments


def _caption_contains_math_fragment(caption: str, fragment: str) -> bool:
    caption_compact = re.sub(r"\s+", "", str(caption or ""))
    return bool(fragment and fragment in caption_compact)


def _phase25_internal_path_leaks(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:system|constraints|requirements|optimization|ambiguity|channel|rectifier|uncertainty)\.[A-Za-z0-9_]+",
            str(text or ""),
        )
    )


def validate_phase25_contract_consistency(
    run_dir: str | Path,
    plan: dict[str, Any],
    df: pd.DataFrame,
    figure_outputs: list[dict[str, Any]] | None = None,
    caption_paths: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Validate that Phase 2.5 figures, rows, metadata, and captions agree.

    This gate is intentionally domain-agnostic: it does not know whether the
    paper is about SWIPT, UAVs, RIS, or another wireless optimization topic. It
    only checks that the active artifacts are internally consistent before a
    figure can be treated as paper-ready evidence.
    """

    run_dir = Path(run_dir)
    phase25_dir = run_dir / "phase2-5"
    errors: list[str] = []
    warnings: list[str] = []
    checks: list[dict[str, Any]] = []
    output_by_id = {
        str(item.get("figure_id") or ""): item
        for item in (figure_outputs or [])
        if isinstance(item, dict)
    }
    caption_file = Path(str((caption_paths or {}).get("figure_captions_path") or phase25_dir / "figure_captions.md"))
    captions = _caption_blocks_from_file(caption_file)

    for raw_fig in plan.get("figure_specs", []):
        if not isinstance(raw_fig, dict):
            continue
        fig = normalize_figure_spec(raw_fig, plan.get("primary_metric", {}))
        figure_id = str(fig.get("figure_id") or fig.get("id") or "").strip() or "figure"
        metric = _get_figure_metric_name(fig).strip()
        sweep_param = _get_figure_sweep_param(fig).strip()
        required_sweep = _get_figure_required_sweep(fig).strip()
        encoding = fig.get("encoding", {}) if isinstance(fig.get("encoding"), dict) else {}
        x_axis = encoding.get("x", {}) if isinstance(encoding.get("x"), dict) else {}
        declared_x_sweep_id = str(x_axis.get("sweep_id") or "").strip()
        x_label = _get_figure_x_display(fig)
        y_label = _safe_display(fig.get("metric", {}).get("display_name", metric))
        resolved_sweep = _validation_plan_sweep_id_for_param(
            phase25_dir,
            sweep_param,
            required_sweep or declared_x_sweep_id,
        )
        subset = _figure_subset_for_selection(df, fig)
        row_swept_params = sorted(
            {
                str(value).strip()
                for value in subset.get("swept_param", pd.Series(dtype=str)).dropna().astype(str).tolist()
                if str(value).strip()
            }
        ) if not subset.empty and "swept_param" in subset.columns else []
        row_required_sweeps = sorted(
            {
                str(value).strip()
                for value in subset.get("required_sweep", pd.Series(dtype=str)).dropna().astype(str).tolist()
                if str(value).strip()
            }
        ) if not subset.empty and "required_sweep" in subset.columns else []
        figure_errors: list[str] = []
        figure_warnings: list[str] = []

        if not sweep_param:
            figure_errors.append(f"{figure_id}: missing x-axis sweep_param in figure spec.")
        if not metric:
            figure_errors.append(f"{figure_id}: missing y-axis metric in figure spec.")
        if required_sweep and declared_x_sweep_id and required_sweep != declared_x_sweep_id:
            figure_errors.append(
                f"{figure_id}: required_sweep '{required_sweep}' disagrees with x.sweep_id '{declared_x_sweep_id}'."
            )
        if resolved_sweep:
            for field_name, value in (("required_sweep", required_sweep), ("x.sweep_id", declared_x_sweep_id)):
                if value and value != resolved_sweep:
                    figure_errors.append(
                        f"{figure_id}: {field_name} '{value}' is inconsistent with x.sweep_param "
                        f"'{sweep_param}' (declared sweep id is '{resolved_sweep}')."
                    )
        elif sweep_param:
            figure_warnings.append(
                f"{figure_id}: no unique declared validation-plan sweep id was found for x.sweep_param '{sweep_param}'."
            )
        if subset.empty:
            figure_errors.append(f"{figure_id}: no validation rows match the declared figure contract.")
        if row_swept_params and sweep_param and row_swept_params != [sweep_param]:
            figure_errors.append(
                f"{figure_id}: validation rows use swept_param values {row_swept_params}, "
                f"but the figure declares '{sweep_param}'."
            )
        if required_sweep and row_required_sweeps and row_required_sweeps != [required_sweep]:
            figure_errors.append(
                f"{figure_id}: validation rows use required_sweep values {row_required_sweeps}, "
                f"but the figure declares '{required_sweep}'."
            )
        if metric and metric not in df.columns:
            figure_errors.append(f"{figure_id}: metric '{metric}' is absent from validation results.")
        output_meta = output_by_id.get(figure_id, {})
        if output_meta:
            output_x = str(output_meta.get("x_axis_param") or "").strip()
            output_sweep = str(output_meta.get("required_sweep") or "").strip()
            output_metric = str(output_meta.get("y_metric") or "").strip()
            if output_x and sweep_param and output_x != sweep_param:
                figure_errors.append(
                    f"{figure_id}: rendered metadata x_axis_param '{output_x}' differs from figure spec '{sweep_param}'."
                )
            if output_sweep and required_sweep and output_sweep != required_sweep:
                figure_errors.append(
                    f"{figure_id}: rendered metadata required_sweep '{output_sweep}' differs from figure spec '{required_sweep}'."
                )
            if output_metric and metric and output_metric != metric:
                figure_errors.append(
                    f"{figure_id}: rendered metadata y_metric '{output_metric}' differs from figure spec '{metric}'."
                )
        else:
            figure_warnings.append(f"{figure_id}: rendered figure metadata is unavailable for consistency checking.")
        caption = captions.get(figure_id, "")
        if caption:
            if _phase25_internal_path_leaks(caption):
                figure_errors.append(f"{figure_id}: caption leaks an internal schema path instead of public paper notation.")
            for label_name, label in (("x-axis", x_label), ("y-axis", y_label)):
                if _phase25_internal_path_leaks(label):
                    figure_errors.append(f"{figure_id}: {label_name} label leaks an internal schema path: '{label}'.")
                for fragment in _math_fragments_for_consistency(label):
                    if not _caption_contains_math_fragment(caption, fragment):
                        figure_warnings.append(
                            f"{figure_id}: caption does not repeat the {label_name} notation '${fragment}$'."
                        )
        else:
            figure_warnings.append(f"{figure_id}: no caption block found for consistency checking.")

        errors.extend(figure_errors)
        warnings.extend(figure_warnings)
        checks.append(
            {
                "figure_id": figure_id,
                "ok": not figure_errors,
                "x_axis_param": sweep_param,
                "required_sweep": required_sweep,
                "resolved_validation_plan_sweep": resolved_sweep,
                "y_metric": metric,
                "row_swept_params": row_swept_params,
                "row_required_sweeps": row_required_sweeps,
                "caption_checked": bool(caption),
                "errors": figure_errors,
                "warnings": figure_warnings,
            }
        )

    return {
        "status": "passed" if not errors else "failed",
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "checks": checks,
    }


def _mean_for_figure_method(
    subset: pd.DataFrame,
    method_id: str,
    metric: str,
    *,
    figure_intent: str = "",
) -> float | None:
    if subset.empty or "method" not in subset.columns or metric not in subset.columns:
        return None
    rows = subset[subset["method"].astype(str) == str(method_id)]
    if rows.empty:
        return None
    rows = _successful_evidence_rows(rows, metric, figure_intent)
    if rows.empty:
        return None
    values = pd.to_numeric(rows[metric], errors="coerce").dropna()
    values = values[values.apply(lambda value: math.isfinite(float(value)))]
    if values.empty:
        return None
    return float(values.mean())


def _method_metric_degenerate_near_zero(
    subset: pd.DataFrame,
    *,
    proposed_method: str,
    benchmark_method: str,
    metric: str,
    higher_is_better: bool,
    figure_intent: str = "",
) -> dict[str, Any]:
    if (
        subset.empty
        or "method" not in subset.columns
        or metric not in subset.columns
        or not higher_is_better
        or _is_violation_or_feasibility_metric(metric)
    ):
        return {"degenerate": False}
    proposed_rows = _successful_evidence_rows(
        subset[subset["method"].astype(str) == str(proposed_method)],
        metric,
        figure_intent,
    )
    benchmark_rows = _successful_evidence_rows(
        subset[subset["method"].astype(str) == str(benchmark_method)],
        metric,
        figure_intent,
    )
    if proposed_rows.empty or benchmark_rows.empty:
        return {"degenerate": False}
    proposed_values = pd.to_numeric(proposed_rows[metric], errors="coerce").abs().dropna()
    benchmark_values = pd.to_numeric(benchmark_rows[metric], errors="coerce").abs().dropna()
    proposed_values = proposed_values[proposed_values.apply(lambda value: math.isfinite(float(value)))]
    benchmark_values = benchmark_values[benchmark_values.apply(lambda value: math.isfinite(float(value)))]
    if proposed_values.empty or benchmark_values.empty:
        return {"degenerate": False}
    proposed_median_abs = float(proposed_values.median())
    benchmark_median_abs = float(benchmark_values.median())
    proposed_mean_abs = float(proposed_values.mean())
    benchmark_mean_abs = float(benchmark_values.mean())
    proposed_reference = max(proposed_median_abs, proposed_mean_abs)
    benchmark_reference = max(benchmark_median_abs, benchmark_mean_abs)
    baseline_floor = max(1.0e-8, 1.0e-3 * proposed_reference)
    degenerate = proposed_reference > 1.0e-6 and benchmark_reference <= baseline_floor
    return {
        "degenerate": bool(degenerate),
        "proposed_median_abs": proposed_median_abs,
        "benchmark_median_abs": benchmark_median_abs,
        "proposed_mean_abs": proposed_mean_abs,
        "benchmark_mean_abs": benchmark_mean_abs,
        "baseline_floor": baseline_floor,
        "reason": "practical_benchmark_metric_is_degenerate_near_zero" if degenerate else "ok",
    }


def _select_single_benchmark_for_figure(df: pd.DataFrame, plan: dict[str, Any], figure_spec: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    """Choose all practical benchmarks that can support the displayed claim.

    Phase 2.4 can execute several candidate baselines. The final figure should
    not cherry-pick the weakest one: every executed, comparable, non-degenerate
    practical baseline for which the proposed method shows a positive gain is
    retained in the plotted method set.
    """
    fig = normalize_figure_spec(figure_spec, plan.get("primary_metric", {}))
    present = _present_methods(df)
    proposed_method = select_comparison_proposed_method(df, plan)
    requested = _get_figure_methods(fig, present)
    plan_methods = [
        _method_id_from_plan_item(method)
        for method in plan.get("compared_methods", [])
        if isinstance(method, dict)
    ]
    candidate_pool: list[str] = []
    for method in [*requested, *plan_methods, *present]:
        method = str(method or "").strip()
        if method and method not in candidate_pool:
            candidate_pool.append(method)
    candidates = [
        method
        for method in candidate_pool
        if method in present
        and method != proposed_method
        and not _method_is_ineligible_display_benchmark(plan, method)
    ]
    if not candidates:
        candidates = [method for method in present if method != proposed_method and not _method_is_ineligible_display_benchmark(plan, method)]
    practical = [
        method
        for method in candidates
        if not _is_optimal_reference_method(plan, method)
    ]
    candidates = practical or candidates
    if not candidates:
        return [proposed_method], {
            "selected_benchmark": "",
            "reason": "no_non_proposed_method_available",
            "candidate_scores": [],
        }

    metric_obj = fig.get("metric", {}) if isinstance(fig.get("metric"), dict) else {}
    metric = str(metric_obj.get("name") or plan.get("primary_metric", {}).get("name") or "objective")
    if metric not in df.columns:
        metric = str(plan.get("primary_metric", {}).get("name") or metric)
    higher_is_better = bool(metric_obj.get("higher_is_better", _metric_higher_is_better(metric, True)))
    subset = _figure_subset_for_selection(df, fig)
    figure_intent = _figure_intent(fig)
    x_field = _get_figure_x_field(fig)
    proposed_mean = _mean_for_figure_method(subset, proposed_method, metric, figure_intent=figure_intent)
    scores: list[dict[str, Any]] = []
    for order, method_id in enumerate(candidates):
        pair_methods = [proposed_method, method_id]
        pair_subset = subset[subset["method"].astype(str).isin(pair_methods)].copy() if "method" in subset.columns else subset
        baseline_mean = _mean_for_figure_method(pair_subset, method_id, metric, figure_intent=figure_intent)
        paired_success_points = _paired_success_x_points(
            pair_subset,
            x_field=x_field,
            methods_required=pair_methods,
            metric=metric,
            figure_intent=figure_intent,
        )
        paired_raw_points = _paired_method_x_points(pair_subset, x_field=x_field, methods_required=pair_methods)
        proposed_success_rate = _method_success_rate(pair_subset, proposed_method)
        benchmark_success_rate = _method_success_rate(pair_subset, method_id)
        degeneracy = _method_metric_degenerate_near_zero(
            pair_subset,
            proposed_method=proposed_method,
            benchmark_method=method_id,
            metric=metric,
            higher_is_better=higher_is_better,
            figure_intent=figure_intent,
        )
        pair_evidence = _successful_evidence_rows(pair_subset, metric, figure_intent)
        varies, variation = _metric_varies_across_sweep(pair_evidence, y_metric=metric, x_field=x_field)
        collapse = _curve_collapse_report(
            pair_evidence,
            method_id=proposed_method,
            metric=metric,
            x_field=x_field,
            figure_spec=fig,
            higher_is_better=higher_is_better,
        )
        if proposed_mean is None or baseline_mean is None:
            gain = None
            score = -10_000.0 - float(_method_display_priority(plan, method_id))
        else:
            gain = (proposed_mean - baseline_mean) if higher_is_better else (baseline_mean - proposed_mean)
            if degeneracy.get("degenerate"):
                score = -10_000.0 - float(_method_display_priority(plan, method_id)) - 0.001 * order
            else:
                denom = max(abs(float(baseline_mean)), 1.0e-9)
                relative_gain = float(gain / denom)
                score = 1000.0 * relative_gain - float(_method_display_priority(plan, method_id)) - 0.001 * order
                score += 10.0 * min(paired_success_points, 12)
                score += 8.0 if varies else -40.0
                score -= 80.0 if paired_success_points < 3 else 0.0
                score -= 80.0 if min(proposed_success_rate, benchmark_success_rate) < 0.8 else 0.0
                score -= 120.0 if collapse.get("collapse_detected") else 0.0
        scores.append(
            {
                "method": method_id,
                "metric": metric,
                "proposed_mean": proposed_mean,
                "benchmark_mean": baseline_mean,
                "gain": gain,
                "paired_success_points": paired_success_points,
                "paired_raw_points": paired_raw_points,
                "proposed_success_rate": proposed_success_rate,
                "benchmark_success_rate": benchmark_success_rate,
                "benchmark_degeneracy": degeneracy,
                "metric_variation": variation,
                "resource_sweep_collapse": collapse,
                "display_priority": _method_display_priority(plan, method_id),
                "score": score,
            }
        )
    scores.sort(key=lambda item: float(item.get("score", -10_000.0)), reverse=True)
    viable_scores = [
        item
        for item in scores
        if item.get("benchmark_mean") is not None
        and item.get("gain") is not None
        and float(item.get("gain") or 0.0) > 0.0
        and int(item.get("paired_success_points") or 0) >= 3
        and min(float(item.get("proposed_success_rate") or 0.0), float(item.get("benchmark_success_rate") or 0.0)) >= 0.8
        and not bool((item.get("resource_sweep_collapse") or {}).get("collapse_detected"))
        and not bool((item.get("benchmark_degeneracy") or {}).get("degenerate"))
    ]
    if viable_scores:
        def viable_display_key(item: dict[str, Any]) -> tuple[float, float, str]:
            priority = float(item.get("display_priority") or _method_display_priority(plan, str(item.get("method") or "")))
            baseline_mean = item.get("benchmark_mean")
            try:
                strength = float(baseline_mean)
            except (TypeError, ValueError):
                strength = 0.0
            # Stronger practical baselines should appear before weaker ones in
            # the legend.  For higher-is-better metrics this means larger
            # baseline values; for lower-is-better metrics this means smaller
            # baseline values.
            strength_key = -strength if higher_is_better else strength
            return (priority, strength_key, str(item.get("method") or ""))

        ranked_scores = sorted(viable_scores, key=viable_display_key)
    else:
        ranked_scores = scores
    selected_methods = [
        str(item.get("method") or "")
        for item in ranked_scores
        if str(item.get("method") or "").strip()
    ]
    if not viable_scores:
        selected_methods = selected_methods[:1]
    selected_methods = list(dict.fromkeys(method for method in selected_methods if method != proposed_method))
    selected = selected_methods[0] if selected_methods else ""
    return [proposed_method, *selected_methods], {
        "selected_benchmark": selected,
        "selected_benchmarks": selected_methods,
        "reason": "all_viable_gain_supporting_benchmarks_for_current_figure"
        if viable_scores
        else "best_available_benchmark_for_current_figure_no_fully_viable_gain_support",
        "candidate_scores": scores,
        "viable_candidate_methods": [str(item.get("method") or "") for item in viable_scores],
    }


def _metric_relative_span(subset: pd.DataFrame, metric: str, x_field: str, method_id: str = "proposed") -> tuple[float, float, int]:
    if subset.empty or metric not in subset.columns or x_field not in subset.columns:
        return 0.0, 0.0, 0
    rows = subset
    if "method" in rows.columns and method_id:
        method_rows = rows[rows["method"].astype(str) == str(method_id)]
        if not method_rows.empty:
            rows = method_rows
    grouped = (
        rows.groupby(x_field, dropna=False)[metric]
        .apply(lambda s: pd.to_numeric(s, errors="coerce").dropna().mean())
        .dropna()
    )
    if len(grouped) < 2:
        return 0.0, 0.0, int(len(grouped))
    values = grouped.to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return 0.0, 0.0, int(values.size)
    span = float(np.nanmax(values) - np.nanmin(values))
    scale = max(float(np.nanmax(np.abs(values))), 1.0e-9)
    return span, float(abs(span) / scale), int(len(values))


def _mechanism_metric_priority(metric: str, figure_spec: dict[str, Any]) -> float:
    text = " ".join(
        str(figure_spec.get(key) or "")
        for key in ("figure_id", "purpose", "primary_message", "chart_intent", "required_sweep")
    ).lower()
    metric_lower = str(metric or "").lower()
    score = 0.0
    if "eh" in text or "harvest" in text or "energy requirement" in text:
        if "harvest" in metric_lower:
            score += 45.0
        if "eh_requirement_margin" in metric_lower or "margin" in metric_lower:
            score += 35.0
        if "rho" in metric_lower or "splitting" in metric_lower:
            score += 25.0
    if "feasib" in text or "constraint" in text or "boundary" in text:
        if "violation" in metric_lower or "margin" in metric_lower:
            score += 35.0
        if "feasible" in metric_lower:
            score += 20.0
    if "power" in text:
        if "power" in metric_lower:
            score += 25.0
    if "rate" in text or "throughput" in text:
        if "rate" in metric_lower or "throughput" in metric_lower:
            score += 15.0
    if _is_objective_metric(metric):
        score -= 40.0
    if _is_runtime_metric(metric):
        score -= 80.0
    if str(metric).startswith("actual_used"):
        score -= 60.0
    return score


def _select_mechanism_metric_from_data(df: pd.DataFrame, plan: dict[str, Any], figure_spec: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    fig = normalize_figure_spec(figure_spec, plan.get("primary_metric", {}))
    subset = _figure_subset_for_selection(df, fig)
    if subset.empty:
        return "", {"reason": "empty_figure_subset"}
    x_field = _get_figure_x_field(fig)
    if x_field not in subset.columns and "swept_value" in subset.columns:
        x_field = "swept_value"
    if x_field not in subset.columns:
        return "", {"reason": "missing_x_field", "x_field": x_field}
    proposed_id = select_comparison_proposed_method(df, plan)
    current_metric = _get_figure_metric_name(fig)
    candidates: list[dict[str, Any]] = []
    ignored = set(NON_METRIC_COLUMNS) | set(TABLE_METADATA_COLUMNS) | {"swept_value"}
    for column in subset.columns:
        if column in ignored or column.startswith("system_") or column.startswith("constraints_") or column.startswith("optimization_"):
            continue
        if column.startswith("actual_used") or column.endswith("_actual_used"):
            continue
        series = pd.to_numeric(subset[column], errors="coerce")
        if series.dropna().empty:
            continue
        span, relative_span, points = _metric_relative_span(subset, column, x_field, proposed_id)
        if points < 3 or relative_span < 5.0e-3:
            continue
        varies, variation = _metric_varies_across_sweep(subset, y_metric=column, x_field=x_field)
        if not varies:
            continue
        priority = _mechanism_metric_priority(column, fig)
        if column == current_metric:
            priority -= 5.0
        candidates.append(
            {
                "metric": column,
                "span": span,
                "relative_span": relative_span,
                "points": points,
                "priority": priority,
                "variation": variation,
                "score": priority + 100.0 * min(relative_span, 1.0),
            }
        )
    candidates.sort(key=lambda item: float(item.get("score", -1.0e9)), reverse=True)
    if not candidates:
        return "", {"reason": "no_responsive_physical_metric", "current_metric": current_metric}
    selected = str(candidates[0]["metric"])
    return selected, {
        "reason": "selected_responsive_mechanism_metric_from_executed_data",
        "current_metric": current_metric,
        "selected_metric": selected,
        "candidates": candidates[:8],
    }


def repair_mechanism_figure_metrics_from_data(plan: dict[str, Any], df: pd.DataFrame) -> dict[str, Any]:
    if not isinstance(plan, dict) or df.empty:
        return plan
    payload = copy.deepcopy(plan)
    repaired: list[dict[str, Any]] = []
    figures = payload.get("figure_specs", [])
    if not isinstance(figures, list):
        return payload
    for fig in figures:
        if not isinstance(fig, dict):
            repaired.append(fig)
            continue
        normalized = normalize_figure_spec(fig, payload.get("primary_metric", {}))
        intent = _figure_intent(normalized)
        current_metric = _get_figure_metric_name(normalized)
        subset = _figure_subset_for_selection(df, normalized)
        x_field = _get_figure_x_field(normalized)
        current_span, current_relative_span, _ = _metric_relative_span(
            subset,
            current_metric,
            x_field if x_field in subset.columns else "swept_value",
            select_comparison_proposed_method(df, payload),
        )
        weak_current = current_relative_span < 5.0e-3
        if intent in MECHANISM_INTENTS and (_is_objective_metric(current_metric) or weak_current):
            selected, selection = _select_mechanism_metric_from_data(df, payload, normalized)
            normalized["phase25_metric_repair"] = {
                **selection,
                "current_relative_span": current_relative_span,
                "current_span": current_span,
            }
            if selected and selected != current_metric:
                metric_obj = normalized.get("metric", {}) if isinstance(normalized.get("metric"), dict) else {}
                metric_obj["name"] = selected
                metric_obj["display_name"] = _metric_name_to_default_label(selected, selected)
                metric_obj["higher_is_better"] = _metric_higher_is_better(selected, bool(metric_obj.get("higher_is_better", True)))
                metric_obj["aggregation"] = metric_obj.get("aggregation", "mean")
                normalized["metric"] = metric_obj
                normalized["primary_message"] = (
                    f"{normalized.get('primary_message') or normalized.get('purpose') or ''} "
                    f"Metric repaired from `{current_metric}` to `{selected}` because the original y-axis was not responsive in the executed pilot data."
                ).strip()
        repaired.append(normalized)
    payload["figure_specs"] = repaired
    return payload


def select_single_benchmark_methods_for_figures(plan: dict[str, Any], df: pd.DataFrame) -> dict[str, Any]:
    if not isinstance(plan, dict) or df.empty or "method" not in df.columns:
        return plan
    payload = copy.deepcopy(plan)
    selected_figures: list[dict[str, Any]] = []
    figure_scores: list[tuple[float, int, dict[str, Any]]] = []
    for idx, fig in enumerate(payload.get("figure_specs", [])):
        if not isinstance(fig, dict):
            continue
        normalized = normalize_figure_spec(fig, payload.get("primary_metric", {}))
        methods, selection = _select_single_benchmark_for_figure(df, payload, normalized)
        normalized["methods"] = methods
        normalized["benchmark_selection"] = selection
        metric = _get_figure_metric_name(normalized)
        subset = _figure_subset_for_selection(df, normalized)
        if methods:
            subset = subset[subset["method"].astype(str).isin(methods)] if "method" in subset.columns else subset
        x_field = _get_figure_x_field(normalized)
        varies = False
        variation = {"reason": "not_evaluated"}
        if metric in subset.columns and x_field in subset.columns:
            varies, variation = _metric_varies_across_sweep(subset, y_metric=metric, x_field=x_field)
        selected_benchmark = str(selection.get("selected_benchmark") or "")
        best_gain = 0.0
        paired_success_points = 0
        min_success_rate = 0.0
        collapse_detected = False
        scores = selection.get("candidate_scores", [])
        if isinstance(scores, list) and scores:
            raw_gain = scores[0].get("gain")
            if isinstance(raw_gain, (int, float)) and math.isfinite(float(raw_gain)):
                best_gain = float(raw_gain)
            paired_success_points = int(scores[0].get("paired_success_points") or 0)
            min_success_rate = float(
                min(
                    float(scores[0].get("proposed_success_rate") or 0.0),
                    float(scores[0].get("benchmark_success_rate") or 0.0),
                )
            )
            collapse_obj = scores[0].get("resource_sweep_collapse", {})
            collapse_detected = bool(collapse_obj.get("collapse_detected")) if isinstance(collapse_obj, dict) else False
        normalized["semantic_metric_variation_precheck"] = variation
        score = 0.0
        score += 30.0 if selected_benchmark else -30.0
        score += 25.0 if best_gain > 0.0 else 0.0
        score += 15.0 if varies else -10.0
        score += min(20.0, 4.0 * paired_success_points)
        score += 10.0 if min_success_rate >= 0.95 else -30.0
        score -= 35.0 if paired_success_points < 3 else 0.0
        score -= 45.0 if collapse_detected else 0.0
        if str(normalized.get("chart_intent") or "").lower() == "main_comparison":
            score += 10.0
        if _is_diagnostic_primary_metric(metric) and str(normalized.get("chart_intent") or "").lower() == "main_comparison":
            score -= 20.0
        normalized["phase25_evidence_screen"] = {
            "score": score,
            "best_gain": best_gain,
            "paired_success_points": paired_success_points,
            "min_method_success_rate": min_success_rate,
            "resource_sweep_collapse_detected": collapse_detected,
        }
        figure_scores.append((score, -idx, normalized))
    figure_scores.sort(reverse=True, key=lambda item: (item[0], item[1]))
    usable_scores = [item for item in figure_scores if item[0] > 0.0]
    if len(usable_scores) >= 2:
        selected_pool = usable_scores[:3]
    else:
        selected_pool = figure_scores[: min(3, len(figure_scores))]
    selected_figures = [item[2] for item in selected_pool]
    if len(selected_figures) > 2 and selected_pool[2][0] <= 0.0:
        selected_figures = selected_figures[:2]
    if len(selected_figures) >= 2:
        common_scores: dict[str, dict[str, Any]] = {}
        for fig in selected_figures:
            selection = fig.get("benchmark_selection", {}) if isinstance(fig.get("benchmark_selection"), dict) else {}
            for item in selection.get("candidate_scores", []):
                if not isinstance(item, dict):
                    continue
                method_id = str(item.get("method") or "").strip()
                if not method_id or _method_is_ineligible_display_benchmark(payload, method_id):
                    continue
                if item.get("benchmark_mean") is None or int(item.get("paired_success_points") or 0) < 3:
                    continue
                if bool((item.get("benchmark_degeneracy") or {}).get("degenerate")):
                    continue
                entry = common_scores.setdefault(method_id, {"count": 0, "score": 0.0})
                entry["count"] += 1
                entry["score"] += float(item.get("score") or 0.0)
        common_candidates = [
            (int(entry["count"]), float(entry["score"]), method_id)
            for method_id, entry in common_scores.items()
            if int(entry.get("count") or 0) >= min(2, len(selected_figures))
        ]
        used_fallback_common = False
        if not common_candidates:
            used_fallback_common = True
            fallback_scores: dict[str, dict[str, Any]] = {}
            for fig in selected_figures:
                selection = fig.get("benchmark_selection", {}) if isinstance(fig.get("benchmark_selection"), dict) else {}
                for item in selection.get("candidate_scores", []):
                    if not isinstance(item, dict):
                        continue
                    method_id = str(item.get("method") or "").strip()
                    if not method_id or _method_is_ineligible_display_benchmark(payload, method_id):
                        continue
                    if item.get("benchmark_mean") is None or int(item.get("paired_success_points") or 0) < 3:
                        continue
                    entry = fallback_scores.setdefault(method_id, {"count": 0, "score": 0.0})
                    entry["count"] += 1
                    entry["score"] += float(item.get("score") or 0.0)
            common_candidates = [
                (int(entry["count"]), float(entry["score"]), method_id)
                for method_id, entry in fallback_scores.items()
                if int(entry.get("count") or 0) >= min(2, len(selected_figures))
            ]
        if common_candidates:
            common_candidates.sort(reverse=True)
            common_methods = [method_id for _count, _score, method_id in common_candidates]
            for fig in selected_figures:
                selection = fig.get("benchmark_selection", {}) if isinstance(fig.get("benchmark_selection"), dict) else {}
                selection["common_supporting_benchmarks"] = common_methods
                selection["common_benchmark_alignment"] = {
                    "enabled": True,
                    "reason": "same_viable_benchmark_set_recorded_across_paper_figures"
                    if not used_fallback_common
                    else "same_best_available_benchmark_set_recorded_across_paper_figures",
                    "does_not_drop_figure_specific_supported_baselines": True,
                }
                fig["benchmark_selection"] = selection
    if selected_figures:
        payload["figure_specs"] = selected_figures
        payload["_phase25_figure_selection"] = [
            {
                "figure_id": str(fig.get("figure_id") or fig.get("id") or ""),
                "methods": fig.get("methods", []),
                "benchmark_selection": fig.get("benchmark_selection", {}),
                "semantic_metric_variation_precheck": fig.get("semantic_metric_variation_precheck", {}),
                "phase25_evidence_screen": fig.get("phase25_evidence_screen", {}),
            }
            for fig in selected_figures
        ]
    payload["table_specs"] = []
    return payload


def _line_like_chart(chart_type: str) -> bool:
    return chart_type in {"line", "scatter", "scatter_trend", "scatter_with_trend", "convergence"}


def _figure_row_mask(df: pd.DataFrame, figure_ids: list[str]) -> pd.Series | None:
    clean_ids = [str(figure_id).strip() for figure_id in figure_ids if str(figure_id).strip()]
    if df.empty or not clean_ids:
        return None
    mask: pd.Series | None = None
    for column in ("figure_id", "case_id", "case_name"):
        if column not in df.columns:
            continue
        series = df[column].astype(str)
        column_mask = pd.Series(False, index=df.index)
        for figure_id in clean_ids:
            if column == "figure_id":
                column_mask = column_mask | series.eq(figure_id)
            else:
                column_mask = column_mask | series.eq(figure_id) | series.str.startswith(f"{figure_id}_")
        mask = column_mask if mask is None else mask | column_mask
    return mask


def _filter_rows_for_figure(df: pd.DataFrame, figure_id: str) -> pd.DataFrame:
    mask = _figure_row_mask(df, [figure_id])
    if mask is not None and bool(mask.any()):
        return df[mask].copy()
    return df


def _filter_rows_for_required_sweep(df: pd.DataFrame, required_sweep: str) -> pd.DataFrame:
    sweep_id = str(required_sweep or "").strip().lower()
    if df.empty or not sweep_id:
        return df
    mask: pd.Series | None = None
    for column in ("required_sweep", "sweep_id", "case_id", "case_name", "scenario_name"):
        if column not in df.columns:
            continue
        series = df[column].astype(str).str.lower()
        column_mask = (
            series.eq(sweep_id)
            | series.str.startswith(f"{sweep_id}_")
            | series.str.contains(sweep_id, regex=False)
        )
        mask = column_mask if mask is None else mask | column_mask
    if mask is not None and bool(mask.any()):
        return df[mask].copy()
    return df


def _success_mask(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(False, index=df.index)
    mask = pd.Series(True, index=df.index)
    if "status" in df.columns:
        mask = mask & ~df["status"].astype(str).str.lower().isin(FAIL_STATUSES)
    elif "success" in df.columns:
        mask = mask & df["success"].astype(bool)
    elif "feasible" in df.columns:
        mask = mask & df["feasible"].astype(bool)
    return mask


def _strict_feasible_success_mask(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(False, index=df.index)
    mask = _success_mask(df)
    if "feasible" in df.columns:
        mask = mask & df["feasible"].astype(bool)
    if "success" in df.columns and "status" not in df.columns:
        mask = mask & df["success"].astype(bool)
    return mask


def _metric_needs_success_rows(metric: str, figure_intent: str = "") -> bool:
    lowered = str(metric or "").strip().lower()
    intent = str(figure_intent or "").strip().lower()
    if any(token in lowered for token in ("feasible", "feasibility", "violation", "outage", "infeasible")):
        return False
    if intent in {"feasibility_boundary"}:
        return False
    return True


def _successful_evidence_rows(subset: pd.DataFrame, metric: str, figure_intent: str = "") -> pd.DataFrame:
    if subset.empty or not _metric_needs_success_rows(metric, figure_intent):
        return subset
    strict_mask = _strict_feasible_success_mask(subset)
    if bool(strict_mask.any()):
        return subset[strict_mask].copy()
    return subset[_success_mask(subset)].copy()


def _paired_success_x_points(
    subset: pd.DataFrame,
    *,
    x_field: str,
    methods_required: list[str],
    metric: str,
    figure_intent: str = "",
) -> int:
    if subset.empty or x_field not in subset.columns or "method" not in subset.columns or metric not in subset.columns:
        return 0
    evidence = _successful_evidence_rows(subset, metric, figure_intent)
    evidence = evidence[pd.to_numeric(evidence[metric], errors="coerce").notna()].copy()
    return _paired_method_x_points(evidence, x_field=x_field, methods_required=methods_required)


def _method_success_rate(subset: pd.DataFrame, method_id: str) -> float:
    if subset.empty or "method" not in subset.columns:
        return 0.0
    rows = subset[subset["method"].astype(str) == str(method_id)]
    if rows.empty:
        return 0.0
    return float(_success_mask(rows).mean())


def _resource_sweep_expects_non_decreasing(figure_spec: dict[str, Any], metric: str, higher_is_better: bool) -> bool:
    if not higher_is_better:
        return False
    metric_lower = str(metric or "").strip().lower()
    if any(token in metric_lower for token in ("violation", "gap", "runtime", "time", "iteration", "power_consumption")):
        return False
    text = " ".join(
        str(value or "")
        for value in (
            _get_figure_sweep_param(figure_spec),
            _get_figure_x_display(figure_spec),
            _get_figure_required_sweep(figure_spec),
            figure_spec.get("purpose", ""),
            figure_spec.get("primary_message", ""),
        )
    ).lower()
    resource_tokens = (
        "pmax",
        "power budget",
        "budget",
        "snr",
        "antenna",
        "bandwidth",
        "blocklength",
        "resource",
    )
    return any(token in text for token in resource_tokens)


def _curve_collapse_report(
    subset: pd.DataFrame,
    *,
    method_id: str,
    metric: str,
    x_field: str,
    figure_spec: dict[str, Any],
    higher_is_better: bool,
) -> dict[str, Any]:
    if (
        subset.empty
        or "method" not in subset.columns
        or metric not in subset.columns
        or x_field not in subset.columns
        or not _resource_sweep_expects_non_decreasing(figure_spec, metric, higher_is_better)
    ):
        return {"collapse_detected": False, "reason": "not_applicable"}
    rows = _successful_evidence_rows(subset[subset["method"].astype(str) == str(method_id)].copy(), metric, _figure_intent(figure_spec))
    if rows.empty:
        return {"collapse_detected": False, "reason": "no_successful_rows"}
    grouped = (
        rows.groupby(x_field, dropna=False)[metric]
        .apply(lambda s: pd.to_numeric(s, errors="coerce").dropna().mean())
        .dropna()
    )
    if len(grouped) < 4:
        return {"collapse_detected": False, "reason": "too_few_points"}
    grouped = grouped.sort_index()
    x_values = [float(x) for x in grouped.index.tolist()]
    values = grouped.to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        return {"collapse_detected": False, "reason": "non_finite_values"}
    drops = []
    for idx in range(1, len(values)):
        prev = float(values[idx - 1])
        curr = float(values[idx])
        if curr < prev:
            drops.append((x_values[idx - 1], x_values[idx], prev, curr, (prev - curr) / max(abs(prev), 1.0e-9)))
    worst = max(drops, key=lambda item: item[4], default=None)
    collapse = bool(worst and worst[4] >= 0.35)
    return {
        "collapse_detected": collapse,
        "worst_drop_ratio": float(worst[4]) if worst else 0.0,
        "from_x": float(worst[0]) if worst else None,
        "to_x": float(worst[1]) if worst else None,
        "from_value": float(worst[2]) if worst else None,
        "to_value": float(worst[3]) if worst else None,
    }


def _expected_non_decreasing_trend_report(
    subset: pd.DataFrame,
    *,
    method_id: str,
    metric: str,
    x_field: str,
    figure_spec: dict[str, Any],
    higher_is_better: bool,
) -> dict[str, Any]:
    if (
        subset.empty
        or "method" not in subset.columns
        or metric not in subset.columns
        or x_field not in subset.columns
        or not _resource_sweep_expects_non_decreasing(figure_spec, metric, higher_is_better)
    ):
        return {"expected_non_decreasing": False, "violation": False, "reason": "not_applicable"}
    rows = _successful_evidence_rows(subset[subset["method"].astype(str) == str(method_id)].copy(), metric, _figure_intent(figure_spec))
    if rows.empty:
        return {"expected_non_decreasing": True, "violation": False, "reason": "no_successful_rows"}
    grouped = (
        rows.groupby(x_field, dropna=False)[metric]
        .apply(lambda s: pd.to_numeric(s, errors="coerce").dropna().mean())
        .dropna()
    )
    if len(grouped) < 2:
        return {"expected_non_decreasing": True, "violation": False, "reason": "too_few_points"}
    grouped = grouped.sort_index()
    x_values = [float(x) for x in grouped.index.tolist()]
    values = grouped.to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        return {"expected_non_decreasing": True, "violation": False, "reason": "non_finite_values"}
    drops: list[tuple[float, float, float, float, float]] = []
    for idx in range(1, len(values)):
        prev = float(values[idx - 1])
        curr = float(values[idx])
        drop_ratio = (prev - curr) / max(abs(prev), 1.0e-9)
        if drop_ratio > 0:
            drops.append((x_values[idx - 1], x_values[idx], prev, curr, drop_ratio))
    worst = max(drops, key=lambda item: item[4], default=None)
    first = float(values[0])
    last = float(values[-1])
    first_last_change_ratio = (last - first) / max(abs(first), 1.0e-9)
    # Small dips can be Monte-Carlo noise. A visible negative trend in a resource
    # sweep is different: it conflicts with the declared experiment hypothesis.
    violation = bool((worst and worst[4] >= 0.05) or first_last_change_ratio <= -0.05)
    return {
        "expected_non_decreasing": True,
        "violation": violation,
        "worst_drop_ratio": float(worst[4]) if worst else 0.0,
        "first_to_last_change_ratio": float(first_last_change_ratio),
        "from_x": float(worst[0]) if worst else None,
        "to_x": float(worst[1]) if worst else None,
        "from_value": float(worst[2]) if worst else None,
        "to_value": float(worst[3]) if worst else None,
        "x_values": x_values,
        "values": [float(v) for v in values.tolist()],
    }


def _actual_dimension_consistency_report(subset: pd.DataFrame) -> dict[str, Any]:
    if subset.empty:
        return {"mismatch_detected": False, "mismatches": []}
    pairs = [
        ("system.K", "actual_used_system_K", "actual_used_quick_K"),
        ("system.M", "actual_used_system_M", "actual_used_quick_M"),
        ("system.N", "actual_used_system_N", "actual_used_quick_N"),
        ("association.L_max", "actual_used_association_L_max", "actual_used_quick_L_max"),
    ]
    mismatches: list[dict[str, Any]] = []
    for label, requested_col, used_col in pairs:
        if requested_col not in subset.columns or used_col not in subset.columns:
            continue
        requested = pd.to_numeric(subset[requested_col], errors="coerce").dropna()
        used = pd.to_numeric(subset[used_col], errors="coerce").dropna()
        if requested.empty or used.empty:
            continue
        requested_values = sorted({float(v) for v in requested.tolist()})
        used_values = sorted({float(v) for v in used.tolist()})
        if requested_values != used_values:
            mismatches.append(
                {
                    "dimension": label,
                    "requested_values": requested_values,
                    "actual_used_values": used_values,
                }
            )
    return {"mismatch_detected": bool(mismatches), "mismatches": mismatches}


def _domain_column_contamination_report(df: pd.DataFrame, plan: dict[str, Any]) -> dict[str, Any]:
    family = str(plan.get("problem_family") or "").strip().lower()
    if not family or any(token in family for token in ("sensing", "swipt", "position", "localization")):
        return {"contamination_detected": False, "columns": []}
    forbidden_tokens = ("sensing", "harvest", "rectifier", "position_update", "separation_violation")
    columns = [
        str(column)
        for column in df.columns
        if any(token in str(column).strip().lower() for token in forbidden_tokens)
    ]
    return {
        "contamination_detected": bool(columns),
        "columns": sorted(columns),
        "problem_family": family,
    }


def _filter_rows_for_active_figures(df: pd.DataFrame, figure_ids: list[str]) -> pd.DataFrame:
    mask = _figure_row_mask(df, figure_ids)
    if mask is not None and bool(mask.any()):
        return df[mask].copy()
    return df


def get_chart_quality_policy(chart_type: str) -> dict[str, Any]:
    if chart_type in {"line", "scatter", "scatter_trend", "scatter_with_trend"}:
        return {
            "quick_points": 6,
            "paper_min_points": LINE_PAPER_MIN_X_POINTS,
            "paper_preferred_points": LINE_PAPER_PREFERRED_X_POINTS,
            "high_conf_points": LINE_HIGH_CONF_X_POINTS,
            "paper_min_seeds": PAPER_MINIMUM_SEEDS,
            "paper_preferred_seeds": PAPER_PREFERRED_SEEDS,
            "high_conf_seeds": HIGH_CONFIDENCE_SEEDS,
        }
    if chart_type in {"grouped_bar", "bar", "categorical_summary", "ablation_bar"}:
        return {
            "quick_points": 3,
            "paper_min_points": BAR_PAPER_MIN_CATEGORIES,
            "paper_preferred_points": BAR_PAPER_PREFERRED_CATEGORIES,
            "high_conf_points": BAR_PAPER_PREFERRED_CATEGORIES,
            "paper_min_seeds": PAPER_MINIMUM_SEEDS,
            "paper_preferred_seeds": PAPER_PREFERRED_SEEDS,
            "high_conf_seeds": HIGH_CONFIDENCE_SEEDS,
        }
    if chart_type == "box":
        return {
            "quick_points": 2,
            "paper_min_points": 2,
            "paper_preferred_points": 3,
            "high_conf_points": 3,
            "paper_min_seeds": BOX_PAPER_MIN_SAMPLES,
            "paper_preferred_seeds": BOX_PAPER_PREFERRED_SAMPLES,
            "high_conf_seeds": HIGH_CONFIDENCE_SEEDS,
        }
    if chart_type == "convergence":
        return {
            "quick_points": 5,
            "paper_min_points": CONVERGENCE_MIN_ITERATIONS,
            "paper_preferred_points": CONVERGENCE_MIN_ITERATIONS,
            "high_conf_points": CONVERGENCE_MIN_ITERATIONS,
            "paper_min_seeds": PAPER_MINIMUM_SEEDS,
            "paper_preferred_seeds": PAPER_PREFERRED_SEEDS,
            "high_conf_seeds": HIGH_CONFIDENCE_SEEDS,
        }
    if chart_type == "heatmap":
        return {
            "quick_points": 4,
            "paper_min_points": 8,
            "paper_preferred_points": 12,
            "high_conf_points": 15,
            "paper_min_seeds": PAPER_MINIMUM_SEEDS,
            "paper_preferred_seeds": PAPER_PREFERRED_SEEDS,
            "high_conf_seeds": HIGH_CONFIDENCE_SEEDS,
        }
    return {
        "quick_points": 4,
        "paper_min_points": LINE_PAPER_MIN_X_POINTS,
        "paper_preferred_points": LINE_PAPER_PREFERRED_X_POINTS,
        "high_conf_points": LINE_HIGH_CONF_X_POINTS,
        "paper_min_seeds": PAPER_MINIMUM_SEEDS,
        "paper_preferred_seeds": PAPER_PREFERRED_SEEDS,
        "high_conf_seeds": HIGH_CONFIDENCE_SEEDS,
    }


def _paired_success_rows_for_comparative_plot(
    subset: pd.DataFrame,
    *,
    methods: list[str],
    x_field: str,
) -> pd.DataFrame:
    if subset.empty or len(methods) < 2 or "method" not in subset.columns or x_field not in subset.columns:
        return subset
    seed_column = _detect_seed_column(subset)
    if not seed_column or seed_column not in subset.columns:
        return subset
    required_methods = {str(method) for method in methods if str(method).strip()}
    if len(required_methods) < 2:
        return subset
    keep_indices: list[Any] = []
    for (_x_value, _seed), group in subset.groupby([x_field, seed_column], dropna=False):
        present_methods = {str(method) for method in group["method"].dropna().astype(str).tolist()}
        if required_methods.issubset(present_methods):
            keep_indices.extend(group.index.tolist())
    if not keep_indices:
        return subset.iloc[0:0].copy()
    return subset.loc[keep_indices].copy()


def _paired_success_seed_coverage(
    subset: pd.DataFrame,
    *,
    methods: list[str],
    x_field: str,
    metric: str,
    figure_intent: str = "",
) -> dict[str, Any]:
    if subset.empty or "method" not in subset.columns or x_field not in subset.columns:
        return {"rows": [], "min_rate": 0.0, "min_paired_seeds": 0, "reliable_x_values": []}
    seed_column = _detect_seed_column(subset)
    if not seed_column or seed_column not in subset.columns:
        return {"rows": [], "min_rate": 0.0, "min_paired_seeds": 0, "reliable_x_values": []}
    required_methods = {str(method) for method in methods if str(method).strip()}
    if len(required_methods) < 2 or metric not in subset.columns:
        return {"rows": [], "min_rate": 0.0, "min_paired_seeds": 0, "reliable_x_values": []}
    finite_subset = subset[pd.to_numeric(subset[metric], errors="coerce").notna()].copy()
    success_subset = finite_subset[_strict_feasible_success_mask(finite_subset)].copy()
    rows: list[dict[str, Any]] = []
    reliable_x_values: list[Any] = []
    for x_value, x_rows in finite_subset.groupby(x_field, dropna=False):
        total_seeds = int(x_rows[seed_column].nunique())
        paired_success_seeds = 0
        if total_seeds:
            success_x = success_subset[success_subset[x_field] == x_value]
            for _seed, seed_rows in success_x.groupby(seed_column, dropna=False):
                present_methods = {str(method) for method in seed_rows["method"].dropna().astype(str).tolist()}
                if required_methods.issubset(present_methods):
                    paired_success_seeds += 1
        paired_success_rate = float(paired_success_seeds / total_seeds) if total_seeds else 0.0
        row = {
            "x_value": _json_safe(x_value),
            "total_seeds": total_seeds,
            "paired_success_seeds": paired_success_seeds,
            "paired_success_rate": paired_success_rate,
        }
        rows.append(row)
        if paired_success_seeds >= MIN_PREVIEW_PAIRED_SUCCESS_SEEDS and paired_success_rate >= MIN_PREVIEW_PAIRED_SUCCESS_RATE:
            reliable_x_values.append(_json_safe(x_value))
    if not rows:
        return {"rows": [], "min_rate": 0.0, "min_paired_seeds": 0, "reliable_x_values": []}
    return {
        "rows": rows,
        "min_rate": min(float(row["paired_success_rate"]) for row in rows),
        "min_paired_seeds": min(int(row["paired_success_seeds"]) for row in rows),
        "reliable_x_values": reliable_x_values,
    }


def _filter_undercovered_curve_points(
    subset: pd.DataFrame,
    *,
    methods: list[str],
    x_field: str,
    metric: str,
    figure_intent: str,
    coverage_subset: pd.DataFrame | None = None,
    min_remaining_points: int = 3,
) -> pd.DataFrame:
    coverage = _paired_success_seed_coverage(
        coverage_subset if coverage_subset is not None else subset,
        methods=methods,
        x_field=x_field,
        metric=metric,
        figure_intent=figure_intent,
    )
    reliable_values = set(coverage.get("reliable_x_values", []))
    if len(reliable_values) < min_remaining_points:
        return subset
    comparable = subset[x_field].map(_json_safe)
    return subset[comparable.isin(reliable_values)].copy()


def aggregate_for_figure(df: pd.DataFrame, figure_spec: dict[str, Any]) -> pd.DataFrame:
    figure_spec = normalize_figure_spec(figure_spec)
    chart_type = str(figure_spec.get("chart_type", "line"))
    sweep_param = _get_figure_sweep_param(figure_spec)
    metric = _get_figure_metric_name(figure_spec)
    methods = _get_figure_methods(figure_spec)
    x_field = _get_figure_x_field(figure_spec)
    figure_id = str(figure_spec.get("figure_id", "figure"))
    subset = df[df["method"].isin(methods)].copy()
    subset = _filter_rows_for_figure(subset, figure_id)
    subset = _filter_rows_for_required_sweep(subset, _get_figure_required_sweep(figure_spec))
    if sweep_param and "swept_param" in subset.columns:
        subset = subset[subset["swept_param"] == sweep_param]
    if metric not in subset.columns:
        if metric in {"sum_power_dBm", "total_power_dBm"}:
            source_metric = "sum_power_W" if metric == "sum_power_dBm" else "total_power_W"
            if source_metric in subset.columns:
                linear_power = pd.to_numeric(subset[source_metric], errors="coerce").clip(lower=1.0e-15)
                subset[metric] = 10.0 * np.log10(linear_power * 1000.0)
    if metric not in subset.columns:
        raise ValueError(f"missing_metric_for_{figure_spec.get('figure_id', 'figure')}")
    finite_metric = pd.to_numeric(subset[metric], errors="coerce")
    subset = subset[finite_metric.notna()].copy()
    raw_finite_subset = subset.copy()
    subset = _successful_evidence_rows(subset, metric, _figure_intent(figure_spec))
    if subset.empty:
        raise ValueError(f"no_successful_finite_rows_for_{figure_spec.get('figure_id', 'figure')}")
    if x_field not in subset.columns:
        if chart_type in {"line", "scatter", "scatter_trend", "scatter_with_trend", "convergence"} and "swept_value" in subset.columns:
            x_field = "swept_value"
        elif chart_type in {"grouped_bar", "bar", "box"} and "scenario_name" in subset.columns:
            x_field = "scenario_name"
        else:
            raise ValueError(f"missing_x_field_for_{figure_spec.get('figure_id', 'figure')}")
    if chart_type in {"line", "scatter", "scatter_trend", "scatter_with_trend", "convergence"}:
        if len(methods) >= 2:
            subset = _paired_success_rows_for_comparative_plot(subset, methods=methods, x_field=x_field)
            if subset.empty:
                raise ValueError(f"no_paired_successful_rows_for_{figure_spec.get('figure_id', 'figure')}")
            subset = _filter_undercovered_curve_points(
                subset,
                methods=methods,
                x_field=x_field,
                metric=metric,
                figure_intent=_figure_intent(figure_spec),
                coverage_subset=raw_finite_subset,
            )
        if subset.empty:
            raise ValueError(f"no_reliably_covered_rows_for_{figure_spec.get('figure_id', 'figure')}")

    if chart_type in {"line", "scatter", "scatter_trend", "scatter_with_trend", "convergence"}:
        group_keys = ["method", x_field]
    elif chart_type in {"grouped_bar", "bar", "box", "categorical_summary", "ablation_bar"}:
        group_keys = [x_field, "method"]
    elif chart_type == "heatmap":
        y_field = str(figure_spec.get("encoding", {}).get("facet", {}).get("field") or "")
        if not y_field or y_field not in subset.columns:
            raise ValueError(f"missing_heatmap_y_field_for_{figure_spec.get('figure_id', 'figure')}")
        group_keys = [x_field, y_field]
    else:
        group_keys = ["method", x_field]

    if chart_type == "box":
        grouped = (
            subset.groupby(group_keys, dropna=False)[metric]
            .agg(
                mean_metric="mean",
                std_metric="std",
                count="count",
                sample_values=lambda s: [float(v) for v in pd.to_numeric(s, errors="coerce").dropna().tolist()],
            )
            .reset_index()
        )
    else:
        grouped = (
            subset.groupby(group_keys, dropna=False)[metric]
            .agg(["mean", "std", "count"])
            .reset_index()
            .rename(columns={"mean": "mean_metric", "std": "std_metric", "count": "count"})
        )
    seed_column = _detect_seed_column(subset)
    if seed_column:
        seed_counts = (
            subset.groupby(group_keys, dropna=False)[seed_column]
            .nunique()
            .reset_index(name="num_unique_seeds")
        )
    else:
        seed_counts = grouped[group_keys].copy()
        seed_counts["num_unique_seeds"] = 0
    grouped = grouped.merge(seed_counts, on=group_keys, how="left")
    grouped["stderr_metric"] = grouped.apply(
        lambda row: float(row["std_metric"] / math.sqrt(row["count"])) if pd.notna(row["std_metric"]) and int(row["count"]) >= 2 else math.nan,
        axis=1,
    )
    grouped["ci95_metric"] = grouped["stderr_metric"].apply(lambda v: float(1.96 * v) if pd.notna(v) else math.nan)
    feasible_rates = subset.groupby(group_keys, dropna=False)["feasible"].mean().reset_index(name="feasible_rate")
    finite_counts = subset.groupby(group_keys, dropna=False)["finite_primary_metric"].sum().reset_index(name="finite_count")
    grouped = grouped.merge(feasible_rates, on=group_keys, how="left")
    grouped = grouped.merge(finite_counts, on=group_keys, how="left")
    grouped["swept_param"] = sweep_param
    grouped["chart_type"] = chart_type
    grouped["x_field"] = x_field
    if x_field in grouped.columns and "x_value" not in grouped.columns:
        grouped["x_value"] = grouped[x_field]
    if chart_type in {"grouped_bar", "bar", "box", "categorical_summary", "ablation_bar"}:
        grouped = grouped.rename(columns={x_field: "category"})
    elif chart_type == "heatmap":
        y_field = str(figure_spec.get("encoding", {}).get("facet", {}).get("field") or "")
        if y_field in grouped.columns:
            grouped = grouped.rename(columns={x_field: "grid_x", y_field: "grid_y"})
    grouped = grouped.sort_values([col for col in ("method", "x_value", "category") if col in grouped.columns])
    return grouped


def write_curve_data(curve_df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    curve_df.to_csv(output_path, index=False, encoding="utf-8-sig")


def _figure_intent(figure_spec: dict[str, Any]) -> str:
    for key in ("chart_intent", "intent", "claim_type"):
        value = str(figure_spec.get(key) or "").strip().lower()
        if value:
            return value
    purpose = str(figure_spec.get("purpose") or figure_spec.get("claim") or "").strip().lower()
    if any(token in purpose for token in ("feasibility", "boundary", "violation")):
        return "feasibility_boundary"
    if any(token in purpose for token in ("ablation", "mechanism")):
        return "mechanism_ablation"
    if any(token in purpose for token in ("sensitivity", "sweep", "tradeoff", "trade-off")):
        return "sensitivity"
    if any(token in purpose for token in ("convergence", "iteration", "residual")):
        return "convergence"
    return "main_comparison"


def _is_objective_metric(metric: str) -> bool:
    return str(metric or "").strip().lower() in OBJECTIVE_METRIC_NAMES


def _is_violation_or_feasibility_metric(metric: str) -> bool:
    lowered = str(metric or "").strip().lower()
    return any(token in lowered for token in ("violation", "feasible", "feasibility", "outage"))


def _is_deterministic_boundary_metric(metric: str, figure_intent: str) -> bool:
    metric_lower = str(metric or "").strip().lower()
    intent_lower = str(figure_intent or "").strip().lower()
    if any(token in intent_lower for token in ("feasibility", "boundary", "constraint", "outage")):
        return _is_violation_or_feasibility_metric(metric_lower)
    return any(token in metric_lower for token in ("violation", "feasible", "feasibility", "outage"))


def _subset_has_true_flag(subset: pd.DataFrame, columns: list[str]) -> bool:
    if subset.empty:
        return False
    for column in columns:
        if column not in subset.columns:
            continue
        series = subset[column]
        if pd.api.types.is_bool_dtype(series):
            if bool(series.fillna(False).any()):
                return True
            continue
        normalized = series.astype(str).str.strip().str.lower()
        if bool(normalized.isin({"1", "true", "yes", "y"}).any()):
            return True
    return False


def _is_runtime_metric(metric: str) -> bool:
    lowered = str(metric or "").strip().lower()
    return any(token in lowered for token in ("runtime", "solve_time", "update_time", "time_ms", "time_sec", "latency"))


def _subset_has_proxy_marker(subset: pd.DataFrame) -> bool:
    if subset.empty:
        return False
    if _subset_has_true_flag(subset, ["runtime_proxy_used", "complexity_proxy_used"]):
        return True
    for column in ("algorithm_approximation", "approximation_mode", "proxy_mode"):
        if column not in subset.columns:
            continue
        normalized = subset[column].astype(str).str.strip().str.lower()
        nonempty = normalized[~normalized.isin({"", "none", "false", "0", "nan"})]
        if nonempty.empty:
            continue
        proxy_like = nonempty.str.contains(
            r"proxy|surrogate|fallback|unavailable|approx(?:imation|imate)?",
            regex=True,
            na=False,
        )
        if bool(proxy_like.any()):
            return True
    return False


def _metric_varies_across_sweep(subset: pd.DataFrame, *, y_metric: str, x_field: str) -> tuple[bool, dict[str, Any]]:
    if subset.empty or y_metric not in subset.columns or x_field not in subset.columns or "method" not in subset.columns:
        return False, {"reason": "insufficient_columns"}
    rows: list[dict[str, Any]] = []
    any_varying = False
    for method, method_df in subset.groupby("method", dropna=False):
        grouped = (
            method_df.groupby(x_field, dropna=False)[y_metric]
            .apply(lambda s: pd.to_numeric(s, errors="coerce").dropna().mean())
            .dropna()
        )
        if len(grouped) < 2:
            rows.append({"method": str(method), "num_x_points": int(len(grouped)), "varies": False, "max_abs_delta": 0.0})
            continue
        values = grouped.to_numpy(dtype=float)
        max_abs_delta = float(np.nanmax(values) - np.nanmin(values))
        scale = max(float(np.nanmax(np.abs(values))), 1.0)
        varies = bool(max_abs_delta > max(1e-9, 1e-6 * scale))
        any_varying = any_varying or varies
        rows.append(
            {
                "method": str(method),
                "num_x_points": int(len(grouped)),
                "varies": varies,
                "max_abs_delta": max_abs_delta,
            }
        )
    return any_varying, {"by_method": rows}


def _paired_method_x_points(subset: pd.DataFrame, *, x_field: str, methods_required: list[str]) -> int:
    if subset.empty or x_field not in subset.columns or "method" not in subset.columns:
        return 0
    required = {str(item) for item in methods_required if str(item)}
    if not required:
        return int(subset[x_field].nunique())
    count = 0
    for _x_value, group in subset.groupby(x_field, dropna=False):
        present = set(group["method"].dropna().astype(str).tolist())
        if required.issubset(present):
            count += 1
    return count


def check_data_sufficiency(
    df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    plan: dict[str, Any],
    monte_carlo_report: dict[str, Any],
    quick_mode: bool,
    run_mode: str = "",
) -> dict[str, Any]:
    figure_reports: list[dict[str, Any]] = []
    mode_label = str(run_mode or ("scout" if quick_mode else "paper")).strip().lower()
    draft_mode_issue = "medium_mode_only" if mode_label == "medium" else "quick_mode_only"
    overall_status = draft_mode_issue if quick_mode else "needs_more_phase24_runs"
    compared_methods = [item.get("name", "") for item in plan.get("compared_methods", [])]
    for fig in plan.get("figure_specs", []):
        fig = normalize_figure_spec(fig, plan.get("primary_metric", {}))
        figure_id = fig.get("figure_id", "figure")
        chart_type = str(fig.get("chart_type", "line"))
        sweep_param = _get_figure_sweep_param(fig)
        y_metric = _get_figure_metric_name(fig)
        methods_required = _get_figure_methods(fig, compared_methods) or compared_methods
        x_field = _get_figure_x_field(fig)
        policy = get_chart_quality_policy(chart_type)
        data_requirements = fig.get("data_requirements", {})
        explicit_min_points = data_requirements.get("min_points")
        min_point_floor = 3 if chart_type in {"line", "scatter", "scatter_trend", "scatter_with_trend"} else 2
        min_points = (
            max(int(explicit_min_points), min_point_floor)
            if explicit_min_points is not None
            else int(policy["paper_min_points"])
        )
        preferred_points = max(int(data_requirements.get("preferred_points", policy["paper_preferred_points"])), min_points)
        min_samples = max(int(data_requirements.get("min_samples_per_group", policy["paper_min_seeds"])), int(policy["paper_min_seeds"]))
        preferred_samples = max(int(data_requirements.get("preferred_samples_per_group", policy["paper_preferred_seeds"])), int(policy["paper_preferred_seeds"]))
        high_conf_points = int(policy["high_conf_points"])
        high_conf_seeds = int(policy["high_conf_seeds"])

        subset = _filter_rows_for_figure(df.copy(), str(figure_id))
        subset = _filter_rows_for_required_sweep(subset, _get_figure_required_sweep(fig))
        if sweep_param and "swept_param" in subset.columns:
            subset = subset[subset["swept_param"] == sweep_param]
        if x_field not in subset.columns:
            if chart_type in {"grouped_bar", "bar", "box"} and "scenario_name" in subset.columns:
                x_field = "scenario_name"
            elif "swept_value" in subset.columns:
                x_field = "swept_value"
        all_x_values = _safe_sorted_json_values(subset[x_field].dropna().tolist()) if not subset.empty and x_field in subset.columns else []
        figure_intent = _figure_intent(fig)
        raw_analysis_subset = pd.DataFrame()
        if y_metric in subset.columns:
            finite_metric = pd.to_numeric(subset[y_metric], errors="coerce")
            raw_analysis_subset = subset[finite_metric.notna()].copy()
            analysis_subset = _successful_evidence_rows(raw_analysis_subset, y_metric, figure_intent)
            success_subset = analysis_subset
        else:
            success_subset = pd.DataFrame()
            analysis_subset = success_subset
        x_values = _safe_sorted_json_values(analysis_subset[x_field].dropna().tolist()) if not analysis_subset.empty and x_field in analysis_subset.columns else []
        num_x_points = len(x_values)
        policy_min_points = int(policy["paper_min_points"])
        min_points_adjustment_reason = ""
        if (
            not quick_mode
            and explicit_min_points is not None
            and min_points > num_x_points
            and num_x_points >= policy_min_points
            and all_x_values
            and num_x_points >= len(all_x_values)
        ):
            # LLM-authored figure specs sometimes request 12+ line-plot points
            # while the promoted sweep plan has already exhausted a complete
            # 10/11-point WCL-style grid with strong Monte Carlo coverage. Do
            # not launch another expensive paper sweep solely to chase an
            # internally inconsistent point count.
            min_points = num_x_points
            preferred_points = max(preferred_points, min_points)
            min_points_adjustment_reason = "capped_to_completed_paper_grid"
        discrete_grid_reason = _completed_discrete_integer_grid_reason(
            figure_spec=fig,
            sweep_param=sweep_param,
            x_values=x_values,
            all_x_values=all_x_values,
            min_points=min_points,
        )
        if not quick_mode and discrete_grid_reason:
            min_points = num_x_points
            preferred_points = max(preferred_points, min_points)
            min_points_adjustment_reason = discrete_grid_reason
        mc_figure = next((item for item in monte_carlo_report.get("figures", []) if item.get("figure_id") == figure_id), {})
        mc_rows = mc_figure.get("rows", [])
        methods_present = sorted(analysis_subset["method"].dropna().astype(str).unique().tolist()) if not analysis_subset.empty else []
        seeds_values = [int(row.get("num_unique_seeds", 0)) for row in mc_rows] if mc_rows else []
        seeds_summary = {
            "min": int(min(seeds_values)) if seeds_values else 0,
            "median": float(statistics.median(seeds_values)) if seeds_values else 0.0,
            "max": int(max(seeds_values)) if seeds_values else 0,
        }
        blocking_issues: list[str] = []
        if chart_type in {"line", "scatter", "scatter_trend", "scatter_with_trend"} and num_x_points < min_points:
            blocking_issues.append("too_few_x_points")
        elif chart_type in {"grouped_bar", "bar", "categorical_summary", "ablation_bar"} and num_x_points < min_points:
            blocking_issues.append("too_few_categories")
        elif chart_type == "box" and seeds_summary["min"] < BOX_PAPER_MIN_SAMPLES:
            blocking_issues.append("too_few_samples_for_box")
        elif chart_type == "convergence" and num_x_points < CONVERGENCE_MIN_ITERATIONS:
            blocking_issues.append("too_few_iterations")
        elif chart_type == "heatmap" and num_x_points < min_points:
            blocking_issues.append("insufficient_heatmap_coverage")
        if seeds_summary["min"] < PAPER_MINIMUM_SEEDS:
            blocking_issues.append("too_few_seeds_per_point")
        if monte_carlo_report.get("unknown_seed_coverage", False):
            blocking_issues.append("missing_seed_column")
        missing_methods = sorted(set(methods_required) - set(methods_present))
        if missing_methods:
            blocking_issues.append("missing_method_curve")
        if y_metric not in analysis_subset.columns:
            blocking_issues.append("missing_primary_metric")
        else:
            finite_ok = analysis_subset[y_metric].notna().all() and analysis_subset[y_metric].apply(lambda x: math.isfinite(float(x))).all()
            if not finite_ok:
                blocking_issues.append("non_finite_metric")
        semantic_variation: dict[str, Any] = {"reason": "not_evaluated"}
        metric_varies = False
        trend_report: dict[str, Any] = {"reason": "not_evaluated"}
        dimension_report: dict[str, Any] = {"mismatch_detected": False, "mismatches": []}
        if _is_objective_metric(y_metric) and figure_intent in MECHANISM_INTENTS:
            blocking_issues.append("objective_y_axis_for_mechanism_claim")
        if y_metric in analysis_subset.columns and x_field in analysis_subset.columns:
            metric_varies, semantic_variation = _metric_varies_across_sweep(analysis_subset, y_metric=y_metric, x_field=x_field)
            if num_x_points >= 2 and not metric_varies:
                blocking_issues.append("metric_constant_across_sweep")
            if (
                select_comparison_proposed_method(df, plan) in methods_required
                or "proposed" in methods_required
            ):
                proposed_id = select_comparison_proposed_method(df, plan)
                collapse = _curve_collapse_report(
                    analysis_subset,
                    method_id=proposed_id,
                    metric=y_metric,
                    x_field=x_field,
                    figure_spec=fig,
                    higher_is_better=bool(
                        (fig.get("metric", {}) if isinstance(fig.get("metric", {}), dict) else {}).get(
                            "higher_is_better",
                            _metric_higher_is_better(y_metric, True),
                        )
                    ),
                )
                if collapse.get("collapse_detected"):
                    blocking_issues.append("resource_sweep_nonmonotone_collapse")
                    semantic_variation["resource_sweep_collapse"] = collapse
                trend_report = _expected_non_decreasing_trend_report(
                    analysis_subset,
                    method_id=proposed_id,
                    metric=y_metric,
                    x_field=x_field,
                    figure_spec=fig,
                    higher_is_better=bool(
                        (fig.get("metric", {}) if isinstance(fig.get("metric", {}), dict) else {}).get(
                            "higher_is_better",
                            _metric_higher_is_better(y_metric, True),
                        )
                    ),
                )
                if trend_report.get("violation"):
                    blocking_issues.append("expected_non_decreasing_trend_violation")
                    semantic_variation["expected_trend_violation"] = trend_report
        dimension_report = _actual_dimension_consistency_report(raw_analysis_subset)
        if dimension_report.get("mismatch_detected"):
            if quick_mode:
                semantic_variation["actual_dimension_truncation"] = dimension_report
            else:
                blocking_issues.append("actual_dimension_mismatch")
                semantic_variation["actual_dimension_mismatch"] = dimension_report
        if _is_runtime_metric(y_metric) and _subset_has_proxy_marker(analysis_subset):
            blocking_issues.append("runtime_metric_is_proxy_not_measured")
        if _subset_has_proxy_marker(analysis_subset) and figure_intent in {"main_comparison", "mechanism_ablation"}:
            blocking_issues.append("plotted_metric_uses_algorithm_proxy")
        comparable_points = 0
        if not comparison_df.empty and "swept_param" in comparison_df.columns:
            comp_subset = comparison_df[(comparison_df["swept_param"] == sweep_param) & comparison_df["comparable"].astype(bool)]
            comp_subset = _filter_rows_for_required_sweep(comp_subset, _get_figure_required_sweep(fig))
            if not comp_subset.empty:
                compare_field = x_field if x_field in comp_subset.columns else "swept_value"
                comparable_points = int(comp_subset[compare_field].nunique()) if compare_field in comp_subset.columns else 0
        paired_method_points = _paired_method_x_points(analysis_subset, x_field=x_field, methods_required=methods_required)
        paired_success_coverage = _paired_success_seed_coverage(
            raw_analysis_subset,
            methods=methods_required,
            x_field=x_field,
            metric=y_metric,
            figure_intent=figure_intent,
        )
        coverage_rows = paired_success_coverage.get("rows", [])
        low_coverage_rows = [
            row for row in coverage_rows
            if int(row.get("paired_success_seeds", 0) or 0) < MIN_PREVIEW_PAIRED_SUCCESS_SEEDS
            or float(row.get("paired_success_rate", 0.0) or 0.0) < MIN_PREVIEW_PAIRED_SUCCESS_RATE
        ]
        if low_coverage_rows and not _is_violation_or_feasibility_metric(y_metric):
            blocking_issues.append("low_paired_success_rate_per_x")
        if paired_method_points < num_x_points:
            blocking_issues.append("insufficient_comparable_points")
        effective_x_values = list(x_values)
        if low_coverage_rows and coverage_rows and paired_success_coverage.get("reliable_x_values"):
            effective_x_values = list(paired_success_coverage.get("reliable_x_values", []))
        effective_num_x_points = len(effective_x_values)
        if (
            chart_type in {"line", "scatter", "scatter_trend", "scatter_with_trend", "convergence"}
            and not _is_violation_or_feasibility_metric(y_metric)
            and effective_num_x_points < min_points
        ):
            blocking_issues.append("too_few_effective_x_points_after_feasibility_filter")
        if (
            mode_label == "medium"
            and chart_type in {"line", "scatter", "scatter_trend", "scatter_with_trend"}
            and not _is_violation_or_feasibility_metric(y_metric)
            and effective_num_x_points < _medium_target_points(chart_type)
        ):
            blocking_issues.append("too_few_medium_x_points_after_filter")
        planned_missing_after_filter = False
        if len(all_x_values) > num_x_points:
            if num_x_points >= min_points:
                planned_missing_after_filter = True
            else:
                blocking_issues.append("planned_x_points_missing_after_feasibility_filter")
        zero_variance_rows = [
            row
            for row in mc_rows
            if "zero_variance_across_all_seeds" in row.get("warnings", [])
        ]
        repeated_rows = [
            row
            for row in mc_rows
            if "repeated_identical_outputs_across_seeds" in row.get("warnings", [])
        ]
        proposed_mc_rows = [row for row in mc_rows if str(row.get("method", "")).strip().lower() == "proposed"]
        zero_variance_blocking = bool(mc_rows) and (
            len(zero_variance_rows) == len(mc_rows)
            or (bool(proposed_mc_rows) and all("zero_variance_across_all_seeds" in row.get("warnings", []) for row in proposed_mc_rows))
        )
        repeated_outputs_blocking = bool(mc_rows) and (
            len(repeated_rows) == len(mc_rows)
            or (bool(proposed_mc_rows) and all("repeated_identical_outputs_across_seeds" in row.get("warnings", []) for row in proposed_mc_rows))
        )
        mc_warnings = sorted({warning for row in mc_rows for warning in row.get("warnings", [])})
        if planned_missing_after_filter:
            mc_warnings.append("planned_x_points_missing_after_feasibility_filter_diagnostic_only")
        if zero_variance_rows and not zero_variance_blocking:
            mc_warnings = [warning for warning in mc_warnings if warning != "zero_variance_across_all_seeds"]
            mc_warnings.append("isolated_zero_variance_groups")
        if repeated_rows and not repeated_outputs_blocking:
            mc_warnings = [warning for warning in mc_warnings if warning != "repeated_identical_outputs_across_seeds"]
            if "isolated_repeated_output_groups" not in mc_warnings:
                mc_warnings.append("isolated_repeated_output_groups")
        mc_warnings = sorted(dict.fromkeys(mc_warnings))
        deterministic_boundary_valid = (
            _is_deterministic_boundary_metric(y_metric, figure_intent)
            and metric_varies
            and num_x_points >= min_points
            and paired_method_points >= num_x_points
            and not missing_methods
            and y_metric in analysis_subset.columns
        )
        deterministic_sweep_valid = (
            chart_type in {"line", "scatter", "box"}
            and figure_intent in MECHANISM_INTENTS
            and metric_varies
            and not _is_runtime_metric(y_metric)
            and not _subset_has_proxy_marker(analysis_subset)
            and num_x_points >= min_points
            and paired_method_points >= num_x_points
            and not missing_methods
            and y_metric in analysis_subset.columns
        )
        deterministic_evidence_valid = deterministic_boundary_valid or deterministic_sweep_valid
        displayed_warnings = list(mc_warnings)
        if deterministic_evidence_valid and (
            "zero_variance_across_all_seeds" in mc_warnings
            or "repeated_identical_outputs_across_seeds" in mc_warnings
        ):
            displayed_warnings.append("deterministic_sweep_zero_seed_variance")
        if zero_variance_blocking and "zero_variance_across_all_seeds" in mc_warnings and not deterministic_evidence_valid:
            blocking_issues.append("zero_variance_across_seeds")
        if repeated_outputs_blocking and "repeated_identical_outputs_across_seeds" in mc_warnings and not deterministic_evidence_valid:
            blocking_issues.append("repeated_identical_outputs_across_seeds")
        # Final WCL plots intentionally avoid error bars; seed coverage is
        # tracked through stability diagnostics and `too_few_seeds_per_point`.
        has_error_bars = False
        if quick_mode:
            blocking_issues.append(draft_mode_issue)

        mc_blocking_warnings = {"missing_seed_column", "too_few_seeds"}
        if not deterministic_evidence_valid:
            if repeated_outputs_blocking:
                mc_blocking_warnings.add("repeated_identical_outputs_across_seeds")
            if zero_variance_blocking:
                mc_blocking_warnings.add("zero_variance_across_all_seeds")
        monte_carlo_valid = not any(w in mc_warnings for w in mc_blocking_warnings)
        paper_minimum_ready = (
            (not quick_mode)
            and seeds_summary["min"] >= PAPER_MINIMUM_SEEDS
            and num_x_points >= min_points
            and not missing_methods
            and not blocking_issues
            and monte_carlo_valid
            and y_metric in analysis_subset.columns
        )
        counts_toward_paper_minimum = _figure_counts_toward_paper_minimum(fig, y_metric)
        if not counts_toward_paper_minimum:
            displayed_warnings.append("diagnostic_only_not_counted_as_primary_paper_figure")
        paper_preferred_ready = (
            paper_minimum_ready
            and seeds_summary["min"] >= PAPER_PREFERRED_SEEDS
            and num_x_points >= preferred_points
        )
        high_confidence_ready = (
            paper_preferred_ready
            and seeds_summary["min"] >= HIGH_CONFIDENCE_SEEDS
            and num_x_points >= high_conf_points
        )
        quality_level = "draft_only"
        if paper_minimum_ready:
            quality_level = "paper_minimum_ready"
        if paper_preferred_ready:
            quality_level = "paper_preferred_ready"
        if high_confidence_ready:
            quality_level = "high_confidence_ready"
        paper_ready = paper_minimum_ready
        figure_reports.append(
            {
                "figure_id": figure_id,
                "chart_type": chart_type,
                "purpose": fig.get("purpose", ""),
                "x_axis_param": sweep_param,
                "required_sweep": _get_figure_required_sweep(fig),
                "y_metric": y_metric,
                "figure_intent": figure_intent,
                "semantic_metric_variation": semantic_variation,
                "expected_trend_report": trend_report,
                "actual_dimension_consistency": dimension_report,
                "methods_required": methods_required,
                "methods_present": methods_present,
                "num_x_points": num_x_points,
                "x_values": x_values,
                "effective_x_values": effective_x_values,
                "effective_num_x_points": effective_num_x_points,
                "all_requested_x_values": all_x_values,
                "seeds_per_point_summary": seeds_summary,
                "comparable_points": comparable_points,
                "paired_method_points": paired_method_points,
                "paired_success_seed_coverage": paired_success_coverage,
                "finite_points": int(success_subset[y_metric].notna().sum()) if y_metric in success_subset.columns else 0,
                "feasible_points": int(success_subset["feasible"].astype(bool).sum()) if "feasible" in success_subset.columns else 0,
                "has_error_bars": has_error_bars,
                "monte_carlo_valid": monte_carlo_valid,
                "deterministic_boundary_valid": deterministic_boundary_valid,
                "deterministic_sweep_valid": deterministic_sweep_valid,
                "paper_ready": paper_ready,
                "draft_only": not paper_ready,
                "quality_level": quality_level,
                "paper_minimum_ready": paper_minimum_ready,
                "paper_preferred_ready": paper_preferred_ready,
                "high_confidence_ready": high_confidence_ready,
                "counts_toward_paper_minimum": counts_toward_paper_minimum,
                "blocking_issues": sorted(dict.fromkeys(blocking_issues)),
                "warnings": sorted(dict.fromkeys(displayed_warnings)),
                "suggested_min_x_points": min_points,
                "suggested_preferred_x_points": preferred_points,
                "min_points_adjustment_reason": min_points_adjustment_reason,
                "suggested_min_seeds_per_point": min_samples,
                "suggested_preferred_seeds_per_point": preferred_samples,
                "suggested_high_confidence_x_points": high_conf_points,
                "suggested_high_confidence_seeds_per_point": high_conf_seeds,
            }
        )
    if quick_mode:
        overall_status = draft_mode_issue
    else:
        eligible_reports = [item for item in figure_reports if item.get("counts_toward_paper_minimum")]
        ready_eligible_reports = [
            item
            for item in eligible_reports
            if item.get("quality_level") in {"paper_minimum_ready", "paper_preferred_ready", "high_confidence_ready"}
        ]
        if len(ready_eligible_reports) < MIN_PAPER_READY_FIGURES:
            overall_status = "needs_more_phase24_runs"
        elif all(item["quality_level"] == "high_confidence_ready" for item in ready_eligible_reports):
            overall_status = "high_confidence_ready"
        elif all(item["quality_level"] in {"paper_preferred_ready", "high_confidence_ready"} for item in ready_eligible_reports):
            overall_status = "paper_preferred_ready"
        elif all(item["quality_level"] in {"paper_minimum_ready", "paper_preferred_ready", "high_confidence_ready"} for item in ready_eligible_reports):
            overall_status = "paper_minimum_ready"
        else:
            overall_status = "needs_more_phase24_runs"
    ready_count = sum(
        1
        for item in figure_reports
        if item.get("counts_toward_paper_minimum")
        and item.get("quality_level") in {"paper_minimum_ready", "paper_preferred_ready", "high_confidence_ready"}
    )
    global_blocking_issues: list[str] = []
    contamination_report = _domain_column_contamination_report(df, plan)
    if contamination_report.get("contamination_detected"):
        global_blocking_issues.append(
            "Result table contains out-of-domain diagnostic columns for this problem family: "
            + ", ".join(contamination_report.get("columns", []))
        )
    if not quick_mode and ready_count < MIN_PAPER_READY_FIGURES:
        global_blocking_issues.append(
            f"Only {ready_count} non-diagnostic paper-ready figures are available; at least {MIN_PAPER_READY_FIGURES} are required."
        )
    return {
        "overall_status": overall_status,
        "minimum_paper_ready_figures": MIN_PAPER_READY_FIGURES,
        "paper_ready_main_figure_count": ready_count,
        "global_blocking_issues": global_blocking_issues,
        "domain_column_contamination": contamination_report,
        "figures": figure_reports,
    }


def _remove_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


def _prepare_figure_paths(output_dir: Path, figure_id: str, paper_ready: bool) -> tuple[Path, Path]:
    suffix = "" if paper_ready else "_draft"
    png_path = output_dir / f"{figure_id}{suffix}.png"
    pdf_path = output_dir / f"{figure_id}{suffix}.pdf"
    alt_png_path = output_dir / f"{figure_id}{'_draft' if paper_ready else ''}.png"
    alt_pdf_path = output_dir / f"{figure_id}{'_draft' if paper_ready else ''}.pdf"
    _remove_if_exists(alt_png_path)
    _remove_if_exists(alt_pdf_path)
    return png_path, pdf_path


def render_line_figure(
    curve_df: pd.DataFrame,
    figure_spec: dict[str, Any],
    output_dir: str | Path,
    paper_ready: bool,
    *,
    plan: dict[str, Any],
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_id = figure_spec.get("figure_id", "figure")
    png_path, pdf_path = _prepare_figure_paths(output_dir, figure_id, paper_ready)

    y_label = _safe_display(figure_spec.get("metric", {}).get("display_name", _get_figure_metric_name(figure_spec)))
    x_label = _get_figure_x_display(figure_spec)
    error_display = str(figure_spec.get("error_display", "none")).strip().lower()
    chart_type = str(figure_spec.get("chart_type", "line")).strip().lower()
    fig, ax = plt.subplots(figsize=(IEEE_SINGLE_COLUMN_WIDTH_IN, IEEE_COMPACT_HEIGHT_IN), dpi=300)
    sort_col = "x_value" if "x_value" in curve_df.columns else "swept_value"
    curve_df = curve_df.sort_values(["method", sort_col])
    has_error_bars = False
    present_methods = [str(method) for method in curve_df["method"].astype(str).unique().tolist()]
    requested_methods = [str(method) for method in _get_figure_methods(figure_spec) if str(method) in present_methods]
    ordered_methods = requested_methods + [method for method in present_methods if method not in requested_methods]
    for idx, method in enumerate(ordered_methods):
        group = curve_df[curve_df["method"].astype(str) == str(method)].copy()
        group = group.sort_values(sort_col)
        x = pd.to_numeric(group[sort_col], errors="coerce").to_numpy(dtype=float)
        y = group["mean_metric"].to_numpy(dtype=float)
        yerr = None
        error_col = {"ci95": "ci95_metric", "stderr": "stderr_metric", "std": "std_metric"}.get(error_display, "")
        if error_col and error_col in group.columns and pd.notna(group[error_col]).sum() >= 1 and group["count"].min() >= 2:
            yerr = group[error_col].fillna(0.0).to_numpy(dtype=float)
            has_error_bars = True
        is_scatter_like = chart_type in {"scatter", "scatter_trend", "scatter_with_trend"}
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            marker=FIGURE_MARKERS[idx % len(FIGURE_MARKERS)],
            color=FIGURE_COLORS[idx % len(FIGURE_COLORS)],
            linestyle="none" if is_scatter_like else "-",
            linewidth=1.15,
            markersize=3.4 if is_scatter_like else 3.0,
            capsize=2.0 if yerr is not None else 0.0,
            elinewidth=0.65,
            markeredgewidth=0.65,
            markerfacecolor="white",
            label=_method_display_name(plan, str(method), long=False),
        )
        trend_guide = chart_type in {"scatter_trend", "scatter_with_trend"} or bool(figure_spec.get("trend_guide"))
        if trend_guide and is_scatter_like and len(np.unique(x[np.isfinite(x)])) >= 4:
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() >= 4:
                x_fit = x[mask]
                y_fit = y[mask]
                degree = min(2, max(1, int(figure_spec.get("trend_degree", 2))))
                if len(np.unique(x_fit)) <= degree:
                    degree = 1
                xx = np.linspace(float(np.min(x_fit)), float(np.max(x_fit)), 160)
                try:
                    coeffs = np.polyfit(x_fit, y_fit, degree)
                    yy = np.polyval(coeffs, xx)
                    ax.plot(
                        xx,
                        yy,
                        linestyle="--",
                        linewidth=0.95,
                        color=FIGURE_COLORS[idx % len(FIGURE_COLORS)],
                        alpha=0.72,
                    )
                except Exception:
                    pass
    metric_key = _get_figure_metric_name(figure_spec).strip().lower()
    if metric_key in {"spectral_radius_f", "rho_f"} or "spectral radius" in y_label.lower():
        ax.axhline(1.0, color="#6b7280", linestyle="--", linewidth=0.8, label=r"$\rho(F)=1$")
    if metric_key in {"sum_power_w", "total_power_w", "power_total"} or ("power" in y_label.lower() and "(w)" in y_label.lower()):
        _, top = ax.get_ylim()
        finite_y = pd.to_numeric(curve_df.get("mean_metric", pd.Series(dtype=float)), errors="coerce").dropna()
        if not finite_y.empty:
            top = max(float(top), float(finite_y.max()) * 1.08)
        ax.set_ylim(bottom=0.0, top=top)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, linestyle=":", linewidth=0.45, alpha=0.45)
    _style_axis(ax)
    num_methods = int(curve_df["method"].nunique()) if "method" in curve_df.columns else 1
    _finish_ieee_figure(fig, ax, legend=True, ncol=2 if num_methods > 4 else 1)
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return {
        "figure_id": figure_id,
        "filename_png": png_path.name,
        "filename_pdf": pdf_path.name,
        "png_path": str(png_path),
        "pdf_path": str(pdf_path),
        "draft_or_final": "final" if paper_ready else "draft",
        "paper_ready": paper_ready,
        "chart_type": chart_type,
        "display_policy_note": str(figure_spec.get("display_policy_note", "")),
        "x_axis_param": _get_figure_sweep_param(figure_spec),
        "required_sweep": _get_figure_required_sweep(figure_spec),
        "y_metric": _get_figure_metric_name(figure_spec),
        "num_x_points": int(curve_df[sort_col].nunique()) if sort_col in curve_df.columns else 0,
        "methods": ordered_methods,
        "method_display_names_short": {
            str(method): _method_display_name(plan, str(method), long=False)
            for method in ordered_methods
        },
        "method_display_names_long": {
            str(method): _method_display_name(plan, str(method), long=True)
            for method in ordered_methods
        },
        "has_error_bars": has_error_bars,
        "error_display": error_display,
        "error_display_label": _error_display_label(error_display),
    }


def render_grouped_bar_figure(
    curve_df: pd.DataFrame,
    figure_spec: dict[str, Any],
    output_dir: str | Path,
    paper_ready: bool,
    *,
    plan: dict[str, Any],
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_id = figure_spec.get("figure_id", "figure")
    png_path, pdf_path = _prepare_figure_paths(output_dir, figure_id, paper_ready)
    y_label = _safe_display(figure_spec.get("metric", {}).get("display_name", _get_figure_metric_name(figure_spec)))
    x_label = _get_figure_x_display(figure_spec)
    error_display = str(figure_spec.get("error_display", "none")).strip().lower()
    fig, ax = plt.subplots(figsize=(IEEE_SINGLE_COLUMN_WIDTH_IN, IEEE_COMPACT_HEIGHT_IN), dpi=300)
    categories = [str(v) for v in curve_df["category"].dropna().astype(str).unique().tolist()] if "category" in curve_df.columns else []
    methods = sorted(curve_df["method"].astype(str).unique().tolist())
    if not categories or not methods:
        raise ValueError(f"insufficient_grouped_bar_data_for_{figure_id}")
    x = np.arange(len(categories), dtype=float)
    width = min(0.34, 0.72 / max(len(methods), 1))
    has_error_bars = False
    for idx, method in enumerate(methods):
        group = curve_df[curve_df["method"].astype(str) == method].copy()
        group["category"] = group["category"].astype(str)
        group = group.set_index("category").reindex(categories)
        heights = pd.to_numeric(group["mean_metric"], errors="coerce").fillna(np.nan).to_numpy(dtype=float)
        error_col = {"ci95": "ci95_metric", "stderr": "stderr_metric", "std": "std_metric"}.get(error_display, "")
        yerr = None
        if error_col and error_col in group.columns and pd.notna(group[error_col]).sum() >= 1 and pd.to_numeric(group["count"], errors="coerce").fillna(0).min() >= 2:
            yerr = pd.to_numeric(group[error_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            has_error_bars = True
        ax.bar(
            x + (idx - (len(methods) - 1) / 2.0) * width,
            heights,
            width=width,
            color=FIGURE_COLORS[idx % len(FIGURE_COLORS)],
            edgecolor="white",
            linewidth=0.5,
            label=_method_display_name(plan, method, long=False),
            yerr=yerr,
            capsize=2.0 if yerr is not None else 0.0,
            error_kw={"elinewidth": 0.65, "capthick": 0.65},
        )
    ax.set_xticks(x)
    rotation = 25 if any(len(_format_category_label(v)) > 8 for v in categories) else 0
    ax.set_xticklabels([_format_category_label(v) for v in categories], rotation=rotation, ha="right" if rotation else "center")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.45, alpha=0.45)
    _style_axis(ax)
    _finish_ieee_figure(fig, ax, legend=True, ncol=1)
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return {
        "figure_id": figure_id,
        "filename_png": png_path.name,
        "filename_pdf": pdf_path.name,
        "png_path": str(png_path),
        "pdf_path": str(pdf_path),
        "draft_or_final": "final" if paper_ready else "draft",
        "paper_ready": paper_ready,
        "x_axis_param": _get_figure_sweep_param(figure_spec),
        "required_sweep": _get_figure_required_sweep(figure_spec),
        "y_metric": _get_figure_metric_name(figure_spec),
        "num_x_points": len(categories),
        "methods": methods,
        "method_display_names_short": {method: _method_display_name(plan, method, long=False) for method in methods},
        "method_display_names_long": {method: _method_display_name(plan, method, long=True) for method in methods},
        "has_error_bars": has_error_bars,
        "error_display": error_display,
        "error_display_label": _error_display_label(error_display),
    }


def render_box_figure(
    curve_df: pd.DataFrame,
    figure_spec: dict[str, Any],
    output_dir: str | Path,
    paper_ready: bool,
    *,
    plan: dict[str, Any],
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_id = figure_spec.get("figure_id", "figure")
    png_path, pdf_path = _prepare_figure_paths(output_dir, figure_id, paper_ready)
    y_label = _safe_display(figure_spec.get("metric", {}).get("display_name", _get_figure_metric_name(figure_spec)))
    x_label = _get_figure_x_display(figure_spec)
    fig, ax = plt.subplots(figsize=(IEEE_SINGLE_COLUMN_WIDTH_IN, IEEE_TALL_HEIGHT_IN), dpi=300)
    categories = [str(v) for v in curve_df["category"].dropna().astype(str).unique().tolist()] if "category" in curve_df.columns else []
    methods = sorted(curve_df["method"].astype(str).unique().tolist())
    if not categories or not methods or "sample_values" not in curve_df.columns:
        raise ValueError(f"insufficient_box_data_for_{figure_id}")
    data = []
    positions = []
    colors = []
    base_positions = np.arange(len(categories), dtype=float)
    width = min(0.26, 0.55 / max(len(methods), 1))
    for cat_idx, cat in enumerate(categories):
        for method_idx, method in enumerate(methods):
            row = curve_df[(curve_df["category"].astype(str) == cat) & (curve_df["method"].astype(str) == method)]
            if row.empty:
                continue
            sample_values = row.iloc[0].get("sample_values", [])
            if not sample_values:
                continue
            data.append(sample_values)
            positions.append(base_positions[cat_idx] + (method_idx - (len(methods) - 1) / 2.0) * width)
            colors.append(FIGURE_COLORS[method_idx % len(FIGURE_COLORS)])
    if not data:
        raise ValueError(f"empty_box_samples_for_{figure_id}")
    box = ax.boxplot(
        data,
        positions=positions,
        widths=width * 0.78,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#111111", "linewidth": 0.8},
        whiskerprops={"linewidth": 0.65},
        capprops={"linewidth": 0.65},
    )
    for idx, patch in enumerate(box.get("boxes", [])):
        patch.set_facecolor(colors[idx % len(colors)])
        patch.set_alpha(0.68)
        patch.set_edgecolor("#333333")
        patch.set_linewidth(0.55)
    ax.set_xticks(base_positions)
    rotation = 25 if any(len(_format_category_label(v)) > 8 for v in categories) else 0
    ax.set_xticklabels([_format_category_label(v) for v in categories], rotation=rotation, ha="right" if rotation else "center")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.45, alpha=0.45)
    _style_axis(ax)
    handles = [
        plt.Line2D([0], [0], color=FIGURE_COLORS[idx % len(FIGURE_COLORS)], marker="s", linestyle="none", markersize=5)
        for idx, _method in enumerate(methods)
    ]
    labels = [_method_display_name(plan, method, long=False) for method in methods]
    ax.legend(handles, labels, frameon=True, fancybox=False, edgecolor="#b8b8b8", facecolor="white", framealpha=0.92, fontsize=7, loc="best")
    _finish_ieee_figure(fig, ax, legend=False)
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return {
        "figure_id": figure_id,
        "filename_png": png_path.name,
        "filename_pdf": pdf_path.name,
        "png_path": str(png_path),
        "pdf_path": str(pdf_path),
        "draft_or_final": "final" if paper_ready else "draft",
        "paper_ready": paper_ready,
        "x_axis_param": _get_figure_sweep_param(figure_spec),
        "y_metric": _get_figure_metric_name(figure_spec),
        "num_x_points": len(categories),
        "methods": methods,
        "method_display_names_short": {method: _method_display_name(plan, method, long=False) for method in methods},
        "method_display_names_long": {method: _method_display_name(plan, method, long=True) for method in methods},
        "has_error_bars": False,
    }


def render_heatmap_figure(
    curve_df: pd.DataFrame,
    figure_spec: dict[str, Any],
    output_dir: str | Path,
    paper_ready: bool,
    *,
    plan: dict[str, Any],
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_id = figure_spec.get("figure_id", "figure")
    png_path, pdf_path = _prepare_figure_paths(output_dir, figure_id, paper_ready)
    if "grid_x" not in curve_df.columns or "grid_y" not in curve_df.columns:
        raise ValueError(f"insufficient_heatmap_data_for_{figure_id}")
    metric_name = _safe_display(figure_spec.get("metric", {}).get("display_name", _get_figure_metric_name(figure_spec)))
    x_label = _get_figure_x_display(figure_spec)
    y_label = _safe_display(figure_spec.get("encoding", {}).get("facet", {}).get("display_name", "Grid Y"))
    pivot = curve_df.pivot_table(index="grid_y", columns="grid_x", values="mean_metric", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(IEEE_SINGLE_COLUMN_WIDTH_IN, IEEE_TALL_HEIGHT_IN), dpi=300)
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", origin="lower")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([_safe_display(v) for v in pivot.columns.tolist()], fontsize=7)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([_safe_display(v) for v in pivot.index.tolist()], fontsize=7)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(metric_name, fontsize=7)
    cbar.ax.tick_params(labelsize=7)
    _style_axis(ax)
    _finish_ieee_figure(fig, ax, legend=False)
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    methods = sorted(curve_df["method"].astype(str).unique().tolist()) if "method" in curve_df.columns else []
    return {
        "figure_id": figure_id,
        "filename_png": png_path.name,
        "filename_pdf": pdf_path.name,
        "png_path": str(png_path),
        "pdf_path": str(pdf_path),
        "draft_or_final": "final" if paper_ready else "draft",
        "paper_ready": paper_ready,
        "x_axis_param": _get_figure_sweep_param(figure_spec),
        "y_metric": _get_figure_metric_name(figure_spec),
        "num_x_points": int(len(pivot.columns)),
        "methods": methods,
        "method_display_names_short": {method: _method_display_name(plan, method, long=False) for method in methods},
        "method_display_names_long": {method: _method_display_name(plan, method, long=True) for method in methods},
        "has_error_bars": False,
    }


def _line_curve_is_locally_noisy(curve_df: pd.DataFrame) -> bool:
    """Detect when a solid line would overstate noisy Monte Carlo means."""
    if curve_df.empty or "method" not in curve_df.columns or "mean_metric" not in curve_df.columns:
        return False
    sort_col = "x_value" if "x_value" in curve_df.columns else "swept_value"
    if sort_col not in curve_df.columns:
        return False
    for _, group in curve_df.groupby(curve_df["method"].astype(str)):
        ordered = group.sort_values(sort_col)
        x = pd.to_numeric(ordered[sort_col], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(ordered["mean_metric"], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        if len(np.unique(x)) < 8 or len(y) < 8:
            continue
        diffs = np.diff(y)
        nonzero = diffs[np.abs(diffs) > 1e-9]
        if len(nonzero) < 5:
            continue
        signs = np.sign(nonzero)
        reversals = int(np.sum(signs[1:] * signs[:-1] < 0))
        reversal_rate = reversals / max(1, len(signs) - 1)
        y_range = float(np.nanmax(y) - np.nanmin(y))
        if y_range <= max(1e-9, 0.03 * max(1.0, abs(float(np.nanmean(y))))):
            continue
        if reversals >= 3 and reversal_rate >= 0.30:
            return True
    return False


def _maybe_use_scatter_trend_for_noisy_line(curve_df: pd.DataFrame, figure_spec: dict[str, Any]) -> dict[str, Any]:
    chart_type = str(figure_spec.get("chart_type", "line")).strip().lower()
    if chart_type != "line":
        return figure_spec
    if str(figure_spec.get("force_line", "")).strip().lower() in {"1", "true", "yes"}:
        return figure_spec
    if _figure_intent(figure_spec) in {"convergence", "algorithm_convergence"}:
        return figure_spec
    if not _line_curve_is_locally_noisy(curve_df):
        return figure_spec
    updated = dict(figure_spec)
    updated["chart_type"] = "scatter_trend"
    updated["trend_guide"] = True
    updated["display_mode"] = "scatter_with_trend_guide"
    updated["display_policy_note"] = (
        "Connected line replaced by scatter plus dashed trend guide because local Monte Carlo wiggles would overstate point-to-point continuity."
    )
    return updated


def render_figure(curve_df: pd.DataFrame, figure_spec: dict[str, Any], output_dir: str | Path, paper_ready: bool, *, plan: dict[str, Any]) -> dict[str, Any]:
    figure_spec = dict(figure_spec)
    if paper_ready:
        if str(figure_spec.get("error_display", "")).strip().lower() in {"", "ci95", "stderr", "std", "iqr", "box"}:
            figure_spec["error_display"] = "none"
        figure_spec["force_line"] = True
        if str(figure_spec.get("chart_type", "")).strip().lower() in {"scatter", "scatter_trend", "scatter_with_trend"}:
            figure_spec["chart_type"] = "line"
            figure_spec["trend_guide"] = False
            figure_spec["display_mode"] = "line_no_error_bars"
        figure_spec["display_policy_note"] = "Paper-ready clean curve; seed stability diagnostics are stored in JSON/CSV, not plotted as uncertainty graphics."
    figure_spec = _maybe_use_scatter_trend_for_noisy_line(curve_df, figure_spec)
    chart_type = str(figure_spec.get("chart_type", "line"))
    if chart_type in {"line", "scatter", "scatter_trend", "scatter_with_trend", "convergence"}:
        return render_line_figure(curve_df, figure_spec, output_dir, paper_ready, plan=plan)
    if chart_type in {"grouped_bar", "bar", "categorical_summary", "ablation_bar"}:
        return render_grouped_bar_figure(curve_df, figure_spec, output_dir, paper_ready, plan=plan)
    if chart_type == "box":
        return render_box_figure(curve_df, figure_spec, output_dir, paper_ready, plan=plan)
    if chart_type == "heatmap":
        return render_heatmap_figure(curve_df, figure_spec, output_dir, paper_ready, plan=plan)
    raise ValueError(f"unsupported_chart_type_for_rendering:{chart_type}")


def _metric_display_name(plan: dict[str, Any], metric: str) -> str:
    primary = plan.get("primary_metric", {}) if isinstance(plan.get("primary_metric"), dict) else {}
    if str(primary.get("name", "")) == metric:
        return _safe_display(str(primary.get("display_name") or metric))
    if metric in {"objective", "objective_value"}:
        primary_name = str(primary.get("name", "")).strip()
        if primary_name in {"sum_power_W", "sum_power_w"}:
            return "Sum transmit power (W)"
        if primary_name in {"sum_power_dBm", "sum_power_dbm"}:
            return "Sum transmit power (dBm)"
        return "Objective value"
    for item in plan.get("secondary_metrics", []):
        if isinstance(item, dict) and str(item.get("name", "")) == metric:
            return _safe_display(str(item.get("display_name") or metric))
    display_map = {
        "objective": "Objective value",
        "objective_value": "Objective value",
        "rate_bpsHz": "Rate (bps/Hz)",
        "weighted_sum_rate_bpsHz": "Weighted sum-rate (bps/Hz)",
        "sum_rate_bpsHz": "Sum-rate (bps/Hz)",
        "sum_rate_bps_hz": "Sum-rate (bps/Hz)",
        "min_user_rate_bpsHz": "Minimum user rate (bps/Hz)",
        "max_user_rate_bpsHz": "Maximum user rate (bps/Hz)",
        "rate_fairness_jain_index": "Jain fairness index",
        "total_runtime_ms": "Total runtime (ms)",
        "w_update_time_ms": r"$\mathbf{w}$-update time (ms)",
        "gamma_update_time_ms": r"$\gamma$-update time (ms)",
        "solve_time_ms": "Solver time (ms)",
        "fp_fixed_point_gap": "FP fixed-point gap",
        "A_gamma_min_eigenvalue": r"Min. eigenvalue of $A(\gamma)$",
        "per_antenna_violation_max_dB": "Max. per-antenna violation (dB)",
        "per_antenna_violation_linear_max": "Max. per-antenna violation",
        "lambda_star_active_count": "Active dual constraints",
        "spectral_efficiency": "Spectral efficiency (bps/Hz)",
        "radar_SNR_dB": "Radar SNR (dB)",
        "radar_snr_dB": "Radar SNR (dB)",
        "radar_snr": "Radar SNR",
        "sensing_gain": "Sensing gain",
        "sensing_metric": "Sensing metric",
        "sensing_beampattern_gain": "Sensing beampattern gain",
        "harvested_power_mW": "Harvested power (mW)",
        "harvested_energy_mW": "Harvested energy (mW)",
        "true_harvested_energy_mW": "True harvested energy (mW)",
        "feasible": "Feasibility",
        "R_c": "Rate",
        "R_c_bpsHz": "Rate (bps/Hz)",
        "tr_CRB": "tr(CRB)",
        "crb_trace": "tr(CRB)",
        "P_EH_total": "Harvested power",
        "P_EH_mW": "Harvested power (mW)",
        "eh_total_mW": "Harvested power (mW)",
        "P_in_actual_mW": "EH input power (mW)",
        "optimal_rho": r"Structural separation $\rho$",
        "rho": r"Structural separation $\rho$",
        "P_EH_per_element": "EH per element",
        "sensing_power_ratio": "Sensing power ratio",
        "constraint_violation": "Constraint violation",
        "constraint_violation_C2": "C2 violation",
        "constraint_violation_C3": "C3 violation",
        "constraint_violation_C7": "C7 violation",
        "rank_W": "rank(W)",
        "SDR_rank_gap": "SDR rank gap",
        "power_consumption": "Power",
        "total_power": "Power",
    }
    return display_map.get(metric, metric.replace("_", " "))


def _available_table_metric_names(comparison_df: pd.DataFrame) -> list[str]:
    metrics: list[str] = []
    for col in comparison_df.columns:
        if not col.startswith("proposed_"):
            continue
        metric = col[len("proposed_") :]
        if f"baseline_{metric}" not in comparison_df.columns:
            continue
        if metric in NON_METRIC_COLUMNS or metric in TABLE_METADATA_COLUMNS:
            continue
        if metric in {"status", "success", "finite_primary_metric"}:
            continue
        if pd.api.types.is_numeric_dtype(comparison_df[col]) or pd.api.types.is_bool_dtype(comparison_df[col]):
            metrics.append(metric)
    return list(dict.fromkeys(metrics))


def _table_columns_from_spec(
    comparison_df: pd.DataFrame,
    table_spec: dict[str, Any],
    plan: dict[str, Any],
) -> list[str]:
    requested = [str(item) for item in table_spec.get("columns", []) if str(item).strip()]
    available_metrics = _available_table_metric_names(comparison_df)
    primary_metric = str(plan.get("primary_metric", {}).get("name", "objective"))
    secondary_metrics = [
        str(item.get("name"))
        for item in plan.get("secondary_metrics", [])
        if isinstance(item, dict) and str(item.get("name", "")) in available_metrics and str(item.get("name", "")) != "feasible"
    ]
    columns = list(requested)
    if not columns:
        columns = ["scenario", f"proposed_{primary_metric}_mean", f"baseline_{primary_metric}_mean", "relative_gain_percent"]
    if "scenario" not in columns:
        columns.insert(0, "scenario")
    if "relative_gain_percent" not in columns and primary_metric in available_metrics:
        columns.append("relative_gain_percent")
    for metric in secondary_metrics:
        for col in (f"proposed_{metric}_mean", f"baseline_{metric}_mean"):
            if col not in columns:
                columns.append(col)
    for col in ("proposed_feasibility_rate", "baseline_feasibility_rate"):
        if col not in columns and "feasible" in available_metrics:
            columns.append(col)
    cleaned: list[str] = []
    for col in columns:
        if col in {"proposed_metric", "baseline_metric"}:
            metric = primary_metric
            col = col.replace("_metric", f"_{metric}_mean")
        if col == "scenario" or col == "relative_gain_percent" or col.endswith("_feasibility_rate"):
            cleaned.append(col)
            continue
        parsed = _parse_table_metric_column(col)
        if parsed is None:
            continue
        _, metric, _ = parsed
        if metric in available_metrics:
            cleaned.append(col)
    deduped = list(dict.fromkeys(cleaned))
    if len(deduped) <= 10:
        return deduped

    compact: list[str] = []
    for col in ["scenario", f"proposed_{primary_metric}_mean", f"baseline_{primary_metric}_mean", "relative_gain_percent", "proposed_feasibility_rate", "baseline_feasibility_rate"]:
        if col in deduped and col not in compact:
            compact.append(col)
    metric_pairs: list[str] = []
    for metric in secondary_metrics:
        for col in (f"proposed_{metric}_mean", f"baseline_{metric}_mean"):
            if col in deduped and col not in compact and col not in metric_pairs:
                metric_pairs.append(col)
        if len(metric_pairs) >= 6:
            break
    compact.extend(metric_pairs[:6])
    return compact


def _parse_table_metric_column(column: str) -> tuple[str, str, str] | None:
    for prefix in ("proposed_", "baseline_"):
        if column.startswith(prefix) and column.endswith("_mean"):
            return prefix[:-1], column[len(prefix) : -len("_mean")], "mean"
    return None


def _table_display_column(column: str, plan: dict[str, Any]) -> str:
    proposed_short = _method_display_name(plan, "proposed", long=False)
    baseline_id = str(plan.get("_active_baseline_method") or "baseline")
    baseline_short = _method_display_name(plan, baseline_id, long=False)
    if column == "scenario":
        return "Scenario"
    if column == "relative_gain_percent":
        if str(plan.get("_primary_claim_mode") or "") == "optimal_reference_equivalence":
            return f"Optimality gap to {baseline_short} [%]"
        return "Primary improvement [%]"
    if column == "proposed_feasibility_rate":
        return f"{proposed_short} feasibility"
    if column == "baseline_feasibility_rate":
        return f"{baseline_short} feasibility"
    parsed = _parse_table_metric_column(column)
    if parsed:
        method, metric, _ = parsed
        method_label = proposed_short if method == "proposed" else baseline_short
        return f"{method_label} {_metric_display_name(plan, metric)}"
    return column.replace("_", " ")


def _table_display_row_key(value: Any) -> Any:
    text = str(value)
    lower = text.lower()
    if lower in {"constraints.e_min_mw", "e_min_mw"}:
        return r"$E_{\min}$ (mW) sweep"
    if lower in {"optimization.lambda_s", "lambda_s"}:
        return r"$\lambda_s$ sweep"
    if "pmax" in lower:
        return "Pmax_dBm sweep"
    if "n_eh_fraction" in lower or "eh_fraction" in lower:
        return "N_EH_fraction sweep"
    cleaned = re.sub(r"(?i)_paper_sweep$", "", text).replace("system.", "").replace("RIS.", "")
    return _safe_display(cleaned).replace("_", " ")


def render_table(
    comparison_df: pd.DataFrame,
    table_spec: dict[str, Any],
    output_dir: str | Path,
    *,
    plan: dict[str, Any],
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table_id = table_spec.get("table_id", "table_1")
    group_by = table_spec.get("group_by", "")
    if group_by and group_by in comparison_df.columns:
        grouped = comparison_df.groupby(group_by, dropna=False)
    else:
        grouped = [("overall", comparison_df)]
    requested_columns = _table_columns_from_spec(comparison_df, table_spec, plan)
    rows: list[dict[str, Any]] = []
    for key, group in grouped:
        if isinstance(group, pd.DataFrame) and not group.empty:
            row: dict[str, Any] = {}
            comparable_group = group[group["comparable"].astype(bool)] if "comparable" in group.columns else group
            for column in requested_columns:
                if column == "scenario":
                    row[column] = _table_display_row_key(key)
                elif column == "relative_gain_percent":
                    source = comparable_group if not comparable_group.empty else group
                    if "relative_gain" in source.columns:
                        values = pd.to_numeric(source.get("relative_gain"), errors="coerce")
                        if str(plan.get("_primary_claim_mode") or "") == "optimal_reference_equivalence":
                            values = values.abs()
                        row[column] = float(values.mean() * 100.0)
                    else:
                        row[column] = None
                elif column == "proposed_feasibility_rate":
                    row[column] = float(group["proposed_feasible"].astype(bool).mean()) if "proposed_feasible" in group.columns else None
                elif column == "baseline_feasibility_rate":
                    row[column] = float(group["baseline_feasible"].astype(bool).mean()) if "baseline_feasible" in group.columns else None
                else:
                    parsed = _parse_table_metric_column(column)
                    if parsed is None:
                        continue
                    method, metric, _ = parsed
                    source_col = f"{method}_{metric}"
                    row[column] = float(pd.to_numeric(group[source_col], errors="coerce").mean()) if source_col in group.columns else None
            rows.append(row)
    table_df = pd.DataFrame(rows)
    rename_map = {column: _table_display_column(column, plan) for column in requested_columns}
    if not table_df.empty:
        table_df = table_df[[col for col in requested_columns if col in table_df.columns]]
        table_df = table_df.rename(columns=rename_map)
        percent_col = rename_map.get("relative_gain_percent", "Primary improvement [%]")
        if percent_col in table_df.columns:
            values = pd.to_numeric(table_df[percent_col], errors="coerce")
            table_df[percent_col] = [
                "" if pd.isna(value) else f"{float(value):.3g}%"
                for value in values
            ]
    csv_path = output_dir / f"{table_id}.csv"
    md_path = output_dir / f"{table_id}.md"
    table_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    md_text = (
        table_df.to_markdown(index=False)
        if not table_df.empty
        else "| " + " | ".join(_table_display_column(col, plan) for col in requested_columns) + " |\n"
        + "| " + " | ".join(["---"] + ["---:" for _ in requested_columns[1:]]) + " |\n"
    )
    md_path.write_text(md_text, encoding="utf-8")
    return {
        "table_id": table_id,
        "filename_csv": csv_path.name,
        "filename_md": md_path.name,
        "csv_path": str(csv_path),
        "md_path": str(md_path),
        "num_rows": int(len(table_df)),
        "status": "generated",
        "display_columns": list(table_df.columns),
    }


def _raw_table_metric_candidates(df: pd.DataFrame, table_spec: dict[str, Any], plan: dict[str, Any]) -> list[str]:
    requested = [str(item).strip() for item in table_spec.get("columns", []) if str(item).strip()]
    normalized_requested: list[str] = []
    for item in requested:
        item = re.sub(r"^(proposed|baseline)_", "", item)
        item = re.sub(r"_(mean|median|std|rate)$", "", item)
        normalized_requested.append(item)
    preferred = [
        *normalized_requested,
    ]
    primary = str(plan.get("primary_metric", {}).get("name", "")).strip()
    if primary:
        preferred.append(primary)
    for item in plan.get("secondary_metrics", []):
        if isinstance(item, dict):
            metric_name = str(item.get("name") or "").strip()
            if metric_name:
                preferred.append(metric_name)
    ignored = set(NON_METRIC_COLUMNS) | set(TABLE_METADATA_COLUMNS) | {
        "method",
        "scenario",
        "scenario_name",
        "swept_param",
        "swept_value",
        "case_id",
        "case_name",
        "seed",
        "feasible",
        "success",
        "status",
    }
    semantic_aliases = {
        "sumratebpshz": "sum_rate_bps_hz",
        "radarsnrdb": "radar_snr_db",
        "maxconstraintviolation": "max_constraint_violation",
        "constraintviolationmax": "max_constraint_violation",
        "optimalrho": "rho",
    }
    seen_semantic_keys: set[str] = set()
    metrics: list[str] = []
    for metric in preferred:
        if not metric or metric in ignored or metric in metrics or metric not in df.columns:
            continue
        if pd.api.types.is_numeric_dtype(df[metric]) or pd.to_numeric(df[metric], errors="coerce").notna().any():
            semantic_key = semantic_aliases.get(re.sub(r"[^a-z0-9]", "", metric.lower()), re.sub(r"[^a-z0-9]", "", metric.lower()))
            if semantic_key in seen_semantic_keys:
                continue
            metrics.append(metric)
            seen_semantic_keys.add(semantic_key)
        if len(metrics) >= 5:
            break
    return metrics


def render_evidence_table(
    results_df: pd.DataFrame,
    table_spec: dict[str, Any],
    output_dir: str | Path,
    *,
    plan: dict[str, Any],
) -> dict[str, Any]:
    """Render a method-by-evidence table from raw validation rows."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table_id = table_spec.get("table_id", "table_1")
    df = results_df.copy()
    if df.empty or "method" not in df.columns:
        raise ValueError("raw_results_missing_method_rows")
    active_figure_ids = [
        str(fig.get("figure_id", "")).strip()
        for fig in plan.get("figure_specs", [])
        if isinstance(fig, dict) and str(fig.get("figure_id", "")).strip()
    ]
    if active_figure_ids:
        df = _filter_rows_for_active_figures(df, active_figure_ids)
    figure_methods: list[str] = []
    for fig in plan.get("figure_specs", []):
        for method in _get_figure_methods(normalize_figure_spec(fig, plan.get("primary_metric", {})), []):
            if method not in figure_methods:
                figure_methods.append(method)
    if figure_methods:
        df = df[df["method"].astype(str).isin(figure_methods)].copy()
    if df.empty:
        raise ValueError("raw_results_empty_after_method_filter")
    metrics = _raw_table_metric_candidates(df, table_spec, plan)
    if not metrics:
        raise ValueError("raw_results_missing_table_metrics")
    group_keys = [key for key in ["swept_param", "method"] if key in df.columns]
    if not group_keys:
        group_keys = ["method"]
    rows: list[dict[str, Any]] = []
    for key, group in df.groupby(group_keys, dropna=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        key_map = dict(zip(group_keys, key_tuple))
        row: dict[str, Any] = {
            "Evidence sweep": _table_display_row_key(key_map.get("swept_param", "overall")),
            "Method": _method_display_name(plan, str(key_map.get("method", "method")), long=False),
            "Feasibility": float(group["feasible"].astype(bool).mean()) if "feasible" in group.columns else None,
            "Runs": int(len(group)),
        }
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[_metric_display_name(plan, metric)] = float(values.mean()) if values.notna().any() else None
        rows.append(row)
    table_df = pd.DataFrame(rows)
    if not table_df.empty:
        sort_cols = [col for col in ["Evidence sweep", "Method"] if col in table_df.columns]
        table_df = table_df.sort_values(sort_cols).reset_index(drop=True)
    csv_path = output_dir / f"{table_id}.csv"
    md_path = output_dir / f"{table_id}.md"
    table_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    md_path.write_text(table_df.to_markdown(index=False) if not table_df.empty else "", encoding="utf-8")
    return {
        "table_id": table_id,
        "filename_csv": csv_path.name,
        "filename_md": md_path.name,
        "csv_path": str(csv_path),
        "md_path": str(md_path),
        "num_rows": int(len(table_df)),
        "status": "generated",
        "table_mode": "method_by_evidence_sweep",
        "display_columns": list(table_df.columns),
    }


def write_phase25_experiment_summary_json(output_path: str | Path, payload: dict[str, Any]) -> None:
    _write_json(Path(output_path), payload)


def _registry_number(value: Any) -> float | int | bool | None:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
    else:
        try:
            numeric = float(str(value).strip())
        except Exception:
            return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _registry_numeric_columns(df: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for column in df.columns:
        if column in {"case_id", "case_name", "scenario_name", "method", "status", "message", "rejection_reason"}:
            continue
        series = df[column]
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            columns.append(str(column))
            continue
        converted = pd.to_numeric(series, errors="coerce")
        if converted.notna().sum() >= max(1, min(3, len(series))):
            columns.append(str(column))
    return columns


def _registry_table_rows(phase25_dir: Path, table_outputs: list[dict[str, Any]], limit: int = 80) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for table_meta in table_outputs:
        if not isinstance(table_meta, dict):
            continue
        csv_path_text = str(table_meta.get("csv_path") or "")
        csv_path = Path(csv_path_text) if csv_path_text else phase25_dir / "tables" / str(table_meta.get("filename_csv") or "")
        if not csv_path.exists():
            continue
        try:
            with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                for index, row in enumerate(reader):
                    cleaned = {str(key).strip(): str(value).strip() for key, value in row.items() if str(key).strip()}
                    rows.append(
                        {
                            "table_id": str(table_meta.get("table_id") or csv_path.stem),
                            "row_index": index,
                            "source": str(csv_path),
                            "values": cleaned,
                        }
                    )
                    if len(rows) >= limit:
                        return rows
        except Exception:
            continue
    return rows


def build_phase25_verified_registry(
    *,
    phase25_dir: Path,
    phase25_status: str,
    plan: dict[str, Any],
    df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    figure_outputs: list[dict[str, Any]],
    table_outputs: list[dict[str, Any]],
    plot_quality_report: dict[str, Any],
    relative_gains: list[float],
) -> dict[str, Any]:
    primary_metric = str((plan.get("primary_metric") or {}).get("name") or "objective")
    numeric_columns = _registry_numeric_columns(df)
    comparison_numeric_columns = _registry_numeric_columns(comparison_df) if not comparison_df.empty else []
    figure_specs = {
        str(item.get("figure_id", "")): item
        for item in plan.get("figure_specs", [])
        if isinstance(item, dict)
    }

    raw_records: list[dict[str, Any]] = []
    sort_columns = [column for column in ["figure_id", "case_id", "swept_param", "swept_value", "method", "seed"] if column in df.columns]
    raw_df = df.sort_values(sort_columns).reset_index(drop=True) if sort_columns and not df.empty else df
    for row_index, row in raw_df.head(250).iterrows():
        metrics: dict[str, Any] = {}
        for column in numeric_columns:
            number = _registry_number(row.get(column))
            if number is not None:
                metrics[column] = number
        raw_records.append(
            {
                "source": "phase2-4/solver/outputs validation results",
                "row_index": int(row_index),
                "case_id": str(row.get("case_id", "")),
                "figure_id": str(row.get("figure_id", "")),
                "method": str(row.get("method", "")),
                "swept_param": str(row.get("swept_param", "")),
                "swept_value": _registry_number(row.get("swept_value")),
                "seed": _registry_number(row.get("seed")),
                "metrics": metrics,
            }
        )

    comparison_records: list[dict[str, Any]] = []
    if not comparison_df.empty:
        sort_columns = [column for column in ["case_id", "swept_param", "swept_value", "baseline_method"] if column in comparison_df.columns]
        comp_df = comparison_df.sort_values(sort_columns).reset_index(drop=True) if sort_columns else comparison_df
        for row_index, row in comp_df.head(250).iterrows():
            metrics: dict[str, Any] = {}
            for column in comparison_numeric_columns:
                number = _registry_number(row.get(column))
                if number is not None:
                    metrics[column] = number
            comparison_records.append(
                {
                    "source": "Phase 2.5 proposed-vs-baseline comparison dataframe",
                    "row_index": int(row_index),
                    "case_id": str(row.get("case_id", "")),
                    "baseline_method": str(row.get("baseline_method", "")),
                    "swept_param": str(row.get("swept_param", "")),
                    "swept_value": _registry_number(row.get("swept_value")),
                    "comparable": bool(row.get("comparable", False)),
                    "metrics": metrics,
                }
            )

    figure_registry: list[dict[str, Any]] = []
    quality_by_id = {
        str(item.get("figure_id", "")): item
        for item in plot_quality_report.get("figures", [])
        if isinstance(item, dict)
    }
    for figure in figure_outputs:
        figure_id = str(figure.get("figure_id", ""))
        spec = figure_specs.get(figure_id, {})
        metric = spec.get("metric", {}) if isinstance(spec, dict) else {}
        encoding = spec.get("encoding", {}) if isinstance(spec, dict) else {}
        x_encoding = encoding.get("x", {}) if isinstance(encoding, dict) else {}
        figure_registry.append(
            {
                "figure_id": figure_id,
                "paper_ready": bool(figure.get("paper_ready", False)),
                "chart_type": str(figure.get("chart_type") or spec.get("chart_type") or ""),
                "claim_id": str(spec.get("claim_id") or spec.get("claim") or spec.get("paper_claim_id") or ""),
                "chart_intent": str(spec.get("chart_intent") or spec.get("intent") or ""),
                "x_axis_param": str(x_encoding.get("sweep_param") or x_encoding.get("field") or figure.get("x_axis_param") or ""),
                "y_metric": str(metric.get("name") or figure.get("y_metric") or ""),
                "methods": list(figure.get("methods", [])) if isinstance(figure.get("methods"), list) else [],
                "blocking_issues": list(figure.get("blocking_issues", [])) if isinstance(figure.get("blocking_issues"), list) else [],
                "quality": quality_by_id.get(figure_id, {}),
                "source_pdf": str(figure.get("pdf_path") or figure.get("filename_pdf") or ""),
            }
        )

    gain_summary = {
        "count": int(len(relative_gains)),
        "mean": float(sum(relative_gains) / len(relative_gains)) if relative_gains else 0.0,
        "median": float(statistics.median(relative_gains)) if relative_gains else 0.0,
        "min": float(min(relative_gains)) if relative_gains else 0.0,
        "max": float(max(relative_gains)) if relative_gains else 0.0,
    }

    return {
        "registry_schema": "phase25_verified_registry.v1",
        "status": "verified_experiment_registry",
        "phase25_status": phase25_status,
        "source_policy": (
            "Downstream paper phases may quote only numbers present in summary_numbers, "
            "table_rows, figure_records, comparison_records, or raw_result_records. "
            "If a desired number is absent, write qualitative text or rerun Phase 2.4/5."
        ),
        "primary_metric": plan.get("primary_metric", {}),
        "methods": plan.get("compared_methods", []),
        "paper_claims_to_test": plan.get("paper_claims_to_test", []),
        "summary_numbers": {
            "num_raw_records": int(len(df)),
            "num_comparison_records": int(len(comparison_df)),
            "relative_gain": gain_summary,
        },
        "figures": figure_registry,
        "tables": table_outputs,
        "table_rows": _registry_table_rows(phase25_dir, table_outputs),
        "comparison_records": comparison_records,
        "raw_result_records": raw_records,
        "allowed_numeric_columns": {
            "raw_results": numeric_columns,
            "comparison": comparison_numeric_columns,
            "primary_metric": primary_metric,
        },
    }


def write_plot_quality_report(phase25_dir: Path, report: dict[str, Any]) -> None:
    _write_json(phase25_dir / "plot_quality_report.json", report)
    lines = [f"# Plot Quality Report", "", f"- overall_status: {report.get('overall_status', 'unknown')}", ""]
    for item in report.get("figures", []):
        lines.extend(
            [
                f"## {item.get('figure_id', 'figure')}",
                f"- chart_type: {item.get('chart_type', '')}",
                f"- x_axis_param: {item.get('x_axis_param', '')}",
                f"- y_metric: {item.get('y_metric', '')}",
                f"- num_x_points: {item.get('num_x_points', 0)}",
                f"- x_values: {item.get('x_values', [])}",
                f"- all_requested_x_values: {item.get('all_requested_x_values', [])}",
                f"- seeds_per_point_summary: {item.get('seeds_per_point_summary', {})}",
                f"- methods_present: {item.get('methods_present', [])}",
                f"- has_error_bars: {item.get('has_error_bars', False)}",
                f"- monte_carlo_valid: {item.get('monte_carlo_valid', False)}",
                f"- deterministic_boundary_valid: {item.get('deterministic_boundary_valid', False)}",
                f"- deterministic_sweep_valid: {item.get('deterministic_sweep_valid', False)}",
                f"- paper_ready: {item.get('paper_ready', False)}",
                f"- blocking_issues: {item.get('blocking_issues', [])}",
                f"- warnings: {item.get('warnings', [])}",
                "",
            ]
        )
    (phase25_dir / "plot_quality_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_method_naming_summary(phase25_dir: Path, plan: dict[str, Any]) -> dict[str, str]:
    maps = _build_method_naming_maps(plan)
    records: list[dict[str, Any]] = []
    for record in maps["records"]:
        item = dict(record)
        item["where_used"] = [
            "figure legends",
            "table labels",
            "captions",
            "result summary",
        ]
        records.append(item)
    payload = {"methods": records}
    json_path = phase25_dir / "method_naming_summary.json"
    md_path = phase25_dir / "method_naming_summary.md"
    _write_json(json_path, payload)
    lines = ["# Method Naming Summary", ""]
    for item in records:
        lines.extend(
            [
                f"## {item['internal_name']}",
                f"- role: {item['role']}",
                f"- display_name_short: {item['display_name_short']}",
                f"- display_name_long: {item['display_name_long']}",
                f"- source_of_name: {item['source_of_name']}",
                f"- where_used: {item['where_used']}",
                "",
            ]
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {"json_path": str(json_path), "md_path": str(md_path)}


def write_caption_files(
    phase25_dir: Path,
    plan: dict[str, Any],
    figure_outputs: list[dict[str, Any]],
    table_outputs: list[dict[str, Any]],
) -> dict[str, str]:
    def caption_display(value: Any) -> str:
        text = str(value or "").replace("\n", " ").strip()
        return _safe_display(text)

    def caption_quantity_phrase(label: str, metric_name: str = "") -> str:
        compact = re.sub(r"\s+", "", str(label or ""))
        metric = str(metric_name or "").strip()
        if metric == "worst_case_utility" or compact == r"$\psi$":
            return r"worst-case weighted utility $\psi$"
        if compact in {r"$R_{\mathrm{sum}}$", r"$R_{\rm sum}$"} or metric in {"sum_rate_bpsHz", "weighted_sum_rate_bpsHz"}:
            return f"sum rate {label}" if "$" in str(label or "") else "sum rate"
        if compact in {r"$P_{\mathrm{DC}}$", r"$P_{\rm DC}$"} or metric in {"harvested_energy_mW", "true_harvested_energy_mW", "sum_harvested_energy_mW"}:
            return f"harvested DC power {label}" if "$" in str(label or "") else "harvested DC power"
        return label

    def caption_parameter_phrase(label: str, sweep_param: str = "") -> str:
        compact = re.sub(r"\s+", "", str(label or ""))
        sweep = str(sweep_param or "").strip().lower()
        if compact == r"$P_{\max}$" or "pmax" in sweep or "power" in sweep:
            return r"transmit power budget $P_{\max}$"
        if compact == r"$b$" or "rectifier_b" in sweep or "steepness" in sweep or sweep.endswith(".b"):
            return r"rectifier steepness $b$"
        if compact == r"$\epsilon$" or "epsilon" in sweep:
            return r"outage tolerance $\epsilon$"
        if compact == r"$\beta/\alpha$" or "beta" in sweep or "alpha" in sweep:
            return r"EH-rate weight ratio $\beta/\alpha$"
        if compact in {r"$\gamma_{\min}$", r"$\gamma$"} or "gamma" in sweep or "sinr" in sweep:
            return f"SINR target {label}" if label else "SINR target"
        return label

    figure_specs = {str(item.get("figure_id", "")): item for item in plan.get("figure_specs", []) if isinstance(item, dict)}
    figure_lines = ["# Figure Captions", ""]
    for meta in figure_outputs:
        figure_id = str(meta.get("figure_id", "figure"))
        spec = figure_specs.get(figure_id, {})
        chart_type = str(meta.get("chart_type") or spec.get("chart_type", "line"))
        x_encoding = spec.get("encoding", {}).get("x", {}) if isinstance(spec.get("encoding", {}).get("x", {}), dict) else {}
        metric_spec = spec.get("metric", {}) if isinstance(spec.get("metric", {}), dict) else {}
        x_label = caption_parameter_phrase(
            caption_display(x_encoding.get("display_name", meta.get("x_axis_param", "parameter"))),
            str(x_encoding.get("sweep_param") or meta.get("x_axis_param") or ""),
        )
        y_label = caption_quantity_phrase(
            caption_display(metric_spec.get("display_name", meta.get("y_metric", "metric"))),
            str(metric_spec.get("name") or meta.get("y_metric") or ""),
        )
        context = caption_display(
            spec.get("caption_context")
            or spec.get("fixed_parameter_caption")
            or spec.get("setting")
            or spec.get("scenario")
            or ""
        ).strip().rstrip(".")
        if chart_type in {"line", "scatter", "scatter_trend", "scatter_with_trend"}:
            prefix = "Corresponding " if figure_id.endswith("_2") or figure_id == "figure_2" else ""
            caption_body = f"{prefix}{y_label} versus {x_label}"
            if not prefix and caption_body:
                caption_body = caption_body[0].upper() + caption_body[1:]
        elif chart_type in {"grouped_bar", "bar", "categorical_summary", "ablation_bar"}:
            caption_body = f"{y_label} under different {x_label}"
        elif chart_type == "box":
            caption_body = f"Distribution of {y_label} under different {x_label}"
        elif chart_type == "convergence":
            caption_body = f"Convergence behavior of {y_label}"
        elif chart_type == "heatmap":
            caption_body = f"{y_label} over the selected two-dimensional parameter grid"
        else:
            caption_body = f"{y_label} under the selected experimental setting"
        caption = f"Fig. {figure_id.split('_')[-1]}. {caption_body}"
        if context:
            caption += f", where {context}"
        caption += "."
        figure_lines.extend([f"## {figure_id}", caption, ""])
    figure_path = phase25_dir / "figure_captions.md"
    figure_path.write_text("\n".join(figure_lines), encoding="utf-8")

    table_lines = ["# Table Captions", ""]
    methods = plan.get("compared_methods", [])
    long_map = {str(item.get("internal_name") or item.get("name")): str(item.get("display_name_long") or item.get("display_name_short") or item.get("name")) for item in methods if isinstance(item, dict)}
    table_spec = plan.get("table_specs", [{}])[0] if plan.get("table_specs") else {}
    compared_long = [long_map.get(str(item.get("internal_name") or item.get("name")), "") for item in methods if isinstance(item, dict)]
    for meta in table_outputs:
        table_id = str(meta.get("table_id", "table_1"))
        caption = (
            f"Table {table_id.split('_')[-1]}. {_safe_display(table_spec.get('purpose', 'Overall performance summary')).capitalize()} "
            f"The table compares {', '.join(name for name in compared_long if name)} "
            f"using consistent short labels in the column headers."
        )
        table_lines.extend([f"## {table_id}", caption, ""])
    table_path = phase25_dir / "table_captions.md"
    table_path.write_text("\n".join(table_lines), encoding="utf-8")
    return {"figure_captions_path": str(figure_path), "table_captions_path": str(table_path)}


def write_missing_experiments(phase25_dir: str | Path, plot_quality_report: dict[str, Any]) -> dict[str, str]:
    phase25_dir = Path(phase25_dir)
    missing_md = phase25_dir / "missing_experiments.md"
    paper_plan = phase25_dir / "paper_sweep_plan.json"
    lines = ["# Missing Experiments", "", "The current Phase 2.4 data are insufficient for paper-ready IEEE WCL figures.", ""]
    global_blocking_issues = plot_quality_report.get("global_blocking_issues", [])
    if isinstance(global_blocking_issues, list) and global_blocking_issues:
        lines.extend(["## Global Blocking Issues", ""])
        lines.extend(f"- {item}" for item in global_blocking_issues)
        lines.append("")
    previous_figures_by_id: dict[str, dict[str, Any]] = {}
    for prior_path in (phase25_dir / "paper_sweep_plan_refined.json", phase25_dir / "paper_sweep_plan.json"):
        if not prior_path.exists():
            continue
        try:
            prior_payload = json.loads(prior_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(prior_payload, dict):
            continue
        prior_figures = prior_payload.get("figures") or prior_payload.get("missing_for_figures") or []
        if not isinstance(prior_figures, list):
            continue
        for prior in prior_figures:
            if not isinstance(prior, dict):
                continue
            figure_id = str(prior.get("figure_id") or prior.get("id") or "").strip()
            if figure_id and figure_id not in previous_figures_by_id:
                previous_figures_by_id[figure_id] = prior

    def numeric_values(raw: Any) -> list[float]:
        if not isinstance(raw, list):
            return []
        values: list[float] = []
        for value in raw:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(numeric):
                values.append(numeric)
        return sorted({float(value) for value in values})

    def preserve_better_prior_coverage(figure_payload: dict[str, Any], *, needs_more_x: bool) -> None:
        figure_id = str(figure_payload.get("figure_id") or "").strip()
        previous = previous_figures_by_id.get(figure_id)
        if not previous:
            return
        current_values = numeric_values(figure_payload.get("suggested_values"))
        previous_values = numeric_values(previous.get("suggested_values") or previous.get("suggested_categories_or_values"))
        if not previous_values:
            return
        current_param = str(figure_payload.get("required_sweep_param") or "").strip()
        previous_param = str(previous.get("required_sweep_param") or previous.get("sweep_param") or "").strip()
        current_sweep = str(figure_payload.get("required_sweep") or "").strip()
        previous_sweep = str(previous.get("required_sweep") or previous.get("required_sweep_id") or "").strip()
        current_target = max(2, int(figure_payload.get("preferred_points") or figure_payload.get("required_min_points") or 2))
        prior_axis_matches_current = (
            not current_param
            or not previous_param
            or previous_param == current_param
        )
        if not prior_axis_matches_current:
            figure_payload["discarded_prior_refined_coverage"] = {
                "used": False,
                "reason": "previous refined Phase 2.5 coverage targeted a different x-axis than the current Phase 2.4 figure contract",
                "previous_required_sweep": previous_sweep,
                "previous_required_sweep_param": previous_param,
                "current_required_sweep": current_sweep,
                "current_required_sweep_param": current_param,
            }
            return
        prior_is_better_coverage = len(previous_values) > len(current_values) or (
            len(previous_values) >= current_target and len(previous_values) >= len(current_values)
        )
        if not (needs_more_x and prior_is_better_coverage):
            return
        for key in (
            "required_sweep",
            "required_sweep_param",
            "suggested_categories_or_values",
            "suggested_values",
            "suggested_num_seeds",
            "scout_values",
            "scout_num_seeds",
            "quick_num_seeds",
            "medium_values",
            "medium_num_seeds",
            "methods_to_run",
            "claim_tested",
        ):
            if key in previous and previous.get(key) not in (None, "", []):
                figure_payload[key] = previous[key]
        if previous_sweep:
            figure_payload["required_sweep"] = previous_sweep
        if previous_param:
            figure_payload["required_sweep_param"] = previous_param
            resolved_sweep = _validation_plan_sweep_id_for_param(phase25_dir, previous_param, previous_sweep)
            if resolved_sweep:
                figure_payload["required_sweep"] = resolved_sweep
        figure_payload["preserved_prior_refined_coverage"] = {
            "used": True,
            "reason": "previous refined Phase 2.5 plan had denser or more specific coverage than the deterministic reanalysis fallback",
            "previous_num_values": len(previous_values),
            "current_num_values_before_preserve": len(current_values),
            "previous_required_sweep": previous_sweep,
            "previous_required_sweep_param": previous_param,
        }

    figures_payload: list[dict[str, Any]] = []
    for item in plot_quality_report.get("figures", []):
        if item.get("paper_ready", False):
            continue
        chart_type = str(item.get("chart_type", "line"))
        raw_x_values = [
            float(v)
            for v in item.get("all_requested_x_values", item.get("x_values", []))
            if isinstance(v, (int, float)) or str(v).replace(".", "", 1).replace("-", "", 1).isdigit()
        ]
        needs_more_x = any(
            issue in item.get("blocking_issues", [])
            for issue in (
                "too_few_x_points",
                "too_few_categories",
                "too_few_effective_x_points_after_feasibility_filter",
                "too_few_medium_x_points_after_filter",
            )
        )
        required_sweep = str(item.get("required_sweep") or "").strip()
        validation_plan_values = _validation_plan_sweep_values(
            phase25_dir,
            str(item.get("x_axis_param", "")),
            required_sweep=required_sweep,
        )
        target_points = int(item.get("suggested_min_x_points", LINE_PAPER_MIN_X_POINTS))
        planned_seed_values = validation_plan_values or raw_x_values
        generated_values = _sanitize_suggested_values(
            str(item.get("x_axis_param", "")),
            _suggest_sweep_values(
                planned_seed_values,
                target_points,
                int(item.get("suggested_preferred_x_points", LINE_PAPER_PREFERRED_X_POINTS)),
            ),
        ) if chart_type in {"line", "scatter", "scatter_trend", "scatter_with_trend", "convergence"} or needs_more_x else item.get("x_values", [])
        if validation_plan_values and len(validation_plan_values) >= target_points:
            suggested_values = validation_plan_values
        else:
            suggested_values = sorted({float(v) for v in list(validation_plan_values) + list(generated_values)}) if generated_values or validation_plan_values else []
        coverage = item.get("paired_success_seed_coverage", {}) if isinstance(item.get("paired_success_seed_coverage", {}), dict) else {}
        reliable_values: list[float] = []
        for value in coverage.get("reliable_x_values", []):
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(numeric):
                reliable_values.append(numeric)
        param_norm = str(item.get("x_axis_param", "")).lower()
        last_token = param_norm.replace("/", ".").split(".")[-1]
        preferred_points = int(item.get("suggested_preferred_x_points", LINE_PAPER_PREFERRED_X_POINTS))
        target_value_count = max(target_points, preferred_points)
        needs_more_effective_points = any(
            issue in item.get("blocking_issues", [])
            for issue in ("too_few_effective_x_points_after_feasibility_filter", "too_few_medium_x_points_after_filter")
        )
        used_reliable_span_densification = False
        if (
            needs_more_effective_points
            and chart_type in {"line", "scatter", "scatter_trend", "scatter_with_trend"}
            and not _is_count_like_sweep_param(param_norm)
        ):
            span_seed_values = reliable_values or raw_x_values
            if len(span_seed_values) >= 2:
                # When the current curve is promising but has too few effective
                # x values, add real simulation points inside the observed
                # operating interval instead of extrapolating into a harder
                # regime that may be irrelevant or infeasible.
                suggested_values = _densify_sweep_values_inside_span(span_seed_values, target_value_count)
                raw_x_values = sorted({float(v) for v in span_seed_values})
                used_reliable_span_densification = True
        if reliable_values and "low_paired_success_rate_per_x" in item.get("blocking_issues", []):
            lower_bound = min(reliable_values)
            upper_bound = max(reliable_values)
            suggested_values = [float(v) for v in suggested_values if lower_bound <= float(v) <= upper_bound]
            raw_x_values = sorted({float(v) for v in reliable_values})
        if raw_x_values and _is_count_like_sweep_param(param_norm):
            positive_existing = [float(v) for v in raw_x_values if float(v) > 0]
            min_existing = min(positive_existing) if positive_existing else min(float(v) for v in raw_x_values)
            suggested_values = [v for v in suggested_values if float(v) >= min_existing]
        metric_norm = str(item.get("y_metric", "")).lower()
        if raw_x_values and any(marker in param_norm for marker in ("gamma", "sinr")) and any(marker in metric_norm for marker in ("power", "objective")):
            # Sum-power curves should stay inside the feasible operating range.
            # Feasibility-boundary evidence can use rho/feasibility metrics instead.
            max_existing = max(float(v) for v in raw_x_values)
            suggested_values = [v for v in suggested_values if float(v) <= max_existing]
            step = statistics.median(
                [raw_x_values[i + 1] - raw_x_values[i] for i in range(len(raw_x_values) - 1) if raw_x_values[i + 1] > raw_x_values[i]]
            ) if len(raw_x_values) >= 2 else 5.0
            numeric_values = sorted({float(v) for v in suggested_values})
            while len(numeric_values) < target_points:
                numeric_values.insert(0, round(numeric_values[0] - step, 6) if numeric_values else round(max_existing - step, 6))
                numeric_values = sorted({float(v) for v in numeric_values if float(v) <= max_existing})
            suggested_values = numeric_values
        if raw_x_values and len(suggested_values) < target_points and any(marker in param_norm for marker in ("lambda", "weight")):
            numeric_values = sorted({float(v) for v in suggested_values})
            if len(numeric_values) >= 2:
                diffs = [numeric_values[i + 1] - numeric_values[i] for i in range(len(numeric_values) - 1) if numeric_values[i + 1] > numeric_values[i]]
                step = statistics.median(diffs) if diffs else max(abs(numeric_values[-1]) * 0.2, 0.5)
            else:
                step = max(abs(numeric_values[0]) * 0.2, 0.5) if numeric_values else 0.5
            while len(numeric_values) < target_points:
                numeric_values.append(round(numeric_values[-1] + step, 6) if numeric_values else 0.0)
                numeric_values = sorted({float(v) for v in numeric_values if v >= 0.0})
            suggested_values = numeric_values
        collapse_info = {}
        semantic_info = item.get("semantic_metric_variation", {})
        if isinstance(semantic_info, dict):
            collapse_info = semantic_info.get("resource_sweep_collapse", {}) if isinstance(semantic_info.get("resource_sweep_collapse", {}), dict) else {}
        if "resource_sweep_nonmonotone_collapse" in item.get("blocking_issues", []) and collapse_info.get("collapse_detected"):
            collapse_from = collapse_info.get("from_x")
            try:
                upper = float(collapse_from)
            except (TypeError, ValueError):
                upper = max(raw_x_values) if raw_x_values else None
            if upper is not None:
                lower_candidates = [float(v) for v in raw_x_values if float(v) <= upper]
                lower = min(lower_candidates) if lower_candidates else min(raw_x_values) if raw_x_values else 0.0
                if upper > lower:
                    suggested_values = [
                        round(float(value), 6)
                        for value in np.linspace(lower, upper, max(target_points, len(lower_candidates), 4)).tolist()
                    ]
        numeric_suggested_values: list[float] = []
        for value in suggested_values:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(numeric):
                numeric_suggested_values.append(numeric)
        suggested_values = _sanitize_suggested_values(str(item.get("x_axis_param", "")), numeric_suggested_values) if numeric_suggested_values else []
        scout_values = _select_scout_values(suggested_values, chart_type)
        medium_values = _select_scout_values(suggested_values, chart_type, target_points=_medium_target_points(chart_type))
        methods = item.get("methods_required", [])
        reason = {
            "line": "Need enough x-axis points and Monte Carlo averaging for a paper-quality trend plot.",
            "scatter": "Need enough x-axis points and Monte Carlo averaging for a paper-quality scatter trend plot.",
            "convergence": "Need more iteration trajectories or repeated runs for a paper-quality convergence figure.",
            "grouped_bar": "Need enough representative categories and Monte Carlo averaging for a paper-quality grouped-bar comparison.",
            "bar": "Need enough representative categories and Monte Carlo averaging for a paper-quality bar comparison.",
            "categorical_summary": "Need enough representative categories and Monte Carlo averaging for a paper-quality categorical summary.",
            "ablation_bar": "Need enough representative ablation categories and Monte Carlo averaging for a paper-quality ablation figure.",
            "box": "Need enough samples per group for a meaningful paper-quality box plot.",
            "heatmap": "Need sufficient two-dimensional grid coverage and Monte Carlo averaging for a paper-quality heatmap.",
        }.get(chart_type, "Need more data coverage and Monte Carlo averaging for a paper-quality figure.")
        if "resource_sweep_nonmonotone_collapse" in item.get("blocking_issues", []):
            reason = (
                "The current resource sweep has a non-monotone collapse in the proposed curve; "
                "rerun a denser pre-collapse operating range and audit the Phase 2.4 solver/contract before using the plot as paper evidence."
            )
        suggested_seed_count = min(
            int(item.get("suggested_preferred_seeds_per_point", PAPER_PREFERRED_SEEDS)),
            PAPER_PREFERRED_SEEDS,
        )
        figure_payload = {
            "figure_id": item.get("figure_id", ""),
            "chart_type": chart_type,
            "required_sweep": required_sweep,
            "required_sweep_param": item.get("x_axis_param", ""),
            "current_points": int(item.get("num_x_points", 0)),
            "required_min_points": int(item.get("suggested_min_x_points", LINE_PAPER_MIN_X_POINTS)),
            "preferred_points": int(item.get("suggested_preferred_x_points", LINE_PAPER_PREFERRED_X_POINTS)),
            "current_seeds_per_point": item.get("seeds_per_point_summary", {}),
            "required_min_seeds_per_point": int(item.get("suggested_min_seeds_per_point", PAPER_MINIMUM_SEEDS)),
            "preferred_seeds_per_point": int(item.get("suggested_preferred_seeds_per_point", PAPER_PREFERRED_SEEDS)),
            "suggested_categories_or_values": suggested_values,
            "suggested_values": suggested_values,
            "suggested_num_seeds": suggested_seed_count,
            "scout_values": scout_values,
            "scout_num_seeds": SCOUT_MODE_SEEDS,
            "quick_num_seeds": QUICK_MODE_SEEDS,
            "medium_values": medium_values,
            "medium_num_seeds": MEDIUM_MODE_SEEDS,
            "methods_to_run": methods,
            "reason": reason,
            "two_pass_policy": {
                "scout_first": True,
                "scout_goal": "Run a small set of representative x values with enough seeds per point to identify responsive, stable, high-gain figure candidates before full x-axis expansion.",
                "medium_goal": "After a promising scout result, run a denser medium pass with more x values and seeds to check curve shape before committing to paper-scale Monte Carlo.",
                "promotion_criteria": [
                    "paired feasible proposed-plus-benchmark rows exist at multiple x values",
                    "the plotted physical metric varies across the sweep",
                    "the proposed method shows positive gain over the selected practical benchmark",
                    "no resource-sweep collapse or obvious solver artifact is detected",
                ],
                "expand_after_promotion": "Use medium_values/medium_num_seeds for curve-shape confirmation, then suggested_values/suggested_num_seeds for the promoted final figures.",
            },
            "range_densification_policy": {
                "used": used_reliable_span_densification,
                "rule": "If effective x coverage is insufficient but the current operating interval is reliable, densify within the observed reliable span instead of extrapolating the axis.",
                "basis_values": raw_x_values if used_reliable_span_densification else [],
            },
            "blocking_issues": item.get("blocking_issues", []),
            "current_coverage": {
                "points": int(item.get("num_x_points", 0)),
                "seeds_per_point": item.get("seeds_per_point_summary", {}),
            },
            "missing_coverage": {
                "needs_more_points": "too_few_x_points" in item.get("blocking_issues", []) or "too_few_categories" in item.get("blocking_issues", []),
                "needs_more_effective_points": any(issue in item.get("blocking_issues", []) for issue in ["too_few_effective_x_points_after_feasibility_filter", "too_few_medium_x_points_after_filter"]),
                "needs_more_seeds": "too_few_seeds_per_point" in item.get("blocking_issues", []),
                "needs_monte_carlo_validity": any(issue in item.get("blocking_issues", []) for issue in ["zero_variance_across_seeds", "repeated_identical_outputs_across_seeds", "missing_seed_column"]),
            },
            "suggested_additional_runs": len(suggested_values),
            "estimated_total_cases": int(max(len(suggested_values), 1) * suggested_seed_count),
            "estimated_total_results": int(max(len(suggested_values), 1) * suggested_seed_count * max(len(methods), 1)),
        }
        preserve_better_prior_coverage(figure_payload, needs_more_x=needs_more_x or needs_more_effective_points)
        figures_payload.append(figure_payload)
        lines.extend(
            [
                f"## {figure_payload['figure_id']}",
                f"- required_sweep: {figure_payload['required_sweep']}",
                f"- required_sweep_param: {figure_payload['required_sweep_param']}",
                f"- current_points: {figure_payload['current_points']}",
                f"- required_min_points: {figure_payload['required_min_points']}",
                f"- preferred_points: {figure_payload['preferred_points']}",
                f"- current_seeds_per_point: {figure_payload['current_seeds_per_point']}",
                f"- scout_values: {figure_payload['scout_values']}",
                f"- scout_num_seeds: {figure_payload['scout_num_seeds']}",
                f"- medium_values: {figure_payload['medium_values']}",
                f"- medium_num_seeds: {figure_payload['medium_num_seeds']}",
                f"- suggested_values: {figure_payload['suggested_values']}",
                f"- suggested_num_seeds: {figure_payload['suggested_num_seeds']}",
                f"- quick_num_seeds: {figure_payload['quick_num_seeds']}",
                f"- methods_to_run: {figure_payload['methods_to_run']}",
                f"- blocking_issues: {figure_payload['blocking_issues']}",
                "",
            ]
        )
    missing_md.write_text("\n".join(lines), encoding="utf-8")
    payload = {
        "status": "needs_more_phase24_runs",
        "paper_mode_default": "preferred",
        "quick_mode_available": True,
        "full_run_required_for_paper_ready": True,
        "monte_carlo_policy": {
            "scout_num_seeds": SCOUT_MODE_SEEDS,
            "quick_num_seeds": QUICK_MODE_SEEDS,
            "medium_num_seeds": MEDIUM_MODE_SEEDS,
            "paper_minimum_num_seeds": PAPER_MINIMUM_SEEDS,
            "paper_preferred_num_seeds": PAPER_PREFERRED_SEEDS,
            "high_confidence_num_seeds": HIGH_CONFIDENCE_SEEDS,
        },
        "two_pass_policy": {
            "scout_first": True,
            "scout_mode": "few_x_values_sufficient_seeds_candidate_screening",
            "medium_mode": "moderate_x_axis_and_seed_count_for_curve_shape_confirmation",
            "paper_mode": "expanded_x_axis_and_monte_carlo_for_promoted_figures",
            "selection_rule": "Phase 2.5 promotes figures with feasible paired rows, responsive metrics, positive proposed gain, and no solver-artifact collapse.",
        },
        "figures": figures_payload,
    }
    _write_json(paper_plan, payload)
    return {"missing_experiments_md": str(missing_md), "paper_sweep_plan_json": str(paper_plan)}


def _summary_from_results(results: list[dict[str, Any]], *, objective_higher_is_better: bool = True) -> dict[str, Any]:
    groups: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
    for row in results:
        key = (row.get("case_id", ""), row.get("swept_param", ""), row.get("swept_value", 0.0), row.get("scenario_name", ""), row.get("seed", 0))
        groups.setdefault(key, {})[str(row.get("method", "")).lower()] = row
    comparable: list[dict[str, Any]] = []
    for key, group in groups.items():
        p = group.get("proposed")
        b = group.get("baseline")
        if b is None:
            for method_id, candidate in group.items():
                if method_id != "proposed":
                    b = candidate
                    break
        if p is None or b is None:
            continue
        if not (bool(p.get("feasible", False)) and str(p.get("status", "")).lower() in SUCCESS_STATUSES):
            continue
        if not (bool(b.get("feasible", False)) and str(b.get("status", "")).lower() in SUCCESS_STATUSES):
            continue
        p_obj = float(p.get("objective", 0.0))
        b_obj = float(b.get("objective", 0.0))
        raw_rel = (p_obj - b_obj) / max(abs(b_obj), 1.0e-9)
        rel = raw_rel if objective_higher_is_better else -raw_rel
        proposed_win = p_obj >= b_obj if objective_higher_is_better else p_obj <= b_obj
        comparable.append({"case_id": key[0], "relative_gain": rel, "proposed_win": proposed_win})
    gains = [item["relative_gain"] for item in comparable]
    best = max(comparable, key=lambda item: item["relative_gain"], default=None)
    worst = min(comparable, key=lambda item: item["relative_gain"], default=None)
    return {
        "num_cases": len({(r.get("case_id", ""), r.get("seed", 0)) for r in results}),
        "num_results": len(results),
        "num_failed": sum(1 for r in results if not (bool(r.get("feasible", False)) and str(r.get("status", "")).lower() in SUCCESS_STATUSES)),
        "num_comparable_cases": len(comparable),
        "proposed_win_count": sum(1 for item in comparable if item["proposed_win"]),
        "proposed_win_rate": float(sum(1 for item in comparable if item["proposed_win"]) / len(comparable)) if comparable else 0.0,
        "proposed_mean_relative_gain": float(sum(gains) / len(gains)) if gains else 0.0,
        "proposed_median_relative_gain": float(statistics.median(gains)) if gains else 0.0,
        "best_gain_case_id": str(best["case_id"]) if best else "",
        "worst_gain_case_id": str(worst["case_id"]) if worst else "",
        "all_finite": all(_is_finite_value(r) for r in results),
        "results": results,
    }


def _clone_problem(problem_data_mod, base_problem, *, case_name: str, case_id: str, swept_param: str, swept_value: float, scenario_name: str, updates: dict[str, Any]):
    if hasattr(base_problem, "clone_with"):
        return base_problem.clone_with(
            case_name=case_name,
            case_id=case_id,
            swept_param=swept_param,
            swept_value=swept_value,
            scenario_name=scenario_name,
            updates=updates,
        )
    if hasattr(base_problem, "fields"):
        fields = copy.deepcopy(getattr(base_problem, "fields", {}))
        for key, value in updates.items():
            parts = [part for part in str(key).replace("/", ".").split(".") if part]
            current = fields
            for part in parts[:-1]:
                if not isinstance(current, dict):
                    break
                current = current.setdefault(part, {})
            else:
                if isinstance(current, dict) and parts:
                    current[parts[-1]] = value
        return problem_data_mod.ProblemData(
            fields=fields,
            case_name=case_name,
            case_id=case_id,
            swept_param=swept_param,
            swept_value=swept_value,
            scenario_name=scenario_name,
            validation_plan=copy.deepcopy(getattr(base_problem, "validation_plan", {})),
        )

    base = base_problem
    N = int(updates.get("N", base.N))
    K = int(updates.get("K", base.K))
    alpha = np.asarray(updates.get("alpha", base.alpha), dtype=float)
    q_users = np.asarray(updates.get("q_users", base.q_users), dtype=float)
    R_min = np.asarray(updates.get("R_min", base.R_min), dtype=float)
    if K != len(alpha):
        alpha = np.ones(K, dtype=float) / max(K, 1)
    if K != len(R_min):
        fill = float(np.mean(np.asarray(base.R_min, dtype=float)))
        R_min = np.full(K, fill, dtype=float)
    if K != q_users.shape[0]:
        if K <= q_users.shape[0]:
            q_users = q_users[:K]
        else:
            extra = []
            last = q_users[-1] if len(q_users) else np.array([3.0, 0.0, 0.0], dtype=float)
            for idx in range(q_users.shape[0], K):
                extra.append([float(last[0]), float(last[1] + 0.05 * (idx - q_users.shape[0] + 1)), float(last[2])])
            q_users = np.vstack([q_users, np.asarray(extra, dtype=float)])
    aperture = float(np.max(base.p_nominal[:, 0]) - np.min(base.p_nominal[:, 0])) if np.asarray(base.p_nominal).size else 0.15 * max(N - 1, 1)
    if N == base.p_nominal.shape[0]:
        p_nominal = np.asarray(updates.get("p_nominal", base.p_nominal), dtype=float)
    else:
        p_nominal = np.column_stack((np.linspace(0.0, aperture, N), np.zeros(N), np.zeros(N)))
    problem = problem_data_mod.make_canonical_problem(
        N=N,
        K=K,
        alpha=alpha,
        P_max=float(updates.get("P_max", base.P_max)),
        sigma2=float(updates.get("sigma2", base.sigma2)),
        p_nominal=p_nominal,
        delta=float(updates.get("delta", base.delta)),
        d_min=float(updates.get("d_min", base.d_min)),
        q_users=q_users,
        fc=float(updates.get("fc", base.fc)),
        R_min=R_min,
        case_name=case_name,
    )
    problem.case_id = case_id
    problem.swept_param = swept_param
    problem.swept_value = float(swept_value)
    problem.scenario_name = scenario_name
    return problem


def _apply_seed_realization(problem, seed: int):
    setattr(problem, "realization_id", int(seed))
    if hasattr(problem, "fields"):
        return problem
    rng = np.random.default_rng(int(seed))
    if hasattr(problem, "q_users"):
        q_users = np.asarray(problem.q_users, dtype=float).copy()
        if q_users.ndim == 2 and q_users.shape[1] >= 2 and q_users.shape[0] >= 1:
            spread = float(max(getattr(problem, "delta", 0.2), 1.0e-3))
            lateral_std = max(0.05 * spread, 0.01)
            radial_std = max(0.08 * spread, 0.015)
            q_users[:, 0] = np.maximum(0.5, q_users[:, 0] + rng.normal(0.0, radial_std, size=q_users.shape[0]))
            q_users[:, 1] = q_users[:, 1] + rng.normal(0.0, lateral_std, size=q_users.shape[0])
            if q_users.shape[1] >= 3:
                q_users[:, 2] = q_users[:, 2] + rng.normal(0.0, 0.25 * lateral_std, size=q_users.shape[0])
            problem.q_users = q_users
    if hasattr(problem, "alpha"):
        alpha = np.asarray(problem.alpha, dtype=float).copy()
        if alpha.ndim == 1 and alpha.size >= 1:
            alpha = np.maximum(alpha * (1.0 + rng.normal(0.0, 0.02, size=alpha.size)), 1.0e-6)
            alpha = alpha / max(float(np.sum(alpha)), 1.0e-9)
            problem.alpha = alpha
    return problem


def _apply_sweep_value(problem_data_mod, base_problem, param: str, value: float, case_name: str, case_id: str) -> tuple[Any | None, str | None]:
    param_key = str(param).strip()
    param_norm = param_key.lower()
    updates: dict[str, Any] = {}
    scenario_name = f"{param_norm}_paper_sweep"

    schema_aliases = {
        "pmax": "system.Pmax",
        "p_max": "system.Pmax",
        "power_budget": "system.Pmax",
        "transmit_power": "system.Pmax",
        "m": "system.M",
        "num_ris_elements": "system.M",
        "nt": "system.Nt",
        "n_t": "system.Nt",
        "me": "system.Me",
        "mr": "system.Mr",
        "r_min": "optimization.R_min",
        "rmin_bpshz": "optimization.R_min",
        "qos_rate_bps_hz": "optimization.R_min",
        "lambda1": "optimization.lambda1",
        "lambda2": "optimization.lambda2",
        "lambda3": "optimization.lambda3",
    }
    update_path = schema_aliases.get(param_norm, param_key)
    last_token = update_path.replace("/", ".").split(".")[-1].lower()
    count_like = {"n", "k", "m", "nt", "nr", "me", "mr", "num_users", "num_antennas", "num_elements"}
    if last_token in count_like:
        updates[update_path] = max(1, int(round(float(value))))
    elif param_norm in {"pmax_dbm", "p_max_dbm"}:
        updates["system.Pmax"] = 10 ** ((float(value) - 30.0) / 10.0)
    elif param_norm in {"fc_ghz", "frequency_ghz"}:
        updates[update_path] = float(value) * 1.0e9
    elif hasattr(base_problem, "clone_with") or hasattr(base_problem, "fields"):
        updates[update_path] = float(value)
    elif hasattr(base_problem, param_key):
        updates[param_key] = value
    else:
        return None, f"sweep_param_not_found:{param_key}"
    problem = _clone_problem(problem_data_mod, base_problem, case_name=case_name, case_id=case_id, swept_param=update_path, swept_value=float(value), scenario_name=scenario_name, updates=updates)
    # Some generated ProblemData classes only expose case_name. Attach the
    # harness metadata explicitly so downstream CSVs, caching, and figure
    # filters never receive a stringified None case id.
    try:
        setattr(problem, "case_id", str(case_id))
        setattr(problem, "case_name", str(case_name))
        setattr(problem, "swept_param", str(update_path))
        setattr(problem, "swept_value", float(value))
        setattr(problem, "scenario_name", str(scenario_name))
    except Exception:
        pass
    return problem, None


_PHASE25_PAPER_WORKER_CACHE: dict[str, tuple[Any, Any, Any, Any]] = {}


def _phase25_worker_modules(run_dir: Path) -> tuple[Any, Any, Any, Any]:
    cache_key = str(run_dir.resolve())
    cached = _PHASE25_PAPER_WORKER_CACHE.get(cache_key)
    if cached is not None:
        return cached
    solver_dir = run_dir / "phase2-4" / "solver"
    solver_path_str = str(solver_dir)
    if solver_path_str not in sys.path:
        sys.path.insert(0, solver_path_str)
    problem_data_mod = _load_module("problem_data", solver_dir / "problem_data.py")
    validation_cases_mod = _load_module("validation_cases", solver_dir / "validation_cases.py")
    plugin_mod = _load_module("generated_plugin", solver_dir / "generated_plugin.py")
    canonical = validation_cases_mod.load_canonical_case(run_dir / "phase2-4" / "validation_plan.yaml")
    cached = (problem_data_mod, validation_cases_mod, plugin_mod, canonical)
    _PHASE25_PAPER_WORKER_CACHE[cache_key] = cached
    return cached


def _phase25_solve_methods_for_problem(
    problem: Any,
    seed: int,
    plugin_mod: Any,
    default_methods_to_run: list[str],
) -> list[dict[str, Any]]:
    model = plugin_mod.build_model(problem, seed=seed)
    if getattr(problem, "_model_cache", None) is None:
        setattr(problem, "_model_cache", model)
    runtime_model = getattr(problem, "_model_cache", None) or model
    call_model = runtime_model
    if isinstance(model, dict) and isinstance(runtime_model, dict):
        call_model = dict(runtime_model)
        for key in ("state_init", "operators", "metadata"):
            if key in model and key not in call_model:
                call_model[key] = model[key]
    metadata = model.get("metadata", {}) if isinstance(model, dict) else {}
    runtime_metadata = call_model.get("metadata", {}) if isinstance(call_model, dict) else {}
    max_iter = int(metadata.get("max_iterations", metadata.get("max_iter", 8)))
    if max_iter == 8 and isinstance(runtime_metadata, dict):
        max_iter = int(runtime_metadata.get("max_iterations", runtime_metadata.get("max_iter", max_iter)))
    methods_to_run = [
        str(method).strip()
        for method in getattr(problem, "methods_to_run", default_methods_to_run)
        if str(method).strip()
    ] or list(default_methods_to_run)

    def solve_method(method_name: str) -> tuple[dict[str, Any], float, int]:
        method_key = method_name.strip()
        start = time.perf_counter()
        if method_key == "proposed":
            state = plugin_mod.initial_state(problem, call_model, seed=seed)
            for iteration in range(max_iter):
                state = plugin_mod.proposed_step(problem, call_model, state, iteration)
            iterations = max_iter
        elif hasattr(plugin_mod, "method_solution"):
            state = plugin_mod.method_solution(problem, call_model, method_key, seed=seed)
            iterations = int(state.get("iteration", 1)) if isinstance(state, dict) else 1
        else:
            state = plugin_mod.baseline_solution(problem, call_model, seed=seed)
            if isinstance(state, dict):
                state = dict(state)
                state["method"] = method_key
            iterations = int(state.get("iteration", 1)) if isinstance(state, dict) else 1
        elapsed = time.perf_counter() - start
        metrics = plugin_mod.evaluate_state(problem, call_model, state)
        measured_ms = float(elapsed) * 1000.0
        if isinstance(metrics, dict):
            metrics["measured_solve_time_sec"] = float(elapsed)
            metrics["measured_total_runtime_ms"] = measured_ms
            for runtime_key in ("solve_time_ms", "solver_time_ms", "runtime_ms", "total_runtime_ms", "w_update_time_ms"):
                if runtime_key in metrics:
                    metrics[runtime_key] = measured_ms
            metrics["runtime_proxy_used"] = False
            metrics["runtime_measurement_source"] = "harness_wall_clock"
        return metrics, elapsed, iterations

    results: list[dict[str, Any]] = []
    for method in methods_to_run:
        metrics, elapsed, iterations = solve_method(method)
        violation = metrics.get("constraint_violation", {}) if isinstance(metrics.get("constraint_violation", {}), dict) else {}
        diagnostics = metrics.get("diagnostics", {}) if isinstance(metrics.get("diagnostics", {}), dict) else {}
        row = {
            "case_id": str(getattr(problem, "case_id", None) or getattr(problem, "case_name", "case")),
            "case_name": str(getattr(problem, "case_name", None) or getattr(problem, "case_id", "case")),
            "required_sweep": str(getattr(problem, "required_sweep", "")),
            "seed": int(seed),
            "swept_param": str(getattr(problem, "swept_param", "canonical")),
            "swept_value": float(getattr(problem, "swept_value", 0.0)),
            "scenario_name": str(getattr(problem, "scenario_name", "default")),
            "method": method,
            "status": str(metrics.get("status", "ok")),
            "objective": float(metrics.get("objective", metrics.get("objective_value", 0.0))),
            "feasible": bool(metrics.get("feasible", False)),
            "iterations": int(iterations),
            "solve_time_sec": float(elapsed),
            "power": float(metrics.get("total_power", metrics.get("power_consumption", 0.0))),
            "qos_violation": float(violation.get("qos", 0.0)) if isinstance(violation, dict) else 0.0,
            "separation_violation": float(violation.get("separation", 0.0)) if isinstance(violation, dict) else 0.0,
            "message": str(metrics.get("message", "")),
            "rejection_reason": str(diagnostics.get("rejection_reason", "")),
            "used_position_update": bool(diagnostics.get("used_position_update", False)),
            "objective_delta": float(diagnostics.get("objective_delta", 0.0)),
            "position_step_norm": float(diagnostics.get("position_step_norm", 0.0)),
        }
        for metric_name, metric_value in _flatten_metric_scalars(metrics).items():
            if metric_name not in row or row.get(metric_name) in (None, ""):
                row[metric_name] = metric_value
        results.append(row)
    return results


def _phase25_paper_job_process_worker(job: dict[str, Any]) -> tuple[dict[str, Any] | None, list[dict[str, Any]], list[dict[str, Any]]]:
    run_dir = Path(job["run_dir"])
    problem_data_mod, _validation_cases_mod, plugin_mod, canonical = _phase25_worker_modules(run_dir)
    item = job["item"]
    param = str(job["param"])
    value = float(job["value"])
    seed = int(job["seed"])
    case_name = str(job["case_name"])
    case_id = str(job["case_id"])
    methods_to_run = [str(method).strip() for method in job.get("methods_to_run", []) if str(method).strip()]
    default_methods_to_run = [str(method).strip() for method in job.get("default_methods_to_run", []) if str(method).strip()]
    problem, error = _apply_sweep_value(problem_data_mod, canonical, param, value, case_name, case_id)
    if error is not None or problem is None:
        return None, [{"figure_id": item.get("figure_id", ""), "sweep_param": param, "swept_value": value, "seed": seed, "error": error or "field_not_mutable"}], []
    problem.seed = seed
    problem.required_sweep = str(item.get("required_sweep", ""))
    problem = _apply_seed_realization(problem, seed)
    problem.methods_to_run = methods_to_run
    metadata = {
        "case_id": case_id,
        "figure_id": item.get("figure_id", ""),
        "required_sweep": item.get("required_sweep", ""),
        "chart_type": job.get("chart_type", "line"),
        "seed": seed,
        "swept_param": param,
        "swept_value": value,
        "scenario_name": str(getattr(problem, "scenario_name", "default")),
    }
    case_rows = _phase25_solve_methods_for_problem(problem, seed, plugin_mod, default_methods_to_run)
    for row in case_rows:
        row["case_id"] = case_id
        row["case_name"] = case_name
        row["required_sweep"] = str(item.get("required_sweep", ""))
        row["swept_param"] = param
        row["swept_value"] = float(value)
        row["scenario_name"] = str(getattr(problem, "scenario_name", "default"))
        row["seed"] = int(seed)
    return metadata, [], [row for row in case_rows if str(row.get("method", "")) in methods_to_run]


def run_phase24_paper_sweep_from_plan(run_dir: str | Path, quick: bool = False) -> dict[str, Any]:
    run_dir = Path(run_dir)
    phase25_dir = run_dir / "phase2-5"
    plan_path = phase25_dir / "paper_sweep_plan.json"
    solver_dir = run_dir / "phase2-4" / "solver"
    outputs_dir = solver_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    plan = _read_json(plan_path)
    if not plan:
        raise FileNotFoundError(f"Missing paper sweep plan: {plan_path}")
    sweep_tier = _phase25_quick_sweep_tier(quick=quick)
    output_prefix = {
        "scout": "scout_validation",
        "medium": "medium_validation",
        "paper": "paper_validation",
    }[sweep_tier]
    # Phase-2.5 sweeps use fewer points/seeds by tier, but they must not inherit
    # the Phase-2.4 quick-validation iteration cap.  The generated adapter reads
    # these environment markers when normalizing the algorithm metadata.
    os.environ["WARA_PHASE25_SWEEP_TIER"] = sweep_tier
    os.environ["WARA_RUN_MODE"] = output_prefix
    results_path = outputs_dir / f"{output_prefix}_results.csv"
    summary_path = outputs_dir / f"{output_prefix}_summary.json"
    cases_path = outputs_dir / f"{output_prefix}_cases.json"
    errors_path = outputs_dir / f"{output_prefix}_errors.json"
    output_staleness = _validation_output_staleness(run_dir, output_prefix, include_phase25_plan=True)
    cached_output = _cached_validation_output(run_dir, output_prefix, output_staleness, include_phase25_plan=True)
    if cached_output:
        return cached_output
    experiment_plan = _read_json(phase25_dir / "experiment_plan.json")
    objective_higher_is_better = bool(experiment_plan.get("primary_metric", {}).get("higher_is_better", True))
    final_methods_by_figure = _final_plotted_methods_by_figure(experiment_plan)
    default_methods_to_run = [
        str(method.get("internal_name") or method.get("name") or method.get("id") or "").strip()
        for method in experiment_plan.get("compared_methods", [])
        if isinstance(method, dict) and str(method.get("internal_name") or method.get("name") or method.get("id") or "").strip()
    ] or ["proposed"]

    solver_path_str = str(solver_dir)
    if solver_path_str not in sys.path:
        sys.path.insert(0, solver_path_str)
    problem_data_mod = _load_module("problem_data", solver_dir / "problem_data.py")
    validation_cases_mod = _load_module("validation_cases", solver_dir / "validation_cases.py")
    plugin_mod = _load_module("generated_plugin", solver_dir / "generated_plugin.py")

    canonical = validation_cases_mod.load_canonical_case(run_dir / "phase2-4" / "validation_plan.yaml")
    cases_metadata: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    seen_case_ids: set[tuple[str, int]] = set()
    rows: list[dict[str, Any]] = []
    plan_figures = plan.get("figures")
    if not isinstance(plan_figures, list):
        plan_figures = plan.get("missing_for_figures")
    if not isinstance(plan_figures, list):
        plan_figures = []
    normalized_plan_figures: list[dict[str, Any]] = []
    for item in plan_figures:
        if not isinstance(item, dict):
            continue
        figure_id = str(item.get("figure_id") or item.get("id") or "").strip()
        param = str(item.get("required_sweep_param") or item.get("sweep_param") or "").strip()
        chart_type = str(item.get("chart_type", "line"))
        using_medium_defaults = False
        if sweep_tier == "medium":
            values = item.get("medium_values")
            if not values:
                values = item.get("suggested_values", item.get("suggested_categories_or_values", []))
                using_medium_defaults = True
        elif sweep_tier == "scout":
            values = item.get("scout_values") or item.get("quick_values") or item.get("suggested_values", item.get("suggested_categories_or_values", []))
        else:
            values = item.get("suggested_values", item.get("suggested_categories_or_values", []))
        if not isinstance(values, list):
            values = []
        scalar_values: list[float] = []
        for value in values:
            if isinstance(value, (list, tuple, dict)):
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(numeric):
                scalar_values.append(numeric)
        scalar_values = sorted(dict.fromkeys(scalar_values))
        if sweep_tier == "medium" and using_medium_defaults:
            scalar_values = _select_scout_values(scalar_values, chart_type, target_points=_medium_target_points(chart_type))
        contract_methods = final_methods_by_figure.get(figure_id, [])
        raw_methods = contract_methods or item.get("methods_to_run", default_methods_to_run)
        methods = [
            str(method).strip()
            for method in raw_methods
            if str(method).strip()
        ]
        if not methods:
            methods = list(default_methods_to_run)
        if figure_id and param and scalar_values:
            clone = dict(item)
            requested_sweep_id = str(item.get("required_sweep") or item.get("required_sweep_id") or "").strip()
            resolved_sweep_id = _validation_plan_sweep_id_for_param(phase25_dir, param, requested_sweep_id)
            clone["figure_id"] = figure_id
            clone["required_sweep"] = resolved_sweep_id or requested_sweep_id
            clone["required_sweep_param"] = param
            clone["suggested_values"] = scalar_values
            if sweep_tier == "scout":
                clone["scout_values"] = list(clone["suggested_values"])
            elif sweep_tier == "medium":
                clone["medium_values"] = list(clone["suggested_values"])
            clone["methods_to_run"] = methods
            normalized_plan_figures.append(clone)
    planned_figure_ids = {str(item.get("figure_id", "")) for item in normalized_plan_figures if str(item.get("figure_id", "")).strip()}

    try:
        env_seed_cap = int(os.environ.get("WCL_PHASE25_PAPER_SWEEP_MAX_SEEDS", "0") or "0")
    except ValueError:
        env_seed_cap = 0
    try:
        env_seed_override = int(
            os.environ.get("WARA_PHASE25_PAPER_SWEEP_SEEDS")
            or os.environ.get("WCL_PHASE25_PAPER_SWEEP_SEEDS")
            or "0"
        )
    except ValueError:
        env_seed_override = 0
    try:
        env_value_cap = int(os.environ.get("WCL_PHASE25_PAPER_SWEEP_MAX_VALUES_PER_FIGURE", "0") or "0")
    except ValueError:
        env_value_cap = 0
    try:
        write_every = int(os.environ.get("WCL_PHASE25_PAPER_SWEEP_WRITE_EVERY", "10") or "10")
    except ValueError:
        write_every = 10
    write_every = max(1, write_every)
    runtime_abort_median_sec = 0.0
    runtime_abort_min_rows = 1
    job_timeout_sec = 0
    runtime_abort_reason = ""
    reuse_existing_paper_rows = (not quick) and not bool(output_staleness.get("is_stale"))
    existing_methods_by_case_seed: dict[tuple[str, int], set[str]] = {}
    if reuse_existing_paper_rows and results_path.exists():
        try:
            with results_path.open("r", encoding="utf-8-sig", newline="") as handle:
                for existing in csv.DictReader(handle):
                    existing_row = dict(existing)
                    rows.append(existing_row)
                    case_id = str(existing_row.get("case_id", ""))
                    try:
                        seed = int(existing_row.get("seed", 0) or 0)
                    except (TypeError, ValueError):
                        seed = 0
                    method = str(existing_row.get("method", "")).strip()
                    if case_id and method:
                        existing_methods_by_case_seed.setdefault((case_id, seed), set()).add(method)
        except Exception:
            rows = []
            existing_methods_by_case_seed = {}
    if reuse_existing_paper_rows and cases_path.exists():
        try:
            existing_cases = json.loads(cases_path.read_text(encoding="utf-8"))
            if isinstance(existing_cases, list):
                for existing in existing_cases:
                    if not isinstance(existing, dict):
                        continue
                    cases_metadata.append(existing)
        except Exception:
            cases_metadata = []

    def _run_case(problem, seed: int) -> list[dict[str, Any]]:
        model = plugin_mod.build_model(problem, seed=seed)
        if getattr(problem, "_model_cache", None) is None:
            setattr(problem, "_model_cache", model)
        runtime_model = getattr(problem, "_model_cache", None) or model
        call_model = runtime_model
        if isinstance(model, dict) and isinstance(runtime_model, dict):
            call_model = dict(runtime_model)
            for key in ("state_init", "operators", "metadata"):
                if key in model and key not in call_model:
                    call_model[key] = model[key]
        metadata = model.get("metadata", {}) if isinstance(model, dict) else {}
        runtime_metadata = call_model.get("metadata", {}) if isinstance(call_model, dict) else {}
        max_iter = int(metadata.get("max_iterations", metadata.get("max_iter", 8)))
        if max_iter == 8 and isinstance(runtime_metadata, dict):
            max_iter = int(runtime_metadata.get("max_iterations", runtime_metadata.get("max_iter", max_iter)))
        methods_to_run = [
            str(method).strip()
            for method in getattr(problem, "methods_to_run", default_methods_to_run)
            if str(method).strip()
        ] or list(default_methods_to_run)

        def solve_method(method_name: str) -> tuple[dict[str, Any], float, int]:
            method_key = method_name.strip()
            start = time.perf_counter()
            if method_key == "proposed":
                state = plugin_mod.initial_state(problem, call_model, seed=seed)
                for iteration in range(max_iter):
                    state = plugin_mod.proposed_step(problem, call_model, state, iteration)
                iterations = max_iter
            elif hasattr(plugin_mod, "method_solution"):
                state = plugin_mod.method_solution(problem, call_model, method_key, seed=seed)
                iterations = int(state.get("iteration", 1)) if isinstance(state, dict) else 1
            else:
                state = plugin_mod.baseline_solution(problem, call_model, seed=seed)
                if isinstance(state, dict):
                    state = dict(state)
                    state["method"] = method_key
                iterations = int(state.get("iteration", 1)) if isinstance(state, dict) else 1
            elapsed = time.perf_counter() - start
            metrics = plugin_mod.evaluate_state(problem, call_model, state)
            measured_ms = float(elapsed) * 1000.0
            if isinstance(metrics, dict):
                metrics["measured_solve_time_sec"] = float(elapsed)
                metrics["measured_total_runtime_ms"] = measured_ms
                for runtime_key in ("solve_time_ms", "solver_time_ms", "runtime_ms", "total_runtime_ms", "w_update_time_ms"):
                    if runtime_key in metrics:
                        metrics[runtime_key] = measured_ms
                metrics["runtime_proxy_used"] = False
                metrics["runtime_measurement_source"] = "harness_wall_clock"
            return metrics, elapsed, iterations

        results: list[dict[str, Any]] = []
        for method in methods_to_run:
            metrics, elapsed, iterations = solve_method(method)
            violation = metrics.get("constraint_violation", {}) if isinstance(metrics.get("constraint_violation", {}), dict) else {}
            diagnostics = metrics.get("diagnostics", {}) if isinstance(metrics.get("diagnostics", {}), dict) else {}
            row = {
                "case_id": str(getattr(problem, "case_id", None) or getattr(problem, "case_name", "case")),
                "case_name": str(getattr(problem, "case_name", None) or getattr(problem, "case_id", "case")),
                "required_sweep": str(getattr(problem, "required_sweep", "")),
                "seed": int(seed),
                "swept_param": str(getattr(problem, "swept_param", "canonical")),
                "swept_value": float(getattr(problem, "swept_value", 0.0)),
                "scenario_name": str(getattr(problem, "scenario_name", "default")),
                "method": method,
                "status": str(metrics.get("status", "ok")),
                "objective": float(metrics.get("objective", metrics.get("objective_value", 0.0))),
                "feasible": bool(metrics.get("feasible", False)),
                "iterations": int(iterations),
                "solve_time_sec": float(elapsed),
                "power": float(metrics.get("total_power", metrics.get("power_consumption", 0.0))),
                "qos_violation": float(violation.get("qos", 0.0)) if isinstance(violation, dict) else 0.0,
                "separation_violation": float(violation.get("separation", 0.0)) if isinstance(violation, dict) else 0.0,
                "message": str(metrics.get("message", "")),
                "rejection_reason": str(diagnostics.get("rejection_reason", "")),
                "used_position_update": bool(diagnostics.get("used_position_update", False)),
                "objective_delta": float(diagnostics.get("objective_delta", 0.0)),
                "position_step_norm": float(diagnostics.get("position_step_norm", 0.0)),
            }
            for metric_name, metric_value in _flatten_metric_scalars(metrics).items():
                if metric_name not in row or row.get(metric_name) in (None, ""):
                    row[metric_name] = metric_value
            results.append(row)
        return results

    paper_jobs: list[dict[str, Any]] = []
    planned_seed_counts: list[int] = []
    for item in normalized_plan_figures:
        param = str(item.get("required_sweep_param", ""))
        values = item.get("suggested_values", [])
        if env_value_cap > 0:
            values = list(values)[:env_value_cap]
        chart_type = str(item.get("chart_type", "line"))
        if sweep_tier == "scout":
            num_seeds = int(item.get("scout_num_seeds", item.get("quick_num_seeds", SCOUT_MODE_SEEDS)))
        elif sweep_tier == "medium":
            num_seeds = _medium_num_seeds(item)
        else:
            num_seeds = int(item.get("suggested_num_seeds", PAPER_PREFERRED_SEEDS))
        if sweep_tier == "paper" and env_seed_override > 0:
            num_seeds = max(num_seeds, env_seed_override)
        if env_seed_cap > 0:
            num_seeds = min(num_seeds, env_seed_cap)
        planned_seed_counts.append(max(0, int(num_seeds)))
        methods_to_run = [
            str(method).strip()
            for method in item.get("methods_to_run", default_methods_to_run)
            if str(method).strip()
        ]
        for value in values:
            for seed in range(num_seeds):
                case_name = f"{item.get('figure_id', 'fig')}_{param}_{value}"
                case_id = f"{case_name}"
                if (case_id, seed) in seen_case_ids:
                    continue
                existing_methods = existing_methods_by_case_seed.get((case_id, seed), set())
                if methods_to_run and set(methods_to_run).issubset(existing_methods):
                    seen_case_ids.add((case_id, seed))
                    continue
                seen_case_ids.add((case_id, seed))
                paper_jobs.append(
                    {
                        "item": item,
                        "param": param,
                        "value": float(value),
                        "seed": int(seed),
                        "case_name": case_name,
                        "case_id": case_id,
                        "chart_type": chart_type,
                        "methods_to_run": methods_to_run,
                    }
                )

    def _run_paper_job(job: dict[str, Any]) -> tuple[dict[str, Any] | None, list[dict[str, Any]], list[dict[str, Any]]]:
        item = job["item"]
        param = str(job["param"])
        value = float(job["value"])
        seed = int(job["seed"])
        case_name = str(job["case_name"])
        case_id = str(job["case_id"])
        methods_to_run = [str(method).strip() for method in job.get("methods_to_run", []) if str(method).strip()]
        problem, error = _apply_sweep_value(problem_data_mod, canonical, param, value, case_name, case_id)
        if error is not None or problem is None:
            return None, [{"figure_id": item.get("figure_id", ""), "sweep_param": param, "swept_value": value, "seed": seed, "error": error or "field_not_mutable"}], []
        problem.seed = seed
        problem.required_sweep = str(item.get("required_sweep", ""))
        problem = _apply_seed_realization(problem, seed)
        problem.methods_to_run = methods_to_run
        metadata = {
            "case_id": case_id,
            "figure_id": item.get("figure_id", ""),
            "required_sweep": item.get("required_sweep", ""),
            "chart_type": job.get("chart_type", "line"),
            "seed": seed,
            "swept_param": param,
            "swept_value": value,
            "scenario_name": str(getattr(problem, "scenario_name", "default")),
        }
        case_rows = _run_case(problem, seed)
        for row in case_rows:
            row["case_id"] = case_id
            row["case_name"] = case_name
            row["required_sweep"] = str(item.get("required_sweep", ""))
            row["swept_param"] = param
            row["swept_value"] = float(value)
            row["scenario_name"] = str(getattr(problem, "scenario_name", "default"))
            row["seed"] = int(seed)
        return metadata, [], [row for row in case_rows if str(row.get("method", "")) in methods_to_run]

    def _runtime_abort_triggered() -> str:
        return ""
        if sweep_tier == "paper" or runtime_abort_median_sec <= 0:
            return ""
        row_times: list[float] = []
        proposed_times: list[float] = []
        case_times: dict[tuple[str, str], float] = {}
        for row in rows:
            try:
                value = float(row.get("solve_time_sec", row.get("measured_solve_time_sec", 0.0)) or 0.0)
            except (TypeError, ValueError):
                value = 0.0
            if math.isfinite(value) and value >= 0.0:
                row_times.append(value)
                key = (str(row.get("case_id") or row.get("case_name") or ""), str(row.get("seed") or "0"))
                case_times[key] = case_times.get(key, 0.0) + value
                if str(row.get("method") or "").strip().lower() == "proposed":
                    proposed_times.append(value)
        if len(row_times) < runtime_abort_min_rows:
            return ""
        median_case_time = statistics.median(case_times.values()) if case_times else 0.0
        median_proposed_time = statistics.median(proposed_times) if proposed_times else 0.0
        median_runtime = max(float(median_case_time), float(median_proposed_time))
        if median_runtime <= runtime_abort_median_sec:
            return ""
        return (
            f"{sweep_tier}_runtime_too_slow:"
            f"median_case_time_sec={float(median_case_time):.3f};"
            f"median_proposed_time_sec={float(median_proposed_time):.3f};"
            f"threshold={runtime_abort_median_sec:.3f};"
            f"rows={len(row_times)}"
        )

    def _write_current_outputs(partial: bool, completed_jobs: int, total_jobs: int) -> dict[str, Any]:
        rows.sort(key=lambda row: (str(row.get("case_id", "")), int(row.get("seed", 0) or 0), str(row.get("method", ""))))
        cases_metadata.sort(key=lambda item: (str(item.get("case_id", "")), int(item.get("seed", 0) or 0)))
        if rows:
            fieldnames: list[str] = []
            for row in rows:
                for key in row.keys():
                    if key not in fieldnames:
                        fieldnames.append(key)
            with results_path.open("w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for row in rows:
                    writer.writerow({k: _json_safe(v) for k, v in row.items()})
        else:
            results_path.write_text("", encoding="utf-8")
        summary = _summary_from_results(rows, objective_higher_is_better=objective_higher_is_better)
        summary["quick_mode"] = bool(quick)
        summary["paper_sweep_run_mode"] = sweep_tier
        default_seeds = {"scout": SCOUT_MODE_SEEDS, "medium": MEDIUM_MODE_SEEDS, "paper": PAPER_PREFERRED_SEEDS}[sweep_tier]
        summary["seeds_per_point"] = max(planned_seed_counts) if planned_seed_counts else default_seeds
        summary["actual_completed_jobs"] = int(completed_jobs)
        summary["planned_jobs"] = int(total_jobs)
        summary["partial"] = bool(partial)
        summary["runtime_abort_reason"] = runtime_abort_reason
        summary["runtime_abort_median_threshold_sec"] = float(runtime_abort_median_sec)
        summary["runtime_abort_min_rows"] = int(runtime_abort_min_rows)
        summary["job_timeout_sec"] = int(job_timeout_sec)
        summary["paper_sweep_seed_cap"] = int(env_seed_cap)
        summary["paper_sweep_value_cap"] = int(env_value_cap)
        summary["total_cases"] = len(cases_metadata)
        summary["generated_plugin_path"] = str(solver_dir / "generated_plugin.py")
        summary["validation_output_prefix"] = output_prefix
        summary["validation_output_staleness_before_run"] = output_staleness
        summary["paper_validation_staleness_before_run"] = output_staleness if not quick else _paper_validation_staleness(run_dir)
        summary["reused_existing_paper_rows"] = bool(reuse_existing_paper_rows)
        summary["validation_source_fingerprint"] = _write_validation_source_fingerprint(
            run_dir,
            output_prefix,
            include_phase25_plan=True,
        )
        _write_json(summary_path, summary)
        _write_json(cases_path, cases_metadata)
        _write_json(errors_path, {"errors": errors})
        return summary

    try:
        max_workers = int(os.environ.get("WCL_PHASE25_PAPER_SWEEP_WORKERS", "0") or "0")
    except ValueError:
        max_workers = 0
    if max_workers <= 0:
        max_workers = min(8, max(1, (os.cpu_count() or 1) // 2))
    max_workers = max(1, min(max_workers, 16))
    completed_jobs = len(existing_methods_by_case_seed)
    total_jobs = completed_jobs + len(paper_jobs)
    runtime_preflight_observations = [
        obs
        for obs in [
            _phase25_runtime_observation_from_csv(outputs_dir / "validation_results.csv"),
            _phase25_runtime_observation_from_csv(outputs_dir / "scout_validation_results.csv"),
            _phase25_runtime_observation_from_csv(outputs_dir / "medium_validation_results.csv"),
        ]
        if obs.get("num_cases")
    ]
    runtime_preflight = {}
    if runtime_preflight_observations and total_jobs > 0:
        median_case_time = max(float(obs.get("median_case_time_sec") or 0.0) for obs in runtime_preflight_observations)
        median_proposed_time = max(float(obs.get("median_proposed_time_sec") or 0.0) for obs in runtime_preflight_observations)
        max_case_time = max(float(obs.get("max_case_time_sec") or 0.0) for obs in runtime_preflight_observations)
        max_proposed_time = max(float(obs.get("max_proposed_time_sec") or 0.0) for obs in runtime_preflight_observations)
        estimated_wall_sec = median_case_time * float(total_jobs) / float(max(max_workers, 1))
        estimated_limit_sec = 0.0
        runtime_preflight = {
            "observations": runtime_preflight_observations,
            "median_case_time_sec": median_case_time,
            "median_proposed_time_sec": median_proposed_time,
            "max_case_time_sec": max_case_time,
            "max_proposed_time_sec": max_proposed_time,
            "estimated_wall_time_sec": estimated_wall_sec,
            "estimated_wall_time_limit_sec": estimated_limit_sec,
            "max_workers": max_workers,
            "planned_jobs": total_jobs,
            "job_timeout_sec": job_timeout_sec,
        }
        if estimated_limit_sec > 0 and estimated_wall_sec > estimated_limit_sec:
            runtime_abort_reason = (
                f"{sweep_tier}_runtime_preflight_too_slow:"
                f"estimated_wall_time_sec={estimated_wall_sec:.3f}>limit={estimated_limit_sec:.3f};"
                f"median_case_time_sec={median_case_time:.3f};planned_jobs={total_jobs};workers={max_workers}"
            )
        elif runtime_abort_median_sec > 0 and max(median_case_time, median_proposed_time) > runtime_abort_median_sec and sweep_tier in {"medium", "paper"}:
            runtime_abort_reason = (
                f"{sweep_tier}_runtime_preflight_median_too_slow:"
                f"median_case_time_sec={median_case_time:.3f};"
                f"median_proposed_time_sec={median_proposed_time:.3f};"
                f"threshold={runtime_abort_median_sec:.3f}"
            )
        elif job_timeout_sec > 0 and max(max_case_time, max_proposed_time) > job_timeout_sec and sweep_tier in {"medium", "paper"}:
            runtime_abort_reason = (
                f"{sweep_tier}_runtime_preflight_long_tail:"
                f"max_case_time_sec={max_case_time:.3f};"
                f"max_proposed_time_sec={max_proposed_time:.3f};"
                f"job_timeout_sec={job_timeout_sec}"
            )
    if runtime_abort_reason:
        errors.append(
            {
                "error": runtime_abort_reason,
                "stage": "runtime_preflight",
                "sweep_tier": sweep_tier,
                "runtime_preflight": runtime_preflight,
            }
        )
        summary = _write_current_outputs(partial=True, completed_jobs=0, total_jobs=total_jobs)
        summary["runtime_preflight"] = runtime_preflight
        _write_json(summary_path, summary)
        return {
            "results_csv": str(results_path),
            "summary_json": str(summary_path),
            "cases_json": str(cases_path),
            "errors_json": str(errors_path),
            "num_cases": len(cases_metadata),
            "num_results": len(rows),
            "quick_mode": bool(quick),
            "paper_sweep_run_mode": sweep_tier,
            "validation_output_prefix": output_prefix,
            "runtime_abort_reason": runtime_abort_reason,
            "runtime_preflight": runtime_preflight,
            "validation_output_staleness_before_run": output_staleness,
            "paper_validation_staleness_before_run": output_staleness if not quick else _paper_validation_staleness(run_dir),
            "reused_existing_paper_rows": bool(reuse_existing_paper_rows),
        }
    def _run_paper_job_with_timeout(job: dict[str, Any]):
        return _run_paper_job(job)

    executor_kind = str(os.environ.get("WCL_PHASE25_PAPER_SWEEP_EXECUTOR", "thread") or "thread").strip().lower()

    if total_jobs == 0:
        _write_current_outputs(partial=False, completed_jobs=0, total_jobs=0)
    elif len(paper_jobs) <= 1 or max_workers == 1:
        for job in paper_jobs:
            metadata, job_errors, job_rows = _run_paper_job_with_timeout(job)
            if metadata is not None:
                cases_metadata.append(metadata)
            errors.extend(job_errors)
            rows.extend(job_rows)
            completed_jobs += 1
            timeout_errors = [item for item in job_errors if str(item.get("stage", "")) == "phase25_job_timeout"]
            if timeout_errors:
                runtime_abort_reason = (
                    f"{sweep_tier}_job_timeout:"
                    f"timeout_sec={job_timeout_sec};completed_jobs={completed_jobs};total_jobs={total_jobs}"
                )
                errors.append({"error": runtime_abort_reason, "completed_jobs": completed_jobs, "total_jobs": total_jobs})
                _write_current_outputs(partial=True, completed_jobs=completed_jobs, total_jobs=total_jobs)
                break
            if completed_jobs % write_every == 0 or completed_jobs == total_jobs:
                _write_current_outputs(partial=completed_jobs < total_jobs, completed_jobs=completed_jobs, total_jobs=total_jobs)
                runtime_abort_reason = _runtime_abort_triggered()
                if runtime_abort_reason:
                    errors.append({"error": runtime_abort_reason, "completed_jobs": completed_jobs, "total_jobs": total_jobs})
                    _write_current_outputs(partial=True, completed_jobs=completed_jobs, total_jobs=total_jobs)
                    break
    elif executor_kind in {"process", "processes", "multiprocess", "multiprocessing"}:
        worker_jobs = []
        for job in paper_jobs:
            payload = dict(job)
            payload["run_dir"] = str(run_dir)
            payload["default_methods_to_run"] = list(default_methods_to_run)
            worker_jobs.append(payload)
        context = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=context) as executor:
            future_map = {executor.submit(_phase25_paper_job_process_worker, job): idx for idx, job in enumerate(worker_jobs)}
            for future in as_completed(future_map):
                metadata, job_errors, job_rows = future.result()
                if metadata is not None:
                    cases_metadata.append(metadata)
                errors.extend(job_errors)
                rows.extend(job_rows)
                completed_jobs += 1
                if completed_jobs % write_every == 0 or completed_jobs == total_jobs:
                    _write_current_outputs(partial=completed_jobs < total_jobs, completed_jobs=completed_jobs, total_jobs=total_jobs)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {executor.submit(_run_paper_job, job): idx for idx, job in enumerate(paper_jobs)}
            for future in as_completed(future_map):
                metadata, job_errors, job_rows = future.result()
                if metadata is not None:
                    cases_metadata.append(metadata)
                errors.extend(job_errors)
                rows.extend(job_rows)
                completed_jobs += 1
                if completed_jobs % write_every == 0 or completed_jobs == total_jobs:
                    _write_current_outputs(partial=completed_jobs < total_jobs, completed_jobs=completed_jobs, total_jobs=total_jobs)
    summary = _write_current_outputs(partial=bool(runtime_abort_reason), completed_jobs=completed_jobs, total_jobs=total_jobs)
    return {
        "results_csv": str(results_path),
        "summary_json": str(summary_path),
        "cases_json": str(cases_path),
        "errors_json": str(errors_path),
        "num_cases": len(cases_metadata),
        "num_results": len(rows),
        "quick_mode": bool(quick),
        "paper_sweep_run_mode": sweep_tier,
        "validation_output_prefix": output_prefix,
        "validation_output_staleness_before_run": output_staleness,
        "paper_validation_staleness_before_run": output_staleness if not quick else _paper_validation_staleness(run_dir),
        "reused_existing_paper_rows": bool(reuse_existing_paper_rows),
    }


def run_phase25_analysis(run_dir: str | Path, experiment_plan_path: str | Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    phase25_dir = run_dir / "phase2-5"
    phase25_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = phase25_dir / "figures"
    tables_dir = phase25_dir / "tables"
    summary, df_raw = load_phase24_results(run_dir)
    data_source = str(summary.get("_phase25_data_source", "quick_validation"))
    quick_mode = bool(summary.get("quick_mode", data_source != "paper_validation"))
    run_mode = str(summary.get("paper_sweep_run_mode") or "").strip().lower()
    if not run_mode:
        if data_source == "paper_validation":
            run_mode = "paper"
        elif data_source == "medium_validation":
            run_mode = "medium"
        elif data_source == "scout_validation":
            run_mode = "scout"
        else:
            run_mode = "quick" if quick_mode else "paper"
    df = normalize_results_dataframe(df_raw)
    plan = json.loads(Path(experiment_plan_path).read_text(encoding="utf-8"))
    plan["figure_specs"] = [normalize_figure_spec(fig, plan.get("primary_metric", {})) for fig in plan.get("figure_specs", [])]
    plan = apply_phase25_sweep_display_names(plan, run_dir)
    plan = align_phase25_plan_with_observed_sweep_plan(plan, phase25_dir, df)
    plan = apply_phase25_sweep_display_names(plan, run_dir)
    plan = repair_phase25_primary_metric_from_figures(plan, df)
    plan = repair_mechanism_figure_metrics_from_data(plan, df)
    plan = select_single_benchmark_methods_for_figures(plan, df)
    _write_json(Path(experiment_plan_path), plan)
    available = summarize_available_results(summary, df)
    _write_json(phase25_dir / "available_data_summary.json", available)
    primary_metric = plan.get("primary_metric", {}).get("name", "objective")
    comparison_df = build_per_case_comparison(df, plan)
    if not comparison_df.empty and "baseline_method" in comparison_df.columns:
        plan["_active_baseline_method"] = str(comparison_df["baseline_method"].dropna().astype(str).iloc[0])
    comparison_df = compute_relative_gain(comparison_df, primary_metric=primary_metric, higher_is_better=bool(plan.get("primary_metric", {}).get("higher_is_better", True)))
    active_baseline_method = str(plan.get("_active_baseline_method") or "")
    plan["_primary_claim_mode"] = "optimal_reference_equivalence" if _is_optimal_reference_method(plan, active_baseline_method) else "advantage_over_benchmark"
    strongest_baseline_method = select_strongest_practical_baseline_method(df, plan)
    _write_json(Path(experiment_plan_path), plan)
    monte_carlo_report = run_monte_carlo_check(df, plan, primary_metric=primary_metric, phase25_dir=phase25_dir)
    plot_quality_report = check_data_sufficiency(df, comparison_df, plan, monte_carlo_report, quick_mode, run_mode=run_mode)
    write_plot_quality_report(phase25_dir, plot_quality_report)
    method_naming_paths = write_method_naming_summary(phase25_dir, plan)

    figure_outputs: list[dict[str, Any]] = []
    for fig_spec in plan.get("figure_specs", []):
        fig_quality = next((item for item in plot_quality_report.get("figures", []) if item.get("figure_id") == fig_spec.get("figure_id")), None)
        paper_ready = bool(fig_quality and fig_quality.get("paper_ready", False))
        try:
            curve_df = aggregate_for_figure(df, fig_spec)
            write_curve_data(curve_df, figures_dir / f"{fig_spec.get('figure_id', 'figure')}_curve_data.csv")
            meta = render_figure(curve_df, fig_spec, figures_dir, paper_ready=paper_ready, plan=plan)
            if fig_quality:
                meta["blocking_issues"] = fig_quality.get("blocking_issues", [])
                meta["seeds_per_point_summary"] = fig_quality.get("seeds_per_point_summary", {})
                meta["num_x_points"] = fig_quality.get("num_x_points", meta.get("num_x_points", 0))
                fig_quality["chart_type"] = meta.get("chart_type", fig_quality.get("chart_type"))
                fig_quality["has_error_bars"] = bool(meta.get("has_error_bars", fig_quality.get("has_error_bars", False)))
                fig_quality["error_display"] = meta.get("error_display", fig_quality.get("error_display", ""))
                fig_quality["error_display_label"] = meta.get("error_display_label", fig_quality.get("error_display_label", ""))
                fig_quality["display_policy_note"] = meta.get("display_policy_note", fig_quality.get("display_policy_note", ""))
            figure_outputs.append(meta)
        except Exception as exc:
            figure_outputs.append(
                {
                    "figure_id": fig_spec.get("figure_id", "figure"),
                    "filename_png": "",
                    "filename_pdf": "",
                    "draft_or_final": "draft",
                    "paper_ready": False,
                    "blocking_issues": ["aggregation_failed"],
                    "error": str(exc),
                    "x_axis_param": _get_figure_sweep_param(fig_spec),
                    "y_metric": _get_figure_metric_name(fig_spec),
                }
            )

    table_outputs: list[dict[str, Any]] = []
    for table_spec in plan.get("table_specs", [])[:1]:
        try:
            table_meta = render_table(comparison_df, table_spec, tables_dir, plan=plan)
            raw_sweep_count = int(df["swept_param"].nunique()) if "swept_param" in df.columns and not df.empty else 0
            row_selection = str(table_spec.get("row_selection") or table_spec.get("row_granularity") or "").lower()
            if (
                raw_sweep_count > int(table_meta.get("num_rows", 0))
                or "method" in row_selection
                or "ablation" in row_selection
            ):
                table_meta = render_evidence_table(df, table_spec, tables_dir, plan=plan)
            table_outputs.append(table_meta)
        except Exception as exc:
            table_outputs.append({"table_id": table_spec.get("table_id", "table_1"), "filename_csv": "", "filename_md": "", "status": "missing", "error": str(exc)})
    caption_paths = write_caption_files(phase25_dir, plan, figure_outputs, table_outputs)
    contract_consistency_report = validate_phase25_contract_consistency(
        run_dir,
        plan,
        df,
        figure_outputs=figure_outputs,
        caption_paths=caption_paths,
    )
    contract_consistency_path = phase25_dir / "phase25_contract_consistency_check.json"
    _write_json(contract_consistency_path, contract_consistency_report)

    comparable = comparison_df[comparison_df["comparable"]] if not comparison_df.empty and "comparable" in comparison_df.columns else pd.DataFrame()
    relative_gains = comparable["relative_gain"].tolist() if not comparable.empty and "relative_gain" in comparable.columns else []
    best_case_id = str(comparable.loc[comparable["relative_gain"].idxmax(), "case_id"]) if not comparable.empty else ""
    worst_case_id = str(comparable.loc[comparable["relative_gain"].idxmin(), "case_id"]) if not comparable.empty else ""
    draft_status = "medium_mode_only" if run_mode == "medium" else "quick_mode_only"
    phase25_status = plot_quality_report.get("overall_status", draft_status if quick_mode else "needs_more_phase24_runs")
    proposed_win_rate_now = float(comparable["proposed_win"].mean()) if not comparable.empty and "proposed_win" in comparable.columns else 0.0
    proposed_median_gain_now = float(statistics.median(relative_gains)) if relative_gains else 0.0
    row_level_primary_claim_check = evaluate_primary_claim_check(comparable, plan)
    figure_level_primary_claim_check = evaluate_figure_level_primary_claim_check(
        df,
        plan,
        baseline_method=active_baseline_method,
    )
    if figure_level_primary_claim_check.get("figure_checks"):
        primary_claim_check = dict(figure_level_primary_claim_check)
        primary_claim_check["row_level_diagnostic"] = row_level_primary_claim_check
    else:
        primary_claim_check = row_level_primary_claim_check
    strongest_practical_audit: dict[str, Any] = {}
    if strongest_baseline_method:
        if strongest_baseline_method == active_baseline_method:
            strongest_practical_audit = dict(primary_claim_check)
            strongest_practical_audit["same_as_primary_claim_baseline"] = True
        else:
            audit_plan = dict(plan)
            audit_plan["_forced_comparison_baseline_method"] = strongest_baseline_method
            audit_plan["_active_baseline_method"] = strongest_baseline_method
            audit_comparison = build_per_case_comparison(df, audit_plan)
            audit_comparison = compute_relative_gain(
                audit_comparison,
                primary_metric=primary_metric,
                higher_is_better=bool(plan.get("primary_metric", {}).get("higher_is_better", True)),
            )
            audit_comparable = audit_comparison[audit_comparison["comparable"]] if not audit_comparison.empty and "comparable" in audit_comparison.columns else pd.DataFrame()
            row_level_audit = evaluate_primary_claim_check(audit_comparable, audit_plan)
            figure_level_audit = evaluate_figure_level_primary_claim_check(
                df,
                audit_plan,
                baseline_method=strongest_baseline_method,
            )
            if figure_level_audit.get("figure_checks"):
                strongest_practical_audit = dict(figure_level_audit)
                strongest_practical_audit["row_level_diagnostic"] = row_level_audit
            else:
                strongest_practical_audit = row_level_audit
            strongest_practical_audit["same_as_primary_claim_baseline"] = False
            strongest_practical_audit["num_comparable_cases"] = int(len(audit_comparable))
    plot_quality_report["primary_claim_check"] = primary_claim_check
    plot_quality_report["strongest_practical_baseline_audit"] = strongest_practical_audit
    plot_quality_report["contract_consistency_check"] = contract_consistency_report
    paper_ready_statuses = {"paper_minimum_ready", "paper_preferred_ready", "high_confidence_ready"}
    if (
        phase25_status in paper_ready_statuses
        and not bool(primary_claim_check.get("passes", False))
    ):
        phase25_status = "claim_failure_needs_redesign"
        plot_quality_report["overall_status"] = phase25_status
        plot_quality_report.setdefault("global_blocking_issues", [])
        if comparable.empty:
            plot_quality_report["global_blocking_issues"].append(
                "The data coverage may be paper-ready, but no comparable proposed-vs-baseline cases were found for the primary metric."
            )
        elif primary_claim_check.get("mode") == "optimal_reference_equivalence":
            plot_quality_report["global_blocking_issues"].append(
                "The data coverage may be paper-ready, but the proposed method does not match the declared optimal-reference comparison within tolerance."
            )
        else:
            plot_quality_report["global_blocking_issues"].append(
                "The data coverage may be paper-ready, but the proposed method does not show a positive primary-metric advantage over the declared comparison."
            )
    if phase25_status in paper_ready_statuses and not bool(contract_consistency_report.get("ok", False)):
        phase25_status = "contract_consistency_failed"
        plot_quality_report["overall_status"] = phase25_status
        plot_quality_report.setdefault("global_blocking_issues", [])
        for item in contract_consistency_report.get("errors", []):
            plot_quality_report["global_blocking_issues"].append(f"Contract consistency: {item}")
    write_plot_quality_report(phase25_dir, plot_quality_report)
    extras = {}
    draft_or_blocked_statuses = {"quick_mode_only", "medium_mode_only", "needs_more_phase24_runs", "claim_failure_needs_redesign", "contract_consistency_failed"}
    if phase25_status in draft_or_blocked_statuses:
        extras = write_missing_experiments(phase25_dir, plot_quality_report)
    else:
        ready_lines = [
            "# Missing experiments",
            "",
            f"No additional Phase 2.4 runs are blocking the current Phase 2.5 status `{phase25_status}`.",
            "Do not describe the current figures as draft-only unless `generated_figures_are_draft_only` is true in `phase25_experiment_summary.json`.",
            "Do not reuse stale missing-experiment recommendations from earlier quick-validation passes.",
            "",
            "Paper-ready figures in this pass:",
        ]
        ready_lines.extend(f"- {item['figure_id']}" for item in figure_outputs if item.get("paper_ready", False))
        (phase25_dir / "missing_experiments.md").write_text("\n".join(ready_lines) + "\n", encoding="utf-8")

    verified_registry = build_phase25_verified_registry(
        phase25_dir=phase25_dir,
        phase25_status=phase25_status,
        plan=plan,
        df=df,
        comparison_df=comparison_df,
        figure_outputs=figure_outputs,
        table_outputs=table_outputs,
        plot_quality_report=plot_quality_report,
        relative_gains=relative_gains,
    )
    verified_registry_path = phase25_dir / "phase25_verified_registry.json"
    _write_json(verified_registry_path, verified_registry)

    payload = {
        "phase25_status": phase25_status,
        "data_source": data_source,
        "paper_validation_staleness": summary.get("_phase25_paper_validation_staleness", {}),
        "ignored_stale_paper_validation": summary.get("_phase25_ignored_paper_validation", {}),
        "quick_mode": quick_mode,
        "paper_sweep_run_mode": run_mode,
        "num_cases": int(available.get("case_count", 0)),
        "num_results": int(available.get("result_count", 0)),
        "num_comparable_cases": int(len(comparable)),
        "primary_metric": plan.get("primary_metric", {}),
        "proposed_win_count": int(comparable["proposed_win"].sum()) if not comparable.empty and "proposed_win" in comparable.columns else 0,
        "proposed_win_rate": proposed_win_rate_now,
        "proposed_mean_relative_gain": float(sum(relative_gains) / len(relative_gains)) if relative_gains else 0.0,
        "proposed_median_relative_gain": proposed_median_gain_now,
        "best_gain_case_id": best_case_id,
        "worst_gain_case_id": worst_case_id,
        "primary_claim_check": primary_claim_check,
        "strongest_practical_baseline_audit": strongest_practical_audit,
        "figures": figure_outputs,
        "tables": table_outputs,
        "plot_quality_report": plot_quality_report,
        "plot_quality_report_path": str(phase25_dir / "plot_quality_report.json"),
        "contract_consistency_check": contract_consistency_report,
        "contract_consistency_check_path": str(contract_consistency_path),
        "monte_carlo_check_path": str(phase25_dir / "monte_carlo_check.json"),
        "rejection_reason_counts": summary.get("rejection_reason_counts", {}),
        "infeasible_reason_counts": summary.get("infeasible_reason_counts", {}),
        "overall": {
            "num_cases": int(available.get("case_count", 0)),
            "num_results": int(available.get("result_count", 0)),
            "num_comparable_cases": int(len(comparable)),
            "proposed_win_rate": proposed_win_rate_now,
            "proposed_mean_relative_gain": float(sum(relative_gains) / len(relative_gains)) if relative_gains else 0.0,
            "proposed_median_relative_gain": proposed_median_gain_now,
        },
        "paper_ready_figures": [item["figure_id"] for item in figure_outputs if item.get("paper_ready", False)],
        "draft_figures": [item["figure_id"] for item in figure_outputs if item.get("draft_or_final") == "draft"],
        "paper_minimum_ready": phase25_status in {"paper_minimum_ready", "paper_preferred_ready", "high_confidence_ready"},
        "paper_preferred_ready": phase25_status in {"paper_preferred_ready", "high_confidence_ready"},
        "high_confidence_ready": phase25_status == "high_confidence_ready",
        "generated_figures_are_draft_only": phase25_status in draft_or_blocked_statuses,
        "paper_sweep_plan_path": str(phase25_dir / "paper_sweep_plan.json") if phase25_status in draft_or_blocked_statuses else "",
        "missing_experiments_path": str(phase25_dir / "missing_experiments.md") if phase25_status in draft_or_blocked_statuses else "",
        "figure_captions_path": caption_paths.get("figure_captions_path", ""),
        "table_captions_path": caption_paths.get("table_captions_path", ""),
        "method_naming_summary_json_path": method_naming_paths.get("json_path", ""),
        "method_naming_summary_md_path": method_naming_paths.get("md_path", ""),
        "verified_registry_path": str(verified_registry_path),
        "verified_registry_status": verified_registry.get("status", ""),
        "limitations": [
            "Phase 2.5 uses deterministic aggregation only; scatter-trend figures may add a fitted visual guide, but no plotted sample values are smoothed, interpolated, or LLM-generated.",
            (
                "Current figures meet the paper-minimum data-sufficiency gate but should still be interpreted with claim-specific caution."
                if phase25_status in {"paper_minimum_ready", "paper_preferred_ready", "high_confidence_ready"}
                else "Paper preferred defaults depend on chart type; quick mode can only produce draft figures."
            ),
        ],
    }
    write_phase25_experiment_summary_json(phase25_dir / "phase25_experiment_summary.json", payload)

    paper_ready = phase25_status in {"paper_minimum_ready", "paper_preferred_ready", "high_confidence_ready"}
    quality_gate_payload = {
        "status": "passed" if paper_ready else "blocked",
        "phase25_status": phase25_status,
        "allowed_statuses": sorted(paper_ready_statuses),
        "paper_ready": paper_ready,
        "data_source": data_source,
        "paper_sweep_run_mode": run_mode,
        "paper_ready_figures": payload["paper_ready_figures"],
        "draft_figures": payload["draft_figures"],
        "generated_figures_are_draft_only": payload["generated_figures_are_draft_only"],
        "reason": (
            "Phase 2.5 produced paper-ready experimental evidence."
            if paper_ready
            else "Phase 2.5 evidence is not paper-ready; expand the experiment sweep before final synthesis."
        ),
        "source": "phase25_analysis",
    }
    _write_json(phase25_dir / "phase25_quality_gate.json", quality_gate_payload)

    handoff_path = phase25_dir / "phase2_to_phase3_handoff.json"
    handoff = _read_json(handoff_path)
    if handoff:
        gate_summary = handoff.setdefault("phase2_gate_summary", {})
        if isinstance(gate_summary, dict):
            gate_summary["phase25_gate_ok"] = bool(paper_ready)
            gate_summary["phase25_status"] = phase25_status
            gate_summary["phase25_quality_gate"] = "phase2-5/phase25_quality_gate.json"
        _write_json(handoff_path, handoff)
    return {"summary": payload, "available_data_summary": available, "extras": extras}


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministic Phase 2.5 analysis for IEEE WCL-ready experiment package")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--experiment-plan")
    parser.add_argument("--paper-sweep", action="store_true")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.paper_sweep:
        result = run_phase24_paper_sweep_from_plan(args.run_dir, quick=args.quick)
        print(json.dumps(_json_safe(result), ensure_ascii=False, indent=2))
        return
    if not args.experiment_plan:
        raise ValueError("--experiment-plan is required unless --paper-sweep is used")
    result = run_phase25_analysis(args.run_dir, args.experiment_plan)
    print(json.dumps(_json_safe(result["summary"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
