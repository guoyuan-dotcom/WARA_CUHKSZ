from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .utils import compact_text, read_json, read_text, write_text


CONTRACT_VERSION = "phase2_pre6_subagents_v1"


def _lower_join(*parts: Any) -> str:
    return "\n".join(str(part or "") for part in parts).lower()


def _has_any(text: str, needles: list[str]) -> bool:
    lowered = str(text or "").lower()
    return any(needle.lower() in lowered for needle in needles)


def _has_term(text: str, term: str) -> bool:
    lowered = str(text or "").lower()
    escaped = re.escape(str(term or "").lower())
    if not escaped:
        return False
    if re.fullmatch(r"[a-z0-9_]+", term.lower()):
        return bool(re.search(rf"(?<![a-z0-9_]){escaped}(?![a-z0-9_])", lowered))
    return escaped in lowered


def _has_any_term(text: str, terms: list[str]) -> bool:
    return any(_has_term(text, term) for term in terms)


def _has_concrete_uncertainty_model(text: str) -> bool:
    return _has_any(
        text,
        [
            "norm-bounded",
            "norm bounded",
            "ellipsoidal",
            "ellipsoid",
            "box uncertainty",
            "polyhedral",
            "wasserstein",
            "moment-based",
            "moment based",
            "mean and covariance",
            "cantelli",
            "chebyshev",
            "bernstein",
            "gaussian error",
            "sub-gaussian",
            "scenario approximation",
            "sample approximation",
            "sample-average",
            "sample average",
            "support set",
            "dual norm",
        ],
    )


def _uses_uncertainty_or_chance(text: str) -> bool:
    return _has_any(
        text,
        [
            "distributionally robust",
            "ambiguity",
            "chance constraint",
            "chance-constrained",
            "outage",
            "imperfect csi",
            "csi uncertainty",
            "channel uncertainty",
            "uncertainty radius",
            "robust counterpart",
            "safe counterpart",
        ],
    )


def _shared_waveform_needs_receiver_scope(text: str) -> bool:
    return (
        _has_any(
            text,
            [
                "auxiliary signal",
                "auxiliary waveform",
                "auxiliary covariance",
                "non-information signal",
                "non-information waveform",
                "service signal",
                "service waveform",
                "shared covariance",
                "shared waveform",
                "sensing covariance",
                "energy covariance",
                "common waveform",
                "common signal",
                "artificial noise",
                "jamming signal",
                "pilot signal",
            ],
        )
        and _has_any(text, ["sinr", "interference", "denominator", "treated as noise"])
        and not _has_any(
            text,
            [
                "not decoded",
                "not canceled",
                "not cancelled",
                "non-cancelled",
                "non-canceled",
                "treat it as noise",
                "treated as noise",
                "unknown to the receiver",
                "known and canceled",
                "known and cancelled",
                "can be canceled",
                "can be cancelled",
                "jointly decoded",
                "joint decoding",
                "successive interference cancellation",
                "sic",
            ],
        )
    )


def _covariance_needs_physical_scope(text: str) -> bool:
    return (
        _has_any(
            text,
            [
                "covariance matrix",
                "covariance matrices",
                "transmit covariance",
                "beam covariance",
                "lifted beam",
                "lifted precoder",
                "semidefinite beam",
                "psd matrix",
                "psd variable",
            ],
        )
        and _has_any(text, ["beamforming", "transmit", "downlink", "precoding"])
        and not _has_any(
            text,
            [
                "gaussian signaling",
                "multi-stream",
                "multistream",
                "rank-one",
                "rank one",
                "single-stream",
                "single stream",
                "rank recovery",
                "rank-one recovery",
                "gaussian randomization",
                "eigenmode",
                "covariance-domain",
                "covariance domain",
                "random vector",
                "physical realization",
                "physically realized",
            ],
        )
    )


def _evidence_segments(text: str) -> list[str]:
    return [
        segment.strip()
        for segment in re.split(r"[\r\n.;!?]+", str(text or "").lower())
        if segment.strip()
    ]


_NEGATED_ALGORITHM_CONTEXTS = [
    "not ",
    "no ",
    "without ",
    "do not",
    "don't",
    "should not",
    "must not",
    "never",
    "avoid",
    "unnecessary",
    "not needed",
    "not required",
    "inappropriate",
    "rather than",
    "instead of",
]


def _has_positive_algorithm_marker(text: str, markers: list[str]) -> bool:
    for segment in _evidence_segments(text):
        if not any(marker.lower() in segment for marker in markers):
            continue
        if any(negative in segment for negative in _NEGATED_ALGORITHM_CONTEXTS):
            continue
        return True
    return False


def _has_direct_conic_route(text: str) -> bool:
    direct_markers = [
        "direct conic optimization",
        "one convex socp",
        "full problem is already convex",
        "full problem already convex",
        "solve the final power-minimization socp",
        "solve the final socp",
        "solve the socp",
        "safe socp",
        "convex socp",
        "socp solver",
    ]
    return _has_positive_algorithm_marker(text, direct_markers) or (
        _has_positive_algorithm_marker(text, ["second-order cone", "socp"])
        and _has_positive_algorithm_marker(text, ["convex", "globally solves", "conic"])
    )


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        value = str(item or "").strip()
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _problem_family(topic: str, *texts: str) -> str:
    text = _lower_join(topic, *texts)
    if _has_any(text, ["uav", "unmanned aerial", "drone", "aerial base station", "trajectory"]):
        return "uav_trajectory_optimization"
    if _has_any(text, ["downlink", "beamforming", "precoder", "multiuser mimo"]):
        return "downlink_beamforming"
    if _has_any(text, ["uplink", "power control", "sinr target", "minimum sum transmit power"]):
        return "uplink_power_control"
    if _has_any_term(text, ["ris", "irs"]) or _has_any(text, ["reconfigurable intelligent"]):
        return "ris_assisted_optimization"
    if _has_any(text, ["swipt", "energy harvesting", "harvested energy", "eh receiver"]):
        return "swipt_energy_optimization"
    if _has_any(text, ["resource allocation", "spectrum", "subcarrier", "bandwidth"]):
        return "wireless_resource_allocation"
    return "generic_wireless_optimization"


def _mechanisms(topic: str, *texts: str) -> list[str]:
    text = _lower_join(topic, *texts)
    mechanisms: list[str] = []
    if _has_any(text, ["uplink"]):
        mechanisms.append("uplink")
    if _has_any(text, ["downlink"]):
        mechanisms.append("downlink")
    if _has_any(text, ["uav", "unmanned aerial", "drone", "aerial base station"]):
        mechanisms.append("uav")
    if _has_any(text, ["trajectory", "path planning", "flight path"]):
        mechanisms.append("trajectory")
    if _has_any(text, ["propulsion", "flight energy", "mission energy"]):
        mechanisms.append("propulsion_energy")
    if _has_any(text, ["sinr"]):
        mechanisms.append("sinr")
    if _has_any(text, ["beamforming", "precoder", "beamformer"]):
        mechanisms.append("beamforming")
    if _has_any_term(text, ["ris", "irs"]) or _has_any(text, ["reconfigurable intelligent"]):
        mechanisms.append("ris")
    if _has_any(
        text,
        [
            "swipt",
            "energy harvesting",
            "harvested energy",
            "harvested-energy",
            "harvested dc",
            "harvested-dc",
            "rf-to-dc",
            "rectifier",
            "wireless power",
            "powering",
        ],
    ) or _has_any_term(text, ["eh", "wpt"]):
        mechanisms.append("energy_harvesting")
    if _has_any(text, ["crb", "fisher information", "fim"]):
        mechanisms.append("crb")
    if _has_any(text, ["radar", "sensing", "beampattern", "beam pattern"]):
        mechanisms.append("sensing")
    if _has_any(text, ["secrecy", "eavesdropper", "physical-layer security", "wiretap"]):
        mechanisms.append("secrecy")
    return mechanisms


def _objective_sense(*texts: str) -> str:
    text = _lower_join(*texts)
    # Prefer the declared outer optimization problem over inner robust/worst-case
    # operators such as \min_{\Delta} inside a max-min objective.
    outer_problem_patterns = [
        r"\(\\mathcal\{p\}[_0-9a-z]*\)[^.\n]{0,160}\\(?P<sense>max|min)\s*_",
        r"\(\\mathcal\{p\}[_0-9a-z]*\)[^.\n]{0,160}\b(?P<sense>maximize|minimize)\b",
        r"\bp[_\s-]*0\b[^.\n]{0,160}\\(?P<sense>max|min)\s*_",
        r"original optimization problem[^.\n]{0,240}\\(?P<sense>max|min)\s*_",
    ]
    for pattern in outer_problem_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            sense = str(match.group("sense")).lower()
            return "maximize" if sense.startswith("max") else "minimize"
    rate_fairness_terms = [
        "rate",
        "spectral efficiency",
        "spectral-efficiency",
        "throughput",
        "sinr",
        "fairness",
        "user rate",
        "r_k",
    ]
    max_min_rate_patterns = [
        r"max[- ]min[^.\n;]{0,160}(rate|spectral efficiency|throughput|sinr|fairness)",
        r"max(?:imize|imization)?[^.\n;]{0,160}(minimum|min[- ]user|worst[- ]user|worst user)[^.\n;]{0,160}(rate|spectral efficiency|throughput|sinr)",
        r"\\max\s*_[^.\n;]{0,200}\\min\s*_[^.\n;]{0,200}(r_|rate|spectral|sinr|throughput)",
    ]
    if (
        _has_any(text, ["max-min", "max min", "maximize the minimum", "maximize minimum", "maximise the minimum"])
        and _has_any(text, rate_fairness_terms)
    ) or any(re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL) for pattern in max_min_rate_patterns):
        return "maximize"
    strong_minimize = [
        "transmit power minimization",
        "power minimization",
        "minimize transmit power",
        "minimize total transmit power",
        "minimize sum transmit",
        "minimum sum transmit power",
        "minimize sum power",
        "minimization problem",
    ]
    if _has_any(text, strong_minimize) or re.search(
        r"\bmin(?:imize|imization)?\b[^.\n;]{0,120}(transmit power|sum power|total power|energy consumption)",
        text,
    ):
        return "minimize"
    strong_maximize = [
        "utility maximization",
        "rate-energy utility",
        "maximize weighted",
        "maximize the weighted",
        "maximize minimum",
        "maximize the minimum",
        "max-min fairness",
        "maximize sum",
        "sum-rate maximization",
        "weighted sum-rate maximization",
        "max_{",
    ]
    if _has_any(text, strong_maximize) or re.search(
        r"\bmax(?:imize|imization)?\b[^.\n;]{0,120}(utility|sum[- ]?rate|rate|harvest|energy)",
        text,
    ):
        return "maximize"
    has_latex_max = bool(re.search(r"\\max\s*_", text))
    has_latex_min = bool(re.search(r"\\min\s*_", text))
    if has_latex_max and not has_latex_min:
        return "maximize"
    if has_latex_min and not has_latex_max:
        return "minimize"
    if has_latex_max:
        return "maximize"
    if has_latex_min:
        return "minimize"
    if _has_any(text, ["maximize", "maximization"]):
        return "maximize"
    if _has_any(text, ["minimize", "minimization"]):
        return "minimize"
    return "unknown"


def _objective_from_mathematical_contract(contract_payload: str | dict[str, Any] | None) -> tuple[str, str]:
    if isinstance(contract_payload, dict):
        contract = contract_payload
    else:
        try:
            contract = json.loads(str(contract_payload or "{}"))
        except Exception:
            contract = {}
    if not isinstance(contract, dict):
        return "", ""
    objective = contract.get("objective")
    if isinstance(objective, dict):
        sense_raw = str(objective.get("sense") or "").strip().lower()
        if sense_raw in {"max", "maximize", "maximise", "maximization", "maximisation"}:
            sense = "maximize"
        elif sense_raw in {"min", "minimize", "minimise", "minimization", "minimisation"}:
            sense = "minimize"
        else:
            sense = ""
        bits = [
            str(objective.get("sense") or ""),
            str(objective.get("expression") or ""),
            str(objective.get("meaning") or ""),
            json.dumps(objective.get("terms") or [], ensure_ascii=False),
        ]
    else:
        sense = ""
        bits = [str(objective or "")]
    text = "\n".join(item for item in bits if item.strip())
    return sense, text


def _primary_kpis(
    family: str,
    mechanisms: list[str],
    objective_sense: str,
    source_text: str = "",
) -> list[str]:
    text = str(source_text or "").lower()
    objective_text = _objective_text_window(text).lower()
    objective_context = objective_text or text
    has_rate_objective = _has_any(
        objective_context,
        ["rate", "spectral efficiency", "spectral-efficiency", "throughput", "sinr", "r_k", "bps/hz", "bpshz"],
    )
    has_fairness_objective = _has_any(
        objective_context,
        [
            "max-min",
            "max min",
            "maximize the minimum",
            "maximize minimum",
            "minimum achievable",
            "minimum user",
            "worst-user",
            "worst user",
            "fairness",
        ],
    )
    has_sum_rate_objective = _has_any(
        objective_context,
        ["sum-rate", "sum rate", "weighted sum-rate", "weighted sum rate", "sum spectral efficiency"],
    )
    has_power_objective = _has_any(
        objective_context,
        [
            "transmit power minimization",
            "power minimization",
            "minimize transmit power",
            "minimize total power",
            "minimize sum power",
            "minimum sum transmit power",
        ],
    )
    has_harvesting_objective = _has_any(
        objective_context,
        ["harvested", "harvest", "rf-to-dc", "rectifier", "dc power", "energy receiver"],
    )
    has_sensing_objective = _has_any(
        objective_context,
        ["sensing", "radar", "crb", "beampattern", "illumination", "fisher information"],
    )
    has_secrecy_objective = _has_any(objective_context, ["secrecy", "confidential", "wiretap", "eavesdropper"])

    if family == "uav_trajectory_optimization":
        return ["energy_efficiency_bit_per_J", "sum_rate_bpsHz", "propulsion_energy_J"]
    if family == "uplink_power_control":
        return ["sum_power_W", "sum_power_dBm", "sinr_min_dB"]
    if objective_sense == "maximize" and has_rate_objective and has_fairness_objective:
        return ["min_user_rate_bpsHz", "min_spectral_efficiency_bpsHz", "min_sinr_dB", "sum_rate_bpsHz", "total_tx_power_mW"]
    if objective_sense == "maximize" and has_sum_rate_objective:
        return ["sum_rate_bpsHz", "min_user_rate_bpsHz", "total_tx_power_mW"]
    if has_secrecy_objective or "secrecy" in mechanisms:
        return ["worst_case_min_secrecy_rate_bpsHz", "min_secrecy_rate_bpsHz", "sum_secrecy_rate_bpsHz"]
    if objective_sense == "maximize" and has_harvesting_objective:
        return ["min_harvested_dc_mW", "harvested_energy_mW", "sum_rate_bpsHz", "total_tx_power_mW"]
    if has_sensing_objective and "crb" in mechanisms and objective_sense == "minimize":
        return ["crb", "sensing_mse", "beampattern_error", "total_tx_power_mW"]
    if objective_sense == "maximize" and has_sensing_objective:
        return ["sensing_illumination_mW", "sensing_snr_dB", "sum_rate_bpsHz", "total_tx_power_mW"]
    if objective_sense == "minimize" and has_power_objective:
        kpis = ["P_tx_mW", "sum_power_mW", "total_tx_power_mW"]
        if has_harvesting_objective or "energy_harvesting" in mechanisms:
            kpis.extend(["min_harvested_dc_mW", "harvested_energy_mW"])
        if has_sensing_objective or "sensing" in mechanisms or "crb" in mechanisms:
            kpis.extend(["crb", "sensing_illumination_mW"])
        return _dedupe_preserve_order(kpis)
    if "beamforming" in mechanisms and objective_sense == "maximize":
        return ["min_user_rate_bpsHz", "sum_rate_bpsHz", "min_sinr_dB", "total_tx_power_mW"]
    if "beamforming" in mechanisms and objective_sense == "minimize":
        kpis = ["P_tx_mW", "sum_power_mW", "total_tx_power_mW"]
        if "energy_harvesting" in mechanisms:
            kpis.extend(["harvested_energy_mW", "min_harvested_dc_mW"])
        if "sensing" in mechanisms or "crb" in mechanisms:
            kpis.extend(["sensing_illumination_mW", "crb"])
        return _dedupe_preserve_order(kpis)
    if objective_sense == "minimize":
        kpis = ["P_tx_mW", "objective_value", "sum_power_mW", "total_tx_power_mW"]
        if "energy_harvesting" in mechanisms:
            kpis.extend(["harvested_energy_mW", "min_harvested_dc_mW"])
        if "sensing" in mechanisms or "crb" in mechanisms:
            kpis.extend(["sensing_illumination_mW", "crb"])
        return _dedupe_preserve_order(kpis)
    if "energy_harvesting" in mechanisms:
        return ["sum_rate_bpsHz", "harvested_energy_mW", "min_harvested_dc_mW"]
    if "ris" in mechanisms:
        return ["sum_rate_bpsHz", "min_user_rate_bpsHz"]
    return ["physical_utility"]


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        item = str(value or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _metric_semantic_keywords(metric: str) -> list[str]:
    lowered = str(metric or "").strip().lower()
    keywords: list[str] = []
    if any(token in lowered for token in ("eta_service", "service_level", "service_margin", "normalized_service", "tau")):
        keywords.extend(
            [
                "eta",
                "\\eta",
                "service level",
                "service-level",
                "normalized service",
                "minimum normalized",
                "worst normalized",
                "service balancing",
                "service-balancing",
                "utility",
                "surplus",
            ]
        )
    if any(token in lowered for token in ("harvest", "eh", "dc", "energy")):
        keywords.extend(["harvest", "harvested", "energy", "dc", "eh", "rf-to-dc", "rectifier", "powering"])
    if "min_harvest" in lowered or "worst" in lowered:
        keywords.extend(["worst", "minimum", "min", "max-min", "epigraph", "fairness"])
    if any(token in lowered for token in ("sum_rate", "rate", "throughput", "bpshz", "spectral")):
        keywords.extend(["sum-rate", "sum rate", "rate", "throughput", "spectral efficiency", "sinr"])
    if "secrecy" in lowered:
        keywords.extend(["secrecy", "confidential", "eavesdropper"])
    if any(token in lowered for token in ("sensing", "radar", "crb", "beampattern", "illumination")):
        keywords.extend(["sensing", "radar", "crb", "beampattern", "illumination"])
    if any(token in lowered for token in ("power", "p_tx", "sum_power", "energy_consumption")):
        keywords.extend(["transmit power", "sum power", "total power", "power consumption", "resource"])
    if "efficiency" in lowered or "bit_per_j" in lowered:
        keywords.extend(["energy efficiency", "bit/j", "bit per joule"])
    return _dedupe_preserve_order(keywords)


def _objective_text_window(text: str) -> str:
    source = str(text or "")
    sentences = re.split(r"(?<=[.!?。；;])\s+|\n+", source)
    selected = [
        item
        for item in sentences
        if _has_any(
            item.lower(),
            [
                "objective",
                "maximize",
                "maximization",
                "minimize",
                "minimization",
                "max-min",
                "worst",
                "epigraph",
                "\\max",
                "\\min",
            ],
        )
    ]
    return "\n".join(selected) if selected else source


def _score_metric_against_objective(metric: str, source_text: str, objective_sense: str) -> float:
    metric_text = str(metric or "").strip().lower()
    if not metric_text:
        return -1.0
    full_text = str(source_text or "").lower()
    objective_text = _objective_text_window(full_text).lower()
    score = 0.0
    compact_metric = metric_text.replace("_", " ")
    service_level_objective = _has_any(
        objective_text,
        [
            "eta",
            "\\eta",
            "service level",
            "service-level",
            "normalized service",
            "minimum normalized",
            "worst normalized",
            "service balancing",
            "service-balancing",
        ],
    )
    if metric_text in full_text or compact_metric in full_text:
        score += 1.5
    if metric_text in objective_text or compact_metric in objective_text:
        score += 3.0
    for keyword in _metric_semantic_keywords(metric_text):
        if keyword in full_text:
            score += 1.0
        if keyword in objective_text:
            score += 2.0
    if objective_sense == "minimize" and any(token in metric_text for token in ("power", "p_tx", "energy_consumption", "cost", "latency")):
        score += 1.0
        if re.search(r"\bmin(?:imize|imizes|imization)?\b[^.\n;]{0,120}(transmit power|sum power|total power|energy consumption|resource)", objective_text):
            score += 5.0
        if "p_tx" in metric_text and "transmit power" in objective_text:
            score += 2.0
    if objective_sense == "minimize" and any(token in metric_text for token in ("harvest", "eh", "dc")):
        if _has_any(objective_text, ["constraint", "subject to", "s.t.", "requirement"]):
            score -= 8.0
    if objective_sense == "maximize" and any(token in metric_text for token in ("rate", "harvest", "sensing", "secrecy", "efficiency", "utility")):
        score += 0.5
    max_min_rate_objective = _has_any(
        objective_text,
        [
            "max-min",
            "max min",
            "maximize the minimum",
            "maximize minimum",
            "minimum achievable rate",
            "minimum user rate",
            "worst-user rate",
            "worst user rate",
            "min_k r",
            "\\min",
        ],
    ) and _has_any(
        objective_text,
        ["rate", "spectral efficiency", "spectral-efficiency", "throughput", "sinr", "r_k", "fairness"],
    )
    if objective_sense == "maximize" and max_min_rate_objective:
        if any(token in metric_text for token in ("min_user_rate", "minimum_user_rate", "worst_user_rate", "min_rate")):
            score += 12.0
        elif any(token in metric_text for token in ("sum_rate", "avg_rate", "mean_rate")):
            score += 2.0
        elif any(token in metric_text for token in ("harvest", "sensing", "illumination", "crb", "energy")):
            score -= 8.0
    if any(token in metric_text for token in ("eta_service", "service_level", "service_margin", "normalized_service", "tau")):
        if service_level_objective:
            score += 14.0
    elif service_level_objective and any(token in metric_text for token in ("harvest", "energy", "rate", "sensing")):
        # A service-balancing epigraph may contain rate/EH/sensing constraints,
        # but the optimized paper KPI is the normalized service level, not one
        # decomposed constraint component.
        score -= 6.0
    if "min_harvested_dc" in metric_text and _has_any(objective_text, ["worst", "minimum", "min", "max-min", "epigraph"]):
        score += 3.0
    if "sum_rate" in metric_text and _has_any(objective_text, ["harvest", "rectifier", "rf-to-dc", "powering"]) and not _has_any(objective_text, ["sum-rate", "sum rate", "throughput"]):
        score -= 2.0
    return score


def _objective_aligned_metric(
    *,
    candidate_metrics: list[str],
    source_text: str,
    objective_sense: str,
    fallback_metric: str = "",
) -> str:
    candidates = _dedupe_preserve_order(candidate_metrics)
    if not candidates:
        return str(fallback_metric or "physical_utility")
    scored = [
        (_score_metric_against_objective(metric, source_text, objective_sense), -idx, metric)
        for idx, metric in enumerate(candidates)
    ]
    scored.sort(reverse=True)
    best_score, _, best_metric = scored[0]
    fallback = str(fallback_metric or candidates[0])
    if best_score <= 0:
        return fallback
    return best_metric


def _metric_is_objective_like_name(metric: str) -> bool:
    text = str(metric or "").strip().lower().replace("-", "_")
    if not text:
        return False
    return (
        any(token in text for token in ("objective", "utility", "eta_service", "service_level", "normalized_service", "tau"))
        or ("service" in text and "level" in text)
        or ("margin" in text and not any(token in text for token in ("violation", "gap", "residual")))
    )


def _metric_is_diagnostic_name(metric: str) -> bool:
    text = str(metric or "").strip().lower().replace("-", "_")
    return any(
        token in text
        for token in (
            "feasible",
            "feasibility",
            "violation",
            "residual",
            "status",
            "runtime",
            "solve_time",
            "iteration",
        )
    )


def _supporting_physical_metric(candidate_metrics: list[str], primary_metric: str) -> str:
    primary = str(primary_metric or "").strip()
    for metric in _dedupe_preserve_order(candidate_metrics):
        if metric == primary:
            continue
        if _metric_is_objective_like_name(metric) or _metric_is_diagnostic_name(metric):
            continue
        return metric
    return primary


def _diagnostic_metrics() -> list[str]:
    return ["feasible", "constraint_violation_max"]


def build_problem_contract(
    *,
    topic: str,
    handoff: dict[str, Any],
    system_model_md: str,
    problem_formulation_md: str,
    core_theory_package_md: str,
    mathematical_contract_json: str | dict[str, Any] | None = None,
) -> dict[str, Any]:
    math_sense, math_objective_text = _objective_from_mathematical_contract(mathematical_contract_json)
    family = _problem_family(topic, system_model_md, problem_formulation_md, core_theory_package_md, math_objective_text)
    mechanisms = _mechanisms(topic, system_model_md, problem_formulation_md, core_theory_package_md, math_objective_text)
    sense = math_sense or _objective_sense(math_objective_text, problem_formulation_md, core_theory_package_md, topic)
    source_text = "\n".join([topic, system_model_md, problem_formulation_md, core_theory_package_md, math_objective_text])
    primary_kpis = _primary_kpis(family, mechanisms, sense, source_text)
    service_objective = _has_any(
        source_text.lower(),
        [
            "eta",
            "\\eta",
            "service level",
            "service-level",
            "normalized service",
            "minimum normalized",
            "worst normalized",
            "service balancing",
            "service-balancing",
        ],
    ) and _has_any(source_text.lower(), ["maximize", "maximization", "\\max", "max-min"])
    if service_objective:
        primary_kpis = ["eta_service_level", "min_normalized_service_margin", "service_margin_tau", *primary_kpis]
    if (
        "energy_harvesting" in mechanisms
        and not service_objective
        and _has_any(source_text.lower(), ["worst", "max-min", "minimum harvested", "epigraph"])
    ):
        primary_kpis = ["min_harvested_dc_mW", *primary_kpis]
    primary_kpis = _dedupe_preserve_order(primary_kpis)
    primary_kpis = _dedupe_preserve_order(
        [
            _objective_aligned_metric(
                candidate_metrics=primary_kpis,
                source_text=source_text,
                objective_sense=sense,
                fallback_metric=primary_kpis[0] if primary_kpis else "physical_utility",
            ),
            *primary_kpis,
        ]
    )
    return {
        "contract_version": CONTRACT_VERSION,
        "agent_role": "problem_contract_builder",
        "topic": topic,
        "phase1_title": handoff.get("final_title", ""),
        "problem_family": family,
        "objective_sense": sense,
        "mechanisms": mechanisms,
        "primary_physical_kpis": primary_kpis,
        "diagnostic_metrics": _diagnostic_metrics(),
        "required_contracts": {
            "physical_model_before_surrogate": True,
            "standard_optimization_layout": True,
            "all_theoretical_claims_need_proof_or_scope": True,
            "phase24_must_implement_phase3_algorithm": True,
            "phase25_must_compare_against_fair_practical_benchmark": True,
        },
        "source_artifacts": {
            "system_model_md": "phase2-1/system_model.md",
            "problem_formulation_md": "phase2-1/problem_formulation.md",
            "core_theory_package_md": "phase2-1/core_theory_package.md",
        },
    }


TRACTABILITY_ROUTE_DEFINITIONS: dict[str, dict[str, str]] = {
    "convex_direct": {
        "meaning": "The original problem is convex, or it has an exact clean convex/conic/LP representation under the frozen model.",
        "technical_focus": "Show that the selected variables and constraints preserve the wireless mechanism, derive the exact convex/conic representation, solve it directly, and extract structural or operating-regime insight instead of stopping at 'use a solver'.",
        "claim_scope": "Exactness/global optimality may be claimed only for the stated convex problem and only when the formulation, equivalence mapping, and solver assumptions support it; if the route lacks mechanism preservation or insight, mark the direction as weak rather than padding the method.",
    },
    "structured_nonconvex": {
        "meaning": "The wireless mechanism naturally creates nonconvex coupling, such as interference, bilinear controls, nonlinear hardware/utility response, fractional utility, trajectory coupling, or coupled resource blocks.",
        "technical_focus": "Use a scoped surrogate, alternating, decomposition, block-update, or problem-specific tractable route that preserves the original physical objective for evaluation.",
        "claim_scope": "Nonconvexity can be a contribution only with a credible solution route; stationarity/monotonicity claims need assumptions and proof sketches.",
    },
    "relaxation_recovery": {
        "meaning": "The route relies on lifting, semidefinite or other convex relaxation, rank relaxation, randomization, or recovery from a relaxed solution.",
        "technical_focus": "Separate the relaxed problem, recovery step, and final feasible-solution evaluation.",
        "claim_scope": "Do not transfer optimality or convergence of the relaxed problem to recovered physical variables unless a proof is supplied.",
    },
    "mixed_discrete_or_manifold": {
        "meaning": "The formulation includes scheduling, association, mode selection, integer/binary variables, mobile positions, unit-modulus phases, or manifold-like controls.",
        "technical_focus": "Use decomposition, alternating updates, projection, rounding, bounded search, or scoped manifold/discrete heuristics with shared physical evaluation.",
        "claim_scope": "Avoid global/KKT claims unless the discrete/manifold step is solved exactly under stated assumptions.",
    },
    "heuristic_empirical": {
        "meaning": "The frozen model is meaningful, but the available route is best presented as a reproducible heuristic/proxy with empirical validation rather than theorem-level optimization.",
        "technical_focus": "Expose the executable update, diagnostics, and true objective/constraint evaluator; make paper claims evidence- and regime-based.",
        "claim_scope": "Do not claim theorem-level optimality, exactness, or stationarity; use empirical and mechanism-insight language.",
    },
}


def build_tractability_route_policy(
    *,
    topic: str,
    handoff: dict[str, Any] | None,
    mathematical_contract_json: str,
    system_model_md: str,
    problem_formulation_md: str,
    core_theory_package_md: str,
    problem_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Controller-side route suggestion before Phase 2.2 theory generation.

    The LLM still audits the mathematics, but it should not default every topic
    to the same nonconvex-to-convexification story.
    """

    handoff = handoff or {}
    problem_contract = problem_contract or {}
    text = _lower_join(
        topic,
        json.dumps(handoff, ensure_ascii=False, default=str),
        mathematical_contract_json,
        system_model_md,
        problem_formulation_md,
        core_theory_package_md,
        json.dumps(problem_contract, ensure_ascii=False, default=str),
    )
    route = "heuristic_empirical"
    evidence: list[str] = []

    if _has_positive_algorithm_marker(
        text,
        [
            "sdr",
            "sdp",
            "semidefinite",
            "rank-one",
            "rank one",
            "rank recovery",
            "gaussian randomization",
            "lifting",
            "lifted covariance",
        ],
    ):
        route = "relaxation_recovery"
        evidence.append("relaxation_or_recovery_terms")
    elif _has_positive_algorithm_marker(
        text,
        [
            "binary",
            "integer",
            "mixed-integer",
            "mixed integer",
            "scheduling",
            "association",
            "mode selection",
            "unit-modulus",
            "unit modulus",
            "phase shift",
            "trajectory",
            "uav",
            "waypoint",
            "manifold",
            "movable antenna",
            "antenna coordinate",
            "movable coordinate",
            "spacing constraint",
            "aperture constraint",
        ],
    ):
        route = "mixed_discrete_or_manifold"
        evidence.append("discrete_manifold_or_mobility_terms")
    elif _has_positive_algorithm_marker(
        text,
        [
            "nonconvex",
            "non-convex",
            "nonconcave",
            "non-concave",
            "bilinear",
            "fractional",
            "product",
            "interference coupling",
            "sinr coupling",
            "nonlinear eh",
            "non-linear eh",
            "sigmoid",
            "logistic",
            "saturation",
            "successive convex",
            "sca",
            "wmmse",
            "majorization",
            "minorization",
            "alternating optimization",
        ],
    ):
        route = "structured_nonconvex"
        evidence.append("structured_nonconvexity_or_surrogate_terms")
    elif _has_any(
        text,
        [
            "convex",
            "socp",
            "second-order cone",
            "linear program",
            " lp",
            "`lp`",
            "affine",
            "conic",
            "fixed point",
            "spectral radius",
        ],
    ):
        route = "convex_direct"
        evidence.append("direct_convex_or_reference_solver_terms")

    if _has_any(text, ["heuristic", "proxy", "candidate search"]) and route not in {
        "convex_direct",
        "relaxation_recovery",
    }:
        route = "heuristic_empirical"
        evidence.append("heuristic_or_proxy_terms")

    if route == "heuristic_empirical" and not evidence:
        evidence.append("no_clear_solver_route_detected")

    confidence = "medium" if evidence else "low"
    if route in {"relaxation_recovery", "mixed_discrete_or_manifold", "structured_nonconvex"} and len(evidence) >= 1:
        confidence = "medium"
    if route == "convex_direct" and _has_any(text, ["convex", "socp", "linear program", "fixed point"]):
        confidence = "medium"

    return {
        "contract_version": CONTRACT_VERSION,
        "agent_role": "tractability_route_controller",
        "selected_route": route,
        "confidence": confidence,
        "evidence": evidence,
        "allowed_routes": TRACTABILITY_ROUTE_DEFINITIONS,
        "controller_policy": [
            "Phase 2.2 may override selected_route if the frozen mathematics supports a different route, but it must explain the override.",
            "Do not force nonconvexity; do not reject natural nonconvexity when it has a credible solution route.",
            "If the route is convex_direct, contribution must come from mechanism-preserving formulation, exact tractability, structural/operating-regime insight, or evaluated performance gain; do not stop at 'solve with CVX'.",
            "If the route is structured_nonconvex, nonconvexity can be part of the contribution only when it is mechanism-driven and solvable under scoped assumptions.",
            "If the route is relaxation_recovery, separate relaxed-problem claims from recovered physical-solution claims.",
            "If the route is heuristic_empirical, avoid theorem-level language and require reproducible empirical evidence.",
        ],
    }


def tractability_route_policy_prompt_block(policy: dict[str, Any] | None) -> str:
    payload = policy or {
        "selected_route": "heuristic_empirical",
        "confidence": "low",
        "allowed_routes": TRACTABILITY_ROUTE_DEFINITIONS,
    }
    return (
        "[Controller tractability route policy]\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n"
        "Use this as a routing prior, not as a mathematical proof. Phase 2.2 may choose a different final route "
        "only if the frozen mathematical contract and problem formulation justify it explicitly."
    )


def audit_model_contract(
    *,
    problem_contract: dict[str, Any],
    system_model_md: str,
    problem_formulation_md: str,
    core_theory_package_md: str,
) -> dict[str, Any]:
    text = _lower_join(system_model_md, problem_formulation_md, core_theory_package_md)
    errors: list[str] = []
    warnings: list[str] = []
    family = str(problem_contract.get("problem_family") or "")
    mechanisms = set(problem_contract.get("mechanisms") or [])

    if "energy_harvesting" in mechanisms and _has_any(text, ["independent", "symbol", "stream"]):
        if re.search(r"\|\s*[^|]*(sum|\\sum)[^|]*w[^|]*\|\s*\^?\s*2", text):
            errors.append(
                "EH RF power appears to square a coherent sum of beam vectors even though independent streams are mentioned."
            )
    if "crb" in mechanisms and not _has_any(text, ["fisher", "fim", "jacobian", "derivative"]):
        errors.append("CRB is present but the observation/FIM-level model is not explicit.")
    if "ris" in mechanisms and _has_any(text, ["ue-mounted ris", "ris at the ue", "ris on the ue"]):
        warnings.append("RIS is placed at the UE; prefer infrastructure/receiver-side placement unless explicitly justified.")
    if family == "uplink_power_control":
        if not _has_any(text, ["spectral radius", "interference matrix", "fixed point", "linear program", "lp"]):
            warnings.append("Uplink SINR power control should expose the interference mapping or LP/fixed-point structure.")
    if _has_any(text, ["weighted objective"]) and not any(
        kpi.lower() in text for kpi in problem_contract.get("primary_physical_kpis", [])
    ):
        warnings.append("Weighted objective appears without clearly exposing decomposed physical KPIs.")
    if _uses_uncertainty_or_chance(text) and not _has_concrete_uncertainty_model(text):
        warnings.append(
            "Robust/chance/ambiguity model is under-specified. The formulation must name a concrete uncertainty/ambiguity family and parameters before downstream theory/code generation."
        )
    if _shared_waveform_needs_receiver_scope(text):
        warnings.append(
            "Shared/common/sensing/energy waveform is treated in the SINR/interference model without a receiver decoding/cancellation assumption."
        )
    if _covariance_needs_physical_scope(text):
        warnings.append(
            "Matrix-valued transmit controls are present without physical signaling, rank-one, or recovery/realization scope."
        )
    if re.search(r"epsilon[^.\n]{0,90}(outage|chance)", text) and re.search(
        r"epsilon[^.\n]{0,90}(uncertainty|radius|csi|channel)", text
    ):
        warnings.append("Epsilon may be used for both outage/chance tolerance and uncertainty radius; verify notation roles before freezing the model.")
    if _has_any(text, ["transmit power minimization", "minimize transmit power", "minimum transmit power"]) and not _has_any(
        text, ["sinr", "rate", "harvest", "sensing", "outage", "reliability", "qos", "quality-of-service"]
    ):
        warnings.append("Transmit-power minimization is not tied to a nontrivial service/reliability regime.")

    return {
        "contract_version": CONTRACT_VERSION,
        "agent_role": "model_auditor",
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
    }


def build_algorithm_contract(
    *,
    topic: str,
    problem_contract: dict[str, Any],
    convexity_audit_md: str,
    reformulation_path_md: str,
) -> dict[str, Any]:
    text = _lower_join(topic, convexity_audit_md, reformulation_path_md)
    family = str(problem_contract.get("problem_family") or "")
    route_decision = ""
    route_text = f"{convexity_audit_md}\n{reformulation_path_md}"
    for pattern in (
        r"`?selected_route`?\s*[:=\-]\s*`?([a-z_]+)`?",
        r"\|\s*selected_route\s*\|\s*`?([a-z_]+)`?\s*\|",
        r"selected\s+route\s*[:=\-]\s*`?([a-z_]+)`?",
    ):
        route_match = re.search(pattern, route_text, flags=re.IGNORECASE)
        if route_match and route_match.group(1).lower() in TRACTABILITY_ROUTE_DEFINITIONS:
            route_decision = route_match.group(1).lower()
            break
    if not route_decision:
        for candidate in TRACTABILITY_ROUTE_DEFINITIONS:
            if f"`{candidate}`" in route_text.lower() or re.search(rf"\b{re.escape(candidate)}\b", route_text, flags=re.I):
                route_decision = candidate
                break
    algorithm_family = "unspecified"
    if family == "uplink_power_control" and _has_positive_algorithm_marker(
        text,
        ["fixed point", "spectral radius", "linear program", " lp", "`lp`", "p=(i-f)", "(i-f)^{-1}"],
    ):
        algorithm_family = "fixed_point_or_lp_reference"
    elif route_decision == "convex_direct":
        algorithm_family = "direct_optimization"
    elif route_decision == "relaxation_recovery":
        algorithm_family = "sdp_or_sdr"
    elif route_decision in {"mixed_discrete_or_manifold", "heuristic_empirical"}:
        algorithm_family = "heuristic"
    elif _has_direct_conic_route(text):
        algorithm_family = "direct_optimization"
    elif _has_positive_algorithm_marker(text, ["wmmse"]):
        algorithm_family = "wmmse_block_coordinate"
    elif _has_positive_algorithm_marker(text, ["sca", "successive convex", "mm", "majorization"]):
        algorithm_family = "sca_or_mm"
    elif _has_positive_algorithm_marker(text, ["sdp", "sdr", "semidefinite"]):
        algorithm_family = "sdp_or_sdr"
    elif _has_positive_algorithm_marker(text, ["greedy", "heuristic"]):
        algorithm_family = "heuristic"
    elif route_decision == "structured_nonconvex":
        algorithm_family = "heuristic"
    elif _objective_sense(convexity_audit_md, reformulation_path_md) in {"minimize", "maximize"}:
        algorithm_family = "direct_optimization"

    proof_obligations: list[str] = []
    if _uses_uncertainty_or_chance(text):
        proof_obligations.append(
            "Name the concrete uncertainty/ambiguity model and deterministic counterpart; do not use a Phi-style safe-constraint placeholder."
        )
    if _has_any(text, ["covariance", "\\mathbf{W}", "\\mathbf{S}"]):
        proof_obligations.append("Explain the physical signaling, rank-one restriction, or recovery scope for matrix-valued transmit controls.")
    if _has_any(
        text,
        [
            "shared covariance",
            "shared waveform",
            "sensing covariance",
            "energy covariance",
            "auxiliary signal",
            "auxiliary waveform",
            "service signal",
            "\\mathbf{S}",
        ],
    ):
        proof_obligations.append("Carry the receiver information-pattern assumption into the affected metric and algorithm narrative.")
    if algorithm_family in {"sdp_or_sdr", "sca_or_mm", "wmmse_block_coordinate"}:
        proof_obligations.append("Scope convergence to exact block/surrogate assumptions and state stationarity limits.")
    if algorithm_family == "fixed_point_or_lp_reference":
        proof_obligations.append("State feasibility condition and convergence/reference optimality conditions.")
    proof_obligations.append("Do not use theorem/KKT/global-optimal language unless the proof is present.")

    mechanisms_set = set(problem_contract.get("mechanisms") or [])
    nonlinear_eh_direct = "energy_harvesting" in mechanisms_set and _has_any(
        text, ["nonlinear", "rectifier", "harvested dc", "harvested-dc", "inverse"]
    )
    direct_solver_block = "covariance_sdp_solve" if _has_any(text, ["sdp", "semidefinite", "covariance"]) else "direct_convex_solver_call"
    direct_update_blocks = [direct_solver_block, "feasibility_acceptance_and_kpi_evaluation"]
    if nonlinear_eh_direct:
        direct_update_blocks.insert(0, "inverse_eh_threshold_update")

    if algorithm_family == "direct_optimization":
        execution_contract = {
            "state_keys": ["primal_solution", "method", "iteration", "solver_status"],
            "initialization": "Construct solver coefficients from the frozen variables, parameters, and reformulation-only constants; do not introduce new physical controls.",
            "update_blocks": direct_update_blocks,
            "projection": "Use the convex feasible set directly; if the solver returns an infeasible or invalid instance, reject the candidate rather than relaxing physical constraints.",
            "objective_evaluator": "Evaluate the frozen physical objective for every method with the same units and objective sense declared in the mathematical contract.",
            "constraint_evaluator": "Evaluate all frozen physical constraints and reformulation-equivalent constraints with shared code for proposed and baselines.",
            "stopping_rule": "Stop after the direct convex/conic solver satisfies the configured primal-dual tolerance, or report infeasible/solver_failure without changing the problem.",
            "allowed_approximations": [
                "first-order conic solvers may replace interior-point solvers when dimensions require it",
                "approximations must be disclosed in metadata.approximations",
                "post-processing diagnostics must not replace the reported physical objective or constraints",
            ],
        }
    elif family == "uav_trajectory_optimization":
        execution_contract = {
            "state_keys": ["trajectory_xy", "power_W", "schedule", "method", "iteration"],
            "initialization": "Initialize a feasible closed UAV path within the mission-time and speed constraints; initialize scheduling and power without changing the frozen user geometry.",
            "update_blocks": [
                "trajectory_update_under_speed_and_endpoint_constraints",
                "user_scheduling_or_time_allocation_update",
                "transmit_power_update_under_Pmax",
                "propulsion_energy_aware_feasibility_projection",
            ],
            "projection": "Project UAV slot-to-slot displacement onto max_speed*slot_duration, preserve start/end positions, and clip transmit power to the configured cap.",
            "objective_evaluator": "Evaluate energy efficiency as delivered communication bits divided by communication plus propulsion energy; also emit decomposed sum-rate and propulsion-energy KPIs.",
            "constraint_evaluator": "Evaluate speed, endpoint, minimum-rate, power, and finite-metric violations with shared code for all methods.",
            "stopping_rule": "Stop on max_iterations or small energy-efficiency change after a feasible/non-worsening update.",
            "allowed_approximations": [
                "slot-wise trajectory heuristics may replace unavailable nonconvex trajectory solvers",
                "all approximations must be disclosed in metadata.approximations",
                "the reported energy-efficiency metric must always include true propulsion energy",
            ],
        }
    elif family == "downlink_beamforming" and "energy_harvesting" in set(problem_contract.get("mechanisms") or []):
        execution_contract = {
            "state_keys": ["W_real", "W_imag", "rho", "method", "iteration"],
            "initialization": "Construct feasible beam directions and power-splitting ratios from the canonical SWIPT channel model; never initialize by changing the frozen physical constraints.",
            "update_blocks": [
                "beamforming_update_under_power_budget",
                "rectifier_aware_power_splitting_update",
                "feasibility_projection_or_candidate_rejection",
            ],
            "projection": "Project transmit beams onto the configured BS power budget and keep rho in the open interval (0, 1).",
            "objective_evaluator": "Evaluate the same physical rate-energy utility for every method; also emit decomposed sum-rate and harvested-energy KPIs.",
            "constraint_evaluator": "Evaluate SINR, nonlinear harvested-energy, power-budget, and rho-bound violations using the same functions for proposed and baselines.",
            "stopping_rule": "Stop on max_iterations or small physical-objective change after a feasible/non-worsening update.",
            "allowed_approximations": [
                "closed-form or grid candidate updates may replace unavailable convex subproblem solvers",
                "approximations must be disclosed in metadata.approximations",
                "surrogate objectives must not replace the reported physical KPI definitions",
            ],
        }
    else:
        execution_contract = {
            "state_keys": ["method", "iteration"],
            "initialization": "Use the frozen canonical configuration and declared controls only.",
            "update_blocks": ["proposed_algorithm_update", "feasibility_projection_or_candidate_rejection"],
            "projection": "Project only declared control variables onto the frozen constraints.",
            "objective_evaluator": "Evaluate the frozen physical objective for every method.",
            "constraint_evaluator": "Evaluate all frozen physical constraints with shared code for proposed and baselines.",
            "stopping_rule": "Stop on max_iterations or convergence_tol from the algorithm contract.",
            "allowed_approximations": [
                "lightweight numerical approximations are allowed only when Phase 2.3 is not solver-ready",
                "approximations must be disclosed in metadata.approximations",
            ],
        }

    return {
        "contract_version": CONTRACT_VERSION,
        "agent_role": "algorithm_contract_builder",
        "topic": topic,
        "problem_family": family,
        "tractability_route": route_decision or "not_declared",
        "algorithm_family": algorithm_family,
        "objective_sense": problem_contract.get("objective_sense", "unknown"),
        "algorithm_execution_contract": execution_contract,
        "proof_obligations": proof_obligations,
        "implementation_obligations": [
            "Phase24 proposed method must implement the named algorithm family.",
            "Phase24 baseline methods must optimize/evaluate the same physical objective and constraints.",
            "Phase25 plots must use physical KPIs, not only weighted objectives.",
        ],
    }


def audit_theory_contract(
    *,
    algorithm_contract: dict[str, Any],
    algorithm_md: str,
    convergence_or_complexity_md: str,
) -> dict[str, Any]:
    text = _lower_join(algorithm_md, convergence_or_complexity_md)
    errors: list[str] = []
    warnings: list[str] = []
    algorithm_family = str(algorithm_contract.get("algorithm_family") or "")

    claims_global_or_exact = _has_any(text, ["globally optimal", "global optimum", "exact relaxation"])
    scoped_to_safe_problem = _has_any(
        text,
        [
            "for the conservative socp",
            "for the safe socp",
            "for the conservative approximation",
            "for the safe approximation",
            "for the safe problem",
            "safe transmit power",
            "not global optimality for the original",
            "not global optimality for original",
            "not the global optimum of the original",
            "not claimed for the original",
        ],
    )
    has_proof_scope = _has_any(text, ["proof", "under", "sufficient condition", "reference solver"])
    if claims_global_or_exact and not (has_proof_scope or scoped_to_safe_problem):
        errors.append("Global/exact optimality is claimed without proof scope.")
    if _has_any(text, ["kkt", "stationary point", "monotonic convergence"]) and algorithm_family in {"sdp_or_sdr", "sca_or_mm"}:
        if not _has_any(text, ["under exact", "surrogate", "rank-one", "if", "assumption"]):
            errors.append("Stationarity/convergence language is not scoped to exact surrogate/block assumptions.")
    if algorithm_family == "heuristic" and _has_any(text, ["theorem", "guarantee", "kkt"]):
        warnings.append("Heuristic algorithm should not be presented with theorem-like guarantees unless proved.")
    if "weighted objective" in text and not _has_any(text, ["sum rate", "power", "energy", "sinr", "feasible"]):
        warnings.append("Algorithm/evidence narrative still centers a weighted objective without physical KPIs.")
    if re.search(r"(\\Phi|Phi_m|\bphi_m\b)", text) and (
        not _has_concrete_uncertainty_model(text)
        or _has_any(text, ["selected ambiguity model", "chosen ambiguity model", "denoted by", "placeholder"])
    ):
        warnings.append(
            "Theory still uses a Phi-style safe-counterpart placeholder instead of a concrete uncertainty model and deterministic expression."
        )
    if _uses_uncertainty_or_chance(text) and _has_any(text, ["safe conic", "safe counterpart", "robust counterpart"]):
        if not _has_concrete_uncertainty_model(text):
            warnings.append("Safe/robust counterpart claim lacks a concrete uncertainty or ambiguity model.")
    if _has_any(text, ["rank recovery is unnecessary", "no rank recovery", "without rank recovery"]) and not _has_any(
        text, ["gaussian signaling", "multi-stream", "multistream", "covariance-domain", "high-rank covariance is physically implemented"]
    ):
        warnings.append("Recovery/rank issues are dismissed without explaining the physical implementation of the matrix-valued transmit design.")
    if _shared_waveform_needs_receiver_scope(text):
        warnings.append("An auxiliary/common/service signal appears in a receiver metric without receiver information-pattern scope.")

    return {
        "contract_version": CONTRACT_VERSION,
        "agent_role": "theory_auditor",
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
    }


def build_claim_map(
    *,
    topic: str,
    problem_contract: dict[str, Any],
    algorithm_contract: dict[str, Any],
    algorithm_md: str,
    convergence_or_complexity_md: str,
    experiment_blueprint_md: str = "",
) -> dict[str, Any]:
    family = str(problem_contract.get("problem_family") or "")
    objective_sense = str(problem_contract.get("objective_sense") or "unknown")
    kpis = list(problem_contract.get("primary_physical_kpis") or [])
    claims: list[dict[str, Any]] = []

    if family == "uplink_power_control":
        claims.append(
            {
                "claim_id": "C1_power_saving",
                "claim_type": "performance_advantage",
                "metric": "sum_power_W",
                "direction": "lower_is_better",
                "required_comparator_role": "practical_heuristic",
                "evidence_kind": "sweep_with_multiple_seeds",
            }
        )
        claims.append(
            {
                "claim_id": "C2_operating_regime",
                "claim_type": "operating_regime_sensitivity",
                "metric": "sum_power_W",
                "direction": "lower_is_better",
                "required_comparator_role": "practical_heuristic",
                "evidence_kind": "stress_or_sensitivity_sweep_on_physical_kpi",
                "diagnostic_metrics": _diagnostic_metrics(),
            }
        )
    else:
        candidate_metrics = [
            item
            for item in kpis
            if item not in {"feasible", "constraint_violation_max"}
        ]
        metric = _objective_aligned_metric(
            candidate_metrics=candidate_metrics,
            source_text="\n".join(
                [
                    str(topic or ""),
                    json.dumps(problem_contract, ensure_ascii=False),
                    str(algorithm_md or ""),
                    str(convergence_or_complexity_md or ""),
                    str(experiment_blueprint_md or ""),
                ]
            ),
            objective_sense=objective_sense,
            fallback_metric=next(iter(candidate_metrics), "physical_utility"),
        )
        secondary_metric = (
            _supporting_physical_metric(candidate_metrics, metric)
            if _metric_is_objective_like_name(metric)
            else metric
        )
        direction = "lower_is_better" if objective_sense == "minimize" else "higher_is_better"
        claims.append(
            {
                "claim_id": "C1_main_physical_kpi",
                "claim_type": "performance_advantage",
                "metric": metric,
                "direction": direction,
                "required_comparator_role": "practical_heuristic",
                "evidence_kind": "main_comparison_sweep",
            }
        )
        claims.append(
            {
                "claim_id": "C2_operating_regime",
                "claim_type": "operating_regime_sensitivity",
                "metric": secondary_metric,
                "direction": direction,
                "required_comparator_role": "practical_heuristic",
                "evidence_kind": "stress_or_sensitivity_sweep_on_physical_kpi",
                "diagnostic_metrics": _diagnostic_metrics(),
            }
        )

    return {
        "contract_version": CONTRACT_VERSION,
        "agent_role": "claim_owner",
        "topic": topic,
        "algorithm_family": algorithm_contract.get("algorithm_family", "unspecified"),
        "claims": claims,
        "forbidden_claims": [
            "Do not claim global optimality unless the proof/reference conditions are explicit.",
            "Do not claim proposed superiority beyond the declared comparison baseline; keep stronger-baseline audits as transparent diagnostics.",
            "Do not describe oracle/reference equivalence as engineering gain over a deployable baseline.",
        ],
        "source_summary": compact_text(
            "\n".join([algorithm_md, convergence_or_complexity_md]),
            1800,
        ),
        "phase_boundary": (
            "Claims are inferred from the problem and proposed algorithm only. "
            "Phase 2.4 owns experiment design, benchmark execution coverage, sweeps, figures, and tables."
        ),
    }


def select_wireless_benchmark_plan(
    *,
    topic: str,
    problem_contract: dict[str, Any],
    algorithm_contract: dict[str, Any],
    claim_map: dict[str, Any],
) -> dict[str, Any]:
    family = str(problem_contract.get("problem_family") or "")
    objective_sense = str(problem_contract.get("objective_sense") or "unknown")
    mechanisms = {str(item).strip().lower() for item in problem_contract.get("mechanisms") or []}
    topic_lower = str(topic or "").lower()
    has_energy_harvesting = "energy_harvesting" in mechanisms or "energy harvesting" in topic_lower
    has_sensing = "sensing" in mechanisms or "isac" in topic_lower or "sensing" in topic_lower
    has_power_splitting = (
        "power_splitting" in mechanisms
        or "power splitting" in topic_lower
        or "power-splitting" in topic_lower
        or "ps ratio" in topic_lower
    )
    has_nonlinear_eh = (
        "nonlinear_eh" in mechanisms
        or "nonlinear_energy_harvesting" in mechanisms
        or "nonlinear energy" in topic_lower
        or "nonlinear eh" in topic_lower
    )

    methods: list[dict[str, Any]]
    if family == "uav_trajectory_optimization":
        methods = [
            {
                "id": "proposed",
                "role": "proposed",
                "mandatory_status": "mandatory",
                "display_name": "Proposed",
                "implementation_hint": "Implement the Phase3 trajectory-aware joint user scheduling and transmit-power update under the propulsion-energy model.",
                "fairness_rule": "Same user locations, mission duration, speed limit, altitude, power cap, and propulsion model.",
            },
            {
                "id": "fixed_trajectory_baseline",
                "role": "main_baseline",
                "mandatory_status": "mandatory",
                "display_priority": 1,
                "display_name": "Fixed-Traj.",
                "implementation_hint": "Use a fixed circular or straight-line UAV trajectory while optimizing scheduling/power with the same evaluator.",
                "fairness_rule": "Only trajectory adaptation is disabled; all channel, power, mission, and propulsion assumptions are identical.",
            },
            {
                "id": "nearest_user_greedy",
                "role": "practical_heuristic",
                "mandatory_status": "mandatory",
                "display_priority": 2,
                "display_name": "Nearest-User",
                "implementation_hint": "Move greedily toward the currently least-served nearest user subject to the same speed cap.",
                "fairness_rule": "No oracle future information beyond current user positions and accumulated service deficits.",
            },
            {
                "id": "no_propulsion_awareness",
                "role": "model_diagnostic",
                "mandatory_status": "mandatory",
                "display_priority": 3,
                "display_name": "No-Prop.",
                "implementation_hint": "Optimize communication utility while ignoring propulsion energy during updates, then evaluate with true propulsion energy.",
                "fairness_rule": "Same final objective evaluator; differs only by omitting propulsion awareness in the design step.",
            },
        ]
    elif family == "uplink_power_control":
        methods = [
            {
                "id": "proposed",
                "role": "proposed",
                "mandatory_status": "mandatory",
                "display_name": "Proposed",
                "implementation_hint": "Use the Phase3 fixed-point/LP-compatible power-control update.",
                "fairness_rule": "Same SINR targets, channel gains, noise, and power caps as all baselines.",
            },
            {
                "id": "equal_power_heuristic",
                "role": "practical_heuristic",
                "mandatory_status": "mandatory",
                "display_name": "EQ-Power",
                "implementation_hint": "Allocate the same power to every user, then clip by the shared power cap.",
                "fairness_rule": "No oracle channel optimization beyond using the same feasibility evaluation.",
            },
            {
                "id": "channel_inversion_heuristic",
                "role": "practical_heuristic",
                "mandatory_status": "mandatory",
                "display_name": "Ch.-Inv.",
                "implementation_hint": "Allocate power inversely proportional to direct gain and normalize to budget.",
                "fairness_rule": "Uses only direct-gain information and the same total/cap constraints.",
            },
            {
                "id": "lp_reference",
                "role": "reference_oracle",
                "mandatory_status": "oracle_only",
                "display_name": "LP Ref.",
                "implementation_hint": "Optional internal reference for optimality sanity checks, not the main gain plot.",
                "fairness_rule": "May be shown only as a reference gap, not as a practical benchmark gain claim.",
            },
        ]
    elif family == "downlink_beamforming" and has_power_splitting:
        methods = [
            {
                "id": "proposed",
                "role": "proposed",
                "mandatory_status": "mandatory",
                "display_name": "Proposed",
                "implementation_hint": "Implement the Phase3 rectifier-aware joint beamforming and power-splitting algorithm.",
                "fairness_rule": "Same channels, power budget, SINR floor, EH requirement, and rectifier model.",
            },
            {
                "id": "fixed_ps_baseline",
                "role": "main_baseline",
                "mandatory_status": "mandatory",
                "display_priority": 1,
                "display_name": "Fixed-PS",
                "implementation_hint": "Fix the power-splitting ratios and optimize/use the same beamforming family under the same constraints.",
                "fairness_rule": "Same channels, budgets, constraints, and rectifier nonlinearity; only the PS adaptation is disabled.",
            },
            {
                "id": "linear_eh_baseline",
                "role": "model_diagnostic",
                "mandatory_status": "mandatory",
                "display_priority": 4,
                "display_name": "Linear-EH",
                "implementation_hint": "Use a linearized EH response as a practical approximation while evaluating final metrics with the true nonlinear rectifier.",
                "fairness_rule": "Same beam/power budget and constraints; differs only by the EH model used during the update. Use this mainly as a model-diagnostic curve, not as the default primary gain baseline.",
            },
            {
                "id": "mrt_split_baseline",
                "role": "direction_heuristic",
                "mandatory_status": "mandatory",
                "display_priority": 2,
                "display_name": "MRT-PS",
                "implementation_hint": "Use channel-matched/MRT beam directions with fair power loading and the same safeguarded power-splitting search.",
                "fairness_rule": "Same channels, constraints, power budget, and exact nonlinear evaluation; only the beam direction rule is simplified.",
            },
            {
                "id": "zf_split_baseline",
                "role": "direction_heuristic",
                "mandatory_status": "mandatory",
                "display_priority": 3,
                "display_name": "ZF-PS",
                "implementation_hint": "Use zero-forcing beam directions when dimensionally feasible, fair power loading, and the same safeguarded power-splitting search.",
                "fairness_rule": "Same constraints and exact nonlinear evaluation; mark unavailable rather than relaxing the problem when Nt < K.",
            },
        ]
    elif family == "downlink_beamforming" and (has_energy_harvesting or has_sensing):
        methods = [
            {
                "id": "proposed",
                "role": "proposed",
                "mandatory_status": "mandatory",
                "display_name": "Proposed",
                "implementation_hint": "Implement the Phase3 joint covariance/beamforming design over the frozen downlink controls.",
                "fairness_rule": "Same channels, service requirements, per-transmitter budgets, and physical evaluator.",
            },
            {
                "id": "no_shared_covariance_baseline",
                "role": "main_baseline",
                "mandatory_status": "mandatory",
                "display_priority": 1,
                "display_name": "No-shared-cov.",
                "implementation_hint": "Disable or strongly restrict the shared auxiliary covariance/resource block and satisfy energy/sensing services using communication covariances when feasible.",
                "fairness_rule": "Do not add new physical controls or relax constraints; only remove the shared covariance flexibility and use the same evaluator.",
            },
            {
                "id": "isotropic_shared_covariance_baseline",
                "role": "practical_heuristic",
                "mandatory_status": "mandatory",
                "display_priority": 2,
                "display_name": "Iso-cov.",
                "implementation_hint": "Use an isotropic or block-equal shared covariance/resource block and optimize only the scalar loading plus communication covariances when feasible.",
                "fairness_rule": "Same channels, budgets, service thresholds, and final physical evaluator; only the shared covariance direction is fixed.",
            },
            {
                "id": "mrt_covariance_baseline",
                "role": "direction_heuristic",
                "mandatory_status": "mandatory",
                "display_priority": 3,
                "display_name": "MRT",
                "implementation_hint": "Use channel-matched/MRT communication covariance directions with fair scalar loading and the same shared-resource treatment.",
                "fairness_rule": "Same constraints and evaluator; failures are reported, not repaired by relaxing service thresholds.",
            },
        ]
        if has_nonlinear_eh:
            methods.append(
                {
                    "id": "linear_eh_baseline",
                    "role": "model_diagnostic",
                    "mandatory_status": "diagnostic_optional",
                    "display_priority": 5,
                    "display_name": "Linear-EH",
                    "implementation_hint": "Use a linear EH approximation only during design, then evaluate all final KPIs with the frozen nonlinear EH evaluator.",
                    "fairness_rule": "Use as a diagnostic curve only when the frozen model contains a nonlinear EH mechanism.",
                }
            )
    elif family == "downlink_beamforming":
        methods = [
            {
                "id": "proposed",
                "role": "proposed",
                "mandatory_status": "mandatory",
                "display_name": "Proposed",
                "implementation_hint": "Implement the Phase3 beamforming algorithm with per-user beams/covariances.",
                "fairness_rule": "Same channels, QoS constraints, and power budget.",
            },
            {
                "id": "regularized_zf_heuristic",
                "role": "practical_heuristic",
                "mandatory_status": "mandatory",
                "display_name": "Benchmark",
                "implementation_hint": "Use fixed, strongly regularized ZF directions with the same robust power-loading certificate.",
                "fairness_rule": "Same channel estimates, uncertainty radii, SINR targets, and robust feasibility test.",
            },
            {
                "id": "mrt_or_channel_matched",
                "role": "practical_heuristic",
                "mandatory_status": "mandatory",
                "display_name": "MRT",
                "implementation_hint": "Matched-filter directions with nominal loading; useful as a stress reference, not the main power-gain baseline.",
                "fairness_rule": "No extra CSI or relaxed constraints.",
            },
        ]
    elif family == "ris_assisted_optimization":
        methods = [
            {
                "id": "proposed",
                "role": "proposed",
                "mandatory_status": "mandatory",
                "display_name": "Proposed",
                "implementation_hint": "Implement Phase3 joint/alternating active beam and RIS phase update.",
                "fairness_rule": "Same topology, channels, unit-modulus constraints, and power budget.",
            },
            {
                "id": "random_phase_baseline",
                "role": "practical_heuristic",
                "mandatory_status": "mandatory",
                "display_name": "Random RIS",
                "implementation_hint": "Random unit-modulus phases with fair active beam/power normalization.",
                "fairness_rule": "Same number of RIS elements and same power/QoS evaluation.",
            },
            {
                "id": "fixed_phase_baseline",
                "role": "practical_heuristic",
                "mandatory_status": "mandatory",
                "display_name": "Fixed RIS",
                "implementation_hint": "Fixed all-one or geometry-aligned phase baseline.",
                "fairness_rule": "No relaxed amplitude control unless the hardware model allows it.",
            },
        ]
    else:
        methods = [
            {
                "id": "proposed",
                "role": "proposed",
                "mandatory_status": "mandatory",
                "display_name": "Proposed",
                "implementation_hint": "Implement the Phase3 algorithm without redesigning it in Phase24.",
                "fairness_rule": "Same physical objective, constraints, channels, and seeds.",
            },
            {
                "id": "greedy_heuristic",
                "role": "practical_heuristic",
                "mandatory_status": "mandatory",
                "display_name": "Greedy",
                "implementation_hint": "A simple deployable greedy or myopic engineering baseline.",
                "fairness_rule": "No oracle access; same constraints and evaluation metric.",
            },
            {
                "id": "random_or_equal_allocation",
                "role": "practical_heuristic",
                "mandatory_status": "mandatory",
                "display_name": "Simple",
                "implementation_hint": "Random/equal allocation baseline appropriate to the current variables.",
                "fairness_rule": "Same budget and feasibility evaluation.",
            },
        ]

    return {
        "contract_version": CONTRACT_VERSION,
        "agent_role": "wireless_benchmark_agent",
        "topic": topic,
        "problem_family": family,
        "objective_sense": objective_sense,
        "algorithm_family": algorithm_contract.get("algorithm_family", "unspecified"),
        "compared_methods": methods,
        "main_plot_policy": {
            "main_figures_use_methods": [
                method["id"]
                for method in methods
                if method.get("role") == "proposed"
                or (method.get("mandatory_status") == "mandatory" and method.get("role") != "reference_oracle")
            ][:5],
            "reference_oracle_policy": "Use reference/oracle methods only for internal sanity or explicit optimality-gap plots.",
        },
        "claims_supported": [claim.get("claim_id") for claim in claim_map.get("claims", [])],
    }


def build_experiment_design_contract(
    *,
    problem_contract: dict[str, Any],
    benchmark_plan: dict[str, Any],
    claim_map: dict[str, Any],
    experiment_blueprint_md: str = "",
) -> dict[str, Any]:
    methods = benchmark_plan.get("main_plot_policy", {}).get("main_figures_use_methods") or ["proposed"]
    claims = claim_map.get("claims") or []
    main_claim = claims[0] if claims else {}
    second_claim = claims[1] if len(claims) > 1 else main_claim
    candidate_metrics = [
        item
        for item in problem_contract.get("primary_physical_kpis", [])
        if str(item or "").strip() and str(item or "").strip() not in {"feasible", "constraint_violation_max"}
    ]
    source_text = "\n".join(
        [
            json.dumps(problem_contract, ensure_ascii=False),
            json.dumps(claim_map, ensure_ascii=False),
            str(experiment_blueprint_md or ""),
        ]
    )
    objective_sense = str(problem_contract.get("objective_sense") or "unknown")
    claim_metric = str(main_claim.get("metric") or "").strip()
    main_metric = _objective_aligned_metric(
        candidate_metrics=_dedupe_preserve_order([claim_metric, *candidate_metrics]),
        source_text=source_text,
        objective_sense=objective_sense,
        fallback_metric=claim_metric or "physical_utility",
    )
    second_metric = str(second_claim.get("metric") or main_metric)
    diagnostic_metric_names = {"feasible", "feasibility", "constraint_violation", "constraint_violation_max", "max_constraint_violation"}
    if second_metric.strip().lower() in diagnostic_metric_names:
        second_metric = main_metric
    if _metric_is_objective_like_name(main_metric) and _metric_is_objective_like_name(second_metric):
        second_metric = _supporting_physical_metric(candidate_metrics, main_metric)

    figures = [
        {
            "figure_id": "figure_1",
            "claim_id": main_claim.get("claim_id", "C1_main"),
            "chart_intent": "main_comparison",
            "chart_type": "line",
            "methods_to_run": methods,
            "y_metric": main_metric,
            "evidence_rule": "Run the proposed method against the mandatory practical baseline family over a meaningful physical sweep, then display the clearest gain.",
        },
        {
            "figure_id": "figure_2",
            "claim_id": second_claim.get("claim_id", "C2_reliability"),
            "chart_intent": "stress_or_gain",
            "chart_type": "line_or_bar_selected_by_data",
            "methods_to_run": methods,
            "y_metric": second_metric,
            "evidence_rule": "Sweep a meaningful operating-regime or stress parameter, but keep the y-axis on the paper objective or a system-performance KPI. Feasibility and violation are diagnostics, not the plotted claim metric.",
        },
    ]
    return {
        "contract_version": CONTRACT_VERSION,
        "agent_role": "experiment_design_agent",
        "primary_physical_kpis": problem_contract.get("primary_physical_kpis", []),
        "phase_boundary": {
            "phase2_4_design": "within Phase 2.4, freeze KPI, benchmark set, sweep family, expected trend, and tiered run policy before code generation",
            "phase2_4_implementation": "within Phase 2.4, generate code that implements the frozen experiment contract",
            "phase2_5": "post-run result verification only: assess sufficiency, request denser reruns, or route back to Phase 2.4 experiment-design repair; do not redesign KPI/benchmark/story",
        },
        "figure_contracts": figures,
        "table_contracts": [],
        "hard_rules": [
            "Freeze the paper-facing primary_metric before Phase 2.4 code generation; Phase 2.5 must not choose a new primary KPI after seeing results.",
            "Freeze the final plotted method set before Phase 2.4 code generation; all final figures should use the same proposed-plus-selected-practical-benchmark set.",
            "Every figure contract must include expected_trend, active_regime_note, chart_choice_rationale, and axis_labels using paper notation rather than internal schema paths.",
            "If a scout result shows the selected KPI/benchmark/sweep family is wrong, route back to Phase 2.4 experiment-design repair instead of patching the paper story in Phase 2.5.",
            "Do not use weighted_objective as the only y-axis unless it is the actual physical objective.",
            "Do not include oracle/reference methods in the main gain plot by default.",
            "Do not create a diagnostic-only second figure when the paper needs evidence of advantage or feasibility.",
            "At least one final paper figure must use a decomposed physical KPI tied to the claim, such as rate, sensing SNR/CRB, harvested power, reliability, or energy efficiency.",
            "At least two final paper figures must use non-diagnostic system-performance y_metrics. For scalarized utility objectives, do not make all final figures use only utility/objective; use decomposed physical KPIs for the main evidence when available.",
            "Feasibility, violation, residual, runtime, solver status, and convergence should be emitted as diagnostics and used to filter/qualify claims, never as final paper figure y-axes.",
            "Compute gains only on comparable feasible samples when feasibility differs.",
            "Do not choose transmit power as the experiment story merely because it is easy to report. Use transmit/total power as primary evidence only when the frozen objective is resource minimization; otherwise report the defined utility, rate, sensing, energy, reliability, or tradeoff KPI.",
        ],
        "phase24_experiment_requirements": [
            "Declare research_evidence_contract.primary_metric with name, display_name, and higher_is_better.",
            "Declare the paper claims that the numerical package must be able to support or falsify.",
            "Declare all mandatory compared methods, including practical baselines, ablations, and oracle/reference diagnostics when justified.",
            "Declare canonical configuration fields and sweep axes that the executable solver must consume.",
            "Declare physical KPI metrics, feasibility/violation diagnostics, and actual-used sweep diagnostics that must be emitted.",
            "Declare 2-3 figure evidence candidates before any experiment code is generated.",
            "Declare scout, medium, and paper-level run policies: scout uses few x-axis points but enough seeds; paper mode uses smooth x grids and 80-100 seeds per point.",
            "Declare missing-experiment behavior when quick validation cannot support a paper claim, including whether to densify values or route back to Phase 2.4 experiment-design repair.",
        ],
    }


def _method_contract_for_plan(method: dict[str, Any]) -> dict[str, Any]:
    method_id = str(method.get("id") or "").strip()
    role = str(method.get("role") or "").strip()
    normalized_role = "heuristic" if role == "practical_heuristic" else role
    if role == "reference_oracle":
        normalized_role = "oracle"
    display = str(method.get("display_name") or method_id or "Method").strip()
    return {
        "id": method_id,
        "role": normalized_role or ("proposed" if method_id == "proposed" else "heuristic"),
        "mandatory_status": str(method.get("mandatory_status") or ("mandatory" if normalized_role != "oracle" else "oracle_only")),
        "display_name_short": display,
        "display_name_long": display,
        "scientific_purpose": (
            "Main proposed method from Phase 2.3."
            if method_id == "proposed"
            else "Practical benchmark for measuring engineering gain under the same physical constraints."
        ),
        "implementation_hint": str(method.get("implementation_hint") or "").strip(),
        "fairness_rule": str(method.get("fairness_rule") or "").strip(),
        "display_priority": method.get("display_priority", None),
    }


def _contract_method_ids(methods: list[dict[str, Any]]) -> list[str]:
    ids: list[str] = []
    for method in methods:
        if not isinstance(method, dict):
            continue
        method_id = str(method.get("id") or method.get("internal_name") or method.get("name") or "").strip()
        if method_id and method_id not in ids:
            ids.append(method_id)
    return ids


def _lookup_methods_by_id(benchmark_plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    methods: dict[str, dict[str, Any]] = {}
    for item in benchmark_plan.get("compared_methods", []):
        if not isinstance(item, dict):
            continue
        method_id = str(item.get("id") or "").strip()
        if method_id:
            methods[method_id] = item
    return methods


def build_phase24_execution_contract(
    *,
    validation_plan: dict[str, Any],
    problem_contract: dict[str, Any],
    algorithm_contract: dict[str, Any],
    benchmark_plan: dict[str, Any],
) -> dict[str, Any]:
    """Freeze the small executable interface used by Phase 2.4 code generation."""
    plan = validation_plan if isinstance(validation_plan, dict) else {}
    evidence = plan.get("research_evidence_contract")
    if not isinstance(evidence, dict):
        evidence = plan.get("paper_evidence_contract", {})
    if not isinstance(evidence, dict):
        evidence = {}

    compared_methods = [
        item
        for item in evidence.get("compared_methods", [])
        if isinstance(item, dict) and str(item.get("id") or item.get("internal_name") or item.get("name") or "").strip()
    ]
    active_method_ids: list[str] = []
    for figure in evidence.get("figures", []):
        if not isinstance(figure, dict):
            continue
        methods = figure.get("methods_to_run", [])
        if not isinstance(methods, list):
            continue
        for method in methods:
            if isinstance(method, dict):
                method_id = str(method.get("id") or method.get("internal_name") or method.get("name") or "").strip()
            else:
                method_id = str(method or "").strip()
            if method_id and method_id not in active_method_ids:
                active_method_ids.append(method_id)
    if not active_method_ids:
        active_method_ids = _contract_method_ids(compared_methods)

    sweeps: list[dict[str, Any]] = []
    for sweep in plan.get("sweep_definitions", []):
        if not isinstance(sweep, dict):
            continue
        path = str(sweep.get("canonical_path") or sweep.get("variable") or "").strip()
        values = sweep.get("quick_mode_values", sweep.get("scout_values", sweep.get("values", [])))
        if not isinstance(values, list):
            values = []
        safe_path = re.sub(r"[^A-Za-z0-9]+", "_", path).strip("_")
        sweeps.append(
            {
                "id": str(sweep.get("id") or sweep.get("name") or path or "sweep").strip(),
                "path": path,
                "quick_values": values,
                "actual_used_metric": f"actual_used_{safe_path}" if safe_path else "",
                "description": str(sweep.get("description") or "").strip(),
            }
        )

    required_columns = [
        str(item).strip()
        for item in evidence.get("required_result_columns", [])
        if str(item).strip()
    ]
    required_outputs = plan.get("required_outputs", {})
    scalar_metrics = []
    if isinstance(required_outputs, dict) and isinstance(required_outputs.get("scalar_metrics"), list):
        scalar_metrics = [str(item).strip() for item in required_outputs["scalar_metrics"] if str(item).strip()]
    metadata_columns = {"method", "seed", "swept_param", "swept_value", "scenario_name"}
    required_metrics = _unique(
        [
            "objective",
            "feasible",
            "constraint_violation_max",
            *scalar_metrics,
            *[column for column in required_columns if column not in metadata_columns],
        ]
    )

    return {
        "contract_version": CONTRACT_VERSION,
        "agent_role": "phase24_execution_contract",
        "architecture": "fixed_harness_single_generated_experiment_core",
        "generated_file": "generated_experiment_core.py",
        "problem_family": plan.get("problem_family") or problem_contract.get("problem_family") or "generic_wireless_optimization",
        "objective_sense": plan.get("objective_sense") or problem_contract.get("objective_sense") or "unknown",
        "algorithm_family": plan.get("algorithm_family") or algorithm_contract.get("algorithm_family") or "unspecified",
        "algorithm_execution_contract": algorithm_contract.get("algorithm_execution_contract", {}),
        "canonical_config": plan.get("canonical_config", {}),
        "methods": compared_methods,
        "active_method_ids": active_method_ids,
        "sweeps": sweeps,
        "required_metrics": required_metrics,
        "required_result_columns": required_columns,
        "acceptance_tests": [
            "smoke_import",
            "required_functions_present",
            "finite_metrics",
            "exact_metric_keys",
            "method_ids_implemented",
            "sweeps_consumed",
            "json_serializable_state_and_metrics",
            "feasibility_consistency",
        ],
        "allowed_approximations": [
            "minor numerical simplifications are allowed only when they preserve the declared controls, method ids, and true objective/constraint evaluator",
            "generic proxy, fallback, or candidate-search-only replacements for the proposed algorithm are not allowed",
            "any simplification must be reported in model['metadata']['approximations']",
            "all methods must still be evaluated with the same true objective and constraints",
        ],
        "benchmark_agent_role": benchmark_plan.get("agent_role", "wireless_benchmark_agent"),
    }


def contract_prompt_block(*, benchmark_plan: dict[str, Any], experiment_design_contract: dict[str, Any]) -> str:
    return (
        "\n\n[Structured Phase24 evidence contracts]\n"
        "WirelessBenchmarkAgent output:\n"
        f"{json.dumps(benchmark_plan, ensure_ascii=False, indent=2)}\n\n"
        "ExperimentDesignAgent output:\n"
        f"{json.dumps(experiment_design_contract, ensure_ascii=False, indent=2)}\n"
        "Phase24 validation/code generation must obey these contracts unless a later auditor blocks them."
    )


def audit_implementation_contract(
    *,
    run_dir: Path,
    generated_plugin: str,
    validation_status: dict[str, Any],
) -> dict[str, Any]:
    benchmark_plan = read_json(Path(run_dir) / "phase2-4" / "wireless_benchmark_plan.json") or {}
    method_ids = [
        str(item.get("id") or "")
        for item in benchmark_plan.get("compared_methods", [])
        if isinstance(item, dict) and item.get("id")
    ]
    text = str(generated_plugin or "").lower()
    warnings: list[str] = []
    errors: list[str] = []
    if "proposed_step" not in text and "method_solution" not in text:
        warnings.append("Generated plugin does not visibly expose proposed_step or method_solution; rely on harness validation before Phase25.")
    for method_id in method_ids:
        if method_id not in {"proposed", "lp_reference"} and method_id.lower() not in text:
            warnings.append(f"Benchmark method `{method_id}` is selected but not visibly named in generated plugin.")
    if validation_status.get("status") not in {"ok", "passed", "selected_after_bounded_repairs"}:
        errors.append("Phase24 harness validation did not pass, so implementation cannot be trusted.")
    return {
        "contract_version": CONTRACT_VERSION,
        "agent_role": "implementation_auditor",
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "validated_method_ids": method_ids,
    }


def audit_phase25_evidence(
    *,
    run_dir: Path,
    phase25_result: dict[str, Any] | None,
    phase25_status: str,
) -> dict[str, Any]:
    summary = read_json(Path(run_dir) / "phase2-5" / "phase25_experiment_summary.json") or {}
    benchmark_plan = read_json(Path(run_dir) / "phase2-4" / "wireless_benchmark_plan.json") or {}
    warnings: list[str] = []
    errors: list[str] = []
    figures = summary.get("figures", []) if isinstance(summary, dict) else []
    if not isinstance(figures, list):
        figures = []
    paper_ready_figures = [item for item in figures if isinstance(item, dict) and item.get("paper_ready")]
    non_diagnostic_paper_figures = [
        item
        for item in paper_ready_figures
        if "violation" not in str(item.get("y_metric", "")).lower()
        and "feasible" not in str(item.get("y_metric", "")).lower()
        and "feasibility" not in str(item.get("y_metric", "")).lower()
    ]
    if phase25_status in {"paper_minimum_ready", "paper_preferred_ready", "high_confidence_ready"}:
        methods = benchmark_plan.get("main_plot_policy", {}).get("main_figures_use_methods") or []
        if "proposed" not in methods or len(methods) < 2:
            errors.append("Paper-ready Phase25 evidence lacks a proposed-vs-practical-benchmark main plot contract.")
        if isinstance(summary, dict) and summary.get("generated_figures_are_draft_only"):
            errors.append("Phase25 claims paper readiness while figures are marked draft-only.")
        if len(paper_ready_figures) < 2:
            errors.append(f"Paper-ready Phase25 evidence must include at least two final figures; found {len(paper_ready_figures)}.")
        if len(non_diagnostic_paper_figures) < 2:
            errors.append(
                "Paper-ready Phase25 evidence must include at least two non-diagnostic system-performance figures; "
                f"found {len(non_diagnostic_paper_figures)}."
            )
    else:
        warnings.append(
            f"Phase25 status `{phase25_status}` is not paper-ready; later synthesis must keep claims conservative "
            "and mark the run as needing review."
        )
    return {
        "contract_version": CONTRACT_VERSION,
        "agent_role": "evidence_auditor",
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "phase25_status": phase25_status,
        "paper_ready_figure_count": len(paper_ready_figures),
        "non_diagnostic_paper_ready_figure_count": len(non_diagnostic_paper_figures),
        "phase25_result_keys": sorted(phase25_result.keys()) if isinstance(phase25_result, dict) else [],
    }


def write_json_artifact(path: Path, payload: dict[str, Any]) -> None:
    write_text(path, json.dumps(payload, ensure_ascii=False, indent=2))
