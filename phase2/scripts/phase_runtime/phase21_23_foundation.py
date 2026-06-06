from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from pipeline_core import compact_text, read_json, read_text, write_text
from pipeline_core.subagents import build_tractability_route_policy, tractability_route_policy_prompt_block
from pipeline_core.json_utils import _safe_json_loads
from phase_runtime.llm import create_llm_client
from phase_runtime.prompt_templates import render_prompt_template
from phase_runtime.topic_guardrails import build_wireless_feasibility_guardrail as build_dynamic_wireless_feasibility_guardrail
from phase_runtime.phase24_plan import (
    extract_candidate_block,
    extract_first_candidate_title,
    extract_section,
    shortlist_preview,
)


DEFAULT_PHASE2_CONTRACT_REPAIR_ROUND_LIMIT = 3


def phase2_contract_repair_round_limit() -> int:
    """Maximum bounded contract-repair rounds for Phase 2.1--2.3."""

    for env_name in ("WARA_PHASE2_CONTRACT_REPAIR_ROUNDS", "WCL_PHASE2_CONTRACT_REPAIR_ROUNDS"):
        raw_value = os.environ.get(env_name)
        if raw_value is None:
            continue
        try:
            return max(0, int(raw_value))
        except ValueError:
            continue
    return DEFAULT_PHASE2_CONTRACT_REPAIR_ROUND_LIMIT


def _phase2_max_tokens(env_name: str, default: int) -> int:
    raw = str(os.environ.get(env_name, "")).strip()
    if not raw:
        return default
    try:
        return max(1000, int(raw))
    except ValueError:
        return default


def _phase2_words(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z][a-zA-Z0-9_\-]*", str(text or "").lower()))


def _phase2_has_any(text: str, needles: list[str]) -> bool:
    lowered = str(text or "").lower()
    return any(needle.lower() in lowered for needle in needles)


def _phase2_has_concrete_uncertainty_model(text: str) -> bool:
    """Detect whether robust/chance language names a reproducible uncertainty model."""
    return _phase2_has_any(
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


def _phase2_uses_uncertainty_or_chance(text: str) -> bool:
    return _phase2_has_any(
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


def _phase2_shared_waveform_needs_receiver_scope(text: str) -> bool:
    return (
        _phase2_has_any(
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
        and _phase2_has_any(text, ["sinr", "interference", "denominator", "treated as noise"])
        and not _phase2_has_any(
            text,
            [
                "not decoded",
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


def _phase2_covariance_needs_physical_scope(text: str) -> bool:
    return (
        _phase2_has_any(
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
        and _phase2_has_any(text, ["beamforming", "transmit", "downlink", "precoding"])
        and not _phase2_has_any(
            text,
            [
                "gaussian signaling",
                "Gaussian signaling",
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


def _phase2_contract_payload(contract: dict[str, Any]) -> str:
    try:
        return json.dumps(contract, ensure_ascii=False)
    except TypeError:
        return str(contract)


def _phase2_entry_payload(entry: Any) -> str:
    if isinstance(entry, dict):
        return " ".join(str(value or "") for value in entry.values())
    return str(entry or "")


def _phase2_mentions_dimension(payload: str, dimension: int) -> bool:
    text = str(payload or "").lower()
    compact = re.sub(r"\s+", "", text)
    dim = str(dimension)
    return any(
        marker in compact
        for marker in (
            f"r^{dim}",
            f"r^{{{dim}}}",
            f"\\mathbb{{r}}^{dim}",
            f"\\mathbb{{r}}^{{{dim}}}",
            f"\\mathbbr^{dim}",
            f"\\mathbbr^{{{dim}}}",
        )
    ) or any(
        marker in text
        for marker in (
            f"{dim}d",
            f"{dim}-d",
            f"{dim} dimensional",
            f"{dim}-dimensional",
            "two-dimensional" if dimension == 2 else "three-dimensional",
        )
    )


def _phase2_mentions_position_like(payload: str) -> bool:
    return _phase2_has_any(
        payload,
        [
            "position",
            "coordinate",
            "location",
            "waypoint",
            "trajectory",
            "deployment",
            "mobility",
            "movable",
            "uav",
            "aerial",
            "antenna element",
        ],
    )


def _phase2_mentions_lower_dim_ground_like(payload: str) -> bool:
    return _phase2_has_any(
        payload,
        [
            "ground",
            "user",
            "device",
            "sensor",
            "terminal",
            "iot",
            "receiver",
            "node location",
            "ground-node",
            "ground node",
        ],
    )


def _phase2_has_geometry_resolution(text: str) -> bool:
    return _phase2_has_any(
        text,
        [
            "horizontal coordinate",
            "horizontal position",
            "horizontal trajectory",
            "horizontal component",
            "ground projection",
            "projected",
            "projection",
            "embedded",
            "embed",
            "same-dimensional",
            "same dimensional",
            "full-dimensional",
            "full dimensional",
            "\\bar",
            "w_k^t,0",
            "w_k^\\top,0",
            "altitude is treated separately",
        ],
    )


def _phase2_has_norm_plus_altitude_term(text: str) -> bool:
    compact = re.sub(r"\s+", "", str(text or ""))
    ascii_text = str(text or "")
    norm_diff = bool(
        re.search(
            r"(?:\|\||\\\||\\lVert|\\Vert|norm)[^.\n;]{0,180}(?:q|x|uav|position|trajectory)[^.\n;]{0,140}(?:w|ground|user|device|sensor|terminal)",
            ascii_text,
            flags=re.I,
        )
    ) or bool(
        re.search(
            r"(?:q|x|uav|position|trajectory)[^.\n;]{0,140}(?:w|ground|user|device|sensor|terminal)[^.\n;]{0,180}(?:\|\||\\\||\\rVert|\\Vert|norm)",
            ascii_text,
            flags=re.I,
        )
    )
    altitude_square = bool(
        re.search(
            r"(?:\+|plus)[^.\n;]{0,80}(?:H|h|altitude|height)(?:\[[^\]]+\]|_\{?[^}\s]+\}?|)?(?:\^2|\^\{2\}|squared)",
            ascii_text,
            flags=re.I,
        )
    ) or bool(
        re.search(
            r"(?:\+|plus)[^.\n;]{0,40}(?:\\?[A-Za-z]*H(?:\[[^\]]+\])?)(?:\^2|\^\{2\})",
            compact,
            flags=re.I,
        )
    )
    return norm_diff and altitude_square


def _phase2_geometry_convention_issues(contract: dict[str, Any], *extra_texts: str) -> tuple[list[str], list[str]]:
    """Catch deterministic coordinate-convention mistakes before they reach writing.

    The check is intentionally narrow. It does not infer a wireless model from
    keywords; it only flags a self-contradictory convention that combines a
    full-dimensional mobile/deployment coordinate, lower-dimensional ground
    coordinates, and an additional altitude/height term in the same distance
    expression.
    """

    normalized = normalize_phase2_phase1_mathematical_contract(contract)
    entries = []
    for key in ("controls", "parameters", "random_quantities", "derived_quantities"):
        entries.extend(entry for entry in normalized.get(key, []) if isinstance(entry, dict))
    full_text = "\n".join([_phase2_contract_payload(normalized), *(str(text or "") for text in extra_texts)])
    has_full_dim_mobile = any(
        _phase2_mentions_dimension(_phase2_entry_payload(entry), 3)
        and _phase2_mentions_position_like(_phase2_entry_payload(entry))
        for entry in entries
    )
    has_lower_dim_ground = any(
        _phase2_mentions_dimension(_phase2_entry_payload(entry), 2)
        and _phase2_mentions_lower_dim_ground_like(_phase2_entry_payload(entry))
        for entry in entries
    )
    has_bad_altitude_distance = _phase2_has_norm_plus_altitude_term(full_text)
    has_resolution = _phase2_has_geometry_resolution(full_text)
    errors: list[str] = []
    warnings: list[str] = []
    if has_full_dim_mobile and has_lower_dim_ground and has_bad_altitude_distance and not has_resolution:
        errors.append(
            "Geometry convention mixes a full-dimensional mobile/deployment coordinate with lower-dimensional ground coordinates and an added altitude/height term. "
            "Use horizontal coordinates plus altitude, or full-dimensional coordinates with embedded ground nodes, but not both."
        )
    elif has_full_dim_mobile and has_lower_dim_ground and not has_resolution:
        warnings.append(
            "Geometry convention combines full-dimensional mobile/deployment coordinates with lower-dimensional ground coordinates without an explicit embedding/projection convention."
        )
    return errors, warnings


def _phase2_contract_keywords(value: Any) -> list[str]:
    raw = str(value or "").lower()
    words = re.findall(r"[a-z][a-z0-9_\-]{2,}", raw)
    stop = {
        "the",
        "and",
        "with",
        "under",
        "over",
        "into",
        "from",
        "for",
        "used",
        "uses",
        "user",
        "users",
        "slot",
        "slots",
        "phase",
        "control",
        "controls",
        "parameter",
        "quantity",
        "quantities",
        "optimizer",
        "appears",
        "allowed",
        "mathbb",
        "mathbf",
        "mathrm",
        "mathcal",
    }
    keywords = [w.replace("_", " ").replace("-", " ") for w in words if w not in stop and len(w) >= 4]
    synonym_groups = [
        (["trajectory", "waypoint", "uav", "horizontal"], ["trajectory", "waypoint", "path", "position"]),
        (["schedule", "scheduling", "allocation", "time allocation"], ["schedule", "scheduling", "allocation", "time-sharing"]),
        (["power", "transmit"], ["power", "transmit power", "p_n"]),
        (["beamforming", "precoding", "beamformer"], ["beamforming", "precoding", "beamformer", "covariance"]),
        (["splitting", "rho"], ["power splitting", "splitting ratio", "rho"]),
        (["energy", "efficiency"], ["energy efficiency", "bits per joule", "bit per j", "eta"]),
        (["harvest", "harvested"], ["harvested energy", "energy harvesting", "eh"]),
        (["rate", "throughput"], ["sum rate", "throughput", "rate"]),
        (["propulsion"], ["propulsion", "flight energy"]),
        (["constraint", "feasible"], ["constraint", "feasible", "feasibility", "projection", "violation"]),
    ]
    for triggers, synonyms in synonym_groups:
        if any(trigger in raw for trigger in triggers):
            keywords.extend(synonyms)
    seen: set[str] = set()
    ordered: list[str] = []
    for keyword in keywords:
        keyword = keyword.strip().lower()
        if keyword and keyword not in seen:
            seen.add(keyword)
            ordered.append(keyword)
    return ordered


def _phase2_text_covers_contract_item(text: str, item: Any) -> bool:
    lowered = str(text or "").lower()
    if isinstance(item, dict):
        payload = " ".join(str(item.get(key) or "") for key in ("symbol", "meaning", "definition", "relation", "id"))
        symbol = str(item.get("symbol") or "").strip().lower()
        if symbol:
            compact_text = re.sub(r"[^a-z0-9]+", "", lowered)
            compact_symbol = re.sub(r"[^a-z0-9]+", "", symbol)
            symbol_variants = {
                symbol,
                symbol.replace("_", ""),
                symbol.replace("_", " "),
                compact_symbol,
            }
            if any(variant and variant in lowered for variant in symbol_variants if len(variant) >= 2):
                return True
            if compact_symbol and len(compact_symbol) >= 2 and compact_symbol in compact_text:
                return True
    else:
        payload = str(item or "")
    if any(keyword in lowered for keyword in _phase2_contract_keywords(payload)):
        return True

    # Internal execution-contract ids use snake_case, while LLM-authored method
    # text normally uses prose such as "inverse-EH step" or "solve the SDP".
    # Match compact ids by meaningful token coverage instead of requiring the
    # exact identifier phrase to appear verbatim in the paper-facing algorithm.
    if not re.search(r"[_\-]", payload):
        return False
    normalized_text = re.sub(r"[^a-z0-9]+", " ", lowered)
    normalized_payload = re.sub(r"[^a-z0-9]+", " ", payload.lower())
    tokens = [
        token
        for token in normalized_payload.split()
        if token
        and len(token) >= 3
        and token
        not in {
            "and",
            "the",
            "for",
            "with",
            "under",
            "block",
            "update",
            "step",
        }
    ]
    if len(tokens) < 2:
        return False
    aliases = {
        "eh": ["eh", "energy", "harvested", "rectenna", "rf dc", "rf to dc"],
        "sdp": ["sdp", "semidefinite", "conic"],
        "solve": ["solve", "solves", "solved", "solving", "solver"],
        "threshold": ["threshold", "inverse", "attainable", "attainability"],
        "feasibility": ["feasibility", "feasible", "infeasible", "residual", "acceptance", "accepted", "reject"],
        "evaluation": ["evaluation", "evaluate", "evaluated", "evaluates", "metric", "metrics", "kpi", "kpis"],
        "kpi": ["kpi", "kpis", "metric", "metrics", "objective", "sinr", "harvested", "sensing"],
    }
    hits = 0
    for token in tokens:
        variants = aliases.get(token, [token])
        if any(variant in normalized_text for variant in variants):
            hits += 1
    required = max(2, (len(tokens) + 1) // 2)
    return hits >= required


def _phase2_count_contract_coverage(text: str, items: list[Any]) -> tuple[int, list[str]]:
    covered = 0
    missing: list[str] = []
    for item in items:
        if _phase2_text_covers_contract_item(text, item):
            covered += 1
        else:
            if isinstance(item, dict):
                missing.append(str(item.get("symbol") or item.get("id") or item.get("meaning") or "")[:80])
            else:
                missing.append(str(item)[:80])
    return covered, missing


def _phase2_load_phase_contracts(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    phase1_dir = Path(run_dir) / "phase2-1"
    math_contract = read_json(phase1_dir / "mathematical_contract.frozen.json") or read_json(
        phase1_dir / "mathematical_contract.json"
    ) or {}
    algorithm_contract = read_json(Path(run_dir) / "phase2-2" / "algorithm_contract.json") or {}
    return math_contract, algorithm_contract


def _phase2_term_present(lowered: str, term: str) -> bool:
    term = term.lower()
    if re.fullmatch(r"[a-z0-9_]+", term):
        return re.search(rf"(?<![a-z0-9_]){re.escape(term)}(?![a-z0-9_])", lowered) is not None
    return term in lowered


def _phase2_count_hard_mechanisms(text: str) -> list[str]:
    lowered = str(text or "").lower()
    mechanisms: list[tuple[str, list[str]]] = [
        ("ris", ["ris", "reconfigurable intelligent surface"]),
        ("multiuser", ["multiuser", "multi-user", "multiple users", "k users"]),
        ("swipt_eh", ["swipt", "energy harvesting", "harvested energy", "eh receiver"]),
        ("nonlinear_eh", ["nonlinear energy", "non-linear energy", "logistic eh", "sigmoid"]),
        ("isac_sensing", ["isac", "radar", "sensing", "beampattern"]),
        ("crb", ["crb", "cramer", "cramer-rao", "fisher information", "fim"]),
        ("bistatic", ["bistatic"]),
        ("robust", ["robust", "uncertain", "imperfect csi", "bounded csi"]),
        ("movable_antenna", ["movable antenna", "movable-antenna", "fluid antenna", "fluid-antenna"]),
        ("secrecy", ["secrecy", "eavesdropper"]),
        ("noma", ["noma"]),
    ]
    found: list[str] = []
    for name, terms in mechanisms:
        if any(_phase2_term_present(lowered, term) for term in terms):
            found.append(name)
    return found


def _phase2_original_optimizer_mentions_auxiliary_quantities(text: str) -> bool:
    compact = re.sub(r"\s+", " ", str(text or ""))
    optimizer_windows: list[str] = []
    for pattern in (r"\\(?:max|min)\s*_\s*(.{0,320}?)(?:\\quad|&|\\\\)", r"\b(?:max|min)\s*_\s*(.{0,320}?)(?:\\quad|&|\\\\)"):
        optimizer_windows.extend(re.findall(pattern, compact, flags=re.I))
    for match in re.finditer(r"\\underset\s*(.{0,420}?)\\operatorname\{(?:maximize|minimize)\}", compact, flags=re.I):
        optimizer_windows.append(match.group(1))
    aux_patterns = [
        r"(?<![A-Za-z])p_\{?k",
        r"(?<![A-Za-z])q_\{?k",
        r"(?<![A-Za-z])z_\{?k",
        r"\\Gamma_\{?k",
        r"(?<![A-Za-z])R_\{?k",
        r"slack",
        r"surrogate",
    ]
    return any(re.search(pattern, window, flags=re.I) for window in optimizer_windows for pattern in aux_patterns)


def _phase2_has_swipt_eh(text: str) -> bool:
    return _phase2_has_any(text, ["swipt", "energy harvesting", "harvested energy", "eh user", "eh receiver", "rectifier"])


def _phase2_has_independent_streams(text: str) -> bool:
    return _phase2_has_any(
        text,
        [
            "independent symbols",
            "independent data",
            "s_k",
            "s_i",
            "data symbol",
            "energy symbol",
            "superposition",
            "\\sum",
        ],
    )


def _phase2_detect_coherent_eh_power_formula(text: str) -> bool:
    compact = re.sub(r"\s+", " ", text)
    patterns = [
        r"P[^.]{0,80}(?:rf|RF)[^.]{0,160}\|[^|]{0,120}(?:\\sum|sum)[^|]{0,120}(?:\+|\\plus)[^|]{0,80}\|(?:\^2|\^\{2\})",
        r"\|[^|]{0,120}\((?:\\sum|sum)[^)]{0,120}(?:w|\\mathbf\{w\})[^)]{0,120}(?:\+|\\plus)[^)]{0,80}\)[^|]{0,80}\|(?:\^2|\^\{2\})",
        r"\|[^|]{0,120}\((?:[^)]*w_\{?1\}?[^)]*\+[^)]*w_\{?2\}?|[^)]*w_k[^)]*\+[^)]*w_\\mathrm\{E\})[^)]*\)[^|]{0,80}\|(?:\^2|\^\{2\})",
    ]
    return any(re.search(pattern, compact, flags=re.IGNORECASE) for pattern in patterns)


def _phase2_detect_independent_eh_power_sum(text: str) -> bool:
    compact = re.sub(r"\s+", " ", text)
    lowered = compact.lower()
    if any(
        marker in lowered
        for marker in (
            "per-stream received powers",
            "sum of per-stream powers",
            "sums per-stream",
            "sum of the individual stream powers",
        )
    ):
        return True
    return bool(
        re.search(
            r"(?:E\[|\\mathbb\{E\}|expectation|sum_\{?i|\\sum_\{?i|\\sum_\{?j|\\sum_\{?k)[^.]{0,260}\|[^|]{0,160}(?:w_i|w_j|w_k|\\mathbf\{w\}_i|\\mathbf\{w\}_j|\\mathbf\{w\}_k)[^|]{0,100}\|(?:\^2|\^\{2\})",
            compact,
            flags=re.IGNORECASE,
        )
        or re.search(r"Tr\s*\([^)]*(?:W_i|W_k|\\mathbf\{W\}_i|\\mathbf\{W\}_k)", compact, flags=re.IGNORECASE)
    )


def _phase2_sentence_claims_concavity(text: str, anchor_patterns: list[str]) -> bool:
    sentences = re.split(r"(?<=[.!?])\s+|\n+", re.sub(r"\s+", " ", text.lower()))
    negation_markers = [
        "not concave",
        "not jointly concave",
        "not generally concave",
        "nonconcave",
        "non-concave",
        "is not",
        "cannot be claimed concave",
    ]
    for sentence in sentences:
        if not any(re.search(pattern, sentence, flags=re.IGNORECASE) for pattern in anchor_patterns):
            continue
        if "concave" not in sentence:
            continue
        if any(marker in sentence for marker in negation_markers):
            continue
        return True
    return False


def _phase2_has_unscoped_forbidden_phrase(text: str, phrases: list[str]) -> bool:
    sentences = re.split(r"(?<=[.!?])\s+|\n+", re.sub(r"\s+", " ", text.lower()))
    negation_markers = [
        "do not claim",
        "don't claim",
        "not claim",
        "not a",
        "not necessarily",
        "not guaranteed",
        "no claim",
        "without claiming",
        "rather than claiming",
        "avoid claiming",
        "cannot be claimed",
    ]
    for sentence in sentences:
        if not any(phrase.lower() in sentence for phrase in phrases):
            continue
        plain_sentence = re.sub(r"[*_`]+", "", sentence)
        if any(marker in plain_sentence for marker in negation_markers):
            continue
        if "classification" in plain_sentence and ("proxy" in plain_sentence or "surrogate" in plain_sentence):
            continue
        return True
    return False


def _phase2_has_unscoped_scope_item(text: str, phrases: list[str]) -> bool:
    sentences = re.split(r"(?<=[.!?])\s+|\n+", re.sub(r"\s+", " ", text.lower()))
    negation_markers = [
        "no ",
        "not ",
        "without ",
        "absent",
        "not part of",
        "not included",
        "is not included",
        "are not included",
        "not introduce",
        "do not introduce",
    ]
    for sentence in sentences:
        if not any(phrase.lower() in sentence for phrase in phrases):
            continue
        plain_sentence = re.sub(r"[*_`]+", "", sentence)
        if any(marker in plain_sentence for marker in negation_markers):
            continue
        return True
    return False


def build_wireless_feasibility_guardrail(topic: str, handoff: dict[str, Any] | None = None, *extra_contexts: Any) -> str:
    """Prompt block shared by theory phases to prevent over-scoped wireless topics."""
    if isinstance(handoff, dict):
        context = "\n".join(
            str(handoff.get(key, ""))
            for key in ("final_title", "problem_statement", "wireless_scenario", "objective", "variables", "core_constraints", "reformulation_path")
        )
    else:
        context = str(handoff or "")
    return build_dynamic_wireless_feasibility_guardrail(topic, context, *extra_contexts)


def _phase2_contract_report_path(run_dir: Path, phase_name: str) -> Path:
    return Path(run_dir) / phase_name / "technical_contract_gate.json"


def validate_phase2_phase1_contract(
    *,
    run_dir: Path,
    topic: str,
    handoff: dict[str, Any] | None = None,
    mathematical_contract: dict[str, Any] | None = None,
    system_model_md: str,
    problem_formulation_md: str,
    core_theory_package_md: str,
) -> dict[str, Any]:
    handoff = handoff or {}
    text = f"{topic}\n{system_model_md}\n{problem_formulation_md}\n{core_theory_package_md}"
    lowered = text.lower()
    handoff_scope = "\n".join(
        str(handoff.get(key, ""))
        for key in ("final_title", "problem_statement", "objective", "variables", "core_constraints")
    ).lower()
    mechanisms = _phase2_count_hard_mechanisms(text)
    errors: list[str] = []
    warnings: list[str] = []

    required_sections = {
        "system_model_md": system_model_md,
        "problem_formulation_md": problem_formulation_md,
        "core_theory_package_md": core_theory_package_md,
    }
    for name, value in required_sections.items():
        if len(str(value).strip()) < 120:
            errors.append(f"{name} is empty or too short for Phase 2.1 handoff.")

    if {"crb", "isac_sensing", "ris", "swipt_eh"}.issubset(set(mechanisms)):
        errors.append(
            "Topic/model combines CRB or radar sensing, RIS, and energy harvesting. "
            "This is over-scoped for the automatic WCL pipeline unless Phase 1 explicitly reduces one mechanism."
        )
    handoff_declares_only_generic_beams = (
        "power-splitting" in handoff_scope
        and "beamforming vectors" in handoff_scope
        and "energy beam" not in handoff_scope
        and "dedicated energy" not in handoff_scope
        and "energy-only" not in handoff_scope
        and "\\mathbf{v}" not in handoff_scope
    )
    if handoff_declares_only_generic_beams and _phase2_has_unscoped_scope_item(
        text,
        [
            "dedicated energy-only",
            "energy-only receiver",
            "dedicated eu",
            "energy beam",
            "energy waveform",
            "\\mathbf{v}",
            " v_m",
            "q_m^{",
        ],
    ):
        errors.append(
            "Phase 2.1 introduces dedicated energy receivers or energy-beam variables that are not in the Phase 1 declared variable contract."
        )
    handoff_declares_auxiliary_received_power = (
        "auxiliary" in handoff_scope
        and ("received-power" in handoff_scope or "received power" in handoff_scope or "rectifier" in handoff_scope)
    )
    if handoff_declares_auxiliary_received_power and _phase2_original_optimizer_mentions_auxiliary_quantities(problem_formulation_md):
        errors.append(
            "The original optimization problem includes auxiliary/derived quantities such as received power, rectifier input power, SINR/rate, or slack variables in the optimizer line. "
            "Optimize only physical controllable variables in the original optimization problem and keep auxiliary quantities as definitions or reformulation-only variables."
        )
    if "crb" in mechanisms and not _phase2_has_any(text, ["fisher information", "fim", "derivative", "jacobian"]):
        errors.append("CRB is mentioned but no Fisher information model or derivative-based observation model is defined.")
    if _phase2_has_any(text, ["multiuser", "multi-user", "multiple users", "k users"]) and "sinr" in lowered:
        has_per_user_beams = bool(re.search(r"\b(w_\{?k\}?|W_\{?k\}?|{\bf w}_\{?k\}?|\\mathbf\{w\}_\{?k\}?|\\mathbf\{W\}_\{?k\}?)", text))
        uses_aggregate_cov = bool(re.search(r"\bR[_\^\{\\]*x", text))
        if uses_aggregate_cov and not has_per_user_beams:
            errors.append(
                "Multiuser SINR is defined with an aggregate transmit covariance R_x but no per-user beam/covariance W_k. "
                "Use W_k/w_k for interference-aware unicast beamforming, or change to a common multicast/single-user model."
            )
    if "uplink" in lowered and "power" in lowered and "sinr" in lowered and "spectral radius" in lowered:
        if re.search(r"G[_\{]?k,?j[\}\]]?\s*=.*G[_\{]?k,?j[\}\]]?\s*/\s*G[_\{]?k,?k", text, flags=re.I | re.S) and re.search(
            r"F[_\{]?k,?j[\}\]]?\s*=.*G[_\{]?k,?j[\}\]]?\s*/\s*G[_\{]?k,?k",
            text,
            flags=re.I | re.S,
        ) or (
            "normalized effective channel gain" in lowered
            and "g_{kj}/g_{kk}" in lowered
            and "f_{kj}" in lowered
            and "g_{kj}/g_{kk}" in lowered[lowered.find("f_{kj}") :]
        ):
            errors.append(
                "Uplink power-control model appears to normalize G_kj by G_kk and then divide by G_kk again in F_kj. "
                "Use either raw gains a_k,b_kj with F_kj=gamma_k*b_kj/a_k, or normalized gains c_kj=b_kj/a_k with F_kj=gamma_k*c_kj, but not both."
            )
        if "any monotone norm" in lowered and re.search(r"rho\s*\([^)]*F[^)]*\).*\^t|\\rho\s*\([^)]*F[^)]*\).*\^t", text, flags=re.I | re.S):
            errors.append(
                "Convergence theorem claims a C*rho(F)^t bound in any monotone norm. "
                "State the bound in a suitable weighted sup norm, or use C*(rho(F)+epsilon)^t for arbitrary norms; keep rho(F) as the asymptotic factor."
            )
    if _phase2_has_swipt_eh(text) and _phase2_has_independent_streams(text):
        has_independent_eh_power_sum = _phase2_detect_independent_eh_power_sum(text)
        if _phase2_detect_coherent_eh_power_formula(text) and not has_independent_eh_power_sum:
            errors.append(
                "EH RF input power appears to be modeled as the coherent square of summed beamforming vectors. "
                "For independent data/energy streams, use the received-power expectation: a sum of per-stream powers or covariance traces."
            )
        elif not has_independent_eh_power_sum and _phase2_has_any(text, ["P_m", "P^{\\mathrm{rf}}", "rf power"]):
            warnings.append(
                "SWIPT/EH model mentions independent streams but does not clearly show RF power as an expectation or sum of stream powers."
            )
    if re.search(r"V_\{?m,?m\}?\s*\\?leq\s*1|diag\s*\([^)]*V[^)]*\)\s*\\?leq\s*1", text) and _phase2_has_any(text, ["unit-modulus", "unit modulus", "|theta", "passive ris"]):
        errors.append("RIS unit-modulus hardware is relaxed to V_mm <= 1 without explicitly changing the physical model.")
    declared_scope = f"{topic}\n{handoff_scope}"
    declared_mechanisms = set(_phase2_count_hard_mechanisms(declared_scope))
    if "movable_antenna" in mechanisms and "movable_antenna" not in declared_mechanisms:
        errors.append("Movable-antenna terminology appears although the selected topic does not include movable antennas.")
    if _phase2_has_any(text, ["ue-mounted ris", "ris at the ue", "ris on the ue", "ris-equipped ue"]):
        warnings.append("RIS is placed at/inside a UE. Prefer a fixed receiver-side panel, edge node, facade, or infrastructure-mounted RIS.")

    if _phase2_uses_uncertainty_or_chance(text) and not _phase2_has_concrete_uncertainty_model(text):
        warnings.append(
            "Robust/chance/ambiguity language appears, but the uncertainty or ambiguity model is not concrete enough for a proof or implementation. "
            "This is an advisory signal for the formulation agent; resolve through the prompt/LLM semantic pass instead of keyword-blocking."
        )
    if _phase2_shared_waveform_needs_receiver_scope(text):
        warnings.append(
            "An auxiliary/common/service signal or matrix-valued transmit resource appears in a receiver metric without a receiver decoding/cancellation assumption. "
            "Ask the formulation agent to verify the receiver/protocol information pattern from the current model semantics."
        )
    if _phase2_covariance_needs_physical_scope(text):
        warnings.append(
            "Matrix-valued transmit controls are used without a physical signaling or recovery interpretation. "
            "Ask the formulation/theory agent to verify whether the matrix is a physical covariance, lifted relaxation, or compact notation."
        )
    if re.search(r"epsilon[^.\n]{0,90}(outage|chance)", lowered) and re.search(
        r"epsilon[^.\n]{0,90}(uncertainty|radius|csi|channel)", lowered
    ):
        warnings.append(
            "The symbol epsilon may be used for both outage/chance tolerance and uncertainty radius; ask the formulation agent to verify notation roles from the current artifact."
        )
    if _phase2_has_any(text, ["transmit power minimization", "minimize transmit power", "minimum transmit power"]):
        if not _phase2_has_any(text, ["sinr", "rate", "harvest", "sensing", "outage", "reliability", "qos", "quality-of-service"]):
            warnings.append(
                "Transmit-power minimization appears without nontrivial service/reliability constraints. "
                "Frame it as minimum resource under QoS/reliability requirements, not as a standalone contribution."
            )
    contract_for_geometry = (
        mathematical_contract
        or read_json(Path(run_dir) / "phase2-1" / "mathematical_contract.json")
        or read_json(Path(run_dir) / "phase2-1" / "mathematical_contract.frozen.json")
        or {}
    )
    geometry_errors, geometry_warnings = _phase2_geometry_convention_issues(
        contract_for_geometry,
        system_model_md,
        problem_formulation_md,
        core_theory_package_md,
    )
    errors.extend(geometry_errors)
    warnings.extend(geometry_warnings)
    if _phase2_has_any(text, ["propulsion", "flight energy", "movement energy", "mobility energy"]) and not _phase2_has_any(
        text,
        [
            "propulsion model",
            "flight-energy model",
            "convex",
            "quadratic",
            "rotary-wing",
            "fixed-wing",
            "velocity",
            "speed",
            "P_0",
            "P_i",
            "defined as",
            "is given by",
        ],
    ):
        warnings.append(
            "A propulsion or movement-energy term appears without an explicit model or solver-relevant properties. "
            "Phase 2.1 should define the expression/properties before Phase 2.2 claims tractability."
        )

    report = {"ok": not errors, "errors": errors, "warnings": warnings, "detected_mechanisms": mechanisms}
    write_text(_phase2_contract_report_path(run_dir, "phase2-1"), json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise ValueError("Phase 2.1 technical contract failed: " + "; ".join(errors))
    return report


def validate_phase2_phase2_contract(
    *,
    run_dir: Path,
    convexity_audit_md: str,
    reformulation_path_md: str,
) -> dict[str, Any]:
    text = f"{convexity_audit_md}\n{reformulation_path_md}"
    lowered = text.lower()
    errors: list[str] = []
    warnings: list[str] = []
    if _phase2_sentence_claims_concavity(text, [r"weighted\s+sum-?rate", r"\bsum-?rate\b"]):
        errors.append(
            "A multiuser rate objective is claimed concave; this is false for generic interference unless a valid transform/block is specified."
        )
    if _phase2_sentence_claims_concavity(text, [r"\br_x\b"]):
        errors.append(
            "A rate/radar term is claimed concave in a single aggregate R_x; verify or reformulate with per-user variables."
        )
    if "uplink" in lowered and "power" in lowered and "sinr" in lowered:
        if "dinkelbach" in lowered:
            errors.append("Dinkelbach/fractional-programming baselines are irrelevant for minimum-sum-power SINR LP unless an energy-efficiency fractional objective is explicitly introduced.")
        if "any monotone norm" in lowered and re.search(r"rho\s*\([^)]*F[^)]*\).*\^t|\\rho\s*\([^)]*F[^)]*\).*\^t", text, flags=re.I | re.S):
            errors.append(
                "Convergence-rate claim uses C*rho(F)^t in any monotone norm. Use a weighted sup norm for rho(F), or C_epsilon*(rho(F)+epsilon)^t for arbitrary norms."
            )
        if re.search(r"C\s*\[?\\?rho\s*\([^)]*F[^)]*\)\]?\^t", text) and "weighted" not in lowered and "epsilon" not in lowered:
            warnings.append("A C*rho(F)^t convergence bound appears without specifying weighted norm or epsilon slack.")
    if "r_x^{1/2}" in lowered and "affine" in lowered:
        errors.append("R_x^{1/2} is claimed affine; matrix square root is not an affine map.")
    if (
        re.search(r"fixed[^.\n]{0,80}gamma|gamma[^.\n]{0,80}fixed|fixed[^.\n]{0,80}\\boldsymbol\{\\gamma\}", lowered)
        and _phase2_has_any(text, ["w-subproblem", "beamformer block", "beamforming block"])
        and _phase2_has_any(text, ["convex qcqp", "convex quadratic", "strictly convex"])
        and not _phase2_has_any(text, ["wmmse", "receive filter", "mse weight", "quadratic transform variable", "auxiliary y"])
    ):
        errors.append(
            "A single SINR/gamma auxiliary is used to claim a convex beamformer block. For generic multiuser interference, use WMMSE or a complete quadratic-transform/SCA construction with all auxiliary variables."
        )
    if _phase2_has_swipt_eh(text) and _phase2_has_any(text, ["sigmoid", "logistic", "nonlinear eh", "non-linear eh"]):
        if _phase2_has_any(text, ["convex qcqp", "concave quadratic surrogate", "standard convex qcqp"]):
            if _phase2_has_any(
                text,
                [
                    "(p_m",
                    "p_m^{",
                    "(p_k",
                    "p_k^{",
                    "q_k",
                    "rf-power auxiliary",
                    "auxiliary rf",
                    "epigraph",
                    "lifted scalar",
                    "lifted variables",
                    "rf-power auxiliaries",
                ],
            ):
                warnings.append(
                    "Nonlinear-EH convexity depends on the stated auxiliary RF-power variable and its relaxation; verify the surrogate is concave in the solver variables."
                )
            else:
                errors.append(
                    "Nonlinear-EH SCA/MM is claimed to yield a convex QCQP or concave quadratic surrogate without an auxiliary RF-power variable/valid convexification."
                )
        if re.search(r"\(P(?:_\{?m\}?|_m|\^\{?\\mathrm\{rf\}\}?)[^)]*-P(?:_\{?m\}?|_m)?\^\{\(?t\)?\}\)\s*\^2", text, flags=re.IGNORECASE):
            errors.append(
                "The reformulation squares a quadratic RF-power expression around an iterate; this is generally quartic, not a convex QCQP, unless lifted scalar variables and convex constraints are explicitly introduced."
            )
    if ("standard convex sdp" in lowered or "convex sdp" in lowered) and not _phase2_has_any(
        text,
        ["under fixed", "for fixed", "after linearization", "surrogate", "sufficient condition", "relaxed subproblem"],
    ):
        warnings.append("Convex SDP language appears without clear block/fixed-variable scoping.")
    if "schur complement" in lowered and "sinr" in lowered:
        if _phase2_has_any(text, ["linear in", "W_k", "\\mathbf{W}_k", "lifted"]):
            warnings.append("SINR lifting is scoped, but avoid calling the linear lifted SINR constraint a Schur-complement step unless an actual LMI is shown.")
        else:
            errors.append("SINR constraints are claimed SOC/Schur-complement representable without a precise convex lifted or fixed-beam form.")
    has_phi_placeholder = bool(re.search(r"(\\Phi|Phi_m|\bphi_m\b)", text))
    if has_phi_placeholder and (
        not _phase2_has_concrete_uncertainty_model(text)
        or _phase2_has_any(text, ["selected ambiguity model", "chosen ambiguity model", "denoted by", "placeholder"])
    ):
        warnings.append(
            "The reformulation uses a Phi-style safe-counterpart placeholder without a concrete ambiguity/uncertainty model and reproducible deterministic expression. "
            "This is an advisory signal for the theory agent; resolve it through the Phase 2.2 prompt/LLM route rather than keyword-blocking."
        )
    if _phase2_uses_uncertainty_or_chance(text) and _phase2_has_any(text, ["safe conic", "safe counterpart", "robust counterpart"]):
        if not _phase2_has_concrete_uncertainty_model(text):
            warnings.append(
                "A robust/chance safe counterpart is claimed without naming a concrete model family such as ellipsoidal/norm-bounded, Gaussian, moment/Cantelli, Bernstein, scenario, or Wasserstein."
            )
    if re.search(r"t[_\{]?[a-z0-9\\]*\}?\s*\\?rho|rho[_\{]?[a-z0-9\\]*\}?\s*t[_\{]?", text, flags=re.I) and _phase2_has_any(
        text, ["\\ge 1", "\\geq 1", ">= 1"]
    ):
        if not _phase2_has_any(text, ["rotated second-order cone", "rotated soc", "rsoc", "perspective", "reciprocal epigraph"]):
            warnings.append(
                "A reciprocal/product epigraph such as t*rho >= 1 appears without an explicit convex representation or substitution."
            )
    if _phase2_has_any(text, ["rank recovery is unnecessary", "no rank recovery", "without rank recovery"]) and not _phase2_has_any(
        text, ["gaussian signaling", "multi-stream", "multistream", "covariance-domain", "high-rank covariance is physically implemented"]
    ):
        warnings.append(
            "The route says rank recovery is unnecessary but does not explain the physical implementation of high-rank covariance signaling."
        )
    if _phase2_shared_waveform_needs_receiver_scope(text):
        warnings.append(
            "The reformulation carries an auxiliary/common/service signal or matrix-valued transmit resource into a receiver metric without preserving the receiver information-pattern assumption."
        )
    if re.search(r"epsilon[^.\n]{0,90}(outage|chance)", lowered) and re.search(
        r"epsilon[^.\n]{0,90}(uncertainty|radius|csi|channel)", lowered
    ):
        warnings.append(
            "Epsilon is used for both outage/chance tolerance and uncertainty radius in the theory route; assign distinct symbols before algorithm generation."
        )
    if _phase2_has_any(text, ["propulsion", "flight energy", "movement energy", "mobility energy"]) and not _phase2_has_any(
        text,
        [
            "propulsion model",
            "flight-energy model",
            "convex",
            "quadratic",
            "rotary-wing",
            "fixed-wing",
            "velocity",
            "speed",
            "P_0",
            "P_i",
            "defined as",
            "is given by",
        ],
    ):
        warnings.append(
            "A movement/propulsion-energy term reaches Phase 2.2 without an explicit expression or solver-relevant properties."
        )
    if "perspective" in lowered and _phase2_has_any(text, ["power", "time", "slot"]) and not _phase2_has_any(
        text,
        ["energy variable", "transmit energy", "e_{", "tau p", "time-energy", "time energy"],
    ):
        warnings.append(
            "A perspective route is mentioned for a time/power model without identifying the corresponding energy variable or valid perspective variables."
        )
    report = {"ok": not errors, "errors": errors, "warnings": warnings}
    write_text(_phase2_contract_report_path(run_dir, "phase2-2"), json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise ValueError("Phase 2.2 technical contract failed: " + "; ".join(errors))
    return report


def validate_phase2_phase3_contract(
    *,
    run_dir: Path,
    algorithm_md: str,
    convergence_or_complexity_md: str,
    experiment_blueprint_md: str = "",
) -> dict[str, Any]:
    # Phase 2.3 is now a pure proposed-method/theory phase. Experiment
    # artifacts are intentionally owned by Phase 2.4, so the contract gate only
    # audits algorithm/proof language here.
    _ = experiment_blueprint_md
    text = f"{algorithm_md}\n{convergence_or_complexity_md}"
    mathematical_contract, algorithm_contract = _phase2_load_phase_contracts(Path(run_dir))
    errors: list[str] = []
    warnings: list[str] = []
    checks: dict[str, Any] = {}
    controls = [item for item in mathematical_contract.get("controls") or [] if isinstance(item, dict)]
    if controls:
        covered, missing = _phase2_count_contract_coverage(algorithm_md, controls)
        required = min(len(controls), 2)
        checks["control_coverage"] = {
            "covered": covered,
            "total": len(controls),
            "missing_examples": missing[:5],
        }
        if covered < required:
            errors.append(
                    f"Phase 2.3 algorithm does not cover enough frozen controls from mathematical_contract "
                f"({covered}/{len(controls)} covered, need at least {required})."
            )
    objective = mathematical_contract.get("objective") or {}
    if isinstance(objective, dict) and (objective.get("expression") or objective.get("meaning")):
        objective_payload = " ".join(
            str(objective.get(key) or "") for key in ("sense", "expression", "meaning")
        )
        objective_covered = _phase2_text_covers_contract_item(text, objective_payload)
        checks["objective_alignment"] = {"covered": objective_covered}
        if not objective_covered:
            errors.append("Phase 2.3 does not explicitly align the algorithm/proof with the frozen physical objective.")
    constraints = [item for item in mathematical_contract.get("constraints") or [] if isinstance(item, dict)]
    if constraints:
        has_feasibility_language = _phase2_has_any(
            text,
            ["constraint", "constraints", "feasible", "feasibility", "projection", "violation", "satisfy", "satisfies"],
        )
        checks["constraint_handling"] = {"mentions_feasibility_or_constraints": has_feasibility_language}
        if not has_feasibility_language:
            errors.append("Phase 2.3 does not explain how frozen constraints are preserved, checked, or projected.")
    execution_contract = algorithm_contract.get("algorithm_execution_contract") or {}
    if isinstance(execution_contract, dict) and execution_contract:
        update_blocks = [str(item) for item in execution_contract.get("update_blocks") or [] if str(item).strip()]
        if update_blocks:
            covered, missing = _phase2_count_contract_coverage(algorithm_md, update_blocks)
            required = max(1, (len(update_blocks) + 1) // 2)
            checks["update_block_coverage"] = {
                "covered": covered,
                "total": len(update_blocks),
                "missing_examples": missing[:5],
            }
            if covered < required:
                errors.append(
                    f"Phase 2.3 algorithm does not implement enough update blocks from algorithm_execution_contract "
                    f"({covered}/{len(update_blocks)} covered, need at least {required})."
                )
        objective_evaluator = str(execution_contract.get("objective_evaluator") or "")
        if objective_evaluator:
            evaluator_covered = _phase2_text_covers_contract_item(text, objective_evaluator)
            checks["objective_evaluator_alignment"] = {"covered": evaluator_covered}
            if not evaluator_covered:
                errors.append("Phase 2.3 does not match the objective_evaluator in algorithm_execution_contract.")
        constraint_evaluator = str(execution_contract.get("constraint_evaluator") or "")
        if constraint_evaluator:
            evaluator_covered = _phase2_text_covers_contract_item(text, constraint_evaluator)
            checks["constraint_evaluator_alignment"] = {"covered": evaluator_covered}
            if not evaluator_covered:
                warnings.append("Phase 2.3 only weakly matches the constraint_evaluator in algorithm_execution_contract.")
    report = {"ok": not errors, "errors": errors, "warnings": warnings, "checks": checks}
    write_text(_phase2_contract_report_path(run_dir, "phase2-3"), json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise ValueError("Phase 2.3 technical contract failed: " + "; ".join(errors))
    return report


def card_index(cards_dir: Path, limit: int = 8) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if not cards_dir.exists():
        return rows
    for idx, path in enumerate(sorted(cards_dir.glob("*.md"))):
        if idx >= limit:
            break
        body = read_text(path)
        title = body.splitlines()[0].replace("#", "").strip() if body else path.stem
        rows.append({"id": path.stem, "title": title, "path": str(path)})
    return rows


def build_phase1_handoff(phase1_run: Path, run_dir: Path) -> dict[str, Any]:
    handoff_dir = run_dir / "input_from_phase1"
    handoff_dir.mkdir(parents=True, exist_ok=True)

    hypotheses_md = read_text(phase1_run / "phase3-3" / "hypotheses.md")
    synthesis_md = read_text(phase1_run / "phase3-2" / "synthesis.md")
    topic_taxonomy = read_json(phase1_run / "phase3-3" / "topic_taxonomy.json") or {}
    topic_score = read_json(phase1_run / "phase3-3" / "topic_score.json") or {}
    review_report = read_json(phase1_run / "phase3-3" / "review_report.json") or {}
    shortlist = shortlist_preview(phase1_run / "phase2-5" / "shortlist.jsonl")
    cards = card_index(phase1_run / "phase3-1" / "cards")

    candidate_block = extract_candidate_block(hypotheses_md)
    final_title = extract_first_candidate_title(hypotheses_md)
    problem_statement = extract_section(candidate_block, "Problem statement")
    wireless_scenario = extract_section(candidate_block, "Wireless scenario")
    objective = extract_section(candidate_block, "Objective")
    core_constraints = extract_section(candidate_block, "Core constraints")
    theorem_target = extract_section(candidate_block, "Theorem / proof target")
    reformulation_path = extract_section(candidate_block, "Convexification / reformulation path")
    validation_plan = extract_section(candidate_block, "Minimal validation plan")
    claimed_contribution = extract_section(candidate_block, "Claimed contribution")
    novelty_delta = extract_section(candidate_block, "Novelty delta vs prior art")
    nonconvexity = extract_section(candidate_block, "Source of nonconvexity")

    final_topic_md = (
        f"# Final Topic\n\n"
        f"## Recommended Title\n{final_title or 'TBD'}\n\n"
        f"## Source Run\n{phase1_run}\n\n"
        f"## Topic Score\n- overall: {topic_score.get('overall_score', '-')}\n"
        f"- verdict: {topic_score.get('verdict', '-')}\n"
    )
    problem_statement_md = (
        "# Problem Statement\n\n"
        f"## Wireless Scenario\n{wireless_scenario or 'TBD'}\n\n"
        f"## Problem Statement\n{problem_statement or 'TBD'}\n\n"
        f"## Objective\n{objective or 'TBD'}\n\n"
        f"## Core Constraints\n{core_constraints or 'TBD'}\n"
    )
    algorithm_sketch_md = (
        "# Algorithm Sketch\n\n"
        f"## Reformulation Path\n{reformulation_path or 'TBD'}\n\n"
        f"## Proposed Route\n{claimed_contribution or 'TBD'}\n"
    )
    theorem_targets_md = (
        "# Theorem Targets\n\n"
        f"## Target\n{theorem_target or 'TBD'}\n\n"
        f"## Novelty Delta\n{novelty_delta or 'TBD'}\n"
    )
    validation_targets_md = (
        "# Validation Targets\n\n"
        f"{validation_plan or 'TBD'}\n"
    )

    write_text(handoff_dir / "final_topic.md", final_topic_md)
    write_text(handoff_dir / "problem_statement.md", problem_statement_md)
    write_text(handoff_dir / "algorithm_sketch.md", algorithm_sketch_md)
    write_text(handoff_dir / "theorem_targets.md", theorem_targets_md)
    write_text(handoff_dir / "validation_targets.md", validation_targets_md)
    write_text(handoff_dir / "hypotheses.md", hypotheses_md)
    write_text(handoff_dir / "synthesis.md", synthesis_md)
    write_text(handoff_dir / "shortlist_preview.json", json.dumps(shortlist, ensure_ascii=False, indent=2))
    write_text(handoff_dir / "card_index.json", json.dumps(cards, ensure_ascii=False, indent=2))
    write_text(handoff_dir / "topic_taxonomy.json", json.dumps(topic_taxonomy, ensure_ascii=False, indent=2))
    write_text(handoff_dir / "topic_score.json", json.dumps(topic_score, ensure_ascii=False, indent=2))
    write_text(handoff_dir / "review_report.json", json.dumps(review_report, ensure_ascii=False, indent=2))

    return {
        "phase1_run": str(phase1_run),
        "final_title": final_title,
        "problem_statement": problem_statement,
        "wireless_scenario": wireless_scenario,
        "objective": objective,
        "core_constraints": core_constraints,
        "theorem_target": theorem_target,
        "reformulation_path": reformulation_path,
        "validation_plan": validation_plan,
        "claimed_contribution": claimed_contribution,
        "novelty_delta": novelty_delta,
        "nonconvexity": nonconvexity,
        "handoff_dir": str(handoff_dir),
    }


def render_phase1_ieee_preview_pdf(phase_dir: Path) -> dict[str, str]:
    wrapper_tex = phase_dir / "phase1_ieee_preview.tex"
    wrapper_tex_content = r"""\documentclass[10pt,conference]{IEEEtran}
\usepackage{amsmath,amssymb,amsfonts,bm,mathtools}
\usepackage[hidelinks]{hyperref}

\begin{document}
\title{Phase 2.1 Preview}
\author{}
\maketitle

\section{Preview}
\input{system_model_problem_formulation_ieee_wcl.tex}

\end{document}
"""
    write_text(wrapper_tex, wrapper_tex_content)

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

    pdf_path = phase_dir / "phase1_ieee_preview.pdf"
    log_path = phase_dir / "phase1_ieee_preview.log"
    if not pdf_path.exists():
        stderr_text = ""
        if last_result is not None:
            stderr_text = f"\nSTDOUT:\n{last_result.stdout}\nSTDERR:\n{last_result.stderr}"
        raise RuntimeError(f"Phase 2.1 PDF preview was not generated.{stderr_text}")
    return {
        "preview_tex": str(wrapper_tex),
        "preview_pdf": str(pdf_path),
        "preview_log": str(log_path),
    }


def build_phase2_phase1_prompt(topic: str, handoff: dict[str, Any], topic_taxonomy: dict[str, Any], synthesis_md: str) -> str:
    layers = ((topic_taxonomy.get("input_profile") or {}).get("display_layers") or [])
    layer_lines = []
    for layer in layers:
        label = str(layer.get("label") or layer.get("id") or "")
        tags = ", ".join(str(x) for x in (layer.get("tag_labels") or []))
        layer_lines.append(f"- {label}: {tags}")

    return render_prompt_template(
        "phase2_1/system_model_problem.prompt.yaml",
        wireless_feasibility_guardrail=build_wireless_feasibility_guardrail(topic, handoff, synthesis_md),
        topic=topic,
        handoff_final_title=handoff.get("final_title", ""),
        handoff_problem_statement=handoff.get("problem_statement", ""),
        handoff_wireless_scenario=handoff.get("wireless_scenario", ""),
        handoff_objective=handoff.get("objective", ""),
        handoff_variables=handoff.get("variables", ""),
        handoff_core_constraints=handoff.get("core_constraints", ""),
        handoff_nonconvexity=handoff.get("nonconvexity", ""),
        handoff_theorem_target=handoff.get("theorem_target", ""),
        handoff_reformulation_path=handoff.get("reformulation_path", ""),
        handoff_claimed_contribution=handoff.get("claimed_contribution", ""),
        taxonomy_layer_lines="\n".join(layer_lines),
        synthesis_excerpt=synthesis_md[:5000],
    )


def build_phase2_phase1_latex_prompt(
    *,
    topic: str,
    system_model_md: str,
    problem_formulation_md: str,
    core_theory_package_md: str,
    mathematical_contract_json: str = "",
) -> str:
    return render_prompt_template(
        "phase2_1/latex_system_problem.prompt.yaml",
        topic=topic,
        wireless_feasibility_guardrail=build_wireless_feasibility_guardrail(
            topic,
            mathematical_contract_json,
            system_model_md,
            problem_formulation_md,
            core_theory_package_md,
        ),
        mathematical_contract_json=mathematical_contract_json or "{}",
        system_model_md=system_model_md,
        problem_formulation_md=problem_formulation_md,
        core_theory_package_md=core_theory_package_md,
    )


def build_phase2_phase1_latex_repair_prompt(
    *,
    topic: str,
    current_tex: str,
    issue_summary: str,
    repair_mode: str = "formatting_repair",
    mathematical_contract_json: str = "",
) -> str:
    return render_prompt_template(
        "phase2_1/latex_system_problem_repair.prompt.yaml",
        topic=topic,
        repair_mode=repair_mode,
        mathematical_contract_json=mathematical_contract_json or "{}",
        issue_summary=issue_summary,
        current_tex=current_tex,
    )


def build_phase2_phase2_prompt(
    *,
    topic: str,
    handoff: dict[str, Any],
    mathematical_contract_json: str = "",
    system_model_md: str,
    problem_formulation_md: str,
    core_theory_package_md: str,
    tractability_route_policy: dict[str, Any] | None = None,
) -> str:
    if tractability_route_policy is None:
        tractability_route_policy = build_tractability_route_policy(
            topic=topic,
            handoff=handoff,
            mathematical_contract_json=mathematical_contract_json or "{}",
            system_model_md=system_model_md,
            problem_formulation_md=problem_formulation_md,
            core_theory_package_md=core_theory_package_md,
        )
    return render_prompt_template(
        "phase2_2/convexity_reformulation.prompt.yaml",
        wireless_feasibility_guardrail=build_wireless_feasibility_guardrail(topic, handoff, system_model_md, problem_formulation_md, core_theory_package_md),
        tractability_route_policy=tractability_route_policy_prompt_block(tractability_route_policy),
        topic=topic,
        handoff_final_title=handoff.get("final_title", ""),
        handoff_theorem_target=handoff.get("theorem_target", ""),
        handoff_reformulation_path=handoff.get("reformulation_path", ""),
        mathematical_contract_json=mathematical_contract_json or "{}",
        system_model_md=system_model_md,
        problem_formulation_md=problem_formulation_md,
        core_theory_package_md=core_theory_package_md,
    )


def build_phase2_phase3_prompt(
    *,
    topic: str,
    handoff: dict[str, Any],
    mathematical_contract_json: str = "",
    system_model_md: str,
    problem_formulation_md: str,
    core_theory_package_md: str,
    convexity_audit_md: str,
    reformulation_path_md: str,
    tractability_route_policy: dict[str, Any] | None = None,
) -> str:
    if tractability_route_policy is None:
        tractability_route_policy = build_tractability_route_policy(
            topic=topic,
            handoff=handoff,
            mathematical_contract_json=mathematical_contract_json or "{}",
            system_model_md=system_model_md,
            problem_formulation_md=problem_formulation_md,
            core_theory_package_md=core_theory_package_md,
        )
    return render_prompt_template(
        "phase2_3/algorithm_design.prompt.yaml",
        wireless_feasibility_guardrail=build_wireless_feasibility_guardrail(
            topic,
            handoff,
            mathematical_contract_json,
            system_model_md,
            problem_formulation_md,
            core_theory_package_md,
            convexity_audit_md,
            reformulation_path_md,
        ),
        topic=topic,
        handoff_final_title=handoff.get("final_title", ""),
        mathematical_contract_json=mathematical_contract_json or "{}",
        system_model_md=system_model_md,
        problem_formulation_md=problem_formulation_md,
        core_theory_package_md=core_theory_package_md,
        convexity_audit_md=convexity_audit_md,
        reformulation_path_md=reformulation_path_md,
        tractability_route_policy=tractability_route_policy_prompt_block(tractability_route_policy),
    )


def build_phase2_phase3_latex_prompt(
    *,
    topic: str,
    mathematical_contract_json: str = "",
    algorithm_md: str,
    convergence_or_complexity_md: str,
    benchmark_definition_md: str,
) -> str:
    return render_prompt_template(
        "phase2_3/latex_solution.prompt.yaml",
        topic=topic,
        mathematical_contract_json=mathematical_contract_json or "{}",
        algorithm_md=algorithm_md,
        convergence_or_complexity_md=convergence_or_complexity_md,
        benchmark_definition_md=benchmark_definition_md,
    )


def build_phase2_phase3_latex_repair_prompt(
    *,
    topic: str,
    mathematical_contract_json: str = "",
    current_tex: str,
    issue_summary: str,
    compile_log_tail: str,
) -> str:
    return render_prompt_template(
        "phase2_3/latex_solution_repair.prompt.yaml",
        topic=topic,
        mathematical_contract_json=mathematical_contract_json or "{}",
        issue_summary=issue_summary,
        compile_log_tail=compile_log_tail,
        current_tex=current_tex,
    )


def build_phase3_1_writing_prompt(
    *,
    topic: str,
    mathematical_contract_json: str = "",
    system_model_md: str,
    problem_formulation_md: str,
    core_theory_package_md: str,
    convexity_audit_md: str,
    reformulation_path_md: str,
    algorithm_md: str,
    convergence_or_complexity_md: str,
    benchmark_definition_md: str,
) -> str:
    return render_prompt_template(
        "phase3_1/technical_writing.prompt.yaml",
        topic=topic,
        mathematical_contract_json=compact_text(mathematical_contract_json or "{}", 7000),
        system_model_md=compact_text(system_model_md, 2200),
        problem_formulation_md=compact_text(problem_formulation_md, 2400),
        core_theory_package_md=compact_text(core_theory_package_md, 1600),
        convexity_audit_md=compact_text(convexity_audit_md, 1600),
        reformulation_path_md=compact_text(reformulation_path_md, 2400),
        algorithm_md=compact_text(algorithm_md, 3200),
        convergence_or_complexity_md=compact_text(convergence_or_complexity_md, 1400),
        benchmark_definition_md=compact_text(benchmark_definition_md, 1200),
    )


def build_phase3_1_writing_repair_prompt(
    *,
    topic: str,
    mathematical_contract_json: str = "",
    issue_summary: str,
    compile_log_tail: str,
    current_system_model_problem_formulation_tex: str,
    current_proposed_solution_tex: str,
) -> str:
    return render_prompt_template(
        "phase3_1/technical_writing_repair.prompt.yaml",
        topic=topic,
        mathematical_contract_json=mathematical_contract_json or "{}",
        issue_summary=issue_summary,
        compile_log_tail=compile_log_tail,
        current_system_model_problem_formulation_tex=current_system_model_problem_formulation_tex,
        current_proposed_solution_tex=current_proposed_solution_tex,
    )


def analyze_latex_equation_line_format(tex: str) -> dict[str, Any]:
    """Detect dense equation displays that hurt IEEE WCL readability.

    The writing contract is intentionally conservative: one displayed line
    should carry one primary equality/inequality relation. Formal optimization
    displays may have multiple constraint lines, but each line must still carry
    at most one relation.
    """

    issues: list[dict[str, Any]] = []
    source = str(tex or "").replace("\r\n", "\n")
    env_pattern = re.compile(
        r"\\begin\{(?P<env>equation\*?|align\*?|aligned|split|subequations)\}(?P<body>.*?)\\end\{(?P=env)\}",
        flags=re.S,
    )
    relation_pattern = re.compile(r"(?:\\leq|\\geq|\\le(?![A-Za-z])|\\ge(?![A-Za-z])|\\approx|\\simeq|(?<![!<>])=|(?<!\\)[<>])")

    def clean_math_line(raw_line: str) -> str:
        line = re.sub(r"%.*$", "", raw_line).strip()
        line = re.sub(r"\\label\{[^{}]*\}", "", line)
        line = re.sub(r"\\nonumber|\\notag", "", line)
        line = re.sub(r"\\begin\{[^{}]*\}|\\end\{[^{}]*\}", "", line)
        line = re.sub(r"_\{[^{}]*\}", "_{}", line)
        line = re.sub(r"\^\{[^{}]*\}", "^{}", line)
        # Set-builder predicates are part of a single set definition, e.g.,
        # \mathcal H_k=\{\hat h_k+e:\|e\|\leq\delta\}; do not count the
        # predicate relation as a second primary displayed-equation relation.
        def mask_set_builder(match: re.Match[str]) -> str:
            segment = match.group(0)
            if ":" not in segment and r"\mid" not in segment:
                return segment
            return relation_pattern.sub(r"\\rel", segment)

        line = re.sub(r"\\\{.*?\\\}", mask_set_builder, line)
        line = re.sub(r",?\s*\\quad\s*[A-Za-z](?:_\{\})?\s*=\s*[^,]+(?:,\s*\\ldots\s*,\s*[^,]+)?", "", line)
        line = line.replace("&", "")
        line = line.rstrip(",.;")
        return line.strip()

    for match in env_pattern.finditer(source):
        env = match.group("env")
        body = match.group("body")
        start_line = source[: match.start()].count("\n") + 1
        is_formal_problem = bool(
            re.search(
                r"\\(?:max|min)(?![A-Za-z])|\\(?:text|mathrm)\{\s*s\.?t\.?\s*\}|\\mathrm\{\s*subject\s+to\s*\}",
                body,
                flags=re.I,
            )
        )
        if env == "subequations" and not is_formal_problem:
            issues.append(
                {
                    "line": start_line,
                    "environment": env,
                    "issue": "non_optimization_subequations",
                    "text": "reserve subequations for optimization problems; use ordinary equation or align displays for system-model definitions",
                }
            )
        relation_lines = 0
        raw_lines = re.split(r"\\\\|\n", body)
        for offset, raw_line in enumerate(raw_lines):
            line = clean_math_line(raw_line)
            if not line:
                continue
            relation_count = len(relation_pattern.findall(line))
            if relation_count:
                relation_lines += 1
            if relation_count > 1:
                issues.append(
                    {
                        "line": start_line + offset,
                        "environment": env,
                        "issue": "multiple_primary_relations_on_one_line",
                        "text": line,
                    }
                )
            if r"\qquad" in line and relation_count >= 1:
                issues.append(
                    {
                        "line": start_line + offset,
                        "environment": env,
                        "issue": "joined_definitions_with_qquad",
                        "text": line,
                    }
                )
        if env.startswith("equation") and r"\begin{aligned}" in body and relation_lines > 1 and not is_formal_problem:
            issues.append(
                {
                    "line": start_line,
                    "environment": env,
                    "issue": "multiple_independent_relations_under_one_equation_number",
                    "text": "equation display contains an aligned block with multiple relations",
                }
            )
    for match in re.finditer(r"\\label\{eq:p0[^{}]*\}", source, flags=re.I):
        issues.append(
            {
                "line": source[: match.start()].count("\n") + 1,
                "environment": "latex",
                "issue": "optimization_problem_labeled_as_equation",
                "text": match.group(0),
            }
        )
    for match in re.finditer(r"\\eqref\{eq:p0[^{}]*\}", source, flags=re.I):
        issues.append(
            {
                "line": source[: match.start()].count("\n") + 1,
                "environment": "latex",
                "issue": "optimization_problem_referenced_as_equation",
                "text": match.group(0),
            }
        )
    for match in re.finditer(r"\b[Pp]roblem\s*~?\s*\\eqref\{[^{}]*\}", source):
        issues.append(
            {
                "line": source[: match.start()].count("\n") + 1,
                "environment": "latex",
                "issue": "problem_referenced_with_eqref",
                "text": match.group(0),
            }
        )
    for match in re.finditer(r"\\(?:mathrm|operatorname|text)\{(?:maximi[sz]e|minimi[sz]e)\}", source, flags=re.I):
        issues.append(
            {
                "line": source[: match.start()].count("\n") + 1,
                "environment": "latex",
                "issue": "optimization_uses_full_word_maximize_or_minimize",
                "text": match.group(0),
            }
        )
    for match in re.finditer(r"\\(?:mathrm|text)\{subject(?:\\|\s)+to\}", source, flags=re.I):
        issues.append(
            {
                "line": source[: match.start()].count("\n") + 1,
                "environment": "latex",
                "issue": "optimization_uses_full_subject_to",
                "text": match.group(0),
            }
        )
    display_block_pattern = re.compile(
        r"\\begin\{(?P<env>subequations|align\*?|equation\*?)\}(?P<body>.*?)\\end\{(?P=env)\}",
        flags=re.S,
    )
    p0_header_pattern = re.compile(
        r"\\label\{prob:p0[^{}]*\}"
        r"|(?:\\text\{\(P0\)\}|\(P0\)|\\mathcal\{P\}_0)\s*:"
        r"|(?:\\text\{\(P0\)\}|\(P0\)|\\mathcal\{P\}_0)(?=.{0,100}\\(?:max|min|underset))",
        flags=re.I | re.S,
    )
    for match in display_block_pattern.finditer(source):
        env = match.group("env")
        block = match.group(0)
        body = match.group("body")
        if not p0_header_pattern.search(block):
            continue
        start_line = source[: match.start()].count("\n") + 1
        if env != "subequations" or r"\begin{align}" not in body or r"\begin{align*}" in body:
            issues.append(
                {
                    "line": start_line,
                    "environment": env,
                    "issue": "optimization_problem_constraints_not_numbered",
                    "text": "problem (P0) must use subequations with an inner numbered align environment",
                }
            )
        if re.search(r"\\label\{eq:p0[^{}]*\}", block, flags=re.I):
            continue
        problem_line_labels = re.findall(r"\\label\{(?:obj|con):p0[^{}]*\}", block, flags=re.I)
        raw_problem_lines = [
            line
            for line in re.split(r"\\\\|\n", body)
            if (r"\max" in line or r"\min" in line or relation_pattern.search(clean_math_line(line)))
            and not re.search(r"\\begin\{|\\end\{", line)
        ]
        if raw_problem_lines and len(problem_line_labels) < len(raw_problem_lines):
            issues.append(
                {
                    "line": start_line,
                    "environment": env,
                    "issue": "optimization_problem_lines_missing_constraint_labels",
                    "text": "label the objective with obj:p0_* and every constraint line with con:p0_* labels",
                }
            )
    return {"ok": not issues, "issues": issues}


def analyze_phase3_reformulation_repeats_system_model(tex: str) -> dict[str, Any]:
    """Detect proposed-solution openings that re-display System Model equations."""

    source = str(tex or "").replace("\r\n", "\n")
    match = re.search(
        r"\\subsection\{(?P<title>Problem Reformulation|Solution Approach|Candidate Evaluation)\}(?P<body>.*?)(?=\\subsection\{|\\section\{|$)",
        source,
        flags=re.S,
    )
    if not match:
        return {"ok": True, "issues": []}

    title = match.group("title")
    body = match.group("body")
    issues: list[dict[str, Any]] = []
    repeated_patterns = [
        ("received_power_definition", r"q_k\s*\([^)]*\\mathbf\s*w[^)]*\)\s*="),
        ("rf_input_power_definition", r"p_k\^\{\\mathrm\{in\}\}.*?="),
        ("sinr_definition", r"\\Gamma_k\s*\([^)]*\)\s*="),
        ("rate_definition", r"R_k\s*\([^)]*\)\s*="),
        ("rectifier_offset_definition", r"\\Omega_k\s*="),
        ("harvested_dc_definition", r"P_k\^\{\\mathrm\{dc\}\}.*?="),
        ("harvested_utility_definition", r"U_k\^\{\\mathrm\{eh\}\}.*?="),
        ("original_objective_definition", r"\\mathcal\s*U\s*\([^)]*\)\s*="),
    ]
    for issue_name, pattern in repeated_patterns:
        if re.search(pattern, body, flags=re.S):
            issues.append({"issue": issue_name, "section": title})
    solution_labels = re.findall(
        r"\\label\{eq:solution_(?:received_power|rf_power|sinr|rate|omega|eh|eh_utility|objective)\}",
        body,
    )
    if solution_labels:
        issues.append({"issue": "system_model_equation_labels_in_solution_opening", "labels": solution_labels})
    displayed_equations = len(re.findall(r"\\begin\{(?:equation\*?|align\*?|subequations)\}", body))
    if title == "Problem Reformulation" and displayed_equations > 2 and issues:
        issues.append(
            {
                "issue": "problem_reformulation_restates_system_model",
                "displayed_equations": displayed_equations,
            }
        )
    return {"ok": not issues, "issues": issues}


def phase3_reformulation_repetition_issue_summary(tex: str) -> str:
    report = analyze_phase3_reformulation_repeats_system_model(tex)
    if report.get("ok"):
        return ""
    issues = report.get("issues", [])
    details = "; ".join(str(item) for item in issues[:10])
    return (
        "The proposed-solution opening repeats System Model equations instead of introducing only new "
        "reformulation/algorithm material. Remove displayed definitions for received power, SINR, rate, "
        "RF input power, rectifier output, harvested-energy utility, and original objective when those "
        "quantities are already in mathematical_contract_json. Rename `Problem Reformulation` to "
        "`Solution Approach` or `Candidate Evaluation` if no genuine reformulation is introduced. "
        f"Detected issues: {details}"
    )


def latex_equation_format_issue_summary(tex: str) -> str:
    report = analyze_latex_equation_line_format(tex)
    issues = report.get("issues", []) if isinstance(report, dict) else []
    if not issues:
        return ""
    lines = [
        "Repair equation display formatting: each displayed equation/align line must contain at most one primary equality or inequality relation, independent definitions must not be joined under one equation number, ordinary System Model definitions must not use subequations, and problem (P0) must use numbered subequations with labels on the objective and every constraint line."
    ]
    for item in issues[:8]:
        if not isinstance(item, dict):
            continue
        lines.append(
            f"- line {item.get('line')}: {item.get('issue')} in {item.get('environment')}: {item.get('text')}"
        )
    return "\n".join(lines)


def analyze_latex_overfull_boxes(log_text: str, tex_text: str, *, threshold_pt: float = 5.0) -> dict[str, Any]:
    """Extract actionable overfull-box warnings from a LaTeX compile log."""

    source_lines = str(tex_text or "").replace("\r\n", "\n").splitlines()
    issues: list[dict[str, Any]] = []
    pattern = re.compile(
        r"Overfull\s+\\hbox\s+\((?P<amount>-?\d+(?:\.\d+)?)pt\s+too\s+wide\).*?"
        r"(?:at\s+line|in\s+paragraph\s+at\s+lines?)\s+(?P<start>\d+)(?:--(?P<end>\d+))?",
        flags=re.I,
    )
    for match in pattern.finditer(str(log_text or "")):
        try:
            amount = float(match.group("amount"))
        except (TypeError, ValueError):
            amount = 0.0
        if amount < threshold_pt:
            continue
        start = int(match.group("start"))
        end = int(match.group("end") or start)
        first = max(1, start - 2)
        last = min(len(source_lines), end + 2)
        context = [
            f"{line_no}: {source_lines[line_no - 1]}"
            for line_no in range(first, last + 1)
            if 1 <= line_no <= len(source_lines)
        ]
        issues.append(
            {
                "line": start,
                "end_line": end,
                "amount_pt": amount,
                "context": context,
                "log_excerpt": match.group(0).strip(),
            }
        )
    return {"ok": not issues, "issues": issues}


def latex_overfull_issue_summary(log_text: str, tex_text: str) -> str:
    report = analyze_latex_overfull_boxes(log_text, tex_text)
    issues = report.get("issues", []) if isinstance(report, dict) else []
    if not issues:
        return ""
    lines = [
        "Repair PDF overfull hbox warnings after compilation. Rewrite the reported LaTeX so the PDF has no overfull boxes. Prefer splitting long equations with aligned/split/multline, introducing compact shorthands before the display, or replacing oversized expanded vectors/constraints with named compact definitions. Preserve mathematical meaning, labels, and IEEE WCL style; do not use tiny fonts or \\resizebox as the default fix."
    ]
    for item in issues[:8]:
        if not isinstance(item, dict):
            continue
        lines.append(f"- line {item.get('line')}: overfull by {item.get('amount_pt')}pt")
        context = item.get("context")
        if isinstance(context, list) and context:
            lines.append("  Source context:")
            lines.extend(f"  {line}" for line in context[:6])
    return "\n".join(lines)


def _phase21_latex_contract_display(tex: str) -> str:
    source = str(tex or "")
    p0_match = re.search(r"\\text\{\(P0\)\}|\(P0\)|\\mathcal\{P\}_0", source)
    if not p0_match:
        return ""
    begin_candidates = [
        source.rfind(r"\begin{align", 0, p0_match.start()),
        source.rfind(r"\begin{equation", 0, p0_match.start()),
        source.rfind(r"\[", 0, p0_match.start()),
    ]
    begin = max(begin_candidates)
    if begin < 0:
        begin = max(0, p0_match.start() - 160)
    end_candidates = [
        source.find(r"\end{align", p0_match.end()),
        source.find(r"\end{equation", p0_match.end()),
        source.find(r"\]", p0_match.end()),
    ]
    end_candidates = [index for index in end_candidates if index >= 0]
    end = min(end_candidates) if end_candidates else min(len(source), p0_match.end() + 900)
    if end < len(source):
        line_end = source.find("\n", end)
        end = line_end if line_end >= 0 else end
    return source[begin:end]


def _phase21_balanced_brace_body(text: str, start: int) -> str:
    if start >= len(text) or text[start] != "{":
        return ""
    depth = 0
    body_start = start + 1
    for index in range(start, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[body_start:index]
    return ""


def _phase21_latex_optimizer_window(tex: str) -> str:
    display = _phase21_latex_contract_display(tex)
    if not display:
        return ""
    match = re.search(r"\\(?:max|min)\s*_", display)
    if not match:
        return ""
    rest = display[match.end() :].lstrip()
    if rest.startswith("{"):
        return _phase21_balanced_brace_body(rest, 0)
    terminators = [rest.find(token) for token in (r"\quad", "&", r"\\") if rest.find(token) >= 0]
    end = min(terminators) if terminators else min(len(rest), 320)
    return rest[:end]


def _phase21_normalize_math_symbol(text: str) -> str:
    normalized = str(text or "")
    normalized = normalized.split("(", 1)[0]
    normalized = re.sub(r"\\(?:widetilde|tilde)\s*\{([^{}]+)\}", r"tilde_\1", normalized)
    normalized = re.sub(r"\\(?:widetilde|tilde)\s+([A-Za-z]+)", r"tilde_\1", normalized)
    normalized = re.sub(r"\\(?:overline|bar)\s*\{([^{}]+)\}", r"bar_\1", normalized)
    normalized = re.sub(r"\\(?:overline|bar)\s+([A-Za-z]+)", r"bar_\1", normalized)
    normalized = re.sub(r"\\(?:widehat|hat)\s*\{([^{}]+)\}", r"hat_\1", normalized)
    normalized = re.sub(r"\\(?:widehat|hat)\s+([A-Za-z]+)", r"hat_\1", normalized)
    normalized = re.sub(r"\\hat\{([^{}]+)\}", r"hat_\1", normalized)
    previous = None
    while previous != normalized:
        previous = normalized
        normalized = re.sub(
            r"\\(?:mathbf|boldsymbol|mathrm|operatorname|mathcal|mathbb|text)\{([^{}]*)\}",
            r"\1",
            normalized,
        )
        normalized = re.sub(
            r"\\(?:mathbf|boldsymbol|mathrm|operatorname|mathcal|mathbb|text)\s*\\([A-Za-z]+)",
            r"\\\1",
            normalized,
        )
        normalized = re.sub(
            r"\\(?:mathbf|boldsymbol|mathrm|operatorname|mathcal|mathbb|text)\s+([A-Za-z]+)",
            r"\1",
            normalized,
        )
    normalized = normalized.replace(r"\rho", "rho")
    normalized = normalized.replace(r"\Gamma", "Gamma")
    normalized = normalized.replace(r"\Omega", "Omega")
    greek_aliases = {
        r"\alpha": "alpha",
        r"\beta": "beta",
        r"\gamma": "gamma",
        r"\delta": "delta",
        r"\epsilon": "epsilon",
        r"\eta": "eta",
        r"\theta": "theta",
        r"\lambda": "lambda",
        r"\mu": "mu",
        r"\sigma": "sigma",
        r"\omega": "omega",
    }
    for latex_name, ascii_name in greek_aliases.items():
        normalized = normalized.replace(latex_name, ascii_name)
    normalized = re.sub(r"\\[a-zA-Z]+", "", normalized)
    normalized = re.sub(r"[^A-Za-z0-9_^]", "", normalized)
    normalized = normalized.replace("^", "")
    return normalized


def _phase21_split_symbol_list(symbol: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    brace_depth = 0
    for char in str(symbol or ""):
        if char == "{":
            brace_depth += 1
        elif char == "}" and brace_depth > 0:
            brace_depth -= 1
        if char == "," and brace_depth == 0:
            part = "".join(current).strip()
            if part:
                parts.append(part)
            current = []
            continue
        current.append(char)
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def _phase21_contract_symbol_aliases(symbol: str) -> list[str]:
    aliases: list[str] = []
    raw_symbol = str(symbol or "").strip()
    parts = [raw_symbol] if "(" in raw_symbol else _phase21_split_symbol_list(raw_symbol)
    for part in parts:
        part = part.strip()
        if not part:
            continue
        base = part.split("(", 1)[0].strip()
        aliases.append(base)
        if base.startswith("hat_"):
            aliases.append(base.replace("hat_", r"\hat{", 1) + "}")
    deduped: list[str] = []
    for alias in aliases:
        normalized = _phase21_normalize_math_symbol(alias)
        if normalized and normalized not in deduped:
            deduped.append(normalized)
    return deduped


def latex_math_contract_issue_summary(tex: str, mathematical_contract_json: str) -> str:
    contract = _safe_json_loads(mathematical_contract_json, {})
    if not isinstance(contract, dict):
        return ""
    contract = normalize_phase2_phase1_mathematical_contract(contract)
    optimizer_window = _phase21_latex_optimizer_window(tex)
    optimizer_normalized = _phase21_normalize_math_symbol(optimizer_window) if optimizer_window else ""
    full_tex_normalized = _phase21_normalize_math_symbol(tex)
    issue_lines: list[str] = []
    for key in ("derived_quantities", "reformulation_only"):
        entries = contract.get(key)
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            symbol = str(entry.get("symbol") or "").strip()
            for alias in _phase21_contract_symbol_aliases(symbol):
                if optimizer_normalized and alias and alias in optimizer_normalized:
                    status = str(entry.get("status") or key).strip()
                    issue_lines.append(
                        f"- P0 optimizer promotes `{symbol}` ({status}) into the original control set."
                    )
                    break
                if key == "reformulation_only" and alias and alias in full_tex_normalized:
                    issue_lines.append(
                        f"- Reformulation-only symbol `{symbol}` appears in the System Model or Problem Formulation snippet."
                    )
                    break
    geometry_errors, geometry_warnings = _phase2_geometry_convention_issues(contract, tex)
    for item in geometry_errors:
        issue_lines.append(f"- Coordinate-convention conflict: {item}")
    for item in geometry_warnings:
        issue_lines.append(f"- Coordinate-convention ambiguity: {item}")
    if not issue_lines:
        return ""
    header = (
        "Repair mathematical-role consistency: the original P0 optimizer must contain only "
        "controls from mathematical_contract. Derived quantities must remain defined functions, "
        "parameters/random quantities must not be optimizer variables, and reformulation-only auxiliaries "
        "must not appear in the System Model or Problem Formulation."
    )
    return "\n".join([header, *issue_lines[:12]])


def _extract_latex_issue_summary(log_text: str, tex_text: str) -> str:
    issues: list[str] = []
    lower_log = log_text.lower()
    lower_tex = tex_text.lower()

    if "undefined reference" in lower_log or "rerun to get cross-references right" in lower_log:
        issues.append("Remove or simplify fragile cross-references; every \\eqref must target a label defined inside the snippet.")
    if "has been referenced but does not exist" in lower_log:
        issues.append("Do not attach \\label to subsection headings and avoid hyperref destinations that rely on section/subsection anchors.")
    if "\\begin{remark}" in lower_tex:
        issues.append("Replace remark environment with a plain paragraph.")
    if "\\cite{" in lower_tex:
        issues.append("Remove citation commands from the snippet.")
    if "\\tag{" in lower_tex:
        issues.append("Avoid \\tag; use ordinary labeled equations or subequations.")
    if "\\textbf{step" in lower_tex or "\\textbf{step " in lower_tex:
        issues.append("Replace bold inline step labels with natural narrative prose or a compact algorithm environment.")
    if "\\label{sec:" in lower_tex or "\\label{subsec:" in lower_tex:
        issues.append("Do not attach labels to subsection headings.")
    equation_format_summary = latex_equation_format_issue_summary(tex_text)
    if equation_format_summary:
        issues.append(equation_format_summary)
    overfull_summary = latex_overfull_issue_summary(log_text, tex_text)
    if overfull_summary:
        issues.append(overfull_summary)

    return "\n".join(f"- {item}" for item in issues)


def repair_phase2_phase1_latex_llm(
    *,
    run_dir: Path,
    topic: str,
    current_tex: str,
    issue_summary: str,
    model_profile: str,
    repair_mode: str = "formatting_repair",
    mathematical_contract_json: str = "",
) -> str:
    prompt = build_phase2_phase1_latex_repair_prompt(
        topic=topic,
        current_tex=current_tex,
        issue_summary=issue_summary,
        repair_mode=repair_mode,
        mathematical_contract_json=mathematical_contract_json,
    )
    phase_dir = run_dir / "phase2-1"
    write_text(phase_dir / "phase1_latex_repair_prompt.txt", prompt)
    llm = create_llm_client(model_profile)

    thinking = None
    if model_profile == "kimi-k2.6-thinking":
        thinking = {"type": "enabled"}

    response = llm.chat(
        [{"role": "user", "content": prompt}],
        json_mode=True,
        strip_thinking=True,
        thinking=thinking,
        max_tokens=10000,
    )
    write_text(phase_dir / "phase1_latex_repair_raw_response.txt", response.content)
    payload = _safe_json_loads(response.content, {})
    if not isinstance(payload, dict):
        raise ValueError("Phase 2.1 LaTeX repair call did not return a valid structured object")
    return str(payload.get("ieee_wcl_system_model_problem_formulation_tex") or "").strip()


def repair_phase2_phase3_latex_llm(
    *,
    run_dir: Path,
    topic: str,
    mathematical_contract_json: str = "",
    current_tex: str,
    issue_summary: str,
    compile_log_tail: str,
    model_profile: str,
) -> str:
    prompt = build_phase2_phase3_latex_repair_prompt(
        topic=topic,
        mathematical_contract_json=mathematical_contract_json,
        current_tex=current_tex,
        issue_summary=issue_summary,
        compile_log_tail=compile_log_tail,
    )
    phase_dir = run_dir / "phase2-3"
    write_text(phase_dir / "phase3_latex_repair_prompt.txt", prompt)
    llm = create_llm_client(model_profile)

    thinking = None
    if model_profile == "kimi-k2.6-thinking":
        thinking = {"type": "enabled"}

    response = llm.chat(
        [{"role": "user", "content": prompt}],
        json_mode=True,
        strip_thinking=True,
        thinking=thinking,
        max_tokens=9000,
    )
    write_text(phase_dir / "phase3_latex_repair_raw_response.txt", response.content)
    payload = _safe_json_loads(response.content, {})
    if not isinstance(payload, dict):
        raise ValueError("Phase 2.3 LaTeX repair call did not return a valid structured object")
    return str(payload.get("ieee_wcl_proposed_solution_tex") or "").strip()


def _latex_heading_title_from_markdown(text: str) -> str:
    title = re.sub(r"[#`*]+", "", str(text or "")).strip()
    title = re.sub(r"\s+", " ", title)
    title = title.replace("&", r"\&")
    return title or "Algorithm Details"


def phase3_algorithm_markdown_to_latex_snippet(markdown_text: str) -> str:
    """Convert the Phase 2.3 algorithm contract into paper-native LaTeX.

    Phase 2.3 stores the algorithm design as Markdown because later phases and
    code generators consume it as structured evidence. Paper assembly needs a
    LaTeX section, so keep this deterministic bridge instead of relying on a
    brittle optional LLM field.
    """
    source = str(markdown_text or "").replace("\r\n", "\n").strip()
    if not source:
        return ""

    out: list[str] = []
    list_mode: str | None = None

    def close_list() -> None:
        nonlocal list_mode
        if list_mode:
            out.append(rf"\end{{{list_mode}}}")
            list_mode = None

    def open_list(mode: str) -> None:
        nonlocal list_mode
        if list_mode != mode:
            close_list()
            out.append(rf"\begin{{{mode}}}")
            list_mode = mode

    def append_blank() -> None:
        if out and out[-1] != "":
            out.append("")

    for raw_line in source.split("\n"):
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            close_list()
            append_blank()
            continue

        heading_match = re.match(r"^\s*(#{2,6})\s+(.+?)\s*$", stripped)
        if heading_match:
            close_list()
            append_blank()
            title = _latex_heading_title_from_markdown(heading_match.group(2))
            level = len(heading_match.group(1))
            command = "subsection" if level <= 3 else "paragraph"
            out.append(rf"\{command}{{{title}}}")
            continue

        numbered_match = re.match(r"^\s*\d+\.\s+(.+?)\s*$", stripped)
        if numbered_match:
            open_list("enumerate")
            out.append(r"\item " + numbered_match.group(1))
            continue

        bullet_match = re.match(r"^\s*[-*]\s+(.+?)\s*$", stripped)
        if bullet_match:
            open_list("itemize")
            out.append(r"\item " + bullet_match.group(1))
            continue

        close_list()
        out.append(line)

    close_list()
    cleaned = "\n".join(out)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return ensure_phase3_safe_feasibility_certificate(cleaned) + "\n"


def ensure_phase3_safe_feasibility_certificate(tex: str) -> str:
    # The runtime must not inject proof text for a specific robust-beamforming
    # template. Any feasibility certificate must be generated by the current
    # Phase 2.3 contract and then checked by the normal review/repair gates.
    return str(tex or "").strip()


def namespace_phase3_solution_labels(tex: str) -> str:
    """Avoid label collisions when the proposed-solution snippet is assembled after Phase 2.1."""
    cleaned = str(tex or "")
    label_map = {
        "eq:sinr": "eq:solution_sinr",
        "eq:rate": "eq:solution_rate",
        "eq:rfpower": "eq:solution_rf_power",
        "eq:rf_power": "eq:solution_rf_power",
        "eq:eh": "eq:solution_eh",
        "eq:harvested_energy": "eq:solution_eh",
        "alg:proposed": "alg:solution_proposed",
    }
    for old, new in label_map.items():
        cleaned = cleaned.replace(rf"\label{{{old}}}", rf"\label{{{new}}}")
        cleaned = cleaned.replace(rf"\eqref{{{old}}}", rf"\eqref{{{new}}}")
        cleaned = cleaned.replace(rf"\ref{{{old}}}", rf"\ref{{{new}}}")
    return cleaned


def paperize_phase3_internal_terms(tex: str) -> str:
    """Translate workflow-only phase wording into paper-native phrasing."""
    cleaned = str(tex or "")
    replacements = [
        (
            r"\bwe\s+preserve\s+the\s+exact\s+(?:Phase\s*~?\s*2\.1|system-model)\s+metrics\s+and\s+develop\b",
            "we retain the original physical model and develop",
        ),
        (
            r"\bwe\s+preserve\s+the\s+exact\s+(?:Phase\s*~?\s*2\.1|system-model)\s+metrics\b",
            "we retain the original physical model",
        ),
        (
            r"\bpreserve\s+the\s+exact\s+(?:Phase\s*~?\s*2\.1|system-model)\s+metrics\s+and\s+develop\b",
            "retain the original physical model and develop",
        ),
        (
            r"\bpreserves\s+the\s+exact\s+(?:Phase\s*~?\s*2\.1|system-model)\s+metrics\b",
            "retains the original physical model",
        ),
        (
            r"\bpreserve\s+the\s+exact\s+(?:Phase\s*~?\s*2\.1|system-model)\s+metrics\b",
            "retain the original physical model",
        ),
        (
            r"\bevaluate\s+the\s+original\s+physical\s+metrics\s+directly\s+and\s+develop\b",
            "retain the original physical model and develop",
        ),
        (
            r"\bevaluates\s+the\s+original\s+physical\s+metrics\s+directly\b",
            "retains the original physical model",
        ),
        (
            r"\bevaluate\s+the\s+original\s+physical\s+metrics\s+directly\b",
            "retain the original physical model",
        ),
        (r"\bPhase\s*~?\s*2\.1\s+scope\b", "system-model scope"),
        (r"\bPhase\s*~?\s*2\.1\s+metrics\b", "physical quantities"),
        (r"\bPhase\s*~?\s*2\.1\s+physical equations\b", "original physical equations"),
        (r"\bfrom\s+Phase\s*~?\s*2\.1\b", "defined in the system model"),
        (r"\bPhase\s*~?\s*2\.4\s+and\s+Phase\s*~?\s*2\.5\s+experiments\b", "numerical experiments"),
        (r"\bpreserved\s+for\s+Phase\s*~?\s*2\.4\b", "used in the numerical evaluation"),
        (r"\bPhase\s*~?\s*2\.4\s+(?:outputs|data|results|experiments|evidence)\b", "numerical results"),
        (r"\bPhase\s*~?\s*2\.5\s+(?:figures|summary|evidence|registry|outputs)\b", "final numerical evidence"),
        (r"\bPhase\s*~?\s*2\.4\b", "the numerical evaluation"),
        (r"\bPhase\s*~?\s*2\.5\b", "the final numerical evaluation"),
        (r"\bPhase\s*~?\s*4\s+and\s+Phase\s*~?\s*5\s+experiments\b", "numerical experiments"),
        (r"\bPhase\s*~?\s*5\s+experiments\b", "numerical experiments"),
        (r"\bPhase\s*~?\s*4\s+experiments\b", "numerical experiments"),
        (r"\bsystem-model\s+metrics\b", "physical quantities"),
        (r"\boriginal\s+original\s+physical\s+equations\b", "original physical equations"),
    ]
    for pattern, replacement in replacements:
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\*\*([^*\n]+)\*\*", r"\\emph{\1}", cleaned)
    return cleaned


def sanitize_phase3_latex_snippet(tex: str) -> str:
    cleaned = tex.replace("\r\n", "\n").strip()
    if re.search(r"^\s*#{2,6}\s+", cleaned, flags=re.M):
        cleaned = phase3_algorithm_markdown_to_latex_snippet(cleaned).strip()
    cleaned = ensure_phase3_safe_feasibility_certificate(cleaned)
    cleaned = namespace_phase3_solution_labels(cleaned)
    cleaned = paperize_phase3_internal_terms(cleaned)
    # The wrapper already provides the adaptive section title. If the model
    # emits the same top-level section again, strip it deterministically so the
    # preview never shows a duplicated section title.
    cleaned = re.sub(
        r"^\s*\\section\*?\{[^{}]*\}\s*",
        "",
        cleaned,
        count=1,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"^\s*\\label\{sec:proposed_solution\}\s*",
        "",
        cleaned,
        count=1,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\\begin\{algorithm\}\[[^\]]*\]", r"\\begin{algorithm}[!t]", cleaned)
    cleaned = re.sub(r"\\begin\{algorithm\}(?!\[)", r"\\begin{algorithm}[!t]", cleaned)
    cleaned = compact_long_minimizer_lists(cleaned)
    return cleaned.strip() + "\n"


def load_phase3_proposed_solution_snippet(run_dir: Path) -> str:
    phase3_dir = Path(run_dir) / "phase2-3"
    latex_path = phase3_dir / "proposed_solution_ieee_wcl.tex"
    snippet = ""
    if latex_path.exists():
        snippet = sanitize_phase3_latex_snippet(read_text(latex_path))
    if len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", snippet)) >= 80:
        return snippet

    algorithm_md = read_text(phase3_dir / "algorithm.md")
    snippet = sanitize_phase3_latex_snippet(phase3_algorithm_markdown_to_latex_snippet(algorithm_md))
    if len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", snippet)) >= 80:
        write_text(latex_path, snippet)
        return snippet
    return snippet


def _sanitize_phase3_section_title(title: str, fallback: str = "Proposed Method") -> str:
    cleaned = re.sub(r"\\[A-Za-z]+\*?(?:\[[^\]]*\])?", " ", str(title or ""))
    cleaned = re.sub(r"[{}$]", " ", cleaned)
    cleaned = re.sub(r"[^A-Za-z0-9 /:()-]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .,:;-")
    if not cleaned:
        cleaned = fallback
    return cleaned[:80].strip() or fallback


def infer_phase3_section_title(text: str = "", fallback: str = "Proposed Method") -> str:
    lowered = str(text or "").lower()
    algorithm_terms = [
        r"\begin{algorithm}",
        "algorithmic",
        "proposed algorithm",
        "algorithm flow",
        "wmmse",
        "sca",
        "mmse",
        "alternating",
        "iteration",
        "stopping criterion",
        "candidate search",
        "scalar search",
    ]
    if any(term in lowered for term in algorithm_terms):
        return "Proposed Algorithm"
    method_terms = [
        "method design",
        "solution approach",
        "surrogate",
        "reformulation",
        "decomposition",
        "relaxation",
        "closed form",
        "closed-form",
    ]
    if any(term in lowered for term in method_terms):
        return "Proposed Method"
    return _sanitize_phase3_section_title(fallback)


def load_phase3_section_title(run_dir: Path) -> str:
    phase3_1_dir = _phase3_1_technical_dir(run_dir)
    raw_phase3_1_title = read_text(phase3_1_dir / "section_title.txt").strip()
    if raw_phase3_1_title:
        return _sanitize_phase3_section_title(raw_phase3_1_title)
    phase3_dir = Path(run_dir) / "phase2-3"
    raw_explicit_title = read_text(phase3_dir / "section_title.txt").strip()
    if raw_explicit_title:
        return _sanitize_phase3_section_title(raw_explicit_title)
    title_context = "\n".join(
        [
            read_text(phase3_1_dir / "proposed_solution_ieee_wcl.tex"),
            read_text(phase3_dir / "proposed_solution_ieee_wcl.tex"),
            read_text(phase3_dir / "algorithm.md"),
            read_text(phase3_dir / "convergence_or_complexity.md"),
        ]
    )
    return _sanitize_phase3_section_title(infer_phase3_section_title(title_context))


def _phase3_1_technical_dir(run_dir: Path) -> Path:
    return Path(run_dir) / "phase3-1"


def _unwrap_nonproblem_subequations(tex: str) -> str:
    """Reserve subequations for formal optimization problems only.

    Older technical-section preview artifacts sometimes grouped channel or uncertainty
    definitions under `subequations`, which makes ordinary system-model
    equations look like an optimization problem. Keep true problem displays
    intact, but unwrap ordinary grouped definitions into their inner display.
    """

    pattern = re.compile(
        r"\\begin\{subequations\}(?:\s*\\label\{(?P<label>[^{}]+)\})?(?P<body>.*?)\\end\{subequations\}",
        flags=re.S,
    )

    def repl(match: re.Match[str]) -> str:
        body = match.group("body")
        is_formal_problem = bool(
            re.search(
                r"\\(?:max|min)(?![A-Za-z])|\\underset\s*\{[^{}]*\}\s*\{\\(?:max|min)(?![A-Za-z])\}|\\text\{\s*\(?P\d|\\(?:mathrm|text)\{\s*s\.?t\.?\s*\}|\\mathrm\{\s*subject\s+to\s*\}",
                body,
                flags=re.I | re.S,
            )
        )
        if is_formal_problem:
            return match.group(0)
        inner = body.strip()
        if re.search(r"\\begin\{(?:align|aligned|split|equation)\*?\}", inner):
            return inner
        return "\\begin{align}\n" + inner + "\n\\end{align}"

    return pattern.sub(repl, str(tex or ""))


def compact_short_split_equations(tex: str, max_line_chars: int = 150) -> str:
    r"""Undo over-eager align line breaks for compact single-relation displays.

    LLMs often split short IEEE two-column equations as
    ``lhs&=rhs \nonumber\\ &\quad+continuation`` even when the combined
    expression is readable on one line. Keep genuinely long displays split,
    but convert compact one-label, one-relation cases back to equation.
    """

    source = str(tex or "")
    align_pattern = re.compile(r"\\begin\{align\}(?P<body>.*?)\\end\{align\}", flags=re.S)

    def normalize_math_line(line: str) -> str:
        cleaned = re.sub(r"\\(?:nonumber|notag)\b", "", line)
        cleaned = cleaned.replace("&", "")
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        cleaned = cleaned.replace(r"\quad", " ")
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        cleaned = re.sub(r"([+\-])(?=\\|[A-Za-z0-9])", r"\1 ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def repl(match: re.Match[str]) -> str:
        body = match.group("body").strip()
        labels = re.findall(r"\\label\{[^{}]+\}", body)
        if len(labels) != 1 or r"\nonumber" not in body:
            return match.group(0)
        chunks = [chunk.strip() for chunk in re.split(r"\\\\", body) if chunk.strip()]
        math_chunks: list[str] = []
        for chunk in chunks:
            chunk = re.sub(r"\\label\{[^{}]+\}", "", chunk).strip()
            if chunk:
                math_chunks.append(chunk)
        if len(math_chunks) != 2:
            return match.group(0)
        first, second = math_chunks
        if not re.search(r"&\s*(?:=|\\leq|\\geq|\\le\b|\\ge\b|<|>)", first):
            return match.group(0)
        if not re.match(r"&?\s*(?:\\quad\s*)?[+\-]", second):
            return match.group(0)
        combined = normalize_math_line(first) + " " + normalize_math_line(second)
        combined = re.sub(r"\s+([,.;])$", r"\1", combined).strip()
        if len(combined) > max_line_chars:
            return match.group(0)
        return "\\begin{equation}\n" + combined + "\n" + labels[0] + "\n\\end{equation}"

    return align_pattern.sub(repl, source)


def _extract_latex_braced(text: str, open_brace_index: int) -> tuple[str, int] | None:
    if open_brace_index < 0 or open_brace_index >= len(text) or text[open_brace_index] != "{":
        return None
    depth = 0
    for pos in range(open_brace_index, len(text)):
        char = text[pos]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[open_brace_index + 1 : pos], pos + 1
    return None


def compact_long_sinr_fraction_equations(tex: str, min_chars: int = 130) -> str:
    """Rewrite overlong SINR fraction displays using an unnumbered interference shorthand."""

    source = str(tex or "")
    equation_pattern = re.compile(r"\\begin\{equation\}\s*\n(?P<body>[^\n]+?)\s*\n\\end\{equation\}")

    def repl(match: re.Match[str]) -> str:
        body = match.group("body").strip()
        lowered = body.lower()
        if len(body) < min_chars or r"\frac" not in body or r"\sum" not in body:
            return match.group(0)
        if "sinr" not in lowered and r"\gamma" not in body:
            return match.group(0)
        label_match = re.search(r"\\label\{[^{}]*sinr[^{}]*\}", body, flags=re.I)
        if not label_match:
            return match.group(0)
        label = label_match.group(0)
        expr = (body[: label_match.start()] + body[label_match.end() :]).strip()
        punctuation = ""
        if expr and expr[-1] in ".,;":
            punctuation = expr[-1]
            expr = expr[:-1].rstrip()
        if "=" not in expr:
            return match.group(0)
        lhs, rhs = expr.split("=", 1)
        lhs = lhs.strip()
        rhs = rhs.strip()
        if not rhs.startswith(r"\frac"):
            return match.group(0)
        numerator_result = _extract_latex_braced(rhs, rhs.find("{"))
        if not numerator_result:
            return match.group(0)
        numerator, after_num = numerator_result
        denominator_result = _extract_latex_braced(rhs, rhs.find("{", after_num))
        if not denominator_result:
            return match.group(0)
        denominator, after_den = denominator_result
        if rhs[after_den:].strip():
            return match.group(0)
        den_match = re.match(r"(?P<prefix>.*?)\\left\((?P<inner>.*)\\right\)(?P<suffix>.*)$", denominator)
        if not den_match:
            return match.group(0)
        inner = den_match.group("inner").strip()
        if r"\sum" not in inner:
            return match.group(0)
        noise_split = re.search(r"(?P<interference>.+)\+(?P<noise>\\sigma[^+]+)$", inner)
        if noise_split:
            interference = noise_split.group("interference").strip()
            noise = noise_split.group("noise").strip()
        else:
            interference = inner
            noise = ""
        sub_match = re.search(r"\\gamma_\{?([A-Za-z0-9]+)\}?", lhs)
        arg_match = re.search(r"\(([^()]*)\)", lhs)
        if not sub_match or not arg_match:
            return match.group(0)
        subscript = sub_match.group(1)
        argument = arg_match.group(1).strip()
        shorthand = rf"I_{subscript}({argument})"
        if shorthand in source:
            return match.group(0)
        den_inner = shorthand + (f"+{noise}" if noise else "")
        compact_denominator = den_match.group("prefix") + r"\bigl(" + den_inner + r"\bigr)" + den_match.group("suffix")
        shorthand_display = "\n".join(
            [
                r"\begin{equation*}",
                shorthand + r"\triangleq " + interference + ".",
                r"\end{equation*}",
            ]
        )
        sinr_display = "\n".join(
            [
                r"\begin{equation}",
                r"\begin{aligned}",
                lhs,
                r"&=\frac{" + numerator + "}",
                "{" + compact_denominator + "}" + punctuation,
                r"\end{aligned}" + label,
                r"\end{equation}",
            ]
        )
        return shorthand_display + "\n" + sinr_display

    return equation_pattern.sub(repl, source)


def compact_long_minimizer_lists(tex: str, min_chars: int = 72) -> str:
    """Wrap long optimizer-variable lists in substack without changing variables."""
    source = str(tex or "")
    out: list[str] = []
    idx = 0
    changed = False
    token = r"\min_{"
    while True:
        start = source.find(token, idx)
        if start < 0:
            out.append(source[idx:])
            break
        arg_start = start + len(token) - 1
        depth = 0
        end = None
        for pos in range(arg_start, len(source)):
            char = source[pos]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    end = pos
                    break
        if end is None:
            out.append(source[idx:])
            break
        content = source[arg_start + 1 : end]
        replacement = source[start : end + 1]
        if len(content) >= min_chars and "," in content and r"\substack" not in content:
            parts: list[str] = []
            part_start = 0
            inner_depth = 0
            for pos, char in enumerate(content):
                if char == "{":
                    inner_depth += 1
                elif char == "}":
                    inner_depth = max(0, inner_depth - 1)
                elif char == "," and inner_depth == 0:
                    parts.append(content[part_start : pos + 1].strip())
                    part_start = pos + 1
            tail = content[part_start:].strip()
            if tail:
                parts.append(tail)
            if len(parts) >= 3:
                lines: list[str] = []
                current = ""
                target = max(min_chars // 2, 42)
                for part in parts:
                    candidate = (current + " " + part).strip() if current else part.strip()
                    if current and len(candidate) > target and len(lines) < 2:
                        lines.append(current.strip())
                        current = part.strip()
                    else:
                        current = candidate
                if current:
                    lines.append(current.strip())
                if len(lines) >= 2:
                    replacement = r"\min_{\substack{" + r"\\".join(lines) + "}}"
                    changed = True
        out.append(source[idx:start])
        out.append(replacement)
        idx = end + 1
    revised = "".join(out)
    return revised if changed else source


def normalize_pair_index_conditions_in_problem(tex: str) -> str:
    """Move pair-index inequality ranges out of optimization constraint lines."""

    source = str(tex or "")
    pair_range_pattern = re.compile(
        r",(?:\s|\\\s*)*1\s*\\leq?\s*n\s*<\s*m\s*\\leq?\s*N",
        flags=re.I,
    )
    if not pair_range_pattern.search(source):
        return source
    replacement = r",\quad \forall (n,m)\in\mathcal{E}"
    revised = pair_range_pattern.sub(lambda _match: replacement, source)
    if r"\mathcal{E}" in revised and r"\mathcal{E}\triangleq" not in revised:
        definition = (
            r"Let $\mathcal{E}\triangleq\{(n,m):1\le n<m\le N\}$ denote the antenna-pair index set."
            "\n"
        )
        problem_pos = revised.find(r"\begin{subequations}")
        if problem_pos >= 0:
            revised = revised[:problem_pos] + definition + revised[problem_pos:]
    return revised


def sanitize_phase3_1_system_problem_snippet(tex: str) -> str:
    cleaned = str(tex or "").strip()
    cleaned = re.sub(r"\\documentclass(?:\[[^\]]*\])?\{[^{}]+\}", "", cleaned)
    cleaned = re.sub(r"\\usepackage(?:\[[^\]]*\])?\{[^{}]+\}", "", cleaned)
    cleaned = re.sub(r"\\begin\{document\}|\\end\{document\}", "", cleaned)
    cleaned = _unwrap_nonproblem_subequations(cleaned)
    cleaned = compact_long_sinr_fraction_equations(cleaned)
    cleaned = re.sub(
        r"^\s*\\section\*?\{(?:System Model|System Model and Problem Formulation|Problem Formulation)[^{}]*\}\s*",
        "",
        cleaned,
        count=1,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"^\s*\\label\{sec:(?:system_model|problem_formulation)\}\s*",
        "",
        cleaned,
        count=1,
        flags=re.IGNORECASE,
    )
    cleaned = compact_short_split_equations(cleaned)
    cleaned = normalize_pair_index_conditions_in_problem(cleaned)
    # LLMs sometimes emit a single trailing backslash after an align-row label,
    # e.g., ``\label{eq:x}\``. TeX then merges the next displayed line into the
    # same align row, causing "Multiple \label's" compile failures. Promote only
    # line-ending single slashes to a proper row break; leave valid ``\\`` alone.
    cleaned = re.sub(r"(\\label\{[^{}]+\})\\[ \t]*(?=\n)", r"\1\\\\", cleaned)
    return cleaned.strip() + "\n"


def load_phase3_1_system_model_problem_snippet(run_dir: Path) -> str:
    phase3_1_path = _phase3_1_technical_dir(run_dir) / "system_model_problem_formulation_ieee_wcl.tex"
    snippet = sanitize_phase3_1_system_problem_snippet(read_text(phase3_1_path))
    if len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", snippet)) >= 80:
        return snippet
    legacy_path = Path(run_dir) / "phase2-1" / "system_model_problem_formulation_ieee_wcl.tex"
    return sanitize_phase3_1_system_problem_snippet(read_text(legacy_path))


def load_phase3_1_proposed_solution_snippet(run_dir: Path) -> str:
    phase3_1_path = _phase3_1_technical_dir(run_dir) / "proposed_solution_ieee_wcl.tex"
    snippet = sanitize_phase3_latex_snippet(read_text(phase3_1_path))
    if len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", snippet)) >= 80:
        return snippet
    return load_phase3_proposed_solution_snippet(run_dir)


def render_phase3_1_technical_preview_pdf(phase_dir: Path) -> dict[str, str]:
    wrapper_tex = phase_dir / "phase3_1_technical_preview.tex"
    system_path = phase_dir / "system_model_problem_formulation_ieee_wcl.tex"
    proposed_path = phase_dir / "proposed_solution_ieee_wcl.tex"
    write_text(system_path, sanitize_phase3_1_system_problem_snippet(read_text(system_path)))
    write_text(proposed_path, sanitize_phase3_latex_snippet(read_text(proposed_path)))
    section_title = _sanitize_phase3_section_title(read_text(phase_dir / "section_title.txt") or "Proposed Method")
    wrapper_tex_content = r"""\documentclass[journal]{IEEEtran}
\usepackage{amsmath,amssymb,amsfonts,bm,mathtools}
\usepackage[hidelinks]{hyperref}
\usepackage{algorithm}
\usepackage{algpseudocode}

\begin{document}
\title{Phase 3.1 Technical Preview}
\author{}
\maketitle

\section{System Model and Problem Formulation}
\label{sec:system_model}
\input{system_model_problem_formulation_ieee_wcl.tex}

\section{__PROPOSED_SECTION_TITLE__}
\label{sec:proposed_solution}
\input{proposed_solution_ieee_wcl.tex}

\end{document}
""".replace("__PROPOSED_SECTION_TITLE__", section_title)
    write_text(wrapper_tex, wrapper_tex_content)

    for suffix in (".aux", ".out", ".toc", ".lof", ".lot"):
        stale = phase_dir / f"phase3_1_technical_preview{suffix}"
        if stale.exists():
            stale.unlink()

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

    pdf_path = phase_dir / "phase3_1_technical_preview.pdf"
    log_path = phase_dir / "phase3_1_technical_preview.log"
    if not pdf_path.exists():
        stderr_text = ""
        if last_result is not None:
            stderr_text = f"\nSTDOUT:\n{last_result.stdout}\nSTDERR:\n{last_result.stderr}"
        raise RuntimeError(f"Phase 3.1 PDF preview was not generated.{stderr_text}")
    return {
        "preview_tex": str(wrapper_tex),
        "preview_pdf": str(pdf_path),
        "preview_log": str(log_path),
    }


def render_phase3_ieee_preview_pdf(phase_dir: Path) -> dict[str, str]:
    wrapper_tex = phase_dir / "phase3_ieee_preview.tex"
    snippet_path = phase_dir / "proposed_solution_ieee_wcl.tex"
    if snippet_path.exists():
        write_text(snippet_path, sanitize_phase3_latex_snippet(read_text(snippet_path)))
    elif (phase_dir / "algorithm.md").exists():
        write_text(snippet_path, load_phase3_proposed_solution_snippet(phase_dir.parent))
    section_title = load_phase3_section_title(phase_dir.parent)
    wrapper_tex_content = r"""\documentclass[10pt,conference]{IEEEtran}
\usepackage{amsmath,amssymb,amsfonts,bm,mathtools}
\usepackage[hidelinks]{hyperref}
\usepackage{algorithm}
\usepackage{algpseudocode}

\begin{document}
\title{Phase 2.3 Preview}
\author{}
\maketitle

\section{__PHASE3_SECTION_TITLE__}
\input{proposed_solution_ieee_wcl.tex}

\end{document}
""".replace("__PHASE3_SECTION_TITLE__", section_title)
    write_text(wrapper_tex, wrapper_tex_content)

    for suffix in (".aux", ".out", ".toc", ".lof", ".lot"):
        stale = phase_dir / f"phase3_ieee_preview{suffix}"
        if stale.exists():
            stale.unlink()

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

    pdf_path = phase_dir / "phase3_ieee_preview.pdf"
    log_path = phase_dir / "phase3_ieee_preview.log"
    if not pdf_path.exists():
        stderr_text = ""
        if last_result is not None:
            stderr_text = f"\nSTDOUT:\n{last_result.stdout}\nSTDERR:\n{last_result.stderr}"
        raise RuntimeError(f"Phase 2.3 PDF preview was not generated.{stderr_text}")
    return {
        "preview_tex": str(wrapper_tex),
        "preview_pdf": str(pdf_path),
        "preview_log": str(log_path),
    }


PHASE21_MATH_CONTRACT_KEYS = [
    "controls",
    "parameters",
    "random_quantities",
    "derived_quantities",
    "objective",
    "constraints",
    "reformulation_only",
    "notation_to_preserve",
]


def _phase21_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value in (None, ""):
        return []
    return [value]


def _phase21_entry(raw: Any, status: str, *, appears_in_optimizer: bool = False) -> dict[str, Any]:
    entry = dict(raw) if isinstance(raw, dict) else {"symbol": str(raw or "").strip()}
    entry["status"] = status
    entry["appears_in_optimizer"] = bool(appears_in_optimizer)
    entry["allowed_in_optimizer"] = bool(appears_in_optimizer)
    if not entry.get("meaning"):
        entry["meaning"] = str(entry.get("description") or "").strip()
    return entry


def _phase21_constraint(raw: Any, index: int) -> dict[str, Any]:
    entry = dict(raw) if isinstance(raw, dict) else {"relation": str(raw or "").strip()}
    entry.setdefault("id", str(entry.get("name") or entry.get("symbol") or f"c{index}"))
    entry.setdefault("relation", str(entry.get("expression") or entry.get("symbol") or "").strip())
    entry.setdefault("meaning", str(entry.get("description") or entry.get("name") or "").strip())
    uses = entry.get("uses_symbols")
    if not isinstance(uses, list):
        entry["uses_symbols"] = []
    entry["appears_in_optimizer"] = False
    return entry


def _phase21_entry_aliases(entry: dict[str, Any]) -> set[str]:
    aliases: set[str] = set()
    if not isinstance(entry, dict):
        return aliases
    for key in ("symbol", "definition", "relation"):
        for alias in _phase21_contract_symbol_aliases(str(entry.get(key) or "")):
            if alias:
                aliases.add(alias)
    return aliases


def _phase21_symbol_aliases(entry: dict[str, Any]) -> set[str]:
    aliases: set[str] = set()
    if not isinstance(entry, dict):
        return aliases
    symbol_text = str(entry.get("symbol") or "")
    for alias in _phase21_contract_symbol_aliases(symbol_text):
        if alias:
            aliases.add(alias)
    function_head = symbol_text.split("(", 1)[0].strip()
    for alias in _phase21_contract_symbol_aliases(function_head):
        if alias:
            aliases.add(alias)
    return aliases


def _phase21_tokens_from_values(*values: Any) -> set[str]:
    tokens: set[str] = set()
    for value in values:
        if isinstance(value, list):
            iterable = value
        else:
            iterable = [value]
        for item in iterable:
            for alias in _phase21_contract_symbol_aliases(str(item or "")):
                if alias:
                    tokens.add(alias)
            normalized = _phase21_normalize_math_symbol(str(item or ""))
            if normalized:
                tokens.add(normalized)
    return tokens


def _phase21_control_dependent_symbols(contract: dict[str, Any]) -> set[str]:
    control_aliases = {
        alias
        for entry in contract.get("controls", [])
        if isinstance(entry, dict)
        for alias in _phase21_contract_symbol_aliases(str(entry.get("symbol") or ""))
        if alias
    }
    dependent_aliases = set(control_aliases)
    derived_entries = [entry for entry in contract.get("derived_quantities", []) if isinstance(entry, dict)]
    changed = True
    while changed:
        changed = False
        for entry in derived_entries:
            entry_aliases = _phase21_symbol_aliases(entry)
            if not entry_aliases:
                entry_aliases = _phase21_entry_aliases(entry)
            symbol_arguments = re.findall(r"\(([^)]*)\)", str(entry.get("symbol") or ""))
            dependency_tokens = _phase21_tokens_from_values(entry.get("depends_on"), entry.get("definition"), symbol_arguments)
            if _phase21_alias_sets_overlap(dependency_tokens, dependent_aliases) and not entry_aliases <= dependent_aliases:
                dependent_aliases.update(entry_aliases)
                changed = True
    return dependent_aliases


def _phase21_alias_sets_overlap(left: set[str], right: set[str]) -> bool:
    for a in left:
        for b in right:
            if not a or not b:
                continue
            if a == b:
                return True
            shorter = a if len(a) <= len(b) else b
            longer = b if len(a) <= len(b) else a
            if longer.startswith(f"{shorter}_"):
                return True
            if len(shorter) >= 3 and "_" in shorter and shorter in longer:
                return True
    return False


def _phase21_constraint_depends_on_control(contract: dict[str, Any], constraint: dict[str, Any]) -> bool:
    dependent_aliases = _phase21_control_dependent_symbols(contract)
    constraint_tokens = _phase21_tokens_from_values(
        constraint.get("uses_symbols"),
        constraint.get("relation"),
    )
    if _phase21_alias_sets_overlap(constraint_tokens, dependent_aliases):
        return True
    normalized_relation = _phase21_normalize_math_symbol(str(constraint.get("relation") or ""))
    return any(alias and alias in normalized_relation for alias in dependent_aliases)


def _phase21_objective(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        objective = dict(raw)
    elif isinstance(raw, list):
        objective = {"sense": "", "expression": "", "meaning": "", "terms": raw}
    else:
        objective = {"sense": "", "expression": str(raw or "").strip(), "meaning": "", "terms": []}
    sense = str(objective.get("sense") or objective.get("objective_sense") or "").strip().lower()
    if sense in {"maximize", "maximization"}:
        sense = "max"
    elif sense in {"minimize", "minimization"}:
        sense = "min"
    objective["sense"] = sense
    terms = objective.get("terms")
    normalized_terms: list[dict[str, Any]] = []
    for term in _phase21_list(terms):
        if isinstance(term, dict):
            item = dict(term)
        else:
            item = {"expression": str(term or "").strip()}
        if not isinstance(item.get("uses_symbols"), list):
            item["uses_symbols"] = []
        normalized_terms.append(item)
    objective["terms"] = normalized_terms
    objective.setdefault("expression", "")
    objective.setdefault("meaning", "")
    objective["appears_in_optimizer"] = False
    return objective


def _phase21_normalize_notation_entry(raw: Any) -> dict[str, Any]:
    entry = dict(raw) if isinstance(raw, dict) else {"canonical_symbol": str(raw or "").strip()}
    if not entry.get("canonical_symbol") and entry.get("symbol"):
        entry["canonical_symbol"] = str(entry.get("symbol") or "").strip()
    aliases = entry.get("aliases_forbidden")
    if not isinstance(aliases, list):
        entry["aliases_forbidden"] = []
    entry.pop("appears_in_optimizer", None)
    return entry


def _phase21_convert_legacy_contract(raw: dict[str, Any]) -> dict[str, Any]:
    objective_terms = _phase21_list(raw.get("objective_terms"))
    expression = ""
    if objective_terms:
        fragments = []
        for term in objective_terms:
            if isinstance(term, dict):
                fragments.append(str(term.get("symbol") or term.get("expression") or "").strip())
            else:
                fragments.append(str(term or "").strip())
        expression = " + ".join(fragment for fragment in fragments if fragment)
    return {
        "controls": raw.get("control_variables", []),
        "parameters": raw.get("constraint_parameters", []),
        "random_quantities": raw.get("random_quantities", []),
        "derived_quantities": raw.get("derived_kpis", []),
        "objective": {
            "sense": raw.get("objective_sense", ""),
            "expression": expression,
            "terms": objective_terms,
        },
        "constraints": raw.get("original_constraints", []),
        "reformulation_only": list(_phase21_list(raw.get("reformulation_only_auxiliaries")))
        + list(_phase21_list(raw.get("forbidden_role_promotions"))),
        "notation_to_preserve": raw.get("notation_to_preserve", []),
    }


def normalize_phase2_phase1_mathematical_contract(raw: Any) -> dict[str, Any]:
    if isinstance(raw, str):
        parsed = _safe_json_loads(raw, {})
        raw = parsed if isinstance(parsed, dict) else {}
    if not isinstance(raw, dict):
        raw = {}
    if not any(key in raw for key in PHASE21_MATH_CONTRACT_KEYS) and any(
        key in raw
        for key in (
            "control_variables",
            "derived_kpis",
            "constraint_parameters",
            "objective_terms",
            "original_constraints",
            "reformulation_only_auxiliaries",
            "forbidden_role_promotions",
        )
    ):
        raw = _phase21_convert_legacy_contract(raw)
    elif any(
        key in raw
        for key in (
            "control_variables",
            "derived_kpis",
            "constraint_parameters",
            "objective_terms",
            "original_constraints",
            "reformulation_only_auxiliaries",
            "forbidden_role_promotions",
        )
    ):
        legacy = _phase21_convert_legacy_contract(raw)
        raw = {**legacy, **raw}
    contract: dict[str, Any] = {}
    contract["controls"] = [
        _phase21_entry(entry, "control", appears_in_optimizer=True)
        for entry in _phase21_list(raw.get("controls"))
    ]
    contract["parameters"] = [
        _phase21_entry(entry, "parameter", appears_in_optimizer=False)
        for entry in _phase21_list(raw.get("parameters"))
    ]
    contract["random_quantities"] = [
        _phase21_entry(entry, "random_quantity", appears_in_optimizer=False)
        for entry in _phase21_list(raw.get("random_quantities"))
    ]
    contract["derived_quantities"] = [
        _phase21_entry(entry, "derived_quantity", appears_in_optimizer=False)
        for entry in _phase21_list(raw.get("derived_quantities"))
    ]
    contract["objective"] = _phase21_objective(raw.get("objective"))
    contract["constraints"] = [
        _phase21_constraint(entry, index)
        for index, entry in enumerate(_phase21_list(raw.get("constraints")), start=1)
    ]
    contract["reformulation_only"] = [
        _phase21_entry(entry, "reformulation_only", appears_in_optimizer=False)
        for entry in _phase21_list(raw.get("reformulation_only"))
    ]
    for entry in contract["reformulation_only"]:
        entry["allowed_in_original_problem"] = False
    control_aliases = {
        alias
        for entry in contract["controls"]
        if isinstance(entry, dict)
        for alias in _phase21_contract_symbol_aliases(str(entry.get("symbol") or ""))
        if alias
    }
    if control_aliases:
        for key in ("derived_quantities", "reformulation_only"):
            cleaned_entries: list[dict[str, Any]] = []
            for entry in contract[key]:
                if not isinstance(entry, dict):
                    continue
                aliases = _phase21_contract_symbol_aliases(str(entry.get("symbol") or ""))
                if any(alias in control_aliases for alias in aliases):
                    continue
                cleaned_entries.append(entry)
            contract[key] = cleaned_entries
    contract["notation_to_preserve"] = [
        _phase21_normalize_notation_entry(entry)
        for entry in _phase21_list(raw.get("notation_to_preserve"))
    ]
    return contract


def validate_phase2_phase1_mathematical_contract_schema(contract: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_phase2_phase1_mathematical_contract(contract)
    errors: list[str] = []
    warnings: list[str] = []
    if not normalized.get("controls"):
        errors.append("mathematical_contract.controls is empty.")
    objective = normalized.get("objective")
    if not isinstance(objective, dict) or not str(objective.get("expression") or "").strip():
        errors.append("mathematical_contract.objective.expression is missing.")
    elif str(objective.get("sense") or "").strip().lower() not in {"max", "min"}:
        warnings.append("mathematical_contract.objective.sense should be `max` or `min`.")
    if not normalized.get("constraints"):
        errors.append("mathematical_contract.constraints is empty.")
    control_aliases = {
        alias
        for entry in normalized.get("controls", [])
        if isinstance(entry, dict)
        for alias in _phase21_contract_symbol_aliases(str(entry.get("symbol") or ""))
    }
    for key in ("derived_quantities", "reformulation_only", "parameters", "random_quantities"):
        for entry in normalized.get(key, []):
            if not isinstance(entry, dict):
                continue
            symbol = str(entry.get("symbol") or "").strip()
            if key in {"derived_quantities", "reformulation_only"}:
                for alias in _phase21_contract_symbol_aliases(symbol):
                    if alias and alias in control_aliases:
                        errors.append(f"{key} entry `{symbol}` duplicates a control symbol.")
                        break
            if bool(entry.get("appears_in_optimizer")) or bool(entry.get("allowed_in_optimizer")):
                errors.append(f"{key} entry `{symbol}` is incorrectly marked as optimizer-allowed.")
    parameter_only_constraints = [
        str(item.get("id") or item.get("relation") or "").strip()
        for item in normalized.get("constraints", [])
        if isinstance(item, dict) and not _phase21_constraint_depends_on_control(normalized, item)
    ]
    if parameter_only_constraints:
        errors.append(
            "mathematical_contract.constraints contains parameter-domain or model-admissibility conditions that do not restrict any control variable: "
            + ", ".join(parameter_only_constraints)
            + ". Move near-field validity, positive-noise, weight-domain, geometry-domain, and other parameter assumptions to system_model_md or core_theory_package_md prose, not to the formal constraints of problem (P0)."
        )
    geometry_errors, geometry_warnings = _phase2_geometry_convention_issues(normalized)
    errors.extend(geometry_errors)
    warnings.extend(geometry_warnings)
    return {"ok": not errors, "errors": errors, "warnings": warnings, "normalized_contract": normalized}


def phase2_phase1_mathematical_contract_json(contract: dict[str, Any] | None) -> str:
    return json.dumps(normalize_phase2_phase1_mathematical_contract(contract or {}), ensure_ascii=False, indent=2)


def _phase21_max_tokens() -> int:
    for env_name in ("WARA_PHASE21_MAX_TOKENS", "WCL_PHASE21_MAX_TOKENS"):
        raw_value = os.environ.get(env_name)
        if raw_value is None:
            continue
        try:
            return max(8000, int(raw_value))
        except ValueError:
            continue
    return 24000


def _phase21_compact_json_retry_prompt(base_prompt: str, *, last_error: str) -> str:
    return (
        base_prompt
        + "\n\nCritical compact-JSON retry: the previous Phase 2.1 response could not be parsed as a complete structured object.\n"
        + f"Parser error summary: {last_error}\n\n"
        + "Return the full JSON object again with exactly these top-level keys only: "
        + "mathematical_contract, system_model_md, problem_formulation_md, core_theory_package_md.\n"
        + "This retry must be compact enough to finish within one response: "
        + "use short strings, no Markdown tables, no long tutorials, no repeated equations, and no bibliography-style background. "
        + "Keep each markdown field under roughly 250 words. "
        + "For core_theory_package_md, use at most six bullets covering modeling choice, tractability route, proof scope, implementation obligation, benchmark contrast, and experiment observable. "
        + "Do not omit required mathematical_contract fields, but keep each entry concise. "
        + "Return JSON only; do not wrap it in Markdown fences."
    )


def _phase21_schema_retry_prompt(base_prompt: str, *, last_error: str, round_no: int, round_limit: int) -> str:
    return (
        base_prompt
        + f"\n\nCritical schema retry round {round_no}/{round_limit}: the previous Phase 2.1 response had an invalid mathematical_contract.\n"
        + f"Schema error(s): {last_error}\n\n"
        + "Return the full JSON object again with exactly the same required top-level keys. "
        + "Repair the mathematical_contract lifecycle roles rather than removing the research mechanism. "
        + "Every symbol must have exactly one status: control, parameter, random_quantity, derived_quantity, or reformulation_only. "
        + "Controls are physical decision variables in the original problem; derived quantities are functions; reformulation-only symbols are later algorithmic auxiliaries. "
        + "mathematical_contract.constraints must contain only relations that restrict the physical controls directly or through derived quantities that depend on controls. "
        + "Move pure parameter-domain/model-validity assumptions such as near-field distance validity, positive noise power, nonnegative weights, fixed geometry, or nonzero constants into system_model_md/core_theory_package_md prose, not into constraints. "
        + "Do not duplicate the same symbol across statuses. Decorated auxiliary symbols such as tilde/hat/bar variants are allowed only when their role is clearly different from the original physical control. "
        + "Keep optimizer variables limited to controls and keep the response compact and parseable."
    )


def _phase21_result_gate_score(result: dict[str, Any], *, contract_error: str = "") -> float:
    """Score only real FormulationAgent candidates; never synthesize content."""

    score = 50.0
    for key, cap in (
        ("system_model_md", 15.0),
        ("problem_formulation_md", 20.0),
        ("core_theory_package_md", 10.0),
    ):
        words = len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", str(result.get(key) or "")))
        score += min(cap, words / 12.0)
    contract = result.get("mathematical_contract")
    if isinstance(contract, dict):
        for key in ("controls", "parameters", "derived_quantities", "constraints"):
            if contract.get(key):
                score += 3.0
    if contract_error:
        score -= min(25.0, 5.0 + len(str(contract_error)) / 120.0)
    else:
        score += 30.0
    return max(0.0, score)


def _phase22_result_gate_score(result: dict[str, str], *, contract_error: str = "") -> float:
    """Score real TheoryAgent tractability candidates without replacement text."""

    score = 45.0
    for key, cap in (("convexity_audit_md", 25.0), ("reformulation_path_md", 30.0)):
        text = str(result.get(key) or "")
        words = len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", text))
        score += min(cap, words / 16.0)
    combined = f"{result.get('convexity_audit_md', '')}\n{result.get('reformulation_path_md', '')}".lower()
    for marker in ("nonconvex", "relax", "approx", "subproblem", "feasible", "objective", "constraint"):
        if marker in combined:
            score += 2.0
    if contract_error:
        score -= min(35.0, 8.0 + len(str(contract_error)) / 100.0)
    else:
        score += 35.0
    return max(0.0, score)


def _phase23_result_gate_score(result: dict[str, Any], *, contract_error: str = "") -> float:
    """Score real TheoryAgent algorithm candidates without replacement text."""

    score = 45.0
    for key, cap in (("algorithm_md", 30.0), ("convergence_or_complexity_md", 20.0)):
        text = str(result.get(key) or "")
        words = len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", text))
        score += min(cap, words / 16.0)
    combined = f"{result.get('algorithm_md', '')}\n{result.get('convergence_or_complexity_md', '')}".lower()
    for marker in ("input", "output", "iteration", "update", "constraint", "objective", "complexity", "stopping"):
        if marker in combined:
            score += 2.0
    if contract_error:
        score -= min(35.0, 8.0 + len(str(contract_error)) / 100.0)
    else:
        score += 35.0
    return max(0.0, score)


def run_phase2_phase1_llm(
    *,
    run_dir: Path,
    topic: str,
    handoff: dict[str, Any],
    topic_taxonomy: dict[str, Any],
    synthesis_md: str,
    model_profile: str,
) -> dict[str, Any]:
    prompt = build_phase2_phase1_prompt(topic, handoff, topic_taxonomy, synthesis_md)
    phase_dir = run_dir / "phase2-1"
    write_text(phase_dir / "phase1_prompt.txt", prompt)

    llm = create_llm_client(model_profile)

    thinking = None
    if model_profile == "kimi-k2.6-thinking":
        thinking = {"type": "enabled"}

    repair_round_limit = phase2_contract_repair_round_limit()
    attempts: list[tuple[str, str, dict[str, str] | None, int]] = [("primary", prompt, thinking, 0)]
    if thinking is not None:
        attempts.append(("non_thinking_retry", prompt, None, 0))

    last_error = ""
    llm_candidates: list[dict[str, Any]] = []
    attempt_index = 0
    while attempt_index < len(attempts):
        attempt_name, attempt_prompt, attempt_thinking, repair_round = attempts[attempt_index]
        attempt_index += 1
        try:
            response = llm.chat(
                [{"role": "user", "content": attempt_prompt}],
                json_mode=True,
                strip_thinking=True,
                thinking=attempt_thinking,
                max_tokens=_phase21_max_tokens(),
            )
        except Exception as exc:
            last_error = f"{attempt_name} model call failed: {exc}"
            write_text(phase_dir / "phase1_generation_errors.txt", last_error)
            raise
        write_text(phase_dir / f"phase1_raw_response_{attempt_name}.txt", response.content)
        write_text(phase_dir / "phase1_raw_response.txt", response.content)
        if not str(response.content or "").strip():
            last_error = f"{attempt_name} returned an empty response"
            continue
        payload = _safe_json_loads(response.content, {})
        if not isinstance(payload, dict) or not payload:
            last_error = f"{attempt_name} did not return a valid structured object"
            if not any(name.startswith("compact_json_retry") for name, _, _, _ in attempts):
                retry_prompt = _phase21_compact_json_retry_prompt(prompt, last_error=last_error)
                write_text(phase_dir / "phase1_compact_json_retry_prompt.txt", retry_prompt)
                attempts.append(("compact_json_retry", retry_prompt, None, 0))
            continue
        mathematical_contract = normalize_phase2_phase1_mathematical_contract(payload.get("mathematical_contract"))
        contract_schema_report = validate_phase2_phase1_mathematical_contract_schema(mathematical_contract)
        write_text(
            phase_dir / f"phase1_mathematical_contract_schema_report_{attempt_name}.json",
            json.dumps(contract_schema_report, ensure_ascii=False, indent=2),
        )
        if not contract_schema_report.get("ok", False):
            last_error = (
                f"{attempt_name} did not return a usable mathematical_contract: "
                + "; ".join(str(item) for item in contract_schema_report.get("errors", []))
            )
            existing_schema_retries = max(
                [round_no for name, _, _, round_no in attempts if name.startswith("schema_retry")]
                or [0]
            )
            if repair_round < repair_round_limit and existing_schema_retries < repair_round_limit:
                next_round = existing_schema_retries + 1
                retry_prompt = _phase21_schema_retry_prompt(
                    prompt,
                    last_error=last_error,
                    round_no=next_round,
                    round_limit=repair_round_limit,
                )
                write_text(phase_dir / f"phase1_schema_retry_prompt_round{next_round}.txt", retry_prompt)
                write_text(phase_dir / "phase1_schema_retry_prompt.txt", retry_prompt)
                attempts.append((f"schema_retry_round{next_round}", retry_prompt, None, next_round))
            continue
        mathematical_contract = contract_schema_report["normalized_contract"]
        result = {
            "mathematical_contract": mathematical_contract,
            "mathematical_contract_json": phase2_phase1_mathematical_contract_json(mathematical_contract),
            "system_model_md": str(payload.get("system_model_md") or "").strip(),
            "problem_formulation_md": str(payload.get("problem_formulation_md") or "").strip(),
            "core_theory_package_md": str(payload.get("core_theory_package_md") or "").strip(),
        }
        try:
            validate_phase2_phase1_contract(
                run_dir=run_dir,
                topic=topic,
                handoff=handoff,
                mathematical_contract=mathematical_contract,
                system_model_md=result["system_model_md"],
                problem_formulation_md=result["problem_formulation_md"],
                core_theory_package_md=result["core_theory_package_md"],
            )
        except ValueError as exc:
            last_error = f"{attempt_name} failed contract: {exc}"
            llm_candidates.append(
                {
                    "attempt": attempt_name,
                    "repair_round": repair_round,
                    "score": _phase21_result_gate_score(result, contract_error=str(exc)),
                    "contract_error": str(exc),
                    "result": result,
                }
            )
            existing_contract_retries = max(
                [round_no for name, _, _, round_no in attempts if name.startswith("contract_retry")]
                or [0]
            )
            if repair_round < repair_round_limit and existing_contract_retries < repair_round_limit:
                next_round = existing_contract_retries + 1
                retry_prompt = (
                    prompt
                    + f"\n\nCritical retry round {next_round}/{repair_round_limit}: the previous Phase 2.1 output failed the technical contract gate.\n"
                    + f"Gate error(s): {exc}\n\n"
                    + "Return the full JSON object again with exactly the same required keys. "
                    + "Respect the Phase 1 declared variables as a hard scope contract. "
                    + "Do not add receiver classes, beam families, hardware blocks, protocol phases, RIS phases, "
                    + "sensing/radar variables, or any other physical mechanism absent from Phase 1. "
                    + "If the model has mobility, deployment, movable elements, distances, or path loss, choose one coordinate convention and keep every distance expression dimensionally consistent. "
                    + "Use horizontal coordinates plus altitude, or full-dimensional coordinates with embedded lower-dimensional nodes, but not both. "
                    + "If a movement, propulsion, service-cost, or hardware-cost function appears, define its expression or solver-relevant mathematical properties. "
                    + "Keep mathematical_contract.constraints limited to true restrictions on the physical controls; do not put parameter-domain/model-validity assumptions such as near-field validity, positive noise, nonnegative weights, or fixed geometry into formal P0 constraints. "
                    + "If robust/chance/ambiguity language is used, choose and define a concrete uncertainty model and distinct symbols for outage tolerance versus uncertainty size. "
                    + "If the current artifacts contain matrix-valued transmit controls or auxiliary/common/service signals, state the physical signaling/recovery interpretation and the receiver decoding/cancellation assumption. "
                    + "If the objective is transmitter-resource minimization, tie it to nontrivial QoS/reliability/service constraints and paper-facing physical KPIs. "
                    + "For independent streams, waveforms, or random quantities already declared by Phase 1, "
                    + "use expectation- or covariance-based expressions instead of coherent sums unless the "
                    + "handoff explicitly justifies coherent combining."
                )
                write_text(phase_dir / f"phase1_contract_retry_prompt_round{next_round}.txt", retry_prompt)
                write_text(phase_dir / "phase1_contract_retry_prompt.txt", retry_prompt)
                attempts.append((f"contract_retry_round{next_round}", retry_prompt, None, next_round))
            continue
        llm_candidates.append(
            {
                "attempt": attempt_name,
                "repair_round": repair_round,
                "score": _phase21_result_gate_score(result),
                "contract_error": "",
                "result": result,
            }
        )
        return result

    if llm_candidates:
        selected = max(llm_candidates, key=lambda item: float(item.get("score") or 0.0))
        write_text(
            phase_dir / "phase21_selected_llm_candidate_after_repair_budget.json",
            json.dumps(
                {
                    "selection_policy": "highest_gate_score_among_llm_candidates",
                    "last_error": last_error,
                    "selected_attempt": selected.get("attempt"),
                    "selected_score": selected.get("score"),
                    "candidate_count": len(llm_candidates),
                    "candidates": [
                        {
                            "attempt": item.get("attempt"),
                            "repair_round": item.get("repair_round"),
                            "score": item.get("score"),
                            "contract_error": item.get("contract_error"),
                        }
                        for item in llm_candidates
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
        )
        selected_result = dict(selected["result"])
        selected_result["selected_after_repair_budget"] = True
        selected_result["selection_policy"] = "highest_gate_score_among_llm_candidates"
        selected_result["selection_score"] = selected.get("score")
        selected_result["selection_contract_error"] = selected.get("contract_error")
        return selected_result
    raise ValueError("Phase 2.1 failed to produce a usable structured handoff: " + last_error)


def run_phase2_phase1_latex_llm(
    *,
    run_dir: Path,
    topic: str,
    system_model_md: str,
    problem_formulation_md: str,
    core_theory_package_md: str,
    model_profile: str,
    mathematical_contract_json: str = "",
) -> str:
    prompt = build_phase2_phase1_latex_prompt(
        topic=topic,
        mathematical_contract_json=mathematical_contract_json,
        system_model_md=system_model_md,
        problem_formulation_md=problem_formulation_md,
        core_theory_package_md=core_theory_package_md,
    )
    phase_dir = run_dir / "phase2-1"
    write_text(phase_dir / "phase1_latex_prompt.txt", prompt)
    llm = create_llm_client(model_profile)

    thinking = None
    if model_profile == "kimi-k2.6-thinking":
        thinking = {"type": "enabled"}

    response = llm.chat(
        [{"role": "user", "content": prompt}],
        json_mode=True,
        strip_thinking=True,
        thinking=thinking,
        max_tokens=10000,
    )
    write_text(phase_dir / "phase1_latex_raw_response.txt", response.content)
    payload = _safe_json_loads(response.content, {})
    if not isinstance(payload, dict):
        raise ValueError("Phase 2.1 LaTeX call did not return a valid structured object")
    latex_text = str(payload.get("ieee_wcl_system_model_problem_formulation_tex") or "").strip()
    equation_format_report = analyze_latex_equation_line_format(latex_text)
    write_text(
        phase_dir / "phase1_equation_format_report.json",
        json.dumps(equation_format_report, ensure_ascii=False, indent=2),
    )
    if not equation_format_report.get("ok", False):
        issue_summary = latex_equation_format_issue_summary(latex_text)
        write_text(phase_dir / "phase1_latex_equation_format_issue_summary.txt", issue_summary)
        latex_text = repair_phase2_phase1_latex_llm(
            run_dir=run_dir,
            topic=topic,
            current_tex=latex_text,
            issue_summary=issue_summary,
            model_profile=model_profile,
            repair_mode="formatting_repair",
            mathematical_contract_json=mathematical_contract_json,
        )
        repaired_report = analyze_latex_equation_line_format(latex_text)
        write_text(
            phase_dir / "phase1_equation_format_report_repaired.json",
            json.dumps(repaired_report, ensure_ascii=False, indent=2),
        )
        if not repaired_report.get("ok", False):
            raise ValueError(
                "Phase 2.1 LaTeX violated equation display formatting rules after repair: "
                + latex_equation_format_issue_summary(latex_text)
            )
    semantic_issue_summary = latex_math_contract_issue_summary(latex_text, mathematical_contract_json)
    write_text(phase_dir / "phase1_latex_semantic_consistency_report.txt", semantic_issue_summary or "ok")
    if semantic_issue_summary:
        latex_text = repair_phase2_phase1_latex_llm(
            run_dir=run_dir,
            topic=topic,
            current_tex=latex_text,
            issue_summary=semantic_issue_summary,
            model_profile=model_profile,
            repair_mode="semantic_consistency_repair",
            mathematical_contract_json=mathematical_contract_json,
        )
        semantic_repair_format_report = analyze_latex_equation_line_format(latex_text)
        write_text(
            phase_dir / "phase1_equation_format_report_semantic_repaired.json",
            json.dumps(semantic_repair_format_report, ensure_ascii=False, indent=2),
        )
        if not semantic_repair_format_report.get("ok", False):
            raise ValueError(
                "Phase 2.1 semantic repair introduced equation display formatting issues: "
                + latex_equation_format_issue_summary(latex_text)
            )
        semantic_after_repair = latex_math_contract_issue_summary(latex_text, mathematical_contract_json)
        write_text(
            phase_dir / "phase1_latex_semantic_consistency_report_repaired.txt",
            semantic_after_repair or "ok",
        )
        if semantic_after_repair:
            raise ValueError(
                "Phase 2.1 LaTeX still violates the mathematical_contract after semantic repair: "
                + semantic_after_repair
            )
    return latex_text


def run_phase2_phase2_llm(
    *,
    run_dir: Path,
    topic: str,
    handoff: dict[str, Any],
    mathematical_contract_json: str = "",
    system_model_md: str,
    problem_formulation_md: str,
    core_theory_package_md: str,
    model_profile: str,
    tractability_route_policy: dict[str, Any] | None = None,
) -> dict[str, str]:
    prompt = build_phase2_phase2_prompt(
        topic=topic,
        handoff=handoff,
        mathematical_contract_json=mathematical_contract_json,
        system_model_md=system_model_md,
        problem_formulation_md=problem_formulation_md,
        core_theory_package_md=core_theory_package_md,
        tractability_route_policy=tractability_route_policy,
    )
    phase_dir = run_dir / "phase2-2"
    write_text(phase_dir / "phase2_prompt.txt", prompt)

    llm = create_llm_client(model_profile)

    thinking = None
    if model_profile == "kimi-k2.6-thinking":
        thinking = {"type": "enabled"}

    def _parse_phase22_payload(response_text: str) -> dict[str, str]:
        payload = _safe_json_loads(response_text, {})
        if not isinstance(payload, dict):
            raise ValueError("Phase 2.2 did not return a valid structured object")
        result_payload = {
            "convexity_audit_md": str(payload.get("convexity_audit_md") or "").strip(),
            "reformulation_path_md": str(payload.get("reformulation_path_md") or "").strip(),
        }
        if not result_payload["convexity_audit_md"] or not result_payload["reformulation_path_md"]:
            raise ValueError("Phase 2.2 structured object is missing convexity_audit_md or reformulation_path_md")
        return result_payload

    def _call_and_parse(phase_prompt: str, raw_name: str, phase_thinking: dict[str, str] | None) -> dict[str, str]:
        response = llm.chat(
            [{"role": "user", "content": phase_prompt}],
            json_mode=True,
            strip_thinking=True,
            thinking=phase_thinking,
            max_tokens=_phase2_max_tokens("WARA_PHASE22_MAX_TOKENS", 18000),
        )
        write_text(phase_dir / raw_name, response.content)
        write_text(
            phase_dir / raw_name.replace(".txt", "_metadata.json"),
            json.dumps(
                {
                    "model": response.model,
                    "finish_reason": response.finish_reason,
                    "truncated": response.truncated,
                    "prompt_tokens": response.prompt_tokens,
                    "completion_tokens": response.completion_tokens,
                    "total_tokens": response.total_tokens,
                    "response_chars": len(response.content),
                },
                ensure_ascii=False,
                indent=2,
            ),
        )
        return _parse_phase22_payload(response.content)

    generation_errors: list[str] = []
    llm_candidates: list[dict[str, Any]] = []
    try:
        result = _call_and_parse(prompt, "phase2_raw_response.txt", thinking)
    except Exception as exc:
        generation_errors.append(f"primary: {exc}")
        primary_raw = read_text(phase_dir / "phase2_raw_response.txt")
        compact_retry_prompt = (
            "Phase 2.2 structured retry. The previous response was malformed or truncated.\n\n"
            "Return valid JSON only with exactly these keys:\n"
            "- convexity_audit_md: concise Markdown, 700-1300 words\n"
            "- reformulation_path_md: concise Markdown, 700-1300 words\n\n"
            "Do not include markdown fences around the JSON. Escape all newlines as JSON string content.\n"
            "Keep the route technically closed: name the tractability class, scope any approximation, state solver-relevant subproblem form, "
            "and explain what is provable versus empirical. Do not invent new original variables or change the frozen objective.\n\n"
            "Original Phase 2.2 request excerpt:\n"
            + compact_text(prompt, int(os.environ.get("WARA_PHASE22_RETRY_PROMPT_CHARS", "14000") or 14000))
            + "\n\nPrevious malformed response excerpt:\n"
            + compact_text(primary_raw, 3000)
        )
        write_text(phase_dir / "phase2_compact_structured_retry_prompt.txt", compact_retry_prompt)
        try:
            result = _call_and_parse(compact_retry_prompt, "phase2_raw_response_compact_structured_retry.txt", None)
        except Exception as retry_exc:
            generation_errors.append(f"compact_structured_retry: {retry_exc}")
            write_text(phase_dir / "phase2_generation_errors.txt", "\n".join(generation_errors))
            raise ValueError("Phase 2.2 did not return a valid structured object: " + " | ".join(generation_errors)) from retry_exc
    repair_round_limit = phase2_contract_repair_round_limit()
    last_error: Exception | None = None
    for repair_round in range(0, repair_round_limit + 1):
        try:
            validate_phase2_phase2_contract(
                run_dir=run_dir,
                convexity_audit_md=result["convexity_audit_md"],
                reformulation_path_md=result["reformulation_path_md"],
            )
            llm_candidates.append(
                {
                    "attempt": "initial" if repair_round == 0 else f"repair_round_{repair_round}",
                    "repair_round": repair_round,
                    "score": _phase22_result_gate_score(result),
                    "contract_error": "",
                    "result": result,
                }
            )
            return result
        except ValueError as exc:
            last_error = exc
            llm_candidates.append(
                {
                    "attempt": "initial" if repair_round == 0 else f"repair_round_{repair_round}",
                    "repair_round": repair_round,
                    "score": _phase22_result_gate_score(result, contract_error=str(exc)),
                    "contract_error": str(exc),
                    "result": result,
                }
            )
            if repair_round >= repair_round_limit:
                break
            next_round = repair_round + 1
            write_text(phase_dir / f"phase2_retry_reason_round{next_round}.txt", str(exc))
            write_text(phase_dir / "phase2_retry_reason.txt", str(exc))
            retry_prompt = (
                prompt
                + f"\n\nSuccess-route retry round {next_round}/{repair_round_limit}: the previous Phase 2.2 route was not technically closed.\n"
                + f"Gate feedback: {exc}\n\n"
                + "Return the full JSON object again with exactly the same required keys. "
                + "Do not merely add warnings. Select a concrete tractability route that can become a real paper method: "
                + "define the exact uncertainty/ambiguity model if robustness is used, write the deterministic counterpart or scoped subproblem, "
                + "preserve any coordinate convention from Phase 2.1 when distances, path loss, mobility, deployment, or movable elements are present, "
                + "state the expression or solver-relevant properties for any propulsion, movement, service-cost, or hardware-cost term, "
                + "identify valid variables before claiming any perspective, epigraph, or time-sharing convexity route, "
                + "state physical-realization and receiver-information assumptions when the current artifacts require them, and map the route to an implementable Phase 2.3 algorithm plus Phase 2.4 physical KPI. "
                + "If the current Phase 2.1 model is too under-specified to support a theorem, explicitly route to a scoped heuristic/empirical method rather than writing a black-box Phi-style counterpart."
            )
            write_text(phase_dir / f"phase2_retry_prompt_round{next_round}.txt", retry_prompt)
            write_text(phase_dir / "phase2_retry_prompt.txt", retry_prompt)
            try:
                result = _call_and_parse(retry_prompt, f"phase2_raw_response_retry_round{next_round}.txt", None)
            except Exception as retry_exc:
                last_error = retry_exc
                break

    if last_error is not None:
        if llm_candidates:
            selected = max(llm_candidates, key=lambda item: float(item.get("score") or 0.0))
            write_text(
                phase_dir / "phase22_selected_llm_candidate_after_repair_budget.json",
                json.dumps(
                    {
                        "selection_policy": "highest_gate_score_among_llm_candidates",
                        "last_error": str(last_error),
                        "selected_attempt": selected.get("attempt"),
                        "selected_score": selected.get("score"),
                        "candidate_count": len(llm_candidates),
                        "candidates": [
                            {
                                "attempt": item.get("attempt"),
                                "repair_round": item.get("repair_round"),
                                "score": item.get("score"),
                                "contract_error": item.get("contract_error"),
                            }
                            for item in llm_candidates
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            )
            selected_result = dict(selected["result"])
            selected_result["selected_after_repair_budget"] = True
            selected_result["selection_policy"] = "highest_gate_score_among_llm_candidates"
            selected_result["selection_score"] = selected.get("score")
            selected_result["selection_contract_error"] = selected.get("contract_error")
            return selected_result
        raise last_error
    return result


def run_phase2_phase3_llm(
    *,
    run_dir: Path,
    topic: str,
    handoff: dict[str, Any],
    mathematical_contract_json: str = "",
    system_model_md: str,
    problem_formulation_md: str,
    core_theory_package_md: str,
    convexity_audit_md: str,
    reformulation_path_md: str,
    model_profile: str,
    tractability_route_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prompt = build_phase2_phase3_prompt(
        topic=topic,
        handoff=handoff,
        mathematical_contract_json=mathematical_contract_json,
        system_model_md=system_model_md,
        problem_formulation_md=problem_formulation_md,
        core_theory_package_md=core_theory_package_md,
        convexity_audit_md=convexity_audit_md,
        reformulation_path_md=reformulation_path_md,
        tractability_route_policy=tractability_route_policy,
    )
    phase_dir = run_dir / "phase2-3"
    write_text(phase_dir / "phase3_prompt.txt", prompt)

    llm = create_llm_client(model_profile)

    thinking = None
    if model_profile == "kimi-k2.6-thinking":
        thinking = {"type": "enabled"}

    def _call_and_parse(phase_prompt: str, raw_name: str) -> dict[str, Any]:
        response = llm.chat(
            [{"role": "user", "content": phase_prompt}],
            json_mode=True,
            strip_thinking=True,
            thinking=thinking,
            max_tokens=10000,
        )
        write_text(phase_dir / raw_name, response.content)
        payload = _safe_json_loads(response.content, {})
        if not isinstance(payload, dict):
            raise ValueError("Phase 2.3 did not return a valid structured object")
        deferred_benchmark = (
            "# Deferred to Phase 2.4\n\n"
            "Benchmark, ablation, validation-principle, sweep, metric, and figure design are owned by Phase 2.4. "
            "Phase 2.3 only defines the proposed algorithm and any scoped convergence or complexity discussion."
        )
        deferred_experiments = (
            "# Deferred to Phase 2.4\n\n"
            "Experiment design is intentionally not generated by Phase 2.3. Phase 2.4 must derive the experiment contract "
            "from the frozen mathematical interface, the system/problem section, the reformulation path, and the Phase 2.3 proposed algorithm."
        )
        return {
            "algorithm_md": str(payload.get("algorithm_md") or "").strip(),
            "convergence_or_complexity_md": str(payload.get("convergence_or_complexity_md") or "").strip(),
            "benchmark_definition_md": deferred_benchmark,
            "validation_principles_md": deferred_experiments,
            "experiment_blueprint_md": deferred_experiments,
        }

    def _validate(result: dict[str, Any]) -> None:
        validate_phase2_phase3_contract(
            run_dir=run_dir,
            algorithm_md=result["algorithm_md"],
            convergence_or_complexity_md=result["convergence_or_complexity_md"],
            experiment_blueprint_md=result["experiment_blueprint_md"],
        )

    try:
        result = _call_and_parse(prompt, "phase3_raw_response.txt")
    except Exception:
        raise
    repair_round_limit = phase2_contract_repair_round_limit()
    llm_candidates: list[dict[str, Any]] = []
    last_error: Exception | None = None
    for repair_round in range(0, repair_round_limit + 1):
        try:
            _validate(result)
            llm_candidates.append(
                {
                    "attempt": "initial" if repair_round == 0 else f"repair_round_{repair_round}",
                    "repair_round": repair_round,
                    "score": _phase23_result_gate_score(result),
                    "contract_error": "",
                    "result": result,
                }
            )
            return result
        except ValueError as exc:
            last_error = exc
            llm_candidates.append(
                {
                    "attempt": "initial" if repair_round == 0 else f"repair_round_{repair_round}",
                    "repair_round": repair_round,
                    "score": _phase23_result_gate_score(result, contract_error=str(exc)),
                    "contract_error": str(exc),
                    "result": result,
                }
            )
            if repair_round == 0:
                initial_report_path = _phase2_contract_report_path(run_dir, "phase2-3")
                if initial_report_path.exists():
                    write_text(phase_dir / "technical_contract_gate_initial.json", read_text(initial_report_path))
            if repair_round >= repair_round_limit:
                break
            next_round = repair_round + 1
            write_text(phase_dir / f"phase3_retry_reason_round{next_round}.txt", str(exc))
            write_text(phase_dir / "phase3_retry_reason.txt", str(exc))
            retry_prompt = (
                prompt
                + f"\n\nCritical retry round {next_round}/{repair_round_limit}: the previous Phase 2.3 output failed the technical contract gate.\n"
                + f"Gate error(s): {exc}\n\n"
                + "Return the full JSON object again with exactly the same required keys. "
                + "Keep the retry aligned with the dynamic guardrail and the frozen mathematical contract. "
                + "If a convexity or convergence proof is incomplete, describe the corresponding block as "
                + "a scoped surrogate/proxy or heuristic update and rely on empirical feasibility/objective "
                + "diagnostics. Scope any KKT, monotonicity, stationarity, lifting, relaxation, or "
                + "randomization claim to the proof obligations actually satisfied. Do not introduce "
                + "physical entities, metrics, baselines, or decision variables absent from Phase 2.1."
            )
            write_text(phase_dir / f"phase3_retry_prompt_round{next_round}.txt", retry_prompt)
            write_text(phase_dir / "phase3_retry_prompt.txt", retry_prompt)
            try:
                result = _call_and_parse(retry_prompt, f"phase3_raw_response_retry_round{next_round}.txt")
            except Exception as retry_exc:
                last_error = retry_exc
                break
    if last_error is not None:
        if llm_candidates:
            selected = max(llm_candidates, key=lambda item: float(item.get("score") or 0.0))
            write_text(
                phase_dir / "phase23_selected_llm_candidate_after_repair_budget.json",
                json.dumps(
                    {
                        "selection_policy": "highest_gate_score_among_llm_candidates",
                        "last_error": str(last_error),
                        "selected_attempt": selected.get("attempt"),
                        "selected_score": selected.get("score"),
                        "candidate_count": len(llm_candidates),
                        "candidates": [
                            {
                                "attempt": item.get("attempt"),
                                "repair_round": item.get("repair_round"),
                                "score": item.get("score"),
                                "contract_error": item.get("contract_error"),
                            }
                            for item in llm_candidates
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            )
            selected_result = dict(selected["result"])
            selected_result["selected_after_repair_budget"] = True
            selected_result["selection_policy"] = "highest_gate_score_among_llm_candidates"
            selected_result["selection_score"] = selected.get("score")
            selected_result["selection_contract_error"] = selected.get("contract_error")
            return selected_result
        raise last_error
    return result


def run_phase2_phase3_latex_llm(
    *,
    run_dir: Path,
    topic: str,
    mathematical_contract_json: str = "",
    algorithm_md: str,
    convergence_or_complexity_md: str,
    benchmark_definition_md: str,
    model_profile: str,
) -> str:
    llm = create_llm_client(model_profile)
    prompt = build_phase2_phase3_latex_prompt(
        topic=topic,
        mathematical_contract_json=mathematical_contract_json,
        algorithm_md=algorithm_md,
        convergence_or_complexity_md=convergence_or_complexity_md,
        benchmark_definition_md=benchmark_definition_md,
    )
    phase_dir = run_dir / "phase2-3"
    write_text(phase_dir / "phase3_latex_prompt.txt", prompt)

    thinking = None
    if model_profile == "kimi-k2.6-thinking":
        thinking = {"type": "enabled"}

    response = llm.chat(
        [{"role": "user", "content": prompt}],
        json_mode=True,
        strip_thinking=True,
        thinking=thinking,
        max_tokens=9000,
    )
    write_text(phase_dir / "phase3_latex_raw_response.txt", response.content)
    payload = _safe_json_loads(response.content, {})
    if not isinstance(payload, dict):
        raise ValueError("Phase 2.3 LaTeX call did not return a valid structured object")
    latex_text = str(payload.get("ieee_wcl_proposed_solution_tex") or "").strip()
    equation_format_report = analyze_latex_equation_line_format(latex_text)
    write_text(
        phase_dir / "phase3_equation_format_report.json",
        json.dumps(equation_format_report, ensure_ascii=False, indent=2),
    )
    if not equation_format_report.get("ok", False):
        issue_summary = latex_equation_format_issue_summary(latex_text)
        write_text(phase_dir / "phase3_latex_equation_format_issue_summary.txt", issue_summary)
        repaired_latex_text = repair_phase2_phase3_latex_llm(
            run_dir=run_dir,
            topic=topic,
            mathematical_contract_json=mathematical_contract_json,
            current_tex=latex_text,
            issue_summary=issue_summary,
            compile_log_tail="",
            model_profile=model_profile,
        )
        repaired_report = analyze_latex_equation_line_format(repaired_latex_text)
        write_text(
            phase_dir / "phase3_equation_format_report_repaired.json",
            json.dumps(repaired_report, ensure_ascii=False, indent=2),
        )
        if not repaired_report.get("ok", False):
            raise ValueError(
                "Phase 2.3 LaTeX violated equation display formatting rules after repair: "
                + latex_equation_format_issue_summary(repaired_latex_text)
            )
        latex_text = repaired_latex_text
    repetition_report = analyze_phase3_reformulation_repeats_system_model(latex_text)
    write_text(
        phase_dir / "phase3_reformulation_repetition_report.json",
        json.dumps(repetition_report, ensure_ascii=False, indent=2),
    )
    if not repetition_report.get("ok", False):
        issue_summary = phase3_reformulation_repetition_issue_summary(latex_text)
        write_text(phase_dir / "phase3_reformulation_repetition_issue_summary.txt", issue_summary)
        repaired_latex_text = repair_phase2_phase3_latex_llm(
            run_dir=run_dir,
            topic=topic,
            mathematical_contract_json=mathematical_contract_json,
            current_tex=latex_text,
            issue_summary=issue_summary,
            compile_log_tail="",
            model_profile=model_profile,
        )
        repaired_repetition_report = analyze_phase3_reformulation_repeats_system_model(repaired_latex_text)
        write_text(
            phase_dir / "phase3_reformulation_repetition_report_repaired.json",
            json.dumps(repaired_repetition_report, ensure_ascii=False, indent=2),
        )
        if not repaired_repetition_report.get("ok", False):
            raise ValueError(
                "Phase 2.3 LaTeX repeated System Model equations after repair: "
                + phase3_reformulation_repetition_issue_summary(repaired_latex_text)
            )
        return repaired_latex_text
    return latex_text


def run_phase3_1_writing_llm(
    *,
    run_dir: Path,
    topic: str,
    mathematical_contract_json: str = "",
    system_model_md: str,
    problem_formulation_md: str,
    core_theory_package_md: str,
    convexity_audit_md: str,
    reformulation_path_md: str,
    algorithm_md: str,
    convergence_or_complexity_md: str,
    benchmark_definition_md: str,
    model_profile: str,
) -> dict[str, str]:
    llm = create_llm_client(model_profile)
    prompt = build_phase3_1_writing_prompt(
        topic=topic,
        mathematical_contract_json=mathematical_contract_json,
        system_model_md=system_model_md,
        problem_formulation_md=problem_formulation_md,
        core_theory_package_md=core_theory_package_md,
        convexity_audit_md=convexity_audit_md,
        reformulation_path_md=reformulation_path_md,
        algorithm_md=algorithm_md,
        convergence_or_complexity_md=convergence_or_complexity_md,
        benchmark_definition_md=benchmark_definition_md,
    )
    phase_dir = run_dir / "phase3-1"
    write_text(phase_dir / "phase3_1_prompt.txt", prompt)
    thinking = {"type": "enabled"} if model_profile == "kimi-k2.6-thinking" else None
    try:
        response = llm.chat(
            [{"role": "user", "content": prompt}],
            json_mode=True,
            strip_thinking=True,
            thinking=thinking,
            max_tokens=12000,
        )
    except Exception as exc:  # noqa: BLE001
        write_text(phase_dir / "phase3_1_generation_error.txt", f"{type(exc).__name__}: {exc}\n")
        raise
    write_text(phase_dir / "phase3_1_raw_response.txt", response.content)
    payload = _safe_json_loads(response.content, {})
    if not isinstance(payload, dict):
        raise ValueError("Phase 3.1 technical writing call did not return a valid structured object")
    system_tex = sanitize_phase3_1_system_problem_snippet(str(payload.get("system_model_problem_formulation_tex") or ""))
    proposed_tex = sanitize_phase3_latex_snippet(str(payload.get("proposed_solution_tex") or ""))
    section_title = _sanitize_phase3_section_title(str(payload.get("proposed_section_title") or infer_phase3_section_title(proposed_tex)))
    if len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", system_tex)) < 120:
        raise ValueError("Phase 3.1 system/problem writing output is too short")
    if len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", proposed_tex)) < 120:
        raise ValueError("Phase 3.1 proposed-method writing output is too short")
    return {
        "system_model_problem_formulation_tex": system_tex,
        "proposed_solution_tex": proposed_tex,
        "proposed_section_title": section_title,
    }


def repair_phase3_1_latex_llm(
    *,
    run_dir: Path,
    topic: str,
    mathematical_contract_json: str = "",
    current_system_model_problem_formulation_tex: str,
    current_proposed_solution_tex: str,
    issue_summary: str,
    compile_log_tail: str,
    model_profile: str,
) -> dict[str, str]:
    llm = create_llm_client(model_profile)
    prompt = build_phase3_1_writing_repair_prompt(
        topic=topic,
        mathematical_contract_json=mathematical_contract_json,
        issue_summary=issue_summary,
        compile_log_tail=compile_log_tail,
        current_system_model_problem_formulation_tex=current_system_model_problem_formulation_tex,
        current_proposed_solution_tex=current_proposed_solution_tex,
    )
    phase_dir = run_dir / "phase3-1"
    write_text(phase_dir / "phase3_1_latex_repair_prompt.txt", prompt)
    thinking = {"type": "enabled"} if model_profile == "kimi-k2.6-thinking" else None
    response = llm.chat(
        [{"role": "user", "content": prompt}],
        json_mode=True,
        strip_thinking=True,
        thinking=thinking,
        max_tokens=12000,
    )
    write_text(phase_dir / "phase3_1_latex_repair_raw_response.txt", response.content)
    payload = _safe_json_loads(response.content, {})
    if not isinstance(payload, dict):
        raise ValueError("Phase 3.1 repair call did not return a valid structured object")
    system_tex = sanitize_phase3_1_system_problem_snippet(str(payload.get("system_model_problem_formulation_tex") or ""))
    proposed_tex = sanitize_phase3_latex_snippet(str(payload.get("proposed_solution_tex") or ""))
    section_title = _sanitize_phase3_section_title(str(payload.get("proposed_section_title") or infer_phase3_section_title(proposed_tex)))
    return {
        "system_model_problem_formulation_tex": system_tex,
        "proposed_solution_tex": proposed_tex,
        "proposed_section_title": section_title,
    }


def build_phase3_design_notes() -> str:
    return """
# Phase 2.3 Design Notes

## Phase 2.3 mission

Phase 2.3 is the theoretical derivation and proposed-algorithm phase.

It is responsible for:
- reformulation analysis
- proposed algorithm design
- optional convergence or complexity discussion

## What Phase 2.3 should output

Phase 2.3 should produce:
- `algorithm.md`
- `convergence_or_complexity.md`
- compatibility stubs for older readers: `benchmark_definition.md`, `validation_principles.md`, and `experiment_blueprint.md`

## What Phase 2.3 should not decide

Phase 2.3 should not decide:
- baselines or ablations
- empirical claims
- experiment metrics
- sweep axes or values
- the final WCL figure plan
- the final 2-3 figures
- `paper_sweep_plan.json`
- chart type selection
- Monte Carlo seed counts for paper-ready figures
- final x-axis point counts
- paper-ready sufficiency rules

All experiment design and evidence-contract decisions belong to Phase 2.4. Phase 2.5 only checks, aggregates, and renders data from the Phase 2.4 contract.
""".strip()


def build_phase3_4_design_notes() -> str:
    return """
# Phase 3.4 Design Notes

## Phase 3.4 mission

phase3.4_introduction_reference_phase is the introduction, reference-verification, and full-paper-preview assembly phase.

It:
- separates Phase 1 literature sources from Phase 2 technical sources and Phase 2/3 result sources
- turns the Phase 1 bibliography pool into a verified reference bank
- replaces arXiv leads with formally published citations when high-confidence matches are found
- generates a claim-mapped IEEE WCL-style introduction
- writes a final references.bib with only verified or justified references
- compiles a bibliography-aware full paper preview PDF

phase3.4_introduction_reference_phase does not:
- redesign the algorithm
- change experiments or numerical results
- invent new contributions beyond prior phases
- fabricate venue, DOI, or page metadata

## Default outputs

The default Phase 3.4 package produces:
- phase3_4_source_map.json
- introduction_facts.json
- verified_reference_bank.json
- introduction.tex
- references.bib
- citation_claim_map.json
- reference_quality_report.json
- source_usage_report.md
- a full-paper preview PDF that combines abstract, introduction, technical sections, conclusion, and bibliography
""".strip()


def build_pipeline_experiment_design_notes() -> str:
    return """
# Pipeline Experiment Design

## Phase 2.3: derivation and proposed algorithm

Phase 2.3 is responsible for:
- theoretical reformulation
- proposed algorithm design
- convergence or complexity discussion, if available

Phase 2.3 is not responsible for:
- empirical claims
- benchmark or ablation design
- metric selection
- sweep axes or values
- the final 2-3 figures
- chart type decisions
- `paper_sweep_plan.json`
- paper-ready Monte Carlo sample sizes
- final x-axis point counts
- paper-ready sufficiency rules

## Phase 2.4: experiment contract and quick validation

Phase 2.4 is the only phase that designs executable experiments. It translates the frozen math/problem/theory interface into `generated_plugin.py` and runs a deterministic harness in quick mode.

Its job is to define:
- paper claims to test or falsify
- mandatory compared methods, including proposed, practical baselines, ablations, and justified oracle/reference diagnostics
- canonical configuration fields and sweep axes
- required physical KPI columns, feasibility diagnostics, and actual-used sweep diagnostics
- 2-3 figure evidence candidates and no required table target
- missing-experiment behavior when quick validation is not paper-sufficient

It also verifies:
- the plugin can run
- proposed and baseline both return structured outputs
- objectives, constraints, diagnostics, and serialization work
- draft validation results can be produced
- the executable validation plan follows the Phase 2.4 evidence contract rather than an accidental plotting template

Phase 2.4 quick results are not final paper figures.

## Phase 2.5: experiment planner and sufficiency checker

Phase 2.5 combines:
- Phase 2.4 evidence contract
- Phase 2.4 quick results or paper-sweep results requested by Phase 2.5
- the paper target

Phase 2.5 chooses rendering and compact reporting for the Phase 2.4 predeclared evidence:
- 2-3 figures
- no required table in the current WARA route
- chart types
- primary metric display and method labels

Phase 2.5 then checks whether the data are sufficient for a paper-ready package. If not, it generates `paper_sweep_plan.json`.

Phase 2.5 must not invent a new empirical claim, new benchmark story, or new parameter factor after seeing weak data. If the data fail the Phase 2.4 evidence contract, the correct output is a missing-experiment or claim-failure report.

## Paper-sweep executor

The paper-sweep executor runs `paper_sweep_plan.json` after Phase 2.5 requests denser runs.

It:
- reuses the same fixed harness
- reuses the same `generated_plugin.py`
- does not redesign the algorithm
- does not redesign the figures
- writes paper validation csv/json artifacts

## Phase 2.5 rerun

After paper-sweep execution, Phase 2.5 reruns and prefers the paper validation outputs over the quick validation outputs.

Only this rerun may produce paper-ready figures.

## Phase 3.3: technical sections assembly

Phase 3.3 assembles the current technical paper core:
- Phase 2.1 system model and problem formulation
- Phase 2.3 proposed solution
- Phase 3.2 numerical results

Phase 3.3 also generates:
- abstract
- conclusion

Phase 3.3 does not yet complete the full paper. Introduction and references are outside its current scope.

## Phase 3.4: introduction and references

phase3.4_introduction_reference_phase continues from the Phase 3.3 technical package.

It:
- inherits Phase 1 literature sources as a candidate related-work and bibliography pool
- inherits Phase 2 technical sources as the exact problem/method/benchmark source
- inherits Phase 2/3 result sources as the naming and claim-strength source
- verifies and replaces references before building the final bibliography
- writes the introduction from structured source facts rather than a mixed raw prompt
- compiles a bibliography-aware full paper preview

phase3.4_introduction_reference_phase does not redesign algorithms, experiments, or figures.

## Default paper standard

Quick mode:
- at least 20 seeds per point
- 3 to 5 x points for line-style drafts
- draft only

Paper minimum:
- 80 seeds per point
- at least 10 x points for line plots
- minimum acceptance floor

Paper preferred:
- 100 seeds per point
- about 12 x points for line plots
- 5 to 6 representative categories for grouped-bar plots
- default target

High confidence:
- 100 seeds per point
- about 15 x points for line plots

Quick mode never produces paper-ready figures.
""".strip()
