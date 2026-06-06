from __future__ import annotations

import re
from typing import Any

from phase_runtime.prompt_templates import render_prompt_template


def _lower_join(*parts: Any) -> str:
    return "\n".join(str(part or "") for part in parts).lower()


def _term_present(lowered: str, term: str) -> bool:
    term = term.lower()
    if re.fullmatch(r"[a-z0-9_]+", term):
        return re.search(rf"(?<![a-z0-9_]){re.escape(term)}(?![a-z0-9_])", lowered) is not None
    return term in lowered


def _has_any(lowered: str, terms: list[str]) -> bool:
    return any(_term_present(lowered, term) for term in terms)


_NEGATED_FEATURE_CONTEXTS = [
    "not_controls",
    "not controls",
    "not active",
    "inactive",
    "absent",
    "without ",
    "no ",
    "not ",
    "do not",
    "don't",
    "must not",
    "never",
    "avoid",
    "not part of",
    "not included",
    "not introduce",
    "do not introduce",
    "excluded",
]


def _segments(lowered: str) -> list[str]:
    return [
        segment.strip()
        for segment in re.split(r"[\r\n.;|]+", str(lowered or ""))
        if segment.strip()
    ]


def _has_positive_any(lowered: str, terms: list[str]) -> bool:
    """Detect active mechanisms without treating negated boundary text as evidence."""
    for segment in _segments(lowered):
        if not _has_any(segment, terms):
            continue
        if any(marker in segment for marker in _NEGATED_FEATURE_CONTEXTS):
            continue
        return True
    return False


def detect_topic_features(*parts: Any) -> set[str]:
    """Return coarse technical features used to select prompt guardrails.

    The detector is intentionally conservative: a feature is used only to add a
    scoped guardrail, never to introduce a new paper mechanism.
    """
    text = _lower_join(*parts)
    features: set[str] = set()
    if _has_positive_any(text, ["multiuser", "multi-user", "multiple users", "k users", "interference"]):
        features.add("multiuser")
    if _has_positive_any(text, ["sinr", "snr", "rate", "throughput"]):
        features.add("communication_kpi")
    if _has_positive_any(text, ["downlink", "beamforming", "precoder", "precoding", "miso", "mimo-bc", "broadcast channel"]):
        features.add("downlink_beamforming")
    if _has_positive_any(text, ["uplink", "power control", "sinr target", "minimum sum transmit power", "spectral radius"]):
        features.add("uplink_power_control")
    if _has_positive_any(text, ["sum-rate", "sum rate", "weighted sum-rate", "utility maximization", "rate maximization", "throughput maximization"]):
        features.add("rate_utility")
    if _has_positive_any(text, ["wmmse", "weighted mmse", "mse weight", "receive filter"]):
        features.add("wmmse_declared")
    if _has_positive_any(text, ["quadratic transform", "fractional programming", "fp transform", "dinkelbach"]):
        features.add("fractional_transform_declared")
    if _has_positive_any(
        text,
        [
            "swipt",
            "wireless information and power transfer",
            "wireless power transfer",
            "wireless powered",
            "wpt",
            "energy harvesting",
            "harvested energy",
            "energy causality",
            "battery",
            "rectifier",
            "power splitting",
        ],
    ):
        features.add("swipt_eh")
    if _has_positive_any(text, ["nonlinear eh", "non-linear eh", "nonlinear energy", "non-linear energy", "logistic", "sigmoid", "rectifier"]):
        features.add("nonlinear_eh")
    if _has_positive_any(text, ["ris", "irs", "intelligent reflecting surface", "reconfigurable intelligent surface", "unit-modulus"]):
        features.add("ris")
    if _has_positive_any(text, ["radar", "sensing", "isac", "beampattern"]):
        features.add("sensing")
    if _has_positive_any(text, ["crb", "cramer", "cramer-rao", "fisher information", "fim"]):
        features.add("crb")
    if _has_positive_any(text, ["robust", "imperfect csi", "bounded csi", "channel uncertainty", "uncertain channel"]):
        features.add("robust_csi")
    if _has_positive_any(text, ["noma", "uav", "movable antenna", "fluid antenna", "secrecy", "eavesdropper"]):
        features.add("specialized_mechanism")
    if {"multiuser", "communication_kpi", "downlink_beamforming", "rate_utility"}.issubset(features):
        features.add("multiuser_downlink_rate")
    if {"multiuser", "communication_kpi"}.issubset(features):
        features.add("multiuser_sinr")
    return features


def _feature_label(features: set[str]) -> str:
    visible = sorted(feature for feature in features if feature not in {"communication_kpi"})
    return ", ".join(visible) if visible else "none detected"


def build_wireless_feasibility_guardrail(topic: str, *contexts: Any) -> str:
    features = detect_topic_features(topic, *contexts)
    rules = [
        "Use mechanism-specific rules only when the corresponding mechanism is present in the topic, handoff, mathematical contract, or earlier phase outputs.",
        "Do not import familiar wireless templates, algorithms, metrics, baselines, receivers, or hardware blocks that are absent from the current paper context.",
        "Prefer a paper that closes one clean optimization loop: signal model -> variables -> objective/constraints -> solvable algorithm -> executable experiment.",
    ]
    hard_mechanisms = features.intersection({"ris", "sensing", "crb", "swipt_eh", "nonlinear_eh", "robust_csi", "specialized_mechanism"})
    if len(hard_mechanisms) >= 3:
        rules.append(
            "The detected topic combines several hard mechanisms; keep the scope minimal and require every signal model, variable block, solver step, and metric to be explicit."
        )
    if "multiuser_sinr" in features:
        rules.append(
            "For multiuser SINR/rate models, define per-user streams, beams, powers, or covariance slices before writing desired-signal and interference terms; do not use one aggregate covariance as if it created per-user interference."
        )
    if "uplink_power_control" in features:
        rules.append(
            "For uplink SINR power-control topics, use an effective-gain/interference-gain model and keep fixed-point, spectral-radius, or LP claims scoped to that model."
        )
    if "multiuser_downlink_rate" in features:
        rules.append(
            "For multiuser downlink rate/utility problems with interference, WMMSE, complete quadratic-transform, or scoped SCA constructions are candidate tools only when the inherited reformulation supports them; do not claim that fixing a single SINR/gamma auxiliary makes the beamformer block convex."
        )
    if "swipt_eh" in features:
        rules.append(
            "For SWIPT/EH models, introduce only the receiver architecture and energy variables present in the current system model; RF input power for independent streams should be an expectation or covariance/power sum, not a coherent square of unrelated beam sums."
        )
    if "nonlinear_eh" in features:
        rules.append(
            "For nonlinear rectifier/EH mappings, distinguish RF input, harvested DC output, and any later surrogate variables; do not call a nonlinear-EH SCA/MM block a convex QCQP unless a valid lifted/auxiliary construction proves it."
        )
    if "ris" in features:
        rules.append(
            "For RIS/passive-surface models, preserve the stated hardware constraint such as unit modulus or diagonal lifting; do not add surface variables when the current model has no surface."
        )
    if "crb" in features:
        rules.append(
            "For CRB claims, require an observation model and Fisher-information-level definition; otherwise use sensing SNR, beampattern, or other metrics actually defined by the current paper."
        )
    if "sensing" in features and "crb" not in features:
        rules.append(
            "For sensing/radar topics without a CRB model, keep sensing metrics aligned with the defined observation or beampattern model and do not introduce CRB notation."
        )
    if "robust_csi" in features:
        rules.append(
            "For robust-CSI topics, keep uncertainty sets, safe approximations, and feasibility claims tied to the stated channel-error model."
        )
    return render_prompt_template(
        "shared/wireless_feasibility_guardrail.prompt.yaml",
        detected_mechanisms=_feature_label(features),
        topic_specific_rules="\n".join(f"  - {rule}" for rule in rules),
    )


def build_phase24_codegen_guardrail(topic: str, *contexts: Any) -> str:
    features = detect_topic_features(topic, *contexts)
    rules = [
        "- Implement the algorithm family actually named in Phase 2.3; do not replace the proposed method with a generic proxy, fallback, or candidate-search routine.",
        "- Do not emit metrics, variables, ablations, or physical subsystems that are absent from Phase 1 and Phase 2.1--2.3 plus validation_plan.yaml.",
    ]
    if "multiuser_sinr" in features:
        rules.append(
            "- For multiuser SINR/rate evaluation, carry per-user streams/beams/covariances; do not evaluate all users from one aggregate transmit covariance with synthetic self-interference."
        )
    if "downlink_beamforming" in features:
        rules.append(
            "- For MISO/MIMO downlink code, use one channel convention end-to-end and verify beam matrices have physically compatible dimensions before computing desired and interference powers."
        )
    if "swipt_eh" in features:
        rules.append(
            "- For SWIPT/EH topics, emit the physical communication and energy metrics requested by the validation plan, such as rate, harvested energy, transmit power, and feasibility, using the current model definitions."
        )
    if "wmmse_declared" in features:
        rules.append(
            "- If Phase 2.3 explicitly declares WMMSE, include receive-filter/MMSE-receiver and MSE-weight/eta_k update logic and use it inside proposed_step; do not merely return a fixed beam/rho grid as the proposed method."
        )
    if "swipt_eh" in features and "wmmse_declared" in features:
        rules.append(
            "- For SWIPT plus declared WMMSE/SCA, keep the runnable path compact: update auxiliary filters/weights, update splitting/resource variables with safeguards, and accept only states evaluated by the exact original metrics."
        )
    if "ris" in features:
        rules.append(
            "- For RIS topics, preserve the stated passive-surface hardware semantics and emit only RIS diagnostics requested by the evidence contract."
        )
    if "sensing" in features or "crb" in features:
        rules.append(
            "- For sensing/CRB topics, keep observation dimensions and sensing metrics exactly aligned with Phase 1 and Phase 2.1--2.3; do not add a new sensing model during code generation."
        )
    return "\n".join(rules)


def build_phase24_repair_guardrail(topic: str, *contexts: Any) -> str:
    features = detect_topic_features(topic, *contexts)
    rules = [
        "- keep candidate generation inside the current topic mechanisms only; do not import repair examples from other wireless topics",
    ]
    if "wmmse_declared" in features:
        rules.append(
            "- if validation reports an algorithm-code mismatch for a declared WMMSE/SCA method, add the required receive-filter/MSE-weight/block-update logic; do not add a proxy marker"
        )
    if "multiuser_sinr" in features:
        rules.append(
            "- for SINR/rate repairs, fix the physical desired/interference computation; do not treat one aggregate covariance as every user's desired signal and also as inter-user interference"
        )
    if "downlink_beamforming" in features:
        rules.append(
            "- for MISO/MIMO downlink repairs, verify channel and beam dimensions before changing objective or constraints"
        )
    if "swipt_eh" in features:
        rules.append(
            "- for EH/SWIPT repairs, compute RF input and harvested-energy metrics only from the Phase 1 and Phase 2.1--2.3 signal model and keep independent-stream power expectations correct"
        )
    else:
        rules.append(
            "- this topic has no power-transfer subsystem unless Phase 1 and Phase 2.1--2.3 say otherwise; do not add harvested-power metrics, nonlinear harvester parameters, or EH ablation branches"
        )
    if "ris" in features:
        rules.append(
            "- for RIS repairs, preserve the Phase 1 and Phase 2.1--2.3 RIS hardware constraint exactly; do not change unit-modulus semantics unless the model explicitly allows it"
        )
    else:
        rules.append(
            "- this topic has no passive-surface mechanism unless Phase 1 and Phase 2.1--2.3 say otherwise; do not add phase-matrix variables or surface-specific ablations"
        )
    if "sensing" in features or "crb" in features:
        rules.append(
            "- for sensing/radar repairs, keep the sensing model and dimensions exactly as defined in Phase 1 and Phase 2.1--2.3; do not add a new CRB/radar model"
        )
    else:
        rules.append(
            "- this topic has no additional sensing subsystem unless Phase 1 and Phase 2.1--2.3 say otherwise; do not add sensing-only branches or metrics"
        )
    return "\n".join(rules)
