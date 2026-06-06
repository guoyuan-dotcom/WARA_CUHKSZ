from __future__ import annotations

import re
from typing import Any

from .wireless_ontology import canonical_tag_label, extract_wireless_ontology


_LAYER_ORDER: tuple[str, ...] = ("technology", "scenario", "optimization", "theory")
_LAYER_LABELS: dict[str, str] = {
    "technology": "Technology Layer",
    "scenario": "Scenario Layer",
    "optimization": "Optimization Layer",
    "theory": "Theory Layer",
}

_TAG_TO_LAYER: dict[str, str] = {
    "isacp": "technology",
    "isac": "technology",
    "swipt": "technology",
    "wpt": "technology",
    "localization_positioning": "technology",
    "security_pls": "technology",
    "mec_offloading": "technology",
    "urllc": "technology",
    "xr_immersive": "technology",
    "massive_iot": "technology",
    "siso": "technology",
    "simo": "technology",
    "miso": "technology",
    "ris": "technology",
    "active_ris": "technology",
    "mimo": "technology",
    "massive_mimo": "technology",
    "xl_mimo": "technology",
    "full_duplex": "technology",
    "cell_free": "technology",
    "ofdma": "technology",
    "otfs": "technology",
    "rsma": "technology",
    "noma": "technology",
    "grant_free": "technology",
    "backscatter": "technology",
    "point_to_point": "scenario",
    "uav_aided": "scenario",
    "single_cell": "scenario",
    "multi_cell": "scenario",
    "relay_iab": "scenario",
    "satcom_ntn": "scenario",
    "near_field": "scenario",
    "far_field": "scenario",
    "mmwave": "scenario",
    "thz": "scenario",
    "optical_vlc": "scenario",
    "vehicular_v2x": "scenario",
    "industrial_iot": "scenario",
    "beamforming_precoding": "optimization",
    "power_allocation": "optimization",
    "resource_allocation": "optimization",
    "covariance_design": "optimization",
    "trajectory_design": "optimization",
    "scheduling": "optimization",
    "user_association": "optimization",
    "computation_offloading": "optimization",
    "interference_management": "optimization",
    "nonlinear_eh_model": "theory",
    "imperfect_csi": "theory",
    "statistical_csi": "theory",
    "channel_aging": "theory",
    "chance_constraint": "theory",
    "finite_blocklength": "theory",
    "robust_optimization": "theory",
    "rank_constrained": "theory",
    "fractional_programming": "theory",
    "mixed_integer": "theory",
    "queueing_control": "theory",
    "sdr": "theory",
    "sca": "theory",
    "wmmse": "theory",
    "kkt": "theory",
}

_OPEN_SLOT_LABELS: dict[str, str] = {
    "technology": "open wireless mechanism",
    "scenario": "open deployment scenario",
    "optimization": "open resource-control layer",
    "theory": "open tractability route",
}

_RESEARCH_AXES: tuple[dict[str, Any], ...] = (
    {
        "id": "deployment_topology",
        "label": "deployment / topology level",
        "purpose": "Where nodes are placed and how links are organized.",
        "examples": [
            "point-to-point",
            "single-cell",
            "multi-cell / CoMP",
            "cell-free / distributed APs",
            "relay / IAB",
            "D2D / sidelink",
            "NTN / satellite",
            "UAV / aerial",
            "vehicular / V2X",
            "industrial / indoor",
        ],
    },
    {
        "id": "node_antenna_architecture",
        "label": "node / antenna architecture level",
        "purpose": "What radio architecture creates spatial degrees of freedom.",
        "examples": [
            "SISO / SIMO / MISO / MIMO",
            "massive MIMO",
            "XL-MIMO",
            "hybrid analog-digital beamforming",
            "full duplex",
            "RIS / metasurface",
            "active RIS",
            "movable / fluid antennas",
            "distributed antennas",
        ],
    },
    {
        "id": "propagation_spectrum_regime",
        "label": "propagation / spectrum level",
        "purpose": "Which channel physics or spectrum regime changes coupling.",
        "examples": [
            "sub-6 GHz",
            "mmWave",
            "THz",
            "near-field / spherical wave",
            "far-field / planar wave",
            "wideband / beam squint",
            "high mobility / Doppler",
            "blockage / shadowing",
            "optical / VLC",
        ],
    },
    {
        "id": "access_waveform_resource",
        "label": "access / waveform / resource level",
        "purpose": "How radio resources are divided, reused, or multiplexed.",
        "examples": [
            "OFDMA / OFDM",
            "NOMA",
            "RSMA",
            "OTFS",
            "grant-free access",
            "scheduling",
            "time / frequency / beam allocation",
            "power splitting / time switching",
            "multicast / broadcast",
        ],
    },
    {
        "id": "service_task_level",
        "label": "service / task level",
        "purpose": "What wireless service or integrated task is being optimized.",
        "examples": [
            "communication",
            "sensing / radar",
            "localization / positioning",
            "wireless powering / WPT",
            "SWIPT",
            "ISAC",
            "ISACP",
            "MEC / offloading",
            "physical-layer security",
            "XR / immersive",
            "IoT / massive access",
        ],
    },
    {
        "id": "control_variable_level",
        "label": "optimization-control level",
        "purpose": "Which variables the formulation can actually optimize.",
        "examples": [
            "power",
            "beamformer / precoder",
            "transmit covariance",
            "waveform",
            "user association",
            "scheduling",
            "trajectory / placement",
            "surface coefficients",
            "subcarrier / bandwidth",
            "computation / cache",
        ],
    },
    {
        "id": "information_reliability_level",
        "label": "information / reliability level",
        "purpose": "What uncertainty, feedback, and reliability assumptions matter.",
        "examples": [
            "perfect CSI",
            "imperfect / statistical CSI",
            "channel aging",
            "outage / chance constraints",
            "finite blocklength",
            "robust design",
            "privacy / secrecy",
            "mobility prediction",
        ],
    },
    {
        "id": "hardware_energy_sustainability_level",
        "label": "hardware / energy / sustainability level",
        "purpose": "Whether device response, hardware limits, or energy use is central.",
        "examples": [
            "RF-chain limits",
            "low-resolution ADC/DAC",
            "phase quantization",
            "power amplifier efficiency",
            "energy harvesting",
            "energy conversion response",
            "network energy saving",
            "battery / circuit power",
        ],
    },
    {
        "id": "algorithm_theory_level",
        "label": "algorithm / theory level",
        "purpose": "Which mathematical structure or solution route is natural.",
        "examples": [
            "convex / SOCP / SDP",
            "SCA / MM",
            "fractional programming",
            "WMMSE",
            "alternating / block coordinate optimization",
            "mixed-integer optimization",
            "Lyapunov / queueing control",
            "distributed / federated optimization",
            "learning-assisted optimization",
        ],
    },
    {
        "id": "metric_evidence_level",
        "label": "metric / evidence level",
        "purpose": "Which system-level outcomes can support the paper claim.",
        "examples": [
            "rate / spectral efficiency",
            "latency",
            "reliability / outage",
            "energy efficiency",
            "transmit power",
            "harvested energy",
            "sensing / localization accuracy",
            "coverage",
            "fairness",
            "secrecy",
        ],
    },
)


def _dedupe(values: list[str] | tuple[str, ...]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _taxonomy_profile(text: str) -> dict[str, Any]:
    ontology = extract_wireless_ontology(text)
    layers: dict[str, list[str]] = {layer: [] for layer in _LAYER_ORDER}
    for tag in ontology.get("flat_tags", []):
        layer = _TAG_TO_LAYER.get(str(tag))
        if layer:
            layers[layer].append(str(tag))
    if "isacp" in layers["technology"]:
        for tag in ("isac", "swipt", "wpt"):
            if tag not in layers["technology"]:
                layers["technology"].append(tag)
    layers = {layer: _dedupe(tags) for layer, tags in layers.items()}
    missing = [layer for layer in _LAYER_ORDER if not layers.get(layer)]
    display_layers = [
        {
            "id": layer,
            "label": _LAYER_LABELS[layer],
            "tags": tags,
            "tag_labels": [canonical_tag_label(tag) for tag in tags],
        }
        for layer, tags in layers.items()
        if tags
    ]
    return {
        "layers": layers,
        "display_layers": display_layers,
        "missing_layers": missing,
        "layer_count": len(_LAYER_ORDER) - len(missing),
        "summary": " | ".join(
            canonical_tag_label(tags[0]) for tags in layers.values() if tags
        ),
        "source": "wara_native_wireless_taxonomy",
    }


def _seed_recommendations(profile: dict[str, Any]) -> dict[str, list[str]]:
    layers = profile.get("layers", {}) if isinstance(profile, dict) else {}
    return {layer: list(layers.get(layer, []) or []) for layer in _LAYER_ORDER}


def build_wireless_topic_taxonomy_plan(
    topic: str,
    *,
    context_text: str = "",
    max_blueprints: int = 5,
) -> dict[str, Any]:
    """Build a compact WARA-native taxonomy plan for Phase1 direction search."""

    text = "\n".join(part for part in (topic, context_text) if part)
    profile = _taxonomy_profile(text)
    recommended = _seed_recommendations(profile)
    blueprints: list[dict[str, Any]] = []
    seen_blueprint_axes: set[tuple[str, str, str]] = set()
    technology_options = recommended["technology"] or ["open_technology"]
    scenario_options = recommended["scenario"] or ["open_scenario"]
    optimization_options = recommended["optimization"] or ["open_optimization"]
    theory_options = recommended["theory"] or ["open_theory"]
    for technology in technology_options:
        for scenario in scenario_options:
            for optimization in optimization_options:
                for theory in theory_options:
                    axes = (technology, scenario, optimization)
                    if axes in seen_blueprint_axes:
                        continue
                    seen_blueprint_axes.add(axes)
                    blueprint = {
                        "blueprint_id": f"BP-{len(blueprints) + 1:02d}",
                        "technology": technology,
                        "scenario": scenario,
                        "optimization": optimization,
                        "theory": theory,
                        "technology_label": _taxonomy_label("technology", technology),
                        "scenario_label": _taxonomy_label("scenario", scenario),
                        "optimization_label": _taxonomy_label("optimization", optimization),
                        "theory_label": _taxonomy_label("theory", theory),
                        "title_stub": _blueprint_title_stub(technology, scenario, optimization),
                        "open_slots": [layer for layer, value in {
                            "technology": technology,
                            "scenario": scenario,
                            "optimization": optimization,
                            "theory": theory,
                        }.items() if str(value).startswith("open_")],
                    }
                    blueprints.append(blueprint)
                    if len(blueprints) >= max_blueprints:
                        break
                if len(blueprints) >= max_blueprints:
                    break
            if len(blueprints) >= max_blueprints:
                break
        if len(blueprints) >= max_blueprints:
            break

    missing = list(profile.get("missing_layers") or [])
    prompt_lines = [
        "## WARA Wireless Topic Taxonomy",
        f"- Input summary: {profile.get('summary') or 'No strong seed detected yet'}",
    ]
    for layer in _LAYER_ORDER:
        matched = profile["layers"].get(layer, [])
        rec = recommended.get(layer, [])
        prompt_lines.append(
            f"- {_LAYER_LABELS[layer]}: "
            f"matched={[canonical_tag_label(tag) for tag in matched] or ['None']} | "
            f"locked={[canonical_tag_label(tag) for tag in rec] or ['None; ScoutAgent must infer this from mechanism/gap evidence']}"
        )
    prompt_lines.append("")
    prompt_lines.append("### Candidate Blueprints")
    if blueprints:
        for blueprint in blueprints:
            prompt_lines.append(f"- {blueprint['blueprint_id']}: {blueprint['title_stub']}")
    else:
        prompt_lines.append("- None preselected; the ScoutAgent must make a focused wireless optimization direction.")
    prompt_lines.append("")
    prompt_lines.append("### Wireless Research Axes")
    prompt_lines.append("- Use these axes as a coverage checklist, not as a closed menu or default candidate list.")
    prompt_lines.append("- A strong direction may use one axis or combine axes only when their interaction creates a real optimization gap.")
    prompt_lines.append("- Concrete mechanisms may come from the user topic, retrieved literature, or wireless reasoning even if not listed below.")
    for axis in _RESEARCH_AXES:
        examples = ", ".join(axis["examples"][:8])
        prompt_lines.append(f"- {axis['label']}: {axis['purpose']} Examples include {examples}; not limited to these.")

    return {
        "topic": topic,
        "input_profile": profile,
        "recommended_layers": recommended,
        "missing_layers": missing,
        "blueprints": blueprints,
        "research_axes": list(_RESEARCH_AXES),
        "constraint_lines": [
            "Every candidate must name the wireless mechanism, optimization controls, KPI, and tractability route.",
            "Taxonomy entries and research-axis examples are coverage guides, not defaults. Missing layers must be filled by mechanism reasoning and literature evidence, not by hardcoded technology or solver choices.",
            "If a topic is broad, propose only mechanism combinations that create a concrete coupling, tradeoff, constraint structure, or operating regime.",
            "Do not force nonconvexity; convex, nonconvex, and mixed routes are all acceptable when the contribution is clear.",
        ],
        "prompt_block": "\n".join(prompt_lines).strip(),
    }


def _blueprint_title_stub(technology: str, scenario: str, optimization: str) -> str:
    parts: list[str] = []
    scenario_label = _taxonomy_label("scenario", scenario)
    technology_label = _taxonomy_label("technology", technology)
    if scenario != technology and not scenario.startswith("open_"):
        parts.append(scenario_label)
    if not technology.startswith("open_"):
        parts.append(technology_label)
    if not optimization.startswith("open_"):
        parts.append(_taxonomy_label("optimization", optimization))
    label = " ".join(part for part in parts if part).strip()
    return label or "Open Wireless Optimization Direction"


def _taxonomy_label(layer: str, tag: str) -> str:
    if tag.startswith("open_"):
        return _OPEN_SLOT_LABELS.get(layer, tag.replace("_", " "))
    return canonical_tag_label(tag)


def assess_wireless_topic_taxonomy_candidate(
    candidate_text: str,
    *,
    taxonomy_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    candidate_profile = _taxonomy_profile(candidate_text)
    layers = candidate_profile.get("layers", {})
    layer_coverage = round(candidate_profile.get("layer_count", 0) / len(_LAYER_ORDER), 2)
    recommended = taxonomy_plan.get("recommended_layers", {}) if isinstance(taxonomy_plan, dict) else {}
    overlaps: dict[str, list[str]] = {}
    aligned = 0
    for layer in _LAYER_ORDER:
        overlap = sorted(set(layers.get(layer, []) or []) & set(recommended.get(layer, []) or []))
        overlaps[layer] = overlap
        if overlap:
            aligned += 1
    blueprint_scores: list[dict[str, Any]] = []
    for blueprint in (taxonomy_plan or {}).get("blueprints", []) or []:
        matches = sum(
            1
            for layer in _LAYER_ORDER
            if blueprint.get(layer) in set(layers.get(layer, []) or [])
        )
        blueprint_scores.append(
            {
                "blueprint_id": blueprint.get("blueprint_id"),
                "title_stub": blueprint.get("title_stub"),
                "score": round(matches / len(_LAYER_ORDER), 2),
            }
        )
    blueprint_scores.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
    best = blueprint_scores[0] if blueprint_scores else {}
    return {
        "candidate_profile": candidate_profile,
        "coverage": layer_coverage,
        "layer_coverage": layer_coverage,
        "aligned_layer_count": aligned,
        "missing_layers": list(candidate_profile.get("missing_layers") or []),
        "missing_layer_labels": [_LAYER_LABELS[layer] for layer in candidate_profile.get("missing_layers", [])],
        "recommended_overlap": overlaps,
        "best_blueprint": best,
        "best_blueprint_score": float(best.get("score") or 0.0),
        "meets_minimum": aligned >= 2 and layer_coverage >= 0.5,
    }


def build_wireless_working_topic_label(
    taxonomy_plan: dict[str, Any],
    *,
    fallback_topic: str = "",
    include_optimization_word: bool = True,
) -> str:
    blueprints = taxonomy_plan.get("blueprints", []) if isinstance(taxonomy_plan, dict) else []
    if blueprints:
        first = dict(blueprints[0])
        parts = [
            str(first.get("scenario_label") or "").strip(),
            str(first.get("technology_label") or "").strip(),
            str(first.get("optimization_label") or "").strip(),
        ]
        label = " ".join(part for part in parts if part)
        if label and include_optimization_word and "optimization" not in label.lower():
            label = f"{label} Optimization"
        if label:
            return label
    cleaned = re.sub(r"\s+", " ", fallback_topic or "Wireless Communications Optimization").strip(" :;-")
    return cleaned or "Wireless Communications Optimization"


__all__ = [
    "assess_wireless_topic_taxonomy_candidate",
    "build_wireless_topic_taxonomy_plan",
    "build_wireless_working_topic_label",
]
