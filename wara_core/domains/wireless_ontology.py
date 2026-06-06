from __future__ import annotations

import re
from typing import Any


_LAYER_LABELS: dict[str, str] = {
    "context": "Context",
    "topology": "Topology",
    "spectrum_regime": "Spectrum / Regime",
    "enabler": "Enabler",
    "task": "Task",
    "access_scheme": "Waveform / Access",
    "focus": "Technical Focus",
    "assumptions": "Assumptions",
    "optimization_form": "Optimization Form",
    "solver_family": "Solver Family",
    "metrics": "Metrics",
}

_LAYER_PRIORITY: tuple[str, ...] = (
    "task",
    "topology",
    "enabler",
    "focus",
    "assumptions",
    "metrics",
    "optimization_form",
    "solver_family",
    "spectrum_regime",
    "access_scheme",
    "context",
)

_PATTERNS: dict[str, dict[str, tuple[str, ...]]] = {
    "context": {
        "5g": (r"\b5g\b", r"\bfifth generation\b"),
        "5g_advanced": (r"\b5g[- ]advanced\b", r"\brelease 18\b", r"\brelease 19\b"),
        "b5g": (r"\bb5g\b", r"\bbeyond[- ]?5g\b"),
        "6g": (r"\b6g\b", r"\bsixth generation\b"),
    },
    "topology": {
        "point_to_point": (r"\bpoint[- ]to[- ]point\b", r"\bp2p\b"),
        "single_cell": (r"\bsingle[- ]cell\b",),
        "multi_cell": (r"\bmulti[- ]cell\b", r"\binter[- ]cell\b", r"\bcomp\b"),
        "cell_free": (r"\bcell[- ]free\b", r"\bdistributed massive mimo\b"),
        "relay_iab": (r"\brelay\b", r"\biab\b", r"\bintegrated access and backhaul\b"),
        "uav_aided": (r"\buav\b", r"\bdrone\b", r"\bunmanned aerial vehicle\b"),
        "satcom_ntn": (r"\bsatellite\b", r"\bleo\b", r"\bntn\b", r"\bnon[- ]terrestrial\b"),
        "d2d_sidelink": (r"\bd2d\b", r"\bsidelink\b", r"\bdevice[- ]to[- ]device\b"),
        "vehicular_v2x": (r"\bv2x\b", r"\bvehicular\b", r"\bvehicle[- ]to[- ]everything\b"),
        "industrial_iot": (r"\bindustrial\b", r"\bfactory\b", r"\biiot\b"),
    },
    "spectrum_regime": {
        "sub_6ghz": (r"\bsub[- ]?6\b",),
        "mmwave": (r"\bmmwave\b", r"\bmillimeter wave\b", r"\bmillimetre wave\b"),
        "thz": (r"\bthz\b", r"\bterahertz\b"),
        "near_field": (r"\bnear[- ]field\b", r"\bfresnel\b"),
        "far_field": (r"\bfar[- ]field\b", r"\bfraunhofer\b"),
        "wideband": (r"\bwideband\b", r"\bultra[- ]wideband\b", r"\buwb\b"),
        "optical_vlc": (r"\bvlc\b", r"\bvisible light communication\b", r"\boptical wireless\b"),
    },
    "enabler": {
        "siso": (r"\bsiso\b", r"\bsingle[- ]input[- ]single[- ]output\b"),
        "simo": (r"\bsimo\b", r"\bsingle[- ]input[- ]multiple[- ]output\b"),
        "miso": (r"\bmiso\b", r"\bmultiple[- ]input[- ]single[- ]output\b"),
        "mimo": (r"\bmimo\b", r"\bmultiple[- ]input[- ]multiple[- ]output\b"),
        "massive_mimo": (r"\bmassive mimo\b",),
        "xl_mimo": (r"\bxl[- ]?mimo\b", r"\bextremely large[- ]scale mimo\b", r"\belaa\b"),
        "full_duplex": (r"\bfull[- ]duplex\b", r"\bin[- ]band full[- ]duplex\b"),
        "ris": (r"\bris\b", r"\birs\b", r"\breconfigurable intelligent surfaces?\b", r"\bintelligent reflecting surfaces?\b"),
        "active_ris": (r"\bactive ris\b", r"\bactive reconfigurable intelligent surface\b"),
        "fluid_antenna": (r"\bfluid antenna\b", r"\bfas\b"),
        "movable_antenna": (r"\bmovable antenna\b", r"\bposition[- ]reconfigurable antenna\b"),
    },
    "task": {
        "isacp": (
            r"\bisacp\b",
            r"\bintegrated sensing,?\s+communication,?\s+and powering\b",
            r"\bintegrated sensing,?\s+communication,?\s+and wireless powering\b",
            r"\bintegrated sensing,?\s+communication,?\s+and power transfer\b",
        ),
        "isac": (r"\bisac\b", r"\bjcas\b", r"\bintegrated sensing and communication\b", r"\bjoint sensing and communication\b"),
        "swipt": (r"\bswipt\b", r"\bsimultaneous wireless information and power transfer\b"),
        "wpt": (r"\bwpt\b", r"\bwireless power transfer\b", r"\bwireless energy transfer\b", r"\bwireless powering\b"),
        "localization_positioning": (r"\blocalization\b", r"\bpositioning\b", r"\btracking\b"),
        "security_pls": (r"\bphysical[- ]layer security\b", r"\bsecrecy\b", r"\bsecure transmission\b"),
        "mec_offloading": (r"\bmec\b", r"\bmobile edge computing\b", r"\boffloading\b"),
        "urllc": (r"\burllc\b", r"\bultra[- ]reliable low[- ]latency\b"),
        "xr_immersive": (r"\bxr\b", r"\bar\b", r"\bvr\b", r"\bimmersive\b"),
        "massive_iot": (r"\biot\b", r"\bmassive iot\b", r"\bmassive machine[- ]type\b", r"\bmmtc\b"),
    },
    "access_scheme": {
        "ofdma": (r"\bofdma\b", r"\borthogonal frequency division multiple access\b"),
        "otfs": (r"\botfs\b", r"\borthogonal time frequency space\b"),
        "rsma": (r"\brsma\b", r"\brate[- ]splitting multiple access\b"),
        "noma": (r"\bnoma\b", r"\bnon[- ]orthogonal multiple access\b"),
        "grant_free": (r"\bgrant[- ]free\b", r"\brandom access\b"),
        "backscatter": (r"\bbackscatter\b", r"\bambient backscatter\b"),
    },
    "focus": {
        "beamforming_precoding": (r"\bbeamforming\b", r"\bprecoding\b", r"\bprecoder\b"),
        "power_allocation": (r"\bpower allocation\b", r"\bpower control\b", r"\bpower optimization\b"),
        "resource_allocation": (r"\bresource allocation\b", r"\bresource management\b"),
        "interference_management": (r"\binterference\b", r"\binterference management\b"),
        "scheduling": (r"\bscheduling\b", r"\buser scheduling\b"),
        "trajectory_design": (r"\btrajectory\b", r"\bpath planning\b", r"\bflight path\b"),
        "covariance_design": (r"\bcovariance design\b", r"\btransmit covariance\b"),
        "user_association": (r"\buser association\b", r"\bassociation\b", r"\bclustering\b"),
        "computation_offloading": (r"\bcomputation offloading\b", r"\bcompute allocation\b"),
    },
    "assumptions": {
        "perfect_csi": (r"\bperfect csi\b",),
        "imperfect_csi": (r"\bimperfect csi\b", r"\bcsi uncertainty\b", r"\bchannel uncertainty\b", r"\brobust csi\b"),
        "statistical_csi": (r"\bstatistical csi\b",),
        "channel_aging": (r"\bchannel aging\b", r"\boutdated csi\b"),
        "chance_constraint": (r"\bchance constraint\b", r"\bprobabilistic constraint\b"),
        "discrete_phase_shift": (r"\bdiscrete phase\b", r"\bquantized phase\b"),
        "nonlinear_eh_model": (r"\bnonlinear eh\b", r"\bnon-linear eh\b", r"\bnonlinear energy harvesting\b", r"\bnonlinear rectifier\b", r"\bsigmoid\b", r"\blogistic\b"),
        "finite_blocklength": (r"\bfinite blocklength\b",),
    },
    "optimization_form": {
        "convex": (r"\bconvex\b", r"\bqcqp\b", r"\bsocp\b", r"\bsecond[- ]order cone\b"),
        "fractional_programming": (r"\bfractional programming\b", r"\bdinkelbach\b", r"\bquadratic transform\b"),
        "rank_constrained": (r"\brank[- ]?1\b", r"\brank one\b", r"\brank relaxation\b"),
        "mixed_integer": (r"\bmixed[- ]integer\b", r"\binteger variable\b", r"\bbinary variable\b"),
        "robust_optimization": (r"\brobust optimization\b", r"\bworst[- ]case\b", r"\buncertainty set\b"),
        "multi_objective": (r"\bmulti[- ]objective\b", r"\bpareto\b", r"\btrade[- ]off\b"),
        "min_max": (r"\bmin[- ]max\b", r"\bmax[- ]min\b", r"\bworst user\b"),
        "queueing_control": (r"\bqueue\b", r"\blyapunov\b", r"\bstochastic network optimization\b"),
    },
    "solver_family": {
        "sdr": (r"\bsdr\b", r"\bsemidefinite relaxation\b", r"\bsemi[- ]definite relaxation\b", r"\bsdp\b"),
        "sca": (r"\bsca\b", r"\bsuccessive convex approximation\b", r"\bmajorization[- ]minimization\b", r"\bmm\b"),
        "wmmse": (r"\bwmmse\b", r"\bweighted mmse\b", r"\bweighted minimum mean square error\b"),
        "alternating_optimization": (r"\balternating optimization\b", r"\bblock coordinate descent\b", r"\bbcd\b"),
        "kkt": (r"\bkkt\b", r"\bkarush[- ]kuhn[- ]tucker\b"),
        "bisection": (r"\bbisection\b", r"\bbinary search\b"),
    },
    "metrics": {
        "sum_rate": (r"\bsum[- ]rate\b", r"\bachievable rate\b", r"\brate maximization\b"),
        "sinr": (r"\bsinr\b", r"\bsnr\b"),
        "spectral_efficiency": (r"\bspectral efficiency\b",),
        "energy_efficiency": (r"\benergy efficiency\b", r"\benergy[- ]efficient\b"),
        "harvested_power": (r"\bharvested power\b", r"\bharvested energy\b", r"\benergy harvesting\b", r"\brf powering\b"),
        "crb": (r"\bcrb\b", r"\bcramer[- ]rao\b", r"\bcram[eé]r[- ]rao\b"),
        "outage": (r"\boutage\b",),
        "latency": (r"\blatency\b", r"\bdelay\b"),
        "transmit_power": (r"\btransmit power\b", r"\bpower minimization\b"),
        "coverage": (r"\bcoverage\b",),
        "fairness": (r"\bfairness\b", r"\bproportional fair\b"),
        "secrecy_rate": (r"\bsecrecy rate\b", r"\bsecure rate\b"),
        "localization_error": (r"\blocalization error\b", r"\bpositioning error\b", r"\branging error\b"),
    },
}

_DISPLAY_LABELS: dict[str, str] = {
    "isacp": "ISACP",
    "isac": "ISAC / JCAS",
    "swipt": "SWIPT",
    "wpt": "WPT",
    "ris": "RIS",
    "active_ris": "Active RIS",
    "siso": "SISO",
    "simo": "SIMO",
    "miso": "MISO",
    "uav_aided": "UAV-Aided",
    "cell_free": "Cell-Free",
    "point_to_point": "Point-to-Point",
    "relay_iab": "Relay / IAB",
    "vehicular_v2x": "Vehicular / V2X",
    "industrial_iot": "Industrial IoT",
    "mimo": "MIMO",
    "massive_mimo": "Massive MIMO",
    "xl_mimo": "XL-MIMO",
    "full_duplex": "Full Duplex",
    "thz": "THz",
    "near_field": "Near-Field",
    "far_field": "Far-Field",
    "optical_vlc": "Optical / VLC",
    "ofdma": "OFDMA",
    "otfs": "OTFS",
    "rsma": "RSMA",
    "noma": "NOMA",
    "grant_free": "Grant-Free Access",
    "urllc": "URLLC",
    "xr_immersive": "XR / Immersive",
    "massive_iot": "Massive IoT",
    "mec_offloading": "MEC / Offloading",
    "security_pls": "Physical-Layer Security",
    "localization_positioning": "Localization / Positioning",
    "nonlinear_eh_model": "Nonlinear EH Model",
    "imperfect_csi": "Imperfect CSI",
    "beamforming_precoding": "Beamforming / Precoding",
    "power_allocation": "Power Allocation",
    "resource_allocation": "Resource Allocation",
    "covariance_design": "Covariance Design",
    "trajectory_design": "Trajectory Design",
    "user_association": "User Association",
    "computation_offloading": "Computation Offloading",
    "sdr": "SDR",
    "sca": "SCA",
    "wmmse": "WMMSE",
    "sum_rate": "Sum-Rate",
    "sinr": "SINR",
    "energy_efficiency": "Energy Efficiency",
    "harvested_power": "Harvested Power",
    "transmit_power": "Transmit Power",
    "crb": "CRB",
    "coverage": "Coverage",
    "fairness": "Fairness",
    "secrecy_rate": "Secrecy Rate",
    "localization_error": "Localization Error",
}

_GENERIC_WIRELESS_HINTS: tuple[str, ...] = (
    r"\bwireless\b",
    r"\bbase station\b",
    r"\baccess point\b",
    r"\buser equipment\b",
    r"\bue\b",
    r"\bchannel\b",
    r"\bsinr\b",
    r"\bbeamforming\b",
    r"\bprecoding\b",
    r"\bantenna\b",
    r"\bmimo\b",
    r"\binterference\b",
)


def canonical_tag_label(tag: str) -> str:
    return _DISPLAY_LABELS.get(tag, tag.replace("_", " ").title())


def _ordered_tags(text: str, aliases: dict[str, tuple[str, ...]]) -> list[str]:
    tags: list[str] = []
    for canonical, patterns in aliases.items():
        if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns):
            tags.append(canonical)
    return tags


def _hint_score(text: str) -> int:
    return sum(1 for pattern in _GENERIC_WIRELESS_HINTS if re.search(pattern, text, flags=re.IGNORECASE))


def extract_wireless_ontology(text: str) -> dict[str, Any]:
    """Return a WARA-native layered wireless profile for a topic or artifact."""

    lowered = f" {str(text or '').lower()} "
    layers = {layer: _ordered_tags(lowered, patterns) for layer, patterns in _PATTERNS.items()}
    if "isacp" in layers["task"]:
        for tag in ("isac", "swipt", "wpt"):
            if tag not in layers["task"]:
                layers["task"].append(tag)
    flat_tags = [tag for layer in _LAYER_LABELS for tag in layers.get(layer, [])]
    non_context_count = sum(len(tags) for layer, tags in layers.items() if layer != "context")
    hint_score = _hint_score(lowered)
    is_wireless = bool(non_context_count) or hint_score >= 2 or (bool(layers["context"]) and hint_score >= 1)
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
    primary_tags: list[str] = []
    for layer in _LAYER_PRIORITY:
        for tag in layers.get(layer, []):
            if tag not in primary_tags:
                primary_tags.append(tag)
    primary_tags = primary_tags[:8]
    primary_tag_labels = [canonical_tag_label(tag) for tag in primary_tags]
    return {
        "is_wireless": is_wireless,
        "layers": layers,
        "display_layers": display_layers,
        "flat_tags": flat_tags,
        "flat_tag_labels": [canonical_tag_label(tag) for tag in flat_tags],
        "primary_tags": primary_tags,
        "primary_tag_labels": primary_tag_labels,
        "layer_count": sum(1 for tags in layers.values() if tags),
        "tag_count": len(flat_tags),
        "summary": " | ".join(primary_tag_labels[:6]),
        "hint_score": hint_score,
        "source": "wara_native_wireless_ontology",
    }


def looks_like_wireless_topic(text: str) -> bool:
    return bool(extract_wireless_ontology(text).get("is_wireless"))


__all__ = ["canonical_tag_label", "extract_wireless_ontology", "looks_like_wireless_topic"]
