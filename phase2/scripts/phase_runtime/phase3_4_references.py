from __future__ import annotations

import difflib
import importlib
import json
import os
import re
import shutil
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from pipeline_core import (
    DEFAULT_MODEL_PROFILE,
    compact_text,
    read_json,
    read_text,
    resolve_phase1_run_path,
    write_text,
)
from pipeline_core.json_utils import _safe_json_loads
from phase_runtime.agent_context import build_role_agent_request_json
from phase_runtime.llm import create_llm_client
from phase_runtime.paper_mode import _paper_writing_mode_snapshot
from phase_runtime.prompt_templates import render_prompt_template
from phase_runtime.phase3_figure import (
    build_phase3_figure_diagram_image_prompt,
    build_phase3_figure_diagram_spec_prompt,
)

_TITLE_SMALL_WORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "into",
    "of",
    "on",
    "or",
    "the",
    "to",
    "under",
    "via",
    "with",
    "over",
    "through",
    "toward",
    "towards",
}

_TITLE_ACRONYMS = {
    "5g": "5G",
    "6g": "6G",
    "ai": "AI",
    "b5g": "B5G",
    "bs": "BS",
    "c-ran": "C-RAN",
    "csi": "CSI",
    "d2d": "D2D",
    "dfrc": "DFRC",
    "dr": "DR",
    "eh": "EH",
    "embb": "eMBB",
    "irs": "IRS",
    "isac": "ISAC",
    "isacp": "ISACP",
    "iot": "IoT",
    "los": "LoS",
    "mec": "MEC",
    "mimo": "MIMO",
    "miso": "MISO",
    "ml": "ML",
    "mmwave": "mmWave",
    "mmtc": "mMTC",
    "noma": "NOMA",
    "nlos": "NLoS",
    "o-ran": "O-RAN",
    "ofdm": "OFDM",
    "ofdma": "OFDMA",
    "qoe": "QoE",
    "qos": "QoS",
    "rf": "RF",
    "ris": "RIS",
    "rsma": "RSMA",
    "sdp": "SDP",
    "sdr": "SDR",
    "sinr": "SINR",
    "siso": "SISO",
    "star-ris": "STAR-RIS",
    "swipt": "SWIPT",
    "thz": "THz",
    "uav": "UAV",
    "urllc": "URLLC",
    "wcl": "WCL",
    "wmmse": "WMMSE",
    "wpt": "WPT",
    "xl-mimo": "XL-MIMO",
}

COMMON_WIRELESS_ABBREVIATION_EXPANSIONS: dict[str, str] = {
    "AO": "alternating optimization",
    "AP": "access point",
    "AWGN": "additive white Gaussian noise",
    "BCD": "block coordinate descent",
    "BS": "base station",
    "CCP": "convex-concave procedure",
    "CPU": "central processing unit",
    "CSI": "channel state information",
    "DC": "direct-current",
    "DFRC": "dual-functional radar-communication",
    "DL": "downlink",
    "EH": "energy harvesting",
    "FD": "full-duplex",
    "FI": "Fisher information",
    "FP": "fractional programming",
    "FP-MM-CCP": "fractional-programming, majorization-minimization, and convex-concave-procedure",
    "IRS": "intelligent reflecting surface",
    "ISAC": "integrated sensing and communication",
    "ISACP": "integrated sensing, communication, and powering",
    "KPI": "key performance indicator",
    "LP": "linear program",
    "MICP": "mixed-integer conic program",
    "MILP": "mixed-integer linear program",
    "MIMO": "multiple-input multiple-output",
    "MEC": "mobile edge computing",
    "MISO": "multiple-input single-output",
    "MISOCP": "mixed-integer second-order cone program",
    "MINLP": "mixed-integer nonlinear program",
    "MIQP": "mixed-integer quadratic program",
    "KKT": "Karush-Kuhn-Tucker",
    "MM": "majorization--minimization",
    "MMSE": "minimum mean-square error",
    "MRT": "maximum-ratio transmission",
    "MSE": "mean-square error",
    "NOMA": "non-orthogonal multiple access",
    "OFDM": "orthogonal frequency-division multiplexing",
    "OFDMA": "orthogonal frequency-division multiple access",
    "PSD": "positive semidefinite",
    "QP": "quadratic program",
    "QCP": "quadratically constrained program",
    "QCQP": "quadratically constrained quadratic program",
    "QoS": "quality of service",
    "RF": "radio-frequency",
    "RIS": "reconfigurable intelligent surface",
    "RZF": "regularized zero-forcing",
    "SCA": "successive convex approximation",
    "SDP": "semidefinite programming",
    "SDR": "semidefinite relaxation",
    "SI": "self-interference",
    "SINR": "signal-to-interference-plus-noise ratio",
    "SNR": "signal-to-noise ratio",
    "SOC": "second-order cone",
    "SOCP": "second-order cone programming",
    "STAR-RIS": "simultaneously transmitting and reflecting reconfigurable intelligent surface",
    "SWIPT": "simultaneous wireless information and power transfer",
    "UL": "uplink",
    "ULA": "uniform linear array",
    "UAV": "unmanned aerial vehicle",
    "WSR": "weighted sum rate",
    "WSSR": "weighted sum secrecy rate",
    "WMMSE": "weighted minimum mean-square error",
    "WPT": "wireless power transfer",
    "XL-MIMO": "extremely large-scale multiple-input multiple-output",
    "ZF": "zero-forcing",
    "AO-FP-MM": "alternating optimization with fractional programming and majorization-minimization",
}


def _impl() -> Any:
    return importlib.import_module("phase_runtime_impl")


def _format_ieee_title_part(part: str, *, force_capital: bool = False) -> str:
    if not part:
        return part
    lowered = part.lower()
    if lowered in _TITLE_ACRONYMS:
        return _TITLE_ACRONYMS[lowered]
    if re.fullmatch(r"[A-Z0-9]{2,}", part):
        return part
    if not force_capital and lowered in _TITLE_SMALL_WORDS:
        return lowered
    return part[:1].upper() + part[1:].lower()


def format_ieee_paper_title(title: str) -> str:
    """Return an IEEE-style manuscript title while preserving common wireless acronyms."""
    raw = re.sub(r"\s+", " ", str(title or "").strip())
    if not raw:
        return raw
    tokens = raw.split(" ")
    formatted: list[str] = []
    for index, token in enumerate(tokens):
        leading = re.match(r"^[^A-Za-z0-9]*", token).group(0)
        trailing = re.search(r"[^A-Za-z0-9]*$", token).group(0)
        body = token[len(leading) : len(token) - len(trailing) if trailing else len(token)]
        if not body:
            formatted.append(token)
            continue
        force_capital = index == 0 or index == len(tokens) - 1 or (formatted and formatted[-1].endswith(":"))
        if body.lower() in _TITLE_ACRONYMS:
            new_body = _TITLE_ACRONYMS[body.lower()]
        else:
            pieces = re.split(r"(-)", body)
            new_pieces = [
                piece
                if piece == "-"
                else _format_ieee_title_part(piece, force_capital=force_capital or part_index == 0 and force_capital)
                for part_index, piece in enumerate(pieces)
            ]
            new_body = "".join(new_pieces)
        formatted.append(f"{leading}{new_body}{trailing}")
    return " ".join(formatted)


def paper_title_quality_issues(title: str, *, working_title: str = "") -> list[str]:
    """Return lightweight paper-title issues without affecting technical contracts."""
    cleaned = re.sub(r"\s+", " ", str(title or "").strip())
    if not cleaned:
        return ["empty_paper_title"]
    words = re.findall(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*", cleaned)
    if len(words) < 5:
        return ["too_short_for_paper_title"]
    if len(words) > 18:
        return ["too_long_for_paper_title"]
    lowered = cleaned.lower()
    template_prefixes = (
        "weighted sum-rate ",
        "weighted sum rate ",
        "sum-rate ",
        "sum rate ",
        "max-min ",
        "transmit-power minimization ",
        "power minimization ",
    )
    if lowered.startswith(template_prefixes) and re.search(r"\bfor\b", lowered):
        return ["template_objective_method_scenario_title"]
    if re.search(r"\boptimization\s+for\s+.+\s+under\s+", lowered):
        return ["stacked_optimization_under_title"]
    raw_working = re.sub(r"\s+", " ", str(working_title or "").strip()).lower()
    if raw_working and lowered == raw_working and lowered.startswith(template_prefixes):
        return ["paper_title_reuses_template_working_title"]
    return []


def paper_title_quality_ok(title: str, *, working_title: str = "") -> bool:
    return not paper_title_quality_issues(title, working_title=working_title)


def _phase_handoff_paper_title_candidates(run_dir: Path) -> list[tuple[str, str]]:
    """Return paper-facing title candidates paired with their working title."""
    candidates: list[tuple[str, str]] = []
    manifest = read_json(run_dir / "phase1_handoff_manifest.json") or {}
    if isinstance(manifest, dict):
        candidates.append((str(manifest.get("paper_title") or ""), str(manifest.get("final_title") or "")))
    phase1_handoff = read_json(run_dir / "input_from_phase1" / "phase1_handoff.json") or {}
    if isinstance(phase1_handoff, dict):
        selected = phase1_handoff.get("selected_candidate")
        decision = phase1_handoff.get("selection_decision")
        working = ""
        if isinstance(selected, dict):
            working = str(selected.get("title") or "")
            candidates.append((str(selected.get("paper_title") or ""), working))
        if isinstance(decision, dict):
            candidates.append((str(decision.get("paper_title") or ""), str(decision.get("selected_title") or working)))
    return candidates


def _phase_handoff_title_candidates(run_dir: Path) -> list[str]:
    candidates: list[str] = []
    manifest = read_json(run_dir / "phase1_handoff_manifest.json") or {}
    if isinstance(manifest, dict):
        candidates.append(str(manifest.get("final_title") or ""))
    phase1_handoff = read_json(run_dir / "input_from_phase1" / "phase1_handoff.json") or {}
    if isinstance(phase1_handoff, dict):
        selected = phase1_handoff.get("selected_candidate")
        decision = phase1_handoff.get("selection_decision")
        if isinstance(selected, dict):
            candidates.append(str(selected.get("title") or ""))
        if isinstance(decision, dict):
            candidates.append(str(decision.get("selected_title") or ""))
    candidate_review = read_json(run_dir / "input_from_phase1" / "candidate_review.json") or {}
    if isinstance(candidate_review, dict):
        decision = candidate_review.get("selection_decision")
        if isinstance(decision, dict):
            candidates.append(str(decision.get("selected_title") or ""))
    summary = read_json(run_dir / "phase2_summary.json") or {}
    if isinstance(summary, dict):
        candidates.append(str(summary.get("selected_title") or ""))
    return candidates


def resolve_paper_title(phase_dir: Path, fallback_title: str = "") -> str:
    """Resolve the paper title from frozen Phase 1 artifacts, not the raw user topic."""
    run_dir = Path(phase_dir).parent
    for candidate, working_title in _phase_handoff_paper_title_candidates(run_dir):
        cleaned = re.sub(r"\s+", " ", str(candidate or "").strip())
        if cleaned and paper_title_quality_ok(cleaned, working_title=working_title):
            return format_ieee_paper_title(cleaned)
    for candidate in [*_phase_handoff_title_candidates(run_dir), str(fallback_title or ""), run_dir.name]:
        cleaned = re.sub(r"\s+", " ", str(candidate or "").strip())
        if not cleaned:
            continue
        if cleaned.lower() in {"run", "runs", "phase2", "phase3", "phase3-3", "phase3-4", "phase3-5"}:
            continue
        return format_ieee_paper_title(cleaned)
    return format_ieee_paper_title(str(fallback_title or run_dir.name))


def build_phase3_4_design_notes() -> str:
    return _impl().build_phase3_4_design_notes()


def load_phase3_proposed_solution_snippet(run_dir: Path) -> str:
    return _impl().load_phase3_proposed_solution_snippet(run_dir)


def load_phase3_1_system_model_problem_snippet(run_dir: Path) -> str:
    return _impl().load_phase3_1_system_model_problem_snippet(run_dir)


def load_phase3_1_proposed_solution_snippet(run_dir: Path) -> str:
    return _impl().load_phase3_1_proposed_solution_snippet(run_dir)


def sanitize_phase3_2_numerical_results_tex(tex: str) -> str:
    return _impl().sanitize_phase3_2_numerical_results_tex(tex)


def _phase3_4_load_method_naming(method_naming_summary_json: str, experiment_plan_json: str) -> dict[str, Any]:
    from phase_runtime.phase3_3_sections import _phase3_3_load_method_naming

    return _phase3_3_load_method_naming(method_naming_summary_json, experiment_plan_json)


def _phase3_4_select_methods(methods_payload: dict[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    from phase_runtime.phase3_3_sections import _phase3_3_select_methods

    return _phase3_3_select_methods(methods_payload)


def _prepare_full_paper_preview_inputs(phase_dir: Path, build_dir: Path) -> None:
    return _impl()._prepare_full_paper_preview_inputs(phase_dir, build_dir)


def sanitize_phase3_4_preview_section_tex(tex: str) -> str:
    """Keep full-paper preview compilable when snippets carry line-level labels."""
    cleaned = re.sub(r"(\\subsection\*?\{[^{}]+\})\s*\\label\{[^}]+\}", r"\1", tex)

    def strip_extra_labels(match: re.Match[str]) -> str:
        block = match.group(0)
        seen_label = False

        def label_repl(label_match: re.Match[str]) -> str:
            nonlocal seen_label
            if not seen_label:
                seen_label = True
                return label_match.group(0)
            return ""

        return re.sub(r"\\label\{[^}]+\}", label_repl, block)

    return re.sub(r"\\begin\{equation\}.*?\\end\{equation\}", strip_extra_labels, cleaned, flags=re.S)


def _phase3_4_plaintext_for_abbreviation_check(tex: str) -> str:
    """Convert LaTeX-ish prose to plain text while preserving acronym order."""
    text = str(tex or "").replace("\r\n", "\n")
    text = re.sub(r"%.*", "", text)
    text = re.sub(r"\\begin\{(?:equation|align|aligned|split|multline|gather|subequations|algorithm|algorithmic)\*?\}.*?\\end\{(?:equation|align|aligned|split|multline|gather|subequations|algorithm|algorithmic)\*?\}", " ", text, flags=re.S)
    text = re.sub(r"\$\$.*?\$\$", " ", text, flags=re.S)
    text = re.sub(r"\$.*?\$", " ", text, flags=re.S)
    text = re.sub(r"\\\[.*?\\\]|\\\(.*?\\\)", " ", text, flags=re.S)
    text = re.sub(r"\\cite[a-zA-Z]*\{[^{}]*\}", " ", text)
    text = re.sub(r"\\(?:ref|eqref|label|bibliography|bibliographystyle)\{[^{}]*\}", " ", text)
    text = re.sub(r"\\(?:section|subsection|subsubsection)\*?\{([^{}]*)\}", r"\n\1\n", text)
    text = re.sub(r"\\(?:begin|end)\{[^{}]*\}", " ", text)
    text = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?", " ", text)
    text = text.replace("{", " ").replace("}", " ")
    text = text.replace("~", " ").replace("--", "-")
    return re.sub(r"[ \t]+", " ", text)


def _phase3_4_acronym_candidates(text: str) -> list[re.Match[str]]:
    pattern = (
        r"\b(?:"
        r"[A-Z0-9]+(?:-[A-Z0-9]+)+s?"
        r"|[A-Z]{2,}[A-Z0-9]*"
        r"|[A-Z][a-z]+-[A-Z]{2,}[A-Za-z0-9-]*"
        r"|[A-Z][0-9]+"
        r")\b"
    )
    return list(re.finditer(pattern, text))


def _phase3_4_definition_spans(text: str) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    for match in re.finditer(r"\(([^()]{2,80})\)", text):
        content = match.group(1).strip()
        acronym_pattern = (
            r"\b(?:[A-Z0-9]+(?:-[A-Z0-9]+)+s?|[A-Z]{2,}[A-Z0-9]*s?|"
            r"[A-Z][a-z]+-[A-Z]{2,}[A-Za-z0-9-]*s?|[A-Z][0-9]+s?)\b"
        )
        acronyms: list[str] = []
        for token in re.findall(acronym_pattern, content):
            if not token:
                continue
            candidates = [token]
            if token.endswith("s") and len(token) > 2 and token[-2].isupper():
                candidates.append(token[:-1])
            for candidate in candidates:
                if candidate not in acronyms:
                    acronyms.append(candidate)
        if not acronyms:
            continue
        prefix = text[max(0, match.start() - 140): match.start()].strip()
        if len(re.findall(r"[A-Za-z][A-Za-z-]+", prefix)) < 2:
            continue
        spans.append(
            {
                "start": match.start(),
                "end": match.end(),
                "acronyms": acronyms,
                "definition_context": " ".join((prefix + " (" + content + ")").split())[-220:],
            }
        )
    return spans


def _phase3_4_labeled_definition_spans(text: str) -> list[dict[str, Any]]:
    """Treat benchmark labels such as `MRT: maximum-ratio transmission` as definitions."""
    spans: list[dict[str, Any]] = []
    acronym_pattern = (
        r"(?:[A-Z0-9]+(?:-[A-Z0-9]+)+s?|[A-Z]{2,}[A-Z0-9]*s?|"
        r"[A-Z][a-z]+-[A-Z]{2,}[A-Za-z0-9-]*s?|[A-Z][0-9]+s?)"
    )
    label_pattern = re.compile(
        rf"(?=(?<![A-Za-z0-9-])(?P<acronym>{acronym_pattern})(?![A-Za-z0-9-])"
        rf"\s*:\s*(?P<definition>[^.;\n]{{3,160}}))",
    )
    for match in label_pattern.finditer(text):
        acronym = match.group("acronym").strip()
        definition = " ".join(match.group("definition").split())
        if not acronym or not definition:
            continue
        if len(re.findall(r"[A-Za-z][A-Za-z-]+", definition)) < 1:
            continue
        spans.append(
            {
                "start": match.start("acronym"),
                "end": match.end("definition"),
                "acronyms": [acronym],
                "definition_context": " ".join(text[match.start("acronym"): match.end("definition")].split())[:220],
                "definition_style": "labeled_definition",
            }
        )
    return spans


def _phase3_4_section_for_offset(section_offsets: list[tuple[str, int, int]], offset: int) -> str:
    for section_name, start, end in section_offsets:
        if start <= offset < end:
            return section_name
    return "full_paper"


def analyze_phase3_4_full_paper_abbreviations(section_tex: dict[str, str]) -> dict[str, Any]:
    """Check full-paper acronym first-use discipline across assembled prose sections."""
    ordered_sections = [
        ("abstract", section_tex.get("abstract", "")),
        ("introduction", section_tex.get("introduction", "")),
        ("system_model", section_tex.get("system_model", "")),
        ("proposed_solution", section_tex.get("proposed_solution", "")),
        ("numerical_results", section_tex.get("numerical_results", "")),
        ("conclusion", section_tex.get("conclusion", "")),
    ]
    chunks: list[str] = []
    offsets: list[tuple[str, int, int]] = []
    cursor = 0
    for section_name, tex in ordered_sections:
        plain = _phase3_4_plaintext_for_abbreviation_check(tex)
        if not plain.strip():
            continue
        header = f"\n[{section_name}]\n"
        chunk = header + plain + "\n"
        start = cursor
        cursor += len(chunk)
        offsets.append((section_name, start, cursor))
        chunks.append(chunk)
    text = "".join(chunks)
    definition_spans = _phase3_4_definition_spans(text) + _phase3_4_labeled_definition_spans(text)
    definition_by_token: dict[str, list[dict[str, Any]]] = {}
    for span in definition_spans:
        for acronym in span["acronyms"]:
            definition_by_token.setdefault(acronym, []).append(span)

    allowed = {
        "IEEE",
        "WCL",
        "WARA",
        "CUHKSZ",
        "LaTeX",
        "PDF",
        "JSON",
        "CSV",
        "YAML",
        "Fig",
        "Figs",
        "Sec",
        "Eq",
        "Eqs",
    }
    roman_section_tokens = {"I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"}

    def scope_for_offset(offset: int) -> str:
        section_name = _phase3_4_section_for_offset(offsets, offset)
        return "abstract" if section_name == "abstract" else "main_body"

    undefined: list[dict[str, Any]] = []
    first_uses: dict[str, dict[str, Any]] = {}
    for match in _phase3_4_acronym_candidates(text):
        token = match.group(0)
        if token in allowed or token in roman_section_tokens:
            continue
        if re.fullmatch(r"[PC]\d+", token):
            continue
        token_scope = scope_for_offset(match.start())
        first_use_key = f"{token_scope}:{token}"
        containing_definition = [
            span for span in definition_spans if span["start"] <= match.start() < span["end"] and token in span["acronyms"]
        ]
        if containing_definition:
            first_uses.setdefault(
                first_use_key,
                {
                    "section": _phase3_4_section_for_offset(offsets, match.start()),
                    "scope": token_scope,
                    "term": token,
                    "defined_on_first_use": True,
                    "definition_context": containing_definition[0]["definition_context"],
                },
            )
            continue
        prior_definitions = [
            span
            for span in definition_by_token.get(token, [])
            if span["end"] <= match.start() and scope_for_offset(int(span.get("start") or 0)) == token_scope
        ]
        if prior_definitions:
            first_uses.setdefault(
                first_use_key,
                {
                    "section": _phase3_4_section_for_offset(offsets, match.start()),
                    "scope": token_scope,
                    "term": token,
                    "defined_on_first_use": False,
                    "definition_context": prior_definitions[0]["definition_context"],
                },
            )
            continue
        if "-" in token:
            components = [part for part in token.split("-") if part]
            if len(components) >= 2 and all(
                component in allowed
                or any(
                    span["end"] <= match.start()
                    and scope_for_offset(int(span.get("start") or 0)) == token_scope
                    for span in definition_by_token.get(component, [])
                )
                for component in components
            ):
                first_uses.setdefault(
                    first_use_key,
                    {
                        "section": _phase3_4_section_for_offset(offsets, match.start()),
                        "scope": token_scope,
                        "term": token,
                        "defined_on_first_use": False,
                        "definition_context": "compound acronym formed from previously defined components",
                    },
                )
                continue
        if first_use_key not in first_uses:
            section_name = _phase3_4_section_for_offset(offsets, match.start())
            context = " ".join(text[max(0, match.start() - 90): match.end() + 90].split())
            first_uses[first_use_key] = {
                "section": section_name,
                "scope": token_scope,
                "term": token,
                "defined_on_first_use": False,
                "definition_context": "",
            }
            undefined.append(
                {
                    "term": token,
                    "section": section_name,
                    "scope": token_scope,
                    "context": context,
                    "message": f"`{token}` appears before a full-name-plus-acronym definition.",
                    "suggested_full_form": COMMON_WIRELESS_ABBREVIATION_EXPANSIONS.get(token, ""),
                    "suggested_fix": (
                        f"Define first use as {COMMON_WIRELESS_ABBREVIATION_EXPANSIONS[token]} ({token})."
                        if token in COMMON_WIRELESS_ABBREVIATION_EXPANSIONS
                        else "Define the abbreviation on first use or remove it."
                    ),
                }
            )

    repeated_definitions: list[dict[str, Any]] = []
    for token, spans in sorted(definition_by_token.items()):
        if token in allowed or token in roman_section_tokens:
            continue
        if re.fullmatch(r"[PC]\d+", token):
            continue
        ordered_spans = sorted(
            [
                span
                for span in spans
                if str(span.get("definition_style", "")).strip() != "labeled_definition"
            ],
            key=lambda item: int(item.get("start") or 0),
        )
        spans_by_scope: dict[str, list[dict[str, Any]]] = {}
        for span in ordered_spans:
            spans_by_scope.setdefault(scope_for_offset(int(span.get("start") or 0)), []).append(span)
        for scope_name, scoped_spans in spans_by_scope.items():
            if len(scoped_spans) <= 1:
                continue
            first_span = scoped_spans[0]
            for repeat_span in scoped_spans[1:]:
                repeat_start = int(repeat_span.get("start") or 0)
                repeated_definitions.append(
                    {
                        "term": token,
                        "section": _phase3_4_section_for_offset(offsets, repeat_start),
                        "scope": scope_name,
                        "count": len(scoped_spans),
                        "first_definition_context": str(first_span.get("definition_context") or ""),
                        "repeated_definition_context": str(repeat_span.get("definition_context") or ""),
                        "message": f"`{token}` is defined more than once in the {scope_name} scope.",
                        "suggested_fix": (
                            "Keep the first full-name-plus-acronym definition in this scope, then use only "
                            "the acronym or the full phrase."
                        ),
                    }
                )

    return {
        "ok": not undefined and not repeated_definitions,
        "undefined_abbreviations": undefined,
        "repeated_abbreviation_definitions": repeated_definitions,
        "defined_abbreviations": sorted(definition_by_token),
        "first_uses": first_uses,
        "checked_sections": [name for name, _, _ in offsets],
    }


def analyze_phase3_4_full_paper_abbreviations_from_phase_dir(phase_dir: Path) -> dict[str, Any]:
    phase_dir = Path(phase_dir)
    return analyze_phase3_4_full_paper_abbreviations(
        {
            "abstract": read_text(phase_dir / "abstract.tex"),
            "introduction": read_text(phase_dir / "introduction.tex"),
            "system_model": read_text(phase_dir / "system_model_problem_formulation_section.tex"),
            "proposed_solution": read_text(phase_dir / "proposed_solution_section.tex"),
            "numerical_results": read_text(phase_dir / "numerical_results_section.tex"),
            "conclusion": read_text(phase_dir / "conclusion.tex"),
        }
    )


def _phase3_4_intro_word_count(text: str) -> int:
    body = re.sub(r"%.*", " ", text)
    body = re.sub(r"\\cite\{[^}]*\}", " citation ", body)
    body = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{[^{}]*\})?", " ", body)
    return len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", body))


def ensure_phase3_4_minimum_intro_words(tex: str, minimum_words: int = 0) -> str:
    return tex


def _extract_bib_field(raw_entry: str, field: str) -> str:
    pattern = rf"{re.escape(field)}\s*=\s*[{{\"](.*?)[}}\"]\s*,?\s*(?:\n|$)"
    match = re.search(pattern, raw_entry, flags=re.I | re.S)
    if not match:
        return ""
    return " ".join(match.group(1).replace("\n", " ").split()).strip()


def parse_bib_entries(bib_text: str) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for chunk in re.split(r"(?=^@\w+\{)", bib_text, flags=re.M):
        raw = chunk.strip()
        if not raw.startswith("@"):
            continue
        match = re.match(r"@(\w+)\{([^,]+),", raw, flags=re.S)
        if not match:
            continue
        entry_type, key = match.groups()
        entries.append(
            {
                "entry_type": entry_type.strip(),
                "key": key.strip(),
                "title": _extract_bib_field(raw, "title"),
                "author": _extract_bib_field(raw, "author"),
                "venue": _extract_bib_field(raw, "journal") or _extract_bib_field(raw, "booktitle"),
                "year": _extract_bib_field(raw, "year"),
                "volume": _extract_bib_field(raw, "volume"),
                "number": _extract_bib_field(raw, "number"),
                "pages": _extract_bib_field(raw, "pages"),
                "month": _extract_bib_field(raw, "month"),
                "doi": _extract_bib_field(raw, "doi"),
                "url": _extract_bib_field(raw, "url"),
                "arxiv_id": (
                    re.search(r"arXiv:(\d{4}\.\d{4,5}(?:v\d+)?)", raw, flags=re.I).group(1)
                    if re.search(r"arXiv:(\d{4}\.\d{4,5}(?:v\d+)?)", raw, flags=re.I)
                    else _extract_bib_field(raw, "eprint")
                ),
                "raw": raw.rstrip() + "\n",
            }
        )
    return entries


def _phase3_4_seminal_bibtex_value(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return " and ".join(str(item).strip() for item in value if str(item).strip())
    return str(value or "")


def _phase3_4_seminal_matches_to_bibtex(topic_literature_json_path: Path | None) -> str:
    if topic_literature_json_path is None or not topic_literature_json_path.exists():
        return ""
    payload = read_json(topic_literature_json_path) or {}
    matches = payload.get("seminal_matches", []) if isinstance(payload, dict) else []
    entries: list[str] = []
    for match in matches if isinstance(matches, list) else []:
        if not isinstance(match, dict):
            continue
        key = sanitize_bibtex_text(_phase3_4_seminal_bibtex_value(match.get("cite_key") or match.get("bib_key"))).replace(" ", "")
        title = sanitize_bibtex_text(_phase3_4_seminal_bibtex_value(match.get("title")))
        authors = normalize_bibtex_author_list(_phase3_4_seminal_bibtex_value(match.get("authors")))
        venue = sanitize_bibtex_text(_phase3_4_seminal_bibtex_value(match.get("venue")))
        year = sanitize_bibtex_text(_phase3_4_seminal_bibtex_value(match.get("year")))
        if not key or not title or not venue:
            continue
        entry_type = "inproceedings" if re.search(r"\b(proc\.|conference|workshop|symposium)\b", venue, flags=re.I) else "article"
        venue_field = "booktitle" if entry_type == "inproceedings" else "journal"
        lines = [
            f"@{entry_type}{{{key},",
            f"  title = {{{title}}},",
        ]
        if authors:
            lines.append(f"  author = {{{authors}}},")
        if year:
            lines.append(f"  year = {{{year}}},")
        lines.append(f"  {venue_field} = {{{venue}}},")
        doi = sanitize_bibtex_text(str(match.get("doi") or ""))
        if doi:
            lines.append(f"  doi = {{{doi}}},")
        url = sanitize_bibtex_text(str(match.get("url") or ""))
        if url:
            lines.append(f"  url = {{{url}}},")
        lines.append("}")
        entries.append("\n".join(lines))
    return "\n\n".join(entries).strip()


def _merge_phase3_4_bibtex_blocks(*blocks: str) -> str:
    merged: list[str] = []
    seen_keys: set[str] = set()
    for block in blocks:
        for raw in re.split(r"\n\s*\n(?=@)", str(block or "").strip()):
            entry = raw.strip()
            if not entry:
                continue
            match = re.search(r"@\w+\s*\{\s*([^,\s]+)", entry)
            key = match.group(1).strip().lower() if match else entry[:100].lower()
            if key in seen_keys:
                continue
            seen_keys.add(key)
            merged.append(entry)
    return ("\n\n".join(merged).strip() + "\n") if merged else ""


def _phase3_4_keyword_tokens(*texts: str) -> list[str]:
    stopwords = {
        "about",
        "above",
        "after",
        "algorithm",
        "analysis",
        "additional",
        "antennas",
        "based",
        "base",
        "between",
        "communications",
        "consider",
        "constraint",
        "contains",
        "data",
        "design",
        "downlink",
        "each",
        "entities",
        "evaluated",
        "from",
        "gain",
        "higher",
        "improves",
        "model",
        "method",
        "multiuser",
        "near",
        "optimization",
        "only",
        "performance",
        "problem",
        "proposed",
        "received",
        "results",
        "same",
        "settings",
        "shows",
        "signal",
        "single-antenna",
        "single-cell",
        "serves",
        "system",
        "station",
        "streams",
        "symbols",
        "tested",
        "topology",
        "transmitted",
        "under",
        "user",
        "users",
        "wireless",
        "with",
    }
    tokens: list[str] = []
    seen: set[str] = set()
    for text in texts:
        for token in re.findall(r"[A-Za-z][A-Za-z0-9\-]{3,}", text.lower()):
            if token in stopwords or token.isdigit():
                continue
            if token not in seen:
                seen.add(token)
                tokens.append(token)
    return tokens[:40]


def _phase3_4_reference_clusters(text: str) -> set[str]:
    lower = str(text or "").lower()
    patterns = {
        "isac": ("isac", "isacp", "integrated sensing", "sensing and communication", "sensing communication"),
        "wireless_power": ("wireless power", "power transfer", "powering", "rf power", "rf-power", "wpt", "energy harvesting", "swipt"),
        "near_field": ("near-field", "near field", "xl-mimo", "extremely large", "extra large", "holographic"),
        "mimo": ("mimo", "massive mimo", "multi-antenna", "antenna array"),
        "beamforming": ("beamforming", "precoding", "covariance", "transmit beam", "waveform"),
        "optimization": ("optimization", "resource allocation", "semidefinite", "sdr", "sdp", "sca", "wmmse"),
        "uav": ("uav", "unmanned aerial", "drone", "trajectory"),
        "ris": ("ris", "reconfigurable intelligent", "intelligent reflecting"),
        "cell_free": ("cell-free", "cell free", "distributed antenna"),
        "ofdm": ("ofdm", "ofdma", "orthogonal frequency division"),
        "noma": ("noma", "non-orthogonal multiple access"),
        "security": ("physical layer security", "secure communication", "secrecy"),
    }
    clusters: set[str] = set()
    for label, terms in patterns.items():
        if any(term in lower for term in terms):
            clusters.add(label)
    return clusters


def _phase3_4_scope_mismatch_count(reference_title: str, current_context: str) -> int:
    title = str(reference_title or "").lower()
    context = str(current_context or "").lower()
    mismatches = 0
    optional_mechanisms: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
        "ris": (
            ("ris", "reconfigurable intelligent", "intelligent reflecting"),
            ("ris-assisted", "ris aided", "reconfigurable intelligent", "intelligent reflecting", "phase-shift", "phase shift"),
        ),
        "uav": (
            ("uav", "unmanned aerial", "drone"),
            ("uav-aided", "uav assisted", "unmanned aerial", "drone", "trajectory"),
        ),
        "noma": (
            ("noma", "non-orthogonal multiple access"),
            ("noma", "non-orthogonal multiple access"),
        ),
        "security": (
            ("physical layer security", "secure communication", "secrecy"),
            ("physical layer security", "secure communication", "secrecy"),
        ),
        "cell_free": (
            ("cell-free", "cell free"),
            ("cell-free", "cell free", "distributed antenna"),
        ),
    }
    for title_terms, positive_context_terms in optional_mechanisms.values():
        title_has = any(term in title for term in title_terms)
        context_has = any(term in context for term in positive_context_terms)
        if title_has and not context_has:
            mismatches += 1
    return mismatches


def _phase3_4_entry_has_topic_overlap(entry: dict[str, str], context_text: str) -> bool:
    context_clusters = _phase3_4_reference_clusters(context_text)
    if not context_clusters:
        return True
    entry_text = " ".join(
        [
            str(entry.get("title", "")),
            str(entry.get("venue", "")),
            str(entry.get("raw", "")),
        ]
    )
    entry_clusters = _phase3_4_reference_clusters(entry_text)
    if not entry_clusters:
        return False
    service_clusters = {"isac", "wireless_power", "near_field", "mimo", "uav", "ris", "cell_free", "ofdm", "noma", "security"}
    active_service_clusters = context_clusters & service_clusters
    entry_service_clusters = entry_clusters & service_clusters
    if active_service_clusters:
        if active_service_clusters & entry_service_clusters:
            return True
        if "beamforming" in entry_clusters and entry_service_clusters:
            return True
        return False
    return bool(context_clusters & entry_clusters)


def _phase3_4_text_has_keyword(text: str, token: str) -> bool:
    token = token.strip().lower()
    if len(token) < 4:
        return False
    token_pattern = re.escape(token).replace(r"\-", r"[- ]")
    return bool(re.search(rf"(?<![a-z0-9]){token_pattern}(?![a-z0-9])", text.lower()))


def extract_phase1_focus_keys(*texts: str) -> list[str]:
    keys: set[str] = set()
    for text in texts:
        for token in re.findall(r"\b[a-z][a-z0-9]+20\d{2}[a-z0-9]*\b", text):
            keys.add(token.strip())
    return sorted(keys)


def _extract_markdown_section(text: str, heading: str) -> str:
    if not text.strip() or not heading.strip():
        return ""
    pattern = rf"(?ims)^##\s+{re.escape(heading)}\s*\n(.*?)(?=^##\s+|^#\s+|\Z)"
    match = re.search(pattern, text)
    if not match:
        return ""
    return compact_text(match.group(1).strip(), 900)


def select_reference_pool(
    entries: list[dict[str, str]],
    *,
    context_text: str,
    focus_keys: list[str],
    max_items: int = 32,
    min_items: int = 12,
) -> list[dict[str, str]]:
    keyword_tokens = _phase3_4_keyword_tokens(context_text)
    focus_set = {item.lower() for item in focus_keys}
    context_lower = context_text.lower()
    scored: list[tuple[int, dict[str, str]]] = []
    for entry in entries:
        title_lower = entry.get("title", "").lower()
        venue_lower = entry.get("venue", "").lower()
        key_lower = entry.get("key", "").lower()
        score = 0
        if key_lower in focus_set:
            score += 12
        shared_token_hits = sum(1 for token in keyword_tokens if _phase3_4_text_has_keyword(title_lower, token))
        if shared_token_hits == 0 and key_lower not in focus_set:
            continue
        if key_lower not in focus_set and not _phase3_4_entry_has_topic_overlap(entry, context_text):
            continue
        score += shared_token_hits
        if any(token in venue_lower for token in ["wireless", "communications", "signal processing", "vehicular", "jsac", "wcl", "tcom"]):
            score += 2
        if any(token in title_lower for token in ["survey", "tutorial", "overview", "optimization", "communications"]):
            score += 2
        if any(token in title_lower for token in ["comprehensive survey", "overview", "framework", "algorithm"]):
            score += 2
        if any(token in venue_lower for token in ["surveys", "tutorials", "transactions on wireless communications", "wireless communications letters"]):
            score += 2
        scope_mismatches = _phase3_4_scope_mismatch_count(title_lower, context_lower)
        if scope_mismatches and key_lower not in focus_set:
            continue
        score -= 7 * scope_mismatches
        if score <= 0:
            continue
        scored.append((score, entry))
    scored.sort(key=lambda item: (item[0], item[1].get("year", ""), item[1].get("key", "")), reverse=True)
    selected = [dict(entry) for _, entry in scored[:max_items]]

    # The strict pool above prevents topic drift, but it must not discard a
    # verified Phase-1 bibliography below the paper-level reference target.
    # Backfill with lower-ranked, still related entries from the supplied .bib
    # so Phase 3.4 fails only when the bibliography itself is insufficient.
    if len(entries) >= min_items and len(selected) < min_items and len(selected) < max_items:
        selected_keys = {str(item.get("key", "")).strip().lower() for item in selected}
        relaxed: list[tuple[int, dict[str, str]]] = []
        context_clusters = _phase3_4_reference_clusters(context_text)
        for entry in entries:
            key_lower = str(entry.get("key", "")).strip().lower()
            if not key_lower or key_lower in selected_keys:
                continue
            title_lower = entry.get("title", "").lower()
            venue_lower = entry.get("venue", "").lower()
            entry_text = " ".join([title_lower, venue_lower, str(entry.get("raw", "")).lower()])
            entry_clusters = _phase3_4_reference_clusters(entry_text)
            shared_token_hits = sum(1 for token in keyword_tokens if _phase3_4_text_has_keyword(title_lower, token))
            cluster_hits = len(context_clusters & entry_clusters)
            score = 2 * shared_token_hits + 3 * cluster_hits
            if _phase3_4_entry_has_topic_overlap(entry, context_text):
                score += 6
            if key_lower in focus_set:
                score += 20
            if any(token in venue_lower for token in ["ieee", "wireless", "communications", "signal processing", "vehicular", "jsac", "wcl", "tcom"]):
                score += 4
            if any(token in title_lower for token in ["sensing", "communication", "powering", "beamforming", "optimization", "resource allocation", "wireless"]):
                score += 3
            if str(entry.get("doi", "")).strip():
                score += 2
            if str(entry.get("venue", "")).strip():
                score += 1
            score -= 2 * _phase3_4_scope_mismatch_count(title_lower, context_lower)
            if score > 0:
                relaxed.append((score, entry))
        relaxed.sort(key=lambda item: (item[0], item[1].get("year", ""), item[1].get("key", "")), reverse=True)
        for _, entry in relaxed:
            if len(selected) >= min(min_items, max_items):
                break
            key_lower = str(entry.get("key", "")).strip().lower()
            if key_lower in selected_keys:
                continue
            selected.append(dict(entry))
            selected_keys.add(key_lower)
    return selected[:max_items]


def _phase3_4_valid_reference_count(verified_reference_bank: list[dict[str, Any]]) -> int:
    return sum(
        1
        for item in verified_reference_bank
        if phase3_4_reference_is_final_usable(item)
    )


def sanitize_bibtex_text(text: str) -> str:
    replacements = {
        "&amp;": r"\&",
        "–": "-",
        "—": "-",
        "’": "'",
        "‘": "'",
        "“": '"',
        "”": '"',
        "é": "e",
        "è": "e",
        "á": "a",
        "à": "a",
        "ó": "o",
        "ö": "o",
        "ü": "u",
        "ñ": "n",
    }
    cleaned = text
    for old, new in replacements.items():
        cleaned = cleaned.replace(old, new)
    cleaned = cleaned.encode("ascii", "ignore").decode("ascii")
    return cleaned


def normalize_bibtex_author_list(authors: str) -> str:
    """Normalize common metadata author strings to BibTeX's `and` separator."""
    text = sanitize_bibtex_text(authors or "").strip()
    if not text:
        return ""
    if "," not in text and re.search(r"\s+and\s+", text, flags=re.I):
        return re.sub(r"\s+and\s+", " and ", text, flags=re.I)
    if text.count(",") < 2 and ", and " not in text.lower():
        return text
    parts = [part.strip() for part in text.split(",") if part.strip()]
    cleaned_parts: list[str] = []
    for part in parts:
        part = re.sub(r"^(and|&)\s+", "", part.strip(), flags=re.I)
        if part:
            cleaned_parts.append(part)
    return " and ".join(cleaned_parts) if len(cleaned_parts) >= 2 else text


def extract_citation_keys_from_tex(tex: str) -> list[str]:
    keys: list[str] = []
    for group in re.findall(r"\\cite\w*\{([^}]*)\}", tex):
        for item in group.split(","):
            key = item.strip()
            if key and key not in keys:
                keys.append(key)
    return keys


def normalize_reference_text(text: str) -> str:
    cleaned = sanitize_bibtex_text(text or "").lower()
    cleaned = re.sub(r"[^a-z0-9]+", " ", cleaned)
    return " ".join(cleaned.split())


def _phase3_4_reference_identity(item: dict[str, Any]) -> str:
    doi = re.sub(r"\s+", "", str(item.get("doi", "")).strip().lower())
    if doi:
        return f"doi:{doi}"
    title = normalize_reference_text(str(item.get("final_title", "") or item.get("candidate_title", "")))
    if title:
        return f"title:{title}"
    key = str(item.get("final_bib_key", "") or item.get("candidate_key", "")).strip().lower()
    return f"key:{key}" if key else ""


def _phase3_4_reference_quality_rank(item: dict[str, Any]) -> tuple[int, int, int, int, int]:
    status = str(item.get("verification_status", "")).strip().lower()
    source_type = str(item.get("source_type", "")).strip().lower()
    publisher_venue = (str(item.get("publisher", "")) + " " + str(item.get("venue", ""))).lower()
    status_rank = {
        "verified_published": 4,
        "replaced_by_published_version": 3,
        "arxiv_only": 1,
    }.get(status, 0)
    metadata_rank = 1 if not _phase3_4_reference_metadata_issues(item) else 0
    doi_rank = 1 if str(item.get("doi", "")).strip() else 0
    source_rank = {"journal": 3, "conference": 2, "book": 1, "standard": 1, "arxiv": 0}.get(source_type, 0)
    ieee_rank = 1 if "ieee" in publisher_venue else 0
    return metadata_rank, status_rank, doi_rank, source_rank, ieee_rank


def dedupe_phase3_4_references_by_identity(
    references: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep one final-citation candidate per DOI/title identity.

    The LLM should not have to decide whether two keys are the same paper.  We
    collapse exact DOI/title duplicates before prompts are assembled and again
    before final BibTeX rendering.
    """

    selected: list[dict[str, Any]] = []
    identity_to_index: dict[str, int] = {}
    notes: list[dict[str, Any]] = []
    for position, item in enumerate(references):
        if not isinstance(item, dict):
            continue
        identity = _phase3_4_reference_identity(item)
        if not identity:
            selected.append(dict(item))
            continue
        current_index = identity_to_index.get(identity)
        candidate = dict(item)
        if current_index is None:
            identity_to_index[identity] = len(selected)
            selected.append(candidate)
            continue
        kept = selected[current_index]
        candidate_rank = _phase3_4_reference_quality_rank(candidate)
        kept_rank = _phase3_4_reference_quality_rank(kept)
        candidate_key = str(candidate.get("final_bib_key", "") or candidate.get("candidate_key", ""))
        kept_key = str(kept.get("final_bib_key", "") or kept.get("candidate_key", ""))
        if candidate_rank > kept_rank:
            selected[current_index] = candidate
            notes.append(
                {
                    "dedupe_identity": identity,
                    "kept_key": candidate_key,
                    "removed_key": kept_key,
                    "reason": "higher verified metadata quality for the same DOI/title identity",
                    "position": position,
                }
            )
        else:
            notes.append(
                {
                    "dedupe_identity": identity,
                    "kept_key": kept_key,
                    "removed_key": candidate_key,
                    "reason": "duplicate DOI/title identity",
                    "position": position,
                }
            )
    return selected, notes


def title_similarity(a: str, b: str) -> float:
    na = normalize_reference_text(a)
    nb = normalize_reference_text(b)
    if not na or not nb:
        return 0.0
    seq = difflib.SequenceMatcher(None, na, nb).ratio()
    ta = set(na.split())
    tb = set(nb.split())
    overlap = len(ta & tb) / max(len(ta | tb), 1)
    return 0.7 * seq + 0.3 * overlap


def _extract_markdown_table_first_column(text: str) -> list[str]:
    values: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|") or stripped.count("|") < 2:
            continue
        if re.match(r"^\|\s*-", stripped):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if not cells:
            continue
        cell0 = re.sub(r"[*`]", "", cells[0]).strip()
        if cell0 and cell0.lower() not in {"scenario", "gap", "variant"} and cell0 not in values:
            values.append(cell0)
    return values


def _extract_claim_bullets(text: str, max_items: int = 5) -> list[str]:
    items: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if re.match(r"^[-*]\s+", stripped):
            items.append(re.sub(r"^[-*]\s+", "", stripped))
        elif stripped.lower().startswith("**claimed contribution**"):
            parts = stripped.split("**", 2)
            if parts:
                items.append(stripped.split("**")[-1].strip(": ").strip())
    deduped: list[str] = []
    for item in items:
        item = " ".join(item.split())
        if item and item not in deduped:
            deduped.append(item)
    return deduped[:max_items]


def build_phase3_4_source_map(
    *,
    phase1_files: list[Path],
    phase2_technical_files: list[Path],
    phase2_results_files: list[Path],
) -> dict[str, Any]:
    return {
        "literature_source": {
            "role": "background_and_related_work_context",
            "files": [str(path) for path in phase1_files],
        },
        "technical_source": {
            "role": "current_problem_method_and_benchmark",
            "files": [str(path) for path in phase2_technical_files],
        },
        "results_source": {
            "role": "method_names_evidence_and_claim_strength",
            "files": [str(path) for path in phase2_results_files],
        },
    }


def _extract_algorithmic_tools(*texts: str) -> list[str]:
    known_tools = [
        "WMMSE",
        "SCA",
        "alternating optimization",
        "AO",
        "BCD",
        "fractional programming",
        "quadratic transform",
        "SDR",
        "SOCP",
        "trust-region",
        "penalty method",
        "projection",
    ]
    combined = "\n".join(texts).lower()
    tools: list[str] = []
    for tool in known_tools:
        if tool.lower() in combined and tool not in tools:
            tools.append(tool)
    return tools


def build_phase3_4_introduction_facts(
    *,
    topic: str,
    synthesis_md: str,
    hypotheses_md: str,
    topic_score: dict[str, Any],
    system_model_md: str,
    problem_formulation_md: str,
    reformulation_path_md: str,
    algorithm_md: str,
    benchmark_definition_md: str,
    phase25_summary: dict[str, Any],
    experiment_plan: dict[str, Any],
    method_naming_payload: dict[str, Any],
    phase3_2_manifest: dict[str, Any],
    phase3_3_manifest: dict[str, Any],
    candidate_references: list[dict[str, str]],
) -> dict[str, Any]:
    proposed, baseline = _phase3_4_select_methods(method_naming_payload)
    overall = phase25_summary.get("overall", {}) if isinstance(phase25_summary, dict) else {}
    topic_selection_rationale = {
        "recommended_title": str(topic_score.get("recommended_title", "")).strip() or "not specified",
        "recommended_candidate": str(topic_score.get("recommended_candidate", "")).strip() or "not specified",
        "top_strengths": topic_score.get("top_strengths", []) if isinstance(topic_score.get("top_strengths", []), list) else [],
        "main_risks": topic_score.get("main_risks", []) if isinstance(topic_score.get("main_risks", []), list) else [],
        "minimum_validation_plan": topic_score.get("minimum_validation_plan", []) if isinstance(topic_score.get("minimum_validation_plan", []), list) else [],
        "selection_justification": compact_text(str(topic_score.get("justification", "")).strip(), 900) or "not specified",
        "decision_summary": _extract_markdown_section(hypotheses_md, "Decision Summary") or "not specified",
        "why_this_family_is_appropriate": _extract_markdown_section(hypotheses_md, "Why this family is appropriate") or "not specified",
        "claimed_contribution_candidate": _extract_markdown_section(hypotheses_md, "Claimed contribution") or "not specified",
        "novelty_delta_candidate": _extract_markdown_section(hypotheses_md, "Novelty delta vs prior art") or "not specified",
    }
    return {
        "background": {
            "motivation": compact_text(synthesis_md or topic or "not specified", 1200) or "not specified",
            "related_work_categories": _extract_markdown_table_first_column(synthesis_md)[:10] or ["not specified"],
            "literature_gap_candidates": _extract_markdown_table_first_column(hypotheses_md)[:10] or _extract_claim_bullets(hypotheses_md) or ["not specified"],
            "topic_selection_rationale": topic_selection_rationale,
            "candidate_references": [
                {
                    "key": item.get("key", ""),
                    "title": item.get("title", ""),
                    "venue": item.get("venue", ""),
                    "year": item.get("year", ""),
                }
                for item in candidate_references[:20]
            ],
        },
        "technical_scope": {
            "topic": topic or "not specified",
            "problem": compact_text(problem_formulation_md, 900) or "not specified",
            "proposed_method": {
                "display_name_long": proposed.get("display_name_long") or "not specified",
                "display_name_short": proposed.get("display_name_short") or "not specified",
            },
            "benchmark": {
                "display_name_long": baseline.get("display_name_long") or "not specified",
                "display_name_short": baseline.get("display_name_short") or "not specified",
            },
            "algorithmic_tools": _extract_algorithmic_tools(reformulation_path_md, algorithm_md, benchmark_definition_md) or ["not specified"],
            "contribution_claims": _extract_claim_bullets(hypotheses_md + "\n" + algorithm_md) or ["not specified"],
            "paper_objective": compact_text(problem_formulation_md + "\n" + reformulation_path_md, 1200) or "not specified",
            "benchmark_limitation": compact_text(benchmark_definition_md, 800) or "not specified",
        },
        "result_constraints": {
            "allowed_numerical_claims": [
                {
                    "proposed_win_rate": overall.get("proposed_win_rate", "not specified"),
                    "proposed_mean_relative_gain": overall.get("proposed_mean_relative_gain", "not specified"),
                    "proposed_median_relative_gain": overall.get("proposed_median_relative_gain", "not specified"),
                }
            ],
            "method_names": {
                "proposed_long": proposed.get("display_name_long") or "not specified",
                "proposed_short": proposed.get("display_name_short") or "not specified",
                "benchmark_long": baseline.get("display_name_long") or "not specified",
                "benchmark_short": baseline.get("display_name_short") or "not specified",
            },
            "claim_strength_limits": [
                "Use only numerical values, trends, and figure evidence from results_source manifests and numerical-results text.",
                "Use 'under the considered settings' or 'in the evaluated scenarios' for empirical claims.",
                "Do not claim monotonicity, statistical significance, guarantees, or universal superiority unless explicitly supported.",
            ],
            "paper_claims_to_test": experiment_plan.get("paper_claims_to_test", []) if isinstance(experiment_plan, dict) else [],
            "phase3_2_numbers_used": phase3_2_manifest.get("numbers_used", []) if isinstance(phase3_2_manifest, dict) else [],
            "phase3_3_numbers_used": phase3_3_manifest.get("numbers_used", []) if isinstance(phase3_3_manifest, dict) else [],
            "minimum_reference_target": 12,
            "preferred_reference_target": 14,
        },
    }


def build_phase3_4_current_paper_brief(
    *,
    topic: str,
    system_model_md: str,
    problem_formulation_md: str,
    reformulation_path_md: str,
    algorithm_md: str,
    benchmark_definition_md: str,
    phase25_summary: dict[str, Any],
    experiment_plan: dict[str, Any],
    method_naming_payload: dict[str, Any],
) -> dict[str, Any]:
    proposed, baseline = _phase3_4_select_methods(method_naming_payload)
    current_context = "\n".join(
        [
            topic,
            system_model_md,
            problem_formulation_md,
            reformulation_path_md,
            algorithm_md,
            benchmark_definition_md,
        ]
    )
    overall = phase25_summary.get("overall", {}) if isinstance(phase25_summary, dict) else {}

    topic_axes = [
        compact_text(item, 180)
        for item in _extract_claim_bullets(
            "\n".join([problem_formulation_md, reformulation_path_md, algorithm_md, benchmark_definition_md]),
            max_items=6,
        )
        if compact_text(item, 180)
    ]
    if not topic_axes:
        topic_axes = [
            "the wireless system architecture specified in the current system model",
            "the optimization variables and constraints specified in the current problem formulation",
            "the method and benchmark specified in the current algorithm files",
        ]

    evidence_summary = {
        "proposed_win_rate": overall.get("proposed_win_rate", "not specified"),
        "proposed_mean_relative_gain": overall.get("proposed_mean_relative_gain", "not specified"),
        "proposed_median_relative_gain": overall.get("proposed_median_relative_gain", "not specified"),
        "paper_claims_to_test": experiment_plan.get("paper_claims_to_test", []) if isinstance(experiment_plan, dict) else [],
    }

    return {
        "priority": (
            "This brief is the primary content plan for the Introduction. Use the literature sources "
            "and reference bank only as support for the axes below."
        ),
        "paper_topic": topic or "not specified",
        "central_research_question": (
            "How should the current paper's system variables be jointly designed under the exact "
            "constraints and evidence metrics specified by technical_source?"
        ),
        "must_center_on": topic_axes,
        "system_scope_from_current_paper": compact_text(system_model_md, 1000) or "not specified",
        "optimization_scope_from_current_paper": compact_text(problem_formulation_md, 1000) or "not specified",
        "method_scope_from_current_paper": {
            "proposed_method_long": proposed.get("display_name_long") or "not specified",
            "proposed_method_short": proposed.get("display_name_short") or "not specified",
            "benchmark_long": baseline.get("display_name_long") or "not specified",
            "benchmark_short": baseline.get("display_name_short") or "not specified",
            "reformulation_route": compact_text(reformulation_path_md, 800) or "not specified",
            "algorithm_route": compact_text(algorithm_md, 900) or "not specified",
            "benchmark_definition": compact_text(benchmark_definition_md, 700) or "not specified",
        },
        "positive_introduction_plan": [
            "Open with the practical value of the current topic and system scope.",
            "Review only prior-work themes that directly support the current topic axes.",
            "State the gap in terms of the current optimization coupling, modeling choice, or benchmark limitation.",
            "Describe the proposed method and benchmark using the method names supplied by results_source.",
            "Summarize empirical claims only through the allowed evidence summary.",
        ],
        "reference_use_policy": [
            "References are supporting evidence, not the source of the paper scope.",
            "A cited reference may motivate or contextualize a topic axis, but it must not introduce a different system architecture as the paper's setting.",
            "If a verified reference belongs to a broader adjacent literature, cite it only for the narrow claim that matches this brief.",
        ],
        "allowed_evidence_summary": evidence_summary,
    }


def _http_get_json(url: str) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "wara/1.0 (reference verification)",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=None) as response:
        return json.loads(response.read().decode("utf-8", errors="ignore"))


def _crossref_year(message: dict[str, Any]) -> str:
    for key in ["published-print", "published-online", "issued", "created"]:
        payload = message.get(key) if isinstance(message, dict) else None
        if not isinstance(payload, dict):
            continue
        parts = payload.get("date-parts")
        if isinstance(parts, list) and parts and isinstance(parts[0], list) and parts[0]:
            year = parts[0][0]
            if year:
                return str(year)
    return ""


def _crossref_authors(message: dict[str, Any]) -> str:
    authors = []
    for item in message.get("author", []) if isinstance(message, dict) else []:
        if not isinstance(item, dict):
            continue
        given = str(item.get("given", "")).strip()
        family = str(item.get("family", "")).strip()
        name = " ".join(part for part in [given, family] if part).strip()
        if not name:
            name = str(item.get("name", "")).strip()
        if name:
            authors.append(name)
    return " and ".join(authors)


def _crossref_venue(message: dict[str, Any]) -> str:
    short_container = message.get("short-container-title", []) if isinstance(message, dict) else []
    if isinstance(short_container, list) and short_container:
        short_name = str(short_container[0]).strip()
        if short_name:
            return short_name
    container = message.get("container-title", []) if isinstance(message, dict) else []
    if isinstance(container, list) and container:
        return str(container[0]).strip()
    return ""


def _crossref_publisher(message: dict[str, Any]) -> str:
    publisher = str(message.get("publisher", "") if isinstance(message, dict) else "").strip()
    return publisher


def _crossref_month(message: dict[str, Any]) -> str:
    month_map = {
        1: "Jan.", 2: "Feb.", 3: "Mar.", 4: "Apr.", 5: "May.", 6: "Jun.",
        7: "Jul.", 8: "Aug.", 9: "Sep.", 10: "Oct.", 11: "Nov.", 12: "Dec.",
    }
    for key in ["published-print", "published-online", "issued", "created"]:
        payload = message.get(key) if isinstance(message, dict) else None
        if not isinstance(payload, dict):
            continue
        parts = payload.get("date-parts")
        if isinstance(parts, list) and parts and isinstance(parts[0], list) and len(parts[0]) >= 2:
            month_value = parts[0][1]
            if isinstance(month_value, int) and month_value in month_map:
                return month_map[month_value]
    return ""


def _crossref_volume(message: dict[str, Any]) -> str:
    return sanitize_bibtex_text(str(message.get("volume", "") if isinstance(message, dict) else "").strip())


def _crossref_number(message: dict[str, Any]) -> str:
    value = str(message.get("issue", "") if isinstance(message, dict) else "").strip()
    if not value:
        value = str(message.get("number", "") if isinstance(message, dict) else "").strip()
    return sanitize_bibtex_text(value)


def _crossref_pages(message: dict[str, Any]) -> str:
    value = str(message.get("page", "") if isinstance(message, dict) else "").strip()
    value = sanitize_bibtex_text(value)
    value = value.replace("–", "--").replace("-", "--")
    return re.sub(r"--+", "--", value)


def _crossref_address(message: dict[str, Any]) -> str:
    if not isinstance(message, dict):
        return ""
    event = message.get("event")
    if isinstance(event, dict):
        location = str(event.get("location", "")).strip()
        if location:
            return sanitize_bibtex_text(location)
    location = str(message.get("publisher-location", "")).strip()
    if location:
        return sanitize_bibtex_text(location)
    return ""


def normalize_ieee_venue_name(venue: str, source_type: str = "") -> str:
    raw = sanitize_bibtex_text(venue or "").strip()
    if not raw:
        return raw
    exact = {
        "IEEE Transactions on Wireless Communications": "IEEE Trans. Wireless Commun.",
        "IEEE Trans. Wireless Commun.": "IEEE Trans. Wireless Commun.",
        "IEEE Transactions on Communications": "IEEE Trans. Commun.",
        "IEEE Trans. Commun.": "IEEE Trans. Commun.",
        "IEEE Transactions on Vehicular Technology": "IEEE Trans. Veh. Technol.",
        "IEEE Trans. Veh. Technol.": "IEEE Trans. Veh. Technol.",
        "IEEE Transactions on Signal Processing": "IEEE Trans. Signal Process.",
        "IEEE Trans. Signal Process.": "IEEE Trans. Signal Process.",
        "IEEE Wireless Communications Letters": "IEEE Wireless Commun. Lett.",
        "IEEE Wireless Commun. Lett.": "IEEE Wireless Commun. Lett.",
        "IEEE Communications Letters": "IEEE Commun. Lett.",
        "IEEE Commun. Lett.": "IEEE Commun. Lett.",
        "IEEE Journal on Selected Areas in Communications": "IEEE J. Sel. Areas Commun.",
        "IEEE J. Sel. Areas Commun.": "IEEE J. Sel. Areas Commun.",
        "IEEE Journal of Selected Topics in Signal Processing": "IEEE J. Sel. Topics Signal Process.",
        "IEEE J. Sel. Topics Signal Process.": "IEEE J. Sel. Topics Signal Process.",
        "IEEE Signal Processing Magazine": "IEEE Signal Process. Mag.",
        "IEEE Signal Process. Mag.": "IEEE Signal Process. Mag.",
        "IEEE Communications Magazine": "IEEE Commun. Mag.",
        "IEEE Commun. Mag.": "IEEE Commun. Mag.",
        "IEEE Communications Surveys & Tutorials": "IEEE Commun. Surveys Tuts.",
        "IEEE Communications Surveys \\& Tutorials": "IEEE Commun. Surveys Tuts.",
        "IEEE Commun. Surveys Tuts.": "IEEE Commun. Surveys Tuts.",
        "IEEE Commun. Surv. Tutorials": "IEEE Commun. Surveys Tuts.",
        "IEEE Access": "IEEE Access",
        "IEEE Transactions on Cognitive Communications and Networking": "IEEE Trans. Cogn. Commun. Netw.",
        "IEEE Trans. Cogn. Commun. Netw.": "IEEE Trans. Cogn. Commun. Netw.",
        "2025 IEEE 26th International Workshop on Signal Processing and Artificial Intelligence for Wireless Communications (SPAWC)": "Proc. IEEE SPAWC",
        "2025 IEEE 26th International Workshop on Signal Processing Advances in Wireless Communications (SPAWC)": "Proc. IEEE SPAWC",
        "Proc. IEEE SPAWC": "Proc. IEEE SPAWC",
        "IEEE Consumer Communications and Networking Conference": "Proc. IEEE Consumer Commun. Netw. Conf. (CCNC)",
        "Proc. IEEE Consumer Commun. Netw. Conf. (CCNC)": "Proc. IEEE Consumer Commun. Netw. Conf. (CCNC)",
    }
    if raw in exact:
        raw = exact[raw]
    raw = raw.replace(r"\&", "&")
    return raw.replace("&", r"\&")


def protect_bibtex_title_case(title: str) -> str:
    text = sanitize_bibtex_text(title or "")
    known_tokens = [
        "MIMO", "MISO", "WMMSE", "SCA", "AO", "BCD", "QoS", "CRB", "SINR", "CSI",
    ]
    protected = text
    for token in sorted(known_tokens, key=len, reverse=True):
        protected = re.sub(rf"(?<!\{{)\b{re.escape(token)}\b(?!\}})", "{" + token + "}", protected)
    protected = re.sub(r"\b([A-Z]{2,}(?:-[A-Z0-9]{2,})*)\b", lambda m: "{" + m.group(1) + "}", protected)
    protected = re.sub(r"\b([A-Z]{2,}[A-Za-z0-9-]+)\b", lambda m: "{" + m.group(1) + "}", protected)
    for _ in range(4):
        prior = protected
        protected = re.sub(r"\{+\s*([A-Za-z0-9]{2,})\s*\}+", r"{\1}", protected)
        protected = re.sub(r"\{+\s*([A-Za-z0-9]{2,})\s*\}+-([A-Za-z0-9-]+)", r"{\1}-\2", protected)
        protected = re.sub(r"([A-Za-z0-9-]+)-\{+\s*([A-Za-z0-9]{2,})\s*\}+", r"\1-{\2}", protected)
        protected = re.sub(r"\{([A-Za-z0-9]+)\}-([A-Za-z0-9-]+)\}", r"{\1}-\2", protected)
        protected = re.sub(r"\{([A-Za-z0-9]+)\}-([A-Za-z0-9-]+)\s", r"{\1}-\2 ", protected)
        protected = protected.replace("{{{", "{").replace("}}}", "}")
        if protected == prior:
            break
    return protected


def normalize_bib_month(value: str) -> str:
    raw = sanitize_bibtex_text(value or "").strip().strip(".")
    if not raw:
        return ""
    month_map = {
        "1": "Jan.", "jan": "Jan.", "january": "Jan.",
        "2": "Feb.", "feb": "Feb.", "february": "Feb.",
        "3": "Mar.", "mar": "Mar.", "march": "Mar.",
        "4": "Apr.", "apr": "Apr.", "april": "Apr.",
        "5": "May.", "may": "May.",
        "6": "Jun.", "jun": "Jun.", "june": "Jun.",
        "7": "Jul.", "jul": "Jul.", "july": "Jul.",
        "8": "Aug.", "aug": "Aug.", "august": "Aug.",
        "9": "Sep.", "sep": "Sep.", "sept": "Sep.", "september": "Sep.",
        "10": "Oct.", "oct": "Oct.", "october": "Oct.",
        "11": "Nov.", "nov": "Nov.", "november": "Nov.",
        "12": "Dec.", "dec": "Dec.", "december": "Dec.",
    }
    if raw.lower() in month_map:
        return month_map[raw.lower()]
    if len(raw) >= 3:
        return raw[:1].upper() + raw[1:3].lower() + "."
    return raw


def normalize_bib_address(value: str) -> str:
    """Clean publisher location strings before BibTeX rendering.

    Crossref sometimes returns conference locations as repeated city/country
    pairs, e.g., "Singapore, Singapore". IEEEtran will print that literally, so
    collapse repeated comma-separated components while preserving meaningful
    pairs such as "Paris, France".
    """

    raw = sanitize_bibtex_text(value or "").strip().strip(",")
    if not raw:
        return ""
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts:
        return ""
    cleaned: list[str] = []
    seen: set[str] = set()
    for part in parts:
        norm = re.sub(r"[^a-z0-9]+", "", part.lower())
        if norm and norm in seen:
            continue
        if norm:
            seen.add(norm)
        cleaned.append(part)
    return ", ".join(cleaned)


def _phase3_4_reference_metadata_issues(item: dict[str, Any]) -> list[str]:
    """Return blocking metadata issues for final bibliography eligibility."""

    if not isinstance(item, dict):
        return ["invalid_reference_record"]
    status = str(item.get("verification_status", "")).strip().lower()
    source_type = str(item.get("source_type", "")).strip().lower()
    key = str(item.get("final_bib_key", "") or item.get("candidate_key", "")).strip()
    title = str(item.get("final_title", "") or item.get("candidate_title", "")).strip()
    venue = str(item.get("venue", "")).strip()
    year = str(item.get("year", "")).strip()
    doi = str(item.get("doi", "")).strip()
    volume = str(item.get("volume", "")).strip()
    pages = str(item.get("pages", "")).strip()
    month = normalize_bib_month(str(item.get("month", "")).strip())
    issues: list[str] = []
    if not key:
        issues.append("missing_bib_key")
    if not title:
        issues.append("missing_title")
    if not venue:
        issues.append("missing_venue")
    if not year:
        issues.append("missing_year")
    if status in {"verified_published", "replaced_by_published_version"}:
        # A final IEEE-style bibliography should not accept old seed entries
        # that merely name an IEEE venue. DOI-backed metadata is the reliable
        # source used to fill volume/issue/pages/month.
        if not doi:
            issues.append("missing_doi_for_verified_published_reference")
        if source_type == "journal":
            if not volume:
                issues.append("missing_journal_volume")
            if not pages:
                issues.append("missing_journal_pages")
            if not month:
                issues.append("missing_journal_month")
        elif source_type == "conference":
            if not pages:
                issues.append("missing_conference_pages")
    elif status == "arxiv_only":
        if not (str(item.get("arxiv_id", "")).strip() or str(item.get("doi", "")).strip()):
            issues.append("missing_arxiv_identifier")
    else:
        issues.append("unverified_reference_status")
    return issues


def phase3_4_reference_is_final_usable(item: dict[str, Any]) -> bool:
    valid_statuses = {"verified_published", "replaced_by_published_version", "arxiv_only"}
    if not isinstance(item, dict) or not item.get("included_in_final_bib"):
        return False
    if str(item.get("verification_status", "")).strip().lower() not in valid_statuses:
        return False
    return not _phase3_4_reference_metadata_issues(item)


def _crossref_source_type(message: dict[str, Any]) -> str:
    kind = str(message.get("type", "") if isinstance(message, dict) else "").strip().lower()
    if "journal" in kind:
        return "journal"
    if "proceedings" in kind or "conference" in kind:
        return "conference"
    if "book" in kind:
        return "book"
    if "standard" in kind:
        return "standard"
    return "other"


def _peer_reviewed_from_record(source_type: str, publisher: str, venue: str) -> bool:
    if source_type in {"journal", "conference", "book", "standard"}:
        return True
    venue_lower = (venue or "").lower()
    publisher_lower = (publisher or "").lower()
    return any(token in venue_lower or token in publisher_lower for token in ["ieee", "acm", "springer", "elsevier", "wiley"])


def _build_bibtex_from_reference(entry: dict[str, Any]) -> str:
    raw_bib = sanitize_bibtex_text(str(entry.get("bibtex", "")))

    def entry_field(name: str, bib_field: str | None = None) -> str:
        value = entry.get(name, "")
        if value is None:
            value = ""
        text = sanitize_bibtex_text(str(value)).strip()
        if text and text.lower() not in {"none", "null"}:
            return text
        return sanitize_bibtex_text(_extract_bib_field(raw_bib, bib_field or name))

    bib_key = sanitize_bibtex_text(str(entry.get("final_bib_key", "ref"))).replace(" ", "")
    source_type = str(entry.get("source_type", "journal"))
    entry_type = "@article" if source_type in {"journal", "arxiv"} else "@inproceedings"
    title = protect_bibtex_title_case(entry_field("final_title", "title"))
    # BibTeX requires names to be separated with "and"; Crossref-style
    # metadata strings often use commas, so prefer a verified raw BibTeX author
    # field whenever it is available.
    authors = normalize_bibtex_author_list(_extract_bib_field(raw_bib, "author")) or normalize_bibtex_author_list(
        entry_field("authors", "author")
    )
    if source_type == "conference":
        raw_venue = _extract_bib_field(raw_bib, "booktitle") or entry_field("venue", "booktitle")
    else:
        raw_venue = entry_field("venue", "journal")
    if not raw_venue:
        raw_venue = _extract_bib_field(raw_bib, "booktitle") or _extract_bib_field(raw_bib, "journal")
    venue = normalize_ieee_venue_name(raw_venue, source_type=source_type)
    year = entry_field("year")
    doi = entry_field("doi")
    volume = entry_field("volume")
    number = entry_field("number")
    pages = entry_field("pages").replace("–", "--").replace("-", "--")
    pages = re.sub(r"--+", "--", pages)
    # IEEEtran already formats proceedings compactly from booktitle, location,
    # year, and pages. Omitting month for conference entries avoids verbose
    # references such as "Jun. 2024" in WCL-style bibliography blocks.
    month = "" if source_type == "conference" else normalize_bib_month(entry_field("month"))
    address = ""
    arxiv_id = entry_field("arxiv_id", "eprint")
    lines = [f"{entry_type}{{{bib_key},", f"  author  = {{{authors}}},", f"  title   = {{{title}}},"]
    if source_type == "conference":
        lines.append(f"  booktitle = {{{venue}}},")
    else:
        lines.append(f"  journal = {{{venue}}},")
    if year:
        lines.append(f"  year    = {{{year}}},")
    if volume and source_type != "arxiv":
        lines.append(f"  volume  = {{{volume}}},")
    if number and source_type != "arxiv":
        lines.append(f"  number  = {{{number}}},")
    if pages and source_type != "arxiv":
        lines.append(f"  pages   = {{{pages}}},")
    if month:
        lines.append(f"  month   = {{{month}}},")
    if address and source_type == "conference":
        lines.append(f"  address = {{{address}}},")
    if doi and source_type != "arxiv":
        lines.append(f"  doi     = {{{doi}}},")
    if source_type == "arxiv":
        if arxiv_id:
            lines.append("  archivePrefix = {arXiv},")
            lines.append(f"  eprint  = {{{arxiv_id}}},")
        lines.append("  note    = {Preprint},")
    lines.append("}")
    return "\n".join(lines)


def _lookup_crossref_by_doi(doi: str) -> dict[str, Any]:
    encoded = urllib.parse.quote(doi, safe="")
    data = _http_get_json(f"https://api.crossref.org/works/{encoded}")
    return data.get("message", {}) if isinstance(data, dict) else {}


def _search_crossref_by_title(title: str, rows: int = 5) -> list[dict[str, Any]]:
    query = urllib.parse.quote(title)
    data = _http_get_json(f"https://api.crossref.org/works?rows={rows}&query.title={query}")
    message = data.get("message", {}) if isinstance(data, dict) else {}
    items = message.get("items", []) if isinstance(message, dict) else []
    return [item for item in items if isinstance(item, dict)]


def _pick_best_published_match(entry: dict[str, str], items: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, float]:
    candidate_title = entry.get("title", "")
    candidate_year = str(entry.get("year", "")).strip()
    best_item: dict[str, Any] | None = None
    best_score = 0.0
    for item in items:
        item_title = " ".join(item.get("title", [])) if isinstance(item.get("title"), list) else str(item.get("title", ""))
        similarity = title_similarity(candidate_title, item_title)
        year = _crossref_year(item)
        if year and candidate_year and year == candidate_year:
            similarity += 0.02
        source_type = _crossref_source_type(item)
        if source_type in {"journal", "conference"}:
            similarity += 0.03
        venue_lower = _crossref_venue(item).lower()
        publisher_lower = _crossref_publisher(item).lower()
        if "ieee" in venue_lower or "ieee" in publisher_lower:
            similarity += 0.02
        if similarity > best_score:
            best_score = similarity
            best_item = item
    return best_item, best_score


def verify_reference_entry(entry: dict[str, str]) -> tuple[dict[str, Any], list[str], list[str], bool]:
    warnings: list[str] = []
    queries: list[str] = []
    network_needed = False
    candidate_key = entry.get("key", "")
    candidate_title = entry.get("title", "")
    candidate_venue = entry.get("venue", "")
    candidate_year = entry.get("year", "")
    doi = entry.get("doi", "")
    arxiv_id = entry.get("arxiv_id", "")
    is_arxiv = "arxiv" in candidate_venue.lower() or bool(arxiv_id)
    author_text = str(entry.get("author", ""))
    looks_like_seed_reference = (
        candidate_key.lower().startswith("wirelessseed")
        or "wireless optimization seed library" in author_text.lower()
        or "seed library" in candidate_title.lower()
    )

    base: dict[str, Any] = {
        "candidate_key": candidate_key,
        "final_bib_key": candidate_key,
        "candidate_source": "literature_source references.bib",
        "candidate_title": candidate_title,
        "final_title": candidate_title,
        "authors": sanitize_bibtex_text(entry.get("author", "")),
        "venue": sanitize_bibtex_text(candidate_venue),
        "year": sanitize_bibtex_text(candidate_year),
        "doi": sanitize_bibtex_text(doi),
        "url": "",
        "volume": sanitize_bibtex_text(entry.get("volume", "")),
        "number": sanitize_bibtex_text(entry.get("number", "")),
        "pages": sanitize_bibtex_text(entry.get("pages", "")),
        "month": sanitize_bibtex_text(entry.get("month", "")),
        "address": sanitize_bibtex_text(entry.get("address", "")),
        "arxiv_id": sanitize_bibtex_text(arxiv_id),
        "source_type": "arxiv" if is_arxiv else ("conference" if entry.get("entry_type", "").lower() == "inproceedings" else "journal"),
        "publisher": "arXiv" if is_arxiv else "other",
        "verification_status": "arxiv_only" if is_arxiv else "verified_published",
        "reliability": "medium" if is_arxiv else "high",
        "used_for_claims": [],
        "included_in_final_bib": False,
        "reason_for_inclusion": "",
        "reason_if_excluded": "",
    }

    if looks_like_seed_reference:
        base.update(
            {
                "verification_status": "unverified",
                "reliability": "low",
                "included_in_final_bib": False,
                "reason_if_excluded": "Deterministic seed metadata is not a real publication and cannot support paper claims.",
                "bibtex": "",
            }
        )
        warnings.append("seed_reference_rejected")
        return base, warnings, queries, network_needed

    if not doi and not candidate_venue.strip() and not is_arxiv:
        base.update(
            {
                "verification_status": "unverified",
                "reliability": "low",
                "included_in_final_bib": False,
                "reason_if_excluded": "Reference has neither DOI nor venue metadata; it was not accepted as verified.",
                "bibtex": "",
            }
        )
        warnings.append("missing_doi_and_venue")
        return base, warnings, queries, network_needed

    if doi:
        try:
            network_needed = True
            crossref = _lookup_crossref_by_doi(doi)
            if crossref:
                source_type = _crossref_source_type(crossref)
                base.update(
                    {
                        "final_title": sanitize_bibtex_text(" ".join(crossref.get("title", [])) or candidate_title),
                        "authors": sanitize_bibtex_text(_crossref_authors(crossref) or entry.get("author", "")),
                        "venue": sanitize_bibtex_text(_crossref_venue(crossref) or candidate_venue),
                        "year": sanitize_bibtex_text(_crossref_year(crossref) or candidate_year),
                        "doi": sanitize_bibtex_text(str(crossref.get("DOI", "")) or doi),
                        "url": "",
                        "volume": _crossref_volume(crossref),
                        "number": _crossref_number(crossref),
                        "pages": _crossref_pages(crossref),
                        "month": _crossref_month(crossref),
                        "address": _crossref_address(crossref),
                        "arxiv_id": "",
                        "source_type": source_type,
                        "publisher": sanitize_bibtex_text(_crossref_publisher(crossref) or "other"),
                        "verification_status": ("replaced_by_published_version" if is_arxiv and source_type in {"journal", "conference", "book", "standard"} else "verified_published"),
                        "reliability": "high",
                    }
                )
        except Exception as exc:
            warnings.append(f"doi_lookup_failed: {exc}")

    if not doi and not is_arxiv and candidate_title:
        queries.append(candidate_title)
        try:
            network_needed = True
            items = _search_crossref_by_title(candidate_title, rows=8)
            match, score = _pick_best_published_match(entry, items)
            if match is not None and score >= 0.92:
                base.update(
                    {
                        "final_title": sanitize_bibtex_text(" ".join(match.get("title", [])) or candidate_title),
                        "authors": sanitize_bibtex_text(_crossref_authors(match) or entry.get("author", "")),
                        "venue": sanitize_bibtex_text(_crossref_venue(match) or candidate_venue),
                        "year": sanitize_bibtex_text(_crossref_year(match) or candidate_year),
                        "doi": sanitize_bibtex_text(str(match.get("DOI", ""))),
                        "url": "",
                        "volume": _crossref_volume(match),
                        "number": _crossref_number(match),
                        "pages": _crossref_pages(match),
                        "month": _crossref_month(match),
                        "address": _crossref_address(match),
                        "arxiv_id": "",
                        "source_type": _crossref_source_type(match),
                        "publisher": sanitize_bibtex_text(_crossref_publisher(match) or "other"),
                        "verification_status": "verified_published",
                        "reliability": "high",
                    }
                )
            else:
                warnings.append(f"no_high_confidence_published_match_for_missing_doi(score={score:.2f})")
        except Exception as exc:
            warnings.append(f"crossref_search_failed_for_missing_doi: {exc}")

    if is_arxiv and base.get("verification_status") != "replaced_by_published_version":
        queries.append(candidate_title)
        try:
            network_needed = True
            items = _search_crossref_by_title(candidate_title, rows=6)
            match, score = _pick_best_published_match(entry, items)
            if match is not None and score >= 0.92:
                base.update(
                    {
                        "final_title": sanitize_bibtex_text(" ".join(match.get("title", [])) or candidate_title),
                        "authors": sanitize_bibtex_text(_crossref_authors(match) or entry.get("author", "")),
                        "venue": sanitize_bibtex_text(_crossref_venue(match)),
                        "year": sanitize_bibtex_text(_crossref_year(match) or candidate_year),
                        "doi": sanitize_bibtex_text(str(match.get("DOI", ""))),
                        "url": "",
                        "volume": _crossref_volume(match),
                        "number": _crossref_number(match),
                        "pages": _crossref_pages(match),
                        "month": _crossref_month(match),
                        "address": _crossref_address(match),
                        "arxiv_id": "",
                        "source_type": _crossref_source_type(match),
                        "publisher": sanitize_bibtex_text(_crossref_publisher(match) or "other"),
                        "verification_status": "replaced_by_published_version",
                        "reliability": "high",
                    }
                )
            else:
                base["verification_status"] = "arxiv_only"
                base["reliability"] = "medium"
                warnings.append(f"no_high_confidence_published_match(score={score:.2f})")
        except Exception as exc:
            warnings.append(f"crossref_search_failed: {exc}")

    if base["verification_status"] in {"verified_published", "replaced_by_published_version"}:
        metadata_issues = _phase3_4_reference_metadata_issues(base)
        if metadata_issues:
            base["included_in_final_bib"] = False
            base["verification_status"] = "incomplete_metadata"
            base["reliability"] = "low"
            base["reason_if_excluded"] = (
                "Reference was not accepted into the final bibliography because verified published "
                "metadata is incomplete: "
                + ", ".join(metadata_issues)
            )
            warnings.extend(metadata_issues)
        else:
            base["included_in_final_bib"] = True
            base["reason_for_inclusion"] = "Peer-reviewed or formally published reference verified and suitable for claim support."
    elif base["verification_status"] == "arxiv_only":
        metadata_issues = _phase3_4_reference_metadata_issues(base)
        if metadata_issues:
            base["included_in_final_bib"] = False
            base["verification_status"] = "incomplete_metadata"
            base["reliability"] = "low"
            base["reason_if_excluded"] = (
                "Reference was not accepted into the final bibliography because arXiv metadata is incomplete: "
                + ", ".join(metadata_issues)
            )
            warnings.extend(metadata_issues)
        else:
            base["included_in_final_bib"] = True
            base["reason_for_inclusion"] = "Kept as a closely related preprint because no high-confidence formally published version was found."
            base["publisher"] = base.get("publisher") or "arXiv"
    else:
        base["reason_if_excluded"] = "Reference could not be verified."

    base["bibtex"] = _build_bibtex_from_reference(base)
    return base, warnings, queries, network_needed


def categorize_reference_claim_support(
    entry: dict[str, Any],
    *,
    synthesis_md: str,
    algorithm_md: str,
    benchmark_definition_md: str,
) -> list[str]:
    title_lower = str(entry.get("final_title", "")).lower()
    categories: list[str] = []
    if any(token in title_lower for token in ["survey", "tutorial", "overview", "communications"]):
        categories.append("background")
    context_tokens = set(_phase3_4_keyword_tokens(f"{synthesis_md}\n{algorithm_md}\n{benchmark_definition_md}"))
    if any(_phase3_4_text_has_keyword(title_lower, token) for token in context_tokens):
        categories.append("related work")
    if any(_phase3_4_text_has_keyword(title_lower, token) for token in _phase3_4_keyword_tokens(algorithm_md + "\n" + benchmark_definition_md)):
        categories.append("method foundation")
    if len([token for token in context_tokens if _phase3_4_text_has_keyword(title_lower, token)]) >= 3:
        categories.append("closest related work")
    if not categories:
        categories.append("related work")
    deduped: list[str] = []
    for item in categories:
        if item not in deduped:
            deduped.append(item)
    return deduped


_PHASE33_SUPPLEMENTAL_WIRELESS_REFERENCES: tuple[dict[str, Any], ...] = (
    {
        "key": "SaadBennisChen2020Vision6G",
        "entry_type": "article",
        "author": "Walid Saad and Mehdi Bennis and Mingzhe Chen",
        "title": "A Vision of 6G Wireless Systems: Applications, Trends, Technologies, and Open Research Problems",
        "venue": "IEEE Network",
        "volume": "34",
        "number": "3",
        "pages": "134--142",
        "month": "May",
        "year": "2020",
        "doi": "10.1109/MNET.001.1900287",
        "axes": ["6g_motivation", "wireless_background"],
        "keywords": ["6g", "wireless", "future networks", "motivation"],
        "rationale": "High-level 6G motivation/background reference for why advanced wireless mechanisms matter.",
    },
    {
        "key": "LetaiefChenShiZhang2019Roadmap6G",
        "entry_type": "article",
        "author": "Khaled B. Letaief and Wei Chen and Yuanming Shi and Jun Zhang and Ying-Jun Angela Zhang",
        "title": "The Roadmap to 6G: AI Empowered Wireless Networks",
        "venue": "IEEE Communications Magazine",
        "volume": "57",
        "number": "8",
        "pages": "84--90",
        "month": "Aug.",
        "year": "2019",
        "doi": "10.1109/MCOM.2019.1900271",
        "axes": ["6g_motivation", "wireless_background"],
        "keywords": ["6g", "wireless", "ai", "resource optimization", "motivation"],
        "rationale": "IEEE Communications Magazine 6G roadmap for broad Introduction motivation.",
    },
    {
        "key": "TatariaShafiMolisch2021Vision6G",
        "entry_type": "article",
        "author": "Harsh Tataria and Mansoor Shafi and Andreas F. Molisch and Mischa Dohler and Henrik Sjoland and Fredrik Tufvesson",
        "title": "6G Wireless Systems: Vision, Requirements, Challenges, Insights, and Opportunities",
        "venue": "Proceedings of the IEEE",
        "volume": "109",
        "number": "7",
        "pages": "1166--1199",
        "month": "Jul.",
        "year": "2021",
        "doi": "10.1109/JPROC.2021.3061701",
        "axes": ["6g_motivation", "wireless_background"],
        "keywords": ["6g", "wireless", "requirements", "challenges", "motivation"],
        "rationale": "Broad 6G requirements and opportunities reference for technology motivation.",
    },
    {
        "key": "AndrewsBuzziChoi2014What5G",
        "entry_type": "article",
        "author": "Jeffrey G. Andrews and Stefano Buzzi and Wan Choi and Stephen V. Hanly and Angel Lozano and Anthony C. K. Soong and Jianzhong Charlie Zhang",
        "title": "What Will 5G Be?",
        "venue": "IEEE Journal on Selected Areas in Communications",
        "volume": "32",
        "number": "6",
        "pages": "1065--1082",
        "month": "Jun.",
        "year": "2014",
        "doi": "10.1109/JSAC.2014.2328098",
        "axes": ["wireless_background"],
        "keywords": ["5g", "wireless", "cellular", "network architecture", "motivation"],
        "rationale": "Broad cellular-network evolution reference for wireless motivation.",
    },
    {
        "key": "LarssonEdforsTufvessonMarzetta2014MassiveMIMO",
        "entry_type": "article",
        "author": "Erik G. Larsson and Ove Edfors and Fredrik Tufvesson and Thomas L. Marzetta",
        "title": "Massive MIMO for Next Generation Wireless Systems",
        "venue": "IEEE Communications Magazine",
        "volume": "52",
        "number": "2",
        "pages": "186--195",
        "month": "Feb.",
        "year": "2014",
        "doi": "10.1109/MCOM.2014.6736761",
        "axes": ["mimo_background", "wireless_background"],
        "keywords": ["massive mimo", "mimo", "antenna array", "beamforming", "wireless"],
        "rationale": "Massive-MIMO background for multi-antenna wireless optimization topics.",
    },
    {
        "key": "HeathGonzalezPrelcicRanganRoh2016MmWaveMIMO",
        "entry_type": "article",
        "author": "Robert W. Heath and Nuria Gonzalez-Prelcic and Sundeep Rangan and Wonil Roh and Akbar M. Sayeed",
        "title": "An Overview of Signal Processing Techniques for Millimeter Wave MIMO Systems",
        "venue": "IEEE Journal of Selected Topics in Signal Processing",
        "volume": "10",
        "number": "3",
        "pages": "436--453",
        "month": "Apr.",
        "year": "2016",
        "doi": "10.1109/JSTSP.2016.2523924",
        "axes": ["mimo_background", "mmwave_background", "wireless_background"],
        "keywords": ["millimeter wave", "mmwave", "mimo", "beamforming", "antenna array"],
        "rationale": "mmWave/MIMO signal-processing background for high-frequency or large-array topics.",
    },
    {
        "key": "ZengZhangLim2016UAVCommunications",
        "entry_type": "article",
        "author": "Yong Zeng and Rui Zhang and Teng Joon Lim",
        "title": "Wireless Communications With Unmanned Aerial Vehicles: Opportunities and Challenges",
        "venue": "IEEE Communications Magazine",
        "volume": "54",
        "number": "5",
        "pages": "36--42",
        "month": "May",
        "year": "2016",
        "doi": "10.1109/MCOM.2016.7470933",
        "axes": ["uav_background", "wireless_background"],
        "keywords": ["uav", "unmanned aerial", "trajectory", "mobility", "wireless"],
        "rationale": "UAV wireless communications background for mobility/deployment topics.",
    },
    {
        "key": "NiknamDhillonReed2020FederatedLearningWireless",
        "entry_type": "article",
        "author": "Solmaz Niknam and Harpreet S. Dhillon and Jeffrey H. Reed",
        "title": "Federated Learning for Wireless Communications: Motivation, Opportunities, and Challenges",
        "venue": "IEEE Communications Magazine",
        "volume": "58",
        "number": "6",
        "pages": "46--51",
        "month": "Jun.",
        "year": "2020",
        "doi": "10.1109/MCOM.001.1900461",
        "axes": ["distributed_coordination_background", "wireless_background"],
        "keywords": ["federated learning", "distributed", "edge", "wireless", "coordination"],
        "rationale": "Distributed wireless learning/coordination background for networked optimization topics.",
    },
    {
        "key": "DaiWangDingWangChenHanzo2017NOMASurvey",
        "entry_type": "article",
        "author": "Linglong Dai and Bichai Wang and Yuanwei Liu and Sheng Chen and Lajos Hanzo",
        "title": "A Survey on Non-Orthogonal Multiple Access for 5G Networks: Research Challenges and Future Trends",
        "venue": "IEEE Journal on Selected Areas in Communications",
        "volume": "35",
        "number": "10",
        "pages": "2181--2195",
        "month": "Oct.",
        "year": "2017",
        "doi": "10.1109/JSAC.2017.2725519",
        "axes": ["multiple_access_background", "wireless_background"],
        "keywords": ["noma", "multiple access", "interference", "resource allocation", "wireless"],
        "rationale": "Multiple-access and interference-management background for resource-allocation topics.",
    },
    {
        "key": "LiuCuiMasouros2022ISAC",
        "entry_type": "article",
        "author": "Fan Liu and Yuanhao Cui and Christos Masouros and Jie Xu and Tony Xiao Han and Yonina C. Eldar and Stefano Buzzi",
        "title": "Integrated Sensing and Communications: Toward Dual-Functional Wireless Networks for 6G and Beyond",
        "venue": "IEEE Journal on Selected Areas in Communications",
        "volume": "40",
        "number": "6",
        "pages": "1728--1767",
        "month": "Jun.",
        "year": "2022",
        "doi": "10.1109/JSAC.2022.3156632",
        "axes": ["isac_background", "6g_motivation"],
        "keywords": ["isac", "sensing", "communication", "6g"],
        "rationale": "ISAC survey/tutorial anchor when the topic includes sensing or multi-functional wireless.",
    },
    {
        "key": "ChenRenXuZeng2025ISACP",
        "entry_type": "article",
        "author": "Yilong Chen and Zixiang Ren and Jie Xu and Yong Zeng and Derrick Wing Kwan Ng and Shuguang Cui",
        "title": "Integrated Sensing, Communication, and Powering: Toward Multi-Functional 6G Wireless Networks",
        "venue": "IEEE Communications Magazine",
        "volume": "63",
        "number": "8",
        "pages": "146--153",
        "month": "Aug.",
        "year": "2025",
        "doi": "10.1109/MCOM.006.2400013",
        "axes": ["isacp_background", "6g_motivation"],
        "keywords": ["isacp", "sensing", "communication", "powering", "wireless power", "6g"],
        "rationale": "Direct multi-functional sensing/communication/powering motivation reference.",
    },
    {
        "key": "ClerckxZhangSchober2019WIPT",
        "entry_type": "article",
        "author": "Bruno Clerckx and Rui Zhang and Robert Schober and Derrick Wing Kwan Ng and Dong In Kim and H. Vincent Poor",
        "title": "Fundamentals of Wireless Information and Power Transfer: From RF Energy Harvester Models to Signal and System Designs",
        "venue": "IEEE Journal on Selected Areas in Communications",
        "volume": "37",
        "number": "1",
        "pages": "4--33",
        "month": "Jan.",
        "year": "2019",
        "doi": "10.1109/JSAC.2018.2872615",
        "axes": ["wipt_background", "swipt_background"],
        "keywords": ["swipt", "wipt", "wireless power", "energy harvesting", "powering"],
        "rationale": "WIPT/SWIPT fundamentals and energy-harvesting model background.",
    },
    {
        "key": "WuZhangZhengYouZhang2021IRSTutorial",
        "entry_type": "article",
        "author": "Qingqing Wu and Shuowen Zhang and Beixiong Zheng and Changsheng You and Rui Zhang",
        "title": "Intelligent Reflecting Surface-Aided Wireless Communications: A Tutorial",
        "venue": "IEEE Transactions on Communications",
        "volume": "69",
        "number": "5",
        "pages": "3313--3351",
        "month": "May",
        "year": "2021",
        "doi": "10.1109/TCOMM.2021.3051897",
        "axes": ["ris_background"],
        "keywords": ["ris", "irs", "reconfigurable intelligent surface", "intelligent reflecting surface"],
        "rationale": "RIS tutorial anchor when the selected topic uses programmable surfaces.",
    },
    {
        "key": "LiuOuyangWang2025NearFieldSurvey",
        "entry_type": "article",
        "author": "Yuanwei Liu and Chongjun Ouyang and Zhaolin Wang and Jiaqi Xu and Xidong Mu and A. Lee Swindlehurst",
        "title": "Near-Field Communications: A Comprehensive Survey",
        "venue": "IEEE Communications Surveys & Tutorials",
        "volume": "27",
        "number": "3",
        "pages": "1687--1728",
        "month": "Jun.",
        "year": "2025",
        "doi": "10.1109/COMST.2024.3475884",
        "axes": ["near_field_background", "6g_motivation"],
        "keywords": ["near-field", "near field", "xl-mimo", "extremely large", "holographic"],
        "rationale": "Near-field communications survey for large-aperture or XL-MIMO topics.",
    },
    {
        "key": "ZhuMaZhang2024MovableAntennaModeling",
        "entry_type": "article",
        "author": "Lipeng Zhu and Wenyan Ma and Rui Zhang",
        "title": "Modeling and Performance Analysis for Movable Antenna Enabled Wireless Communications",
        "venue": "IEEE Transactions on Wireless Communications",
        "volume": "23",
        "number": "6",
        "pages": "6234--6250",
        "month": "Jun.",
        "year": "2024",
        "doi": "10.1109/TWC.2023.3330887",
        "axes": ["movable_antenna_background", "mimo_background"],
        "keywords": ["movable antenna", "antenna position", "fluid antenna", "reconfigurable antenna", "wireless"],
        "rationale": "Movable-antenna modeling and performance foundation for position-adaptive array topics.",
    },
    {
        "key": "ZhuMaZhang2023MovableAntennaPerformance",
        "entry_type": "inproceedings",
        "author": "Lipeng Zhu and Wenyan Ma and Rui Zhang",
        "title": "Performance Analysis for Movable Antenna Aided Wireless Communications",
        "venue": "GLOBECOM 2023 - 2023 IEEE Global Communications Conference",
        "pages": "703--708",
        "month": "Dec.",
        "year": "2023",
        "doi": "10.1109/GLOBECOM54140.2023.10437024",
        "axes": ["movable_antenna_background", "mimo_background"],
        "keywords": ["movable antenna", "antenna position", "reconfigurable antenna", "wireless"],
        "rationale": "Early movable-antenna performance analysis reference.",
    },
    {
        "key": "NingYangWuWangMeiYuenBjornson2025MovableAntennaArchitectures",
        "entry_type": "article",
        "author": "Boyu Ning and Songjie Yang and Yafei Wu and Peilan Wang and Weidong Mei and Chau Yuen and Emil Bjornson",
        "title": "Movable Antenna-Enhanced Wireless Communications: General Architectures and Implementation Methods",
        "venue": "IEEE Wireless Communications",
        "volume": "32",
        "number": "5",
        "pages": "108--116",
        "month": "Oct.",
        "year": "2025",
        "doi": "10.1109/MWC.013.2400238",
        "axes": ["movable_antenna_background", "wireless_background"],
        "keywords": ["movable antenna", "antenna position", "architecture", "implementation", "wireless"],
        "rationale": "Movable-antenna architecture and implementation background.",
    },
    {
        "key": "ChengWangZhaoSongLiaoYang2025MovableAntennaTraining",
        "entry_type": "article",
        "author": "Yawen Cheng and Jiajia Wang and Xuanzhi Zhao and Jiacheng Song and Jingfa Liao and Songjie Yang",
        "title": "Bayesian Optimization-Based Antenna Position Training for Movable Antenna Enhanced Wireless Communications",
        "venue": "IEEE Communications Letters",
        "volume": "29",
        "number": "12",
        "pages": "2825--2829",
        "month": "Dec.",
        "year": "2025",
        "doi": "10.1109/LCOMM.2025.3614681",
        "axes": ["movable_antenna_background", "optimization_background"],
        "keywords": ["movable antenna", "antenna position", "bayesian optimization", "training", "wireless"],
        "rationale": "Optimization-oriented movable-antenna training reference.",
    },
    {
        "key": "AbbasZhangTaherkordiSkeie2018MEC",
        "entry_type": "article",
        "author": "Nasir Abbas and Yan Zhang and Amir Taherkordi and Tor Skeie",
        "title": "Mobile Edge Computing: A Survey",
        "venue": "IEEE Internet of Things Journal",
        "volume": "5",
        "number": "1",
        "pages": "450--465",
        "month": "Feb.",
        "year": "2018",
        "doi": "10.1109/JIOT.2017.2750180",
        "axes": ["edge_background"],
        "keywords": ["edge", "offloading", "mobile edge computing", "iot"],
        "rationale": "MEC background reference for edge/offloading topics.",
    },
    {
        "key": "ShafieYangHan2023THz6G",
        "entry_type": "article",
        "author": "Akram Shafie and Nan Yang and Chong Han and Josep Miquel Jornet and Markku Juntti and Thomas Kurner",
        "title": "Terahertz Communications for 6G and Beyond Wireless Networks: Challenges, Key Advancements, and Opportunities",
        "venue": "IEEE Network",
        "volume": "37",
        "number": "3",
        "pages": "162--169",
        "month": "May",
        "year": "2023",
        "doi": "10.1109/MNET.118.2200057",
        "axes": ["thz_background", "6g_motivation"],
        "keywords": ["terahertz", "thz", "sub-thz", "blockage", "6g"],
        "rationale": "THz/6G motivation and challenges reference.",
    },
    {
        "key": "YangChenLi2024SecureSemantic",
        "entry_type": "article",
        "author": "Zhaohui Yang and Mingzhe Chen and Gaolei Li and Yang Yang and Zhaoyang Zhang",
        "title": "Secure Semantic Communications: Fundamentals and Challenges",
        "venue": "IEEE Network",
        "volume": "38",
        "number": "6",
        "pages": "513--520",
        "month": "Nov.",
        "year": "2024",
        "doi": "10.1109/MNET.2024.3411027",
        "axes": ["semantic_background"],
        "keywords": ["semantic", "semantics", "task-oriented", "goal-oriented"],
        "rationale": "Semantic communications background reference for semantic-aware wireless topics.",
    },
)


def phase3_4_supplemental_reference_candidates(context_text: str, *, min_items: int = 12) -> list[dict[str, Any]]:
    """Return DOI-backed background references that broaden a thin topic pool.

    These entries are intentionally generic but still wireless/IEEE-focused.
    They support motivation and high-level background statements; closest-work
    and method claims should still prefer topic-specific references discovered
    by Phase 1 or live academic search.
    """

    lower = str(context_text or "").lower()
    selected: list[dict[str, Any]] = []
    selected_keys: set[str] = set()

    def score(ref: dict[str, Any]) -> int:
        keywords = [str(item).lower() for item in ref.get("keywords", []) if str(item).strip()]
        axes = [str(item).lower() for item in ref.get("axes", []) if str(item).strip()]
        value = 0
        for token in keywords + axes:
            if token and token in lower:
                value += 6
        if "6g" in lower or "wireless" in lower:
            if "6g_motivation" in axes or "wireless_background" in axes:
                value += 3
        return value

    ranked = sorted(
        _PHASE33_SUPPLEMENTAL_WIRELESS_REFERENCES,
        key=lambda item: (score(item), str(item.get("year", "")), str(item.get("key", ""))),
        reverse=True,
    )
    for item in ranked:
        item_score = score(item)
        # Always allow a few 6G/wireless background references, but require
        # topic overlap for specialized supplements such as THz, RIS, or edge.
        axes = set(str(axis).lower() for axis in item.get("axes", []))
        generic_background = bool({"6g_motivation", "wireless_background"} & axes)
        if item_score <= 0 and not generic_background:
            continue
        key = str(item.get("key", "")).strip()
        if not key or key in selected_keys:
            continue
        ref = dict(item)
        ref["source"] = "phase3_4_verified_supplemental_background"
        selected.append(ref)
        selected_keys.add(key)
        if len(selected) >= min_items:
            break
    return selected


def merge_phase3_4_candidate_references(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for group in groups:
        for item in group:
            if not isinstance(item, dict):
                continue
            key = str(item.get("key") or item.get("final_bib_key") or item.get("candidate_key") or "").strip().lower()
            title = normalize_reference_text(str(item.get("title") or item.get("final_title") or item.get("candidate_title") or ""))
            dedupe_key = key or title
            if not dedupe_key or dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            merged.append(dict(item))
    return merged


def build_verified_reference_bank(
    candidate_entries: list[dict[str, str]],
    *,
    synthesis_md: str,
    algorithm_md: str,
    benchmark_definition_md: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], bool]:
    bank: list[dict[str, Any]] = []
    verify_needed: list[dict[str, Any]] = []
    replacement_notes: list[dict[str, Any]] = []
    network_used = False
    for entry in candidate_entries:
        verified, warnings, queries, touched_network = verify_reference_entry(entry)
        network_used = network_used or touched_network
        verified["used_for_claims"] = categorize_reference_claim_support(
            verified,
            synthesis_md=synthesis_md,
            algorithm_md=algorithm_md,
            benchmark_definition_md=benchmark_definition_md,
        )
        verified["verification_warnings"] = warnings
        bank.append(verified)
        if warnings:
            replacement_notes.append(
                {
                    "candidate_key": verified["candidate_key"],
                    "candidate_title": verified["candidate_title"],
                    "verification_status": verified["verification_status"],
                    "warnings": warnings,
                }
            )
        if verified["verification_status"] == "arxiv_only":
            verify_needed.append(
                {
                    "candidate_key": verified["candidate_key"],
                    "candidate_title": verified["candidate_title"],
                    "possible_search_queries": queries,
                    "reason": "No high-confidence formally published version was found via deterministic verification.",
                }
            )
    bank, dedupe_notes = dedupe_phase3_4_references_by_identity(bank)
    for note in dedupe_notes:
        replacement_notes.append(
            {
                "candidate_key": note.get("removed_key", ""),
                "verification_status": "duplicate_removed",
                "warnings": [note.get("reason", "duplicate DOI/title identity")],
                "kept_key": note.get("kept_key", ""),
                "dedupe_identity": note.get("dedupe_identity", ""),
            }
        )
    removed_keys = {str(note.get("removed_key", "")) for note in dedupe_notes if str(note.get("removed_key", "")).strip()}
    if removed_keys:
        verify_needed = [
            item
            for item in verify_needed
            if str(item.get("candidate_key", "")).strip() not in removed_keys
        ]
    return bank, verify_needed, replacement_notes, network_used


def write_reference_replacement_report(
    *,
    verified_bank: list[dict[str, Any]],
    report_path: Path,
) -> None:
    lines = [
        "# Reference Replacement Report",
        "",
        "This report records how the supplied literature source was converted into a verified final reference bank.",
        "",
    ]
    for item in verified_bank:
        status = str(item.get("verification_status", ""))
        lines.append(f"## {item.get('candidate_key', '')}")
        lines.append(f"- candidate title: {item.get('candidate_title', '')}")
        lines.append(f"- final title: {item.get('final_title', '')}")
        lines.append(f"- verification status: {status}")
        lines.append(f"- venue: {item.get('venue', '')}")
        lines.append(f"- publisher: {item.get('publisher', '')}")
        lines.append(f"- used_for_claims: {', '.join(item.get('used_for_claims', [])) or 'not specified'}")
        if status == "replaced_by_published_version":
            lines.append("- note: arXiv candidate was replaced by a formally published version.")
        elif status == "arxiv_only":
            lines.append("- note: kept as arXiv-only because no high-confidence published version was verified.")
        elif status == "verified_published":
            lines.append("- note: existing metadata was verified as formally published.")
        else:
            lines.append(f"- note: {item.get('reason_if_excluded', '') or 'excluded'}")
        lines.append("")
    write_text(report_path, "\n".join(lines).strip() + "\n")


def build_reference_quality_report(
    *,
    final_references: list[dict[str, Any]],
    citation_claim_map: list[dict[str, Any]],
) -> dict[str, Any]:
    total = len(final_references)
    peer_reviewed = sum(
        1 for item in final_references if item.get("verification_status") in {"verified_published", "replaced_by_published_version"}
    )
    ieee_count = sum(1 for item in final_references if "ieee" in str(item.get("publisher", "")).lower() or "ieee" in str(item.get("venue", "")).lower())
    arxiv_only = sum(1 for item in final_references if item.get("verification_status") == "arxiv_only")
    unverified = sum(1 for item in final_references if item.get("verification_status") == "unverified")
    used_keys = {key for item in citation_claim_map for key in item.get("citation_keys", [])}
    introduction_used_keys = {
        key
        for item in citation_claim_map
        if str(item.get("section", "introduction") or "introduction") == "introduction"
        for key in item.get("citation_keys", [])
    }
    technical_used_keys = {
        key
        for item in citation_claim_map
        if str(item.get("section", "") or "") in {"system_model", "proposed_solution", "numerical_results"}
        for key in item.get("citation_keys", [])
    }
    technical_section_counts = {
        section_name: len(
            {
                key
                for item in citation_claim_map
                if str(item.get("section", "") or "") == section_name
                for key in item.get("citation_keys", [])
            }
        )
        for section_name in ["system_model", "proposed_solution", "numerical_results"]
    }
    final_keys = {str(item.get("final_bib_key", "")) for item in final_references}
    unused = sorted(final_keys - used_keys)
    no_claim_mapping = sorted(final_keys - used_keys)
    duplicate_titles = []
    seen_titles: dict[str, str] = {}
    metadata_blockers: list[dict[str, Any]] = []
    for item in final_references:
        norm_title = normalize_reference_text(str(item.get("final_title", "")))
        key = str(item.get("final_bib_key", ""))
        issues = _phase3_4_reference_metadata_issues(item)
        if issues:
            metadata_blockers.append(
                {
                    "key": key,
                    "title": item.get("final_title", ""),
                    "source_type": item.get("source_type", ""),
                    "verification_status": item.get("verification_status", ""),
                    "issues": issues,
                }
            )
        if norm_title in seen_titles:
            duplicate_titles.append({"title": item.get("final_title", ""), "keys": [seen_titles[norm_title], key]})
        else:
            seen_titles[norm_title] = key
    missing_doi = [
        item.get("final_bib_key", "")
        for item in final_references
        if not str(item.get("doi", "")).strip() and item.get("verification_status") != "arxiv_only"
    ]
    return {
        "total_references": total,
        "peer_reviewed_count": peer_reviewed,
        "ieee_count": ieee_count,
        "arxiv_only_count": arxiv_only,
        "arxiv_ratio": (arxiv_only / total) if total else 0.0,
        "unverified_count": unverified,
        "references_used_in_cited_claims": len(used_keys),
        "references_used_in_introduction": len(introduction_used_keys),
        "references_used_in_technical_sections": len(technical_used_keys),
        "technical_section_reference_counts": technical_section_counts,
        "unused_references": unused,
        "references_without_claim_mapping": no_claim_mapping,
        "fabricated_looking_metadata_warnings": metadata_blockers,
        "metadata_blocking_errors": metadata_blockers,
        "missing_doi_warnings": missing_doi,
        "duplicate_arxiv_published_version_warnings": duplicate_titles,
        "ok": not metadata_blockers and not duplicate_titles,
    }


def build_phase3_4_final_reference_count_contract(
    *,
    final_selected_reference_keys: list[str],
    current_valid_reference_keys: set[str],
    introduction_cited_reference_keys: list[str],
    technical_citation_map: list[dict[str, Any]],
    minimum_reference_target: int,
) -> dict[str, Any]:
    final_keys = []
    for key in final_selected_reference_keys:
        if key in current_valid_reference_keys and key not in final_keys:
            final_keys.append(key)
    intro_keys = []
    for key in introduction_cited_reference_keys:
        if key in current_valid_reference_keys and key not in intro_keys:
            intro_keys.append(key)
    technical_keys_by_section: dict[str, list[str]] = {section: [] for section in ["system_model", "proposed_solution", "numerical_results"]}
    for item in technical_citation_map:
        if not isinstance(item, dict):
            continue
        section = str(item.get("section", "") or "").strip()
        if section not in technical_keys_by_section:
            continue
        for key in item.get("citation_keys", []) if isinstance(item.get("citation_keys", []), list) else []:
            key_text = str(key).strip()
            if key_text in current_valid_reference_keys and key_text not in technical_keys_by_section[section]:
                technical_keys_by_section[section].append(key_text)
    technical_keys = []
    for keys in technical_keys_by_section.values():
        for key in keys:
            if key not in technical_keys:
                technical_keys.append(key)
    errors: list[str] = []
    warnings: list[str] = []
    if len(final_keys) < minimum_reference_target:
        errors.append(
            f"Final full-paper citations use {len(final_keys)} valid references < hard target {minimum_reference_target}."
        )
    if not technical_keys:
        errors.append("No verified references are cited in technical sections.")
    if not technical_keys_by_section["system_model"]:
        warnings.append("System model/problem formulation has no verified technical citation.")
    if not technical_keys_by_section["proposed_solution"]:
        warnings.append("Proposed solution has no verified method-foundation citation.")
    if not technical_keys_by_section["numerical_results"]:
        warnings.append("Numerical results has no benchmark/evaluation-context citation.")
    return {
        "ok": not errors,
        "minimum_reference_target": minimum_reference_target,
        "available_valid_references": len(current_valid_reference_keys),
        "final_valid_cited_references": len(final_keys),
        "final_valid_cited_reference_keys": final_keys,
        "introduction_valid_cited_references": len(intro_keys),
        "introduction_valid_cited_reference_keys": intro_keys,
        "technical_valid_cited_references": len(technical_keys),
        "technical_valid_cited_reference_keys": technical_keys,
        "technical_section_reference_counts": {section: len(keys) for section, keys in technical_keys_by_section.items()},
        "technical_section_reference_keys": technical_keys_by_section,
        "errors": errors,
        "warnings": warnings,
        "contract_scope": "full_paper_after_technical_citation_pass",
    }


def write_reference_quality_report_md(report: dict[str, Any], path: Path) -> None:
    metadata_blockers = report.get("metadata_blocking_errors", [])
    text = [
        "# Reference Quality Report",
        "",
        f"- total references: {report.get('total_references', 0)}",
        f"- peer-reviewed count: {report.get('peer_reviewed_count', 0)}",
        f"- IEEE count: {report.get('ieee_count', 0)}",
        f"- arXiv-only count: {report.get('arxiv_only_count', 0)}",
        f"- arXiv ratio: {report.get('arxiv_ratio', 0.0):.3f}",
        f"- unverified count: {report.get('unverified_count', 0)}",
        f"- references used in cited claims: {report.get('references_used_in_cited_claims', report.get('references_used_in_introduction', 0))}",
        f"- references used in introduction: {report.get('references_used_in_introduction', 0)}",
        f"- references used in technical sections: {report.get('references_used_in_technical_sections', 0)}",
        f"- unused references: {', '.join(report.get('unused_references', [])) or 'none'}",
        f"- references without claim mapping: {', '.join(report.get('references_without_claim_mapping', [])) or 'none'}",
        f"- missing DOI warnings: {', '.join(report.get('missing_doi_warnings', [])) or 'none'}",
        f"- metadata blocking errors: {len(metadata_blockers) if isinstance(metadata_blockers, list) else 0}",
    ]
    if isinstance(metadata_blockers, list) and metadata_blockers:
        text.extend(["", "## Metadata Blocking Errors"])
        for item in metadata_blockers:
            if not isinstance(item, dict):
                continue
            key = str(item.get("key", "")).strip() or "unknown"
            issues = item.get("issues", [])
            issue_text = ", ".join(str(issue) for issue in issues) if isinstance(issues, list) else str(issues)
            text.append(f"- {key}: {issue_text}")
    write_text(path, "\n".join(text) + "\n")


def build_phase3_4_paper_facts(
    *,
    topic: str,
    current_paper_brief: dict[str, Any],
    introduction_facts: dict[str, Any],
    verified_reference_bank: list[dict[str, Any]],
) -> dict[str, Any]:
    technical = introduction_facts.get("technical_scope", {}) if isinstance(introduction_facts, dict) else {}
    results = introduction_facts.get("result_constraints", {}) if isinstance(introduction_facts, dict) else {}
    background = introduction_facts.get("background", {}) if isinstance(introduction_facts, dict) else {}
    return {
        "introduction_reference_phase": {
            "topic": topic or "not specified",
            "target_venue": "IEEE WCL",
            "current_paper_brief": current_paper_brief,
            "background": background,
            "technical_scope": technical,
            "result_constraints": results,
            "verified_reference_bank": verified_reference_bank,
            "allowed_claims": [
                "addresses the stated limitation under the considered settings",
                "improves over the benchmark in the evaluated scenarios",
                "supports the value of the added design degrees of freedom",
            ],
            "forbidden_claims": [
                "proves",
                "guarantees",
                "globally optimal",
                "always superior",
                "statistically significant",
            ],
        }
    }


def _score_phase3_4_reference(
    entry: dict[str, Any],
    *,
    desired_categories: set[str],
) -> float:
    score = 0.0
    used_for_claims = {str(item).strip().lower() for item in entry.get("used_for_claims", [])}
    for category in desired_categories:
        if category.lower() in used_for_claims:
            score += 4.0
    verification_status = str(entry.get("verification_status", "")).strip().lower()
    if verification_status in {"verified_published", "replaced_by_published_version"}:
        score += 6.0
    elif verification_status == "arxiv_only":
        score += 1.0
    venue_lower = str(entry.get("venue", "")).lower()
    publisher_lower = str(entry.get("publisher", "")).lower()
    title_lower = str(entry.get("final_title", "")).lower()
    if "ieee" in venue_lower or "ieee" in publisher_lower:
        score += 3.0
    if any(token in title_lower for token in ["survey", "tutorial", "overview", "comprehensive"]):
        score += 2.0
    if any(token in title_lower for token in ["optimization", "communications", "network", "algorithm", "resource"]):
        score += 2.0
    return score


def build_phase3_4_reference_strategy(
    *,
    verified_reference_bank: list[dict[str, Any]],
    introduction_facts: dict[str, Any],
) -> dict[str, Any]:
    results = introduction_facts.get("result_constraints", {}) if isinstance(introduction_facts, dict) else {}
    requested_minimum_reference_target = int(results.get("minimum_reference_target", 12) or 12)
    requested_preferred_reference_target = int(results.get("preferred_reference_target", 14) or 14)
    usable_reference_bank = [
        item
        for item in verified_reference_bank
        if isinstance(item, dict) and phase3_4_reference_is_final_usable(item)
    ]
    peer_reviewed = [
        item
        for item in usable_reference_bank
        if str(item.get("verification_status", "")).strip().lower() in {"verified_published", "replaced_by_published_version"}
    ]
    arxiv_only = [
        item
        for item in usable_reference_bank
        if str(item.get("verification_status", "")).strip().lower() == "arxiv_only"
    ]
    available_reference_count = len(peer_reviewed) + len(arxiv_only)
    minimum_reference_target = requested_minimum_reference_target
    preferred_reference_target = max(requested_preferred_reference_target, minimum_reference_target)

    def pick(limit: int, *categories: str) -> list[str]:
        ranked = sorted(
            usable_reference_bank,
            key=lambda item: (
                _score_phase3_4_reference(item, desired_categories=set(categories)),
                str(item.get("year", "")),
                str(item.get("final_bib_key", "")),
            ),
            reverse=True,
        )
        keys: list[str] = []
        for item in ranked:
            key = str(item.get("final_bib_key", "")).strip()
            if not key or key in keys:
                continue
            if _score_phase3_4_reference(item, desired_categories=set(categories)) <= 0:
                continue
            keys.append(key)
            if len(keys) >= limit:
                break
        return keys

    motivation_keys = pick(4, "background", "method foundation")
    related_work_keys = pick(7, "related work", "closest related work")
    gap_keys = pick(4, "closest related work", "method foundation")
    mandatory_keys: list[str] = []
    for group in [motivation_keys, related_work_keys, gap_keys]:
        for key in group:
            if key not in mandatory_keys:
                mandatory_keys.append(key)
    # Fill to at least the requested minimum, prioritizing peer-reviewed IEEE-style entries.
    for item in sorted(
        peer_reviewed + arxiv_only,
        key=lambda ref: (
            _score_phase3_4_reference(ref, desired_categories={"background", "related work", "closest related work", "method foundation"}),
            str(ref.get("year", "")),
            str(ref.get("final_bib_key", "")),
        ),
        reverse=True,
    ):
        key = str(item.get("final_bib_key", "")).strip()
        if not key or key in mandatory_keys:
            continue
        mandatory_keys.append(key)
        if len(mandatory_keys) >= min(preferred_reference_target, len(usable_reference_bank)):
            break

    return {
        "minimum_reference_target": minimum_reference_target,
        "preferred_reference_target": preferred_reference_target,
        "requested_minimum_reference_target": requested_minimum_reference_target,
        "requested_preferred_reference_target": requested_preferred_reference_target,
        "maximum_arxiv_only_target": 3,
        "motivation_reference_keys": motivation_keys,
        "related_work_reference_keys": related_work_keys,
        "gap_reference_keys": gap_keys,
        "mandatory_reference_keys": mandatory_keys[: min(preferred_reference_target, len(mandatory_keys))],
        "recommended_reference_keys": mandatory_keys[: min(preferred_reference_target, len(mandatory_keys))],
        "available_peer_reviewed_count": len(peer_reviewed),
        "available_arxiv_only_count": len(arxiv_only),
        "reference_count_gate": {
            "ok": available_reference_count >= minimum_reference_target,
            "available_reference_count": available_reference_count,
            "minimum_reference_target": minimum_reference_target,
            "message": (
                "Reference bank meets the target."
                if available_reference_count >= minimum_reference_target
                else "Reference bank is below the IEEE WCL target and must be repaired before final writing."
            ),
        },
    }


def build_phase3_4_introduction_prompt(
    *,
    source_map_json: str,
    current_paper_brief_json: str,
    introduction_facts_json: str,
    verified_reference_bank_json: str,
    reference_quality_report_json: str,
    reference_strategy_json: str,
    writing_agent_request_json: str = "",
    literature_agent_request_json: str = "",
) -> str:
    return render_prompt_template(
        "phase3_4/introduction.prompt.yaml",
        source_map_json=source_map_json,
        current_paper_brief_json=current_paper_brief_json,
        introduction_facts_json=introduction_facts_json,
        verified_reference_bank_json=verified_reference_bank_json,
        reference_quality_report_json=reference_quality_report_json,
        reference_strategy_json=reference_strategy_json,
        writing_agent_request_json=compact_text(writing_agent_request_json, 7000),
        literature_agent_request_json=compact_text(literature_agent_request_json, 7000),
    )


def _phase3_4_demath_intro_body(tex: str) -> str:
    """Keep mathematical notation out of Introduction prose while preserving the final notation paragraph."""

    notation_match = re.search(
        r"\\textit\{Notation:\}|\\emph\{Notation:\}|(?:^|\n)\s*Notation\s*:|Unless otherwise stated",
        tex,
        flags=re.I,
    )
    body = tex[: notation_match.start()] if notation_match else tex
    suffix = tex[notation_match.start():] if notation_match else ""

    def replace_inline_math(match: re.Match[str]) -> str:
        expr = match.group(1).strip()
        normalized = re.sub(r"\s+", "", expr)
        if normalized in {r"\lambda/2", r"{\lambda}/2"} or re.fullmatch(r"\\frac\{\\lambda\}\{2\}", normalized):
            return "half-wavelength"
        if normalized in {r"1/2\lambda", r"0.5\lambda"}:
            return "half-wavelength"
        if normalized in {"p", r"\mathbf{p}", r"\boldsymbol{p}"}:
            return "position"
        if normalized in {"W", r"\mathbf{W}", r"\boldsymbol{W}"}:
            return "precoder"
        if re.fullmatch(r"[A-Za-z]", normalized):
            return "design variable"
        if "," in expr or re.fullmatch(r"[A-Za-z\\{}_,\s]+", expr):
            return "the associated design variables"
        plain = expr
        plain = re.sub(r"\\mathrm\{([^{}]+)\}", r"\1", plain)
        plain = re.sub(r"\\text\{([^{}]+)\}", r"\1", plain)
        plain = plain.replace(r"\lambda", "wavelength")
        plain = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"\1 over \2", plain)
        plain = re.sub(r"\\[A-Za-z]+", "", plain)
        plain = re.sub(r"[{}_^]", " ", plain)
        plain = re.sub(r"\s+", " ", plain).strip()
        return plain if re.search(r"[A-Za-z]", plain) else "the corresponding quantity"

    body = re.sub(r"\$([^$]+)\$", replace_inline_math, body)
    return body + suffix


def normalize_phase3_4_introduction_paragraphs(tex: str) -> str:
    """Restore a paper-facing WCL Introduction skeleton without changing claims."""
    cleaned = str(tex or "").strip()
    if not cleaned:
        return cleaned
    cleaned = cleaned.replace("the associated design variables-update", "position-update")
    cleaned = cleaned.replace("The associated design variables-update", "Position-update")
    cleaned = re.sub(r"\bmRT\b", "MRT", cleaned)
    cleaned = re.sub(r"\bA first thread\b", "Prior work", cleaned)
    cleaned = re.sub(r"\bA second thread\b", "Complementary prior work", cleaned)
    cleaned = re.sub(r"\bthe first line of work\b", "prior work", cleaned, flags=re.I)
    cleaned = re.sub(r"\bthe second line of work\b", "complementary studies", cleaned, flags=re.I)
    cleaned = re.sub(r"(\\section\*?\{Introduction\})\s*", r"\1\n", cleaned, count=1)
    cleaned = re.sub(r"\s*(\\begin\{itemize\})\s*", r"\n\n\1\n", cleaned, count=1)
    cleaned = re.sub(r"\s*(\\end\{itemize\})\s*", r"\n\1\n\n", cleaned, count=1)
    cleaned = re.sub(r"\s*\\item\s+", r"\n\\item ", cleaned)
    paragraph_starts = [
        r"A first thread",
        r"One line of work",
        r"Prior work",
        r"Existing work",
        r"In this letter,",
        r"The remainder of this (?:letter|paper) is organized as follows\.",
        r"\\textit\{Notation:\}",
    ]
    for marker in paragraph_starts:
        cleaned = re.sub(rf"(?<!\n\n)\s+({marker})", r"\n\n\1", cleaned, count=1, flags=re.I)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def sanitize_phase3_4_introduction_tex(text: str) -> str:
    cleaned = text.replace("\r\n", "\n").strip()
    for old, new in {
        "\u9225\u6503": " c",
        "\u9225\ufffd": "-",
        "–": "-",
        "—": "-",
        "’": "'",
        "“": '"',
        "”": '"',
    }.items():
        cleaned = cleaned.replace(old, new)
    cleaned = re.sub(r"^\s*Introduction\s*:?\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\\begin\{abstract\}.*?\\end\{abstract\}", "", cleaned, flags=re.I | re.S)
    cleaned = re.sub(r"\\section\*?\{Conclusion\}.*$", "", cleaned, flags=re.I | re.S)
    cleaned = re.sub(r"\\end\{document\}\s*$", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    if not re.search(r"^\s*\\section\*?\{Introduction\}", cleaned, flags=re.I):
        cleaned = "\\section{Introduction}\n" + cleaned
    cleaned = _phase3_4_demath_intro_body(cleaned)
    cleaned = normalize_phase3_4_introduction_paragraphs(cleaned)
    return cleaned.strip() + "\n"


def ensure_phase3_4_notation_paragraph(tex: str, problem_formulation_md: str = "", system_model_md: str = "") -> str:
    """Ensure the Introduction ends with organization and notation paragraphs."""
    organization = (
        "The remainder of this letter is organized as follows. "
        "Section~\\ref{sec:system_model} presents the system model and problem formulation. "
        "Section~\\ref{sec:proposed_solution} describes the proposed solution. "
        "Section~\\ref{sec:numerical_results} reports the numerical results, and "
        "Section~\\ref{sec:conclusion} concludes the letter."
    )
    notation_sentence = (
        "\\textit{Notation:} Bold lowercase and uppercase letters denote vectors and matrices, respectively. "
        "$(\\cdot)^T$ and $(\\cdot)^H$ denote transpose and Hermitian transpose, respectively. "
        "$\\|\\cdot\\|_2$ denotes the Euclidean norm, and $\\mathbb{C}$ denotes the complex field."
    )
    cleaned = tex.rstrip()
    cleaned = re.sub(
        r"(^|\n)(\s*)Notation\s*:",
        r"\1\2\\textit{Notation:}",
        cleaned,
        count=1,
        flags=re.I,
    )
    has_organization = bool(re.search(r"The remainder of this (?:letter|paper) is organized as follows\.", cleaned, flags=re.I))
    has_notation = bool(re.search(r"\\textit\{Notation:\}|\\emph\{Notation:\}|(?:^|\n)\s*Notation\s*:|Unless otherwise stated", cleaned, flags=re.I))
    if has_notation and not cleaned.rstrip().endswith("."):
        cleaned = cleaned.rstrip() + "."
    if not has_organization:
        if has_notation:
            notation_match = re.search(r"(?:\\textit\{Notation:\}|\\emph\{Notation:\}|Unless otherwise stated)", cleaned, flags=re.I)
            if notation_match:
                insert_at = notation_match.start()
                cleaned = cleaned[:insert_at].rstrip() + "\n\n" + organization + "\n\n" + cleaned[insert_at:].lstrip()
            else:
                cleaned += "\n\n" + organization
        else:
            cleaned += "\n\n" + organization
    if not has_notation:
        cleaned += "\n\n" + notation_sentence
    return cleaned.strip() + "\n"


def _phase3_4_extract_itemize_items(text: str) -> list[str]:
    match = re.search(r"\\begin\{itemize\}(.*?)\\end\{itemize\}", text, flags=re.S)
    if not match:
        return []
    return [
        re.sub(r"\s+", " ", item).strip()
        for item in re.split(r"\\item\b", match.group(1))
        if item.strip()
    ]


def _phase3_4_plain_content_tokens(text: str) -> set[str]:
    stopwords = {
        "about",
        "across",
        "against",
        "also",
        "and",
        "are",
        "because",
        "been",
        "being",
        "between",
        "both",
        "can",
        "current",
        "design",
        "does",
        "each",
        "from",
        "has",
        "have",
        "into",
        "its",
        "letter",
        "method",
        "model",
        "not",
        "our",
        "paper",
        "problem",
        "proposed",
        "same",
        "section",
        "show",
        "such",
        "that",
        "the",
        "their",
        "this",
        "through",
        "under",
        "using",
        "with",
        "without",
    }
    plain = re.sub(r"\\cite\{[^}]*\}", " ", text)
    plain = re.sub(r"\$[^$]*\$", " ", plain)
    plain = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{[^{}]*\})?", " ", plain)
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z-]{3,}", plain)
        if token.lower() not in stopwords
    }


def _phase3_4_intro_plain_sentence_stats(paragraph: str) -> tuple[int, int]:
    plain = str(paragraph or "")
    plain = re.sub(r"\\cite\{[^{}]*\}", "", plain)
    plain = re.sub(r"\$[^$]*\$", " math ", plain)
    plain = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{([^{}]*)\})?", r"\1", plain)
    plain = re.sub(r"[{}]", " ", plain)
    plain = re.sub(r"\s+", " ", plain).strip()
    word_count = len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", plain))
    sentence_count = len(re.findall(r"[.!?](?=\s|$)", plain))
    if word_count and sentence_count == 0:
        sentence_count = 1
    return word_count, sentence_count


def _phase3_4_is_contribution_lead(paragraph: str) -> bool:
    plain = re.sub(r"\\cite\{[^{}]*\}", "", str(paragraph or ""))
    plain = re.sub(r"\s+", " ", plain).strip()
    return bool(
        re.search(
            r"\b(In this (?:letter|paper)|This (?:letter|paper)|We)\b.{0,180}"
            r"\b(contributions? are summarized|main contributions?|summarized as follows|contribute)\b",
            plain,
            flags=re.I,
        )
    )


def phase3_4_intro_precontribution_paragraphs(text: str) -> list[str]:
    """Return argument paragraphs before the contribution list, excluding the lead-in."""
    before_items = str(text or "").split(r"\begin{itemize}", 1)[0]
    before_body = re.sub(r"\\section\*?\{Introduction\}", " ", before_items, flags=re.I).strip()
    paragraphs = [
        part.strip()
        for part in re.split(r"\n\s*\n", before_body)
        if part.strip()
    ]
    if paragraphs and _phase3_4_is_contribution_lead(paragraphs[-1]):
        paragraphs = paragraphs[:-1]
    return paragraphs


def _phase3_4_first_intro_argument_paragraph(text: str) -> str:
    paragraphs = phase3_4_intro_precontribution_paragraphs(text)
    return paragraphs[0] if paragraphs else ""


def analyze_phase3_4_intro_orphan_argument_paragraphs(text: str) -> list[dict[str, Any]]:
    """Find one-sentence Introduction argument paragraphs before the contribution list.

    Organization, notation, and the short "In this letter" contribution lead-in are
    intentionally excluded; the check targets orphan motivation/related-work/gap
    paragraphs that look visibly unfinished in an IEEE letter.
    """
    issues: list[dict[str, Any]] = []
    for idx, paragraph in enumerate(phase3_4_intro_precontribution_paragraphs(text), start=1):
        if re.search(r"\\begin\{itemize\}|\\end\{itemize\}", paragraph):
            continue
        word_count, sentence_count = _phase3_4_intro_plain_sentence_stats(paragraph)
        if word_count >= 10 and sentence_count < 2:
            excerpt = re.sub(r"\s+", " ", paragraph).strip()
            issues.append(
                {
                    "paragraph_index": idx,
                    "sentence_count": sentence_count,
                    "word_count": word_count,
                    "excerpt": excerpt[:240],
                }
            )
    return issues


def analyze_phase3_4_introduction_content_quality(text: str) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    notation_match = re.search(r"(?:\\textit\{Notation:\}|\\emph\{Notation:\}|(?:^|\n)\s*Notation\s*:|Unless otherwise stated)", text, flags=re.I)
    body_without_notation = text[: notation_match.start()] if notation_match else text
    inline_math = re.findall(r"\$[^$]+\$", body_without_notation)
    notation_inline_math = re.findall(r"\$[^$]+\$", text[notation_match.start():] if notation_match else "")
    display_math = re.findall(r"\\begin\{(?:equation|align|multline|gather)\*?\}", text)
    if inline_math or display_math:
        errors.append("Introduction contains mathematical notation outside the final notation paragraph; move variables and equations to the system model.")

    generic_intro_patterns = [
        r"\bis a key direction\b",
        r"\bhas attracted significant attention\b",
        r"\bplays an important role\b",
        r"\bwith the rapid development\b",
        r"\bdespite (?:this|these) progress\b",
        r"\bremains challenging\b",
        r"\bis therefore useful\b",
        r"\bprovide(?:s)? useful insights\b",
        r"\bdemonstrates? the effectiveness\b",
        r"\bthe proposed (?:scheme|method|algorithm|framework) is effective\b",
    ]
    generic_hits = [
        pattern
        for pattern in generic_intro_patterns
        if re.search(pattern, body_without_notation, flags=re.I)
    ]
    if len(generic_hits) >= 2:
        errors.append(
            "Introduction uses multiple generic IEEE-template phrases; rewrite motivation, gap, and contributions around the concrete wireless design tension."
        )

    weak_deictic_study_patterns = [
        r"\bThis (?:letter|paper) studies this (?:bottleneck|challenge|problem|issue)\b",
        r"\bThis (?:letter|paper) investigates this (?:bottleneck|challenge|problem|issue)\b",
        r"\bThis (?:letter|paper) considers this (?:bottleneck|challenge|problem|issue)\b",
    ]
    weak_deictic_study_hits = [
        pattern
        for pattern in weak_deictic_study_patterns
        if re.search(pattern, body_without_notation, flags=re.I)
    ]
    if weak_deictic_study_hits:
        errors.append(
            "Introduction uses a weak deictic study sentence such as 'This letter studies this bottleneck'; "
            "rewrite the opening argument so the exact system setting is part of the concrete gap or contribution transition."
        )

    first_paragraph = _phase3_4_first_intro_argument_paragraph(body_without_notation)
    first_plain = re.sub(r"\\cite\{[^{}]*\}", " ", first_paragraph)
    first_plain = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{([^{}]*)\})?", r"\1", first_plain)
    first_plain = re.sub(r"\s+", " ", first_plain).strip()
    first_para_setup_patterns = [
        r"\b(?:we|this (?:letter|paper)) (?:consider|study|investigate|focus on|address)\b.{0,180}\bwhere\b",
        r"\bnarrowband\b.{0,80}\bsingle-cell\b",
        r"\bsingle-cell\b.{0,80}\bdownlink\b.{0,80}\b(?:where|with)\b",
        r"\b(?:uniform linear array|ULA)\b.{0,100}\bserves\b",
        r"\bserves\b.{0,80}\bsingle-antenna users\b",
        r"\b(?:perfect|imperfect) (?:instantaneous )?(?:CSI|channel state information)\b",
    ]
    first_para_setup_hits = [
        pattern
        for pattern in first_para_setup_patterns
        if re.search(pattern, first_plain, flags=re.I)
    ]
    if first_para_setup_hits:
        errors.append(
            "Introduction first paragraph reads like a system-model setup; keep it at the general background, promise, and challenge level, "
            "and move topology, antenna geometry, user type, CSI assumptions, or narrowband/single-cell details to the contribution transition or Section II."
        )

    prohibited_meta_patterns = [
        r"\btechnical axis\b",
        r"\bfirst literature axis\b",
        r"\bsecond literature axis\b",
        r"\bmissing capability\b",
        r"\bexact current topic\b",
        r"\bcurrent paper\b",
        r"\bthis paper's positive writing plan\b",
    ]
    prohibited_meta_hits = [
        pattern
        for pattern in prohibited_meta_patterns
        if re.search(pattern, body_without_notation, flags=re.I)
    ]
    if prohibited_meta_hits:
        errors.append(
            "Introduction uses meta-structural workflow/taxonomy language instead of IEEE WCL prose; "
            "rewrite it around a concrete bottleneck, closest-work limitation, and direct contribution statement."
        )

    taxonomy_patterns = [
        r"\ba line of work\b",
        r"\banother line of work\b",
        r"\ba complementary line of work\b",
        r"\bprior works (?:have )?(?:studied|investigated|considered|explored)\b",
        r"\bexisting (?:works|studies) (?:have )?(?:studied|investigated|considered|explored)\b",
        r"\brecent (?:works|studies) (?:have )?(?:studied|investigated|considered|explored)\b",
    ]
    taxonomy_hits = [
        pattern
        for pattern in taxonomy_patterns
        if re.search(pattern, body_without_notation, flags=re.I)
    ]
    if len(taxonomy_hits) >= 2:
        warnings.append(
            "Introduction may read like a taxonomy of literatures; related-work paragraphs should be organized around the paper's design conflict and closest-work limitations."
        )
    orphan_argument_paragraphs = analyze_phase3_4_intro_orphan_argument_paragraphs(text)
    if orphan_argument_paragraphs:
        errors.append(
            "Introduction contains one-sentence motivation/related-work/gap paragraph(s) before the contribution list; "
            "merge or expand each orphan paragraph so every argument paragraph has a thesis, support, and gap/transition."
        )

    long_citation_clusters = [
        match.group(0)
        for match in re.finditer(r"\\cite\{([^{}]+)\}", body_without_notation)
        if len([key for key in match.group(1).split(",") if key.strip()]) > 3
    ]
    if long_citation_clusters:
        warnings.append(
            "Introduction contains citation clusters with more than three keys; split or focus them so each sentence makes a precise supported claim."
        )

    detail_patterns = [
        (r"\bone-dimensional search\b|\bscalar search\b|\bper-user update\b|\bper-block update\b|\bblock-update\b", "algorithm implementation detail"),
        (r"\balternates? between\b.{0,120}\b(?:updates?|blocks?|variables?)\b", "algorithm implementation detail"),
        (r"\bexact signal model\b|\bclosed-form metric expression\b|\bfull constraint list\b", "system-model detail"),
        (r"\bminimum .* constraint\b|\bmaximum .* constraint\b|\bsum .* budget\b", "formulation constraint detail"),
        (r"\bkey performance indicators?\b|\bKPIs?\b", "numerical-evidence metric list"),
    ]
    found_details = [
        label
        for pattern, label in detail_patterns
        if re.search(pattern, text, flags=re.I)
    ]
    if found_details:
        errors.append("Introduction includes details that belong in later sections: " + ", ".join(sorted(set(found_details))) + ".")

    result_patterns = [
        r"\d+(?:\.\d+)?\s*\\?%",
        r"\bmedian (?:relative )?gains?\b",
        r"\bmean (?:relative )?gains?\b",
        r"\bwin rate\b",
        r"\bfeasibility rate\b",
        r"\bapproximately\s+\d",
    ]
    if any(re.search(pattern, body_without_notation, flags=re.I) for pattern in result_patterns):
        errors.append("Introduction previews numerical-result values or evaluation statistics; leave them to Numerical Results.")

    itemize_items = _phase3_4_extract_itemize_items(text)
    itemize_match = re.search(r"\\begin\{itemize\}(.*?)\\end\{itemize\}", text, flags=re.S)
    itemize_block = itemize_match.group(1) if itemize_match else ""
    if not itemize_items:
        errors.append("Introduction must end with a concise itemized contribution list.")
    if len(itemize_items) > 3:
        warnings.append(f"Introduction has {len(itemize_items)} contribution bullets; WCL-style framing should use at most three concise bullets.")
    if itemize_items:
        before_items = text.split(r"\begin{itemize}", 1)[0]
        before_body = re.sub(r"\\section\*?\{Introduction\}", " ", before_items, flags=re.I).strip()
        pre_contribution_paragraphs = [
            part.strip()
            for part in re.split(r"\n\s*\n", before_body)
            if part.strip()
        ]
        if len(pre_contribution_paragraphs) != 4:
            warnings.append(
                "Introduction should have exactly three pre-contribution paragraphs plus a short contribution lead-in before itemize; "
                f"found {len(pre_contribution_paragraphs)} text blocks before itemize."
            )
        contribution_lead_in = before_items.rstrip().split("\n\n")[-1]
        prior_tokens = _phase3_4_plain_content_tokens(contribution_lead_in)
        repeated_items: list[int] = []
        for idx, item in enumerate(itemize_items, start=1):
            item_tokens = _phase3_4_plain_content_tokens(item)
            if len(item_tokens) >= 8:
                overlap_ratio = len(item_tokens & prior_tokens) / max(len(item_tokens), 1)
                if overlap_ratio >= 0.62:
                    repeated_items.append(idx)
        if repeated_items:
            errors.append(
                "Contribution bullets repeat earlier prose instead of adding new positioning: "
                + ", ".join(f"item {idx}" for idx in repeated_items)
                + "."
            )
        low_information_items = [
            idx
            for idx, item in enumerate(itemize_items, start=1)
            if re.search(
                r"^\s*(?:we\s+)?(?:formulate|develop|provide|present|conduct|perform)\s+(?:a|an|the)?\s*(?:problem|algorithm|framework|simulations?|method)\b",
                re.sub(r"\\item(?:\[[^\]]*\])?", "", item).strip(),
                flags=re.I,
            )
        ]
        if len(low_information_items) >= 2:
            errors.append(
                "Contribution bullets are low-information section summaries instead of capability claims: "
                + ", ".join(f"item {idx}" for idx in low_information_items)
                + "."
            )
    contribution_outcome_patterns = [
        r"\bshow(?:s|ing)?\b.{0,80}\b(?:improvement|gain|outperform(?:s|ed|ing)?|superior|higher|better)\b",
        r"\bdemonstrat(?:e|es|ing)\b.{0,80}\b(?:improvement|gain|outperform(?:s|ed|ing)?|superior|higher|better)\b",
        r"\bindicat(?:e|es|ing)\b.{0,80}\b(?:improvement|gain|outperform(?:s|ed|ing)?|superior|higher|better|reduce|reduction|lower)\b",
        r"\byield(?:s|ing)?\b.{0,80}\b(?:improvement|gain|outperform(?:s|ed|ing)?|superior|higher|better)\b",
        r"\bconsistent utility improvements\b",
    ]
    contribution_has_outcome = itemize_block and any(
        re.search(pattern, itemize_block, flags=re.I) for pattern in contribution_outcome_patterns
    )
    if contribution_has_outcome:
        scoped_evidence_pattern = (
            r"\bunder the considered settings\b|"
            r"\bin the evaluated scenarios\b|"
            r"\bover the tested configurations\b|"
            r"\bfor the considered settings\b|"
            r"\brelative to\b|"
            r"\bcompared with\b|"
            r"\bagainst\b"
        )
        universal_outcome_pattern = (
            r"\balways\b|\buniversally\b|\bguarantees?\b|\bproves?\b|"
            r"\bstatistically significant\b|\ball (?:settings|scenarios|configurations)\b"
        )
        if not re.search(scoped_evidence_pattern, itemize_block, flags=re.I):
            errors.append(
                "Contribution list claims numerical gains without scope; use a qualitative statement tied to the considered settings and plotted benchmarks."
            )
        if re.search(universal_outcome_pattern, itemize_block, flags=re.I):
            errors.append(
                "Contribution list overstates numerical evidence; avoid universal or proof-like superiority claims in the Introduction."
            )
    has_organization = bool(re.search(r"The remainder of this (?:letter|paper) is organized as follows\.", text, flags=re.I))
    if not has_organization:
        errors.append('Introduction must include an organization paragraph beginning "The remainder of this letter is organized as follows."')
    if not notation_match:
        errors.append('Introduction must end with a concise "\\textit{Notation:}" paragraph.')
    elif not re.search(r"\\textit\{Notation:\}|\\emph\{Notation:\}", text[notation_match.start():], flags=re.I):
        warnings.append("Notation paragraph should begin with \\textit{Notation:}.")

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "inline_math_count": len(inline_math),
        "notation_inline_math_count": len(notation_inline_math),
        "display_math_count": len(display_math),
        "itemize_count": len(itemize_items),
        "generic_intro_phrase_count": len(generic_hits),
        "weak_deictic_study_phrase_count": len(weak_deictic_study_hits),
        "first_paragraph_system_setup_phrase_count": len(first_para_setup_hits),
        "prohibited_meta_phrase_count": len(prohibited_meta_hits),
        "taxonomy_phrase_count": len(taxonomy_hits),
        "long_citation_cluster_count": len(long_citation_clusters),
        "orphan_argument_paragraphs": orphan_argument_paragraphs,
    }


def validate_phase3_4_introduction_contract(run_dir: Path, introduction_tex: str) -> dict[str, Any]:
    """Enforce the Phase 3.4 introduction structure before the paper is assembled."""
    text = str(introduction_tex or "")
    body = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{[^{}]*\})?", " ", text)
    word_count = len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", body))
    errors: list[str] = []
    warnings: list[str] = []
    if not re.search(r"^\s*\\section\*?\{Introduction\}", text, flags=re.I):
        errors.append("Introduction must begin with \\section{Introduction}.")
    if word_count < 350:
        warnings.append(f"Introduction is quite short for a complete WCL opening: {word_count} words < 350.")
    if word_count > 1200:
        warnings.append(f"Introduction may be over-expanded for WCL-style prose: {word_count} words > 1200.")
    if "\\begin{itemize}" not in text:
        errors.append("Introduction is missing the required itemized contribution list.")
    if "The remainder of this letter is organized as follows." not in text:
        errors.append("Introduction is missing the required paper-organization paragraph.")
    notation_pattern = r"\\textit\{Notation:\}|\\emph\{Notation:\}|(?:^|\n)\s*Notation\s*:|Unless otherwise stated"
    if not re.search(notation_pattern, text, flags=re.I):
        errors.append("Introduction is missing the required notation paragraph.")
    if re.search(r"The first .*The second .*The third", re.sub(r"\s+", " ", text), flags=re.I):
        errors.append("Related work uses ordinal card-summary wording instead of natural thematic prose.")
    forbidden_terms = ["pipeline", "Phase 2.4", "Phase 2.5", "LLM", "Codex", "generated_plugin", "draft", "preliminary"]
    found_forbidden = [term for term in forbidden_terms if re.search(rf"\b{re.escape(term)}\b", text)]
    if found_forbidden:
        errors.append("Introduction contains forbidden internal/workflow terms: " + ", ".join(found_forbidden))
    content_quality = analyze_phase3_4_introduction_content_quality(text)
    errors.extend(content_quality["errors"])
    warnings.extend(content_quality["warnings"])
    report = {
        "ok": not errors,
        "word_count": word_count,
        "errors": errors,
        "warnings": warnings,
        "has_itemized_contributions": "\\begin{itemize}" in text,
        "has_organization": "The remainder of this letter is organized as follows." in text,
        "has_notation": bool(re.search(notation_pattern, text, flags=re.I)),
        "content_quality": content_quality,
    }
    phase3_4_dir = Path(run_dir) / "phase3-4"
    write_text(phase3_4_dir / "phase3_4_introduction_contract_report.json", json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise ValueError("Phase 3.4 introduction writing contract failed: " + "; ".join(errors))
    return report


def build_phase3_4_reference_check_prompt(
    *,
    introduction_facts_json: str,
    verified_reference_bank_json: str,
    selected_reference_keys_json: str,
    citation_claim_map_json: str,
    reference_quality_report_json: str,
) -> str:
    return render_prompt_template(
        "phase3_4/reference_check.prompt.yaml",
        introduction_facts_json=introduction_facts_json,
        verified_reference_bank_json=verified_reference_bank_json,
        selected_reference_keys_json=selected_reference_keys_json,
        citation_claim_map_json=citation_claim_map_json,
        reference_quality_report_json=reference_quality_report_json,
    )


def build_phase3_4_technical_citation_prompt(
    *,
    current_paper_brief_json: str,
    verified_reference_bank_json: str,
    reference_quality_report_json: str,
    introduction_citation_claim_map_json: str,
    system_model_problem_formulation_section_tex: str,
    proposed_solution_section_tex: str,
    numerical_results_section_tex: str,
) -> str:
    return render_prompt_template(
        "phase3_4/technical_citation.prompt.yaml",
        current_paper_brief_json=current_paper_brief_json,
        verified_reference_bank_json=verified_reference_bank_json,
        reference_quality_report_json=reference_quality_report_json,
        introduction_citation_claim_map_json=introduction_citation_claim_map_json,
        system_model_problem_formulation_section_tex=system_model_problem_formulation_section_tex,
        proposed_solution_section_tex=proposed_solution_section_tex,
        numerical_results_section_tex=numerical_results_section_tex,
    )


_PHASE33_TECHNICAL_SECTION_KEYS = {
    "system_model_problem_formulation_section_tex": "system_model",
    "proposed_solution_section_tex": "proposed_solution",
    "numerical_results_section_tex": "numerical_results",
}


def _phase3_4_strip_citations_for_compare(tex: str) -> str:
    """Normalize text after removing citations so citation-only edits can be audited."""
    cleaned = re.sub(r"\s*\\cite\w*\{[^{}]*\}", "", str(tex or ""))
    cleaned = re.sub(r"\s+([,.;:])", r"\1", cleaned)
    cleaned = re.sub(r"([(\[{])\s+", r"\1", cleaned)
    cleaned = re.sub(r"\s+([\]\)}])", r"\1", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _phase3_4_inline_math_segments(tex: str) -> list[str]:
    """Return likely inline-math bodies without treating closing dollars as openers."""

    segments: list[str] = []
    text = str(tex or "")
    pattern = re.compile(r"(?<!\\)\$(?![\s,.;:!?])(?P<body>[^$\n]*?)(?<!\\)\$")
    for match in pattern.finditer(text):
        body = match.group("body")
        if body.strip():
            segments.append(body)
    return segments


def _phase3_4_section_has_citable_context(tex: str, section_name: str) -> bool:
    text = re.sub(r"\\begin\{[^{}]+\}.*?\\end\{[^{}]+\}", " ", str(tex or ""), flags=re.S).lower()
    text = re.sub(r"\$[^$]*\$", " ", text)
    common_cues = [
        "channel",
        "fading",
        "path-loss",
        "path loss",
        "uncertainty",
        "csi",
        "beamforming",
        "resource allocation",
        "energy harvesting",
        "receiver",
        "transmitter",
        "uav",
        "ris",
        "mimo",
        "near-field",
        "far-field",
        "robust",
        "optimization",
    ]
    method_cues = [
        "successive convex approximation",
        "sca",
        "alternating optimization",
        "block coordinate",
        "bcd",
        "wmmse",
        "sdr",
        "semidefinite relaxation",
        "gaussian randomization",
        "kkt",
        "penalty",
        "projected",
        "majorization",
        "minorization",
    ]
    benchmark_cues = [
        "benchmark",
        "baseline",
        "comparison",
        "compared with",
        "fixed",
        "linear",
        "equal allocation",
        "random",
        "maximum-ratio",
        "mrt",
        "zero-forcing",
    ]
    if section_name == "system_model":
        cues = common_cues
    elif section_name == "proposed_solution":
        cues = common_cues + method_cues
    else:
        cues = benchmark_cues + ["monte carlo", "simulation setup"]
    return any(cue in text for cue in cues)


_PHASE33_CITATION_TOKEN_STOPWORDS = {
    "about",
    "across",
    "after",
    "also",
    "and",
    "because",
    "before",
    "between",
    "both",
    "from",
    "into",
    "only",
    "over",
    "that",
    "their",
    "then",
    "this",
    "through",
    "under",
    "using",
    "when",
    "where",
    "which",
    "while",
    "with",
}


def _phase3_4_reference_search_text(item: dict[str, Any]) -> str:
    parts: list[str] = []
    for field in [
        "final_title",
        "candidate_title",
        "venue",
        "source_type",
        "publisher",
        "reason_for_inclusion",
    ]:
        value = item.get(field)
        if value:
            parts.append(str(value))
    used_for_claims = item.get("used_for_claims", [])
    if isinstance(used_for_claims, list):
        parts.extend(str(value) for value in used_for_claims if value)
    else:
        parts.append(str(used_for_claims))
    return " ".join(parts).lower()


def _phase3_4_citation_tokens(text: str) -> set[str]:
    tokens = {
        token
        for token in re.findall(r"[A-Za-z][A-Za-z0-9-]{3,}", str(text or "").lower())
        if token not in _PHASE33_CITATION_TOKEN_STOPWORDS
    }
    return tokens


def _phase3_4_claim_map_key_score(
    *,
    key: str,
    section_text: str,
    section_name: str,
    introduction_citation_claim_map: list[dict[str, Any]],
) -> int:
    section_tokens = _phase3_4_citation_tokens(section_text)
    score = 0
    for item in introduction_citation_claim_map:
        if not isinstance(item, dict):
            continue
        citation_keys = item.get("citation_keys", [])
        if key not in {str(value).strip() for value in citation_keys if str(value).strip()}:
            continue
        claim_text = str(item.get("claim_text", ""))
        claim_tokens = _phase3_4_citation_tokens(claim_text)
        score += min(4, len(section_tokens & claim_tokens))
        claim_type = str(item.get("claim_type", "")).lower()
        if section_name == "proposed_solution" and claim_type in {"related_work", "method_foundation"}:
            score += 2
        if section_name == "system_model" and claim_type in {"background", "related_work"}:
            score += 1
        if section_name == "numerical_results" and claim_type in {"related_work", "background"}:
            score += 1
    return score


def _phase3_4_select_section_citation_keys(
    *,
    section_tex: str,
    section_name: str,
    verified_reference_bank: list[dict[str, Any]],
    introduction_citation_claim_map: list[dict[str, Any]],
    valid_reference_keys: set[str],
    max_keys: int = 2,
) -> list[str]:
    section_text = str(section_tex or "")
    section_lower = section_text.lower()
    section_tokens = _phase3_4_citation_tokens(section_text)
    if not section_tokens:
        return []

    model_terms = {
        "channel",
        "fading",
        "path-loss",
        "path loss",
        "uncertainty",
        "csi",
        "receiver",
        "transmitter",
        "mec",
        "edge",
        "semantic",
        "offloading",
        "beamforming",
        "energy",
        "uav",
        "ris",
        "mimo",
    }
    method_terms = {
        "optimization",
        "resource allocation",
        "successive convex approximation",
        "sca",
        "alternating optimization",
        "block coordinate",
        "bcd",
        "wmmse",
        "sdr",
        "semidefinite",
        "relaxation",
        "randomization",
        "finite-blocklength",
        "finite blocklength",
        "offloading",
    }
    benchmark_terms = {
        "benchmark",
        "baseline",
        "heuristic",
        "comparison",
        "equal power",
        "channel inversion",
        "mrt",
        "zero-forcing",
        "random",
    }

    if section_name == "system_model":
        preferred_terms = model_terms
        preferred_claims = {"model", "related", "background", "closest"}
    elif section_name == "proposed_solution":
        preferred_terms = method_terms
        preferred_claims = {"method", "optimization", "resource", "closest", "related"}
    else:
        preferred_terms = benchmark_terms
        preferred_claims = {"benchmark", "evaluation", "related", "background"}

    scored: list[tuple[int, str]] = []
    for item in verified_reference_bank:
        if not isinstance(item, dict) or not phase3_4_reference_is_final_usable(item):
            continue
        key = str(item.get("final_bib_key", "")).strip()
        if not key or key not in valid_reference_keys:
            continue
        ref_text = _phase3_4_reference_search_text(item)
        ref_tokens = _phase3_4_citation_tokens(ref_text)
        score = 0
        score += min(8, len(section_tokens & ref_tokens))
        for term in preferred_terms:
            if term in section_lower and term in ref_text:
                score += 3
        for claim in preferred_claims:
            if claim in ref_text:
                score += 2
        if "ieee" in ref_text:
            score += 1
        if str(item.get("reliability", "")).lower() == "high":
            score += 1
        score += _phase3_4_claim_map_key_score(
            key=key,
            section_text=section_text,
            section_name=section_name,
            introduction_citation_claim_map=introduction_citation_claim_map,
        )
        if score > 0:
            scored.append((score, key))

    scored.sort(key=lambda pair: (-pair[0], pair[1].lower()))
    selected: list[str] = []
    for _, key in scored:
        if key not in selected:
            selected.append(key)
        if len(selected) >= max_keys:
            break
    return selected


def _phase3_4_insert_citation_in_first_prose_sentence(tex: str, citation_keys: list[str]) -> str:
    keys = [str(key).strip() for key in citation_keys if str(key).strip()]
    if not keys:
        return tex
    citation = r"\cite{" + ",".join(keys) + "}"
    lines = str(tex or "").splitlines()
    in_display_env = False
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if re.search(r"\\begin\{(?:equation|align|aligned|split|multline|gather|subequations|algorithm|algorithmic)\*?\}", stripped):
            in_display_env = True
        if in_display_env:
            if re.search(r"\\end\{(?:equation|align|aligned|split|multline|gather|subequations|algorithm|algorithmic)\*?\}", stripped):
                in_display_env = False
            continue
        if stripped.startswith("\\"):
            continue
        if r"\cite" in line:
            continue
        if not re.search(r"[.!?]", line):
            continue
        lines[idx] = re.sub(r"([.!?])", lambda match: f" {citation}{match.group(1)}", line, count=1)
        return "\n".join(lines)
    return tex


def _phase3_4_try_auto_repair_missing_technical_citations(
    *,
    candidate_sections: dict[str, str],
    validation_report: dict[str, Any],
    verified_reference_bank: list[dict[str, Any]],
    introduction_citation_claim_map: list[dict[str, Any]],
    original_sections: dict[str, str],
    valid_reference_keys: set[str],
) -> tuple[dict[str, str], dict[str, Any], list[dict[str, Any]]]:
    errors = [str(item) for item in validation_report.get("errors", [])]
    missing_errors = [
        item
        for item in errors
        if item.endswith(": citable model/method context is present but no technical citation was inserted.")
    ]
    non_repairable_errors = [
        item
        for item in errors
        if item not in missing_errors and item != "No technical-section citations were inserted."
    ]
    if not missing_errors or non_repairable_errors:
        return candidate_sections, validation_report, []

    section_name_to_key = {value: key for key, value in _PHASE33_TECHNICAL_SECTION_KEYS.items()}
    repaired_sections = dict(candidate_sections)
    repair_log: list[dict[str, Any]] = []
    for error in missing_errors:
        section_name = error.split(":", 1)[0].strip()
        section_key = section_name_to_key.get(section_name)
        if not section_key:
            continue
        if extract_citation_keys_from_tex(repaired_sections.get(section_key, "")):
            continue
        citation_keys = _phase3_4_select_section_citation_keys(
            section_tex=repaired_sections.get(section_key, "") or original_sections.get(section_key, ""),
            section_name=section_name,
            verified_reference_bank=verified_reference_bank,
            introduction_citation_claim_map=introduction_citation_claim_map,
            valid_reference_keys=valid_reference_keys,
        )
        if not citation_keys:
            continue
        before = repaired_sections.get(section_key, "")
        after = _phase3_4_insert_citation_in_first_prose_sentence(before, citation_keys)
        if after != before:
            repaired_sections[section_key] = after
            repair_log.append(
                {
                    "section": section_name,
                    "citation_keys": citation_keys,
                    "repair": "inserted citation in the first suitable prose sentence",
                }
            )

    if not repair_log:
        return candidate_sections, validation_report, []

    repaired_report = validate_phase3_4_technical_citation_only_revision(
        original_sections=original_sections,
        revised_sections=repaired_sections,
        valid_reference_keys=valid_reference_keys,
    )
    if repaired_report.get("ok"):
        repaired_report["auto_repaired_missing_technical_citations"] = repair_log
        return repaired_sections, repaired_report, repair_log
    return candidate_sections, validation_report, []


def validate_phase3_4_technical_citation_only_revision(
    *,
    original_sections: dict[str, str],
    revised_sections: dict[str, str],
    valid_reference_keys: set[str],
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    valid_keys = {str(key).strip() for key in valid_reference_keys if str(key).strip()}

    for key, section_name in _PHASE33_TECHNICAL_SECTION_KEYS.items():
        original = str(original_sections.get(key, ""))
        revised = str(revised_sections.get(key, ""))
        if not revised.strip():
            errors.append(f"{section_name}: returned section is empty.")
            continue
        if _phase3_4_strip_citations_for_compare(original) != _phase3_4_strip_citations_for_compare(revised):
            errors.append(f"{section_name}: non-citation technical text was changed.")

        citation_keys = extract_citation_keys_from_tex(revised)
        unknown_keys = [item for item in citation_keys if item not in valid_keys]
        if unknown_keys:
            errors.append(f"{section_name}: unknown citation keys: {', '.join(sorted(set(unknown_keys)))}.")

        math_env_pattern = r"\\begin\{([^{}]+)\}.*?\\end\{\1\}"
        for env_match in re.finditer(math_env_pattern, revised, flags=re.S):
            env_name = env_match.group(1).rstrip("*")
            if env_name in {
                "equation",
                "align",
                "aligned",
                "split",
                "multline",
                "gather",
                "subequations",
                "algorithm",
                "algorithmic",
            } and re.search(r"\\cite\w*\{", env_match.group(0)):
                errors.append(f"{section_name}: citation placed inside {env_name} environment.")
        if any(re.search(r"\\cite\w*\{", segment) for segment in _phase3_4_inline_math_segments(revised)):
            errors.append(f"{section_name}: citation placed inside inline math.")
        if re.search(r"\\(?:caption|label|includegraphics)\s*(?:\[[^\]]*\])?\{[^{}]*\\cite\w*\{", revised, flags=re.S):
            errors.append(f"{section_name}: citation placed inside a caption, label, or includegraphics command.")
        if section_name in {"system_model", "proposed_solution"} and _phase3_4_section_has_citable_context(original, section_name) and not citation_keys:
            errors.append(f"{section_name}: citable model/method context is present but no technical citation was inserted.")
        if section_name == "numerical_results" and _phase3_4_section_has_citable_context(original, section_name) and not citation_keys:
            warnings.append(f"{section_name}: benchmark or evaluation context appears citation-free.")

    total_citations = sum(
        len(extract_citation_keys_from_tex(str(revised_sections.get(key, ""))))
        for key in _PHASE33_TECHNICAL_SECTION_KEYS
    )
    if total_citations == 0:
        errors.append("No technical-section citations were inserted.")

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "technical_citation_count": total_citations,
    }


def build_phase3_4_technical_citation_claim_map(
    section_tex: dict[str, str],
    *,
    final_reference_keys: set[str],
    arxiv_only_keys: set[str],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for key, section_name in _PHASE33_TECHNICAL_SECTION_KEYS.items():
        tex = str(section_tex.get(key, ""))
        prose = re.sub(
            r"\\begin\{(?:equation|align|aligned|split|multline|gather|subequations|algorithm|algorithmic)\*?\}.*?\\end\{(?:equation|align|aligned|split|multline|gather|subequations|algorithm|algorithmic)\*?\}",
            " ",
            tex,
            flags=re.S,
        )
        sentences = re.split(r"(?<=[.!?])\s+", prose.replace("\n", " "))
        for idx, sentence in enumerate(sentences, start=1):
            keys = extract_citation_keys_from_tex(sentence)
            if not keys:
                continue
            if any(item not in final_reference_keys for item in keys):
                support_status = "unsupported"
            elif any(item in arxiv_only_keys for item in keys):
                support_status = "weak_support"
            else:
                support_status = "supported"
            output.append(
                {
                    "sentence_id": f"{section_name}_s{idx}",
                    "section": section_name,
                    "claim_text": " ".join(sentence.split()),
                    "citation_keys": keys,
                    "claim_type": "technical_context",
                    "support_status": support_status,
                    "notes": "" if support_status == "supported" else (
                        "Uses arXiv-only support." if support_status == "weak_support" else "Contains citation keys absent from final verified references."
                    ),
                }
            )
    return output


def call_llm_phase3_4_technical_citation_pass(
    *,
    model_profile: str,
    phase3_4_dir: Path,
    current_paper_brief: dict[str, Any],
    verified_reference_bank: list[dict[str, Any]],
    reference_quality_report: dict[str, Any],
    introduction_citation_claim_map: list[dict[str, Any]],
    system_model_problem_formulation_section_tex: str,
    proposed_solution_section_tex: str,
    numerical_results_section_tex: str,
) -> tuple[dict[str, str], list[dict[str, Any]], dict[str, Any]]:
    valid_reference_keys = {
        str(item.get("final_bib_key", "")).strip()
        for item in verified_reference_bank
        if isinstance(item, dict) and phase3_4_reference_is_final_usable(item)
    }
    original_sections = {
        "system_model_problem_formulation_section_tex": system_model_problem_formulation_section_tex,
        "proposed_solution_section_tex": proposed_solution_section_tex,
        "numerical_results_section_tex": numerical_results_section_tex,
    }
    prompt = build_phase3_4_technical_citation_prompt(
        current_paper_brief_json=compact_text(json.dumps(current_paper_brief, ensure_ascii=False, indent=2), 8000),
        verified_reference_bank_json=compact_text(json.dumps(verified_reference_bank, ensure_ascii=False, indent=2), 16000),
        reference_quality_report_json=compact_text(json.dumps(reference_quality_report, ensure_ascii=False, indent=2), 4000),
        introduction_citation_claim_map_json=compact_text(json.dumps(introduction_citation_claim_map, ensure_ascii=False, indent=2), 8000),
        system_model_problem_formulation_section_tex=system_model_problem_formulation_section_tex,
        proposed_solution_section_tex=proposed_solution_section_tex,
        numerical_results_section_tex=numerical_results_section_tex,
    )
    write_text(phase3_4_dir / "technical_citation_prompt.txt", prompt)
    if str(os.environ.get("WARA_SKIP_TECHNICAL_CITATION_LLM", "")).strip().lower() in {"1", "true", "yes", "on"}:
        initial_report = validate_phase3_4_technical_citation_only_revision(
            original_sections=original_sections,
            revised_sections=original_sections,
            valid_reference_keys=valid_reference_keys,
        )
        candidate_sections, validation_report, repair_log = _phase3_4_try_auto_repair_missing_technical_citations(
            candidate_sections=original_sections,
            validation_report=initial_report,
            verified_reference_bank=verified_reference_bank,
            introduction_citation_claim_map=introduction_citation_claim_map,
            original_sections=original_sections,
            valid_reference_keys=valid_reference_keys,
        )
        if not repair_log:
            candidate_sections = original_sections
            validation_report = initial_report
        validation_report.setdefault("warnings", [])
        validation_report["warnings"].append(
            "Technical-citation LLM pass skipped; deterministic local citation insertion was used with verified reference keys."
        )
        validation_report["bounded_local_citation_pass"] = True
        validation_report["local_citation_insertions"] = repair_log
        technical_citation_map = build_phase3_4_technical_citation_claim_map(
            candidate_sections,
            final_reference_keys=valid_reference_keys,
            arxiv_only_keys={
                str(item.get("final_bib_key", "")).strip()
                for item in verified_reference_bank
                if str(item.get("verification_status", "")).strip().lower() == "arxiv_only"
            },
        )
        write_text(
            phase3_4_dir / "technical_citation_raw_response.txt",
            json.dumps(
                {
                    "sections": candidate_sections,
                    "bounded_local_citation_pass": True,
                    "local_citation_insertions": repair_log,
                },
                ensure_ascii=False,
                indent=2,
            ),
        )
        if repair_log:
            write_text(phase3_4_dir / "technical_citation_auto_repair.json", json.dumps(repair_log, ensure_ascii=False, indent=2))
        write_text(phase3_4_dir / "technical_citation_map.json", json.dumps(technical_citation_map, ensure_ascii=False, indent=2))
        write_text(phase3_4_dir / "technical_citation_contract_report.json", json.dumps(validation_report, ensure_ascii=False, indent=2))
        return candidate_sections, technical_citation_map, validation_report

    llm = create_llm_client(model_profile)
    thinking = {"type": "enabled"} if model_profile == "kimi-k2.6-thinking" else None
    raw_response_text = ""
    payload: dict[str, Any] | None = None
    validation_report: dict[str, Any] = {"ok": False, "errors": ["technical citation pass was not run"], "warnings": []}
    for attempt in range(2):
        attempt_prompt = prompt
        if attempt and payload is not None:
            attempt_prompt = (
                prompt
                + "\n\nCitation-only repair requirement:\n"
                + "- The previous response failed the deterministic citation-only validator.\n"
                + "- Return the same three sections again, but change nothing except adding `\\cite{...}` commands.\n"
                + "- Do not add sentences, remove words, edit equations, edit captions, or use keys outside verified_reference_bank.\n"
                + "- Validator errors:\n"
                + "\n".join(f"  - {item}" for item in validation_report.get("errors", []))
                + "\n\nPrevious response JSON:\n"
                + json.dumps(payload, ensure_ascii=False, indent=2)
            )
        response = llm.chat(
            [{"role": "user", "content": attempt_prompt}],
            json_mode=True,
            thinking=thinking,
            max_tokens=12000,
        )
        raw_response_text = response.content
        payload = _safe_json_loads(response.content, {})
        if not isinstance(payload, dict):
            validation_report = {"ok": False, "errors": ["LLM did not return a JSON object."], "warnings": []}
            continue
        candidate_sections = {
            key: normalize_phase3_4_citation_aliases(str(payload.get(key, "")), valid_reference_keys)
            for key in _PHASE33_TECHNICAL_SECTION_KEYS
        }
        validation_report = validate_phase3_4_technical_citation_only_revision(
            original_sections=original_sections,
            revised_sections=candidate_sections,
            valid_reference_keys=valid_reference_keys,
        )
        if validation_report.get("ok"):
            write_text(phase3_4_dir / "technical_citation_raw_response.txt", raw_response_text)
            technical_citation_map = build_phase3_4_technical_citation_claim_map(
                candidate_sections,
                final_reference_keys=valid_reference_keys,
                arxiv_only_keys={
                    str(item.get("final_bib_key", "")).strip()
                    for item in verified_reference_bank
                    if str(item.get("verification_status", "")).strip().lower() == "arxiv_only"
                },
            )
            if not technical_citation_map and isinstance(payload.get("technical_citation_map"), list):
                technical_citation_map = [
                    item for item in payload.get("technical_citation_map", [])
                    if isinstance(item, dict)
                ]
            write_text(phase3_4_dir / "technical_citation_map.json", json.dumps(technical_citation_map, ensure_ascii=False, indent=2))
            write_text(phase3_4_dir / "technical_citation_contract_report.json", json.dumps(validation_report, ensure_ascii=False, indent=2))
            return candidate_sections, technical_citation_map, validation_report

    if payload is not None and isinstance(payload, dict):
        candidate_sections = {
            key: normalize_phase3_4_citation_aliases(str(payload.get(key, "")), valid_reference_keys)
            for key in _PHASE33_TECHNICAL_SECTION_KEYS
        }
        repaired_sections, repaired_report, repair_log = _phase3_4_try_auto_repair_missing_technical_citations(
            candidate_sections=candidate_sections,
            validation_report=validation_report,
            verified_reference_bank=verified_reference_bank,
            introduction_citation_claim_map=introduction_citation_claim_map,
            original_sections=original_sections,
            valid_reference_keys=valid_reference_keys,
        )
        if repaired_report.get("ok"):
            write_text(phase3_4_dir / "technical_citation_raw_response.txt", raw_response_text)
            write_text(phase3_4_dir / "technical_citation_auto_repair.json", json.dumps(repair_log, ensure_ascii=False, indent=2))
            technical_citation_map = build_phase3_4_technical_citation_claim_map(
                repaired_sections,
                final_reference_keys=valid_reference_keys,
                arxiv_only_keys={
                    str(item.get("final_bib_key", "")).strip()
                    for item in verified_reference_bank
                    if str(item.get("verification_status", "")).strip().lower() == "arxiv_only"
                },
            )
            write_text(phase3_4_dir / "technical_citation_map.json", json.dumps(technical_citation_map, ensure_ascii=False, indent=2))
            write_text(phase3_4_dir / "technical_citation_contract_report.json", json.dumps(repaired_report, ensure_ascii=False, indent=2))
            return repaired_sections, technical_citation_map, repaired_report

    write_text(phase3_4_dir / "technical_citation_raw_response.txt", raw_response_text)
    write_text(phase3_4_dir / "technical_citation_contract_report.json", json.dumps(validation_report, ensure_ascii=False, indent=2))
    raise ValueError("Phase 3.4 technical citation insertion failed: " + "; ".join(validation_report.get("errors", [])))


def write_reference_check_report_md(payload: dict[str, Any], path: Path) -> None:
    lines = ["# Reference Check Report", ""]
    lines.append(str(payload.get("summary", "")).strip() or "No summary.")
    lines.append("")
    issues = payload.get("issues", [])
    if isinstance(issues, list) and issues:
        lines.append("## Issues")
        for item in issues:
            if not isinstance(item, dict):
                continue
            severity = str(item.get("severity", "info"))
            category = str(item.get("category", "general"))
            ref_key = str(item.get("reference_key", "")).strip()
            message = str(item.get("message", "")).strip()
            suffix = f" ({ref_key})" if ref_key else ""
            lines.append(f"- [{severity}/{category}]{suffix} {message}")
        lines.append("")
    keep_keys = payload.get("keep_keys", [])
    if isinstance(keep_keys, list):
        lines.append("## Keep Keys")
        lines.append(", ".join(str(item) for item in keep_keys) or "None")
        lines.append("")
    drop_keys = payload.get("drop_keys", [])
    if isinstance(drop_keys, list):
        lines.append("## Drop Keys")
        lines.append(", ".join(str(item) for item in drop_keys) or "None")
        lines.append("")
    add_keys = payload.get("recommended_additional_keys", [])
    if isinstance(add_keys, list):
        lines.append("## Recommended Additional Keys")
        lines.append(", ".join(str(item) for item in add_keys) or "None")
        lines.append("")
    write_text(path, "\n".join(lines).strip() + "\n")


def call_llm_phase3_4_introduction_writer(
    *,
    run_dir: Path,
    topic: str,
    system_model_md: str,
    problem_formulation_md: str,
    reformulation_path_md: str,
    algorithm_md: str,
    benchmark_definition_md: str,
    convergence_or_complexity_md: str,
    model_profile: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, str]], Path | None]:
    phase3_4_dir = run_dir / "phase3-4"
    phase3_3_dir = run_dir / "phase3-3"
    phase3_2_dir = run_dir / "phase3-2"
    phase25_dir = run_dir / "phase2-5"
    summary_payload = read_json(run_dir / "phase2_summary.json") or {}
    handoff_manifest = read_json(run_dir / "phase1_handoff_manifest.json") or {}
    handoff_dir = Path(handoff_manifest.get("handoff_dir")) if handoff_manifest.get("handoff_dir") else None
    phase1_run_path = resolve_phase1_run_path(str(handoff_manifest.get("phase1_run")) if handoff_manifest.get("phase1_run") else None)
    if phase1_run_path is None:
        phase1_run_raw = summary_payload.get("phase1_run")
        phase1_run_path = resolve_phase1_run_path(str(phase1_run_raw)) if phase1_run_raw else None

    synthesis_md = read_text(handoff_dir / "synthesis.md") if handoff_dir else ""
    if not synthesis_md and phase1_run_path is not None:
        synthesis_md = read_text(phase1_run_path / "phase3-2" / "synthesis.md")
    hypotheses_md = read_text(handoff_dir / "hypotheses.md") if handoff_dir else ""
    if not hypotheses_md and phase1_run_path is not None:
        hypotheses_md = read_text(phase1_run_path / "phase3-3" / "hypotheses.md")
    topic_score = read_json(handoff_dir / "topic_score.json") if handoff_dir else {}
    if not topic_score and phase1_run_path is not None:
        topic_score = read_json(phase1_run_path / "phase3-3" / "topic_score.json")
    topic_score = topic_score or {}

    phase25_summary = read_json(phase25_dir / "phase25_experiment_summary.json") or {}
    experiment_plan = read_json(phase25_dir / "experiment_plan.json") or {}
    method_naming_summary_json = read_text(phase25_dir / "method_naming_summary.json")
    experiment_plan_json = read_text(phase25_dir / "experiment_plan.json")
    method_naming_payload = _phase3_4_load_method_naming(method_naming_summary_json, experiment_plan_json)
    phase3_2_manifest = read_json(phase3_2_dir / "phase3_2_manifest.json") or {}
    phase3_3_manifest = read_json(phase3_3_dir / "phase3_3_manifest.json") or {}

    reference_source_path: Path | None = None
    references_bib_text = ""
    for candidate_path in [
        handoff_dir / "topic_focused_references.bib" if handoff_dir else None,
        handoff_dir / "references.bib" if handoff_dir else None,
        phase1_run_path / "phase2-4" / "references.bib" if phase1_run_path is not None else None,
    ]:
        if candidate_path is None:
            continue
        candidate_text = read_text(candidate_path)
        if candidate_text.strip():
            references_bib_text = candidate_text
            reference_source_path = candidate_path
            break
    references_bib_text = _merge_phase3_4_bibtex_blocks(
        references_bib_text,
        _phase3_4_seminal_matches_to_bibtex(handoff_dir / "topic_focused_literature.json" if handoff_dir else None),
    )
    bib_entries = parse_bib_entries(references_bib_text)
    current_reference_context = "\n".join(
        [
            topic,
            system_model_md,
            problem_formulation_md,
            reformulation_path_md,
            algorithm_md,
            benchmark_definition_md,
        ]
    )
    focus_keys = extract_phase1_focus_keys(current_reference_context)
    topic_selected_references = select_reference_pool(
        bib_entries,
        context_text=current_reference_context,
        focus_keys=focus_keys,
        max_items=32,
    )
    supplemental_references = phase3_4_supplemental_reference_candidates(
        current_reference_context,
        min_items=12,
    )
    candidate_references = merge_phase3_4_candidate_references(
        topic_selected_references,
        supplemental_references,
    )[:48]

    introduction_facts = build_phase3_4_introduction_facts(
        topic=topic,
        synthesis_md=synthesis_md,
        hypotheses_md=hypotheses_md,
        topic_score=topic_score if isinstance(topic_score, dict) else {},
        system_model_md=system_model_md,
        problem_formulation_md=problem_formulation_md,
        reformulation_path_md=reformulation_path_md,
        algorithm_md=algorithm_md,
        benchmark_definition_md=benchmark_definition_md,
        phase25_summary=phase25_summary if isinstance(phase25_summary, dict) else {},
        experiment_plan=experiment_plan if isinstance(experiment_plan, dict) else {},
        method_naming_payload=method_naming_payload,
        phase3_2_manifest=phase3_2_manifest if isinstance(phase3_2_manifest, dict) else {},
        phase3_3_manifest=phase3_3_manifest if isinstance(phase3_3_manifest, dict) else {},
        candidate_references=candidate_references,
    )
    current_paper_brief = build_phase3_4_current_paper_brief(
        topic=topic,
        system_model_md=system_model_md,
        problem_formulation_md=problem_formulation_md,
        reformulation_path_md=reformulation_path_md,
        algorithm_md=algorithm_md,
        benchmark_definition_md=benchmark_definition_md,
        phase25_summary=phase25_summary if isinstance(phase25_summary, dict) else {},
        experiment_plan=experiment_plan if isinstance(experiment_plan, dict) else {},
        method_naming_payload=method_naming_payload,
    )

    verified_reference_bank, verify_needed, replacement_notes, network_used = build_verified_reference_bank(
        candidate_references,
        synthesis_md=current_reference_context,
        algorithm_md=algorithm_md,
        benchmark_definition_md=benchmark_definition_md,
    )

    source_map = build_phase3_4_source_map(
        phase1_files=[
            path
            for path in [
                phase1_run_path / "phase3-2" / "synthesis.md" if phase1_run_path else None,
                phase1_run_path / "phase3-3" / "hypotheses.md" if phase1_run_path else None,
                phase1_run_path / "phase3-3" / "topic_score.json" if phase1_run_path else None,
                phase1_run_path / "phase2-4" / "references.bib" if phase1_run_path else None,
                handoff_dir / "topic_focused_literature.json" if handoff_dir else None,
                handoff_dir / "topic_focused_literature.md" if handoff_dir else None,
                reference_source_path,
            ]
            if path is not None
        ],
        phase2_technical_files=[
            run_dir / "phase2-1" / "system_model.md",
            run_dir / "phase2-1" / "problem_formulation.md",
            run_dir / "phase2-2" / "reformulation_path.md",
            run_dir / "phase2-3" / "algorithm.md",
            run_dir / "phase2-4" / "benchmark_plan.md",
            run_dir / "phase2-4" / "validation_plan.yaml",
            run_dir / "phase2-3" / "convergence_or_complexity.md",
        ],
        phase2_results_files=[
            run_dir / "phase2-5" / "phase25_experiment_summary.json",
            run_dir / "phase2-5" / "experiment_plan.json",
            run_dir / "phase2-5" / "method_naming_summary.json",
            run_dir / "phase3-2" / "phase3_2_manifest.json",
            run_dir / "phase3-3" / "phase3_3_manifest.json",
            run_dir / "phase3-3" / "abstract.tex",
            run_dir / "phase3-3" / "conclusion.tex",
        ],
    )
    source_map["reference_candidate_policy"] = {
        "phase1_topic_selected_reference_count": len(topic_selected_references),
        "supplemental_background_reference_count": len(supplemental_references),
        "merged_candidate_reference_count": len(candidate_references),
        "supplemental_background_reference_keys": [
            str(item.get("key", "")).strip()
            for item in supplemental_references
            if str(item.get("key", "")).strip()
        ],
        "policy": (
            "Phase1/topic-focused references are prioritized. DOI-backed supplemental wireless references "
            "may be added only as motivation/background support and must pass the same final metadata contract."
        ),
    }

    usable_reference_bank = [
        item
        for item in verified_reference_bank
        if isinstance(item, dict) and phase3_4_reference_is_final_usable(item)
    ]
    reference_quality_report = build_reference_quality_report(final_references=usable_reference_bank, citation_claim_map=[])
    paper_facts = build_phase3_4_paper_facts(
        topic=topic,
        current_paper_brief=current_paper_brief,
        introduction_facts=introduction_facts,
        verified_reference_bank=verified_reference_bank,
    )
    reference_strategy = build_phase3_4_reference_strategy(
        verified_reference_bank=verified_reference_bank,
        introduction_facts=introduction_facts,
    )
    preflight_reference_count_report = {
        "ok": _phase3_4_valid_reference_count(verified_reference_bank) >= int(reference_strategy.get("minimum_reference_target", 12)),
        "minimum_reference_target": int(reference_strategy.get("minimum_reference_target", 12)),
        "available_valid_references": _phase3_4_valid_reference_count(verified_reference_bank),
        "candidate_reference_count": len(candidate_references),
        "phase1_topic_selected_reference_count": len(topic_selected_references),
        "supplemental_background_reference_count": len(supplemental_references),
        "message": "Reference bank meets the hard target before writing." if _phase3_4_valid_reference_count(verified_reference_bank) >= int(reference_strategy.get("minimum_reference_target", 12)) else "Reference bank is below the hard target before writing.",
    }
    write_text(phase3_4_dir / "phase3_4_reference_preflight_report.json", json.dumps(preflight_reference_count_report, ensure_ascii=False, indent=2))
    if not preflight_reference_count_report["ok"]:
        raise ValueError(
            "Phase 3.4 reference preflight failed: "
            f"{preflight_reference_count_report['available_valid_references']} valid references < "
            f"hard target {preflight_reference_count_report['minimum_reference_target']}. "
            "Do not write the paper; repair LiteratureAgent/reference retrieval first."
        )
    writing_agent_request_json = build_role_agent_request_json(
        run_dir,
        "writing_agent",
        event="phase3_4_prompt",
        max_chars=7000,
    )
    literature_agent_request_json = build_role_agent_request_json(
        run_dir,
        "literature_agent",
        event="phase3_4_prompt",
        max_chars=7000,
    )
    write_text(phase3_4_dir / "writing_agent_request_excerpt.json", writing_agent_request_json)
    write_text(phase3_4_dir / "literature_agent_request_excerpt.json", literature_agent_request_json)

    prompt = build_phase3_4_introduction_prompt(
        source_map_json=compact_text(json.dumps(source_map, ensure_ascii=False, indent=2), 5000),
        current_paper_brief_json=compact_text(json.dumps(current_paper_brief, ensure_ascii=False, indent=2), 8000),
        introduction_facts_json=compact_text(json.dumps(introduction_facts, ensure_ascii=False, indent=2), 12000),
        verified_reference_bank_json=compact_text(json.dumps(verified_reference_bank, ensure_ascii=False, indent=2), 12000),
        reference_quality_report_json=compact_text(json.dumps(reference_quality_report, ensure_ascii=False, indent=2), 3000),
        reference_strategy_json=compact_text(json.dumps(reference_strategy, ensure_ascii=False, indent=2), 5000),
        writing_agent_request_json=writing_agent_request_json,
        literature_agent_request_json=literature_agent_request_json,
    )
    write_text(phase3_4_dir / "phase3_4_prompt.txt", prompt)

    reuse_phase3_4_response = str(os.environ.get("WARA_REUSE_PHASE3_4_RESPONSE", "")).strip().lower() in {"1", "true", "yes", "on"}
    raw_response_text = ""
    payload: Any = {}
    if reuse_phase3_4_response:
        for cached_response_path in [
            phase3_4_dir / "phase3_4_raw_response.txt",
            phase3_4_dir / "phase3_4_raw_response.txt",
            phase3_4_dir / "phase3_4_raw_response_initial.txt",
            phase3_4_dir / "phase3_4_raw_response_initial.txt",
        ]:
            cached_text = read_text(cached_response_path).strip()
            if not cached_text:
                continue
            cached_payload = _safe_json_loads(cached_text, {})
            if isinstance(cached_payload, dict):
                raw_response_text = cached_text
                payload = cached_payload
                write_text(phase3_4_dir / "phase3_4_reused_response_source.txt", str(cached_response_path))
                break
    llm = create_llm_client(model_profile)
    thinking = {"type": "enabled"} if model_profile == "kimi-k2.6-thinking" else None
    if not isinstance(payload, dict) or not payload:
        response = llm.chat(
            [{"role": "user", "content": prompt}],
            json_mode=True,
            thinking=thinking,
            max_tokens=12000,
        )
        write_text(phase3_4_dir / "phase3_4_raw_response_initial.txt", response.content)
        write_text(phase3_4_dir / "phase3_4_raw_response_initial.txt", response.content)
        payload = _safe_json_loads(response.content, {})
        raw_response_text = response.content
    required_payload_keys = {"introduction_tex", "selected_reference_keys", "citation_claim_map"}
    for repair_round in range(1, 3):
        if isinstance(payload, dict) and required_payload_keys.issubset(set(payload)):
            break
        repair_prompt = (
            "The previous Introduction/Reference response was not a valid structured object for the required schema.\n"
            "Repair only the serialization. Do not change the paper scope, equations, references, or claims unless needed to make valid JSON.\n"
            "Return valid JSON only with exactly these keys: introduction_tex, selected_reference_keys, citation_claim_map.\n"
            "selected_reference_keys must be a JSON array of BibTeX keys. citation_claim_map must be an array of objects.\n\n"
            "Original task prompt:\n"
            + compact_text(prompt, 12000)
            + "\n\nPrevious invalid response:\n"
            + compact_text(raw_response_text, 12000)
        )
        repair_response = llm.chat(
            [{"role": "user", "content": repair_prompt}],
            json_mode=True,
            thinking=thinking,
            max_tokens=12000,
        )
        raw_response_text = repair_response.content
        write_text(phase3_4_dir / f"phase3_4_structured_repair_raw_response_round{repair_round}.txt", raw_response_text)
        payload = _safe_json_loads(raw_response_text, {})
    if not isinstance(payload, dict) or not required_payload_keys.issubset(set(payload)):
        write_text(
            phase3_4_dir / "phase3_4_structured_output_error.json",
            json.dumps(
                {
                    "error": "introduction_reference_phase did not return a valid structured object",
                    "payload_type": type(payload).__name__,
                    "required_keys": sorted(required_payload_keys),
                    "present_keys": sorted(payload.keys()) if isinstance(payload, dict) else [],
                },
                ensure_ascii=False,
                indent=2,
            ),
        )
        raise ValueError("introduction_reference_phase did not return a valid structured object")

    minimum_reference_target = int(reference_strategy.get("minimum_reference_target", 12))
    maximum_arxiv_only_target = int(reference_strategy.get("maximum_arxiv_only_target", 3))
    verified_keys = {
        str(item.get("final_bib_key", "")).strip()
        for item in verified_reference_bank
        if phase3_4_reference_is_final_usable(item)
    }
    recommended_keys = [str(item).strip() for item in reference_strategy.get("recommended_reference_keys", []) if str(item).strip()]
    arxiv_only_keys = {
        str(item.get("final_bib_key", "")).strip()
        for item in verified_reference_bank
        if phase3_4_reference_is_final_usable(item)
        and str(item.get("verification_status", "")).strip().lower() == "arxiv_only"
    }
    augmentation_rounds = 0 if reuse_phase3_4_response else 2
    for _ in range(augmentation_rounds):
        cited_keys_now = extract_citation_keys_from_tex(str(payload.get("introduction_tex", "")))
        unsupported_now = [key for key in cited_keys_now if key not in verified_keys]
        cited_arxiv_only = [key for key in cited_keys_now if key in arxiv_only_keys]
        if len(cited_keys_now) >= minimum_reference_target and not unsupported_now and len(cited_arxiv_only) <= maximum_arxiv_only_target:
            break
        supplemental_keys = [key for key in recommended_keys if key not in cited_keys_now][: max(0, minimum_reference_target - len(cited_keys_now)) + 4]
        retry_prompt = (
            prompt
            + "\n\nCitation augmentation requirement:\n"
            + f"- The current draft cites only {len(cited_keys_now)} distinct references, but the minimum target is {minimum_reference_target}.\n"
            + "- Revise the Introduction so that the motivation and related-work paragraphs use more verified references while preserving the same technical scope and IEEE WCL tone.\n"
            + "- Use additional peer-reviewed references from the supplemental key list below where relevant, especially in the motivation and related-work paragraphs.\n"
            + "- Keep the contribution paragraph tied to technical_source only.\n"
            + "- Do not add unsupported keys.\n"
            + (f"- Replace unsupported citation keys: {unsupported_now}.\n" if unsupported_now else "")
            + (f"- Reduce arXiv-only citations when possible. Current arXiv-only keys in text: {cited_arxiv_only}. Target at most {maximum_arxiv_only_target}.\n" if len(cited_arxiv_only) > maximum_arxiv_only_target else "")
            + (f"- Supplemental verified keys to integrate when relevant: {supplemental_keys}.\n" if supplemental_keys else "")
            + "- Return the same JSON schema only.\n\n"
            + "Previous draft JSON:\n"
            + json.dumps(payload, ensure_ascii=False, indent=2)
        )
        retry_response = llm.chat(
            [{"role": "user", "content": retry_prompt}],
            json_mode=True,
            thinking=thinking,
            max_tokens=12000,
        )
        retry_payload = _safe_json_loads(retry_response.content, {})
        if isinstance(retry_payload, dict):
            payload = retry_payload
            raw_response_text = retry_response.content

    write_text(phase3_4_dir / "phase3_4_raw_response.txt", raw_response_text)
    write_text(phase3_4_dir / "phase3_4_raw_response.txt", raw_response_text)

    reference_verification_needed_path = phase3_4_dir / "reference_verification_needed.json"
    if verify_needed:
        write_text(reference_verification_needed_path, json.dumps(verify_needed, ensure_ascii=False, indent=2))

    return (
        payload,
        paper_facts,
        source_map,
        verified_reference_bank,
        reference_quality_report,
        verify_needed,
        replacement_notes,
        candidate_references,
        phase1_run_path,
    )


def call_llm_phase3_4_reference_check(
    *,
    model_profile: str,
    phase3_4_dir: Path,
    introduction_facts: dict[str, Any],
    verified_reference_bank: list[dict[str, Any]],
    selected_reference_keys: list[str],
    citation_claim_map: list[dict[str, Any]],
    reference_quality_report: dict[str, Any],
) -> dict[str, Any]:
    prompt = build_phase3_4_reference_check_prompt(
        introduction_facts_json=compact_text(json.dumps(introduction_facts, ensure_ascii=False, indent=2), 12000),
        verified_reference_bank_json=compact_text(json.dumps(verified_reference_bank, ensure_ascii=False, indent=2), 12000),
        selected_reference_keys_json=json.dumps(selected_reference_keys, ensure_ascii=False, indent=2),
        citation_claim_map_json=compact_text(json.dumps(citation_claim_map, ensure_ascii=False, indent=2), 9000),
        reference_quality_report_json=json.dumps(reference_quality_report, ensure_ascii=False, indent=2),
    )
    write_text(phase3_4_dir / "reference_check_prompt.txt", prompt)
    if str(os.environ.get("WARA_SKIP_REFERENCE_CHECK_LLM", "")).strip().lower() in {"1", "true", "yes", "on"}:
        payload = {
            "summary": "Reference-check LLM call skipped by bounded release configuration; local reference-bank and citation-contract checks remain active.",
            "issues": [],
            "keep_keys": selected_reference_keys,
            "drop_keys": [],
            "recommended_additional_keys": [],
            "bounded_check_fallback": True,
        }
        write_text(phase3_4_dir / "reference_check_raw_response.txt", json.dumps(payload, ensure_ascii=False, indent=2))
        write_text(phase3_4_dir / "reference_check_report.json", json.dumps(payload, ensure_ascii=False, indent=2))
        write_reference_check_report_md(payload, phase3_4_dir / "reference_check_report.md")
        return payload
    llm = create_llm_client(model_profile)
    try:
        llm.config.max_retries = min(int(getattr(llm.config, "max_retries", 1) or 1), 1)
    except Exception:
        pass
    thinking = {"type": "enabled"} if model_profile == "kimi-k2.6-thinking" else None
    try:
        response = llm.chat(
            [{"role": "user", "content": prompt}],
            json_mode=True,
            thinking=thinking,
            max_tokens=4000,
        )
    except Exception as exc:  # noqa: BLE001 - reference bank checks still run locally below.
        payload = {
            "summary": "Reference-check LLM call did not complete within the bounded repair budget.",
            "issues": [{"severity": "warning", "message": f"{type(exc).__name__}: {exc}"}],
            "keep_keys": selected_reference_keys,
            "drop_keys": [],
            "recommended_additional_keys": [],
            "bounded_check_fallback": True,
        }
        write_text(phase3_4_dir / "reference_check_raw_response.txt", json.dumps(payload, ensure_ascii=False, indent=2))
        write_text(phase3_4_dir / "reference_check_report.json", json.dumps(payload, ensure_ascii=False, indent=2))
        write_reference_check_report_md(payload, phase3_4_dir / "reference_check_report.md")
        return payload
    payload = _safe_json_loads(response.content, {})
    if not isinstance(payload, dict):
        payload = {"summary": "", "issues": [], "keep_keys": [], "drop_keys": [], "recommended_additional_keys": []}
    write_text(phase3_4_dir / "reference_check_raw_response.txt", response.content)
    write_text(phase3_4_dir / "reference_check_report.json", json.dumps(payload, ensure_ascii=False, indent=2))
    write_reference_check_report_md(payload, phase3_4_dir / "reference_check_report.md")
    return payload


def _render_final_references_bib(entries: list[dict[str, Any]]) -> str:
    rendered = [
        sanitize_bibtex_text(_build_bibtex_from_reference(item).strip())
        for item in entries
        if phase3_4_reference_is_final_usable(item)
    ]
    rendered = [item for item in rendered if item]
    return "\n\n".join(rendered) + ("\n" if rendered else "")


def _phase3_4_reference_is_citable_verified_bank_entry(item: dict[str, Any]) -> bool:
    """Allow cited verified-bank entries to remain in final BibTeX.

    The stricter `phase3_4_reference_is_final_usable` is useful when selecting
    references, but final packaging must not drop a key that the current paper
    already cites and that Phase 3.4 marked as verified/included. Dropping it
    creates undefined citations, which is worse than carrying an otherwise
    peer-reviewed entry with incomplete optional metadata.
    """
    valid_statuses = {"verified_published", "replaced_by_published_version", "arxiv_only"}
    if not isinstance(item, dict) or not item.get("included_in_final_bib"):
        return False
    if str(item.get("verification_status", "")).strip().lower() not in valid_statuses:
        return False
    return bool(str(item.get("final_bib_key", "")).strip())


def build_curated_bibliography(
    verified_reference_bank: list[dict[str, Any]],
    selected_keys: list[str],
) -> tuple[str, list[str], list[dict[str, Any]]]:
    strict_by_key = {
        str(item.get("final_bib_key", "")): item
        for item in verified_reference_bank
        if phase3_4_reference_is_final_usable(item)
    }
    citable_by_key = {
        str(item.get("final_bib_key", "")): item
        for item in verified_reference_bank
        if _phase3_4_reference_is_citable_verified_bank_entry(item)
    }
    missing: list[str] = []
    selected_entries: list[dict[str, Any]] = []
    for key in selected_keys:
        entry = strict_by_key.get(key) or citable_by_key.get(key)
        if entry is None:
            missing.append(key)
            continue
        chosen = dict(entry)
        chosen["included_in_final_bib"] = True
        selected_entries.append(chosen)
    selected_entries, _dedupe_notes = dedupe_phase3_4_references_by_identity(selected_entries)
    rendered = [
        sanitize_bibtex_text(_build_bibtex_from_reference(item).strip())
        for item in selected_entries
        if _phase3_4_reference_is_citable_verified_bank_entry(item)
    ]
    rendered = [item for item in rendered if item]
    return "\n\n".join(rendered) + ("\n" if rendered else ""), missing, selected_entries


def render_phase3_4_preview_pdf(phase_dir: Path, title: str, curated_bib_text: str) -> dict[str, str]:
    build_dir = phase_dir.parent / "_phase3_4_preview_build"
    if build_dir.exists():
        shutil.rmtree(build_dir, ignore_errors=True)
    build_dir.mkdir(parents=True, exist_ok=True)
    wrapper_tex = build_dir / "full_paper_preview.tex"
    paper_title = resolve_paper_title(phase_dir, title)
    safe_title = paper_title.replace("\\", " ").replace("{", "(").replace("}", ")")
    proposed_section_title = _impl().load_phase3_section_title(phase_dir.parent)
    write_text(phase_dir / "references.bib", curated_bib_text)
    write_text(phase_dir / "references_curated.bib", curated_bib_text)
    write_text(phase_dir / "references_ieee.bib", curated_bib_text)
    _prepare_full_paper_preview_inputs(phase_dir, build_dir)
    write_text(build_dir / "references.bib", curated_bib_text)
    wrapper_tex_content = f"""\\documentclass[journal]{{IEEEtran}}
\\usepackage{{amsmath,amssymb,amsfonts,bm,mathtools}}
\\usepackage{{graphicx}}
\\usepackage{{booktabs}}
\\usepackage{{algorithm}}
\\usepackage{{algpseudocode}}
\\usepackage[hidelinks]{{hyperref}}

\\begin{{document}}
\\title{{{safe_title}}}
\\author{{WARA CUHKSZ}}
\\maketitle

\\input{{abstract.tex}}
\\input{{introduction.tex}}
\\input{{conceptual_diagram.tex}}

\\section{{System Model and Problem Formulation}}\\label{{sec:system_model}}
\\input{{system_model_problem_formulation_section.tex}}

\\section{{{proposed_section_title}}}\\label{{sec:proposed_solution}}
\\input{{proposed_solution_section.tex}}

\\input{{numerical_results_section.tex}}

\\input{{conclusion.tex}}

\\bibliographystyle{{IEEEtran}}
\\bibliography{{references}}

\\end{{document}}
""".strip()
    write_text(wrapper_tex, wrapper_tex_content + "\n")

    citation_source = "".join(
        read_text(phase_dir / name)
        for name in [
            "introduction.tex",
            "system_model_problem_formulation_section.tex",
            "proposed_solution_section.tex",
            "numerical_results_section.tex",
            "conclusion.tex",
        ]
    )
    commands = [["pdflatex", "-interaction=nonstopmode", wrapper_tex.name]]
    if "\\cite{" in citation_source and curated_bib_text.strip():
        commands.append(["bibtex", wrapper_tex.stem])
    commands.extend(
        [
            ["pdflatex", "-interaction=nonstopmode", wrapper_tex.name],
            ["pdflatex", "-interaction=nonstopmode", wrapper_tex.name],
            ["pdflatex", "-interaction=nonstopmode", wrapper_tex.name],
        ]
    )
    last_result = None
    for cmd in commands:
        last_result = subprocess.run(
            cmd,
            cwd=build_dir,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
        if last_result.returncode != 0:
            stderr_text = f"\nCMD: {' '.join(cmd)}\nSTDOUT:\n{last_result.stdout}\nSTDERR:\n{last_result.stderr}"
            raise RuntimeError(f"Phase 3.4 full paper preview compilation failed.{stderr_text}")

    built_pdf_path = build_dir / "full_paper_preview.pdf"
    built_log_path = build_dir / "full_paper_preview.log"
    if not built_pdf_path.exists():
        stderr_text = ""
        if last_result is not None:
            stderr_text = f"\nSTDOUT:\n{last_result.stdout}\nSTDERR:\n{last_result.stderr}"
        raise RuntimeError(f"Phase 3.4 full paper preview PDF was not generated.{stderr_text}")
    log_text = read_text(built_log_path)
    problem_markers = [
        "There were undefined references.",
        "undefined citations",
        "undefined citation",
        "Citation `",
    ]
    if any(marker in log_text for marker in problem_markers):
        raise RuntimeError(
            "Phase 3.4 full paper preview compiled, but unresolved citations/references remain after the final rerun."
        )
    target_pdf = phase_dir / "full_paper_preview.pdf"
    target_log = phase_dir / "full_paper_preview.log"
    latest_pdf_path = phase_dir / "full_paper_preview_latest.pdf"
    latest_log_path = phase_dir / "full_paper_preview_latest.log"
    try:
        shutil.copyfile(built_pdf_path, target_pdf)
        pdf_path = target_pdf
    except PermissionError:
        shutil.copyfile(built_pdf_path, latest_pdf_path)
        pdf_path = latest_pdf_path
    try:
        shutil.copyfile(built_log_path, target_log)
        log_path = target_log
    except PermissionError:
        shutil.copyfile(built_log_path, latest_log_path)
        log_path = latest_log_path
    shutil.copyfile(wrapper_tex, phase_dir / "full_paper_preview.tex")
    shutil.copyfile(build_dir / "conceptual_diagram.tex", phase_dir / "conceptual_diagram.tex")
    return {
        "preview_tex": str(phase_dir / "full_paper_preview.tex"),
        "preview_pdf": str(pdf_path),
        "preview_log": str(log_path),
        "preview_documentclass": "IEEEtran",
    }


def analyze_phase3_4_forbidden_terms(text: str) -> dict[str, Any]:
    forbidden = [
        "pipeline",
        "phase 1",
        "phase 2",
        "literature source",
        "current technical artifacts",
        "phase 3.3",
        "llm",
        "codex",
        "generated_plugin",
        "draft",
        "preliminary",
        "proves",
        "guarantees",
        "statistically significant",
    ]
    lower = text.lower()
    hits = [
        term
        for term in forbidden
        if re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", lower)
    ]
    return {"ok": not hits, "hits": hits}


def analyze_phase3_4_introduction_structure(text: str) -> dict[str, Any]:
    body = re.sub(r"\\section\*?\{Introduction\}", " ", text, flags=re.I).strip()
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", body) if part.strip()]
    citations = extract_citation_keys_from_tex(text)
    has_itemize = "\\begin{itemize}" in text
    return {
        "has_section": bool(re.search(r"\\section\*?\{Introduction\}", text, flags=re.I)),
        "paragraph_count": len(paragraphs),
        "word_count": len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", body)),
        "citation_count": len(citations),
        "citation_keys": citations,
        "has_contribution_bullets": has_itemize,
        "content_quality": analyze_phase3_4_introduction_content_quality(text),
    }


def build_citation_claim_map(
    *,
    introduction_tex: str,
    llm_claim_map: list[dict[str, Any]],
    final_reference_keys: set[str],
    arxiv_only_keys: set[str],
) -> list[dict[str, Any]]:
    sentences = re.split(r"(?<=[.!?])\s+", introduction_tex.replace("\n", " "))
    sentence_lookup: list[tuple[str, list[str]]] = []
    for idx, sentence in enumerate(sentences, start=1):
        keys = extract_citation_keys_from_tex(sentence)
        if keys:
            sentence_lookup.append((f"s{idx}", keys))
    output: list[dict[str, Any]] = []
    for idx, (sentence_id, keys) in enumerate(sentence_lookup):
        if idx < len(llm_claim_map) and isinstance(llm_claim_map[idx], dict):
            claim_text = str(llm_claim_map[idx].get("claim_text", "")).strip() or sentences[idx].strip()
            claim_type = str(llm_claim_map[idx].get("claim_type", "related_work")).strip() or "related_work"
            mapped_keys = [str(item).strip() for item in llm_claim_map[idx].get("citation_keys", keys) if str(item).strip()]
            if set(mapped_keys) != set(keys):
                mapped_keys = keys
        else:
            claim_text = sentences[idx].strip()
            claim_type = "related_work"
            mapped_keys = keys
        if any(key not in final_reference_keys for key in mapped_keys):
            support_status = "unsupported"
        elif any(key in arxiv_only_keys for key in mapped_keys):
            support_status = "weak_support"
        else:
            support_status = "supported"
        output.append(
            {
                "sentence_id": sentence_id,
                "claim_text": claim_text,
                "citation_keys": mapped_keys,
                "claim_type": claim_type,
                "support_status": support_status,
                "notes": "" if support_status == "supported" else ("Uses arXiv-only support." if support_status == "weak_support" else "Contains citation keys absent from final verified references."),
            }
        )
    return output


def write_references_to_verify_md(items: list[dict[str, Any]], path: Path) -> None:
    lines = ["# References To Verify", ""]
    if not items:
        lines.append("None.")
    for item in items:
        lines.append(f"## {item.get('candidate_key', '')}")
        lines.append(f"- title: {item.get('candidate_title', '')}")
        lines.append(f"- reason: {item.get('reason', '')}")
        queries = item.get("possible_search_queries", [])
        if queries:
            lines.append(f"- possible search queries: {', '.join(queries)}")
        lines.append("")
    write_text(path, "\n".join(lines).strip() + "\n")


def write_source_usage_report(
    *,
    path: Path,
    source_map: dict[str, Any],
    introduction_facts: dict[str, Any],
    verified_reference_bank: list[dict[str, Any]],
    citation_claim_map: list[dict[str, Any]],
    selected_reference_keys: list[str],
) -> None:
    phase1_refs_used = [item.get("final_bib_key", "") for item in verified_reference_bank if item.get("final_bib_key", "") in selected_reference_keys]
    text = [
        "# Source Usage Report",
        "",
        "## current_paper_brief",
        f"- paper topic: {introduction_facts.get('current_paper_brief', {}).get('paper_topic', 'not specified') if isinstance(introduction_facts.get('current_paper_brief', {}), dict) else 'not specified'}",
        f"- topic axes: {', '.join(introduction_facts.get('current_paper_brief', {}).get('must_center_on', [])) if isinstance(introduction_facts.get('current_paper_brief', {}), dict) else 'not specified'}",
        "",
        "## literature_source",
        f"- background and related-work framing were inherited from: {', '.join(source_map.get('literature_source', {}).get('files', [])) or 'not specified'}",
        f"- literature categories used: {', '.join(introduction_facts.get('background', {}).get('related_work_categories', [])) or 'not specified'}",
        f"- final literature source references used: {', '.join(phase1_refs_used) or 'none'}",
        "",
        "## technical_source",
        f"- problem/method/benchmark were inherited from: {', '.join(source_map.get('technical_source', {}).get('files', [])) or 'not specified'}",
        f"- contribution claims kept: {', '.join(introduction_facts.get('technical_scope', {}).get('contribution_claims', [])) or 'not specified'}",
        "",
        "## results_source",
        f"- naming and claim-strength constraints were inherited from: {', '.join(source_map.get('results_source', {}).get('files', [])) or 'not specified'}",
        f"- citations mapped in final paper sections: {len(citation_claim_map)}",
    ]
    write_text(path, "\n".join(text) + "\n")


def normalize_phase3_4_citation_aliases(
    introduction_tex: str,
    valid_reference_keys: set[str] | None = None,
) -> str:
    valid_keys = {str(key).strip() for key in (valid_reference_keys or set()) if str(key).strip()}
    valid_by_lower = {key.lower(): key for key in valid_keys}
    alias_candidates: dict[str, list[str]] = {}

    def resolve_key(key: str) -> str:
        if key in valid_keys:
            return key
        lowered = key.lower()
        if lowered in valid_by_lower:
            return valid_by_lower[lowered]
        candidates = alias_candidates.get(lowered, [])
        for candidate in candidates:
            if candidate in valid_keys:
                return candidate
            candidate_lower = candidate.lower()
            if candidate_lower in valid_by_lower:
                return valid_by_lower[candidate_lower]
        if candidates and not valid_keys:
            return candidates[0]
        return key

    def replace_cite(match: re.Match[str]) -> str:
        keys = [key.strip() for key in match.group(1).split(",") if key.strip()]
        normalized: list[str] = []
        for key in keys:
            mapped = resolve_key(key)
            if mapped not in normalized:
                normalized.append(mapped)
        return "\\cite{" + ",".join(normalized) + "}"

    return re.sub(r"\\cite\{([^}]+)\}", replace_cite, introduction_tex)


def _has_usable_reference_bank(reference_bank: list[dict[str, Any]]) -> bool:
    for item in reference_bank:
        if not isinstance(item, dict) or not phase3_4_reference_is_final_usable(item):
            continue
        key = str(item.get("final_bib_key", "")).strip().lower()
        authors = str(item.get("authors", "")).lower()
        if key.startswith("wirelessseed") or "wireless optimization seed library" in authors:
            continue
        return True
    return False


def run_phase3_4_introduction_references_package(run_dir: Path, paper_target: str = "IEEE WCL") -> dict[str, Any]:
    run_dir = Path(run_dir)
    summary_payload = read_json(run_dir / "phase2_summary.json") or {}
    topic = str(summary_payload.get("topic", run_dir.name))
    model_profile = str(summary_payload.get("model_profile") or DEFAULT_MODEL_PROFILE)
    phase3_4_dir = run_dir / "phase3-4"
    phase3_4_dir.mkdir(parents=True, exist_ok=True)
    write_text(phase3_4_dir / "phase3_4_design_notes.md", build_phase3_4_design_notes())
    write_text(phase3_4_dir / "phase3_4_design_notes.md", build_phase3_4_design_notes())

    system_model_md = read_text(run_dir / "phase2-1" / "system_model.md")
    problem_formulation_md = read_text(run_dir / "phase2-1" / "problem_formulation.md")
    reformulation_path_md = read_text(run_dir / "phase2-2" / "reformulation_path.md")
    algorithm_md = read_text(run_dir / "phase2-3" / "algorithm.md")
    benchmark_definition_md = read_text(run_dir / "phase2-4" / "benchmark_plan.md")
    if not benchmark_definition_md.strip():
        benchmark_definition_md = read_text(run_dir / "phase2-3" / "benchmark_definition.md")
    convergence_or_complexity_md = read_text(run_dir / "phase2-3" / "convergence_or_complexity.md")

    (
        generated,
        paper_facts,
        source_map,
        verified_reference_bank,
        reference_quality_report,
        verify_needed,
        replacement_notes,
        candidate_references,
        phase1_run_path,
    ) = call_llm_phase3_4_introduction_writer(
        run_dir=run_dir,
        topic=topic,
        system_model_md=system_model_md,
        problem_formulation_md=problem_formulation_md,
        reformulation_path_md=reformulation_path_md,
        algorithm_md=algorithm_md,
        benchmark_definition_md=benchmark_definition_md,
        convergence_or_complexity_md=convergence_or_complexity_md,
        model_profile=model_profile,
    )

    if not _has_usable_reference_bank(verified_reference_bank):
        raise ValueError(
            "Phase 3.4 requires usable references from the supplied literature source; "
            "no topic-specific reference seed bank is used."
        )

    current_valid_reference_keys = {
        str(item.get("final_bib_key", "")).strip()
        for item in verified_reference_bank
        if isinstance(item, dict) and phase3_4_reference_is_final_usable(item)
    }

    def prepare_phase3_4_introduction(payload: dict[str, Any]) -> str:
        prepared = ensure_phase3_4_notation_paragraph(
            sanitize_phase3_4_introduction_tex(str(payload.get("introduction_tex") or "")),
            problem_formulation_md=problem_formulation_md,
            system_model_md=system_model_md,
        )
        prepared = ensure_phase3_4_minimum_intro_words(prepared)
        return normalize_phase3_4_citation_aliases(prepared, current_valid_reference_keys)

    introduction_tex = prepare_phase3_4_introduction(generated)
    try:
        validate_phase3_4_introduction_contract(run_dir, introduction_tex)
    except ValueError as initial_contract_error:
        phase3_4_dir = run_dir / "phase3-4"
        repair_prompt = (
            read_text(phase3_4_dir / "phase3_4_prompt.txt")
            + "\n\nIntroduction contract repair requirement:\n"
            + "- The previous Introduction failed the local writing contract below.\n"
            + "- Rewrite introduction_tex only by preserving the same four-part WCL structure, verified citations, organization paragraph, and final notation paragraph.\n"
            + "- Keep the three contribution items non-overlapping: item 1 modeling/formulation idea without constraint lists; item 2 method idea without block-update/search details; item 3 evaluation evidence may state a scoped qualitative gain under the considered settings, but must not include exact values, win rates, feasibility rates, or universal superiority claims.\n"
            + "- Before the final notation paragraph, remove every inline `$...$` math expression and describe the concept in words instead. For example, write half-wavelength fixed array rather than a wavelength-spacing math expression, and write design variables or method names rather than variable tuples.\n"
            + "- Do not add references outside verified_reference_bank.\n"
            + "- Return the same JSON schema only: introduction_tex, selected_reference_keys, citation_claim_map.\n\n"
            + "Contract failure:\n"
            + str(initial_contract_error)
            + "\n\nPrevious draft JSON:\n"
            + json.dumps(generated, ensure_ascii=False, indent=2)
        )
        llm = create_llm_client(model_profile)
        thinking = {"type": "enabled"} if model_profile == "kimi-k2.6-thinking" else None
        repair_response = llm.chat(
            [{"role": "user", "content": repair_prompt}],
            json_mode=True,
            thinking=thinking,
            max_tokens=6500,
        )
        write_text(phase3_4_dir / "phase3_4_raw_response_contract_retry.txt", repair_response.content)
        write_text(phase3_4_dir / "phase3_4_raw_response_contract_retry.txt", repair_response.content)
        repair_payload = _safe_json_loads(repair_response.content, {})
        if not isinstance(repair_payload, dict):
            raise initial_contract_error
        generated = repair_payload
        introduction_tex = prepare_phase3_4_introduction(generated)
        validate_phase3_4_introduction_contract(run_dir, introduction_tex)
        write_text(phase3_4_dir / "phase3_4_raw_response.txt", repair_response.content)
        write_text(phase3_4_dir / "phase3_4_raw_response.txt", repair_response.content)

    selected_reference_keys = [str(item).strip() for item in (generated.get("selected_reference_keys") or []) if str(item).strip()]
    selected_reference_keys = [
        normalize_phase3_4_citation_aliases(f"\\cite{{{key}}}", current_valid_reference_keys)
        .removeprefix("\\cite{")
        .removesuffix("}")
        for key in selected_reference_keys
    ]
    selected_reference_keys = [key for key in selected_reference_keys if key in current_valid_reference_keys]
    llm_claim_map = generated.get("citation_claim_map") if isinstance(generated.get("citation_claim_map"), list) else []
    result_constraints = (
        paper_facts.get("introduction_reference_phase", {}).get("result_constraints", {})
        if isinstance(paper_facts, dict)
        else {}
    )
    requested_reference_target = int(result_constraints.get("minimum_reference_target", 12) or 12)
    minimum_reference_target = requested_reference_target
    cited_keys = extract_citation_keys_from_tex(introduction_tex)
    all_reference_keys: list[str] = []
    for key in cited_keys:
        if key not in all_reference_keys:
            all_reference_keys.append(key)
    valid_cited_reference_keys = [key for key in all_reference_keys if key in current_valid_reference_keys]
    introduction_reference_count_report = {
        "ok": bool(valid_cited_reference_keys),
        "final_paper_minimum_reference_target": minimum_reference_target,
        "available_valid_references": len(current_valid_reference_keys),
        "introduction_valid_cited_references": len(valid_cited_reference_keys),
        "introduction_valid_cited_reference_keys": valid_cited_reference_keys,
        "hard_blocking": False,
        "contract_scope": "introduction_before_technical_citation_pass",
        "message": (
            "Introduction citations are present; the hard reference-count contract is enforced after technical citations are inserted."
            if valid_cited_reference_keys
            else "Introduction has no verified citations; the final full-paper reference contract will still be checked after technical citation insertion."
        ),
    }
    write_text(phase3_4_dir / "phase3_4_introduction_reference_count_report.json", json.dumps(introduction_reference_count_report, ensure_ascii=False, indent=2))

    verified_key_bank = {
        str(item.get("final_bib_key", ""))
        for item in verified_reference_bank
        if phase3_4_reference_is_final_usable(item)
    }
    arxiv_only_keys = {
        str(item.get("final_bib_key", ""))
        for item in verified_reference_bank
        if phase3_4_reference_is_final_usable(item) and item.get("verification_status") == "arxiv_only"
    }
    introduction_citation_claim_map = build_citation_claim_map(
        introduction_tex=introduction_tex,
        llm_claim_map=llm_claim_map,
        final_reference_keys=verified_key_bank,
        arxiv_only_keys=arxiv_only_keys,
    )
    introduction_claim_mapped_keys: list[str] = []
    for item in introduction_citation_claim_map:
        for key in item.get("citation_keys", []):
            if key not in introduction_claim_mapped_keys:
                introduction_claim_mapped_keys.append(key)

    _preliminary_bib_text, preliminary_missing_reference_keys, preliminary_reference_entries = build_curated_bibliography(
        verified_reference_bank,
        introduction_claim_mapped_keys or cited_keys,
    )
    reference_quality_report = build_reference_quality_report(
        final_references=preliminary_reference_entries,
        citation_claim_map=introduction_citation_claim_map,
    )
    reference_check_payload = call_llm_phase3_4_reference_check(
        model_profile=model_profile,
        phase3_4_dir=phase3_4_dir,
        introduction_facts=paper_facts.get("introduction_reference_phase", {}),
        verified_reference_bank=verified_reference_bank,
        selected_reference_keys=introduction_claim_mapped_keys or cited_keys,
        citation_claim_map=introduction_citation_claim_map,
        reference_quality_report=reference_quality_report,
    )

    introduction_facts = paper_facts.get("introduction_reference_phase", {})
    reference_strategy = build_phase3_4_reference_strategy(
        verified_reference_bank=verified_reference_bank,
        introduction_facts=introduction_facts,
    )
    write_text(phase3_4_dir / "phase3_4_source_map.json", json.dumps(source_map, ensure_ascii=False, indent=2))
    write_text(phase3_4_dir / "introduction_facts.json", json.dumps({
        "current_paper_brief": introduction_facts.get("current_paper_brief", {}),
        "background": introduction_facts.get("background", {}),
        "technical_scope": introduction_facts.get("technical_scope", {}),
        "result_constraints": introduction_facts.get("result_constraints", {}),
    }, ensure_ascii=False, indent=2))
    write_text(phase3_4_dir / "reference_strategy.json", json.dumps(reference_strategy, ensure_ascii=False, indent=2))
    write_text(phase3_4_dir / "verified_reference_bank.json", json.dumps(verified_reference_bank, ensure_ascii=False, indent=2))
    write_text(phase3_4_dir / "introduction_citation_claim_map.json", json.dumps(introduction_citation_claim_map, ensure_ascii=False, indent=2))
    write_references_to_verify_md(verify_needed, phase3_4_dir / "references_to_verify.md")
    write_reference_replacement_report(verified_bank=verified_reference_bank, report_path=phase3_4_dir / "reference_replacement_report.md")
    write_text(phase3_4_dir / "introduction.tex", introduction_tex)

    abstract_tex = read_text(run_dir / "phase3-3" / "abstract.tex")
    conclusion_tex = read_text(run_dir / "phase3-3" / "conclusion.tex")
    phase1_snippet = sanitize_phase3_4_preview_section_tex(load_phase3_1_system_model_problem_snippet(run_dir).strip())
    phase3_snippet = sanitize_phase3_4_preview_section_tex(load_phase3_1_proposed_solution_snippet(run_dir))
    phase3_2_snippet = sanitize_phase3_4_preview_section_tex(
        sanitize_phase3_2_numerical_results_tex(read_text(run_dir / "phase3-2" / "numerical_results_section.tex"))
    )
    technical_sections, technical_citation_map, technical_citation_report = call_llm_phase3_4_technical_citation_pass(
        model_profile=model_profile,
        phase3_4_dir=phase3_4_dir,
        current_paper_brief=paper_facts.get("introduction_reference_phase", {}).get("current_paper_brief", {}),
        verified_reference_bank=verified_reference_bank,
        reference_quality_report=reference_quality_report,
        introduction_citation_claim_map=introduction_citation_claim_map,
        system_model_problem_formulation_section_tex=phase1_snippet + "\n",
        proposed_solution_section_tex=phase3_snippet,
        numerical_results_section_tex=phase3_2_snippet,
    )
    phase1_snippet = technical_sections["system_model_problem_formulation_section_tex"]
    phase3_snippet = technical_sections["proposed_solution_section_tex"]
    phase3_2_snippet = technical_sections["numerical_results_section_tex"]

    citation_claim_map = introduction_citation_claim_map + technical_citation_map
    final_cited_keys: list[str] = []
    for text in [introduction_tex, phase1_snippet, phase3_snippet, phase3_2_snippet]:
        for key in extract_citation_keys_from_tex(text):
            if key in current_valid_reference_keys and key not in final_cited_keys:
                final_cited_keys.append(key)
    final_selected_reference_keys: list[str] = []
    for key in final_cited_keys:
        if key in current_valid_reference_keys and key not in final_selected_reference_keys:
            final_selected_reference_keys.append(key)
    final_bib_text, missing_reference_keys, final_reference_entries = build_curated_bibliography(
        verified_reference_bank,
        final_selected_reference_keys or final_cited_keys,
    )
    reference_count_report = build_phase3_4_final_reference_count_contract(
        final_selected_reference_keys=final_selected_reference_keys,
        current_valid_reference_keys=current_valid_reference_keys,
        introduction_cited_reference_keys=valid_cited_reference_keys,
        technical_citation_map=technical_citation_map,
        minimum_reference_target=minimum_reference_target,
    )
    write_text(phase3_4_dir / "phase3_4_reference_count_contract_report.json", json.dumps(reference_count_report, ensure_ascii=False, indent=2))
    if not reference_count_report["ok"]:
        raise ValueError(
            "Phase 3.4 final full-paper reference contract failed: "
            + "; ".join(reference_count_report.get("errors", []))
            + " Repair the LiteratureAgent/reference bank or technical citation placement; do not lower the target."
        )
    reference_quality_report = build_reference_quality_report(
        final_references=final_reference_entries,
        citation_claim_map=citation_claim_map,
    )
    write_text(phase3_4_dir / "citation_claim_map.json", json.dumps(citation_claim_map, ensure_ascii=False, indent=2))
    write_text(phase3_4_dir / "citation_map.json", json.dumps(citation_claim_map, ensure_ascii=False, indent=2))
    write_text(phase3_4_dir / "reference_quality_report.json", json.dumps(reference_quality_report, ensure_ascii=False, indent=2))
    write_reference_quality_report_md(reference_quality_report, phase3_4_dir / "reference_quality_report.md")
    write_source_usage_report(
        path=phase3_4_dir / "source_usage_report.md",
        source_map=source_map,
        introduction_facts=paper_facts.get("introduction_reference_phase", {}),
        verified_reference_bank=verified_reference_bank,
        citation_claim_map=citation_claim_map,
        selected_reference_keys=final_selected_reference_keys,
    )
    write_text(phase3_4_dir / "references.bib", final_bib_text)
    write_text(phase3_4_dir / "references_curated.bib", final_bib_text)
    write_text(phase3_4_dir / "references_ieee.bib", final_bib_text)
    write_text(phase3_4_dir / "abstract.tex", abstract_tex)
    write_text(phase3_4_dir / "conclusion.tex", conclusion_tex)
    write_text(phase3_4_dir / "system_model_problem_formulation_section.tex", phase1_snippet)
    write_text(phase3_4_dir / "proposed_solution_section.tex", phase3_snippet)
    write_text(phase3_4_dir / "numerical_results_section.tex", phase3_2_snippet)

    preview = render_phase3_4_preview_pdf(phase3_4_dir, topic, final_bib_text)
    full_paper_abbreviation_report = analyze_phase3_4_full_paper_abbreviations_from_phase_dir(phase3_4_dir)
    write_text(
        phase3_4_dir / "full_paper_abbreviation_report.json",
        json.dumps(full_paper_abbreviation_report, ensure_ascii=False, indent=2),
    )
    introduction_word_count = len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", introduction_tex))
    technical = paper_facts.get("introduction_reference_phase", {}).get("technical_scope", {})
    proposed = technical.get("proposed_method", {}) if isinstance(technical, dict) else {}
    benchmark = technical.get("benchmark", {}) if isinstance(technical, dict) else {}
    method_names_used = {
        "proposed_display_name_long": str(proposed.get("display_name_long", "")),
        "proposed_display_name_short": str(proposed.get("display_name_short", "")),
        "benchmark_display_name_long": str(benchmark.get("display_name_long", "")),
        "benchmark_display_name_short": str(benchmark.get("display_name_short", "")),
    }
    manifest = {
        "paper_target": paper_target,
        "phase": "phase3",
        "phase_id": "phase3.4",
        "phase_name": "phase3.4_introduction_reference_phase",
        "paper_writing_mode": _paper_writing_mode_snapshot(),
        "title": topic,
        "input_files_used": source_map,
        "source_map_path": str(phase3_4_dir / "phase3_4_source_map.json"),
        "introduction_facts_path": str(phase3_4_dir / "introduction_facts.json"),
        "verified_reference_bank_path": str(phase3_4_dir / "verified_reference_bank.json"),
        "reference_strategy_path": str(phase3_4_dir / "reference_strategy.json"),
        "references_bib_path": str(phase3_4_dir / "references.bib"),
        "references_ieee_bib_path": str(phase3_4_dir / "references_ieee.bib"),
        "references_to_verify_path": str(phase3_4_dir / "references_to_verify.md"),
        "reference_replacement_report_path": str(phase3_4_dir / "reference_replacement_report.md"),
        "reference_quality_report_path": str(phase3_4_dir / "reference_quality_report.json"),
        "reference_check_report_path": str(phase3_4_dir / "reference_check_report.json"),
        "full_paper_abbreviation_report_path": str(phase3_4_dir / "full_paper_abbreviation_report.json"),
        "reference_check_prompt_path": str(phase3_4_dir / "reference_check_prompt.txt"),
        "reference_check_raw_response_path": str(phase3_4_dir / "reference_check_raw_response.txt"),
        "technical_citation_prompt_path": str(phase3_4_dir / "technical_citation_prompt.txt"),
        "technical_citation_raw_response_path": str(phase3_4_dir / "technical_citation_raw_response.txt"),
        "technical_citation_map_path": str(phase3_4_dir / "technical_citation_map.json"),
        "technical_citation_contract_report_path": str(phase3_4_dir / "technical_citation_contract_report.json"),
        "citation_claim_map_path": str(phase3_4_dir / "citation_claim_map.json"),
        "source_usage_report_path": str(phase3_4_dir / "source_usage_report.md"),
        "introduction_path": str(phase3_4_dir / "introduction.tex"),
        "preview_pdf_path": preview.get("preview_pdf", str(phase3_4_dir / "full_paper_preview.pdf")),
        "prompt_path": str(phase3_4_dir / "phase3_4_prompt.txt"),
        "raw_response_path": str(phase3_4_dir / "phase3_4_raw_response.txt"),
        "word_count_introduction": introduction_word_count,
        "method_names_used": method_names_used,
        "numbers_used": [],
        "selected_reference_keys": final_selected_reference_keys,
        "introduction_selected_reference_keys": selected_reference_keys,
        "citation_keys_in_text": final_cited_keys,
        "introduction_citation_keys": cited_keys,
        "missing_reference_keys": missing_reference_keys,
        "preliminary_missing_reference_keys_before_technical_citations": preliminary_missing_reference_keys,
        "technical_citation_report": technical_citation_report,
        "reference_target": reference_strategy,
        "peer_reviewed_count": reference_quality_report.get("peer_reviewed_count", 0),
        "arxiv_only_count": reference_quality_report.get("arxiv_only_count", 0),
        "verified_reference_bank_summary": {
            "total_candidates": len(candidate_references),
            "total_verified_bank": len(verified_reference_bank),
            "replaced_by_published_version": sum(1 for item in verified_reference_bank if item.get("verification_status") == "replaced_by_published_version"),
            "verified_published": sum(1 for item in verified_reference_bank if item.get("verification_status") == "verified_published"),
            "arxiv_only": sum(1 for item in verified_reference_bank if item.get("verification_status") == "arxiv_only"),
            "unverified": sum(1 for item in verified_reference_bank if item.get("verification_status") == "unverified"),
        },
        "forbidden_terms_check": analyze_phase3_4_forbidden_terms(introduction_tex),
        "introduction_structure_check": analyze_phase3_4_introduction_structure(introduction_tex),
        "paper_facts": paper_facts,
        "preview": preview,
        "reference_check": reference_check_payload,
        "full_paper_abbreviation_check": full_paper_abbreviation_report,
    }
    write_text(phase3_4_dir / "phase3_4_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest
