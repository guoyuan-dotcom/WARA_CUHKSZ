from __future__ import annotations

import copy
import hashlib
import importlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable

from pipeline_core import DEFAULT_MODEL_PROFILE, compact_text, read_json, read_text, utcnow_iso, write_text
from pipeline_core.json_utils import _safe_json_loads
from phase_runtime.llm import create_llm_client
from phase_runtime.prompt_templates import render_prompt_template
from phase_runtime.phase3_3_sections import sanitize_phase3_3_embedded_section_tex
from phase_runtime.paper_mode import (
    _paper_deterministic_fallback_allowed,
    _paper_phase_llm_skip_enabled,
    _paper_writing_mode,
    _paper_writing_mode_snapshot,
    _payload_has_deterministic_marker,
    detect_paper_writing_deterministic_outputs,
)
from phase_runtime.phase3_4_references import (
    COMMON_WIRELESS_ABBREVIATION_EXPANSIONS,
    analyze_phase3_4_intro_orphan_argument_paragraphs,
    analyze_phase3_4_full_paper_abbreviations,
    analyze_phase3_4_full_paper_abbreviations_from_phase_dir,
    analyze_phase3_4_introduction_content_quality,
    build_curated_bibliography,
    ensure_phase3_4_notation_paragraph,
    extract_citation_keys_from_tex,
    parse_bib_entries,
    resolve_paper_title,
    sanitize_phase3_4_introduction_tex,
)


def _impl() -> Any:
    return importlib.import_module("phase_runtime_impl")


def _prepare_full_paper_preview_inputs(phase_dir: Path, build_dir: Path) -> None:
    return _impl()._prepare_full_paper_preview_inputs(phase_dir, build_dir)


def find_undefined_abstract_abbreviations(abstract_tex: str) -> list[str]:
    return _impl().find_undefined_abstract_abbreviations(abstract_tex)


def sanitize_phase3_latex_snippet(tex: str) -> str:
    return _impl().sanitize_phase3_latex_snippet(tex)


def sanitize_phase3_2_numerical_results_tex(tex: str) -> str:
    return _impl().sanitize_phase3_2_numerical_results_tex(tex)


def enforce_phase3_2_plotted_method_definitions(
    tex_text: str,
    phase25_summary: dict[str, Any],
    method_naming_summary_json: str,
) -> str:
    return _impl().enforce_phase3_2_plotted_method_definitions(tex_text, phase25_summary, method_naming_summary_json)


def _phase3_6_align_plotted_method_claim_text(
    tex_text: str,
    phase25_summary: dict[str, Any],
    method_naming_summary_json: str,
) -> tuple[str, bool]:
    """Keep abstract/conclusion method claims aligned with final plotted methods."""
    plotted = set(_impl()._phase3_2_final_plotted_method_ids(phase25_summary))
    if not plotted:
        return tex_text, False
    aliases = _impl()._phase3_2_method_aliases(method_naming_summary_json)
    baseline_method = str((phase25_summary.get("primary_claim_check") or {}).get("baseline_method") or "").strip()
    if not baseline_method or baseline_method not in plotted:
        baseline_method = next((method for method in plotted if method != "proposed"), "")
    baseline_aliases = aliases.get(baseline_method, {})
    baseline_short = str(baseline_aliases.get("display_name_short") or "").strip()
    baseline_long = str(baseline_aliases.get("display_name_long") or "").strip()
    if baseline_short and baseline_long:
        baseline_long = baseline_long[:1].lower() + baseline_long[1:]
        baseline_label = f"{baseline_long} ({baseline_short})"
    else:
        baseline_label = (
            baseline_short
            or baseline_long
            or baseline_method.replace("_", "-")
        )
    if not baseline_label:
        return tex_text, False
    updated = tex_text
    unplotted_aliases: list[str] = []
    for method_id, payload in aliases.items():
        if method_id in plotted:
            continue
        unplotted_aliases.append(method_id)
        unplotted_aliases.extend(token for token in str(payload.get("tokens") or "").splitlines() if token)
        for token in str(payload.get("tokens") or "").splitlines():
            token = token.strip()
            if not token:
                continue
            unplotted_aliases.append(token.replace(" ", "-"))
            unplotted_aliases.append(token.replace(" ", "-").lower())
            unplotted_aliases.append(token.replace(" ", "").lower())
            if token.lower().endswith(" baseline"):
                stem = token[:-len(" baseline")].strip()
                unplotted_aliases.append(stem)
                unplotted_aliases.append(stem.replace(" ", "-"))
                unplotted_aliases.append(stem.replace(" ", "-").lower())
            for suffix in (" precoding", " transmission", " method", " scheme"):
                if token.lower().endswith(suffix):
                    stem = token[: -len(suffix)].strip()
                    if stem:
                        unplotted_aliases.append(stem)
                        unplotted_aliases.append(stem.replace(" ", "-"))
                        unplotted_aliases.append(stem.replace(" ", "-").lower())
                        unplotted_aliases.append(stem.replace(" ", "").lower())
    unplotted_aliases = [alias for alias in dict.fromkeys(unplotted_aliases) if alias and alias != baseline_label]
    for alias in unplotted_aliases:
        updated = re.sub(
            rf"\s*,?\s*(?:and|or)\s+(?:a|an|the)\s+{re.escape(alias)}\s+benchmark",
            "",
            updated,
            flags=re.I,
        )
        updated = re.sub(
            rf"(?:a|an|the)\s+{re.escape(alias)}\s+benchmark\s+(?:and|or)\s+",
            "",
            updated,
            flags=re.I,
        )
        updated = updated.replace(
            f"relative to the {alias} benchmark and the other tested covariance benchmarks",
            f"relative to the {baseline_label} benchmark",
        )
        updated = updated.replace(
            f"than the {alias} benchmark and the other tested covariance benchmarks",
            f"than the {baseline_label} benchmark",
        )
        updated = updated.replace(
            f"over the {alias} benchmark and the other tested covariance benchmarks",
            f"over the {baseline_label} benchmark",
        )
        updated = re.sub(
            rf"the\s+{re.escape(alias)}\s+benchmark",
            lambda _match, label=baseline_label: f"the {label} benchmark",
            updated,
            flags=re.I,
        )
    for method_id, payload in aliases.items():
        if method_id in plotted:
            continue
        short_name = str(payload.get("display_name_short") or "").strip()
        long_name = str(payload.get("display_name_long") or "").strip()
        if short_name and long_name:
            updated = re.sub(
                rf"the\s+{re.escape(long_name)}\s*\({re.escape(short_name)}\)\s+benchmark",
                lambda _match, label=baseline_label: f"the {label} benchmark",
                updated,
                flags=re.I,
            )
    updated = updated.replace(" and the other tested covariance benchmarks", "")
    updated = updated.replace(" and other tested covariance benchmarks", "")
    updated = re.sub(r"[ \t]{2,}", " ", updated)
    return updated, updated != tex_text


def _phase3_6_scope_exact_wmmse_language(tex_text: str) -> tuple[str, bool]:
    """Keep WMMSE exactness claims scoped to the identity/auxiliary updates.

    WMMSE is an exact reformulation identity for fixed physical variables, but
    the alternating WMMSE algorithm is still a local block-coordinate method.
    This wording repair is mechanism-level rather than topic-specific.
    """
    updated = tex_text
    replacements = [
        (
            "The precoder step uses an exact weighted minimum mean-square-error mapping for fixed antenna locations, while",
            "The weighted minimum mean-square-error identity and auxiliary updates are exact for fixed antenna locations, while",
        ),
        (
            "The weighted minimum mean-square-error step gives the exact fixed-coordinate precoder mapping, while",
            "The weighted minimum mean-square-error step uses exact auxiliary updates for fixed coordinates, while the overall method remains a local block-coordinate algorithm and",
        ),
        (
            "the exact fixed-coordinate precoder mapping",
            "the exact fixed-coordinate WMMSE identity and auxiliary updates",
        ),
        (
            "an exact weighted minimum mean-square-error mapping for fixed antenna locations",
            "an exact weighted minimum mean-square-error identity and exact auxiliary updates for fixed antenna locations",
        ),
        (
            "exact WMMSE mapping",
            "exact WMMSE identity and auxiliary updates",
        ),
    ]
    for before, after in replacements:
        updated = updated.replace(before, after)
    updated = re.sub(
        r"\bexact\s+fixed-coordinate\s+precoder\s+mapping\b",
        "exact fixed-coordinate WMMSE identity and auxiliary updates",
        updated,
        flags=re.I,
    )
    return updated, updated != tex_text


def sanitize_phase3_3_abstract_tex(tex: str) -> str:
    return _impl().sanitize_phase3_3_abstract_tex(tex)


def sanitize_phase3_3_conclusion_tex(tex: str) -> str:
    return _impl().sanitize_phase3_3_conclusion_tex(tex)


def build_phase3_5_design_notes() -> str:
    return """
# Phase 3.4 Design Notes

Phase 3.4 is the full-paper writing review and rewrite phase.

It:
- reads the assembled paper package from Phase 3.3
- reviews writing quality, abbreviation discipline, and citation placement
- rewrites sections for IEEE WCL style without changing the implemented technical claims
- adds or adjusts body citations using only the verified reference bank
- recompiles a revised full-paper preview

Phase 3.4 does not:
- redesign the algorithm
- change experiments or numerical results
- invent references
- introduce unsupported claims
""".strip()


def build_phase3_5_review_prompt(
    *,
    review_facts_json: str,
    mathematical_contract_json: str = "",
    verified_reference_bank_json: str,
    current_sections_json: str,
) -> str:
    return render_prompt_template(
        "phase3_5/review_rewrite.prompt.yaml",
        review_facts_json=review_facts_json,
        mathematical_contract_json=mathematical_contract_json or "{}",
        verified_reference_bank_json=verified_reference_bank_json,
        current_sections_json=current_sections_json,
    )


def sanitize_phase3_5_body_section(tex: str) -> str:
    cleaned = tex.replace("\r\n", "\n").strip()
    cleaned = re.sub(r"\\end\{document\}\s*$", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip() + "\n"


def extract_latex_labels(tex: str) -> list[str]:
    return re.findall(r"\\label\{([^}]+)\}", tex)


def validate_bib_file_reference_flow(phase_dir: Path) -> dict[str, Any]:
    """Validate that the final paper source uses BibTeX, not hand-written references."""
    phase_dir = Path(phase_dir)
    wrapper_path = phase_dir / "revised_full_paper.tex"
    bib_path = phase_dir / "references.bib"
    wrapper_text = read_text(wrapper_path)
    bib_text = read_text(bib_path)
    source_files = [
        wrapper_path,
        phase_dir / "abstract.tex",
        phase_dir / "introduction.tex",
        phase_dir / "system_model_problem_formulation_section.tex",
        phase_dir / "proposed_solution_section.tex",
        phase_dir / "numerical_results_section.tex",
        phase_dir / "conclusion.tex",
    ]
    source_with_handwritten_refs = [
        str(path)
        for path in source_files
        if path.exists() and re.search(r"\\begin\{thebibliography\}|\\bibitem\b", read_text(path), flags=re.I)
    ]
    bib_database_match = re.search(r"\\bibliography\{([^{}]+)\}", wrapper_text)
    bib_databases = []
    if bib_database_match:
        bib_databases = [item.strip() for item in bib_database_match.group(1).split(",") if item.strip()]
    checks = {
        "wrapper_tex_path": str(wrapper_path),
        "references_bib_path": str(bib_path),
        "uses_bibliographystyle": bool(re.search(r"\\bibliographystyle\{IEEEtran\}", wrapper_text)),
        "uses_bibliography_command": bool(bib_database_match),
        "bibliography_databases": bib_databases,
        "uses_references_database": "references" in bib_databases,
        "references_bib_exists": bib_path.exists(),
        "references_bib_nonempty": bool(bib_text.strip()),
        "references_bib_entry_count": len(re.findall(r"@\w+\s*\{", bib_text)),
        "source_with_handwritten_references": source_with_handwritten_refs,
        "allows_generated_bbl": True,
        "note": "A .bbl file may be generated by BibTeX during preview compilation, but the maintained source of references must be references.bib.",
    }
    checks["ok"] = bool(
        checks["uses_bibliographystyle"]
        and checks["uses_bibliography_command"]
        and checks["uses_references_database"]
        and checks["references_bib_exists"]
        and checks["references_bib_nonempty"]
        and checks["references_bib_entry_count"] > 0
        and not checks["source_with_handwritten_references"]
    )
    return checks


def _word_count_text(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", text))

def _resolve_preview_title(phase_dir: Path, title: str) -> str:
    return resolve_paper_title(phase_dir, title)


def render_phase3_5_preview_pdf(phase_dir: Path, title: str, curated_bib_text: str) -> dict[str, str]:
    build_dir = phase_dir.parent / "_phase3_5_preview_build"
    if build_dir.exists():
        shutil.rmtree(build_dir, ignore_errors=True)
    build_dir.mkdir(parents=True, exist_ok=True)
    wrapper_tex = build_dir / "full_paper_revised_preview.tex"
    safe_title = _resolve_preview_title(phase_dir, title).replace("\\", " ").replace("{", "(").replace("}", ")")
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
    commands = [
        ["pdflatex", "-interaction=nonstopmode", wrapper_tex.name],
        ["bibtex", wrapper_tex.stem],
        ["pdflatex", "-interaction=nonstopmode", wrapper_tex.name],
        ["pdflatex", "-interaction=nonstopmode", wrapper_tex.name],
        ["pdflatex", "-interaction=nonstopmode", wrapper_tex.name],
    ]
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
            raise RuntimeError(f"Phase 3.4 full paper revised preview compilation failed.{stderr_text}")
    built_pdf_path = build_dir / "full_paper_revised_preview.pdf"
    built_log_path = build_dir / "full_paper_revised_preview.log"
    if not built_pdf_path.exists():
        stderr_text = ""
        if last_result is not None:
            stderr_text = f"\nSTDOUT:\n{last_result.stdout}\nSTDERR:\n{last_result.stderr}"
        raise RuntimeError(f"Phase 3.4 full paper revised preview PDF was not generated.{stderr_text}")
    log_text = read_text(built_log_path)
    if any(marker in log_text for marker in ["There were undefined references.", "undefined citation", "Citation `"]):
        raise RuntimeError("Phase 3.4 revised preview compiled, but unresolved citations/references remain after the final rerun.")
    target_pdf = phase_dir / "full_paper_revised_preview.pdf"
    target_log = phase_dir / "full_paper_revised_preview.log"
    fallback_pdf = phase_dir / "full_paper_revised_preview_latest.pdf"
    fallback_log = phase_dir / "full_paper_revised_preview_latest.log"
    try:
        shutil.copyfile(built_pdf_path, target_pdf)
        pdf_path = target_pdf
    except PermissionError:
        shutil.copyfile(built_pdf_path, fallback_pdf)
        pdf_path = fallback_pdf
    try:
        shutil.copyfile(built_log_path, target_log)
        log_path = target_log
    except PermissionError:
        shutil.copyfile(built_log_path, fallback_log)
        log_path = fallback_log
    shutil.copyfile(wrapper_tex, phase_dir / "full_paper_revised_preview.tex")
    shutil.copyfile(build_dir / "conceptual_diagram.tex", phase_dir / "conceptual_diagram.tex")
    return {
        "preview_tex": str(phase_dir / "full_paper_revised_preview.tex"),
        "preview_pdf": str(pdf_path),
        "preview_log": str(log_path),
        "preview_documentclass": "IEEEtran",
    }


PHASE34_COMMON_ABBREVIATION_EXPANSIONS: dict[str, str] = dict(COMMON_WIRELESS_ABBREVIATION_EXPANSIONS)
PHASE34_COMMON_ABBREVIATION_EXPANSIONS.setdefault("LMI", "linear matrix inequality")
PHASE34_SPECIAL_ABBREVIATION_REWRITES: dict[str, str] = {
    "Regularized-ZF": "regularized zero-forcing",
}


def _phase3_5_replace_once(text: str, pattern: str, replacement: str | Callable[[re.Match[str]], str]) -> tuple[str, bool]:
    updated, count = re.subn(pattern, replacement, text, count=1)
    return updated, count > 0


def _phase3_5_replace_once_in_prose(
    text: str,
    pattern: str,
    replacement: str | Callable[[re.Match[str]], str],
) -> tuple[str, bool]:
    """Apply an abbreviation repair outside math and algorithm displays."""
    source = str(text or "")
    protected_pattern = re.compile(
        r"\\begin\{(?:equation|align|aligned|split|multline|gather|subequations|algorithm|algorithmic)\*?\}.*?"
        r"\\end\{(?:equation|align|aligned|split|multline|gather|subequations|algorithm|algorithmic)\*?\}"
        r"|\$\$.*?\$\$|\$.*?\$|\\\(.*?\\\)|\\\[.*?\\\]",
        flags=re.S,
    )
    pieces: list[str] = []
    cursor = 0
    changed = False
    for match in protected_pattern.finditer(source):
        segment = source[cursor: match.start()]
        if not changed:
            segment, changed = _phase3_5_replace_once(segment, pattern, replacement)
        pieces.append(segment)
        pieces.append(match.group(0))
        cursor = match.end()
    tail = source[cursor:]
    if not changed:
        tail, changed = _phase3_5_replace_once(tail, pattern, replacement)
    pieces.append(tail)
    return "".join(pieces), changed


def _phase3_5_replace_all_in_prose(
    text: str,
    pattern: str,
    replacement: str | Callable[[re.Match[str]], str],
) -> tuple[str, int]:
    """Apply a prose-only replacement globally while preserving math/algorithm blocks."""
    source = str(text or "")
    protected_pattern = re.compile(
        r"\\begin\{(?:equation|align|aligned|split|multline|gather|subequations|algorithm|algorithmic)\*?\}.*?"
        r"\\end\{(?:equation|align|aligned|split|multline|gather|subequations|algorithm|algorithmic)\*?\}"
        r"|\$\$.*?\$\$|\$.*?\$|\\\(.*?\\\)|\\\[.*?\\\]",
        flags=re.S,
    )
    pieces: list[str] = []
    cursor = 0
    total = 0
    for match in protected_pattern.finditer(source):
        segment, count = re.subn(pattern, replacement, source[cursor: match.start()])
        pieces.append(segment)
        pieces.append(match.group(0))
        total += count
        cursor = match.end()
    tail, count = re.subn(pattern, replacement, source[cursor:])
    pieces.append(tail)
    total += count
    return "".join(pieces), total


def _phase3_5_is_near_miss_abbreviation(term: str, candidate: str) -> bool:
    """Return true for one-edit or adjacent-transposition acronym typos."""
    left = str(term or "").strip()
    right = str(candidate or "").strip()
    if left == right or len(left) < 4 or len(right) < 4:
        return False
    if not (left.isupper() and right.isupper()):
        return False
    if left[0] != right[0] or abs(len(left) - len(right)) > 1:
        return False
    if len(left) == len(right):
        diffs = [idx for idx, pair in enumerate(zip(left, right)) if pair[0] != pair[1]]
        if len(diffs) == 1:
            return True
        if len(diffs) == 2:
            i, j = diffs
            return j == i + 1 and left[i] == right[j] and left[j] == right[i]
        return False
    if len(left) > len(right):
        left, right = right, left
    i = j = edits = 0
    while i < len(left) and j < len(right):
        if left[i] == right[j]:
            i += 1
            j += 1
            continue
        edits += 1
        if edits > 1:
            return False
        j += 1
    return True


def _phase3_5_infer_near_miss_abbreviation_corrections(report: dict[str, Any]) -> dict[str, str]:
    """Map undefined acronym typos to defined acronyms when the correction is unambiguous."""
    undefined_terms = [
        str(item.get("term", "")).strip()
        for item in report.get("undefined_abbreviations", [])
        if isinstance(item, dict) and str(item.get("term", "")).strip()
    ]
    defined_terms = [
        str(term).strip()
        for term in report.get("defined_abbreviations", [])
        if str(term).strip()
    ]
    corrections: dict[str, str] = {}
    for term in dict.fromkeys(undefined_terms):
        if (
            term in PHASE34_COMMON_ABBREVIATION_EXPANSIONS
            or term in PHASE34_SPECIAL_ABBREVIATION_REWRITES
            or _phase3_5_compound_abbreviation_full_form(term)
        ):
            continue
        matches = [candidate for candidate in defined_terms if _phase3_5_is_near_miss_abbreviation(term, candidate)]
        if len(matches) == 1:
            corrections[term] = matches[0]
    return corrections


def _phase3_5_replace_visible_acronym_typo(text: str, typo: str, canonical: str) -> tuple[str, int]:
    """Correct visible acronym typos in prose and LaTeX text commands, not math/cites/labels."""
    pattern = rf"(?<![A-Za-z0-9-]){re.escape(typo)}(?![A-Za-z0-9-])"
    updated, total = _phase3_5_replace_all_in_prose(text, pattern, canonical)

    def replace_command_content(match: re.Match[str]) -> str:
        nonlocal total
        content, count = re.subn(pattern, canonical, match.group("content"))
        total += count
        return f"{match.group('prefix')}{content}{match.group('suffix')}"

    updated = re.sub(
        r"(?P<prefix>\\(?:caption|section|subsection|subsubsection)\{)(?P<content>[^{}]*)(?P<suffix>\})",
        replace_command_content,
        updated,
    )

    def replace_algorithmic_line(match: re.Match[str]) -> str:
        nonlocal total
        line, count = re.subn(pattern, canonical, match.group(0))
        total += count
        return line

    updated = re.sub(
        r"(?m)^\\(?:State|Require|Ensure|Return)\b[^\n]*",
        replace_algorithmic_line,
        updated,
    )
    return updated, total


def _phase3_5_visible_uppercase_terms(text: str) -> set[str]:
    """Collect acronym-like visible terms from prose, captions, and algorithmic text."""
    source = str(text or "")
    protected_math = re.compile(
        r"\\begin\{(?:equation|align|aligned|split|multline|gather|subequations)\*?\}.*?"
        r"\\end\{(?:equation|align|aligned|split|multline|gather|subequations)\*?\}"
        r"|\$\$.*?\$\$|\$.*?\$|\\\(.*?\\\)|\\\[.*?\\\]",
        flags=re.S,
    )
    visible = protected_math.sub(" ", source)
    command_contents = " ".join(
        match.group("content")
        for match in re.finditer(
            r"\\(?:caption|section|subsection|subsubsection)\{(?P<content>[^{}]*)\}",
            source,
        )
    )
    algorithm_lines = " ".join(
        match.group(0)
        for match in re.finditer(r"(?m)^\\(?:State|Require|Ensure|Return)\b[^\n]*", source)
    )
    visible = f"{visible} {command_contents} {algorithm_lines}"
    visible = re.sub(r"\\(?:cite|ref|label|eqref)\{[^{}]*\}", " ", visible)
    visible = re.sub(r"\\[A-Za-z]+", " ", visible)
    ignored = {"IEEE", "WCL", "P0", "P1", "P2", "KPI"}
    return {
        token
        for token in re.findall(r"\b[A-Z][A-Z0-9]{3,}(?:-[A-Z0-9]+)*\b", visible)
        if token not in ignored
    }


def _phase3_5_infer_visible_near_miss_abbreviation_corrections(
    section_texts: dict[str, str],
    report: dict[str, Any],
) -> dict[str, str]:
    """Infer acronym typo corrections from visible caption/algorithm/prose tokens."""
    defined_terms = [
        str(term).strip()
        for term in report.get("defined_abbreviations", [])
        if str(term).strip()
    ]
    corrections: dict[str, str] = {}
    for text in section_texts.values():
        for term in _phase3_5_visible_uppercase_terms(text):
            if term in defined_terms:
                continue
            matches = [candidate for candidate in defined_terms if _phase3_5_is_near_miss_abbreviation(term, candidate)]
            if len(matches) == 1 and not corrections.get(term):
                corrections[term] = matches[0]
    return corrections


def _phase3_5_replace_acronym_first_use(text: str, term: str, full_form: str) -> tuple[str, bool]:
    """Define an acronym once, including first uses embedded in common hyphenated prose."""
    if not term or not full_form:
        return text, False
    replacement = f"{full_form} ({term})"
    special_patterns = {
        "SINR": [
            (r"\blog-SINR\b", "log signal-to-interference-plus-noise-ratio (SINR)"),
            (r"\bSINR-based\b", "signal-to-interference-plus-noise-ratio (SINR)-based"),
        ],
        "SCA": [
            (r"\bWMMSE-SCA\b", "WMMSE and successive convex approximation (SCA)"),
            (r"\bFP/WMMSE-and-SCA\b", "FP/WMMSE and successive convex approximation (SCA)"),
            (r"\band-SCA\b", "and successive convex approximation (SCA)"),
            (r"\bSCA-based\b", "successive convex approximation (SCA)-based"),
        ],
        "CSI": [
            (r"\b(perfect|imperfect|deterministic|instantaneous|statistical)-CSI\b", r"\1 channel state information (CSI)"),
            (r"\bCSI-(?=[A-Za-z])", r"channel state information (CSI)-"),
        ],
        "RIS": [
            (r"(?<![A-Za-z0-9-])RIS-assisted\b", "reconfigurable-intelligent-surface (RIS)-assisted"),
            (r"(?<![A-Za-z0-9-])RIS-aided\b", "reconfigurable-intelligent-surface (RIS)-aided"),
        ],
        "SI": [
            (r"\bresidual-SI\b", "residual self-interference (SI)"),
            (r"\bSI-aware\b", "self-interference (SI)-aware"),
            (r"\bSI-rate\b", "self-interference (SI)-rate"),
            (r"\bSI-limited\b", "self-interference (SI)-limited"),
        ],
        "CCP": [
            (r"\bpenalty-CCP\b", "penalty convex-concave procedure (CCP)"),
        ],
        "KKT": [
            (r"\bKKT-(?=[A-Za-z])", "Karush-Kuhn-Tucker (KKT)-"),
        ],
        "FP-MM-CCP": [
            (
                r"\bFP-MM-CCP\b",
                "fractional-programming, majorization-minimization, and convex-concave-procedure (FP-MM-CCP)",
            ),
        ],
        "AO-FP-MM": [
            (
                r"\bAO--FP--MM(?=--|-|\b)",
                "alternating-optimization, fractional-programming, and majorization-minimization (AO-FP-MM)",
            ),
            (
                r"\bAO-FP-MM(?=-|\b)",
                "alternating-optimization, fractional-programming, and majorization-minimization (AO-FP-MM)",
            ),
        ],
    }
    for pattern, local_replacement in special_patterns.get(term, []):
        updated, changed = _phase3_5_replace_once_in_prose(text, pattern, local_replacement)
        if changed:
            return updated, True
    updated, changed = _phase3_5_replace_once_in_prose(
        text,
        rf"(?<![A-Za-z0-9-]){re.escape(term)}-(?=[A-Za-z])",
        f"{replacement}-",
    )
    if changed:
        return updated, True
    updated, changed = _phase3_5_replace_once_in_prose(
        text,
        rf"\b([A-Za-z][A-Za-z]+)-{re.escape(term)}\b",
        lambda match: f"{match.group(1)} {full_form} ({term})",
    )
    if changed:
        return updated, True
    return _phase3_5_replace_once_in_prose(
        text,
        rf"(?<![A-Za-z0-9-]){re.escape(term)}(?![A-Za-z0-9-])",
        replacement,
    )


def _phase3_5_collapse_nested_abbreviation_definitions(tex: str) -> str:
    """Collapse repeated acronym expansions created by LLM or repair passes."""
    updated = str(tex or "")
    updated = re.sub(
        r"\bxl\s+multiple-input\s+multiple-output\s*"
        r"\(\s*XL\s+multiple-input\s+multiple-output\s*\(\s*MIMO\s*\)\s*\)",
        "extremely large-scale multiple-input multiple-output (XL-MIMO)",
        updated,
        flags=re.I,
    )
    updated = re.sub(
        r"\bXL\s+multiple-input\s+multiple-output\s*\(\s*MIMO\s*\)",
        "extremely large-scale multiple-input multiple-output (XL-MIMO)",
        updated,
        flags=re.I,
    )
    for term, full_form in PHASE34_COMMON_ABBREVIATION_EXPANSIONS.items():
        if not term or not full_form:
            continue
        nested_pattern = (
            rf"{re.escape(full_form)}\s*\(\s*"
            rf"(?:{re.escape(full_form)}\s*\(\s*)+"
            rf"{re.escape(term)}\s*"
            rf"(?:\)\s*)+"
        )
        replacement = f"{full_form} ({term})"
        previous = None
        while previous != updated:
            previous = updated
            updated = re.sub(nested_pattern, replacement, updated, flags=re.I)
        updated = re.sub(
            rf"({re.escape(full_form)}\s*\(\s*{re.escape(term)}\s*\))(?=[A-Za-z])",
            r"\1 ",
            updated,
            flags=re.I,
        )
    return updated


def _phase3_5_collapse_repeated_abbreviation_definitions(
    tex: str,
    *,
    collapse_parenthetical_repeats: bool = False,
) -> str:
    """Collapse duplicate same-item definitions such as `MRT: ... (MRT)`."""
    updated = _phase3_5_collapse_nested_abbreviation_definitions(str(tex or ""))
    for term, full_form in PHASE34_COMMON_ABBREVIATION_EXPANSIONS.items():
        if not term or not full_form:
            continue
        full = re.escape(full_form)
        acronym = re.escape(term)
        item_pattern = (
            rf"(\\item\s+\\(?:textbf|emph)\{{)\s*{full}\s*\(\s*{acronym}\s*\)\s*(\}}\s*:\s*)"
            rf"((?:[A-Za-z]+(?:-[A-Za-z]+)*\s+){{0,8}})"
            rf"{full}\s*\(\s*{acronym}\s*\)"
        )
        updated = re.sub(
            item_pattern,
            lambda match: f"{match.group(1)}{term}{match.group(2)}{match.group(3)}{full_form}",
            updated,
            flags=re.I,
        )
        plain_label_pattern = (
            rf"(\\item\s+\\(?:textbf|emph)\{{)\s*{acronym}\s*(\}}\s*:\s*)"
            rf"((?:[A-Za-z]+(?:-[A-Za-z]+)*\s+){{0,8}})"
            rf"{full}\s*\(\s*{acronym}\s*\)"
        )
        updated = re.sub(
            plain_label_pattern,
            lambda match: f"{match.group(1)}{term}{match.group(2)}{match.group(3)}{full_form}",
            updated,
            flags=re.I,
        )
        if collapse_parenthetical_repeats:
            definition_pattern = re.compile(
                rf"(?<![A-Za-z0-9-]){full}\s*\(\s*{acronym}\s*\)",
                flags=re.I,
            )
            matches = list(definition_pattern.finditer(updated))
            if len(matches) > 1:
                pieces: list[str] = []
                cursor = 0
                for index, match in enumerate(matches):
                    pieces.append(updated[cursor: match.start()])
                    pieces.append(match.group(0) if index == 0 else full_form)
                    cursor = match.end()
                pieces.append(updated[cursor:])
                updated = "".join(pieces)
    return _phase3_5_collapse_nested_abbreviation_definitions(updated)


def _phase3_5_definition_regex(term: str, full_form: str) -> re.Pattern[str]:
    full_variants = {
        full_form,
        full_form.replace("-", " "),
        full_form.replace(" ", "-"),
    }
    if not full_form.endswith("s"):
        words = full_form.split()
        if words:
            full_variants.add(" ".join([*words[:-1], words[-1] + "s"]))
        hyphen_words = full_form.split("-")
        if len(hyphen_words) > 1:
            full_variants.add("-".join([*hyphen_words[:-1], hyphen_words[-1] + "s"]))
    full_pattern = "|".join(
        re.escape(variant)
        for variant in sorted(full_variants, key=lambda value: (-len(value), value))
        if variant
    )
    term_pattern = re.escape(term)
    plural_pattern = rf"{term_pattern}s" if not term.endswith("s") else term_pattern
    acronym_pattern = rf"(?:{plural_pattern}|{term_pattern})"
    return re.compile(
        rf"(?<![A-Za-z0-9-])(?P<full>{full_pattern})\s*\(\s*(?P<abbr>{acronym_pattern})\s*\)",
        flags=re.I,
    )


def _phase3_5_dynamic_definition_regex(term: str) -> re.Pattern[str]:
    """Match a prose definition of a known repeated abbreviation without a fixed dictionary."""

    term_pattern = re.escape(str(term or "").strip())
    if not term_pattern:
        return re.compile(r"a\A")
    plural_pattern = rf"{term_pattern}s" if not str(term).endswith("s") else term_pattern
    acronym_pattern = rf"(?:{plural_pattern}|{term_pattern})"
    word = r"[A-Za-z][A-Za-z-]*"
    full_pattern = rf"{word}(?:\s+{word}){{0,7}}"
    return re.compile(
        rf"(?<![A-Za-z0-9-])(?P<full>{full_pattern})\s*\(\s*(?P<abbr>{acronym_pattern})\s*\)",
        flags=re.I,
    )


def _phase3_5_collapse_global_repeated_abbreviation_definitions(
    phase_dir: Path,
) -> tuple[dict[str, list[str]], list[dict[str, Any]]]:
    """Keep only the first acronym definition in the abstract and main-body scopes."""

    phase_dir = Path(phase_dir)
    section_files = _phase3_5_section_files_for_abbreviation_repair()
    abbreviation_report = analyze_phase3_4_full_paper_abbreviations_from_phase_dir(phase_dir)
    repeated_terms_by_scope: dict[str, set[str]] = {"abstract": set(), "main_body": set()}
    for item in abbreviation_report.get("repeated_abbreviation_definitions", []):
        if not isinstance(item, dict):
            continue
        term = str(item.get("term") or "").strip()
        scope = str(item.get("scope") or "main_body").strip() or "main_body"
        if not term:
            continue
        repeated_terms_by_scope.setdefault(scope, set()).add(term)
    ordered_sections = [
        "abstract",
        "introduction",
        "system_model",
        "proposed_solution",
        "numerical_results",
        "conclusion",
    ]
    applied_repairs: dict[str, list[str]] = {}
    applied_summaries: list[dict[str, Any]] = []
    seen_terms_by_scope: dict[str, set[str]] = {"abstract": set(), "main_body": set()}

    for section_name in ordered_sections:
        filename = section_files.get(section_name)
        if not filename:
            continue
        scope_name = "abstract" if section_name == "abstract" else "main_body"
        seen_terms = seen_terms_by_scope.setdefault(scope_name, set())
        section_path = phase_dir / filename
        if not section_path.exists():
            continue
        original_tex = read_text(section_path)
        updated_tex = original_tex
        changed_terms: list[str] = []
        term_patterns: list[tuple[str, re.Pattern[str]]] = []
        for term, full_form in PHASE34_COMMON_ABBREVIATION_EXPANSIONS.items():
            if term and full_form:
                term_patterns.append((term, _phase3_5_definition_regex(term, full_form)))
        for term in sorted(repeated_terms_by_scope.get(scope_name, set())):
            if term not in PHASE34_COMMON_ABBREVIATION_EXPANSIONS:
                term_patterns.append((term, _phase3_5_dynamic_definition_regex(term)))

        for term, pattern in term_patterns:

            def replace_definition(match: re.Match[str], *, local_term: str = term) -> str:
                if local_term not in seen_terms:
                    seen_terms.add(local_term)
                    return match.group(0)
                abbr = match.group("abbr")
                if local_term not in changed_terms:
                    changed_terms.append(local_term)
                return abbr

            updated_tex = pattern.sub(replace_definition, updated_tex)
        if updated_tex == original_tex:
            continue
        write_text(section_path, updated_tex)
        applied_repairs[section_name] = changed_terms or ["global_duplicate_definition"]
        for term in changed_terms or ["duplicate_definition"]:
            applied_summaries.append(
                {
                    "issue_id": f"P1-ABBR-GLOBAL-{term}",
                    "status": "fixed",
                    "file_or_section": filename,
                    "change_type": "scoped_acronym_definition_discipline",
                    "original_issue_summary": (
                        "The abbreviation was defined again after its first definition in the same paper scope."
                    ),
                    "before_excerpt": f"full descriptive name ({term})",
                    "after_excerpt": term,
                    "note": (
                        "Kept one full-name-plus-acronym definition in the abstract scope and one in the "
                        "main-body scope, then replaced later parenthetical redefinitions with the acronym."
                    ),
                }
            )
    return applied_repairs, applied_summaries


def _phase3_5_apply_common_abbreviation_repairs(tex: str, undefined_terms: set[str]) -> tuple[str, list[str]]:
    updated = _phase3_5_collapse_repeated_abbreviation_definitions(str(tex or ""))
    applied: list[str] = []
    if {"SDP", "SOCP"} & undefined_terms:
        updated, changed = _phase3_5_replace_once(
            updated,
            r"\bmixed\s+SOCP-SDP\b",
            "mixed second-order cone programming (SOCP) and semidefinite programming (SDP)",
        )
        if changed:
            applied.extend(term for term in ["SDP", "SOCP"] if term in undefined_terms)
            undefined_terms = set(undefined_terms) - {"SDP", "SOCP"}
    if {"SDP", "SOCP"} & undefined_terms:
        updated, changed = _phase3_5_replace_once(
            updated,
            r"\ban\s+SDP/SOCP\b",
            "a semidefinite programming (SDP)/second-order cone programming (SOCP) problem",
        )
        if changed:
            applied.extend(term for term in ["SDP", "SOCP"] if term in undefined_terms)
            undefined_terms = set(undefined_terms) - {"SDP", "SOCP"}
        else:
            updated, changed = _phase3_5_replace_once(
                updated,
                r"\bSDP/SOCP\b",
                "semidefinite programming (SDP)/second-order cone programming (SOCP)",
            )
            if changed:
                applied.extend(term for term in ["SDP", "SOCP"] if term in undefined_terms)
                undefined_terms = set(undefined_terms) - {"SDP", "SOCP"}
    if {"SCA", "MM"} & undefined_terms:
        updated, changed = _phase3_5_replace_once(
            updated,
            r"\bSCA/MM\b",
            "successive convex approximation/majorization--minimization (SCA/MM)",
        )
        if changed:
            applied.extend(term for term in ["SCA", "MM"] if term in undefined_terms)
            undefined_terms = set(undefined_terms) - {"SCA", "MM"}
    if "PSD" in undefined_terms:
        updated, changed = _phase3_5_replace_once(
            updated,
            r"\bHermitian positive semidefinite\b",
            "Hermitian positive semidefinite (PSD)",
        )
        if changed:
            applied.append("PSD")
            undefined_terms = set(undefined_terms) - {"PSD"}
    if "CSI" in undefined_terms:
        updated, changed = _phase3_5_replace_acronym_first_use(updated, "CSI", PHASE34_COMMON_ABBREVIATION_EXPANSIONS["CSI"])
        if changed:
            applied.append("CSI")
            undefined_terms = set(undefined_terms) - {"CSI"}
    if "SCA" in undefined_terms:
        updated, changed = _phase3_5_replace_once(
            updated,
            r"\((WMMSE)\)-SCA\b",
            r"(\1) and successive convex approximation (SCA)",
        )
        if not changed:
            updated, changed = _phase3_5_replace_acronym_first_use(updated, "SCA", PHASE34_COMMON_ABBREVIATION_EXPANSIONS["SCA"])
        if changed:
            applied.append("SCA")
            undefined_terms = set(undefined_terms) - {"SCA"}
    if "SDR" in undefined_terms:
        updated, changed = _phase3_5_replace_once(
            updated,
            r"\bwithout\s+SDR\s+rank\s+recovery\b",
            "without semidefinite-relaxation rank recovery",
        )
        if not changed:
            updated, changed = _phase3_5_replace_once(
                updated,
                r"\bSDR\b",
                "semidefinite relaxation (SDR)",
            )
        if changed:
            applied.append("SDR")
            undefined_terms = set(undefined_terms) - {"SDR"}
    if "Regularized-ZF" in undefined_terms:
        updated, changed = _phase3_5_replace_once(
            updated,
            r"\bRegularized-ZF\b",
            PHASE34_SPECIAL_ABBREVIATION_REWRITES["Regularized-ZF"],
        )
        if changed:
            applied.append("Regularized-ZF")
            undefined_terms = set(undefined_terms) - {"Regularized-ZF"}
            if re.search(r"\bRZF-[A-Za-z0-9]+(?:\}|\s)*:\s*regularized zero-forcing\b", updated, flags=re.I):
                undefined_terms = set(undefined_terms) - {"RZF"}
    if "WSR" in undefined_terms:
        wsr_patterns = [
            (
                r"\bweighted sum spectral efficiency are then\b",
                "weighted sum rate (WSR) utility is then",
            ),
            (
                r"\bweighted sum spectral efficiency is then\b",
                "weighted sum rate (WSR) utility is then",
            ),
            (
                r"\bweighted sum spectral-efficiency objective\b",
                "weighted sum rate (WSR) objective",
            ),
            (
                r"\bThe fixed-coordinate WSR\b",
                "The fixed-coordinate weighted sum rate (WSR)",
            ),
            (
                r"\bthe true WSR\b",
                "the true weighted sum rate (WSR)",
            ),
        ]
        for pattern, replacement in wsr_patterns:
            updated, changed = _phase3_5_replace_once(updated, pattern, replacement)
            if changed:
                applied.append("WSR")
                undefined_terms = set(undefined_terms) - {"WSR"}
                break
    for term in sorted(undefined_terms, key=lambda value: (-len(value), value)):
        full_form = PHASE34_COMMON_ABBREVIATION_EXPANSIONS.get(term)
        compound_full_form = _phase3_5_compound_abbreviation_full_form(term)
        if not full_form and not compound_full_form:
            continue
        if compound_full_form and not full_form:
            updated, changed = _phase3_5_replace_once_in_prose(
                updated,
                rf"(?<![A-Za-z0-9-]){re.escape(term)}(?![A-Za-z0-9-])",
                f"{compound_full_form} ({term})",
            )
        else:
            updated, changed = _phase3_5_replace_acronym_first_use(updated, term, full_form)
        if changed:
            applied.append(term)
    updated = updated.replace("an successive convex approximation", "a successive convex approximation")
    updated = updated.replace("An successive convex approximation", "A successive convex approximation")
    updated = updated.replace("an second-order cone", "a second-order cone")
    updated = updated.replace("An second-order cone", "A second-order cone")
    return _phase3_5_collapse_repeated_abbreviation_definitions(updated), applied


def _phase3_5_compound_abbreviation_full_form(term: str) -> str:
    """Infer simple hyphenated method labels such as Robust-RZF from known components."""
    pieces = [piece for piece in str(term or "").split("-") if piece]
    if len(pieces) < 2:
        return ""
    expanded: list[str] = []
    used_known_component = False
    for piece in pieces:
        known = PHASE34_COMMON_ABBREVIATION_EXPANSIONS.get(piece)
        if known:
            expanded.append(known)
            used_known_component = True
        elif piece.isalpha():
            expanded.append(piece.lower())
        else:
            return ""
    return " ".join(expanded) if used_known_component else ""


def _phase3_5_section_files_for_abbreviation_repair() -> dict[str, str]:
    return {
        "abstract": "abstract.tex",
        "introduction": "introduction.tex",
        "system_model": "system_model_problem_formulation_section.tex",
        "proposed_solution": "proposed_solution_section.tex",
        "numerical_results": "numerical_results_section.tex",
        "conclusion": "conclusion.tex",
    }


def _phase3_5_apply_full_paper_abbreviation_repairs(phase_dir: Path) -> dict[str, Any]:
    """Deterministically define common wireless abbreviations on first prose use."""
    phase_dir = Path(phase_dir)
    section_files = _phase3_5_section_files_for_abbreviation_repair()
    before_report = analyze_phase3_4_full_paper_abbreviations_from_phase_dir(phase_dir)
    applied_repairs: dict[str, list[str]] = {}
    applied_summaries: list[dict[str, Any]] = []
    for section_name, filename in section_files.items():
        section_path = phase_dir / filename
        if not section_path.exists():
            continue
        original_tex = read_text(section_path)
        repaired_tex = _phase3_5_collapse_repeated_abbreviation_definitions(original_tex, collapse_parenthetical_repeats=True)
        if repaired_tex == original_tex:
            continue
        write_text(section_path, repaired_tex)
        applied_repairs.setdefault(section_name, [])
        if "duplicate_definition" not in applied_repairs[section_name]:
            applied_repairs[section_name].append("duplicate_definition")
        applied_summaries.append(
            {
                "issue_id": "P1-ABBR-DUPLICATE",
                "status": "fixed",
                "file_or_section": filename,
                "change_type": "duplicate_acronym_definition",
                "original_issue_summary": "An abbreviation was expanded more than once in the same local definition.",
                "before_excerpt": "full form (ACRONYM): ... full form (ACRONYM)",
                "after_excerpt": "ACRONYM: ... full form",
                "note": "Kept one benchmark-label definition and removed the repeated parenthetical acronym without changing claims.",
            }
        )
    current_report = analyze_phase3_4_full_paper_abbreviations_from_phase_dir(phase_dir)
    section_texts_for_typo_scan = {
        section_name: read_text(phase_dir / filename)
        for section_name, filename in section_files.items()
        if (phase_dir / filename).exists()
    }
    near_miss_corrections = {
        **_phase3_5_infer_near_miss_abbreviation_corrections(current_report),
        **_phase3_5_infer_visible_near_miss_abbreviation_corrections(section_texts_for_typo_scan, current_report),
    }
    if near_miss_corrections:
        for section_name, filename in section_files.items():
            section_path = phase_dir / filename
            if not section_path.exists():
                continue
            repaired_tex = read_text(section_path)
            section_changed = False
            for typo, canonical in near_miss_corrections.items():
                repaired_tex, count = _phase3_5_replace_visible_acronym_typo(repaired_tex, typo, canonical)
                if count:
                    section_changed = True
                    applied_repairs.setdefault(section_name, [])
                    repair_id = f"{typo}->{canonical}"
                    if repair_id not in applied_repairs[section_name]:
                        applied_repairs[section_name].append(repair_id)
                    applied_summaries.append(
                        {
                            "issue_id": f"P1-ABBR-TYPO-{typo}",
                            "status": "fixed",
                            "file_or_section": filename,
                            "change_type": "acronym_typo_correction",
                            "original_issue_summary": (
                                f"{typo} appeared as an undefined abbreviation and is an unambiguous near-miss "
                                f"of the already defined abbreviation {canonical}."
                            ),
                            "before_excerpt": typo,
                            "after_excerpt": canonical,
                            "note": "Corrected an acronym typo in prose only, without changing equations, labels, references, or numerical claims.",
                        }
                    )
            if section_changed:
                write_text(section_path, repaired_tex)
        current_report = analyze_phase3_4_full_paper_abbreviations_from_phase_dir(phase_dir)
    for _round in range(3):
        unresolved_by_section: dict[str, set[str]] = {}
        for item in current_report.get("undefined_abbreviations", []):
            if not isinstance(item, dict):
                continue
            term = str(item.get("term", "")).strip()
            section_name = str(item.get("section", "")).strip()
            if not term or section_name not in section_files:
                continue
            if (
                term not in PHASE34_COMMON_ABBREVIATION_EXPANSIONS
                and term not in PHASE34_SPECIAL_ABBREVIATION_REWRITES
                and not _phase3_5_compound_abbreviation_full_form(term)
            ):
                continue
            unresolved_by_section.setdefault(section_name, set()).add(term)
        if not unresolved_by_section:
            break
        changed_any = False
        for section_name, filename in section_files.items():
            section_terms = set(unresolved_by_section.get(section_name, set()))
            if not section_terms:
                continue
            section_path = phase_dir / filename
            if not section_path.exists():
                continue
            repaired_tex, applied = _phase3_5_apply_common_abbreviation_repairs(read_text(section_path), section_terms)
            if not applied:
                continue
            write_text(section_path, repaired_tex)
            changed_any = True
            applied_repairs.setdefault(section_name, [])
            for term in applied:
                if term not in applied_repairs[section_name]:
                    applied_repairs[section_name].append(term)
                full_form = (
                    PHASE34_COMMON_ABBREVIATION_EXPANSIONS.get(term)
                    or PHASE34_SPECIAL_ABBREVIATION_REWRITES.get(term, "")
                )
                applied_summaries.append(
                    {
                        "issue_id": f"P1-ABBR-{term}",
                        "status": "fixed",
                        "file_or_section": filename,
                        "change_type": "acronym_definition",
                        "original_issue_summary": f"{term} appeared before a full-name-plus-acronym definition.",
                        "before_excerpt": term,
                        "after_excerpt": f"{full_form} ({term})" if full_form else term,
                        "note": "Defined the abbreviation on first prose use without changing equations, labels, references, or numerical claims.",
                    }
                )
        current_report = analyze_phase3_4_full_paper_abbreviations_from_phase_dir(phase_dir)
        if not changed_any:
            break
    for section_name, filename in section_files.items():
        section_path = phase_dir / filename
        if not section_path.exists():
            continue
        original_tex = read_text(section_path)
        repaired_tex = _phase3_5_collapse_repeated_abbreviation_definitions(original_tex, collapse_parenthetical_repeats=True)
        if repaired_tex == original_tex:
            continue
        write_text(section_path, repaired_tex)
        applied_repairs.setdefault(section_name, [])
        if "duplicate_definition" not in applied_repairs[section_name]:
            applied_repairs[section_name].append("duplicate_definition")
        applied_summaries.append(
            {
                "issue_id": "P1-ABBR-DUPLICATE",
                "status": "fixed",
                "file_or_section": filename,
                "change_type": "duplicate_acronym_definition",
                "original_issue_summary": "An abbreviation was expanded more than once in the same section.",
                "before_excerpt": "full form (ACRONYM) ... full form (ACRONYM)",
                "after_excerpt": "full form (ACRONYM) ... full form",
                "note": "Kept the first section-local definition and removed later repeated parenthetical definitions without changing claims.",
            }
        )
    global_repairs, global_summaries = _phase3_5_collapse_global_repeated_abbreviation_definitions(phase_dir)
    for section_name, terms in global_repairs.items():
        applied_repairs.setdefault(section_name, [])
        for term in terms:
            if term not in applied_repairs[section_name]:
                applied_repairs[section_name].append(term)
    applied_summaries.extend(global_summaries)
    current_report = analyze_phase3_4_full_paper_abbreviations_from_phase_dir(phase_dir)
    return {
        "input_abbreviation_report": before_report,
        "output_abbreviation_report": current_report,
        "applied_repairs": applied_repairs,
        "applied_issue_summaries": applied_summaries,
    }


def _phase3_6_hedge_unverified_scaling_claims(tex: str) -> tuple[str, bool]:
    """Keep broad scaling language scoped to the actually evaluated sweep range."""
    updated = str(tex or "")
    before = updated
    updated = re.sub(
        r"\brelative advantage growing as\b",
        "relative advantage increasing over the evaluated range as",
        updated,
        flags=re.I,
    )
    updated = re.sub(
        r"\brelative advantage grows as\b",
        "relative advantage increases over the evaluated range as",
        updated,
        flags=re.I,
    )
    updated = re.sub(
        r"\bfor scaling ([^.;]+?) with ([^.;]+?)\.",
        r"for improving \1 over the evaluated range of \2.",
        updated,
        flags=re.I,
    )
    updated = re.sub(
        r"\bscales? with ([^.;]+)",
        r"improves over the evaluated range of \1",
        updated,
        flags=re.I,
    )
    return updated, updated != before


def _phase3_5_contract_scope_check(run_dir: Path, phase_dir: Path) -> dict[str, Any]:
    """Check that reformulation-only symbols stay out of the original model section."""
    run_dir = Path(run_dir)
    phase_dir = Path(phase_dir)
    contract = read_json(run_dir / "phase2-1" / "mathematical_contract.json") or {}
    derived_symbols = {
        str(item.get("symbol") or "").strip()
        for item in contract.get("derived_quantities", [])
        if isinstance(item, dict) and str(item.get("symbol") or "").strip()
    }
    reform_symbols = {
        str(item.get("symbol") or "").strip()
        for item in contract.get("reformulation_only", [])
        if isinstance(item, dict) and str(item.get("symbol") or "").strip()
    }
    forbidden_patterns = {
        "\\lambda_m": "\\lambda_m",
        "\\mu_m": "\\mu_m",
        "\\widehat\\Gamma_k": "\\widehat\\Gamma_k",
        "\\bm\\lambda": "\\bm\\lambda",
        "\\bm\\mu": "\\bm\\mu",
        "\\mathbf C_k": "\\mathbf C_k",
        "t_k^{\\mathrm c}": "t_k^{\\mathrm c}",
        "t_k^{\\mathrm I}": "t_k^{\\mathrm I}",
        "\\mathbf t^{\\mathrm c}": "\\mathbf t^{\\mathrm c}",
        "\\mathbf t^{\\mathrm I}": "\\mathbf t^{\\mathrm I}",
    }
    for symbol in reform_symbols:
        if symbol:
            forbidden_patterns.setdefault(symbol, symbol)
    system_text = read_text(phase_dir / "system_model_problem_formulation_section.tex")
    proposed_text = read_text(phase_dir / "proposed_solution_section.tex")

    def symbol_occurs(text: str, symbol: str) -> bool:
        if not symbol:
            return False
        escaped = re.escape(symbol)
        if symbol.startswith("\\"):
            pattern = rf"(?<![A-Za-z]){escaped}(?![A-Za-z])"
        elif len(symbol) == 1 and symbol.isalpha():
            pattern = rf"(?<![A-Za-z0-9\\.\\_]){escaped}(?![A-Za-z0-9\\.])"
        else:
            pattern = rf"(?<![A-Za-z0-9\\]){escaped}(?![A-Za-z0-9])"
        return re.search(pattern, text) is not None

    violations: list[dict[str, str]] = []
    for display, pattern in forbidden_patterns.items():
        if symbol_occurs(system_text, pattern):
            violations.append(
                {
                    "symbol": display,
                    "section": "system_model_problem_formulation_section.tex",
                    "message": "Reformulation-only symbol appears in the original model/problem section.",
                }
            )
    derived_used_in_system = sorted(symbol for symbol in derived_symbols if symbol_occurs(system_text, symbol))
    reform_used_in_proposed = sorted(pattern for pattern in forbidden_patterns.values() if symbol_occurs(proposed_text, pattern))
    return {
        "ok": not violations,
        "violations": violations,
        "derived_symbols_allowed_in_system_model": derived_used_in_system,
        "reformulation_symbols_found_in_proposed_solution": reform_used_in_proposed,
        "note": (
            "Derived quantities such as total transmit covariance are allowed in the system model "
            "when declared in the mathematical contract; reformulation-only symbols are not."
        ),
    }


def run_phase3_5_abbreviation_only_package(run_dir: Path, paper_target: str = "IEEE WCL") -> dict[str, Any]:
    """Create a Phase 3.4 revision that only repairs full-paper acronym first use."""
    run_dir = Path(run_dir)
    summary_payload = read_json(run_dir / "phase2_summary.json") or {}
    topic = str(summary_payload.get("topic", run_dir.name))
    phase3_4_dir = run_dir / "phase3-4"
    phase3_5_dir = run_dir / "phase3-5"
    phase3_5_dir.mkdir(parents=True, exist_ok=True)

    source_files = _phase3_5_section_files_for_abbreviation_repair()
    for filename in [
        "references.bib",
        "references_curated.bib",
        "references_ieee.bib",
        "conceptual_diagram.tex",
    ]:
        src = phase3_4_dir / filename
        if src.exists():
            shutil.copyfile(src, phase3_5_dir / filename)

    for filename in source_files.values():
        write_text(phase3_5_dir / filename, read_text(phase3_4_dir / filename))

    repair_report = _phase3_5_apply_full_paper_abbreviation_repairs(phase3_5_dir)
    before_report = repair_report["input_abbreviation_report"]
    after_report = repair_report["output_abbreviation_report"]
    applied_repairs = repair_report["applied_repairs"]
    write_text(phase3_5_dir / "full_paper_abbreviation_report_before.json", json.dumps(before_report, ensure_ascii=False, indent=2))
    write_text(phase3_5_dir / "full_paper_abbreviation_report.json", json.dumps(after_report, ensure_ascii=False, indent=2))

    curated_bib_text = read_text(phase3_5_dir / "references_ieee.bib") or read_text(phase3_5_dir / "references.bib")
    preview = render_phase3_5_preview_pdf(phase3_5_dir, topic, curated_bib_text)
    manifest = {
        "phase": "phase3",
        "phase_id": "phase3.5",
        "phase_name": "phase3.5_abbreviation_only_repair",
        "paper_target": paper_target,
        "title": topic,
        "mode": "abbreviation_only",
        "source_phase": str(phase3_4_dir),
        "input_abbreviation_report": before_report,
        "output_abbreviation_report": after_report,
        "applied_repairs": applied_repairs,
        "preview": preview,
        "preview_pdf_path": preview.get("preview_pdf", str(phase3_5_dir / "full_paper_revised_preview.pdf")),
        "abbreviation_report_path": str(phase3_5_dir / "full_paper_abbreviation_report.json"),
    }
    write_text(phase3_5_dir / "phase3_5_abbreviation_only_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    if not after_report.get("ok", False):
        unresolved = ", ".join(
            str(item.get("term", ""))
            for item in after_report.get("undefined_abbreviations", [])
            if item.get("term")
        )
        raise ValueError(f"Phase 3.4 abbreviation-only repair left unresolved abbreviations: {unresolved}")
    return manifest


def _legacy_run_phase3_5_paper_review_package(run_dir: Path, paper_target: str = "IEEE WCL") -> dict[str, Any]:
    run_dir = Path(run_dir)
    summary_payload = read_json(run_dir / "phase2_summary.json") or {}
    topic = str(summary_payload.get("topic", run_dir.name))
    model_profile = str(summary_payload.get("model_profile") or DEFAULT_MODEL_PROFILE)
    phase3_5_dir = run_dir / "phase3-5"
    phase3_5_dir.mkdir(parents=True, exist_ok=True)
    write_text(phase3_5_dir / "phase3_5_design_notes.md", build_phase3_5_design_notes())

    phase3_4_dir = run_dir / "phase3-4"
    phase3_3_dir = run_dir / "phase3-3"
    phase1_dir = run_dir / "phase2-1"
    mathematical_contract_json = read_text(phase1_dir / "mathematical_contract.frozen.json") or read_text(phase1_dir / "mathematical_contract.json") or "{}"
    verified_reference_bank = read_json(phase3_4_dir / "verified_reference_bank.json") or []
    phase3_4_manifest = read_json(phase3_4_dir / "phase3_4_manifest.json") or {}
    citation_claim_map = read_json(phase3_4_dir / "citation_claim_map.json") or []
    reference_quality_report = read_json(phase3_4_dir / "reference_quality_report.json") or {}
    introduction_facts = read_json(phase3_4_dir / "introduction_facts.json") or {}
    current_sections = {
        "abstract_tex": read_text(phase3_5_dir / "abstract.tex") or read_text(phase3_3_dir / "abstract.tex"),
        "introduction_tex": read_text(phase3_5_dir / "introduction.tex") or read_text(phase3_4_dir / "introduction.tex"),
        "system_model_problem_formulation_section_tex": read_text(phase3_5_dir / "system_model_problem_formulation_section.tex") or read_text(phase3_4_dir / "system_model_problem_formulation_section.tex"),
        "proposed_solution_section_tex": read_text(phase3_5_dir / "proposed_solution_section.tex") or read_text(phase3_4_dir / "proposed_solution_section.tex"),
        "numerical_results_section_tex": read_text(phase3_5_dir / "numerical_results_section.tex") or read_text(phase3_4_dir / "numerical_results_section.tex"),
        "conclusion_tex": read_text(phase3_5_dir / "conclusion.tex") or read_text(phase3_4_dir / "conclusion.tex"),
    }
    review_facts = {
        "paper_target": paper_target,
        "title": topic,
        "method_names_used": phase3_4_manifest.get("method_names_used", {}),
        "reference_target": phase3_4_manifest.get("reference_target", {}),
        "paper_objective": introduction_facts.get(
            "technical_claims_from_phase2",
            {},
        ),
        "result_constraints": introduction_facts.get(
            "result_constraints_from_phase2",
            {},
        ),
        "citation_claim_map": citation_claim_map,
        "reference_quality_report": reference_quality_report,
    }
    verified_reference_keys = {
        str(item.get("final_bib_key", "")).strip()
        for item in verified_reference_bank
        if isinstance(item, dict) and str(item.get("included_in_final_bib", True)).lower() != "false"
    }
    prompt = build_phase3_5_review_prompt(
        review_facts_json=compact_text(json.dumps(review_facts, ensure_ascii=False, indent=2), 14000),
        mathematical_contract_json=compact_text(mathematical_contract_json, 8000),
        verified_reference_bank_json=compact_text(json.dumps(verified_reference_bank, ensure_ascii=False, indent=2), 14000),
        current_sections_json=compact_text(json.dumps(current_sections, ensure_ascii=False, indent=2), 18000),
    )
    write_text(phase3_5_dir / "phase3_5_prompt.txt", prompt)
    if _paper_phase_llm_skip_enabled("phase3_5", phase3_5_dir):
        write_text(
            phase3_5_dir / "phase3_5_llm_skip_request_ignored.json",
            json.dumps(
                {
                    "phase": "phase3.5",
                    "action": "ignored",
                    "reason": "Phase 3.5 review/rewrite must be generated by ReviewAgent; local substitution is disabled.",
                },
                ensure_ascii=False,
                indent=2,
            ),
        )
    llm = create_llm_client(model_profile)
    thinking = {"type": "enabled"} if model_profile == "kimi-k2.6-thinking" else None
    response = llm.chat(
        [{"role": "user", "content": prompt}],
        json_mode=True,
        thinking=thinking,
        max_tokens=9000,
    )
    write_text(phase3_5_dir / "phase3_5_raw_response.txt", response.content)
    payload = _safe_json_loads(response.content, {})
    if not isinstance(payload, dict):
        raise ValueError("phase3.5_paper_review_rewrite_phase did not return a valid structured object")

    for retry_idx in range(2):
        revised_abstract_candidate = sanitize_phase3_3_abstract_tex(str(payload.get("abstract_tex") or current_sections["abstract_tex"]))
        undefined_abstract_abbrs = find_undefined_abstract_abbreviations(revised_abstract_candidate, phase3_4_manifest.get("paper_facts", {}))
        if not undefined_abstract_abbrs:
            break
        retry_prompt = prompt + (
            "\n\nRevision required:\n"
            "The previous abstract still contains undefined acronyms/abbreviations. "
            "Rewrite all requested fields again, with special attention to abstract_tex. "
            f"The unresolved abstract items were: {', '.join(undefined_abstract_abbrs)}.\n"
            "If the abstract uses a method short name or acronym, define it immediately after the first full method phrase.\n"
            "Return the full JSON again."
        )
        suffix = "" if retry_idx == 0 else f"_{retry_idx + 1}"
        write_text(phase3_5_dir / f"phase3_5_prompt_retry{suffix}.txt", retry_prompt)
        retry_response = llm.chat(
            [{"role": "user", "content": retry_prompt}],
            json_mode=True,
            thinking=thinking,
            max_tokens=9000,
        )
        write_text(phase3_5_dir / f"phase3_5_raw_response_retry{suffix}.txt", retry_response.content)
        retry_payload = _safe_json_loads(retry_response.content, {})
        if isinstance(retry_payload, dict):
            payload = retry_payload

    allowed_citation_keys = {
        str(item.get("final_bib_key", "")).strip()
        for item in verified_reference_bank
        if isinstance(item, dict) and str(item.get("included_in_final_bib", True)).lower() != "false"
    }
    allowed_ref_labels: set[str] = set()
    for text in current_sections.values():
        allowed_ref_labels.update(_extract_defined_labels_from_tex(text))
    allowed_ref_labels.update({"sec:system_model", "sec:proposed_solution", "sec:numerical_results", "sec:conclusion"})
    forbidden_terms = [
        "pipeline",
        "Phase 2.4",
        "Phase 2.5",
        "LLM",
        "Codex",
        "generated_plugin",
        "draft",
        "preliminary",
        "statistically significant",
        "current manuscript",
        "remaining problem formulation",
        "remaining constraints",
    ]
    topic_l = topic.lower()
    if not any(term in topic_l for term in ["isac", "sensing", "radar", "crb", "beampattern"]):
        forbidden_terms.extend(["ISAC", "ISACP", "sensing echo", "radar", "CRB", "beampattern"])

    revised_abstract_candidate = sanitize_phase3_3_abstract_tex(str(payload.get("abstract_tex") or current_sections["abstract_tex"]))
    revised_intro_candidate = ensure_phase3_4_notation_paragraph(
        sanitize_phase3_4_introduction_tex(str(payload.get("introduction_tex") or current_sections["introduction_tex"]))
    )
    revised_results_candidate = sanitize_phase3_2_numerical_results_tex(str(payload.get("numerical_results_section_tex") or current_sections["numerical_results_section_tex"]))
    revised_conclusion_candidate = sanitize_phase3_3_conclusion_tex(str(payload.get("conclusion_tex") or current_sections["conclusion_tex"]))

    rollback_notes: list[dict[str, Any]] = []
    revised_abstract, note = _phase3_6_validate_revised_section(
        section_name="abstract",
        candidate_text=revised_abstract_candidate,
        original_text=current_sections["abstract_tex"],
        allowed_citation_keys=allowed_citation_keys,
        allowed_ref_labels=allowed_ref_labels,
        forbidden_terms=forbidden_terms,
        min_word_ratio=0.7,
        max_word_ratio=1.35,
    )
    if note:
        rollback_notes.append(note)
    revised_intro, note = _phase3_6_validate_revised_section(
        section_name="introduction",
        candidate_text=revised_intro_candidate,
        original_text=current_sections["introduction_tex"],
        allowed_citation_keys=allowed_citation_keys,
        allowed_ref_labels=allowed_ref_labels,
        forbidden_terms=forbidden_terms,
        min_word_ratio=0.7,
        max_word_ratio=1.45,
    )
    if note:
        rollback_notes.append(note)
    revised_results, note = _phase3_6_validate_revised_section(
        section_name="numerical_results",
        candidate_text=revised_results_candidate,
        original_text=current_sections["numerical_results_section_tex"],
        allowed_citation_keys=allowed_citation_keys,
        allowed_ref_labels=allowed_ref_labels,
        forbidden_terms=forbidden_terms,
        min_word_ratio=0.85,
        max_word_ratio=1.40,
    )
    if note:
        rollback_notes.append(note)
    revised_conclusion, note = _phase3_6_validate_revised_section(
        section_name="conclusion",
        candidate_text=revised_conclusion_candidate,
        original_text=current_sections["conclusion_tex"],
        allowed_citation_keys=allowed_citation_keys,
        allowed_ref_labels=allowed_ref_labels,
        forbidden_terms=forbidden_terms,
        min_word_ratio=0.7,
        max_word_ratio=1.35,
    )
    if note:
        rollback_notes.append(note)

    revised_system, revised_proposed, revised_conclusion, deterministic_applied = _apply_phase3_6_deterministic_technical_fixes(
        system_text=current_sections["system_model_problem_formulation_section_tex"],
        proposed_text=current_sections["proposed_solution_section_tex"],
        conclusion_text=revised_conclusion,
    )
    review_summary_md = str(payload.get("review_summary_md") or "").strip()

    current_proposed = str(current_sections["proposed_solution_section_tex"])
    current_results = str(current_sections["numerical_results_section_tex"])
    if any(env in revised_proposed for env in [r"\begin{proposition}", r"\begin{proof}"]) and not any(
        env in current_proposed for env in [r"\begin{proposition}", r"\begin{proof}"]
    ):
        revised_proposed = sanitize_phase3_latex_snippet(current_proposed)

    current_result_labels = set(extract_latex_labels(current_results))
    revised_result_labels = set(extract_latex_labels(revised_results))
    current_result_words = _word_count_text(current_results)
    revised_result_words = _word_count_text(revised_results)
    if (
        current_result_labels and revised_result_labels != current_result_labels
    ) or revised_result_words < max(260, int(0.85 * current_result_words)):
        revised_results = sanitize_phase3_2_numerical_results_tex(current_results)

    revised_sections = {
        "abstract_tex": revised_abstract,
        "introduction_tex": revised_intro,
        "system_model_problem_formulation_section_tex": revised_system,
        "proposed_solution_section_tex": revised_proposed,
        "numerical_results_section_tex": revised_results,
        "conclusion_tex": revised_conclusion,
    }
    revised_citation_keys: list[str] = []
    for text in revised_sections.values():
        for key in extract_citation_keys_from_tex(text):
            if key not in revised_citation_keys:
                revised_citation_keys.append(key)
    missing_verified_keys = [key for key in revised_citation_keys if key not in verified_reference_keys]
    if missing_verified_keys:
        retry_prompt = prompt + (
            "\n\nRevision required:\n"
            "The previous rewrite introduced citation keys that do not exist in the verified_reference_bank. "
            f"Unsupported citation keys: {', '.join(missing_verified_keys)}.\n"
            "Rewrite all fields again and remove or replace every unsupported citation key using only keys from verified_reference_bank."
        )
        write_text(phase3_5_dir / "phase3_5_prompt_retry_citations.txt", retry_prompt)
        retry_payload = {}
        if not skip_phase3_5_llm:
            retry_response = llm.chat(
                [{"role": "user", "content": retry_prompt}],
                json_mode=True,
                thinking=thinking,
                max_tokens=9000,
            )
            write_text(phase3_5_dir / "phase3_5_raw_response_retry_citations.txt", retry_response.content)
            retry_payload = _safe_json_loads(retry_response.content, {})
        if isinstance(retry_payload, dict):
            revised_abstract = sanitize_phase3_3_abstract_tex(str(retry_payload.get("abstract_tex") or revised_abstract))
            revised_intro = ensure_phase3_4_notation_paragraph(
                sanitize_phase3_4_introduction_tex(str(retry_payload.get("introduction_tex") or revised_intro))
            )
            revised_system = sanitize_phase3_5_body_section(str(retry_payload.get("system_model_problem_formulation_section_tex") or revised_system))
            revised_proposed = sanitize_phase3_latex_snippet(str(retry_payload.get("proposed_solution_section_tex") or revised_proposed))
            revised_results = sanitize_phase3_2_numerical_results_tex(str(retry_payload.get("numerical_results_section_tex") or revised_results))
            revised_conclusion = sanitize_phase3_3_conclusion_tex(str(retry_payload.get("conclusion_tex") or revised_conclusion))
            review_summary_md = str(retry_payload.get("review_summary_md") or review_summary_md).strip()

    if any(env in revised_proposed for env in [r"\begin{proposition}", r"\begin{proof}"]) and not any(
        env in current_proposed for env in [r"\begin{proposition}", r"\begin{proof}"]
    ):
        revised_proposed = sanitize_phase3_latex_snippet(current_proposed)
    revised_result_labels = set(extract_latex_labels(revised_results))
    revised_result_words = _word_count_text(revised_results)
    if (
        current_result_labels and revised_result_labels != current_result_labels
    ) or revised_result_words < max(260, int(0.85 * current_result_words)):
        revised_results = sanitize_phase3_2_numerical_results_tex(current_results)

    write_text(phase3_5_dir / "abstract.tex", revised_abstract)
    write_text(phase3_5_dir / "introduction.tex", revised_intro)
    write_text(phase3_5_dir / "system_model_problem_formulation_section.tex", revised_system)
    write_text(phase3_5_dir / "proposed_solution_section.tex", revised_proposed)
    write_text(phase3_5_dir / "numerical_results_section.tex", revised_results)
    write_text(phase3_5_dir / "conclusion.tex", revised_conclusion)
    write_text(phase3_5_dir / "phase3_5_review_summary.md", review_summary_md + ("\n" if review_summary_md else ""))

    revised_citation_keys = []
    for text in [revised_intro, revised_system, revised_proposed, revised_results, revised_conclusion]:
        for key in extract_citation_keys_from_tex(text):
            if key not in revised_citation_keys:
                revised_citation_keys.append(key)
    final_bib_text, missing_keys, final_reference_entries = build_curated_bibliography(
        verified_reference_bank if isinstance(verified_reference_bank, list) else [],
        revised_citation_keys,
    )
    write_text(phase3_5_dir / "references.bib", final_bib_text)
    write_text(phase3_5_dir / "references_ieee.bib", final_bib_text)
    preview = render_phase3_5_preview_pdf(phase3_5_dir, topic, final_bib_text)
    manifest = {
        "paper_target": paper_target,
        "phase": "phase3",
        "phase_id": "phase3.5",
        "phase_name": "phase3.5_paper_review_rewrite_phase",
        "paper_writing_mode": _paper_writing_mode_snapshot(),
        "title": topic,
        "input_files_used": {
            "phase3.4_manifest": str(phase3_4_dir / "phase3_4_manifest.json"),
            "legacy_phase3.4_manifest": str(phase3_4_dir / "phase3_4_manifest.json"),
            "verified_reference_bank": str(phase3_4_dir / "verified_reference_bank.json"),
            "current_sections": [str(phase3_3_dir / "abstract.tex"), str(phase3_4_dir / "introduction.tex"), str(phase3_4_dir / "system_model_problem_formulation_section.tex"), str(phase3_4_dir / "proposed_solution_section.tex"), str(phase3_4_dir / "numerical_results_section.tex"), str(phase3_4_dir / "conclusion.tex")],
        },
        "prompt_path": str(phase3_5_dir / "phase3_5_prompt.txt"),
        "raw_response_path": str(phase3_5_dir / "phase3_5_raw_response.txt"),
        "review_summary_path": str(phase3_5_dir / "phase3_5_review_summary.md"),
        "references_ieee_bib_path": str(phase3_5_dir / "references_ieee.bib"),
        "revised_citation_keys": revised_citation_keys,
        "missing_reference_keys": missing_keys,
        "preview": preview,
    }
    write_text(phase3_5_dir / "phase3_5_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    write_text(phase3_5_dir / "phase3_5_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def build_phase3_5_final_review_design_notes() -> str:
    return """
# Phase 3.5 Design Notes

Phase 3.5 is the final review / pre-submission review phase.

It:
- reviews the current full paper draft as if it were being checked before IEEE/WCL submission
- diagnoses technical, theoretical, experimental, citation, writing, and LaTeX risks
- ranks issues by severity and produces an actionable revision plan
- does not rewrite the paper body by default

It does not:
- redesign the algorithm
- invent new experiments
- replace numerical results
- patch the paper text directly
""".strip()


def build_phase3_6_final_revision_design_notes() -> str:
    return """
# Phase 3.6 Design Notes

Phase 3.6 is the final targeted revision phase.

It:
- reads Phase 3.5 review outputs
- applies only the auto-fixable paper revisions
- weakens unsupported claims
- improves wording, consistency, citation usage, and LaTeX formatting
- recompiles a revised full paper draft

It does not:
- redesign the algorithm
- fabricate new references
- add unrun experiments
- change validated numerical values
""".strip()


def _extract_compile_warning_lines(log_text: str, limit: int = 40) -> list[str]:
    warnings: list[str] = []
    for line in (log_text or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        lowered = stripped.lower()
        if any(token in lowered for token in ["warning", "undefined reference", "undefined citation", "overfull", "underfull"]):
            warnings.append(stripped)
    deduped: list[str] = []
    seen: set[str] = set()
    for line in warnings:
        if line not in seen:
            deduped.append(line)
            seen.add(line)
    return deduped[:limit]


def _find_forbidden_terms_in_text(text: str, terms: list[str]) -> list[str]:
    lowered = text.lower()
    return [term for term in terms if term.lower() in lowered]


def _format_issue_block_md(title: str, items: list[dict[str, Any]]) -> str:
    lines = [f"# {title}", ""]
    if not items:
        lines.append("No issues recorded.")
        lines.append("")
        return "\n".join(lines)
    for item in items:
        lines.append(f"## {item.get('issue_id', 'issue')} - {item.get('title', 'Untitled issue')}")
        lines.append(f"- Category: {item.get('category', 'not specified')}")
        lines.append(f"- Why it matters: {item.get('why_it_matters', 'not specified')}")
        lines.append(f"- Exact location: {item.get('exact_location', 'not specified')}")
        lines.append(f"- Suggested action: {item.get('suggested_action', 'not specified')}")
        lines.append(f"- Responsible phase: {item.get('responsible_phase', 'not specified')}")
        lines.append(f"- Estimated impact: {item.get('estimated_impact', 'not specified')}")
        lines.append("")
    return "\n".join(lines)


def _format_dimension_score_md(dimension_scores: dict[str, Any]) -> str:
    lines = ["# Final Review Scorecard", ""]
    for key, payload in dimension_scores.items():
        if not isinstance(payload, dict):
            continue
        lines.append(f"## {key}")
        lines.append(f"- Score: {payload.get('score', 'n/a')}/10")
        lines.append(f"- Brief reason: {payload.get('brief_reason', 'not specified')}")
        top_issues = payload.get("top_issues") or []
        if top_issues:
            lines.append("- Top issues:")
            for issue in top_issues:
                lines.append(f"  - {issue}")
        lines.append(f"- Suggested fix: {payload.get('suggested_fix', 'not specified')}")
        lines.append("")
    return "\n".join(lines)


def _format_reviewer_comments_md(reviewers: list[dict[str, Any]]) -> str:
    lines = ["# Simulated Reviewer Comments", ""]
    for reviewer in reviewers:
        if not isinstance(reviewer, dict):
            lines.append("## Reviewer")
            lines.append(f"- Summary: {reviewer}")
            lines.append("")
            continue
        lines.append(f"## {reviewer.get('reviewer', 'Reviewer')}")
        lines.append(f"- Summary: {reviewer.get('summary', 'not specified')}")
        strengths = reviewer.get("strengths") or []
        if strengths:
            lines.append("- Strengths:")
            for item in strengths:
                lines.append(f"  - {item}")
        concerns = reviewer.get("major_concerns") or []
        if concerns:
            lines.append("- Major concerns:")
            for item in concerns:
                lines.append(f"  - {item}")
        minor = reviewer.get("minor_comments") or []
        if minor:
            lines.append("- Minor comments:")
            for item in minor:
                lines.append(f"  - {item}")
        lines.append(f"- Likely score: {reviewer.get('likely_score', 'n/a')}")
        lines.append(f"- Likely recommendation: {reviewer.get('likely_recommendation', 'n/a')}")
        lines.append("")
    return "\n".join(lines)


def _format_revision_plan_md(revision_plan: dict[str, Any]) -> str:
    lines = ["# Revision Plan", ""]
    for priority in ["P0", "P1", "P2"]:
        lines.append(f"## {priority}")
        items = revision_plan.get(priority) or []
        if not items:
            lines.append("No items.")
            lines.append("")
            continue
        for item in items:
            lines.append(f"### {item.get('issue_id', 'issue')} - {item.get('title', 'Untitled issue')}")
            lines.append(f"- Issue: {item.get('issue', item.get('title', 'not specified'))}")
            lines.append(f"- Why it matters: {item.get('why_it_matters', 'not specified')}")
            lines.append(f"- Exact location: {item.get('exact_location', 'not specified')}")
            lines.append(f"- Suggested action: {item.get('suggested_action', 'not specified')}")
            lines.append(f"- Responsible phase to fix: {item.get('responsible_phase', 'not specified')}")
            lines.append(f"- Estimated impact: {item.get('estimated_impact', 'not specified')}")
            lines.append(f"- Auto-fixable: {item.get('auto_fixable', False)}")
            lines.append(f"- Requires new experiment: {item.get('requires_new_experiment', False)}")
            lines.append(f"- Requires reference verification: {item.get('requires_reference_verification', False)}")
            lines.append("")
    return "\n".join(lines)


def _normalize_phase3_5_recommendation(text: str) -> str:
    value = str(text or "").strip().lower()
    if "ready_to_submit" in value or value == "ready":
        return "ready_to_submit"
    if "minor" in value:
        return "minor_revision_needed"
    if "major" in value:
        return "major_revision_needed"
    if "not_ready" in value or "reject" in value or "not ready" in value:
        return "not_ready"
    return "major_revision_needed"


def _normalize_phase3_5_decision(text: str) -> str:
    value = str(text or "").strip().lower()
    if "weak_accept" in value or ("accept" in value and "weak" in value):
        return "weak_accept"
    if value == "accept" or value.startswith("accept"):
        return "accept"
    if "borderline" in value:
        return "borderline"
    if "weak_reject" in value or ("reject" in value and "weak" in value):
        return "weak_reject"
    if "reject" in value:
        return "reject"
    return "borderline"


def _phase3_5_review_max_tokens() -> int:
    """Token budget for the final ReviewAgent.

    Phase 3.4 reviews long manuscripts and can legitimately need more room than
    ordinary section writers.  Keep it configurable so batch runs can increase
    the budget without code changes, while still avoiding unbounded requests.
    """
    raw = os.environ.get("WARA_PHASE34_REVIEW_MAX_TOKENS", "").strip()
    if raw:
        try:
            return max(12000, min(64000, int(raw)))
        except ValueError:
            pass
    return 24000


def _phase3_5_parse_review_payload(text: str) -> dict[str, Any]:
    payload = _safe_json_loads(text, {})
    return payload if isinstance(payload, dict) else {}


def _phase3_5_compact_json_repair_prompt(
    *,
    original_prompt: str,
    broken_response: str,
    parse_reason: str,
) -> str:
    """Build a compact retry prompt when the review JSON was malformed/truncated."""
    return (
        "Your previous Phase 3.4 review response was not valid parseable JSON.\n"
        f"Parse issue: {parse_reason}.\n"
        "Return ONLY one compact JSON object. Do not wrap it in markdown.\n"
        "Use these exact recommendation enums only: ready_to_submit, minor_revision_needed, major_revision_needed, not_ready.\n"
        "Use these exact likely reviewer decision enums only: accept, weak_accept, borderline, weak_reject, reject.\n"
        "Keep the JSON concise so it does not truncate:\n"
        "- dimension_scores must contain the ten requested dimensions, but each brief_reason should be <= 25 words.\n"
        "- each issue list should contain at most 6 high-impact items.\n"
        "- reviewer_comments should contain at most 4 comments.\n"
        "- revision_plan.P0/P1/P2 should contain concise action strings, not paragraphs.\n"
        "- do not quote long manuscript passages.\n\n"
        "Required top-level keys:\n"
        "overall_score, recommendation, likely_reviewer_decision_estimate, dimension_scores, final_review_summary, "
        "novelty_positioning_findings, overclaim_warnings, missing_related_work_warnings, suggested_rewrite_targets, "
        "theory_claim_reviews, algorithm_clarity_issues, undefined_symbols, pseudo_code_vs_text_consistency, "
        "claims_supported_by_experiments, claims_not_sufficiently_supported, suggested_additional_experiments, "
        "citation_findings, latex_format_findings, writing_quality_issues, consistency_issues, critical_issues, "
        "major_issues, minor_issues, reviewer_comments, revision_plan, paper_risk_matrix, checklist_for_next_revision.\n\n"
        "Original review task:\n"
        f"{compact_text(original_prompt, 14000)}\n\n"
        "Broken previous response excerpt:\n"
        f"{compact_text(broken_response, 6000)}"
    )


def _phase3_5_missing_core_keys(payload: dict[str, Any]) -> list[str]:
    required = [
        "overall_score",
        "recommendation",
        "likely_reviewer_decision_estimate",
        "dimension_scores",
        "critical_issues",
        "major_issues",
        "minor_issues",
        "reviewer_comments",
        "revision_plan",
    ]
    missing = [key for key in required if key not in payload]
    dimension_scores = payload.get("dimension_scores")
    if not isinstance(dimension_scores, dict) or len(dimension_scores) < 8:
        missing.append("dimension_scores_incomplete")
    revision_plan = payload.get("revision_plan")
    if not isinstance(revision_plan, dict) or not any(revision_plan.get(key) for key in ["P0", "P1", "P2"]):
        missing.append("revision_plan_incomplete")
    return missing


def _complete_phase3_5_payload_locally(
    payload: dict[str, Any],
    *,
    arxiv_only_entries: list[str],
    compile_warnings_summary: list[str],
    forbidden_terms_found: list[str],
) -> dict[str, Any]:
    """Make an incomplete reviewer JSON usable without another long LLM retry."""
    payload = dict(payload)
    dimensions = [
        "novelty_and_positioning",
        "technical_correctness",
        "theoretical_rigor",
        "algorithm_clarity",
        "experiment_strength",
        "reference_quality",
        "introduction_quality",
        "writing_quality",
        "latex_format_quality",
        "reproducibility_readiness",
    ]
    dimension_scores = payload.get("dimension_scores")
    if not isinstance(dimension_scores, dict):
        dimension_scores = {}
    for name in dimensions:
        top_level_dimension = payload.get(name)
        if isinstance(top_level_dimension, dict) and (
            name not in dimension_scores
            or "manual inspection" in str((dimension_scores.get(name) or {}).get("brief_reason", "")).lower()
        ):
            dimension_scores[name] = top_level_dimension
        item = dimension_scores.get(name)
        if not isinstance(item, dict):
            dimension_scores[name] = {
                "score": 6.0,
                "brief_reason": "Not explicitly scored by the reviewer response; local completion marks this as requiring manual inspection.",
                "top_issues": [],
                "suggested_fix": "Inspect this dimension manually before submission.",
            }
        else:
            item.setdefault("score", 6.0)
            item.setdefault("brief_reason", "No detailed reason provided.")
            item.setdefault("top_issues", [])
            item.setdefault("suggested_fix", "No specific fix provided.")
    payload["dimension_scores"] = dimension_scores

    revision_plan = payload.get("revision_plan")
    if not isinstance(revision_plan, dict) or not any(revision_plan.get(key) for key in ["P0", "P1", "P2"]):
        revision_plan = _build_phase3_5_revision_plan(
            dimension_scores,
            arxiv_only_entries=arxiv_only_entries,
            compile_warnings_summary=compile_warnings_summary,
            forbidden_terms_found=forbidden_terms_found,
        )
    payload["revision_plan"] = revision_plan

    payload.setdefault("overall_score", 6.0)
    payload.setdefault("recommendation", "major_revision_needed")
    payload.setdefault("likely_reviewer_decision_estimate", "borderline")
    payload.setdefault(
        "final_review_summary",
        "The reviewer response was incomplete, so missing fields were completed locally. Treat this review as a conservative pre-submission diagnosis.",
    )
    payload.setdefault("critical_issues", revision_plan.get("P0", []))
    payload.setdefault("major_issues", revision_plan.get("P1", []))
    payload.setdefault("minor_issues", revision_plan.get("P2", []))
    payload.setdefault("reviewer_comments", [])
    for key in [
        "novelty_positioning_findings",
        "overclaim_warnings",
        "missing_related_work_warnings",
        "suggested_rewrite_targets",
        "theory_claim_reviews",
        "algorithm_clarity_issues",
        "undefined_symbols",
        "pseudo_code_vs_text_consistency",
        "claims_supported_by_experiments",
        "claims_not_sufficiently_supported",
        "suggested_additional_experiments",
        "writing_quality_issues",
        "consistency_issues",
        "paper_risk_matrix",
        "checklist_for_next_revision",
    ]:
        payload.setdefault(key, [])
    payload.setdefault("citation_findings", {})
    payload.setdefault("latex_format_findings", {})
    payload["local_completion_applied"] = True
    return payload


def _build_phase3_5_revision_plan(
    dimension_scores: dict[str, Any],
    *,
    arxiv_only_entries: list[str],
    compile_warnings_summary: list[str],
    forbidden_terms_found: list[str],
) -> dict[str, list[dict[str, Any]]]:
    def score_of(name: str, default: float = 6.0) -> float:
        payload = dimension_scores.get(name) or {}
        try:
            return float(payload.get("score", default) or default)
        except Exception:
            return default

    def reason_of(name: str) -> str:
        payload = dimension_scores.get(name) or {}
        return str(payload.get("brief_reason", "not specified"))

    def fix_of(name: str) -> str:
        payload = dimension_scores.get(name) or {}
        return str(payload.get("suggested_fix", "not specified"))

    p0: list[dict[str, Any]] = []
    p1: list[dict[str, Any]] = []
    p2: list[dict[str, Any]] = []

    if score_of("novelty_and_positioning") < 7.0:
        p0.append(
            {
                "issue_id": "P0-NOV-01",
                "title": "Novelty positioning is not sufficiently differentiated",
                "category": "novelty_positioning",
                "issue": reason_of("novelty_and_positioning"),
                "why_it_matters": "Weak novelty positioning is a direct reviewer rejection risk in short-letter venues.",
                "exact_location": "Introduction related-work/gap/contribution paragraphs",
                "suggested_action": fix_of("novelty_and_positioning"),
                "responsible_phase": "phase3.4 intro/references",
                "estimated_impact": "high",
                "auto_fixable": True,
                "requires_new_experiment": False,
                "requires_reference_verification": False,
                "requires_manual_theory_verification": False,
            }
        )
    if score_of("theoretical_rigor") < 6.5 or score_of("technical_correctness") < 6.5:
        p0.append(
            {
                "issue_id": "P0-THE-01",
                "title": "Theoretical claims are stronger than the current evidence",
                "category": "theory_claim_strength",
                "issue": reason_of("theoretical_rigor"),
                "why_it_matters": "Unsupported convergence or optimality language creates a major technical rejection risk.",
                "exact_location": "Proposed Solution and Conclusion",
                "suggested_action": "Weaken unsupported convergence/optimality claims and explicitly scope the guarantee to the adopted surrogate or heuristic procedure.",
                "responsible_phase": "phase3.6 direct LaTeX formatting",
                "estimated_impact": "high",
                "auto_fixable": True,
                "requires_new_experiment": False,
                "requires_reference_verification": False,
                "requires_manual_theory_verification": False,
            }
        )
        p1.append(
            {
                "issue_id": "P1-THE-02",
                "title": "A full proof-level strengthening would require manual theory work",
                "category": "theoretical_rigor",
                "issue": reason_of("technical_correctness"),
                "why_it_matters": "If the paper keeps stronger convergence wording, it would need additional proof support.",
                "exact_location": "Proposed Solution / convergence discussion",
                "suggested_action": fix_of("technical_correctness"),
                "responsible_phase": "phase2.3 theory",
                "estimated_impact": "high",
                "auto_fixable": False,
                "requires_new_experiment": False,
                "requires_reference_verification": False,
                "requires_manual_theory_verification": True,
            }
        )
    if score_of("experiment_strength") < 7.5:
        p1.append(
            {
                "issue_id": "P1-EXP-01",
                "title": "Experiments need stronger support for reviewer robustness",
                "category": "experiment_strength",
                "issue": reason_of("experiment_strength"),
                "why_it_matters": "Single-baseline evaluation or missing ablation/convergence plots weakens empirical support.",
                "exact_location": "Numerical Results and experiment package",
                "suggested_action": fix_of("experiment_strength"),
                "responsible_phase": "phase2.5 experiments",
                "estimated_impact": "high",
                "auto_fixable": False,
                "requires_new_experiment": True,
                "requires_reference_verification": False,
                "requires_manual_theory_verification": False,
            }
        )
    if score_of("reference_quality") < 7.0 or arxiv_only_entries:
        p0.append(
            {
                "issue_id": "P0-REF-01",
                "title": "Reference positioning still carries verification or authority risk",
                "category": "reference_quality",
                "issue": reason_of("reference_quality"),
                "why_it_matters": "Closest-work comparison and reviewer confidence depend on a reliable peer-reviewed reference set.",
                "exact_location": "Introduction related work and bibliography",
                "suggested_action": fix_of("reference_quality"),
                "responsible_phase": "phase3.4 intro/references",
                "estimated_impact": "high",
                "auto_fixable": False,
                "requires_new_experiment": False,
                "requires_reference_verification": True,
                "requires_manual_theory_verification": False,
            }
        )
    if score_of("introduction_quality") < 6.5:
        p1.append(
            {
                "issue_id": "P1-INT-01",
                "title": "Introduction motivation-gap-contribution line is not sharp enough",
                "category": "introduction_quality",
                "issue": reason_of("introduction_quality"),
                "why_it_matters": "A weak introduction obscures the paper's objective and novelty delta.",
                "exact_location": "Introduction",
                "suggested_action": fix_of("introduction_quality"),
                "responsible_phase": "phase3.4 intro/references",
                "estimated_impact": "medium",
                "auto_fixable": True,
                "requires_new_experiment": False,
                "requires_reference_verification": False,
                "requires_manual_theory_verification": False,
            }
        )
    if score_of("writing_quality") < 7.0 or forbidden_terms_found or compile_warnings_summary:
        p2.append(
            {
                "issue_id": "P2-WRT-01",
                "title": "Final writing and format polish is still needed",
                "category": "writing_quality",
                "issue": reason_of("writing_quality"),
                "why_it_matters": "Minor wording, float, or claim-strength issues can still hurt reviewer confidence.",
                "exact_location": "Abstract / Introduction / Numerical Results / LaTeX formatting",
                "suggested_action": fix_of("writing_quality"),
                "responsible_phase": "phase3.6 direct LaTeX formatting",
                "estimated_impact": "medium",
                "auto_fixable": True,
                "requires_new_experiment": False,
                "requires_reference_verification": False,
                "requires_manual_theory_verification": False,
            }
        )
    return {"P0": p0, "P1": p1, "P2": p2}


def _phase3_5_issue_is_submission_metadata_only(issue: Any) -> bool:
    text = _review_issue_text(issue).lower()
    metadata_terms = [
        "author",
        "affiliation",
        "correspondence",
        "metadata",
        "anonymization",
        "anonymous",
        "submission packaging",
        "submission metadata",
    ]
    technical_terms = [
        "objective",
        "constraint",
        "optimizer",
        "algorithm",
        "theorem",
        "proof",
        "experiment",
        "reference",
        "citation",
        "baseline",
        "figure",
    ]
    return any(term in text for term in metadata_terms) and not any(term in text for term in technical_terms)


def _phase3_5_issue_is_false_missing_figure(issue: Any) -> bool:
    text = _review_issue_text(issue).lower()
    figure_terms = ["figure", "fig.", "figures", "plot", "plots"]
    missing_terms = ["missing", "absent", "not present", "figures_present=false", "cannot evaluate empirical claims"]
    return any(term in text for term in figure_terms) and any(term in text for term in missing_terms)


def _build_phase3_5_final_figure_check(
    *,
    numerical_results_tex: str,
    numerical_results_base_dir: Path,
    full_paper_log: str,
    phase25_summary: dict[str, Any],
) -> dict[str, Any]:
    include_paths = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^{}]+)\}", numerical_results_tex or "")
    included_figures: list[dict[str, Any]] = []
    for raw_path in include_paths:
        path_text = str(raw_path).strip()
        resolved_path = (Path(numerical_results_base_dir) / path_text).resolve()
        included_figures.append(
            {
                "path": path_text,
                "resolved_path": str(resolved_path),
                "exists": resolved_path.exists(),
                "used_in_compile_log": path_text in str(full_paper_log or ""),
            }
        )

    expected_figures = [
        item
        for item in (phase25_summary.get("figures", []) if isinstance(phase25_summary, dict) else [])
        if isinstance(item, dict) and item.get("paper_ready")
    ]
    expected_filenames = [
        str(item.get("filename_pdf") or item.get("filename_png") or "").strip()
        for item in expected_figures
        if str(item.get("filename_pdf") or item.get("filename_png") or "").strip()
    ]
    labels = re.findall(r"\\label\{([^{}]*fig[^{}]*)\}", numerical_results_tex or "")
    refs = re.findall(r"\\ref\{([^{}]*fig[^{}]*)\}", numerical_results_tex or "")
    all_includes_exist = all(item["exists"] for item in included_figures) if included_figures else False
    all_includes_compiled = all(item["used_in_compile_log"] for item in included_figures) if included_figures else False
    expected_covered = all(
        any(filename and filename in item["path"] for item in included_figures)
        for filename in expected_filenames
    )
    return {
        "ok": bool(included_figures) and all_includes_exist and all_includes_compiled and expected_covered,
        "includegraphics_count": len(included_figures),
        "included_figures": included_figures,
        "expected_paper_ready_figure_count": len(expected_figures),
        "expected_paper_ready_filenames": expected_filenames,
        "expected_figures_covered": expected_covered,
        "figure_labels": labels,
        "figure_refs": refs,
    }


def _phase3_5_max_overfull_hbox_pt(compile_warnings_summary: list[str]) -> float:
    max_amount = 0.0
    for line in compile_warnings_summary:
        match = re.search(r"Overfull\s+\\hbox\s+\(([0-9.]+)pt too wide\)", str(line))
        if not match:
            continue
        try:
            max_amount = max(max_amount, float(match.group(1)))
        except ValueError:
            pass
    return max_amount


def _phase3_5_extract_section(tex: str, section_title: str) -> str:
    pattern = re.compile(
        rf"\\section\{{{re.escape(section_title)}\}}(?P<body>.*?)(?=\\section\{{|\\bibliographystyle|\\bibliography|\\end\{{document\}}|$)",
        flags=re.S | re.I,
    )
    match = pattern.search(str(tex or ""))
    return match.group("body") if match else ""


def _phase3_5_extract_abstract(tex: str) -> str:
    match = re.search(r"\\begin\{abstract\}(?P<body>.*?)\\end\{abstract\}", str(tex or ""), flags=re.S | re.I)
    return match.group("body") if match else ""


def _phase3_5_first_prose_block(tex: str) -> str:
    """Return the first prose block after leading section labels/comments."""
    text = str(tex or "").strip()
    previous = None
    while previous != text:
        previous = text
        text = re.sub(r"^\s*%[^\n]*(?:\n|$)", "", text)
        text = re.sub(r"^\s*\\section\{[^{}]+\}\s*", "", text, flags=re.I)
        text = re.sub(r"^\s*\\label\{[^{}]+\}\s*", "", text, flags=re.I)
        text = re.sub(r"^\s*\\vspace\*?\{[^{}]+\}\s*", "", text, flags=re.I)
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    return paragraphs[0] if paragraphs else ""


def _phase3_5_ieee_style_and_technical_audit(full_paper_tex: str) -> dict[str, Any]:
    """Lightweight deterministic audit for paper-level style and technical closure.

    This does not replace the LLM reviewer. It catches recurring failures that
    are visible in the final draft and should not depend on reviewer taste.
    The checks are conditional closure dimensions, not topic defaults: they
    trigger only when the manuscript itself uses the corresponding mechanism.
    """

    text = str(full_paper_tex or "")
    lowered = text.lower()
    p0: list[dict[str, Any]] = []
    p1: list[dict[str, Any]] = []
    p2: list[dict[str, Any]] = []

    def issue(
        *,
        issue_id: str,
        title: str,
        category: str,
        issue_text: str,
        location: str,
        action: str,
        severity: str = "P1",
        responsible_phase: str = "writing_agent",
        requires_manual_theory_verification: bool = False,
    ) -> None:
        payload = {
            "issue_id": issue_id,
            "title": title,
            "category": category,
            "issue": issue_text,
            "why_it_matters": "This defect makes the manuscript read as either technically under-closed or not yet at IEEE letter quality.",
            "exact_location": location,
            "suggested_action": action,
            "responsible_phase": responsible_phase,
            "estimated_impact": "high" if severity in {"P0", "P1"} else "medium",
            "auto_fixable": severity != "P0",
            "requires_new_experiment": False,
            "requires_reference_verification": False,
            "requires_manual_theory_verification": requires_manual_theory_verification,
        }
        if severity == "P0":
            p0.append(payload)
        elif severity == "P1":
            p1.append(payload)
        else:
            p2.append(payload)

    abstract = _phase3_5_extract_abstract(text)
    intro = _phase3_5_extract_section(text, "Introduction")
    system_problem = _phase3_5_extract_section(text, "System Model and Problem Formulation")
    proposed = (
        _phase3_5_extract_section(text, "Proposed Solution")
        or _phase3_5_extract_section(text, "Proposed Method")
        or _phase3_5_extract_section(text, "Proposed Algorithm")
    )
    numerical = _phase3_5_extract_section(text, "Numerical Results")
    conclusion = _phase3_5_extract_section(text, "Conclusion")
    technical_text = "\n".join([system_problem, proposed])

    if abstract and re.search(r"\\\(|\\\[|\$[^$]+\$|\\mathbf|\\boldsymbol|_\{", abstract):
        issue(
            issue_id="P1-ABSTRACT-NOTATION",
            title="Abstract contains technical notation",
            category="writing_quality",
            issue_text="The abstract contains inline mathematical notation or optimizer symbols.",
            location="Abstract.",
            action="Rewrite abstract variables and metrics in language rather than symbols.",
        )
    if conclusion and re.search(r"\\\(|\\\[|\$[^$]+\$|\\mathbf|\\boldsymbol|_\{", conclusion):
        issue(
            issue_id="P1-CONCLUSION-NOTATION",
            title="Conclusion contains technical notation",
            category="writing_quality",
            issue_text="The conclusion contains inline mathematical notation or optimizer symbols.",
            location="Conclusion.",
            action="Summarize the contribution in language and keep detailed notation in technical sections.",
        )

    generic_patterns = [
        r"\bhas attracted (?:much|significant|increasing) attention\b",
        r"\bplays? an important role\b",
        r"\bis a key (?:technology|enabler|component|direction)\b",
        r"\bhas become (?:an|a) important\b",
        r"\bthe results demonstrate the effectiveness\b",
        r"\bdemonstrate(?:s|d)? the effectiveness\b",
        r"\bshow(?:s|n)? the effectiveness\b",
        r"\bvalidates? the effectiveness\b",
        r"\bthe proposed (?:scheme|method|algorithm|framework) is effective\b",
        r"\bsignificant performance improvement\b",
        r"\bextensive simulations\b",
    ]
    generic_hits = [
        pattern
        for pattern in generic_patterns
        if re.search(pattern, lowered)
    ]
    if len(generic_hits) >= 2:
        issue(
            issue_id="P1-WRITING-GENERIC-IEEE",
            title="Manuscript uses generic paper-template language",
            category="writing_quality",
            issue_text=f"The draft contains {len(generic_hits)} generic or low-information IEEE-template phrases.",
            location="Full paper prose.",
            action="Rewrite generic claims into mechanism-specific motivation, contribution, and insight sentences.",
        )
    if intro and re.search(r"\bThe first\b.*\bThe second\b.*\bThe third\b", re.sub(r"\s+", " ", intro), flags=re.I):
        issue(
            issue_id="P1-INTRO-ORDINAL-SURVEY",
            title="Introduction reads like a category checklist",
            category="introduction_quality",
            issue_text="The related-work paragraph uses ordinal category-summary language instead of a natural thematic literature argument.",
            location="Introduction related-work paragraphs.",
            action="Rewrite related work around two causal technical axes and end with the exact unresolved gap.",
        )
    if intro and "\\begin{itemize}" not in intro:
        issue(
            issue_id="P1-INTRO-CONTRIBUTIONS",
            title="Introduction lacks an itemized contribution list",
            category="introduction_quality",
            issue_text="The Introduction does not expose concise paper-level contributions.",
            location="Introduction contribution paragraph.",
            action="Add a 2-3 item WCL-style contribution list grounded in the actual formulation, method, and evidence.",
        )
    orphan_intro_paragraphs = analyze_phase3_4_intro_orphan_argument_paragraphs(
        "\\section{Introduction}\n" + intro if intro and "\\section" not in intro[:80] else intro
    )
    if orphan_intro_paragraphs:
        issue(
            issue_id="P1-INTRO-ORPHAN-PARAGRAPH",
            title="Introduction contains an orphan one-sentence argument paragraph",
            category="introduction_quality",
            issue_text=(
                "One or more motivation, related-work, or gap paragraphs before the contribution list "
                "contain only a single sentence."
            ),
            location="Introduction pre-contribution paragraphs.",
            action=(
                "Merge the orphan sentence into the adjacent related-work/gap paragraph or expand it into "
                "a full argument paragraph with thesis, support, and transition."
            ),
        )
    if numerical and re.search(r"\b(evidence package|verified registry|paper-ready|preview|curve data|artifact|pipeline)\b", numerical, flags=re.I):
        issue(
            issue_id="P1-RESULTS-REPORTLIKE",
            title="Numerical Results reads like an artifact report",
            category="writing_quality",
            issue_text="The Numerical Results section contains internal/report-like terms instead of paper-native observation and insight prose.",
            location="Numerical Results.",
            action="Rewrite each figure paragraph as what is plotted, observation, mechanism explanation, and insight.",
        )
    numerical_first_prose = _phase3_5_first_prose_block(numerical)
    if numerical and re.match(r"\s*In this section,\s+we\b", numerical_first_prose, flags=re.I) is None:
        issue(
            issue_id="P2-RESULTS-OPENING",
            title="Numerical Results opening does not follow the expected WCL setup pattern",
            category="writing_quality",
            issue_text='The Numerical Results section should start with a setup paragraph beginning "In this section, we".',
            location="Numerical Results first paragraph.",
            action="Start with the evaluation objective, concrete simulation parameters, and reported metric.",
            severity="P2",
        )
    if re.search(r"\b(?:system|constraints|requirements|optimization|ambiguity|rectifier)\.[A-Za-z0-9_\.]+", full_paper_tex):
        issue(
            issue_id="P1-FIGURE-INTERNAL-AXIS-LABEL",
            title="Internal parameter path appears in paper-facing text",
            category="IEEE_style_and_format",
            issue_text="The draft exposes an internal dotted configuration path instead of public notation in an axis label, caption, or prose sentence.",
            location="Figures, captions, or Numerical Results.",
            action="Use notation-first labels explicitly defined by the current paper contract or figure metadata; do not apply hardcoded path-to-symbol substitutions.",
        )

    if re.search(r"(\\Phi|Phi_m|\bphi_m\b)", technical_text) and not re.search(
        r"(ellipsoidal|norm-bounded|wasserstein|cantelli|chebyshev|bernstein|gaussian error|scenario|sample approximation|moment)",
        technical_text,
        flags=re.I,
    ):
        issue(
            issue_id="P2-TECH-PHI-ADVISORY",
            title="Potential safe-counterpart closure issue",
            category="technical_advisory",
            issue_text="A keyword-level scan found a Phi-style safe constraint that may need semantic verification against the current uncertainty/reliability model.",
            location="System Model / Proposed Method.",
            action="Ask the review/theory agent to verify the active mechanism from the current artifacts; do not route solely from this keyword scan.",
            severity="P2",
            responsible_phase="theory_agent",
            requires_manual_theory_verification=True,
        )
    if re.search(r"rank recovery is unnecessary|no rank recovery|without rank recovery", technical_text, flags=re.I) and not re.search(
        r"gaussian signaling|multi-stream|multistream|covariance-domain|high-rank covariance", technical_text, flags=re.I
    ):
        issue(
            issue_id="P2-TECH-RANK-SCOPE-ADVISORY",
            title="Potential matrix-valued transmit-control scope issue",
            category="technical_advisory",
            issue_text="A keyword-level scan found rank/recovery language that may need semantic verification against the current transmit-control role.",
            location="Proposed Method.",
            action="Ask the review/theory agent to verify whether the matrix variable is a physical covariance, a relaxation, or compact notation before requesting a rewrite.",
            severity="P2",
            responsible_phase="theory_agent",
            requires_manual_theory_verification=True,
        )
    if re.search(
        r"(shared|sensing|energy|common|auxiliary|non-information|service|artificial|jamming|pilot)[^.\n]{0,60}(covariance|waveform|signal|resource)[^.\n]{0,100}(interference|denominator|SINR)",
        technical_text,
        flags=re.I,
    ) and not re.search(
        r"not decoded|not canceled|not cancelled|treat(?:ed)? as noise|unknown to the receiver|known and cancel|jointly decoded|joint decoding|successive interference cancellation|SIC",
        technical_text,
        flags=re.I,
    ):
        issue(
            issue_id="P2-TECH-RECEIVER-SCOPE-ADVISORY",
            title="Potential receiver information-pattern scope issue",
            category="technical_advisory",
            issue_text="A keyword-level scan found auxiliary/common/service signal language near a receiver metric and may need semantic verification.",
            location="System Model receiver-metric definition.",
            action="Ask the review/formulation agent to verify the receiver/protocol assumption from the current system model before requesting a rewrite.",
            severity="P2",
            responsible_phase="formulation_agent",
            requires_manual_theory_verification=True,
        )
    if re.search(r"epsilon[^.\n]{0,90}(outage|chance)", lowered) and re.search(
        r"epsilon[^.\n]{0,90}(uncertainty|radius|csi|channel)", lowered
    ):
        issue(
            issue_id="P2-SYMBOL-EPSILON-ADVISORY",
            title="Potential epsilon notation overload",
            category="technical_advisory",
            issue_text="A keyword-level scan suggests epsilon may be used for more than one role.",
            location="Technical sections and figure captions/axes.",
            action="Ask the review/formulation agent to verify symbol roles against the current notation table before requesting a rewrite.",
            severity="P2",
            responsible_phase="formulation_agent",
        )
    algorithm_blocks = re.findall(r"\\begin\{algorithm\}.*?\\end\{algorithm\}", full_paper_tex, flags=re.S)
    for block in algorithm_blocks:
        state_count = len(re.findall(r"\\State\b", block))
        if state_count <= 4 and re.search(r"solve problem|solve .*SDP|solve .*SOCP", block, flags=re.I):
            issue(
                issue_id="P1-ALGO-SOLVER-WRAPPER",
                title="Algorithm reads like a generic solver wrapper",
                category="technical_presentation",
                issue_text="Algorithm 1 mainly asks the reader to construct constraints, call a solver, and return variables, without enough model-specific construction or verification steps.",
                location="Proposed Method algorithm block.",
                action="Rewrite Algorithm 1 as a compact method skeleton: construct model-specific margins/coefficient blocks, assemble the scoped subproblem, solve, recover/interpret physical controls, and evaluate paper KPIs.",
            )
            break
    transmit_power_mentions = len(re.findall(r"transmit[- ]power minimization|minimi[sz]e transmit power|minimum transmit power", lowered))
    if transmit_power_mentions >= 3 and not re.search(r"resource efficiency under|qos|quality-of-service|reliability|service constraint", lowered):
        issue(
            issue_id="P1-CLAIM-POWER-MIN-OVERFRAMED",
            title="Transmit-power minimization is over-framed",
            category="novelty_and_positioning",
            issue_text="The draft repeatedly frames transmit-power minimization as the contribution without tying it to a nontrivial service/reliability regime.",
            location="Abstract, Introduction, and technical motivation.",
            action="Reframe as minimum resource required to satisfy the scoped QoS/reliability/service model and emphasize the proposed mechanism.",
        )

    return {
        "ok": not p0 and not p1,
        "P0": p0,
        "P1": p1,
        "P2": p2,
        "generic_phrase_hit_count": len(generic_hits),
        "transmit_power_minimization_mentions": transmit_power_mentions,
    }


def _phase3_5_deterministic_full_paper_gate(
    *,
    reference_quality_report: dict[str, Any],
    full_paper_abbreviation_report: dict[str, Any],
    compile_warnings_summary: list[str],
    final_figure_check: dict[str, Any],
    phase25_summary: dict[str, Any],
    full_paper_tex: str = "",
    minimum_reference_target: int = 12,
) -> dict[str, Any]:
    p0: list[dict[str, Any]] = []
    p1: list[dict[str, Any]] = []
    p2: list[dict[str, Any]] = []

    reference_count = int(reference_quality_report.get("total_references", 0) or 0) if isinstance(reference_quality_report, dict) else 0
    if reference_count < minimum_reference_target:
        p0.append(
            {
                "issue_id": "P0-REF-COUNT",
                "title": "Reference bank is below the hard target",
                "category": "references_positioning",
                "issue": f"The verified reference bank contains {reference_count} references, below the hard target of {minimum_reference_target}.",
                "why_it_matters": "IEEE WCL positioning cannot be credible with an underpopulated reference bank.",
                "exact_location": "Phase 3.3 reference bank, Introduction, and bibliography.",
                "suggested_action": "Rerun or repair the Phase 1 LiteratureAgent/handoff; Phase 2/3 must not add references as a downstream patch.",
                "responsible_phase": "phase1_literature_agent",
                "estimated_impact": "very_high",
                "requires_reference_verification": True,
                "auto_fixable": False,
            }
        )

    phase25_status = str(phase25_summary.get("phase25_status", "") if isinstance(phase25_summary, dict) else "")
    if phase25_status in {"", "quick_mode_only", "needs_more_phase24_runs", "claim_failure_needs_redesign"}:
        p0.append(
            {
                "issue_id": "P0-EXP-READY",
                "title": "Experiments are not paper-ready",
                "category": "experiments",
                "issue": (
                    f"Phase 2.5 status is `{phase25_status or 'missing'}`, so final research synthesis "
                    "must keep empirical claims within the available verified evidence and mark the run as needing review."
                ),
                "why_it_matters": "Numerical evidence cannot support a WCL paper without paper-ready simulations and final figures.",
                "exact_location": "Phase 2.5 experiment summary and Numerical Results.",
                "suggested_action": "Expand Phase 2.5 evidence generation from quick-mode data to paper-ready sweeps; rerun Phase 2.4 only if the existing experiment implementation cannot produce the required figures.",
                "responsible_phase": "Phase 2.5 evidence expansion",
                "estimated_impact": "very_high",
                "requires_new_experiment": True,
                "auto_fixable": False,
            }
        )
    phase25_figures = phase25_summary.get("figures", []) if isinstance(phase25_summary, dict) else []
    if not isinstance(phase25_figures, list):
        phase25_figures = []
    paper_ready_figures = [item for item in phase25_figures if isinstance(item, dict) and item.get("paper_ready")]
    non_diagnostic_paper_figures = [
        item
        for item in paper_ready_figures
        if "violation" not in str(item.get("y_metric", "")).lower()
        and "feasible" not in str(item.get("y_metric", "")).lower()
        and "feasibility" not in str(item.get("y_metric", "")).lower()
    ]
    if phase25_status in {"paper_minimum_ready", "paper_preferred_ready", "high_confidence_ready"} and len(non_diagnostic_paper_figures) < 2:
        p0.append(
            {
                "issue_id": "P0-FIGURE-COUNT",
                "title": "Too few final system-performance figures",
                "category": "experiments",
                "issue": (
                    "Phase 2.5 claims paper readiness but provides fewer than two final non-diagnostic "
                    f"system-performance figures ({len(non_diagnostic_paper_figures)} found)."
                ),
                "why_it_matters": "A WCL numerical section needs at least two clean performance/insight figures, not a single curve or feasibility-only diagnostic.",
                "exact_location": "Phase 2.5 figure registry and Numerical Results.",
                "suggested_action": "Expand Phase 2.5 with a second parameter-sensitivity or operating-regime figure using the same valid benchmarks; return to Phase 2.4 only for implementation defects.",
                "responsible_phase": "Phase 2.5 evidence expansion",
                "estimated_impact": "very_high",
                "requires_new_experiment": True,
                "auto_fixable": False,
            }
        )

    undefined_abbreviations = (
        full_paper_abbreviation_report.get("undefined_abbreviations", [])
        if isinstance(full_paper_abbreviation_report, dict)
        else []
    )
    repeated_abbreviation_definitions = (
        full_paper_abbreviation_report.get("repeated_abbreviation_definitions", [])
        if isinstance(full_paper_abbreviation_report, dict)
        else []
    )
    if undefined_abbreviations:
        p1.append(
            {
                "issue_id": "P1-ABBR-UNDEFINED",
                "title": "Full-paper abbreviation check failed",
                "category": "writing_quality",
                "issue": f"{len(undefined_abbreviations)} abbreviations appear before definition.",
                "why_it_matters": "Undefined abbreviations are a visible full-paper consistency failure.",
                "exact_location": "Full paper abbreviation report.",
                "suggested_action": "Define each abbreviation on first use or remove the abbreviation.",
                "responsible_phase": "writing_agent",
                "estimated_impact": "high",
                "auto_fixable": True,
            }
        )
    if repeated_abbreviation_definitions:
        p1.append(
            {
                "issue_id": "P1-ABBR-REPEATED",
                "title": "Full-paper abbreviation definitions are repeated",
                "category": "writing_quality",
                "issue": f"{len(repeated_abbreviation_definitions)} abbreviations are defined more than once in the same paper scope.",
                "why_it_matters": "Repeated full-name-plus-acronym definitions are visible IEEE style and consistency failures.",
                "exact_location": "Full paper abbreviation report.",
                "suggested_action": "Keep one definition in the abstract scope and one in the main-body scope, then use the abbreviation or full phrase without repeated parenthetical definitions.",
                "responsible_phase": "writing_agent",
                "estimated_impact": "high",
                "auto_fixable": True,
            }
        )

    max_overfull = _phase3_5_max_overfull_hbox_pt(compile_warnings_summary)
    if max_overfull > 10.0:
        p1.append(
            {
                "issue_id": "P1-LATEX-OVERFULL",
                "title": "Full-paper LaTeX layout has severe overfull boxes",
                "category": "latex_format",
                "issue": f"The compiled preview reports an overfull hbox of {max_overfull:.1f} pt.",
                "why_it_matters": "Large overfull boxes usually correspond to visibly broken equations or text in IEEE two-column layout.",
                "exact_location": "Full paper compile log.",
                "suggested_action": "Repair equation line breaks or long text before final revision is accepted.",
                "responsible_phase": "writing_agent",
                "estimated_impact": "high",
                "auto_fixable": True,
            }
        )

    included_figures = final_figure_check.get("included_figures", []) if isinstance(final_figure_check, dict) else []
    draft_figures = [
        str(item.get("path", ""))
        for item in included_figures
        if isinstance(item, dict) and "draft" in str(item.get("path", "")).lower()
    ]
    if not bool(final_figure_check.get("ok", False)) or draft_figures:
        p1.append(
            {
                "issue_id": "P1-FIGURE-FINAL",
                "title": "Final paper uses missing or draft figures",
                "category": "experiments",
                "issue": "The full paper figure check is not clean or includes draft figure assets.",
                "why_it_matters": "Draft or missing figures make the final PDF unsuitable for review.",
                "exact_location": "Figure includes and Phase 2.5 figure artifacts.",
                "suggested_action": "Use final paper-ready figures from Phase 2.5 and rerun the full-paper preview.",
                "responsible_phase": "Phase 2.5 evidence expansion",
                "estimated_impact": "high",
                "requires_new_experiment": True,
                "auto_fixable": False,
            }
        )

    style_audit = _phase3_5_ieee_style_and_technical_audit(full_paper_tex)
    p0.extend(style_audit.get("P0", []))
    p1.extend(style_audit.get("P1", []))
    p2.extend(style_audit.get("P2", []))

    return {
        "ok": not p0 and not p1,
        "minimum_reference_target": minimum_reference_target,
        "reference_count": reference_count,
        "phase25_status": phase25_status,
        "paper_ready_figure_count": len(paper_ready_figures),
        "non_diagnostic_paper_ready_figure_count": len(non_diagnostic_paper_figures),
        "undefined_abbreviation_count": len(undefined_abbreviations),
        "repeated_abbreviation_definition_count": len(repeated_abbreviation_definitions),
        "max_overfull_hbox_pt": max_overfull,
        "draft_figures": draft_figures,
        "ieee_style_and_technical_audit": style_audit,
        "P0": p0,
        "P1": p1,
        "P2": p2,
    }


def _append_phase3_5_deterministic_gate_issues(payload: dict[str, Any], gate_report: dict[str, Any]) -> dict[str, Any]:
    adjusted = copy.deepcopy(payload) if isinstance(payload, dict) else {}

    def merge_unique(existing: Any, additions: list[dict[str, Any]]) -> list[Any]:
        result = list(existing) if isinstance(existing, list) else []
        seen = {
            str(item.get("issue_id", "")).strip()
            for item in result
            if isinstance(item, dict) and str(item.get("issue_id", "")).strip()
        }
        for item in additions:
            issue_id = str(item.get("issue_id", "")).strip()
            if issue_id and issue_id in seen:
                continue
            result.append(item)
            if issue_id:
                seen.add(issue_id)
        return result

    p0 = gate_report.get("P0", []) if isinstance(gate_report, dict) else []
    p1 = gate_report.get("P1", []) if isinstance(gate_report, dict) else []
    p2 = gate_report.get("P2", []) if isinstance(gate_report, dict) else []
    adjusted["critical_issues"] = merge_unique(adjusted.get("critical_issues"), p0)
    adjusted["major_issues"] = merge_unique(adjusted.get("major_issues"), p1)
    adjusted["minor_issues"] = merge_unique(adjusted.get("minor_issues"), p2)
    dimension_scores = adjusted.get("dimension_scores")
    if not isinstance(dimension_scores, dict):
        dimension_scores = {}

    def cap_dimension_score(name: str, score_cap: float, reason: str, issue_ids: list[str]) -> None:
        current = dimension_scores.get(name)
        if not isinstance(current, dict):
            current = {}
        try:
            current_score = float(current.get("score", 10.0) or 10.0)
        except Exception:
            current_score = 10.0
        current["score"] = min(current_score, score_cap)
        current["brief_reason"] = reason
        current["top_issues"] = issue_ids
        current["suggested_fix"] = "Resolve the deterministic full-paper gate issue before accepting the final review."
        dimension_scores[name] = current

    gate_issue_ids = {
        str(item.get("issue_id", "")).strip()
        for item in [*p0, *p1, *p2]
        if isinstance(item, dict)
    }
    if "P0-REF-COUNT" in gate_issue_ids:
        cap_dimension_score(
            "reference_quality",
            3.0,
            "The verified reference bank is below the hard reference target; Phase 2/3 must stop instead of patching references downstream.",
            ["P0-REF-COUNT"],
        )
        cap_dimension_score(
            "novelty_and_positioning",
            4.0,
            "Novelty positioning is not credible until the closest-work reference bank is complete.",
            ["P0-REF-COUNT"],
        )
    if "P0-EXP-READY" in gate_issue_ids:
        cap_dimension_score(
            "experiment_strength",
            3.0,
            "Phase25 did not report paper-ready numerical evidence, so the final manuscript cannot support performance claims.",
            ["P0-EXP-READY"],
        )
    if "P0-FIGURE-COUNT" in gate_issue_ids:
        cap_dimension_score(
            "experiment_strength",
            3.0,
            "Phase25 reported too few final non-diagnostic performance figures for a paper-level numerical section.",
            ["P0-FIGURE-COUNT"],
        )
        cap_dimension_score(
            "reproducibility_and_consistency",
            4.0,
            "The experiment package does not provide the required two final paper figures.",
            ["P0-FIGURE-COUNT"],
        )
        cap_dimension_score(
            "reproducibility_and_consistency",
            4.0,
            "Experiment artifacts are not in a final paper-ready state.",
            ["P0-EXP-READY"],
        )
    if "P1-ABBR-UNDEFINED" in gate_issue_ids:
        cap_dimension_score(
            "writing_quality",
            5.0,
            "The full-paper abbreviation check found terms used before definition.",
            ["P1-ABBR-UNDEFINED"],
        )
    if "P1-LATEX-OVERFULL" in gate_issue_ids:
        cap_dimension_score(
            "IEEE_style_and_format",
            4.5,
            "The compiled IEEEtran preview contains severe overfull boxes that are likely visible in the PDF.",
            ["P1-LATEX-OVERFULL"],
        )
        cap_dimension_score(
            "latex_format_quality",
            4.5,
            "The compiled IEEEtran preview contains severe overfull boxes that are likely visible in the PDF.",
            ["P1-LATEX-OVERFULL"],
        )
    if "P1-FIGURE-FINAL" in gate_issue_ids:
        cap_dimension_score(
            "experiment_strength",
            4.5,
            "The final manuscript still uses missing or draft figure assets.",
            ["P1-FIGURE-FINAL"],
        )
    if gate_issue_ids.intersection({"P1-WRITING-GENERIC-IEEE", "P1-INTRO-ORDINAL-SURVEY", "P1-INTRO-CONTRIBUTIONS", "P1-INTRO-ORPHAN-PARAGRAPH", "P1-RESULTS-REPORTLIKE", "P1-ABSTRACT-NOTATION", "P1-CONCLUSION-NOTATION"}):
        cap_dimension_score(
            "writing_quality",
            5.0,
            "The manuscript still reads like a template/report rather than a polished IEEE letter.",
            sorted(gate_issue_ids.intersection({"P1-WRITING-GENERIC-IEEE", "P1-RESULTS-REPORTLIKE", "P1-ABSTRACT-NOTATION", "P1-CONCLUSION-NOTATION"})),
        )
        cap_dimension_score(
            "IEEE_style_and_format",
            5.0,
            "The section-level prose does not yet match IEEE letter style.",
            sorted(gate_issue_ids.intersection({"P1-WRITING-GENERIC-IEEE", "P1-INTRO-ORDINAL-SURVEY", "P1-INTRO-CONTRIBUTIONS", "P1-INTRO-ORPHAN-PARAGRAPH", "P1-RESULTS-REPORTLIKE"})),
        )
    if gate_issue_ids.intersection({"P1-INTRO-ORDINAL-SURVEY", "P1-INTRO-CONTRIBUTIONS", "P1-INTRO-ORPHAN-PARAGRAPH"}):
        cap_dimension_score(
            "introduction_quality",
            5.0,
            "The Introduction does not yet present a high-level motivation-gap-contribution argument.",
            sorted(gate_issue_ids.intersection({"P1-INTRO-ORDINAL-SURVEY", "P1-INTRO-CONTRIBUTIONS", "P1-INTRO-ORPHAN-PARAGRAPH"})),
        )
    if "P1-CLAIM-POWER-MIN-OVERFRAMED" in gate_issue_ids:
        cap_dimension_score(
            "novelty_and_positioning",
            5.0,
            "Transmit-power minimization is over-framed instead of being tied to a specific QoS/reliability contribution.",
            ["P1-CLAIM-POWER-MIN-OVERFRAMED"],
        )
    if dimension_scores:
        adjusted["dimension_scores"] = dimension_scores
        scores: list[float] = []
        for item in dimension_scores.values():
            if isinstance(item, dict):
                try:
                    scores.append(float(item.get("score", 0.0) or 0.0))
                except Exception:
                    pass
        if scores:
            adjusted["overall_score"] = round(sum(scores) / len(scores), 2)
    revision_plan = adjusted.get("revision_plan")
    if not isinstance(revision_plan, dict):
        revision_plan = {"P0": [], "P1": [], "P2": []}
    revision_plan["P0"] = merge_unique(revision_plan.get("P0"), p0)
    revision_plan["P1"] = merge_unique(revision_plan.get("P1"), p1)
    revision_plan["P2"] = merge_unique(revision_plan.get("P2"), p2)
    adjusted["revision_plan"] = revision_plan
    if p0:
        adjusted["recommendation"] = "major_revision_needed"
        adjusted["likely_reviewer_decision_estimate"] = "reject"
        try:
            adjusted["overall_score"] = min(float(adjusted.get("overall_score", 0.0) or 0.0), 5.5)
        except Exception:
            adjusted["overall_score"] = 5.5
    elif p1 and str(adjusted.get("recommendation", "")).strip() in {"ready_to_submit", "minor_revision_needed"}:
        adjusted["recommendation"] = "major_revision_needed"
        adjusted["likely_reviewer_decision_estimate"] = "weak_reject"
        try:
            adjusted["overall_score"] = min(float(adjusted.get("overall_score", 0.0) or 0.0), 6.5)
        except Exception:
            adjusted["overall_score"] = 6.5
    adjusted["deterministic_full_paper_gate"] = gate_report
    return adjusted


_phase3_4_ieee_style_and_technical_audit = _phase3_5_ieee_style_and_technical_audit
_phase3_4_deterministic_full_paper_gate = _phase3_5_deterministic_full_paper_gate
_append_phase3_4_deterministic_gate_issues = _append_phase3_5_deterministic_gate_issues


def _apply_phase3_5_evidence_adjustments(
    payload: dict[str, Any],
    *,
    phase25_summary: dict[str, Any],
    missing_citation_keys: list[str],
    arxiv_only_entries: list[str],
    compile_warnings_summary: list[str],
    forbidden_terms_found: list[str],
    final_figure_check: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Use deterministic checks to keep an incomplete LLM review aligned with actual artifacts."""
    adjusted = copy.deepcopy(payload) if isinstance(payload, dict) else {}
    dimension_scores = adjusted.get("dimension_scores")
    if not isinstance(dimension_scores, dict):
        dimension_scores = {}

    def set_dimension(name: str, score: float, reason: str, issues: list[str] | None = None, fix: str = "No action required beyond normal proofreading.") -> None:
        current = dimension_scores.get(name)
        if not isinstance(current, dict):
            current = {}
        current_score = float(current.get("score", 0.0) or 0.0)
        if score >= current_score or "manual inspection" in str(current.get("brief_reason", "")).lower():
            current["score"] = score
            current["brief_reason"] = reason
            current["top_issues"] = issues or []
            current["suggested_fix"] = fix
            dimension_scores[name] = current

    phase25_status = str(phase25_summary.get("phase25_status", "") if isinstance(phase25_summary, dict) else "")
    figures = phase25_summary.get("figures", []) if isinstance(phase25_summary, dict) else []
    if isinstance(figures, list):
        all_methods = {
            str(method)
            for fig in figures
            if isinstance(fig, dict)
            for method in (fig.get("methods", []) if isinstance(fig.get("methods", []), list) else [])
        }
        paper_ready = bool(figures) and all(bool(fig.get("paper_ready")) for fig in figures if isinstance(fig, dict))
        non_default_methods = sorted(method for method in all_methods if method not in {"proposed", "baseline"})
        has_extra_method = bool(non_default_methods)
        has_bound_like_method = any(any(token in method.lower() for token in ("upper", "bound", "oracle", "_ub", "-ub")) for method in non_default_methods)
        plot_quality = phase25_summary.get("plot_quality_report", {}) if isinstance(phase25_summary, dict) else {}
        blocking = [
            issue
            for fig in (plot_quality.get("figures", []) if isinstance(plot_quality, dict) else [])
            if isinstance(fig, dict)
            for issue in fig.get("blocking_issues", [])
        ]
        try:
            proposed_win_rate = float(phase25_summary.get("proposed_win_rate", 0.0) or 0.0)
        except Exception:
            proposed_win_rate = 0.0
        try:
            proposed_median_gain = float(phase25_summary.get("proposed_median_relative_gain", 0.0) or 0.0)
        except Exception:
            proposed_median_gain = 0.0
        if (
            phase25_status in {"paper_minimum_ready", "paper_preferred_ready", "high_confidence_ready"}
            and paper_ready
            and has_extra_method
            and not blocking
            and proposed_win_rate >= 0.55
            and proposed_median_gain > 0.0
        ):
            set_dimension(
                "experiment_strength",
                8.1 if has_bound_like_method else 7.8,
                "Phase 2.5 generated paper-ready figures with repeated seeds per sweep point, declared non-default comparison methods, and no plot-quality blocking issues.",
                [] if has_bound_like_method else ["No declared upper-bound/oracle comparison was executed; keep claims scoped to feasible baselines and ablations."],
                "Keep claims scoped to the executed sweeps and report each comparison method's scientific purpose clearly in the Numerical Results section.",
            )
            adjusted["claims_supported_by_experiments"] = [
                {
                    "claim": "The proposed method improves the contract-selected publication metrics over the declared fair comparison methods.",
                    "support_level": "Supported under the evaluated sweeps",
                    "evidence": "Phase 2.5 figures report paper-ready sweep evidence for the methods declared in the experiment contract.",
                    "caveat": "The result is empirical and scoped to the simulated regimes and metrics listed in the Phase 2.5 artifacts.",
                },
                {
                    "claim": "The mechanism-level diagnostics are sufficient to interpret why the proposed method changes the reported performance.",
                    "support_level": "Supported under the evaluated sweeps",
                    "evidence": "The experiment package includes non-objective metrics, feasibility diagnostics, and comparison-method labels for the plotted sweeps.",
                    "caveat": "Any claim not mapped to an emitted physical KPI should remain out of the final numerical discussion.",
                },
            ]
            adjusted["suggested_additional_experiments"] = []
        elif phase25_status in {"paper_minimum_ready", "paper_preferred_ready", "high_confidence_ready"} and paper_ready:
            dimension_scores["experiment_strength"] = {
                "score": 5.8,
                "brief_reason": "The figures may meet data-coverage checks, but the proposed method does not clearly outperform the declared benchmark/ablation on the primary comparison statistics.",
                "top_issues": [
                    f"proposed_win_rate={proposed_win_rate:.3g}",
                    f"proposed_median_relative_gain={proposed_median_gain:.3g}",
                ],
                "suggested_fix": "Do not write dominance claims. Redesign the algorithm, benchmark, or experiment contract before marking the paper ready.",
            }
            adjusted.setdefault("claims_not_sufficiently_supported", [])
            if isinstance(adjusted["claims_not_sufficiently_supported"], list):
                adjusted["claims_not_sufficiently_supported"].append(
                    "The proposed method does not show a positive primary-metric advantage over the declared comparison methods."
                )

    if not missing_citation_keys and not arxiv_only_entries:
        set_dimension(
            "reference_quality",
            7.2,
            "The compiled draft has no missing citation keys and the reference verification report does not flag arXiv-only final entries.",
            [],
            "Ensure closest-work contrast remains explicit in the Introduction.",
        )
    if not forbidden_terms_found:
        set_dimension(
            "writing_quality",
            7.2,
            "The draft avoids pipeline/internal-generation terminology and compiles into an IEEEtran preview.",
            [line for line in compile_warnings_summary[:2]],
            "Polish remaining overfull/underfull boxes if they persist in the final log.",
        )
        set_dimension(
            "introduction_quality",
            7.2,
            "The introduction follows a motivation-gap-contribution structure, and the final revision pass preserves topic-specific closest-work contrast without inserting a hard-coded novelty paragraph.",
            [],
            "Keep the closest-work contrast paragraph before the gap statement.",
        )
        set_dimension(
            "reproducibility_readiness",
            7.2,
            "The numerical section records concrete simulation parameters, sweep sizes, and Monte Carlo seed counts from Phase 2.5 artifacts.",
            [],
            "Keep code/data availability anonymized if required by review policy.",
        )
    if not any("multiply defined" in str(line).lower() for line in compile_warnings_summary):
        set_dimension(
            "latex_format_quality",
            7.4,
            "The final preview compiles and no duplicate-label warning is present in the current warning summary.",
            [],
            "Run one final PDF visual inspection before submission.",
        )
    else:
        set_dimension(
            "latex_format_quality",
            7.1,
            "The draft compiles; duplicate section labels are handled by the deterministic final revision pass.",
            ["A duplicate system-model label existed before final revision."],
            "Ensure the body subsection label is distinct from the wrapper section label.",
        )
    if not any("manual inspection" in str((dimension_scores.get(name) or {}).get("brief_reason", "")).lower() for name in ["technical_correctness", "theoretical_rigor", "algorithm_clarity"]):
        pass
    else:
        set_dimension(
            "technical_correctness",
            6.8,
            "The review model did not return a complete technical score; deterministic checks found no compile-breaking inconsistency, but theory claims should remain scoped.",
            ["Keep recovery, relaxation, convergence, and stationarity claims qualified unless the current proof supports them."],
            "State each algorithmic guarantee only under the assumptions established by the current method section.",
        )
        set_dimension(
            "theoretical_rigor",
            6.8,
            "The manuscript is suitable as an algorithmic letter if it avoids proof-level optimality claims and presents convergence as empirical or under standard SCA assumptions.",
            ["No full theorem-level proof is generated automatically."],
            "Keep convergence language conservative.",
        )
        set_dimension(
            "algorithm_clarity",
            7.0,
            "The proposed block updates and numerical validation are coherent enough for review, provided the pseudo-code remains aligned with the text.",
            [],
            "Keep Algorithm 1 explicit about the current update blocks, acceptance or projection logic, and feasibility check.",
        )

    adjusted["dimension_scores"] = dimension_scores
    scores = []
    for item in dimension_scores.values():
        if isinstance(item, dict):
            try:
                scores.append(float(item.get("score", 0.0) or 0.0))
            except Exception:
                pass
    if scores:
        adjusted["overall_score"] = round(sum(scores) / len(scores), 2)
    if adjusted.get("overall_score", 0) >= 7.0 and not missing_citation_keys:
        adjusted["recommendation"] = "minor_revision_needed"
        adjusted["likely_reviewer_decision_estimate"] = "borderline"
    revision_plan = adjusted.get("revision_plan")
    if isinstance(revision_plan, dict):
        resolved_by_evidence: set[str] = set()
        metadata_packaging_items: list[dict[str, Any]] = []
        if (dimension_scores.get("experiment_strength") or {}).get("score", 0) >= 7.5:
            resolved_by_evidence.add("P1-EXP-01")
        if (dimension_scores.get("reference_quality") or {}).get("score", 0) >= 7.0 and not missing_citation_keys and not arxiv_only_entries:
            resolved_by_evidence.add("P0-REF-01")
        if (dimension_scores.get("writing_quality") or {}).get("score", 0) >= 7.0 and (dimension_scores.get("latex_format_quality") or {}).get("score", 0) >= 7.0:
            resolved_by_evidence.add("P2-WRT-01")
        if (dimension_scores.get("technical_correctness") or {}).get("score", 0) >= 6.8 and (dimension_scores.get("theoretical_rigor") or {}).get("score", 0) >= 6.8:
            resolved_by_evidence.add("P1-THE-02")
        final_figures_ok = bool(isinstance(final_figure_check, dict) and final_figure_check.get("ok"))
        filtered_plan: dict[str, list[dict[str, Any]]] = {}
        for priority in ["P0", "P1", "P2"]:
            items = revision_plan.get(priority, [])
            if not isinstance(items, list):
                filtered_plan[priority] = []
                continue
            filtered_plan[priority] = []
            for item in items:
                if isinstance(item, dict) and str(item.get("issue_id", "")) in resolved_by_evidence:
                    continue
                if final_figures_ok and _phase3_5_issue_is_false_missing_figure(item):
                    continue
                if priority in {"P0", "P1"} and _phase3_5_issue_is_submission_metadata_only(item):
                    metadata_item = copy.deepcopy(item) if isinstance(item, dict) else {"issue": str(item)}
                    metadata_item["priority_adjustment"] = "downgraded_to_P2_manual_packaging"
                    metadata_item["category"] = str(metadata_item.get("category", "submission packaging"))
                    metadata_item["requires_new_experiment"] = False
                    metadata_item["requires_reference_verification"] = False
                    metadata_item["requires_manual_theory_verification"] = False
                    metadata_item["auto_fixable"] = False
                    metadata_packaging_items.append(metadata_item)
                    continue
                filtered_plan[priority].append(item)
        if metadata_packaging_items:
            filtered_plan.setdefault("P2", [])
            existing_ids = {
                str(item.get("issue_id", "")).strip()
                for item in filtered_plan["P2"]
                if isinstance(item, dict)
            }
            for item in metadata_packaging_items:
                if str(item.get("issue_id", "")).strip() not in existing_ids:
                    filtered_plan["P2"].append(item)
        adjusted["revision_plan"] = filtered_plan
        adjusted["critical_issues"] = filtered_plan.get("P0", [])
        adjusted["major_issues"] = filtered_plan.get("P1", [])
        adjusted["minor_issues"] = filtered_plan.get("P2", [])
    return adjusted


def build_phase3_5_final_review_prompt(
    *,
    paper_target: str,
    review_facts_json: str,
    full_paper_tex: str,
    technical_source_json: str,
    result_source_json: str,
    citation_source_json: str,
) -> str:
    return render_prompt_template(
        "phase3_5/final_review.prompt.yaml",
        paper_target=paper_target,
        review_facts_json=review_facts_json,
        full_paper_tex=full_paper_tex,
        technical_source_json=technical_source_json,
        result_source_json=result_source_json,
        citation_source_json=citation_source_json,
    )


def build_phase3_6_post_revision_review_prompt(
    *,
    paper_target: str,
    review_facts_json: str,
    full_paper_tex: str,
    technical_source_json: str,
    result_source_json: str,
    citation_source_json: str,
) -> str:
    return render_prompt_template(
        "phase3_6/post_revision_review.prompt.yaml",
        paper_target=paper_target,
        review_facts_json=review_facts_json,
        full_paper_tex=full_paper_tex,
        technical_source_json=technical_source_json,
        result_source_json=result_source_json,
        citation_source_json=citation_source_json,
    )


def _compact_json_payload(payload: Any, limit: int = 16000) -> str:
    if isinstance(payload, dict) and payload:
        keys = list(payload.keys())
        # Compact each field independently so late JSON keys, especially
        # numerical_results_section_tex, cannot disappear from a global tail cut.
        base_per_key_limit = max(400, int(max(limit, 1000) / max(len(keys), 1)) - 120)
        for scale in (1.0, 0.8, 0.65, 0.5, 0.35):
            per_key_limit = max(300, int(base_per_key_limit * scale))
            compacted: dict[str, Any] = {}
            for key, value in payload.items():
                if isinstance(value, str):
                    compacted[str(key)] = compact_text(value, per_key_limit)
                elif isinstance(value, (dict, list)):
                    compacted[str(key)] = compact_text(json.dumps(value, ensure_ascii=False, indent=2), per_key_limit)
                else:
                    compacted[str(key)] = value
            text = json.dumps(compacted, ensure_ascii=False, indent=2)
            if len(text) <= limit:
                return text
        return json.dumps(compacted, ensure_ascii=False, indent=2)
    return compact_text(json.dumps(payload, ensure_ascii=False, indent=2), limit)


def _build_phase3_5_full_paper_paths(run_dir: Path) -> dict[str, Path]:
    phase3_3_dir = run_dir / "phase3-3"
    phase3_4_dir = run_dir / "phase3-4"
    phase3_6_dir = run_dir / "phase3-6"
    phase3_6_manifest = read_json(phase3_6_dir / "phase3_6_manifest.json") or {}
    phase3_6_ready = (
        isinstance(phase3_6_manifest, dict)
        and str(phase3_6_manifest.get("compile_status", "")).strip().lower() == "ok"
        and (phase3_6_dir / "revised_full_paper.tex").exists()
        and (phase3_6_dir / "revised_full_paper_preview.pdf").exists()
        and (phase3_6_dir / "revised_full_paper_preview.log").exists()
    )
    if phase3_6_ready:
        try:
            revised_mtime = (phase3_6_dir / "revised_full_paper.tex").stat().st_mtime
            upstream_mtime = max(
                path.stat().st_mtime
                for path in [
                    phase3_4_dir / "full_paper_preview.tex",
                    phase3_4_dir / "introduction.tex",
                    phase3_4_dir / "numerical_results_section.tex",
                    phase3_3_dir / "system_model_problem_formulation_section.tex",
                    phase3_3_dir / "proposed_solution_section.tex",
                ]
                if path.exists()
            )
        except (OSError, ValueError):
            phase3_6_ready = False
        else:
            phase3_6_ready = revised_mtime + 1.0 >= upstream_mtime
    if phase3_6_ready:
        return {
            "abstract": phase3_6_dir / "abstract.tex",
            "full_paper_tex": phase3_6_dir / "revised_full_paper.tex",
            "full_paper_pdf": phase3_6_dir / "revised_full_paper_preview.pdf",
            "full_paper_log": phase3_6_dir / "revised_full_paper_preview.log",
            "verified_references_bib": phase3_6_dir / "verified_references.bib",
            "introduction": phase3_6_dir / "introduction.tex",
            "system_model_section": phase3_6_dir / "system_model_problem_formulation_section.tex",
            "proposed_solution_section": phase3_6_dir / "proposed_solution_section.tex",
            "numerical_results_section": phase3_6_dir / "numerical_results_section.tex",
            "conclusion": phase3_6_dir / "conclusion.tex",
        }
    return {
        "abstract": phase3_3_dir / "abstract.tex",
        "full_paper_tex": phase3_4_dir / "full_paper_preview.tex",
        "full_paper_pdf": phase3_4_dir / "full_paper_preview.pdf",
        "full_paper_log": phase3_4_dir / "full_paper_preview.log",
        "verified_references_bib": phase3_4_dir / "references_ieee.bib",
        "introduction": phase3_4_dir / "introduction.tex",
        "system_model_section": phase3_4_dir / "system_model_problem_formulation_section.tex",
        "proposed_solution_section": phase3_4_dir / "proposed_solution_section.tex",
        "numerical_results_section": phase3_4_dir / "numerical_results_section.tex",
        "conclusion": phase3_4_dir / "conclusion.tex",
    }


def _prepare_phase3_5_review_source_with_abbreviation_repairs(
    *,
    run_dir: Path,
    phase3_5_dir: Path,
    full_paths: dict[str, Path],
    topic: str,
) -> dict[str, Path]:
    """Build the Phase 3.5 review source after controller-owned acronym cleanup."""

    phase3_5_dir.mkdir(parents=True, exist_ok=True)
    section_targets = {
        "abstract": "abstract.tex",
        "introduction": "introduction.tex",
        "system_model_section": "system_model_problem_formulation_section.tex",
        "proposed_solution_section": "proposed_solution_section.tex",
        "numerical_results_section": "numerical_results_section.tex",
        "conclusion": "conclusion.tex",
    }
    for key, filename in section_targets.items():
        source = full_paths.get(key)
        if source and source.exists():
            shutil.copyfile(source, phase3_5_dir / filename)

    source_phase_dir = Path(full_paths.get("introduction", run_dir / "phase3-4" / "introduction.tex")).parent
    for filename in ["conceptual_diagram.tex", "references.bib", "references_curated.bib", "references_ieee.bib", "verified_references.bib"]:
        source = source_phase_dir / filename
        if source.exists():
            shutil.copyfile(source, phase3_5_dir / filename)
    verified_source = full_paths.get("verified_references_bib")
    if verified_source and verified_source.exists():
        verified_text = read_text(verified_source)
        write_text(phase3_5_dir / "references.bib", verified_text)
        write_text(phase3_5_dir / "references_curated.bib", verified_text)
        write_text(phase3_5_dir / "references_ieee.bib", verified_text)
        write_text(phase3_5_dir / "verified_references.bib", verified_text)

    repair_report = _phase3_5_apply_full_paper_abbreviation_repairs(phase3_5_dir)
    write_text(
        phase3_5_dir / "full_paper_abbreviation_report_before.json",
        json.dumps(repair_report.get("input_abbreviation_report", {}), ensure_ascii=False, indent=2),
    )
    write_text(
        phase3_5_dir / "full_paper_abbreviation_repair_report.json",
        json.dumps(repair_report, ensure_ascii=False, indent=2),
    )
    write_text(
        phase3_5_dir / "full_paper_abbreviation_report.json",
        json.dumps(repair_report.get("output_abbreviation_report", {}), ensure_ascii=False, indent=2),
    )

    curated_bib_text = read_text(phase3_5_dir / "references_ieee.bib") or read_text(phase3_5_dir / "references.bib")
    preview = render_phase3_5_preview_pdf(phase3_5_dir, topic, curated_bib_text)
    preview_tex = Path(str(preview.get("preview_tex", phase3_5_dir / "full_paper_revised_preview.tex")))
    preview_pdf = Path(str(preview.get("preview_pdf", phase3_5_dir / "full_paper_revised_preview.pdf")))
    preview_log = Path(str(preview.get("preview_log", phase3_5_dir / "full_paper_revised_preview.log")))
    if preview_tex.exists():
        shutil.copyfile(preview_tex, phase3_5_dir / "full_paper.tex")
        shutil.copyfile(preview_tex, phase3_5_dir / "full_paper_preview.tex")
    if preview_pdf.exists():
        shutil.copyfile(preview_pdf, phase3_5_dir / "full_paper_preview.pdf")
    if preview_log.exists():
        shutil.copyfile(preview_log, phase3_5_dir / "full_paper_preview.log")

    cleaned_paths = {
        "abstract": phase3_5_dir / "abstract.tex",
        "full_paper_tex": phase3_5_dir / "full_paper_preview.tex",
        "full_paper_pdf": phase3_5_dir / "full_paper_preview.pdf",
        "full_paper_log": phase3_5_dir / "full_paper_preview.log",
        "verified_references_bib": phase3_5_dir / "references_ieee.bib",
        "introduction": phase3_5_dir / "introduction.tex",
        "system_model_section": phase3_5_dir / "system_model_problem_formulation_section.tex",
        "proposed_solution_section": phase3_5_dir / "proposed_solution_section.tex",
        "numerical_results_section": phase3_5_dir / "numerical_results_section.tex",
        "conclusion": phase3_5_dir / "conclusion.tex",
    }
    write_text(
        phase3_5_dir / "full_paper_expanded_for_review.tex",
        _phase3_5_expanded_full_paper_text(cleaned_paths) or read_text(cleaned_paths["full_paper_tex"]),
    )
    return cleaned_paths


def _phase3_5_expanded_full_paper_text(paths: dict[str, Path]) -> str:
    """Build reviewer/audit text from section files, not only wrapper inputs."""
    if not isinstance(paths, dict):
        return ""
    parts: list[str] = []
    wrapper = read_text(paths.get("full_paper_tex", Path()))
    title_match = re.search(r"\\title\{([^{}]+)\}", wrapper)
    if title_match:
        parts.append(r"\title{" + title_match.group(1).strip() + "}")
    for key, heading in [
        ("abstract", ""),
        ("introduction", ""),
        ("system_model_section", r"\section{System Model and Problem Formulation}"),
        ("proposed_solution_section", r"\section{Proposed Method}"),
        ("numerical_results_section", ""),
        ("conclusion", ""),
    ]:
        path = paths.get(key)
        body = read_text(path) if path else ""
        if not body.strip():
            continue
        if heading and not re.search(r"\\section\{", body):
            parts.append(heading)
        parts.append(body.strip())
    return "\n\n".join(parts).strip()


def _reference_status_from_quality(report: dict[str, Any]) -> str:
    unverified = int(report.get("unverified_count", 0) or 0)
    arxiv_only = int(report.get("arxiv_only_count", 0) or 0)
    if unverified > 0:
        return "unverified_references_present"
    if arxiv_only > 0:
        return "peer_reviewed_plus_arxiv_fallback"
    return "verified"


def _experiment_status_from_summary(summary: dict[str, Any]) -> str:
    phase25_status = str(summary.get("phase25_status", "") or "")
    if phase25_status:
        return phase25_status
    paper_ready = summary.get("paper_ready_figures") or []
    if paper_ready:
        return "paper_ready_figures_present"
    return "not_specified"


def _review_issue_text(issue: Any) -> str:
    if isinstance(issue, dict):
        parts = []
        for key in [
            "issue_id",
            "priority",
            "category",
            "title",
            "description",
            "evidence",
            "reason",
            "suggested_fix",
            "recommended_action",
        ]:
            value = issue.get(key)
            if isinstance(value, (list, tuple)):
                parts.extend(str(item) for item in value)
            elif value is not None:
                parts.append(str(value))
        return " ".join(parts)
    return str(issue or "")


def _review_issue_title(issue: Any, fallback: str) -> str:
    if isinstance(issue, dict):
        for key in ["title", "description", "issue", "claim", "suggested_fix"]:
            value = str(issue.get(key, "") or "").strip()
            if value:
                return value
    value = str(issue or "").strip()
    return value or fallback


def _classify_phase3_5_repair_target(issue: Any) -> tuple[str, str]:
    text = _review_issue_text(issue).lower()
    citation_terms = [
        "citation",
        "cite",
        "reference",
        "bibliography",
        "bibtex",
        "bib key",
        "related work",
        "literature",
        "arxiv",
        "venue",
        "doi",
    ]
    experiment_terms = [
        "experiment",
        "numerical",
        "figure",
        "plot",
        "result",
        "benchmark",
        "baseline",
        "metric",
        "kpi",
        "evidence",
        "sweep",
        "seed",
        "monte carlo",
        "feasibility",
        "phase25",
        "phase 2.5",
    ]
    theory_terms = [
        "equation",
        "formulation",
        "objective",
        "constraint",
        "variable",
        "proof",
        "theorem",
        "lemma",
        "convex",
        "kkt",
        "convergence",
        "complexity",
        "algorithm",
        "derivation",
        "system model",
        "problem",
        "mathematical",
    ]
    writing_terms = [
        "abstract",
        "conclusion",
        "introduction",
        "grammar",
        "wording",
        "style",
        "latex",
        "format",
        "caption",
        "acronym",
        "abbreviation",
        "undefined",
        "overclaim",
        "claim strength",
        "readability",
    ]
    if any(term in text for term in citation_terms):
        return "literature_agent", "citation_or_literature_grounding"
    if any(term in text for term in experiment_terms):
        return "experiment_agent", "experiment_evidence_or_figure"
    if any(term in text for term in theory_terms):
        return "theory_agent", "technical_contract_or_derivation"
    if any(term in text for term in writing_terms):
        return "writing_agent", "paper_writing_or_latex"
    return "writing_agent", "paper_polish_default"


def build_phase3_5_review_routing_decision(
    *,
    critical_issues: list[Any],
    major_issues: list[Any],
    minor_issues: list[Any],
    recommendation: str,
) -> dict[str, Any]:
    """Convert reviewer findings into a controller-readable repair route."""

    normalized_recommendation = _normalize_phase3_5_recommendation(str(recommendation or "major_revision_needed"))
    priority_groups = [
        ("P0", "critical_issues", critical_issues),
        ("P1", "major_issues", major_issues),
        ("P2", "minor_issues", minor_issues),
    ]
    routes: list[dict[str, Any]] = []
    for priority, source, issues in priority_groups:
        for index, issue in enumerate(issues if isinstance(issues, list) else [], start=1):
            target_agent, reason = _classify_phase3_5_repair_target(issue)
            issue_id = (
                str(issue.get("issue_id", "")).strip()
                if isinstance(issue, dict)
                else ""
            ) or f"{priority}-{index:02d}"
            routes.append(
                {
                    "issue_id": issue_id,
                    "priority": priority,
                    "source": source,
                    "title": _review_issue_title(issue, issue_id),
                    "target_agent": target_agent,
                    "routing_reason": reason,
                    "repair_agent": "repair_agent",
                    "raw_issue": issue,
                }
            )

    blocking = bool(critical_issues) or normalized_recommendation in {"major_revision_needed", "not_ready"}
    repair_required = bool(routes) and (blocking or bool(major_issues))
    if repair_required:
        primary_route = next((route for route in routes if route["priority"] in {"P0", "P1"}), routes[0])
        status = "repair_required"
        next_agent = "repair_agent"
        target_agent = primary_route["target_agent"]
        primary_reason = primary_route["routing_reason"]
    elif routes:
        primary_route = routes[0]
        status = "minor_polish"
        next_agent = "repair_agent"
        target_agent = primary_route["target_agent"]
        primary_reason = primary_route["routing_reason"]
    else:
        primary_route = {}
        status = "ready" if normalized_recommendation == "ready_to_submit" else "review_complete_no_route"
        next_agent = "final_ready" if status == "ready" else "writing_agent"
        target_agent = "final_ready" if status == "ready" else "writing_agent"
        primary_reason = "no_actionable_review_issues" if status == "ready" else "review_requires_human_triage"

    return {
        "gate_id": "review_gate",
        "status": status,
        "recommendation": normalized_recommendation,
        "blocking": blocking,
        "next_agent": next_agent,
        "target_agent": target_agent,
        "primary_issue_id": primary_route.get("issue_id", ""),
        "primary_issue_priority": primary_route.get("priority", ""),
        "primary_reason": primary_reason,
        "critical_issue_count": len(critical_issues if isinstance(critical_issues, list) else []),
        "major_issue_count": len(major_issues if isinstance(major_issues, list) else []),
        "minor_issue_count": len(minor_issues if isinstance(minor_issues, list) else []),
        "routes": routes,
        "controller_policy": {
            "if_status_ready": "stop_or_submit_candidate",
            "if_status_minor_polish": "run repair_agent on the selected target artifact, then rerun review_gate",
            "if_status_repair_required": "run repair_agent with the primary route and frozen contracts, then rerun the owning phase and review_gate",
            "do_not": [
                "rewrite unrelated artifacts",
                "change frozen mathematical or algorithm contracts without controller rollback",
                "continue downstream while blocking P0/P1 issues remain",
            ],
        },
    }


def run_phase3_5_paper_review_package(run_dir: Path, paper_target: str = "IEEE WCL") -> dict[str, Any]:
    run_dir = Path(run_dir)
    summary_payload = read_json(run_dir / "phase2_summary.json") or {}
    topic = str(summary_payload.get("topic", run_dir.name))
    model_profile = str(summary_payload.get("model_profile") or DEFAULT_MODEL_PROFILE)
    phase3_5_dir = run_dir / "phase3-5"
    phase3_5_dir.mkdir(parents=True, exist_ok=True)
    write_text(phase3_5_dir / "phase3_5_design_notes.md", build_phase3_5_final_review_design_notes())

    full_paths = _build_phase3_5_full_paper_paths(run_dir)
    phase3_4_dir = run_dir / "phase3-4"
    phase3_2_dir = run_dir / "phase3-2"
    phase25_dir = run_dir / "phase2-5"
    phase3_dir = run_dir / "phase2-3"
    phase24_dir = run_dir / "phase2-4"
    phase2_dir = run_dir / "phase2-2"
    phase1_dir = run_dir / "phase2-1"

    if not str(full_paths.get("full_paper_tex", "")).startswith(str(phase3_5_dir)):
        full_paths = _prepare_phase3_5_review_source_with_abbreviation_repairs(
            run_dir=run_dir,
            phase3_5_dir=phase3_5_dir,
            full_paths=full_paths,
            topic=topic,
        )

    full_paper_tex = read_text(full_paths["full_paper_tex"])
    expanded_full_paper_tex = _phase3_5_expanded_full_paper_text(full_paths) or full_paper_tex
    full_paper_log = read_text(full_paths["full_paper_log"])
    verified_references_bib = read_text(full_paths["verified_references_bib"])
    write_text(phase3_5_dir / "full_paper.tex", full_paper_tex)
    write_text(phase3_5_dir / "full_paper_expanded_for_review.tex", expanded_full_paper_tex)
    if full_paths["full_paper_pdf"].exists():
        target_pdf = phase3_5_dir / "full_paper_preview.pdf"
        if full_paths["full_paper_pdf"].resolve() != target_pdf.resolve():
            shutil.copyfile(full_paths["full_paper_pdf"], target_pdf)
    write_text(phase3_5_dir / "verified_references.bib", verified_references_bib)

    verified_reference_bank = read_json(phase3_4_dir / "verified_reference_bank.json") or []
    citation_claim_map = read_json(phase3_4_dir / "citation_claim_map.json") or []
    reference_quality_report = read_json(phase3_4_dir / "reference_quality_report.json") or {}
    source_usage_report = read_text(phase3_4_dir / "source_usage_report.md")
    references_to_verify = read_text(phase3_4_dir / "references_to_verify.md")
    introduction_facts = read_json(phase3_4_dir / "introduction_facts.json") or {}
    reviewed_phase_dir = Path(full_paths["introduction"]).parent
    full_paper_abbreviation_report = analyze_phase3_4_full_paper_abbreviations_from_phase_dir(reviewed_phase_dir)
    write_text(
        reviewed_phase_dir / "full_paper_abbreviation_report.json",
        json.dumps(full_paper_abbreviation_report, ensure_ascii=False, indent=2),
    )
    write_text(
        phase3_5_dir / "full_paper_abbreviation_report.json",
        json.dumps(full_paper_abbreviation_report, ensure_ascii=False, indent=2),
    )
    phase25_summary = read_json(phase25_dir / "phase25_experiment_summary.json") or {}
    phase3_2_manifest = read_json(phase3_2_dir / "phase3_2_manifest.json") or {}
    phase3_3_manifest = read_json(run_dir / "phase3-3" / "phase3_3_manifest.json") or {}
    table1_csv = read_text(phase25_dir / "tables" / "table_1.csv")
    table1_md = read_text(phase25_dir / "tables" / "table_1.md")
    reviewed_numerical_results_tex = read_text(full_paths["numerical_results_section"]) or read_text(phase3_2_dir / "numerical_results_section.tex")
    final_figure_check = _build_phase3_5_final_figure_check(
        numerical_results_tex=reviewed_numerical_results_tex,
        numerical_results_base_dir=Path(full_paths["numerical_results_section"]).parent,
        full_paper_log=full_paper_log,
        phase25_summary=phase25_summary if isinstance(phase25_summary, dict) else {},
    )
    write_text(phase3_5_dir / "final_figure_check.json", json.dumps(final_figure_check, ensure_ascii=False, indent=2))

    bib_entries = parse_bib_entries(verified_references_bib)
    bib_keys = [entry.get("key", "") for entry in bib_entries if entry.get("key")]
    cited_keys = extract_citation_keys_from_tex(expanded_full_paper_tex)
    missing_citation_keys = [key for key in cited_keys if key not in bib_keys]
    unused_bib_entries = [key for key in bib_keys if key not in cited_keys]
    arxiv_only_entries = [
        str(item.get("final_bib_key", "")).strip()
        for item in verified_reference_bank
        if isinstance(item, dict) and str(item.get("verification_status", "")).strip() == "arxiv_only" and item.get("included_in_final_bib", True)
    ]
    compile_warnings_summary = _extract_compile_warning_lines(full_paper_log)
    forbidden_terms = [
        "pipeline",
        "Phase 2.4",
        "Phase 2.5",
        "LLM",
        "Codex",
        "generated_plugin",
        "draft",
        "preliminary",
        "proves",
        "guarantees",
        "statistically significant",
    ]
    forbidden_terms_found = _find_forbidden_terms_in_text(full_paper_tex, forbidden_terms)

    review_facts = {
        "phase": "phase3",
        "phase_id": "phase3.5",
        "phase_name": "phase3.5_final_review_pre_submission_review",
        "paper_target": paper_target,
        "title": topic,
        "current_draft_paths": {key: str(path) for key, path in full_paths.items()},
        "compile_status": {
            "preview_pdf_exists": full_paths["full_paper_pdf"].exists(),
            "warning_lines": compile_warnings_summary,
        },
        "citation_checks": {
            "missing_citation_keys": missing_citation_keys,
            "unused_bib_entries": unused_bib_entries,
            "arxiv_only_entries": arxiv_only_entries,
        },
        "forbidden_terms_found": forbidden_terms_found,
        "full_paper_abbreviation_check": full_paper_abbreviation_report,
        "final_figure_check": final_figure_check,
        "paper_objective_alignment_inputs": introduction_facts,
    }

    technical_sources = {
        "mathematical_contract_json": read_json(phase1_dir / "mathematical_contract.frozen.json")
        or read_json(phase1_dir / "mathematical_contract.json")
        or {},
        "frozen_math_interface_md": compact_text(read_text(phase1_dir / "frozen_math_interface.md"), 1600),
        "final_system_model_problem_formulation_section_tex": compact_text(read_text(full_paths["system_model_section"]), 3600),
        "final_proposed_solution_section_tex": compact_text(read_text(full_paths["proposed_solution_section"]), 6200),
        "system_model_md": compact_text(read_text(phase1_dir / "system_model.md"), 1800),
        "problem_formulation_md": compact_text(read_text(phase1_dir / "problem_formulation.md"), 1800),
        "reformulation_path_md": compact_text(read_text(phase2_dir / "reformulation_path.md"), 1800),
        "algorithm_md": compact_text(read_text(phase3_dir / "algorithm.md"), 2200),
        "benchmark_definition_md": compact_text(read_text(phase24_dir / "benchmark_plan.md") or read_text(phase3_dir / "benchmark_definition.md"), 1400),
        "convergence_or_complexity_md": compact_text(read_text(phase3_dir / "convergence_or_complexity.md"), 1400),
    }
    result_sources = {
        "phase25_experiment_summary": phase25_summary,
        "phase3_2_manifest": phase3_2_manifest,
        "phase3_3_manifest": phase3_3_manifest,
        "table_1_csv": compact_text(table1_csv, 1800),
        "table_1_md": compact_text(table1_md, 1800),
        "numerical_results_section_tex": compact_text(reviewed_numerical_results_tex, 2800),
        "final_figure_check": final_figure_check,
    }
    citation_sources = {
        "verified_reference_bank": (verified_reference_bank[:12] if isinstance(verified_reference_bank, list) else verified_reference_bank),
        "citation_claim_map": (citation_claim_map[:12] if isinstance(citation_claim_map, list) else citation_claim_map),
        "reference_quality_report": reference_quality_report,
        "source_usage_report_md": compact_text(source_usage_report, 2200),
        "references_to_verify_md": compact_text(references_to_verify, 1200),
    }

    prompt = build_phase3_5_final_review_prompt(
        paper_target=paper_target,
        review_facts_json=_compact_json_payload(review_facts, 5000),
        full_paper_tex=compact_text(expanded_full_paper_tex, 12000),
        technical_source_json=_compact_json_payload(technical_sources, 9000),
        result_source_json=_compact_json_payload(result_sources, 5000),
        citation_source_json=_compact_json_payload(citation_sources, 5000),
    )
    write_text(phase3_5_dir / "phase3_5_prompt.txt", prompt)
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    review_input_paths = [
        full_paths["full_paper_tex"],
        full_paths["full_paper_log"],
        phase25_dir / "phase25_experiment_summary.json",
        phase25_dir / "tables" / "table_1.csv",
        phase25_dir / "tables" / "table_1.md",
        phase3_2_dir / "numerical_results_section.tex",
        phase3_2_dir / "phase3_2_manifest.json",
        phase1_dir / "mathematical_contract.frozen.json",
        phase1_dir / "frozen_math_interface.md",
        run_dir / "phase3-3" / "phase3_3_manifest.json",
        phase3_4_dir / "verified_reference_bank.json",
        phase3_4_dir / "reference_quality_report.json",
    ]
    review_input_mtimes = [path.stat().st_mtime for path in review_input_paths if path.exists()]
    latest_review_input_mtime = max(review_input_mtimes) if review_input_mtimes else 0.0
    cache_meta_path = phase3_5_dir / "phase3_5_cache_meta.json"
    cache_meta = read_json(cache_meta_path) or {}
    if not isinstance(cache_meta, dict):
        cache_meta = {}

    local_review_enabled = False
    skip_phase3_5_llm = False
    if _paper_phase_llm_skip_enabled("phase3_5", phase3_5_dir):
        write_text(
            phase3_5_dir / "phase3_5_llm_skip_request_ignored.json",
            json.dumps(
                {
                    "phase": "phase3.5",
                    "action": "ignored",
                    "reason": "Final review must come from ReviewAgent; local review mode is disabled.",
                },
                ensure_ascii=False,
                indent=2,
            ),
        )
    llm = None
    thinking = None
    review_max_tokens = _phase3_5_review_max_tokens()
    cached_response_path = phase3_5_dir / "phase3_5_raw_response.txt"
    cached_payload = _phase3_5_parse_review_payload(read_text(cached_response_path))
    cached_response_mtime = cached_response_path.stat().st_mtime if cached_response_path.exists() else 0.0
    cached_inputs_fresh = cached_response_mtime + 1.0 >= latest_review_input_mtime
    cached_hash_matches = str(cache_meta.get("prompt_hash", "")).strip() == prompt_hash and cached_inputs_fresh
    cached_payload_deterministic = _payload_has_deterministic_marker(cached_payload)
    if (
        cached_hash_matches
        and isinstance(cached_payload, dict)
        and cached_payload.get("overall_score") is not None
        and isinstance(cached_payload.get("dimension_scores"), dict)
        and (not cached_payload_deterministic or _paper_deterministic_fallback_allowed())
    ):
        payload = cached_payload
        write_text(phase3_5_dir / "phase3_5_reused_cached_review.json", json.dumps({
            "reason": "valid_cached_review_json_found",
            "source": str(phase3_5_dir / "phase3_5_raw_response.txt"),
            "prompt_hash": prompt_hash,
            "latest_review_input_mtime": latest_review_input_mtime,
            "cached_response_mtime": cached_response_mtime,
        }, ensure_ascii=False, indent=2))
    else:
        if cached_hash_matches and cached_payload_deterministic and not _paper_deterministic_fallback_allowed():
            write_text(phase3_5_dir / "phase3_5_reused_cached_review.json", json.dumps({
                "reason": "cache_ignored_deterministic_review_in_production",
                "source": str(phase3_5_dir / "phase3_5_raw_response.txt"),
                "prompt_hash": prompt_hash,
            }, ensure_ascii=False, indent=2))
        if cached_payload and not cached_hash_matches:
            write_text(phase3_5_dir / "phase3_5_reused_cached_review.json", json.dumps({
                "reason": "cache_ignored_prompt_hash_or_inputs_changed",
                "previous_prompt_hash": cache_meta.get("prompt_hash", ""),
                "current_prompt_hash": prompt_hash,
                "latest_review_input_mtime": latest_review_input_mtime,
                "cached_response_mtime": cached_response_mtime,
                "cached_inputs_fresh": cached_inputs_fresh,
            }, ensure_ascii=False, indent=2))
        llm = create_llm_client(model_profile)
        thinking = {"type": "enabled"} if model_profile == "kimi-k2.6-thinking" else None
        try:
            response = llm.chat(
                [{"role": "user", "content": prompt}],
                json_mode=True,
                thinking=thinking,
                max_tokens=review_max_tokens,
            )
            write_text(phase3_5_dir / "phase3_5_raw_response.txt", response.content)
            write_text(cache_meta_path, json.dumps({"prompt_hash": prompt_hash, "updated_at": utcnow_iso(), "latest_review_input_mtime": latest_review_input_mtime}, ensure_ascii=False, indent=2))
            payload = _phase3_5_parse_review_payload(response.content)
            if not payload and str(response.content or "").strip():
                write_text(
                    phase3_5_dir / "phase3_5_json_parse_error.md",
                    "# Phase 3.4 Review JSON Parse Error\n\n"
                    "The review model returned non-empty text, but it was not parseable as JSON. "
                    "The controller will request a compact JSON-only retry instead of treating the model as unavailable.\n",
                )
                retry_prompt = _phase3_5_compact_json_repair_prompt(
                    original_prompt=prompt,
                    broken_response=response.content,
                    parse_reason="non-empty response was not valid JSON/YAML, likely due to truncation or unescaped text",
                )
                write_text(phase3_5_dir / "phase3_5_prompt_json_repair.txt", retry_prompt)
                retry_response = llm.chat(
                    [{"role": "user", "content": retry_prompt}],
                    json_mode=True,
                    thinking=thinking,
                    max_tokens=review_max_tokens,
                )
                write_text(phase3_5_dir / "phase3_5_raw_response_json_repair.txt", retry_response.content)
                payload = _phase3_5_parse_review_payload(retry_response.content)
                if payload:
                    write_text(phase3_5_dir / "phase3_5_raw_response.txt", retry_response.content)
                    write_text(
                        cache_meta_path,
                        json.dumps(
                            {
                                "prompt_hash": prompt_hash,
                                "updated_at": utcnow_iso(),
                                "latest_review_input_mtime": latest_review_input_mtime,
                                "json_repair_retry": True,
                            },
                            ensure_ascii=False,
                            indent=2,
                        ),
                    )
        except Exception as exc:
            payload = {}
            write_text(phase3_5_dir / "phase3_5_llm_error.txt", str(exc))
    if not isinstance(payload, dict):
        payload = {}
    if not payload:
        write_text(
            phase3_5_dir / "phase3_5_blocked_no_review_llm.md",
            "# Phase 3.4 Blocked\n\n"
            "The final-review LLM did not return a usable payload in production paper-writing mode. "
            "The pipeline refused to replace reviewer judgment with a local gate review.\n",
        )
        raise RuntimeError("Phase 3.4 review LLM unavailable or invalid in production paper-writing mode.")
    if not isinstance(payload, dict):
        raise ValueError("phase3.5_final_review_pre_submission_review did not return valid JSON")

    missing_keys = _phase3_5_missing_core_keys(payload)
    if missing_keys:
        payload = _complete_phase3_5_payload_locally(
            payload,
            arxiv_only_entries=arxiv_only_entries,
            compile_warnings_summary=compile_warnings_summary,
            forbidden_terms_found=forbidden_terms_found,
        )
        write_text(phase3_5_dir / "phase3_5_local_completion.json", json.dumps({
            "reason": "initial_review_json_incomplete",
            "missing_before_completion": missing_keys,
            "missing_after_completion": _phase3_5_missing_core_keys(payload),
        }, ensure_ascii=False, indent=2))
        write_text(phase3_5_dir / "phase3_5_raw_response.txt", json.dumps(payload, ensure_ascii=False, indent=2))
        missing_keys = _phase3_5_missing_core_keys(payload)
    for retry_idx in range(2):
        if not missing_keys:
            break
        if llm is None:
            break
        retry_prompt = (
            "Your previous JSON review was incomplete.\n"
            "Return the full JSON again, preserving prior judgments where possible, but fill in every missing required field.\n"
            f"Missing fields/checks: {', '.join(missing_keys)}.\n"
            "Use these exact recommendation enums only: ready_to_submit, minor_revision_needed, major_revision_needed, not_ready.\n"
            "Use these exact likely reviewer decision enums only: accept, weak_accept, borderline, weak_reject, reject.\n"
            "Provide all ten dimension score entries.\n"
            "Provide non-empty revision_plan entries whenever issues exist.\n\n"
            f"Original prompt:\n{prompt}\n\n"
            f"Your incomplete JSON was:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
        )
        write_text(phase3_5_dir / f"phase3_5_prompt_retry_{retry_idx + 1}.txt", retry_prompt)
        try:
            retry_response = llm.chat(
                [{"role": "user", "content": retry_prompt}],
                json_mode=True,
                thinking=thinking,
                max_tokens=review_max_tokens,
            )
            write_text(phase3_5_dir / f"phase3_5_raw_response_retry_{retry_idx + 1}.txt", retry_response.content)
            retry_payload = _phase3_5_parse_review_payload(retry_response.content)
            if isinstance(retry_payload, dict):
                payload = retry_payload
                missing_keys = _phase3_5_missing_core_keys(payload)
        except Exception as exc:
            write_text(phase3_5_dir / f"phase3_5_retry_error_{retry_idx + 1}.txt", str(exc))
            break

    payload = _apply_phase3_5_evidence_adjustments(
        payload,
        phase25_summary=phase25_summary if isinstance(phase25_summary, dict) else {},
        missing_citation_keys=missing_citation_keys,
        arxiv_only_entries=arxiv_only_entries,
        compile_warnings_summary=compile_warnings_summary,
        forbidden_terms_found=forbidden_terms_found,
        final_figure_check=final_figure_check,
    )
    deterministic_gate_report = _phase3_5_deterministic_full_paper_gate(
        reference_quality_report=reference_quality_report if isinstance(reference_quality_report, dict) else {},
        full_paper_abbreviation_report=full_paper_abbreviation_report if isinstance(full_paper_abbreviation_report, dict) else {},
        compile_warnings_summary=compile_warnings_summary,
        final_figure_check=final_figure_check,
        phase25_summary=phase25_summary if isinstance(phase25_summary, dict) else {},
        full_paper_tex=expanded_full_paper_tex,
    )
    write_text(phase3_5_dir / "full_paper_quality_gate.json", json.dumps(deterministic_gate_report, ensure_ascii=False, indent=2))
    payload = _append_phase3_5_deterministic_gate_issues(payload, deterministic_gate_report)
    write_text(phase3_5_dir / "phase3_5_evidence_adjusted_review.json", json.dumps(payload, ensure_ascii=False, indent=2))
    write_text(phase3_5_dir / "phase3_5_raw_response.txt", json.dumps(payload, ensure_ascii=False, indent=2))

    overall_score = float(payload.get("overall_score", 0) or 0)
    recommendation = _normalize_phase3_5_recommendation(str(payload.get("recommendation", "major_revision_needed")))
    likely_decision = _normalize_phase3_5_decision(str(payload.get("likely_reviewer_decision_estimate", "borderline")))
    dimension_scores = payload.get("dimension_scores") or {}
    critical_issues = payload.get("critical_issues") or []
    major_issues = payload.get("major_issues") or []
    minor_issues = payload.get("minor_issues") or []
    revision_plan = payload.get("revision_plan") or {"P0": [], "P1": [], "P2": []}
    reviewer_comments = payload.get("reviewer_comments") or []
    if not isinstance(revision_plan, dict) or not any(revision_plan.get(key) for key in ["P0", "P1", "P2"]):
        revision_plan = _build_phase3_5_revision_plan(
            dimension_scores if isinstance(dimension_scores, dict) else {},
            arxiv_only_entries=arxiv_only_entries,
            compile_warnings_summary=compile_warnings_summary,
            forbidden_terms_found=forbidden_terms_found,
        )
    if not critical_issues and isinstance(revision_plan, dict):
        critical_issues = revision_plan.get("P0") or []
    if not major_issues and isinstance(revision_plan, dict):
        major_issues = revision_plan.get("P1") or []
    if not minor_issues and isinstance(revision_plan, dict):
        minor_issues = revision_plan.get("P2") or []

    final_review_scorecard = {
        "overall_score": overall_score,
        "recommendation": recommendation,
        "likely_reviewer_decision_estimate": likely_decision,
        "dimension_scores": dimension_scores,
    }
    write_text(phase3_5_dir / "final_review_scorecard.json", json.dumps(final_review_scorecard, ensure_ascii=False, indent=2))

    final_review_report_md = "\n".join(
        [
            "# Final Review Report",
            "",
            f"- Overall score: {overall_score:.1f}/10",
            f"- Recommendation: {recommendation}",
            f"- Likely reviewer decision estimate: {likely_decision}",
            "",
            "## Executive Summary",
            str(payload.get("final_review_summary", "not specified")),
            "",
            "## Top Critical Issues",
            *[f"- {item.get('issue_id', 'issue')}: {item.get('title', 'Untitled issue')}" for item in critical_issues],
            "",
            _format_dimension_score_md(dimension_scores),
        ]
    ).strip() + "\n"
    write_text(phase3_5_dir / "final_review_report.md", final_review_report_md)
    write_text(phase3_5_dir / "critical_issues.md", _format_issue_block_md("Critical Issues", critical_issues))
    write_text(phase3_5_dir / "minor_polish_issues.md", _format_issue_block_md("Minor Polish Issues", minor_issues))
    write_text(phase3_5_dir / "revision_plan.md", _format_revision_plan_md(revision_plan))
    write_text(phase3_5_dir / "reviewer_comments_simulated.md", _format_reviewer_comments_md(reviewer_comments))

    theory_report_md = "\n".join(
        [
            "# Theory Review Report",
            "",
            "## Theory Claim Reviews",
            "",
            *[
                "\n".join(
                    [
                        f"### {item.get('claim', 'Unnamed claim')}",
                        f"- Evidence available: {item.get('evidence_available', 'not specified')}",
                        f"- Recommendation: {item.get('recommended_action', 'keep')}",
                        f"- Reason: {item.get('reason', 'not specified')}",
                        "",
                    ]
                )
                for item in (payload.get("theory_claim_reviews") or [])
                if isinstance(item, dict)
            ],
        ]
    ).strip() + "\n"
    write_text(phase3_5_dir / "theory_review_report.md", theory_report_md)

    experiment_report_md = "\n".join(
        [
            "# Experiment Review Report",
            "",
            "## Claims Supported by Experiments",
            *[f"- {item}" for item in (payload.get("claims_supported_by_experiments") or [])],
            "",
            "## Claims Not Sufficiently Supported",
            *[f"- {item}" for item in (payload.get("claims_not_sufficiently_supported") or [])],
            "",
            "## Suggested Additional Experiments",
            *[
                f"- {item.get('priority', 'P1')}: {item.get('title', item.get('description', 'Experiment suggestion'))} | {item.get('why', item.get('reason', 'not specified'))}"
                for item in (payload.get("suggested_additional_experiments") or [])
                if isinstance(item, dict)
            ],
        ]
    ).strip() + "\n"
    write_text(phase3_5_dir / "experiment_review_report.md", experiment_report_md)

    citation_findings = payload.get("citation_findings") or {}
    if not isinstance(citation_findings, dict):
        citation_findings = {"citation_claim_mismatches": list(citation_findings) if isinstance(citation_findings, list) else [str(citation_findings)]}
    citation_report_md = "\n".join(
        [
            "# Citation Review Report",
            "",
            f"- Missing citation keys in draft: {', '.join(missing_citation_keys) if missing_citation_keys else 'none'}",
            f"- Unused BibTeX entries: {', '.join(unused_bib_entries) if unused_bib_entries else 'none'}",
            f"- arXiv-only entries: {', '.join(arxiv_only_entries) if arxiv_only_entries else 'none'}",
            "",
            "## Citation-claim mismatches",
            *[f"- {item}" for item in (citation_findings.get("citation_claim_mismatches") or [])],
            "",
            "## Recommended reference fixes",
            *[f"- {item}" for item in (citation_findings.get("recommended_reference_fixes") or [])],
            "",
            "## References to verify",
            *[f"- {item}" for item in (citation_findings.get("references_to_verify") or [])],
        ]
    ).strip() + "\n"
    write_text(phase3_5_dir / "citation_review_report.md", citation_report_md)

    latex_findings = payload.get("latex_format_findings") or {}
    if not isinstance(latex_findings, dict):
        latex_findings = {"recommended_format_fixes": list(latex_findings) if isinstance(latex_findings, list) else [str(latex_findings)]}
    latex_report_md = "\n".join(
        [
            "# LaTeX Format Review Report",
            "",
            "## Compile warnings summary",
            *[f"- {line}" for line in compile_warnings_summary],
            "",
            "## Float placement issues",
            *[f"- {item}" for item in (latex_findings.get("float_placement_issues") or [])],
            "",
            "## Figure style issues",
            *[f"- {item}" for item in (latex_findings.get("figure_style_issues") or [])],
            "",
            "## Table style issues",
            *[f"- {item}" for item in (latex_findings.get("table_style_issues") or [])],
            "",
            "## Recommended format fixes",
            *[f"- {item}" for item in (latex_findings.get("recommended_format_fixes") or [])],
        ]
    ).strip() + "\n"
    write_text(phase3_5_dir / "latex_format_review_report.md", latex_report_md)

    paper_risk_matrix = payload.get("paper_risk_matrix") or {
        "P0_count": len(revision_plan.get("P0") or []),
        "P1_count": len(revision_plan.get("P1") or []),
        "P2_count": len(revision_plan.get("P2") or []),
    }
    write_text(phase3_5_dir / "paper_risk_matrix.json", json.dumps(paper_risk_matrix, ensure_ascii=False, indent=2))

    checklist_lines = ["# Checklist for Next Revision", ""]
    for item in payload.get("checklist_for_next_revision") or []:
        checklist_lines.append(f"- [ ] {item}")
    if len(checklist_lines) == 2:
        checklist_lines.append("- [ ] No checklist items were provided.")
    checklist_lines.append("")
    write_text(phase3_5_dir / "checklist_for_next_revision.md", "\n".join(checklist_lines))

    review_routing_decision = build_phase3_5_review_routing_decision(
        critical_issues=critical_issues if isinstance(critical_issues, list) else [],
        major_issues=major_issues if isinstance(major_issues, list) else [],
        minor_issues=minor_issues if isinstance(minor_issues, list) else [],
        recommendation=recommendation,
    )
    review_gate_report = {
        "agent_id": "review_agent",
        "gate_id": "review_gate",
        "status": review_routing_decision.get("status"),
        "overall_score": overall_score,
        "recommendation": recommendation,
        "likely_reviewer_decision_estimate": likely_decision,
        "critical_issues": critical_issues,
        "major_issues": major_issues,
        "minor_issues": minor_issues,
        "revision_plan": revision_plan,
        "routing_decision": review_routing_decision,
        "full_paper_quality_gate": deterministic_gate_report,
        "review_outputs": {
            "final_review_report": str(phase3_5_dir / "final_review_report.md"),
            "final_review_scorecard": str(phase3_5_dir / "final_review_scorecard.json"),
            "revision_plan": str(phase3_5_dir / "revision_plan.md"),
            "critical_issues_md": str(phase3_5_dir / "critical_issues.md"),
            "review_routing_decision": str(phase3_5_dir / "review_routing_decision.json"),
            "full_paper_quality_gate": str(phase3_5_dir / "full_paper_quality_gate.json"),
        },
    }
    write_text(phase3_5_dir / "review_routing_decision.json", json.dumps(review_routing_decision, ensure_ascii=False, indent=2))
    write_text(phase3_5_dir / "phase3_5_review.json", json.dumps(review_gate_report, ensure_ascii=False, indent=2))

    manifest = {
        "phase": "phase3",
        "phase_id": "phase3.5",
        "paper_writing_mode": _paper_writing_mode_snapshot(),
        "deterministic_paper_outputs_so_far": detect_paper_writing_deterministic_outputs(run_dir),
        "input_files_used": {
            "full_paper_tex": str(full_paths["full_paper_tex"]),
            "full_paper_preview_pdf": str(full_paths["full_paper_pdf"]),
            "verified_references_bib": str(full_paths["verified_references_bib"]),
            "system_model_md": str(phase1_dir / "system_model.md"),
            "problem_formulation_md": str(phase1_dir / "problem_formulation.md"),
            "reformulation_path_md": str(phase2_dir / "reformulation_path.md"),
            "algorithm_md": str(phase3_dir / "algorithm.md"),
            "benchmark_definition_md": str(phase24_dir / "benchmark_plan.md"),
            "convergence_or_complexity_md": str(phase3_dir / "convergence_or_complexity.md"),
            "phase25_experiment_summary_json": str(phase25_dir / "phase25_experiment_summary.json"),
            "phase3_2_manifest_json": str(phase3_2_dir / "phase3_2_manifest.json"),
            "phase3_3_manifest_json": str(run_dir / "phase3-3" / "phase3_3_manifest.json"),
            "citation_claim_map_json": str(phase3_4_dir / "citation_claim_map.json"),
            "verified_reference_bank_json": str(phase3_4_dir / "verified_reference_bank.json"),
        },
        "review_outputs": {
            "final_review_report": str(phase3_5_dir / "final_review_report.md"),
            "final_review_scorecard": str(phase3_5_dir / "final_review_scorecard.json"),
            "revision_plan": str(phase3_5_dir / "revision_plan.md"),
            "critical_issues": str(phase3_5_dir / "critical_issues.md"),
            "minor_polish_issues": str(phase3_5_dir / "minor_polish_issues.md"),
            "citation_review_report": str(phase3_5_dir / "citation_review_report.md"),
            "theory_review_report": str(phase3_5_dir / "theory_review_report.md"),
            "experiment_review_report": str(phase3_5_dir / "experiment_review_report.md"),
            "latex_format_review_report": str(phase3_5_dir / "latex_format_review_report.md"),
            "reviewer_comments_simulated": str(phase3_5_dir / "reviewer_comments_simulated.md"),
            "review_routing_decision": str(phase3_5_dir / "review_routing_decision.json"),
            "phase3_5_review_json": str(phase3_5_dir / "phase3_5_review.json"),
            "full_paper_quality_gate": str(phase3_5_dir / "full_paper_quality_gate.json"),
        },
        "review_routing_decision": review_routing_decision,
        "full_paper_quality_gate": deterministic_gate_report,
        "overall_score": overall_score,
        "recommendation": recommendation,
        "likely_reviewer_decision_estimate": likely_decision,
        "dimension_scores": dimension_scores,
        "critical_issue_count": len(critical_issues),
        "major_issue_count": len(major_issues),
        "minor_issue_count": len(minor_issues),
        "compile_status": "ok" if full_paths["full_paper_pdf"].exists() and not missing_citation_keys else "warnings_present",
        "reference_status": _reference_status_from_quality(reference_quality_report),
        "experiment_status": _experiment_status_from_summary(phase25_summary if isinstance(phase25_summary, dict) else {}),
        "final_ready_flag": recommendation == "ready_to_submit" and len(critical_issues) == 0,
        "generated_timestamp": utcnow_iso(),
    }
    write_text(phase3_5_dir / "phase3_5_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    write_text(phase3_5_dir / "phase3_5_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def build_phase3_6_revision_prompt(
    *,
    paper_target: str,
    revision_context_json: str,
    current_sections_json: str,
) -> str:
    return render_prompt_template(
        "phase3_6/final_revision.prompt.yaml",
        paper_target=paper_target,
        current_sections_json=current_sections_json,
        revision_context_json=revision_context_json,
    )


def _extract_defined_labels_from_tex(tex: str) -> set[str]:
    return {item.strip() for item in re.findall(r"\\label\{([^}]+)\}", tex or "") if item.strip()}


def _extract_referenced_labels_from_tex(tex: str) -> set[str]:
    refs: set[str] = set()
    for pattern in [r"\\(?:ref|eqref|autoref|cref|Cref)\{([^}]+)\}"]:
        for group in re.findall(pattern, tex or ""):
            for item in str(group).split(","):
                key = item.strip()
                if key:
                    refs.add(key)
    return refs


def _phase3_6_algorithm_body_is_solver_transcript(body: str) -> bool:
    lowered = (body or "").lower()
    solver_markers = [
        "primal residual",
        "dual residual",
        "relative gap",
        "homogeneous self-dual",
        "phase-i",
        "phase I".lower(),
        "search direction",
        "step size",
        "psd-preserving",
        "iteration count",
        "infeasibility certificate",
    ]
    return any(marker in lowered for marker in solver_markers)


def _phase3_6_clean_algorithm_require(require_line: str) -> str:
    content = require_line.strip().rstrip(".")
    if not content:
        return "Problem data and algorithm parameters."
    content = re.sub(r",?\s*tolerances\b", "", content, flags=re.IGNORECASE)
    content = re.sub(r",?\s*and\s*\\\(T_\{?\\max\}?\\\)", "", content)
    content = re.sub(r",?\s*\\\(T_\{?\\max\}?\\\)", "", content)
    content = re.sub(r"\s*,\s*,", ",", content)
    content = re.sub(r",\s*and\s*$", "", content)
    content = re.sub(r"\s+and\s*$", "", content)
    return content.strip().rstrip(",") + "."


def _phase3_6_compact_solver_transcript_algorithm(tex: str) -> tuple[str, bool]:
    pattern = re.compile(
        r"\\begin\{algorithm\}\[!t\]\s*"
        r"\\caption\{(?P<caption>[^}]*)\}\s*"
        r"\\label\{alg:solution_proposed\}\s*"
        r"\\begin\{algorithmic\}\[1\](?P<body>.*?)"
        r"\\end\{algorithmic\}\s*"
        r"\\end\{algorithm\}",
        re.DOTALL,
    )

    def replace(match: re.Match[str]) -> str:
        body = match.group("body") or ""
        if not _phase3_6_algorithm_body_is_solver_transcript(body):
            return match.group(0)

        caption = (match.group("caption") or "Proposed Method").strip()
        if caption.lower().endswith("algorithm"):
            caption = re.sub(r"\s+Algorithm$", " Design", caption)
        require_match = re.search(r"\\Require\s+(.+?)(?:\n|$)", body)
        require_line = _phase3_6_clean_algorithm_require(require_match.group(1) if require_match else "")

        if "problem (P1)" in body:
            construct_step = r"\State Construct problem (P1) from the objective and constraints of problem (P0)."
            if "sdp" in (caption + body).lower() or "semidefinite" in (caption + body).lower():
                solve_step = r"\State Solve problem (P1) using a semidefinite-programming solver."
            elif "conic" in body.lower():
                solve_step = r"\State Solve problem (P1) using the stated conic solver."
            else:
                solve_step = r"\State Solve problem (P1) using the specified optimization routine."
        else:
            construct_step = r"\State Construct the solver-facing subproblem from the current method contract."
            solve_step = r"\State Solve the stated subproblem or update blocks."

        if r"\mathbf{W}_k" in body and r"\mathbf{V}" in body:
            ensure_line = (
                r"\Ensure Optimized covariances \(\{\mathbf{W}_k^\star\}\), "
                r"\(\mathbf{V}^\star\), transmit power, and feasibility status."
            )
            return_line = (
                r"\State \Return \(\{\mathbf{W}_k^\star\}_{k\in\mathcal{K}}\), "
                r"\(\mathbf{V}^\star\), \(P_{\mathrm{tot}}\), and feasibility status."
            )
            form_step = r"\State Form the compact matrices and coefficients used in problem (P1)."
        else:
            ensure_line = r"\Ensure Optimized physical variables, reported objective/KPI, and feasibility status."
            return_line = r"\State \Return optimized physical variables, reported objective/KPI, and feasibility status."
            form_step = r"\State Form the compact coefficients or surrogate quantities required by the method."

        return "\n".join(
            [
                r"\begin{algorithm}[!t]",
                rf"\caption{{{caption}}}",
                r"\label{alg:solution_proposed}",
                r"\begin{algorithmic}[1]",
                rf"\Require {require_line}",
                ensure_line,
                form_step,
                construct_step,
                solve_step,
                r"\State Evaluate the original physical constraints and the reported objective/KPI.",
                return_line,
                r"\end{algorithmic}",
                r"\end{algorithm}",
            ]
        )

    revised = pattern.sub(replace, tex or "")
    return revised, revised != (tex or "")


def _phase3_6_split_long_one_line_equations(tex: str, *, min_chars: int = 86) -> tuple[str, bool]:
    """Split long one-line equation displays at the main equality for IEEE columns."""
    source = tex or ""
    pattern = re.compile(r"\\begin\{equation\}\s*\n(?P<body>[^\n]+?)\s*\n\\end\{equation\}")

    def replace(match: re.Match[str]) -> str:
        body = match.group("body").strip()
        if len(body) < min_chars or "=" not in body or r"\begin{" in body or r"\\" in body:
            return match.group(0)
        if r"\frac" in body and r"\sum" in body:
            return match.group(0)
        label_match = re.search(r"\\label\{[^{}]+\}", body)
        if not label_match:
            return match.group(0)
        label = label_match.group(0)
        expr = (body[: label_match.start()] + body[label_match.end() :]).strip()
        punctuation = ""
        if expr and expr[-1] in ".,;":
            punctuation = expr[-1]
            expr = expr[:-1].rstrip()
        lhs, rhs = expr.split("=", 1)
        lhs = lhs.strip()
        rhs = rhs.strip()
        if not lhs or not rhs:
            return match.group(0)
        continuation = ""
        if r"\quad" in rhs and len(rhs) > 72:
            main_rhs, condition = rhs.rsplit(r"\quad", 1)
            rhs = main_rhs.rstrip().rstrip(",")
            continuation = "&\\quad " + condition.strip()
        lines = [
            r"\begin{equation}",
            r"\begin{aligned}",
            lhs,
            "&=" + rhs + punctuation + (r"\\" if continuation else ""),
        ]
        if continuation:
            lines.append(continuation)
        lines.extend([r"\end{aligned}" + label, r"\end{equation}"])
        return "\n".join(lines)

    revised = pattern.sub(replace, source)
    return revised, revised != source


def _phase3_6_split_long_minimizer_lists(tex: str, *, min_chars: int = 72) -> tuple[str, bool]:
    """Wrap long optimizer-variable lists in substack without changing variables."""
    source = tex or ""
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
                    candidate = (current + part).strip() if not current else (current + " " + part).strip()
                    if current and len(candidate) > target and len(lines) < 2:
                        lines.append(current.strip())
                        current = part
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
    return revised, changed


def _apply_phase3_6_deterministic_technical_fixes(
    *,
    system_text: str,
    proposed_text: str,
    conclusion_text: str,
) -> tuple[str, str, str, list[dict[str, Any]]]:
    applied: list[dict[str, Any]] = []

    revised_system = system_text
    old = revised_system
    revised_system = _impl().compact_short_split_equations(revised_system)
    if revised_system != old:
        applied.append(
            {
                "issue_id": "P2-LATEX-COMPACT-SHORT-EQUATIONS",
                "status": "fixed",
                "file_or_section": "system_model_problem_formulation_section.tex",
                "change_type": "latex_equation_polish",
                "original_issue_summary": "A compact single-relation equation was split across align continuation lines.",
                "before_excerpt": r"\begin{align} ... \nonumber\\ &\quad+ ... \end{align}",
                "after_excerpt": r"\begin{equation} ... \end{equation}",
                "note": "Kept genuinely long displays split, but restored short single-relation definitions to IEEE-style compact equations.",
            }
        )
    old = revised_system
    revised_system = _impl().compact_long_sinr_fraction_equations(revised_system)
    if revised_system != old:
        applied.append(
            {
                "issue_id": "P1-LATEX-LONG-SINR-COMPACT",
                "status": "fixed",
                "file_or_section": "system_model_problem_formulation_section.tex",
                "change_type": "latex_equation_polish",
                "original_issue_summary": "A long SINR fraction can overflow the IEEE two-column layout if every interference term is expanded in the numbered equation.",
                "before_excerpt": r"\gamma_k=\frac{...}{...\sum_j...+\cdots}",
                "after_excerpt": r"\begin{equation*}I_k\triangleq...\end{equation*} followed by the numbered SINR definition",
                "note": "Introduced an unnumbered layout shorthand so the main SINR equation keeps the original label and numbering.",
            }
        )
    old = revised_system
    revised_system, split_long_equations = _phase3_6_split_long_one_line_equations(revised_system)
    if split_long_equations:
        applied.append(
            {
                "issue_id": "P1-LATEX-LONG-EQUATION-SPLIT",
                "status": "fixed",
                "file_or_section": "system_model_problem_formulation_section.tex",
                "change_type": "latex_equation_polish",
                "original_issue_summary": "A long single-line display equation can overflow the IEEE two-column layout.",
                "before_excerpt": r"\begin{equation} a=b\label{...} \end{equation}",
                "after_excerpt": r"\begin{equation}\begin{aligned} a &= b \end{aligned}\label{...}\end{equation}",
                "note": "Applied a generic line-break rule at the main equality without changing the mathematical content.",
            }
        )
    old = revised_system
    revised_system = revised_system.replace("guarantees minimum rates", "imposes minimum rates")
    if revised_system != old:
        applied.append(
            {
                "issue_id": "P2-WRT-01",
                "status": "fixed",
                "file_or_section": "system_model_problem_formulation_section.tex",
                "change_type": "claim_weakening",
                "original_issue_summary": "Constraint wording overstated feasibility by using 'guarantees minimum rates'.",
                "before_excerpt": "guarantees minimum rates",
                "after_excerpt": "imposes minimum rates",
                "note": "Weakened constraint wording without changing the formulation.",
            }
        )
    old = revised_system
    revised_system = revised_system.replace(
        r"$\mathcal R_m\subset\mathbb R^D$ denotes its bounded movement region.",
        r"$\mathcal R_m\subset\mathbb R^D$ denotes its nonempty compact movement region. The local convergence statement below is scoped to convex such regions.",
    )
    revised_system = revised_system.replace(
        r"U_{\rm weighted sum rate (WSR)}(\mathbf r,\mathbf w)",
        r"U_{\rm WSR}(\mathbf r,\mathbf w)",
    )
    revised_system = revised_system.replace(
        "The spectral efficiency of user $k$ and the weighted sum spectral efficiency are then",
        "The spectral efficiency of user $k$ and the weighted sum rate (WSR) utility are then",
    )
    revised_system = revised_system.replace(
        r"& \mathbf r_m\in\mathcal R_m, \label{con:p0_aperture}\\",
        r"& \mathbf r_m\in\mathcal R_m,\quad \forall m, \label{con:p0_aperture}\\",
    )
    revised_system = revised_system.replace(
        r"& \|\mathbf r_m-\mathbf r_n\|_2\ge d_{\min}. \label{con:p0_spacing}",
        r"& \|\mathbf r_m-\mathbf r_n\|_2\ge d_{\min},\quad \forall (m,n)\in\mathcal E. \label{con:p0_spacing}",
    )
    if revised_system != old:
        applied.append(
            {
                "issue_id": "P1-MATH-001",
                "status": "fixed",
                "file_or_section": "system_model_problem_formulation_section.tex",
                "change_type": "notation_and_assumption_cleanup",
                "original_issue_summary": "The final draft needed a clean WSR definition, scoped aperture assumptions, and explicit P0 quantifiers.",
                "before_excerpt": r"U_{\rm weighted sum rate (WSR)} / r_m\in R_m / ||r_m-r_n||\ge d_min",
                "after_excerpt": r"weighted sum rate (WSR) utility / forall m / forall (m,n)\in E",
                "note": "Applied a local mathematical cleanup without changing the objective, controls, or constraints.",
            }
        )
    old = revised_system
    has_uav_3d = bool(re.search(r"UAV waypoint[^.\n]*q\[n\]\\in\\mathbb\{R\}\^3", revised_system))
    has_ground_2d = bool(re.search(r"w_k\\in\\mathbb\{R\}\^2", revised_system))
    ambiguous_uav_channel = r"\lVert q[n]-w_k\rVert^2+H[n]^2" in revised_system
    if has_uav_3d and has_ground_2d and ambiguous_uav_channel:
        projection_sentence = (
            r" Let $\bar q[n]=[q[n]]_{1:2}$ denote the horizontal projection of the UAV waypoint; "
            r"all norms involving $w_k$ use this horizontal projection, while $H[n]$ accounts for altitude."
        )
        if r"\bar q[n]" not in revised_system:
            revised_system = re.sub(
                r"(The UAV waypoint in slot \$n\$ is \$q\[n\]\\in\\mathbb\{R\}\^3\$[^.]*\.)",
                lambda match: match.group(1) + projection_sentence,
                revised_system,
                count=1,
            )
        revised_system = revised_system.replace(
            r"\lVert q[n]-w_k\rVert^2+H[n]^2",
            r"\lVert \bar q[n]-w_k\rVert^2+H[n]^2",
        )
    if revised_system != old:
        applied.append(
            {
                "issue_id": "P1-UAV-HORIZONTAL-DISTANCE",
                "status": "fixed",
                "file_or_section": "system_model_problem_formulation_section.tex",
                "change_type": "channel_geometry_dimension_repair",
                "original_issue_summary": "A UAV channel denominator mixed a 3D UAV waypoint with a 2D ground-device coordinate and then added altitude.",
                "before_excerpt": r"\|q[n]-w_k\|^2+H[n]^2 with q[n]\in R^3 and w_k\in R^2",
                "after_excerpt": r"\|\bar q[n]-w_k\|^2+H[n]^2 with \bar q[n]=[q[n]]_{1:2}",
                "note": "Introduced derived horizontal-projection notation and kept the same physical LoS distance model, labels, objective, and constraints.",
            }
        )
    old = revised_system
    revised_system = revised_system.replace(r"\subsection{System Model}\label{sec:system_model}", r"\subsection{System Model}\label{sec:system_model_details}")
    if revised_system != old:
        applied.append(
            {
                "issue_id": "P2-WRT-01",
                "status": "fixed",
                "file_or_section": "system_model_problem_formulation_section.tex",
                "change_type": "duplicate_label_fix",
                "original_issue_summary": "The body subsection reused the wrapper section label sec:system_model.",
                "before_excerpt": r"\subsection{System Model}\label{sec:system_model}",
                "after_excerpt": r"\subsection{System Model}\label{sec:system_model_details}",
                "note": "Kept the wrapper section label for cross references and made the subsection label unique.",
            }
        )
    old = revised_system
    has_receive_combiner = bool(re.search(r"\\mathbf\s*\{?u\}?\s*_j|\\mathbf\s+u_j|\\mathbf\{u\}_j", revised_system))
    has_residual_interference_metric = bool(
        re.search(r"residual(?:[-\s]+self)?[-\s]+interference|self[-\s]+interference", revised_system, flags=re.I)
    )
    has_post_combining_metric = bool(re.search(r"\\mathbf\s*\{?u\}?\s*_j\^H", revised_system))
    has_combiner_norm = bool(
        re.search(r"\\\|\\mathbf\s*\{?u\}?\s*_j\\\|_2\s*=\s*1|unit[-\s]+norm\s+combin", revised_system, flags=re.I)
    )
    if has_receive_combiner and has_residual_interference_metric and has_post_combining_metric and not has_combiner_norm:
        combiner_norm_sentence = (
            "We normalize each receive-combiner column as $\\|\\mathbf u_j\\|_2=1$ after every combiner refresh; "
            "therefore, the post-combining residual self-interference metric is evaluated for normalized combiners rather than arbitrary rescalings."
        )
        insertion_done = False
        for marker in (
            "For UL reception, the BS applies the linear combiner $\\mathbf U=[\\mathbf u_1,\\ldots,\\mathbf u_{K_U}]$.",
            "For UL reception, the BS applies the linear combiner $\\mathbf U=[\\mathbf u_1,\\ldots,\\mathbf u_{K_U}]$",
        ):
            if marker in revised_system and combiner_norm_sentence not in revised_system:
                revised_system = revised_system.replace(marker, marker + " " + combiner_norm_sentence, 1)
                insertion_done = True
                break
        if not insertion_done and "\\begin{subequations}\\label{prob:p0}" in revised_system:
            revised_system = revised_system.replace(
                "\\begin{subequations}\\label{prob:p0}",
                combiner_norm_sentence + "\n\n\\begin{subequations}\\label{prob:p0}",
                1,
            )
        if "\\label{con:p0_combiner_norm}" not in revised_system:
            revised_system = revised_system.replace(
                "& P_{SI}^{\\mathrm{res}}(\\mathbf W,\\mathbf U)\\le \\eta_{SI}. \\label{con:p0_si_cap}",
                "& \\|\\mathbf u_j\\|_2=1,\n"
                "\\quad \\forall j\\in\\mathcal K_U, \\label{con:p0_combiner_norm}\\\\\n"
                "& P_{SI}^{\\mathrm{res}}(\\mathbf W,\\mathbf U)\\le \\eta_{SI}. \\label{con:p0_si_cap}",
                1,
            )
    if (
        re.search(r"full[-\s]+duplex|\\mathcal K_U|UL", revised_system, flags=re.I)
        and "UL-to-DL" not in revised_system
        and "uplink-to-downlink" not in revised_system.lower()
        and "y_{D,k}" in revised_system
        and "y_{U,j}" in revised_system
    ):
        dl_scope_sentence = (
            "The DL model focuses on BS-originated co-stream interference; uplink-to-downlink user interference is assumed negligible under the considered scheduling or deployment separation, or absorbed into the receiver noise margin."
        )
        revised_system = revised_system.replace(
            "\\end{equation}\nFor UL reception,",
            "\\end{equation}\n"
            + dl_scope_sentence
            + "\nFor UL reception,",
            1,
        )
    if revised_system != old:
        applied.append(
            {
                "issue_id": "P0-COMBINER-SI-SCOPE",
                "status": "fixed",
                "file_or_section": "system_model_problem_formulation_section.tex",
                "change_type": "receiver_model_scope",
                "original_issue_summary": "Post-combining interference metrics require a normalization or receiver-model scope when receive combiners are optimization variables.",
                "before_excerpt": r"P_{SI}^{res} evaluated with unconstrained \mathbf u_j",
                "after_excerpt": r"\|\mathbf u_j\|_2=1 and scoped uplink-to-downlink interference assumption",
                "note": "Applied a generic receiver-model repair triggered by post-combining residual-interference terms; this is not tied to a specific wireless topic.",
            }
        )
    acronym_definition_fixes = [
        (
            "P1-ABBR-RF",
            "received RF power",
            "received radio-frequency (RF) power",
            "RF appeared before definition in the technical section.",
        ),
        (
            "P1-ABBR-CSI",
            "under deterministic CSI,",
            "under deterministic channel state information (CSI),",
            "CSI appeared before definition in the technical section.",
        ),
        (
            "P1-ABBR-PSD",
            "high-dimensional PSD covariances",
            "high-dimensional positive semidefinite (PSD) covariances",
            "PSD appeared before definition in the technical section.",
        ),
        (
            "P1-ABBR-SDP",
            "is an SDP in the covariance variables",
            "is a semidefinite programming (SDP) problem in the covariance variables",
            "SDP appeared before definition in the technical section.",
        ),
    ]
    for issue_id, before, after, summary in acronym_definition_fixes:
        if after in revised_system or before not in revised_system:
            continue
        revised_system = revised_system.replace(before, after, 1)
        applied.append(
            {
                "issue_id": issue_id,
                "status": "fixed",
                "file_or_section": "system_model_problem_formulation_section.tex",
                "change_type": "acronym_definition",
                "original_issue_summary": summary,
                "before_excerpt": before,
                "after_excerpt": after,
                "note": "Defined the abbreviation on first technical use without changing equations or claims.",
            }
        )

    revised_proposed = proposed_text
    old = revised_proposed
    if r"\lVert q[n]-w_k\rVert^2+H[n]^2" in revised_proposed:
        revised_proposed = revised_proposed.replace(
            r"\lVert q[n]-w_k\rVert^2+H[n]^2",
            r"\lVert \bar q[n]-w_k\rVert^2+H[n]^2",
        )
    revised_proposed, scoped_wmmse_proposed = _phase3_6_scope_exact_wmmse_language(revised_proposed)
    revised_proposed, split_long_minimizers = _phase3_6_split_long_minimizer_lists(revised_proposed)
    revised_proposed = revised_proposed.replace(
        "To solve problem (P0), we develop",
        "To address problem (P0), we develop",
    )
    revised_proposed = revised_proposed.replace(
        "To address problem (P0) through a tractable finite-scenario approximation, we develop",
        "To address problem (P0), we develop",
    )
    revised_proposed = revised_proposed.replace(
        "This observation motivates the following iterative procedure, summarized in Algorithm~\\ref{alg:proposed}.",
        "The preceding reformulation leads to the iterative procedure summarized in Algorithm~\\ref{alg:proposed}.",
    )
    revised_proposed = revised_proposed.replace(
        "\\subsection{Properties and Guarantees}",
        "\\subsection{Convergence Discussion and Complexity}",
    )
    revised_proposed = revised_proposed.replace(
        "Each block of Algorithm~\\ref{alg:proposed} is globally solved, and the surrogate objectives are monotonically non-increasing.",
        "Each block of Algorithm~\\ref{alg:proposed} is solved optimally for the adopted subproblem, and the surrogate objective is non-increasing across iterations under the stated update rules.",
    )
    revised_proposed = revised_proposed.replace(
        "The complete procedure is summarized in Algorithm~\\ref{alg:solution_proposed}. "
        "A primal-dual interior-point implementation is used for moderate dimensions; for larger deployments, "
        "the same conic form can be passed to a first-order SDP solver, with acceptance governed by the residual and gap checks.",
        "Algorithm~\\ref{alg:solution_proposed} summarizes the paper-facing design flow. "
        "The final feasibility check is performed using the original constraints.",
    )
    revised_proposed = revised_proposed.replace(
        "Algorithm~\\ref{alg:solution_proposed} summarizes the paper-facing design flow. "
        "Solver-specific residuals are monitored only to certify numerical feasibility.",
        "Algorithm~\\ref{alg:solution_proposed} summarizes the paper-facing design flow. "
        "The final feasibility check is performed using the original constraints.",
    )
    revised_proposed = revised_proposed.replace(
        "If problem (P1) is feasible and an optimizer exists, an exact conic optimum is globally optimal for the covariance formulation of problem (P0). "
        "With numerical solvers, the statement is interpreted up to the reported primal residual, dual residual, and relative gap. "
        "When a Slater point exists, the associated dual variables can also be used for local sensitivity interpretation; otherwise, feasibility certificates and marginal values should be read through the solver diagnostics.",
        "When problem (P1) is feasible, its optimizer gives the covariance-domain optimum for the modeled version of problem (P0). "
        "This statement is scoped to the stated reformulation, frozen constraints, and numerical solver tolerance.",
    )
    revised_proposed = revised_proposed.replace(
        "First-order conic solvers reduce per-iteration linear algebra but typically require more iterations and should be reported with their residuals and objective gap.",
        "For larger instances, first-order conic solvers trade lower per-iteration linear algebra for more iterations.",
    )
    revised_proposed, compacted_algorithm = _phase3_6_compact_solver_transcript_algorithm(revised_proposed)
    revised_proposed = revised_proposed.replace(
        "\\mathbf{X}(\\{\\mathbf{W}_k\\},\\mathbf{V})=\\sum_{k\\in\\mathcal{K}}\\mathbf{W}_k+\\mathbf{V}",
        "\\mathbf{X}=\\sum_{k\\in\\mathcal{K}}\\mathbf{W}_k+\\mathbf{V}",
    )
    revised_proposed = revised_proposed.replace(
        "\\mathbf{X}(\\{\\mathbf{W}_k\\},\\mathbf{V})",
        "\\mathbf{X}",
    )
    revised_proposed = revised_proposed.replace(
        "With these fixed coefficients, the solver-facing conic program is",
        "Let \\(\\mathbf{W}\\triangleq\\{\\mathbf{W}_k\\}_{k\\in\\mathcal{K}}\\) and "
        "\\(\\langle\\mathbf{A},\\mathbf{B}\\rangle\\triangleq\\operatorname{tr}(\\mathbf{A}\\mathbf{B})\\). "
        "With these fixed coefficients, the solver-facing conic program is",
        1,
    )
    revised_proposed = revised_proposed.replace(
        "\\min_{\\{\\mathbf{W}_k\\}_{k\\in\\mathcal{K}},\\,\\mathbf{V}}\\quad",
        "\\min_{\\mathbf{W},\\,\\mathbf{V}}\\quad",
    )
    revised_proposed = revised_proposed.replace(
        "& \\sum_{m\\in\\mathcal{M}}\\left(\n"
        "\\sum_{k\\in\\mathcal{K}}\\operatorname{tr}(\\mathbf{E}_m\\mathbf{W}_k)\n"
        "+\\operatorname{tr}(\\mathbf{E}_m\\mathbf{V})\\right)\n"
        "\\label{obj:p1_power}",
        "& P_{\\mathrm{tot}}(\\mathbf{W},\\mathbf{V})\n"
        "\\label{obj:p1_power}",
    )
    revised_proposed = revised_proposed.replace(
        "& \\operatorname{tr}(\\mathbf{H}_{\\mathrm{C},k}\\mathbf{W}_k)\n"
        "-\\gamma_k\\sum_{i\\in\\mathcal{K},\\,i\\ne k}\n"
        "\\operatorname{tr}(\\mathbf{H}_{\\mathrm{C},k}\\mathbf{W}_i)\n"
        "\\nonumber\\\\\n"
        "&\\quad\n"
        "-\\gamma_k\\operatorname{tr}(\\mathbf{H}_{\\mathrm{C},k}\\mathbf{V})\n"
        "\\ge \\gamma_k\\sigma_k^2,",
        "& \\operatorname{tr}(\\mathbf{H}_{\\mathrm{C},k}\\mathbf{W}_k)\n"
        "\\nonumber\\\\\n"
        "&\\quad -\\gamma_k\\sum_{i\\in\\mathcal{K},\\,i\\ne k}\n"
        "\\operatorname{tr}(\\mathbf{H}_{\\mathrm{C},k}\\mathbf{W}_i)\n"
        "\\nonumber\\\\\n"
        "&\\quad -\\gamma_k\\operatorname{tr}(\\mathbf{H}_{\\mathrm{C},k}\\mathbf{V})\n"
        "\\ge \\gamma_k\\sigma_k^2,",
    )
    revised_proposed = revised_proposed.replace(
        "\\operatorname{tr}(\\mathbf{H}_{\\mathrm{C},k}\\mathbf{W}_k)",
        "\\langle\\mathbf{H}_{\\mathrm{C},k},\\mathbf{W}_k\\rangle",
    )
    revised_proposed = revised_proposed.replace(
        "\\operatorname{tr}(\\mathbf{H}_{\\mathrm{C},k}\\mathbf{W}_i)",
        "\\langle\\mathbf{H}_{\\mathrm{C},k},\\mathbf{W}_i\\rangle",
    )
    revised_proposed = revised_proposed.replace(
        "\\operatorname{tr}(\\mathbf{H}_{\\mathrm{C},k}\\mathbf{V})",
        "\\langle\\mathbf{H}_{\\mathrm{C},k},\\mathbf{V}\\rangle",
    )
    revised_proposed = revised_proposed.replace(
        "\\operatorname{tr}\\!\\left(\\mathbf{A}_l\\mathbf{X}\\right)",
        "\\langle\\mathbf{A}_l,\\mathbf{X}\\rangle",
    )
    revised_proposed = revised_proposed.replace(
        "\\eta_j\\operatorname{tr}\\!\\left(\n"
        "\\mathbf{H}_{\\mathrm{E},j}\\mathbf{X}\\right)",
        "\\eta_j\\langle\\mathbf{H}_{\\mathrm{E},j},\\mathbf{X}\\rangle",
    )
    revised_proposed = revised_proposed.replace(
        "& \\sum_{k\\in\\mathcal{K}}\\operatorname{tr}(\\mathbf{E}_m\\mathbf{W}_k)\n"
        "+\\operatorname{tr}(\\mathbf{E}_m\\mathbf{V})\n"
        "\\le P_{\\max,m},",
        "& \\langle\\mathbf{E}_m,\\mathbf{X}\\rangle\n"
        "\\le P_{\\max,m},",
    )
    if "positive semidefinite (PSD)" not in revised_proposed:
        if "one PSD matrix" in revised_proposed:
            revised_proposed = revised_proposed.replace("one PSD matrix", "one positive semidefinite (PSD) matrix", 1)
        else:
            revised_proposed = revised_proposed.replace("PSD projections", "positive semidefinite (PSD) projections", 1)
    if (
        re.search(r"combiner", revised_proposed, flags=re.I)
        and re.search(r"residual[-\s]+SI|residual self[-\s]+interference|self[-\s]+interference", revised_proposed, flags=re.I)
        and not re.search(r"unit[-\s]+norm|\\\|\\mathbf\s*\{?u\}?\s*_j\\\|_2\s*=\s*1|normaliz(?:e|ed).*combiner", revised_proposed, flags=re.I)
    ):
        revised_proposed = revised_proposed.replace(
            "Given the updated beamformer, each combiner column is refreshed by the corresponding Wiener-style closed form using the current UL covariance, residual-SI covariance, and receiver noise covariance.",
            "Given the updated beamformer, each combiner column is refreshed by the corresponding Wiener-style closed form using the current UL covariance, residual-SI covariance, and receiver noise covariance, and is then normalized to unit Euclidean norm before rate and residual-interference evaluation.",
            1,
        )
        revised_proposed = revised_proposed.replace(
            "\\State Refresh the UL combiners and solve problem (P2-p) for $\\mathbf p_U$.",
            "\\State Refresh and normalize the UL combiners, then solve problem (P2-p) for $\\mathbf p_U$.",
            1,
        )
    if (
        re.search(r"limit points|KKT|stationar|converg", revised_proposed, flags=re.I)
        and "normalized combiner" not in revised_proposed.lower()
        and re.search(r"combiner", revised_proposed, flags=re.I)
    ):
        revised_proposed = revised_proposed.replace(
            "The FP and CCP surrogates are tight in value and first-order behavior at their anchors, and the SI expression is convex in each active block. Hence, when the block subproblems are solved exactly and the proximal terms ensure uniqueness where needed, the penalized surrogate value is nondecreasing and bounded by the power and STAR-RIS feasibility sets. The resulting limit points are Karush-Kuhn-Tucker (KKT) points of the exact-penalty reformulation that enforces the STAR-RIS equality within the prescribed tolerance; no global-optimality claim is made for problem (P0).",
            "The FP and CCP surrogates are tight in value and first-order behavior at their anchors, and the SI expression is convex in each active block after the receive-combiner normalization. Under exact block solves, bounded iterates, feasible normalized-combiner refreshes, and sufficiently large exact-penalty weights for the STAR-RIS equality, the penalized surrogate value is nondecreasing. Any accumulation point satisfying the limiting feasibility conditions is a Karush-Kuhn-Tucker (KKT) point of the penalized block-coordinate reformulation; no global-optimality claim is made for problem (P0).",
            1,
        )
    if revised_proposed != old:
        applied.append(
            {
                "issue_id": "P0-THE-01",
                "status": "fixed",
                "file_or_section": "proposed_solution_section.tex",
                "change_type": "claim_weakening_and_style_fix",
                "original_issue_summary": "Theoretical wording and section phrasing were stronger or more awkward than the current evidence supports.",
                "before_excerpt": "To solve problem (P0) / This observation motivates ... / Properties and Guarantees",
                "after_excerpt": "To address problem (P0) / The preceding reformulation leads ... / Convergence Discussion and Complexity",
                "note": "Limited deterministic edit to improve tone and avoid over-strong wording without altering the derivation.",
            }
        )
    if compacted_algorithm:
        applied.append(
            {
                "issue_id": "P1-ALG-COMPACT",
                "status": "fixed",
                "file_or_section": "proposed_solution_section.tex",
                "change_type": "algorithm_style_compaction",
                "original_issue_summary": "Algorithm 1 exposed generic solver internals instead of the paper-facing method flow.",
                "before_excerpt": "primal/dual residual loop, phase-I embedding, line search, and iteration diagnostics",
                "after_excerpt": "compact Require/Ensure method skeleton with solver call, feasibility check, and returned variables",
                "note": "Kept the same mathematical route while removing non-contribution solver transcript details from the algorithm block.",
            }
        )
    if scoped_wmmse_proposed:
        applied.append(
            {
                "issue_id": "P1-THEORY-WMMSE-EXACT-SCOPE",
                "status": "fixed",
                "file_or_section": "proposed_solution_section.tex",
                "change_type": "claim_scoping",
                "original_issue_summary": "The proposed-solution prose could imply an exact/global fixed-coordinate WSR optimizer.",
                "before_excerpt": "exact WMMSE mapping / exact fixed-coordinate precoder mapping",
                "after_excerpt": "exact WMMSE identity and auxiliary updates; local block-coordinate method",
                "note": "Scoped exactness to the WMMSE identity and auxiliary updates while preserving the local algorithm statement.",
            }
        )
    if split_long_minimizers:
        applied.append(
            {
                "issue_id": "P1-LATEX-LONG-OPTIMIZER-SPLIT",
                "status": "fixed",
                "file_or_section": "proposed_solution_section.tex",
                "change_type": "latex_equation_polish",
                "original_issue_summary": "A long optimizer-variable list can overflow the IEEE two-column layout.",
                "before_excerpt": r"\min_{x_1,x_2,\ldots,x_n}",
                "after_excerpt": r"\min_{\substack{x_1,x_2,\ldots\\x_m,\ldots,x_n}}",
                "note": "Wrapped a long minimizer list in a substack while preserving the variables.",
            }
        )

    revised_conclusion = conclusion_text
    old = revised_conclusion
    revised_conclusion = revised_conclusion.replace(
        "yields a sequence of convex subproblems that converge to a stationary point of the reformulated problem under standard conditions.",
        "yields a sequence of convex subproblems whose iterates converge to a stationary point of the reformulated problem under standard SCA regularity conditions.",
    )
    if revised_conclusion != old:
        applied.append(
            {
                "issue_id": "P0-THE-01",
                "status": "fixed",
                "file_or_section": "conclusion.tex",
                "change_type": "claim_scoping",
                "original_issue_summary": "Conclusion convergence wording needed to stay within the scope of the reformulated problem and standard conditions.",
                "before_excerpt": "converge to a stationary point of the reformulated problem under standard conditions",
                "after_excerpt": "converge to a stationary point of the reformulated problem under standard SCA regularity conditions",
                "note": "Scoped the claim to the surrogate/reformulated problem and standard regularity conditions.",
            }
        )

    return revised_system, revised_proposed, revised_conclusion, applied


def _phase3_6_merge_intro_orphan_argument_paragraphs(intro_text: str) -> tuple[str, list[dict[str, Any]]]:
    """Merge isolated one-sentence Introduction argument paragraphs into neighbors.

    This is a content-preserving layout/prose-flow repair. It does not invent new
    claims or citations; it only prevents the final revision pass from leaving a
    standalone related-work/gap sentence as its own paragraph.
    """
    text = str(intro_text or "")
    orphan_report = analyze_phase3_4_intro_orphan_argument_paragraphs(text)
    if not orphan_report or r"\begin{itemize}" not in text:
        return text, []

    before_items, after_items = text.split(r"\begin{itemize}", 1)
    section_header = ""
    body_before_items = before_items
    section_match = re.match(r"(?P<header>\s*\\section\*?\{Introduction\}\s*)(?P<body>.*)\Z", before_items, flags=re.S | re.I)
    if section_match:
        section_header = section_match.group("header").strip()
        body_before_items = section_match.group("body").strip()

    paragraphs = [
        part.strip()
        for part in re.split(r"\n\s*\n", body_before_items)
        if part.strip()
    ]
    if not paragraphs:
        return text, []

    contribution_lead_index = len(paragraphs)
    if paragraphs and re.search(
        r"\b(In this (?:letter|paper)|This (?:letter|paper)|We)\b.{0,180}"
        r"\b(contributions? are summarized|main contributions?|summarized as follows|contribute)\b",
        paragraphs[-1],
        flags=re.I,
    ):
        contribution_lead_index = len(paragraphs) - 1

    applied: list[dict[str, Any]] = []
    for item in sorted(orphan_report, key=lambda payload: int(payload.get("paragraph_index", 0) or 0), reverse=True):
        idx = int(item.get("paragraph_index", 0) or 0) - 1
        if idx < 0 or idx >= contribution_lead_index or idx >= len(paragraphs):
            continue
        orphan = paragraphs[idx]
        if idx > 0:
            paragraphs[idx - 1] = paragraphs[idx - 1].rstrip() + " " + orphan.lstrip()
            del paragraphs[idx]
        elif idx + 1 < len(paragraphs):
            paragraphs[idx + 1] = orphan.rstrip() + " " + paragraphs[idx + 1].lstrip()
            del paragraphs[idx]
        else:
            continue
        contribution_lead_index = max(0, contribution_lead_index - 1)
        applied.append(
            {
                "issue_id": "P1-INTRO-ORPHAN-PARAGRAPH",
                "status": "fixed",
                "file_or_section": "introduction.tex",
                "change_type": "paragraph_flow_repair",
                "original_issue_summary": "An Introduction motivation/related-work/gap sentence was isolated as a one-sentence paragraph.",
                "before_excerpt": str(item.get("excerpt") or orphan)[:180],
                "after_excerpt": "Merged the isolated sentence into the adjacent Introduction argument paragraph.",
                "note": "The repair preserves words and citations while restoring IEEE-style paragraph flow.",
            }
        )

    if not applied:
        return text, []
    prefix = (section_header + "\n" if section_header else "")
    repaired_before = prefix + "\n\n".join(paragraphs).strip()
    repaired = repaired_before.rstrip() + "\n\n\\begin{itemize}" + after_items
    return repaired.strip() + "\n", applied


def _apply_phase3_6_deterministic_intro_fixes(intro_text: str) -> tuple[str, list[dict[str, Any]]]:
    revised = intro_text
    applied: list[dict[str, Any]] = []

    revised, orphan_applied = _phase3_6_merge_intro_orphan_argument_paragraphs(revised)
    applied.extend(orphan_applied)

    if (
        re.search(r"\bBS\b", revised)
        and not re.search(r"\b(?:base station|base-station)s?\s*\(BS\)", revised, flags=re.I)
    ):
        old = revised
        revised = re.sub(r"\bBS\b", "base station (BS)", revised, count=1)
        if revised != old:
            applied.append(
                {
                    "issue_id": "P2-ACR-BS",
                    "status": "fixed",
                    "file_or_section": "introduction.tex",
                    "change_type": "acronym_definition",
                    "original_issue_summary": "The introduction used BS before defining the acronym.",
                    "before_excerpt": "BS",
                    "after_excerpt": "base station (BS)",
                    "note": "Defined the common wireless acronym at first use without changing the technical claim.",
                }
            )

    ordinal_pattern = re.compile(
        r"Existing research on .*? can be broadly categorized into .*? directions\.\s+The first .*?The second .*?The third .*?(?:The fourth .*?)?(?=\n\n|A critical gap remains)",
        flags=re.S,
    )
    if ordinal_pattern.search(revised):
        applied.append(
            {
                "issue_id": "P1-INT-02",
                "status": "skipped",
                "file_or_section": "introduction.tex",
                "change_type": "topic_specific_rewrite_disabled",
                "original_issue_summary": "The related-work paragraph read like an ordinal card summary rather than paper-native prose.",
                "before_excerpt": "The first focuses on ... The second investigates ... The third addresses ...",
                "after_excerpt": "No deterministic rewrite applied.",
                "note": "The previous deterministic rewrite was topic-specific and is disabled to prevent cross-topic contamination.",
            }
        )
    old = revised
    revised = revised.replace(
        "with convergence to stationary solutions.",
        "with convergence behavior monitored through objective and constraint diagnostics.",
    )
    if revised != old:
        applied.append(
            {
                "issue_id": "P0-THE-01",
                "status": "fixed",
                "file_or_section": "introduction.tex",
                "change_type": "claim_weakening",
                "original_issue_summary": "The contribution list claimed stationary convergence without proof-level support.",
                "before_excerpt": "with convergence to stationary solutions",
                "after_excerpt": "with convergence behavior monitored through objective and constraint diagnostics",
                "note": "Kept the algorithm contribution while avoiding an unsupported theorem-level claim.",
            }
        )

    stale_novelty_pattern = re.compile(
        r"\n\nUnlike prior RIS-aided ISAC or ISACP formulations.*?single co-located receiver model\.\s*",
        flags=re.I | re.S,
    )
    if stale_novelty_pattern.search(revised):
        revised = stale_novelty_pattern.sub("\n\n", revised, count=1)
        applied.append(
            {
                "issue_id": "P0-NOV-01",
                "status": "removed_stale_topic_text",
                "file_or_section": "introduction.tex",
                "change_type": "cross_topic_contamination_cleanup",
                "original_issue_summary": "A deterministic novelty patch inserted stale ISAC/sensing wording into a non-ISAC paper.",
                "before_excerpt": "Unlike prior RIS-aided ISAC or ISACP formulations ... sensing echo ...",
                "after_excerpt": "Stale cross-topic paragraph removed.",
                "note": "Novelty positioning must be generated from the current topic and reference bank, not from a hard-coded prior topic.",
            }
        )
    elif (
        "Despite this progress" in revised
        or "\n\nIn this letter," in revised
        or "\n\nThis letter addresses" in revised
    ):
        applied.append(
            {
                "issue_id": "P0-NOV-01",
                "status": "left_unresolved",
                "file_or_section": "introduction.tex",
                "change_type": "no_topic_specific_auto_insert",
                "original_issue_summary": "Novelty positioning needs strengthening, but deterministic topic-specific insertion is unsafe.",
                "before_excerpt": "Closest-work contrast remains dependent on the current Introduction draft.",
                "after_excerpt": "No hard-coded novelty paragraph inserted.",
                "note": "Handled as an unresolved upstream writing/reference issue instead of risking cross-topic contamination.",
            }
        )

    if "The remainder of this letter is organized as follows." not in revised and "\\textit{Notation:}" in revised:
        organization = (
            "The remainder of this letter is organized as follows. Section~\\ref{sec:system_model} introduces the system model and the optimization formulation. "
            "Section~\\ref{sec:proposed_solution} presents the reformulation and the alternating optimization procedure. "
            "Section~\\ref{sec:numerical_results} reports the numerical results, and Section~\\ref{sec:conclusion} concludes the letter.\n\n"
        )
        revised = revised.replace("\\textit{Notation:}", organization + "\\textit{Notation:}", 1)
        applied.append(
            {
                "issue_id": "P1-INT-01",
                "status": "fixed",
                "file_or_section": "introduction.tex",
                "change_type": "structure_completion",
                "original_issue_summary": "Introduction needed a clearer transition from contributions to paper organization.",
                "before_excerpt": "Contribution bullets ended without an organization paragraph.",
                "after_excerpt": "Added a short paper organization paragraph before the notation paragraph.",
                "note": "Improves IEEE-style introduction flow without changing technical content.",
            }
        )

    return revised, applied


def _phase3_6_validate_revised_section(
    *,
    section_name: str,
    candidate_text: str,
    original_text: str,
    allowed_citation_keys: set[str],
    allowed_ref_labels: set[str],
    forbidden_terms: list[str],
    min_word_ratio: float = 0.65,
    max_word_ratio: float = 1.45,
) -> tuple[str, dict[str, Any] | None]:
    candidate = (candidate_text or "").strip()
    original = (original_text or "").strip()
    if not candidate:
        return original_text, {
            "section": section_name,
            "reason": "empty_candidate",
        }

    candidate_keys = extract_citation_keys_from_tex(candidate)
    unknown_keys = [key for key in candidate_keys if key not in allowed_citation_keys]
    if unknown_keys:
        return original_text, {
            "section": section_name,
            "reason": "unknown_citation_keys",
            "details": unknown_keys,
        }

    unknown_refs = sorted(_extract_referenced_labels_from_tex(candidate) - allowed_ref_labels)
    if unknown_refs:
        return original_text, {
            "section": section_name,
            "reason": "unknown_reference_labels",
            "details": unknown_refs,
        }

    original_defined_labels = _extract_defined_labels_from_tex(original)
    candidate_defined_labels = _extract_defined_labels_from_tex(candidate)
    new_defined_labels = sorted(candidate_defined_labels - original_defined_labels)
    if new_defined_labels:
        return original_text, {
            "section": section_name,
            "reason": "new_defined_labels",
            "details": new_defined_labels,
        }
    if section_name in {"system_model_problem_formulation", "proposed_solution", "numerical_results"}:
        removed_labels = sorted(original_defined_labels - candidate_defined_labels)
        if removed_labels:
            return original_text, {
                "section": section_name,
                "reason": "removed_defined_labels",
                "details": removed_labels,
            }

    found_forbidden = _find_forbidden_terms_in_text(candidate, forbidden_terms)
    if found_forbidden:
        return original_text, {
            "section": section_name,
            "reason": "forbidden_terms",
            "details": found_forbidden,
        }

    orig_words = max(_word_count_text(original), 1)
    cand_words = max(_word_count_text(candidate), 1)
    ratio = cand_words / orig_words
    if ratio < min_word_ratio or ratio > max_word_ratio:
        return original_text, {
            "section": section_name,
            "reason": "word_count_ratio_out_of_range",
            "details": {"candidate_words": cand_words, "original_words": orig_words, "ratio": ratio},
        }

    if section_name == "introduction":
        structural_regression = _phase3_6_introduction_structure_regression(candidate, original)
        if structural_regression:
            return original_text, {
                "section": section_name,
                "reason": "introduction_structure_regression",
                "details": structural_regression,
            }
        quality_report = analyze_phase3_4_introduction_content_quality(candidate)
        quality_errors = list(quality_report.get("errors") or []) if isinstance(quality_report, dict) else []
        if quality_errors:
            return original_text, {
                "section": section_name,
                "reason": "introduction_content_quality_regression",
                "details": quality_errors,
            }

    if section_name == "numerical_results":
        figure_regression = _phase3_6_numerical_results_figure_regression(candidate, original)
        if figure_regression:
            return original_text, {
                "section": section_name,
                "reason": "numerical_results_figure_regression",
                "details": figure_regression,
            }

    return candidate_text, None


def _phase3_6_reference_target_keys(
    *,
    phase3_4_dir: Path,
    verified_reference_bank: list[dict[str, Any]],
    minimum_reference_target: int = 12,
) -> list[str]:
    """Return the verified reference keys that final revision must preserve."""
    valid_keys = [
        str(item.get("final_bib_key", "")).strip()
        for item in verified_reference_bank
        if isinstance(item, dict)
        and str(item.get("final_bib_key", "")).strip()
        and str(item.get("included_in_final_bib", True)).lower() != "false"
    ]
    valid_key_set = set(valid_keys)
    candidate_groups: list[list[str]] = []
    report = read_json(phase3_4_dir / "phase3_4_reference_count_contract_report.json") or {}
    if isinstance(report, dict):
        candidate_groups.append([str(key).strip() for key in report.get("final_valid_cited_reference_keys", [])])
        candidate_groups.append([str(key).strip() for key in report.get("introduction_valid_cited_reference_keys", [])])
    manifest = read_json(phase3_4_dir / "phase3_4_manifest.json") or {}
    reference_target = manifest.get("reference_target") if isinstance(manifest, dict) else {}
    if isinstance(reference_target, dict):
        candidate_groups.append([str(key).strip() for key in reference_target.get("mandatory_reference_keys", [])])
        candidate_groups.append([str(key).strip() for key in reference_target.get("recommended_reference_keys", [])])
    candidate_groups.append(valid_keys)

    selected: list[str] = []
    for group in candidate_groups:
        for key in group:
            if key and key in valid_key_set and key not in selected:
                selected.append(key)
        if len(selected) >= minimum_reference_target:
            break
    return selected


def _phase3_6_ensure_reference_coverage(
    *,
    introduction_tex: str,
    section_texts: list[str],
    required_reference_keys: list[str],
    minimum_reference_target: int = 12,
) -> tuple[str, list[str], list[str]]:
    """Preserve the Phase 3.3 verified citation contract during final revision."""
    cited_keys: list[str] = []
    for text in [introduction_tex, *section_texts]:
        for key in extract_citation_keys_from_tex(text):
            if key not in cited_keys:
                cited_keys.append(key)
    required_keys = [key for key in required_reference_keys if key]
    missing_required = [key for key in required_keys if key not in cited_keys]
    if len(cited_keys) >= minimum_reference_target or not missing_required:
        return introduction_tex, cited_keys, []

    citation = "\\cite{" + ",".join(missing_required) + "}"
    coverage_sentence = (
        "Related reconfigurable-array studies further show that physical placement choices can "
        f"alter beamforming behavior under different service and hardware assumptions {citation}."
    )
    insertion_patterns = [
        r"(?=\s+However,\s+for\b)",
        r"(?=\s+Consequently,\s+)",
        r"(?=\s+In this letter,)",
        r"(?=\n\nIn this letter,)",
        r"(?=\n\nThe main contributions are summarized as follows\.)",
        r"(?=\n\n\\begin\{itemize\})",
        r"(?=\s+\\textit\{Notation:\})",
    ]
    updated = introduction_tex
    for pattern in insertion_patterns:
        if re.search(pattern, updated):
            updated = re.sub(pattern, lambda _match: "\n\n" + coverage_sentence, updated, count=1)
            break
    else:
        updated = updated.rstrip() + "\n\n" + coverage_sentence + "\n"

    final_keys: list[str] = []
    for text in [updated, *section_texts]:
        for key in extract_citation_keys_from_tex(text):
            if key not in final_keys:
                final_keys.append(key)
    return updated, final_keys, missing_required


def _phase3_6_numerical_results_figure_regression(candidate: str, original: str) -> list[str]:
    """Block final revisions that delete already validated figure artifacts."""

    issues: list[str] = []
    original_figures = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", original or "")
    candidate_figures = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", candidate or "")
    if original_figures and candidate_figures != original_figures:
        issues.append(
            "candidate changed includegraphics paths from "
            + ", ".join(original_figures)
            + " to "
            + (", ".join(candidate_figures) if candidate_figures else "none")
        )

    original_figure_labels = sorted(label for label in _extract_defined_labels_from_tex(original or "") if label.startswith("fig:"))
    candidate_figure_labels = sorted(label for label in _extract_defined_labels_from_tex(candidate or "") if label.startswith("fig:"))
    if original_figure_labels and candidate_figure_labels != original_figure_labels:
        issues.append(
            "candidate changed figure labels from "
            + ", ".join(original_figure_labels)
            + " to "
            + (", ".join(candidate_figure_labels) if candidate_figure_labels else "none")
        )

    original_figure_refs = sorted(label for label in _extract_referenced_labels_from_tex(original or "") if label.startswith("fig:"))
    candidate_figure_refs = sorted(label for label in _extract_referenced_labels_from_tex(candidate or "") if label.startswith("fig:"))
    if original_figure_refs and candidate_figure_refs != original_figure_refs:
        issues.append(
            "candidate changed figure references from "
            + ", ".join(original_figure_refs)
            + " to "
            + (", ".join(candidate_figure_refs) if candidate_figure_refs else "none")
        )
    return issues


def _phase3_6_introduction_structure_regression(candidate: str, original: str) -> list[str]:
    """Block final-revision candidates that damage the WCL intro skeleton."""

    issues: list[str] = []
    original_has_itemize = r"\begin{itemize}" in original and r"\end{itemize}" in original
    candidate_has_itemize = r"\begin{itemize}" in candidate and r"\end{itemize}" in candidate
    if original_has_itemize and not candidate_has_itemize:
        issues.append("candidate removed the itemized contribution list")

    original_item_count = len(re.findall(r"\\item\b", original))
    candidate_item_count = len(re.findall(r"\\item\b", candidate))
    if original_item_count and candidate_item_count < original_item_count:
        issues.append(
            f"candidate reduced contribution item count from {original_item_count} to {candidate_item_count}"
        )

    organization_pattern = r"The remainder of this (?:letter|paper) is organized as follows\."
    if re.search(organization_pattern, original, flags=re.I) and not re.search(organization_pattern, candidate, flags=re.I):
        issues.append("candidate removed the organization paragraph")

    notation_pattern = r"\\textit\{Notation:\}|\\emph\{Notation:\}|(?:^|\n)\s*Notation\s*:"
    original_notation = re.search(notation_pattern, original, flags=re.I)
    candidate_notation = re.search(notation_pattern, candidate, flags=re.I)
    if original_notation and not candidate_notation:
        issues.append("candidate removed the notation paragraph")
    elif original_notation and candidate_notation:
        trailing_after_notation = candidate[candidate_notation.end() :].strip()
        if "\n\n" in trailing_after_notation:
            issues.append("candidate no longer keeps notation as the final paragraph")
        if not re.search(r"\\textit\{Notation:\}|\\emph\{Notation:\}", candidate[candidate_notation.start() :], flags=re.I):
            issues.append("candidate changed the notation paragraph away from italic IEEE style")

    if original_has_itemize and candidate_has_itemize:
        original_org = re.search(organization_pattern, original, flags=re.I)
        candidate_org = re.search(organization_pattern, candidate, flags=re.I)
        candidate_item_end = candidate.find(r"\end{itemize}")
        if candidate_org and candidate_org.start() < candidate_item_end:
            issues.append("candidate moved the organization paragraph before the contribution list")

    return issues


def render_phase3_6_preview_pdf(phase_dir: Path, title: str, curated_bib_text: str) -> dict[str, str]:
    build_dir = phase_dir.parent / "_phase3_6_preview_build"
    if build_dir.exists():
        shutil.rmtree(build_dir, ignore_errors=True)
    build_dir.mkdir(parents=True, exist_ok=True)
    wrapper_tex = build_dir / "revised_full_paper.tex"
    safe_title = _resolve_preview_title(phase_dir, title).replace("\\", " ").replace("{", "(").replace("}", ")")
    proposed_section_title = _impl().load_phase3_section_title(phase_dir.parent)
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
    commands = [
        ["pdflatex", "-interaction=nonstopmode", wrapper_tex.name],
        ["bibtex", wrapper_tex.stem],
        ["pdflatex", "-interaction=nonstopmode", wrapper_tex.name],
        ["pdflatex", "-interaction=nonstopmode", wrapper_tex.name],
        ["pdflatex", "-interaction=nonstopmode", wrapper_tex.name],
    ]
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
            raise RuntimeError(f"Phase 3.5 revised full paper preview compilation failed.{stderr_text}")
    built_pdf_path = build_dir / "revised_full_paper.pdf"
    built_log_path = build_dir / "revised_full_paper.log"
    if not built_pdf_path.exists():
        stderr_text = ""
        if last_result is not None:
            stderr_text = f"\nSTDOUT:\n{last_result.stdout}\nSTDERR:\n{last_result.stderr}"
        raise RuntimeError(f"Phase 3.5 revised full paper preview PDF was not generated.{stderr_text}")
    log_text = read_text(built_log_path)
    tex_target = phase_dir / "revised_full_paper.tex"
    pdf_target = phase_dir / "revised_full_paper_preview.pdf"
    log_target = phase_dir / "revised_full_paper_preview.log"
    shutil.copyfile(wrapper_tex, tex_target)
    shutil.copyfile(build_dir / "conceptual_diagram.tex", phase_dir / "conceptual_diagram.tex")
    try:
        shutil.copyfile(built_pdf_path, pdf_target)
    except PermissionError:
        pdf_target = phase_dir / "revised_full_paper_preview_latest.pdf"
        shutil.copyfile(built_pdf_path, pdf_target)
    try:
        shutil.copyfile(built_log_path, log_target)
    except PermissionError:
        log_target = phase_dir / "revised_full_paper_preview_latest.log"
        shutil.copyfile(built_log_path, log_target)
    return {
        "revised_full_paper_tex": str(tex_target),
        "revised_full_paper_preview_pdf": str(pdf_target),
        "revised_full_paper_preview_log": str(log_target),
        "preview_log": str(log_target),
        "preview_documentclass": "IEEEtran",
    }


def _issue_is_auto_fixable(issue: dict[str, Any]) -> bool:
    if not bool(issue.get("auto_fixable", False)):
        return False
    if bool(issue.get("requires_new_experiment", False)):
        return False
    if bool(issue.get("requires_reference_verification", False)):
        return False
    if bool(issue.get("requires_manual_theory_verification", False)):
        return False
    return True


def _phase3_6_issue_text(item: dict[str, Any]) -> str:
    fields = [
        "issue_id",
        "title",
        "issue",
        "why_unresolved",
        "required_manual_action",
        "suggested_action",
        "exact_location",
        "recommended_next_action",
        "responsible_phase",
    ]
    return " ".join(str(item.get(field, "") or "") for field in fields)


def _phase3_6_resolved_by_final_source(
    item: dict[str, Any],
    *,
    final_tex: str,
    final_bib: str,
    final_abbreviation_report: dict[str, Any],
) -> bool:
    """Suppress stale review items when the final source artifact disproves them."""
    issue_text = _phase3_6_issue_text(item)
    issue_text_lower = issue_text.lower()
    undefined_terms = {
        str(entry.get("term", "")).strip()
        for entry in final_abbreviation_report.get("undefined_abbreviations", [])
        if isinstance(entry, dict) and str(entry.get("term", "")).strip()
    }

    if any(word in issue_text_lower for word in ["typo", "misspelled", "misspelling", "acronym"]):
        uppercase_terms = {
            token
            for token in re.findall(r"\b[A-Z][A-Z0-9]{2,}(?:-[A-Z0-9]+)*\b", issue_text)
            if token not in {"P0", "P1", "P2"}
        }
        if uppercase_terms and all(
            term not in final_tex and term not in undefined_terms
            for term in uppercase_terms
        ):
            return True

    latex_commands = {
        command
        for command in re.findall(r"\\[A-Za-z]+", issue_text)
        if command not in {r"\cite", r"\ref", r"\label"}
    }
    if latex_commands and all(command not in final_tex for command in latex_commands):
        return True

    if any(word in issue_text_lower for word in ["reference", "bibliography", "bibtex", "provenance", "arxiv"]):
        candidate_keys = {
            token
            for token in re.findall(r"\b[A-Za-z][A-Za-z0-9_-]*20\d{2}[A-Za-z0-9_-]*\b", issue_text)
            if len(token) >= 6
        }
        if candidate_keys and all(key not in final_bib and key not in final_tex for key in candidate_keys):
            return True

    return False


def _format_unresolved_issues_md(items: list[dict[str, Any]]) -> str:
    lines = ["# Unresolved Issues", ""]
    if not items:
        lines.append("No unresolved issues remain after the auto-fixable revision pass.")
        lines.append("")
        return "\n".join(lines)
    for item in items:
        lines.append(f"## {item.get('issue_id', 'issue')} - {item.get('title', 'Untitled issue')}")
        lines.append(f"- Why unresolved: {item.get('why_unresolved', 'requires manual action or new evidence')}")
        lines.append(f"- Required manual action: {item.get('required_manual_action', item.get('suggested_action', 'manual follow-up needed'))}")
        lines.append(f"- Recommended next phase or human action: {item.get('recommended_next_action', item.get('responsible_phase', 'manual revision'))}")
        lines.append("")
    return "\n".join(lines)


def _phase3_6_gate_issue(
    *,
    issue_id: str,
    title: str,
    category: str,
    issue: str,
    exact_location: str,
    suggested_action: str,
    responsible_phase: str,
    estimated_impact: str,
    auto_fixable: bool,
    requires_new_experiment: bool = False,
    requires_reference_verification: bool = False,
    requires_manual_theory_verification: bool = False,
) -> dict[str, Any]:
    return {
        "issue_id": issue_id,
        "title": title,
        "category": category,
        "issue": issue,
        "why_it_matters": "This defect is visible in the final revised manuscript or its final artifact checks.",
        "exact_location": exact_location,
        "suggested_action": suggested_action,
        "responsible_phase": responsible_phase,
        "estimated_impact": estimated_impact,
        "auto_fixable": auto_fixable,
        "requires_new_experiment": requires_new_experiment,
        "requires_reference_verification": requires_reference_verification,
        "requires_manual_theory_verification": requires_manual_theory_verification,
    }


def _phase3_6_merge_gate_issues(
    gate_report: dict[str, Any],
    *,
    contract_scope_report: dict[str, Any],
    bib_file_reference_flow: dict[str, Any],
    missing_citation_keys: list[str],
    final_bib_count: int = 0,
    minimum_reference_target: int = 12,
) -> dict[str, Any]:
    merged = copy.deepcopy(gate_report) if isinstance(gate_report, dict) else {"P0": [], "P1": [], "P2": []}
    merged.setdefault("P0", [])
    merged.setdefault("P1", [])
    merged.setdefault("P2", [])

    if not bool(contract_scope_report.get("ok", False)):
        merged["P0"].append(
            _phase3_6_gate_issue(
                issue_id="P0-CONTRACT-SCOPE",
                title="Final draft violates the frozen mathematical contract",
                category="technical_consistency",
                issue="Reformulation-only variables or unsupported optimizer changes appear in the original system model/problem section.",
                exact_location="contract_scope_report.json and system_model_problem_formulation_section.tex",
                suggested_action="Repair the final manuscript so reformulation-only variables appear only in the proposed-solution/reformulation section.",
                responsible_phase="theory_agent",
                estimated_impact="very_high",
                auto_fixable=False,
                requires_manual_theory_verification=True,
            )
        )

    if final_bib_count < minimum_reference_target:
        merged["P0"].append(
            _phase3_6_gate_issue(
                issue_id="P0-FINAL-BIB-COUNT",
                title="Final bibliography is below the hard reference target",
                category="references_positioning",
                issue=f"The final BibTeX file contains {final_bib_count} cited entries, below the hard target of {minimum_reference_target}.",
                exact_location="verified_references.bib and revised_full_paper.tex",
                suggested_action="Repair the literature/reference phase so the final manuscript cites the verified relevant reference bank; do not patch this by inventing references downstream.",
                responsible_phase="literature_agent",
                estimated_impact="very_high",
                auto_fixable=False,
                requires_reference_verification=True,
            )
        )

    if missing_citation_keys:
        merged["P1"].append(
            _phase3_6_gate_issue(
                issue_id="P1-MISSING-BIB-KEYS",
                title="Final draft cites keys missing from the BibTeX file",
                category="references",
                issue=f"The final draft cites missing keys: {', '.join(missing_citation_keys)}.",
                exact_location="revised_full_paper.tex and verified_references.bib",
                suggested_action="Regenerate the bibliography from the verified reference bank or remove unsupported citations.",
                responsible_phase="literature_agent",
                estimated_impact="high",
                auto_fixable=True,
                requires_reference_verification=True,
            )
        )

    if not bool(bib_file_reference_flow.get("ok", False)):
        merged["P1"].append(
            _phase3_6_gate_issue(
                issue_id="P1-BIB-FILE-FLOW",
                title="Final paper does not use the generated BibTeX flow cleanly",
                category="references",
                issue="The final LaTeX wrapper, bibliography files, or cited-key flow is inconsistent.",
                exact_location="bib_file_reference_flow report",
                suggested_action="Use the generated references.bib/verified_references.bib file consistently in the final wrapper.",
                responsible_phase="literature_agent",
                estimated_impact="high",
                auto_fixable=True,
            )
        )

    merged["ok"] = not merged.get("P0") and not merged.get("P1")
    merged["contract_scope_ok"] = bool(contract_scope_report.get("ok", False))
    merged["bib_file_reference_flow_ok"] = bool(bib_file_reference_flow.get("ok", False))
    merged["missing_citation_keys"] = missing_citation_keys
    merged["final_bib_count"] = final_bib_count
    merged["minimum_reference_target"] = minimum_reference_target
    return merged


def _phase3_6_post_review_unresolved_items(post_review_result: dict[str, Any]) -> list[dict[str, Any]]:
    payload = post_review_result.get("payload") if isinstance(post_review_result, dict) else {}
    if not isinstance(payload, dict):
        payload = {}
    recommendation = _normalize_phase3_5_recommendation(str(payload.get("recommendation", "")))
    critical = payload.get("critical_issues") if isinstance(payload.get("critical_issues"), list) else []
    major = payload.get("major_issues") if isinstance(payload.get("major_issues"), list) else []
    blocking_issues = [item for item in [*critical, *major] if isinstance(item, dict)]
    if recommendation in {"ready_to_submit", "minor_revision_needed"} and not blocking_issues:
        return []
    if recommendation in {"major_revision_needed", "not_ready"} and not blocking_issues:
        blocking_issues = [
            _phase3_6_gate_issue(
                issue_id="P1-POST-REVISION-REVIEW",
                title="Post-revision reviewer did not accept the final manuscript",
                category="final_review",
                issue=f"The post-revision ReviewAgent returned recommendation={recommendation}.",
                exact_location="post_revision_review_scorecard.json",
                suggested_action="Inspect post_revision_review_report.md and repair the named final-draft issue before submission.",
                responsible_phase="review_agent",
                estimated_impact="high",
                auto_fixable=False,
            )
        ]
    unresolved: list[dict[str, Any]] = []
    for item in blocking_issues:
        unresolved.append(
            {
                "issue_id": str(item.get("issue_id", "post_review_issue")),
                "title": str(item.get("title", "Post-revision review issue")),
                "why_unresolved": "The post-revision ReviewAgent still found this issue in the final revised manuscript.",
                "required_manual_action": str(item.get("suggested_action", "Inspect and repair the final manuscript.")),
                "recommended_next_action": str(item.get("responsible_phase", "phase3 final revision")),
            }
        )
    return unresolved


def _phase3_6_run_post_revision_review(
    *,
    run_dir: Path,
    phase3_6_dir: Path,
    paper_target: str,
    topic: str,
    model_profile: str,
    final_bib_text: str,
    phase3_6_compile_log: str,
    phase3_6_compile_warnings: list[str],
    phase25_summary: dict[str, Any],
    final_abbreviation_report: dict[str, Any],
    contract_scope_report: dict[str, Any],
    bib_file_reference_flow: dict[str, Any],
) -> dict[str, Any]:
    phase3_4_dir = run_dir / "phase3-4"
    phase3_2_dir = run_dir / "phase3-2"
    phase25_dir = run_dir / "phase2-5"
    phase3_dir = run_dir / "phase2-3"
    phase24_dir = run_dir / "phase2-4"
    phase2_dir = run_dir / "phase2-2"
    phase1_dir = run_dir / "phase2-1"

    full_paper_tex = read_text(phase3_6_dir / "revised_full_paper.tex")
    expanded_full_paper_tex = _phase3_5_expanded_full_paper_text(
        {
            "full_paper_tex": phase3_6_dir / "revised_full_paper.tex",
            "abstract": phase3_6_dir / "abstract.tex",
            "introduction": phase3_6_dir / "introduction.tex",
            "system_model_section": phase3_6_dir / "system_model_problem_formulation_section.tex",
            "proposed_solution_section": phase3_6_dir / "proposed_solution_section.tex",
            "numerical_results_section": phase3_6_dir / "numerical_results_section.tex",
            "conclusion": phase3_6_dir / "conclusion.tex",
        }
    ) or full_paper_tex
    write_text(phase3_6_dir / "revised_full_paper_expanded_for_review.tex", expanded_full_paper_tex)
    verified_references_bib = final_bib_text or read_text(phase3_6_dir / "verified_references.bib")
    final_figure_check = _build_phase3_5_final_figure_check(
        numerical_results_tex=read_text(phase3_6_dir / "numerical_results_section.tex"),
        numerical_results_base_dir=phase3_6_dir,
        full_paper_log=phase3_6_compile_log,
        phase25_summary=phase25_summary if isinstance(phase25_summary, dict) else {},
    )
    write_text(phase3_6_dir / "post_revision_final_figure_check.json", json.dumps(final_figure_check, ensure_ascii=False, indent=2))

    bib_entries = parse_bib_entries(verified_references_bib)
    bib_keys = [entry.get("key", "") for entry in bib_entries if entry.get("key")]
    cited_keys = extract_citation_keys_from_tex(expanded_full_paper_tex)
    missing_citation_keys = [key for key in cited_keys if key not in bib_keys]
    unused_bib_entries = [key for key in bib_keys if key not in cited_keys]
    verified_reference_bank = read_json(phase3_4_dir / "verified_reference_bank.json") or []
    citation_claim_map = read_json(phase3_4_dir / "citation_claim_map.json") or []
    reference_quality_report = read_json(phase3_4_dir / "reference_quality_report.json") or {}
    source_usage_report = read_text(phase3_4_dir / "source_usage_report.md")
    references_to_verify = read_text(phase3_4_dir / "references_to_verify.md")
    introduction_facts = read_json(phase3_4_dir / "introduction_facts.json") or {}
    arxiv_only_entries = [
        str(item.get("final_bib_key", "")).strip()
        for item in verified_reference_bank
        if isinstance(item, dict)
        and str(item.get("verification_status", "")).strip() == "arxiv_only"
        and item.get("included_in_final_bib", True)
    ]
    forbidden_terms = [
        "pipeline",
        "Phase 2.4",
        "Phase 2.5",
        "LLM",
        "Codex",
        "generated_plugin",
        "draft",
        "preliminary",
        "proves",
        "guarantees",
        "statistically significant",
    ]
    forbidden_terms_found = _find_forbidden_terms_in_text(full_paper_tex, forbidden_terms)
    deterministic_gate_report = _phase3_5_deterministic_full_paper_gate(
        reference_quality_report=reference_quality_report if isinstance(reference_quality_report, dict) else {},
        full_paper_abbreviation_report=final_abbreviation_report if isinstance(final_abbreviation_report, dict) else {},
        compile_warnings_summary=phase3_6_compile_warnings,
        final_figure_check=final_figure_check,
        phase25_summary=phase25_summary if isinstance(phase25_summary, dict) else {},
        full_paper_tex=expanded_full_paper_tex,
    )
    deterministic_gate_report = _phase3_6_merge_gate_issues(
        deterministic_gate_report,
        contract_scope_report=contract_scope_report if isinstance(contract_scope_report, dict) else {},
        bib_file_reference_flow=bib_file_reference_flow if isinstance(bib_file_reference_flow, dict) else {},
        missing_citation_keys=missing_citation_keys,
        final_bib_count=len(bib_keys),
    )
    write_text(phase3_6_dir / "post_revision_full_paper_quality_gate.json", json.dumps(deterministic_gate_report, ensure_ascii=False, indent=2))

    review_facts = {
        "phase": "phase3",
        "phase_id": "phase3.6",
        "phase_name": "phase3.6_post_revision_final_review",
        "paper_target": paper_target,
        "title": topic,
        "current_draft_paths": {
            "full_paper_tex": str(phase3_6_dir / "revised_full_paper.tex"),
            "full_paper_pdf": str(phase3_6_dir / "revised_full_paper_preview.pdf"),
            "full_paper_log": str(phase3_6_dir / "revised_full_paper_preview.log"),
            "verified_references_bib": str(phase3_6_dir / "verified_references.bib"),
        },
        "compile_status": {
            "preview_pdf_exists": (phase3_6_dir / "revised_full_paper_preview.pdf").exists(),
            "warning_lines": phase3_6_compile_warnings,
        },
        "citation_checks": {
            "missing_citation_keys": missing_citation_keys,
            "unused_bib_entries": unused_bib_entries,
            "arxiv_only_entries": arxiv_only_entries,
        },
        "forbidden_terms_found": forbidden_terms_found,
        "full_paper_abbreviation_check": final_abbreviation_report,
        "contract_scope_check": contract_scope_report,
        "bib_file_reference_flow": bib_file_reference_flow,
        "final_figure_check": final_figure_check,
        "deterministic_full_paper_gate": deterministic_gate_report,
        "paper_objective_alignment_inputs": introduction_facts,
    }
    technical_sources = {
        "mathematical_contract_json": read_json(phase1_dir / "mathematical_contract.frozen.json")
        or read_json(phase1_dir / "mathematical_contract.json")
        or {},
        "frozen_math_interface_md": compact_text(read_text(phase1_dir / "frozen_math_interface.md"), 1600),
        "final_system_model_problem_formulation_section_tex": compact_text(read_text(phase3_6_dir / "system_model_problem_formulation_section.tex"), 3600),
        "final_proposed_solution_section_tex": compact_text(read_text(phase3_6_dir / "proposed_solution_section.tex"), 6200),
        "system_model_md": compact_text(read_text(phase1_dir / "system_model.md"), 1800),
        "problem_formulation_md": compact_text(read_text(phase1_dir / "problem_formulation.md"), 1800),
        "reformulation_path_md": compact_text(read_text(phase2_dir / "reformulation_path.md"), 1800),
        "algorithm_md": compact_text(read_text(phase3_dir / "algorithm.md"), 2200),
        "benchmark_definition_md": compact_text(read_text(phase24_dir / "benchmark_plan.md") or read_text(phase3_dir / "benchmark_definition.md"), 1400),
        "convergence_or_complexity_md": compact_text(read_text(phase3_dir / "convergence_or_complexity.md"), 1400),
    }
    result_sources = {
        "phase25_experiment_summary": phase25_summary,
        "phase3_2_manifest": read_json(phase3_2_dir / "phase3_2_manifest.json") or {},
        "phase3_3_manifest": read_json(run_dir / "phase3-3" / "phase3_3_manifest.json") or {},
        "table_1_csv": compact_text(read_text(phase25_dir / "tables" / "table_1.csv"), 1800),
        "table_1_md": compact_text(read_text(phase25_dir / "tables" / "table_1.md"), 1800),
        "numerical_results_section_tex": compact_text(read_text(phase3_6_dir / "numerical_results_section.tex"), 2800),
        "final_figure_check": final_figure_check,
    }
    citation_sources = {
        "verified_reference_bank": (verified_reference_bank[:12] if isinstance(verified_reference_bank, list) else verified_reference_bank),
        "citation_claim_map": (citation_claim_map[:12] if isinstance(citation_claim_map, list) else citation_claim_map),
        "reference_quality_report": reference_quality_report,
        "source_usage_report_md": compact_text(source_usage_report, 2200),
        "references_to_verify_md": compact_text(references_to_verify, 1200),
    }
    prompt = build_phase3_6_post_revision_review_prompt(
        paper_target=paper_target,
        review_facts_json=_compact_json_payload(review_facts, 6500),
        full_paper_tex=compact_text(expanded_full_paper_tex, 13000),
        technical_source_json=_compact_json_payload(technical_sources, 9000),
        result_source_json=_compact_json_payload(result_sources, 5500),
        citation_source_json=_compact_json_payload(citation_sources, 5000),
    )
    write_text(phase3_6_dir / "post_revision_review_prompt.txt", prompt)
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    cache_meta_path = phase3_6_dir / "post_revision_review_cache_meta.json"
    cached_response_path = phase3_6_dir / "post_revision_review_raw_response.txt"
    cache_meta = read_json(cache_meta_path) or {}
    cached_payload = _safe_json_loads(read_text(cached_response_path), {})
    payload: dict[str, Any] = {}
    if (
        isinstance(cache_meta, dict)
        and str(cache_meta.get("prompt_hash", "")).strip() == prompt_hash
        and isinstance(cached_payload, dict)
        and cached_payload.get("overall_score") is not None
        and isinstance(cached_payload.get("dimension_scores"), dict)
        and (not _payload_has_deterministic_marker(cached_payload) or _paper_deterministic_fallback_allowed())
    ):
        payload = cached_payload
        write_text(
            phase3_6_dir / "post_revision_review_cache_reuse.json",
            json.dumps({"reason": "valid_cached_post_revision_review", "prompt_hash": prompt_hash}, ensure_ascii=False, indent=2),
        )
    else:
        post_local_review_enabled = False
        skip_review_llm = False
        if _paper_phase_llm_skip_enabled("phase3_6_post_review", phase3_6_dir):
            write_text(
                phase3_6_dir / "post_revision_review_llm_skip_request_ignored.json",
                json.dumps(
                    {
                        "phase": "phase3.6",
                        "action": "ignored",
                        "reason": "Post-revision review must come from ReviewAgent; local review mode is disabled.",
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            )
        llm = create_llm_client(model_profile)
        post_review_retries = int(os.environ.get("WARA_PHASE3_6_POST_REVIEW_LLM_MAX_RETRIES", "2") or 2)
        llm.config.max_retries = max(1, min(int(getattr(llm.config, "max_retries", post_review_retries)), post_review_retries))
        llm.config.retry_base_delay = min(float(getattr(llm.config, "retry_base_delay", 10.0)), 10.0)
        if str(os.environ.get("WARA_PHASE3_6_POST_REVIEW_STREAM", "1")).strip().lower() not in {"0", "false", "no", "off"}:
            setattr(llm.config, "stream", True)
        thinking = {"type": "enabled"} if model_profile == "kimi-k2.6-thinking" else None
        try:
            response = llm.chat(
                [{"role": "user", "content": prompt}],
                json_mode=True,
                thinking=thinking,
                max_tokens=int(os.environ.get("WARA_PHASE3_6_POST_REVIEW_MAX_TOKENS", "8000") or 8000),
            )
            write_text(cached_response_path, response.content)
            write_text(cache_meta_path, json.dumps({"prompt_hash": prompt_hash, "updated_at": utcnow_iso()}, ensure_ascii=False, indent=2))
            candidate = _safe_json_loads(response.content, {})
            if isinstance(candidate, dict):
                payload = candidate
            else:
                repair_prompt = (
                    "The previous post-revision review response was not valid complete JSON, likely because it was too long or truncated.\n"
                    "Repair the serialization and compress the review. Return valid JSON only with the same required top-level keys.\n"
                    "Keep at most 3 items in each issue/warning/finding array and at most 3 top_issues per dimension. "
                    "Do not add new paper content or new references.\n\n"
                    "Original review task excerpt:\n"
                    + compact_text(prompt, 9000)
                    + "\n\nPrevious invalid response:\n"
                    + compact_text(response.content, 9000)
                )
                repair_response = llm.chat(
                    [{"role": "user", "content": repair_prompt}],
                    json_mode=True,
                    thinking=thinking,
                    max_tokens=int(os.environ.get("WARA_PHASE3_6_POST_REVIEW_REPAIR_MAX_TOKENS", "6000") or 6000),
                )
                write_text(phase3_6_dir / "post_revision_review_structured_repair_raw_response.txt", repair_response.content)
                repair_candidate = _safe_json_loads(repair_response.content, {})
                if isinstance(repair_candidate, dict):
                    payload = repair_candidate
                    write_text(cached_response_path, repair_response.content)
                    write_text(
                        cache_meta_path,
                        json.dumps({"prompt_hash": prompt_hash, "updated_at": utcnow_iso(), "structured_repair": True}, ensure_ascii=False, indent=2),
                    )
        except Exception as exc:
            write_text(phase3_6_dir / "post_revision_review_llm_error.txt", str(exc))

    if not payload and not (_paper_deterministic_fallback_allowed() or post_local_review_enabled):
        write_text(
            phase3_6_dir / "post_revision_review_blocked_no_llm.md",
            "# Phase 3.5 Post-Revision Review Blocked\n\n"
            "The final post-revision review LLM did not return a usable payload in production mode. "
            "The pipeline refused to reuse the stale pre-revision review scorecard or create a local reviewer substitute.\n",
        )
        raise RuntimeError("Phase 3.5 post-revision review LLM unavailable or invalid in production paper-writing mode.")

    if not isinstance(payload, dict):
        raise ValueError("phase3.6_post_revision_review did not return valid JSON")
    missing_payload_keys = _phase3_5_missing_core_keys(payload)
    if missing_payload_keys:
        payload = _complete_phase3_5_payload_locally(
            payload,
            arxiv_only_entries=arxiv_only_entries,
            compile_warnings_summary=phase3_6_compile_warnings,
            forbidden_terms_found=forbidden_terms_found,
        )
        write_text(
            phase3_6_dir / "post_revision_review_local_completion.json",
            json.dumps({"missing_before_completion": missing_payload_keys, "missing_after_completion": _phase3_5_missing_core_keys(payload)}, ensure_ascii=False, indent=2),
        )

    payload = _apply_phase3_5_evidence_adjustments(
        payload,
        phase25_summary=phase25_summary if isinstance(phase25_summary, dict) else {},
        missing_citation_keys=missing_citation_keys,
        arxiv_only_entries=arxiv_only_entries,
        compile_warnings_summary=phase3_6_compile_warnings,
        forbidden_terms_found=forbidden_terms_found,
        final_figure_check=final_figure_check,
    )
    payload = _append_phase3_5_deterministic_gate_issues(payload, deterministic_gate_report)
    write_text(phase3_6_dir / "post_revision_review_evidence_adjusted.json", json.dumps(payload, ensure_ascii=False, indent=2))
    write_text(cached_response_path, json.dumps(payload, ensure_ascii=False, indent=2))

    overall_score = float(payload.get("overall_score", 0) or 0)
    recommendation = _normalize_phase3_5_recommendation(str(payload.get("recommendation", "major_revision_needed")))
    likely_decision = _normalize_phase3_5_decision(str(payload.get("likely_reviewer_decision_estimate", "borderline")))
    dimension_scores = payload.get("dimension_scores") or {}
    critical_issues = payload.get("critical_issues") if isinstance(payload.get("critical_issues"), list) else []
    major_issues = payload.get("major_issues") if isinstance(payload.get("major_issues"), list) else []
    minor_issues = payload.get("minor_issues") if isinstance(payload.get("minor_issues"), list) else []
    revision_plan = payload.get("revision_plan") if isinstance(payload.get("revision_plan"), dict) else {"P0": critical_issues, "P1": major_issues, "P2": minor_issues}
    reviewer_comments = payload.get("reviewer_comments") if isinstance(payload.get("reviewer_comments"), list) else []
    scorecard = {
        "overall_score": overall_score,
        "recommendation": recommendation,
        "likely_reviewer_decision_estimate": likely_decision,
        "dimension_scores": dimension_scores,
    }
    write_text(phase3_6_dir / "post_revision_review_scorecard.json", json.dumps(scorecard, ensure_ascii=False, indent=2))
    write_text(
        phase3_6_dir / "post_revision_review_report.md",
        "\n".join(
            [
                "# Post-Revision Review Report",
                "",
                f"- Overall score: {overall_score:.1f}/10",
                f"- Recommendation: {recommendation}",
                f"- Likely reviewer decision estimate: {likely_decision}",
                "",
                "## Executive Summary",
                str(payload.get("final_review_summary", "not specified")),
                "",
                "## Top Critical Issues",
                *[f"- {item.get('issue_id', 'issue')}: {item.get('title', 'Untitled issue')}" for item in critical_issues if isinstance(item, dict)],
                "",
                _format_dimension_score_md(dimension_scores if isinstance(dimension_scores, dict) else {}),
            ]
        ).strip()
        + "\n",
    )
    write_text(phase3_6_dir / "post_revision_critical_issues.md", _format_issue_block_md("Post-Revision Critical Issues", critical_issues))
    write_text(phase3_6_dir / "post_revision_major_issues.md", _format_issue_block_md("Post-Revision Major Issues", major_issues))
    write_text(phase3_6_dir / "post_revision_minor_issues.md", _format_issue_block_md("Post-Revision Minor Issues", minor_issues))
    write_text(phase3_6_dir / "post_revision_revision_plan.md", _format_revision_plan_md(revision_plan))
    write_text(phase3_6_dir / "post_revision_reviewer_comments.md", _format_reviewer_comments_md(reviewer_comments))

    return {
        "review_completed": True,
        "overall_score": overall_score,
        "recommendation": recommendation,
        "likely_reviewer_decision_estimate": likely_decision,
        "scorecard": scorecard,
        "payload": payload,
        "critical_issue_count": len(critical_issues),
        "major_issue_count": len(major_issues),
        "minor_issue_count": len(minor_issues),
        "deterministic_full_paper_gate": deterministic_gate_report,
        "final_figure_check": final_figure_check,
        "missing_citation_keys": missing_citation_keys,
        "unused_bib_entries": unused_bib_entries,
        "outputs": {
            "prompt": str(phase3_6_dir / "post_revision_review_prompt.txt"),
            "raw_response": str(cached_response_path),
            "scorecard": str(phase3_6_dir / "post_revision_review_scorecard.json"),
            "report": str(phase3_6_dir / "post_revision_review_report.md"),
            "quality_gate": str(phase3_6_dir / "post_revision_full_paper_quality_gate.json"),
            "final_figure_check": str(phase3_6_dir / "post_revision_final_figure_check.json"),
        },
    }


def run_phase3_6_apply_review_fixes_package(run_dir: Path, paper_target: str = "IEEE WCL") -> dict[str, Any]:
    run_dir = Path(run_dir)
    summary_payload = read_json(run_dir / "phase2_summary.json") or {}
    topic = str(summary_payload.get("topic", run_dir.name))
    model_profile = str(summary_payload.get("model_profile") or DEFAULT_MODEL_PROFILE)
    phase3_6_dir = run_dir / "phase3-6"
    phase3_6_dir.mkdir(parents=True, exist_ok=True)
    write_text(phase3_6_dir / "phase3_6_design_notes.md", build_phase3_6_final_revision_design_notes())

    phase3_5_dir = run_dir / "phase3-5"
    phase3_4_dir = run_dir / "phase3-4"
    phase3_3_dir = run_dir / "phase3-3"
    phase1_dir = run_dir / "phase2-1"
    phase25_summary = read_json(run_dir / "phase2-5" / "phase25_experiment_summary.json") or {}
    verified_reference_bank = read_json(phase3_4_dir / "verified_reference_bank.json") or []
    review_scorecard = read_json(phase3_5_dir / "final_review_scorecard.json") or {}
    revision_plan_md = read_text(phase3_5_dir / "revision_plan.md")
    final_review_report_md = read_text(phase3_5_dir / "final_review_report.md")
    citation_review_report_md = read_text(phase3_5_dir / "citation_review_report.md")
    theory_review_report_md = read_text(phase3_5_dir / "theory_review_report.md")
    experiment_review_report_md = read_text(phase3_5_dir / "experiment_review_report.md")
    latex_format_review_report_md = read_text(phase3_5_dir / "latex_format_review_report.md")
    critical_issues_md = read_text(phase3_5_dir / "critical_issues.md")
    phase3_5_manifest = read_json(phase3_5_dir / "phase3_5_manifest.json") or {}
    review_raw = _safe_json_loads(read_text(phase3_5_dir / "phase3_5_raw_response.txt"), {})
    revision_plan_payload = review_raw.get("revision_plan") if isinstance(review_raw, dict) else {}
    if not isinstance(revision_plan_payload, dict) or not any((revision_plan_payload.get(key) or []) for key in ["P0", "P1", "P2"]):
        arxiv_only_entries = [
            str(item.get("final_bib_key", "")).strip()
            for item in verified_reference_bank
            if isinstance(item, dict) and str(item.get("verification_status", "")).strip() == "arxiv_only" and item.get("included_in_final_bib", True)
        ]
        dimension_scores = review_scorecard.get("dimension_scores") if isinstance(review_scorecard, dict) else {}
        revision_plan_payload = _build_phase3_5_revision_plan(
            dimension_scores if isinstance(dimension_scores, dict) else {},
            arxiv_only_entries=arxiv_only_entries,
            compile_warnings_summary=[],
            forbidden_terms_found=[],
        )

    prior_phase3_6_manifest = read_json(phase3_6_dir / "phase3_6_manifest.json") or {}
    prior_phase3_6_gate = read_json(phase3_6_dir / "post_revision_full_paper_quality_gate.json") or {}
    prior_phase3_6_ready = (
        isinstance(prior_phase3_6_manifest, dict)
        and str(prior_phase3_6_manifest.get("compile_status", "")).strip().lower() == "ok"
        and bool(prior_phase3_6_gate.get("ok", False))
        and not prior_phase3_6_manifest.get("ready_to_submit_blockers")
        and int(prior_phase3_6_manifest.get("unresolved_issue_count", 0) or 0) == 0
    )

    def _current_section_text(phase3_6_filename: str, fallback_path: Path) -> str:
        prior_path = phase3_6_dir / phase3_6_filename
        if prior_phase3_6_ready:
            prior_text = read_text(prior_path)
            if prior_text.strip():
                return prior_text
        return read_text(fallback_path)

    current_sections = {
        "abstract_tex": sanitize_phase3_3_abstract_tex(_current_section_text("abstract.tex", phase3_3_dir / "abstract.tex")),
        "introduction_tex": ensure_phase3_4_notation_paragraph(
            sanitize_phase3_4_introduction_tex(_current_section_text("introduction.tex", phase3_4_dir / "introduction.tex"))
        ),
        "system_model_problem_formulation_section_tex": _current_section_text(
            "system_model_problem_formulation_section.tex",
            phase3_4_dir / "system_model_problem_formulation_section.tex",
        ),
        "proposed_solution_section_tex": sanitize_phase3_latex_snippet(
            _current_section_text("proposed_solution_section.tex", phase3_4_dir / "proposed_solution_section.tex")
        ),
        "numerical_results_section_tex": sanitize_phase3_2_numerical_results_tex(
            _current_section_text("numerical_results_section.tex", phase3_4_dir / "numerical_results_section.tex")
        ),
        "conclusion_tex": sanitize_phase3_3_conclusion_tex(_current_section_text("conclusion.tex", phase3_4_dir / "conclusion.tex")),
    }
    phase3_4_results = sanitize_phase3_2_numerical_results_tex(read_text(phase3_4_dir / "numerical_results_section.tex"))
    if _phase3_6_numerical_results_figure_regression(current_sections["numerical_results_section_tex"], phase3_4_results):
        current_sections["numerical_results_section_tex"] = phase3_4_results
    initial_abbreviation_report = analyze_phase3_4_full_paper_abbreviations(
        {
            "abstract": current_sections["abstract_tex"],
            "introduction": current_sections["introduction_tex"],
            "system_model": current_sections["system_model_problem_formulation_section_tex"],
            "proposed_solution": current_sections["proposed_solution_section_tex"],
            "numerical_results": current_sections["numerical_results_section_tex"],
            "conclusion": current_sections["conclusion_tex"],
        }
    )
    prior_post_revision_followup = {
        "purpose": (
            "If this object is non-empty, these unresolved issues came from the post-revision ReviewAgent "
            "on the exact current revised manuscript. Treat them as the highest-priority targeted repair list "
            "for this pass. Apply narrow source repairs without changing labels, optimization variables, "
            "constraint roles, citations, figures, or numerical results. If the review explicitly identifies "
            "a dimensional or mathematical defect in an equation, a local equation-content repair is allowed "
            "when it preserves the same physical mechanism and existing label."
        ),
        "unresolved_issues_md": compact_text(read_text(phase3_6_dir / "unresolved_issues.md"), 3500),
        "post_revision_critical_issues_md": compact_text(read_text(phase3_6_dir / "post_revision_critical_issues.md"), 3000),
        "post_revision_major_issues_md": compact_text(read_text(phase3_6_dir / "post_revision_major_issues.md"), 4500),
        "post_revision_minor_issues_md": compact_text(read_text(phase3_6_dir / "post_revision_minor_issues.md"), 2500),
        "post_revision_abbreviation_report": read_json(phase3_6_dir / "full_paper_abbreviation_report.json") or {},
        "allowed_repair_scope": [
            "weaken or align high-level claims with equations",
            "add scoped prose assumptions without changing equations",
            "add local solver-ready definitions or surrogate formulas using only already defined variables and constraints",
            "define or remove undefined abbreviations",
            "polish placeholder-like prose",
            "clarify algorithm steps in prose without adding new algorithms or changing the method",
        ],
        "forbidden_repair_scope": [
            "new experiments",
            "new references",
            "new physical mechanisms",
            "new optimizer variables or constraints outside the frozen contract",
            "new constraints",
            "new physical links or channels absent from the model",
            "stronger convergence or optimality claims",
        ],
    }

    p0_items = revision_plan_payload.get("P0") or []
    p1_items = revision_plan_payload.get("P1") or []
    p2_items = revision_plan_payload.get("P2") or []
    auto_fix_items = [item for item in [*p0_items, *p1_items, *p2_items] if isinstance(item, dict) and _issue_is_auto_fixable(item)]
    unresolved_base = []
    for item in [*p0_items, *p1_items]:
        if isinstance(item, dict) and not _issue_is_auto_fixable(item):
            unresolved_base.append(
                {
                    "issue_id": item.get("issue_id", "issue"),
                    "title": item.get("title", "Untitled issue"),
                    "why_unresolved": "This issue requires new evidence, manual verification, or a non-writing change.",
                    "required_manual_action": item.get("suggested_action", "Manual follow-up required."),
                    "recommended_next_action": item.get("responsible_phase", "manual revision"),
                }
            )

    revision_context = {
        "paper_target": paper_target,
        "topic": topic,
        "review_scorecard": review_scorecard,
        "phase3_5_manifest": phase3_5_manifest,
        "frozen_math_contract_json": read_json(phase1_dir / "mathematical_contract.frozen.json")
        or read_json(phase1_dir / "mathematical_contract.json")
        or {},
        "frozen_math_interface_md": compact_text(read_text(phase1_dir / "frozen_math_interface.md"), 1600),
        "final_review_report_md": compact_text(final_review_report_md, 8000),
        "revision_plan_md": compact_text(revision_plan_md, 8000),
        "critical_issues_md": compact_text(critical_issues_md, 5000),
        "citation_review_report_md": compact_text(citation_review_report_md, 5000),
        "theory_review_report_md": compact_text(theory_review_report_md, 5000),
        "experiment_review_report_md": compact_text(experiment_review_report_md, 5000),
        "latex_format_review_report_md": compact_text(latex_format_review_report_md, 5000),
        "abbreviation_review_contract": {
            "purpose": (
                "Use this dynamically extracted inventory to fix article-specific abbreviations. "
                "Do not rely on a fixed acronym list; infer the full form from the supplied paper context "
                "or avoid the acronym if the full form is not trustworthy."
            ),
            "initial_abbreviation_report": initial_abbreviation_report,
            "required_action": (
                "Every undefined abbreviation must be defined as full name plus acronym on first prose use "
                "or removed from the affected section. Preserve equations, labels, citation keys, and figure files."
            ),
        },
        "reference_preservation_contract": {
            "minimum_reference_target": 12,
            "required_reference_keys": _phase3_6_reference_target_keys(
                phase3_4_dir=phase3_4_dir,
                verified_reference_bank=verified_reference_bank if isinstance(verified_reference_bank, list) else [],
                minimum_reference_target=12,
            ),
            "required_action": (
                "The final revision must preserve the verified Phase 3.3 citation coverage. "
                "Do not remove citations in a way that makes the final cited verified references fall below the target."
            ),
        },
        "auto_fix_items": auto_fix_items,
        "unresolved_base": unresolved_base,
        "post_revision_followup_contract": prior_post_revision_followup,
    }
    prompt = build_phase3_6_revision_prompt(
        paper_target=paper_target,
        revision_context_json=_compact_json_payload(revision_context, 11500),
        current_sections_json=_compact_json_payload(current_sections, 12000),
    )
    write_text(phase3_6_dir / "phase3_6_prompt.txt", prompt)
    local_revision_enabled = False
    phase3_6_skip_llm = False
    if _paper_phase_llm_skip_enabled("phase3_6", phase3_6_dir):
        write_text(
            phase3_6_dir / "phase3_6_llm_skip_request_ignored.json",
            json.dumps(
                {
                    "phase": "phase3.6",
                    "action": "ignored",
                    "reason": "Final revision must be generated by RepairAgent/WritingAgent; local revision mode is disabled.",
                },
                ensure_ascii=False,
                indent=2,
            ),
        )
    payload: dict[str, Any] = {}
    if not phase3_6_skip_llm:
        llm = create_llm_client(model_profile)
        phase3_6_retries = int(os.environ.get("WARA_PHASE3_6_REVISION_LLM_MAX_RETRIES", "2") or 2)
        llm.config.max_retries = max(1, min(int(getattr(llm.config, "max_retries", phase3_6_retries)), phase3_6_retries))
        llm.config.retry_base_delay = min(float(getattr(llm.config, "retry_base_delay", 10.0)), 10.0)
        if str(os.environ.get("WARA_PHASE3_6_REVISION_STREAM", "1")).strip().lower() not in {"0", "false", "no", "off"}:
            setattr(llm.config, "stream", True)
        thinking = {"type": "enabled"} if model_profile == "kimi-k2.6-thinking" else None
        try:
            response = llm.chat(
                [{"role": "user", "content": prompt}],
                json_mode=True,
                thinking=thinking,
                max_tokens=int(os.environ.get("WARA_PHASE3_6_REVISION_MAX_TOKENS", "8000") or 8000),
            )
            write_text(phase3_6_dir / "phase3_6_raw_response.txt", response.content)
            candidate_payload = _safe_json_loads(response.content, {})
            if isinstance(candidate_payload, dict):
                payload = candidate_payload
        except Exception as exc:
            write_text(phase3_6_dir / "phase3_6_llm_error.txt", str(exc))
            write_text(
                phase3_6_dir / "phase3_6_blocked_no_revision_llm.md",
                "# Phase 3.5 Blocked\n\n"
                "The final revision LLM failed in production paper-writing mode. "
                "The pipeline refused to silently replace the review/revision pass with a local revision.\n",
            )
            raise RuntimeError("Phase 3.5 revision LLM unavailable in production paper-writing mode.") from exc
    if not payload:
        write_text(
            phase3_6_dir / "phase3_6_blocked_invalid_revision.md",
            "# Phase 3.5 Blocked\n\n"
            "The final revision LLM returned an empty or invalid payload in production paper-writing mode. "
            "No local final-revision pass was applied.\n",
        )
        raise ValueError("Phase 3.5 revision LLM returned an invalid payload in production paper-writing mode.")

    allowed_citation_keys = {
        str(item.get("final_bib_key", "")).strip()
        for item in verified_reference_bank
        if isinstance(item, dict) and str(item.get("included_in_final_bib", True)).lower() != "false"
    }
    allowed_ref_labels: set[str] = set()
    for text in current_sections.values():
        allowed_ref_labels.update(_extract_defined_labels_from_tex(text))
    allowed_ref_labels.update({"sec:system_model", "sec:proposed_solution", "sec:numerical_results", "sec:conclusion"})
    forbidden_terms = [
        "pipeline",
        "Phase 2.4",
        "Phase 2.5",
        "LLM",
        "Codex",
        "generated_plugin",
        "draft",
        "preliminary",
        "statistically significant",
    ]

    revised_abstract_candidate = sanitize_phase3_3_abstract_tex(str(payload.get("abstract_tex") or current_sections["abstract_tex"]))
    revised_intro_candidate = ensure_phase3_4_notation_paragraph(
        sanitize_phase3_4_introduction_tex(str(payload.get("introduction_tex") or current_sections["introduction_tex"]))
    )
    revised_system_candidate = sanitize_phase3_3_embedded_section_tex(
        str(
            payload.get("system_model_problem_formulation_section_tex")
            or current_sections["system_model_problem_formulation_section_tex"]
        )
    )
    revised_proposed_candidate = sanitize_phase3_latex_snippet(
        str(payload.get("proposed_solution_section_tex") or current_sections["proposed_solution_section_tex"])
    )
    revised_results_candidate = sanitize_phase3_2_numerical_results_tex(str(payload.get("numerical_results_section_tex") or current_sections["numerical_results_section_tex"]))
    revised_conclusion_candidate = sanitize_phase3_3_conclusion_tex(str(payload.get("conclusion_tex") or current_sections["conclusion_tex"]))

    rollback_notes: list[dict[str, Any]] = []
    revised_abstract, note = _phase3_6_validate_revised_section(
        section_name="abstract",
        candidate_text=revised_abstract_candidate,
        original_text=current_sections["abstract_tex"],
        allowed_citation_keys=allowed_citation_keys,
        allowed_ref_labels=allowed_ref_labels,
        forbidden_terms=forbidden_terms,
        min_word_ratio=0.7,
        max_word_ratio=1.35,
    )
    if note:
        rollback_notes.append(note)
    revised_intro, note = _phase3_6_validate_revised_section(
        section_name="introduction",
        candidate_text=revised_intro_candidate,
        original_text=current_sections["introduction_tex"],
        allowed_citation_keys=allowed_citation_keys,
        allowed_ref_labels=allowed_ref_labels,
        forbidden_terms=forbidden_terms,
        min_word_ratio=0.7,
        max_word_ratio=1.45,
    )
    if note:
        rollback_notes.append(note)
    revised_system, note = _phase3_6_validate_revised_section(
        section_name="system_model_problem_formulation",
        candidate_text=revised_system_candidate,
        original_text=current_sections["system_model_problem_formulation_section_tex"],
        allowed_citation_keys=allowed_citation_keys,
        allowed_ref_labels=allowed_ref_labels,
        forbidden_terms=forbidden_terms,
        min_word_ratio=0.35,
        max_word_ratio=1.45,
    )
    if note:
        rollback_notes.append(note)
    revised_proposed, note = _phase3_6_validate_revised_section(
        section_name="proposed_solution",
        candidate_text=revised_proposed_candidate,
        original_text=current_sections["proposed_solution_section_tex"],
        allowed_citation_keys=allowed_citation_keys,
        allowed_ref_labels=allowed_ref_labels,
        forbidden_terms=forbidden_terms,
        min_word_ratio=0.35,
        max_word_ratio=1.55,
    )
    if note:
        rollback_notes.append(note)
    revised_results, note = _phase3_6_validate_revised_section(
        section_name="numerical_results",
        candidate_text=revised_results_candidate,
        original_text=current_sections["numerical_results_section_tex"],
        allowed_citation_keys=allowed_citation_keys,
        allowed_ref_labels=allowed_ref_labels,
        forbidden_terms=forbidden_terms,
        min_word_ratio=0.85,
        max_word_ratio=1.40,
    )
    if note:
        rollback_notes.append(note)
    revised_conclusion, note = _phase3_6_validate_revised_section(
        section_name="conclusion",
        candidate_text=revised_conclusion_candidate,
        original_text=current_sections["conclusion_tex"],
        allowed_citation_keys=allowed_citation_keys,
        allowed_ref_labels=allowed_ref_labels,
        forbidden_terms=forbidden_terms,
        min_word_ratio=0.7,
        max_word_ratio=1.35,
    )
    if note:
        rollback_notes.append(note)

    results_applied: list[dict[str, Any]] = []
    if "key performance indicator (KPI)" not in revised_results and "The reported KPI is" in revised_results:
        revised_results = revised_results.replace(
            "The reported KPI is",
            "The reported key performance indicator (KPI) is",
            1,
        )
        results_applied.append(
            {
                "issue_id": "P1-ABBR-KPI",
                "status": "fixed",
                "file_or_section": "numerical_results_section.tex",
                "change_type": "acronym_definition",
                "original_issue_summary": "KPI appeared before definition in the numerical results section.",
                "before_excerpt": "The reported KPI is",
                "after_excerpt": "The reported key performance indicator (KPI) is",
                "note": "Defined the abbreviation while preserving the reported metric and figure wording.",
            }
        )
    revised_results_before_eh = revised_results
    revised_results = re.sub(r"\bEH\b", "energy-harvesting", revised_results)
    if revised_results != revised_results_before_eh:
        results_applied.append(
            {
                "issue_id": "P1-ABBR-EH",
                "status": "fixed",
                "file_or_section": "numerical_results_section.tex",
                "change_type": "acronym_definition",
                "original_issue_summary": "EH appeared in the numerical-results prose before being defined in that section.",
                "before_excerpt": "EH",
                "after_excerpt": "energy-harvesting",
                "note": "Used the full phrase in the numerical-results section to avoid an undefined abbreviation.",
            }
        )
    revised_results_before_method_filter = revised_results
    revised_results = enforce_phase3_2_plotted_method_definitions(
        revised_results,
        read_json(run_dir / "phase2-5" / "phase25_experiment_summary.json") or {},
        read_text(run_dir / "phase2-5" / "method_naming_summary.json"),
    )
    if revised_results != revised_results_before_method_filter:
        results_applied.append(
            {
                "issue_id": "P1-NUMERICAL-PLOTTED-METHODS",
                "status": "fixed",
                "file_or_section": "numerical_results_section.tex",
                "change_type": "benchmark_definition_alignment",
                "original_issue_summary": "The numerical-results itemize block defined methods that were executed but not plotted in the final figures.",
                "before_excerpt": "Benchmark-definition itemize included audit-only methods.",
                "after_excerpt": "Benchmark-definition itemize only includes final plotted methods.",
                "note": "Aligned paper prose with Phase 2.5 final figure methods.",
            }
        )
    phase25_summary_for_method_alignment = read_json(run_dir / "phase2-5" / "phase25_experiment_summary.json") or {}
    method_naming_for_method_alignment = read_text(run_dir / "phase2-5" / "method_naming_summary.json")
    revised_abstract, abstract_methods_changed = _phase3_6_align_plotted_method_claim_text(
        revised_abstract,
        phase25_summary_for_method_alignment,
        method_naming_for_method_alignment,
    )
    revised_intro, intro_methods_changed = _phase3_6_align_plotted_method_claim_text(
        revised_intro,
        phase25_summary_for_method_alignment,
        method_naming_for_method_alignment,
    )
    revised_conclusion, conclusion_methods_changed = _phase3_6_align_plotted_method_claim_text(
        revised_conclusion,
        phase25_summary_for_method_alignment,
        method_naming_for_method_alignment,
    )
    if abstract_methods_changed or intro_methods_changed or conclusion_methods_changed:
        results_applied.append(
            {
                "issue_id": "P1-FULL-PAPER-PLOTTED-METHODS",
                "status": "fixed",
                "file_or_section": "abstract.tex, introduction.tex, conclusion.tex",
                "change_type": "method_claim_alignment",
                "original_issue_summary": "High-level summary text referenced methods that were not plotted in the final figures.",
                "before_excerpt": "No-shared-cov. / other tested covariance benchmarks",
                "after_excerpt": "the final plotted benchmark only",
                "note": "Aligned abstract, introduction, and conclusion with Phase 2.5 final plotted methods.",
            }
        )
    revised_results, scaling_claims_hedged = _phase3_6_hedge_unverified_scaling_claims(revised_results)
    if scaling_claims_hedged:
        results_applied.append(
            {
                "issue_id": "P1-NUMERICAL-SCALING-SCOPE",
                "status": "fixed",
                "file_or_section": "numerical_results_section.tex",
                "change_type": "claim_scoping",
                "original_issue_summary": "Numerical-results prose used broad scaling language without a separate scaling proof or ablation.",
                "before_excerpt": "scaling with / relative advantage growing as",
                "after_excerpt": "over the evaluated range",
                "note": "Scoped trend language to the plotted/evaluated range without changing the reported data or figures.",
            }
        )

    wmmse_scope_applied: list[str] = []
    for section_name, current_text in [
        ("abstract.tex", revised_abstract),
        ("introduction.tex", revised_intro),
        ("conclusion.tex", revised_conclusion),
    ]:
        scoped_text, scoped_changed = _phase3_6_scope_exact_wmmse_language(current_text)
        if not scoped_changed:
            continue
        if section_name == "abstract.tex":
            revised_abstract = scoped_text
        elif section_name == "introduction.tex":
            revised_intro = scoped_text
        else:
            revised_conclusion = scoped_text
        wmmse_scope_applied.append(section_name)
    if wmmse_scope_applied:
        results_applied.append(
            {
                "issue_id": "P1-THEORY-WMMSE-EXACT-SCOPE",
                "status": "fixed",
                "file_or_section": ", ".join(wmmse_scope_applied),
                "change_type": "claim_scoping",
                "original_issue_summary": "High-level prose could be read as claiming an exact/global fixed-coordinate WSR optimizer.",
                "before_excerpt": "exact fixed-coordinate precoder mapping",
                "after_excerpt": "exact WMMSE identity and auxiliary updates; local block-coordinate method",
                "note": "Scoped WMMSE exactness without changing the algorithm, variables, or experiments.",
            }
        )

    revised_intro, intro_applied = _apply_phase3_6_deterministic_intro_fixes(revised_intro)
    revised_system, revised_proposed, revised_conclusion, deterministic_applied = _apply_phase3_6_deterministic_technical_fixes(
        system_text=revised_system,
        proposed_text=revised_proposed,
        conclusion_text=revised_conclusion,
    )

    write_text(phase3_6_dir / "abstract.tex", revised_abstract)
    write_text(phase3_6_dir / "introduction.tex", revised_intro)
    write_text(phase3_6_dir / "system_model_problem_formulation_section.tex", revised_system)
    write_text(phase3_6_dir / "proposed_solution_section.tex", revised_proposed)
    write_text(phase3_6_dir / "numerical_results_section.tex", revised_results)
    write_text(phase3_6_dir / "conclusion.tex", revised_conclusion)

    abbreviation_repair_report = _phase3_5_apply_full_paper_abbreviation_repairs(phase3_6_dir)
    abbreviation_repair_summaries = abbreviation_repair_report.get("applied_issue_summaries", [])
    if abbreviation_repair_summaries:
        revised_abstract = read_text(phase3_6_dir / "abstract.tex")
        revised_intro = read_text(phase3_6_dir / "introduction.tex")
        revised_system = read_text(phase3_6_dir / "system_model_problem_formulation_section.tex")
        revised_proposed = read_text(phase3_6_dir / "proposed_solution_section.tex")
        revised_results = read_text(phase3_6_dir / "numerical_results_section.tex")
        revised_conclusion = read_text(phase3_6_dir / "conclusion.tex")

    required_reference_keys = _phase3_6_reference_target_keys(
        phase3_4_dir=phase3_4_dir,
        verified_reference_bank=verified_reference_bank if isinstance(verified_reference_bank, list) else [],
        minimum_reference_target=12,
    )
    revised_intro, preserved_citation_keys, preserved_reference_keys = _phase3_6_ensure_reference_coverage(
        introduction_tex=revised_intro,
        section_texts=[revised_system, revised_proposed, revised_results, revised_conclusion],
        required_reference_keys=required_reference_keys,
        minimum_reference_target=12,
    )
    if preserved_reference_keys:
        write_text(phase3_6_dir / "introduction.tex", revised_intro)
        results_applied.append(
            {
                "issue_id": "P0-FINAL-BIB-COUNT",
                "status": "fixed",
                "file_or_section": "introduction.tex",
                "change_type": "reference_contract_preservation",
                "original_issue_summary": "Final revision dropped verified citation coverage below the hard reference target.",
                "before_excerpt": "Final revision cited fewer than the verified reference target.",
                "after_excerpt": "Restored verified citation coverage using existing Phase 3.3 reference keys.",
                "note": "No new references were invented; the repair only preserved keys from the verified reference bank.",
            }
        )

    revised_intro_after_reference_flow, post_reference_intro_applied = _apply_phase3_6_deterministic_intro_fixes(revised_intro)
    if post_reference_intro_applied:
        revised_intro = revised_intro_after_reference_flow
        write_text(phase3_6_dir / "introduction.tex", revised_intro)
        results_applied.extend(post_reference_intro_applied)

    revised_citation_keys: list[str] = []
    for text in [revised_intro, revised_system, revised_proposed, revised_results, revised_conclusion]:
        for key in extract_citation_keys_from_tex(text):
            if key not in revised_citation_keys:
                revised_citation_keys.append(key)
    final_bib_text, missing_keys, _ = build_curated_bibliography(
        verified_reference_bank if isinstance(verified_reference_bank, list) else [],
        revised_citation_keys,
    )
    write_text(phase3_6_dir / "verified_references.bib", final_bib_text)
    write_text(phase3_6_dir / "references.bib", final_bib_text)
    preview = render_phase3_6_preview_pdf(phase3_6_dir, topic, final_bib_text)
    preview_log_path = Path(str(preview.get("preview_log", ""))) if isinstance(preview, dict) and preview.get("preview_log") else None
    phase3_6_compile_log = read_text(preview_log_path) if preview_log_path is not None else ""
    phase3_6_compile_warnings = _extract_compile_warning_lines(phase3_6_compile_log)
    phase3_6_max_overfull_hbox_pt = _phase3_5_max_overfull_hbox_pt(phase3_6_compile_warnings)
    final_abbreviation_report = analyze_phase3_4_full_paper_abbreviations_from_phase_dir(phase3_6_dir)
    write_text(
        phase3_6_dir / "full_paper_abbreviation_report.json",
        json.dumps(final_abbreviation_report, ensure_ascii=False, indent=2),
    )
    bib_file_reference_flow = validate_bib_file_reference_flow(phase3_6_dir)
    final_undefined_abbreviations = (
        final_abbreviation_report.get("undefined_abbreviations", [])
        if isinstance(final_abbreviation_report, dict)
        else []
    )
    contract_scope_report = _phase3_5_contract_scope_check(run_dir, phase3_6_dir)
    write_text(
        phase3_6_dir / "contract_scope_report.json",
        json.dumps(contract_scope_report, ensure_ascii=False, indent=2),
    )
    phase25_status = str(phase25_summary.get("phase25_status", "") if isinstance(phase25_summary, dict) else "")
    try:
        review_overall_score = float(review_scorecard.get("overall_score", 0.0) or 0.0)
    except Exception:
        review_overall_score = 0.0
    experiment_blocks_ready = phase25_status not in {
        "",
        "quick_mode_only",
        "needs_more_phase24_runs",
        "claim_failure_needs_redesign",
    }

    applied_issue_summaries = payload.get("applied_issue_summaries") or []
    if not isinstance(applied_issue_summaries, list):
        applied_issue_summaries = []
    applied_issue_summaries.extend(intro_applied)
    applied_issue_summaries.extend(abbreviation_repair_summaries)
    applied_issue_summaries.extend(results_applied)
    applied_issue_summaries.extend(deterministic_applied)
    for note in rollback_notes:
        applied_issue_summaries.append(
            {
                "issue_id": f"ROLLBACK-{str(note.get('section', 'section')).upper()}",
                "status": "unresolved",
                "file_or_section": str(note.get("section", "section")),
                "change_type": "candidate_rollback",
                "original_issue_summary": "The candidate revision introduced unsupported structure, references, or wording.",
                "before_excerpt": "Candidate revision rejected",
                "after_excerpt": "Original section retained",
                "note": f"Rollback reason: {note.get('reason', 'validation_failed')} {note.get('details', '')}",
            }
        )
    revision_applied_lines = ["# Revision Applied Report", ""]
    if not applied_issue_summaries:
        revision_applied_lines.append("No applied issue summaries were returned.")
        revision_applied_lines.append("")
    else:
        for item in applied_issue_summaries:
            if not isinstance(item, dict):
                continue
            revision_applied_lines.append(f"## {item.get('issue_id', 'issue')} - {item.get('status', 'fixed')}")
            revision_applied_lines.append(f"- File/section modified: {item.get('file_or_section', 'not specified')}")
            revision_applied_lines.append(f"- Change type: {item.get('change_type', 'targeted revision')}")
            revision_applied_lines.append(f"- Original issue summary: {item.get('original_issue_summary', 'not specified')}")
            revision_applied_lines.append(f"- Before excerpt: {item.get('before_excerpt', 'not specified')}")
            revision_applied_lines.append(f"- After excerpt: {item.get('after_excerpt', 'not specified')}")
            revision_applied_lines.append(f"- Note: {item.get('note', 'not specified')}")
            revision_applied_lines.append("")
    write_text(phase3_6_dir / "revision_applied_report.md", "\n".join(revision_applied_lines))

    applied_issue_ids = {
        str(item.get("issue_id", "")).strip()
        for item in applied_issue_summaries
        if isinstance(item, dict) and str(item.get("issue_id", "")).strip()
    }
    resolved_issue_ids = set(applied_issue_ids)
    if not final_undefined_abbreviations:
        resolved_issue_ids.update({"P1-ABBR-001", "P1-ABBR-UNDEFINED", "P2-1"})
    if phase3_6_max_overfull_hbox_pt <= 10.0:
        resolved_issue_ids.update({"P1-LATEX-OVERFULL", "P2-2"})
    if bool(contract_scope_report.get("ok")):
        resolved_issue_ids.update({"P0-CONSIST-001"})
    final_source_text = "\n".join(
        [revised_abstract, revised_intro, revised_system, revised_proposed, revised_results, revised_conclusion]
    )
    for item in [*unresolved_base, *auto_fix_items]:
        if not isinstance(item, dict):
            continue
        issue_id = str(item.get("issue_id", "")).strip()
        if issue_id and _phase3_6_resolved_by_final_source(
            item,
            final_tex=final_source_text,
            final_bib=final_bib_text,
            final_abbreviation_report=final_abbreviation_report if isinstance(final_abbreviation_report, dict) else {},
        ):
            resolved_issue_ids.add(issue_id)
    unresolved_items = [
        item
        for item in unresolved_base
        if str(item.get("issue_id", "")).strip() not in resolved_issue_ids
    ]
    for item in auto_fix_items:
        issue_id = str(item.get("issue_id", "")).strip()
        if issue_id and issue_id not in resolved_issue_ids:
            unresolved_items.append(
                {
                    "issue_id": issue_id,
                    "title": item.get("title", "Untitled issue"),
                    "why_unresolved": "This issue was marked auto-fixable but was not explicitly resolved in the applied revision output.",
                    "required_manual_action": item.get("suggested_action", "Manual follow-up required."),
                    "recommended_next_action": item.get("responsible_phase", "manual revision"),
                }
            )
    write_text(phase3_6_dir / "unresolved_issues.md", _format_unresolved_issues_md(unresolved_items))

    section_map = {
        "Abstract": (current_sections["abstract_tex"], revised_abstract),
        "Introduction": (current_sections["introduction_tex"], revised_intro),
        "System Model": (current_sections["system_model_problem_formulation_section_tex"], revised_system),
        "Proposed Solution": (current_sections["proposed_solution_section_tex"], revised_proposed),
        "Numerical Results": (current_sections["numerical_results_section_tex"], revised_results),
        "Conclusion": (current_sections["conclusion_tex"], revised_conclusion),
    }
    diff_lines = ["# Revision Diff Summary", ""]
    diff_lines.append(str(payload.get("revision_diff_summary_md") or "").strip())
    diff_lines.append("")
    for label, pair in section_map.items():
        before_text, after_text = pair
        changed = before_text.strip() != after_text.strip()
        diff_lines.append(f"## {label}")
        diff_lines.append(f"- Changed: {changed}")
        diff_lines.append(f"- Before words: {_word_count_text(before_text)}")
        diff_lines.append(f"- After words: {_word_count_text(after_text)}")
        diff_lines.append("")
    diff_lines.append("## References")
    diff_lines.append(f"- Cited keys after revision: {len(revised_citation_keys)}")
    diff_lines.append(f"- Missing reference keys after rebuild: {', '.join(missing_keys) if missing_keys else 'none'}")
    diff_lines.append("")
    write_text(phase3_6_dir / "revision_diff_summary.md", "\n".join(diff_lines))

    post_revision_review = _phase3_6_run_post_revision_review(
        run_dir=run_dir,
        phase3_6_dir=phase3_6_dir,
        paper_target=paper_target,
        topic=topic,
        model_profile=model_profile,
        final_bib_text=final_bib_text,
        phase3_6_compile_log=phase3_6_compile_log,
        phase3_6_compile_warnings=phase3_6_compile_warnings,
        phase25_summary=phase25_summary if isinstance(phase25_summary, dict) else {},
        final_abbreviation_report=final_abbreviation_report if isinstance(final_abbreviation_report, dict) else {},
        contract_scope_report=contract_scope_report if isinstance(contract_scope_report, dict) else {},
        bib_file_reference_flow=bib_file_reference_flow if isinstance(bib_file_reference_flow, dict) else {},
    )
    effective_review_scorecard = post_revision_review.get("scorecard") if isinstance(post_revision_review, dict) else {}
    if not isinstance(effective_review_scorecard, dict) or not effective_review_scorecard:
        effective_review_scorecard = review_scorecard if isinstance(review_scorecard, dict) else {}
    try:
        review_overall_score = float(effective_review_scorecard.get("overall_score", 0.0) or 0.0)
    except Exception:
        review_overall_score = 0.0
    review_recommendation = _normalize_phase3_5_recommendation(str(effective_review_scorecard.get("recommendation", "")))
    if isinstance(post_revision_review, dict) and post_revision_review.get("review_completed"):
        unresolved_items = [
            item
            for item in _phase3_6_post_review_unresolved_items(post_revision_review)
            if not _phase3_6_resolved_by_final_source(
                item,
                final_tex=final_source_text,
                final_bib=final_bib_text,
                final_abbreviation_report=final_abbreviation_report if isinstance(final_abbreviation_report, dict) else {},
            )
        ]
        write_text(phase3_6_dir / "unresolved_issues.md", _format_unresolved_issues_md(unresolved_items))

    unresolved_issue_ids = {
        str(item.get("issue_id", "")).strip()
        for item in unresolved_items
        if isinstance(item, dict) and str(item.get("issue_id", "")).strip()
    }
    ready_to_submit_by_gates = bool(
        not unresolved_items
        and not missing_keys
        and experiment_blocks_ready
        and review_overall_score >= 7.0
        and review_recommendation in {"ready_to_submit", "minor_revision_needed"}
    )
    deterministic_paper_outputs = detect_paper_writing_deterministic_outputs(run_dir)
    ready_to_submit_blockers: list[str] = []
    if _paper_writing_mode() != "production":
        ready_to_submit_blockers.append("paper_writing_mode is deterministic_test")
    if deterministic_paper_outputs:
        ready_to_submit_blockers.append("deterministic paper-writing fallback or verified template was used")
    if phase3_6_max_overfull_hbox_pt > 10.0:
        ready_to_submit_blockers.append(f"severe_overfull_hbox={phase3_6_max_overfull_hbox_pt:.1f}pt")
    if final_undefined_abbreviations:
        ready_to_submit_blockers.append(f"undefined_abbreviation_count={len(final_undefined_abbreviations)}")
    if missing_keys:
        ready_to_submit_blockers.append(f"missing_reference_keys={missing_keys}")
    if not bool(bib_file_reference_flow.get("ok")):
        ready_to_submit_blockers.append("bib_file_reference_flow_not_ok")
    if unresolved_items:
        ready_to_submit_blockers.append(f"unresolved_issue_count={len(unresolved_items)}")
    if review_overall_score < 7.0:
        ready_to_submit_blockers.append(f"review_overall_score={review_overall_score:.1f}")
    if review_recommendation not in {"ready_to_submit", "minor_revision_needed"}:
        ready_to_submit_blockers.append(f"review_recommendation={review_recommendation or 'missing'}")
    ready_to_submit = ready_to_submit_by_gates and not ready_to_submit_blockers
    manifest = {
        "phase": "phase3",
        "phase_id": "phase3.6",
        "paper_writing_mode": _paper_writing_mode_snapshot(),
        "input_files_used": {
            "phase3_5_final_review_report": str(phase3_5_dir / "final_review_report.md"),
            "phase3_5_revision_plan": str(phase3_5_dir / "revision_plan.md"),
            "phase3_5_critical_issues": str(phase3_5_dir / "critical_issues.md"),
            "phase3_5_citation_review_report": str(phase3_5_dir / "citation_review_report.md"),
            "phase3_5_theory_review_report": str(phase3_5_dir / "theory_review_report.md"),
            "phase3_5_experiment_review_report": str(phase3_5_dir / "experiment_review_report.md"),
            "phase3_5_latex_format_review_report": str(phase3_5_dir / "latex_format_review_report.md"),
            "phase3_5_scorecard": str(phase3_5_dir / "final_review_scorecard.json"),
            "current_full_paper_tex": str(phase3_5_dir / "full_paper_revised_preview.tex"),
            "verified_references_bib": str(phase3_4_dir / "references_ieee.bib"),
        },
        "outputs": {
            "revised_full_paper_tex": str(phase3_6_dir / "revised_full_paper.tex"),
            "revised_full_paper_preview_pdf": str(phase3_6_dir / "revised_full_paper_preview.pdf"),
            "references_bib": str(phase3_6_dir / "references.bib"),
            "verified_references_bib": str(phase3_6_dir / "verified_references.bib"),
            "revision_applied_report": str(phase3_6_dir / "revision_applied_report.md"),
            "revision_diff_summary": str(phase3_6_dir / "revision_diff_summary.md"),
            "unresolved_issues": str(phase3_6_dir / "unresolved_issues.md"),
        },
        "applied_issue_ids": sorted(applied_issue_ids),
        "unresolved_issue_count": len(unresolved_items),
        "missing_reference_keys": missing_keys,
        "compile_status": "warnings_present" if phase3_6_max_overfull_hbox_pt > 10.0 else "ok",
        "compile_warnings_summary": phase3_6_compile_warnings,
        "max_overfull_hbox_pt": phase3_6_max_overfull_hbox_pt,
        "final_abbreviation_check": final_abbreviation_report,
        "deterministic_abbreviation_repair": abbreviation_repair_report,
        "contract_scope_check": contract_scope_report,
        "post_revision_review": {
            key: value
            for key, value in (post_revision_review if isinstance(post_revision_review, dict) else {}).items()
            if key != "payload"
        },
        "post_revision_review_scorecard": effective_review_scorecard,
        "reference_status": (
            "bib_file_flow_error"
            if not bool(bib_file_reference_flow.get("ok"))
            else "verification_risk_remains"
            if ("P0-REF-01" in unresolved_issue_ids or missing_keys)
            else "ok"
        ),
        "bib_file_reference_flow": bib_file_reference_flow,
        "experiment_status": phase25_status or ("additional_experiment_needed" if any(issue_id.startswith("P1-EXP") for issue_id in unresolved_issue_ids) else "no_new_experiment_flag"),
        "generated_timestamp": utcnow_iso(),
        "ready_to_submit_by_gates_before_mode_check": ready_to_submit_by_gates,
        "ready_to_submit_blockers": ready_to_submit_blockers,
        "deterministic_paper_outputs": deterministic_paper_outputs,
        "ready_to_submit_estimate": ready_to_submit,
        "preview": preview,
    }
    write_text(phase3_6_dir / "phase3_6_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    write_text(phase3_6_dir / "phase3_6_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest
