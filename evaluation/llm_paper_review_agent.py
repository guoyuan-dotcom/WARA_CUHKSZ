#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
PHASE2_SCRIPTS = REPO_ROOT / "phase2" / "scripts"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PHASE2_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(PHASE2_SCRIPTS))


RESEARCH_VALIDITY_RUBRIC: list[dict[str, Any]] = [
    {
        "key": "problem_definition_and_scope",
        "label": "Problem definition and scope",
        "max": 10,
        "description": "Clarity, boundary, and appropriateness of the research problem.",
    },
    {
        "key": "novelty_and_positioning",
        "label": "Novelty and positioning",
        "max": 10,
        "description": "Claimed contribution and relation to prior work.",
    },
    {
        "key": "system_model_and_technical_correctness",
        "label": "System model and technical correctness",
        "max": 15,
        "description": "Correctness of the model, assumptions, notation, and wireless reasoning.",
    },
    {
        "key": "method_validity_and_formulation_alignment",
        "label": "Method validity and formulation alignment",
        "max": 15,
        "description": "Consistency between formulation, method, and claimed solution route.",
    },
    {
        "key": "evidence_validity_and_experiment_design",
        "label": "Evidence validity and experiment design",
        "max": 20,
        "description": "Execution provenance, baselines, scenarios, sweeps, and empirical support.",
    },
    {
        "key": "result_interpretation_and_claim_support",
        "label": "Result interpretation and claim support",
        "max": 15,
        "description": "Whether numerical trends support the stated claims.",
    },
    {
        "key": "scientific_writing_and_presentation",
        "label": "Scientific writing and presentation",
        "max": 10,
        "description": "Organization, readability, figure/table quality, and technical clarity.",
    },
    {
        "key": "reference_grounding",
        "label": "Reference grounding",
        "max": 5,
        "description": "Citation relevance and grounding of prior-work claims.",
    },
]


OPTIMIZATION_MATURITY_RUBRIC: list[dict[str, Any]] = [
    {
        "key": "optimization_framing",
        "label": "Optimization framing",
        "max": 15,
        "description": "Wireless bottleneck, controllable resources, objective, and contribution mechanism.",
    },
    {
        "key": "formulation_quality",
        "label": "Formulation quality",
        "max": 20,
        "description": "Variables, assumptions, objective, and constraints of the optimization problem.",
    },
    {
        "key": "solution_method_validity",
        "label": "Solution-method validity",
        "max": 20,
        "description": "Reformulation, algorithmic route, approximation scope, and method--formulation alignment.",
    },
    {
        "key": "benchmark_design",
        "label": "Benchmark design",
        "max": 15,
        "description": "Baselines, scenarios, sweeps, ablations, and fairness of comparison.",
    },
    {
        "key": "evidence_strength",
        "label": "Evidence strength",
        "max": 20,
        "description": "Stability, interpretability, and sufficiency of numerical evidence.",
    },
    {
        "key": "scholarly_positioning",
        "label": "Scholarly positioning",
        "max": 10,
        "description": "Relation to prior wireless optimization literature.",
    },
]


REVIEW_PROFILES: dict[str, dict[str, Any]] = {
    "research_validity": {
        "label": "Manuscript-Level Research Validity",
        "score_name": "research_validity_score",
        "rubric": RESEARCH_VALIDITY_RUBRIC,
        "purpose": (
            "Evaluate the complete manuscript as a compact wireless-communications research paper, "
            "including scope, novelty, technical correctness, evidence, claim support, writing, and references."
        ),
    },
    "optimization_maturity": {
        "label": "Optimization Research Maturity",
        "score_name": "optimization_maturity_score",
        "rubric": OPTIMIZATION_MATURITY_RUBRIC,
        "purpose": (
            "Evaluate the optimization core of the paper, focusing on the chain from wireless bottleneck "
            "to formulation, solution method, benchmark design, numerical evidence, and scholarly positioning."
        ),
    },
}


SYSTEM_PROMPT = """You are a strict IEEE Wireless Communications Letters-style scoring agent for wireless optimization manuscripts.
You evaluate only the final paper content provided to you.
Do not assume access to hidden code, hidden simulations, WARA artifacts, author intent, or unstated experiment provenance.
Treat text-only assertions of experiments as unsupported unless the manuscript itself gives credible setup details, baselines, metrics, figures/tables, and numerical values.
Return only a valid JSON object following the requested schema."""


def extract_pdf_text(pdf_path: Path) -> str:
    if not pdf_path.exists():
        return ""
    reader_cls: Any | None = None
    try:
        from pypdf import PdfReader  # type: ignore
        reader_cls = PdfReader
    except Exception:
        try:
            from PyPDF2 import PdfReader  # type: ignore
            reader_cls = PdfReader
        except Exception:
            reader_cls = None
    if reader_cls is not None:
        try:
            reader = reader_cls(str(pdf_path))
            chunks: list[str] = []
            for index, page in enumerate(reader.pages, start=1):
                try:
                    text = page.extract_text() or ""
                except Exception:
                    text = ""
                chunks.append(f"\n[PDF page {index}]\n{text}")
            extracted = "\n".join(chunks)
            if extracted.strip():
                return extracted
        except Exception:
            pass
    try:
        result = subprocess.run(
            ["pdftotext", "-layout", str(pdf_path), "-"],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout
    except Exception:
        pass
    return ""


def clean_extracted_text(text: str) -> str:
    text = text.replace("\x00", "")
    text = "".join(ch for ch in text if ch in "\n\r\t" or ord(ch) >= 32)
    text = re.sub(r"[ \t]{3,}", "  ", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def pdf_text_payload(pdf_path: Path, *, max_chars: int) -> dict[str, Any]:
    pdf_text = clean_extracted_text(extract_pdf_text(pdf_path))
    combined = "=== TEXT EXTRACTED FROM PAPER PDF ===\n" + pdf_text
    truncated = False
    if len(combined) > max_chars:
        combined = combined[:max_chars] + "\n\n[TRUNCATED_FOR_SCORING_AGENT]\n"
        truncated = True
    return {
        "paper_source": str(pdf_path),
        "text": combined,
        "truncated": truncated,
        "char_count": len(combined),
        "input_mode": "pdf_only",
        "text_extraction_failed": not bool(pdf_text.strip()),
    }


def json_from_text(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        cleaned = fence.group(1).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            return json.loads(cleaned[start : end + 1])
        raise


def _rubric_total(rubric: list[dict[str, Any]]) -> int:
    return sum(int(item["max"]) for item in rubric)


def _dimension_schema(rubric: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        item["key"]: {
            "score": f"<integer 0-{item['max']}>",
            "max_score": item["max"],
            "justification": "short reason",
        }
        for item in rubric
    }


def build_prompt(
    paper_id: str,
    *,
    profile_id: str,
    profile: dict[str, Any],
    text_payload: dict[str, Any],
) -> str:
    rubric = profile["rubric"]
    total = _rubric_total(rubric)
    rubric_lines = "\n".join(
        f"- {item['label']} ({item['max']} pts): {item['description']}" for item in rubric
    )
    return f"""
Score the following wireless optimization manuscript using only the text extracted from its PDF.
The paper id is only an anonymized identifier: {paper_id}.

Scoring profile: {profile['label']}
Purpose: {profile['purpose']}

Use this {total}-point rubric:
{rubric_lines}

Scoring instructions:
- Assign integer scores between 0 and the maximum for each dimension.
- Calibrate as a strict manuscript scoring agent, not as a template-completion checker.
- Reward concise letter-style papers only when their technical content and evidence are credible in the manuscript itself.
- Penalize vague algorithms, disconnected formulations, absent baselines, unsupported numerical evidence, and claims that exceed the shown evidence.
- For evidence-related dimensions, use only the manuscript-level evidence visible in the PDF. Do not infer hidden experiment execution.
- If a manuscript has no credible numerical/evaluation section with concrete setup and figures/tables/numbers, give very low evidence and claim-support scores.
- If the mathematical formulation is shallow, internally inconsistent, or disconnected from the method/results, cap method-related dimensions accordingly.
- If the text was truncated, mention any uncertainty in the confidence field.

Return JSON with this schema:
{{
  "paper_id": "{paper_id}",
  "profile": "{profile_id}",
  "profile_label": "{profile['label']}",
  "overall_score": <integer 0-{total}>,
  "dimension_scores": {json.dumps(_dimension_schema(rubric), ensure_ascii=False, indent=2)},
  "summary": "one short paragraph",
  "strengths": ["..."],
  "weaknesses": ["..."],
  "confidence": <number between 0 and 1>
}}

For each dimension_scores entry, use:
{{
  "score": <integer>,
  "max_score": <integer>,
  "justification": "short reason"
}}

Paper text extracted from the PDF:
{text_payload['text']}
"""


def normalize_profile_review(payload: dict[str, Any], *, fallback_paper_id: str, profile_id: str) -> dict[str, Any]:
    profile = REVIEW_PROFILES[profile_id]
    rubric = profile["rubric"]
    dims = payload.get("dimension_scores")
    if not isinstance(dims, dict):
        dims = {}
    normalized_dims: dict[str, Any] = {}
    total = 0
    for item in rubric:
        key = item["key"]
        raw = dims.get(key, {})
        if not isinstance(raw, dict):
            raw = {}
        try:
            score = int(raw.get("score", 0))
        except Exception:
            score = 0
        score = max(0, min(int(item["max"]), score))
        total += score
        normalized_dims[key] = {
            "label": item["label"],
            "score": score,
            "max_score": int(item["max"]),
            "justification": str(raw.get("justification", "")).strip(),
        }

    try:
        reported = int(payload.get("overall_score", total))
    except Exception:
        reported = total
    if abs(reported - total) > 2:
        reported = total

    score_max = _rubric_total(rubric)
    normalized = {
        "paper_id": str(payload.get("paper_id") or fallback_paper_id),
        "profile": profile_id,
        "profile_label": str(payload.get("profile_label") or profile["label"]),
        "score_name": profile["score_name"],
        "overall_score": max(0, min(score_max, reported)),
        "max_score": score_max,
        "dimension_scores": normalized_dims,
        "summary": str(payload.get("summary", "")).strip(),
        "strengths": payload.get("strengths") if isinstance(payload.get("strengths"), list) else [],
        "weaknesses": payload.get("weaknesses") if isinstance(payload.get("weaknesses"), list) else [],
    }
    normalized["strengths"] = [str(item) for item in normalized["strengths"] if str(item).strip()]
    normalized["weaknesses"] = [str(item) for item in normalized["weaknesses"] if str(item).strip()]
    try:
        confidence = float(payload.get("confidence", 0.5))
    except Exception:
        confidence = 0.5
    normalized["confidence"] = max(0.0, min(1.0, confidence))
    return normalized


def create_review_llm_client(model_profile: str) -> Any:
    from phase_runtime.llm import create_llm_client  # noqa: PLC0415

    return create_llm_client(model_profile)


def review_pdf(
    pdf_path: Path,
    *,
    paper_id: str,
    llm: Any,
    model_profile: str,
    max_chars: int,
    max_tokens: int,
    prompt_out: Path | None,
) -> dict[str, Any]:
    text_payload = pdf_text_payload(pdf_path, max_chars=max_chars)
    if text_payload.get("text_extraction_failed"):
        raise RuntimeError(f"Could not extract text from PDF: {pdf_path}")

    prompts: dict[str, str] = {}
    reviews: dict[str, Any] = {}
    for profile_id, profile in REVIEW_PROFILES.items():
        prompt = build_prompt(paper_id, profile_id=profile_id, profile=profile, text_payload=text_payload)
        prompts[profile_id] = prompt
        if prompt_out is None:
            response = llm.chat(
                [{"role": "user", "content": prompt}],
                system=SYSTEM_PROMPT,
                max_tokens=max_tokens,
                temperature=0.1,
                json_mode=True,
            )
            reviews[profile_id] = normalize_profile_review(
                json_from_text(response.content),
                fallback_paper_id=paper_id,
                profile_id=profile_id,
            )

    if prompt_out is not None:
        prompt_out.parent.mkdir(parents=True, exist_ok=True)
        prompt_text = "\n\n".join(
            f"===== {profile_id} =====\n{SYSTEM_PROMPT}\n\n{prompt}" for profile_id, prompt in prompts.items()
        )
        prompt_out.write_text(prompt_text, encoding="utf-8")
        return {
            "paper_id": paper_id,
            "pdf": str(pdf_path),
            "paper_source": text_payload["paper_source"],
            "dry_run": True,
            "prompt_path": str(prompt_out),
            "profiles": list(REVIEW_PROFILES),
        }

    payload: dict[str, Any] = {
        "paper_id": paper_id,
        "pdf": str(pdf_path),
        "paper_source": text_payload["paper_source"],
        "input_mode": text_payload.get("input_mode"),
        "model_profile": model_profile,
        "review_input_truncated": text_payload["truncated"],
        "review_input_char_count": text_payload["char_count"],
        "reviews": reviews,
        "rubric_profiles": {
            key: {
                "label": value["label"],
                "score_name": value["score_name"],
                "rubric": value["rubric"],
            }
            for key, value in REVIEW_PROFILES.items()
        },
    }
    for profile_id, review in reviews.items():
        payload[REVIEW_PROFILES[profile_id]["score_name"]] = review.get("overall_score")
    return payload


def write_csv(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    reviews = payload.get("reviews") if isinstance(payload.get("reviews"), dict) else {}
    fieldnames = [
        "paper_id",
        "research_validity_score",
        "optimization_maturity_score",
        "research_validity_confidence",
        "optimization_maturity_confidence",
    ]
    for profile_id, profile in REVIEW_PROFILES.items():
        for item in profile["rubric"]:
            fieldnames.append(f"{profile_id}.{item['key']}")

    row: dict[str, Any] = {
        "paper_id": payload.get("paper_id", ""),
        "research_validity_score": payload.get("research_validity_score", ""),
        "optimization_maturity_score": payload.get("optimization_maturity_score", ""),
        "research_validity_confidence": reviews.get("research_validity", {}).get("confidence", ""),
        "optimization_maturity_confidence": reviews.get("optimization_maturity", {}).get("confidence", ""),
    }
    for profile_id, profile in REVIEW_PROFILES.items():
        dims = reviews.get(profile_id, {}).get("dimension_scores", {})
        if not isinstance(dims, dict):
            dims = {}
        for item in profile["rubric"]:
            value = dims.get(item["key"], {})
            row[f"{profile_id}.{item['key']}"] = value.get("score", "") if isinstance(value, dict) else ""

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Independent LLM manuscript scoring agent for a single wireless optimization PDF."
    )
    parser.add_argument("--pdf", required=True, help="Path to the paper PDF to score.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--csv", default="")
    parser.add_argument("--paper-id", default="", help="Optional anonymized id used in the scoring output.")
    parser.add_argument("--model-profile", default="openai-gpt-5.5")
    parser.add_argument("--max-chars", type=int, default=70000)
    parser.add_argument("--max-tokens", type=int, default=12000)
    parser.add_argument("--dry-run-prompt", default="", help="Write both scoring prompts to this file and do not call the LLM.")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists() or pdf_path.suffix.lower() != ".pdf":
        raise SystemExit(f"PDF not found or not a .pdf file: {pdf_path}")
    paper_id = args.paper_id.strip() or pdf_path.stem
    prompt_out = Path(args.dry_run_prompt) if args.dry_run_prompt else None
    llm = None if prompt_out is not None else create_review_llm_client(args.model_profile)

    print(f"[scoring] {paper_id}", flush=True)
    payload = review_pdf(
        pdf_path,
        paper_id=paper_id,
        llm=llm,
        model_profile=args.model_profile,
        max_chars=args.max_chars,
        max_tokens=args.max_tokens,
        prompt_out=prompt_out,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.csv:
        write_csv(payload, Path(args.csv))
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
