from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pipeline_core import read_json, read_text, write_text


_TRUTHY_ENV_VALUES = {"1", "true", "yes", "on"}


def _env_truthy(name: str) -> bool:
    return str(os.environ.get(name, "")).strip().lower() in _TRUTHY_ENV_VALUES


def _paper_writing_mode() -> str:
    raw = str(os.environ.get("WCL_PAPER_WRITING_MODE", "")).strip().lower()
    if not raw:
        return "production"
    aliases = {
        "prod": "production",
        "llm": "production",
        "live": "production",
        "test": "deterministic_test",
        "deterministic": "deterministic_test",
        "offline": "deterministic_test",
    }
    normalized = aliases.get(raw, raw)
    return normalized if normalized in {"production", "deterministic_test"} else "production"


def _paper_deterministic_fallback_allowed() -> bool:
    return False


def _paper_phase_template_env(phase: str) -> str:
    return f"WCL_{phase.upper()}_USE_VERIFIED_TEMPLATE"


def _paper_phase_template_requested(phase: str) -> bool:
    return _env_truthy(_paper_phase_template_env(phase))


def _paper_writing_mode_snapshot() -> dict[str, Any]:
    return {
        "mode": _paper_writing_mode(),
        "deterministic_fallback_allowed": _paper_deterministic_fallback_allowed(),
        "requested_verified_templates": {
            phase: _paper_phase_template_requested(phase)
            for phase in ["phase3_2", "phase3_3", "phase3_4", "phase3_5", "phase3_6"]
        },
        "phase3_5_skip_llm_requested": _env_truthy("WCL_PHASE34_SKIP_LLM"),
        "allow_deterministic_paper_fallback_requested": _env_truthy("WCL_ALLOW_DETERMINISTIC_PAPER_FALLBACK"),
    }


def _write_paper_writing_mode_notice(phase_dir: Path, phase: str, requested_envs: list[str]) -> None:
    if not requested_envs:
        return
    payload = {
        "phase": phase,
        "paper_writing_mode": _paper_writing_mode(),
        "requested_envs": requested_envs,
        "deterministic_fallback_allowed": _paper_deterministic_fallback_allowed(),
        "action": "ignored_deterministic_request_in_production",
        "message": (
            "Deterministic paper-writing templates and LLM-skip paths are disabled in production mode. "
            "Set WCL_PAPER_WRITING_MODE=deterministic_test only for offline smoke tests."
        ),
    }
    write_text(Path(phase_dir) / "paper_writing_mode_notice.json", json.dumps(payload, ensure_ascii=False, indent=2))


def _paper_phase_template_enabled(phase: str, phase_dir: Path | None = None) -> bool:
    requested = _paper_phase_template_requested(phase)
    if requested and not _paper_deterministic_fallback_allowed() and phase_dir is not None:
        _write_paper_writing_mode_notice(Path(phase_dir), phase, [_paper_phase_template_env(phase)])
    return requested and _paper_deterministic_fallback_allowed()


def _paper_phase_llm_skip_enabled(phase: str, phase_dir: Path | None = None) -> bool:
    requested_envs: list[str] = []
    if _paper_phase_template_requested(phase):
        requested_envs.append(_paper_phase_template_env(phase))
    if phase == "phase3_5" and _env_truthy("WCL_PHASE34_SKIP_LLM"):
        requested_envs.append("WCL_PHASE34_SKIP_LLM")
    if phase == "phase3_6" and _env_truthy("WCL_PHASE10_SKIP_LLM"):
        requested_envs.append("WCL_PHASE10_SKIP_LLM")
    if requested_envs and not _paper_deterministic_fallback_allowed() and phase_dir is not None:
        _write_paper_writing_mode_notice(Path(phase_dir), phase, requested_envs)
    return bool(requested_envs) and _paper_deterministic_fallback_allowed()


def _text_has_deterministic_marker(text: str) -> bool:
    lowered = str(text or "").lower()
    markers = [
        "verified_template",
        "deterministic fallback",
        "deterministic targeted revision",
        "fallback review",
    ]
    return any(marker in lowered for marker in markers)


def _payload_has_deterministic_marker(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    if bool(payload.get("fallback_used")) or bool(payload.get("fallback")):
        return True
    for key in ["source", "fallback_reason", "summary", "review_summary_md", "final_review_summary", "revision_diff_summary_md"]:
        if _text_has_deterministic_marker(str(payload.get(key, ""))):
            return True
    return False


def detect_paper_writing_deterministic_outputs(run_dir: Path) -> list[dict[str, str]]:
    run_dir = Path(run_dir)
    checks = [
        ("phase3_2", run_dir / "phase3-2" / "phase3_2_raw_response.txt"),
        ("phase3_3", run_dir / "phase3-3" / "phase3_3_raw_response.txt"),
        ("phase3_4", run_dir / "phase3-4" / "phase3_4_raw_response.txt"),
        ("phase3_4_reference_check", run_dir / "phase3-4" / "reference_check_raw_response.txt"),
        ("phase3_4_technical_citation", run_dir / "phase3-4" / "technical_citation_raw_response.txt"),
        ("phase3_5", run_dir / "phase3-5" / "phase3_5_raw_response.txt"),
        ("phase3_6", run_dir / "phase3-6" / "phase3_6_raw_response.txt"),
    ]
    markers: list[dict[str, str]] = []
    for phase, path in checks:
        if not path.exists():
            continue
        raw = read_text(path)
        payload = _safe_json_loads(raw, {})
        if _payload_has_deterministic_marker(payload) or _text_has_deterministic_marker(raw):
            markers.append({"phase": phase, "path": str(path)})
    cache_meta = read_json(run_dir / "phase3-5" / "phase3_5_cache_meta.json") or {}
    if isinstance(cache_meta, dict) and bool(cache_meta.get("fallback")):
        markers.append({"phase": "phase3_5_cache", "path": str(run_dir / "phase3-5" / "phase3_5_cache_meta.json")})
    return markers


def _safe_json_loads(text: str, default: Any) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return default
