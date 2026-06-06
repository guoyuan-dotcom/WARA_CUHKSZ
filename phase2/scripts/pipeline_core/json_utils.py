from __future__ import annotations

import json
import re
from typing import Any

import yaml


_JSON_FENCE_PATTERN = re.compile(r"```(?:json)?\s*\n(.*?)```", re.DOTALL)


def _load_structured_candidate(candidate: str) -> Any:
    candidate = str(candidate or "").strip()
    if not candidate:
        raise ValueError("empty candidate")
    try:
        return json.loads(candidate)
    except (json.JSONDecodeError, ValueError, RecursionError):
        pass
    parsed = yaml.safe_load(candidate)
    if isinstance(parsed, (dict, list)):
        return parsed
    raise ValueError("candidate is not a structured mapping/list payload")


def safe_json_loads(text: str, default: Any) -> Any:
    """Parse structured JSON/YAML from noisy LLM text."""
    if not text or not text.strip():
        return default

    try:
        return _load_structured_candidate(text)
    except Exception:
        pass

    for match in _JSON_FENCE_PATTERN.finditer(text):
        try:
            return _load_structured_candidate(match.group(1).strip())
        except Exception:
            continue

    candidates: list[str] = []
    brace_depth = 0
    start = -1
    for index, char in enumerate(text):
        if char == "{":
            if brace_depth == 0:
                start = index
            brace_depth += 1
        elif char == "}":
            brace_depth -= 1
            if brace_depth == 0 and start >= 0:
                candidates.append(text[start : index + 1])
                start = -1

    candidates.sort(key=len, reverse=True)
    for candidate in candidates:
        try:
            parsed = _load_structured_candidate(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue

    bracket_depth = 0
    start = -1
    for index, char in enumerate(text):
        if char == "[":
            if bracket_depth == 0:
                start = index
            bracket_depth += 1
        elif char == "]":
            bracket_depth -= 1
            if bracket_depth == 0 and start >= 0:
                try:
                    parsed = _load_structured_candidate(text[start : index + 1])
                    if isinstance(parsed, list):
                        return parsed
                except Exception:
                    pass
                start = -1

    return default


_safe_json_loads = safe_json_loads
