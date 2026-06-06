from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PHASE2_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PHASE2_ROOT.parent
PHASE3_ROOT = WORKSPACE_ROOT / "phase3"
PROMPT_ROOT = PHASE2_ROOT / "prompts"
PHASE3_PROMPT_ROOT = PHASE3_ROOT / "prompts"


def _is_phase3_prompt(relative_path: str | Path) -> bool:
    first_part = str(relative_path).replace("\\", "/").split("/", 1)[0]
    return first_part.startswith("phase3_")


def resolve_prompt_path(relative_path: str | Path) -> Path:
    rel_path = Path(relative_path)
    if rel_path.is_absolute():
        return rel_path
    if _is_phase3_prompt(relative_path):
        candidates = [PHASE3_PROMPT_ROOT / rel_path, PROMPT_ROOT / rel_path]
    else:
        candidates = [PROMPT_ROOT / rel_path, PHASE3_PROMPT_ROOT / rel_path]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def load_prompt_yaml(relative_path: str | Path) -> dict[str, Any]:
    path = resolve_prompt_path(relative_path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Prompt YAML must be a mapping: {path}")
    return payload


def load_prompt_payload(relative_path: str | Path) -> dict[str, Any]:
    path = resolve_prompt_path(relative_path)
    payload = load_prompt_yaml(relative_path)
    template = payload.get("template")
    if not isinstance(template, str) or not template.strip():
        raise ValueError(f"Prompt template is empty or invalid: {path}")
    return payload


def render_prompt_template(relative_path: str | Path, **variables: str) -> str:
    payload = load_prompt_payload(relative_path)
    rendered = str(payload["template"]).strip()
    required_variables = [str(name) for name in payload.get("variables") or variables.keys()]
    missing = [name for name in required_variables if name not in variables]
    if missing:
        raise ValueError(f"Missing prompt variables for {relative_path}: {', '.join(missing)}")

    sentinels: dict[str, str] = {}
    for index, key in enumerate(variables):
        # Use exact token replacement instead of str.format because prompt text
        # contains LaTeX braces such as \section{Introduction}. Insert via
        # sentinels first so pasted code/f-strings like f"{solver}:{status}" are
        # not mistaken for unresolved prompt variables after substitution.
        sentinel = f"__WARA_PROMPT_VAR_{index}_{key}__"
        rendered = rendered.replace("{" + key + "}", sentinel)
        sentinels[sentinel] = str(variables[key])

    unresolved = [name for name in required_variables if "{" + name + "}" in rendered]
    if unresolved:
        raise ValueError(f"Unresolved prompt variables in {relative_path}: {', '.join(unresolved)}")
    for sentinel, value in sentinels.items():
        rendered = rendered.replace(sentinel, value)
    return rendered.strip()
