from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from .experiment_agent import ExperimentAgent


PROMPT_ROOT = Path(__file__).resolve().parents[1] / "prompts"
EXPERIMENT_AGENT_PROMPT = "agents/experiment_agent.prompt.yaml"


def _load_prompt(relative_path: str) -> dict[str, Any]:
    path = PROMPT_ROOT / relative_path
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Prompt YAML must be a mapping: {path}")
    template = payload.get("template")
    if not isinstance(template, str) or not template.strip():
        raise ValueError(f"Prompt template is empty or invalid: {path}")
    return payload


def _render_prompt(relative_path: str, **variables: str) -> str:
    payload = _load_prompt(relative_path)
    rendered = str(payload["template"]).strip()
    required_variables = [str(name) for name in payload.get("variables") or variables.keys()]
    missing = [name for name in required_variables if name not in variables]
    if missing:
        raise ValueError(f"Missing prompt variables for {relative_path}: {', '.join(missing)}")

    for key, value in variables.items():
        rendered = rendered.replace("{" + key + "}", str(value))

    unresolved = [name for name in required_variables if "{" + name + "}" in rendered]
    if unresolved:
        raise ValueError(f"Unresolved prompt variables in {relative_path}: {', '.join(unresolved)}")
    return rendered.strip()


def _compact_json(payload: dict[str, Any], *, max_chars: int) -> str:
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    if len(text) <= max_chars:
        return text
    head_chars = max_chars * 3 // 4
    tail_chars = max_chars - head_chars
    return (
        text[:head_chars].rstrip()
        + "\n\n... [ExperimentAgent request compacted for prompt budget] ...\n\n"
        + text[-tail_chars:].lstrip()
    )


def build_experiment_agent_task_prompt(
    *,
    run_dir: Path,
    task_kind: str,
    output_contract: str,
    legacy_task_prompt: str = "",
    request_payload: dict[str, Any] | None = None,
    request_max_chars: int = 70000,
    write_request: bool = True,
) -> str:
    """Render a WARA ExperimentAgent prompt around a phase-specific task.

    The legacy task prompt remains available only as a compatibility contract
    for output shape and harness details; the ExperimentAgent request is the
    authoritative context boundary.
    """

    agent = ExperimentAgent(Path(run_dir))
    payload = request_payload or agent.build_request_payload()
    if write_request:
        agent.write_request_payload()
    return _render_prompt(
        EXPERIMENT_AGENT_PROMPT,
        task_kind=task_kind,
        output_contract=output_contract.strip(),
        experiment_agent_request_json=_compact_json(payload, max_chars=request_max_chars),
        legacy_task_prompt=str(legacy_task_prompt or "").strip(),
    )
