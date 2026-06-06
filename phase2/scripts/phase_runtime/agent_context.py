from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pipeline_core import compact_text, write_text
from wara_core.agents import RoleAgent


def build_role_agent_request_json(
    run_dir: Path,
    agent_id: str,
    *,
    event: str,
    max_chars: int = 7000,
    include_payloads: bool = True,
) -> str:
    """Bootstrap a role agent and return a compact request payload JSON string."""

    run_dir = Path(run_dir)
    agent = RoleAgent(run_dir, agent_id)
    snapshot = agent.bootstrap(event=event, message=f"Synchronized {agent_id} request for prompt assembly.")
    request_path = agent.write_request_payload(include_payloads=include_payloads)
    payload: dict[str, Any] = {
        "agent_id": agent.id,
        "snapshot": snapshot.to_dict(),
        "request_path": str(request_path),
        "request": agent.build_request_payload(include_payloads=include_payloads),
    }
    target = run_dir / "agent-sync" / f"{agent.id}_{event}.json"
    write_text(target, json.dumps(payload, ensure_ascii=False, indent=2))
    return compact_text(json.dumps(payload, ensure_ascii=False, indent=2), max_chars)
