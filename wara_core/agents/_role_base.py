from __future__ import annotations

from pathlib import Path

from phase2.agents.controller import ArtifactWorkspace
from phase2.agents.role_agent import RoleAgent


class WaraRoleAgent(RoleAgent):
    """Thin named wrapper around the generic artifact-mediated role agent."""

    agent_id = ""

    def __init__(self, run_dir: Path, *, workspace: ArtifactWorkspace | None = None) -> None:
        super().__init__(run_dir, self.agent_id, workspace=workspace)
