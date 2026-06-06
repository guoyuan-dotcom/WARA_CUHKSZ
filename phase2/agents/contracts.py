from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ArtifactSpec:
    """A declared artifact boundary for an agent."""

    id: str
    path_hint: str
    kind: str = "text"
    required: bool = True
    frozen_required: bool = False
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GateSpec:
    """A quality gate that must pass before downstream agents rely on output."""

    id: str
    purpose: str
    checks: tuple[str, ...]
    failure_route: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AgentContract:
    """Stable interface for a role-specialized WARA agent."""

    id: str
    role: str
    phase_hints: tuple[str, ...]
    input_artifacts: tuple[ArtifactSpec, ...]
    output_artifacts: tuple[ArtifactSpec, ...]
    tools: tuple[str, ...] = ()
    frozen_contracts: tuple[str, ...] = ()
    allowed_actions: tuple[str, ...] = ()
    forbidden_actions: tuple[str, ...] = ()
    gates: tuple[GateSpec, ...] = ()
    repair_policy: str = "repair_only_reported_issue"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["input_artifact_ids"] = [item.id for item in self.input_artifacts]
        payload["output_artifact_ids"] = [item.id for item in self.output_artifacts]
        return payload


@dataclass
class AgentRunRecord:
    """Runtime record written by the controller after an agent-like task runs."""

    agent_id: str
    status: str
    input_artifacts: list[str]
    output_artifacts: list[str]
    gate_ids: list[str] = field(default_factory=list)
    phase_hint: str = ""
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_agent_contract(contract: AgentContract) -> list[str]:
    """Return human-readable contract errors without mutating the contract."""

    errors: list[str] = []
    if not contract.id.strip():
        errors.append("agent id is empty")
    if not contract.role.strip():
        errors.append(f"{contract.id}: role is empty")
    if not contract.input_artifacts:
        errors.append(f"{contract.id}: input_artifacts is empty")
    if not contract.output_artifacts:
        errors.append(f"{contract.id}: output_artifacts is empty")

    input_ids = [item.id for item in contract.input_artifacts]
    output_ids = [item.id for item in contract.output_artifacts]
    if len(input_ids) != len(set(input_ids)):
        errors.append(f"{contract.id}: duplicate input artifact ids")
    if len(output_ids) != len(set(output_ids)):
        errors.append(f"{contract.id}: duplicate output artifact ids")

    for frozen_id in contract.frozen_contracts:
        if frozen_id not in input_ids:
            errors.append(f"{contract.id}: frozen contract `{frozen_id}` is not declared as an input")
    for gate in contract.gates:
        if not gate.failure_route.strip():
            errors.append(f"{contract.id}: gate `{gate.id}` has no failure route")
        if not gate.checks:
            errors.append(f"{contract.id}: gate `{gate.id}` has no checks")

    return errors
