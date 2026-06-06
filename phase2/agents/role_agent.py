from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .controller import AgentController, ArtifactWorkspace
from .registry import get_agent_contract


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def _read_json(path: Path) -> Any:
    text = _read_text(path)
    if not text.strip():
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _read_yaml(path: Path) -> Any:
    text = _read_text(path)
    if not text.strip():
        return None
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return None


def _read_payload(path: Path, kind: str) -> Any:
    normalized_kind = str(kind or "text").lower()
    if normalized_kind == "json":
        return _read_json(path)
    if normalized_kind == "yaml":
        return _read_yaml(path)
    if normalized_kind in {"directory", "pdf"}:
        return {"path": str(path), "exists": path.exists()}
    return _read_text(path)


@dataclass
class RoleAgentSnapshot:
    """Topic-agnostic state snapshot for any WARA role agent."""

    run_dir: str
    agent_id: str
    status: str
    present_inputs: dict[str, str]
    present_outputs: dict[str, str]
    missing_inputs: list[str]
    missing_outputs: list[str]
    frozen_contracts_present: list[str]
    gate_ids: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RoleAgent:
    """Generic artifact-mediated facade for non-experiment WARA agents.

    ExperimentAgent has extra Phase 2.4 compatibility logic, but the other role
    agents still need the same basic controller-facing boundary: declared
    inputs, declared outputs, frozen contracts, gates, and a narrow request
    payload. This class provides that without replacing the existing runtime.
    """

    def __init__(self, run_dir: Path, agent_id: str, *, workspace: ArtifactWorkspace | None = None) -> None:
        self.run_dir = Path(run_dir)
        self.id = str(agent_id or "").strip().lower()
        self.contract = get_agent_contract(self.id)
        self.controller = AgentController(self.run_dir, workspace=workspace)

    def bootstrap(
        self,
        *,
        event: str = "bootstrap",
        message: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RoleAgentSnapshot:
        self.controller.bootstrap_known_phase_artifacts()
        snapshot = self.snapshot()
        run_metadata = snapshot.to_dict()
        if metadata:
            run_metadata["event_metadata"] = metadata
        self.controller.record_phase_agent_run(
            phase_id=",".join(self.contract.phase_hints),
            agent_id=self.id,
            status=snapshot.status,
            output_artifacts=list(snapshot.present_outputs),
            message=f"{event}: {message or 'Bootstrapped role-agent state from existing artifacts.'}",
            metadata=run_metadata,
        )
        return snapshot

    def snapshot(self) -> RoleAgentSnapshot:
        present_inputs: dict[str, str] = {}
        present_outputs: dict[str, str] = {}
        missing_inputs: list[str] = []
        missing_outputs: list[str] = []
        frozen_present: list[str] = []
        notes: list[str] = []

        for spec in self.contract.input_artifacts:
            path = self.run_dir / spec.path_hint
            if path.exists():
                present_inputs[spec.id] = spec.path_hint
                if spec.frozen_required or spec.id in self.contract.frozen_contracts:
                    frozen_present.append(spec.id)
            elif spec.required:
                missing_inputs.append(spec.id)

        for spec in self.contract.output_artifacts:
            path = self.run_dir / spec.path_hint
            if path.exists():
                present_outputs[spec.id] = spec.path_hint
                if spec.id in self.contract.frozen_contracts:
                    frozen_present.append(spec.id)
            elif spec.required:
                missing_outputs.append(spec.id)

        for frozen_id in self.contract.frozen_contracts:
            if frozen_id not in set(frozen_present):
                notes.append(f"Frozen contract `{frozen_id}` is not present for {self.id}.")

        if not missing_outputs:
            status = "ready"
            if missing_inputs:
                notes.append(
                    "Some historical input artifacts are missing, but all declared outputs are present."
                )
        elif missing_inputs:
            status = "blocked"
        else:
            status = "ready_to_run"

        return RoleAgentSnapshot(
            run_dir=str(self.run_dir),
            agent_id=self.id,
            status=status,
            present_inputs=present_inputs,
            present_outputs=present_outputs,
            missing_inputs=missing_inputs,
            missing_outputs=missing_outputs,
            frozen_contracts_present=sorted(set(frozen_present)),
            gate_ids=[gate.id for gate in self.contract.gates],
            notes=notes,
        )

    def build_request_payload(self, *, include_payloads: bool = True) -> dict[str, Any]:
        snapshot = self.snapshot()
        inputs: dict[str, Any] = {}
        for spec in self.contract.input_artifacts:
            path = self.run_dir / spec.path_hint
            if not path.exists():
                continue
            entry: dict[str, Any] = {
                "path": spec.path_hint,
                "kind": spec.kind,
                "frozen_required": spec.frozen_required,
            }
            if include_payloads:
                entry["payload"] = _read_payload(path, spec.kind)
            inputs[spec.id] = entry

        outputs = {
            spec.id: {
                "path": spec.path_hint,
                "kind": spec.kind,
                "required": spec.required,
                "description": spec.description,
            }
            for spec in self.contract.output_artifacts
        }
        return {
            "agent_id": self.id,
            "role": self.contract.role,
            "phase_hints": list(self.contract.phase_hints),
            "allowed_actions": list(self.contract.allowed_actions),
            "forbidden_actions": list(self.contract.forbidden_actions),
            "gates": [gate.to_dict() for gate in self.contract.gates],
            "snapshot": snapshot.to_dict(),
            "inputs": inputs,
            "expected_outputs": outputs,
            "frozen_contracts": list(self.contract.frozen_contracts),
        }

    def write_request_payload(self, path: Path | None = None, *, include_payloads: bool = True) -> Path:
        target = path or (self.run_dir / "agent-requests" / f"{self.id}_request.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.build_request_payload(include_payloads=include_payloads), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return target
