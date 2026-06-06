from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .contracts import AgentContract, AgentRunRecord, ArtifactSpec
from .registry import build_default_agent_registry, phase_to_agent_ids


AGENT_WORKSPACE_MANIFEST = "agent_workspace_manifest.json"


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


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


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


@dataclass
class RegisteredArtifact:
    id: str
    path: str
    producer_agent: str
    kind: str = "text"
    frozen: bool = False
    version: int = 1
    created_at: str = field(default_factory=_utcnow_iso)
    metadata: dict[str, Any] = field(default_factory=dict)


class ArtifactWorkspace:
    """Artifact registry shared by WARA agents.

    This is intentionally small and file-backed so it can be layered over the
    existing runtime without replacing it all at once.
    """

    def __init__(self, root_dir: Path, manifest_name: str = AGENT_WORKSPACE_MANIFEST) -> None:
        self.root_dir = Path(root_dir)
        self.manifest_path = self.root_dir / manifest_name
        self.manifest = self._load()

    def _load(self) -> dict[str, Any]:
        payload = _read_json(self.manifest_path)
        if isinstance(payload, dict):
            payload.setdefault("version", "wara_agent_workspace")
            payload.setdefault("artifacts", {})
            payload.setdefault("runs", [])
            return payload
        return {
            "version": "wara_agent_workspace",
            "created_at": _utcnow_iso(),
            "artifacts": {},
            "runs": [],
        }

    def persist(self) -> None:
        _write_json(self.manifest_path, self.manifest)

    def register_artifact(
        self,
        artifact_id: str,
        path: str | Path,
        *,
        producer_agent: str,
        kind: str = "text",
        frozen: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> RegisteredArtifact:
        relative_path = self._relative(path)
        prior = self.manifest["artifacts"].get(artifact_id)
        version = int(prior.get("version", 0)) + 1 if isinstance(prior, dict) else 1
        if isinstance(prior, dict) and prior.get("frozen") and prior.get("path") != relative_path:
            raise ValueError(f"Cannot replace frozen artifact `{artifact_id}` without explicit rollback.")
        record = RegisteredArtifact(
            id=artifact_id,
            path=relative_path,
            producer_agent=producer_agent,
            kind=kind,
            frozen=frozen,
            version=version,
            metadata=metadata or {},
        )
        self.manifest["artifacts"][artifact_id] = asdict(record)
        self.persist()
        return record

    def freeze(self, artifact_id: str, *, reason: str) -> None:
        artifact = self.manifest["artifacts"].get(artifact_id)
        if not isinstance(artifact, dict):
            raise KeyError(f"Unknown artifact `{artifact_id}`.")
        artifact["frozen"] = True
        artifact["frozen_at"] = _utcnow_iso()
        artifact["freeze_reason"] = reason
        self.persist()

    def resolve(self, artifact_id: str) -> Path:
        artifact = self.manifest["artifacts"].get(artifact_id)
        if not isinstance(artifact, dict):
            raise KeyError(f"Unknown artifact `{artifact_id}`.")
        return self.root_dir / str(artifact.get("path", ""))

    def read_payload(self, artifact_id: str) -> Any:
        artifact = self.manifest["artifacts"].get(artifact_id)
        if not isinstance(artifact, dict):
            raise KeyError(f"Unknown artifact `{artifact_id}`.")
        path = self.root_dir / str(artifact.get("path", ""))
        kind = str(artifact.get("kind") or "text").lower()
        if kind in {"json", "jsonl"}:
            return _read_json(path)
        return _read_text(path)

    def validate_inputs(self, contract: AgentContract) -> list[str]:
        errors: list[str] = []
        for spec in contract.input_artifacts:
            artifact = self.manifest["artifacts"].get(spec.id)
            if not isinstance(artifact, dict):
                if spec.required:
                    errors.append(f"{contract.id}: missing required artifact `{spec.id}`")
                continue
            path = self.root_dir / str(artifact.get("path", ""))
            if spec.required and not path.exists():
                errors.append(f"{contract.id}: artifact `{spec.id}` path does not exist: {path}")
            if spec.frozen_required and not artifact.get("frozen"):
                errors.append(f"{contract.id}: artifact `{spec.id}` must be frozen")
        return errors

    def build_context(self, contract: AgentContract, *, include_payloads: bool = True) -> dict[str, Any]:
        context: dict[str, Any] = {
            "agent_id": contract.id,
            "role": contract.role,
            "inputs": {},
            "frozen_contracts": {},
            "allowed_actions": list(contract.allowed_actions),
            "forbidden_actions": list(contract.forbidden_actions),
        }
        for spec in contract.input_artifacts:
            artifact = self.manifest["artifacts"].get(spec.id)
            if not isinstance(artifact, dict):
                continue
            entry: dict[str, Any] = dict(artifact)
            if include_payloads:
                entry["payload"] = self.read_payload(spec.id)
            context["inputs"][spec.id] = entry
            if spec.frozen_required or spec.id in contract.frozen_contracts:
                context["frozen_contracts"][spec.id] = entry
        return context

    def record_run(self, record: AgentRunRecord) -> None:
        payload = record.to_dict()
        payload["recorded_at"] = _utcnow_iso()
        self.manifest["runs"].append(payload)
        self.persist()

    def _relative(self, path: str | Path) -> str:
        path_obj = Path(path)
        if not path_obj.is_absolute():
            return str(path_obj)
        try:
            return str(path_obj.relative_to(self.root_dir))
        except ValueError:
            return str(path_obj)


class AgentController:
    """Minimal WARA controller facade over the existing runtime."""

    def __init__(
        self,
        run_dir: Path,
        *,
        registry: dict[str, AgentContract] | None = None,
        workspace: ArtifactWorkspace | None = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.registry = registry or build_default_agent_registry()
        self.workspace = workspace or ArtifactWorkspace(self.run_dir)

    def agents_for_phase(self, phase_id: str) -> tuple[AgentContract, ...]:
        return tuple(self.registry[agent_id] for agent_id in phase_to_agent_ids(phase_id))

    def context_for_agent(self, agent_id: str, *, include_payloads: bool = True) -> dict[str, Any]:
        contract = self.registry[agent_id]
        errors = self.workspace.validate_inputs(contract)
        if errors:
            raise ValueError("; ".join(errors))
        return self.workspace.build_context(contract, include_payloads=include_payloads)

    def record_phase_agent_run(
        self,
        *,
        phase_id: str,
        agent_id: str,
        status: str,
        output_artifacts: list[str],
        message: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        contract = self.registry[agent_id]
        self.workspace.record_run(
            AgentRunRecord(
                agent_id=agent_id,
                status=status,
                phase_hint=phase_id,
                input_artifacts=[item.id for item in contract.input_artifacts],
                output_artifacts=output_artifacts,
                gate_ids=[gate.id for gate in contract.gates],
                message=message,
                metadata=metadata or {},
            )
        )

    def bootstrap_known_phase_artifacts(self) -> None:
        """Register common artifacts from the current phase layout if they exist."""

        for agent in self.registry.values():
            for spec in agent.output_artifacts:
                relative_path = self._existing_phase_artifact_path(agent.id, spec)
                if not relative_path:
                    continue
                self.workspace.register_artifact(
                    spec.id,
                    relative_path,
                    producer_agent=agent.id,
                    kind=spec.kind,
                    frozen=spec.id in {"mathematical_contract", "algorithm_contract"},
                    metadata={"bootstrapped_from_phase_layout": True},
                )

    def _existing_phase_artifact_path(self, agent_id: str, spec: ArtifactSpec) -> str:
        candidates = [spec.path_hint]
        if agent_id in {"experiment_agent", "implementation_agent"}:
            candidates.extend(
                {
                    "experiment_plan": [
                        "phase2-4-simple/experiment_plan.json",
                        "phase2-5/experiment_plan.json",
                    ],
                    "main_experiment_code": [
                        "phase2-4-simple/focused_experiment.py",
                        "phase2-4-simple/simple_experiment.py",
                        "phase2-4-simple/preview_experiment.py",
                    ],
                    "validation_results": [
                        "phase2-4-simple/outputs/simple_results.csv",
                    ],
                    "figure_data": [
                        "phase2-4-simple/figures",
                    ],
                    "experiment_report": [
                        "phase2-4-simple/outputs/simple_summary.json",
                    ],
                }.get(spec.id, [])
            )
        for relative_path in candidates:
            if (self.run_dir / relative_path).exists():
                return relative_path
        return ""

    def missing_inputs_by_agent(self) -> dict[str, list[str]]:
        output: dict[str, list[str]] = {}
        for agent_id, contract in self.registry.items():
            missing = self.workspace.validate_inputs(contract)
            if missing:
                output[agent_id] = missing
        return output
