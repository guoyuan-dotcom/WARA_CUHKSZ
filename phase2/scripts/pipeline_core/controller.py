from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .utils import read_json, read_text, utcnow_iso, write_text


CONTROLLER_MANIFEST = "controller_manifest.json"
DEFAULT_REVIEW_REPAIR_ROUND_LIMIT = 3


@dataclass
class ArtifactRef:
    id: str
    path: str
    kind: str = "text"
    producer: str = ""
    frozen: bool = False
    version: int = 1
    reason: str = ""


@dataclass
class AgentSpec:
    id: str
    role: str
    input_artifacts: list[str]
    output_artifacts: list[str]
    tools: list[str] = field(default_factory=list)
    frozen_contracts: list[str] = field(default_factory=list)
    allowed_actions: list[str] = field(default_factory=list)
    forbidden_actions: list[str] = field(default_factory=list)
    acceptance_criteria: list[str] = field(default_factory=list)
    failure_modes: list[str] = field(default_factory=list)
    repair_policy: str = "repair_only_reported_issue"


@dataclass
class GateResult:
    gate_id: str
    ok: bool
    artifact_ids: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checked_at: str = ""

    def __post_init__(self) -> None:
        if not self.checked_at:
            self.checked_at = utcnow_iso()


@dataclass
class ControllerDecision:
    action: str
    reason: str
    target_agent: str = ""
    owner_agent: str = ""
    input_artifacts: list[str] = field(default_factory=list)
    gate_id: str = ""
    repair_scope: str = ""
    rerun_phase: str = ""
    decided_at: str = ""

    def __post_init__(self) -> None:
        if not self.decided_at:
            self.decided_at = utcnow_iso()


class WaraController:
    """Artifact registry, context assembler, gate recorder, and failure router.

    The controller intentionally does not write research content. It records what
    artifacts are current, which contracts are frozen, which gate failed, and which
    bounded repair route is allowed next.
    """

    def __init__(self, run_dir: Path, *, manifest_name: str = CONTROLLER_MANIFEST) -> None:
        self.run_dir = Path(run_dir)
        self.manifest_name = manifest_name
        self.manifest_path = self.run_dir / manifest_name
        self.manifest = self._load_manifest()

    def _load_manifest(self) -> dict[str, Any]:
        payload = read_json(self.manifest_path)
        if isinstance(payload, dict):
            payload.setdefault("artifacts", {})
            payload.setdefault("agents", {})
            payload.setdefault("gates", [])
            payload.setdefault("decisions", [])
            return payload
        return {
            "controller_version": "wara_controller_v1",
            "created_at": utcnow_iso(),
            "artifacts": {},
            "agents": {},
            "gates": [],
            "decisions": [],
        }

    def persist(self) -> None:
        write_text(self.manifest_path, json.dumps(self.manifest, ensure_ascii=False, indent=2))

    def register_agent(self, spec: AgentSpec) -> None:
        self.manifest["agents"][spec.id] = asdict(spec)
        self.persist()

    def register_artifact(self, ref: ArtifactRef) -> None:
        existing = self.manifest["artifacts"].get(ref.id)
        if isinstance(existing, dict) and existing.get("frozen") and existing.get("path") != ref.path:
            raise ValueError(f"Cannot replace frozen artifact `{ref.id}` without rollback/reopen.")
        if isinstance(existing, dict) and existing.get("path") == ref.path:
            ref.version = int(existing.get("version") or 1) + 1
        self.manifest["artifacts"][ref.id] = asdict(ref)
        self.persist()

    def freeze_artifact(self, artifact_id: str, *, reason: str) -> None:
        artifact = self.manifest["artifacts"].get(artifact_id)
        if not isinstance(artifact, dict):
            raise KeyError(f"Unknown artifact `{artifact_id}`.")
        artifact["frozen"] = True
        artifact["reason"] = reason
        artifact["frozen_at"] = utcnow_iso()
        self.persist()

    def artifact_path(self, artifact_id: str) -> Path:
        artifact = self.manifest["artifacts"].get(artifact_id)
        if not isinstance(artifact, dict):
            raise KeyError(f"Unknown artifact `{artifact_id}`.")
        return self.run_dir / str(artifact.get("path", ""))

    def assemble_context(self, agent_id: str) -> dict[str, Any]:
        spec = self.manifest["agents"].get(agent_id)
        if not isinstance(spec, dict):
            raise KeyError(f"Unknown agent `{agent_id}`.")
        context: dict[str, Any] = {"agent_id": agent_id, "artifacts": {}, "frozen_contracts": {}}
        for artifact_id in spec.get("input_artifacts", []):
            context["artifacts"][artifact_id] = self._read_artifact_payload(artifact_id)
        for artifact_id in spec.get("frozen_contracts", []):
            artifact = self.manifest["artifacts"].get(artifact_id)
            if not isinstance(artifact, dict) or not artifact.get("frozen"):
                raise ValueError(f"Agent `{agent_id}` requires frozen contract `{artifact_id}`.")
            context["frozen_contracts"][artifact_id] = self._read_artifact_payload(artifact_id)
        return context

    def _read_artifact_payload(self, artifact_id: str) -> Any:
        artifact = self.manifest["artifacts"].get(artifact_id)
        if not isinstance(artifact, dict):
            raise KeyError(f"Unknown artifact `{artifact_id}`.")
        path = self.run_dir / str(artifact.get("path", ""))
        kind = str(artifact.get("kind") or "text").lower()
        if kind in {"json", "jsonl"}:
            return read_json(path)
        return read_text(path)

    def record_gate(self, result: GateResult) -> ControllerDecision:
        payload = asdict(result)
        self.manifest["gates"].append(payload)
        decision = self.route_failure(result)
        self.manifest["decisions"].append(asdict(decision))
        self.persist()
        return decision

    def record_review_routing(
        self,
        routing_decision: dict[str, Any],
        *,
        source_path: str = "phase3-5/review_routing_decision.json",
    ) -> ControllerDecision:
        """Record the Phase 3.5 ReviewAgent routing output as a controller gate.

        Phase 3.5 is the first gate whose output is already a routing decision:
        it names the owner agent whose artifact must be repaired. The controller
        records that decision without rewriting paper content.
        """

        routing = routing_decision if isinstance(routing_decision, dict) else {}
        decision = self.route_review_routing(routing)
        routes = routing.get("routes") if isinstance(routing.get("routes"), list) else []
        issue_titles = [
            str(item.get("title") or item.get("issue_id") or "review issue")
            for item in routes
            if isinstance(item, dict)
        ]
        status = str(routing.get("status") or "").strip()
        gate_ok = decision.action == "final_ready"
        gate = GateResult(
            gate_id=str(routing.get("gate_id") or "review_gate"),
            ok=gate_ok,
            artifact_ids=["phase3_5_review", "review_routing_decision"],
            errors=[] if gate_ok or status == "minor_polish" else issue_titles,
            warnings=[] if str(routing.get("status")) != "minor_polish" else issue_titles,
        )
        self.manifest["review_routing"] = {
            "source_path": source_path,
            "recorded_at": utcnow_iso(),
            "routing_decision": routing,
            "controller_decision": asdict(decision),
        }
        self.manifest["review_repair_policy"] = {
            "max_auto_repair_rounds": _review_repair_round_limit(),
            "completed_auto_repair_rounds": self._review_repair_round_count(str(routing.get("gate_id") or "review_gate"))
            + (1 if decision.action == "repair" else 0),
            "env_overrides": ["WARA_MAX_REVIEW_REPAIR_ROUNDS", "WCL_PHASE34_REPAIR_ROUNDS"],
        }
        self.manifest["gates"].append(asdict(gate))
        self.manifest["decisions"].append(asdict(decision))
        self.persist()
        return decision

    def _review_repair_round_count(self, gate_id: str = "review_gate") -> int:
        count = 0
        for item in self.manifest.get("decisions", []):
            if not isinstance(item, dict):
                continue
            if item.get("gate_id") == gate_id and item.get("action") == "repair":
                count += 1
        return count

    def route_review_routing(self, routing_decision: dict[str, Any]) -> ControllerDecision:
        routing = routing_decision if isinstance(routing_decision, dict) else {}
        status = str(routing.get("status") or "").strip()
        gate_id = str(routing.get("gate_id") or "review_gate")
        owner_agent = str(routing.get("target_agent") or "").strip()
        if owner_agent == "implementation_agent":
            owner_agent = "experiment_agent"
        next_agent = str(routing.get("next_agent") or "").strip() or "repair_agent"
        if next_agent == "implementation_agent":
            next_agent = "experiment_agent"
        primary_issue_id = str(routing.get("primary_issue_id") or "").strip()
        primary_reason = str(routing.get("primary_reason") or "").strip()
        rerun_phase = _review_owner_rerun_phase(owner_agent, routing)
        repair_round_limit = _review_repair_round_limit()
        completed_repair_rounds = self._review_repair_round_count(gate_id)

        if status == "ready":
            return ControllerDecision(
                action="final_ready",
                target_agent="final_ready",
                reason="Review gate found no actionable repair issues.",
                gate_id=gate_id,
                repair_scope="none",
                rerun_phase="",
            )
        if status in {"repair_required", "minor_polish"}:
            if completed_repair_rounds >= repair_round_limit:
                return ControllerDecision(
                    action="stop",
                    target_agent="manual_triage",
                    owner_agent=owner_agent,
                    reason=(
                        f"Review repair round limit reached "
                        f"({completed_repair_rounds}/{repair_round_limit}); "
                        "stop automatic repair and require manual triage."
                    ),
                    gate_id=gate_id,
                    repair_scope="manual_triage_after_review_repair_limit",
                    rerun_phase="",
                    input_artifacts=["phase3_5_review", "review_routing_decision"],
                )
            return ControllerDecision(
                action="repair",
                target_agent=next_agent,
                owner_agent=owner_agent,
                reason=primary_reason or f"Review gate routed {primary_issue_id or 'an issue'} to {owner_agent or 'repair'}.",
                gate_id=gate_id,
                repair_scope=owner_agent or "controller_selected_artifact",
                rerun_phase=rerun_phase,
                input_artifacts=["phase3_5_review", "review_routing_decision"],
            )
        return ControllerDecision(
            action="stop",
            reason="Review gate did not provide a safe automatic route.",
            gate_id=gate_id,
            owner_agent=owner_agent,
            repair_scope=owner_agent,
            rerun_phase=rerun_phase,
            input_artifacts=["phase3_5_review", "review_routing_decision"],
        )

    def route_failure(self, result: GateResult) -> ControllerDecision:
        if result.ok:
            return ControllerDecision(action="continue", reason=f"{result.gate_id} passed", gate_id=result.gate_id)
        text = "\n".join([*result.errors, *result.warnings]).lower()
        if any(token in text for token in ("json", "schema", "malformed", "yaml", "parse")):
            return ControllerDecision(
                action="repair",
                target_agent="schema_repair_agent",
                reason="Malformed or schema-invalid artifact.",
                gate_id=result.gate_id,
                repair_scope="schema_only",
                input_artifacts=result.artifact_ids,
            )
        if any(token in text for token in ("formulation", "control variable", "derived quantity", "original problem")):
            return ControllerDecision(
                action="repair",
                target_agent="formulation_repair_agent",
                reason="Frozen mathematical formulation is inconsistent.",
                gate_id=result.gate_id,
                repair_scope="formulation_contract",
                input_artifacts=result.artifact_ids,
            )
        if any(
            token in text
            for token in (
                "phi-style",
                "safe-counterpart",
                "safe counterpart",
                "ambiguity model",
                "uncertainty model",
                "rank recovery",
                "physical covariance-domain",
                "receiver decoding",
                "technical closure",
            )
        ):
            return ControllerDecision(
                action="repair",
                target_agent="theory_agent",
                reason="Technical closure is incomplete and needs bounded upstream formulation/theory revision.",
                gate_id=result.gate_id,
                repair_scope="technical_closure_or_formulation_route",
                rerun_phase="phase2.2",
                input_artifacts=result.artifact_ids,
            )
        if any(token in text for token in ("import", "syntax", "compile", "function", "method id")):
            return ControllerDecision(
                action="repair",
                target_agent="implementation_repair_agent",
                reason="Generated implementation failed interface or import checks.",
                gate_id=result.gate_id,
                repair_scope="implementation_only",
                input_artifacts=result.artifact_ids,
            )
        if any(token in text for token in ("experiment_responsiveness", "weakly responsive", "metric is constant", "main y_metric")):
            return ControllerDecision(
                action="repair",
                target_agent="experiment_design_repair_agent",
                reason="Figure metric, sweep, or operating-regime design is not responsive enough for paper evidence.",
                gate_id=result.gate_id,
                repair_scope="figure_sweep_metric_or_operating_regime",
                input_artifacts=result.artifact_ids,
            )
        if any(token in text for token in ("missing metric", "required metric", "csv column", "sweep", "finite")):
            return ControllerDecision(
                action="repair",
                target_agent="experiment_code_repair_agent",
                reason="Validation evidence contract is not satisfied.",
                gate_id=result.gate_id,
                repair_scope="metrics_or_validation_adapter",
                input_artifacts=result.artifact_ids,
            )
        if any(token in text for token in ("unsupported claim", "claim", "evidence", "paper-ready", "quick_mode")):
            return ControllerDecision(
                action="repair",
                target_agent="analysis_or_writing_repair_agent",
                reason="Claims are not supported by verified evidence.",
                gate_id=result.gate_id,
                repair_scope="claim_scope_or_missing_experiments",
                input_artifacts=result.artifact_ids,
            )
        return ControllerDecision(
            action="stop",
            reason="Gate failed with no safe automatic repair route.",
            gate_id=result.gate_id,
            input_artifacts=result.artifact_ids,
        )


def _review_route_text(route: dict[str, Any]) -> str:
    raw_issue = route.get("raw_issue") if isinstance(route.get("raw_issue"), dict) else {}
    fields = [
        route.get("issue_id"),
        route.get("title"),
        route.get("routing_reason"),
        route.get("source"),
        raw_issue.get("issue_id"),
        raw_issue.get("title"),
        raw_issue.get("category"),
        raw_issue.get("issue"),
        raw_issue.get("why_it_matters"),
        raw_issue.get("exact_location"),
        raw_issue.get("suggested_action"),
        raw_issue.get("responsible_phase"),
    ]
    return "\n".join(str(item or "") for item in fields).lower()


def _review_route_targets_agent(route: dict[str, Any], owner_agent: str) -> bool:
    target_agent = str(route.get("target_agent", "")).strip()
    if target_agent == "implementation_agent":
        target_agent = "experiment_agent"
    normalized_owner = "experiment_agent" if owner_agent == "implementation_agent" else owner_agent
    return target_agent == normalized_owner


def _review_route_is_phase25_evidence_expansion(route: dict[str, Any]) -> bool:
    text = _review_route_text(route)
    phase25_terms = (
        "phase 2.5",
        "phase2.5",
        "phase25",
        "paper-ready",
        "paper ready",
        "quick-mode",
        "quick_mode",
        "draft figure",
        "draft figures",
        "final figure",
        "final figures",
        "figure registry",
        "evidence package",
        "experiment summary",
        "verified final figures",
        "paper-ready simulations",
        "numerical validation",
        "submission-grade support",
    )
    return any(term in text for term in phase25_terms)


def _review_routes_phase25_evidence_expansion(owner_agent: str, routing: dict[str, Any] | None) -> bool:
    owner = "experiment_agent" if owner_agent == "implementation_agent" else str(owner_agent or "").strip()
    if owner != "experiment_agent" or not isinstance(routing, dict):
        return False
    routes = routing.get("routes") if isinstance(routing.get("routes"), list) else []
    candidate_routes = [
        route
        for route in routes
        if isinstance(route, dict)
        and _review_route_targets_agent(route, owner)
        and str(route.get("priority", "")).strip() in {"P0", "P1"}
    ]
    if not candidate_routes and str(routing.get("target_agent") or "").strip() in {"experiment_agent", "implementation_agent"}:
        candidate_routes = [routing]
    return any(_review_route_is_phase25_evidence_expansion(route) for route in candidate_routes)


def _review_owner_rerun_phase(owner_agent: str, routing: dict[str, Any] | None = None) -> str:
    owner = str(owner_agent or "").strip()
    if owner == "implementation_agent":
        owner = "experiment_agent"
    if _review_routes_phase25_evidence_expansion(owner, routing):
        return "phase2.5"
    return {
        "formulation_agent": "phase2.1",
        "theory_agent": "phase2.3",
        "experiment_agent": "phase2.4",
        "validation_agent": "phase2.5",
        "literature_agent": "phase3.4",
        "writing_agent": "phase3.6",
        "repair_agent": "phase3.6",
        "final_ready": "",
    }.get(owner, "")


def _review_repair_round_limit() -> int:
    for env_name in ("WARA_MAX_REVIEW_REPAIR_ROUNDS", "WCL_PHASE34_REPAIR_ROUNDS"):
        raw_value = os.environ.get(env_name)
        if raw_value is None:
            continue
        try:
            return max(0, int(raw_value))
        except ValueError:
            continue
    return DEFAULT_REVIEW_REPAIR_ROUND_LIMIT
