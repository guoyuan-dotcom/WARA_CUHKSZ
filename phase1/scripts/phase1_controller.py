from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import wara_phase1_pipeline as pipeline
except ModuleNotFoundError:  # pragma: no cover - package import path fallback
    from phase1.scripts import wara_phase1_pipeline as pipeline


@dataclass
class Phase1Controller:
    """Controller for WARA-native Phase 1 artifact production.

    The controller keeps the existing Phase 1 artifacts stable while making the
    orchestration explicit: agent runs, gates, decisions, and frozen contracts
    are recorded in a machine-readable manifest.
    """

    topic: str
    run_dir: Path
    model_profile: str = pipeline.DEFAULT_MODEL_PROFILE
    max_tokens: int = pipeline.DEFAULT_MAX_TOKENS
    tail_root: Path | None = None
    llm_client: pipeline.ChatClient | None = None
    manifest: dict[str, Any] = field(init=False)
    handoff_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.topic = self.topic.strip()
        if len(self.topic) < 5:
            raise ValueError("topic too short")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        tail_root = self.tail_root or pipeline.WORKSPACE_ROOT / "outputs" / "paper_runs" / "shared" / "phase1_tail"
        self.handoff_dir = tail_root / self.run_dir.name
        self.handoff_dir.mkdir(parents=True, exist_ok=True)
        self.manifest = {
            "controller_version": "wara_phase1_controller_v1",
            "phase": "phase1",
            "phase_step": "phase1",
            "phase1_design": "wara_native_4_phase_controller",
            "phase_architecture": {
                "phase1.1": "Research Framing",
                "phase1.2": "Evidence Grounding",
                "phase1.3": "Direction Contract",
                "phase1.4": "WARA Handoff",
            },
            "topic": self.topic,
            "run_dir": str(self.run_dir),
            "handoff_dir": str(self.handoff_dir),
            "created_at": pipeline.utcnow_iso(),
            "agents": [],
            "artifacts": [],
            "gates": [],
            "decisions": [],
            "frozen_contracts": [],
            "llm_trace": [],
        }

    def run(self) -> pipeline.WaraPhase1Result:
        trace: list[dict[str, Any]] = []
        try:
            result = self._run_impl(trace)
        except Exception as exc:
            self._record_decision(
                action="stop",
                reason=f"Phase 1 stopped after {type(exc).__name__}: {exc}",
                target="phase1_controller",
            )
            self.manifest["status"] = "failed"
            self.manifest["completed_at"] = pipeline.utcnow_iso()
            self._persist_manifest()
            raise
        self.manifest["llm_trace"] = trace
        self.manifest["status"] = "completed"
        self.manifest["completed_at"] = pipeline.utcnow_iso()
        self._persist_manifest()
        return result

    def _run_impl(self, trace: list[dict[str, Any]]) -> pipeline.WaraPhase1Result:
        client = self.llm_client or pipeline.make_llm_client(self.model_profile)

        scope_context = pipeline.build_wireless_scope_context(self.topic)
        self._record_agent(
            agent_id="scout_agent",
            role="deterministic wireless scope and topic taxonomy assembly",
            inputs=["user_topic"],
            outputs=["wireless_scope", "scope_contract", "taxonomy_prompt_block"],
        )
        self._record_gate(
            gate_id="scope_gate",
            ok=True,
            artifacts=["wireless_scope", "scope_contract"],
            notes=["Wireless scope context assembled before any LLM direction generation."],
        )
        self._record_decision(
            action="continue",
            reason="Scope context is ready; proceed to research object generation.",
            target="scout_agent",
            gate_id="scope_gate",
        )

        object_system, object_user = pipeline.build_research_object_prompt(self.topic, scope_context)
        research_payload = pipeline.call_json_agent(
            client,
            agent_id="scout_agent",
            system=object_system,
            user=object_user,
            max_tokens=self.max_tokens,
            trace=trace,
        )
        pipeline.validate_research_object_payload(research_payload)
        self._record_agent(
            agent_id="scout_agent",
            role="generate research object, mechanism hypothesis, and Phase 2 seeds",
            inputs=["user_topic", "scope_contract", "taxonomy_prompt_block"],
            outputs=["research_object", "research_frame"],
        )
        self._record_gate(
            gate_id="research_object_gate",
            ok=True,
            artifacts=["research_object", "research_frame"],
            notes=["Research-object schema accepted."],
        )
        self._record_decision(
            action="continue",
            reason="Research object is structurally valid; proceed to evidence grounding.",
            target="literature_agent",
            gate_id="research_object_gate",
        )

        research_frame = pipeline.build_research_frame_payload(self.topic, scope_context, research_payload)
        self._write_phase_artifact(1, "goal.md", pipeline.render_goal_markdown(self.topic), "goal", "markdown")
        self._write_phase_artifact(
            1,
            "topic_intake.json",
            pipeline.dump_json(
                {
                    "topic": self.topic,
                    "phase1_design": "wara_native_4_phase_controller",
                    "controller": "artifact_mediated",
                    "generated_at": pipeline.utcnow_iso(),
                }
            ),
            "topic_intake",
            "json",
        )
        self._write_phase_artifact(1, "wireless_scope.json", pipeline.dump_json(scope_context), "wireless_scope", "json")
        self._write_phase_artifact(
            1,
            "scope_contract.json",
            pipeline.dump_json(scope_context["scope_contract"]),
            "scope_contract",
            "json",
            frozen=True,
            freeze_reason="Phase 1 scope boundary controls what downstream agents may add or forbid.",
        )
        self._write_phase_artifact(
            1,
            "taxonomy_prompt_block.md",
            scope_context["taxonomy_plan"].get("prompt_block", ""),
            "taxonomy_prompt_block",
            "markdown",
        )
        self._write_phase_artifact(
            1,
            "scope_contract.md",
            pipeline.render_scope_markdown(self.topic, scope_context),
            "scope_contract_markdown",
            "markdown",
        )
        self._write_phase_artifact(1, "research_object_request.md", object_user, "research_object_request", "prompt")
        self._write_phase_artifact(
            1,
            "research_object.json",
            pipeline.dump_json(research_payload),
            "research_object",
            "json",
        )
        self._write_phase_artifact(
            1,
            "research_object.md",
            pipeline.render_research_object_markdown(research_payload),
            "research_object_markdown",
            "markdown",
        )
        self._write_phase_artifact(
            1,
            "research_frame.json",
            pipeline.dump_json(research_frame),
            "research_frame",
            "json",
        )
        pipeline.write_phase_status(
            self.run_dir,
            1,
            (
                "goal.md",
                "topic_intake.json",
                "wireless_scope.json",
                "scope_contract.json",
                "taxonomy_prompt_block.md",
                "scope_contract.md",
                "research_object_request.md",
                "research_object.json",
                "research_object.md",
                "research_frame.json",
            ),
        )

        evidence_pack = pipeline.build_grounded_evidence_pack(
            self.topic,
            scope_context,
            research_payload,
            artifact_dir=pipeline.phase1_step_dir(self.run_dir, 2),
        )
        grounding_report = pipeline.validate_literature_grounding_contract(evidence_pack)
        self._record_agent(
            agent_id="literature_agent",
            role="assemble abstract-grounded literature cards, gap signals, citation policy, and retrieved references",
            inputs=["research_object", "scope_contract"],
            outputs=["evidence_pack"],
        )
        self._record_gate(
            gate_id="evidence_grounding_gate",
            ok=True,
            artifacts=["evidence_pack"],
            notes=[
                "Evidence pack assembled; citation verification remains a downstream responsibility.",
                (
                    f"{grounding_report['abstract_or_pdf_card_count']} abstract/PDF-backed cards meet "
                    f"minimum {grounding_report['minimum_abstract_or_pdf_cards']}."
                ),
            ],
        )
        self._record_decision(
            action="continue",
            reason="Evidence pack is available; proceed to direction contract generation.",
            target="scout_agent",
            gate_id="evidence_grounding_gate",
        )
        self._write_phase_artifact(2, "evidence_pack.json", pipeline.dump_json(evidence_pack), "evidence_pack", "json")
        self._write_phase_artifact(
            2,
            "evidence_pack.md",
            pipeline.render_evidence_pack_markdown(evidence_pack),
            "evidence_pack_markdown",
            "markdown",
        )
        pipeline.write_phase_status(self.run_dir, 2, ("evidence_pack.json", "evidence_pack.md"))

        direction_system, direction_user = pipeline.build_direction_contract_prompt(self.topic, research_frame, evidence_pack)
        contract_payload = pipeline.call_json_agent(
            client,
            agent_id="scout_agent",
            system=direction_system,
            user=direction_user,
            max_tokens=self.max_tokens,
            trace=trace,
        )
        contract_payload = pipeline.normalize_direction_contract_payload(contract_payload)
        pipeline.validate_direction_contract_payload(contract_payload)
        self._record_agent(
            agent_id="scout_agent",
            role="select research direction and produce Phase 2-facing contracts",
            inputs=["research_frame", "evidence_pack", "scope_contract"],
            outputs=[
                "candidate_directions",
                "selected_direction",
                "contract_bundle",
                "problem_contract_seed",
                "novelty_contract",
                "proof_contract",
                "validation_contract",
            ],
        )
        self._record_gate(
            gate_id="direction_contract_gate",
            ok=True,
            artifacts=["selected_direction", "contract_bundle"],
            notes=["Direction contract schema accepted and selected direction is ready to freeze."],
        )
        self._record_decision(
            action="freeze",
            reason="Selected direction and Phase 2-facing seed contracts define the handoff boundary.",
            target="selected_direction",
            gate_id="direction_contract_gate",
        )

        candidates = pipeline.coerce_list(contract_payload.get("candidate_directions"))
        candidate_review = pipeline.normalize_candidate_review(contract_payload, scope_context)
        selected_evidence_pack = pipeline.build_grounded_evidence_pack(
            self.topic,
            scope_context,
            research_payload,
            contract_payload,
            artifact_dir=pipeline.phase1_step_dir(self.run_dir, 4),
        )
        evidence_pack = pipeline.merge_evidence_packs(evidence_pack, selected_evidence_pack)
        try:
            grounding_contract_report = pipeline.validate_literature_grounding_contract(evidence_pack)
            reference_contract_report = pipeline.validate_evidence_pack_reference_contract(evidence_pack)
        except ValueError as exc:
            self._record_gate(
                gate_id="reference_bank_gate",
                ok=False,
                artifacts=["selected_evidence_pack", "topic_focused_references"],
                notes=[str(exc)],
            )
            self._record_decision(
                action="stop",
                reason=str(exc),
                target="literature_agent",
                gate_id="reference_bank_gate",
            )
            raise
        self._record_gate(
            gate_id="reference_bank_gate",
            ok=True,
            artifacts=["selected_evidence_pack", "topic_focused_references"],
            notes=[
                (
                    f"{grounding_contract_report['abstract_or_pdf_card_count']} abstract/PDF-backed cards meet "
                    f"minimum {grounding_contract_report['minimum_abstract_or_pdf_cards']}."
                ),
                f"{reference_contract_report['reference_count']} references meet hard target "
                f"{reference_contract_report['minimum_reference_target']}."
            ],
        )
        selected = dict(contract_payload["selected_candidate"])
        selected_title = str(selected.get("title") or "").strip()
        self._write_phase_artifact(3, "direction_contract_request.md", direction_user, "direction_contract_request", "prompt")
        self._write_phase_artifact(
            3,
            "candidate_directions.json",
            pipeline.dump_json({"candidates": candidates}),
            "candidate_directions",
            "json",
        )
        self._write_phase_artifact(
            3,
            "candidate_directions.md",
            pipeline.render_candidates_markdown(candidates),
            "candidate_directions_markdown",
            "markdown",
        )
        self._write_phase_artifact(
            3,
            "selected_direction.json",
            pipeline.dump_json(selected),
            "selected_direction",
            "json",
            frozen=True,
            freeze_reason="Selected direction is the Phase 1-to-Phase 2 topic contract.",
        )
        self._write_phase_artifact(
            3,
            "selected_evidence_pack.json",
            pipeline.dump_json(evidence_pack),
            "selected_evidence_pack",
            "json",
        )
        self._write_phase_artifact(
            3,
            "selected_evidence_pack.md",
            pipeline.render_evidence_pack_markdown(evidence_pack),
            "selected_evidence_pack_markdown",
            "markdown",
        )
        self._write_phase_artifact(
            3,
            "candidate_review.json",
            pipeline.dump_json(candidate_review),
            "candidate_review",
            "json",
        )
        self._write_phase_artifact(3, "review_report.json", pipeline.dump_json(candidate_review), "phase1_review_report", "json")
        self._write_phase_artifact(
            3,
            "contract_bundle.json",
            pipeline.dump_json(contract_payload),
            "contract_bundle",
            "json",
            frozen=True,
            freeze_reason="Contract bundle contains the Phase 2 seed contracts used by downstream agents.",
        )
        self._freeze_contract("problem_contract_seed", "phase1-3/contract_bundle.json", "Frozen seed for Phase 2 formulation.")
        self._freeze_contract("novelty_contract", "phase1-3/contract_bundle.json", "Frozen novelty boundary for Phase 2 and writing.")
        self._freeze_contract("proof_contract", "phase1-3/contract_bundle.json", "Frozen proof/theory route seed.")
        self._freeze_contract("validation_contract", "phase1-3/contract_bundle.json", "Frozen validation intent seed.")
        pipeline.write_phase_status(
            self.run_dir,
            3,
            (
                "direction_contract_request.md",
                "candidate_directions.json",
                "candidate_directions.md",
                "selected_direction.json",
                "selected_evidence_pack.json",
                "selected_evidence_pack.md",
                "candidate_review.json",
                "review_report.json",
                "contract_bundle.json",
            ),
        )

        handoff_payload = pipeline.finalize_handoff_payload(
            self.topic,
            contract_payload,
            evidence_pack,
            candidate_review,
            self.run_dir,
        )
        pipeline.validate_handoff_payload(handoff_payload)
        self._record_agent(
            agent_id="controller",
            role="finalize and mirror Phase 1 handoff artifacts for Phase 2",
            inputs=["contract_bundle", "selected_evidence_pack", "candidate_review"],
            outputs=["phase1_handoff", "phase1_tail_summary", "mirrored_handoff_artifacts"],
        )
        self._record_gate(
            gate_id="handoff_gate",
            ok=True,
            artifacts=["phase1_handoff"],
            notes=["Handoff payload contains required Phase 2 seed contracts."],
        )
        self._record_decision(
            action="export",
            reason="Phase 1 handoff is valid; mirror artifacts for downstream phases.",
            target="phase1_handoff",
            gate_id="handoff_gate",
        )
        hypotheses_md = pipeline.render_hypotheses_markdown(handoff_payload)
        topic_score = pipeline.render_topic_score(handoff_payload, candidate_review)
        summary = pipeline.render_tail_summary(self.run_dir, handoff_payload, topic_score)

        self._write_phase_artifact(
            4,
            "phase1_handoff.json",
            pipeline.dump_json(handoff_payload),
            "phase1_handoff",
            "json",
            frozen=True,
            freeze_reason="Phase 1 handoff is the exported contract consumed by Phase 2.",
        )
        self._write_phase_artifact(4, "hypotheses.md", hypotheses_md, "hypotheses", "markdown")
        self._write_phase_artifact(4, "topic_score.json", pipeline.dump_json(topic_score), "topic_score", "json")
        self._write_phase_artifact(4, "review_report.json", pipeline.dump_json(candidate_review), "handoff_review_report", "json")
        self._write_phase_artifact(4, "phase1_tail_summary.json", pipeline.dump_json(summary), "phase1_tail_summary", "json")
        self._write_phase_artifact(4, "controller_trace.json", pipeline.dump_json({"steps": trace}), "controller_trace", "json")
        pipeline.write_phase_status(
            self.run_dir,
            4,
            (
                "phase1_handoff.json",
                "hypotheses.md",
                "topic_score.json",
                "review_report.json",
                "phase1_tail_summary.json",
                "controller_trace.json",
            ),
        )

        pipeline.mirror_handoff_artifacts(
            source_run_dir=self.run_dir,
            handoff_dir=self.handoff_dir,
            handoff_payload=handoff_payload,
            evidence_pack=evidence_pack,
            candidates=candidates,
            candidate_review=candidate_review,
            hypotheses_md=hypotheses_md,
            topic_score=topic_score,
            summary=summary,
        )
        pipeline.write_run_summary(self.run_dir, self.handoff_dir, self.topic, selected_title, trace)

        return pipeline.WaraPhase1Result(
            run_id=self.run_dir.name,
            run_dir=self.run_dir,
            handoff_dir=self.handoff_dir,
            handoff_file=self.handoff_dir / "phase1_handoff.json",
            selected_title=selected_title,
        )

    def _write_phase_artifact(
        self,
        phase_num: int,
        name: str,
        content: str,
        artifact_id: str,
        kind: str,
        *,
        frozen: bool = False,
        freeze_reason: str = "",
    ) -> None:
        pipeline.write_phase_artifact(self.run_dir, phase_num, name, content)
        path = f"phase1-{phase_num}/{name}"
        self._record_artifact(
            artifact_id=artifact_id,
            path=path,
            kind=kind,
            producer=f"phase1.{phase_num}",
            frozen=frozen,
            freeze_reason=freeze_reason,
        )

    def _record_agent(self, *, agent_id: str, role: str, inputs: list[str], outputs: list[str]) -> None:
        self.manifest["agents"].append(
            {
                "id": agent_id,
                "role": role,
                "input_artifacts": inputs,
                "output_artifacts": outputs,
                "status": "completed",
                "recorded_at": pipeline.utcnow_iso(),
            }
        )

    def _record_artifact(
        self,
        *,
        artifact_id: str,
        path: str,
        kind: str,
        producer: str,
        frozen: bool = False,
        freeze_reason: str = "",
    ) -> None:
        artifact = {
            "id": artifact_id,
            "path": path,
            "kind": kind,
            "producer": producer,
            "frozen": frozen,
            "recorded_at": pipeline.utcnow_iso(),
        }
        if freeze_reason:
            artifact["freeze_reason"] = freeze_reason
        self.manifest["artifacts"].append(artifact)
        if frozen:
            self._freeze_contract(artifact_id, path, freeze_reason or "Frozen by Phase 1 controller.")

    def _record_gate(self, *, gate_id: str, ok: bool, artifacts: list[str], notes: list[str] | None = None) -> None:
        self.manifest["gates"].append(
            {
                "id": gate_id,
                "ok": ok,
                "artifacts": artifacts,
                "notes": notes or [],
                "checked_at": pipeline.utcnow_iso(),
            }
        )

    def _record_decision(self, *, action: str, reason: str, target: str, gate_id: str | None = None) -> None:
        decision = {
            "action": action,
            "target": target,
            "reason": reason,
            "recorded_at": pipeline.utcnow_iso(),
        }
        if gate_id:
            decision["gate_id"] = gate_id
        self.manifest["decisions"].append(decision)

    def _freeze_contract(self, artifact_id: str, path: str, reason: str) -> None:
        frozen = self.manifest["frozen_contracts"]
        if any(item.get("artifact_id") == artifact_id and item.get("path") == path for item in frozen):
            return
        frozen.append(
            {
                "artifact_id": artifact_id,
                "path": path,
                "reason": reason,
                "frozen_at": pipeline.utcnow_iso(),
            }
        )

    def _persist_manifest(self) -> None:
        self.manifest["updated_at"] = pipeline.utcnow_iso()
        pipeline.write_text(self.run_dir / "phase1_controller_manifest.json", pipeline.dump_json(self.manifest))
