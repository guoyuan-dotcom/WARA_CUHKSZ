from __future__ import annotations

import json
import csv
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


def _csv_columns(path: Path) -> list[str]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            return [str(item).strip() for item in next(reader, []) if str(item).strip()]
    except (FileNotFoundError, OSError, StopIteration):
        return []


@dataclass
class ExperimentAgentSnapshot:
    """Current experiment-loop state derived from an existing Phase 2/3 run."""

    run_dir: str
    status: str
    present_artifacts: dict[str, str]
    missing_artifacts: list[str]
    frozen_contracts_present: list[str]
    phase24_status: str = "unknown"
    phase25_status: str = "unknown"
    figure_count: int = 0
    benchmark_count: int = 0
    metric_columns: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ExperimentAgent:
    """Runtime facade for the Phase 2.4 ExperimentAgent boundary."""

    id = "experiment_agent"

    def __init__(self, run_dir: Path, *, workspace: ArtifactWorkspace | None = None) -> None:
        self.run_dir = Path(run_dir)
        self.contract = get_agent_contract(self.id)
        self.controller = AgentController(self.run_dir, workspace=workspace)

    def bootstrap(
        self,
        *,
        event: str = "bootstrap",
        message: str = "Bootstrapped ExperimentAgent state from existing Phase 2.4 artifacts.",
        metadata: dict[str, Any] | None = None,
    ) -> ExperimentAgentSnapshot:
        self.controller.bootstrap_known_phase_artifacts()
        snapshot = self.snapshot()
        run_metadata = snapshot.to_dict()
        if metadata:
            run_metadata["event_metadata"] = metadata
        self.controller.record_phase_agent_run(
            phase_id="phase2.4",
            agent_id=self.id,
            status=snapshot.status,
            output_artifacts=list(snapshot.present_artifacts),
            message=f"{event}: {message}",
            metadata=run_metadata,
        )
        return snapshot

    def snapshot(self) -> ExperimentAgentSnapshot:
        present: dict[str, str] = {}
        missing: list[str] = []
        frozen_present: list[str] = []
        for spec in self.contract.input_artifacts + self.contract.output_artifacts:
            path = next((candidate for candidate in self._artifact_candidates(spec.id, spec.path_hint) if candidate.exists()), None)
            if path is not None:
                present[spec.id] = str(path.relative_to(self.run_dir))
                if spec.frozen_required or spec.id in self.contract.frozen_contracts:
                    frozen_present.append(spec.id)
            elif spec.required:
                missing.append(spec.id)

        phase24_manifest = _read_json(self.run_dir / "phase2-4" / "phase24_validation_manifest.json") or {}
        simple_manifest = (
            _read_json(self.run_dir / "phase2-4-simple" / "focused_experiment_manifest.json")
            or _read_json(self.run_dir / "phase2-4-simple" / "two_call_preview_manifest.json")
            or _read_json(self.run_dir / "phase2-4-simple" / "simple_experiment_manifest.json")
            or {}
        )
        phase25_summary = _read_json(self.run_dir / "phase2-5" / "phase25_experiment_summary.json") or {}
        experiment_plan = (
            _read_yaml(self.run_dir / "phase2-4" / "validation_plan.yaml")
            or _read_json(self.run_dir / "phase2-4-simple" / "experiment_plan.json")
            or _read_json(self.run_dir / "phase2-5" / "experiment_plan.json")
            or {}
        )

        phase24_status = str(phase24_manifest.get("status") or phase24_manifest.get("compile_status") or "unknown")
        simple_status = str(simple_manifest.get("status") or "unknown")
        phase25_status = str(
            phase25_summary.get("phase25_status")
            or phase25_summary.get("overall_status")
            or phase25_summary.get("status")
            or "unknown"
        )
        figure_count = self._figure_count(experiment_plan, phase25_summary)
        benchmark_count = self._benchmark_count(experiment_plan, phase25_summary)
        metric_columns = self._metric_columns(experiment_plan, phase25_summary)
        for csv_path in (
            self.run_dir / "phase2-4" / "solver" / "outputs" / "validation_results.csv",
            self.run_dir / "phase2-4-simple" / "outputs" / "simple_results.csv",
        ):
            metric_columns.extend(_csv_columns(csv_path))
        metric_columns = sorted(set(metric_columns))
        notes: list[str] = []
        if "mathematical_contract" not in frozen_present:
            notes.append("Frozen mathematical contract is not registered/present for ExperimentAgent.")
        if "algorithm_contract" not in frozen_present:
            notes.append("Frozen algorithm contract is not registered/present for ExperimentAgent.")
        simple_path_ready = simple_status == "ok" and phase25_status in {
            "paper_minimum_ready",
            "paper_preferred_ready",
            "high_confidence_ready",
        }
        if phase24_status not in {"ok", "unknown"} and not simple_path_ready:
            notes.append(f"Phase 2.4 validation status is `{phase24_status}`.")
        if phase25_status not in {"paper_minimum_ready", "paper_preferred_ready", "high_confidence_ready", "unknown"}:
            notes.append(f"Phase 2.5 evidence status is `{phase25_status}`.")

        status = "ready" if not missing and not notes else ("partial" if present else "empty")
        return ExperimentAgentSnapshot(
            run_dir=str(self.run_dir),
            status=status,
            present_artifacts=present,
            missing_artifacts=missing,
            frozen_contracts_present=sorted(set(frozen_present)),
            phase24_status=phase24_status,
            phase25_status=phase25_status,
            figure_count=figure_count,
            benchmark_count=benchmark_count,
            metric_columns=metric_columns,
            notes=notes,
        )

    def _artifact_candidates(self, artifact_id: str, primary_path_hint: str) -> tuple[Path, ...]:
        """Return topic-agnostic legacy and focused Phase 2.4 artifact paths."""

        alternatives: dict[str, tuple[str, ...]] = {
            "experiment_plan": (
                "phase2-4/validation_plan.yaml",
                "phase2-4-simple/experiment_plan.json",
                "phase2-5/experiment_plan.json",
            ),
            "main_experiment_code": (
                "phase2-4/solver/generated_plugin.py",
                "phase2-4-simple/focused_experiment.py",
                "phase2-4-simple/simple_experiment.py",
                "phase2-4-simple/preview_experiment.py",
            ),
            "validation_results": (
                "phase2-4/solver/outputs/validation_results.csv",
                "phase2-4-simple/outputs/simple_results.csv",
            ),
            "figure_data": (
                "phase2-5/figures",
                "phase2-4-simple/figures",
            ),
            "experiment_report": (
                "phase2-5/phase25_experiment_summary.json",
                "phase2-4-simple/outputs/simple_summary.json",
            ),
        }
        hints = (primary_path_hint,) + tuple(item for item in alternatives.get(artifact_id, ()) if item != primary_path_hint)
        return tuple(self.run_dir / hint for hint in hints)

    def build_request_payload(self) -> dict[str, Any]:
        """Assemble the narrow context future LLM calls should receive."""

        snapshot = self.snapshot()
        payload = {
            "agent_id": self.id,
            "role": self.contract.role,
            "allowed_actions": list(self.contract.allowed_actions),
            "forbidden_actions": list(self.contract.forbidden_actions),
            "gates": [gate.to_dict() for gate in self.contract.gates],
            "snapshot": snapshot.to_dict(),
            "inputs": {
                "mathematical_contract": _read_json(self.run_dir / "phase2-1" / "mathematical_contract.json"),
                "system_model_md": _read_text(self.run_dir / "phase2-1" / "system_model.md"),
                "problem_formulation_md": _read_text(self.run_dir / "phase2-1" / "problem_formulation.md"),
                "algorithm_contract": _read_json(self.run_dir / "phase2-2" / "algorithm_contract.json"),
                "reformulation_path_md": _read_text(self.run_dir / "phase2-2" / "reformulation_path.md"),
                "algorithm_description_md": _read_text(self.run_dir / "phase2-3" / "algorithm.md"),
                "paper_target": _read_json(self.run_dir / "phase2_summary.json"),
            },
            "existing_experiment_artifacts": {
                "validation_plan": _read_yaml(self.run_dir / "phase2-4" / "validation_plan.yaml"),
                "phase24_execution_contract": _read_json(self.run_dir / "phase2-4" / "phase24_execution_contract.json"),
                "wireless_benchmark_plan": _read_json(self.run_dir / "phase2-4" / "wireless_benchmark_plan.json"),
                "experiment_design_contract": _read_json(self.run_dir / "phase2-4" / "experiment_design_contract.json"),
                "phase25_summary": _read_json(self.run_dir / "phase2-5" / "phase25_experiment_summary.json"),
                "phase24_simple_summary": _read_json(self.run_dir / "phase2-4-simple" / "outputs" / "simple_summary.json"),
                "phase24_preview_quality": _read_json(self.run_dir / "phase2-4-simple" / "outputs" / "preview_quality_report.json"),
                "phase24_benchmark_selection": _read_json(self.run_dir / "phase2-4-simple" / "outputs" / "benchmark_selection_report.json"),
                "phase24_figure_selection": _read_json(self.run_dir / "phase2-4-simple" / "outputs" / "figure_selection_report.json"),
            },
            "expected_outputs": [artifact.to_dict() for artifact in self.contract.output_artifacts],
        }
        return payload

    def write_request_payload(self, path: Path | None = None) -> Path:
        target = path or (self.run_dir / "phase2-4" / "experiment_agent_request.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.build_request_payload(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return target

    @staticmethod
    def _figure_count(plan: Any, summary: Any) -> int:
        if isinstance(summary, dict):
            figures = summary.get("figures") or summary.get("figure_summaries")
            if isinstance(figures, list):
                return len(figures)
        if isinstance(plan, dict):
            for key in ("figure_specs", "figures", "figure_candidates"):
                figures = plan.get(key)
                if isinstance(figures, list):
                    return len(figures)
            contract = plan.get("paper_evidence_contract") or plan.get("research_evidence_contract")
            if isinstance(contract, dict) and isinstance(contract.get("figures"), list):
                return len(contract["figures"])
        return 0

    @staticmethod
    def _benchmark_count(plan: Any, summary: Any) -> int:
        if isinstance(summary, dict):
            methods = summary.get("methods") or summary.get("compared_methods") or summary.get("plotted_methods")
            if isinstance(methods, list):
                return len(methods)
            if isinstance(methods, dict):
                return len(methods)
        if isinstance(plan, dict):
            for key in ("compared_methods", "methods"):
                methods = plan.get(key)
                if isinstance(methods, list):
                    return len(methods)
            contract = plan.get("paper_evidence_contract") or plan.get("research_evidence_contract")
            if isinstance(contract, dict) and isinstance(contract.get("compared_methods"), list):
                return len(contract["compared_methods"])
        return 0

    @staticmethod
    def _metric_columns(plan: Any, summary: Any) -> list[str]:
        columns: list[str] = []
        if isinstance(summary, dict):
            for key in ("metric_columns", "required_columns", "result_columns"):
                values = summary.get(key)
                if isinstance(values, list):
                    columns.extend(str(item) for item in values if str(item).strip())
        if isinstance(plan, dict):
            required = plan.get("required_outputs")
            if isinstance(required, dict):
                values = required.get("scalar_metrics")
                if isinstance(values, list):
                    columns.extend(str(item) for item in values if str(item).strip())
            contract = plan.get("paper_evidence_contract") or plan.get("research_evidence_contract")
            if isinstance(contract, dict):
                values = contract.get("required_result_columns")
                if isinstance(values, list):
                    columns.extend(str(item) for item in values if str(item).strip())
        return sorted(set(columns))
