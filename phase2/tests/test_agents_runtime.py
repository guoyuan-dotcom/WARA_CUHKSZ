from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from wara_core.agents import (
    CONTENT_AGENT_IDS,
    AgentController,
    ArtifactWorkspace,
    ExperimentAgent,
    RoleAgent,
    build_default_agent_registry,
    build_experiment_agent_task_prompt,
    phase_to_agent_ids,
    validate_agent_contract,
)
from phase2.scripts.pipeline_core.flow import _sync_experiment_agent
from phase2.scripts.run_phase24_simple_llm_experiment import publish_phase24_simple_as_phase25


class Phase2AgentsRuntimeTests(unittest.TestCase):
    def test_default_registry_defines_role_agents(self) -> None:
        registry = build_default_agent_registry()

        self.assertEqual(tuple(registry), CONTENT_AGENT_IDS)
        for contract in registry.values():
            self.assertEqual(validate_agent_contract(contract), [])
            self.assertGreater(len(contract.allowed_actions), 0)
            self.assertGreater(len(contract.forbidden_actions), 0)

        self.assertEqual(phase_to_agent_ids("2.4"), ("experiment_agent",))
        self.assertEqual(phase_to_agent_ids("2.5"), ("validation_agent",))
        self.assertEqual(phase_to_agent_ids("phase3.1"), ("writing_agent",))
        self.assertEqual(phase_to_agent_ids("phase3.2"), ("analysis_agent",))
        self.assertEqual(phase_to_agent_ids("3.2"), ("analysis_agent",))
        self.assertEqual(phase_to_agent_ids("phase3.4"), ("literature_agent", "writing_agent"))
        self.assertEqual(phase_to_agent_ids("phase3.6"), ("repair_agent", "writing_agent"))
        self.assertEqual(registry["literature_agent"].output_artifacts[1].path_hint, "phase3-4/verified_reference_bank.json")
        self.assertEqual(registry["writing_agent"].output_artifacts[0].path_hint, "phase3-1/system_model_problem_formulation_ieee_wcl.tex")

    def test_artifact_workspace_enforces_frozen_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "phase2-1").mkdir()
            (root / "phase2-1" / "mathematical_contract.json").write_text(
                '{"controls": []}', encoding="utf-8"
            )
            workspace = ArtifactWorkspace(root)
            controller = AgentController(root, workspace=workspace)

            workspace.register_artifact(
                "mathematical_contract",
                "phase2-1/mathematical_contract.json",
                producer_agent="formulation_agent",
                kind="json",
                frozen=False,
            )
            errors = workspace.validate_inputs(controller.registry["theory_agent"])
            self.assertTrue(any("must be frozen" in item for item in errors))

            workspace.freeze("mathematical_contract", reason="formulation gate passed")
            errors = workspace.validate_inputs(controller.registry["theory_agent"])
            self.assertTrue(any("system_model" in item for item in errors))
            self.assertFalse(any("mathematical_contract" in item and "frozen" in item for item in errors))

    def test_experiment_agent_builds_narrow_request_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            for rel in ["phase2-1", "phase2-2", "phase2-3", "phase2-4/solver", "phase2-5"]:
                (run_dir / rel).mkdir(parents=True)
            (run_dir / "phase2_summary.json").write_text('{"topic": "demo"}', encoding="utf-8")
            (run_dir / "phase2-1" / "mathematical_contract.json").write_text(
                '{"objective": {"sense": "maximize", "expression": "U"}}',
                encoding="utf-8",
            )
            (run_dir / "phase2-2" / "algorithm_contract.json").write_text(
                '{"algorithm_execution_contract": {"state_keys": ["x"]}}',
                encoding="utf-8",
            )
            (run_dir / "phase2-3" / "algorithm.md").write_text("Use SCA.", encoding="utf-8")
            (run_dir / "phase2-4" / "validation_plan.yaml").write_text(
                """
paper_evidence_contract:
  compared_methods:
    - id: proposed
    - id: fixed_baseline
  figures:
    - figure_id: fig1
required_outputs:
  scalar_metrics:
    - sum_rate_bpsHz
""".strip(),
                encoding="utf-8",
            )
            (run_dir / "phase2-4" / "phase24_validation_manifest.json").write_text(
                '{"status": "ok"}', encoding="utf-8"
            )
            (run_dir / "phase2-4" / "solver" / "generated_plugin.py").write_text(
                "def build_model():\n    return {}\n", encoding="utf-8"
            )
            (run_dir / "phase2-5" / "phase25_experiment_summary.json").write_text(
                '{"phase25_status": "paper_minimum_ready"}', encoding="utf-8"
            )

            agent = ExperimentAgent(run_dir)
            snapshot = agent.snapshot()
            payload = agent.build_request_payload()
            request_path = agent.write_request_payload()

            self.assertEqual(snapshot.phase24_status, "ok")
            self.assertEqual(snapshot.phase25_status, "paper_minimum_ready")
            self.assertEqual(snapshot.figure_count, 1)
            self.assertEqual(snapshot.benchmark_count, 2)
            self.assertIn("sum_rate_bpsHz", snapshot.metric_columns)
            self.assertEqual(payload["agent_id"], "experiment_agent")
            self.assertIn("change the mathematical objective", payload["forbidden_actions"])
            self.assertTrue(request_path.exists())
            self.assertEqual(json.loads(request_path.read_text(encoding="utf-8"))["agent_id"], "experiment_agent")

    def test_role_agent_bootstraps_theory_boundary_from_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            for rel in ["phase2-1", "phase2-2", "phase2-3"]:
                (run_dir / rel).mkdir(parents=True)
            (run_dir / "phase2-1" / "mathematical_contract.json").write_text(
                '{"controls": [{"id": "x"}], "objective": {"expression": "U"}}',
                encoding="utf-8",
            )
            (run_dir / "phase2-1" / "system_model.md").write_text("System model.", encoding="utf-8")
            (run_dir / "phase2-1" / "problem_formulation.md").write_text("Problem formulation.", encoding="utf-8")

            agent = RoleAgent(run_dir, "theory_agent")
            snapshot = agent.bootstrap(event="unit_test_theory")
            request_path = agent.write_request_payload()
            manifest = json.loads((run_dir / "agent_workspace_manifest.json").read_text(encoding="utf-8"))
            request = json.loads(request_path.read_text(encoding="utf-8"))

            self.assertEqual(snapshot.status, "ready_to_run")
            self.assertEqual(snapshot.missing_inputs, [])
            self.assertIn("algorithm_contract", snapshot.missing_outputs)
            self.assertIn("mathematical_contract", snapshot.frozen_contracts_present)
            self.assertTrue(manifest["artifacts"]["mathematical_contract"]["frozen"])
            self.assertEqual(request["agent_id"], "theory_agent")
            self.assertIn("system_model", request["inputs"])
            self.assertEqual(request["expected_outputs"]["algorithm_contract"]["path"], "phase2-2/algorithm_contract.json")

    def test_role_agent_bootstraps_writing_boundary_with_phase_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            for rel in ["phase2-1", "phase2-2", "phase2-5", "phase3-1", "phase3-2", "phase3-3", "phase3-4"]:
                (run_dir / rel).mkdir(parents=True)
            (run_dir / "phase2-1" / "mathematical_contract.json").write_text("{}", encoding="utf-8")
            (run_dir / "phase2-2" / "algorithm_contract.json").write_text("{}", encoding="utf-8")
            (run_dir / "phase2-5" / "phase25_experiment_summary.json").write_text(
                '{"phase25_status": "paper_minimum_ready"}',
                encoding="utf-8",
            )
            (run_dir / "phase3-1" / "system_model_problem_formulation_ieee_wcl.tex").write_text(
                "System model and problem formulation.", encoding="utf-8"
            )
            (run_dir / "phase3-2" / "numerical_results_section.tex").write_text("Numerical results.", encoding="utf-8")
            (run_dir / "phase3-3" / "phase3_3_technical_sections_preview.tex").write_text("Technical sections.", encoding="utf-8")
            (run_dir / "phase3-4" / "full_paper_preview.pdf").write_text("pdf placeholder", encoding="utf-8")
            (run_dir / "phase3-4" / "verified_reference_bank.json").write_text("[]", encoding="utf-8")

            agent = RoleAgent(run_dir, "writing_agent")
            snapshot = agent.bootstrap(event="unit_test_writing")
            request = agent.build_request_payload()

            self.assertEqual(snapshot.status, "ready")
            self.assertEqual(snapshot.missing_inputs, [])
            self.assertEqual(snapshot.missing_outputs, [])
            self.assertEqual(snapshot.present_outputs["technical_sections"], "phase3-3/phase3_3_technical_sections_preview.tex")
            self.assertIn("numerical_results", request["inputs"])
            self.assertEqual(request["inputs"]["reference_bank"]["payload"], [])

    def test_experiment_agent_prompt_wraps_phase_task_with_narrow_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            for rel in ["phase2-1", "phase2-2", "phase2-3", "phase2-4/solver", "phase2-5"]:
                (run_dir / rel).mkdir(parents=True)
            (run_dir / "phase2_summary.json").write_text('{"topic": "demo"}', encoding="utf-8")
            (run_dir / "phase2-1" / "mathematical_contract.json").write_text(
                '{"objective": {"expression": "psi"}}', encoding="utf-8"
            )
            (run_dir / "phase2-1" / "system_model.md").write_text("System model marker.", encoding="utf-8")
            (run_dir / "phase2-1" / "problem_formulation.md").write_text("Problem marker.", encoding="utf-8")
            (run_dir / "phase2-2" / "algorithm_contract.json").write_text(
                '{"method_ids": ["proposed"]}', encoding="utf-8"
            )
            (run_dir / "phase2-2" / "reformulation_path.md").write_text("Reformulation marker.", encoding="utf-8")
            (run_dir / "phase2-3" / "algorithm.md").write_text("Algorithm marker.", encoding="utf-8")

            prompt = build_experiment_agent_task_prompt(
                run_dir=run_dir,
                task_kind="validation_plan",
                output_contract="Return JSON with validation_plan_yaml.",
                legacy_task_prompt="Legacy schema marker.",
            )

            self.assertIn("WARA ExperimentAgent", prompt)
            self.assertIn("validation_plan", prompt)
            self.assertIn("Return JSON with validation_plan_yaml.", prompt)
            self.assertIn("Legacy schema marker.", prompt)
            self.assertIn("mathematical_contract", prompt)
            self.assertIn("System model marker.", prompt)
            self.assertIn("Algorithm marker.", prompt)
            self.assertTrue((run_dir / "phase2-4" / "experiment_agent_request.json").exists())

    def test_experiment_agent_marks_present_but_failed_evidence_as_partial(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            for rel in ["phase2-1", "phase2-2", "phase2-3", "phase2-4/solver/outputs", "phase2-5/figures"]:
                (run_dir / rel).mkdir(parents=True)
            (run_dir / "phase2_summary.json").write_text('{"topic": "demo"}', encoding="utf-8")
            (run_dir / "phase2-1" / "mathematical_contract.json").write_text("{}", encoding="utf-8")
            (run_dir / "phase2-2" / "algorithm_contract.json").write_text("{}", encoding="utf-8")
            (run_dir / "phase2-3" / "algorithm.md").write_text("Algorithm.", encoding="utf-8")
            (run_dir / "phase2-4" / "validation_plan.yaml").write_text("paper_evidence_contract: {}\n", encoding="utf-8")
            (run_dir / "phase2-4" / "phase24_validation_manifest.json").write_text(
                '{"status": "smoke_failed"}', encoding="utf-8"
            )
            (run_dir / "phase2-4" / "solver" / "generated_plugin.py").write_text("# code\n", encoding="utf-8")
            (run_dir / "phase2-4" / "solver" / "outputs" / "validation_results.csv").write_text(
                "method,objective\nproposed,1\n", encoding="utf-8"
            )
            (run_dir / "phase2-5" / "phase25_experiment_summary.json").write_text(
                '{"phase25_status": "needs_more_phase24_runs"}', encoding="utf-8"
            )

            snapshot = ExperimentAgent(run_dir).snapshot()

            self.assertEqual(snapshot.missing_artifacts, [])
            self.assertEqual(snapshot.status, "partial")
            self.assertIn("Phase 2.4 validation status is `smoke_failed`.", snapshot.notes)
            self.assertIn("Phase 2.5 evidence status is `needs_more_phase24_runs`.", snapshot.notes)

    def test_experiment_agent_accepts_simple_ready_path_over_legacy_phase24_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            for rel in ["phase2-1", "phase2-2", "phase2-3", "phase2-4/solver/outputs", "phase2-4-simple", "phase2-5/figures"]:
                (run_dir / rel).mkdir(parents=True)
            (run_dir / "phase2_summary.json").write_text('{"topic": "demo"}', encoding="utf-8")
            (run_dir / "phase2-1" / "mathematical_contract.json").write_text("{}", encoding="utf-8")
            (run_dir / "phase2-2" / "algorithm_contract.json").write_text("{}", encoding="utf-8")
            (run_dir / "phase2-3" / "algorithm.md").write_text("Algorithm.", encoding="utf-8")
            (run_dir / "phase2-4" / "validation_plan.yaml").write_text("paper_evidence_contract: {}\n", encoding="utf-8")
            (run_dir / "phase2-4" / "phase24_validation_manifest.json").write_text(
                '{"status": "smoke_failed"}', encoding="utf-8"
            )
            (run_dir / "phase2-4" / "solver" / "generated_plugin.py").write_text("# code\n", encoding="utf-8")
            (run_dir / "phase2-4" / "solver" / "outputs" / "validation_results.csv").write_text(
                "method,objective\nproposed,1\n", encoding="utf-8"
            )
            (run_dir / "phase2-4-simple" / "focused_experiment_manifest.json").write_text(
                '{"status": "ok"}', encoding="utf-8"
            )
            (run_dir / "phase2-5" / "phase25_experiment_summary.json").write_text(
                '{"phase25_status": "paper_minimum_ready"}', encoding="utf-8"
            )

            snapshot = ExperimentAgent(run_dir).snapshot()

            self.assertEqual(snapshot.status, "ready")
            self.assertNotIn("Phase 2.4 validation status is `smoke_failed`.", snapshot.notes)

    def test_experiment_agent_accepts_focused_artifacts_without_legacy_solver(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            for rel in ["phase2-1", "phase2-2", "phase2-3", "phase2-4-simple/outputs", "phase2-5/figures"]:
                (run_dir / rel).mkdir(parents=True)
            (run_dir / "phase2_summary.json").write_text('{"topic": "generic wireless demo"}', encoding="utf-8")
            (run_dir / "phase2-1" / "mathematical_contract.json").write_text("{}", encoding="utf-8")
            (run_dir / "phase2-2" / "algorithm_contract.json").write_text("{}", encoding="utf-8")
            (run_dir / "phase2-3" / "algorithm.md").write_text("Algorithm.", encoding="utf-8")
            (run_dir / "phase2-4-simple" / "experiment_plan.json").write_text('{"methods": ["proposed", "baseline"]}', encoding="utf-8")
            (run_dir / "phase2-4-simple" / "focused_experiment.py").write_text("# generated focused experiment\n", encoding="utf-8")
            (run_dir / "phase2-4-simple" / "focused_experiment_manifest.json").write_text('{"status": "ok"}', encoding="utf-8")
            (run_dir / "phase2-4-simple" / "outputs" / "simple_results.csv").write_text(
                "figure_id,method,seed,swept_param,swept_value,throughput_bpsHz,feasible_numeric\n"
                "fig_gain,proposed,1,power,1,10,1\n"
                "fig_gain,baseline,1,power,1,8,1\n",
                encoding="utf-8",
            )
            (run_dir / "phase2-5" / "phase25_experiment_summary.json").write_text(
                '{"phase25_status": "paper_minimum_ready", "plotted_methods": ["proposed", "baseline"]}',
                encoding="utf-8",
            )

            agent = ExperimentAgent(run_dir)
            snapshot = agent.bootstrap(event="focused_only_unit_test")
            manifest = json.loads((run_dir / "agent_workspace_manifest.json").read_text(encoding="utf-8"))

            self.assertEqual(snapshot.status, "ready")
            self.assertEqual(snapshot.present_artifacts["experiment_plan"], "phase2-4-simple/experiment_plan.json")
            self.assertEqual(snapshot.present_artifacts["main_experiment_code"], "phase2-4-simple/focused_experiment.py")
            self.assertEqual(snapshot.present_artifacts["validation_results"], "phase2-4-simple/outputs/simple_results.csv")
            self.assertEqual(snapshot.benchmark_count, 2)
            self.assertIn("throughput_bpsHz", snapshot.metric_columns)
            self.assertEqual(manifest["artifacts"]["experiment_plan"]["path"], "phase2-4-simple/experiment_plan.json")
            self.assertEqual(manifest["artifacts"]["main_experiment_code"]["path"], "phase2-4-simple/focused_experiment.py")

    def test_phase24_to_phase25_adapter_uses_dynamic_figure_and_metric_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            out_dir = run_dir / "phase2-4-simple"
            outputs = out_dir / "outputs"
            figures = out_dir / "figures"
            outputs.mkdir(parents=True)
            figures.mkdir(parents=True)
            (run_dir / "phase2_summary.json").write_text('{"topic": "generic wireless demo"}', encoding="utf-8")
            (outputs / "simple_results.csv").write_text(
                "figure_id,method,seed,swept_param,swept_value,throughput_bpsHz,feasible_numeric,num_users_actual\n"
                "fig_gain,proposed,1,power_budget,1,10,1,4\n"
                "fig_gain,baseline,1,power_budget,1,8,1,4\n"
                "fig_gain,proposed,2,power_budget,2,12,1,4\n"
                "fig_gain,baseline,2,power_budget,2,9,1,4\n"
                "fig_sensitivity,proposed,1,mobility,1,9,1,4\n"
                "fig_sensitivity,baseline,1,mobility,1,7,1,4\n",
                encoding="utf-8",
            )
            (outputs / "simple_summary.json").write_text(
                json.dumps(
                    {
                        "preview_passed": True,
                        "methods": ["proposed", "baseline"],
                        "primary_benchmark": "baseline",
                        "primary_claim": "The proposed design improves throughput.",
                        "claim_evidence": {"primary_metric": "throughput_bpsHz", "primary_metric_symbol": "$T$"},
                    }
                ),
                encoding="utf-8",
            )
            (outputs / "preview_quality_report.json").write_text('{"preview_passed": true}', encoding="utf-8")
            (outputs / "benchmark_selection_report.json").write_text(
                json.dumps(
                    {
                        "final_plotted_method_set": ["proposed", "baseline"],
                        "primary_practical_benchmark": "baseline",
                        "method_display_names": {"proposed": "Proposed", "baseline": "Baseline"},
                    }
                ),
                encoding="utf-8",
            )
            (outputs / "figure_selection_report.json").write_text(
                json.dumps(
                    {
                        "figures": [
                            {
                                "figure_id": "fig_gain",
                                "filename": "figures/custom_gain.png",
                                "x_axis_param": "power_budget",
                                "x_axis_label": "$B$",
                                "y_metric": "throughput_bpsHz",
                                "y_axis_label": "$T$",
                            },
                            {
                                "figure_id": "fig_sensitivity",
                                "filename": "figures/custom_sensitivity.png",
                                "x_axis_param": "mobility",
                                "x_axis_label": "$v$",
                                "y_metric": "throughput_bpsHz",
                                "y_axis_label": "$T$",
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )

            old_min_x = os.environ.get("WARA_PHASE24_SIMPLE_MIN_X_POINTS")
            old_min_seeds = os.environ.get("WARA_PHASE24_SIMPLE_MIN_SEEDS")
            os.environ["WARA_PHASE24_SIMPLE_MIN_X_POINTS"] = "1"
            os.environ["WARA_PHASE24_SIMPLE_MIN_SEEDS"] = "1"
            export = publish_phase24_simple_as_phase25(run_dir, out_dir)
            if old_min_x is None:
                os.environ.pop("WARA_PHASE24_SIMPLE_MIN_X_POINTS", None)
            else:
                os.environ["WARA_PHASE24_SIMPLE_MIN_X_POINTS"] = old_min_x
            if old_min_seeds is None:
                os.environ.pop("WARA_PHASE24_SIMPLE_MIN_SEEDS", None)
            else:
                os.environ["WARA_PHASE24_SIMPLE_MIN_SEEDS"] = old_min_seeds
            summary = json.loads((run_dir / "phase2-5" / "phase25_experiment_summary.json").read_text(encoding="utf-8"))

            self.assertTrue(export["paper_minimum_ready"])
            self.assertEqual(summary["primary_metric"]["name"], "throughput_bpsHz")
            self.assertEqual(summary["plotted_methods"], ["proposed", "baseline"])
            self.assertEqual(summary["figures"][0]["source_phase24_figure_id"], "fig_gain")
            self.assertEqual(summary["figures"][0]["x_axis_param"], "power_budget")
            self.assertEqual(summary["figures"][0]["y_metric"], "throughput_bpsHz")
            self.assertEqual(summary["figures"][1]["source_phase24_figure_id"], "fig_sensitivity")
            self.assertEqual(summary["figures"][1]["x_axis_param"], "mobility")
            self.assertEqual(summary["claim_evidence"]["proposed_feasibility_rate"], 1.0)

    def test_flow_sync_writes_experiment_agent_event_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            for rel in ["phase2-1", "phase2-2", "phase2-3", "phase2-4/solver/outputs", "phase2-5/figures"]:
                (run_dir / rel).mkdir(parents=True)
            (run_dir / "phase2_summary.json").write_text('{"topic": "demo"}', encoding="utf-8")
            (run_dir / "phase2-1" / "mathematical_contract.json").write_text("{}", encoding="utf-8")
            (run_dir / "phase2-2" / "algorithm_contract.json").write_text("{}", encoding="utf-8")
            (run_dir / "phase2-3" / "algorithm.md").write_text("Algorithm.", encoding="utf-8")
            (run_dir / "phase2-4" / "validation_plan.yaml").write_text("paper_evidence_contract: {}\n", encoding="utf-8")
            (run_dir / "phase2-4" / "phase24_validation_manifest.json").write_text('{"status": "ok"}', encoding="utf-8")
            (run_dir / "phase2-4" / "solver" / "generated_plugin.py").write_text("# code\n", encoding="utf-8")
            (run_dir / "phase2-4" / "solver" / "outputs" / "validation_results.csv").write_text("method,objective\nproposed,1\n", encoding="utf-8")
            (run_dir / "phase2-5" / "phase25_experiment_summary.json").write_text(
                '{"phase25_status": "paper_minimum_ready"}', encoding="utf-8"
            )

            payload = _sync_experiment_agent(
                run_dir,
                event="unit_test_event",
                extra_metadata={"marker": "flow-sync"},
            )

            event_path = run_dir / "phase2-4" / "experiment_agent_unit_test_event.json"
            request_path = run_dir / "phase2-4" / "experiment_agent_request.json"
            manifest_path = run_dir / "agent_workspace_manifest.json"
            self.assertTrue(event_path.exists())
            self.assertTrue(request_path.exists())
            self.assertTrue(manifest_path.exists())
            self.assertEqual(payload["event"], "unit_test_event")
            self.assertEqual(payload["snapshot"]["phase24_status"], "ok")
            self.assertEqual(json.loads(event_path.read_text(encoding="utf-8"))["metadata"]["marker"], "flow-sync")


if __name__ == "__main__":
    unittest.main()
