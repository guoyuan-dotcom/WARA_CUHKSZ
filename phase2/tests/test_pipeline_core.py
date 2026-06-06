from __future__ import annotations

import ast
import json
import os
import sys
import tempfile
import unittest
from unittest import mock
from collections import Counter
from pathlib import Path

import yaml


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
PHASE1_ENGINE_DIR = Path(__file__).resolve().parents[2] / "phase1" / "engine"
if str(PHASE1_ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE1_ENGINE_DIR))

from pipeline_core import (
    AgentSpec,
    ArtifactRef,
    GateResult,
    Phase2FlowCallbacks,
    Phase2RunState,
    Phase2RunSummary,
    WaraController,
    build_wara_phase1_handoff,
    build_algorithm_contract,
    build_claim_map,
    build_experiment_design_contract,
    build_problem_contract,
    execute_phase3_flow,
    execute_phase2_flow,
    find_default_phase1_handoff,
    build_tractability_route_policy,
    looks_like_phase1_handoff,
    make_run_id,
    make_phase2_phase_flow,
    normalize_model_profile,
    select_wireless_benchmark_plan,
)
from pipeline_core.flow import (
    _phase3_1_compile_issues_should_block,
    _phase3_1_technical_writing_contract,
    _phase25_auto_paper_run_limit,
    _phase25_is_paper_ready,
    _phase25_paper_quality_requires_phase24_design_revision,
    _phase25_sweep_mode_for_round,
    _phase25_defer_initial_refiner_redesign_until_sweep,
    _phase24_selected_candidate_can_continue,
    _run_phase25_auto_paper_expansion,
    _phase3_5_review_blocks_downstream,
    _phase3_6_final_revision_ready,
    _sync_role_agent,
)
from phase_runtime.phase24_plan import normalize_phase24_validation_plan_yaml
from phase_runtime.phase21_23_foundation import validate_phase2_phase1_mathematical_contract_schema, validate_phase2_phase3_contract
from run_phase_pipeline import topic_from_wara_phase1_handoff, validate_phase1_quality_gate, validate_wara_phase1_handoff_gate


class Phase2PipelineCoreTests(unittest.TestCase):
    def test_phase_flow_starts_with_first_phase_running(self) -> None:
        phases = make_phase2_phase_flow()
        self.assertEqual(len(phases), 11)
        self.assertEqual(phases[0]["status"], "running")
        self.assertEqual(phases[3]["phase_id"], "phase2.4")
        self.assertEqual(phases[5]["phase_step"], 6)
        self.assertEqual(phases[0]["phase"], "phase2")
        self.assertEqual(phases[6]["phase"], "phase3")
        self.assertEqual(phases[6]["phase_id"], "phase3.2")
        self.assertEqual(phases[-1]["phase_id"], "phase3.6")
        self.assertEqual(phases[5]["phase_id"], "phase3.1")
        self.assertIn("Technical Sections", phases[5]["name"])
        self.assertTrue(all(phase["status"] == "ready" for phase in phases[1:]))

    def test_backend_model_profile_accepts_kimi_profiles(self) -> None:
        self.assertEqual(normalize_model_profile("kimi-k2.6-no-thinking"), "kimi-k2.6-no-thinking")
        self.assertEqual(normalize_model_profile("kimi-k2.6-thinking"), "kimi-k2.6-thinking")
        self.assertEqual(normalize_model_profile("kimi"), "kimi-k2.6-no-thinking")
        self.assertEqual(normalize_model_profile("gpt-5.5"), "openai-gpt-5.5")
        self.assertEqual(normalize_model_profile("codex"), "openai-gpt-5.3-codex")
        self.assertEqual(normalize_model_profile("deepseek"), "deepseek-chat")
        self.assertEqual(normalize_model_profile("deepseek-r1"), "deepseek-reasoner")

    def test_phase31_contract_blocks_ambiguous_multi_equation_leadin(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "phase3-1").mkdir(parents=True)
            report = _phase3_1_technical_writing_contract(
                run_dir=run_dir,
                system_tex="",
                proposed_tex=(
                    "For the communication terms, define\n"
                    "\\begin{align}\n"
                    "a_k &\\triangleq b_k.\\label{eq:a}\\\\\n"
                    "c_k &= d_k.\\label{eq:c}\n"
                    "\\end{align}\n"
                ),
            )
        self.assertFalse(report["ok"])
        self.assertEqual(report["checks"]["ambiguous_multi_equation_leadin_count"], 1)
        self.assertTrue(any("multiple displayed equations" in error for error in report["errors"]))

    def test_phase31_contract_allows_and_respectively_multi_equation_style(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "phase3-1").mkdir(parents=True)
            report = _phase3_1_technical_writing_contract(
                run_dir=run_dir,
                system_tex="",
                proposed_tex=(
                    "The quantities are defined as\n"
                    "\\begin{align}\n"
                    "a_k &\\triangleq b_k,\\label{eq:a}\\\\\n"
                    "\\text{and}\\quad c_k &= d_k,\\label{eq:c}\n"
                    "\\end{align}\n"
                    "respectively.\n"
                ),
            )
        self.assertTrue(report["ok"], report)
        self.assertEqual(report["checks"]["ambiguous_multi_equation_leadin_count"], 0)

    def test_phase24_selected_candidate_allows_usable_evidence_after_bounded_repairs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            phase24 = run_dir / "phase2-4"
            outputs = phase24 / "solver" / "outputs"
            outputs.mkdir(parents=True)
            rows = ["method,figure_id,sweep_id,swept_param,swept_value,seed,objective,sum_rate_bpsHz"]
            for seed in range(10):
                rows.append(f"proposed,figure_1,sweep_a,power,{seed % 2},{seed},1.2,1.2")
                rows.append(f"benchmark,figure_1,sweep_a,power,{seed % 2},{seed},1.0,1.0")
            (outputs / "validation_results.csv").write_text(
                "\n".join(rows) + "\n",
                encoding="utf-8",
            )
            (phase24 / "phase24_evidence_contract_check.json").write_text(
                json.dumps({"ok": True}),
                encoding="utf-8",
            )
            (phase24 / "phase24_method_semantics_check.json").write_text(
                json.dumps({"ok": True}),
                encoding="utf-8",
            )
            (phase24 / "phase24_experiment_responsiveness_check.json").write_text(
                json.dumps({"ok": False, "checks": [{"relative_metric_span": 0.05, "num_x_values": 3}]}),
                encoding="utf-8",
            )
            (phase24 / "phase24_pilot_gain_check.json").write_text(
                json.dumps(
                    {
                        "ok": True,
                        "pilot_win_rate": 1.0,
                        "pilot_median_relative_gain": 0.2,
                    }
                ),
                encoding="utf-8",
            )
            self.assertTrue(
                _phase24_selected_candidate_can_continue(
                    run_dir,
                    {"status": "experiment_responsiveness_failed", "design_repair_recommended": True},
                )
            )
        self.assertEqual(normalize_model_profile(None), "kimi-k2.6-no-thinking")

    def test_phase25_auto_paper_sweep_failure_is_structured(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            phase25_dir = run_dir / "phase2-5"
            phase25_dir.mkdir(parents=True)
            (phase25_dir / "paper_sweep_plan.json").write_text(
                json.dumps({"figures": [{"figure_id": "figure_1"}]}),
                encoding="utf-8",
            )

            def fail_paper_sweep(_: Path, __: bool = False) -> dict:
                raise RuntimeError("paper sweep crashed")

            def unreachable(*_args, **_kwargs):
                raise AssertionError("phase25 reanalysis should not run after sweep failure")

            callbacks = Phase2FlowCallbacks(
                build_pipeline_experiment_design_notes=lambda: "notes",
                build_phase1_handoff=lambda _phase1, _run: {},
                build_phase3_design_notes=lambda: "phase3 notes",
                build_phase24_design_notes=lambda: "phase24 notes",
                extract_latex_issue_summary=lambda _log, _tex: "",
                render_phase1_ieee_preview_pdf=lambda _phase: {"preview_pdf": "phase1.pdf"},
                render_phase3_ieee_preview_pdf=lambda _phase: {"preview_pdf": "phase3.pdf"},
                render_phase3_1_technical_preview_pdf=lambda _phase: {"preview_pdf": "phase3_1.pdf"},
                repair_phase2_phase1_latex_llm=lambda **_: "",
                repair_phase2_phase3_latex_llm=lambda **_: "",
                repair_phase3_1_latex_llm=lambda **_: {},
                repair_phase2_phase24_plugin_llm=lambda **_: "",
                run_phase3_6_apply_review_fixes_package=unreachable,
                run_phase2_phase1_latex_llm=lambda **_: "latex",
                run_phase2_phase1_llm=lambda **_: {},
                run_phase2_phase2_llm=lambda **_: {},
                run_phase2_phase3_latex_llm=lambda **_: "latex",
                run_phase2_phase3_llm=lambda **_: {},
                run_phase3_1_writing_llm=lambda **_: {},
                run_phase2_phase24_benchmark_llm=lambda **_: {},
                run_phase2_phase24_plugin_llm=lambda **_: "",
                run_phase2_phase24_validation_llm=lambda **_: "",
                run_phase24_paper_sweep_from_plan=fail_paper_sweep,
                run_phase25_wcl_package=unreachable,
                run_phase3_2_numerical_results_package=unreachable,
                run_phase3_3_technical_sections_package=unreachable,
                run_phase3_4_introduction_references_package=unreachable,
                run_phase3_5_paper_review_package=unreachable,
                phase24_validation_allows_repair=lambda _status: False,
                phase24_validation_error_text=lambda _run, _status: "",
                validate_phase24_evidence_contract_design=lambda _run: {"ok": True, "errors": [], "warnings": []},
                validate_phase2_phase24_plugin_bundle=lambda _run: {"status": "ok"},
                write_phase2_phase24_fixed_harness=lambda _run: None,
            )

            result, status, manifest = _run_phase25_auto_paper_expansion(
                run_dir=run_dir,
                callbacks=callbacks,
                initial_result={"phase25_status": "needs_more_phase24_runs"},
                initial_status="needs_more_phase24_runs",
            )

            self.assertEqual(result["phase25_status"], "needs_more_phase24_runs")
            self.assertEqual(status, "needs_more_phase24_runs")
            self.assertEqual(manifest["reason"], "auto_expansion_failed:RuntimeError")
            self.assertFalse(manifest["rounds"][0]["paper_ready_after"])
            self.assertEqual(manifest["rounds"][0]["mode"], "scout")
            self.assertEqual(manifest["rounds"][0]["sweep_result"]["error"], "paper sweep crashed")
            self.assertEqual(manifest["rounds"][0]["sweep_result"]["validation_output_prefix"], "scout_validation")
            self.assertTrue((run_dir / "phase2-4" / "phase25_auto_paper_sweep_round1_error.json").exists())
            self.assertTrue((run_dir / "phase2-5" / "phase25_auto_expansion_manifest.json").exists())

    def test_phase25_quick_evidence_does_not_skip_auto_expansion(self) -> None:
        previous = os.environ.get("WARA_PHASE25_BOUNDED_BUDGET_CONTINUE")
        os.environ["WARA_PHASE25_BOUNDED_BUDGET_CONTINUE"] = "1"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                run_dir = Path(tmp)
                phase25_dir = run_dir / "phase2-5"
                phase25_dir.mkdir(parents=True)
                (phase25_dir / "paper_sweep_plan.json").write_text(
                    json.dumps({"figures": [{"figure_id": "figure_1"}]}),
                    encoding="utf-8",
                )
                (phase25_dir / "phase25_experiment_summary.json").write_text(
                    json.dumps(
                        {
                            "phase25_status": "quick_mode_only",
                            "num_comparable_cases": 160,
                            "proposed_win_rate": 1.0,
                            "proposed_median_relative_gain": 0.12,
                            "primary_claim_check": {
                                "passes": True,
                                "proposed_win_rate": 1.0,
                                "proposed_median_relative_gain": 0.12,
                            },
                            "strongest_practical_baseline_audit": {
                                "passes": True,
                                "proposed_win_rate": 1.0,
                                "proposed_median_relative_gain": 0.12,
                            },
                        }
                    ),
                    encoding="utf-8",
                )

                class TrackingCallbacks:
                    sweep_modes: list[str] = []

                    def run_phase24_paper_sweep_from_plan(self, *_args, **_kwargs) -> dict:
                        self.sweep_modes.append(os.environ.get("WARA_PHASE25_SWEEP_TIER", ""))
                        return {"ok": True}

                    def run_phase25_wcl_package(self, *_args, **_kwargs) -> dict:
                        payload = {"phase25_status": "paper_minimum_ready", "paper_minimum_ready": True}
                        (phase25_dir / "phase25_experiment_summary.json").write_text(
                            json.dumps(payload),
                            encoding="utf-8",
                        )
                        return payload

                callbacks = TrackingCallbacks()

                result, status, manifest = _run_phase25_auto_paper_expansion(
                    run_dir=run_dir,
                    callbacks=callbacks,
                    initial_result={"phase25_status": "quick_mode_only"},
                    initial_status="quick_mode_only",
                )

                self.assertEqual(result["phase25_status"], "paper_minimum_ready")
                self.assertEqual(status, "paper_minimum_ready")
                self.assertEqual(manifest["reason"], "paper_ready_after_auto_expansion")
                self.assertEqual(manifest["rounds"][0]["mode"], "scout")
                self.assertEqual(callbacks.sweep_modes, ["scout"])
        finally:
            if previous is None:
                os.environ.pop("WARA_PHASE25_BOUNDED_BUDGET_CONTINUE", None)
            else:
                os.environ["WARA_PHASE25_BOUNDED_BUDGET_CONTINUE"] = previous

    def test_phase25_refiner_phase24_revision_status_propagates_to_final_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            phase25_dir = run_dir / "phase2-5"
            phase25_dir.mkdir(parents=True)
            for name, payload in {
                "experiment_plan.json": {"primary_metric": {"name": "weighted_sum_rate_bpsHz"}},
                "available_data_summary.json": {},
                "phase25_experiment_summary.json": {"phase25_status": "needs_more_phase24_runs"},
                "paper_sweep_plan.json": {"figures": [{"figure_id": "figure_1"}]},
            }.items():
                (phase25_dir / name).write_text(json.dumps(payload), encoding="utf-8")
            (phase25_dir / "missing_experiments.md").write_text("not enough evidence", encoding="utf-8")

            class UnreachableCallbacks:
                def run_phase24_paper_sweep_from_plan(self, *_args, **_kwargs) -> dict:
                    raise AssertionError("paper sweep should not run after phase2.4 redesign request")

                def run_phase25_wcl_package(self, *_args, **_kwargs) -> dict:
                    raise AssertionError("phase2.5 reanalysis should not run after phase2.4 redesign request")

            with mock.patch(
                "pipeline_core.flow._phase25_refine_sweep_plan",
                return_value={"status": "requires_phase24_design_revision", "notes": ["experiment design mismatch"]},
            ):
                _result, status, manifest = _run_phase25_auto_paper_expansion(
                    run_dir=run_dir,
                    callbacks=UnreachableCallbacks(),
                    initial_result={"phase25_status": "needs_more_phase24_runs"},
                    initial_status="needs_more_phase24_runs",
                )

            self.assertEqual(status, "requires_phase24_design_revision")
            self.assertEqual(manifest["final_phase25_status"], "requires_phase24_design_revision")
            self.assertEqual(manifest["reason"], "requires_phase24_design_revision_before_paper_sweep")
            self.assertEqual(manifest["rounds"], [])

    def test_phase25_initial_refiner_redesign_is_deferred_when_quick_design_is_valid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "phase2-4").mkdir(parents=True)
            (run_dir / "phase2-5").mkdir(parents=True)
            (run_dir / "phase2-4" / "phase24_evidence_contract_design_check.json").write_text(
                json.dumps({"ok": True, "errors": []}),
                encoding="utf-8",
            )
            (run_dir / "phase2-5" / "paper_sweep_plan.json").write_text(
                json.dumps({"figures": [{"figure_id": "figure_1"}]}),
                encoding="utf-8",
            )
            (run_dir / "phase2-5" / "phase25_experiment_summary.json").write_text(
                json.dumps({"primary_metric": {"name": "service_margin_tau"}}),
                encoding="utf-8",
            )

            self.assertTrue(
                _phase25_defer_initial_refiner_redesign_until_sweep(
                    run_dir,
                    "quick_mode_only",
                    {"status": "requires_phase24_design_revision", "notes": ["sparse quick data"]},
                )
            )

    def test_phase25_unpromising_scout_after_redesign_budget_routes_to_phase24_revision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            phase25_dir = run_dir / "phase2-5"
            phase25_dir.mkdir(parents=True)
            for name, payload in {
                "experiment_plan.json": {"primary_metric": {"name": "weighted_sum_rate_bpsHz"}},
                "available_data_summary.json": {},
                "phase25_experiment_summary.json": {"phase25_status": "needs_more_phase24_runs"},
                "paper_sweep_plan.json": {"figures": [{"figure_id": "figure_1"}]},
            }.items():
                (phase25_dir / name).write_text(json.dumps(payload), encoding="utf-8")
            (phase25_dir / "missing_experiments.md").write_text("needs scout evidence", encoding="utf-8")

            class Callbacks:
                def run_phase24_paper_sweep_from_plan(self, _path: Path, quick: bool = False) -> dict:
                    return {"ok": True, "quick_mode": quick, "validation_output_prefix": "scout_validation"}

                def run_phase25_wcl_package(self, path: Path) -> dict:
                    payload = {
                        "phase25_status": "quick_mode_only",
                        "num_comparable_cases": 80,
                        "proposed_win_rate": 0.25,
                        "proposed_median_relative_gain": -0.05,
                        "primary_claim_check": {"passes": False},
                        "strongest_practical_baseline_audit": {"passes": False},
                    }
                    (path / "phase2-5" / "phase25_experiment_summary.json").write_text(json.dumps(payload), encoding="utf-8")
                    return payload

            old_env = {name: os.environ.get(name) for name in [
                "WARA_PHASE25_AUTO_PAPER_MODE",
                "WARA_PHASE25_AUTO_PAPER_RUNS",
                "WARA_PHASE25_REDESIGN_ROUNDS",
                "WARA_PHASE25_COVERAGE_EXTENSION_ROUNDS",
            ]}
            try:
                os.environ["WARA_PHASE25_AUTO_PAPER_MODE"] = "auto"
                os.environ["WARA_PHASE25_AUTO_PAPER_RUNS"] = "1"
                os.environ["WARA_PHASE25_REDESIGN_ROUNDS"] = "0"
                os.environ["WARA_PHASE25_COVERAGE_EXTENSION_ROUNDS"] = "0"
                with mock.patch(
                    "pipeline_core.flow._phase25_refine_sweep_plan",
                    return_value={"status": "ok", "figures": [{"figure_id": "figure_1"}]},
                ):
                    _result, status, manifest = _run_phase25_auto_paper_expansion(
                        run_dir=run_dir,
                        callbacks=Callbacks(),
                        initial_result={"phase25_status": "needs_more_phase24_runs"},
                        initial_status="needs_more_phase24_runs",
                    )
            finally:
                for name, value in old_env.items():
                    if value is None:
                        os.environ.pop(name, None)
                    else:
                        os.environ[name] = value

            self.assertEqual(status, "requires_phase24_design_revision")
            self.assertEqual(manifest["final_phase25_status"], "requires_phase24_design_revision")
            self.assertEqual(manifest["reason"], "requires_phase24_design_revision_after_unpromising_scout")
            self.assertEqual(manifest["rounds"][0]["mode"], "scout")

    def test_phase25_paper_quality_routes_noncoverage_blockers_to_phase24_revision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            phase25_dir = run_dir / "phase2-5"
            phase25_dir.mkdir(parents=True)
            (phase25_dir / "plot_quality_report.json").write_text(
                json.dumps(
                    {
                        "figures": [
                            {
                                "figure_id": "figure_1",
                                "counts_toward_paper_minimum": True,
                                "required_sweep": "aperture_sweep",
                                "x_axis_param": "geometry.aperture",
                                "y_metric": "weighted_sum_rate_bpsHz",
                                "blocking_issues": ["metric_constant_across_sweep"],
                            },
                            {
                                "figure_id": "figure_2",
                                "counts_toward_paper_minimum": True,
                                "required_sweep": "aperture_sweep",
                                "x_axis_param": "geometry.aperture",
                                "y_metric": "weighted_sum_rate_bpsHz",
                                "blocking_issues": [],
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )

            result = _phase25_paper_quality_requires_phase24_design_revision(run_dir)

            self.assertTrue(result["requires_phase24_design_revision"])
            issues = [issue for item in result["design_blockers"] for issue in item["blocking_issues"]]
            self.assertIn("metric_constant_across_sweep", issues)
            self.assertIn("duplicate_final_figure_story", issues)
            preserve_ids = [item["figure_id"] for item in result["repair_scope"]["preserve_figures"]]
            redesign_ids = [item["figure_id"] for item in result["repair_scope"]["redesign_figures"]]
            self.assertIn("figure_2", preserve_ids)
            self.assertIn("figure_1", redesign_ids)

    def test_phase25_paper_quality_allows_pure_coverage_blockers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            phase25_dir = run_dir / "phase2-5"
            phase25_dir.mkdir(parents=True)
            (phase25_dir / "plot_quality_report.json").write_text(
                json.dumps(
                    {
                        "figures": [
                            {
                                "figure_id": "figure_1",
                                "counts_toward_paper_minimum": True,
                                "required_sweep": "load_sweep",
                                "x_axis_param": "system.K",
                                "y_metric": "weighted_sum_rate_bpsHz",
                                "blocking_issues": ["too_few_x_points", "too_few_seeds_per_point"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            result = _phase25_paper_quality_requires_phase24_design_revision(run_dir)

            self.assertFalse(result["requires_phase24_design_revision"])

    def test_phase25_promising_paper_sweep_gets_coverage_extension(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            phase25_dir = run_dir / "phase2-5"
            phase25_dir.mkdir(parents=True)
            (phase25_dir / "paper_sweep_plan.json").write_text(
                json.dumps({"figures": [{"figure_id": "figure_1", "suggested_values": [1, 2, 3]}]}),
                encoding="utf-8",
            )
            calls: list[bool] = []
            statuses = iter(["needs_more_phase24_runs", "paper_minimum_ready"])

            def paper_sweep(_: Path, quick: bool = False) -> dict:
                calls.append(quick)
                return {"ok": True, "num_cases": 100, "quick_mode": quick}

            def reanalyze(run_path: Path, *_args, **_kwargs) -> dict:
                status = next(statuses)
                summary = {
                    "phase25_status": status,
                    "num_comparable_cases": 80,
                    "proposed_win_rate": 1.0,
                    "proposed_median_relative_gain": 0.08,
                    "primary_claim_check": {"passes": True},
                    "strongest_practical_baseline_audit": {"passes": True},
                }
                (run_path / "phase2-5" / "phase25_experiment_summary.json").write_text(
                    json.dumps(summary),
                    encoding="utf-8",
                )
                return {"phase25_status": status}

            def unreachable(*_args, **_kwargs):
                raise AssertionError("downstream phases are not part of this controller unit test")

            callbacks = Phase2FlowCallbacks(
                build_pipeline_experiment_design_notes=lambda: "notes",
                build_phase1_handoff=lambda _phase1, _run: {},
                build_phase3_design_notes=lambda: "phase3 notes",
                build_phase24_design_notes=lambda: "phase24 notes",
                extract_latex_issue_summary=lambda _log, _tex: "",
                render_phase1_ieee_preview_pdf=lambda _phase: {"preview_pdf": "phase1.pdf"},
                render_phase3_ieee_preview_pdf=lambda _phase: {"preview_pdf": "phase3.pdf"},
                render_phase3_1_technical_preview_pdf=lambda _phase: {"preview_pdf": "phase3_1.pdf"},
                repair_phase2_phase1_latex_llm=lambda **_: "",
                repair_phase2_phase3_latex_llm=lambda **_: "",
                repair_phase3_1_latex_llm=lambda **_: {},
                repair_phase2_phase24_plugin_llm=lambda **_: "",
                run_phase3_6_apply_review_fixes_package=unreachable,
                run_phase2_phase1_latex_llm=lambda **_: "latex",
                run_phase2_phase1_llm=lambda **_: {},
                run_phase2_phase2_llm=lambda **_: {},
                run_phase2_phase3_latex_llm=lambda **_: "latex",
                run_phase2_phase3_llm=lambda **_: {},
                run_phase3_1_writing_llm=lambda **_: {},
                run_phase2_phase24_benchmark_llm=lambda **_: {},
                run_phase2_phase24_plugin_llm=lambda **_: "",
                run_phase2_phase24_validation_llm=lambda **_: "",
                run_phase24_paper_sweep_from_plan=paper_sweep,
                run_phase25_wcl_package=reanalyze,
                run_phase3_2_numerical_results_package=unreachable,
                run_phase3_3_technical_sections_package=unreachable,
                run_phase3_4_introduction_references_package=unreachable,
                run_phase3_5_paper_review_package=unreachable,
                phase24_validation_allows_repair=lambda _status: False,
                phase24_validation_error_text=lambda _run, _status: "",
                validate_phase24_evidence_contract_design=lambda _run: {"ok": True, "errors": [], "warnings": []},
                validate_phase2_phase24_plugin_bundle=lambda _run: {"status": "ok"},
                write_phase2_phase24_fixed_harness=lambda _run: None,
            )

            old_env = {name: os.environ.get(name) for name in [
                "WARA_PHASE25_AUTO_PAPER_MODE",
                "WARA_PHASE25_AUTO_PAPER_RUNS",
                "WARA_PHASE25_REDESIGN_ROUNDS",
                "WARA_PHASE25_COVERAGE_EXTENSION_ROUNDS",
            ]}
            try:
                os.environ["WARA_PHASE25_AUTO_PAPER_MODE"] = "paper"
                os.environ["WARA_PHASE25_AUTO_PAPER_RUNS"] = "1"
                os.environ["WARA_PHASE25_REDESIGN_ROUNDS"] = "0"
                os.environ["WARA_PHASE25_COVERAGE_EXTENSION_ROUNDS"] = "1"
                result, status, manifest = _run_phase25_auto_paper_expansion(
                    run_dir=run_dir,
                    callbacks=callbacks,
                    initial_result={"phase25_status": "needs_more_phase24_runs"},
                    initial_status="needs_more_phase24_runs",
                )
            finally:
                for name, value in old_env.items():
                    if value is None:
                        os.environ.pop(name, None)
                    else:
                        os.environ[name] = value

            self.assertEqual(status, "paper_minimum_ready")
            self.assertEqual(result["phase25_status"], "paper_minimum_ready")
            self.assertEqual(calls, [False, False])
            self.assertEqual(manifest["reason"], "paper_ready_after_auto_expansion")
            self.assertEqual(len(manifest["coverage_extensions"]), 1)

    def test_run_state_persists_transitions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            summary = Phase2RunSummary(
                run_id=make_run_id("demo topic"),
                topic="demo topic",
                created_at="2026-04-28T00:00:00+00:00",
                root=str(run_dir),
                phase1_run=None,
                model_profile="kimi-k2.6-no-thinking",
                phases=make_phase2_phase_flow(),
            )
            state = Phase2RunState(run_dir, summary)
            state.persist()
            state.complete_phase(0, 1)
            state.block_phase(4)

            payload = json.loads((run_dir / "phase2_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["phases"][0]["status"], "done")
            self.assertEqual(payload["phases"][1]["status"], "running")
            self.assertEqual(payload["phases"][4]["status"], "blocked")

    def test_controller_freezes_contract_and_assembles_bounded_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "phase2-1").mkdir()
            (run_dir / "phase2-1" / "mathematical_contract.json").write_text(
                json.dumps({"controls": [{"symbol": "w_k"}]}),
                encoding="utf-8",
            )
            (run_dir / "phase2-2").mkdir()
            (run_dir / "phase2-2" / "algorithm_contract.json").write_text(
                json.dumps({"algorithm_execution_contract": {"state_keys": ["W"]}}),
                encoding="utf-8",
            )

            controller = WaraController(run_dir)
            controller.register_artifact(
                ArtifactRef(
                    id="mathematical_contract",
                    path="phase2-1/mathematical_contract.json",
                    kind="json",
                    producer="formulation_agent",
                )
            )
            controller.register_artifact(
                ArtifactRef(
                    id="algorithm_contract",
                    path="phase2-2/algorithm_contract.json",
                    kind="json",
                    producer="theory_agent",
                )
            )
            controller.freeze_artifact("mathematical_contract", reason="MathContractGate passed")
            controller.register_agent(
                AgentSpec(
                    id="experiment_agent",
                    role="implement solver from frozen contracts",
                    input_artifacts=["algorithm_contract"],
                    output_artifacts=["generated_experiment_core"],
                    frozen_contracts=["mathematical_contract"],
                )
            )

            context = controller.assemble_context("experiment_agent")
            self.assertEqual(context["artifacts"]["algorithm_contract"]["algorithm_execution_contract"]["state_keys"], ["W"])
            self.assertEqual(context["frozen_contracts"]["mathematical_contract"]["controls"][0]["symbol"], "w_k")

    def test_controller_routes_gate_failures_to_bounded_repair_agents(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = WaraController(Path(tmp))
            decision = controller.record_gate(
                GateResult(
                    gate_id="ImplementationGate",
                    ok=False,
                    artifact_ids=["generated_experiment_core"],
                    errors=["Python import failed: missing required function proposed_step"],
                )
            )
            self.assertEqual(decision.action, "repair")
            self.assertEqual(decision.target_agent, "implementation_repair_agent")
            self.assertEqual(decision.repair_scope, "implementation_only")

    def test_phase3_1_technical_closure_report_is_advisory_not_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "phase3-1").mkdir(parents=True)
            report = _phase3_1_technical_writing_contract(
                run_dir=run_dir,
                system_tex=r"\section{System Model} Chance constraints use outage tolerance epsilon.",
                proposed_tex=(
                    r"For each margin, the selected ambiguity model provides "
                    r"$\Phi_m\ge0$ with $(\bm\lambda,\bm\mu)\in\mathcal C_\Phi$. "
                    r"The safe conic counterpart guarantees reliability."
                ),
            )

            self.assertTrue(report["ok"])
            self.assertTrue(report["advisory_only"])
            self.assertTrue(any("Phi-style" in item for item in report["warnings"]))
            self.assertTrue((run_dir / "phase3-1" / "phase3_1_technical_writing_contract_report.json").exists())

    def test_phase3_1_technical_closure_report_warns_on_solver_wrapper_algorithm_block(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "phase3-1").mkdir(parents=True)
            report = _phase3_1_technical_writing_contract(
                run_dir=run_dir,
                system_tex=r"\section{System Model} A generic model is considered.",
                proposed_tex=(
                    r"\begin{algorithm}[!t]"
                    r"\caption{Wrapper}"
                    r"\begin{algorithmic}[1]"
                    r"\Require Inputs."
                    r"\Ensure Solution."
                    r"\State Form problem (P1)."
                    r"\State Solve problem (P1)."
                    r"\State Return the solution."
                    r"\end{algorithmic}"
                    r"\end{algorithm}"
                ),
            )

            self.assertTrue(report["ok"])
            self.assertTrue(any("generic solver wrapper" in item or "route-specific construction" in item for item in report["warnings"]))
            self.assertFalse(report["checks"]["algorithm_block_has_route_specific_steps"])

    def test_phase3_1_contract_blocks_formula_list_system_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "phase3-1").mkdir(parents=True)
            system_tex = (
                r"\begin{equation} a=b \end{equation}"
                r"\begin{equation} c=d \end{equation}"
                r"\begin{equation} e=f \end{equation}"
                r"\begin{equation} g=h \end{equation}"
                r"\begin{equation} i=j \end{equation}"
                r"\begin{equation} k=l \end{equation}"
            )
            report = _phase3_1_technical_writing_contract(
                run_dir=run_dir,
                system_tex=system_tex,
                proposed_tex=r"\begin{algorithm}[!t]\begin{algorithmic}[1]\State Construct model.\State Update variables.\State Evaluate KPI.\State Return physical solution.\end{algorithmic}\end{algorithm}",
            )

            self.assertFalse(report["ok"])
            self.assertFalse(report["advisory_only"])
            self.assertTrue(any("equation-list-like" in item for item in report["errors"]))
            self.assertEqual(report["checks"]["system_equation_count"], 6)

    def test_controller_routes_responsiveness_failures_to_experiment_design_repair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = WaraController(Path(tmp))
            decision = controller.record_gate(
                GateResult(
                    gate_id="implementation_gate",
                    ok=False,
                    artifact_ids=["validation_plan", "generated_plugin"],
                    errors=[
                        "experiment_responsiveness: figure_2 metric harvested_energy_mW is too weakly responsive across sw_Qmin_mW"
                    ],
                )
            )
            self.assertEqual(decision.action, "repair")
            self.assertEqual(decision.target_agent, "experiment_design_repair_agent")
            self.assertEqual(decision.repair_scope, "figure_sweep_metric_or_operating_regime")

    def test_controller_records_phase3_5_review_routing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = WaraController(Path(tmp))
            decision = controller.record_review_routing(
                {
                    "gate_id": "review_gate",
                    "status": "repair_required",
                    "next_agent": "repair_agent",
                    "target_agent": "experiment_agent",
                    "primary_issue_id": "P0-EXP-01",
                    "primary_reason": "experiment_evidence_or_figure",
                    "routes": [
                        {
                            "issue_id": "P0-EXP-01",
                            "title": "Figure evidence does not support the claim",
                            "target_agent": "experiment_agent",
                        }
                    ],
                }
            )

            self.assertEqual(decision.action, "repair")
            self.assertEqual(decision.target_agent, "repair_agent")
            self.assertEqual(decision.owner_agent, "experiment_agent")
            self.assertEqual(decision.rerun_phase, "phase2.4")
            manifest = json.loads((Path(tmp) / "controller_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["review_routing"]["controller_decision"]["owner_agent"], "experiment_agent")
            self.assertFalse(manifest["gates"][-1]["ok"])

    def test_phase3_5_writing_review_routes_to_phase3_6_even_when_major(self) -> None:
        self.assertFalse(
            _phase3_5_review_blocks_downstream(
                {
                    "status": "repair_required",
                    "recommendation": "major_revision_needed",
                    "blocking": True,
                    "routes": [
                        {
                            "priority": "P1",
                            "target_agent": "writing_agent",
                            "issue_id": "P1-LATEX-OVERFULL",
                        }
                    ],
                }
            )
        )

    def test_phase3_6_final_revision_gate_requires_clean_manifest(self) -> None:
        ready, blockers = _phase3_6_final_revision_ready(
            {
                "ready_to_submit_estimate": False,
                "unresolved_issue_count": 2,
                "missing_reference_keys": ["RefA"],
                "compile_status": "ok",
                "reference_status": "ok",
                "experiment_status": "paper_minimum_ready",
            }
        )

        self.assertFalse(ready)
        self.assertTrue(any("ready_to_submit_estimate" in item for item in blockers))
        self.assertTrue(any("unresolved_issue_count=2" in item for item in blockers))
        self.assertTrue(any("missing_reference_keys" in item for item in blockers))

    def test_controller_stops_phase3_5_repair_after_round_limit(self) -> None:
        old_limit = os.environ.get("WCL_PHASE34_REPAIR_ROUNDS")
        old_global_limit = os.environ.get("WARA_MAX_REVIEW_REPAIR_ROUNDS")
        os.environ["WCL_PHASE34_REPAIR_ROUNDS"] = "2"
        os.environ.pop("WARA_MAX_REVIEW_REPAIR_ROUNDS", None)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                controller = WaraController(Path(tmp))
                routing = {
                    "gate_id": "review_gate",
                    "status": "repair_required",
                    "next_agent": "repair_agent",
                    "target_agent": "writing_agent",
                    "primary_issue_id": "P0-WRT-01",
                    "primary_reason": "paper_polish_default",
                    "routes": [
                        {
                            "issue_id": "P0-WRT-01",
                            "title": "Claims need another writing pass",
                            "target_agent": "writing_agent",
                        }
                    ],
                }

                first = controller.record_review_routing(routing)
                second = controller.record_review_routing(routing)
                third = controller.record_review_routing(routing)

                self.assertEqual(first.action, "repair")
                self.assertEqual(second.action, "repair")
                self.assertEqual(third.action, "stop")
                self.assertEqual(third.target_agent, "manual_triage")
                self.assertIn("round limit reached", third.reason)
                manifest = json.loads((Path(tmp) / "controller_manifest.json").read_text(encoding="utf-8"))
                self.assertEqual(manifest["review_repair_policy"]["max_auto_repair_rounds"], 2)
                self.assertEqual(manifest["review_repair_policy"]["completed_auto_repair_rounds"], 2)
        finally:
            if old_limit is None:
                os.environ.pop("WCL_PHASE34_REPAIR_ROUNDS", None)
            else:
                os.environ["WCL_PHASE34_REPAIR_ROUNDS"] = old_limit
            if old_global_limit is None:
                os.environ.pop("WARA_MAX_REVIEW_REPAIR_ROUNDS", None)
            else:
                os.environ["WARA_MAX_REVIEW_REPAIR_ROUNDS"] = old_global_limit

    def test_main_script_has_no_duplicate_top_level_function_names(self) -> None:
        script_path = SCRIPTS_DIR / "run_phase_pipeline.py"
        tree = ast.parse(script_path.read_text(encoding="utf-8"))
        names = [
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        ]
        duplicates = {name: count for name, count in Counter(names).items() if count > 1}
        self.assertEqual(duplicates, {})

    def test_main_script_imports_flow_module(self) -> None:
        script_path = SCRIPTS_DIR / "run_phase_pipeline.py"
        text = script_path.read_text(encoding="utf-8")
        self.assertIn("execute_phase2_flow", text)
        self.assertIn("Phase2FlowCallbacks", text)

    def test_phase1_quality_gate_blocks_revise_handoff(self) -> None:
        previous = os.environ.pop("WARA_ALLOW_WEAK_PHASE1", None)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                run_dir = Path(tmp)
                phase3_4 = run_dir / "phase3-4"
                phase3_4.mkdir(parents=True)
                (phase3_4 / "topic_score.json").write_text(
                    json.dumps({"overall_score": 6.3, "verdict": "revise"}),
                    encoding="utf-8",
                )
                (phase3_4 / "review_report.json").write_text(
                    json.dumps({"overall_recommendation": "revise"}),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(SystemExit, "quality gate blocked"):
                    validate_phase1_quality_gate(run_dir)
        finally:
            if previous is not None:
                os.environ["WARA_ALLOW_WEAK_PHASE1"] = previous

    def test_phase25_ready_statuses_are_strict(self) -> None:
        self.assertTrue(_phase25_is_paper_ready("paper_minimum_ready"))
        self.assertTrue(_phase25_is_paper_ready("paper_preferred_ready"))
        self.assertTrue(_phase25_is_paper_ready("high_confidence_ready"))
        self.assertFalse(_phase25_is_paper_ready("needs_more_phase24_runs"))
        self.assertFalse(_phase25_is_paper_ready("claim_failure_needs_redesign"))

    def test_phase25_auto_sweep_mode_progresses_by_round(self) -> None:
        previous = os.environ.pop("WARA_PHASE25_AUTO_PAPER_MODE", None)
        previous_legacy = os.environ.pop("WCL_PHASE25_AUTO_PAPER_MODE", None)
        previous_runs = os.environ.pop("WARA_PHASE25_AUTO_PAPER_RUNS", None)
        previous_legacy_runs = os.environ.pop("WCL_PHASE25_AUTO_PAPER_RUNS", None)
        try:
            self.assertEqual(_phase25_auto_paper_run_limit(), 10)
            self.assertEqual(_phase25_sweep_mode_for_round(1), "scout")
            self.assertEqual(_phase25_sweep_mode_for_round(2), "medium")
            self.assertEqual(_phase25_sweep_mode_for_round(3), "paper")
            os.environ["WARA_PHASE25_AUTO_PAPER_RUNS"] = "2"
            self.assertEqual(_phase25_auto_paper_run_limit(), 2)
            os.environ.pop("WARA_PHASE25_AUTO_PAPER_RUNS", None)
            os.environ["WARA_PHASE25_AUTO_PAPER_MODE"] = "medium"
            self.assertEqual(_phase25_sweep_mode_for_round(1), "medium")
            self.assertEqual(_phase25_sweep_mode_for_round(3), "medium")
        finally:
            if previous is not None:
                os.environ["WARA_PHASE25_AUTO_PAPER_MODE"] = previous
            else:
                os.environ.pop("WARA_PHASE25_AUTO_PAPER_MODE", None)
            if previous_legacy is not None:
                os.environ["WCL_PHASE25_AUTO_PAPER_MODE"] = previous_legacy
            else:
                os.environ.pop("WCL_PHASE25_AUTO_PAPER_MODE", None)
            if previous_runs is not None:
                os.environ["WARA_PHASE25_AUTO_PAPER_RUNS"] = previous_runs
            else:
                os.environ.pop("WARA_PHASE25_AUTO_PAPER_RUNS", None)
            if previous_legacy_runs is not None:
                os.environ["WCL_PHASE25_AUTO_PAPER_RUNS"] = previous_legacy_runs
            else:
                os.environ.pop("WCL_PHASE25_AUTO_PAPER_RUNS", None)

    def test_phase25_default_auto_expansion_runs_scout_medium_then_paper(self) -> None:
        previous_mode = os.environ.pop("WARA_PHASE25_AUTO_PAPER_MODE", None)
        previous_legacy_mode = os.environ.pop("WCL_PHASE25_AUTO_PAPER_MODE", None)
        previous_runs = os.environ.pop("WARA_PHASE25_AUTO_PAPER_RUNS", None)
        previous_legacy_runs = os.environ.pop("WCL_PHASE25_AUTO_PAPER_RUNS", None)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                run_dir = Path(tmp)
                phase25_dir = run_dir / "phase2-5"
                phase25_dir.mkdir(parents=True)
                (phase25_dir / "paper_sweep_plan.json").write_text(json.dumps({"figures": [{"figure_id": "figure_1"}]}), encoding="utf-8")
                calls: list[tuple[str, bool]] = []
                reanalysis_calls = {"count": 0}

                class Callbacks:
                    def run_phase24_paper_sweep_from_plan(self, path: Path, quick: bool = False) -> dict:
                        mode = os.environ.get("WARA_PHASE25_SWEEP_TIER", "")
                        calls.append((mode, quick))
                        return {"validation_output_prefix": f"{mode}_validation", "quick_mode": quick}

                    def run_phase25_wcl_package(self, path: Path) -> dict:
                        reanalysis_calls["count"] += 1
                        status = "paper_minimum_ready" if reanalysis_calls["count"] >= 3 else "needs_more_phase24_runs"
                        payload = {
                            "phase25_status": status,
                            "figures": [],
                            "num_comparable_cases": 4,
                            "proposed_win_rate": 1.0,
                            "proposed_median_relative_gain": 0.1,
                            "primary_claim_check": {"passes": True, "proposed_win_rate": 1.0, "proposed_median_relative_gain": 0.1},
                            "strongest_practical_baseline_audit": {"passes": True, "proposed_win_rate": 1.0, "proposed_median_relative_gain": 0.1},
                        }
                        (path / "phase2-5" / "phase25_experiment_summary.json").write_text(json.dumps(payload), encoding="utf-8")
                        return payload

                _result, status, manifest = _run_phase25_auto_paper_expansion(
                    run_dir=run_dir,
                    callbacks=Callbacks(),
                    initial_result={"phase25_status": "needs_more_phase24_runs"},
                    initial_status="needs_more_phase24_runs",
                )

                self.assertEqual(status, "paper_minimum_ready")
                self.assertEqual(calls, [("scout", True), ("medium", True), ("paper", False)])
                self.assertEqual([item["mode"] for item in manifest["rounds"]], ["scout", "medium", "paper"])
            self.assertEqual(manifest["round_limit"], 10)
        finally:
            if previous_mode is not None:
                os.environ["WARA_PHASE25_AUTO_PAPER_MODE"] = previous_mode
            else:
                os.environ.pop("WARA_PHASE25_AUTO_PAPER_MODE", None)
            if previous_legacy_mode is not None:
                os.environ["WCL_PHASE25_AUTO_PAPER_MODE"] = previous_legacy_mode
            else:
                os.environ.pop("WCL_PHASE25_AUTO_PAPER_MODE", None)
            if previous_runs is not None:
                os.environ["WARA_PHASE25_AUTO_PAPER_RUNS"] = previous_runs
            else:
                os.environ.pop("WARA_PHASE25_AUTO_PAPER_RUNS", None)
            if previous_legacy_runs is not None:
                os.environ["WCL_PHASE25_AUTO_PAPER_RUNS"] = previous_legacy_runs
            else:
                os.environ.pop("WCL_PHASE25_AUTO_PAPER_RUNS", None)

    def test_wara_phase1_handoff_normalizes_for_phase2(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "native-tail"
            source.mkdir()
            (source / "phase1_handoff.json").write_text(
                json.dumps(
                    {
                        "selected_candidate": {
                            "title": "Nonlinear SWIPT Utility Maximization",
                            "problem_statement": "Optimize a nonlinear SWIPT utility.",
                            "wireless_scenario": "Single-cell SWIPT downlink.",
                        },
                        "problem_contract_seed": {
                            "objective": "maximize nonlinear rate-energy utility",
                            "constraints": ["power budget", "rate floor", "rectifier operating region"],
                        },
                        "novelty_contract": {
                            "claim_boundary": "Novelty comes from nonlinear EH optimization structure.",
                            "main_risk": "Prior nonlinear SWIPT work may already cover it.",
                        },
                        "proof_contract": {"target_claims": ["monotone SCA convergence"]},
                        "validation_contract": {"figures": ["utility vs EH nonlinearity"]},
                        "kill_criteria": ["Prior art already proves the same claim."],
                    }
                ),
                encoding="utf-8",
            )
            (source / "evidence_pack.json").write_text(
                json.dumps(
                    {
                        "source_phase1_run": "/legacy/phase1",
                        "synthesis_excerpt": "SWIPT nonlinear EH synthesis.",
                        "topic_score": {"overall_score": 8.2},
                    }
                ),
                encoding="utf-8",
            )
            (source / "candidate_review.json").write_text(
                json.dumps({"selection_decision": {"selected_title": "Nonlinear SWIPT Utility Maximization"}}),
                encoding="utf-8",
            )
            (source / "topic_focused_literature.json").write_text(
                json.dumps(
                    {
                        "selected_title": "Nonlinear SWIPT Utility Maximization",
                        "references": [{"key": "FocusedRef", "doi": "10.1109/TWC.2013.031813.120224"}],
                        "search_report": {"reference_count": 1},
                    }
                ),
                encoding="utf-8",
            )
            (source / "topic_focused_references.bib").write_text(
                "@article{FocusedRef,\n"
                "  title = {MIMO Broadcasting for Simultaneous Wireless Information and Power Transfer},\n"
                "  doi = {10.1109/TWC.2013.031813.120224},\n"
                "  year = {2013}\n"
                "}\n",
                encoding="utf-8",
            )

            run_dir = root / "phase2-run"
            old_minimum = os.environ.get("WARA_PHASE1_REFERENCE_MIN")
            os.environ["WARA_PHASE1_REFERENCE_MIN"] = "1"
            try:
                handoff = build_wara_phase1_handoff(source, run_dir)
            finally:
                if old_minimum is None:
                    os.environ.pop("WARA_PHASE1_REFERENCE_MIN", None)
                else:
                    os.environ["WARA_PHASE1_REFERENCE_MIN"] = old_minimum
            self.assertEqual(handoff["source_kind"], "wara_phase1_handoff")
            self.assertEqual(handoff["final_title"], "Nonlinear SWIPT Utility Maximization")
            self.assertIn("nonlinear rate-energy", handoff["objective"])
            self.assertTrue((run_dir / "input_from_phase1" / "phase1_handoff.json").exists())
            self.assertIn(
                "SWIPT nonlinear EH synthesis",
                (run_dir / "input_from_phase1" / "synthesis.md").read_text(encoding="utf-8"),
            )
            self.assertIn(
                "FocusedRef",
                (run_dir / "input_from_phase1" / "topic_focused_references.bib").read_text(encoding="utf-8"),
            )
            self.assertIn(
                "FocusedRef",
                (run_dir / "input_from_phase1" / "references.bib").read_text(encoding="utf-8"),
            )
            self.assertEqual(handoff["topic_focused_reference_count"], 1)

    def test_wara_phase1_handoff_adapter_blocks_thin_reference_bank(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "handoff"
            source.mkdir()
            (source / "phase1_handoff.json").write_text(
                json.dumps(
                    {
                        "selected_candidate": {"title": "Thin Reference Topic"},
                        "problem_contract_seed": {"controls": ["x"]},
                        "novelty_contract": {"claim_boundary": "demo"},
                        "proof_contract": {"target_claims": ["demo"]},
                        "validation_contract": {"metrics": ["U"]},
                        "kill_criteria": ["demo"],
                    }
                ),
                encoding="utf-8",
            )
            (source / "evidence_pack.json").write_text(json.dumps({}), encoding="utf-8")
            (source / "candidate_review.json").write_text(json.dumps({}), encoding="utf-8")
            (source / "topic_focused_literature.json").write_text(
                json.dumps({"references": [{"key": "OnlyOne2025"}]}),
                encoding="utf-8",
            )
            (source / "topic_focused_references.bib").write_text("@article{OnlyOne2025,title={Only One}}\n", encoding="utf-8")
            (source / "topic_focused_literature.md").write_text("Only one reference.\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "1 references < hard target 12"):
                build_wara_phase1_handoff(source, Path(tmp) / "phase2-run")

    def test_wara_phase1_handoff_gate_blocks_missing_contract(self) -> None:
        previous = os.environ.pop("WARA_ALLOW_WEAK_PHASE1", None)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                source = Path(tmp)
                (source / "phase1_handoff.json").write_text(
                    json.dumps({"selected_candidate": {"title": "Weak"}}),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(SystemExit, "handoff gate blocked"):
                    validate_wara_phase1_handoff_gate(source)
        finally:
            if previous is not None:
                os.environ["WARA_ALLOW_WEAK_PHASE1"] = previous

    def test_wara_phase1_handoff_detection_requires_native_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)
            (source / "phase1_handoff.json").write_text(
                json.dumps(
                    {
                        "selected_candidate": {"title": "Native"},
                        "problem_contract_seed": {"objective": "maximize utility"},
                    }
                ),
                encoding="utf-8",
            )
            self.assertTrue(looks_like_phase1_handoff(source))
            self.assertEqual(topic_from_wara_phase1_handoff(source), "Native")

    def test_wireless_benchmark_agent_selects_practical_uplink_baselines(self) -> None:
        problem_contract = build_problem_contract(
            topic="Single-cell uplink power control for minimum sum transmit power under SINR constraints",
            handoff={},
            system_model_md="An uplink system uses SINR targets and interference gains.",
            problem_formulation_md="Minimize sum transmit power subject to SINR constraints.",
            core_theory_package_md="The fixed point uses a spectral radius feasibility condition.",
        )
        algorithm_contract = build_algorithm_contract(
            topic="uplink power control",
            problem_contract=problem_contract,
            convexity_audit_md="The problem is an LP/fixed point under standard interference mapping.",
            reformulation_path_md="Use fixed point and LP reference.",
        )
        claim_map = build_claim_map(
            topic="uplink power control",
            problem_contract=problem_contract,
            algorithm_contract=algorithm_contract,
            algorithm_md="fixed point algorithm",
            convergence_or_complexity_md="feasibility follows spectral radius condition",
            experiment_blueprint_md="compare sum power and feasibility",
        )
        plan = select_wireless_benchmark_plan(
            topic="uplink power control",
            problem_contract=problem_contract,
            algorithm_contract=algorithm_contract,
            claim_map=claim_map,
        )

        method_ids = [method["id"] for method in plan["compared_methods"]]
        self.assertIn("equal_power_heuristic", method_ids)
        self.assertIn("channel_inversion_heuristic", method_ids)
        self.assertIn("lp_reference", method_ids)
        self.assertEqual(plan["main_plot_policy"]["main_figures_use_methods"][0], "proposed")
        self.assertNotIn("lp_reference", plan["main_plot_policy"]["main_figures_use_methods"])
        self.assertNotIn("ris", problem_contract["mechanisms"])

    def test_wireless_benchmark_agent_avoids_ps_when_no_power_splitting_mechanism(self) -> None:
        problem_contract = {
            "problem_family": "downlink_beamforming",
            "objective_sense": "minimize",
            "mechanisms": ["downlink", "beamforming", "energy_harvesting", "sensing"],
            "primary_physical_kpis": ["P_tx_mW", "harvested_energy_mW", "sensing_illumination_mW"],
        }
        algorithm_contract = {"algorithm_family": "direct_covariance_optimization"}
        claim_map = {"claims": [{"claim_id": "C1_main_physical_kpi"}]}

        plan = select_wireless_benchmark_plan(
            topic="cell-free integrated sensing communication and powering",
            problem_contract=problem_contract,
            algorithm_contract=algorithm_contract,
            claim_map=claim_map,
        )

        method_ids = [method["id"] for method in plan["compared_methods"]]
        self.assertIn("no_shared_covariance_baseline", method_ids)
        self.assertIn("isotropic_shared_covariance_baseline", method_ids)
        self.assertNotIn("fixed_ps_baseline", method_ids)
        self.assertNotIn("linear_eh_baseline", method_ids)
        self.assertIn("no_shared_covariance_baseline", plan["main_plot_policy"]["main_figures_use_methods"])

    def test_algorithm_contract_ignores_negated_algorithm_names(self) -> None:
        problem_contract = build_problem_contract(
            topic="Conic Beamforming for Reliable Downlink QoS",
            handoff={},
            system_model_md="A downlink beamforming system uses per-user SINR constraints.",
            problem_formulation_md="Minimize transmit power subject to safe SOCP robust QoS constraints.",
            core_theory_package_md="The safe margin gives a convex second-order cone program.",
        )

        algorithm_contract = build_algorithm_contract(
            topic="Conic Beamforming for Reliable Downlink QoS",
            problem_contract=problem_contract,
            convexity_audit_md=(
                "The proposed safe formulation is a convex SOCP. "
                "No DC programming, SDR, WMMSE, SCA, or BCD is required."
            ),
            reformulation_path_md=(
                "Recommended method: direct conic optimization, not BCD/SCA/MM/WMMSE/SDR. "
                "Build one convex SOCP and solve it with a conic solver."
            ),
        )

        self.assertEqual(algorithm_contract["algorithm_family"], "direct_optimization")

    def test_problem_contract_minimize_is_not_confused_by_pmax(self) -> None:
        problem_contract = build_problem_contract(
            topic="Nonlinear-EH-Aware Monostatic ISACP Covariance Design",
            handoff={},
            system_model_md="A downlink covariance system includes nonlinear harvested-DC constraints and sensing illumination.",
            problem_formulation_md=(
                "The base station minimizes total transmit power subject to SINR, harvested-DC, "
                "and sensing constraints. The optional cap is denoted P_{\\max}."
            ),
            core_theory_package_md="The nonlinear RF-to-DC rectifier is handled by an inverse threshold.",
        )

        self.assertEqual(problem_contract["objective_sense"], "minimize")
        self.assertIn("energy_harvesting", problem_contract["mechanisms"])
        self.assertEqual(problem_contract["primary_physical_kpis"][0], "P_tx_mW")
        self.assertNotIn("constraint_violation_max", problem_contract["primary_physical_kpis"])
        self.assertNotIn("feasible", problem_contract["primary_physical_kpis"])
        self.assertIn("constraint_violation_max", problem_contract["diagnostic_metrics"])

    def test_problem_contract_prefers_outer_max_over_inner_worst_case_min(self) -> None:
        problem_contract = build_problem_contract(
            topic="RIS-aided physical-layer security for multiuser MIMO with imperfect CSI",
            handoff={"final_title": "Worst-Case Robust Max-Min Secrecy Rate Maximization"},
            system_model_md="A downlink RIS-assisted secrecy system has bounded cascaded CSI errors.",
            problem_formulation_md=(
                "The worst-case per-user secrecy rate is "
                "$\\underline{R}_k=\\inf_{\\Delta\\in\\mathcal{U}}R_k(\\Delta)$. "
                "Original Optimization Problem $(\\mathcal{P}_0)$: "
                "$\\max_{\\mathbf{p},\\theta,t}\\ t$ subject to "
                "$\\underline{R}_k\\ge t$ and a sum-power budget."
            ),
            core_theory_package_md="The robust evaluator contains inner minimum operators over CSI errors.",
        )

        self.assertEqual(problem_contract["objective_sense"], "maximize")
        self.assertIn("worst_case_min_secrecy_rate_bpsHz", problem_contract["primary_physical_kpis"])
        self.assertNotEqual(problem_contract["primary_physical_kpis"][0], "P_tx_mW")

    def test_problem_contract_uses_frozen_max_min_rate_objective_for_kpis(self) -> None:
        mathematical_contract = {
            "objective": {
                "sense": "max",
                "expression": "min_{k in [K]} R_k(w,a)",
                "meaning": "Maximize the minimum achievable rate across all users (max-min fairness).",
                "terms": [{"expression": "min_k R_k"}],
            }
        }
        problem_contract = build_problem_contract(
            topic="cell-free massive MIMO",
            handoff={"final_title": "Joint Precoding and Association for Max-Min Fairness in Cell-Free Massive MIMO"},
            system_model_md="A downlink cell-free massive MIMO system with AP-user association and beamforming.",
            problem_formulation_md=(
                "The formulation contains binary AP-user association, beamforming vectors, and the minimum achievable rate. "
                "It is a mixed-integer nonconvex max-min fairness problem."
            ),
            core_theory_package_md="Use target-rate bisection and conic feasibility checks.",
            mathematical_contract_json=json.dumps(mathematical_contract),
        )
        algorithm_contract = build_algorithm_contract(
            topic="cell-free massive MIMO",
            problem_contract=problem_contract,
            convexity_audit_md="The problem is mixed-integer and nonconvex.",
            reformulation_path_md="Use bisection over the target minimum user rate.",
        )
        claim_map = build_claim_map(
            topic="cell-free massive MIMO",
            problem_contract=problem_contract,
            algorithm_contract=algorithm_contract,
            algorithm_md="The proposed method maximizes the minimum user rate via target-rate bisection.",
            convergence_or_complexity_md="Each target-rate feasibility test is solved by a certified conic oracle.",
        )

        self.assertEqual(problem_contract["objective_sense"], "maximize")
        self.assertEqual(problem_contract["primary_physical_kpis"][0], "min_user_rate_bpsHz")
        self.assertNotIn("harvested_energy_mW", problem_contract["primary_physical_kpis"])
        self.assertNotIn("sensing_illumination_mW", problem_contract["primary_physical_kpis"])
        self.assertIn(claim_map["claims"][0]["metric"], {"min_user_rate_bpsHz", "min_spectral_efficiency_bpsHz"})
        self.assertEqual(claim_map["claims"][0]["direction"], "higher_is_better")

    def test_claim_owner_aligns_main_metric_with_harvested_dc_objective(self) -> None:
        problem_contract = build_problem_contract(
            topic="SWIPT-enabled ISAC with nonlinear energy harvesting",
            handoff={},
            system_model_md="The downlink system supports information decoding, sensing, and nonlinear RF-to-DC energy harvesting.",
            problem_formulation_md=(
                "The frozen problem maximizes the worst harvested DC power using an epigraph "
                "variable t subject to SINR, sensing, and total-power constraints."
            ),
            core_theory_package_md="A bisection method solves an SDP feasibility oracle for each harvested-DC epigraph value.",
        )
        algorithm_contract = build_algorithm_contract(
            topic="SWIPT-enabled ISAC with nonlinear energy harvesting",
            problem_contract=problem_contract,
            convexity_audit_md="The fixed epigraph query is an SDP.",
            reformulation_path_md="Use inverse nonlinear-EH thresholds and bisection over the harvested-DC epigraph.",
        )
        claim_map = build_claim_map(
            topic="SWIPT-enabled ISAC with nonlinear energy harvesting",
            problem_contract=problem_contract,
            algorithm_contract=algorithm_contract,
            algorithm_md="The proposed method maximizes the minimum harvested DC power via bisection over t.",
            convergence_or_complexity_md="The SDP oracle is exact for the covariance-domain formulation.",
        )

        self.assertIn(problem_contract["primary_physical_kpis"][0], {"min_harvested_dc_mW", "harvested_energy_mW"})
        self.assertIn(claim_map["claims"][0]["metric"], {"min_harvested_dc_mW", "harvested_energy_mW"})

    def test_problem_contract_aligns_service_balancing_objective_to_eta_metric(self) -> None:
        problem_contract = build_problem_contract(
            topic="Integrated sensing, communication, and powering",
            handoff={},
            system_model_md=(
                "A downlink ISACP system serves communication users, energy receivers, and sensing targets "
                "with covariance-domain beamforming."
            ),
            problem_formulation_md=(
                "The frozen problem maximizes eta, the minimum normalized robust communication, powering, "
                "and sensing service level, subject to transmit-power, SINR, harvested-energy, and sensing constraints."
            ),
            core_theory_package_md="Use bisection over eta and solve each fixed-eta robust conic feasibility problem.",
        )

        self.assertEqual(problem_contract["primary_physical_kpis"][0], "eta_service_level")
        self.assertIn("min_harvested_dc_mW", problem_contract["primary_physical_kpis"])

    def test_experiment_design_corrects_stale_secondary_claim_metric_to_objective_kpi(self) -> None:
        problem_contract = {
            "problem_family": "downlink_beamforming",
            "objective_sense": "maximize",
            "primary_physical_kpis": ["min_harvested_dc_mW", "sum_rate_bpsHz", "harvested_energy_mW"],
        }
        benchmark_plan = {"main_plot_policy": {"main_figures_use_methods": ["proposed", "fixed_baseline"]}}
        stale_claim_map = {
            "source_summary": "The proposed method maximizes the worst harvested DC epigraph under SINR constraints.",
            "claims": [
                {
                    "claim_id": "C1_main_physical_kpi",
                    "metric": "sum_rate_bpsHz",
                    "direction": "higher_is_better",
                }
            ],
        }

        design = build_experiment_design_contract(
            problem_contract=problem_contract,
            benchmark_plan=benchmark_plan,
            claim_map=stale_claim_map,
        )

        self.assertEqual(design["figure_contracts"][0]["y_metric"], "min_harvested_dc_mW")

    def test_algorithm_contract_uses_table_route_for_convex_sdp(self) -> None:
        problem_contract = build_problem_contract(
            topic="Nonlinear-EH-Aware Monostatic ISACP Covariance Design",
            handoff={},
            system_model_md="A downlink covariance system includes nonlinear harvested-DC constraints and sensing illumination.",
            problem_formulation_md="Minimize total transmit power over covariance matrices subject to SINR, harvested-DC, and sensing constraints.",
            core_theory_package_md="The nonlinear rectifier is handled by an inverse threshold and the covariance problem is an SDP.",
        )

        algorithm_contract = build_algorithm_contract(
            topic="Nonlinear-EH-Aware Monostatic ISACP Covariance Design",
            problem_contract=problem_contract,
            convexity_audit_md="| item | decision |\n|---|---|\n| selected_route | `convex_direct` |\nThe covariance formulation is an exact SDP.",
            reformulation_path_md="Compute inverse EH thresholds and solve the covariance SDP directly.",
        )

        update_blocks = algorithm_contract["algorithm_execution_contract"]["update_blocks"]
        self.assertEqual(algorithm_contract["tractability_route"], "convex_direct")
        self.assertEqual(algorithm_contract["algorithm_family"], "direct_optimization")
        self.assertEqual(algorithm_contract["objective_sense"], "minimize")
        self.assertIn("inverse_eh_threshold_update", update_blocks)
        self.assertIn("covariance_sdp_solve", update_blocks)

        bullet_contract = build_algorithm_contract(
            topic="Nonlinear-EH-Aware Monostatic ISACP Covariance Design",
            problem_contract=problem_contract,
            convexity_audit_md="- `selected_route`: `convex_direct`\nThe problem is exactly representable as an SDP.",
            reformulation_path_md="Use inverse EH thresholding and solve the SDP directly.",
        )
        self.assertEqual(bullet_contract["tractability_route"], "convex_direct")
        self.assertEqual(bullet_contract["algorithm_family"], "direct_optimization")

    def test_phase3_contract_accepts_prose_for_execution_blocks(self) -> None:
        run_dir = Path(tempfile.mkdtemp(prefix="phase3-contract-prose-"))
        (run_dir / "phase2-1").mkdir(parents=True, exist_ok=True)
        (run_dir / "phase2-2").mkdir(parents=True, exist_ok=True)
        (run_dir / "phase2-3").mkdir(parents=True, exist_ok=True)
        (run_dir / "phase2-1" / "mathematical_contract.frozen.json").write_text(
            json.dumps(
                {
                    "controls": [
                        {"symbol": "\\mathbf W_k", "meaning": "communication covariance matrix"},
                        {"symbol": "\\mathbf S", "meaning": "non-information covariance matrix"},
                    ],
                    "objective": {"sense": "minimize", "expression": "P_tx", "meaning": "minimize transmit power"},
                    "constraints": [{"id": "C-SINR"}, {"id": "C-EH"}, {"id": "C-SENS"}],
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "phase2-2" / "algorithm_contract.json").write_text(
            json.dumps(
                {
                    "algorithm_execution_contract": {
                        "update_blocks": [
                            "inverse_eh_threshold_update",
                            "covariance_sdp_solve",
                            "feasibility_acceptance_and_kpi_evaluation",
                        ],
                        "objective_evaluator": "Evaluate the frozen physical objective for every method.",
                        "constraint_evaluator": "Evaluate all frozen physical constraints with shared code.",
                    }
                }
            ),
            encoding="utf-8",
        )
        algorithm_md = (
            "The method validates the inverse-EH representation, computes an attainable RF threshold, "
            "and then solves the covariance-domain SDP over the communication covariance matrix and the non-information covariance matrix. "
            "After the conic solver returns, it checks feasibility residuals and evaluates the physical KPIs."
        )

        report = validate_phase2_phase3_contract(
            run_dir=run_dir,
            algorithm_md=algorithm_md,
            convergence_or_complexity_md="The objective evaluator reports transmit power and all frozen constraints are evaluated.",
        )

        self.assertTrue(report["ok"])

    def test_phase3_1_overfull_only_compile_issue_is_nonblocking(self) -> None:
        overfull_only = (
            "- Repair PDF overfull hbox warnings after compilation.\n"
            "- line 59: overfull by 25.76747pt\n"
            "  Source context:\n"
            "  57: P_{\\mathrm{tx}}=\\operatorname{Tr}(\\mathbf Q)."
        )
        equation_issue = "- Repair equation display formatting: independent definitions must not be joined."
        fatal_issue = "! LaTeX Error: Missing $ inserted."

        self.assertFalse(_phase3_1_compile_issues_should_block(overfull_only))
        self.assertFalse(_phase3_1_compile_issues_should_block(equation_issue))
        self.assertTrue(_phase3_1_compile_issues_should_block(fatal_issue))

    def test_tractability_route_policy_branches_convex_and_nonconvex_topics(self) -> None:
        convex_policy = build_tractability_route_policy(
            topic="Conic Beamforming for Reliable Downlink QoS",
            handoff={},
            mathematical_contract_json="{}",
            system_model_md="A downlink beamforming system uses per-user SINR constraints.",
            problem_formulation_md="Minimize transmit power subject to a convex SOCP safe QoS formulation.",
            core_theory_package_md="The safe formulation is solved by direct conic optimization.",
        )
        nonconvex_policy = build_tractability_route_policy(
            topic="Nonlinear EH-aware SWIPT beamforming",
            handoff={},
            mathematical_contract_json="{}",
            system_model_md="Power splitting receivers use a nonlinear EH saturation curve.",
            problem_formulation_md="The rate-energy objective has SINR coupling and nonlinear EH.",
            core_theory_package_md="Use SCA/MM with a scoped surrogate and true physical evaluation.",
        )

        self.assertEqual(convex_policy["selected_route"], "convex_direct")
        self.assertEqual(nonconvex_policy["selected_route"], "structured_nonconvex")
        self.assertIn("structured_nonconvex", nonconvex_policy["allowed_routes"])

        movable_policy = build_tractability_route_policy(
            topic="Movable antennas for downlink WSR",
            handoff={},
            mathematical_contract_json="{}",
            system_model_md="Movable antenna coordinates and precoders are optimized under aperture and spacing constraints.",
            problem_formulation_md="The WSR problem has SINR coupling and exponential coordinate-dependent channels.",
            core_theory_package_md="Use WMMSE and SCA without SDR rank recovery.",
        )
        self.assertNotEqual(movable_policy["selected_route"], "relaxation_recovery")
        self.assertIn(movable_policy["selected_route"], {"mixed_discrete_or_manifold", "structured_nonconvex"})

    def test_validation_plan_normalizer_avoids_threshold_self_metric_as_main_figure(self) -> None:
        raw_plan = {
            "problem_family": "downlink_beamforming",
            "objective_sense": "maximize",
            "research_evidence_contract": {
                "compared_methods": [
                    {"id": "proposed", "role": "proposed", "mandatory_status": "mandatory"},
                    {"id": "fixed_ps_baseline", "role": "main_baseline", "mandatory_status": "mandatory"},
                ],
                "required_result_columns": [
                    "method",
                    "seed",
                    "swept_param",
                    "swept_value",
                    "objective",
                    "sum_rate_bpsHz",
                    "harvested_energy_mW",
                    "constraint_violation_max",
                ],
                "figures": [
                    {
                        "id": "figure_1",
                        "claim": "main performance comparison",
                        "chart_intent": "main_comparison",
                        "y_metric": "sum_rate_bpsHz",
                        "required_sweep": "sw_Pmax_mW",
                        "methods_to_run": ["proposed", "fixed_ps_baseline"],
                    },
                    {
                        "id": "figure_2",
                        "claim": "EH threshold stress should expose nonlinear mechanism",
                        "chart_intent": "eh_stress_mechanism",
                        "y_metric": "harvested_energy_mW",
                        "required_sweep": "sw_Qmin_mW",
                        "methods_to_run": ["proposed", "fixed_ps_baseline"],
                    },
                    {
                        "id": "figure_3",
                        "claim": "rectifier sensitivity mechanism",
                        "chart_intent": "rectifier_robustness_diagnostic",
                        "y_metric": "constraint_violation_max",
                        "required_sweep": "sw_rectifier_b",
                        "methods_to_run": ["proposed", "fixed_ps_baseline"],
                    },
                ],
            },
            "canonical_config": {
                "constraints": {"P_max_mW": 100.0},
                "requirements": {"Q_min_mW": 0.02},
                "rectifier": {"steepness_b": 4.0},
            },
            "sweep_definitions": [
                {
                    "id": "sw_Pmax_mW",
                    "variable": "$P_{\\max}$",
                    "canonical_path": "constraints.P_max_mW",
                    "quick_values": [50, 100, 200],
                    "paper_values": [40, 60, 80, 100],
                },
                {
                    "id": "sw_Qmin_mW",
                    "variable": "$Q_{\\min}$",
                    "canonical_path": "requirements.Q_min_mW",
                    "quick_values": [0.01, 0.03, 0.06],
                    "paper_values": [0.01, 0.03, 0.06, 0.1],
                },
                {
                    "id": "sw_rectifier_b",
                    "variable": "$b$",
                    "canonical_path": "rectifier.steepness_b",
                    "quick_values": [2, 4, 8],
                    "paper_values": [2, 4, 8, 12],
                },
            ],
            "required_outputs": {
                "scalar_metrics": [
                    "objective",
                    "sum_rate_bpsHz",
                    "harvested_energy_mW",
                    "constraint_violation_max",
                ]
            },
        }

        normalized = yaml.safe_load(normalize_phase24_validation_plan_yaml(yaml.safe_dump(raw_plan, sort_keys=False)))
        figures = normalized["research_evidence_contract"]["figures"]

        self.assertEqual(len(figures), 2)
        self.assertEqual(figures[0]["required_sweep"], "sw_Pmax_mW")
        self.assertEqual(figures[1]["required_sweep"], "sw_rectifier_b")
        self.assertNotEqual(figures[1]["y_metric"], "constraint_violation_max")
        self.assertNotEqual(figures[1]["y_metric"], "harvested_energy_mW")

    def test_validation_plan_normalizer_avoids_saturating_raw_metric_for_mechanism_sweep(self) -> None:
        raw_plan = {
            "problem_family": "downlink_beamforming",
            "objective_sense": "maximize",
            "research_evidence_contract": {
                "compared_methods": [
                    {"id": "proposed", "role": "proposed", "mandatory_status": "mandatory"},
                    {"id": "linear_eh_baseline", "role": "model_diagnostic", "mandatory_status": "mandatory"},
                ],
                "required_result_columns": [
                    "method",
                    "seed",
                    "swept_param",
                    "swept_value",
                    "objective",
                    "sum_rate_bpsHz",
                    "harvested_energy_mW",
                ],
                "figures": [
                    {
                        "id": "figure_1",
                        "claim": "main physical KPI comparison",
                        "chart_intent": "main_comparison",
                        "y_metric": "sum_rate_bpsHz",
                        "required_sweep": "sw_Pmax_mW",
                        "methods_to_run": ["proposed", "linear_eh_baseline"],
                    },
                    {
                        "id": "figure_2",
                        "claim": "nonlinear rectifier steepness should expose mechanism sensitivity",
                        "chart_intent": "mechanism_sensitivity",
                        "y_metric": "harvested_energy_mW",
                        "required_sweep": "sw_rectifier_b",
                        "methods_to_run": ["proposed", "linear_eh_baseline"],
                    },
                ],
            },
            "canonical_config": {
                "constraints": {"P_max_mW": 100.0},
                "rectifier": {"steepness_b": 4.0},
            },
            "sweep_definitions": [
                {
                    "id": "sw_Pmax_mW",
                    "variable": "$P_{\\max}$",
                    "canonical_path": "constraints.P_max_mW",
                    "quick_values": [50, 100, 200],
                    "paper_values": [40, 60, 80, 100],
                },
                {
                    "id": "sw_rectifier_b",
                    "variable": "$b$",
                    "canonical_path": "rectifier.steepness_b",
                    "quick_values": [2, 4, 8],
                    "paper_values": [2, 4, 8, 12],
                },
            ],
            "required_outputs": {
                "scalar_metrics": [
                    "objective",
                    "sum_rate_bpsHz",
                    "harvested_energy_mW",
                ]
            },
        }

        normalized = yaml.safe_load(normalize_phase24_validation_plan_yaml(yaml.safe_dump(raw_plan, sort_keys=False)))
        figures = normalized["research_evidence_contract"]["figures"]

        self.assertEqual(figures[1]["required_sweep"], "sw_rectifier_b")
        self.assertIn(figures[1]["y_metric"], {"objective", "sum_rate_bpsHz"})
        self.assertNotEqual(figures[1]["y_metric"], "harvested_energy_mW")

    def test_math_contract_validator_does_not_split_latex_subscript_commas(self) -> None:
        report = validate_phase2_phase1_mathematical_contract_schema(
            {
                "controls": [
                    {"symbol": "a_{k,n}", "meaning": "slot scheduling", "appears_in_optimizer": True},
                    {"symbol": "\\mathbf q_n", "meaning": "UAV position", "appears_in_optimizer": True},
                    {"symbol": "p_n", "meaning": "slot power", "appears_in_optimizer": True},
                ],
                "parameters": [{"symbol": "P_0,P_i", "meaning": "propulsion constants"}],
                "derived_quantities": [
                    {
                        "symbol": "d_{k,n}",
                        "meaning": "UAV-user distance",
                        "definition": "d_{k,n}=sqrt(H^2+||q_n-w_k||^2)",
                    }
                ],
                "random_quantities": [],
                "objective": {"sense": "max", "expression": "energy_efficiency_bit_per_J"},
                "constraints": [{"name": "speed", "expression": "||q_{n+1}-q_n|| <= Vmax dt"}],
                "reformulation_only": [],
            }
        )
        self.assertTrue(report["ok"], report["errors"])

    def test_phase25_gate_blocks_later_paper_phases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            docs_dir = Path(tmp) / "docs"
            summary = Phase2RunSummary(
                run_id="phase2-test",
                topic="demo",
                created_at="2026-04-30T00:00:00+00:00",
                root=str(run_dir),
                phase1_run=None,
                model_profile="test",
                phases=make_phase2_phase_flow(),
            )
            state = Phase2RunState(run_dir, summary)
            state.persist()
            later_phase_called = {"value": False}

            def phase25(_: Path) -> dict:
                return {"phase25_status": "needs_more_phase24_runs"}

            def fail_later(_: Path) -> dict:
                later_phase_called["value"] = True
                raise AssertionError("later paper phases must not run when Phase25 is blocked")

            callbacks = Phase2FlowCallbacks(
                build_pipeline_experiment_design_notes=lambda: "notes",
                build_phase1_handoff=lambda _phase1, _run: {},
                build_phase3_design_notes=lambda: "phase3 notes",
                build_phase24_design_notes=lambda: "phase24 notes",
                extract_latex_issue_summary=lambda _log, _tex: "",
                render_phase1_ieee_preview_pdf=lambda _phase: {"preview_pdf": "phase1.pdf"},
                render_phase3_ieee_preview_pdf=lambda _phase: {"preview_pdf": "phase3.pdf"},
                render_phase3_1_technical_preview_pdf=lambda _phase: {"preview_pdf": "phase3_1.pdf"},
                repair_phase2_phase1_latex_llm=lambda **_: "",
                repair_phase2_phase3_latex_llm=lambda **_: "",
                repair_phase3_1_latex_llm=lambda **_: {
                    "system_model_problem_formulation_tex": "system latex",
                    "proposed_solution_tex": "proposed latex",
                    "proposed_section_title": "Proposed Method",
                },
                repair_phase2_phase24_plugin_llm=lambda **_: "",
                run_phase3_6_apply_review_fixes_package=fail_later,
                run_phase2_phase1_latex_llm=lambda **_: "latex",
                run_phase2_phase1_llm=lambda **_: {
                    "system_model_md": "system",
                    "problem_formulation_md": "problem",
                    "core_theory_package_md": "theory",
                },
                run_phase2_phase2_llm=lambda **_: {
                    "convexity_audit_md": "audit",
                    "reformulation_path_md": "path",
                },
                run_phase2_phase3_latex_llm=lambda **_: "algorithm latex",
                run_phase2_phase3_llm=lambda **_: {
                    "algorithm_md": "algorithm",
                    "convergence_or_complexity_md": "complexity",
                    "benchmark_definition_md": "benchmark",
                    "validation_principles_md": "validation",
                    "experiment_blueprint_md": "blueprint",
                },
                run_phase3_1_writing_llm=lambda **_: {
                    "system_model_problem_formulation_tex": "system latex",
                    "proposed_solution_tex": "proposed latex",
                    "proposed_section_title": "Proposed Method",
                },
                run_phase2_phase24_benchmark_llm=lambda **_: {
                    "benchmark_plan_md": "bench",
                    "solver_readme_md": "readme",
                },
                run_phase2_phase24_plugin_llm=lambda **_: "def build_model(config=None):\n    return {}\n",
                run_phase2_phase24_validation_llm=lambda **_: "cases: []\n",
                run_phase24_paper_sweep_from_plan=lambda _run, _quick=False: {},
                run_phase25_wcl_package=phase25,
                run_phase3_2_numerical_results_package=fail_later,
                run_phase3_3_technical_sections_package=fail_later,
                run_phase3_4_introduction_references_package=fail_later,
                run_phase3_5_paper_review_package=fail_later,
                phase24_validation_allows_repair=lambda _status: False,
                phase24_validation_error_text=lambda _run, _status: "",
                validate_phase24_evidence_contract_design=lambda _run: {"ok": True, "errors": [], "warnings": []},
                validate_phase2_phase24_plugin_bundle=lambda _run: {"status": "ok"},
                write_phase2_phase24_fixed_harness=lambda _run: None,
            )

            execute_phase2_flow(
                run_dir=run_dir,
                state=state,
                topic="demo",
                model_profile="test",
                phase1_run=None,
                docs_dir=docs_dir,
                callbacks=callbacks,
                stop_after_phase="5",
            )

            payload = json.loads((run_dir / "phase2_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["phases"][4]["status"], "blocked")
            self.assertFalse(later_phase_called["value"])
            self.assertTrue((run_dir / "phase2-1" / "problem_contract.json").exists())
            self.assertTrue((run_dir / "phase2-2" / "tractability_route_policy.json").exists())
            self.assertTrue((run_dir / "phase2-2" / "algorithm_contract.json").exists())
            self.assertTrue((run_dir / "phase2-3" / "claim_map.json").exists())
            self.assertFalse((run_dir / "phase3-1" / "phase3_1_preview_manifest.json").exists())
            self.assertTrue((run_dir / "phase2-4" / "wireless_benchmark_plan.json").exists())
            self.assertTrue((run_dir / "phase2-4" / "experiment_design_contract.json").exists())
            self.assertTrue((run_dir / "phase2-4" / "implementation_audit.json").exists())
            self.assertTrue((run_dir / "phase2-5" / "evidence_audit.json").exists())
            self.assertTrue((run_dir / "agent-sync" / "formulation_agent_phase2_formulation_complete.json").exists())
            self.assertTrue((run_dir / "agent-sync" / "theory_agent_phase2_reformulation_complete.json").exists())
            self.assertTrue((run_dir / "agent-sync" / "theory_agent_phase2_algorithm_complete.json").exists())
            self.assertTrue((run_dir / "agent-requests" / "formulation_agent_request.json").exists())
            self.assertTrue((run_dir / "agent-requests" / "theory_agent_request.json").exists())
            workspace = json.loads((run_dir / "agent_workspace_manifest.json").read_text(encoding="utf-8"))
            synced_agents = [item["agent_id"] for item in workspace["runs"]]
            self.assertIn("formulation_agent", synced_agents)
            self.assertIn("theory_agent", synced_agents)
            controller_manifest = json.loads((run_dir / "phase2_controller_manifest.json").read_text(encoding="utf-8"))
            controller_index = json.loads((run_dir / "controller_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(controller_manifest["phase"], "phase2")
            self.assertEqual(controller_index["phase"], "phase2_phase3_split")
            self.assertIn("phase2_controller_manifest", controller_index)
            self.assertIn("phase3_controller_manifest", controller_index)
            self.assertIn("phase_architecture", controller_manifest)
            self.assertIn("formulation_agent", controller_manifest["agents"])
            self.assertIn("theory_agent", controller_manifest["agents"])
            self.assertIn("experiment_agent", controller_manifest["agents"])
            self.assertNotIn("writing_agent", controller_manifest["agents"])
            self.assertTrue(controller_manifest["artifacts"]["mathematical_contract"]["frozen"])
            self.assertTrue(controller_manifest["artifacts"]["algorithm_contract"]["frozen"])
            self.assertTrue(controller_manifest["artifacts"]["validation_plan"]["frozen"])
            gate_ids = [item["gate_id"] for item in controller_manifest["gates"]]
            self.assertIn("formulation_gate", gate_ids)
            self.assertIn("algorithm_contract_gate", gate_ids)
            self.assertIn("theory_gate", gate_ids)
            self.assertIn("phase24_experiment_design_gate", gate_ids)
            self.assertIn("implementation_gate", gate_ids)
            self.assertIn("phase25_evidence_gate", gate_ids)

    def test_phase25_auto_paper_sweep_runs_before_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            docs_dir = Path(tmp) / "docs"
            summary = Phase2RunSummary(
                run_id="phase2-test",
                topic="demo",
                created_at="2026-04-30T00:00:00+00:00",
                root=str(run_dir),
                phase1_run=None,
                model_profile="test",
                phases=make_phase2_phase_flow(),
            )
            state = Phase2RunState(run_dir, summary)
            state.persist()
            calls = {"phase25": 0, "paper_sweep": 0, "phase3_2": 0}

            def phase25(path: Path) -> dict:
                calls["phase25"] += 1
                phase25_dir = path / "phase2-5"
                phase25_dir.mkdir(parents=True, exist_ok=True)
                if calls["phase25"] == 1:
                    (phase25_dir / "paper_sweep_plan.json").write_text(
                        json.dumps({"figures": [{"figure_id": "figure_1", "required_sweep_param": "P_max", "suggested_values": [1.0, 2.0]}]}),
                        encoding="utf-8",
                    )
                    (phase25_dir / "phase25_experiment_summary.json").write_text(
                        json.dumps({"phase25_status": "needs_more_phase24_runs", "figures": []}),
                        encoding="utf-8",
                    )
                    return {"phase25_status": "needs_more_phase24_runs", "figures": []}
                ready = {
                    "phase25_status": "paper_minimum_ready",
                    "figures": [
                        {"figure_id": "figure_1", "paper_ready": True, "y_metric": "U"},
                        {"figure_id": "figure_2", "paper_ready": True, "y_metric": "U"},
                    ],
                }
                (phase25_dir / "phase25_experiment_summary.json").write_text(json.dumps(ready), encoding="utf-8")
                return ready

            def paper_sweep(path: Path, quick: bool = False) -> dict:
                calls["paper_sweep"] += 1
                self.assertTrue(quick)
                outputs_dir = path / "phase2-4" / "solver" / "outputs"
                outputs_dir.mkdir(parents=True, exist_ok=True)
                (outputs_dir / "scout_validation_results.csv").write_text("case_id,method,U\nc1,proposed,1.0\n", encoding="utf-8")
                (outputs_dir / "scout_validation_summary.json").write_text("{}", encoding="utf-8")
                (outputs_dir / "scout_validation_cases.json").write_text("[]", encoding="utf-8")
                return {"validation_output_prefix": "scout_validation", "num_results": 1}

            def fail_phase3_2(_: Path) -> dict:
                calls["phase3_2"] += 1
                raise AssertionError("Phase31 must not run when --stop-phase 2.5 is requested")

            callbacks = Phase2FlowCallbacks(
                build_pipeline_experiment_design_notes=lambda: "notes",
                build_phase1_handoff=lambda _phase1, _run: {},
                build_phase3_design_notes=lambda: "phase3 notes",
                build_phase24_design_notes=lambda: "phase24 notes",
                extract_latex_issue_summary=lambda _log, _tex: "",
                render_phase1_ieee_preview_pdf=lambda _phase: {"preview_pdf": "phase1.pdf"},
                render_phase3_ieee_preview_pdf=lambda _phase: {"preview_pdf": "phase3.pdf"},
                render_phase3_1_technical_preview_pdf=lambda _phase: {"preview_pdf": "phase3_1.pdf"},
                repair_phase2_phase1_latex_llm=lambda **_: "",
                repair_phase2_phase3_latex_llm=lambda **_: "",
                repair_phase3_1_latex_llm=lambda **_: {
                    "system_model_problem_formulation_tex": "system latex",
                    "proposed_solution_tex": "proposed latex",
                    "proposed_section_title": "Proposed Method",
                },
                repair_phase2_phase24_plugin_llm=lambda **_: "",
                run_phase3_6_apply_review_fixes_package=fail_phase3_2,
                run_phase2_phase1_latex_llm=lambda **_: "latex",
                run_phase2_phase1_llm=lambda **_: {
                    "system_model_md": "system",
                    "problem_formulation_md": "problem",
                    "core_theory_package_md": "theory",
                },
                run_phase2_phase2_llm=lambda **_: {
                    "convexity_audit_md": "audit",
                    "reformulation_path_md": "path",
                },
                run_phase2_phase3_latex_llm=lambda **_: "algorithm latex",
                run_phase2_phase3_llm=lambda **_: {
                    "algorithm_md": "algorithm",
                    "convergence_or_complexity_md": "complexity",
                    "benchmark_definition_md": "benchmark",
                    "validation_principles_md": "validation",
                    "experiment_blueprint_md": "blueprint",
                },
                run_phase3_1_writing_llm=lambda **_: {
                    "system_model_problem_formulation_tex": "system latex",
                    "proposed_solution_tex": "proposed latex",
                    "proposed_section_title": "Proposed Method",
                },
                run_phase2_phase24_benchmark_llm=lambda **_: {
                    "benchmark_plan_md": "bench",
                    "solver_readme_md": "readme",
                },
                run_phase2_phase24_plugin_llm=lambda **_: "def build_model(config=None):\n    return {}\n",
                run_phase2_phase24_validation_llm=lambda **_: "cases: []\n",
                run_phase24_paper_sweep_from_plan=paper_sweep,
                run_phase25_wcl_package=phase25,
                run_phase3_2_numerical_results_package=fail_phase3_2,
                run_phase3_3_technical_sections_package=fail_phase3_2,
                run_phase3_4_introduction_references_package=fail_phase3_2,
                run_phase3_5_paper_review_package=fail_phase3_2,
                phase24_validation_allows_repair=lambda _status: False,
                phase24_validation_error_text=lambda _run, _status: "",
                validate_phase24_evidence_contract_design=lambda _run: {"ok": True, "errors": [], "warnings": []},
                validate_phase2_phase24_plugin_bundle=lambda _run: {"status": "ok"},
                write_phase2_phase24_fixed_harness=lambda _run: None,
            )

            execute_phase2_flow(
                run_dir=run_dir,
                state=state,
                topic="demo",
                model_profile="test",
                phase1_run=None,
                docs_dir=docs_dir,
                callbacks=callbacks,
                stop_after_phase="2.5",
            )

            payload = json.loads((run_dir / "phase2_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["phases"][4]["status"], "done")
            self.assertEqual(calls["phase25"], 2)
            self.assertEqual(calls["paper_sweep"], 1)
            self.assertEqual(calls["phase3_2"], 0)
            auto_manifest = json.loads((run_dir / "phase2-5" / "phase25_auto_expansion_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(auto_manifest["final_phase25_status"], "paper_minimum_ready")
            self.assertEqual(auto_manifest["rounds"][0]["mode"], "scout")
            self.assertEqual(auto_manifest["rounds"][0]["sweep_result"]["validation_output_prefix"], "scout_validation")

    def test_stop_after_phase25_stops_after_ready_evidence_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            docs_dir = Path(tmp) / "docs"
            summary = Phase2RunSummary(
                run_id="phase2-test",
                topic="demo",
                created_at="2026-04-30T00:00:00+00:00",
                root=str(run_dir),
                phase1_run=None,
                model_profile="test",
                phases=make_phase2_phase_flow(),
            )
            state = Phase2RunState(run_dir, summary)
            state.persist()
            phase3_2_called = {"value": False}

            def phase25(path: Path) -> dict:
                phase25_dir = path / "phase2-5"
                phase25_dir.mkdir(parents=True, exist_ok=True)
                (phase25_dir / "phase25_experiment_summary.json").write_text(
                    json.dumps({
                        "phase25_status": "paper_minimum_ready",
                        "figures": [
                            {"figure_id": "figure_1", "paper_ready": True, "y_metric": "sum_rate_bpsHz"},
                            {"figure_id": "figure_2", "paper_ready": True, "y_metric": "P_tx_mW"},
                        ],
                    }),
                    encoding="utf-8",
                )
                return {
                    "phase25_status": "paper_minimum_ready",
                    "figures": [
                        {"figure_id": "figure_1", "paper_ready": True, "y_metric": "sum_rate_bpsHz"},
                        {"figure_id": "figure_2", "paper_ready": True, "y_metric": "P_tx_mW"},
                    ],
                }

            def fail_phase3_2(_: Path) -> dict:
                phase3_2_called["value"] = True
                raise AssertionError("Phase31 must not run when --stop-phase 2.5 is requested")

            callbacks = Phase2FlowCallbacks(
                build_pipeline_experiment_design_notes=lambda: "notes",
                build_phase1_handoff=lambda _phase1, _run: {},
                build_phase3_design_notes=lambda: "phase3 notes",
                build_phase24_design_notes=lambda: "phase24 notes",
                extract_latex_issue_summary=lambda _log, _tex: "",
                render_phase1_ieee_preview_pdf=lambda _phase: {"preview_pdf": "phase1.pdf"},
                render_phase3_ieee_preview_pdf=lambda _phase: {"preview_pdf": "phase3.pdf"},
                render_phase3_1_technical_preview_pdf=lambda _phase: {"preview_pdf": "phase3_1.pdf"},
                repair_phase2_phase1_latex_llm=lambda **_: "",
                repair_phase2_phase3_latex_llm=lambda **_: "",
                repair_phase3_1_latex_llm=lambda **_: {
                    "system_model_problem_formulation_tex": "system latex",
                    "proposed_solution_tex": "proposed latex",
                    "proposed_section_title": "Proposed Method",
                },
                repair_phase2_phase24_plugin_llm=lambda **_: "",
                run_phase3_6_apply_review_fixes_package=fail_phase3_2,
                run_phase2_phase1_latex_llm=lambda **_: "latex",
                run_phase2_phase1_llm=lambda **_: {
                    "system_model_md": "system",
                    "problem_formulation_md": "problem",
                    "core_theory_package_md": "theory",
                },
                run_phase2_phase2_llm=lambda **_: {
                    "convexity_audit_md": "audit",
                    "reformulation_path_md": "path",
                },
                run_phase2_phase3_latex_llm=lambda **_: "algorithm latex",
                run_phase2_phase3_llm=lambda **_: {
                    "algorithm_md": "algorithm",
                    "convergence_or_complexity_md": "complexity",
                    "benchmark_definition_md": "benchmark",
                    "validation_principles_md": "validation",
                    "experiment_blueprint_md": "blueprint",
                },
                run_phase3_1_writing_llm=lambda **_: {
                    "system_model_problem_formulation_tex": "system latex",
                    "proposed_solution_tex": "proposed latex",
                    "proposed_section_title": "Proposed Method",
                },
                run_phase2_phase24_benchmark_llm=lambda **_: {
                    "benchmark_plan_md": "bench",
                    "solver_readme_md": "readme",
                },
                run_phase2_phase24_plugin_llm=lambda **_: "def build_model(config=None):\n    return {}\n",
                run_phase2_phase24_validation_llm=lambda **_: "cases: []\n",
                run_phase24_paper_sweep_from_plan=lambda _run, _quick=False: {},
                run_phase25_wcl_package=phase25,
                run_phase3_2_numerical_results_package=fail_phase3_2,
                run_phase3_3_technical_sections_package=fail_phase3_2,
                run_phase3_4_introduction_references_package=fail_phase3_2,
                run_phase3_5_paper_review_package=fail_phase3_2,
                phase24_validation_allows_repair=lambda _status: False,
                phase24_validation_error_text=lambda _run, _status: "",
                validate_phase24_evidence_contract_design=lambda _run: {"ok": True, "errors": [], "warnings": []},
                validate_phase2_phase24_plugin_bundle=lambda _run: {"status": "ok"},
                write_phase2_phase24_fixed_harness=lambda _run: None,
            )

            execute_phase2_flow(
                run_dir=run_dir,
                state=state,
                topic="demo",
                model_profile="test",
                phase1_run=None,
                docs_dir=docs_dir,
                callbacks=callbacks,
                stop_after_phase="2.5",
            )

            payload = json.loads((run_dir / "phase2_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["phases"][4]["status"], "done")
            self.assertEqual(payload["phases"][5]["status"], "ready")
            self.assertFalse(phase3_2_called["value"])
            manifest = json.loads((run_dir / "phase2_controller_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["gates"][-1]["gate_id"], "phase25_evidence_gate")
            self.assertTrue(manifest["gates"][-1]["ok"])
            self.assertTrue((run_dir / "phase2-5" / "phase2_to_phase3_handoff.json").exists())
            self.assertFalse((run_dir / "phase3_controller_manifest.json").exists())

    def test_phase3_5_review_routing_blocks_owner_phase_instead_of_phase3_6(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            docs_dir = Path(tmp) / "docs"
            summary = Phase2RunSummary(
                run_id="phase2-test",
                topic="demo",
                created_at="2026-04-30T00:00:00+00:00",
                root=str(run_dir),
                phase1_run=None,
                model_profile="test",
                phases=make_phase2_phase_flow(),
            )
            state = Phase2RunState(run_dir, summary)
            state.persist()
            phase3_6_called = {"value": False}

            def phase25(path: Path) -> dict:
                phase25_dir = path / "phase2-5"
                phase25_dir.mkdir(parents=True, exist_ok=True)
                (phase25_dir / "phase25_experiment_summary.json").write_text(
                    json.dumps({
                        "phase25_status": "paper_minimum_ready",
                        "figures": [
                            {"figure_id": "figure_1", "paper_ready": True, "y_metric": "sum_rate_bpsHz"},
                            {"figure_id": "figure_2", "paper_ready": True, "y_metric": "P_tx_mW"},
                        ],
                    }),
                    encoding="utf-8",
                )
                return {
                    "phase25_status": "paper_minimum_ready",
                    "figures": [
                        {"figure_id": "figure_1", "paper_ready": True, "y_metric": "sum_rate_bpsHz"},
                        {"figure_id": "figure_2", "paper_ready": True, "y_metric": "P_tx_mW"},
                    ],
                }

            def phase3_2(path: Path) -> dict:
                phase3_2_dir = path / "phase3-2"
                phase3_2_dir.mkdir(parents=True, exist_ok=True)
                (phase3_2_dir / "numerical_results_section.tex").write_text("Results.", encoding="utf-8")
                (phase3_2_dir / "phase3_2_manifest.json").write_text("{}", encoding="utf-8")
                return {"status": "ok"}

            def phase3_3(path: Path) -> dict:
                phase3_3_dir = path / "phase3-3"
                phase3_3_dir.mkdir(parents=True, exist_ok=True)
                (phase3_3_dir / "phase3_3_technical_sections_preview.tex").write_text("Technical.", encoding="utf-8")
                (phase3_3_dir / "phase3_3_manifest.json").write_text("{}", encoding="utf-8")
                return {"status": "ok"}

            def phase3_4(path: Path) -> dict:
                phase3_4_dir = path / "phase3-4"
                phase3_4_dir.mkdir(parents=True, exist_ok=True)
                (phase3_4_dir / "verified_reference_bank.json").write_text("[]", encoding="utf-8")
                (phase3_4_dir / "citation_claim_map.json").write_text("[]", encoding="utf-8")
                (phase3_4_dir / "full_paper_preview.pdf").write_text("pdf", encoding="utf-8")
                return {"status": "ok"}

            def phase3_5(path: Path) -> dict:
                phase3_5_dir = path / "phase3-5"
                phase3_5_dir.mkdir(parents=True, exist_ok=True)
                routing = {
                    "gate_id": "review_gate",
                    "status": "repair_required",
                    "next_agent": "repair_agent",
                    "target_agent": "experiment_agent",
                    "primary_issue_id": "P0-EXP-01",
                    "primary_reason": "experiment_evidence_or_figure",
                    "routes": [
                        {
                            "issue_id": "P0-EXP-01",
                            "title": "Numerical evidence does not support the claim",
                            "target_agent": "experiment_agent",
                        }
                    ],
                }
                (phase3_5_dir / "review_routing_decision.json").write_text(
                    json.dumps(routing),
                    encoding="utf-8",
                )
                (phase3_5_dir / "phase3_5_review.json").write_text(
                    json.dumps({"routing_decision": routing}),
                    encoding="utf-8",
                )
                (phase3_5_dir / "final_review_report.md").write_text("Review.", encoding="utf-8")
                return {"review_routing_decision": routing}

            def phase3_6(_: Path) -> dict:
                phase3_6_called["value"] = True
                raise AssertionError("Phase10 must not run for experiment-agent review repair")

            callbacks = Phase2FlowCallbacks(
                build_pipeline_experiment_design_notes=lambda: "notes",
                build_phase1_handoff=lambda _phase1, _run: {},
                build_phase3_design_notes=lambda: "phase3 notes",
                build_phase24_design_notes=lambda: "phase24 notes",
                extract_latex_issue_summary=lambda _log, _tex: "",
                render_phase1_ieee_preview_pdf=lambda _phase: {"preview_pdf": "phase1.pdf"},
                render_phase3_ieee_preview_pdf=lambda _phase: {"preview_pdf": "phase3.pdf"},
                render_phase3_1_technical_preview_pdf=lambda _phase: {"preview_pdf": "phase3_1.pdf"},
                repair_phase2_phase1_latex_llm=lambda **_: "",
                repair_phase2_phase3_latex_llm=lambda **_: "",
                repair_phase3_1_latex_llm=lambda **_: {
                    "system_model_problem_formulation_tex": "system latex",
                    "proposed_solution_tex": "proposed latex",
                    "proposed_section_title": "Proposed Method",
                },
                repair_phase2_phase24_plugin_llm=lambda **_: "",
                run_phase3_6_apply_review_fixes_package=phase3_6,
                run_phase2_phase1_latex_llm=lambda **_: "latex",
                run_phase2_phase1_llm=lambda **_: {
                    "system_model_md": "system",
                    "problem_formulation_md": "problem",
                    "core_theory_package_md": "theory",
                },
                run_phase2_phase2_llm=lambda **_: {
                    "convexity_audit_md": "audit",
                    "reformulation_path_md": "path",
                },
                run_phase2_phase3_latex_llm=lambda **_: "algorithm latex",
                run_phase2_phase3_llm=lambda **_: {
                    "algorithm_md": "algorithm",
                    "convergence_or_complexity_md": "complexity",
                    "benchmark_definition_md": "benchmark",
                    "validation_principles_md": "validation",
                    "experiment_blueprint_md": "blueprint",
                },
                run_phase3_1_writing_llm=lambda **_: {
                    "system_model_problem_formulation_tex": "system latex",
                    "proposed_solution_tex": "proposed latex",
                    "proposed_section_title": "Proposed Method",
                },
                run_phase2_phase24_benchmark_llm=lambda **_: {
                    "benchmark_plan_md": "bench",
                    "solver_readme_md": "readme",
                },
                run_phase2_phase24_plugin_llm=lambda **_: "def build_model(config=None):\n    return {}\n",
                run_phase2_phase24_validation_llm=lambda **_: "cases: []\n",
                run_phase24_paper_sweep_from_plan=lambda _run, _quick=False: {},
                run_phase25_wcl_package=phase25,
                run_phase3_2_numerical_results_package=phase3_2,
                run_phase3_3_technical_sections_package=phase3_3,
                run_phase3_4_introduction_references_package=phase3_4,
                run_phase3_5_paper_review_package=phase3_5,
                phase24_validation_allows_repair=lambda _status: False,
                phase24_validation_error_text=lambda _run, _status: "",
                validate_phase24_evidence_contract_design=lambda _run: {"ok": True, "errors": [], "warnings": []},
                validate_phase2_phase24_plugin_bundle=lambda _run: {"status": "ok"},
                write_phase2_phase24_fixed_harness=lambda _run: None,
            )

            execute_phase2_flow(
                run_dir=run_dir,
                state=state,
                topic="demo",
                model_profile="test",
                phase1_run=None,
                docs_dir=docs_dir,
                callbacks=callbacks,
            )

            phase2_payload = json.loads((run_dir / "phase2_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(phase2_payload["phases"][4]["status"], "done")
            self.assertEqual(phase2_payload["phases"][5]["status"], "ready")
            self.assertFalse((run_dir / "phase3_controller_manifest.json").exists())

            execute_phase3_flow(
                run_dir=run_dir,
                state=state,
                topic="demo",
                model_profile="test",
                phase1_run=None,
                callbacks=callbacks,
                phase1_handoff=None,
            )

            payload = json.loads((run_dir / "phase2_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["phases"][3]["status"], "blocked")
            self.assertEqual(payload["phases"][9]["status"], "done")
            self.assertFalse(phase3_6_called["value"])
            decision = json.loads((run_dir / "phase3-5" / "controller_review_decision.json").read_text(encoding="utf-8"))
            self.assertEqual(decision["owner_agent"], "experiment_agent")
            self.assertEqual(decision["rerun_phase"], "phase2.4")
            phase3_manifest = json.loads((run_dir / "phase3_controller_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(phase3_manifest["phase"], "phase3")
            self.assertIn("analysis_agent", phase3_manifest["agents"])
            self.assertIn("writing_agent", phase3_manifest["agents"])
            self.assertNotIn("experiment_agent", phase3_manifest["agents"])
            self.assertIn("phase2_to_phase3_handoff", phase3_manifest["artifacts"])
            phase3_gate_ids = [item["gate_id"] for item in phase3_manifest["gates"]]
            self.assertIn("phase2_to_phase3_handoff_gate", phase3_gate_ids)
            self.assertIn("phase3_2_numerical_results_gate", phase3_gate_ids)
            phase2_manifest = json.loads((run_dir / "phase2_controller_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(phase2_manifest["phase"], "phase2")
            self.assertNotIn("phase3_2_numerical_results_gate", [item["gate_id"] for item in phase2_manifest["gates"]])
            self.assertTrue((run_dir / "agent-sync" / "analysis_agent_phase3_numerical_results_complete.json").exists())
            self.assertTrue((run_dir / "agent-sync" / "repair_agent_phase3_repair_routed.json").exists())

    def test_sync_role_agent_records_literature_and_writing_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            for rel in ["phase2-1", "phase2-2", "phase2-5", "phase3-1", "phase3-2", "phase3-3", "phase3-4"]:
                (run_dir / rel).mkdir(parents=True)
            (run_dir / "phase2-1" / "mathematical_contract.json").write_text("{}", encoding="utf-8")
            (run_dir / "phase2-2" / "algorithm_contract.json").write_text("{}", encoding="utf-8")
            (run_dir / "phase2-5" / "phase25_experiment_summary.json").write_text(
                json.dumps({
                    "phase25_status": "paper_minimum_ready",
                    "figures": [
                        {"figure_id": "figure_1", "paper_ready": True, "y_metric": "sum_rate_bpsHz"},
                        {"figure_id": "figure_2", "paper_ready": True, "y_metric": "P_tx_mW"},
                    ],
                }),
                encoding="utf-8",
            )
            (run_dir / "phase3-1" / "system_model_problem_formulation_ieee_wcl.tex").write_text(
                "System.", encoding="utf-8"
            )
            (run_dir / "phase3-2" / "numerical_results_section.tex").write_text("Results.", encoding="utf-8")
            (run_dir / "phase3-3" / "phase3_3_technical_sections_preview.tex").write_text("Technical.", encoding="utf-8")
            (run_dir / "phase3-4" / "verified_reference_bank.json").write_text("[]", encoding="utf-8")
            (run_dir / "phase3-4" / "citation_claim_map.json").write_text("[]", encoding="utf-8")
            (run_dir / "phase3-4" / "full_paper_preview.pdf").write_text("pdf", encoding="utf-8")

            literature = _sync_role_agent(run_dir, "literature_agent", event="unit_literature")
            writing = _sync_role_agent(run_dir, "writing_agent", event="unit_writing")

            self.assertEqual(literature["snapshot"]["status"], "ready")
            self.assertEqual(writing["snapshot"]["status"], "ready")
            self.assertTrue((run_dir / "agent-sync" / "literature_agent_unit_literature.json").exists())
            self.assertTrue((run_dir / "agent-sync" / "writing_agent_unit_writing.json").exists())
            self.assertTrue((run_dir / "agent-requests" / "literature_agent_request.json").exists())
            self.assertTrue((run_dir / "agent-requests" / "writing_agent_request.json").exists())
            workspace = json.loads((run_dir / "agent_workspace_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(workspace["artifacts"]["reference_bank"]["path"], "phase3-4/verified_reference_bank.json")
            self.assertEqual(workspace["artifacts"]["technical_sections"]["path"], "phase3-3/phase3_3_technical_sections_preview.tex")


if __name__ == "__main__":
    unittest.main()
