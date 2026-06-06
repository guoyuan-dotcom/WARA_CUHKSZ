from __future__ import annotations

import unittest

import json
import os
import tempfile
import time
from pathlib import Path
import sys


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from phase25_analysis import (  # noqa: E402
    _apply_sweep_value,
    _is_count_like_sweep_param,
    _is_violation_or_feasibility_metric,
    _metric_varies_across_sweep,
    _paired_method_x_points,
    _prefer_phase24_output_paths,
    _safe_display,
    _sanitize_suggested_values,
    _validation_output_staleness,
    _validation_plan_sweep_values,
    _write_validation_source_fingerprint,
    aggregate_for_figure,
    align_phase25_plan_with_observed_sweep_plan,
    apply_phase25_sweep_display_names,
    build_per_case_comparison,
    check_data_sufficiency,
    evaluate_figure_level_primary_claim_check,
    evaluate_primary_claim_check,
    load_phase24_results,
    normalize_figure_spec,
    render_evidence_table,
    render_table,
    repair_mechanism_figure_metrics_from_data,
    run_phase24_paper_sweep_from_plan,
    select_comparison_baseline_method,
    select_single_benchmark_methods_for_figures,
    select_strongest_practical_baseline_method,
    validate_phase25_contract_consistency,
    write_missing_experiments,
)
from phase_runtime.phase25_planning import _normalize_phase25_refined_sweep_plan  # noqa: E402

import pandas as pd  # noqa: E402


class _SchemaProblem:
    def __init__(self) -> None:
        self.fields = {
            "system": {"M": 64, "Pmax": 1.0},
            "optimization": {"lambda3": 0.3},
        }
        self.validation_plan = {"canonical_config": self.fields}

    def clone_with(self, *, case_name, case_id, swept_param, swept_value, scenario_name, updates):
        clone = _SchemaProblem()
        clone.case_name = case_name
        clone.case_id = case_id
        clone.swept_param = swept_param
        clone.swept_value = swept_value
        clone.scenario_name = scenario_name
        clone.updates = updates
        return clone


class Phase25AnalysisTests(unittest.TestCase):
    def _write_phase25_staleness_fixture(self, run_dir: Path) -> None:
        solver_dir = run_dir / "phase2-4" / "solver"
        outputs_dir = solver_dir / "outputs"
        phase25_dir = run_dir / "phase2-5"
        outputs_dir.mkdir(parents=True, exist_ok=True)
        phase25_dir.mkdir(parents=True, exist_ok=True)
        (solver_dir / "generated_plugin.py").write_text("# plugin\n", encoding="utf-8")
        (solver_dir / "generated_experiment_core.py").write_text("# core\n", encoding="utf-8")
        (solver_dir / "problem_data.py").write_text("# problem\n", encoding="utf-8")
        (solver_dir / "validation_cases.py").write_text("# cases\n", encoding="utf-8")
        (run_dir / "phase2-4" / "validation_plan.yaml").write_text("canonical_config: {}\n", encoding="utf-8")
        (phase25_dir / "experiment_plan.json").write_text('{"primary_metric":{"higher_is_better":true}}\n', encoding="utf-8")
        (phase25_dir / "paper_sweep_plan.json").write_text('{"figures":[]}\n', encoding="utf-8")

    def test_validation_staleness_uses_content_fingerprint_not_mtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            self._write_phase25_staleness_fixture(run_dir)
            outputs_dir = run_dir / "phase2-4" / "solver" / "outputs"
            (outputs_dir / "paper_validation_results.csv").write_text("case_id,method\nc1,proposed\n", encoding="utf-8")
            (outputs_dir / "paper_validation_summary.json").write_text(
                json.dumps({"partial": False, "planned_jobs": 1, "actual_completed_jobs": 1}),
                encoding="utf-8",
            )
            _write_validation_source_fingerprint(run_dir, "paper_validation")

            time.sleep(0.01)
            (run_dir / "phase2-4" / "validation_plan.yaml").write_text("canonical_config: {}\n", encoding="utf-8")
            self.assertFalse(_validation_output_staleness(run_dir, "paper_validation")["is_stale"])

            (run_dir / "phase2-4" / "validation_plan.yaml").write_text("canonical_config:\n  changed: true\n", encoding="utf-8")
            stale = _validation_output_staleness(run_dir, "paper_validation")
            self.assertTrue(stale["is_stale"])
            self.assertEqual(stale["reason"], "paper_validation_source_content_changed")

    def test_preferred_phase24_outputs_skip_partial_paper_sweep(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            self._write_phase25_staleness_fixture(run_dir)
            outputs_dir = run_dir / "phase2-4" / "solver" / "outputs"
            (outputs_dir / "paper_validation_results.csv").write_text("case_id,method\nc1,proposed\n", encoding="utf-8")
            (outputs_dir / "paper_validation_summary.json").write_text(
                json.dumps({"partial": True, "planned_jobs": 10, "actual_completed_jobs": 4}),
                encoding="utf-8",
            )
            _write_validation_source_fingerprint(run_dir, "paper_validation")
            (outputs_dir / "medium_validation_results.csv").write_text("case_id,method\nm1,proposed\n", encoding="utf-8")
            (outputs_dir / "medium_validation_summary.json").write_text(
                json.dumps({"partial": False, "planned_jobs": 1, "actual_completed_jobs": 1}),
                encoding="utf-8",
            )
            _write_validation_source_fingerprint(run_dir, "medium_validation")

            _summary_path, _results_path, label = _prefer_phase24_output_paths(run_dir)

        self.assertEqual(label, "medium_validation")

    def test_paper_sweep_resume_reuses_existing_figure_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            solver_dir = run_dir / "phase2-4" / "solver"
            outputs_dir = solver_dir / "outputs"
            phase25_dir = run_dir / "phase2-5"
            outputs_dir.mkdir(parents=True, exist_ok=True)
            phase25_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "phase2-4" / "validation_plan.yaml").write_text("canonical_config: {}\n", encoding="utf-8")
            (phase25_dir / "experiment_plan.json").write_text(
                json.dumps(
                    {
                        "primary_metric": {"name": "objective", "higher_is_better": True},
                        "compared_methods": [
                            {"internal_name": "proposed"},
                            {"internal_name": "baseline"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (phase25_dir / "paper_sweep_plan.json").write_text(
                json.dumps(
                    {
                        "figures": [
                            {
                                "figure_id": "figure_1",
                                "required_sweep_param": "system.x",
                                "suggested_values": [1.0, 2.0],
                                "suggested_num_seeds": 2,
                                "methods_to_run": ["proposed", "baseline"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            (solver_dir / "problem_data.py").write_text(
                """
class ProblemData:
    def __init__(self, fields=None, case_name="canonical", case_id="canonical", swept_param="canonical", swept_value=0.0, scenario_name="default", validation_plan=None):
        self.fields = fields or {"system": {"x": 0.0}}
        self.case_name = case_name
        self.case_id = case_id
        self.swept_param = swept_param
        self.swept_value = swept_value
        self.scenario_name = scenario_name
        self.validation_plan = validation_plan or {}

    def clone_with(self, *, case_name, case_id, swept_param, swept_value, scenario_name, updates):
        fields = {"system": {"x": float(updates.get("system.x", swept_value))}}
        return ProblemData(fields=fields, case_name=case_name, case_id=case_id, swept_param=swept_param, swept_value=swept_value, scenario_name=scenario_name, validation_plan=self.validation_plan)
""",
                encoding="utf-8",
            )
            (solver_dir / "validation_cases.py").write_text(
                """
from problem_data import ProblemData

def load_canonical_case(_path):
    return ProblemData()
""",
                encoding="utf-8",
            )
            (solver_dir / "generated_plugin.py").write_text(
                """
def build_model(problem, seed=0):
    return {"metadata": {"max_iterations": 1}}

def initial_state(problem, model, seed=0):
    return {"method": "proposed"}

def proposed_step(problem, model, state, iteration):
    return state

def method_solution(problem, model, method_key, seed=0):
    return {"method": method_key}

def evaluate_state(problem, model, state):
    method = state.get("method", "baseline") if isinstance(state, dict) else "baseline"
    return {"status": "ok", "feasible": True, "objective": 2.0 if method == "proposed" else 1.0}
""",
                encoding="utf-8",
            )
            (outputs_dir / "paper_validation_results.csv").write_text(
                "case_id,case_name,required_sweep,seed,swept_param,swept_value,scenario_name,method,status,objective,feasible,iterations,solve_time_sec\n"
                "figure_1_system.x_1.0,figure_1_system.x_1.0,,0,system.x,1.0,default,proposed,ok,2.0,True,1,0.001\n"
                "figure_1_system.x_1.0,figure_1_system.x_1.0,,0,system.x,1.0,default,baseline,ok,1.0,True,1,0.001\n",
                encoding="utf-8-sig",
            )
            (outputs_dir / "paper_validation_cases.json").write_text(
                json.dumps(
                    [
                        {
                            "case_id": "figure_1_system.x_1.0",
                            "figure_id": "figure_1",
                            "seed": 0,
                            "swept_param": "system.x",
                            "swept_value": 1.0,
                        }
                    ]
                ),
                encoding="utf-8",
            )
            (outputs_dir / "paper_validation_errors.json").write_text('{"errors":[]}', encoding="utf-8")
            (outputs_dir / "paper_validation_summary.json").write_text(
                json.dumps({"partial": True, "planned_jobs": 4, "actual_completed_jobs": 1}),
                encoding="utf-8",
            )
            _write_validation_source_fingerprint(run_dir, "paper_validation", include_phase25_plan=True)

            result = run_phase24_paper_sweep_from_plan(run_dir, quick=False)
            summary = json.loads((outputs_dir / "paper_validation_summary.json").read_text(encoding="utf-8"))
            rows = pd.read_csv(outputs_dir / "paper_validation_results.csv", encoding="utf-8-sig")

        self.assertFalse(summary["partial"])
        self.assertEqual(summary["planned_jobs"], 4)
        self.assertEqual(summary["actual_completed_jobs"], 4)
        self.assertEqual(result["num_cases"], 4)
        self.assertEqual(len(rows), 8)
        self.assertEqual(rows.groupby(["case_id", "seed"]).size().shape[0], 4)

    def test_sanitize_nonnegative_weight_sweep_values(self) -> None:
        values = _sanitize_suggested_values("optimization.lambda3", [-2.0, -0.5, 0.0, 0.3, 1.0])
        self.assertEqual(values, [0.0, 0.3, 1.0])

    def test_sanitize_mw_requirement_sweep_values_are_positive(self) -> None:
        values = _sanitize_suggested_values("requirements.Gamma_s_mW", [-1.0, 0.0, 1.5])
        self.assertEqual(values, [1.5])

    def test_sanitize_spacing_meter_sweep_values_stay_continuous(self) -> None:
        values = _sanitize_suggested_values("constraints.d_min_m", [0.04, 0.1, 0.16])
        self.assertEqual(values, [0.04, 0.1, 0.16])
        self.assertFalse(_is_count_like_sweep_param("constraints.d_min_m"))

    def test_safe_display_preserves_latex_axis_notation(self) -> None:
        self.assertEqual(_safe_display(r"$P_{\max}$"), r"$P_{\max}$")
        self.assertEqual(_safe_display(r"d_{\min}/\lambda"), r"$d_{\min}/\lambda$")
        self.assertEqual(
            _safe_display(r"length of \mathcal R_m / \lambda"),
            r"length of $\mathcal{R}_m/\lambda$",
        )

    def test_sweep_display_names_use_validation_plan_not_raw_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            phase24_dir = run_dir / "phase2-4"
            phase24_dir.mkdir(parents=True)
            (phase24_dir / "validation_plan.yaml").write_text(
                """
sweep_definitions:
  - id: sweep_delta
    canonical_path: uncertainty.delta_h_norm
    variable: "$\\\\delta_{h,k}$"
""",
                encoding="utf-8",
            )
            plan = {
                "primary_metric": {"name": "P_tx_mW", "display_name": "$P_{\\rm tx}$", "higher_is_better": False},
                "figure_specs": [
                    {
                        "figure_id": "figure_1",
                        "chart_type": "scatter",
                        "metric": {"name": "P_tx_mW", "display_name": "$P_{\\rm tx}$", "higher_is_better": False},
                        "encoding": {
                            "x": {
                                "field": "swept_value",
                                "sweep_param": "uncertainty.delta_h_norm",
                                "display_name": "uncertainty.delta_h_norm",
                            }
                        },
                        "methods": ["proposed", "baseline"],
                    }
                ],
            }

            repaired = apply_phase25_sweep_display_names(plan, run_dir)

        self.assertEqual(repaired["figure_specs"][0]["encoding"]["x"]["display_name"], "$\\delta_{h,k}$")

    def test_refined_sweep_plan_updates_stale_figure_axis_when_rows_use_new_param(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            phase25_dir = run_dir / "phase2-5"
            phase25_dir.mkdir(parents=True)
            (phase25_dir / "paper_sweep_plan_refined.json").write_text(
                json.dumps(
                    {
                        "missing_for_figures": [
                            {
                                "figure_id": "figure_1",
                                "required_sweep_param": "requirements.Gamma_s_mW",
                                "suggested_values": [1.0, 2.0, 3.0],
                                "methods_to_run": ["proposed", "iso"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            plan = {
                "primary_metric": {"name": "achieved_tau", "higher_is_better": True},
                "figure_specs": [
                    {
                        "figure_id": "figure_1",
                        "chart_type": "line",
                        "metric": {"name": "achieved_tau", "higher_is_better": True},
                        "encoding": {
                            "x": {
                                "type": "numeric",
                                "field": "swept_value",
                                "sweep_param": "requirements.gamma_dB",
                                "display_name": "$\\gamma_0$",
                            }
                        },
                        "methods": ["proposed", "old_baseline"],
                    }
                ],
            }
            df = pd.DataFrame(
                [
                    {
                        "case_id": "figure_1_requirements.Gamma_s_mW_1.0",
                        "swept_param": "requirements.Gamma_s_mW",
                        "swept_value": 1.0,
                        "method": "proposed",
                        "achieved_tau": 1.0,
                        "status": "ok",
                    }
                ]
            )

            aligned = align_phase25_plan_with_observed_sweep_plan(plan, phase25_dir, df)

        fig = aligned["figure_specs"][0]
        self.assertEqual(fig["encoding"]["x"]["sweep_param"], "requirements.Gamma_s_mW")
        self.assertEqual(fig["encoding"]["x"]["field"], "swept_value")
        self.assertEqual(fig["methods"], ["proposed", "iso"])

    def test_validation_plan_sweep_values_prefers_phase24_paper_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            phase25_dir = Path(tmp) / "phase2-5"
            phase24_dir = Path(tmp) / "phase2-4"
            phase25_dir.mkdir(parents=True)
            phase24_dir.mkdir(parents=True)
            (phase24_dir / "validation_plan.yaml").write_text(
                """
sweep_definitions:
  - id: sweep_P_max_mW
    variable: $P_{\\max}$
    canonical_path: constraints.P_max_mW
    scout_values: [140.0, 250.0, 420.0]
    paper_values: [120.0, 150.0, 180.0, 210.0]
""",
                encoding="utf-8",
            )

            values = _validation_plan_sweep_values(phase25_dir, "constraints.P_max_mW", required_sweep="sweep_P_max_mW")
            values_by_path = _validation_plan_sweep_values(phase25_dir, "constraints.P_max_mW")

        self.assertEqual(values, [120.0, 150.0, 180.0, 210.0])
        self.assertEqual(values_by_path, [120.0, 150.0, 180.0, 210.0])

    def test_meter_geometry_sweeps_are_not_treated_as_integer_counts(self) -> None:
        values = [1.0, 1.08, 1.15, 1.23, 1.31, 1.38, 1.46, 1.54, 1.62, 1.69, 1.77, 1.85, 1.92, 2.0]

        self.assertFalse(_is_count_like_sweep_param("system.aperture.side_length_m"))
        self.assertFalse(_is_count_like_sweep_param("system.min_spacing_m"))
        self.assertEqual(_sanitize_suggested_values("system.aperture.side_length_m", values), values)

    def test_phase25_contract_consistency_blocks_mismatched_sweep_id_and_param(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            phase24_dir = run_dir / "phase2-4"
            phase25_dir = run_dir / "phase2-5"
            phase24_dir.mkdir(parents=True)
            phase25_dir.mkdir(parents=True)
            (phase24_dir / "validation_plan.yaml").write_text(
                """
sweep_definitions:
  - id: sweep_delta
    canonical_path: uncertainty.delta
    paper_values: [0.0, 0.1]
  - id: sweep_power
    canonical_path: constraints.Pmax
    paper_values: [10.0, 20.0]
""",
                encoding="utf-8",
            )
            (phase25_dir / "figure_captions.md").write_text(
                "# Figure Captions\n\n"
                "## figure_1\n"
                "Fig. 1. Sum rate $R_{\\mathrm{sum}}$ versus transmit power budget $P_{\\max}$.\n",
                encoding="utf-8",
            )
            plan = {
                "primary_metric": {"name": "sum_rate_bpsHz", "display_name": r"sum rate $R_{\mathrm{sum}}$ (bps/Hz)", "higher_is_better": True},
                "figure_specs": [
                    {
                        "figure_id": "figure_1",
                        "required_sweep": "sweep_delta",
                        "chart_type": "line",
                        "metric": {"name": "sum_rate_bpsHz", "display_name": r"sum rate $R_{\mathrm{sum}}$ (bps/Hz)", "higher_is_better": True},
                        "encoding": {
                            "x": {
                                "field": "swept_value",
                                "sweep_param": "constraints.Pmax",
                                "sweep_id": "sweep_delta",
                                "display_name": r"transmit power budget $P_{\max}$",
                            }
                        },
                        "methods": ["proposed", "baseline"],
                    }
                ],
            }
            df = pd.DataFrame(
                [
                    {
                        "figure_id": "figure_1",
                        "required_sweep": "sweep_delta",
                        "swept_param": "constraints.Pmax",
                        "swept_value": 10.0,
                        "method": "proposed",
                        "sum_rate_bpsHz": 5.0,
                        "status": "ok",
                        "success": True,
                        "feasible": True,
                    }
                ]
            )

            report = validate_phase25_contract_consistency(
                run_dir,
                plan,
                df,
                figure_outputs=[
                    {
                        "figure_id": "figure_1",
                        "x_axis_param": "constraints.Pmax",
                        "required_sweep": "sweep_delta",
                        "y_metric": "sum_rate_bpsHz",
                    }
                ],
            )

        self.assertFalse(report["ok"])
        self.assertTrue(any("inconsistent with x.sweep_param" in item for item in report["errors"]))

    def test_phase25_contract_consistency_accepts_aligned_sweep_rows_and_caption(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            phase24_dir = run_dir / "phase2-4"
            phase25_dir = run_dir / "phase2-5"
            phase24_dir.mkdir(parents=True)
            phase25_dir.mkdir(parents=True)
            (phase24_dir / "validation_plan.yaml").write_text(
                """
sweep_definitions:
  - id: sweep_delta
    canonical_path: uncertainty.delta
    paper_values: [0.0, 0.1]
  - id: sweep_power
    canonical_path: constraints.Pmax
    paper_values: [10.0, 20.0]
""",
                encoding="utf-8",
            )
            (phase25_dir / "figure_captions.md").write_text(
                "# Figure Captions\n\n"
                "## figure_1\n"
                "Fig. 1. Sum rate $R_{\\mathrm{sum}}$ versus transmit power budget $P_{\\max}$.\n",
                encoding="utf-8",
            )
            plan = {
                "primary_metric": {"name": "sum_rate_bpsHz", "display_name": r"sum rate $R_{\mathrm{sum}}$ (bps/Hz)", "higher_is_better": True},
                "figure_specs": [
                    {
                        "figure_id": "figure_1",
                        "required_sweep": "sweep_power",
                        "chart_type": "line",
                        "metric": {"name": "sum_rate_bpsHz", "display_name": r"sum rate $R_{\mathrm{sum}}$ (bps/Hz)", "higher_is_better": True},
                        "encoding": {
                            "x": {
                                "field": "swept_value",
                                "sweep_param": "constraints.Pmax",
                                "sweep_id": "sweep_power",
                                "display_name": r"transmit power budget $P_{\max}$",
                            }
                        },
                        "methods": ["proposed", "baseline"],
                    }
                ],
            }
            rows = []
            for method, value in [("proposed", 5.0), ("baseline", 4.0)]:
                rows.append(
                    {
                        "figure_id": "figure_1",
                        "required_sweep": "sweep_power",
                        "swept_param": "constraints.Pmax",
                        "swept_value": 10.0,
                        "method": method,
                        "sum_rate_bpsHz": value,
                        "status": "ok",
                        "success": True,
                        "feasible": True,
                    }
                )

            report = validate_phase25_contract_consistency(
                run_dir,
                plan,
                pd.DataFrame(rows),
                figure_outputs=[
                    {
                        "figure_id": "figure_1",
                        "x_axis_param": "constraints.Pmax",
                        "required_sweep": "sweep_power",
                        "y_metric": "sum_rate_bpsHz",
                    }
                ],
            )

        self.assertTrue(report["ok"])
        self.assertEqual(report["errors"], [])

    def test_phase25_benchmark_selection_prefers_viable_full_sweep_benchmark(self) -> None:
        rows = []
        for x in [1.0, 2.0, 3.0]:
            rows.append({"figure_id": "figure_1", "swept_param": "power", "swept_value": x, "method": "proposed", "sum_rate_bpsHz": 10.0 + x, "feasible": True, "success": True})
            rows.append({"figure_id": "figure_1", "swept_param": "power", "swept_value": x, "method": "fixed_ps_baseline", "sum_rate_bpsHz": 9.8 + x, "feasible": True, "success": True})
            rows.append({"figure_id": "figure_1", "swept_param": "power", "swept_value": x, "method": "mrt_split_baseline", "sum_rate_bpsHz": 2.0, "feasible": x == 1.0, "success": x == 1.0})
        df = pd.DataFrame(rows)
        plan = {
            "primary_metric": {"name": "sum_rate_bpsHz", "higher_is_better": True},
            "compared_methods": [
                {"name": "proposed", "role": "proposed"},
                {"name": "fixed_ps_baseline", "role": "main_baseline", "display_priority": 0},
                {"name": "mrt_split_baseline", "role": "direction_heuristic", "display_priority": 2},
            ],
            "figure_specs": [
                {
                    "figure_id": "figure_1",
                    "chart_intent": "main_comparison",
                    "metric": {"name": "sum_rate_bpsHz", "higher_is_better": True},
                    "encoding": {"x": {"field": "swept_value", "sweep_param": "power"}},
                    "methods": ["proposed", "fixed_ps_baseline", "mrt_split_baseline"],
                }
            ],
        }

        repaired = select_single_benchmark_methods_for_figures(plan, df)

        self.assertEqual(repaired["figure_specs"][0]["methods"], ["proposed", "fixed_ps_baseline"])
        self.assertIn("fixed_ps_baseline", repaired["figure_specs"][0]["benchmark_selection"]["viable_candidate_methods"])

    def test_aggregate_for_figure_uses_paired_successful_seeds(self) -> None:
        rows = [
            {"figure_id": "figure_1", "swept_param": "gamma", "swept_value": 2.0, "seed": 0, "method": "proposed", "P_tx_mW": 100.0, "feasible": True, "success": True, "status": "ok", "finite_primary_metric": True},
            {"figure_id": "figure_1", "swept_param": "gamma", "swept_value": 2.0, "seed": 0, "method": "mrt", "P_tx_mW": 0.0, "feasible": False, "success": False, "status": "infeasible", "finite_primary_metric": True},
            {"figure_id": "figure_1", "swept_param": "gamma", "swept_value": 2.0, "seed": 1, "method": "proposed", "P_tx_mW": 2.0, "feasible": True, "success": True, "status": "ok", "finite_primary_metric": True},
            {"figure_id": "figure_1", "swept_param": "gamma", "swept_value": 2.0, "seed": 1, "method": "mrt", "P_tx_mW": 3.0, "feasible": True, "success": True, "status": "ok", "finite_primary_metric": True},
        ]
        fig = {
            "figure_id": "figure_1",
            "chart_type": "line",
            "metric": {"name": "P_tx_mW", "higher_is_better": False},
            "encoding": {"x": {"field": "swept_value", "sweep_param": "gamma"}},
            "methods": ["proposed", "mrt"],
        }

        curve = aggregate_for_figure(pd.DataFrame(rows), fig)
        proposed = curve[curve["method"] == "proposed"].iloc[0]
        mrt = curve[curve["method"] == "mrt"].iloc[0]

        self.assertEqual(float(proposed["mean_metric"]), 2.0)
        self.assertEqual(float(mrt["mean_metric"]), 3.0)
        self.assertEqual(int(proposed["num_unique_seeds"]), 1)

    def test_aggregate_for_figure_filters_undercovered_stress_tail(self) -> None:
        rows = []
        for x_value in [0.0, 0.03, 0.06, 0.12]:
            for seed in range(6):
                feasible = x_value < 0.12 or seed < 2
                for method, base in [("proposed", 1.0), ("mrt", 1.4)]:
                    rows.append(
                        {
                            "figure_id": "figure_2",
                            "swept_param": "uncertainty.delta",
                            "swept_value": x_value,
                            "seed": seed,
                            "method": method,
                            "P_tx_mW": base + x_value + 0.01 * seed if feasible else 0.0,
                            "feasible": feasible,
                            "success": feasible,
                            "status": "ok",
                            "finite_primary_metric": True,
                        }
                    )
        fig = {
            "figure_id": "figure_2",
            "chart_type": "line",
            "chart_intent": "robustness_stress",
            "metric": {"name": "P_tx_mW", "higher_is_better": False},
            "encoding": {"x": {"field": "swept_value", "sweep_param": "uncertainty.delta"}},
            "methods": ["proposed", "mrt"],
        }

        curve = aggregate_for_figure(pd.DataFrame(rows), fig)

        self.assertEqual(set(curve["x_value"].tolist()), {0.0, 0.03, 0.06})
        self.assertNotIn(0.12, set(curve["x_value"].tolist()))

    def test_comparison_baseline_uses_selected_main_figure_benchmark(self) -> None:
        df = pd.DataFrame(
            {
                "method": ["proposed", "fixed_ps_baseline", "mrt_split_baseline"],
                "sum_rate_bpsHz": [10.0, 9.0, 2.0],
                "feasible": [True, True, True],
                "success": [True, True, True],
            }
        )
        plan = {
            "primary_metric": {"name": "sum_rate_bpsHz", "higher_is_better": True},
            "compared_methods": [
                {"name": "proposed", "role": "proposed"},
                {"name": "fixed_ps_baseline", "role": "main_baseline", "display_priority": 0},
                {"name": "mrt_split_baseline", "role": "direction_heuristic", "display_priority": 2},
            ],
            "figure_specs": [
                {
                    "figure_id": "figure_1",
                    "chart_intent": "main_comparison",
                    "metric": {"name": "sum_rate_bpsHz", "higher_is_better": True},
                    "methods": ["proposed", "fixed_ps_baseline"],
                }
            ],
        }

        self.assertEqual(select_comparison_baseline_method(df, plan), "fixed_ps_baseline")

    def test_comparison_baseline_uses_strongest_gain_showing_plotted_benchmark(self) -> None:
        df = pd.DataFrame(
            {
                "method": ["proposed", "weak_baseline", "strong_baseline"],
                "sum_rate_bpsHz": [10.0, 4.0, 9.0],
                "feasible": [True, True, True],
                "success": [True, True, True],
            }
        )
        plan = {
            "primary_metric": {"name": "sum_rate_bpsHz", "higher_is_better": True},
            "compared_methods": [
                {"name": "proposed", "role": "proposed"},
                {"name": "weak_baseline", "role": "main_baseline", "display_priority": 2},
                {"name": "strong_baseline", "role": "practical_heuristic", "display_priority": 3},
            ],
            "_active_baseline_method": "weak_baseline",
            "_comparison_baseline_method": "weak_baseline",
            "figure_specs": [
                {
                    "figure_id": "figure_1",
                    "chart_intent": "main_comparison",
                    "metric": {"name": "sum_rate_bpsHz", "higher_is_better": True},
                    "methods": ["proposed", "weak_baseline", "strong_baseline"],
                }
            ],
        }

        self.assertEqual(select_comparison_baseline_method(df, plan), "strong_baseline")

        plan["_forced_comparison_baseline_method"] = "weak_baseline"
        self.assertEqual(select_comparison_baseline_method(df, plan), "weak_baseline")

    def test_schema_driven_paper_sweep_update_uses_nested_path(self) -> None:
        problem, error = _apply_sweep_value(None, _SchemaProblem(), "optimization.lambda3", 1.0, "case", "case")
        self.assertIsNone(error)
        self.assertIsNotNone(problem)
        self.assertEqual(problem.updates, {"optimization.lambda3": 1.0})
        self.assertEqual(problem.swept_param, "optimization.lambda3")

    def test_schema_driven_count_sweep_is_integer(self) -> None:
        problem, error = _apply_sweep_value(None, _SchemaProblem(), "system.M", 127.6, "case", "case")
        self.assertIsNone(error)
        self.assertEqual(problem.updates, {"system.M": 128})

    def test_figure_spec_uses_top_level_y_metric_before_objective_fallback(self) -> None:
        spec = normalize_figure_spec(
            {
                "figure_id": "figure_2",
                "chart_type": "line",
                "y_metric": "sensing_gain",
                "x_axis": {"source": "swept_value", "sweep_param": "system.partition_fraction"},
            },
            {"name": "objective", "display_name": "Weighted objective"},
        )
        self.assertEqual(spec["metric"]["name"], "sensing_gain")
        self.assertEqual(spec["metric"]["display_name"], "Sensing gain")

    def test_violation_metric_direction_is_lower_better_even_if_plan_says_otherwise(self) -> None:
        spec = normalize_figure_spec(
            {
                "figure_id": "figure_2",
                "chart_type": "line",
                "metric": {
                    "name": "max_constraint_violation",
                    "display_name": "Bad copied label",
                    "higher_is_better": True,
                },
                "x_axis": {"source": "swept_value", "sweep_param": "constraints.E_min_mW"},
            },
            {"name": "objective", "display_name": "Weighted objective", "higher_is_better": True},
        )
        self.assertFalse(spec["metric"]["higher_is_better"])

    def test_metric_variation_detector_rejects_flat_sweep(self) -> None:
        df = pd.DataFrame(
            {
                "method": ["proposed", "proposed", "baseline", "baseline"],
                "swept_value": [0.0, 1.0, 0.0, 1.0],
                "sensing_gain": [3.0, 3.0, 2.0, 2.0],
            }
        )
        varies, detail = _metric_varies_across_sweep(df, y_metric="sensing_gain", x_field="swept_value")
        self.assertFalse(varies)
        self.assertEqual(len(detail["by_method"]), 2)

    def test_violation_metrics_are_not_treated_as_success_only_metrics(self) -> None:
        self.assertTrue(_is_violation_or_feasibility_metric("any_constraint_violation"))
        self.assertTrue(_is_violation_or_feasibility_metric("feasibility_rate"))
        self.assertFalse(_is_violation_or_feasibility_metric("Ps_dB"))

    def test_deterministic_feasibility_boundary_zero_seed_variance_is_not_blocking(self) -> None:
        rows = []
        x_values = list(range(10))
        for method in ["proposed", "no_rho"]:
            for x_value in x_values:
                for seed in range(80):
                    rows.append(
                        {
                            "method": method,
                            "swept_param": "constraints.E_min_mW",
                            "swept_value": float(x_value),
                            "seed": seed,
                            "status": "ok",
                            "feasible": bool(x_value < 5),
                            "objective": 1.0 + x_value,
                            "finite_primary_metric": True,
                        }
                    )
        df = pd.DataFrame(rows)
        plan = {
            "primary_metric": {"name": "objective", "higher_is_better": True},
            "compared_methods": [{"name": "proposed"}, {"name": "no_rho"}],
            "figure_specs": [
                {
                    "figure_id": "figure_2",
                    "chart_type": "line",
                    "chart_intent": "feasibility_boundary",
                    "methods": ["proposed", "no_rho"],
                    "metric": {"name": "feasible", "display_name": "Feasibility rate"},
                    "encoding": {
                        "x": {"field": "swept_value", "sweep_param": "constraints.E_min_mW"},
                        "group": {"field": "method"},
                        "facet": {"field": None},
                    },
                    "data_requirements": {"min_points": 10, "preferred_points": 10, "min_samples_per_group": 80, "preferred_samples_per_group": 100},
                }
            ],
        }
        mc_rows = []
        for method in ["proposed", "no_rho"]:
            for x_value in x_values:
                mc_rows.append(
                    {
                        "figure_id": "figure_2",
                        "method": method,
                        "x_value": float(x_value),
                        "num_unique_seeds": 80,
                        "warnings": ["repeated_identical_outputs_across_seeds", "zero_variance_across_all_seeds"],
                    }
                )
        mc_report = {
            "unknown_seed_coverage": False,
            "figures": [{"figure_id": "figure_2", "rows": mc_rows}],
        }
        report = check_data_sufficiency(df, pd.DataFrame(), plan, mc_report, quick_mode=False)
        figure = report["figures"][0]
        self.assertTrue(figure["deterministic_boundary_valid"])
        self.assertTrue(figure["monte_carlo_valid"])
        self.assertTrue(figure["paper_ready"])
        self.assertFalse(figure["counts_toward_paper_minimum"])
        self.assertEqual(report["overall_status"], "needs_more_phase24_runs")
        self.assertTrue(any("non-diagnostic paper-ready figures" in item for item in report["global_blocking_issues"]))
        self.assertNotIn("zero_variance_across_seeds", figure["blocking_issues"])

    def test_complete_discrete_integer_count_grid_can_be_paper_ready_with_nine_points(self) -> None:
        rows = []
        x_values = list(range(2, 11))
        for method, offset in [("proposed", 3.0), ("regularized_heuristic", 0.0)]:
            for x_value in x_values:
                for seed in range(80):
                    rows.append(
                        {
                            "figure_id": "figure_users",
                            "swept_param": "network.num_users",
                            "swept_value": float(x_value),
                            "seed": seed,
                            "method": method,
                            "status": "ok",
                            "feasible": True,
                            "success": True,
                            "min_rate_bpsHz": 20.0 - 0.5 * x_value + offset + 0.001 * seed,
                            "finite_primary_metric": True,
                        }
                    )
        plan = {
            "primary_metric": {"name": "min_rate_bpsHz", "higher_is_better": True},
            "compared_methods": [{"name": "proposed"}, {"name": "regularized_heuristic"}],
            "figure_specs": [
                {
                    "figure_id": "figure_users",
                    "chart_type": "line",
                    "chart_intent": "scalability_stress",
                    "methods": ["proposed", "regularized_heuristic"],
                    "metric": {"name": "min_rate_bpsHz", "higher_is_better": True},
                    "encoding": {
                        "x": {"field": "swept_value", "sweep_param": "network.num_users", "display_name": "$K$"},
                        "group": {"field": "method"},
                    },
                    "data_requirements": {"min_points": 10, "preferred_points": 14, "min_samples_per_group": 80, "preferred_samples_per_group": 100},
                }
            ],
        }
        mc_report = {
            "unknown_seed_coverage": False,
            "figures": [
                {
                    "figure_id": "figure_users",
                    "rows": [
                        {"figure_id": "figure_users", "method": method, "x_value": float(x_value), "num_unique_seeds": 80, "warnings": []}
                        for method in ["proposed", "regularized_heuristic"]
                        for x_value in x_values
                    ],
                }
            ],
        }

        report = check_data_sufficiency(pd.DataFrame(rows), pd.DataFrame(), plan, mc_report, quick_mode=False)
        figure = report["figures"][0]

        self.assertTrue(figure["paper_ready"])
        self.assertNotIn("too_few_x_points", figure["blocking_issues"])
        self.assertNotIn("too_few_effective_x_points_after_feasibility_filter", figure["blocking_issues"])
        self.assertEqual(figure["suggested_min_x_points"], 9)
        self.assertEqual(figure["min_points_adjustment_reason"], "capped_to_completed_discrete_integer_grid")

    def test_missing_experiments_densifies_reliable_numeric_span(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            phase25_dir = Path(tmp) / "phase2-5"
            phase25_dir.mkdir(parents=True)
            observed_values = [0.01, 0.018, 0.026, 0.035, 0.045, 0.058, 0.072]
            report = {
                "figures": [
                    {
                        "figure_id": "figure_1",
                        "chart_type": "line",
                        "x_axis_param": "requirements.E_receiver_min_mW",
                        "y_metric": "weighted_sum_rate_bpsHz",
                        "num_x_points": len(observed_values),
                        "x_values": observed_values,
                        "all_requested_x_values": observed_values,
                        "suggested_min_x_points": 12,
                        "suggested_preferred_x_points": 14,
                        "suggested_preferred_seeds_per_point": 100,
                        "methods_required": ["proposed", "robust_baseline"],
                        "blocking_issues": [
                            "too_few_effective_x_points_after_feasibility_filter",
                            "too_few_x_points",
                        ],
                        "paired_success_seed_coverage": {
                            "reliable_x_values": observed_values,
                            "min_rate": 0.87,
                        },
                    }
                ]
            }

            write_missing_experiments(phase25_dir, report)
            plan = json.loads((phase25_dir / "paper_sweep_plan.json").read_text(encoding="utf-8"))
            values = plan["figures"][0]["suggested_values"]

        self.assertEqual(len(values), 14)
        self.assertEqual(values[0], 0.01)
        self.assertEqual(values[-1], 0.072)
        self.assertTrue(all(0.01 <= value <= 0.072 for value in values))
        self.assertTrue(plan["figures"][0]["range_densification_policy"]["used"])

    def test_missing_experiments_handles_zero_count_like_sweep_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            phase25_dir = Path(tmp) / "phase2-5"
            phase25_dir.mkdir(parents=True)
            report = {
                "figures": [
                    {
                        "figure_id": "figure_1",
                        "chart_type": "line",
                        "x_axis_param": "constraints.sensing_min",
                        "y_metric": "weighted_sum_rate_bpsHz",
                        "num_x_points": 1,
                        "x_values": [0.0],
                        "all_requested_x_values": [0.0],
                        "suggested_min_x_points": 4,
                        "suggested_preferred_x_points": 4,
                        "methods_required": ["proposed", "baseline"],
                        "blocking_issues": ["too_few_x_points"],
                    }
                ]
            }

            write_missing_experiments(phase25_dir, report)
            plan = json.loads((phase25_dir / "paper_sweep_plan.json").read_text(encoding="utf-8"))

        values = plan["figures"][0]["suggested_values"]
        self.assertGreaterEqual(len(values), 1)
        self.assertTrue(all(float(value) >= 0.0 for value in values))

    def test_missing_experiments_preserves_prior_refined_dense_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            phase25_dir = Path(tmp) / "phase2-5"
            phase25_dir.mkdir(parents=True)
            refined_values = [round(0.05 + idx * 0.025, 6) for idx in range(14)]
            (phase25_dir / "paper_sweep_plan_refined.json").write_text(
                json.dumps(
                    {
                        "figures": [
                            {
                                "figure_id": "figure_2",
                                "required_sweep": "spacing_constraint_sweep",
                                "required_sweep_param": "system.min_spacing_m",
                                "suggested_values": refined_values,
                                "scout_values": [0.05, 0.15, 0.25, 0.375],
                                "medium_values": [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35],
                                "suggested_num_seeds": 100,
                                "methods_to_run": ["proposed", "mrt"],
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            report = {
                "figures": [
                    {
                        "figure_id": "figure_2",
                        "chart_type": "line",
                        "x_axis_param": "system.min_spacing_m",
                        "y_metric": "weighted_sum_rate_bpsHz",
                        "num_x_points": 2,
                        "x_values": [1.0, 2.0],
                        "all_requested_x_values": [1.0, 2.0],
                        "suggested_min_x_points": 10,
                        "suggested_preferred_x_points": 14,
                        "suggested_preferred_seeds_per_point": 100,
                        "methods_required": ["proposed", "mrt"],
                        "blocking_issues": ["too_few_x_points", "too_few_effective_x_points_after_feasibility_filter"],
                        "paired_success_seed_coverage": {"reliable_x_values": [1.0, 2.0], "min_rate": 1.0},
                    }
                ]
            }

            write_missing_experiments(phase25_dir, report)
            plan = json.loads((phase25_dir / "paper_sweep_plan.json").read_text(encoding="utf-8"))
            figure = plan["figures"][0]

        self.assertEqual(figure["required_sweep"], "spacing_constraint_sweep")
        self.assertEqual(figure["required_sweep_param"], "system.min_spacing_m")
        self.assertEqual(figure["suggested_values"], refined_values)
        self.assertTrue(figure["preserved_prior_refined_coverage"]["used"])

    def test_missing_experiments_discards_prior_refined_coverage_when_axis_changed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            phase25_dir = Path(tmp) / "phase2-5"
            phase25_dir.mkdir(parents=True)
            (phase25_dir / "paper_sweep_plan_refined.json").write_text(
                json.dumps(
                    {
                        "figures": [
                            {
                                "figure_id": "figure_1",
                                "required_sweep": "aperture_size_sweep",
                                "required_sweep_param": "geometry.aperture.side_length_m",
                                "suggested_values": [0.24, 0.3, 0.36, 0.42, 0.48, 0.54, 0.6],
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            report = {
                "figures": [
                    {
                        "figure_id": "figure_1",
                        "chart_type": "line",
                        "x_axis_param": "constraints.P_max_W",
                        "y_metric": "weighted_sum_rate_bpsHz",
                        "num_x_points": 3,
                        "x_values": [0.05, 0.2, 1.0],
                        "all_requested_x_values": [0.05, 0.2, 1.0],
                        "suggested_min_x_points": 10,
                        "suggested_preferred_x_points": 14,
                        "suggested_preferred_seeds_per_point": 100,
                        "methods_required": ["proposed", "mrt"],
                        "blocking_issues": ["too_few_x_points"],
                    }
                ]
            }

            write_missing_experiments(phase25_dir, report)
            plan = json.loads((phase25_dir / "paper_sweep_plan.json").read_text(encoding="utf-8"))
            figure = plan["figures"][0]

        self.assertEqual(figure["required_sweep_param"], "constraints.P_max_W")
        self.assertNotEqual(figure["suggested_values"], [0.24, 0.3, 0.36, 0.42, 0.48, 0.54, 0.6])
        self.assertFalse(figure["discarded_prior_refined_coverage"]["used"])

    def test_refined_sweep_normalizer_accepts_dense_axis_change_with_valid_sweep(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            phase24_dir = run_dir / "phase2-4"
            phase25_dir = run_dir / "phase2-5"
            phase24_dir.mkdir(parents=True)
            phase25_dir.mkdir(parents=True)
            (phase24_dir / "validation_plan.yaml").write_text(
                """
sweep_definitions:
  - id: aperture_size_sweep
    canonical_path: system.aperture.side_length_m
  - id: spacing_constraint_sweep
    canonical_path: system.min_spacing_m
""",
                encoding="utf-8",
            )
            (phase25_dir / "paper_sweep_plan.json").write_text(
                json.dumps(
                    {
                        "figures": [
                            {
                                "figure_id": "figure_2",
                                "required_sweep": "aperture_size_sweep",
                                "required_sweep_param": "system.aperture.side_length_m",
                                "suggested_values": [1.0, 2.0],
                                "methods_to_run": ["proposed", "mrt"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            payload = {
                "status": "paper_sweep_plan_refined",
                "missing_for_figures": [
                    {
                        "figure_id": "figure_2",
                        "required_sweep_param": "system.min_spacing_m",
                        "suggested_values": [0.05, 0.075, 0.1, 0.125, 0.15, 0.175],
                        "methods_to_run": ["proposed", "mrt"],
                    }
                ],
            }

            normalized = _normalize_phase25_refined_sweep_plan(payload, phase25_dir)
            figure = normalized["figures"][0]

        self.assertEqual(figure["required_sweep"], "spacing_constraint_sweep")
        self.assertEqual(figure["required_sweep_param"], "system.min_spacing_m")
        self.assertEqual(len(figure["suggested_values"]), 6)

    def test_refined_sweep_normalizer_clamps_values_to_phase24_range(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            phase24_dir = run_dir / "phase2-4"
            phase25_dir = run_dir / "phase2-5"
            phase24_dir.mkdir(parents=True)
            phase25_dir.mkdir(parents=True)
            phase24_values = [0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16]
            (phase24_dir / "validation_plan.yaml").write_text(
                """
sweep_definitions:
  - id: spacing_threshold_sweep
    canonical_path: constraints.d_min_m
    paper_values: [0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16]
""",
                encoding="utf-8",
            )
            (phase25_dir / "paper_sweep_plan.json").write_text(
                json.dumps(
                    {
                        "figures": [
                            {
                                "figure_id": "figure_2",
                                "required_sweep": "spacing_threshold_sweep",
                                "required_sweep_param": "constraints.d_min_m",
                                "suggested_values": [0.04, 0.1, 0.16],
                                "methods_to_run": ["proposed", "mrt"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            payload = {
                "status": "needs_more_phase24_runs",
                "missing_for_figures": [
                    {
                        "figure_id": "figure_2",
                        "required_sweep": "spacing_threshold_sweep",
                        "required_sweep_param": "constraints.d_min_m",
                        "suggested_values": [0.5, 1.0, 2.0],
                        "replace_existing_values": True,
                    }
                ],
            }

            normalized = _normalize_phase25_refined_sweep_plan(payload, phase25_dir)
            figure = normalized["figures"][0]

        self.assertEqual(figure["suggested_values"], phase24_values)
        self.assertEqual(figure["scout_values"][0], phase24_values[0])
        self.assertEqual(figure["scout_values"][-1], phase24_values[-1])

    def test_numeric_sensitivity_box_is_coerced_to_line(self) -> None:
        fig = normalize_figure_spec(
            {
                "figure_id": "figure_2",
                "chart_intent": "sensitivity",
                "chart_type": "box",
                "metric": {"name": "optimal_rho"},
                "encoding": {
                    "x": {"type": "numeric", "field": "swept_value", "sweep_param": "constraints.E_min_mW"},
                    "group": {"field": "method"},
                },
                "error_display": "iqr",
            }
        )
        self.assertEqual(fig["chart_type"], "line")
        self.assertEqual(fig["error_display"], "ci95")

    def test_ordered_numeric_scatter_is_coerced_to_line(self) -> None:
        fig = normalize_figure_spec(
            {
                "figure_id": "figure_2",
                "chart_intent": "robustness_stress",
                "chart_type": "scatter",
                "metric": {"name": "P_tx_mW"},
                "encoding": {
                    "x": {"type": "numeric", "field": "swept_value", "sweep_param": "uncertainty.delta_h_norm"},
                    "group": {"field": "method"},
                },
            }
        )
        self.assertEqual(fig["chart_type"], "line")

    def test_figure_aggregation_ignores_stale_same_sweep_rows_from_other_figures(self) -> None:
        df = pd.DataFrame(
            {
                "case_id": [
                    "figure_2_constraints.E_min_mW_1.0",
                    "figure_4_rho_sensitivity_constraints.E_min_mW_1.0",
                    "figure_2_constraints.E_min_mW_1.0",
                ],
                "method": ["proposed", "proposed", "rho_fixed_half"],
                "swept_param": ["constraints.E_min_mW", "constraints.E_min_mW", "constraints.E_min_mW"],
                "swept_value": [1.0, 1.0, 1.0],
                "optimal_rho": [0.2, 0.8, 0.5],
                "feasible": [True, True, True],
                "finite_primary_metric": [True, True, True],
                "seed": [0, 0, 0],
            }
        )
        fig = normalize_figure_spec(
            {
                "figure_id": "figure_2",
                "chart_type": "line",
                "methods": ["proposed", "rho_fixed_half"],
                "metric": {"name": "optimal_rho"},
                "encoding": {
                    "x": {"type": "numeric", "field": "swept_value", "sweep_param": "constraints.E_min_mW"},
                    "group": {"field": "method"},
                },
            }
        )
        curve = aggregate_for_figure(df, fig)
        proposed = curve[curve["method"] == "proposed"]
        self.assertEqual(float(proposed["mean_metric"].iloc[0]), 0.2)

    def test_validation_plan_values_respect_required_sweep_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            phase24_dir = run_dir / "phase2-4"
            phase25_dir = run_dir / "phase2-5"
            phase24_dir.mkdir()
            phase25_dir.mkdir()
            (phase24_dir / "validation_plan.yaml").write_text(
                """
sweep_definitions:
  - id: sinr_feasible_sweep
    variable: constraints.gamma_target
    paper_mode_values: [0, 2, 4, 6, 8, 9.5, 10, 10.5]
  - id: sinr_stress_sweep
    variable: constraints.gamma_target
    paper_mode_values: [9, 9.5, 10, 10.5, 11, 11.5, 12]
""".strip(),
                encoding="utf-8",
            )
            feasible = _validation_plan_sweep_values(phase25_dir, "constraints.gamma_target", required_sweep="sinr_feasible_sweep")
            stress = _validation_plan_sweep_values(phase25_dir, "constraints.gamma_target", required_sweep="sinr_stress_sweep")
        self.assertEqual(feasible, [0.0, 2.0, 4.0, 6.0, 8.0, 9.5, 10.0, 10.5])
        self.assertEqual(stress, [9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0])

    def test_figure_aggregation_uses_required_sweep_before_same_parameter_fallback(self) -> None:
        rows = []
        for sweep_id, values in {
            "sinr_feasible_sweep": [0.0, 3.0],
            "sinr_stress_sweep": [11.0, 11.5],
        }.items():
            for value in values:
                rows.append(
                    {
                        "case_id": f"{sweep_id}_{value}",
                        "method": "proposed",
                        "swept_param": "constraints.gamma_target",
                        "swept_value": value,
                        "sum_power_dBm": 10.0 + value,
                        "feasible": True,
                        "finite_primary_metric": True,
                        "seed": 0,
                    }
                )
        fig = normalize_figure_spec(
            {
                "figure_id": "figure_power_vs_sinr_target",
                "required_sweep": "sinr_feasible_sweep",
                "chart_type": "line",
                "methods": ["proposed"],
                "metric": {"name": "sum_power_dBm"},
                "encoding": {
                    "x": {
                        "type": "numeric",
                        "field": "swept_value",
                        "sweep_param": "constraints.gamma_target",
                        "sweep_id": "sinr_feasible_sweep",
                    },
                    "group": {"field": "method"},
                },
            }
        )
        curve = aggregate_for_figure(pd.DataFrame(rows), fig)
        self.assertEqual(sorted(curve["x_value"].tolist()), [0.0, 3.0])

    def test_isolated_boundary_zero_variance_does_not_block_power_curve(self) -> None:
        rows = []
        x_values = [float(value) for value in range(10)]
        for method in ["proposed", "equal_power_heuristic"]:
            for x_value in x_values:
                for seed in range(80):
                    metric = 10.0 + x_value + 0.01 * seed
                    if method == "equal_power_heuristic" and x_value == 9.0:
                        metric = 30.0
                    rows.append(
                        {
                            "case_id": f"figure_power_vs_sinr_target_constraints.gamma_target_{x_value}",
                            "method": method,
                            "swept_param": "constraints.gamma_target",
                            "swept_value": x_value,
                            "seed": seed,
                            "status": "ok",
                            "feasible": True,
                            "sum_power_dBm": metric,
                            "finite_primary_metric": True,
                        }
                    )
        plan = {
            "primary_metric": {"name": "sum_power_dBm", "higher_is_better": False},
            "compared_methods": [{"name": "proposed"}, {"name": "equal_power_heuristic"}],
            "figure_specs": [
                {
                    "figure_id": "figure_power_vs_sinr_target",
                    "required_sweep": "sinr_feasible_sweep",
                    "chart_type": "line",
                    "methods": ["proposed", "equal_power_heuristic"],
                    "metric": {"name": "sum_power_dBm"},
                    "encoding": {
                        "x": {
                            "field": "swept_value",
                            "sweep_param": "constraints.gamma_target",
                            "sweep_id": "sinr_feasible_sweep",
                        },
                        "group": {"field": "method"},
                    },
                    "data_requirements": {"min_points": 10, "preferred_points": 10, "min_samples_per_group": 80, "preferred_samples_per_group": 100},
                }
            ],
        }
        mc_rows = []
        for method in ["proposed", "equal_power_heuristic"]:
            for x_value in x_values:
                warnings = []
                if method == "equal_power_heuristic" and x_value == 9.0:
                    warnings = ["repeated_identical_outputs_across_seeds", "zero_variance_across_all_seeds"]
                mc_rows.append(
                    {
                        "figure_id": "figure_power_vs_sinr_target",
                        "method": method,
                        "x_value": x_value,
                        "num_unique_seeds": 80,
                        "warnings": warnings,
                    }
                )
        mc_report = {
            "unknown_seed_coverage": False,
            "figures": [{"figure_id": "figure_power_vs_sinr_target", "rows": mc_rows}],
        }
        report = check_data_sufficiency(pd.DataFrame(rows), pd.DataFrame(), plan, mc_report, quick_mode=False)
        figure = report["figures"][0]
        self.assertTrue(figure["paper_ready"])
        self.assertNotIn("zero_variance_across_seeds", figure["blocking_issues"])
        self.assertIn("isolated_zero_variance_groups", figure["warnings"])

    def test_mechanism_line_accepts_fixed_ablation_zero_seed_variance(self) -> None:
        rows = []
        x_values = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0]
        for x_value in x_values:
            for seed in range(80):
                rows.append(
                    {
                        "case_id": f"figure_2_constraints.E_min_mW_{x_value}",
                        "method": "proposed",
                        "swept_param": "constraints.E_min_mW",
                        "swept_value": x_value,
                        "seed": seed,
                        "status": "ok",
                        "feasible": True,
                        "objective": 1.0,
                        "finite_primary_metric": True,
                        "optimal_rho": 0.1 + 0.02 * x_value + 0.001 * seed,
                    }
                )
                rows.append(
                    {
                        "case_id": f"figure_2_constraints.E_min_mW_{x_value}",
                        "method": "rho_fixed_half",
                        "swept_param": "constraints.E_min_mW",
                        "swept_value": x_value,
                        "seed": seed,
                        "status": "ok",
                        "feasible": True,
                        "objective": 1.0,
                        "finite_primary_metric": True,
                        "optimal_rho": 0.5,
                    }
                )
        df = pd.DataFrame(rows)
        plan = {
            "primary_metric": {"name": "objective", "higher_is_better": True},
            "compared_methods": [{"name": "proposed"}, {"name": "rho_fixed_half"}],
            "figure_specs": [
                normalize_figure_spec(
                    {
                        "figure_id": "figure_2",
                        "chart_intent": "sensitivity",
                        "chart_type": "box",
                        "methods": ["proposed", "rho_fixed_half"],
                        "metric": {"name": "optimal_rho"},
                        "encoding": {
                            "x": {"type": "numeric", "field": "swept_value", "sweep_param": "constraints.E_min_mW"},
                            "group": {"field": "method"},
                        },
                        "data_requirements": {"min_points": 7, "preferred_points": 7, "min_samples_per_group": 80, "preferred_samples_per_group": 100},
                    }
                )
            ],
        }
        mc_report = {
            "unknown_seed_coverage": False,
            "figures": [
                {
                    "figure_id": "figure_2",
                    "rows": [
                        {
                            "method": method,
                            "x_value": x_value,
                            "num_unique_seeds": 80,
                            "warnings": ["repeated_identical_outputs_across_seeds", "zero_variance_across_all_seeds"] if method == "rho_fixed_half" else [],
                        }
                        for method in ["proposed", "rho_fixed_half"]
                        for x_value in x_values
                    ],
                }
            ],
        }
        report = check_data_sufficiency(df, pd.DataFrame(), plan, mc_report, quick_mode=False)
        figure = report["figures"][0]
        self.assertEqual(figure["blocking_issues"], [])
        self.assertTrue(figure["deterministic_sweep_valid"])
        self.assertTrue(figure["paper_minimum_ready"])

    def test_paired_method_points_count_finite_plot_coverage(self) -> None:
        df = pd.DataFrame(
            {
                "method": ["proposed", "baseline", "proposed"],
                "swept_value": [0.0, 0.0, 1.0],
                "Ps_dB": [1.0, 0.5, 2.0],
            }
        )
        self.assertEqual(_paired_method_x_points(df, x_field="swept_value", methods_required=["proposed", "baseline"]), 1)

    def test_comparison_uses_declared_baseline_when_method_not_named_baseline(self) -> None:
        df = pd.DataFrame(
            {
                "case_id": ["c0", "c0"],
                "case_name": ["c0", "c0"],
                "swept_param": ["system.Pmax", "system.Pmax"],
                "swept_value": [1.0, 1.0],
                "scenario_name": ["power", "power"],
                "seed": [0, 0],
                "method": ["proposed", "fixed_phase"],
                "status": ["ok", "ok"],
                "feasible": [True, True],
                "objective": [2.0, 1.0],
                "success": [True, True],
                "finite_primary_metric": [True, True],
            }
        )
        plan = {
            "primary_metric": {"name": "objective", "higher_is_better": True},
            "compared_methods": [
                {"name": "proposed", "role": "proposed"},
                {"name": "fixed_phase", "role": "main_baseline"},
            ],
        }
        comparison = build_per_case_comparison(df, plan)
        self.assertFalse(comparison.empty)
        self.assertEqual(comparison["baseline_method"].iloc[0], "fixed_phase")
        self.assertAlmostEqual(float(comparison["relative_gain"].iloc[0]), 1.0)

    def test_comparison_prefers_practical_heuristic_over_optimal_reference(self) -> None:
        df = pd.DataFrame(
            {
                "case_id": ["c0", "c0", "c0"],
                "case_name": ["c0", "c0", "c0"],
                "swept_param": ["constraints.gamma_target"] * 3,
                "swept_value": [1.0, 1.0, 1.0],
                "scenario_name": ["gamma"] * 3,
                "seed": [0, 0, 0],
                "method": ["proposed", "centralized_lp", "equal_power_heuristic"],
                "status": ["ok", "ok", "ok"],
                "feasible": [True, True, True],
                "sum_power_W": [1.0, 1.0, 2.0],
                "success": [True, True, True],
                "finite_primary_metric": [True, True, True],
            }
        )
        plan = {
            "primary_metric": {"name": "sum_power_W", "higher_is_better": False},
            "compared_methods": [
                {"name": "proposed", "role": "proposed"},
                {"name": "centralized_lp", "role": "main_baseline", "display_name_long": "Centralized LP Optimal"},
                {"name": "equal_power_heuristic", "role": "heuristic", "display_name_short": "EQ-Power"},
            ],
        }

        self.assertEqual(select_comparison_baseline_method(df, plan), "equal_power_heuristic")
        comparison = build_per_case_comparison(df, plan)

        self.assertEqual(comparison["baseline_method"].iloc[0], "equal_power_heuristic")
        self.assertAlmostEqual(float(comparison["relative_gain"].iloc[0]), 0.5)

    def test_comparison_uses_gain_showing_claim_baseline_and_keeps_strongest_audit(self) -> None:
        df = pd.DataFrame(
            {
                "case_id": ["c0"] * 5,
                "case_name": ["c0"] * 5,
                "swept_param": ["system.Pmax"] * 5,
                "swept_value": [1.0] * 5,
                "scenario_name": ["power"] * 5,
                "seed": [0] * 5,
                "method": ["proposed", "fixed_ps_baseline", "linear_eh_baseline", "mrt_split_baseline", "zf_split_baseline"],
                "status": ["ok"] * 5,
                "feasible": [True] * 5,
                "sum_rate_bpsHz": [10.0, 7.0, 10.0, 3.0, 11.0],
                "success": [True] * 5,
                "finite_primary_metric": [True] * 5,
            }
        )
        plan = {
            "primary_metric": {"name": "sum_rate_bpsHz", "higher_is_better": True},
            "compared_methods": [
                {"name": "proposed", "role": "proposed"},
                {"name": "fixed_ps_baseline", "role": "main_baseline", "display_priority": 1},
                {"name": "linear_eh_baseline", "role": "model_diagnostic", "display_priority": 4},
                {"name": "mrt_split_baseline", "role": "direction_heuristic", "display_priority": 2},
                {"name": "zf_split_baseline", "role": "direction_heuristic", "display_priority": 3},
            ],
            "figure_specs": [
                {"figure_id": "figure_1", "methods": ["proposed", "fixed_ps_baseline", "linear_eh_baseline", "mrt_split_baseline", "zf_split_baseline"], "metric": {"name": "sum_rate_bpsHz"}},
            ],
        }

        self.assertEqual(select_comparison_baseline_method(df, plan), "fixed_ps_baseline")
        self.assertEqual(select_strongest_practical_baseline_method(df, plan), "zf_split_baseline")

    def test_single_benchmark_selection_skips_invalid_and_redundant_methods(self) -> None:
        rows = []
        for figure_id, swept_param in [("figure_1", "requirements.gamma_dB"), ("figure_2", "requirements.E_min_mW")]:
            for x in [1.0, 2.0, 3.0]:
                rows.append(
                    {
                        "figure_id": figure_id,
                        "swept_param": swept_param,
                        "swept_value": x,
                        "method": "proposed",
                        "P_tx_mW": x,
                        "feasible": True,
                        "success": True,
                    }
                )
                rows.append(
                    {
                        "figure_id": figure_id,
                        "swept_param": swept_param,
                        "swept_value": x,
                        "method": "no_shared_covariance_baseline",
                        "P_tx_mW": x + 1.0,
                        "feasible": True,
                        "success": True,
                    }
                )
                rows.append(
                    {
                        "figure_id": figure_id,
                        "swept_param": swept_param,
                        "swept_value": x,
                        "method": "linear_eh_baseline",
                        "P_tx_mW": x,
                        "feasible": True,
                        "success": True,
                    }
                )
                rows.append(
                    {
                        "figure_id": figure_id,
                        "swept_param": swept_param,
                        "swept_value": x,
                        "method": "fixed_ps_baseline",
                        "P_tx_mW": 1.0e9,
                        "feasible": False,
                        "success": False,
                    }
                )
        df = pd.DataFrame(rows)
        plan = {
            "primary_metric": {"name": "P_tx_mW", "higher_is_better": False},
            "compared_methods": [
                {"name": "proposed", "role": "proposed"},
                {
                    "name": "fixed_ps_baseline",
                    "role": "main_baseline",
                    "implementation_hint": "No power-splitting control exists in the frozen model; mark unavailable.",
                },
                {
                    "name": "linear_eh_baseline",
                    "role": "model_diagnostic",
                    "implementation_hint": "Redundant with proposed under the frozen linear EH model; do not plot as a distinct curve.",
                },
                {
                    "name": "no_shared_covariance_baseline",
                    "role": "main_baseline",
                    "implementation_hint": "Disable the shared covariance flexibility while preserving the same evaluator.",
                },
            ],
            "figure_specs": [
                {
                    "figure_id": "figure_1",
                    "chart_intent": "main_comparison",
                    "methods": ["proposed", "fixed_ps_baseline", "linear_eh_baseline"],
                    "metric": {"name": "P_tx_mW", "higher_is_better": False},
                    "encoding": {"x": {"field": "swept_value", "sweep_param": "requirements.gamma_dB"}},
                },
                {
                    "figure_id": "figure_2",
                    "chart_intent": "energy_service_sensitivity",
                    "methods": ["proposed", "fixed_ps_baseline", "linear_eh_baseline"],
                    "metric": {"name": "P_tx_mW", "higher_is_better": False},
                    "encoding": {"x": {"field": "swept_value", "sweep_param": "requirements.E_min_mW"}},
                },
            ],
        }

        repaired = select_single_benchmark_methods_for_figures(plan, df)

        self.assertEqual(repaired["figure_specs"][0]["methods"], ["proposed", "no_shared_covariance_baseline"])
        self.assertEqual(repaired["figure_specs"][1]["methods"], ["proposed", "no_shared_covariance_baseline"])
        self.assertEqual(
            repaired["figure_specs"][0]["benchmark_selection"]["common_benchmark_alignment"]["reason"],
            "same_viable_benchmark_set_recorded_across_paper_figures",
        )

    def test_final_figure_retains_all_gain_supporting_practical_baselines(self) -> None:
        rows = []
        for figure_id, swept_param in [("figure_1", "users.angular_separation_deg"), ("figure_2", "users.radial_separation_m")]:
            for x in [1.0, 2.0, 3.0, 4.0]:
                for method, value in [
                    ("proposed", 10.0 + x),
                    ("strong_baseline", 8.0 + 0.4 * x),
                    ("weak_baseline", 3.0 + 0.2 * x),
                ]:
                    rows.append(
                        {
                            "figure_id": figure_id,
                            "swept_param": swept_param,
                            "swept_value": x,
                            "method": method,
                            "sum_rate_bpsHz": value,
                            "feasible": True,
                            "success": True,
                            "finite_primary_metric": True,
                        }
                    )
        df = pd.DataFrame(rows)
        plan = {
            "primary_metric": {"name": "sum_rate_bpsHz", "higher_is_better": True},
            "compared_methods": [
                {"name": "proposed", "role": "proposed"},
                {"name": "strong_baseline", "role": "practical_heuristic", "display_priority": 1},
                {"name": "weak_baseline", "role": "practical_heuristic", "display_priority": 2},
            ],
            "figure_specs": [
                {
                    "figure_id": "figure_1",
                    "chart_intent": "main_comparison",
                    "methods": ["proposed", "strong_baseline", "weak_baseline"],
                    "metric": {"name": "sum_rate_bpsHz", "higher_is_better": True},
                    "encoding": {"x": {"field": "swept_value", "sweep_param": "users.angular_separation_deg"}},
                },
                {
                    "figure_id": "figure_2",
                    "chart_intent": "stress_or_gain",
                    "methods": ["proposed", "strong_baseline", "weak_baseline"],
                    "metric": {"name": "sum_rate_bpsHz", "higher_is_better": True},
                    "encoding": {"x": {"field": "swept_value", "sweep_param": "users.radial_separation_m"}},
                },
            ],
        }

        repaired = select_single_benchmark_methods_for_figures(plan, df)

        self.assertEqual(
            repaired["figure_specs"][0]["methods"],
            ["proposed", "strong_baseline", "weak_baseline"],
        )
        self.assertEqual(
            repaired["figure_specs"][1]["methods"],
            ["proposed", "strong_baseline", "weak_baseline"],
        )
        self.assertEqual(
            repaired["figure_specs"][0]["benchmark_selection"]["selected_benchmarks"],
            ["strong_baseline", "weak_baseline"],
        )

    def test_single_benchmark_selection_skips_degenerate_near_zero_baseline(self) -> None:
        rows = []
        for figure_id, swept_param in [("figure_1", "uncertainty.delta"), ("figure_2", "channel.eve_gain")]:
            for x in [1.0, 2.0, 3.0]:
                rows.extend(
                    [
                        {
                            "figure_id": figure_id,
                            "swept_param": swept_param,
                            "swept_value": x,
                            "method": "proposed",
                            "worst_case_min_secrecy_rate_bpsHz": 2.0 + x,
                            "feasible": True,
                            "success": True,
                        },
                        {
                            "figure_id": figure_id,
                            "swept_param": swept_param,
                            "swept_value": x,
                            "method": "zero_benchmark",
                            "worst_case_min_secrecy_rate_bpsHz": 0.0,
                            "feasible": True,
                            "success": True,
                        },
                        {
                            "figure_id": figure_id,
                            "swept_param": swept_param,
                            "swept_value": x,
                            "method": "usable_benchmark",
                            "worst_case_min_secrecy_rate_bpsHz": 1.0 + 0.2 * x,
                            "feasible": True,
                            "success": True,
                        },
                    ]
                )
        df = pd.DataFrame(rows)
        plan = {
            "primary_metric": {"name": "worst_case_min_secrecy_rate_bpsHz", "higher_is_better": True},
            "compared_methods": [
                {"name": "proposed", "role": "proposed"},
                {"name": "zero_benchmark", "role": "main_baseline"},
                {"name": "usable_benchmark", "role": "main_baseline"},
            ],
            "figure_specs": [
                {
                    "figure_id": "figure_1",
                    "chart_intent": "main_comparison",
                    "methods": ["proposed", "zero_benchmark", "usable_benchmark"],
                    "metric": {"name": "worst_case_min_secrecy_rate_bpsHz", "higher_is_better": True},
                    "encoding": {"x": {"field": "swept_value", "sweep_param": "uncertainty.delta"}},
                },
                {
                    "figure_id": "figure_2",
                    "chart_intent": "stress_or_gain",
                    "methods": ["proposed", "zero_benchmark", "usable_benchmark"],
                    "metric": {"name": "worst_case_min_secrecy_rate_bpsHz", "higher_is_better": True},
                    "encoding": {"x": {"field": "swept_value", "sweep_param": "channel.eve_gain"}},
                },
            ],
        }

        repaired = select_single_benchmark_methods_for_figures(plan, df)

        self.assertEqual(repaired["figure_specs"][0]["methods"], ["proposed", "usable_benchmark"])
        self.assertEqual(repaired["figure_specs"][1]["methods"], ["proposed", "usable_benchmark"])
        candidate_scores = repaired["figure_specs"][0]["benchmark_selection"]["candidate_scores"]
        zero_score = next(item for item in candidate_scores if item["method"] == "zero_benchmark")
        self.assertTrue(zero_score["benchmark_degeneracy"]["degenerate"])

    def test_no_oracle_fairness_text_does_not_make_heuristic_an_optimal_reference(self) -> None:
        comparison = pd.DataFrame(
            {
                "relative_gain": [0.25, 0.30],
                "proposed_win": [True, True],
                "baseline_method": ["equal_power_heuristic", "equal_power_heuristic"],
                "comparable": [True, True],
            }
        )
        plan = {
            "_active_baseline_method": "equal_power_heuristic",
            "compared_methods": [
                {"name": "proposed", "role": "proposed"},
                {
                    "name": "equal_power_heuristic",
                    "role": "heuristic",
                    "display_name_short": "EQ-Power",
                    "fairness_rule": "No oracle channel optimization beyond using the same feasibility evaluation.",
                },
            ],
        }
        check = evaluate_primary_claim_check(comparison, plan)

        self.assertEqual(check["mode"], "advantage_over_benchmark")
        self.assertTrue(check["passes"])

    def test_safe_display_repairs_plain_axis_subscripts(self) -> None:
        self.assertEqual(_safe_display("CSI uncertainty radius r_CSI"), r"CSI uncertainty radius $r_{\mathrm{CSI}}$")
        self.assertEqual(_safe_display("Eve channel gain factor eta_e"), r"Eve channel gain factor $\eta_e$")
        self.assertEqual(_safe_display(r"Eve channel gain factor $\eta_e$"), r"Eve channel gain factor $\eta_e$")
        self.assertEqual(_safe_display(r"Eve channel gain factor $\\\\eta_e$"), r"Eve channel gain factor $\eta_e$")

    def test_figure_level_claim_uses_aggregated_monte_carlo_curves(self) -> None:
        rows = []
        for x_value in [1.0, 2.0, 3.0]:
            for seed, baseline_value in enumerate([0.0, 0.0, 10.0, 10.0, 10.0]):
                case_id = f"x{x_value}_seed{seed}"
                rows.append(
                    {
                        "case_id": case_id,
                        "case_name": case_id,
                        "figure_id": "figure_1",
                        "scenario_name": "power",
                        "swept_param": "system.Pmax",
                        "swept_value": x_value,
                        "seed": seed,
                        "method": "proposed",
                        "status": "ok",
                        "feasible": True,
                        "success": True,
                        "finite_primary_metric": True,
                        "sum_rate_bpsHz": 7.0 + 0.1 * x_value,
                    }
                )
                rows.append(
                    {
                        "case_id": case_id,
                        "case_name": case_id,
                        "figure_id": "figure_1",
                        "scenario_name": "power",
                        "swept_param": "system.Pmax",
                        "swept_value": x_value,
                        "seed": seed,
                        "method": "mrt",
                        "status": "ok",
                        "feasible": True,
                        "success": True,
                        "finite_primary_metric": True,
                        "sum_rate_bpsHz": baseline_value,
                    }
                )
        df = pd.DataFrame(rows)
        plan = {
            "primary_metric": {"name": "sum_rate_bpsHz", "higher_is_better": True},
            "_active_baseline_method": "mrt",
            "compared_methods": [
                {"name": "proposed", "role": "proposed"},
                {"name": "mrt", "role": "main_baseline"},
            ],
            "figure_specs": [
                {
                    "figure_id": "figure_1",
                    "chart_intent": "main_comparison",
                    "chart_type": "line",
                    "methods": ["proposed", "mrt"],
                    "metric": {"name": "sum_rate_bpsHz", "higher_is_better": True},
                    "encoding": {"x": {"field": "swept_value", "sweep_param": "system.Pmax"}},
                }
            ],
        }

        row_level = evaluate_primary_claim_check(build_per_case_comparison(df, plan), plan)
        figure_level = evaluate_figure_level_primary_claim_check(df, plan, baseline_method="mrt")

        self.assertFalse(row_level["passes"])
        self.assertEqual(row_level["proposed_win_rate"], 0.4)
        self.assertTrue(figure_level["passes"], figure_level)
        self.assertEqual(figure_level["mode"], "figure_level_aggregate_advantage")

    def test_primary_claim_rejects_degenerate_near_zero_practical_baseline(self) -> None:
        comparison = pd.DataFrame(
            {
                "relative_gain": [4.0e10, 3.5e10],
                "proposed_win": [True, True],
                "baseline_method": ["regularized_zf_heuristic", "regularized_zf_heuristic"],
                "comparable": [True, True],
                "proposed_sum_rate_bpsHz": [40.0, 35.0],
                "baseline_sum_rate_bpsHz": [0.0, 1.0e-10],
            }
        )
        plan = {
            "primary_metric": {"name": "sum_rate_bpsHz", "higher_is_better": True},
            "_active_baseline_method": "regularized_zf_heuristic",
            "compared_methods": [
                {"name": "proposed", "role": "proposed"},
                {"name": "regularized_zf_heuristic", "role": "main_baseline"},
            ],
        }

        check = evaluate_primary_claim_check(comparison, plan)

        self.assertFalse(check["passes"])
        self.assertTrue(check["baseline_degenerate"])
        self.assertEqual(check["reason"], "practical_benchmark_metric_is_degenerate_near_zero")

    def test_optimal_reference_equivalence_passes_without_positive_gain(self) -> None:
        df = pd.DataFrame(
            {
                "case_id": ["c0", "c0"],
                "case_name": ["c0", "c0"],
                "swept_param": ["constraints.gamma_target", "constraints.gamma_target"],
                "swept_value": [1.0, 1.0],
                "scenario_name": ["gamma", "gamma"],
                "seed": [0, 0],
                "method": ["proposed", "centralized_lp"],
                "status": ["ok", "ok"],
                "feasible": [True, True],
                "sum_power_W": [1.0, 1.0],
                "success": [True, True],
                "finite_primary_metric": [True, True],
            }
        )
        plan = {
            "primary_metric": {"name": "sum_power_W", "higher_is_better": False},
            "compared_methods": [
                {"name": "proposed", "role": "proposed"},
                {"name": "centralized_lp", "role": "main_baseline", "display_name_long": "Centralized LP Optimal"},
            ],
        }
        comparison = build_per_case_comparison(df, plan)
        plan["_active_baseline_method"] = str(comparison["baseline_method"].iloc[0])
        check = evaluate_primary_claim_check(comparison[comparison["comparable"]], plan)

        self.assertEqual(check["mode"], "optimal_reference_equivalence")
        self.assertTrue(check["passes"])

    def test_comparison_treats_solver_optimal_status_as_success(self) -> None:
        df = pd.DataFrame(
            {
                "case_id": ["c0", "c0"],
                "case_name": ["c0", "c0"],
                "swept_param": ["system.Pmax", "system.Pmax"],
                "swept_value": [1.0, 1.0],
                "scenario_name": ["power", "power"],
                "seed": [0, 0],
                "method": ["proposed", "fixed_phase"],
                "status": ["optimal", "optimal"],
                "feasible": [True, True],
                "objective": [2.0, 1.0],
                "success": [False, False],
                "finite_primary_metric": [True, True],
            }
        )
        plan = {
            "primary_metric": {"name": "objective", "higher_is_better": True},
            "compared_methods": [
                {"name": "proposed", "role": "proposed"},
                {"name": "fixed_phase", "role": "main_baseline"},
            ],
        }

        comparison = build_per_case_comparison(df, plan)

        self.assertTrue(bool(comparison["comparable"].iloc[0]))
        self.assertTrue(bool(comparison["proposed_win"].iloc[0]))

    def test_phase25_ignores_paper_results_older_than_generated_plugin(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            solver_dir = run_dir / "phase2-4" / "solver"
            outputs_dir = solver_dir / "outputs"
            outputs_dir.mkdir(parents=True)
            (outputs_dir / "validation_summary.json").write_text(json.dumps({"quick_mode": True}), encoding="utf-8")
            (outputs_dir / "validation_results.csv").write_text(
                "\n".join(
                    [
                        "case_id,method,status,objective,feasible",
                        "quick_case,proposed,ok,1.0,True",
                    ]
                ),
                encoding="utf-8",
            )
            (outputs_dir / "paper_validation_summary.json").write_text(json.dumps({"quick_mode": False}), encoding="utf-8")
            stale_paper = outputs_dir / "paper_validation_results.csv"
            stale_paper.write_text(
                "\n".join(
                    [
                        "case_id,method,status,objective,feasible,rho,M_eh",
                        "paper_case,no_rho,ok,2.0,True,0.5,16",
                    ]
                ),
                encoding="utf-8",
            )
            old_time = time.time() - 100.0
            os.utime(stale_paper, (old_time, old_time))
            os.utime(outputs_dir / "paper_validation_summary.json", (old_time, old_time))
            plugin_path = solver_dir / "generated_plugin.py"
            plugin_path.write_text("# current plugin marker\n", encoding="utf-8")

            summary, df = load_phase24_results(run_dir)

        self.assertEqual(summary["_phase25_data_source"], "quick_validation")
        self.assertTrue(summary["_phase25_paper_validation_staleness"]["is_stale"])
        self.assertTrue(summary["_phase25_ignored_paper_validation"]["is_stale"])
        self.assertEqual(df["case_id"].tolist(), ["quick_case"])

    def test_render_table_marks_percent_values_to_prevent_double_scaling(self) -> None:
        comparison = pd.DataFrame(
            {
                "swept_value": [1.0],
                "comparable": [True],
                "relative_gain": [0.0134],
                "proposed_objective": [2.0],
                "baseline_objective": [1.0],
                "proposed_feasible": [True],
                "baseline_feasible": [True],
            }
        )
        plan = {
            "primary_metric": {"name": "objective", "display_name": "Objective"},
            "compared_methods": [
                {"name": "proposed", "role": "proposed"},
                {"name": "fixed_phase", "role": "main_baseline", "display_name_short": "Fixed"},
            ],
            "_active_baseline_method": "fixed_phase",
        }
        table_spec = {
            "table_id": "table_1",
            "group_by": "swept_value",
            "columns": ["scenario", "relative_gain_percent"],
        }
        with tempfile.TemporaryDirectory() as tmp:
            output = render_table(comparison, table_spec, Path(tmp), plan=plan)
            csv_text = Path(output["csv_path"]).read_text(encoding="utf-8-sig")

        self.assertIn("1.34%", csv_text)

    def test_evidence_table_keeps_ablation_sweeps_without_pairwise_baseline(self) -> None:
        df = pd.DataFrame(
            {
                "swept_param": ["optimization.lambda_s", "optimization.lambda_s", "EH.steepness_a", "EH.steepness_a"],
                "method": ["proposed", "fixed_phase", "proposed", "linear_eh"],
                "feasible": [True, True, True, True],
                "sum_rate_bps_hz": [10.0, 9.0, 8.0, 7.0],
                "sum_rate_bpsHz": [10.0, 9.0, 8.0, 7.0],
                "radar_snr_dB": [20.0, 18.0, 19.0, 17.0],
                "true_harvested_energy_mW": [3.0, 2.5, 4.0, 3.2],
                "optimal_rho": [0.3, 0.5, 0.4, 0.0],
            }
        )
        plan = {
            "primary_metric": {"name": "objective", "display_name": "Objective"},
            "compared_methods": [
                {"name": "proposed", "internal_name": "proposed", "role": "proposed", "display_name_short": "Proposed"},
                {"name": "fixed_phase", "internal_name": "fixed_phase", "role": "main_baseline", "display_name_short": "Fixed"},
                {"name": "linear_eh", "internal_name": "linear_eh", "role": "model_ablation", "display_name_short": "Linear-EH"},
            ],
            "figure_specs": [
                {"figure_id": "figure_1", "methods": ["proposed", "fixed_phase"], "metric": {"name": "radar_snr_dB"}},
                {"figure_id": "figure_2", "methods": ["proposed", "linear_eh"], "metric": {"name": "true_harvested_energy_mW"}},
            ],
        }
        table_spec = {
            "table_id": "table_1",
            "row_selection": "method x evidence_sweep",
            "columns": ["sum_rate_bps_hz", "sum_rate_bpsHz", "radar_snr_dB", "true_harvested_energy_mW", "optimal_rho"],
        }
        with tempfile.TemporaryDirectory() as tmp:
            meta = render_evidence_table(df, table_spec, Path(tmp), plan=plan)
            csv_text = Path(meta["csv_path"]).read_text(encoding="utf-8-sig")

        self.assertEqual(meta["num_rows"], 4)
        sum_rate_columns = [col for col in meta["display_columns"] if "Sum" in col and "bps" in col]
        self.assertEqual(len(sum_rate_columns), 1)
        self.assertIn("EH steepness", csv_text)
        self.assertIn("Linear-EH", csv_text)

    def test_mechanism_figure_metric_repair_prefers_responsive_physical_kpi(self) -> None:
        rows = []
        for x, rate, harvested in [
            (0.0, 27.1512, 22.5),
            (1.0, 27.1512, 22.8),
            (2.0, 27.1510, 23.0),
            (3.0, 27.1503, 24.0),
            (4.0, 27.1479, 27.0),
            (6.0, 27.1398, 34.8),
            (8.0, 27.1312, 39.7),
            (10.0, 27.0228, 40.0),
        ]:
            rows.append(
                {
                    "case_id": f"figure_sum_rate_vs_eh_requirement_{x}",
                    "figure_id": "figure_sum_rate_vs_eh_requirement",
                    "required_sweep": "eh_requirement_sweep",
                    "swept_param": "constraints.E_min_mW",
                    "swept_value": x,
                    "method": "proposed",
                    "feasible": True,
                    "sum_rate_bpsHz": rate,
                    "sum_harvested_energy_mW": harvested,
                    "min_harvested_energy_mW": harvested / 4.0,
                    "average_rho": 0.999 - 0.002 * x,
                }
            )
        plan = {
            "primary_metric": {"name": "sum_rate_bpsHz", "display_name": "Sum rate"},
            "compared_methods": [{"name": "proposed", "role": "proposed"}],
            "figure_specs": [
                {
                    "figure_id": "figure_sum_rate_vs_eh_requirement",
                    "chart_intent": "mechanism_ablation",
                    "required_sweep": "eh_requirement_sweep",
                    "metric": {"name": "sum_rate_bpsHz"},
                    "encoding": {"x": {"field": "swept_value", "sweep_param": "constraints.E_min_mW"}},
                    "methods": ["proposed"],
                }
            ],
        }

        repaired = repair_mechanism_figure_metrics_from_data(plan, pd.DataFrame(rows))

        metric = repaired["figure_specs"][0]["metric"]["name"]
        self.assertIn(metric, {"sum_harvested_energy_mW", "min_harvested_energy_mW"})
        self.assertIn("phase25_metric_repair", repaired["figure_specs"][0])


if __name__ == "__main__":
    unittest.main()
