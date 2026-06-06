from __future__ import annotations

import json
import importlib.util
import os
import tempfile
import textwrap
import unittest
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "phase2" / "scripts"

import sys

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from phase_runtime_impl import merge_phase24_method_solution_branches, normalize_phase24_generated_plugin_source, validate_phase2_phase24_plugin_bundle, validate_phase2_phase24_plugin_interfaces, validate_phase2_phase24_schema_alignment, validate_phase24_algorithm_code_contract, validate_phase24_basic_evidence_quality, validate_phase24_evidence_contract_design, validate_phase24_evidence_contract_outputs, validate_phase24_experiment_responsiveness, validate_phase24_pilot_gain, validate_phase24_method_semantics, _phase24_numerical_runtime_warning_report, _phase24_validation_allows_repair, write_phase2_phase24_fixed_harness, write_phase24_split_plugin_package
from phase_runtime.phase24_codegen import PHASE24_SPLIT_ADAPTER_VERSION, build_phase24_split_plugin_adapter, preserve_phase24_required_exports
from phase_runtime.phase24_validation import _phase24_iteration_cap_mismatch_errors, _validate_phase24_split_codegen_package
from phase_runtime.phase24_plugin_generation import _phase24_method_fidelity_contract
from phase_runtime.phase24_plan import _phase24_publication_metric_for_target
from continue_phase2_from_phase24 import _load_required_mathematical_contract_json, _phase24_repair_round_limit, _phase24_design_repair_round_limit, _phase24_inactive_axis_directive_for_feedback, _phase25_auto_paper_run_limit, _phase25_impl_repair_round_limit, _phase25_refiner_requires_phase24_design_revision, _phase25_objective_metric_alignment_precheck, _phase25_design_revision_feedback_for_phase24


class Phase24SchemaHarnessTests(unittest.TestCase):
    def test_phase24_method_fidelity_contract_preserves_solver_route(self) -> None:
        contract = _phase24_method_fidelity_contract(
            algorithm_md=(
                "The proposed method alternates an RIS phase update with an SDP/SDR "
                "subproblem, CVXPY implementation, rank-one recovery from the leading "
                "eigenvector, and SCA linearization."
            ),
            phase24_execution_contract={
                "algorithm_family": "sdp_or_sdr",
                "algorithm_execution_contract": {"update_blocks": ["proposed_algorithm_update"]},
            },
        )

        self.assertTrue(contract["route_requires_cvxpy_solver_path"])
        self.assertTrue(contract["route_uses_rank_or_recovery_logic"])
        self.assertTrue(contract["route_uses_surface_phase_or_unit_modulus_update"])
        self.assertIn("cp.Problem", contract["required_solver_code_markers"])
        self.assertIn("used_theta_update", contract["required_update_diagnostics"])

    def test_phase24_method_fidelity_contract_does_not_treat_ris_phase_as_position(self) -> None:
        contract = _phase24_method_fidelity_contract(
            algorithm_md=(
                "The RIS phase block solves a lifted SDP and recovers a unit-modulus "
                "reflection vector. The text mentions post-recovery position in prose "
                "but no trajectory, deployment, mobility, or movable-antenna control is optimized."
            ),
            phase24_execution_contract={"algorithm_family": "sdp_or_sdr"},
        )

        self.assertTrue(contract["route_uses_surface_phase_or_unit_modulus_update"])
        self.assertFalse(contract["route_uses_position_or_deployment_updates"])
        self.assertNotIn("used_position_update", contract["required_update_diagnostics"])

    def test_phase24_codegen_prompt_defaults_do_not_encourage_numpy_proxy(self) -> None:
        source = (SCRIPTS_DIR / "phase_runtime" / "phase24_plugin_generation.py").read_text(encoding="utf-8")

        self.assertIn("Prioritize method fidelity over line count", source)
        self.assertIn("method_fidelity_contract.route_requires_cvxpy_solver_path", source)
        self.assertNotIn("Use compact helper functions and simple numpy formulas", source)

    def test_phase24_figure_metric_aligns_with_y_axis_kpi_context(self) -> None:
        metric = _phase24_publication_metric_for_target(
            {
                "y_metric": "sum_rate_bpsHz",
                "axis_labels": {"y": "minimum harvested DC power $\\min_l P_l^{\\mathrm{DC}}$ (mW)"},
                "intended_insight": "Audit the nonlinear rectifier operating regime.",
                "required_metrics": ["sum_rate_bpsHz", "min_harvested_dc_mW", "harvested_energy_mW"],
            },
            ["sum_rate_bpsHz", "min_harvested_dc_mW", "harvested_energy_mW"],
            1,
            {"id": "rectifier_turn_on_sweep", "canonical_path": "rectifier.turn_on_b_mW"},
        )

        self.assertEqual(metric, "min_harvested_dc_mW")

    def test_phase24_figure_metric_keeps_rate_when_y_axis_is_rate(self) -> None:
        metric = _phase24_publication_metric_for_target(
            {
                "y_metric": "sum_rate_bpsHz",
                "axis_labels": {"y": "sum rate $R_{\\mathrm{sum}}$ (bps/Hz)"},
                "claim": "Compare rate under nonlinear energy harvesting and sensing constraints.",
                "required_metrics": ["sum_rate_bpsHz", "min_harvested_dc_mW", "harvested_energy_mW"],
            },
            ["sum_rate_bpsHz", "min_harvested_dc_mW", "harvested_energy_mW"],
            0,
            {"id": "sinr_target_sweep", "canonical_path": "requirements.gamma_dB"},
        )

        self.assertEqual(metric, "sum_rate_bpsHz")

    def test_phase24_figure_metric_keeps_rate_under_sinr_sweep_when_axis_is_rate(self) -> None:
        metric = _phase24_publication_metric_for_target(
            {
                "y_metric": "sum_rate_bpsHz",
                "axis_labels": {"y": "sum rate $R_{\\mathrm{sum}}$ (bps/Hz)"},
                "trend_hypothesis": "Increasing the SINR target should expose the rate-energy tradeoff.",
                "required_sweep": "sinr_target_sweep",
                "required_sweep_param": "requirements.gamma_dB",
                "required_metrics": ["sum_rate_bpsHz", "objective", "min_harvested_dc_mW"],
            },
            ["sum_rate_bpsHz", "objective", "min_harvested_dc_mW"],
            0,
            {"id": "sinr_target_sweep", "canonical_path": "requirements.gamma_dB"},
        )

        self.assertEqual(metric, "sum_rate_bpsHz")

    def test_compact_diagnostics_maps_common_position_update_aliases(self) -> None:
        source = (SCRIPTS_DIR / "phase_runtime" / "phase24_validation.py").read_text(encoding="utf-8")

        self.assertIn('"used_p_update"', source)
        self.assertIn('"p_update_norm"', source)
        self.assertIn('"delta_objective"', source)

    def test_continue_phase24_uses_shared_repair_round_limit_env(self) -> None:
        old_wara = os.environ.get("WARA_PHASE24_REPAIR_ROUNDS")
        old_wcl = os.environ.get("WCL_PHASE24_REPAIR_ROUNDS")
        try:
            os.environ.pop("WARA_PHASE24_REPAIR_ROUNDS", None)
            os.environ.pop("WCL_PHASE24_REPAIR_ROUNDS", None)
            self.assertEqual(_phase24_repair_round_limit(), 10)
            os.environ["WCL_PHASE24_REPAIR_ROUNDS"] = "4"
            self.assertEqual(_phase24_repair_round_limit(), 4)
            os.environ["WARA_PHASE24_REPAIR_ROUNDS"] = "1"
            self.assertEqual(_phase24_repair_round_limit(), 1)
        finally:
            if old_wara is None:
                os.environ.pop("WARA_PHASE24_REPAIR_ROUNDS", None)
            else:
                os.environ["WARA_PHASE24_REPAIR_ROUNDS"] = old_wara
            if old_wcl is None:
                os.environ.pop("WCL_PHASE24_REPAIR_ROUNDS", None)
            else:
                os.environ["WCL_PHASE24_REPAIR_ROUNDS"] = old_wcl

    def test_continue_phase24_defaults_phase25_auto_expansion_to_medium(self) -> None:
        old_wara = os.environ.get("WARA_PHASE25_AUTO_PAPER_RUNS")
        old_wcl = os.environ.get("WCL_PHASE25_AUTO_PAPER_RUNS")
        try:
            os.environ.pop("WARA_PHASE25_AUTO_PAPER_RUNS", None)
            os.environ.pop("WCL_PHASE25_AUTO_PAPER_RUNS", None)
            self.assertEqual(_phase25_auto_paper_run_limit(), 10)
            os.environ["WCL_PHASE25_AUTO_PAPER_RUNS"] = "2"
            self.assertEqual(_phase25_auto_paper_run_limit(), 2)
            os.environ["WARA_PHASE25_AUTO_PAPER_RUNS"] = "1"
            self.assertEqual(_phase25_auto_paper_run_limit(), 1)
        finally:
            if old_wara is None:
                os.environ.pop("WARA_PHASE25_AUTO_PAPER_RUNS", None)
            else:
                os.environ["WARA_PHASE25_AUTO_PAPER_RUNS"] = old_wara
            if old_wcl is None:
                os.environ.pop("WCL_PHASE25_AUTO_PAPER_RUNS", None)
            else:
                os.environ["WCL_PHASE25_AUTO_PAPER_RUNS"] = old_wcl

    def test_continue_phase24_defaults_design_repair_to_ten_rounds(self) -> None:
        old_wara = os.environ.get("WARA_PHASE24_DESIGN_REPAIR_ROUNDS")
        old_wcl = os.environ.get("WCL_PHASE24_DESIGN_REPAIR_ROUNDS")
        try:
            os.environ.pop("WARA_PHASE24_DESIGN_REPAIR_ROUNDS", None)
            os.environ.pop("WCL_PHASE24_DESIGN_REPAIR_ROUNDS", None)
            self.assertEqual(_phase24_design_repair_round_limit(), 10)
            os.environ["WCL_PHASE24_DESIGN_REPAIR_ROUNDS"] = "2"
            self.assertEqual(_phase24_design_repair_round_limit(), 2)
            os.environ["WARA_PHASE24_DESIGN_REPAIR_ROUNDS"] = "0"
            self.assertEqual(_phase24_design_repair_round_limit(), 0)
        finally:
            if old_wara is None:
                os.environ.pop("WARA_PHASE24_DESIGN_REPAIR_ROUNDS", None)
            else:
                os.environ["WARA_PHASE24_DESIGN_REPAIR_ROUNDS"] = old_wara
            if old_wcl is None:
                os.environ.pop("WCL_PHASE24_DESIGN_REPAIR_ROUNDS", None)
            else:
                os.environ["WCL_PHASE24_DESIGN_REPAIR_ROUNDS"] = old_wcl

    def test_continue_phase24_defaults_phase25_claim_failure_impl_repair_ten_times(self) -> None:
        old_wara = os.environ.get("WARA_PHASE25_IMPL_REPAIR_ROUNDS")
        old_wcl = os.environ.get("WCL_PHASE25_IMPL_REPAIR_ROUNDS")
        try:
            os.environ.pop("WARA_PHASE25_IMPL_REPAIR_ROUNDS", None)
            os.environ.pop("WCL_PHASE25_IMPL_REPAIR_ROUNDS", None)
            self.assertEqual(_phase25_impl_repair_round_limit(), 10)
            os.environ["WCL_PHASE25_IMPL_REPAIR_ROUNDS"] = "2"
            self.assertEqual(_phase25_impl_repair_round_limit(), 2)
            os.environ["WARA_PHASE25_IMPL_REPAIR_ROUNDS"] = "0"
            self.assertEqual(_phase25_impl_repair_round_limit(), 0)
        finally:
            if old_wara is None:
                os.environ.pop("WARA_PHASE25_IMPL_REPAIR_ROUNDS", None)
            else:
                os.environ["WARA_PHASE25_IMPL_REPAIR_ROUNDS"] = old_wara
            if old_wcl is None:
                os.environ.pop("WCL_PHASE25_IMPL_REPAIR_ROUNDS", None)
            else:
                os.environ["WCL_PHASE25_IMPL_REPAIR_ROUNDS"] = old_wcl

    def test_phase25_objective_metric_alignment_uses_frozen_objective_not_constraint_context(self) -> None:
        run_dir = Path(tempfile.mkdtemp(prefix="phase25-objective-family-"))
        (run_dir / "phase2-1").mkdir(parents=True)
        (run_dir / "phase2-2").mkdir()
        (run_dir / "phase2-3").mkdir()
        (run_dir / "phase2-5").mkdir()
        (run_dir / "phase2-1" / "mathematical_contract.json").write_text(
            json.dumps(
                {
                    "objective": {
                        "sense": "max",
                        "expression": "sum_k omega_k R_k",
                        "meaning": "Weighted sum communication rate under fixed power and sensing-information requirements.",
                        "terms": [{"expression": "omega_k R_k"}],
                    }
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "phase2-2" / "reformulation_path.md").write_text(
            "The sensing constraint is represented by a safe beamforming condition.",
            encoding="utf-8",
        )
        (run_dir / "phase2-3" / "algorithm.md").write_text(
            "The algorithm repeatedly checks sensing, radar, and beampattern constraints.",
            encoding="utf-8",
        )
        (run_dir / "phase2-5" / "phase25_experiment_summary.json").write_text(
            json.dumps({"primary_metric": {"name": "weighted_sum_rate_bpsHz"}}),
            encoding="utf-8",
        )

        result = _phase25_objective_metric_alignment_precheck(run_dir)

        self.assertFalse(result["requires_phase24_design_revision"], result)

    def test_phase25_feedback_replaces_mismatched_primary_metric_instead_of_preserving_it(self) -> None:
        run_dir = Path(tempfile.mkdtemp(prefix="phase25-metric-feedback-"))
        for rel in ("phase2-1", "phase2-2", "phase2-3", "phase2-4", "phase2-5"):
            (run_dir / rel).mkdir(parents=True, exist_ok=True)
        (run_dir / "phase2-1" / "mathematical_contract.json").write_text(
            json.dumps(
                {
                    "objective": {
                        "sense": "max",
                        "expression": "sum_k omega_k R_k",
                        "meaning": "Weighted sum communication rate.",
                    }
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "phase2-4" / "validation_plan.yaml").write_text(
            yaml.safe_dump(
                {
                    "research_evidence_contract": {
                        "primary_metric": {"name": "min_harvested_dc_mW"},
                        "required_result_columns": ["min_harvested_dc_mW", "weighted_sum_rate_bpsHz", "sum_rate_bpsHz"],
                    },
                    "required_outputs": {
                        "scalar_metrics": ["min_harvested_dc_mW", "weighted_sum_rate_bpsHz", "sum_rate_bpsHz"]
                    },
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "phase2-5" / "phase25_experiment_summary.json").write_text(
            json.dumps({"phase25_status": "requires_phase24_design_revision", "primary_metric": {"name": "min_harvested_dc_mW"}}),
            encoding="utf-8",
        )
        (run_dir / "phase2-5" / "phase25_auto_expansion_manifest.json").write_text(
            json.dumps({"final_phase25_status": "requires_phase24_design_revision", "reason": "primary_metric_family_mismatch_with_frozen_objective"}),
            encoding="utf-8",
        )

        feedback = _phase25_design_revision_feedback_for_phase24(run_dir)

        self.assertIn("Primary KPI mismatch is the root failure", feedback)
        self.assertIn("Do not preserve the previous primary y-metric", feedback)
        self.assertIn("weighted_sum_rate_bpsHz", feedback)

    def test_phase25_feedback_uses_service_margin_candidates_for_tau_objective(self) -> None:
        run_dir = Path(tempfile.mkdtemp(prefix="phase25-tau-feedback-"))
        for rel in ("phase2-1", "phase2-2", "phase2-3", "phase2-4", "phase2-5"):
            (run_dir / rel).mkdir(parents=True, exist_ok=True)
        (run_dir / "phase2-1" / "mathematical_contract.json").write_text(
            json.dumps(
                {
                    "objective": {
                        "sense": "max",
                        "expression": "\\tau",
                        "meaning": "Maximize the worst normalized deterministic surplus across communication, sensing, and EH services.",
                    }
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "phase2-4" / "validation_plan.yaml").write_text(
            yaml.safe_dump(
                {
                    "research_evidence_contract": {
                        "primary_metric": {"name": "min_harvested_dc_mW"},
                        "required_result_columns": ["min_harvested_dc_mW", "service_margin_tau", "min_normalized_service_margin"],
                    },
                    "required_outputs": {
                        "scalar_metrics": ["min_harvested_dc_mW", "service_margin_tau", "min_normalized_service_margin"]
                    },
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "phase2-5" / "phase25_experiment_summary.json").write_text(
            json.dumps({"phase25_status": "requires_phase24_design_revision", "primary_metric": {"name": "min_harvested_dc_mW"}}),
            encoding="utf-8",
        )
        (run_dir / "phase2-5" / "phase25_auto_expansion_manifest.json").write_text(
            json.dumps({"final_phase25_status": "requires_phase24_design_revision", "reason": "primary_metric_family_mismatch_with_frozen_objective"}),
            encoding="utf-8",
        )

        feedback = _phase25_design_revision_feedback_for_phase24(run_dir)

        self.assertIn("Primary KPI mismatch is the root failure", feedback)
        self.assertIn("service_margin_tau", feedback)
        self.assertIn("min_normalized_service_margin", feedback)

    def test_phase24_design_gate_blocks_service_objective_metric_mismatch(self) -> None:
        run_dir = Path(tempfile.mkdtemp(prefix="phase24-service-metric-mismatch-"))
        for rel in ("phase2-1", "phase2-2", "phase2-3", "phase2-4"):
            (run_dir / rel).mkdir(parents=True, exist_ok=True)
        (run_dir / "phase2-1" / "mathematical_contract.json").write_text(
            json.dumps(
                {
                    "objective": {
                        "sense": "max",
                        "expression": "eta",
                        "meaning": "Maximize the minimum normalized communication, powering, and sensing service level.",
                    }
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "phase2-2" / "reformulation_path.md").write_text(
            "Use bisection over eta and solve the fixed-eta feasibility problem.",
            encoding="utf-8",
        )
        (run_dir / "phase2-3" / "algorithm.md").write_text(
            "The proposed algorithm maximizes the robust normalized service level eta.",
            encoding="utf-8",
        )
        (run_dir / "phase2-4" / "validation_plan.yaml").write_text(
            yaml.safe_dump(
                {
                    "research_evidence_contract": {
                        "primary_metric": {"name": "min_harvested_dc_mW", "higher_is_better": True},
                        "compared_methods": [
                            {
                                "id": "proposed",
                                "role": "proposed",
                                "scientific_purpose": "Solve the frozen service-balancing problem.",
                                "implementation_hint": "Run the proposed eta-bisection solver.",
                                "fairness_rule": "Use the same channels, power budget, constraints, and metric.",
                            },
                            {
                                "id": "baseline",
                                "role": "main_baseline",
                                "scientific_purpose": "Compare against a reproducible practical benchmark.",
                                "implementation_hint": "Run a fixed-covariance benchmark.",
                                "fairness_rule": "Use the same channels, power budget, constraints, and metric.",
                            },
                        ],
                        "required_result_columns": [
                            "method",
                            "seed",
                            "swept_param",
                            "swept_value",
                            "scenario_name",
                            "objective",
                            "feasible",
                            "min_harvested_dc_mW",
                            "eta_service_level",
                        ],
                        "figures": [
                            {
                                "id": "figure_1",
                                "claim": "Main service-balancing comparison.",
                                "chart_intent": "main_comparison",
                                "required_sweep": "power_sweep",
                                "y_metric": "min_harvested_dc_mW",
                                "methods_to_run": ["proposed", "baseline"],
                                "axis_labels": {"x": "transmit power budget $P_{\\max}$", "y": "minimum harvested DC power $P_{\\rm dc}^{\\min}$"},
                            },
                            {
                                "id": "figure_2",
                                "claim": "Complementary service-level comparison.",
                                "chart_intent": "stress_or_gain",
                                "required_sweep": "uncertainty_sweep",
                                "y_metric": "eta_service_level",
                                "methods_to_run": ["proposed", "baseline"],
                                "axis_labels": {"x": "CSI uncertainty radius $r_h$", "y": "minimum normalized service level $\\eta$"},
                            },
                        ],
                    },
                    "sweep_definitions": [
                        {"id": "power_sweep", "canonical_path": "resources.P_max_W", "scout_values": [0.4, 0.8, 1.2]},
                        {"id": "uncertainty_sweep", "canonical_path": "uncertainty.r_h_common", "scout_values": [0.0, 0.04, 0.08]},
                    ],
                    "required_outputs": {"scalar_metrics": ["min_harvested_dc_mW", "eta_service_level"]},
                }
            ),
            encoding="utf-8",
        )

        result = validate_phase24_evidence_contract_design(run_dir)

        self.assertFalse(result["ok"], result)
        self.assertIn("primary_metric", "\n".join(result["errors"]))
        self.assertIn("eta_service_level", "\n".join(result["errors"]))

    def test_phase24_design_gate_blocks_stale_design_contract_kpi(self) -> None:
        run_dir = Path(tempfile.mkdtemp(prefix="phase24-stale-design-kpi-"))
        for rel in ("phase2-1", "phase2-2", "phase2-3", "phase2-4"):
            (run_dir / rel).mkdir(parents=True, exist_ok=True)
        (run_dir / "phase2-1" / "mathematical_contract.json").write_text(
            json.dumps(
                {
                    "objective": {
                        "sense": "max",
                        "expression": "min_{k in [K]} R_k(w,a)",
                        "meaning": "Maximize the minimum achievable rate across all users (max-min fairness).",
                    }
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "phase2-4" / "experiment_design_contract.json").write_text(
            json.dumps(
                {
                    "primary_physical_kpis": ["P_tx_mW", "harvested_energy_mW", "sensing_illumination_mW"],
                    "figure_contracts": [
                        {
                            "figure_id": "figure_1",
                            "y_metric": "sensing_illumination_mW",
                            "methods_to_run": ["proposed", "baseline"],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "phase2-4" / "validation_plan.yaml").write_text(
            yaml.safe_dump(
                {
                    "research_evidence_contract": {
                        "primary_metric": {"name": "min_user_rate", "higher_is_better": True},
                        "compared_methods": [
                            {
                                "id": "proposed",
                                "role": "proposed",
                                "scientific_purpose": "Solve the frozen max-min rate problem.",
                                "implementation_hint": "Run the proposed target-rate solver.",
                                "fairness_rule": "Use the same channels, AP power budgets, and rate metric.",
                            },
                            {
                                "id": "baseline",
                                "role": "main_baseline",
                                "scientific_purpose": "Compare against a reproducible fixed-association benchmark.",
                                "implementation_hint": "Run a fixed-association power-loading benchmark.",
                                "fairness_rule": "Use the same channels, AP power budgets, and rate metric.",
                            },
                        ],
                        "required_result_columns": [
                            "method",
                            "seed",
                            "swept_param",
                            "swept_value",
                            "scenario_name",
                            "objective",
                            "feasible",
                            "min_user_rate",
                            "sum_rate",
                        ],
                        "figures": [
                            {
                                "id": "figure_1",
                                "claim": "The proposed method improves worst-user rate.",
                                "chart_intent": "main_comparison",
                                "required_sweep": "power_sweep",
                                "y_metric": "min_user_rate",
                                "methods_to_run": ["proposed", "baseline"],
                                "axis_labels": {
                                    "x": "AP transmit-power budget $P_m^{\\max}$",
                                    "y": "minimum user rate $\\min_k R_k$",
                                },
                            },
                            {
                                "id": "figure_2",
                                "claim": "The rate gain remains under a load sweep.",
                                "chart_intent": "stress_or_gain",
                                "required_sweep": "load_sweep",
                                "y_metric": "min_user_rate",
                                "methods_to_run": ["proposed", "baseline"],
                                "axis_labels": {
                                    "x": "number of users $K$",
                                    "y": "minimum user rate $\\min_k R_k$",
                                },
                            },
                        ],
                    },
                    "sweep_definitions": [
                        {"id": "power_sweep", "canonical_path": "system.Pmax_W", "scout_values": [0.5, 1.0]},
                        {"id": "load_sweep", "canonical_path": "system.K", "scout_values": [4, 6]},
                    ],
                    "required_outputs": {"scalar_metrics": ["min_user_rate", "sum_rate"]},
                }
            ),
            encoding="utf-8",
        )

        result = validate_phase24_evidence_contract_design(run_dir)

        self.assertFalse(result["ok"], result)
        errors = "\n".join(result["errors"])
        self.assertIn("experiment_design_contract", errors)
        self.assertIn("sensing_illumination_mW", errors)
        self.assertIn("rate/fairness KPI", errors)

    def test_phase25_refiner_design_revision_status_stops_auto_expansion(self) -> None:
        self.assertTrue(
            _phase25_refiner_requires_phase24_design_revision(
                {
                    "status": "requires_phase24_design_revision",
                    "notes": ["The KPI and claim family cannot be repaired by adding seeds."],
                }
            )
        )
        self.assertTrue(
            _phase25_refiner_requires_phase24_design_revision(
                {"status": "ok", "notes": ["Phase 2.4 experiment-design repair must redesign the validation claim before reruns."]}
            )
        )
        self.assertFalse(_phase25_refiner_requires_phase24_design_revision({"status": "ok", "figures": []}))

    def test_phase25_feedback_preserves_ready_figures_and_repairs_failed_figures(self) -> None:
        run_dir = Path(tempfile.mkdtemp(prefix="phase25-selective-feedback-"))
        phase25_dir = run_dir / "phase2-5"
        phase25_dir.mkdir(parents=True)
        (phase25_dir / "phase25_experiment_summary.json").write_text(
            json.dumps(
                {
                    "phase25_status": "needs_more_phase24_runs",
                    "primary_metric": {"name": "weighted_sum_rate_bpsHz"},
                    "num_comparable_cases": 2800,
                    "proposed_win_rate": 1.0,
                    "proposed_median_relative_gain": 0.08,
                }
            ),
            encoding="utf-8",
        )
        (phase25_dir / "plot_quality_report.json").write_text(
            json.dumps(
                {
                    "figures": [
                        {
                            "figure_id": "figure_1",
                            "counts_toward_paper_minimum": True,
                            "required_sweep": "inactive_sweep",
                            "x_axis_param": "model.inactive_axis",
                            "y_metric": "weighted_sum_rate_bpsHz",
                            "num_x_points": 14,
                            "blocking_issues": ["metric_constant_across_sweep"],
                        },
                        {
                            "figure_id": "figure_2",
                            "counts_toward_paper_minimum": True,
                            "required_sweep": "active_sweep",
                            "x_axis_param": "model.active_axis",
                            "y_metric": "weighted_sum_rate_bpsHz",
                            "num_x_points": 14,
                            "blocking_issues": [],
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )

        feedback = _phase25_design_revision_feedback_for_phase24(run_dir)

        self.assertIn("Selective figure repair contract", feedback)
        self.assertIn("Preserve these already paper-ready final figures", feedback)
        self.assertIn("figure_2", feedback)
        self.assertIn("Redesign or replace only these failed final figures", feedback)
        self.assertIn("figure_1", feedback)

    def test_runtime_feedback_forbids_repeated_inactive_axes(self) -> None:
        run_dir = Path(tempfile.mkdtemp(prefix="phase24-inactive-axis-"))
        phase24_dir = run_dir / "phase2-4"
        phase24_dir.mkdir(parents=True)
        (phase24_dir / "phase24_runtime_design_feedback_round1.txt").write_text(
            "figure_2: metric `weighted_sum_rate_bpsHz` is constant across "
            "required_sweep=spacing_threshold_sweep (executable path `constraints.d_min_m`) for method=proposed.",
            encoding="utf-8",
        )
        responsiveness = {
            "blocking_errors": [
                "figure_1: metric `weighted_sum_rate_bpsHz` is constant across "
                "required_sweep=aperture_side_length_sweep (executable path `geometry.aperture.side_length_m`) for method=proposed."
            ],
            "checks": [
                {
                    "figure_id": "figure_1",
                    "required_sweep": "aperture_side_length_sweep",
                    "metric": "weighted_sum_rate_bpsHz",
                    "metric_span": 0.0,
                    "relative_metric_span": 0.0,
                    "sweep_consumption_proven": True,
                },
                {
                    "figure_id": "figure_2",
                    "required_sweep": "spacing_threshold_sweep",
                    "metric": "weighted_sum_rate_bpsHz",
                    "metric_span": 1.0,
                    "relative_metric_span": 0.1,
                    "sweep_consumption_proven": True,
                    "num_x_values": 3,
                },
            ],
        }

        directive = _phase24_inactive_axis_directive_for_feedback(
            run_dir=run_dir,
            responsiveness=responsiveness,
            repair_round=2,
        )

        self.assertIn("forbidden as final-figure x-axes", directive)
        self.assertIn("spacing_threshold_sweep", directive)
        self.assertIn("aperture_side_length_sweep", directive)
        self.assertIn("replacing the inactive axis is mandatory", directive)
        self.assertIn("Preserve their story", directive)

    def test_phase25_alignment_precheck_blocks_objective_metric_family_mismatch(self) -> None:
        run_dir = Path(tempfile.mkdtemp(prefix="phase25-alignment-precheck-"))
        (run_dir / "phase2-1").mkdir(parents=True)
        (run_dir / "phase2-2").mkdir()
        (run_dir / "phase2-3").mkdir()
        (run_dir / "phase2-5").mkdir()
        (run_dir / "phase2-1" / "mathematical_contract.frozen.json").write_text(
            json.dumps(
                {
                    "objective": {
                        "sense": "max",
                        "expression": "t",
                        "meaning": "Maximize the worst harvested DC power across nonlinear energy receivers.",
                    }
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "phase2-2" / "reformulation_path.md").write_text(
            "Use inverse RF-to-DC rectifier thresholds and bisection over the harvested-DC epigraph.",
            encoding="utf-8",
        )
        (run_dir / "phase2-3" / "algorithm.md").write_text(
            "The proposed method solves the max-min harvested DC covariance problem.",
            encoding="utf-8",
        )
        (run_dir / "phase2-5" / "phase25_experiment_summary.json").write_text(
            json.dumps({"primary_metric": {"name": "sum_rate_bpsHz"}}),
            encoding="utf-8",
        )

        result = _phase25_objective_metric_alignment_precheck(run_dir)

        self.assertTrue(result["requires_phase24_design_revision"], result)
        self.assertEqual(result["objective_family"], "energy")

    def test_continue_phase24_requires_non_empty_mathematical_contract(self) -> None:
        run_dir = Path(tempfile.mkdtemp(prefix="phase24-continue-math-contract-"))
        phase1_dir = run_dir / "phase2-1"
        phase1_dir.mkdir(parents=True, exist_ok=True)

        with self.assertRaisesRegex(ValueError, "mathematical_contract"):
            _load_required_mathematical_contract_json(run_dir)

        (phase1_dir / "mathematical_contract.json").write_text(
            json.dumps(
                {
                    "controls": [{"symbol": "w_k", "status": "control"}],
                    "objective": {"sense": "max", "expression": "sum_k R_k"},
                    "constraints": [{"id": "power_budget", "relation": "sum_k ||w_k||^2 <= P_max"}],
                }
            ),
            encoding="utf-8",
        )

        loaded = json.loads(_load_required_mathematical_contract_json(run_dir))

        self.assertEqual(loaded["controls"][0]["symbol"], "w_k")
        self.assertEqual(loaded["objective"]["sense"], "max")

    def _make_run(self, validation_plan: dict) -> Path:
        evidence = validation_plan.get("paper_evidence_contract")
        if isinstance(evidence, dict):
            methods = evidence.get("compared_methods")
            if isinstance(methods, list):
                for method in methods:
                    if not isinstance(method, dict):
                        continue
                    method_id = str(method.get("id") or "method")
                    method.setdefault("scientific_purpose", f"Evaluate {method_id} as a declared comparison method for the current toy claim.")
                    method.setdefault("implementation_hint", f"Implement {method_id} by changing the toy state according to its declared role.")
                    method.setdefault("fairness_rule", "Use the same toy data, power budget, constraints, and evaluation metrics as other methods.")
        run_dir = Path(tempfile.mkdtemp(prefix="phase24-schema-harness-"))
        phase24_dir = run_dir / "phase2-4"
        solver_dir = phase24_dir / "solver"
        solver_dir.mkdir(parents=True, exist_ok=True)
        (phase24_dir / "validation_plan.yaml").write_text(yaml.safe_dump(validation_plan, sort_keys=False), encoding="utf-8")
        (solver_dir / "validation_plan.yaml").write_text(yaml.safe_dump(validation_plan, sort_keys=False), encoding="utf-8")
        write_phase2_phase24_fixed_harness(run_dir)
        return run_dir

    def test_evidence_contract_outputs_accepts_swept_canonical_path_alias(self) -> None:
        run_dir = self._make_run(
            {
                "problem_family": "toy_wireless",
                "paper_evidence_contract": {
                    "compared_methods": [
                        {"id": "proposed", "role": "proposed"},
                        {"id": "baseline", "role": "main_baseline"},
                    ],
                    "required_result_columns": [
                        "method",
                        "seed",
                        "scenario_name",
                        "swept_param",
                        "swept_value",
                        "sweep_canonical_path",
                        "sum_rate_bpsHz",
                    ],
                },
            }
        )
        outputs = run_dir / "phase2-4" / "solver" / "outputs"
        outputs.mkdir(parents=True, exist_ok=True)
        (outputs / "validation_results.csv").write_text(
            "method,seed,scenario_name,swept_param,swept_value,swept_canonical_path,sum_rate_bpsHz\n"
            "proposed,0,main,resources.P_max_W,1.0,resources.P_max_W,3.5\n",
            encoding="utf-8",
        )

        result = validate_phase24_evidence_contract_outputs(run_dir)

        self.assertTrue(result["ok"], result["errors"])

    def test_repair_merge_preserves_existing_method_branches(self) -> None:
        current = """
def method_solution(problem, model, method, seed=0):
    if method == "comm_only":
        return {"method": "comm_only", "rho": 0.0}
    elif method == "fixed_phase":
        return {"method": "fixed_phase"}
    else:
        return {"method": method}
"""
        repaired = """
def method_solution(problem, model, method, seed=0):
    if method == "rho_fixed_half":
        return {"method": "rho_fixed_half", "rho": 0.5}
    elif method == "linear_eh":
        return {"method": "linear_eh"}
    else:
        return {"method": method}
"""

        merged = merge_phase24_method_solution_branches(current, repaired)

        self.assertIn('method == "comm_only"', merged)
        self.assertIn('method == "fixed_phase"', merged)
        self.assertIn('method == "rho_fixed_half"', merged)
        self.assertIn('method == "linear_eh"', merged)

    def test_preserve_required_exports_keeps_private_helper_dependencies(self) -> None:
        current = """
def _fixed_benchmark(problem, model, seed=0):
    return {"method": "benchmark", "x": 1.0}

def baseline_solution(problem, model, seed=0):
    return _fixed_benchmark(problem, model, seed=seed)
"""
        repaired = """
def build_model(problem, seed=0):
    return {}
"""

        merged = preserve_phase24_required_exports(textwrap.dedent(current), textwrap.dedent(repaired))

        self.assertIn("def baseline_solution", merged)
        self.assertIn("def _fixed_benchmark", merged)

    def test_schema_alignment_blocks_undeclared_problem_attrs(self) -> None:
        run_dir = self._make_run(
            {
                "problem_family": "toy_wireless",
                "canonical_config": {"system": {"Pmax": 1.0}},
                "required_outputs": {"scalar_metrics": ["toy_metric", "sum_rate_bpsHz"]},
            }
        )
        plugin = """
def build_model(problem, seed=0):
    return {"state_init": {}, "operators": {"channel_from_state": channel_from_state, "project_state": project_state, "evaluate_state": evaluate_state}, "metadata": {"max_iterations": 1}}

def initial_state(problem, model, seed=0):
    return {"x": problem.N}

def proposed_step(problem, model, state, iteration):
    return dict(state)

def baseline_solution(problem, model, seed=0):
    return {"x": problem.N}

def channel_from_state(problem, state):
    return {}

def project_state(problem, state):
    return dict(state)

def evaluate_state(problem, model, state):
    return {"status": "feasible", "objective": 1.0, "feasible": True, "constraint_violation": {}, "toy_metric": 1.0}
"""
        (run_dir / "phase2-4" / "solver" / "generated_plugin.py").write_text(textwrap.dedent(plugin), encoding="utf-8")

        result = validate_phase2_phase24_plugin_bundle(run_dir)

        self.assertEqual(result["status"], "schema_alignment_failed")
        error_text = Path(result["error_path"]).read_text(encoding="utf-8")
        self.assertIn("undeclared ProblemData fields", error_text)
        self.assertIn("N", error_text)

    def test_schema_alignment_allows_local_cvxpy_problem_solver_attrs(self) -> None:
        run_dir = self._make_run(
            {
                "problem_family": "toy_wireless",
                "canonical_config": {"system": {"Pmax": 1.0}},
                "required_outputs": {"scalar_metrics": ["toy_metric"]},
            }
        )
        plugin = """
import cvxpy as cp

def build_model(problem, seed=0):
    x = cp.Variable(nonneg=True)
    problem = cp.Problem(cp.Minimize(x), [x >= 1])
    problem.solve()
    return {"solver_status": problem.status, "solver_value": problem.value}
"""
        (run_dir / "phase2-4" / "solver" / "generated_plugin.py").write_text(textwrap.dedent(plugin), encoding="utf-8")

        result = validate_phase2_phase24_schema_alignment(run_dir)

        self.assertTrue(result["ok"], result["errors"])

    def test_schema_driven_plugin_runs_with_declared_metrics(self) -> None:
        run_dir = self._make_run(
            {
                "problem_family": "toy_wireless",
                "canonical_config": {
                    "system": {"Pmax": 1.0},
                },
                "sweep_definitions": {
                    "power": {"variable": "system.Pmax", "values": [0.5, 1.0]},
                },
                "required_outputs": {"scalar_metrics": ["toy_metric"]},
                "paper_evidence_contract": {
                    "compared_methods": [
                        {"id": "proposed", "role": "proposed", "display_name_short": "Proposed", "display_name_long": "Proposed toy method"},
                        {"id": "baseline", "role": "main_baseline", "display_name_short": "Baseline", "display_name_long": "Baseline toy method"},
                    ],
                    "required_result_columns": ["method", "seed", "scenario_name", "swept_param", "swept_value", "toy_metric", "sum_rate_bpsHz"],
                },
                "figure_targets": [
                    {
                        "id": "figure_1",
                        "claim": "Toy metric improves with available power.",
                        "chart_intent": "main_comparison",
                        "required_sweep": "power",
                        "y_metric": "sum_rate_bpsHz",
                        "methods_to_run": ["proposed", "baseline"],
                    }
                ],
            }
        )
        plugin = """
def build_model(problem, seed=0):
    return {"state_init": {"x": 0.0}, "operators": {"channel_from_state": channel_from_state, "project_state": project_state, "evaluate_state": evaluate_state}, "metadata": {"max_iterations": 2}}

def initial_state(problem, model, seed=0):
    return {"x": float(problem.get("system.Pmax", 1.0))}

def proposed_step(problem, model, state, iteration):
    new_state = dict(state)
    new_state["x"] = float(new_state.get("x", 0.0)) + 0.1
    return new_state

def baseline_solution(problem, model, seed=0):
    return {"x": 0.5 * float(problem.get("system.Pmax", 1.0))}

def channel_from_state(problem, state):
    return {"gain": float(problem.get("system.Pmax", 1.0))}

def project_state(problem, state):
    return dict(state)

def evaluate_state(problem, model, state):
    objective = float(state.get("x", 0.0))
    return {"status": "feasible", "objective": objective, "feasible": True, "constraint_violation": {}, "toy_metric": objective, "sum_rate_bpsHz": objective}
"""
        (run_dir / "phase2-4" / "solver" / "generated_plugin.py").write_text(textwrap.dedent(plugin), encoding="utf-8")

        result = validate_phase2_phase24_plugin_bundle(run_dir)

        self.assertEqual(result["status"], "ok")
        summary = json.loads((run_dir / "phase2-4" / "solver" / "outputs" / "validation_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["validation_mode"], "schema_driven")
        self.assertIn("toy_metric", summary["required_metric_keys"])

    def test_evidence_outputs_reject_placeholder_zero_required_metric(self) -> None:
        run_dir = self._make_run(
            {
                "problem_family": "toy_wireless",
                "canonical_config": {"system": {"Pmax": 1.0}},
                "required_outputs": {"scalar_metrics": ["sum_power_W", "average_rho"]},
                "paper_evidence_contract": {
                    "compared_methods": [
                        {"id": "proposed", "role": "proposed"},
                        {"id": "baseline", "role": "main_baseline"},
                    ],
                    "required_result_columns": [
                        "method",
                        "seed",
                        "scenario_name",
                        "swept_param",
                        "swept_value",
                        "sum_power_W",
                        "average_rho",
                    ],
                    "figures": [
                        {
                            "id": "fig_power",
                            "y_metric": "sum_power_W",
                            "required_sweep": "power",
                            "methods_to_run": ["proposed", "baseline"],
                        }
                    ],
                },
                "sweep_definitions": {"power": {"variable": "system.Pmax", "values": [0.5, 1.0]}},
            }
        )
        outputs = run_dir / "phase2-4" / "solver" / "outputs"
        outputs.mkdir(parents=True, exist_ok=True)
        (outputs / "validation_results.csv").write_text(
            "\n".join(
                [
                    "method,seed,scenario_name,swept_param,swept_value,sum_power_W,total_power_W,average_rho,rho_min,rho_max",
                    "proposed,0,case,system.Pmax,0.5,0.0,0.5,0.0,0.2,0.8",
                    "baseline,0,case,system.Pmax,0.5,0.0,0.4,0.0,0.5,0.5",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        result = validate_phase24_evidence_contract_outputs(run_dir)

        self.assertFalse(result["ok"])
        errors = "\n".join(result["errors"])
        self.assertIn("placeholder zero values", errors)
        self.assertIn("sum_power_W", errors)
        self.assertIn("average_rho", errors)

    def test_evidence_outputs_accept_dotted_contract_columns_via_flattened_aliases(self) -> None:
        run_dir = self._make_run(
            {
                "problem_family": "toy_wireless",
                "canonical_config": {"system": {"Pmax": 1.0}},
                "required_outputs": {"scalar_metrics": ["sum_rate_bpsHz"]},
                "paper_evidence_contract": {
                    "compared_methods": [
                        {"id": "proposed", "role": "proposed"},
                        {"id": "baseline", "role": "main_baseline"},
                    ],
                    "required_result_columns": [
                        "method",
                        "seed",
                        "scenario_name",
                        "swept_param",
                        "swept_value",
                        "figure_id",
                        "sum_rate_bpsHz",
                        "actual_channel.blockage.p_unblocked_common",
                        "actual_channel.uncertainty.sigma_e_common",
                        "actual_outage.epsilon",
                    ],
                    "figures": [
                        {
                            "id": "fig_rate",
                            "y_metric": "sum_rate_bpsHz",
                            "required_sweep": "power",
                            "methods_to_run": ["proposed", "baseline"],
                        }
                    ],
                },
                "sweep_definitions": {"power": {"variable": "system.Pmax", "values": [0.5, 1.0]}},
            }
        )
        outputs = run_dir / "phase2-4" / "solver" / "outputs"
        outputs.mkdir(parents=True, exist_ok=True)
        (outputs / "validation_results.csv").write_text(
            "\n".join(
                [
                    "method,seed,scenario_name,swept_param,swept_value,figure_id,sum_rate_bpsHz,actual_channel_blockage_p_unblocked_common,diagnostics_actual_used_channel_uncertainty_sigma_e_common,actual_outage_epsilon",
                    "proposed,0,case,system.Pmax,0.5,fig_rate,3.0,0.8,0.02,0.05",
                    "baseline,0,case,system.Pmax,0.5,fig_rate,2.0,0.8,0.02,0.05",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        result = validate_phase24_evidence_contract_outputs(run_dir)

        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual(result["errors"], [])

    def test_evidence_outputs_reject_missing_declared_figure_methods(self) -> None:
        run_dir = self._make_run(
            {
                "problem_family": "toy_wireless",
                "canonical_config": {"system": {"Pmax": 1.0}},
                "required_outputs": {"scalar_metrics": ["sum_rate_bpsHz"]},
                "paper_evidence_contract": {
                    "compared_methods": [
                        {"id": "proposed", "role": "proposed"},
                        {"id": "baseline_a", "role": "main_baseline"},
                        {"id": "baseline_b", "role": "main_baseline"},
                    ],
                    "required_result_columns": [
                        "method",
                        "seed",
                        "scenario_name",
                        "swept_param",
                        "swept_value",
                        "figure_id",
                        "sum_rate_bpsHz",
                    ],
                    "figures": [
                        {
                            "id": "fig_rate",
                            "y_metric": "sum_rate_bpsHz",
                            "required_sweep": "power",
                            "methods_to_run": ["proposed", "baseline_a", "baseline_b"],
                        }
                    ],
                },
                "sweep_definitions": {"power": {"variable": "system.Pmax", "values": [0.5, 1.0]}},
            }
        )
        outputs = run_dir / "phase2-4" / "solver" / "outputs"
        outputs.mkdir(parents=True, exist_ok=True)
        (outputs / "validation_results.csv").write_text(
            "\n".join(
                [
                    "method,seed,scenario_name,swept_param,swept_value,figure_id,sum_rate_bpsHz",
                    "proposed,0,case,system.Pmax,0.5,fig_rate,3.0",
                    "baseline_a,0,case,system.Pmax,0.5,fig_rate,2.0",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        result = validate_phase24_evidence_contract_outputs(run_dir)

        self.assertFalse(result["ok"])
        errors = "\n".join(result["errors"])
        self.assertIn("missing rows for declared methods_to_run", errors)
        self.assertIn("baseline_b", errors)

    def test_evidence_contract_design_is_required(self) -> None:
        run_dir = self._make_run(
            {
                "problem_family": "toy_wireless",
                "canonical_config": {"system": {"Pmax": 1.0}},
                "required_outputs": {"scalar_metrics": ["toy_metric"]},
            }
        )

        result = validate_phase24_evidence_contract_design(run_dir)

        self.assertFalse(result["ok"])
        self.assertIn("paper_evidence_contract is missing", "\n".join(result["errors"]))

    def test_algorithm_contract_prevents_comparison_only_wmmse_sdr_false_positive(self) -> None:
        run_dir = self._make_run(
            {
                "problem_family": "uplink_power_control",
                "canonical_config": {"system": {"K": 4}, "constraints": {"gamma_target": 3.0}},
                "required_outputs": {"scalar_metrics": ["sum_power_W"]},
                "paper_evidence_contract": {
                    "compared_methods": [
                        {"id": "proposed", "role": "proposed"},
                        {"id": "equal_power_heuristic", "role": "heuristic"},
                    ],
                    "required_result_columns": ["method", "seed", "scenario_name", "swept_param", "swept_value", "sum_power_W"],
                },
            }
        )
        phase2_dir = run_dir / "phase2-2"
        phase3_dir = run_dir / "phase2-3"
        phase2_dir.mkdir(parents=True, exist_ok=True)
        phase3_dir.mkdir(parents=True, exist_ok=True)
        (phase2_dir / "algorithm_contract.json").write_text(
            json.dumps({"algorithm_family": "fixed_point_or_lp_reference"}),
            encoding="utf-8",
        )
        (phase2_dir / "reformulation_path.md").write_text(
            "This LP/fixed-point route does not require SDR, WMMSE, or Gaussian randomization.",
            encoding="utf-8",
        )
        (phase3_dir / "algorithm.md").write_text(
            "Use fixed-point power control. WMMSE/SCA/SDR are not used for this uplink LP.",
            encoding="utf-8",
        )
        plugin = """
def build_model(problem, seed=0):
    return {"state_init": {"p": 1.0}, "operators": {"channel_from_state": channel_from_state, "project_state": project_state, "evaluate_state": evaluate_state}, "metadata": {"max_iterations": 1}}

def initial_state(problem, model, seed=0):
    return {"p": 1.0}

def proposed_step(problem, model, state, iteration):
    return {"p": float(state.get("p", 1.0)), "iteration": iteration + 1}

def baseline_solution(problem, model, seed=0):
    return {"p": 2.0}

def channel_from_state(problem, state):
    return {}

def project_state(problem, state):
    return dict(state)

def evaluate_state(problem, model, state):
    return {"objective": float(state.get("p", 1.0)), "feasible": True, "constraint_violation": 0.0, "sum_power_W": float(state.get("p", 1.0)), "gamma_target": 3.0, "sinr": 1.0, "constraint_residual": 0.0}
"""
        (run_dir / "phase2-4" / "solver" / "generated_plugin.py").write_text(textwrap.dedent(plugin), encoding="utf-8")

        result = validate_phase24_algorithm_code_contract(run_dir)

        self.assertTrue(result["ok"], result["errors"])

    def test_algorithm_contract_blocks_experiment_fallback_proxy(self) -> None:
        run_dir = self._make_run(
            {
                "problem_family": "downlink_beamforming",
                "canonical_config": {"system": {"K": 4, "Nt": 8, "Pmax": 1.0}},
                "required_outputs": {"scalar_metrics": ["sum_rate_bpsHz"]},
                "paper_evidence_contract": {
                    "compared_methods": [
                        {"id": "proposed", "role": "proposed"},
                        {"id": "fixed_ps_baseline", "role": "heuristic"},
                    ],
                    "required_result_columns": ["method", "seed", "scenario_name", "swept_param", "swept_value", "sum_rate_bpsHz"],
                },
            }
        )
        phase2_dir = run_dir / "phase2-2"
        phase3_dir = run_dir / "phase2-3"
        phase2_dir.mkdir(parents=True, exist_ok=True)
        phase3_dir.mkdir(parents=True, exist_ok=True)
        (phase2_dir / "algorithm_contract.json").write_text(
            json.dumps({"algorithm_family": "direct_optimization"}),
            encoding="utf-8",
        )
        (phase2_dir / "reformulation_path.md").write_text(
            "Use successive convex approximation with a convex beamformer subproblem.",
            encoding="utf-8",
        )
        (phase3_dir / "algorithm.md").write_text(
            "The proposed method is a WMMSE-SCA algorithm with receive filters, MSE weights, and a convex surrogate solve.",
            encoding="utf-8",
        )
        (run_dir / "phase2-4" / "phase24_generated_plugin_fallback_reason.txt").write_text(
            "Used deterministic SWIPT power-splitting reference plugin.",
            encoding="utf-8",
        )
        plugin = """
def proposed_step(problem, model, state, iteration):
    return _best_candidate(model["metadata"], "proposed")

def _best_candidate(metadata, method):
    return {"algorithm_family": "wmmse_lightweight_candidate_search", "sum_rate_bpsHz": 1.0}
"""
        (run_dir / "phase2-4" / "solver" / "generated_plugin.py").write_text(textwrap.dedent(plugin), encoding="utf-8")

        result = validate_phase24_algorithm_code_contract(run_dir)

        self.assertFalse(result["ok"])
        joined = "\n".join(result["errors"])
        self.assertIn("experiment fallback", joined.lower())
        self.assertIn("fallback", joined.lower())
        self.assertIn("WMMSE", "\n".join(result["repair_advice"]))

    def test_algorithm_contract_accepts_fp_wmmse_projected_gradient_sca(self) -> None:
        run_dir = self._make_run(
            {
                "problem_family": "movable_antenna_miso_downlink",
                "canonical_config": {"scenario": {"K": 4, "N": 8}},
                "required_outputs": {"scalar_metrics": ["R_wsr_bpsHz"]},
                "paper_evidence_contract": {
                    "compared_methods": [
                        {"id": "proposed", "role": "proposed"},
                        {"id": "fixed_array", "role": "main_baseline"},
                    ],
                    "required_result_columns": ["method", "seed", "scenario_name", "swept_param", "swept_value", "R_wsr_bpsHz"],
                },
            }
        )
        phase2_dir = run_dir / "phase2-2"
        phase3_dir = run_dir / "phase2-3"
        phase2_dir.mkdir(parents=True, exist_ok=True)
        phase3_dir.mkdir(parents=True, exist_ok=True)
        (phase2_dir / "algorithm_contract.json").write_text(
            json.dumps({"algorithm_family": "wmmse_block_coordinate"}),
            encoding="utf-8",
        )
        (phase2_dir / "reformulation_path.md").write_text(
            "Use FP/WMMSE on W and SCA with projected gradient on p.",
            encoding="utf-8",
        )
        (phase3_dir / "algorithm.md").write_text(
            "The proposed method uses FP/WMMSE equivalence, gamma_k updates, QCQP dual bisection, and SCA projected gradient.",
            encoding="utf-8",
        )
        core = """
def _fp_gamma(H, W, sigma2):
    gamma = H @ W
    return gamma

def _w_qcqp_bisection(H, gamma, mu, Pmax):
    # QCQP dual bisection on the power multiplier for the FP/WMMSE equivalent W block.
    return H

def _p_pg_update(p, model, W, gamma):
    # Projected-gradient SCA majorizer with spacing_penalty and project_to_box.
    spacing_penalty = 0.0
    return p

def proposed_step(problem, model, state, iteration):
    gamma_k = _fp_gamma(0, 0, 0)
    W = _w_qcqp_bisection(0, gamma_k, 0, 1)
    p = _p_pg_update(0, model, W, gamma_k)
    for _ in range(1):
        pass
    return {"W": W, "p": p, "iteration": iteration + 1, "sinr": 1.0, "R_wsr_bpsHz": 1.0}
"""
        (run_dir / "phase2-4" / "solver" / "generated_experiment_core.py").write_text(textwrap.dedent(core), encoding="utf-8")
        (run_dir / "phase2-4" / "solver" / "generated_plugin.py").write_text("from generated_experiment_core import *\n", encoding="utf-8")

        result = validate_phase24_algorithm_code_contract(run_dir)

        self.assertTrue(result["ok"], result["errors"])

    def test_algorithm_contract_accepts_sensing_logdet_as_fim_surrogate_metric(self) -> None:
        run_dir = self._make_run(
            {
                "problem_family": "isac_waveform",
                "canonical_config": {"sensing": {"fim_dimension": 2}},
                "required_outputs": {"scalar_metrics": ["sensing_logdet"]},
                "paper_evidence_contract": {
                    "required_result_columns": ["method", "seed", "swept_param", "swept_value", "sensing_logdet"],
                },
            }
        )
        phase3_dir = run_dir / "phase2-3"
        phase3_dir.mkdir(parents=True, exist_ok=True)
        (phase3_dir / "algorithm.md").write_text(
            "The method optimizes a sensing Fisher information surrogate jointly with communication and power transfer.",
            encoding="utf-8",
        )
        plugin = """
def build_model(problem, seed=0):
    return {}

def evaluate_state(problem, model, state):
    sensing_logdet = 1.0
    return {"objective": sensing_logdet, "feasible": True, "constraint_violation": 0.0, "sensing_logdet": sensing_logdet}
"""
        (run_dir / "phase2-4" / "solver" / "generated_plugin.py").write_text(textwrap.dedent(plugin), encoding="utf-8")

        result = validate_phase24_algorithm_code_contract(run_dir)

        self.assertTrue(result["ok"], result["errors"])

    def test_algorithm_contract_accepts_hyphenated_projected_gradient_sca_phrase(self) -> None:
        run_dir = self._make_run(
            {
                "problem_family": "toy_wireless_sca",
                "canonical_config": {"scenario": {"K": 3}},
                "required_outputs": {"scalar_metrics": ["sum_rate_bpsHz"]},
                "paper_evidence_contract": {
                    "compared_methods": [
                        {"id": "proposed", "role": "proposed"},
                        {"id": "baseline", "role": "main_baseline"},
                    ],
                    "required_result_columns": ["method", "seed", "scenario_name", "swept_param", "swept_value", "sum_rate_bpsHz"],
                },
            }
        )
        phase2_dir = run_dir / "phase2-2"
        phase3_dir = run_dir / "phase2-3"
        phase2_dir.mkdir(parents=True, exist_ok=True)
        phase3_dir.mkdir(parents=True, exist_ok=True)
        (phase2_dir / "algorithm_contract.json").write_text(
            json.dumps({"algorithm_family": "sca_or_mm"}),
            encoding="utf-8",
        )
        (phase2_dir / "reformulation_path.md").write_text(
            "Use a successive convex approximation route with a projected-gradient update.",
            encoding="utf-8",
        )
        (phase3_dir / "algorithm.md").write_text(
            "The proposed SCA method uses a projected-gradient majorizer update for the wireless controls.",
            encoding="utf-8",
        )
        core = """
def proposed_step(problem, model, state, iteration):
    # Projected-gradient SCA majorizer with spacing_penalty and project_to_box.
    value = 1.0
    for _ in range(2):
        value = value + 0.1
    return {"iteration": iteration + 1, "sum_rate_bpsHz": value}
"""
        (run_dir / "phase2-4" / "solver" / "generated_experiment_core.py").write_text(textwrap.dedent(core), encoding="utf-8")
        (run_dir / "phase2-4" / "solver" / "generated_plugin.py").write_text("from generated_experiment_core import *\n", encoding="utf-8")

        result = validate_phase24_algorithm_code_contract(run_dir)

        self.assertTrue(result["ok"], result["errors"])

    def test_split_generated_core_runs_through_adapter(self) -> None:
        run_dir = self._make_run(
            {
                "problem_family": "toy_wireless",
                "canonical_config": {"system": {"Pmax": 1.0}},
                "sweep_definitions": {"power": {"variable": "system.Pmax", "values": [0.5, 1.0]}},
                "required_outputs": {"scalar_metrics": ["sum_rate_bpsHz"]},
                "paper_evidence_contract": {
                    "compared_methods": [
                        {"id": "proposed", "role": "proposed"},
                        {"id": "baseline", "role": "main_baseline"},
                    ],
                    "required_result_columns": ["method", "seed", "scenario_name", "swept_param", "swept_value", "sum_rate_bpsHz"],
                },
                "figure_targets": [
                    {
                        "id": "figure_1",
                        "claim": "Rate changes with power.",
                        "chart_intent": "main_comparison",
                        "required_sweep": "power",
                        "y_metric": "sum_rate_bpsHz",
                        "methods_to_run": ["proposed", "baseline"],
                    }
                ],
            }
        )
        core = """
def build_model(problem, seed=0):
    return {"state_init": {"x": 0.0}, "operators": {"channel_from_state": channel_from_state, "project_state": project_state, "evaluate_state": evaluate_state}, "metadata": {"max_iterations": 1}}

def initial_state(problem, model, seed=0):
    return {"x": float(problem.get("system.Pmax", 1.0))}

def proposed_step(problem, model, state, iteration):
    new_state = dict(state)
    new_state["x"] = float(new_state.get("x", 0.0)) + 0.1
    return new_state

def baseline_solution(problem, model, seed=0):
    return {"x": 0.5 * float(problem.get("system.Pmax", 1.0))}

def channel_from_state(problem, state):
    return {"gain": float(problem.get("system.Pmax", 1.0))}

def project_state(problem, state):
    return dict(state)

def evaluate_state(problem, model, state):
    objective = float(state.get("x", 0.0))
    return {"status": "feasible", "objective": objective, "feasible": True, "constraint_violation": {}, "sum_rate_bpsHz": objective}
"""
        solver_dir = run_dir / "phase2-4" / "solver"
        adapter = write_phase24_split_plugin_package(run_dir / "phase2-4", solver_dir, textwrap.dedent(core))

        self.assertIn("def method_solution", adapter)
        self.assertTrue((solver_dir / "generated_experiment_core.py").exists())
        self.assertTrue((solver_dir / "generated_plugin.py").exists())

    def test_split_adapter_normalizes_incomplete_build_model_shape(self) -> None:
        run_dir = self._make_run(
            {
                "problem_family": "toy_wireless",
                "canonical_config": {"system": {"Pmax": 1.0}},
                "sweep_definitions": {"power": {"variable": "system.Pmax", "values": [0.5, 1.0]}},
                "required_outputs": {"scalar_metrics": ["sum_rate_bpsHz"]},
                "paper_evidence_contract": {
                    "compared_methods": [
                        {"id": "proposed", "role": "proposed"},
                        {"id": "baseline", "role": "main_baseline"},
                    ],
                    "required_result_columns": ["method", "seed", "scenario_name", "swept_param", "swept_value", "sum_rate_bpsHz"],
                    "figures": [
                        {
                            "id": "figure_1",
                            "claim": "Toy evidence changes with power.",
                            "chart_intent": "main_comparison",
                            "required_sweep": "power",
                            "y_metric": "sum_rate_bpsHz",
                            "methods_to_run": ["proposed", "baseline"],
                        }
                    ],
                },
            }
        )
        core = """
def build_model(problem, seed=0):
    return {"problem_family": "toy_wireless"}

def initial_state(problem, model, seed=0):
    return {"x": float(problem.get("system.Pmax", 1.0))}

def proposed_step(problem, model, state, iteration):
    return dict(state)

def baseline_solution(problem, model, seed=0):
    return {"x": 0.5}

def channel_from_state(problem, state):
    return {}

def project_state(problem, state):
    return dict(state)

def evaluate_state(problem, model, state):
    value = float(state.get("x", 0.0))
    return {"status": "feasible", "objective": value, "feasible": True, "constraint_violation": {}, "sum_rate_bpsHz": value}
"""
        solver_dir = run_dir / "phase2-4" / "solver"
        adapter = write_phase24_split_plugin_package(run_dir / "phase2-4", solver_dir, textwrap.dedent(core))
        (solver_dir / "generated_plugin.py").write_text(adapter, encoding="utf-8")

        result = validate_phase2_phase24_plugin_bundle(run_dir)

        self.assertEqual(result["status"], "ok")

    def test_split_codegen_package_check_detects_missing_current_core(self) -> None:
        run_dir = Path(tempfile.mkdtemp(prefix="phase24-codegen-package-"))
        phase24_dir = run_dir / "phase2-4"
        solver_dir = phase24_dir / "solver"
        solver_dir.mkdir(parents=True)
        adapter = build_phase24_split_plugin_adapter()
        (solver_dir / "generated_plugin.py").write_text(adapter, encoding="utf-8")
        (phase24_dir / "phase24_split_code_manifest.json").write_text(
            json.dumps(
                {
                    "mode": "split_generated_core_with_deterministic_adapter",
                    "codegen_version": PHASE24_SPLIT_ADAPTER_VERSION,
                    "generated_experiment_core_sha256": "missing",
                }
            ),
            encoding="utf-8",
        )

        result = _validate_phase24_split_codegen_package(phase24_dir)

        self.assertFalse(result["ok"])
        self.assertIn("missing solver/generated_experiment_core.py", "\n".join(result["errors"]))

    def test_split_codegen_package_check_detects_stale_adapter_version(self) -> None:
        run_dir = Path(tempfile.mkdtemp(prefix="phase24-codegen-version-"))
        phase24_dir = run_dir / "phase2-4"
        solver_dir = phase24_dir / "solver"
        solver_dir.mkdir(parents=True)
        core = """
def build_model(problem, seed=0): return {}
def initial_state(problem, model, seed=0): return {}
def proposed_step(problem, model, state, iteration): return state
def baseline_solution(problem, model, seed=0): return {}
def evaluate_state(problem, model, state): return {"objective": 1.0, "feasible": True}
"""
        write_phase24_split_plugin_package(phase24_dir, solver_dir, textwrap.dedent(core))
        adapter = (solver_dir / "generated_plugin.py").read_text(encoding="utf-8")
        (solver_dir / "generated_plugin.py").write_text(
            adapter.replace(PHASE24_SPLIT_ADAPTER_VERSION, "old-version"),
            encoding="utf-8",
        )

        result = _validate_phase24_split_codegen_package(phase24_dir)

        self.assertFalse(result["ok"])
        self.assertIn("not the current deterministic Phase 2.4 adapter", "\n".join(result["errors"]))

    def test_evidence_contract_blocks_objective_only_mechanism_figures(self) -> None:
        run_dir = self._make_run(
            {
                "problem_family": "toy_wireless",
                "canonical_config": {"system": {"Pmax": 1.0}},
                "sweep_definitions": {"power": {"variable": "system.Pmax", "values": [0.5, 1.0]}},
                "required_outputs": {"scalar_metrics": ["objective_value"]},
                "paper_evidence_contract": {
                    "compared_methods": [
                        {"id": "proposed", "role": "proposed"},
                        {"id": "baseline", "role": "main_baseline"},
                    ],
                    "required_result_columns": ["method", "seed", "scenario_name", "swept_param", "swept_value", "objective_value"],
                },
                "figure_targets": [
                    {
                        "id": "figure_2",
                        "claim": "Mechanism ablation should use a physical KPI.",
                        "chart_intent": "mechanism_ablation",
                        "required_sweep": "power",
                        "y_metric": "objective_value",
                        "methods_to_run": ["proposed", "baseline"],
                    }
                ],
            }
        )

        result = validate_phase24_evidence_contract_design(run_dir)

        self.assertFalse(result["ok"])
        self.assertIn("objective-like", "\n".join(result["errors"]))

    def test_evidence_contract_treats_named_utility_as_objective_like(self) -> None:
        run_dir = self._make_run(
            {
                "problem_family": "ris_isac",
                "canonical_config": {"system": {"Pmax": 1.0}},
                "sweep_definitions": {"power": {"variable": "system.Pmax", "values": [0.5, 1.0]}},
                "required_outputs": {"scalar_metrics": ["utility_U_alpha", "sum_rate_bpsHz", "sensing_snr_dB"]},
                "paper_evidence_contract": {
                    "compared_methods": [
                        {"id": "proposed", "role": "proposed"},
                        {"id": "baseline", "role": "main_baseline"},
                    ],
                    "required_result_columns": [
                        "method",
                        "seed",
                        "scenario_name",
                        "swept_param",
                        "swept_value",
                        "utility_U_alpha",
                        "sum_rate_bpsHz",
                        "sensing_snr_dB",
                    ],
                    "figures": [
                        {
                            "id": "figure_1",
                            "claim": "Scalarized utility alone is not the physical evidence story.",
                            "chart_intent": "main_comparison",
                            "required_sweep": "power",
                            "y_metric": "utility_U_alpha",
                            "methods_to_run": ["proposed", "baseline"],
                        }
                    ],
                },
            }
        )

        result = validate_phase24_evidence_contract_design(run_dir)

        self.assertFalse(result["ok"])
        self.assertIn("objective-like", "\n".join(result["errors"]))

    def test_evidence_contract_allows_objective_when_physical_kpi_figure_exists(self) -> None:
        run_dir = self._make_run(
            {
                "problem_family": "toy_wireless",
                "canonical_config": {"system": {"Pmax": 1.0}, "rectifier": {"steepness_b": 6.0}},
                "sweep_definitions": {
                    "power": {"variable": "system.Pmax", "values": [0.5, 1.0]},
                    "rectifier_b": {"variable": "rectifier.steepness_b", "values": [3.0, 6.0, 9.0]},
                },
                "required_outputs": {"scalar_metrics": ["objective", "sum_rate_bpsHz"]},
                "paper_evidence_contract": {
                    "compared_methods": [
                        {
                            "id": "proposed",
                            "role": "proposed",
                            "scientific_purpose": "Evaluate the proposed wireless optimizer.",
                            "implementation_hint": "Use the generated proposed method.",
                            "fairness_rule": "Same channels and constraints.",
                        },
                        {
                            "id": "baseline",
                            "role": "main_baseline",
                            "scientific_purpose": "Evaluate a practical comparison method.",
                            "implementation_hint": "Use the same evaluator with the baseline rule.",
                            "fairness_rule": "Same channels and constraints.",
                        },
                    ],
                    "required_result_columns": [
                        "method",
                        "seed",
                        "scenario_name",
                        "swept_param",
                        "swept_value",
                        "objective",
                        "sum_rate_bpsHz",
                    ],
                    "figures": [
                        {
                            "id": "figure_1",
                            "claim": "Main rate performance changes with power.",
                            "chart_intent": "main_comparison",
                            "required_sweep": "power",
                            "y_metric": "sum_rate_bpsHz",
                            "axis_labels": {"x": "power budget $P_{\\max}$", "y": "sum rate $R_{\\rm sum}$"},
                            "chart_choice_rationale": "Line plot compares system throughput over a resource budget.",
                            "expected_trend": "The sum rate increases with the available transmit resource.",
                            "active_regime_note": "The power sweep keeps the rate constraint active while preserving method fairness.",
                            "methods_to_run": ["proposed", "baseline"],
                        },
                        {
                            "id": "figure_2",
                            "claim": "The paper-defined utility reflects rectifier operating-regime sensitivity.",
                            "chart_intent": "mechanism_sensitivity",
                            "required_sweep": "rectifier_b",
                            "y_metric": "objective",
                            "axis_labels": {"x": "rectifier steepness $b$", "y": "paper utility $U$"},
                            "chart_choice_rationale": "Line plot connects the utility objective to the nonlinear operating regime.",
                            "expected_trend": "The utility changes as the rectifier response becomes sharper.",
                            "active_regime_note": "The rectifier sweep activates the energy-conversion mechanism rather than a solver diagnostic.",
                            "methods_to_run": ["proposed", "baseline"],
                        },
                    ],
                },
            }
        )

        result = validate_phase24_evidence_contract_design(run_dir)

        self.assertTrue(result["ok"], result["errors"])

    def test_evidence_contract_treats_wsr_bpshz_as_physical_kpi(self) -> None:
        run_dir = self._make_run(
            {
                "problem_family": "movable_antenna_miso_downlink",
                "canonical_config": {"scenario": {"Pmax": 1.0}},
                "sweep_definitions": {
                    "aperture": {"variable": "scenario.aperture_diameter_lambda", "values": [2.0, 4.0, 8.0]},
                    "load": {"variable": "scenario.K", "values": [2, 4, 6]},
                },
                "required_outputs": {"scalar_metrics": ["R_wsr_bpsHz", "min_user_rate_bpsHz"]},
                "paper_evidence_contract": {
                    "compared_methods": [
                        {
                            "id": "proposed",
                            "role": "proposed",
                            "scientific_purpose": "Evaluate joint wireless optimization.",
                            "implementation_hint": "Run proposed updates.",
                            "fairness_rule": "Same channels and constraints.",
                        },
                        {
                            "id": "fixed_array",
                            "role": "main_baseline",
                            "scientific_purpose": "Evaluate fixed-array baseline.",
                            "implementation_hint": "Freeze array and optimize beams.",
                            "fairness_rule": "Same channels and constraints.",
                        },
                    ],
                    "required_result_columns": [
                        "method",
                        "seed",
                        "scenario_name",
                        "swept_param",
                        "swept_value",
                        "R_wsr_bpsHz",
                        "min_user_rate_bpsHz",
                    ],
                    "figures": [
                        {
                            "id": "figure_1",
                            "claim": "WSR improves with aperture.",
                            "chart_intent": "main_comparison",
                            "required_sweep": "aperture",
                            "y_metric": "R_wsr_bpsHz",
                            "methods_to_run": ["proposed", "fixed_array"],
                        },
                        {
                            "id": "figure_2",
                            "claim": "WSR remains robust as load grows.",
                            "chart_intent": "stress_or_gain",
                            "required_sweep": "load",
                            "y_metric": "R_wsr_bpsHz",
                            "methods_to_run": ["proposed", "fixed_array"],
                        },
                    ],
                },
            }
        )

        result = validate_phase24_evidence_contract_design(run_dir)

        self.assertTrue(result["ok"], result["errors"])

    def test_evidence_contract_treats_tx_power_notation_as_physical_kpi(self) -> None:
        run_dir = self._make_run(
            {
                "problem_family": "downlink_beamforming",
                "canonical_config": {"system": {"Pmax": 1.0}},
                "sweep_definitions": {
                    "sinr_target": {"variable": "requirements.gamma_dB", "values": [0.0, 4.0, 8.0]},
                    "energy_target": {"variable": "requirements.E_min_mW", "values": [0.1, 0.2, 0.3]},
                },
                "required_outputs": {"scalar_metrics": ["P_tx_mW", "SINR_min_dB", "constraint_violation_max"]},
                "paper_evidence_contract": {
                    "compared_methods": [
                        {
                            "id": "proposed",
                            "role": "proposed",
                            "scientific_purpose": "Evaluate the proposed wireless optimizer.",
                            "implementation_hint": "Solve the declared transmit-power minimization model.",
                            "fairness_rule": "Same channels, targets, budgets, and evaluator.",
                        },
                        {
                            "id": "mrt_split_baseline",
                            "role": "direction_heuristic",
                            "scientific_purpose": "Compare against a practical fixed beam-direction heuristic.",
                            "implementation_hint": "Restrict directions and optimize feasible powers.",
                            "fairness_rule": "Same channels, targets, budgets, and evaluator.",
                        },
                    ],
                    "required_result_columns": [
                        "method",
                        "seed",
                        "scenario_name",
                        "swept_param",
                        "swept_value",
                        "P_tx_mW",
                        "SINR_min_dB",
                        "constraint_violation_max",
                    ],
                    "figures": [
                        {
                            "id": "figure_1",
                            "claim": "Transmit power changes with the SINR target.",
                            "chart_intent": "main_comparison",
                            "required_sweep": "sinr_target",
                            "y_metric": "P_tx_mW",
                            "methods_to_run": ["proposed", "mrt_split_baseline"],
                        },
                        {
                            "id": "figure_2",
                            "claim": "Transmit power changes with the energy-service target.",
                            "chart_intent": "mechanism_sensitivity",
                            "required_sweep": "energy_target",
                            "y_metric": "P_tx_mW",
                            "methods_to_run": ["proposed", "mrt_split_baseline"],
                        },
                    ],
                },
            }
        )

        result = validate_phase24_evidence_contract_design(run_dir)

        self.assertTrue(result["ok"], result["errors"])

    def test_evidence_contract_treats_service_margin_and_resource_fraction_as_paper_kpis(self) -> None:
        run_dir = self._make_run(
            {
                "problem_family": "downlink_beamforming",
                "canonical_config": {"requirements": {"gamma_dB": 3.0}, "rectifier": {"b_mW": 0.05}},
                "sweep_definitions": {
                    "sinr_target": {"variable": "requirements.gamma_dB", "values": [0.0, 3.0, 6.0]},
                    "rectifier_turn_on": {"variable": "rectifier.b_mW", "values": [0.03, 0.06, 0.09]},
                },
                "required_outputs": {"scalar_metrics": ["achieved_tau", "shared_power_fraction", "feasible"]},
                "paper_evidence_contract": {
                    "claims": [
                        {"id": "C1", "statement": "The proposed design improves the service margin.", "primary_kpi": "achieved_tau"},
                        {"id": "C2", "statement": "The shared resource becomes active in the operating regime.", "primary_kpi": "shared_power_fraction"},
                    ],
                    "compared_methods": [
                        {
                            "id": "proposed",
                            "role": "proposed",
                            "scientific_purpose": "Evaluate the proposed wireless optimizer.",
                            "implementation_hint": "Run the declared service-margin method.",
                            "fairness_rule": "Same channels, constraints, and evaluator.",
                        },
                        {
                            "id": "restricted_shared_resource",
                            "role": "main_baseline",
                            "scientific_purpose": "Evaluate a practical restriction on the shared resource.",
                            "implementation_hint": "Restrict the shared resource and use the same evaluator.",
                            "fairness_rule": "Same channels, constraints, and evaluator.",
                        },
                    ],
                    "required_result_columns": [
                        "method",
                        "seed",
                        "scenario_name",
                        "swept_param",
                        "swept_value",
                        "achieved_tau",
                        "shared_power_fraction",
                        "feasible",
                    ],
                    "figures": [
                        {
                            "id": "figure_1",
                            "claim": "C1",
                            "chart_intent": "main_comparison",
                            "required_sweep": "sinr_target",
                            "y_metric": "achieved_tau",
                            "axis_labels": {"x": "SINR target $\\gamma$", "y": "service margin $\\tau$"},
                            "chart_choice_rationale": "Line plot shows how the paper objective changes with service demand.",
                            "expected_trend": "The service margin decreases as the SINR target becomes more stringent.",
                            "active_regime_note": "The SINR target activates the service bottleneck without changing the frozen model.",
                            "methods_to_run": ["proposed", "restricted_shared_resource"],
                        },
                        {
                            "id": "figure_2",
                            "claim": "C2",
                            "chart_intent": "mechanism_sensitivity",
                            "required_sweep": "rectifier_turn_on",
                            "y_metric": "shared_power_fraction",
                            "axis_labels": {"x": "rectifier turn-on level $b$", "y": "shared-power fraction $\\rho_s$"},
                            "chart_choice_rationale": "Line plot exposes the mechanism allocation rather than another abstract objective curve.",
                            "expected_trend": "The shared-power fraction adapts as the energy-conversion operating point changes.",
                            "active_regime_note": "The rectifier sweep activates the resource-sharing mechanism while keeping methods comparable.",
                            "methods_to_run": ["proposed", "restricted_shared_resource"],
                        },
                    ],
                },
            }
        )

        result = validate_phase24_evidence_contract_design(run_dir)

        self.assertTrue(result["ok"], result["errors"])

    def test_evidence_contract_rejects_diagnostic_only_second_paper_axis(self) -> None:
        run_dir = self._make_run(
            {
                "problem_family": "downlink_beamforming",
                "canonical_config": {"system": {"Pmax": 1.0}},
                "sweep_definitions": {
                    "sinr_target": {"variable": "requirements.gamma_dB", "values": [0.0, 4.0, 8.0]},
                    "budget_boundary": {"variable": "constraints.P_max_mW", "values": [3.0, 5.0, 7.0]},
                },
                "required_outputs": {"scalar_metrics": ["P_tx_mW", "constraint_violation_max"]},
                "paper_evidence_contract": {
                    "compared_methods": [
                        {
                            "id": "proposed",
                            "role": "proposed",
                            "scientific_purpose": "Evaluate the proposed wireless optimizer.",
                            "implementation_hint": "Solve the declared transmit-power minimization model.",
                            "fairness_rule": "Same channels, targets, budgets, and evaluator.",
                        },
                        {
                            "id": "benchmark",
                            "role": "main_baseline",
                            "scientific_purpose": "Evaluate a practical comparison method.",
                            "implementation_hint": "Use a restricted feasible design.",
                            "fairness_rule": "Same channels, targets, budgets, and evaluator.",
                        },
                    ],
                    "required_result_columns": [
                        "method",
                        "seed",
                        "scenario_name",
                        "swept_param",
                        "swept_value",
                        "P_tx_mW",
                        "constraint_violation_max",
                    ],
                    "figures": [
                        {
                            "id": "figure_1",
                            "claim": "Transmit power changes with the SINR target.",
                            "chart_intent": "main_comparison",
                            "required_sweep": "sinr_target",
                            "y_metric": "P_tx_mW",
                            "methods_to_run": ["proposed", "benchmark"],
                        },
                        {
                            "id": "figure_2",
                            "claim": "Constraint diagnostics qualify the stress regime.",
                            "chart_intent": "feasibility_boundary",
                            "required_sweep": "budget_boundary",
                            "y_metric": "constraint_violation_max",
                            "methods_to_run": ["proposed", "benchmark"],
                        },
                    ],
                },
            }
        )

        result = validate_phase24_evidence_contract_design(run_dir)

        self.assertFalse(result["ok"])
        error_text = "\n".join(result["errors"])
        self.assertIn("At least two", error_text)
        self.assertIn("diagnostic", error_text)

    def test_evidence_contract_rejects_power_metric_against_power_budget_sweep(self) -> None:
        run_dir = self._make_run(
            {
                "problem_family": "downlink_beamforming",
                "canonical_config": {"system": {"Pmax": 1.0}},
                "sweep_definitions": {
                    "power_budget": {"variable": "constraints.P_max_mW", "values": [3.0, 5.0, 7.0]},
                    "sinr_target": {"variable": "requirements.gamma_dB", "values": [0.0, 4.0, 8.0]},
                },
                "required_outputs": {"scalar_metrics": ["P_tx_mW"]},
                "paper_evidence_contract": {
                    "compared_methods": [
                        {
                            "id": "proposed",
                            "role": "proposed",
                            "scientific_purpose": "Evaluate the proposed wireless optimizer.",
                            "implementation_hint": "Solve the declared transmit-power minimization model.",
                            "fairness_rule": "Same channels, targets, budgets, and evaluator.",
                        },
                        {
                            "id": "benchmark",
                            "role": "main_baseline",
                            "scientific_purpose": "Evaluate a practical comparison method.",
                            "implementation_hint": "Use a restricted feasible design.",
                            "fairness_rule": "Same channels, targets, budgets, and evaluator.",
                        },
                    ],
                    "required_result_columns": [
                        "method",
                        "seed",
                        "scenario_name",
                        "swept_param",
                        "swept_value",
                        "P_tx_mW",
                    ],
                    "figures": [
                        {
                            "id": "figure_1",
                            "claim": "Transmit power should not be plotted against its inactive upper bound.",
                            "chart_intent": "main_comparison",
                            "required_sweep": "power_budget",
                            "required_sweep_param": "constraints.P_max_mW",
                            "y_metric": "P_tx_mW",
                            "methods_to_run": ["proposed", "benchmark"],
                        },
                        {
                            "id": "figure_2",
                            "claim": "Transmit power changes with demand.",
                            "chart_intent": "stress_or_gain",
                            "required_sweep": "sinr_target",
                            "required_sweep_param": "requirements.gamma_dB",
                            "y_metric": "P_tx_mW",
                            "methods_to_run": ["proposed", "benchmark"],
                        },
                    ],
                },
            }
        )

        result = validate_phase24_evidence_contract_design(run_dir)

        self.assertFalse(result["ok"])
        self.assertIn("resource upper bound", "\n".join(result["errors"]))

    def test_evidence_contract_rejects_placeholder_methods(self) -> None:
        run_dir = Path(tempfile.mkdtemp(prefix="phase24-placeholder-methods-"))
        phase24_dir = run_dir / "phase2-4"
        solver_dir = phase24_dir / "solver"
        solver_dir.mkdir(parents=True, exist_ok=True)
        validation_plan = {
            "problem_family": "toy_wireless",
            "canonical_config": {"system": {"Pmax": 1.0}},
            "sweep_definitions": {"power": {"variable": "system.Pmax", "values": [0.5, 1.0]}},
            "required_outputs": {"scalar_metrics": ["sum_rate_bpsHz"]},
            "paper_evidence_contract": {
                "compared_methods": [
                    {"id": "proposed", "role": "proposed"},
                    {"id": "baseline", "role": "main_baseline", "scientific_purpose": "generic baseline"},
                ],
                "required_result_columns": ["method", "seed", "scenario_name", "swept_param", "swept_value", "sum_rate_bpsHz"],
            },
            "figure_targets": [
                {
                    "id": "figure_1",
                    "claim": "Rate changes with power.",
                    "chart_intent": "main_comparison",
                    "required_sweep": "power",
                    "y_metric": "sum_rate_bpsHz",
                    "methods_to_run": ["proposed", "baseline"],
                }
            ],
        }
        (phase24_dir / "validation_plan.yaml").write_text(yaml.safe_dump(validation_plan, sort_keys=False), encoding="utf-8")
        (solver_dir / "validation_plan.yaml").write_text(yaml.safe_dump(validation_plan, sort_keys=False), encoding="utf-8")

        result = validate_phase24_evidence_contract_design(run_dir)

        self.assertFalse(result["ok"])
        self.assertIn("missing executable contract fields", "\n".join(result["errors"]))
        self.assertIn("placeholder/generic", "\n".join(result["errors"]))

    def test_interface_blocks_model_rebuild_inside_helpers(self) -> None:
        run_dir = self._make_run(
            {
                "problem_family": "toy_wireless",
                "canonical_config": {"system": {"Pmax": 1.0}},
                "required_outputs": {"scalar_metrics": ["toy_metric"]},
            }
        )
        plugin = """
def build_model(problem, seed=0):
    return {"state_init": {}, "operators": {"channel_from_state": channel_from_state, "project_state": project_state, "evaluate_state": evaluate_state}, "metadata": {"max_iterations": 1}}

def initial_state(problem, model, seed=0):
    return {"x": 1.0}

def proposed_step(problem, model, state, iteration):
    return dict(state)

def baseline_solution(problem, model, seed=0):
    return {"x": 0.5}

def channel_from_state(problem, state):
    model = build_model(problem, seed=0)
    return {"x": model["metadata"]["max_iterations"]}

def project_state(problem, state):
    return dict(state)

def evaluate_state(problem, model, state):
    return {"status": "feasible", "objective": 1.0, "feasible": True, "constraint_violation": {}, "toy_metric": 1.0}
"""
        solver_dir = run_dir / "phase2-4" / "solver"
        (solver_dir / "generated_plugin.py").write_text(textwrap.dedent(plugin), encoding="utf-8")

        result = validate_phase2_phase24_plugin_interfaces(solver_dir)

        self.assertFalse(result["ok"])
        self.assertIn("calls build_model with seed=0", "\n".join(result["errors"]))

    def test_interface_blocks_method_solution_passthrough(self) -> None:
        run_dir = self._make_run(
            {
                "problem_family": "toy_wireless",
                "canonical_config": {"system": {"Pmax": 1.0}},
                "paper_evidence_contract": {
                    "compared_methods": [
                        {"id": "proposed", "role": "proposed"},
                        {"id": "baseline", "role": "main_baseline"},
                        {"id": "no_coupling", "role": "mechanism_ablation"},
                    ]
                },
                "required_outputs": {"scalar_metrics": ["toy_metric"]},
            }
        )
        plugin = """
def build_model(problem, seed=0):
    return {"state_init": {}, "operators": {"channel_from_state": channel_from_state, "project_state": project_state, "evaluate_state": evaluate_state}, "metadata": {"max_iterations": 1}}

def initial_state(problem, model, seed=0):
    return {"x": 1.0}

def proposed_step(problem, model, state, iteration):
    return dict(state)

def baseline_solution(problem, model, seed=0):
    return {"x": 0.5}

def method_solution(problem, model, method, seed=0):
    state = baseline_solution(problem, model, seed=seed)
    state["method"] = method
    return state

def channel_from_state(problem, state):
    return {}

def project_state(problem, state):
    return dict(state)

def evaluate_state(problem, model, state):
    return {"status": "feasible", "objective": 1.0, "feasible": True, "constraint_violation": {}, "toy_metric": 1.0}
"""
        solver_dir = run_dir / "phase2-4" / "solver"
        (solver_dir / "generated_plugin.py").write_text(textwrap.dedent(plugin), encoding="utf-8")

        result = validate_phase2_phase24_plugin_interfaces(solver_dir)

        self.assertFalse(result["ok"])
        self.assertIn("does not explicitly implement", "\n".join(result["errors"]))

    def test_interface_blocks_harness_signature_drift(self) -> None:
        run_dir = self._make_run(
            {
                "problem_family": "toy_wireless",
                "canonical_config": {"system": {"Pmax": 1.0}},
                "paper_evidence_contract": {
                    "compared_methods": [
                        {"id": "proposed", "role": "proposed"},
                        {"id": "baseline", "role": "main_baseline"},
                    ]
                },
                "required_outputs": {"scalar_metrics": ["toy_metric"]},
            }
        )
        plugin = """
def build_model(problem, seed=0):
    return {"state_init": {}, "operators": {"channel_from_state": channel_from_state, "project_state": project_state, "evaluate_state": evaluate_state}, "metadata": {"max_iterations": 1}}

def initial_state(problem, model, seed=0):
    return {"x": 1.0}

def proposed_step(problem, model, state):
    return dict(state)

def baseline_solution(problem, model, seed=0):
    return {"x": 0.5}

def method_solution(method, problem=None, model=None, seed=0):
    if method == "proposed":
        return {"x": 1.0, "method": "proposed"}
    return {"x": 0.5, "method": method}

def channel_from_state(problem, state):
    return {}

def project_state(problem, state):
    return dict(state)

def evaluate_state(problem, model, state):
    return {"status": "feasible", "objective": 1.0, "feasible": True, "constraint_violation": {}, "toy_metric": 1.0}
"""
        solver_dir = run_dir / "phase2-4" / "solver"
        (solver_dir / "generated_plugin.py").write_text(textwrap.dedent(plugin), encoding="utf-8")

        result = validate_phase2_phase24_plugin_interfaces(solver_dir)

        self.assertFalse(result["ok"])
        errors = "\n".join(result["errors"])
        self.assertIn("signature mismatch for proposed_step", errors)
        self.assertIn("signature mismatch for method_solution", errors)

    def test_split_core_normalizes_common_repair_signature_drift(self) -> None:
        run_dir = self._make_run(
            {
                "problem_family": "toy_wireless",
                "canonical_config": {"system": {"Pmax": 1.0}},
                "paper_evidence_contract": {
                    "compared_methods": [
                        {"id": "proposed", "role": "proposed"},
                        {"id": "baseline", "role": "main_baseline"},
                    ],
                    "figures": [{"id": "figure_1", "methods_to_run": ["proposed", "baseline"], "y_metric": "toy_metric"}],
                },
                "required_outputs": {"scalar_metrics": ["toy_metric"]},
            }
        )
        core = """
def build_model(problem, seed=0):
    return {"state_init": {}, "metadata": {"max_iterations": 1}}

def initial_state(problem, model, seed=0):
    return {"x": 1.0, "seed": seed}

def proposed_step(state, model, seed=0, **kwargs):
    updated = dict(state)
    updated["x"] = updated.get("x", 0.0) + 1.0 + 0.0 * seed
    return updated

def baseline_solution(problem, model, seed=0):
    return {"x": 0.5, "method": "baseline", "seed": seed}

def evaluate_state(state, model, seed=0, **kwargs):
    return {"status": "ok", "objective": float(state["x"]), "feasible": True, "constraint_violation": {}, "toy_metric": float(state["x"]) + 0.0 * seed}

def method_solution(method, model=None, seed=0, **kwargs):
    state = {"x": 1.0, "method": method, "seed": seed}
    if method == "proposed":
        return proposed_step(state, model, seed=seed)
    return {"x": 0.5, "method": method, "seed": seed}

def run_point(problem, method, seed=0):
    model = build_model(problem, seed=seed)
    return evaluate_state(method_solution(method, model, seed=seed), model, seed=seed)
"""
        phase24_dir = run_dir / "phase2-4"
        solver_dir = phase24_dir / "solver"
        adapter = write_phase24_split_plugin_package(phase24_dir, solver_dir, textwrap.dedent(core))
        (solver_dir / "generated_plugin.py").write_text(adapter, encoding="utf-8")

        normalized = (solver_dir / "generated_experiment_core.py").read_text(encoding="utf-8")
        self.assertIn("def proposed_step(problem, model, state, iteration=0, **kwargs):", normalized)
        self.assertIn("def evaluate_state(problem, model, state, **kwargs):", normalized)
        self.assertIn("def method_solution(problem, model, method, seed=0, **kwargs):", normalized)
        result = validate_phase2_phase24_plugin_interfaces(solver_dir)

        self.assertTrue(result["ok"], result["errors"])

    def test_interface_accepts_core_registry_with_method_aware_solver_dispatch(self) -> None:
        run_dir = self._make_run(
            {
                "problem_family": "toy_wireless",
                "canonical_config": {"system": {"Pmax": 1.0}},
                "paper_evidence_contract": {
                    "compared_methods": [
                        {"id": "proposed", "role": "proposed"},
                        {"id": "no_coupling", "role": "mechanism_ablation"},
                    ],
                    "figures": [
                        {
                            "id": "figure_1",
                            "methods_to_run": ["proposed", "no_coupling"],
                            "y_metric": "toy_metric",
                        }
                    ],
                },
                "required_outputs": {"scalar_metrics": ["toy_metric"]},
            }
        )
        solver_dir = run_dir / "phase2-4" / "solver"
        adapter = """
import generated_experiment_core as _core

def build_model(problem, seed=0):
    return _core.build_model(problem, seed=seed)

def initial_state(problem, model, seed=0):
    return _core.initial_state(problem, model, seed=seed)

def proposed_step(problem, model, state, iteration):
    return _core.proposed_step(problem, model, state, iteration)

def baseline_solution(problem, model, seed=0):
    return _core.baseline_solution(problem, model, seed=seed)

def evaluate_state(problem, model, state):
    return _core.evaluate_state(problem, model, state)

def method_solution(problem, model, method, seed=0):
    return _core.method_solution(problem, model, method, seed=seed)
"""
        core = """
ACTIVE_METHOD_IDS = ("proposed", "no_coupling")

def build_model(problem, seed=0):
    return {"state_init": {}, "operators": {"channel_from_state": channel_from_state, "project_state": project_state, "evaluate_state": evaluate_state}, "metadata": {"max_iterations": 1}}

def initial_state(problem, model, seed=0):
    return {"x": 1.0}

def proposed_step(problem, model, state, iteration):
    return _solve_method(model, "proposed", seed)

def baseline_solution(problem, model, seed=0):
    return method_solution(problem, model, "no_coupling", seed=seed)

def _solve_method(model, method, seed=0):
    if method == "no_coupling":
        return {"x": 0.5, "method": "no_coupling"}
    return {"x": 1.0, "method": "proposed"}

def method_solution(problem, model, method, seed=0):
    if method not in ACTIVE_METHOD_IDS:
        raise ValueError(method)
    return _solve_method(model, method, seed=seed)

def channel_from_state(problem, state):
    return {}

def project_state(problem, state):
    return dict(state)

def evaluate_state(problem, model, state):
    return {"status": "feasible", "objective": 1.0, "feasible": True, "constraint_violation": {}, "toy_metric": state.get("x", 0.0)}
"""
        (solver_dir / "generated_plugin.py").write_text(textwrap.dedent(adapter), encoding="utf-8")
        (solver_dir / "generated_experiment_core.py").write_text(textwrap.dedent(core), encoding="utf-8")

        result = validate_phase2_phase24_plugin_interfaces(solver_dir)

        self.assertTrue(result["ok"], result["errors"])

    def test_interface_allows_private_cached_model_fallback_helper(self) -> None:
        run_dir = self._make_run(
            {
                "problem_family": "toy_wireless",
                "canonical_config": {"system": {"Pmax": 1.0}},
                "required_outputs": {"scalar_metrics": ["toy_metric"]},
            }
        )
        plugin = """
def build_model(problem, seed=0):
    return {"state_init": {}, "operators": {"channel_from_state": channel_from_state, "project_state": project_state, "evaluate_state": evaluate_state}, "metadata": {"max_iterations": 1}, "hhat": [1], "gamma_linear": 1.0}

def initial_state(problem, model, seed=0):
    return {"x": 1.0}

def proposed_step(problem, model, state, iteration):
    return dict(state)

def baseline_solution(problem, model, seed=0):
    return {"x": 0.5}

def _evaluate_state_core(problem_or_model, state, seed=0):
    if isinstance(problem_or_model, dict) and "hhat" in problem_or_model and "gamma_linear" in problem_or_model:
        model = problem_or_model
    else:
        model = build_model(problem_or_model, seed=seed)
    return {"status": "feasible", "objective": model["gamma_linear"], "feasible": True, "constraint_violation": {}, "toy_metric": state.get("x", 0.0)}

def channel_from_state(problem, state):
    return {}

def project_state(problem, state):
    return dict(state)

def evaluate_state(problem, model, state):
    return _evaluate_state_core(model, state, seed=0)
"""
        solver_dir = run_dir / "phase2-4" / "solver"
        (solver_dir / "generated_plugin.py").write_text(textwrap.dedent(plugin), encoding="utf-8")

        result = validate_phase2_phase24_plugin_interfaces(solver_dir)

        self.assertTrue(result["ok"], result["errors"])

    def test_interface_does_not_force_unused_optional_compared_methods(self) -> None:
        run_dir = self._make_run(
            {
                "problem_family": "toy_wireless",
                "canonical_config": {"system": {"Pmax": 1.0}},
                "paper_evidence_contract": {
                    "compared_methods": [
                        {"id": "proposed", "role": "proposed"},
                        {"id": "baseline", "role": "main_baseline"},
                        {"id": "future_ablation", "role": "mechanism_ablation"},
                    ],
                    "figures": [
                        {
                            "id": "figure_1",
                            "methods_to_run": ["proposed", "baseline"],
                            "y_metric": "toy_metric",
                        }
                    ],
                },
                "required_outputs": {"scalar_metrics": ["toy_metric"]},
            }
        )
        plugin = """
def build_model(problem, seed=0):
    return {"state_init": {}, "operators": {"channel_from_state": channel_from_state, "project_state": project_state, "evaluate_state": evaluate_state}, "metadata": {"max_iterations": 1}}

def initial_state(problem, model, seed=0):
    return {"x": 1.0}

def proposed_step(problem, model, state, iteration):
    return dict(state)

def baseline_solution(problem, model, seed=0):
    return {"x": 0.5}

def channel_from_state(problem, state):
    return {}

def project_state(problem, state):
    return dict(state)

def evaluate_state(problem, model, state):
    return {"status": "feasible", "objective": 1.0, "feasible": True, "constraint_violation": 0.0, "toy_metric": 1.0}
"""
        solver_dir = run_dir / "phase2-4" / "solver"
        (solver_dir / "generated_plugin.py").write_text(textwrap.dedent(plugin), encoding="utf-8")

        result = validate_phase2_phase24_plugin_interfaces(solver_dir)

        self.assertTrue(result["ok"], result)

    def test_method_semantics_blocks_no_rho_with_nonzero_reported_rho(self) -> None:
        run_dir = self._make_run(
            {
                "problem_family": "toy_wireless",
                "canonical_config": {"system": {"Pmax": 1.0}},
                "paper_evidence_contract": {
                    "compared_methods": [
                        {"id": "proposed", "role": "proposed"},
                        {
                            "id": "no_rho",
                            "role": "mechanism_ablation",
                            "display_name_short": "No-rho",
                            "scientific_purpose": "No structural separation, rho=0",
                        },
                    ]
                },
            }
        )
        outputs_dir = run_dir / "phase2-4" / "solver" / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)
        (outputs_dir / "validation_results.csv").write_text(
            "\n".join(
                [
                    "method,status,objective,feasible,constraint_violation,rho",
                    "proposed,ok,1.0,True,0.0,0.2",
                    "no_rho,ok,0.9,True,0.0,0.5",
                ]
            ),
            encoding="utf-8",
        )

        result = validate_phase24_method_semantics(run_dir)

        self.assertFalse(result["ok"])
        self.assertIn("no-rho/no-structural-separation", "\n".join(result["errors"]))

    def test_method_semantics_ignores_stale_paper_results_from_previous_plugin(self) -> None:
        run_dir = self._make_run(
            {
                "problem_family": "toy_wireless",
                "canonical_config": {"system": {"Pmax": 1.0}},
                "paper_evidence_contract": {
                    "compared_methods": [
                        {"id": "proposed", "role": "proposed"},
                        {
                            "id": "no_rho",
                            "role": "mechanism_ablation",
                            "scientific_purpose": "No structural separation, rho=0",
                        },
                    ]
                },
            }
        )
        outputs_dir = run_dir / "phase2-4" / "solver" / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)
        (outputs_dir / "validation_results.csv").write_text(
            "\n".join(
                [
                    "method,status,objective,feasible,constraint_violation,rho,M_eh",
                    "proposed,ok,1.0,True,0.0,0.2,4",
                    "no_rho,ok,0.9,True,0.0,0.0,0",
                ]
            ),
            encoding="utf-8",
        )
        stale_paper = outputs_dir / "paper_validation_results.csv"
        stale_paper.write_text(
            "\n".join(
                [
                    "method,status,objective,feasible,constraint_violation,rho,M_eh",
                    "no_rho,ok,0.9,True,0.0,0.5,8",
                ]
            ),
            encoding="utf-8",
        )
        old_time = time.time() - 100.0
        os.utime(stale_paper, (old_time, old_time))
        plugin_path = run_dir / "phase2-4" / "solver" / "generated_plugin.py"
        plugin_path.write_text("# current plugin marker\n", encoding="utf-8")

        result = validate_phase24_method_semantics(run_dir)

        self.assertTrue(result["ok"], result)
        self.assertIn("ignored stale", "\n".join(result["warnings"]))

    def test_method_semantics_blocks_duplicate_active_plotted_methods(self) -> None:
        run_dir = self._make_run(
            {
                "problem_family": "toy_wireless",
                "canonical_config": {"system": {"Pmax": 1.0}},
                "paper_evidence_contract": {
                    "compared_methods": [
                        {"id": "proposed", "role": "proposed"},
                        {"id": "benchmark_a", "role": "main_baseline"},
                        {"id": "benchmark_b", "role": "main_baseline"},
                    ],
                    "figures": [
                        {
                            "id": "figure_1",
                            "methods_to_run": ["proposed", "benchmark_a", "benchmark_b"],
                            "y_metric": "toy_metric",
                        }
                    ],
                },
            }
        )
        outputs_dir = run_dir / "phase2-4" / "solver" / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)
        (outputs_dir / "validation_results.csv").write_text(
            "\n".join(
                [
                    "case_id,seed,swept_param,swept_value,scenario_name,method,objective,toy_metric,feasible",
                    "c0,0,x,0,s,proposed,1.2,1.2,True",
                    "c0,0,x,0,s,benchmark_a,1.0,1.0,True",
                    "c0,0,x,0,s,benchmark_b,1.0,1.0,True",
                    "c1,0,x,1,s,proposed,1.3,1.3,True",
                    "c1,0,x,1,s,benchmark_a,1.1,1.1,True",
                    "c1,0,x,1,s,benchmark_b,1.1,1.1,True",
                    "c2,0,x,2,s,proposed,1.4,1.4,True",
                    "c2,0,x,2,s,benchmark_a,1.2,1.2,True",
                    "c2,0,x,2,s,benchmark_b,1.2,1.2,True",
                ]
            ),
            encoding="utf-8",
        )

        result = validate_phase24_method_semantics(run_dir)

        self.assertFalse(result["ok"])
        self.assertIn("numerically identical", "\n".join(result["errors"]))

    def test_interface_blocks_no_rho_branch_with_nonzero_literal(self) -> None:
        run_dir = self._make_run(
            {
                "problem_family": "toy_wireless",
                "canonical_config": {"system": {"Pmax": 1.0}},
                "paper_evidence_contract": {
                    "compared_methods": [
                        {"id": "proposed", "role": "proposed"},
                        {
                            "id": "no_rho",
                            "role": "mechanism_ablation",
                            "scientific_purpose": "No structural separation, rho=0",
                        },
                    ]
                },
                "required_outputs": {"scalar_metrics": ["toy_metric"]},
            }
        )
        plugin = """
def build_model(problem, seed=0):
    return {"state_init": {}, "operators": {"channel_from_state": channel_from_state, "project_state": project_state, "evaluate_state": evaluate_state}, "metadata": {"max_iterations": 1}}

def initial_state(problem, model, seed=0):
    return {"rho": 0.2}

def proposed_step(problem, model, state, iteration):
    return dict(state)

def baseline_solution(problem, model, seed=0):
    return {"rho": 0.0}

def method_solution(problem, model, method, seed=0):
    if method == "no_rho":
        state = initial_state(problem, model, seed)
        state["rho"] = 0.5
        state["method"] = "no_rho"
        return state
    return baseline_solution(problem, model, seed)

def channel_from_state(problem, state):
    return {}

def project_state(problem, state):
    return dict(state)

def evaluate_state(problem, model, state):
    return {"status": "feasible", "objective": 1.0, "feasible": True, "constraint_violation": 0.0, "toy_metric": 1.0, "rho": float(state.get("rho", 0.0))}
"""
        solver_dir = run_dir / "phase2-4" / "solver"
        (solver_dir / "generated_plugin.py").write_text(textwrap.dedent(plugin), encoding="utf-8")

        result = validate_phase2_phase24_plugin_interfaces(solver_dir)

        self.assertFalse(result["ok"])
        self.assertIn("assigns a nonzero rho literal", "\n".join(result["errors"]))

    def test_interface_blocks_ris_quadratic_double_transpose(self) -> None:
        run_dir = self._make_run(
            {
                "problem_family": "toy_wireless",
                "canonical_config": {"system": {"Pmax": 1.0}},
                "required_outputs": {"scalar_metrics": ["toy_metric"]},
            }
        )
        plugin = """
import numpy as np

def build_model(problem, seed=0):
    return {"state_init": {}, "operators": {"channel_from_state": channel_from_state, "project_state": project_state, "evaluate_state": evaluate_state}, "metadata": {"max_iterations": 1}}

def initial_state(problem, model, seed=0):
    return {"V": np.eye(2)}

def _herm(x):
    return x.conj().T

def proposed_step(problem, model, state, iteration):
    A_radar = np.ones((1, 2), dtype=complex)
    A_radar_h = _herm(A_radar).T
    state = dict(state)
    state["V"] = state["V"] + 0.01 * (A_radar_h @ A_radar)
    return state

def baseline_solution(problem, model, seed=0):
    return {"V": np.eye(2)}

def channel_from_state(problem, state):
    return {}

def project_state(problem, state):
    return dict(state)

def evaluate_state(problem, model, state):
    return {"status": "feasible", "objective": 1.0, "feasible": True, "constraint_violation": 0.0, "toy_metric": 1.0}
"""
        solver_dir = run_dir / "phase2-4" / "solver"
        (solver_dir / "generated_plugin.py").write_text(textwrap.dedent(plugin), encoding="utf-8")

        result = validate_phase2_phase24_plugin_interfaces(solver_dir)

        self.assertFalse(result["ok"])
        self.assertIn("double-transpose", "\n".join(result["errors"]))

    def test_plugin_source_normalizer_repairs_common_phase24_math_bugs(self) -> None:
        source = """
def _compute_Pin(rho, M):
    M_eh = max(1, int(np.floor(rho * M)))
    return M_eh

def _update_V(A_radar):
    return _herm(A_radar).T @ A_radar
"""
        repaired = normalize_phase24_generated_plugin_source(textwrap.dedent(source))

        self.assertIn("0 if rho <= 0 else max(1, int(np.floor(rho * M)))", repaired)
        self.assertIn("_herm(A_radar) @ A_radar", repaired)
        self.assertNotIn("_herm(A_radar).T @ A_radar", repaired)

    def test_plugin_source_normalizer_repairs_harness_signature_drift(self) -> None:
        source = """
def proposed_step(problem, model, state):
    return dict(state)

def method_solution(method, problem=None, model=None, seed=0):
    if method == "proposed":
        return {"method": method, "x": 1.0}
    return {"method": method, "x": 0.5}
"""
        repaired = normalize_phase24_generated_plugin_source(textwrap.dedent(source))

        self.assertIn("def proposed_step(problem, model, state, iteration=0):", repaired)
        self.assertIn("def method_solution(problem, model, method, seed=0, **kwargs):", repaired)

    def test_plugin_source_normalizer_tolerates_internal_keyword_drift(self) -> None:
        source = """
def _solve_power_sca(model, phi, W, p0):
    return p0, {"status": "ok"}

def method_solution(problem, model, method, seed=0):
    return _solve_power_sca(model, 1, 2, 3, enforce_floor=True, max_inner=4)[1]
"""
        repaired = normalize_phase24_generated_plugin_source(textwrap.dedent(source))

        self.assertIn("def _solve_power_sca(model, phi, W, p0, **kwargs):", repaired)

    def test_plugin_source_normalizer_populates_training_scenario_cache(self) -> None:
        source = """
def _make_scenarios_for_relay(model, m, n, seed, adversarial=True):
    return [{"relay": m, "seed": seed}]

def _construct_model(problem, seed=0):
    model = {"relay_count_M": 2, "training_scenarios_Ntr": 3}
    return model

def initial_state(problem, model, seed=0):
    return {"scenario": model["training_scenarios"][0][0]}
"""
        repaired = normalize_phase24_generated_plugin_source(textwrap.dedent(source))

        self.assertIn('model["training_scenarios"] = [', repaired)
        namespace: dict[str, object] = {}
        exec(repaired, namespace)
        model = namespace["_construct_model"]({}, seed=5)
        self.assertEqual(len(model["training_scenarios"]), 2)

    def test_plugin_source_normalizer_adds_relay_count_alias(self) -> None:
        source = """
def _construct_model(problem, seed=0):
    model = {"relay_count_M": 3}
    return model

def _select(model):
    return list(range(model["M"]))
"""
        repaired = normalize_phase24_generated_plugin_source(textwrap.dedent(source))

        self.assertIn('model["M"] = int(model.get("relay_count_M", 0))', repaired)
        namespace: dict[str, object] = {}
        exec(repaired, namespace)
        self.assertEqual(namespace["_select"](namespace["_construct_model"]({}, seed=0)), [0, 1, 2])

    def test_plugin_source_normalizer_matches_solver_return_schema_to_callers(self) -> None:
        source = """
import numpy as np

def _solve_power_sca(model, phi, W, p0):
    p_cur = p0
    used_solver = True
    status = "optimal"
    last_value = 1.0
    return p_cur, {
        "used_power_sca_update": bool(used_solver),
        "power_sca_status": status,
        "power_sca_objective": float(last_value) if np.isfinite(last_value) else None,
    }

def baseline_solution(problem, model, seed=0):
    sol = _solve_power_sca(model, 1, 2, 3)
    return {"ok": sol["ok"], "p": sol["p"]}
"""
        repaired = normalize_phase24_generated_plugin_source(textwrap.dedent(source))

        self.assertIn('"ok": bool(used_solver', repaired)
        namespace: dict[str, object] = {}
        exec(repaired, namespace)
        self.assertEqual(namespace["baseline_solution"]({}, {}, seed=0)["p"], 3)

    def test_plugin_source_normalizer_adds_selected_relay_beams(self) -> None:
        source = '''
def initial_state(problem, model, seed=0):
    relays = [{"beams": {"x": 1.0}, "eta": 1.0, "mean_rate": 2.0}]
    selected = 0
    return {
        "method": "proposed",
        "iteration": 0,
        "relays": relays,
        "selected_relay": int(selected),
    }

def proposed_step(problem, model, state, iteration):
    return state["beams"]
'''
        repaired = normalize_phase24_generated_plugin_source(textwrap.dedent(source))

        self.assertIn('"beams": relays[selected]["beams"]', repaired)
        namespace: dict[str, object] = {}
        exec(repaired, namespace)
        state = namespace["initial_state"]({}, {}, seed=0)
        self.assertEqual(namespace["proposed_step"]({}, {}, state, 0), {"x": 1.0})

    def test_plugin_source_normalizer_repairs_make_state_argument_order(self) -> None:
        source = """
import numpy as np
from typing import Any, Dict

def _make_state(method: str, iteration: int, phi: np.ndarray, p: np.ndarray, W: np.ndarray, extra: Dict[str, Any] = None) -> Dict[str, Any]:
    return {"method": method, "iteration": int(iteration), "p": np.asarray(p, dtype=float).tolist()}

def baseline_solution(problem, model, seed=0):
    return _make_state("baseline", np.array([1.0]), np.array([2.0]), np.eye(1), 0, {})
"""
        repaired = normalize_phase24_generated_plugin_source(textwrap.dedent(source))

        self.assertIn("def _make_state(method: str, phi: np.ndarray, p: np.ndarray, W: np.ndarray, iteration: int", repaired)
        namespace: dict[str, object] = {}
        exec(repaired, namespace)
        self.assertEqual(namespace["baseline_solution"]({}, {}, seed=0)["p"], [2.0])

    def test_plugin_source_normalizer_sanitizes_sdr_recovery_matrix_factors(self) -> None:
        source = """
import numpy as np

def recover_from_solver_matrix(X):
    Xv = np.asarray(X.value, dtype=complex)
    Xv = 0.5 * (Xv + Xv.conj().T)
    vals, vecs = np.linalg.eigh(Xv)
    vals = np.maximum(np.real(vals), 0.0)
    L = vecs @ np.diag(np.sqrt(np.maximum(vals, 0.0)))
    return bool(np.isfinite(L).all())
"""
        repaired = normalize_phase24_generated_plugin_source(textwrap.dedent(source))

        self.assertIn("np.nan_to_num(Xv", repaired)
        self.assertIn("try:", repaired)
        self.assertIn("safe_vals = np.nan_to_num", repaired)
        self.assertIn("safe_vecs = np.clip", repaired)
        self.assertIn("L = safe_vecs * np.sqrt(safe_vals)[None, :]", repaired)
        self.assertNotIn("safe_vecs @ np.diag", repaired)
        namespace: dict[str, object] = {}
        exec(repaired, namespace)

        class Box:
            value = __import__("numpy").array([[float("inf"), 0.0], [0.0, float("nan")]], dtype=complex)

        self.assertTrue(namespace["recover_from_solver_matrix"](Box()))

    def test_plugin_source_normalizer_sanitizes_sdp_trace_objective_matrix(self) -> None:
        source = """
import cvxpy as cp
import numpy as np

def build_problem(Qsum, X):
    Qsum = 0.5 * (Qsum + Qsum.conj().T)
    return cp.Problem(cp.Maximize(cp.real(cp.trace(Qsum @ X))), [])
"""
        repaired = normalize_phase24_generated_plugin_source(textwrap.dedent(source))

        self.assertIn("Qsum = np.nan_to_num(Qsum", repaired)
        self.assertIn("Qsum = np.clip(np.real(Qsum)", repaired)
        self.assertIn("cp.sum(cp.multiply(Qsum.T, X))", repaired)
        self.assertNotIn("cp.trace(Qsum @ X)", repaired)

    def test_plugin_source_normalizer_sanitizes_randomization_vector_products(self) -> None:
        source = """
import numpy as np

def _complex_normal(rng, shape, scale):
    return np.asarray([complex(float("inf"), 0.0), 1.0 + 0.0j])

def sample(L, rng):
    z = L @ _complex_normal(rng, (2,), 1.0)
    return bool(np.isfinite(z).all())
"""
        repaired = normalize_phase24_generated_plugin_source(textwrap.dedent(source))

        self.assertIn("random_vec = _complex_normal", repaired)
        self.assertIn("np.sum(L * random_vec[None, :], axis=1)", repaired)
        self.assertNotIn("z = L @ _complex_normal", repaired)
        namespace: dict[str, object] = {}
        exec(repaired, namespace)
        self.assertTrue(namespace["sample"](__import__("numpy").eye(2), None))

    def test_interface_blocks_no_rho_partition_helper_with_min_one(self) -> None:
        run_dir = self._make_run(
            {
                "problem_family": "toy_wireless",
                "canonical_config": {"system": {"Pmax": 1.0}},
                "paper_evidence_contract": {
                    "compared_methods": [
                        {"id": "proposed", "role": "proposed"},
                        {
                            "id": "no_rho",
                            "role": "mechanism_ablation",
                            "scientific_purpose": "No structural separation, rho=0",
                        },
                    ]
                },
                "required_outputs": {"scalar_metrics": ["toy_metric"]},
            }
        )
        plugin = """
import numpy as np

def build_model(problem, seed=0):
    return {"state_init": {}, "operators": {"channel_from_state": channel_from_state, "project_state": project_state, "evaluate_state": evaluate_state}, "metadata": {"max_iterations": 1}}

def initial_state(problem, model, seed=0):
    return {"rho": 0.0}

def proposed_step(problem, model, state, iteration):
    return dict(state)

def baseline_solution(problem, model, seed=0):
    return {"rho": 0.0}

def method_solution(problem, model, method, seed=0):
    if method == "no_rho":
        state = initial_state(problem, model, seed)
        state["method"] = "no_rho"
        return state
    return baseline_solution(problem, model, seed)

def _compute_Pin(rho, M):
    M_eh = max(1, int(np.floor(rho * M)))
    return float(M_eh), M_eh

def channel_from_state(problem, state):
    return {}

def project_state(problem, state):
    return dict(state)

def evaluate_state(problem, model, state):
    _, M_eh = _compute_Pin(float(state.get("rho", 0.0)), 8)
    return {"status": "feasible", "objective": 1.0, "feasible": True, "constraint_violation": 0.0, "toy_metric": 1.0, "M_eh": M_eh}
"""
        solver_dir = run_dir / "phase2-4" / "solver"
        (solver_dir / "generated_plugin.py").write_text(textwrap.dedent(plugin), encoding="utf-8")

        result = validate_phase2_phase24_plugin_interfaces(solver_dir)

        self.assertFalse(result["ok"])
        self.assertIn("forces `M_eh`", "\n".join(result["errors"]))

    def test_interface_blocks_aggregate_covariance_sinr_antipattern(self) -> None:
        run_dir = self._make_run(
            {
                "problem_family": "toy_wireless",
                "canonical_config": {"system": {"Pmax": 1.0}},
                "required_outputs": {"scalar_metrics": ["toy_metric"]},
            }
        )
        plugin = """
import numpy as np

def build_model(problem, seed=0):
    return {"state_init": {}, "operators": {"channel_from_state": channel_from_state, "project_state": project_state, "evaluate_state": evaluate_state}, "metadata": {"max_iterations": 1}}

def initial_state(problem, model, seed=0):
    return {"Rx": np.eye(2)}

def proposed_step(problem, model, state, iteration):
    return dict(state)

def baseline_solution(problem, model, seed=0):
    return {"Rx": np.eye(2)}

def channel_from_state(problem, state):
    return {}

def project_state(problem, state):
    return dict(state)

def _herm(x):
    return x.conj().T

def evaluate_state(problem, model, state):
    Rx = state["Rx"]
    H = np.ones((2, 2))
    K = 2
    c2 = 0.0
    for k in range(K):
        hk = H[:, k:k+1]
        sig = float(np.real(_herm(hk) @ Rx @ hk).item())
        intf = 0.0
        for j in range(K):
            if j == k:
                continue
            hj = H[:, j:j+1]
            intf += float(np.real(_herm(hj) @ Rx @ hj).item())
        sinr = sig / (intf + 1e-3)
        c2 = max(c2, 1.0 - sinr)
    return {"status": "feasible", "objective": 1.0, "feasible": True, "constraint_violation": c2, "toy_metric": 1.0}
"""
        solver_dir = run_dir / "phase2-4" / "solver"
        (solver_dir / "generated_plugin.py").write_text(textwrap.dedent(plugin), encoding="utf-8")

        result = validate_phase2_phase24_plugin_interfaces(solver_dir)

        self.assertFalse(result["ok"])
        self.assertIn("aggregate covariance Rx", "\n".join(result["errors"]))

    def test_interface_blocks_effective_proxy_with_synthetic_k_minus_one_interference(self) -> None:
        run_dir = self._make_run(
            {
                "problem_family": "toy_wireless",
                "canonical_config": {"system": {"Pmax": 1.0}},
                "required_outputs": {"scalar_metrics": ["toy_metric"]},
            }
        )
        plugin = """
import numpy as np

def build_model(problem, seed=0):
    return {"state_init": {}, "operators": {"channel_from_state": channel_from_state, "project_state": project_state, "evaluate_state": evaluate_state}, "metadata": {"max_iterations": 1}}

def initial_state(problem, model, seed=0):
    return {"Rx": np.eye(2)}

def proposed_step(problem, model, state, iteration):
    return dict(state)

def baseline_solution(problem, model, seed=0):
    return {"Rx": np.eye(2)}

def channel_from_state(problem, state):
    return {}

def project_state(problem, state):
    return dict(state)

def _compute_effective_snr_proxy(Rx, K):
    rates = []
    for k in range(K):
        sig = 1.0 / K
        intf = 1.0 * (K - 1) / K
        sinr = sig / (intf + 1e-3)
        rates.append(sinr)
    return rates

def evaluate_state(problem, model, state):
    rates = _compute_effective_snr_proxy(state["Rx"], 2)
    return {"status": "feasible", "objective": float(sum(rates)), "feasible": True, "constraint_violation": 0.0, "toy_metric": 1.0}
"""
        solver_dir = run_dir / "phase2-4" / "solver"
        (solver_dir / "generated_plugin.py").write_text(textwrap.dedent(plugin), encoding="utf-8")

        result = validate_phase2_phase24_plugin_interfaces(solver_dir)

        self.assertFalse(result["ok"])
        self.assertIn("synthetic (K-1) interference", "\n".join(result["errors"]))

    def test_responsiveness_reports_nearly_flat_required_figure_metric_as_advice(self) -> None:
        run_dir = self._make_run(
            {
                "problem_family": "toy_wireless",
                "canonical_config": {"constraints": {"E_min_mW": 0.0}},
                "sweep_definitions": [
                    {
                        "id": "eh_requirement_sweep",
                        "canonical_path": "constraints.E_min_mW",
                        "quick_mode_values": [0.0, 5.0, 10.0],
                    }
                ],
                "figure_targets": [
                    {
                        "id": "figure_sum_rate_vs_eh_requirement",
                        "required_sweep": "eh_requirement_sweep",
                        "y_metric": "sum_rate_bpsHz",
                        "methods_to_run": ["proposed"],
                    }
                ],
            }
        )
        output_dir = run_dir / "phase2-4" / "solver" / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "validation_results.csv").write_text(
            "\n".join(
                [
                    "case_id,scenario_name,swept_param,swept_value,method,sum_rate_bpsHz,actual_used_constraints_E_min_mW",
                    "eh_requirement_sweep_0,eh_requirement_sweep,constraints.E_min_mW,0.0,proposed,10.00000,0.0",
                    "eh_requirement_sweep_1,eh_requirement_sweep,constraints.E_min_mW,5.0,proposed,10.00005,5.0",
                    "eh_requirement_sweep_2,eh_requirement_sweep,constraints.E_min_mW,10.0,proposed,10.00010,10.0",
                ]
            ),
            encoding="utf-8",
        )

        result = validate_phase24_experiment_responsiveness(run_dir)

        self.assertTrue(result["ok"], result)
        self.assertIn("too weakly responsive", "\n".join(result["repair_advice"]))

    def test_responsiveness_can_be_strict_when_debugging(self) -> None:
        run_dir = self._make_run(
            {
                "problem_family": "toy_wireless",
                "canonical_config": {"constraints": {"E_min_mW": 0.0}},
                "sweep_definitions": [
                    {
                        "id": "eh_requirement_sweep",
                        "canonical_path": "constraints.E_min_mW",
                        "quick_mode_values": [0.0, 5.0, 10.0],
                    }
                ],
                "figure_targets": [
                    {
                        "id": "figure_sum_rate_vs_eh_requirement",
                        "required_sweep": "eh_requirement_sweep",
                        "y_metric": "sum_rate_bpsHz",
                        "methods_to_run": ["proposed"],
                    }
                ],
            }
        )
        output_dir = run_dir / "phase2-4" / "solver" / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "validation_results.csv").write_text(
            "\n".join(
                [
                    "case_id,scenario_name,swept_param,swept_value,method,sum_rate_bpsHz,actual_used_constraints_E_min_mW",
                    "eh_requirement_sweep_0,eh_requirement_sweep,constraints.E_min_mW,0.0,proposed,10.00000,0.0",
                    "eh_requirement_sweep_1,eh_requirement_sweep,constraints.E_min_mW,5.0,proposed,10.00005,5.0",
                    "eh_requirement_sweep_2,eh_requirement_sweep,constraints.E_min_mW,10.0,proposed,10.00010,10.0",
                ]
            ),
            encoding="utf-8",
        )

        old_value = os.environ.get("WARA_PHASE24_STRICT_RESEARCH_GATE")
        try:
            os.environ["WARA_PHASE24_STRICT_RESEARCH_GATE"] = "1"
            result = validate_phase24_experiment_responsiveness(run_dir)
        finally:
            if old_value is None:
                os.environ.pop("WARA_PHASE24_STRICT_RESEARCH_GATE", None)
            else:
                os.environ["WARA_PHASE24_STRICT_RESEARCH_GATE"] = old_value

        self.assertFalse(result["ok"], result)
        self.assertIn("too weakly responsive", "\n".join(result["errors"]))

    def test_responsiveness_allows_flat_zero_violation_diagnostic(self) -> None:
        run_dir = self._make_run(
            {
                "problem_family": "toy_wireless",
                "canonical_config": {
                    "constraints": {"E_min_mW": 0.0},
                    "algorithm": {"primal_dual_tolerance": 1.0e-4},
                },
                "sweep_definitions": [
                    {
                        "id": "eh_requirement_sweep",
                        "canonical_path": "constraints.E_min_mW",
                        "quick_mode_values": [0.0, 5.0, 10.0],
                    }
                ],
                "figure_targets": [
                    {
                        "id": "figure_violation_vs_eh_requirement",
                        "required_sweep": "eh_requirement_sweep",
                        "y_metric": "constraint_violation_max",
                        "methods_to_run": ["proposed"],
                    }
                ],
            }
        )
        output_dir = run_dir / "phase2-4" / "solver" / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "validation_results.csv").write_text(
            "\n".join(
                [
                    "case_id,scenario_name,swept_param,swept_value,method,constraint_violation_max",
                    "eh_requirement_sweep_0,eh_requirement_sweep,constraints.E_min_mW,0.0,proposed,0.0",
                    "eh_requirement_sweep_1,eh_requirement_sweep,constraints.E_min_mW,5.0,proposed,0.0",
                    "eh_requirement_sweep_2,eh_requirement_sweep,constraints.E_min_mW,10.0,proposed,0.0",
                ]
            ),
            encoding="utf-8",
        )

        result = validate_phase24_experiment_responsiveness(run_dir)

        self.assertTrue(result["ok"], result)
        self.assertIn("reliability evidence", "\n".join(result["warnings"]))

    def test_validation_cases_consume_canonical_path_and_scout_values(self) -> None:
        run_dir = self._make_run(
            {
                "problem_family": "downlink_covariance_isacp",
                "canonical_config": {"requirements": {"E_dc_mW": 0.06}},
                "sweep_definitions": [
                    {
                        "id": "sweep_E_dc_req",
                        "variable": "E_m^DC",
                        "canonical_path": "requirements.E_dc_mW",
                        "scout_values": [0.03, 0.07, 0.11],
                        "paper_values": [0.025, 0.035, 0.045, 0.055],
                    }
                ],
            }
        )
        solver_dir = run_dir / "phase2-4" / "solver"

        old_path = list(sys.path)
        old_problem_data = sys.modules.pop("problem_data", None)
        try:
            sys.path.insert(0, str(solver_dir))
            spec = importlib.util.spec_from_file_location("validation_cases_under_test", solver_dir / "validation_cases.py")
            self.assertIsNotNone(spec)
            module = importlib.util.module_from_spec(spec)
            assert spec is not None and spec.loader is not None
            spec.loader.exec_module(module)

            cases = module.make_validation_cases(solver_dir / "validation_plan.yaml")
        finally:
            if old_problem_data is not None:
                sys.modules["problem_data"] = old_problem_data
            else:
                sys.modules.pop("problem_data", None)
            sys.path[:] = old_path

        self.assertEqual(len(cases), 4)
        self.assertEqual([case.swept_param for case in cases[1:]], ["requirements.E_dc_mW"] * 3)
        self.assertEqual([case.swept_value for case in cases[1:]], [0.03, 0.07, 0.11])
        self.assertEqual([case.get("requirements.E_dc_mW") for case in cases[1:]], [0.03, 0.07, 0.11])

    def test_validation_cases_apply_context_overrides_linked_paths_and_seed_cap(self) -> None:
        run_dir = self._make_run(
            {
                "problem_family": "toy_wireless",
                "canonical_config": {
                    "uncertainty": {"delta_h": 0.01, "delta_g": 0.01},
                    "constraints": {"P_max_mW": 10.0},
                },
                "research_evidence_contract": {
                    "two_pass_policy": {"scout_seeds_per_point": 5},
                },
                "sweep_definitions": [
                    {
                        "id": "sweep_delta",
                        "canonical_path": "uncertainty.delta_h",
                        "scout_values": [0.02, 0.04],
                        "context_overrides": {"constraints.P_max_mW": 20.0},
                        "linked_paths": [{"canonical_path": "uncertainty.delta_g", "same_value": True}],
                    }
                ],
            }
        )
        solver_dir = run_dir / "phase2-4" / "solver"

        old_path = list(sys.path)
        old_problem_data = sys.modules.pop("problem_data", None)
        old_cap = os.environ.get("WARA_PHASE24_QUICK_SEED_CAP")
        try:
            os.environ["WARA_PHASE24_QUICK_SEED_CAP"] = "2"
            sys.path.insert(0, str(solver_dir))
            spec = importlib.util.spec_from_file_location("validation_cases_under_test_context", solver_dir / "validation_cases.py")
            self.assertIsNotNone(spec)
            module = importlib.util.module_from_spec(spec)
            assert spec is not None and spec.loader is not None
            spec.loader.exec_module(module)

            cases = module.make_validation_cases(solver_dir / "validation_plan.yaml")
        finally:
            if old_cap is None:
                os.environ.pop("WARA_PHASE24_QUICK_SEED_CAP", None)
            else:
                os.environ["WARA_PHASE24_QUICK_SEED_CAP"] = old_cap
            if old_problem_data is not None:
                sys.modules["problem_data"] = old_problem_data
            else:
                sys.modules.pop("problem_data", None)
            sys.path[:] = old_path

        self.assertEqual(len(cases), 5)
        sweep_cases = cases[1:]
        self.assertEqual([getattr(case, "seed") for case in sweep_cases], [0, 1, 0, 1])
        self.assertEqual([case.get("constraints.P_max_mW") for case in sweep_cases], [20.0] * 4)
        self.assertEqual([case.get("uncertainty.delta_g") for case in sweep_cases], [0.02, 0.02, 0.04, 0.04])

    def test_validation_cases_cap_phase24_quick_sweep_values(self) -> None:
        run_dir = self._make_run(
            {
                "problem_family": "toy_wireless",
                "canonical_config": {"constraints": {"P_max_mW": 10.0}},
                "sweep_definitions": [
                    {
                        "id": "sweep_power",
                        "canonical_path": "constraints.P_max_mW",
                        "scout_values": [10.0, 12.0, 14.0, 16.0, 18.0],
                    }
                ],
            }
        )
        solver_dir = run_dir / "phase2-4" / "solver"

        old_path = list(sys.path)
        old_problem_data = sys.modules.pop("problem_data", None)
        old_cap = os.environ.get("WARA_PHASE24_QUICK_VALUES_PER_SWEEP_CAP")
        try:
            os.environ["WARA_PHASE24_QUICK_VALUES_PER_SWEEP_CAP"] = "2"
            sys.path.insert(0, str(solver_dir))
            spec = importlib.util.spec_from_file_location("validation_cases_under_test_value_cap", solver_dir / "validation_cases.py")
            self.assertIsNotNone(spec)
            module = importlib.util.module_from_spec(spec)
            assert spec is not None and spec.loader is not None
            spec.loader.exec_module(module)

            cases = module.make_validation_cases(solver_dir / "validation_plan.yaml")
        finally:
            if old_cap is None:
                os.environ.pop("WARA_PHASE24_QUICK_VALUES_PER_SWEEP_CAP", None)
            else:
                os.environ["WARA_PHASE24_QUICK_VALUES_PER_SWEEP_CAP"] = old_cap
            if old_problem_data is not None:
                sys.modules["problem_data"] = old_problem_data
            else:
                sys.modules.pop("problem_data", None)
            sys.path[:] = old_path

        self.assertEqual(len(cases), 3)
        self.assertEqual([case.swept_value for case in cases[1:]], [10.0, 12.0])

    def test_phase24_quick_runtime_caps_are_in_fixed_harness_and_adapter(self) -> None:
        run_validation_source = (ROOT / "phase2" / "scripts" / "phase24_harness_templates" / "generic_run_validation.py").read_text(encoding="utf-8")
        adapter_source = build_phase24_split_plugin_adapter()

        self.assertIn("WARA_PHASE24_QUICK_METHOD_CAP", run_validation_source)
        self.assertIn("WARA_PHASE24_QUICK_MAX_ITERATIONS", run_validation_source)
        self.assertIn("_is_quick_validation_mode", run_validation_source)
        self.assertIn("swept_canonical_path", run_validation_source)
        self.assertIn("WARA_PHASE24_QUICK_MAX_ITERATIONS", adapter_source)
        self.assertIn("_is_quick_validation_mode", adapter_source)
        self.assertIn('algorithm["max_iterations"]', adapter_source)

    def test_phase24_adapter_does_not_cap_iterations_in_paper_sweep_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            declared_iterations = 37
            (tmp_path / "generated_experiment_core.py").write_text(
                textwrap.dedent(
                    """
                    def build_model(problem, seed=0):
                        return {
                            "metadata": {"max_iterations": DECLARED_ITERATIONS},
                            "algorithm": {"max_iterations": DECLARED_ITERATIONS},
                            "state_init": {},
                        }

                    def channel_from_state(problem, state):
                        return {}

                    def project_state(problem, state):
                        return state

                    def evaluate_state(problem, model, state):
                        return {"objective": 1.0, "feasible": True}

                    def initial_state(problem, model, seed=0):
                        return {}

                    def proposed_step(problem, model, state, iteration):
                        return {"iteration": iteration + 1}

                    def baseline_solution(problem, model, seed=0):
                        return {"method": "baseline"}
                    """
                ).replace("DECLARED_ITERATIONS", str(declared_iterations)),
                encoding="utf-8",
            )
            (tmp_path / "generated_plugin.py").write_text(build_phase24_split_plugin_adapter(), encoding="utf-8")
            old_path = list(sys.path)
            old_modules = {name: sys.modules.get(name) for name in ("generated_experiment_core", "generated_plugin_under_test")}
            old_env = {
                name: os.environ.get(name)
                for name in ("WARA_PHASE24_QUICK_MAX_ITERATIONS", "WARA_PHASE25_SWEEP_TIER", "WCL_PHASE25_SWEEP_TIER", "WARA_RUN_MODE")
            }
            try:
                sys.path.insert(0, str(tmp_path))
                spec = importlib.util.spec_from_file_location("generated_plugin_under_test", tmp_path / "generated_plugin.py")
                self.assertIsNotNone(spec)
                module = importlib.util.module_from_spec(spec)
                assert spec is not None and spec.loader is not None
                spec.loader.exec_module(module)

                os.environ["WARA_PHASE24_QUICK_MAX_ITERATIONS"] = "3"
                os.environ.pop("WARA_PHASE25_SWEEP_TIER", None)
                os.environ.pop("WCL_PHASE25_SWEEP_TIER", None)
                os.environ.pop("WARA_RUN_MODE", None)
                quick_model = module.build_model(object(), seed=0)
                self.assertEqual(quick_model["metadata"]["max_iterations"], 3)
                self.assertEqual(quick_model["algorithm"]["max_iterations"], 3)

                os.environ["WARA_PHASE25_SWEEP_TIER"] = "paper"
                os.environ["WARA_RUN_MODE"] = "paper_validation"
                paper_model = module.build_model(object(), seed=0)
                self.assertEqual(paper_model["metadata"]["max_iterations"], declared_iterations)
                self.assertEqual(paper_model["algorithm"]["max_iterations"], declared_iterations)
            finally:
                sys.path[:] = old_path
                for name, value in old_modules.items():
                    if value is None:
                        sys.modules.pop(name, None)
                    else:
                        sys.modules[name] = value
                for name, value in old_env.items():
                    if value is None:
                        os.environ.pop(name, None)
                    else:
                        os.environ[name] = value

    def test_phase24_gate_rejects_generated_core_iteration_cap_below_plan(self) -> None:
        validation_plan = textwrap.dedent(
            """
            canonical_config:
              algorithm:
                max_iterations: 100
            """
        )
        generated_core = "max_iter_cfg = 100\nmax_iter = max(1, min(max_iter_cfg, 60))\n"
        errors = _phase24_iteration_cap_mismatch_errors(generated_core, validation_plan)

        self.assertEqual(len(errors), 1)
        self.assertIn("cap 60", errors[0])
        self.assertIn("declared max_iterations is 100", errors[0])

        compliant_core = "max_iter_cfg = 100\nmax_iter = max(1, max_iter_cfg)\n"
        self.assertEqual(_phase24_iteration_cap_mismatch_errors(compliant_core, validation_plan), [])

    def test_run_validation_synthesizes_constraint_violation_from_specific_violation_fields(self) -> None:
        import types

        template_path = SCRIPTS_DIR / "phase24_harness_templates" / "generic_run_validation.py"
        spec = importlib.util.spec_from_file_location("generic_run_validation_under_test", template_path)
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        old_modules = {name: sys.modules.get(name) for name in ("generated_plugin", "problem_data", "validation_cases")}
        try:
            generated_plugin = types.ModuleType("generated_plugin")
            generated_plugin.baseline_solution = lambda *args, **kwargs: {}
            generated_plugin.build_model = lambda *args, **kwargs: {}
            generated_plugin.evaluate_state = lambda *args, **kwargs: {}
            generated_plugin.initial_state = lambda *args, **kwargs: {}
            generated_plugin.proposed_step = lambda *args, **kwargs: {}
            problem_data = types.ModuleType("problem_data")
            problem_data.ProblemData = object
            problem_data.SolverResult = object
            problem_data.result_to_dict = lambda value: value
            problem_data.save_csv = lambda *args, **kwargs: None
            problem_data.save_json = lambda *args, **kwargs: None
            validation_cases = types.ModuleType("validation_cases")
            validation_cases.make_validation_cases = lambda *args, **kwargs: []
            sys.modules["generated_plugin"] = generated_plugin
            sys.modules["problem_data"] = problem_data
            sys.modules["validation_cases"] = validation_cases
            spec.loader.exec_module(module)
        finally:
            for name, old_value in old_modules.items():
                if old_value is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = old_value

        metrics = module._normalize_metric_aliases(
            {
                "objective": 1.0,
                "feasible": True,
                "toy_metric": 1.0,
                "power_violation_W": 0.02,
                "unit_modulus_violation": 0.001,
            },
            iterations=1,
            elapsed=0.1,
            trace=[],
        )

        self.assertAlmostEqual(metrics["max_constraint_violation"], 0.02)
        self.assertAlmostEqual(metrics["constraint_violation_max"], 0.02)
        self.assertAlmostEqual(metrics["constraint_violation"], 0.02)

    def test_numerical_runtime_warning_report_blocks_serious_numpy_warnings(self) -> None:
        report = _phase24_numerical_runtime_warning_report(
            {
                "phase24_validation_stderr": "\n".join(
                    [
                        "/tmp/generated_experiment_core.py:137: RuntimeWarning: invalid value encountered in matmul",
                        "  S += Ck.conj().T @ Ck",
                        "/tmp/generated_experiment_core.py:141: RuntimeWarning: overflow encountered in multiply",
                        "  q = q * scale",
                    ]
                )
            }
        )

        self.assertFalse(report["ok"])
        self.assertEqual(len(report["findings"]), 2)
        self.assertIn("invalid value encountered in matmul", "\n".join(report["errors"]))
        self.assertIn("overflow encountered in multiply", "\n".join(report["errors"]))

    def test_numerical_runtime_warning_report_ignores_external_cvxpy_solver_warning(self) -> None:
        report = _phase24_numerical_runtime_warning_report(
            {
                "phase24_validation_stderr": "\n".join(
                    [
                        "/Users/user/site-packages/cvxpy/problems/problem.py:1539: UserWarning: Solution may be inaccurate.",
                        "  warnings.warn(",
                        "/Users/user/site-packages/cvxpy/atoms/elementwise/log.py:35: RuntimeWarning: invalid value encountered in log",
                        "  return np.log(values[0])",
                    ]
                )
            }
        )

        self.assertTrue(report["ok"])
        self.assertFalse(report["findings"])
        self.assertTrue(report["ignored_external_solver_warnings"])

    def test_numerical_runtime_warning_failure_allows_repair(self) -> None:
        self.assertTrue(_phase24_validation_allows_repair({"status": "numerical_runtime_warning_failed"}))

    def test_basic_evidence_quality_uses_declared_solver_tolerance(self) -> None:
        run_dir = self._make_run(
            {
                "problem_family": "downlink_covariance_isacp",
                "canonical_config": {"algorithm": {"primal_dual_tolerance": 1.0e-4}},
            }
        )
        output_dir = run_dir / "phase2-4" / "solver" / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "validation_results.csv").write_text(
            "\n".join(
                [
                    "case_id,method,status,objective,feasible,constraint_violation_max",
                    "canonical,proposed,ok,1.0,True,0.00075",
                ]
            ),
            encoding="utf-8",
        )

        result = validate_phase24_basic_evidence_quality(run_dir)

        self.assertTrue(result["ok"], result)
        self.assertEqual(result["num_feasible_proposed_rows"], 1)
        self.assertGreaterEqual(result["quality_tolerance"], 0.001)

    def test_basic_evidence_quality_uses_declared_eps_feas_tolerance(self) -> None:
        run_dir = self._make_run(
            {
                "problem_family": "downlink_covariance_isacp",
                "canonical_config": {"algorithm": {"eps_feas": 1.0e-5}},
            }
        )
        output_dir = run_dir / "phase2-4" / "solver" / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "validation_results.csv").write_text(
            "\n".join(
                [
                    "case_id,method,status,objective,feasible,constraint_violation_max",
                    "canonical,proposed,ok,1.0,True,0.00003",
                ]
            ),
            encoding="utf-8",
        )

        result = validate_phase24_basic_evidence_quality(run_dir)

        self.assertTrue(result["ok"], result)
        self.assertEqual(result["num_feasible_proposed_rows"], 1)
        self.assertGreaterEqual(result["quality_tolerance"], 0.0001)

    def test_basic_evidence_quality_reports_dormant_proposed_mechanism_as_advice(self) -> None:
        run_dir = self._make_run(
            {
                "problem_family": "movable_antenna_wireless_optimization",
                "mechanism": "movable antenna position update",
                "canonical_config": {"algorithm": {"eps_feas": 1.0e-4}},
            }
        )
        output_dir = run_dir / "phase2-4" / "solver" / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "validation_results.csv").write_text(
            "\n".join(
                [
                    "case_id,method,status,objective,feasible,constraint_violation_max,used_position_update,position_step_norm",
                    "canonical,proposed,ok,1.0,True,0.0,False,0.0",
                    "sweep1,proposed,ok,1.2,True,0.0,False,0.0",
                ]
            ),
            encoding="utf-8",
        )

        result = validate_phase24_basic_evidence_quality(run_dir)

        self.assertTrue(result["ok"], result)
        self.assertTrue(
            any("dormant mechanism/update diagnostics" in error for error in result["repair_advice"]),
            result,
        )

    def test_basic_evidence_quality_treats_feasible_rate_as_diagnostic_only(self) -> None:
        run_dir = self._make_run(
            {
                "problem_family": "generic_wireless_optimization",
                "canonical_config": {"algorithm": {"eps_feas": 1.0e-4}},
            }
        )
        output_dir = run_dir / "phase2-4" / "solver" / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "validation_results.csv").write_text(
            "\n".join(
                [
                    "case_id,method,status,objective,feasible,constraint_violation_max",
                    "canonical,proposed,ok,1.0,True,0.0",
                    "stress1,proposed,ok,0.1,False,1.0",
                    "stress2,proposed,ok,0.2,False,1.0",
                    "stress3,proposed,ok,0.3,False,1.0",
                ]
            ),
            encoding="utf-8",
        )

        result = validate_phase24_basic_evidence_quality(run_dir)

        self.assertTrue(result["ok"], result)
        self.assertEqual(result["num_feasible_proposed_rows"], 1)
        self.assertAlmostEqual(result["feasible_rate"], 0.25)
        self.assertFalse(result["feasible_rate_gate_enabled"])
        self.assertFalse(any("too few feasible quick-validation rows" in error for error in result["repair_advice"]), result)

    def test_method_semantics_rejects_cvxpy_required_rows_that_use_non_cvxpy_substitute(self) -> None:
        run_dir = self._make_run(
            {
                "problem_family": "downlink_beamforming",
                "paper_evidence_contract": {
                    "compared_methods": [{"id": "proposed", "role": "proposed"}],
                    "figures": [
                        {
                            "id": "figure_1",
                            "required_sweep": "power",
                            "y_metric": "sum_rate_bpsHz",
                            "methods_to_run": ["proposed"],
                        }
                    ],
                },
            }
        )
        (run_dir / "phase2-4" / "phase24_method_fidelity_contract.json").write_text(
            json.dumps({"route_requires_cvxpy_solver_path": True}),
            encoding="utf-8",
        )
        outputs = run_dir / "phase2-4" / "solver" / "outputs"
        outputs.mkdir(parents=True, exist_ok=True)
        (outputs / "validation_results.csv").write_text(
            "\n".join(
                [
                    "case_id,method,status,feasible,sum_rate_bpsHz,cvxpy_solver_used,solver_status",
                    "canonical,proposed,ok,True,1.0,0,cvxpy_failed:solver_error;kkt_nu=1.0",
                ]
            ),
            encoding="utf-8",
        )

        result = validate_phase24_method_semantics(run_dir)

        self.assertFalse(result["ok"], result)
        self.assertIn("cvxpy_solver_used=0", "\n".join(result["errors"]))
        self.assertIn("solver-path substitution", "\n".join(result["errors"]))

    def test_method_semantics_accepts_cvxpy_required_rows_when_solver_path_succeeds(self) -> None:
        run_dir = self._make_run(
            {
                "problem_family": "downlink_beamforming",
                "paper_evidence_contract": {
                    "compared_methods": [{"id": "proposed", "role": "proposed"}],
                    "figures": [
                        {
                            "id": "figure_1",
                            "required_sweep": "power",
                            "y_metric": "sum_rate_bpsHz",
                            "methods_to_run": ["proposed"],
                        }
                    ],
                },
            }
        )
        (run_dir / "phase2-4" / "phase24_method_fidelity_contract.json").write_text(
            json.dumps({"route_requires_cvxpy_solver_path": True}),
            encoding="utf-8",
        )
        outputs = run_dir / "phase2-4" / "solver" / "outputs"
        outputs.mkdir(parents=True, exist_ok=True)
        (outputs / "validation_results.csv").write_text(
            "\n".join(
                [
                    "case_id,method,status,feasible,sum_rate_bpsHz,cvxpy_solver_used,solver_status",
                    "canonical,proposed,ok,True,1.0,1,optimal",
                ]
            ),
            encoding="utf-8",
        )

        result = validate_phase24_method_semantics(run_dir)

        self.assertTrue(result["ok"], result)

    def test_pilot_gain_accepts_paired_seed_positive_gain(self) -> None:
        run_dir = self._make_run(
            {
                "problem_family": "downlink_beamforming",
                "paper_evidence_contract": {
                    "primary_metric": {"name": "sum_rate_bpsHz", "higher_is_better": True},
                    "compared_methods": [
                        {"id": "proposed", "role": "proposed"},
                        {"id": "baseline", "role": "main_baseline"},
                    ],
                },
            }
        )
        outputs = run_dir / "phase2-4" / "solver" / "outputs"
        outputs.mkdir(parents=True, exist_ok=True)
        lines = ["case_id,seed,swept_param,swept_value,scenario_name,method,status,feasible,sum_rate_bpsHz"]
        for x_idx, swept_value in enumerate([1.0, 2.0, 3.0]):
            for seed in range(20):
                lines.append(f"case{x_idx},{seed},load,{swept_value},pilot,baseline,ok,True,1.0")
                lines.append(f"case{x_idx},{seed},load,{swept_value},pilot,proposed,ok,True,1.2")
        (outputs / "validation_results.csv").write_text("\n".join(lines), encoding="utf-8")

        result = validate_phase24_pilot_gain(run_dir)

        self.assertTrue(result["ok"], result)
        self.assertEqual(result["num_qualified_x_groups"], 3)
        self.assertGreater(result["pilot_median_relative_gain"], 0.0)

    def test_pilot_gain_rejects_nonpositive_paired_seed_gain(self) -> None:
        run_dir = self._make_run(
            {
                "problem_family": "downlink_beamforming",
                "paper_evidence_contract": {
                    "primary_metric": {"name": "sum_rate_bpsHz", "higher_is_better": True},
                    "compared_methods": [
                        {"id": "proposed", "role": "proposed"},
                        {"id": "baseline", "role": "main_baseline"},
                    ],
                },
            }
        )
        outputs = run_dir / "phase2-4" / "solver" / "outputs"
        outputs.mkdir(parents=True, exist_ok=True)
        lines = ["case_id,seed,swept_param,swept_value,scenario_name,method,status,feasible,sum_rate_bpsHz"]
        for x_idx, swept_value in enumerate([1.0, 2.0, 3.0]):
            for seed in range(20):
                lines.append(f"case{x_idx},{seed},load,{swept_value},pilot,baseline,ok,True,1.2")
                lines.append(f"case{x_idx},{seed},load,{swept_value},pilot,proposed,ok,True,1.0")
        (outputs / "validation_results.csv").write_text("\n".join(lines), encoding="utf-8")

        result = validate_phase24_pilot_gain(run_dir)

        self.assertFalse(result["ok"], result)
        self.assertIn("pilot median relative gain", "\n".join(result["errors"]))


if __name__ == "__main__":
    unittest.main()
