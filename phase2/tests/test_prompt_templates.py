from __future__ import annotations

import ast
import json
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from phase_runtime_impl import (
    _build_phase3_4_revision_plan,
    _build_phase3_4_full_paper_paths,
    _apply_phase3_4_evidence_adjustments,
    _phase24_concept_appears,
    _phase2_count_hard_mechanisms,
    _normalize_phase25_refined_sweep_plan,
    align_phase25_plan_with_phase24_contract,
    _extract_latex_issue_summary,
    analyze_latex_equation_line_format,
    analyze_latex_overfull_boxes,
    analyze_phase3_2_cross_topic_contamination,
    analyze_phase3_reformulation_repeats_system_model,
    build_phase3_2_dynamic_latex_table,
    build_phase3_2_numerical_results_prompt,
    build_phase3_3_abstract_conclusion_prompt,
    build_phase3_4_introduction_prompt,
    build_phase3_5_review_routing_decision,
    build_phase3_figure_diagram_image_prompt,
    build_phase3_figure_diagram_spec_prompt,
    build_phase3_figure_direct_image_prompt,
    build_default_phase3_figure_diagram_spec,
    build_phase2_phase1_latex_prompt,
    build_phase2_phase2_prompt,
    build_phase2_phase3_latex_prompt,
    build_phase2_phase3_prompt,
    build_phase3_1_writing_prompt,
    build_phase3_4_review_prompt,
    build_phase3_5_revision_prompt,
    build_phase3_5_post_revision_review_prompt,
    compact_short_split_equations,
    compact_long_sinr_fraction_equations,
    _apply_phase3_6_deterministic_intro_fixes,
    _apply_phase3_6_deterministic_technical_fixes,
    _phase3_6_validate_revised_section,
    build_phase3_4_reference_check_prompt,
    build_phase2_phase24_plugin_prompt,
    build_phase2_phase24_validation_prompt,
    build_phase25_sweep_refiner_prompt,
    enforce_phase3_2_axis_value_consistency,
    enforce_phase3_2_plotted_method_definitions,
    filter_phase3_2_method_naming_summary_for_plotted_methods,
    find_phase3_3_abstract_notation_issues,
    find_phase3_3_conclusion_notation_issues,
    infer_phase3_section_title,
    latex_math_contract_issue_summary,
    normalize_phase2_phase1_mathematical_contract,
    normalize_phase24_validation_plan_yaml,
    render_phase3_figure_diagram_from_spec,
    render_phase3_2_setup_paragraph,
    sanitize_phase3_1_system_problem_snippet,
    sanitize_phase3_latex_snippet,
    sanitize_phase3_2_numerical_results_tex,
    sanitize_phase3_3_abstract_tex,
    sanitize_phase3_3_conclusion_tex,
    validate_phase2_phase1_contract,
    validate_phase2_phase1_mathematical_contract_schema,
    validate_phase2_phase2_contract,
    validate_phase2_phase3_contract,
    validate_phase3_figure_assets,
    validate_phase3_2_paper_evidence_gate,
)
from phase_runtime.phase24_plan import sanitize_phase24_validation_plan_yaml
from phase_runtime.phase24_plugin_generation import (
    _phase24_should_stream_llm,
    _phase24_synthesize_solver_readme_from_benchmark_plan,
    extract_phase24_priority_feedback,
)
from phase_runtime.prompt_templates import render_prompt_template
from phase25_analysis import _safe_display, build_per_case_comparison, check_data_sufficiency, compute_relative_gain, run_monte_carlo_check, write_caption_files
from phase_runtime.phase3_3_sections import _prepare_full_paper_preview_inputs, analyze_phase3_3_abstract_structure
from phase_runtime.phase3_3_sections import _phase3_3_assert_no_internal_paper_terms, _phase3_3_assert_required_section_not_empty
from phase_runtime.phase3_4_references import (
    _build_bibtex_from_reference,
    analyze_phase3_4_intro_orphan_argument_paragraphs,
    analyze_phase3_4_introduction_content_quality,
    analyze_phase3_4_full_paper_abbreviations,
    analyze_phase3_4_full_paper_abbreviations_from_phase_dir,
    analyze_phase3_4_forbidden_terms,
    build_phase3_4_reference_strategy,
    ensure_phase3_4_minimum_intro_words,
    build_phase3_4_technical_citation_prompt,
    build_phase3_4_final_reference_count_contract,
    build_reference_quality_report,
    build_curated_bibliography,
    dedupe_phase3_4_references_by_identity,
    format_ieee_paper_title,
    normalize_bib_address,
    normalize_phase3_4_introduction_paragraphs,
    normalize_bibtex_author_list,
    paper_title_quality_issues,
    phase3_4_reference_is_final_usable,
    phase3_4_supplemental_reference_candidates,
    resolve_paper_title,
    sanitize_phase3_4_preview_section_tex,
    sanitize_phase3_4_introduction_tex,
    select_reference_pool,
    validate_phase3_4_technical_citation_only_revision,
)
from phase_runtime.phase3_5_review import (
    _append_phase3_4_deterministic_gate_issues,
    _phase3_5_apply_common_abbreviation_repairs,
    _phase3_5_apply_full_paper_abbreviation_repairs,
    _phase3_5_contract_scope_check,
    _phase3_6_post_review_unresolved_items,
    _phase3_4_deterministic_full_paper_gate,
    _phase3_4_ieee_style_and_technical_audit,
    _phase3_6_align_plotted_method_claim_text,
    _phase3_6_scope_exact_wmmse_language,
    _phase3_6_split_long_minimizer_lists,
    _phase3_6_split_long_one_line_equations,
    validate_bib_file_reference_flow,
)
from phase_runtime.topic_guardrails import build_wireless_feasibility_guardrail, detect_topic_features


class Phase2PromptTemplateTests(unittest.TestCase):
    def test_phase24_openai_codegen_streaming_defaults_off(self) -> None:
        self.assertFalse(_phase24_should_stream_llm("openai-gpt-5.5", "WARA_TEST_STREAM_FLAG"))
        self.assertTrue(_phase24_should_stream_llm("deepseek-chat", "WARA_TEST_STREAM_FLAG"))

    def test_phase24_benchmark_readme_can_be_synthesized_from_valid_plan(self) -> None:
        readme = _phase24_synthesize_solver_readme_from_benchmark_plan(
            "# Benchmark Plan\n\n- `proposed`\n- `mrt`\n"
        )
        self.assertIn("validation_plan.yaml", readme)
        self.assertIn("Benchmark summary", readme)
        self.assertIn("proposed", readme)

    def test_phase24_priority_feedback_extracts_controller_repair_contract(self) -> None:
        long_prefix = "generic contract\n" * 1000
        blueprint = (
            long_prefix
            + "[Controller-enforced figure-axis repair contract]\n"
            + "aperture_side_length_sweep is forbidden as final-figure x-axis.\n"
            + "[Phase 2.4 experiment-design feedback from previous Phase 2.5 result verification]\n"
            + "other feedback"
        )

        priority = extract_phase24_priority_feedback(blueprint)

        self.assertIn("High-priority controller feedback", priority)
        self.assertIn("aperture_side_length_sweep", priority)
        self.assertIn("forbidden as final-figure x-axis", priority)

    def test_paper_title_prefers_phase1_handoff_and_uses_ieee_title_case(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            phase_dir = run_dir / "phase3-5"
            phase_dir.mkdir(parents=True)
            (run_dir / "phase1_handoff_manifest.json").write_text(
                json.dumps(
                    {
                        "final_title": (
                            "distributionally robust chance-constrained resource allocation "
                            "for integrated sensing, communication, and powering"
                        )
                    }
                ),
                encoding="utf-8",
            )
            title = resolve_paper_title(
                phase_dir,
                "integrated sensing, communication, and powering with robust resource allocation",
            )

        self.assertEqual(
            title,
            "Distributionally Robust Chance-Constrained Resource Allocation for Integrated Sensing, Communication, and Powering",
        )
        self.assertEqual(
            format_ieee_paper_title("swipt-enabled ris-assisted mimo resource allocation for 6g"),
            "SWIPT-Enabled RIS-Assisted MIMO Resource Allocation for 6G",
        )

    def test_paper_title_prefers_paper_facing_title_over_working_contract_title(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            phase_dir = run_dir / "phase3-5"
            phase_dir.mkdir(parents=True)
            (run_dir / "phase1_handoff_manifest.json").write_text(
                json.dumps(
                    {
                        "final_title": "Weighted Sum-Rate Beamfocusing for Multiuser Radiative Near-Field Communications",
                        "paper_title": "Near-Field Beamfocusing for Multiuser Downlink Communications",
                    }
                ),
                encoding="utf-8",
            )

            title = resolve_paper_title(phase_dir, "")

        self.assertEqual(title, "Near-Field Beamfocusing for Multiuser Downlink Communications")
        self.assertIn(
            "template_objective_method_scenario_title",
            paper_title_quality_issues(
                "Weighted Sum-Rate Beamfocusing for Multiuser Radiative Near-Field Communications",
                working_title="Weighted Sum-Rate Beamfocusing for Multiuser Radiative Near-Field Communications",
            ),
        )

    def test_paper_title_rejects_template_paper_title_and_keeps_working_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            phase_dir = run_dir / "phase3-5"
            phase_dir.mkdir(parents=True)
            (run_dir / "phase1_handoff_manifest.json").write_text(
                json.dumps(
                    {
                        "final_title": "Power-Constrained RIS Beamforming for Integrated Sensing and Communication",
                        "paper_title": "Weighted Sum-Rate Optimization for RIS-Assisted ISAC Under Power Constraints",
                    }
                ),
                encoding="utf-8",
            )

            title = resolve_paper_title(phase_dir, "")

        self.assertEqual(title, "Power-Constrained RIS Beamforming for Integrated Sensing and Communication")

    def test_phase3_5_deterministic_full_paper_gate_turns_visible_defects_into_blockers(self) -> None:
        gate = _phase3_4_deterministic_full_paper_gate(
            reference_quality_report={"total_references": 1},
            full_paper_abbreviation_report={"undefined_abbreviations": ["ISAC", "PSD"]},
            compile_warnings_summary=["Overfull \\hbox (24.5pt too wide) in paragraph at lines 10--12"],
            final_figure_check={
                "ok": True,
                "included_figures": [{"path": "../phase2-5/figures/figure_1_draft.pdf", "exists": True}],
            },
            phase25_summary={"phase25_status": "quick_mode_only"},
        )

        p0_ids = {item["issue_id"] for item in gate["P0"]}
        p1_ids = {item["issue_id"] for item in gate["P1"]}
        self.assertFalse(gate["ok"])
        self.assertIn("P0-REF-COUNT", p0_ids)
        self.assertIn("P0-EXP-READY", p0_ids)
        self.assertIn("P1-ABBR-UNDEFINED", p1_ids)
        self.assertIn("P1-LATEX-OVERFULL", p1_ids)
        self.assertIn("P1-FIGURE-FINAL", p1_ids)

        payload = {
            "overall_score": 8.2,
            "recommendation": "ready_to_submit",
            "likely_reviewer_decision_estimate": "weak_accept",
            "dimension_scores": {
                "reference_quality": {"score": 8.0, "brief_reason": "looks fine"},
                "experiment_strength": {"score": 8.0, "brief_reason": "looks fine"},
            },
            "revision_plan": {"P0": [], "P1": [], "P2": []},
        }
        adjusted = _append_phase3_4_deterministic_gate_issues(payload, gate)

        self.assertEqual(adjusted["recommendation"], "major_revision_needed")
        self.assertEqual(adjusted["likely_reviewer_decision_estimate"], "reject")
        self.assertLessEqual(adjusted["overall_score"], 5.5)
        self.assertLessEqual(adjusted["dimension_scores"]["reference_quality"]["score"], 3.0)
        self.assertLessEqual(adjusted["dimension_scores"]["experiment_strength"]["score"], 3.0)

    def test_intro_orphan_argument_paragraphs_are_detected_and_repaired(self) -> None:
        intro = r"""\section{Introduction}
Programmable wireless surfaces help secure blocked links by creating controllable propagation paths. The resulting design is difficult because the same reflected path carries useful and leaked signals.

Prior work has optimized active and passive beamforming for rate and security objectives \cite{a,b}. However, these formulations do not fully expose how the controllable resource should be redistributed when the robust margin is user-limited.

Related reconfigurable-array studies show that placement choices can alter beamforming behavior under different service assumptions \cite{c}.

In this letter, we study a robust surface-aided downlink design. The main contributions are summarized as follows:

\begin{itemize}
\item We formulate the coupled robust design problem.
\end{itemize}

The remainder of this letter is organized as follows. Section~II presents the model.

\textit{Notation:} Boldface letters denote vectors and matrices.
"""

        issues = analyze_phase3_4_intro_orphan_argument_paragraphs(intro)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0]["paragraph_index"], 3)

        repaired, applied = _apply_phase3_6_deterministic_intro_fixes(intro)
        self.assertIn("P1-INTRO-ORPHAN-PARAGRAPH", {item["issue_id"] for item in applied})
        self.assertEqual(analyze_phase3_4_intro_orphan_argument_paragraphs(repaired), [])
        self.assertNotIn("\n\nRelated reconfigurable-array studies", repaired)

    def test_phase3_5_gate_blocks_orphan_intro_argument_paragraphs(self) -> None:
        full_paper = r"""\begin{abstract}
This letter studies robust wireless optimization.
\end{abstract}
\section{Introduction}
Programmable wireless surfaces help secure blocked links by creating controllable propagation paths. The resulting design is difficult because the same reflected path carries useful and leaked signals.

Prior work has optimized active and passive beamforming for rate and security objectives \cite{a,b}. However, these formulations do not fully expose how the controllable resource should be redistributed when the robust margin is user-limited.

Related reconfigurable-array studies show that placement choices can alter beamforming behavior under different service assumptions \cite{c}.

In this letter, we study a robust surface-aided downlink design. The main contributions are summarized as follows:
\begin{itemize}
\item We formulate the coupled robust design problem.
\end{itemize}
\section{System Model and Problem Formulation}
The system model is defined.
\section{Proposed Solution}
The proposed solution is described.
\section{Numerical Results}
In this section, we evaluate the reported metric.
\section{Conclusion}
This letter studied robust wireless optimization.
"""
        gate = _phase3_4_deterministic_full_paper_gate(
            reference_quality_report={"total_references": 12},
            full_paper_abbreviation_report={},
            compile_warnings_summary=[],
            final_figure_check={"ok": True, "included_figures": [{"path": "figure_1.pdf", "exists": True}]},
            phase25_summary={
                "phase25_status": "paper_minimum_ready",
                "figures": [
                    {"paper_ready": True, "y_metric": "secrecy_rate"},
                    {"paper_ready": True, "y_metric": "secrecy_rate"},
                ],
            },
            full_paper_tex=full_paper,
        )

        self.assertFalse(gate["ok"])
        self.assertIn("P1-INTRO-ORPHAN-PARAGRAPH", {item["issue_id"] for item in gate["P1"]})

    def test_phase3_5_abbreviation_repair_collapses_nested_definitions_and_lmi(self) -> None:
        text = (
            "The channel state information (channel state information (CSI)) model uses an LMI. "
            "The positive semidefinite (positive semidefinite (PSD))cone is used. "
            "A mixed SOCP-SDP is solved for a CSI-ball model. "
            "The notation is Hermitian positive semidefinite."
        )
        repaired, applied = _phase3_5_apply_common_abbreviation_repairs(text, {"CSI", "LMI", "PSD", "SOCP", "SDP"})

        self.assertIn("channel state information (CSI)", repaired)
        self.assertIn("linear matrix inequality (LMI)", repaired)
        self.assertIn("positive semidefinite (PSD)", repaired)
        self.assertIn("positive semidefinite (PSD) cone", repaired)
        self.assertIn("mixed second-order cone programming (SOCP) and semidefinite programming (SDP)", repaired)
        self.assertIn("Hermitian positive semidefinite (PSD)", repaired)
        self.assertNotIn("channel state information (channel state information", repaired)
        self.assertNotIn("positive semidefinite (positive semidefinite", repaired)
        self.assertIn("LMI", applied)

        hyphenated, _ = _phase3_5_apply_common_abbreviation_repairs("A CSI-ball model is used.", {"CSI"})
        self.assertIn("channel state information (CSI)-ball", hyphenated)

    def test_phase3_5_abbreviation_repair_defines_dynamic_common_optimization_terms(self) -> None:
        text = (
            "The MMSE receiver is refreshed before the WSR update, and the KKT form is used. "
            "The setting is perfect-CSI, the method is WMMSE-SCA, and it avoids SDR rank recovery."
        )
        repaired, applied = _phase3_5_apply_common_abbreviation_repairs(
            text,
            {"MMSE", "WSR", "KKT", "CSI", "SCA", "SDR"},
        )

        self.assertIn("minimum mean-square error (MMSE) receiver", repaired)
        self.assertIn("weighted sum rate (WSR) update", repaired)
        self.assertIn("Karush-Kuhn-Tucker (KKT) form", repaired)
        self.assertIn("perfect channel state information (CSI)", repaired)
        self.assertIn("WMMSE and successive convex approximation (SCA)", repaired)
        self.assertIn("semidefinite relaxation (SDR) rank recovery", repaired)
        self.assertEqual(set(applied), {"MMSE", "WSR", "KKT", "CSI", "SCA", "SDR"})

    def test_phase3_5_abbreviation_repair_handles_hyphenated_terms_and_qp(self) -> None:
        text = (
            "The log-SINR sum is handled by an FP/WMMSE-and-SCA method. "
            "The position block becomes a convex QP and the beamforming block is a convex QCQP."
        )
        repaired, applied = _phase3_5_apply_common_abbreviation_repairs(
            text,
            {"SINR", "SCA", "QP", "QCQP"},
        )

        self.assertIn("log signal-to-interference-plus-noise-ratio (SINR) sum", repaired)
        self.assertIn("FP/WMMSE and successive convex approximation (SCA) method", repaired)
        self.assertIn("convex quadratic program (QP)", repaired)
        self.assertIn("convex quadratically constrained quadratic program (QCQP)", repaired)
        self.assertNotIn("an successive", repaired)
        self.assertEqual(set(applied), {"SINR", "SCA", "QP", "QCQP"})

    def test_phase3_5_abbreviation_repair_defines_mixed_integer_conic_terms(self) -> None:
        text = (
            "The design is SINR-feasible. "
            "The feasibility test is represented by a zero-objective MISOCP. "
            "The fixed-target condition is written as an SOC inequality. "
            "\\item \\textbf{RZF-fixed}: Regularized-ZF fixed-cluster loading is compared."
        )
        repaired, applied = _phase3_5_apply_common_abbreviation_repairs(
            text,
            {"SINR", "MISOCP", "SOC", "RZF", "Regularized-ZF"},
        )

        self.assertIn("signal-to-interference-plus-noise ratio (SINR)-feasible", repaired)
        self.assertIn("zero-objective mixed-integer second-order cone program (MISOCP)", repaired)
        self.assertIn("a second-order cone (SOC) inequality", repaired)
        self.assertIn("RZF-fixed}: regularized zero-forcing fixed-cluster loading", repaired)
        self.assertNotIn("regularized zero-forcing (RZF)-fixed", repaired)
        self.assertEqual(set(applied), {"SINR", "MISOCP", "SOC", "Regularized-ZF"})

    def test_phase3_5_abbreviation_repair_avoids_algorithm_blocks(self) -> None:
        text = (
            "\\begin{algorithm}[!t]\n"
            "\\begin{algorithmic}[1]\n"
            "\\State Initialize with MRT scaled to the power budget.\n"
            "\\end{algorithmic}\n"
            "\\end{algorithm}\n"
            "The MRT direction is used only as an ablation in the figure discussion."
        )
        repaired, applied = _phase3_5_apply_common_abbreviation_repairs(text, {"MRT"})

        self.assertIn("\\State Initialize with MRT scaled", repaired)
        self.assertIn("maximum-ratio transmission (MRT) direction", repaired)
        self.assertEqual(applied, ["MRT"])

    def test_abbreviation_check_accepts_labeled_benchmark_definition(self) -> None:
        report = analyze_phase3_4_full_paper_abbreviations(
            {
                "numerical_results": (
                    "\\item \\textbf{MRT}: fixed-layout maximum-ratio transmission, "
                    "which uses channel-matched beams. The MRT benchmark is compared."
                )
            }
        )

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["undefined_abbreviations"], [])
        self.assertEqual(report["repeated_abbreviation_definitions"], [])

    def test_abbreviation_check_treats_hyphenated_method_label_as_single_token(self) -> None:
        report = analyze_phase3_4_full_paper_abbreviations(
            {
                "introduction": (
                    "Movable-antenna (MA) arrays are used. "
                    "For multiple-input single-output (MISO) downlinks, the MA-MISO model is adopted. "
                    "The organization paragraph mentions the maximum-ratio-transmission movable-antenna benchmark (MRT-MA). "
                    "The MRT-MA benchmark is discussed later."
                )
            }
        )

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["undefined_abbreviations"], [])
        self.assertEqual(report["repeated_abbreviation_definitions"], [])

    def test_abbreviation_check_accepts_suffixed_labeled_definition(self) -> None:
        report = analyze_phase3_4_full_paper_abbreviations(
            {
                "numerical_results": (
                    "\\item \\textbf{RZF-fixed}: regularized zero-forcing fixed-cluster loading. "
                    "The RZF-fixed curve is reported."
                )
            }
        )

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["undefined_abbreviations"], [])

    def test_abbreviation_check_flags_repeated_local_definition(self) -> None:
        report = analyze_phase3_4_full_paper_abbreviations(
            {
                "numerical_results": (
                    "\\item \\textbf{maximum-ratio transmission (MRT)}: "
                    "fixed-layout maximum-ratio transmission (MRT), which uses channel-matched beams."
                )
            }
        )

        self.assertFalse(report["ok"])
        self.assertEqual(report["undefined_abbreviations"], [])
        self.assertEqual(report["repeated_abbreviation_definitions"][0]["term"], "MRT")

    def test_phase3_5_contract_scope_allows_physical_eta_parameters(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            phase_dir = run_dir / "phase3-5"
            (run_dir / "phase2-1").mkdir(parents=True)
            phase_dir.mkdir(parents=True)
            (run_dir / "phase2-1" / "mathematical_contract.json").write_text(
                json.dumps(
                    {
                        "parameters": [{"symbol": r"\eta_k"}],
                        "derived_quantities": [],
                        "reformulation_only": [{"symbol": r"\lambda_k"}],
                    }
                ),
                encoding="utf-8",
            )
            (phase_dir / "system_model_problem_formulation_section.tex").write_text(
                r"$Q_k=\eta_k P_k$.",
                encoding="utf-8",
            )
            (phase_dir / "proposed_solution_section.tex").write_text(
                r"Use $\lambda_k$ only in the robust counterpart.",
                encoding="utf-8",
            )
            report = _phase3_5_contract_scope_check(run_dir, phase_dir)

        self.assertTrue(report["ok"])

    def test_phase3_5_contract_scope_does_not_match_reform_symbol_inside_latex_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            phase_dir = run_dir / "phase3-5"
            (run_dir / "phase2-1").mkdir(parents=True)
            phase_dir.mkdir(parents=True)
            (run_dir / "phase2-1" / "mathematical_contract.json").write_text(
                json.dumps(
                    {
                        "derived_quantities": [{"symbol": r"R_{\mathrm{wsr}}"}],
                        "reformulation_only": [{"symbol": "u_k"}],
                    }
                ),
                encoding="utf-8",
            )
            (phase_dir / "system_model_problem_formulation_section.tex").write_text(
                r"The utility is $R_{\mathrm{wsr}}=\sum_k \mu_k R_k$.",
                encoding="utf-8",
            )
            (phase_dir / "proposed_solution_section.tex").write_text(
                r"The auxiliary $u_k$ is updated in the reformulation.",
                encoding="utf-8",
            )

            report = _phase3_5_contract_scope_check(run_dir, phase_dir)

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["reformulation_symbols_found_in_proposed_solution"], ["u_k"])

    def test_phase3_5_gate_blocks_paper_ready_claim_with_only_one_final_figure(self) -> None:
        gate = _phase3_4_deterministic_full_paper_gate(
            reference_quality_report={"total_references": 12},
            full_paper_abbreviation_report={"undefined_abbreviations": []},
            compile_warnings_summary=[],
            final_figure_check={
                "ok": True,
                "included_figures": [{"path": "../phase2-5/figures/figure_1.pdf", "exists": True}],
            },
            phase25_summary={
                "phase25_status": "paper_minimum_ready",
                "figures": [{"figure_id": "figure_1", "paper_ready": True, "y_metric": "sum_rate_bpsHz"}],
            },
        )

        self.assertFalse(gate["ok"])
        self.assertIn("P0-FIGURE-COUNT", {item["issue_id"] for item in gate["P0"]})
        self.assertEqual(gate["non_diagnostic_paper_ready_figure_count"], 1)

    def test_phase3_5_style_audit_flags_black_box_technical_and_generic_writing(self) -> None:
        full_paper = r"""
        \begin{abstract}
        This problem has attracted significant attention. The proposed method is effective for \(P_{\rm tx}\).
        \end{abstract}
        \section{Introduction}
        Wireless optimization plays an important role in future networks.
        The first stream of work studies models. The second stream studies solvers. The third stream studies experiments.
        \section{System Model and Problem Formulation}
        The shared sensing covariance is placed in the SINR denominator as interference.
        \section{Proposed Solution}
        The selected ambiguity model gives the safe counterpart \Phi_m\ge 0. No rank recovery is needed.
        \section{Numerical Results}
        The verified registry shows curve data from the artifact.
        \section{Conclusion}
        The method improves \(P_{\rm tx}\).
        """

        audit = _phase3_4_ieee_style_and_technical_audit(full_paper)
        p0_ids = {item["issue_id"] for item in audit["P0"]}
        p1_ids = {item["issue_id"] for item in audit["P1"]}
        p2_ids = {item["issue_id"] for item in audit["P2"]}

        self.assertFalse(audit["ok"])
        self.assertNotIn("P0-TECH-PHI-BLACKBOX", p0_ids)
        self.assertIn("P2-TECH-PHI-ADVISORY", p2_ids)
        self.assertIn("P1-WRITING-GENERIC-IEEE", p1_ids)
        self.assertIn("P1-ABSTRACT-NOTATION", p1_ids)
        self.assertIn("P1-CONCLUSION-NOTATION", p1_ids)
        self.assertIn("P1-RESULTS-REPORTLIKE", p1_ids)

    def test_phase3_5_style_audit_accepts_results_opening_after_section_label(self) -> None:
        full_paper = r"""
        \section{Numerical Results}
        \label{sec:numerical_results}
        In this section, we evaluate the proposed method with $K=4$ users.
        """
        report = _phase3_4_ieee_style_and_technical_audit(full_paper)

        self.assertNotIn("P2-RESULTS-OPENING", {item["issue_id"] for item in report["P2"]})

    def test_topic_feature_detection_ignores_negated_mechanism_boundaries(self) -> None:
        features = detect_topic_features(
            "movable antennas",
            "not_controls: RIS coefficients, UAV trajectory, CSI uncertainty sets",
            "Uncertainty/reliability is inactive. No robust or chance constraint is part of the base contract.",
            "The downlink uses movable antenna coordinates and WMMSE precoding for weighted sum rate.",
        )

        self.assertIn("downlink_beamforming", features)
        self.assertIn("wmmse_declared", features)
        self.assertIn("specialized_mechanism", features)
        self.assertNotIn("ris", features)
        self.assertNotIn("robust_csi", features)

    def test_phase3_5_gate_caps_scores_for_technical_and_style_failures(self) -> None:
        gate = _phase3_4_deterministic_full_paper_gate(
            reference_quality_report={"total_references": 12},
            full_paper_abbreviation_report={"undefined_abbreviations": []},
            compile_warnings_summary=[],
            final_figure_check={"ok": True, "included_figures": []},
            phase25_summary={
                "phase25_status": "paper_minimum_ready",
                "figures": [
                    {"paper_ready": True, "y_metric": "sum_rate"},
                    {"paper_ready": True, "y_metric": "power"},
                ],
            },
            full_paper_tex=r"""
            \begin{abstract}The proposed method is effective for \(P_{\rm tx}\).\end{abstract}
            \section{System Model and Problem Formulation}
            \section{Proposed Solution}The safe counterpart is \Phi_m\ge0.
            \section{Numerical Results}The evidence package shows the curve data.
            \section{Conclusion}The method improves \(P_{\rm tx}\).
            """,
        )
        payload = {
            "overall_score": 8.8,
            "recommendation": "ready_to_submit",
            "dimension_scores": {
                "technical_correctness": {"score": 8.0},
                "theoretical_rigor": {"score": 8.0},
                "writing_quality": {"score": 8.0},
                "IEEE_style_and_format": {"score": 8.0},
            },
            "revision_plan": {"P0": [], "P1": [], "P2": []},
        }

        adjusted = _append_phase3_4_deterministic_gate_issues(payload, gate)

        self.assertEqual(adjusted["recommendation"], "major_revision_needed")
        self.assertEqual(adjusted["dimension_scores"]["technical_correctness"]["score"], 8.0)
        self.assertLessEqual(adjusted["dimension_scores"]["writing_quality"]["score"], 5.0)

    def test_phase3_4_reference_strategy_does_not_lower_requested_target(self) -> None:
        strategy = build_phase3_4_reference_strategy(
            verified_reference_bank=[
                {
                    "final_bib_key": "OnlyOne2025",
                    "verification_status": "verified_published",
                    "venue": "IEEE Wireless Communications Letters",
                    "final_title": "Only One Complete Reference",
                    "year": "2025",
                    "volume": "14",
                    "number": "1",
                    "pages": "1--5",
                    "month": "Jan.",
                    "doi": "10.1109/LWC.2025.0000001",
                    "source_type": "journal",
                    "included_in_final_bib": True,
                    "used_for_claims": ["background"],
                }
            ],
            introduction_facts={
                "result_constraints": {
                    "minimum_reference_target": 12,
                    "preferred_reference_target": 14,
                }
            },
        )

        self.assertEqual(strategy["minimum_reference_target"], 12)
        self.assertFalse(strategy["reference_count_gate"]["ok"])
        self.assertEqual(strategy["reference_count_gate"]["available_reference_count"], 1)

    def test_phase3_4_reference_pool_excludes_generic_or_offscope_references(self) -> None:
        context = "near-field XL-MIMO integrated sensing communication and powering beamforming optimization"
        entries = [
            {
                "key": "gupta2015survey",
                "title": "A Survey of 5G Network: Architecture and Emerging Technologies",
                "venue": "IEEE Access",
                "year": "2015",
            },
            {
                "key": "basar2019wireless",
                "title": "Wireless Communications Through Reconfigurable Intelligent Surfaces",
                "venue": "IEEE Access",
                "year": "2019",
            },
            {
                "key": "li2023nearfield",
                "title": "Near-Field Beamforming Optimization for Holographic XL-MIMO Multiuser Systems",
                "venue": "IEEE Transactions on Communications",
                "year": "2023",
            },
            {
                "key": "sun2025optimal",
                "title": "Optimal Beamforming for Multi-Functional Integrated Sensing, Communication, and Powering Systems",
                "venue": "Electronics Letters",
                "year": "2025",
            },
        ]

        selected = select_reference_pool(entries, context_text=context, focus_keys=[], max_items=10)
        keys = {item["key"] for item in selected}

        self.assertIn("li2023nearfield", keys)
        self.assertIn("sun2025optimal", keys)
        self.assertNotIn("gupta2015survey", keys)
        self.assertNotIn("basar2019wireless", keys)

    def test_phase3_4_final_reference_requires_complete_ieee_metadata(self) -> None:
        incomplete = {
            "final_bib_key": "SparseSeed2014",
            "final_title": "Sparse Seed Reference",
            "authors": "A. Author",
            "venue": "IEEE Journal on Selected Areas in Communications",
            "year": "2014",
            "source_type": "journal",
            "verification_status": "verified_published",
            "included_in_final_bib": True,
        }

        self.assertFalse(phase3_4_reference_is_final_usable(incomplete))
        report = build_reference_quality_report(final_references=[incomplete], citation_claim_map=[])
        self.assertFalse(report["ok"])
        self.assertTrue(report["metadata_blocking_errors"])

    def test_phase3_4_reference_bibtex_omits_conference_address(self) -> None:
        self.assertEqual(normalize_bib_address("Singapore, Singapore"), "Singapore")
        bib = _build_bibtex_from_reference(
            {
                "final_bib_key": "ConfRef2024",
                "final_title": "Conference Reference",
                "authors": "A. Author",
                "venue": "IEEE Global Communications Conference",
                "year": "2024",
                "pages": "10--15",
                "doi": "10.1109/GLOBE.2024.000001",
                "source_type": "conference",
                "verification_status": "verified_published",
                "included_in_final_bib": True,
                "address": "Singapore, Singapore",
            }
        )

        self.assertNotIn("address =", bib)
        self.assertNotIn("Singapore, Singapore", bib)

    def test_phase3_4_reference_bank_dedupes_same_doi_and_title_before_llm(self) -> None:
        duplicate_refs = [
            {
                "final_bib_key": "arxivVersion",
                "final_title": "Integrated Wireless Optimization",
                "authors": "A. Author",
                "venue": "IEEE Commun. Mag.",
                "year": "2025",
                "volume": "63",
                "number": "1",
                "pages": "10--20",
                "month": "jan",
                "doi": "10.1109/MCOM.2025.000001",
                "source_type": "journal",
                "verification_status": "replaced_by_published_version",
                "included_in_final_bib": True,
            },
            {
                "final_bib_key": "publishedVersion",
                "final_title": "Integrated Wireless Optimization",
                "authors": "A. Author",
                "venue": "IEEE Commun. Mag.",
                "year": "2025",
                "volume": "63",
                "number": "1",
                "pages": "10--20",
                "month": "jan",
                "doi": "10.1109/MCOM.2025.000001",
                "source_type": "journal",
                "verification_status": "verified_published",
                "included_in_final_bib": True,
            },
        ]

        deduped, notes = dedupe_phase3_4_references_by_identity(duplicate_refs)
        self.assertEqual([item["final_bib_key"] for item in deduped], ["publishedVersion"])
        self.assertEqual(notes[0]["removed_key"], "arxivVersion")

        bib, missing, entries = build_curated_bibliography(duplicate_refs, ["arxivVersion", "publishedVersion"])
        self.assertFalse(missing)
        self.assertEqual(len(entries), 1)
        self.assertEqual(bib.count("@article"), 1)

        report = build_reference_quality_report(final_references=duplicate_refs, citation_claim_map=[])
        self.assertFalse(report["ok"])
        self.assertTrue(report["duplicate_arxiv_published_version_warnings"])

    def test_phase24_validation_plan_normalizer_extracts_metric_names_from_specs(self) -> None:
        yaml_text = """
canonical_config:
  power:
    Pmax_W: 1.0
sweep_definitions:
  sweep_power:
    variable: power.Pmax_W
    values: [0.5, 1.0, 1.5]
required_outputs:
  scalar_metrics:
    - name: certified_target_rate_bpsHz
      role: primary_paper_kpi
    - "{'name': 'min_certified_state_rate_bpsHz', 'role': 'objective_consistency_check'}"
paper_evidence_contract:
  primary_metric:
    name: certified_target_rate_bpsHz
  compared_methods:
    - id: proposed
      display_name_short: Proposed
    - id: baseline
      display_name_short: Baseline
  required_result_columns:
    - method
    - seed
    - "{'name': 'nominal_selected_rate_bpsHz', 'role': 'physical_secondary_kpi'}"
  figures:
    - id: figure_1
      required_sweep: sweep_power
      y_metric:
        name: certified_target_rate_bpsHz
      required_metrics:
        - "{'name': 'sum_power_W', 'role': 'resource_diagnostic'}"
"""
        normalized = yaml.safe_load(normalize_phase24_validation_plan_yaml(yaml_text))
        scalar_metrics = normalized["required_outputs"]["scalar_metrics"]
        required_columns = normalized["paper_evidence_contract"]["required_result_columns"]
        figure_metrics = normalized["paper_evidence_contract"]["figures"][0]["required_metrics"]

        self.assertIn("certified_target_rate_bpsHz", scalar_metrics)
        self.assertIn("min_certified_state_rate_bpsHz", scalar_metrics)
        self.assertIn("nominal_selected_rate_bpsHz", required_columns)
        self.assertTrue(all("{'name'" not in item for item in scalar_metrics + required_columns + figure_metrics))

    def test_phase3_4_supplemental_references_are_doi_backed_background_not_topic_replacement(self) -> None:
        refs = phase3_4_supplemental_reference_candidates(
            "6G wireless movable antenna beamforming optimization with antenna position control",
            min_items=12,
        )
        keys = {item["key"] for item in refs}

        self.assertGreaterEqual(len(refs), 12)
        self.assertIn("SaadBennisChen2020Vision6G", keys)
        self.assertIn("ZhuMaZhang2024MovableAntennaModeling", keys)
        self.assertTrue(all(str(item.get("doi", "")).startswith("10.1109/") for item in refs))
        self.assertTrue(all(str(item.get("rationale", "")).strip() for item in refs))

    def test_phase_runtime_prompts_are_external_yaml_templates(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        phase2_prompt_root = repo_root / "phase2" / "prompts"
        phase3_prompt_root = repo_root / "phase3" / "prompts"

        def prompt_path(relative_path: str) -> Path:
            root = phase3_prompt_root if relative_path.startswith("phase3_") else phase2_prompt_root
            return root / relative_path

        expected_templates = [
            "shared/wireless_feasibility_guardrail.prompt.yaml",
            "phase2_1/system_model_problem.prompt.yaml",
            "phase2_1/latex_system_problem.prompt.yaml",
            "phase2_1/latex_system_problem_repair.prompt.yaml",
            "phase2_2/convexity_reformulation.prompt.yaml",
            "phase2_3/algorithm_design.prompt.yaml",
            "phase2_3/latex_solution.prompt.yaml",
            "phase2_3/latex_solution_repair.prompt.yaml",
            "phase3_1/technical_writing.prompt.yaml",
            "phase3_1/technical_writing_repair.prompt.yaml",
            "phase2_4/solver_package.prompt.yaml",
            "phase2_4/validation_plan.prompt.yaml",
            "phase2_4/benchmark_readme.prompt.yaml",
            "phase2_4/code_file.prompt.yaml",
            "phase2_4/plugin_core.prompt.yaml",
            "phase2_4/plugin_repair.prompt.yaml",
            "phase2_5/experiment_planner.prompt.yaml",
            "phase2_5/result_writer.prompt.yaml",
            "phase2_5/sweep_refiner.prompt.yaml",
            "phase3_2/method_mechanism_summary.prompt.yaml",
            "phase3_2/paper_objective_summary.prompt.yaml",
            "phase3_2/figure_to_claim_summary.prompt.yaml",
            "phase3_2/numerical_results.prompt.yaml",
            "phase3_3/abstract_conclusion.prompt.yaml",
            "phase3_4/introduction.prompt.yaml",
            "phase3_4/reference_check.prompt.yaml",
            "phase3_4/technical_citation.prompt.yaml",
            "phase3_figure/diagram_spec.prompt.yaml",
            "phase3_figure/diagram_image.prompt.yaml",
            "phase3_figure/direct_image.prompt.yaml",
            "phase3_5/review_rewrite.prompt.yaml",
            "phase3_5/final_review.prompt.yaml",
            "phase3_6/final_revision.prompt.yaml",
            "phase3_6/post_revision_review.prompt.yaml",
            "agents/experiment_agent.prompt.yaml",
        ]
        prompt_data_files = [
            "phase2_4/code_file_rules.yaml",
        ]
        for rel_path in expected_templates:
            path = prompt_path(rel_path)
            self.assertTrue(path.exists(), f"missing prompt template: {path}")
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            self.assertIsInstance(payload, dict)
            self.assertIsInstance(payload.get("variables"), list)
            self.assertIsInstance(payload.get("template"), str)
            self.assertGreater(len(payload["template"].strip()), 40)
        for rel_path in prompt_data_files:
            path = prompt_path(rel_path)
            self.assertTrue(path.exists(), f"missing prompt data file: {path}")
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            self.assertIsInstance(payload.get("rules"), dict)
            self.assertIn("problem_data.py", payload["rules"])

        runtime_files = [
            SCRIPTS_DIR / "phase_runtime" / "phase21_23_foundation.py",
            SCRIPTS_DIR / "phase_runtime" / "phase24_plugin_generation.py",
            SCRIPTS_DIR / "phase_runtime" / "phase25_planning.py",
            SCRIPTS_DIR / "phase_runtime" / "phase3_2_writing.py",
            SCRIPTS_DIR / "phase_runtime" / "phase3_3_sections.py",
            SCRIPTS_DIR / "phase_runtime" / "phase3_4_references.py",
            SCRIPTS_DIR / "phase_runtime" / "phase3_5_review.py",
        ]
        for runtime_file in runtime_files:
            source = runtime_file.read_text(encoding="utf-8")
            self.assertNotIn('return f"""', source, f"inline f-string prompt remains in {runtime_file}")

    def test_phase3_6_post_revision_review_prompt_is_final_draft_specific(self) -> None:
        prompt = build_phase3_5_post_revision_review_prompt(
            paper_target="IEEE WCL",
            full_paper_tex="\\section{Numerical Results} Proposed versus MRT.",
            review_facts_json=json.dumps(
                {
                    "full_paper_abbreviation_check": {"ok": True, "undefined_abbreviations": []},
                    "final_figure_check": {"ok": True, "plotted_methods": ["proposed", "mrt"]},
                    "contract_scope_check": {"ok": True},
                }
            ),
            technical_source_json="{}",
            result_source_json="{}",
            citation_source_json="{}",
        )

        self.assertIn("Judge the final revised draft, not the pre-revision draft", prompt)
        self.assertIn("dynamic paper-specific acronym inventory", prompt)
        self.assertIn("must not claim unplotted baselines", prompt)
        self.assertIn("If contract_scope_check is clean", prompt)

    def test_phase3_6_post_revision_review_replaces_stale_unresolved_items(self) -> None:
        ready_review = {
            "payload": {
                "recommendation": "minor_revision_needed",
                "critical_issues": [],
                "major_issues": [],
                "minor_issues": [{"issue_id": "P2-META", "title": "Metadata polish"}],
            }
        }
        blocking_review = {
            "payload": {
                "recommendation": "major_revision_needed",
                "critical_issues": [],
                "major_issues": [
                    {
                        "issue_id": "P1-FIG",
                        "title": "Figure evidence mismatch",
                        "suggested_action": "Align final claims with plotted figures.",
                        "responsible_phase": "analysis_agent",
                    }
                ],
            }
        }

        self.assertEqual(_phase3_6_post_review_unresolved_items(ready_review), [])
        unresolved = _phase3_6_post_review_unresolved_items(blocking_review)
        self.assertEqual(len(unresolved), 1)
        self.assertEqual(unresolved[0]["issue_id"], "P1-FIG")
        self.assertIn("post-revision ReviewAgent", unresolved[0]["why_unresolved"])

    def test_compact_short_split_equations_restores_single_line_equation(self) -> None:
        tex = (
            "\\begin{align}\n"
            "I_k&=\\sum_{i\\in\\mathcal K\\setminus\\{k\\}}\\mathbf h_k^H\\mathbf W_i\\mathbf h_k\n"
            "\\nonumber\\\\\n"
            "&\\quad+\\mathbf h_k^H\\mathbf S\\mathbf h_k .\n"
            "\\label{eq:interference_power}\n"
            "\\end{align}\n"
        )

        compacted = compact_short_split_equations(tex)

        self.assertIn("\\begin{equation}", compacted)
        self.assertIn("\\end{equation}", compacted)
        self.assertIn("I_k=\\sum_{i\\in\\mathcal K\\setminus\\{k\\}}\\mathbf h_k^H\\mathbf W_i\\mathbf h_k + \\mathbf h_k^H\\mathbf S\\mathbf h_k.", compacted)
        self.assertIn("\\label{eq:interference_power}", compacted)
        self.assertNotIn("\\nonumber", compacted)

    def test_phase3_6_splits_long_one_line_equations_for_ieee_columns(self) -> None:
        tex = (
            "\\begin{equation}\n"
            "Q_k(\\mathbf h_k)=\\eta_k(1-\\rho_k)(\\mathbf h_k^H\\mathbf R_x\\mathbf h_k+\\sigma_k^2),"
            "\\quad k\\in\\mathcal K_{\\rm PS}.\\label{eq:swipt_eh_definition}\n"
            "\\end{equation}\n"
        )

        revised, changed = _phase3_6_split_long_one_line_equations(tex, min_chars=40)

        self.assertTrue(changed)
        self.assertIn("\\begin{aligned}", revised)
        self.assertIn("Q_k(\\mathbf h_k)\n&=", revised)
        self.assertIn("&\\quad k\\in\\mathcal K_{\\rm PS}", revised)
        self.assertIn("\\label{eq:swipt_eh_definition}", revised)

    def test_phase3_1_compacts_long_sinr_with_unnumbered_shorthand(self) -> None:
        tex = (
            "\\begin{equation}\n"
            "\\gamma_k(\\mathbf h_k)=\\frac{\\bar\\rho_k\\mathbf h_k^H\\mathbf W_k\\mathbf h_k}"
            "{\\bar\\rho_k\\left(\\sum_{j\\in\\mathcal K,\\ j\\neq k}\\mathbf h_k^H\\mathbf W_j\\mathbf h_k"
            "+\\mathbf h_k^H\\mathbf S\\mathbf h_k+\\sigma_k^2\\right)+\\nu_k^2}.\\label{eq:sinr_definition}\n"
            "\\end{equation}\n"
        )

        revised = compact_long_sinr_fraction_equations(tex, min_chars=40)

        self.assertIn("\\begin{equation*}", revised)
        self.assertIn("I_k(\\mathbf h_k)\\triangleq", revised)
        self.assertIn("\\begin{equation}\n\\begin{aligned}", revised)
        self.assertIn("I_k(\\mathbf h_k)+\\sigma_k^2", revised)
        self.assertEqual(revised.count("\\label{eq:sinr_definition}"), 1)
        self.assertNotIn("eq:interference_power", revised)

    def test_phase3_6_does_not_mechanically_split_long_sinr_fraction(self) -> None:
        tex = (
            "\\begin{equation}\n"
            "\\gamma_k(\\mathbf h_k)=\\frac{\\bar\\rho_k\\mathbf h_k^H\\mathbf W_k\\mathbf h_k}"
            "{\\bar\\rho_k\\left(\\sum_{j\\in\\mathcal K,\\ j\\neq k}\\mathbf h_k^H\\mathbf W_j\\mathbf h_k"
            "+\\mathbf h_k^H\\mathbf S\\mathbf h_k+\\sigma_k^2\\right)+\\nu_k^2}.\\label{eq:sinr_definition}\n"
            "\\end{equation}\n"
        )

        revised, changed = _phase3_6_split_long_one_line_equations(tex, min_chars=40)

        self.assertFalse(changed)
        self.assertEqual(revised, tex)

    def test_phase3_6_splits_long_minimizer_lists_for_ieee_columns(self) -> None:
        tex = (
            r"\text{(P1)}\quad\min_{\mathbf W,\mathbf S,\boldsymbol\rho,\boldsymbol\tau,"
            r"\boldsymbol\zeta,\boldsymbol\lambda,\boldsymbol\mu,\boldsymbol\beta}\quad"
        )

        revised, changed = _phase3_6_split_long_minimizer_lists(tex, min_chars=40)

        self.assertTrue(changed)
        self.assertIn(r"\min_{\substack{", revised)
        self.assertIn(r"\boldsymbol\beta", revised)
        self.assertNotIn(r"\min_{\mathbf W,\mathbf S,\boldsymbol\rho,\boldsymbol\tau,\boldsymbol\zeta", revised)

    def test_phase3_sanitizer_splits_long_minimizer_lists_before_preview(self) -> None:
        tex = (
            r"\begin{subequations}\begin{align}"
            r"\text{(P1)}\quad\min_{\mathbf W,\mathbf S,\boldsymbol\rho,\boldsymbol\tau,"
            r"\boldsymbol\zeta,\boldsymbol\lambda,\boldsymbol\mu,\boldsymbol\beta}\quad"
            r"& \operatorname{tr}(\mathbf R_x)"
            r"\end{align}\end{subequations}"
        )

        cleaned = sanitize_phase3_latex_snippet(tex)

        self.assertIn(r"\min_{\substack{", cleaned)
        self.assertIn(r"\boldsymbol\beta", cleaned)

    def test_metric_prompts_do_not_hardcode_undefined_worst_case_alias(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        checked_files = [
            repo_root / "phase2" / "prompts" / "phase2_4" / "solver_package.prompt.yaml",
            repo_root / "phase2" / "prompts" / "phase2_4" / "validation_plan.prompt.yaml",
            repo_root / "phase2" / "prompts" / "phase2_5" / "experiment_planner.prompt.yaml",
            repo_root / "phase2" / "prompts" / "phase2_5" / "result_writer.prompt.yaml",
            repo_root / "phase2" / "prompts" / "phase2_5" / "sweep_refiner.prompt.yaml",
            repo_root / "phase3" / "prompts" / "phase3_2" / "numerical_results.prompt.yaml",
            repo_root / "phase2" / "scripts" / "phase_runtime" / "phase3_2_writing.py",
            repo_root / "phase2" / "scripts" / "phase25_analysis.py",
            repo_root / "phase2" / "scripts" / "run_phase24_simple_llm_experiment.py",
        ]
        for path in checked_files:
            self.assertNotIn(r"U_{\mathrm{wc}}", path.read_text(encoding="utf-8"), str(path))

    def test_phase25_captions_use_reference_paper_style(self) -> None:
        plan = {
            "compared_methods": [
                {
                    "internal_name": "proposed",
                    "display_name_short": "Proposed",
                    "display_name_long": "Proposed method",
                }
            ],
            "figure_specs": [
                {
                    "figure_id": "figure_1",
                    "chart_type": "line",
                    "metric": {"name": "worst_case_utility", "display_name": r"$\psi$"},
                    "encoding": {"x": {"display_name": r"$P_{\max}$", "sweep_param": "Pmax"}},
                    "caption_context": r"$K=3$ and $N_t=4$",
                },
                {
                    "figure_id": "figure_2",
                    "chart_type": "line",
                    "metric": {"name": "worst_case_utility", "display_name": r"$\psi$"},
                    "encoding": {"x": {"display_name": r"$b$", "sweep_param": "rectifier_b"}},
                    "caption_context": r"$P_{\max}=1.5$",
                },
            ],
            "table_specs": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            write_caption_files(
                Path(tmp),
                plan,
                [
                    {"figure_id": "figure_1", "methods": ["proposed"], "draft_or_final": "draft"},
                    {"figure_id": "figure_2", "methods": ["proposed"], "draft_or_final": "final"},
                ],
                [],
            )
            text = (Path(tmp) / "figure_captions.md").read_text(encoding="utf-8")

        self.assertIn(r"Fig. 1. Worst-case weighted utility $\psi$ versus transmit power budget $P_{\max}$, where $K=3$ and $N_t=4$.", text)
        self.assertIn(r"Fig. 2. Corresponding worst-case weighted utility $\psi$ versus rectifier steepness $b$, where $P_{\max}=1.5$.", text)
        self.assertNotIn("The compared methods are", text)
        self.assertNotIn("The legend uses", text)
        self.assertNotIn("Paper-ready", text)
        self.assertNotIn("Draft figure", text)

    def test_phase25_internal_axis_paths_do_not_invent_csi_notation(self) -> None:
        self.assertEqual(_safe_display("ambiguity.channel_uncertainty_radius"), "ambiguity channel uncertainty radius")
        plan = {
            "compared_methods": [{"internal_name": "proposed", "display_name_short": "Proposed"}],
            "figure_specs": [
                {
                    "figure_id": "figure_1",
                    "chart_type": "line",
                    "metric": {"name": "worst_case_utility", "display_name": r"$\psi$"},
                    "encoding": {
                        "x": {
                            "display_name": r"$r_{\mathrm{amb}}$",
                            "sweep_param": "ambiguity.channel_uncertainty_radius",
                        }
                    },
                    "caption_context": r"$K=3$",
                }
            ],
            "table_specs": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            write_caption_files(
                Path(tmp),
                plan,
                [{"figure_id": "figure_1", "methods": ["proposed"], "draft_or_final": "final"}],
                [],
            )
            text = (Path(tmp) / "figure_captions.md").read_text(encoding="utf-8")

        self.assertIn(r"$r_{\mathrm{amb}}$", text)
        self.assertNotIn(r"CSI uncertainty radius $\delta_h$", text)
        self.assertNotIn("ambiguity.channel", text)
        self.assertNotIn(r"$\delta_h$", text)

    def test_phase3_4_introduction_quality_blocks_generic_template_language(self) -> None:
        report = analyze_phase3_4_introduction_content_quality(
            r"""
            \section{Introduction}
            Wireless optimization is a key direction and has attracted significant attention.

            Prior works study several systems.

            Despite this progress, the problem remains challenging.

            Motivated by this gap, this letter makes the following contributions.
            \begin{itemize}
            \item We formulate a problem.
            \item We develop an algorithm.
            \item We provide simulations.
            \end{itemize}

            The remainder of this letter is organized as follows.

            \textit{Notation:} Bold letters denote vectors.
            """
        )
        self.assertFalse(report["ok"])
        self.assertGreaterEqual(report["generic_intro_phrase_count"], 2)
        self.assertTrue(any("generic IEEE-template phrases" in item for item in report["errors"]))

    def test_phase3_4_introduction_quality_warns_taxonomy_and_citation_clusters(self) -> None:
        report = analyze_phase3_4_introduction_content_quality(
            r"""
            \section{Introduction}
            Modern wireless systems require joint resource decisions under coupled service constraints.

            A line of work studies one resource axis for such networks \cite{a,b,c,d}.

            A complementary line of work investigates another resource axis, but leaves the coupled design consequence unresolved.

            Motivated by this gap, this letter makes the following contributions.
            \begin{itemize}
            \item It introduces a coupled modeling capability.
            \item It derives a tractable solution capability.
            \end{itemize}

            The remainder of this letter is organized as follows.

            \textit{Notation:} Bold letters denote vectors.
            """
        )
        self.assertGreaterEqual(report["taxonomy_phrase_count"], 2)
        self.assertEqual(report["long_citation_cluster_count"], 1)
        joined_warnings = " ".join(report["warnings"]).lower()
        self.assertIn("taxonomy", joined_warnings)
        self.assertIn("citation clusters", joined_warnings)

    def test_phase3_4_introduction_quality_blocks_meta_structural_language(self) -> None:
        report = analyze_phase3_4_introduction_content_quality(
            r"""
            \section{Introduction}
            Modern wireless systems require joint resource decisions under coupled service constraints.

            Covariance optimization provides one technical axis for this setting.

            Robust design forms a second technical axis, but the missing capability is a deterministic design for the current paper.

            In this letter, we address this problem.
            \begin{itemize}
            \item It introduces a coupled modeling element.
            \item It derives a tractable solution element.
            \end{itemize}

            The remainder of this letter is organized as follows.

            \textit{Notation:} Bold letters denote vectors.
            """
        )
        self.assertFalse(report["ok"])
        self.assertGreaterEqual(report["prohibited_meta_phrase_count"], 3)
        self.assertTrue(any("meta-structural" in item for item in report["errors"]))

    def test_phase3_4_introduction_sanitizer_demaths_body_only(self) -> None:
        tex = sanitize_phase3_4_introduction_tex(
            r"""
            \section{Introduction}
            The benchmark is a fixed $\lambda/2$ ULA and the method optimizes $(W,p)$ jointly.
            The roadmap discusses the proposed $p$-update benchmark.

            \textit{Notation:} $(\cdot)^H$ denotes Hermitian transpose.
            """
        )

        before_notation = tex.split(r"\textit{Notation:}", 1)[0]
        self.assertIn("fixed half-wavelength ULA", before_notation)
        self.assertIn("the associated design variables jointly", before_notation)
        self.assertIn("proposed position-update benchmark", before_notation)
        self.assertNotIn("associated design variables-update", before_notation)
        self.assertNotRegex(before_notation, r"\$[^$]+\$")
        self.assertIn(r"$(\cdot)^H$", tex.split(r"\textit{Notation:}", 1)[1])

    def test_phase3_4_introduction_normalizer_restores_wcl_paragraph_skeleton(self) -> None:
        tex = normalize_phase3_4_introduction_paragraphs(
            r"\section{Introduction} Motivation sentence. A first thread studies related systems. "
            r"A second thread studies methods. In this letter, we contribute. \begin{itemize} \item First. \item Second. \end{itemize} "
            r"The remainder of this letter is organized as follows. Section reports mRT with proposed the associated design variables-update. "
            r"\textit{Notation:} Bold letters denote vectors."
        )

        self.assertIn("\\section{Introduction}\nMotivation sentence.", tex)
        self.assertIn("\n\nPrior work studies", tex)
        self.assertIn("Complementary prior work studies methods.", tex)
        self.assertIn("\n\nIn this letter, we contribute.", tex)
        self.assertIn("\\begin{itemize}\n\\item First.\n\\item Second.\n\\end{itemize}", tex)
        self.assertIn("\n\nThe remainder of this letter is organized as follows.", tex)
        self.assertIn("MRT with proposed position-update", tex)
        self.assertNotIn("associated design variables-update", tex)
        self.assertIn("\n\n\\textit{Notation:}", tex)

    def test_phase28_prompts_are_external_yaml_templates(self) -> None:
        prompt_dir = Path(__file__).resolve().parents[2] / "phase3" / "prompts" / "phase3_4"
        prompt_files = [
            prompt_dir / "introduction.prompt.yaml",
            prompt_dir / "reference_check.prompt.yaml",
            prompt_dir / "technical_citation.prompt.yaml",
        ]
        for prompt_path in prompt_files:
            self.assertTrue(prompt_path.exists(), f"missing prompt template: {prompt_path}")
            payload = yaml.safe_load(prompt_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["phase_id"], "phase3.4")
            self.assertIsInstance(payload["template"], str)
            self.assertIsInstance(payload["variables"], list)
            self.assertGreater(len(payload["template"].strip()), 200)

        source = (SCRIPTS_DIR / "phase_runtime" / "phase3_4_references.py").read_text(encoding="utf-8")
        self.assertNotIn("You are writing the Introduction section for phase3.3", source)
        self.assertIn("render_prompt_template", source)

    def test_phase28_intro_prompt_renders_from_yaml_without_latex_brace_breakage(self) -> None:
        prompt = build_phase3_4_introduction_prompt(
            source_map_json='{"source": "source-map-marker"}',
            current_paper_brief_json='{"brief": "current-paper-brief-marker"}',
            introduction_facts_json='{"facts": "intro-facts-marker"}',
            verified_reference_bank_json='{"refs": "verified-reference-bank-marker"}',
            reference_quality_report_json='{"quality": "reference-quality-marker"}',
            reference_strategy_json='{"strategy": "reference-strategy-marker"}',
            writing_agent_request_json='{"agent_id": "writing_agent", "marker": "writing-request-marker"}',
            literature_agent_request_json='{"agent_id": "literature_agent", "marker": "literature-request-marker"}',
        )

        self.assertIn(r"\section{Introduction}", prompt)
        self.assertIn("selected_reference_keys", prompt)
        self.assertIn("verified_reference_bank", prompt)
        self.assertIn("source-map-marker", prompt)
        self.assertIn("current-paper-brief-marker", prompt)
        self.assertIn("Topic-first writing rule", prompt)
        self.assertIn("this WCL introduction flow", prompt)
        self.assertIn("Paragraph 2 - closest prior-work context", prompt)
        self.assertIn("Paragraph 3 - second context thread plus gap", prompt)
        self.assertIn("Organization paragraph", prompt)
        self.assertIn("Notation paragraph", prompt)
        self.assertIn("The notation paragraph must be the final paragraph", prompt)
        self.assertIn("why this exact problem is timely", prompt)
        self.assertIn("right minimal response", prompt)
        self.assertIn("must be selected from current_paper_brief_json", prompt)
        self.assertIn("what existing works optimize or assume", prompt)
        self.assertIn("what coupling, constraint, or modeling feature they miss", prompt)
        self.assertIn("why that omission matters for the current paper's objective", prompt)
        self.assertIn("specific modeling element, solution element, or evaluation contrast", prompt)
        self.assertIn("broad wireless buzzwords", prompt)
        self.assertIn("In this letter, we", prompt)
        self.assertIn("The main contributions are summarized as follows", prompt)
        self.assertIn("Do not describe algorithm implementation details", prompt)
        self.assertIn("scoped qualitative gain", prompt)
        self.assertIn("universal superiority claims", prompt)
        self.assertIn("half-wavelength fixed array", prompt)
        self.assertIn("Do not write a full constraint list", prompt)
        self.assertIn("Do not preview numerical values", prompt)
        self.assertIn("First-pass authoring principle", prompt)
        self.assertIn("submission-quality on the first generation", prompt)
        self.assertIn("Do not write the Introduction as a taxonomy of literatures", prompt)
        self.assertIn("No paragraph may be only a citation-supported survey summary", prompt)
        self.assertIn("If two adjacent paragraphs could be swapped without changing the argument", prompt)
        self.assertIn("The gap sentence must be the intellectual hinge", prompt)
        self.assertIn("citation clusters readable", prompt)
        self.assertIn("causal motivation -> closest-work context -> gap argument", prompt)
        self.assertIn("WCL style evidence from highly cited IEEE Wireless Communications Letters", prompt)
        self.assertIn("technical axis", prompt)
        self.assertIn("Do not recast the paper objective into a common objective", prompt)
        self.assertNotIn("about 650 to 900 words", prompt)
        self.assertIn("intro-facts-marker", prompt)
        self.assertIn("verified-reference-bank-marker", prompt)
        self.assertIn("writing-request-marker", prompt)
        self.assertIn("literature-request-marker", prompt)
        self.assertIn("Agent boundaries", prompt)
        self.assertNotIn("{source_map_json}", prompt)
        self.assertNotIn("{current_paper_brief_json}", prompt)
        self.assertNotIn("{verified_reference_bank_json}", prompt)

    def test_phase26_and_phase27_prompts_include_writing_agent_request(self) -> None:
        phase3_2_prompt = build_phase3_2_numerical_results_prompt(
            topic="topic-marker",
            system_model_md="system-model-marker",
            problem_formulation_md="problem-formulation-marker",
            algorithm_md="algorithm-marker",
            benchmark_definition_md="benchmark-marker",
            method_mechanism_summary="method-marker",
            paper_objective_summary="objective-marker",
            figure_to_claim_summary="figure-claim-marker",
            figure_observations_summary="observation-marker",
            figure_evidence_summary_json='{"evidence": "evidence-marker"}',
            claim_constraints_summary="claim-constraint-marker",
            simulation_setup_facts_json='{"setup": "setup-marker"}',
            phase25_summary_json='{"phase25": "summary-marker"}',
            verified_registry_json='{"registry": "registry-marker"}',
            figure_captions_md="caption-marker",
            writing_agent_request_json='{"agent_id": "analysis_agent", "marker": "phase3_2-analysis-request"}',
        )
        self.assertIn("AnalysisAgent boundary", phase3_2_prompt)
        self.assertIn("phase3_2-analysis-request", phase3_2_prompt)
        self.assertIn("only the methods that actually appear", phase3_2_prompt)
        self.assertIn("First-pass authoring principle", phase3_2_prompt)
        self.assertIn("mature paper section on the first attempt", phase3_2_prompt)
        self.assertNotIn("{writing_agent_request_json}", phase3_2_prompt)
        self.assertNotIn("{analysis_agent_request_json}", phase3_2_prompt)

        phase3_3_prompt = build_phase3_3_abstract_conclusion_prompt(
            topic="topic-marker",
            paper_facts_json='{"facts": "paper-facts-marker"}',
            writing_agent_request_json='{"agent_id": "writing_agent", "marker": "phase3_3-writing-request"}',
        )
        self.assertIn("WritingAgent boundary", phase3_3_prompt)
        self.assertIn("phase3_3-writing-request", phase3_3_prompt)
        self.assertIn("Do not use inline mathematical notation", phase3_3_prompt)
        self.assertIn("Do not use inline mathematical notation, optimizer symbols, or equation variables in the conclusion", phase3_3_prompt)
        self.assertIn("Prefer full-language phrases over letter tags", phase3_3_prompt)
        self.assertIn("First-pass authoring principle", phase3_3_prompt)
        self.assertIn("submission-quality on the first generation", phase3_3_prompt)
        self.assertIn("Do not write the abstract as a five-sentence checklist", phase3_3_prompt)
        self.assertIn("operating conflict -> missing capability -> proposed response -> technical mechanism", phase3_3_prompt)
        self.assertIn("These five moves are a logic contract", phase3_3_prompt)
        self.assertNotIn("{writing_agent_request_json}", phase3_3_prompt)

    def test_phase3_3_abstract_sanitizer_removes_optimizer_notation(self) -> None:
        tex = (
            "\\begin{abstract}\n"
            "The method optimizes the user covariances \\(\\{\\mathbf{W}_k\\}\\) "
            "and the shared covariance \\(\\mathbf{V}\\) directly. "
            "Numerical results reduce optimized transmit power \\(P_{\\rm tx}\\) "
            "against the maximum-ratio-transmission (MRT) benchmark.\n"
            "\\end{abstract}"
        )

        cleaned = sanitize_phase3_3_abstract_tex(tex)

        self.assertIn("optimized transmit-design variables", cleaned)
        self.assertIn("optimized transmit power", cleaned)
        self.assertNotIn("\\mathbf", cleaned)
        self.assertNotIn("P_{\\rm tx}", cleaned)
        self.assertNotIn("(MRT)", cleaned)
        self.assertNotIn("MRT benchmark", cleaned)
        self.assertEqual(find_phase3_3_abstract_notation_issues(cleaned), [])

    def test_phase3_3_abstract_structure_reports_checklist_style(self) -> None:
        report = analyze_phase3_3_abstract_structure(
            "\\begin{abstract}\n"
            "Integrated wireless design couples service and resource constraints. "
            "Existing methods do not jointly optimize all relevant controls. "
            "This letter proposes a design method. "
            "The proposed method uses a tractable optimization route. "
            "Numerical results support the method under the considered settings.\n"
            "\\end{abstract}\n"
            "\\begin{IEEEkeywords}Wireless optimization, resource allocation\\end{IEEEkeywords}"
        )
        self.assertGreaterEqual(report["checklist_like_sentence_count"], 3)
        self.assertGreaterEqual(report["weak_gap_template_count"], 1)
        self.assertTrue(report["narrative_warning"])

    def test_phase3_3_conclusion_sanitizer_removes_optimizer_notation(self) -> None:
        tex = (
            "\\section{Conclusion}\n"
            "The proposed method optimizes the communication covariances "
            "\\(\\{\\mathbf{W}_k\\}\\) and the shared covariance \\(\\mathbf{V}\\). "
            "The results show lower optimized transmit power \\(P_{\\rm tx}\\) "
            "than the maximum-ratio-transmission (MRT) benchmark."
        )

        cleaned = sanitize_phase3_3_conclusion_tex(tex)

        self.assertIn("optimized transmit-design variables", cleaned)
        self.assertIn("optimized transmit power", cleaned)
        self.assertNotIn("\\mathbf", cleaned)
        self.assertNotIn("P_{\\rm tx}", cleaned)
        self.assertNotIn("(MRT)", cleaned)
        self.assertNotIn("MRT benchmark", cleaned)
        self.assertEqual(find_phase3_3_conclusion_notation_issues(cleaned), [])

    def test_phase3_2_filters_method_metadata_to_final_plotted_methods(self) -> None:
        method_json = json.dumps(
            {
                "methods": [
                    {"internal_name": "proposed", "display_name_short": "Proposed"},
                    {
                        "internal_name": "mrt_covariance_baseline",
                        "display_name_short": "MRT",
                        "display_name_long": "Channel-matched covariance directions",
                    },
                    {"internal_name": "isotropic_shared_covariance_baseline", "display_name_short": "Iso-cov."},
                ]
            }
        )
        phase25_summary = {
            "figures": [
                {"paper_ready": True, "draft_or_final": "final", "methods": ["proposed", "mrt_covariance_baseline"]},
                {"paper_ready": True, "draft_or_final": "final", "methods": ["proposed", "mrt_covariance_baseline"]},
            ]
        }

        filtered = json.loads(filter_phase3_2_method_naming_summary_for_plotted_methods(method_json, phase25_summary))

        self.assertEqual(
            [item["internal_name"] for item in filtered["methods"]],
            ["proposed", "mrt_covariance_baseline"],
        )

    def test_phase3_2_removes_unplotted_benchmark_definitions_from_itemize(self) -> None:
        method_json = json.dumps(
            {
                "methods": [
                    {"internal_name": "proposed", "display_name_short": "Proposed"},
                    {
                        "internal_name": "mrt_covariance_baseline",
                        "display_name_short": "MRT",
                        "display_name_long": "Channel-matched covariance directions",
                    },
                    {"internal_name": "isotropic_shared_covariance_baseline", "display_name_short": "Iso-cov."},
                ]
            }
        )
        phase25_summary = {
            "figures": [
                {"paper_ready": True, "draft_or_final": "final", "methods": ["proposed", "mrt_covariance_baseline"]},
                {"paper_ready": True, "draft_or_final": "final", "methods": ["proposed", "mrt_covariance_baseline"]},
            ]
        }
        tex = (
            "\\begin{itemize}\n"
            "\\item \\textbf{Proposed}: solves the full problem.\n"
            "\\item \\textbf{Iso-cov.}: constrains the shared covariance.\n"
            "\\item \\textbf{MRT}: fixes channel-matched directions.\n"
            "\\end{itemize}"
        )

        cleaned = enforce_phase3_2_plotted_method_definitions(tex, phase25_summary, method_json)

        self.assertIn("Proposed", cleaned)
        self.assertIn("MRT", cleaned)
        self.assertNotIn("Iso-cov.", cleaned)

    def test_phase3_6_aligns_abstract_claims_to_plotted_benchmark(self) -> None:
        method_json = json.dumps(
            {
                "methods": [
                    {"internal_name": "proposed", "display_name_short": "Proposed"},
                    {
                        "internal_name": "mrt_covariance_baseline",
                        "display_name_short": "MRT",
                        "display_name_long": "Channel-matched covariance directions",
                    },
                    {"internal_name": "no_shared_covariance_baseline", "display_name_short": "No-shared-cov.", "display_name_long": "Communication covariance only"},
                ]
            }
        )
        phase25_summary = {
            "primary_claim_check": {"baseline_method": "mrt_covariance_baseline"},
            "figures": [
                {"paper_ready": True, "draft_or_final": "final", "methods": ["proposed", "mrt_covariance_baseline"]},
            ],
        }
        text = (
            "Proposed achieves lower power than the Communication covariance only "
            "(No-shared-cov.) benchmark and the other tested covariance benchmarks."
        )

        updated, changed = _phase3_6_align_plotted_method_claim_text(text, phase25_summary, method_json)

        self.assertTrue(changed)
        self.assertIn("than the channel-matched covariance directions (MRT) benchmark", updated)
        self.assertNotIn("Communication covariance only", updated)
        self.assertNotIn("No-shared-cov.", updated)

    def test_phase3_6_aligns_unplotted_precoding_variant_to_plotted_benchmark(self) -> None:
        method_json = json.dumps(
            {
                "methods": [
                    {"internal_name": "proposed", "display_name_short": "Proposed"},
                    {
                        "internal_name": "regularized_zf_heuristic",
                        "display_name_short": "RZF",
                        "display_name_long": "Fixed-layout regularized zero-forcing precoding",
                    },
                    {
                        "internal_name": "mrt_or_channel_matched",
                        "display_name_short": "MRT",
                        "display_name_long": "Fixed-layout maximum-ratio transmission",
                    },
                ]
            }
        )
        phase25_summary = {
            "primary_claim_check": {"baseline_method": "mrt_or_channel_matched"},
            "figures": [
                {"paper_ready": True, "draft_or_final": "final", "methods": ["proposed", "mrt_or_channel_matched"]},
            ],
        }
        text = "Numerical results compare the Proposed WMMSE-SCA scheme with the fixed-layout regularized zero-forcing benchmark."

        updated, changed = _phase3_6_align_plotted_method_claim_text(text, phase25_summary, method_json)

        self.assertTrue(changed)
        self.assertIn("fixed-layout maximum-ratio transmission (MRT) benchmark", updated)
        self.assertNotIn("regularized zero-forcing", updated)

    def test_phase3_6_method_alignment_preserves_introduction_paragraph_breaks(self) -> None:
        method_json = json.dumps(
            {
                "methods": [
                    {"internal_name": "proposed", "display_name_short": "Proposed"},
                    {
                        "internal_name": "mrt_or_channel_matched",
                        "display_name_short": "MRT",
                        "display_name_long": "Fixed-layout maximum-ratio transmission",
                    },
                    {
                        "internal_name": "regularized_zf_heuristic",
                        "display_name_short": "RZF",
                        "display_name_long": "Fixed-layout regularized zero-forcing",
                    },
                ]
            }
        )
        phase25_summary = {
            "primary_claim_check": {"baseline_method": "mrt_or_channel_matched"},
            "figures": [
                {"paper_ready": True, "draft_or_final": "final", "methods": ["proposed", "mrt_or_channel_matched"]},
            ],
        }
        text = (
            "\\section{Introduction}\n"
            "Motivation paragraph.\n\n"
            "In this letter, we compare against the fixed-layout regularized zero-forcing benchmark.\n\n"
            "\\textit{Notation:} Bold letters denote vectors."
        )

        updated, changed = _phase3_6_align_plotted_method_claim_text(text, phase25_summary, method_json)

        self.assertTrue(changed)
        self.assertIn("\n\nIn this letter", updated)
        self.assertIn("\n\n\\textit{Notation:}", updated)
        self.assertIn("fixed-layout maximum-ratio transmission (MRT) benchmark", updated)

    def test_phase3_6_scopes_exact_wmmse_language_to_identity(self) -> None:
        revised, changed = _phase3_6_scope_exact_wmmse_language(
            "The precoder step uses an exact weighted minimum mean-square-error mapping for fixed antenna locations, "
            "while the coordinate step is local. The weighted minimum mean-square-error step gives the exact "
            "fixed-coordinate precoder mapping, while the coordinate step remains feasible."
        )

        self.assertTrue(changed)
        self.assertIn("identity and auxiliary updates are exact", revised)
        self.assertIn("local block-coordinate algorithm", revised)
        self.assertNotIn("exact weighted minimum mean-square-error mapping", revised)
        self.assertNotIn("exact fixed-coordinate precoder mapping", revised)

    def test_phase29_review_routing_targets_owning_agent(self) -> None:
        experiment_decision = build_phase3_5_review_routing_decision(
            critical_issues=[
                {
                    "issue_id": "P0-EXP-01",
                    "title": "Figure evidence does not support the numerical claim",
                    "description": "The benchmark and metric evidence in the numerical result are inconsistent.",
                }
            ],
            major_issues=[],
            minor_issues=[],
            recommendation="major_revision_needed",
        )
        self.assertEqual(experiment_decision["status"], "repair_required")
        self.assertEqual(experiment_decision["next_agent"], "repair_agent")
        self.assertEqual(experiment_decision["target_agent"], "experiment_agent")

        citation_decision = build_phase3_5_review_routing_decision(
            critical_issues=[],
            major_issues=[
                {
                    "issue_id": "P1-REF-01",
                    "title": "Missing citation support",
                    "description": "A related-work claim uses an unsupported bibliography key.",
                }
            ],
            minor_issues=[],
            recommendation="major_revision_needed",
        )
        self.assertEqual(citation_decision["target_agent"], "literature_agent")

    def test_phase28_reference_check_prompt_renders_from_yaml(self) -> None:
        prompt = build_phase3_4_reference_check_prompt(
            introduction_facts_json='{"facts": "intro-facts-marker"}',
            verified_reference_bank_json='{"refs": "verified-reference-bank-marker"}',
            selected_reference_keys_json='["keep-me"]',
            citation_claim_map_json='[{"claim": "claim-map-marker"}]',
            reference_quality_report_json='{"quality": "reference-quality-marker"}',
        )

        self.assertIn("keep_keys", prompt)
        self.assertIn("drop_keys", prompt)
        self.assertIn("Do not invent references", prompt)
        self.assertIn("verified-reference-bank-marker", prompt)
        self.assertIn("claim-map-marker", prompt)
        self.assertNotIn("{selected_reference_keys_json}", prompt)
        self.assertNotIn("{citation_claim_map_json}", prompt)

    def test_phase28_technical_citation_prompt_is_citation_only(self) -> None:
        prompt = build_phase3_4_technical_citation_prompt(
            current_paper_brief_json='{"brief": "brief-marker"}',
            verified_reference_bank_json='{"refs": [{"final_bib_key": "refA"}]}',
            reference_quality_report_json='{"quality": "quality-marker"}',
            introduction_citation_claim_map_json='[{"claim": "intro-claim-marker"}]',
            system_model_problem_formulation_section_tex="System sentence.",
            proposed_solution_section_tex="Method sentence.",
            numerical_results_section_tex="Result sentence.",
        )

        self.assertIn("citation-insertion pass only", prompt)
        self.assertIn("Do not rewrite", prompt)
        self.assertIn("identical to the supplied section text except for adding", prompt)
        self.assertIn("Do not place citations inside display equations", prompt)
        self.assertIn("full-paper reference target is enforced across Introduction plus technical sections", prompt)
        self.assertIn("System Model and Problem Formulation should cite", prompt)
        self.assertIn("Numerical Results should cite benchmark/evaluation context", prompt)
        self.assertIn("brief-marker", prompt)
        self.assertIn("intro-claim-marker", prompt)
        self.assertNotIn("{current_paper_brief_json}", prompt)

    def test_phase28_technical_citation_validator_allows_only_cites(self) -> None:
        originals = {
            "system_model_problem_formulation_section_tex": "The channel follows a bounded uncertainty model.",
            "proposed_solution_section_tex": "The problem is handled by an iterative approximation.",
            "numerical_results_section_tex": "Fig.~\\ref{fig:a} reports the utility.",
        }
        revised = {
            "system_model_problem_formulation_section_tex": "The channel follows a bounded uncertainty model \\cite{refA}.",
            "proposed_solution_section_tex": "The problem is handled by an iterative approximation \\cite{refB}.",
            "numerical_results_section_tex": originals["numerical_results_section_tex"],
        }

        report = validate_phase3_4_technical_citation_only_revision(
            original_sections=originals,
            revised_sections=revised,
            valid_reference_keys={"refA", "refB"},
        )

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["technical_citation_count"], 2)

        changed = dict(revised)
        changed["proposed_solution_section_tex"] = "The problem is solved by an iterative approximation \\cite{refB}."
        changed_report = validate_phase3_4_technical_citation_only_revision(
            original_sections=originals,
            revised_sections=changed,
            valid_reference_keys={"refA", "refB"},
        )
        self.assertFalse(changed_report["ok"], changed_report)
        self.assertIn("non-citation technical text was changed", " ".join(changed_report["errors"]))

        prose_after_math = {
            "system_model_problem_formulation_section_tex": r"The set $\mathcal K$ models users \cite{refA}, where $K$ is fixed.",
            "proposed_solution_section_tex": r"The update preserves $\rho_k$ feasibility \cite{refB}.",
            "numerical_results_section_tex": r"The metric $\psi$ increases in the evaluated regime.",
        }
        prose_report = validate_phase3_4_technical_citation_only_revision(
            original_sections={
                "system_model_problem_formulation_section_tex": r"The set $\mathcal K$ models users, where $K$ is fixed.",
                "proposed_solution_section_tex": r"The update preserves $\rho_k$ feasibility.",
                "numerical_results_section_tex": prose_after_math["numerical_results_section_tex"],
            },
            revised_sections=prose_after_math,
            valid_reference_keys={"refA", "refB"},
        )
        self.assertTrue(prose_report["ok"], prose_report)

        citation_inside_math = dict(prose_after_math)
        citation_inside_math["numerical_results_section_tex"] = r"The metric $\psi \cite{refA}$ increases."
        math_report = validate_phase3_4_technical_citation_only_revision(
            original_sections={
                "system_model_problem_formulation_section_tex": prose_after_math["system_model_problem_formulation_section_tex"],
                "proposed_solution_section_tex": prose_after_math["proposed_solution_section_tex"],
                "numerical_results_section_tex": r"The metric $\psi$ increases.",
            },
            revised_sections=citation_inside_math,
            valid_reference_keys={"refA", "refB"},
        )
        self.assertFalse(math_report["ok"], math_report)
        self.assertIn("citation placed inside inline math", " ".join(math_report["errors"]))

        no_technical_citations = validate_phase3_4_technical_citation_only_revision(
            original_sections=originals,
            revised_sections=originals,
            valid_reference_keys={"refA", "refB"},
        )
        self.assertFalse(no_technical_citations["ok"], no_technical_citations)
        self.assertIn("No technical-section citations were inserted", " ".join(no_technical_citations["errors"]))

    def test_phase3_4_final_reference_contract_counts_full_paper_not_intro_only(self) -> None:
        valid_keys = {f"ref{i}" for i in range(1, 13)}
        report = build_phase3_4_final_reference_count_contract(
            final_selected_reference_keys=[f"ref{i}" for i in range(1, 13)],
            current_valid_reference_keys=valid_keys,
            introduction_cited_reference_keys=["ref1", "ref2", "ref3", "ref4"],
            technical_citation_map=[
                {"section": "system_model", "citation_keys": ["ref5", "ref6"]},
                {"section": "proposed_solution", "citation_keys": ["ref7", "ref8", "ref9"]},
                {"section": "numerical_results", "citation_keys": ["ref10", "ref11", "ref12"]},
            ],
            minimum_reference_target=12,
        )

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["introduction_valid_cited_references"], 4)
        self.assertEqual(report["final_valid_cited_references"], 12)
        self.assertEqual(report["technical_valid_cited_references"], 8)

        thin_report = build_phase3_4_final_reference_count_contract(
            final_selected_reference_keys=["ref1", "ref2"],
            current_valid_reference_keys=valid_keys,
            introduction_cited_reference_keys=["ref1", "ref2"],
            technical_citation_map=[],
            minimum_reference_target=12,
        )
        self.assertFalse(thin_report["ok"])
        self.assertTrue(any("hard target 12" in item for item in thin_report["errors"]))
        self.assertTrue(any("No verified references are cited in technical sections" in item for item in thin_report["errors"]))

    def test_phase3_figure_prompts_are_graph_based_and_generic(self) -> None:
        prompt_root = Path(__file__).resolve().parents[2] / "phase3" / "prompts" / "phase3_figure"
        for prompt_path in [
            prompt_root / "diagram_spec.prompt.yaml",
            prompt_root / "diagram_image.prompt.yaml",
            prompt_root / "direct_image.prompt.yaml",
        ]:
            self.assertTrue(prompt_path.exists(), f"missing prompt template: {prompt_path}")
            payload = yaml.safe_load(prompt_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["phase_id"], "phase3.figure")
            self.assertIsInstance(payload["template"], str)
            self.assertIsInstance(payload["variables"], list)

        prompt = build_phase3_figure_diagram_spec_prompt(
            topic="marker-topic",
            current_paper_brief_json='{"brief": "brief-marker"}',
            system_model_md="system-model-marker",
            problem_formulation_md="problem-formulation-marker",
            proposed_solution_md="proposed-solution-marker",
            numerical_evidence_json='{"evidence": "evidence-marker"}',
        )
        self.assertIn("diagram_type", prompt)
        self.assertIn("nodes", prompt)
        self.assertIn("edges", prompt)
        self.assertIn("containers", prompt)
        self.assertIn("branches", prompt)
        self.assertIn("paper_placement", prompt)
        self.assertIn("figure*", prompt)
        self.assertIn(r"0.7\\linewidth", prompt)
        self.assertIn("optimization_structure", prompt)
        self.assertIn("algorithm_flow", prompt)
        self.assertIn("Do not assume", prompt)
        self.assertIn("unless they are explicitly present", prompt)
        self.assertIn("If the paper material is insufficient to justify a physical system diagram", prompt)
        self.assertIn("brief-marker", prompt)
        self.assertIn("system-model-marker", prompt)
        self.assertNotIn('"transmitters"', prompt)
        self.assertNotIn('"receivers"', prompt)
        self.assertNotIn('"channels"', prompt)
        self.assertNotIn("{current_paper_brief_json}", prompt)

        image_prompt = build_phase3_figure_diagram_image_prompt(
            diagram_spec_json='{"diagram_type":"algorithm_flow","nodes":[{"id":"step","label":"Update","role":"algorithm_step"}],"edges":[]}'
        )
        self.assertIn("Use only the nodes, edges, containers, branches, outputs, and labels", image_prompt)
        self.assertIn("Do not infer or add any missing domain element", image_prompt)
        self.assertIn("For \"optimization_structure\"", image_prompt)
        self.assertIn(r"\begin{figure*}[!t]", image_prompt)
        self.assertIn(r"0.7\linewidth", image_prompt)
        self.assertIn("Do not include equations", image_prompt)
        self.assertIn("Do not include a legend box", image_prompt)
        self.assertIn("Do not draw math-variable labels", image_prompt)
        self.assertIn('"algorithm_flow"', image_prompt)
        self.assertNotIn("{diagram_spec_json}", image_prompt)

    def test_phase3_figure_structured_renderer_creates_valid_assets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            spec = build_default_phase3_figure_diagram_spec(
                topic="Nonlinear SWIPT Utility Maximization",
                system_model_md="A BS serves users with power-splitting SWIPT and nonlinear rectifier.",
                problem_formulation_md="Joint beamforming and power splitting maximize rate-energy utility.",
                proposed_solution_md="The solver updates beamforming and rho.",
            )
            result = render_phase3_figure_diagram_from_spec(spec, Path(tmp))
            png_path = Path(result["png_path"])
            pdf_path = Path(result["pdf_path"])
            self.assertTrue(png_path.exists())
            self.assertTrue(pdf_path.exists())
            asset_report = validate_phase3_figure_assets(png_path, pdf_path)
            self.assertTrue(asset_report["ok"], asset_report)
            self.assertGreaterEqual(asset_report["png_size"][0], 900)

    def test_phase3_figure_direct_image_prompt_preserves_spec_labels(self) -> None:
        spec = build_default_phase3_figure_diagram_spec(
            topic="Nonlinear SWIPT Utility Maximization",
            system_model_md="A BS serves users with power-splitting SWIPT and nonlinear rectifier.",
            problem_formulation_md="Joint beamforming and power splitting maximize rate-energy utility.",
            proposed_solution_md="The solver updates beamforming and rho.",
        )
        prompt = build_phase3_figure_direct_image_prompt(spec)

        self.assertIn("publication-ready", prompt)
        self.assertIn(r"\begin{figure*}[!t]", prompt)
        self.assertIn(r"0.7\linewidth", prompt)
        self.assertIn("inserted immediately after Introduction", prompt)
        self.assertIn("Use only the visible labels listed below", prompt)
        self.assertIn('visible label "Power Splitting"', prompt)
        self.assertIn('"BS" -> "Beamforming"', prompt)
        self.assertIn("Do not include equations", prompt)
        self.assertIn("Do not include a legend box", prompt)
        self.assertIn("Do not draw math-variable labels", prompt)
        self.assertNotIn('edge label "rho"', prompt)
        self.assertNotIn('edge label "rho_k"', prompt)
        self.assertNotIn('edge label "w_k"', prompt)
        self.assertNotIn("{diagram_spec_json}", prompt)

    def test_full_paper_preview_places_conceptual_diagram_after_introduction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            phase_dir = Path(tmp) / "run" / "phase3-3"
            build_dir = Path(tmp) / "run" / "_preview_build"
            phase_dir.mkdir(parents=True)
            build_dir.mkdir(parents=True)
            for name in [
                "abstract.tex",
                "introduction.tex",
                "system_model_problem_formulation_section.tex",
                "proposed_solution_section.tex",
                "numerical_results_section.tex",
                "conclusion.tex",
            ]:
                (phase_dir / name).write_text("% placeholder\n", encoding="utf-8")
            (phase_dir / "figures").mkdir()
            (phase_dir / "figures" / "conceptual_diagram.png").write_bytes(b"mock-png")
            (phase_dir / "conceptual_diagram_caption.txt").write_text(
                "Conceptual overview of the proposed framework.", encoding="utf-8"
            )

            _prepare_full_paper_preview_inputs(phase_dir, build_dir)

            figure_tex = (build_dir / "conceptual_diagram.tex").read_text(encoding="utf-8")
            self.assertIn(r"\begin{figure*}[!t]", figure_tex)
            self.assertIn(r"\includegraphics[width=0.7\linewidth]{figures/conceptual_diagram.png}", figure_tex)
            self.assertIn(r"\label{fig:conceptual_diagram}", figure_tex)
            self.assertIn(r"\end{figure*}", figure_tex)
            self.assertTrue((build_dir / "figures" / "conceptual_diagram.png").exists())

        phase3_4_source = (SCRIPTS_DIR / "phase_runtime" / "phase3_4_references.py").read_text(encoding="utf-8")
        intro_index = phase3_4_source.index(r"\input{{introduction.tex}}")
        diagram_index = phase3_4_source.index(r"\input{{conceptual_diagram.tex}}")
        system_index = phase3_4_source.index(r"\section{{System Model and Problem Formulation}}")
        self.assertLess(intro_index, diagram_index)
        self.assertLess(diagram_index, system_index)

    def test_full_paper_preview_prefers_phase_figure_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            phase_dir = run_dir / "phase3-3"
            build_dir = run_dir / "_preview_build"
            figure_dir = run_dir / "phase3-figure" / "figures"
            phase_dir.mkdir(parents=True)
            build_dir.mkdir(parents=True)
            figure_dir.mkdir(parents=True)
            for name in [
                "abstract.tex",
                "introduction.tex",
                "system_model_problem_formulation_section.tex",
                "proposed_solution_section.tex",
                "numerical_results_section.tex",
                "conclusion.tex",
            ]:
                (phase_dir / name).write_text("% placeholder\n", encoding="utf-8")
            (figure_dir / "conceptual_diagram.pdf").write_bytes(b"%PDF-1.4\n" + b"x" * 2048)
            manifest = {
                "ok": True,
                "primary_asset": "figures/conceptual_diagram.pdf",
                "figures": [
                    {
                        "id": "conceptual_diagram",
                        "label": "fig:structured_concept",
                        "caption": "Structured conceptual diagram from manifest.",
                        "path": "figures/conceptual_diagram.pdf",
                    }
                ],
            }
            (run_dir / "phase3-figure" / "figure_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )

            _prepare_full_paper_preview_inputs(phase_dir, build_dir)

            figure_tex = (build_dir / "conceptual_diagram.tex").read_text(encoding="utf-8")
            self.assertIn(r"\begin{figure*}[!t]", figure_tex)
            self.assertIn(r"\includegraphics[width=0.7\linewidth]{figures/conceptual_diagram.pdf}", figure_tex)
            self.assertIn("Structured conceptual diagram from manifest.", figure_tex)
            self.assertIn(r"\label{fig:structured_concept}", figure_tex)
            self.assertIn(r"\end{figure*}", figure_tex)
            self.assertTrue((build_dir / "figures" / "conceptual_diagram.pdf").exists())

    def test_phase28_reference_pool_prefers_current_scope(self) -> None:
        entries = [
            {
                "key": "current_scope",
                "title": "Queue-Aware Resource Allocation for Low-Latency Wireless Networks",
                "venue": "IEEE Transactions on Wireless Communications",
                "year": "2021",
            },
            {
                "key": "adjacent_unrelated",
                "title": "Vision Transformer Models for Medical Image Classification",
                "venue": "IEEE Wireless Communications Letters",
                "year": "2024",
            },
            {
                "key": "substring_trap",
                "title": "Embodied AI Agents for Team Collaboration in Co-located Work",
                "venue": "Other",
                "year": "2026",
            },
        ]
        context = "Queue-aware resource allocation for low-latency wireless networks"
        pool = select_reference_pool(entries, context_text=context, focus_keys=[], max_items=5)
        keys = [item["key"] for item in pool]

        self.assertIn("current_scope", keys)
        self.assertNotIn("adjacent_unrelated", keys)
        self.assertNotIn("substring_trap", keys)

    def test_phase28_final_bibliography_forces_journal_abbreviations(self) -> None:
        bib_text, missing, entries = build_curated_bibliography(
            [
                {
                    "final_bib_key": "full_name_seed",
                    "final_title": "Example {SWIPT} Paper",
                    "authors": "A. Author and B. Writer",
                    "venue": "IEEE Transactions on Wireless Communications",
                    "year": "2024",
                    "volume": "1",
                    "number": "2",
                    "pages": "1--9",
                    "month": "Jan.",
                    "doi": "10.1109/example",
                    "source_type": "journal",
                    "verification_status": "verified_published",
                    "included_in_final_bib": True,
                    "bibtex": "@article{full_name_seed,\n  author={A. Author and B. Writer},\n  journal={IEEE Transactions on Wireless Communications}\n}",
                },
                {
                    "final_bib_key": "jsac_seed",
                    "final_title": "Another Example",
                    "authors": "C. Author",
                    "venue": "IEEE Journal on Selected Areas in Communications",
                    "year": "2024",
                    "volume": "2",
                    "number": "3",
                    "pages": "10--18",
                    "month": "Mar.",
                    "doi": "10.1109/jsac.example",
                    "source_type": "journal",
                    "verification_status": "verified_published",
                    "included_in_final_bib": True,
                    "bibtex": "@article{jsac_seed,\n  journal={IEEE Journal on Selected Areas in Communications}\n}",
                },
                {
                    "final_bib_key": "ccnc_seed",
                    "final_title": "Conference Example",
                    "authors": "D. Author",
                    "venue": "IEEE Consumer Communications and Networking Conference",
                    "year": "2024",
                    "month": "Jan.",
                    "pages": "20--25",
                    "doi": "10.1109/ccnc.example",
                    "source_type": "conference",
                    "verification_status": "verified_published",
                    "included_in_final_bib": True,
                    "bibtex": "@inproceedings{ccnc_seed,\n  author={D. Author},\n  booktitle={IEEE Consumer Communications and Networking Conference}\n}",
                },
            ],
            ["full_name_seed", "jsac_seed", "ccnc_seed"],
        )

        self.assertEqual(missing, [])
        self.assertEqual(len(entries), 3)
        self.assertIn("journal = {IEEE Trans. Wireless Commun.}", bib_text)
        self.assertIn("journal = {IEEE J. Sel. Areas Commun.}", bib_text)
        self.assertIn("booktitle = {Proc. IEEE Consumer Commun. Netw. Conf. (CCNC)}", bib_text)
        self.assertIn("author  = {A. Author and B. Writer}", bib_text)
        self.assertNotIn("author  = {A. Author, B. Writer}", bib_text)
        self.assertNotIn("IEEE Transactions on Wireless Communications", bib_text)
        self.assertNotIn("IEEE Journal on Selected Areas in Communications", bib_text)
        self.assertNotIn("booktitle = {IEEE Consumer Communications and Networking Conference}", bib_text)
        ccnc_block = bib_text.split("@inproceedings{ccnc_seed", 1)[1]
        self.assertNotIn("month   =", ccnc_block)

    def test_prompt_builders_do_not_embed_topic_specific_leftovers(self) -> None:
        script_path = SCRIPTS_DIR / "phase_runtime_impl.py"
        source = script_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        forbidden_terms = [
            "movable antenna",
            "movable-antenna",
            "ma-ao",
            "fixed-pos",
            "fixed-position wmmse",
            "antenna movable range",
            "weighted sum-rate",
            "wsr",
            "eh weight",
            "ris sizes",
            "linear eh",
            "bistatic isacp",
            "crb-rate",
            "lambda3",
            "bcd-sca",
            "leh_oris",
            "nleh_fris",
            "so_ub",
        ]
        offenders: dict[str, list[str]] = {}
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            if "prompt" not in node.name.lower():
                continue
            segment = ast.get_source_segment(source, node) or ""
            lowered = segment.lower()
            hits = [term for term in forbidden_terms if term in lowered]
            if hits:
                offenders[node.name] = hits
        self.assertEqual(offenders, {})

    def test_prompts_require_standard_formulation_and_semantic_metrics(self) -> None:
        source = (SCRIPTS_DIR / "phase_runtime_impl.py").read_text(encoding="utf-8")
        self.assertIn("subject to", source)
        self.assertIn("evidence mapping table", source)
        self.assertIn("Do not use weighted objective as the primary observable metric", source)
        self.assertIn("Choose each figure's y_metric from the claim semantics", source)
        self.assertIn("RF input power is the expectation", source)
        self.assertIn("squared deviation of a quadratic RF-power expression", source)

    def test_latex_generation_prompts_force_single_relation_displays(self) -> None:
        prompt_root = Path(__file__).resolve().parents[1] / "prompts"
        for relative in [
            "phase2_1/system_model_problem.prompt.yaml",
            "phase2_1/latex_system_problem.prompt.yaml",
            "phase2_1/latex_system_problem_repair.prompt.yaml",
            "phase2_3/algorithm_design.prompt.yaml",
            "phase2_3/latex_solution.prompt.yaml",
            "phase2_3/latex_solution_repair.prompt.yaml",
        ]:
            template = yaml.safe_load((prompt_root / relative).read_text(encoding="utf-8"))["template"]
            self.assertTrue(
                "chained equalit" in template or "chained definitions" in template,
                relative,
            )
            self.assertIn("chained inequalities", template, relative)

        phase21_latex = yaml.safe_load(
            (prompt_root / "phase2_1/latex_system_problem.prompt.yaml").read_text(encoding="utf-8")
        )["template"]
        phase23_latex = yaml.safe_load(
            (prompt_root / "phase2_3/latex_solution.prompt.yaml").read_text(encoding="utf-8")
        )["template"]
        self.assertIn("do not preserve compact multi-relation equations", phase21_latex)
        self.assertIn("write each independent physical quantity or performance relation", phase21_latex)
        self.assertIn("Do not put \\label directly on \\subsection commands", phase21_latex)
        self.assertIn("do not preserve compact multi-relation equations", phase23_latex)
        self.assertIn("physical metrics defined in the System Model", phase23_latex)
        self.assertIn("eq:solution_metric", phase23_latex)
        self.assertIn("do not reuse generic labels", phase23_latex)
        self.assertIn("do not emit phrases such as `Phase 2.1`", phase23_latex)
        self.assertIn("paper-native language", phase23_latex)

    def test_phase21_system_model_prompts_require_restrained_ieee_prose(self) -> None:
        prompt_root = Path(__file__).resolve().parents[1] / "prompts" / "phase2_1"
        markdown_prompt = yaml.safe_load((prompt_root / "system_model_problem.prompt.yaml").read_text(encoding="utf-8"))["template"]
        latex_prompt = yaml.safe_load((prompt_root / "latex_system_problem.prompt.yaml").read_text(encoding="utf-8"))["template"]
        repair_prompt = yaml.safe_load((prompt_root / "latex_system_problem_repair.prompt.yaml").read_text(encoding="utf-8"))["template"]

        self.assertIn("mathematical_contract", markdown_prompt)
        self.assertIn("Separate mathematical status from usage", markdown_prompt)
        self.assertIn("status=\"control\"", markdown_prompt)
        self.assertIn("status=\"derived_quantity\"", markdown_prompt)
        self.assertIn("objective must be an object", markdown_prompt)
        self.assertIn("constraints must be a list", markdown_prompt)
        self.assertIn("controls", markdown_prompt)
        self.assertIn("derived_quantities", markdown_prompt)
        self.assertIn("reformulation_only", markdown_prompt)
        self.assertIn("notation_to_preserve", markdown_prompt)
        self.assertIn("A derived quantity such as a performance KPI", markdown_prompt)
        self.assertIn("A constraint is a relation", markdown_prompt)
        self.assertIn("mathematical status from usage", markdown_prompt)
        self.assertIn("Every optimizer variable is listed under controls", markdown_prompt)
        self.assertIn("Maintain clear logical flow", markdown_prompt)
        self.assertIn("restrained derivation", markdown_prompt)
        self.assertIn("not as a tutorial", markdown_prompt)
        self.assertIn("Phase 1 handoff", markdown_prompt)
        self.assertIn("Scope guards are generation rules", markdown_prompt)
        self.assertIn("not manuscript content", markdown_prompt)
        self.assertIn("physical control variables", markdown_prompt)
        self.assertIn("derived auxiliary quantities", markdown_prompt)
        self.assertIn("original problem display", markdown_prompt)
        self.assertIn("only physical controllable variables", markdown_prompt)
        self.assertIn("problem (P0)", markdown_prompt)
        self.assertIn("not as an equation number", markdown_prompt)
        self.assertIn("\\(\\max\\) or \\(\\min\\)", markdown_prompt)
        self.assertIn("use \"s.t.\"", markdown_prompt)
        self.assertIn("is given by", markdown_prompt)
        self.assertIn("First-person plural", markdown_prompt)
        self.assertIn("resource-splitting", markdown_prompt)
        self.assertIn("noise terms or impairments", markdown_prompt)
        self.assertIn("nonlinear hardware", markdown_prompt)
        self.assertIn("expectation- or covariance-based transition", markdown_prompt)
        self.assertIn("Do not explain every equation", markdown_prompt)
        self.assertIn("Avoid tutorial-style exposition", markdown_prompt)
        self.assertIn("critically", markdown_prompt)
        self.assertIn("contract-driven rather than exhaustive", markdown_prompt)
        self.assertIn("Do not target a fixed word count", markdown_prompt)
        self.assertIn("Phase 2.2 reformulation notes", markdown_prompt)
        self.assertIn("This phase should define only the original system model", markdown_prompt)
        self.assertIn("First-pass quality principle", markdown_prompt)
        self.assertIn("credible research construction", markdown_prompt)
        self.assertIn("closed research path", markdown_prompt)
        self.assertIn("Output-length contract", markdown_prompt)
        self.assertIn("roughly 350 words per markdown field", markdown_prompt)
        self.assertNotIn("500-900 words", markdown_prompt)
        self.assertNotIn("400-800 words", markdown_prompt)
        self.assertNotIn("exactly one role", markdown_prompt)
        self.assertNotIn("OBJECTIVE TERM", markdown_prompt)
        self.assertNotIn("DERIVED KPI", markdown_prompt)
        self.assertNotIn("CONTROL VARIABLE", markdown_prompt)
        self.assertNotIn("power-splitting ratio", markdown_prompt)
        self.assertNotIn("ID/EH branches", markdown_prompt)

        phase21_source = (SCRIPTS_DIR / "phase_runtime" / "phase21_23_foundation.py").read_text(encoding="utf-8")
        self.assertIn("compact_json_retry", phase21_source)
        self.assertIn("Keep each markdown field under roughly 250 words", phase21_source)

        self.assertIn("Maintain clear logical flow", latex_prompt)
        self.assertIn("provided topic, mathematical_contract, system_model_md, and problem_formulation_md", latex_prompt)
        self.assertIn("Scope guards are generation rules", latex_prompt)
        self.assertIn("not manuscript content", latex_prompt)
        self.assertIn("physical control variables", latex_prompt)
        self.assertIn("derived auxiliary quantities", latex_prompt)
        self.assertIn("optimizer line must include only physical controllable variables", latex_prompt)
        self.assertIn("numbered optimization layout", latex_prompt)
        self.assertIn(r"\begin{subequations}\label{prob:p0}", latex_prompt)
        self.assertIn("objective line and every constraint line must have a label", latex_prompt)
        self.assertIn("constraint-oriented labels", latex_prompt)
        self.assertIn("Do not use `eq:p0_*` labels", latex_prompt)
        self.assertIn("problem (P0)", latex_prompt)
        self.assertIn("Do not write `Problem \\eqref", latex_prompt)
        self.assertIn("Preserve the mathematical_contract exactly", latex_prompt)
        self.assertIn("mathematical status of each contract entry is immutable", latex_prompt)
        self.assertIn("optimizer line must be copied from the controls list only", latex_prompt)
        self.assertIn("derived_quantity may appear in objectives or constraints", latex_prompt)
        self.assertIn("reformulation_only auxiliary must not appear", latex_prompt)
        self.assertIn("parameter or random_quantity must never appear", latex_prompt)
        self.assertIn("If markdown prose conflicts with mathematical_contract", latex_prompt)
        self.assertIn("Treat core_theory_package_md as later-phase reformulation notes only", latex_prompt)
        self.assertIn("use `\\max` or `\\min`", latex_prompt)
        self.assertIn("use `\\mathrm{s.t.}` or `\\text{s.t.}`", latex_prompt)
        self.assertIn("is given by", latex_prompt)
        self.assertIn("First-person plural", latex_prompt)
        self.assertIn("restrained IEEE prose", latex_prompt)
        self.assertIn("not a tutorial", latex_prompt)
        self.assertIn("Avoid overfull two-column displays", latex_prompt)
        self.assertIn("introduce a local unnumbered shorthand for the aggregate interference", latex_prompt)
        self.assertIn("resource-splitting", latex_prompt)
        self.assertIn("noise terms or impairments", latex_prompt)
        self.assertIn("nonlinear hardware", latex_prompt)
        self.assertIn("expectation- or covariance-based transition", latex_prompt)
        self.assertIn("Do not explain every equation", latex_prompt)
        self.assertIn("Avoid tutorial-style exposition", latex_prompt)
        self.assertIn("meta-writing", latex_prompt)
        self.assertNotIn("ID/EH branch", latex_prompt)
        self.assertNotIn("antenna noise", latex_prompt)

        self.assertIn("restrained IEEE prose", repair_prompt)
        self.assertIn("physical control variables", repair_prompt)
        self.assertIn("derived auxiliary quantities", repair_prompt)
        self.assertIn("direct problem references", repair_prompt)
        self.assertIn(r"\begin{subequations}\label{prob:p0}", repair_prompt)
        self.assertIn("number and label the objective plus every constraint line", repair_prompt)
        self.assertIn("Constraint references", repair_prompt)
        self.assertIn("Do not use `eq:p0_*` labels", repair_prompt)
        self.assertIn("formatting_repair", repair_prompt)
        self.assertIn("semantic_consistency_repair", repair_prompt)
        self.assertIn("repair mathematical-status or usage drift", repair_prompt)
        self.assertIn("Preserve the mathematical_contract exactly", repair_prompt)
        self.assertIn("mathematical status of each contract entry is immutable", repair_prompt)
        self.assertIn("derived_quantity may appear in objectives or constraints", repair_prompt)
        self.assertIn("use `\\max` or `\\min`", repair_prompt)
        self.assertIn("use `\\mathrm{s.t.}` or `\\text{s.t.}`", repair_prompt)
        self.assertIn("expectation- or covariance-based explanation", repair_prompt)
        self.assertIn("do not expand it into tutorial-style exposition", repair_prompt)
        self.assertIn("PDF overfull hbox warnings", repair_prompt)
        self.assertIn("without overfull hbox warnings", repair_prompt)
        self.assertIn("introduce a local unnumbered shorthand for the aggregate interference", repair_prompt)

    def test_phase21_latex_prompt_receives_mathematical_contract(self) -> None:
        prompt = build_phase2_phase1_latex_prompt(
            topic="contract-test-topic",
            mathematical_contract_json='{"controls":[{"symbol":"w_k","status":"control"}]}',
            system_model_md="system draft",
            problem_formulation_md="problem draft",
            core_theory_package_md="theory draft",
        )

        self.assertIn("Mathematical contract:", prompt)
        self.assertIn("status", prompt)
        self.assertIn("control", prompt)
        self.assertIn("contract-test-topic", prompt)
        self.assertIn("system draft", prompt)

    def test_frozen_math_contract_flows_across_later_phase_prompts(self) -> None:
        contract = json.dumps(
            {
                "controls": [{"symbol": "w_k", "status": "control"}],
                "derived_quantities": [{"symbol": "R_k(w)", "status": "derived_quantity"}],
                "objective": {"sense": "max", "expression": "sum_k R_k(w)"},
                "constraints": [{"id": "power", "relation": "sum_k ||w_k||^2 <= Pmax"}],
                "reformulation_only": [{"symbol": "W_k", "status": "reformulation_only"}],
            }
        )
        handoff = {"final_title": "Frozen Contract Test"}

        phase22_prompt = build_phase2_phase2_prompt(
            topic="generic wireless optimization",
            handoff=handoff,
            mathematical_contract_json=contract,
            system_model_md="system",
            problem_formulation_md="problem",
            core_theory_package_md="theory",
        )
        phase23_prompt = build_phase2_phase3_prompt(
            topic="generic wireless optimization",
            handoff=handoff,
            mathematical_contract_json=contract,
            system_model_md="system",
            problem_formulation_md="problem",
            core_theory_package_md="theory",
            convexity_audit_md="audit",
            reformulation_path_md="reformulation",
        )
        self.assertIn("Controller tractability route policy", phase22_prompt)
        self.assertIn("selected_route", phase22_prompt)
        self.assertIn("convex_direct", phase22_prompt)
        self.assertIn("structured_nonconvex", phase22_prompt)
        self.assertIn("mechanism-preserving variable choice", phase22_prompt)
        self.assertIn("key mechanism disappears", phase22_prompt)
        self.assertIn("Technical Closure Plan", phase22_prompt)
        self.assertIn("concrete model family", phase22_prompt)
        self.assertIn("First-pass quality principle", phase22_prompt)
        self.assertIn("construct the route that downstream agents implement", phase22_prompt)
        self.assertIn("Controller tractability route policy", phase23_prompt)
        self.assertIn("which tractability route was selected", phase23_prompt)
        self.assertIn("Do not present the contribution as merely calling CVX/CVXPY", phase23_prompt)
        self.assertIn("preserve the wireless mechanism and physical KPIs", phase23_prompt)
        self.assertIn("active constraints", phase23_prompt)
        self.assertIn("Success-first principle", phase23_prompt)
        self.assertIn("Do not use black-box placeholders", phase23_prompt)
        self.assertIn("First-pass quality principle", phase23_prompt)
        self.assertIn("initial algorithm design must already be implementable", phase23_prompt)
        phase23_latex_prompt = build_phase2_phase3_latex_prompt(
            topic="generic wireless optimization",
            mathematical_contract_json=contract,
            algorithm_md="algorithm",
            convergence_or_complexity_md="complexity",
            benchmark_definition_md="benchmark",
        )
        phase24_prompt = build_phase2_phase24_plugin_prompt(
            topic="generic wireless optimization",
            mathematical_contract_json=contract,
            system_model_md="system",
            problem_formulation_md="problem",
            reformulation_path_md="reformulation",
            algorithm_md="algorithm",
            benchmark_definition_md="benchmark",
            experiment_blueprint_md="blueprint",
            validation_plan_summary="summary",
            problem_data_contract_summary="contract",
        )
        phase3_5_prompt = build_phase3_4_review_prompt(
            review_facts_json="{}",
            mathematical_contract_json=contract,
            verified_reference_bank_json="[]",
            current_sections_json="{}",
        )
        phase3_6_prompt = build_phase3_5_revision_prompt(
            paper_target="IEEE WCL",
            current_sections_json="{}",
            revision_context_json=json.dumps({"frozen_math_contract_json": json.loads(contract)}),
        )

        for prompt in [phase22_prompt, phase23_prompt, phase23_latex_prompt, phase24_prompt, phase3_5_prompt, phase3_6_prompt]:
            self.assertIn("frozen", prompt.lower())
            self.assertIn("w_k", prompt)
        self.assertIn("Reformulation-only variables must not be pushed back", phase22_prompt)
        self.assertIn("must not modify the original optimizer", phase23_prompt)
        self.assertIn("must not redefine original controls", phase24_prompt)
        self.assertIn("Technical Closure Plan", phase24_prompt)
        self.assertIn("Do not introduce auxiliary/reformulation variables", phase3_5_prompt)
        self.assertIn("read-only original-problem interface", phase3_6_prompt)
        self.assertIn("abstract_tex and conclusion_tex must not contain inline mathematical notation", phase3_6_prompt)
        self.assertIn("abstract_tex and conclusion_tex should prefer full-language phrases over letter tags", phase3_6_prompt)
        self.assertIn("If the abstract or Introduction has weak narrative logic", phase3_6_prompt)
        self.assertIn("Do not convert the Introduction into a taxonomy of literatures", phase3_6_prompt)

    def test_phase3_6_rejects_introduction_structure_regression(self) -> None:
        original = (
            r"\section{Introduction}"
            "\n\nMotivation paragraph with a verified claim \\cite{refA}."
            "\n\nRelated work paragraph."
            "\n\nMotivated by this gap, this letter makes the following contributions."
            "\n\\begin{itemize}"
            "\n\\item We add a concrete modeling capability."
            "\n\\item We add a concrete solution capability."
            "\n\\item We add a concrete evaluation contrast."
            "\n\\end{itemize}"
            "\n\nThe remainder of this letter is organized as follows. "
            r"Section~\ref{sec:system_model} presents the system model. "
            r"Section~\ref{sec:proposed_solution} describes the solution. "
            r"Section~\ref{sec:numerical_results} reports results, and "
            r"Section~\ref{sec:conclusion} concludes the letter."
            "\n\n"
            r"\textit{Notation:} Bold lowercase letters denote vectors."
        )
        bad_candidate = (
            r"\section{Introduction}"
            "\n\nMotivation paragraph with a verified claim \\cite{refA}."
            "\n\nThis letter addresses the gap with a concise formulation, solution, and evaluation paragraph."
            "\n\nThe remainder of this letter is organized as follows. "
            r"Section~\ref{sec:system_model} presents the system model. "
            r"Section~\ref{sec:proposed_solution} describes the solution. "
            r"Section~\ref{sec:numerical_results} reports results, and "
            r"Section~\ref{sec:conclusion} concludes the letter."
            "\n\n"
            r"\textit{Notation:} Bold lowercase letters denote vectors."
        )

        revised, note = _phase3_6_validate_revised_section(
            section_name="introduction",
            candidate_text=bad_candidate,
            original_text=original,
            allowed_citation_keys={"refA"},
            allowed_ref_labels={"sec:system_model", "sec:proposed_solution", "sec:numerical_results", "sec:conclusion"},
            forbidden_terms=[],
        )

        self.assertEqual(revised, original)
        self.assertIsNotNone(note)
        self.assertEqual(note["reason"], "introduction_structure_regression")
        self.assertIn("candidate removed the itemized contribution list", note["details"])

    def test_phase3_6_accepts_scoped_qualitative_gain_in_contribution(self) -> None:
        original = (
            r"\section{Introduction}"
            "\n\nA downlink transmitter faces a concrete wireless bottleneck under shared resources \\cite{refA}. "
            "The bottleneck couples service quality, robustness, and resource usage, so a useful design must preserve the physical tradeoff rather than optimizing a detached surrogate."
            "\n\nPrior robust designs address part of this setting while leaving the coupled receiver-service model unresolved. "
            "This gap matters because benchmark comparisons are only meaningful when the same service constraints and reported performance metric are enforced consistently."
            "\n\nHowever, the studied setting requires a single robust covariance formulation that certifies decoding, energy delivery, and sensing illumination. "
            "This motivates a formulation and solution route that connect the technical construction directly to the numerical claims."
            "\n\nIn this letter, we address this problem. The main contributions are summarized as follows."
            "\n\\begin{itemize}"
            "\n\\item We introduce a robust covariance formulation for the coupled service model."
            "\n\\item We derive a deterministic conic counterpart with scoped certificates."
            "\n\\item We define a benchmark comparison without previewing numerical outcomes."
            "\n\\end{itemize}"
            "\n\nThe remainder of this letter is organized as follows. "
            r"Section~\ref{sec:system_model} presents the system model. "
            r"Section~\ref{sec:proposed_solution} describes the solution. "
            r"Section~\ref{sec:numerical_results} reports results, and "
            r"Section~\ref{sec:conclusion} concludes the letter."
            "\n\n"
            r"\textit{Notation:} Bold lowercase letters denote vectors."
        )
        candidate = original.replace(
            "We define a benchmark comparison without previewing numerical outcomes.",
            "Numerical results over the tested configurations indicate that the proposed method improves the reported system utility against the tested benchmark.",
        )

        revised, note = _phase3_6_validate_revised_section(
            section_name="introduction",
            candidate_text=candidate,
            original_text=original,
            allowed_citation_keys={"refA"},
            allowed_ref_labels={"sec:system_model", "sec:proposed_solution", "sec:numerical_results", "sec:conclusion"},
            forbidden_terms=[],
        )

        self.assertEqual(revised, candidate)
        self.assertIsNone(note)

    def test_phase3_6_rejects_unscoped_gain_in_contribution(self) -> None:
        original = (
            r"\section{Introduction}"
            "\n\nA wireless technology enables shared service delivery and exposes a resource bottleneck \\cite{refA}. "
            "This bottleneck couples resource allocation, reliability, and implementation constraints, so a useful design must preserve both the physical model and the final performance metric."
            "\n\nPrior methods leave the coupled robust design unresolved. "
            "In particular, they simplify either the channel-dependent resource coupling or the benchmark comparison, which limits how directly the resulting method can support a paper-level performance claim."
            "\n\nHowever, the considered setting requires a joint formulation. "
            "This motivates a formulation and solution route that keep the model, algorithm, and numerical evidence aligned from the beginning rather than treating experiments as an afterthought."
            "\n\nIn this letter, we address this problem. The main contributions are summarized as follows."
            "\n\\begin{itemize}"
            "\n\\item We introduce a robust covariance formulation."
            "\n\\item We derive a deterministic conic counterpart."
            "\n\\item We show that the proposed method always outperforms existing methods."
            "\n\\end{itemize}"
            "\n\nThe remainder of this letter is organized as follows. "
            r"Section~\ref{sec:system_model} presents the system model. "
            r"Section~\ref{sec:proposed_solution} describes the solution. "
            r"Section~\ref{sec:numerical_results} reports results, and "
            r"Section~\ref{sec:conclusion} concludes the letter."
            "\n\n"
            r"\textit{Notation:} Bold lowercase letters denote vectors."
        )

        revised, note = _phase3_6_validate_revised_section(
            section_name="introduction",
            candidate_text=original,
            original_text=original.replace("always outperforms existing methods", "is evaluated against a benchmark"),
            allowed_citation_keys={"refA"},
            allowed_ref_labels={"sec:system_model", "sec:proposed_solution", "sec:numerical_results", "sec:conclusion"},
            forbidden_terms=[],
        )

        self.assertNotEqual(revised, original)
        self.assertIsNotNone(note)
        self.assertEqual(note["reason"], "introduction_content_quality_regression")
        self.assertTrue(any("overstates numerical evidence" in item for item in note["details"]))

    def test_phase3_5_review_prefers_compiled_phase3_6_revised_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            phase3_3_dir = run_dir / "phase3-3"
            phase3_4_dir = run_dir / "phase3-4"
            phase3_6_dir = run_dir / "phase3-6"
            phase3_3_dir.mkdir(parents=True)
            phase3_4_dir.mkdir(parents=True)
            phase3_6_dir.mkdir(parents=True)

            for path in [
                phase3_3_dir / "abstract.tex",
                phase3_4_dir / "full_paper_preview.tex",
                phase3_4_dir / "full_paper_preview.pdf",
                phase3_4_dir / "full_paper_preview.log",
                phase3_4_dir / "references_ieee.bib",
            ]:
                path.write_text("phase3_4", encoding="utf-8")
            for name in [
                "abstract.tex",
                "revised_full_paper.tex",
                "revised_full_paper_preview.pdf",
                "revised_full_paper_preview.log",
                "verified_references.bib",
                "introduction.tex",
                "system_model_problem_formulation_section.tex",
                "proposed_solution_section.tex",
                "numerical_results_section.tex",
                "conclusion.tex",
            ]:
                (phase3_6_dir / name).write_text("phase3_6", encoding="utf-8")
            (phase3_6_dir / "phase3_6_manifest.json").write_text(
                json.dumps({"compile_status": "ok"}),
                encoding="utf-8",
            )

            paths = _build_phase3_4_full_paper_paths(run_dir)

            self.assertEqual(paths["full_paper_tex"], phase3_6_dir / "revised_full_paper.tex")
            self.assertEqual(paths["full_paper_pdf"], phase3_6_dir / "revised_full_paper_preview.pdf")
            self.assertEqual(paths["verified_references_bib"], phase3_6_dir / "verified_references.bib")

    def test_phase3_6_reference_flow_requires_bib_file_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            phase3_6_dir = Path(tmp)
            for name in [
                "abstract.tex",
                "introduction.tex",
                "system_model_problem_formulation_section.tex",
                "proposed_solution_section.tex",
                "numerical_results_section.tex",
                "conclusion.tex",
            ]:
                (phase3_6_dir / name).write_text("Section text with \\cite{RefA}.", encoding="utf-8")
            (phase3_6_dir / "revised_full_paper.tex").write_text(
                "\\documentclass{IEEEtran}\n"
                "\\begin{document}\n"
                "\\input{abstract.tex}\n"
                "\\bibliographystyle{IEEEtran}\n"
                "\\bibliography{references}\n"
                "\\end{document}\n",
                encoding="utf-8",
            )
            (phase3_6_dir / "references.bib").write_text(
                "@article{RefA, author={A. Author}, title={Paper}, journal={IEEE Wireless Commun. Lett.}, year={2025}}\n",
                encoding="utf-8",
            )

            report = validate_bib_file_reference_flow(phase3_6_dir)

            self.assertTrue(report["ok"])
            self.assertTrue(report["uses_references_database"])
            self.assertEqual(report["references_bib_entry_count"], 1)
            self.assertEqual(report["source_with_handwritten_references"], [])

            (phase3_6_dir / "conclusion.tex").write_text(
                "\\begin{thebibliography}{1}\\bibitem{RefA} A. Author.\\end{thebibliography}",
                encoding="utf-8",
            )
            handwritten_report = validate_bib_file_reference_flow(phase3_6_dir)

            self.assertFalse(handwritten_report["ok"])
            self.assertTrue(handwritten_report["source_with_handwritten_references"])

    def test_abbreviation_check_ignores_roman_section_numbers(self) -> None:
        report = analyze_phase3_4_full_paper_abbreviations(
            {
                "introduction": (
                    "The remainder of this letter is organized as follows. "
                    "Section II gives the model, Section III gives the algorithm, and Section IV gives results."
                )
            }
        )

        self.assertTrue(report["ok"])
        self.assertEqual(report["undefined_abbreviations"], [])

    def test_abbreviation_check_treats_plural_definition_as_singular_acronym(self) -> None:
        report = analyze_phase3_4_full_paper_abbreviations(
            {
                "abstract": (
                    "The access points (APs) coordinate transmission under per-AP power limits."
                )
            }
        )

        self.assertTrue(report["ok"])
        self.assertEqual(report["undefined_abbreviations"], [])

    def test_full_paper_abbreviation_repair_defines_common_wireless_terms(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            phase_dir = Path(tmp)
            files = {
                "abstract.tex": "Abstract text without acronyms.",
                "introduction.tex": "Integrated sensing, communication, and powering motivates the model.",
                "system_model_problem_formulation_section.tex": (
                    "The SWIPT receiver observes RF power, and the WPT receiver harvests DC power. "
                    "The SINR target is imposed under CSI uncertainty."
                ),
                "proposed_solution_section.tex": "The safe counterpart is an SDP/SOCP.",
                "numerical_results_section.tex": (
                    "\\item \\emph{maximum-ratio transmission (MRT)}: "
                    "fixed-layout maximum-ratio transmission (MRT), which is compared."
                ),
                "conclusion.tex": "Conclusion text.",
            }
            for filename, content in files.items():
                (phase_dir / filename).write_text(content, encoding="utf-8")

            repair = _phase3_5_apply_full_paper_abbreviation_repairs(phase_dir)
            report = analyze_phase3_4_full_paper_abbreviations_from_phase_dir(phase_dir)
            proposed_text = (phase_dir / "proposed_solution_section.tex").read_text(encoding="utf-8")
            results_text = (phase_dir / "numerical_results_section.tex").read_text(encoding="utf-8")

            self.assertTrue(report["ok"])
            self.assertIn("SWIPT", repair["applied_repairs"]["system_model"])
            self.assertIn("a semidefinite programming (SDP)/second-order cone programming (SOCP) problem", proposed_text)
            self.assertIn("\\item \\emph{MRT}: fixed-layout maximum-ratio transmission", results_text)
            self.assertNotIn("maximum-ratio transmission (MRT): fixed-layout maximum-ratio transmission (MRT)", results_text)

    def test_full_paper_abbreviation_repair_normalizes_malformed_xl_mimo_definition(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            phase_dir = Path(tmp)
            files = {
                "abstract.tex": "Abstract text without acronyms.",
                "introduction.tex": (
                    "Codebook and capacity-oriented xl multiple-input multiple-output "
                    "(XL multiple-input multiple-output (MIMO)) studies motivate the setting."
                ),
                "system_model_problem_formulation_section.tex": "System text.",
                "proposed_solution_section.tex": "Proposed text.",
                "numerical_results_section.tex": "Numerical text.",
                "conclusion.tex": "Conclusion text.",
            }
            for filename, content in files.items():
                (phase_dir / filename).write_text(content, encoding="utf-8")

            _phase3_5_apply_full_paper_abbreviation_repairs(phase_dir)
            report = analyze_phase3_4_full_paper_abbreviations_from_phase_dir(phase_dir)
            intro_text = (phase_dir / "introduction.tex").read_text(encoding="utf-8")

            self.assertTrue(report["ok"])
            self.assertIn("extremely large-scale multiple-input multiple-output (XL-MIMO)", intro_text)
            self.assertNotIn("XL multiple-input multiple-output (MIMO)", intro_text)

    def test_contract_scope_check_allows_derived_x_but_blocks_reformulation_symbols(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            phase_dir = run_dir / "phase3-5"
            (run_dir / "phase2-1").mkdir(parents=True)
            phase_dir.mkdir(parents=True)
            (run_dir / "phase2-1" / "mathematical_contract.json").write_text(
                json.dumps(
                    {
                        "derived_quantities": [{"symbol": "\\mathbf X"}],
                        "reformulation_only": [{"symbol": "\\lambda_m"}, {"symbol": "\\mu_m"}],
                    }
                ),
                encoding="utf-8",
            )
            (phase_dir / "system_model_problem_formulation_section.tex").write_text(
                "The total covariance $\\mathbf X$ is derived from the physical controls.",
                encoding="utf-8",
            )
            (phase_dir / "proposed_solution_section.tex").write_text(
                "The conic counterpart uses $\\bm\\lambda$ and $\\bm\\mu$.",
                encoding="utf-8",
            )

            ok_report = _phase3_5_contract_scope_check(run_dir, phase_dir)
            (phase_dir / "system_model_problem_formulation_section.tex").write_text(
                "The original model incorrectly uses $\\bm\\lambda$.",
                encoding="utf-8",
            )
            bad_report = _phase3_5_contract_scope_check(run_dir, phase_dir)

        self.assertTrue(ok_report["ok"])
        self.assertIn("\\mathbf X", ok_report["derived_symbols_allowed_in_system_model"])
        self.assertFalse(bad_report["ok"])
        self.assertEqual(bad_report["violations"][0]["symbol"], "\\bm\\lambda")

    def test_phase3_6_intro_fix_defines_bs_at_first_use(self) -> None:
        revised, applied = _apply_phase3_6_deterministic_intro_fixes(
            "We formulate a robust downlink problem with a BS power budget and nonlinear rectifier constraints."
        )

        self.assertIn("base station (BS) power budget", revised)
        self.assertTrue(any(item.get("issue_id") == "P2-ACR-BS" for item in applied))

    def test_phase3_6_technical_fix_scopes_solve_p0_language(self) -> None:
        _, revised_proposed, _, applied = _apply_phase3_6_deterministic_technical_fixes(
            system_text="System.",
            proposed_text="To solve problem (P0), we develop a finite-scenario method.",
            conclusion_text="Conclusion.",
        )

        self.assertIn("To address problem (P0)", revised_proposed)
        self.assertIn("finite-scenario method", revised_proposed)
        self.assertNotIn("To solve problem (P0)", revised_proposed)
        self.assertTrue(any(item.get("issue_id") == "P0-THE-01" for item in applied))

    def test_phase3_6_technical_fix_shortens_x_shorthand_for_two_column_layout(self) -> None:
        _, revised_proposed, _, applied = _apply_phase3_6_deterministic_technical_fixes(
            system_text="System.",
            proposed_text=(
                "With these fixed coefficients, the solver-facing conic program is\n"
                "\\min_{\\{\\mathbf{W}_k\\}_{k\\in\\mathcal{K}},\\,\\mathbf{V}}\\quad\n"
                "& \\sum_{m\\in\\mathcal{M}}\\left(\n"
                "\\sum_{k\\in\\mathcal{K}}\\operatorname{tr}(\\mathbf{E}_m\\mathbf{W}_k)\n"
                "+\\operatorname{tr}(\\mathbf{E}_m\\mathbf{V})\\right)\n"
                "\\label{obj:p1_power}\n"
                "& \\operatorname{tr}(\\mathbf{H}_{\\mathrm{C},k}\\mathbf{W}_k)\n"
                "-\\gamma_k\\sum_{i\\in\\mathcal{K},\\,i\\ne k}\n"
                "\\operatorname{tr}(\\mathbf{H}_{\\mathrm{C},k}\\mathbf{W}_i)\n"
                "\\nonumber\\\\\n"
                "&\\quad\n"
                "-\\gamma_k\\operatorname{tr}(\\mathbf{H}_{\\mathrm{C},k}\\mathbf{V})\n"
                "\\ge \\gamma_k\\sigma_k^2,\n"
                "\\begin{equation}\n"
                "\\mathbf{X}(\\{\\mathbf{W}_k\\},\\mathbf{V})=\\sum_{k\\in\\mathcal{K}}\\mathbf{W}_k+\\mathbf{V}\n"
                "\\end{equation}\n"
                "\\operatorname{tr}\\!\\left(\\mathbf{A}_l\\mathbf{X}(\\{\\mathbf{W}_k\\},\\mathbf{V})\\right)\n"
                "& \\sum_{k\\in\\mathcal{K}}\\operatorname{tr}(\\mathbf{E}_m\\mathbf{W}_k)\n"
                "+\\operatorname{tr}(\\mathbf{E}_m\\mathbf{V})\n"
                "\\le P_{\\max,m},"
            ),
            conclusion_text="Conclusion.",
        )

        self.assertIn("\\mathbf{X}=\\sum_{k\\in\\mathcal{K}}\\mathbf{W}_k+\\mathbf{V}", revised_proposed)
        self.assertIn("\\langle\\mathbf{A}_l,\\mathbf{X}\\rangle", revised_proposed)
        self.assertIn("\\langle\\mathbf{H}_{\\mathrm{C},k},\\mathbf{W}_i\\rangle", revised_proposed)
        self.assertIn("\\langle\\mathbf{E}_m,\\mathbf{X}\\rangle", revised_proposed)
        self.assertIn("\\langle\\mathbf{A},\\mathbf{B}\\rangle\\triangleq\\operatorname{tr}(\\mathbf{A}\\mathbf{B})", revised_proposed)
        self.assertIn("\\min_{\\mathbf{W},\\,\\mathbf{V}}\\quad", revised_proposed)
        self.assertIn("& P_{\\mathrm{tot}}(\\mathbf{W},\\mathbf{V})", revised_proposed)
        self.assertIn("&\\quad -\\gamma_k\\sum_{i\\in\\mathcal{K},\\,i\\ne k}", revised_proposed)
        self.assertNotIn("\\mathbf{X}(\\{\\mathbf{W}_k\\},\\mathbf{V})", revised_proposed)
        self.assertNotIn("\\min_{\\{\\mathbf{W}_k\\}_{k\\in\\mathcal{K}},\\,\\mathbf{V}}\\quad", revised_proposed)
        self.assertTrue(any(item.get("issue_id") == "P0-THE-01" for item in applied))

    def test_phase3_6_technical_fix_compacts_solver_transcript_algorithm(self) -> None:
        _, revised_proposed, _, applied = _apply_phase3_6_deterministic_technical_fixes(
            system_text="System.",
            proposed_text=(
                "The complete procedure is summarized in Algorithm~\\ref{alg:solution_proposed}. "
                "A primal-dual interior-point implementation is used for moderate dimensions; for larger deployments, "
                "the same conic form can be passed to a first-order SDP solver, with acceptance governed by the residual and gap checks.\n"
                "\\begin{algorithm}[!t]\n"
                "\\caption{Covariance-Domain SDP Algorithm}\n"
                "\\label{alg:solution_proposed}\n"
                "\\begin{algorithmic}[1]\n"
                "\\Require Channels, sensing matrices, thresholds, conversion coefficients, AP budgets, tolerances, and \\(T_{\\max}\\).\n"
                "\\Statex \\hspace{\\algorithmicindent}Tolerances are \\(\\epsilon_{\\mathrm{p}}\\), \\(\\epsilon_{\\mathrm{d}}\\), and \\(\\epsilon_{\\mathrm{g}}\\).\n"
                "\\State Construct problem (P1).\n"
                "\\Repeat\n"
                "\\State Evaluate primal, dual, PSD-cone residuals, and relative gap.\n"
                "\\State Choose a PSD-preserving step size that reduces conic residuals.\n"
                "\\Until{\\(t=T_{\\max}\\)}\n"
                "\\State \\Return \\(\\{\\mathbf{W}_k^\\star\\}_{k\\in\\mathcal{K}}\\), \\(\\mathbf{V}^\\star\\), residuals, gap, and iteration count.\n"
                "\\end{algorithmic}\n"
                "\\end{algorithm}\n"
                "If problem (P1) is feasible and an optimizer exists, an exact conic optimum is globally optimal for the covariance formulation of problem (P0). "
                "With numerical solvers, the statement is interpreted up to the reported primal residual, dual residual, and relative gap. "
                "When a Slater point exists, the associated dual variables can also be used for local sensitivity interpretation; otherwise, feasibility certificates and marginal values should be read through the solver diagnostics.\n"
                "First-order conic solvers reduce per-iteration linear algebra but typically require more iterations and should be reported with their residuals and objective gap."
            ),
            conclusion_text="Conclusion.",
        )

        self.assertIn("Algorithm~\\ref{alg:solution_proposed} summarizes the paper-facing design flow", revised_proposed)
        self.assertIn("final feasibility check is performed using the original constraints", revised_proposed)
        self.assertIn("\\caption{Covariance-Domain SDP Design}", revised_proposed)
        self.assertIn("\\Ensure Optimized covariances", revised_proposed)
        self.assertIn("\\State Solve problem (P1) using a semidefinite-programming solver.", revised_proposed)
        self.assertNotIn("\\Statex", revised_proposed)
        self.assertNotIn("\\Repeat", revised_proposed)
        self.assertNotIn("primal, dual", revised_proposed)
        self.assertNotIn("primal residual", revised_proposed)
        self.assertNotIn("dual residual", revised_proposed)
        self.assertNotIn("Solver-specific residuals", revised_proposed)
        self.assertNotIn("relative gap", revised_proposed)
        self.assertNotIn("iteration count", revised_proposed)
        self.assertNotIn("T_{\\max}", revised_proposed)
        self.assertIn("This statement is scoped to the stated reformulation", revised_proposed)
        self.assertTrue(any(item.get("issue_id") == "P1-ALG-COMPACT" for item in applied))

    def test_phase3_6_technical_fix_defines_common_wireless_abbreviations(self) -> None:
        revised_system, _, _, applied = _apply_phase3_6_deterministic_technical_fixes(
            system_text=(
                "Meanwhile, the received RF power at energy receiver j follows as stated. "
                "Consequently, under deterministic CSI, the problem is convex. "
                "The burden is high-dimensional PSD covariances. "
                "Therefore, problem (P0) is an SDP in the covariance variables."
            ),
            proposed_text="Proposed.",
            conclusion_text="Conclusion.",
        )

        self.assertIn("received radio-frequency (RF) power", revised_system)
        self.assertIn("deterministic channel state information (CSI)", revised_system)
        self.assertIn("positive semidefinite (PSD) covariances", revised_system)
        self.assertIn("semidefinite programming (SDP) problem", revised_system)
        self.assertTrue(any(item.get("issue_id") == "P1-ABBR-RF" for item in applied))
        self.assertTrue(any(item.get("issue_id") == "P1-ABBR-CSI" for item in applied))
        self.assertTrue(any(item.get("issue_id") == "P1-ABBR-PSD" for item in applied))
        self.assertTrue(any(item.get("issue_id") == "P1-ABBR-SDP" for item in applied))

    def test_phase3_6_technical_fix_normalizes_wsr_quantifiers_and_scenario_phrase(self) -> None:
        revised_system, revised_proposed, _, applied = _apply_phase3_6_deterministic_technical_fixes(
            system_text=(
                r"The coordinate of element $m$ is $\mathbf r_m\in\mathbb R^D$, where "
                r"$\mathcal R_m\subset\mathbb R^D$ denotes its bounded movement region. "
                "The spectral efficiency of user $k$ and the weighted sum spectral efficiency are then\n"
                r"\begin{equation}U_{\rm weighted sum rate (WSR)}(\mathbf r,\mathbf w)=\sum_k R_k\end{equation}"
                r"\begin{subequations}\begin{align}"
                r"& \mathbf r_m\in\mathcal R_m, \label{con:p0_aperture}\\"
                r"& \|\mathbf r_m-\mathbf r_n\|_2\ge d_{\min}. \label{con:p0_spacing}"
                r"\end{align}\end{subequations}"
            ),
            proposed_text="To address problem (P0) through a tractable finite-scenario approximation, we develop a method.",
            conclusion_text="Conclusion.",
        )

        self.assertIn("weighted sum rate (WSR) utility", revised_system)
        self.assertIn(r"U_{\rm WSR}", revised_system)
        self.assertIn(r"\forall m", revised_system)
        self.assertIn(r"\forall (m,n)\in\mathcal E", revised_system)
        self.assertNotIn("finite-scenario approximation", revised_proposed)
        self.assertTrue(any(item.get("issue_id") == "P1-MATH-001" for item in applied))

    def test_phase3_5_downgrades_author_metadata_from_p0_to_p2(self) -> None:
        issue = {
            "issue_id": "P0-META",
            "title": "Finalize author and submission metadata",
            "category": "submission readiness",
            "issue": "The author field appears placeholder-like.",
            "auto_fixable": False,
            "requires_new_experiment": False,
            "requires_reference_verification": False,
            "requires_manual_theory_verification": False,
        }

        adjusted = _apply_phase3_4_evidence_adjustments(
            {
                "overall_score": 7.1,
                "recommendation": "minor_revision_needed",
                "likely_reviewer_decision_estimate": "borderline",
                "dimension_scores": {},
                "revision_plan": {"P0": [issue], "P1": [], "P2": []},
            },
            phase25_summary={},
            missing_citation_keys=[],
            arxiv_only_entries=[],
            compile_warnings_summary=[],
            forbidden_terms_found=[],
        )

        self.assertEqual(adjusted["revision_plan"]["P0"], [])
        self.assertEqual(adjusted["critical_issues"], [])
        self.assertEqual(adjusted["revision_plan"]["P2"][0]["issue_id"], "P0-META")
        self.assertEqual(adjusted["revision_plan"]["P2"][0]["priority_adjustment"], "downgraded_to_P2_manual_packaging")

    def test_phase3_5_filters_false_missing_figure_issue_when_final_figures_compile(self) -> None:
        issue = {
            "issue_id": "P0-FIG",
            "title": "Possible missing numerical figures",
            "category": "experiments",
            "issue": "The numerical section references figures, but figures may be missing.",
            "auto_fixable": False,
            "requires_new_experiment": False,
            "requires_reference_verification": False,
            "requires_manual_theory_verification": False,
        }

        adjusted = _apply_phase3_4_evidence_adjustments(
            {
                "overall_score": 7.1,
                "recommendation": "minor_revision_needed",
                "likely_reviewer_decision_estimate": "borderline",
                "dimension_scores": {},
                "revision_plan": {"P0": [issue], "P1": [], "P2": []},
            },
            phase25_summary={},
            missing_citation_keys=[],
            arxiv_only_entries=[],
            compile_warnings_summary=[],
            forbidden_terms_found=[],
            final_figure_check={"ok": True},
        )

        self.assertEqual(adjusted["revision_plan"]["P0"], [])
        self.assertEqual(adjusted["critical_issues"], [])

    def test_phase23_latex_blocks_system_model_repetition_in_reformulation(self) -> None:
        prompt = build_phase2_phase3_latex_prompt(
            topic="generic wireless optimization",
            mathematical_contract_json=json.dumps(
                {
                    "derived_quantities": [
                        {"symbol": "q_k(w)", "status": "derived_quantity"},
                        {"symbol": "Gamma_k(w,rho)", "status": "derived_quantity"},
                        {"symbol": "R_k(w,rho)", "status": "derived_quantity"},
                    ]
                }
            ),
            algorithm_md="Use candidate evaluation without redefining the system model.",
            convergence_or_complexity_md="finite heuristic",
            benchmark_definition_md="baseline",
        )
        self.assertIn("Do not restate the System Model", prompt)
        self.assertIn("must be referenced textually", prompt)
        self.assertIn("at most one compact display", prompt)
        self.assertIn("Solution Approach", prompt)
        self.assertIn("Avoid overfull two-column displays", prompt)
        self.assertIn("Do not repeat the final convexity/technical-difficulty paragraph", prompt)
        self.assertIn("start from the solver idea", prompt)

        bad = (
            r"\subsection{Problem Reformulation}"
            "\nThe received RF signal power before splitting is"
            "\n\\begin{equation}"
            r"q_k(\mathbf w)=\sum_j|\mathbf h_k^H\mathbf w_j|^2."
            r"\label{eq:solution_received_power}"
            "\n\\end{equation}"
            "\nThe SINR is"
            "\n\\begin{equation}"
            r"\Gamma_k(\mathbf w,\rho_k)=\frac{\rho_k|\mathbf h_k^H\mathbf w_k|^2}{\sigma^2}."
            r"\label{eq:solution_sinr}"
            "\n\\end{equation}"
            "\n\\subsection{Proposed Algorithm}"
        )
        good = (
            r"\subsection{Solution Approach}"
            "\nThe method keeps the original physical variables and evaluates candidates using the rate and "
            "harvested-energy metrics defined in the system model."
            "\n\\subsection{Proposed Algorithm}"
        )
        self.assertFalse(analyze_phase3_reformulation_repeats_system_model(bad)["ok"])
        self.assertTrue(analyze_phase3_reformulation_repeats_system_model(good)["ok"])

    def test_phase23_solution_prompt_requires_intro_and_algorithm_block(self) -> None:
        contract = json.dumps(
            {
                "controls": [{"symbol": "x", "status": "control"}],
                "objective": {"sense": "max", "expression": "U(x)"},
                "constraints": [{"id": "budget", "relation": "x <= P"}],
            }
        )
        design_prompt = build_phase2_phase3_prompt(
            topic="generic wireless optimization",
            handoff={"final_title": "Prompt Structure Test"},
            mathematical_contract_json=contract,
            system_model_md="system",
            problem_formulation_md="problem",
            core_theory_package_md="theory",
            convexity_audit_md="audit",
            reformulation_path_md="reformulation",
        )
        latex_prompt = build_phase2_phase3_latex_prompt(
            topic="generic wireless optimization",
            mathematical_contract_json=contract,
            algorithm_md="algorithm flow",
            convergence_or_complexity_md="complexity",
            benchmark_definition_md="benchmark",
        )
        phase3_1_prompt = build_phase3_1_writing_prompt(
            topic="generic wireless optimization",
            mathematical_contract_json=contract,
            system_model_md="system facts",
            problem_formulation_md="problem facts",
            core_theory_package_md="nonconvexity notes",
            convexity_audit_md="audit",
            reformulation_path_md="path",
            algorithm_md="algorithm flow",
            convergence_or_complexity_md="complexity",
            benchmark_definition_md="benchmark",
        )

        self.assertIn('distinct compact paper-facing "Algorithm flow" block', design_prompt)
        self.assertIn("4-7 numbered executable steps", design_prompt)
        self.assertIn("not the paper-facing Algorithm 1", design_prompt)
        self.assertIn("do not restate the original system model", design_prompt)
        self.assertIn("do not open by explaining why the original problem is nonconvex or hard", design_prompt)
        self.assertIn("selected route must be supported by the current mathematical structure", design_prompt)
        self.assertIn("call it heuristic/proxy-based", design_prompt)
        self.assertIn("identify active closure dimensions from explicit artifacts, not keywords", design_prompt)
        self.assertIn("Technical Contribution Contract", design_prompt)
        self.assertIn("active contribution mode", design_prompt)
        self.assertIn("do not fabricate an iteration loop for a direct one-shot convex/conic method", design_prompt)
        self.assertIn('must not collapse to only "assemble problem, solve, return"', design_prompt)
        self.assertIn("Start with one short unheaded section-level introduction paragraph", latex_prompt)
        self.assertIn("Use at most two subsections by default", latex_prompt)
        self.assertIn("Do not create separate subsections for every block update", latex_prompt)
        self.assertIn("must include exactly one standard `algorithm` environment", latex_prompt)
        self.assertIn("compact paper-facing method skeleton", latex_prompt)
        self.assertIn("Do not list generic solver internals", latex_prompt)
        self.assertIn("Do not assume that the wrapper section title is always", latex_prompt)
        self.assertIn("Do not open this section by restating why the problem is hard", latex_prompt)
        self.assertIn("To solve problem (P0), we develop", latex_prompt)
        self.assertIn("For scalar resource updates", latex_prompt)
        self.assertIn("standard convex optimization tools", latex_prompt)
        self.assertIn("resolve the active contribution mode", latex_prompt)
        self.assertIn("direct exact/conic/certified route", latex_prompt)
        self.assertIn('must not reduce to only "form problem, solve problem, return solution"', latex_prompt)
        self.assertIn("Phase 3.1, the first paper-writing checkpoint", phase3_1_prompt)
        self.assertIn("writes the two technical paper sections together", phase3_1_prompt)
        self.assertIn("Do not change the mathematics", phase3_1_prompt)
        self.assertIn("First-pass authoring principle", phase3_1_prompt)
        self.assertIn("infer the active technical-closure obligations from the frozen artifacts, not from isolated keywords", phase3_1_prompt)
        self.assertIn("resolve the active contribution mode", phase3_1_prompt)
        self.assertIn("Proof obligations are conditional", phase3_1_prompt)
        self.assertIn('do not let the block collapse to only "assemble problem, solve, return"', phase3_1_prompt)
        self.assertIn("End this section with a compact convexity/technical-difficulty", phase3_1_prompt)
        self.assertIn("Do not repeat the System Model or the convexity/technical-difficulty diagnosis", phase3_1_prompt)
        self.assertIn("compact paper-facing method skeleton", phase3_1_prompt)
        self.assertIn("No symbol is first defined twice", phase3_1_prompt)
        self.assertIn("not a tutorial and not an agent-generated list of equations", phase3_1_prompt)
        self.assertIn("is given by", phase3_1_prompt)
        self.assertIn("can be written as", phase3_1_prompt)
        self.assertIn("where ... denotes", phase3_1_prompt)
        self.assertIn("When one sentence introduces two displayed relations", phase3_1_prompt)
        self.assertIn(r"\text{and}\quad", phase3_1_prompt)
        self.assertIn("respectively", phase3_1_prompt)
        self.assertIn("Avoid dumping all variables", phase3_1_prompt)
        self.assertIn("Do not explain every equation", phase3_1_prompt)
        self.assertIn("Avoid tutorial-style exposition", phase3_1_prompt)
        self.assertIn("Avoid emphatic or AI-like", phase3_1_prompt)
        self.assertIn("Avoid meta-writing", phase3_1_prompt)
        self.assertIn("formal problem must use a numbered optimization layout", phase3_1_prompt)
        self.assertIn("Do not use `align*`", phase3_1_prompt)
        self.assertIn("Do not write objective and constraints as disconnected prose", phase3_1_prompt)
        self.assertIn("After the display, explain the role of each constraint group", phase3_1_prompt)
        self.assertIn("what engineering decision is being made", phase3_1_prompt)

    def test_phase23_section_title_inference_is_content_adaptive(self) -> None:
        self.assertEqual(
            infer_phase3_section_title(r"\begin{algorithm}[!t]\caption{Method}\end{algorithm}"),
            "Proposed Algorithm",
        )
        self.assertEqual(
            infer_phase3_section_title("The method uses a surrogate reformulation and decomposition."),
            "Proposed Method",
        )

    def test_phase21_semantic_contract_guard_blocks_role_promotion(self) -> None:
        contract = json.dumps(
            {
                "controls": [
                    {"symbol": "w_k", "status": "control"},
                    {"symbol": "rho_k", "status": "control"},
                ],
                "derived_quantities": [
                    {"symbol": "q_k(w,rho)", "status": "derived_quantity"},
                    {"symbol": "Gamma_k(w,rho)", "status": "derived_quantity"},
                    {"symbol": "R_k(w,rho)", "status": "derived_quantity"},
                ],
                "objective": {
                    "sense": "max",
                    "expression": "U(w,rho)",
                    "terms": [{"expression": "R_k(w,rho)", "uses_symbols": ["R_k"]}],
                },
                "constraints": [{"id": "q_region", "relation": "q_k(w,rho)<=q_max", "uses_symbols": ["q_k"]}],
                "reformulation_only": [
                    {"symbol": "W_k", "status": "reformulation_only"},
                    {"symbol": "hat_q_k", "status": "reformulation_only"},
                ],
            }
        )
        good_tex = (
            r"\begin{align*}"
            r"\text{(P0)}\quad"
            r"\max_{\{\mathbf{w}_k\}_{k\in\mathcal{K}},\,\{\rho_k\}_{k\in\mathcal{K}}}\quad & U(\mathbf{w},\boldsymbol{\rho})\\"
            r"\text{s.t.}\quad & q_k(\mathbf{w},\boldsymbol{\rho})\leq q_k^{\max}"
            r"\end{align*}"
        )
        bad_tex = (
            r"\begin{align*}"
            r"\text{(P0)}\quad"
            r"\max_{\{\mathbf{w}_k,\rho_k,q_k,\mathbf{W}_k,\hat{q}_k\}_{k\in\mathcal{K}}}\quad & U(\mathbf{w},\boldsymbol{\rho})\\"
            r"\text{s.t.}\quad & q_k\leq q_k^{\max}"
            r"\end{align*}"
        )

        self.assertEqual(latex_math_contract_issue_summary(good_tex, contract), "")
        summary = latex_math_contract_issue_summary(bad_tex, contract)
        self.assertIn("mathematical-role consistency", summary)
        self.assertIn("q_k(w,rho)", summary)
        self.assertIn("W_k", summary)
        self.assertIn("hat_q_k", summary)

    def test_phase21_contract_normalizer_separates_status_from_usage(self) -> None:
        normalized = normalize_phase2_phase1_mathematical_contract(
            {
                "controls": [
                    {
                        "symbol": "w_k",
                        "status": "control",
                    }
                ],
                "derived_quantities": [
                    {
                        "symbol": "q_k(w,rho)",
                        "status": "derived_quantity",
                        "used_in": ["objective", "constraint:q_region"],
                        "appears_in_optimizer": True,
                    }
                ],
                "objective": {
                    "sense": "max",
                    "expression": "U(w,rho)",
                    "terms": [{"expression": "R_k(w,rho)", "uses_symbols": ["R_k"]}],
                },
                "constraints": [
                    {
                        "id": "q_region",
                        "relation": "q_k(w,rho)<=q_max",
                        "uses_symbols": ["q_k", "q_max"],
                    }
                ],
                "reformulation_only": [{"symbol": "W_k", "status": "reformulation_only"}],
            }
        )

        self.assertTrue(normalized["controls"][0]["appears_in_optimizer"])
        self.assertFalse(normalized["derived_quantities"][0]["appears_in_optimizer"])
        self.assertEqual(normalized["derived_quantities"][0]["status"], "derived_quantity")
        self.assertEqual(normalized["constraints"][0]["id"], "q_region")
        self.assertFalse(normalized["reformulation_only"][0]["allowed_in_original_problem"])
        report = validate_phase2_phase1_mathematical_contract_schema(normalized)
        self.assertTrue(report["ok"])

    def test_phase21_contract_normalizer_removes_duplicate_noncontrol_roles(self) -> None:
        report = validate_phase2_phase1_mathematical_contract_schema(
            {
                "controls": [
                    {"symbol": "\\gamma_k", "meaning": "optimization variable"},
                    {"symbol": "\\eta_k", "meaning": "auxiliary optimization variable"},
                ],
                "derived_quantities": [
                    {"symbol": "\\gamma_k", "meaning": "duplicate derived expression"},
                    {"symbol": "r_k", "meaning": "rate expression"},
                ],
                "reformulation_only": [
                    {"symbol": "\\eta_k", "meaning": "duplicate auxiliary"},
                    {"symbol": "z_k", "meaning": "true reformulation-only variable"},
                ],
                "objective": {"sense": "max", "expression": "sum_k r_k"},
                "constraints": [{"id": "c1", "relation": "\\gamma_k <= r_k"}],
            }
        )

        self.assertTrue(report["ok"])
        normalized = report["normalized_contract"]
        self.assertEqual([item["symbol"] for item in normalized["derived_quantities"]], ["r_k"])
        self.assertEqual([item["symbol"] for item in normalized["reformulation_only"]], ["z_k"])

    def test_phase21_rejects_parameter_domain_conditions_as_p0_constraints(self) -> None:
        report = validate_phase2_phase1_mathematical_contract_schema(
            {
                "controls": [{"symbol": "\\mathbf w_k", "meaning": "beamforming vector"}],
                "parameters": [
                    {"symbol": "r_k", "meaning": "fixed user range"},
                    {"symbol": "\\omega_k", "meaning": "fixed user weight"},
                    {"symbol": "\\sigma_k^2", "meaning": "fixed noise variance"},
                ],
                "objective": {"sense": "max", "expression": "\\sum_k \\omega_k R_k"},
                "derived_quantities": [
                    {
                        "symbol": "R_k",
                        "definition": "\\log_2(1+\\mathrm{SINR}_k(\\mathbf w))",
                        "depends_on": ["\\mathbf w_k"],
                    }
                ],
                "constraints": [
                    {"id": "C-PWR", "relation": "\\sum_k\\|\\mathbf w_k\\|_2^2\\le P_{\\max}", "uses_symbols": ["\\mathbf w_k"]},
                    {"id": "C-NF", "relation": "r_k\\in(R_{\\rm react},R_F)", "uses_symbols": ["r_k"]},
                    {"id": "C-WGT", "relation": "\\omega_k\\ge0", "uses_symbols": ["\\omega_k"]},
                    {"id": "C-NOISE", "relation": "\\sigma_k^2>0", "uses_symbols": ["\\sigma_k^2"]},
                ],
            }
        )

        self.assertFalse(report["ok"])
        error_text = " ".join(report["errors"])
        self.assertIn("parameter-domain", error_text)
        self.assertIn("C-NF", error_text)
        self.assertIn("C-WGT", error_text)
        self.assertIn("C-NOISE", error_text)
        self.assertNotIn("C-PWR", error_text)

    def test_phase21_allows_ris_unit_modulus_control_constraint(self) -> None:
        report = validate_phase2_phase1_mathematical_contract_schema(
            {
                "controls": [
                    {
                        "symbol": "\\{\\mathbf w_k\\}_{k=1}^K",
                        "meaning": "BS active beamforming vectors for independent downlink streams",
                        "domain": "\\mathbf w_k\\in\\mathbb C^M",
                    },
                    {
                        "symbol": "\\boldsymbol\\theta",
                        "meaning": "passive RIS reflection coefficients",
                        "domain": "\\boldsymbol\\theta\\in\\mathbb C^N",
                    },
                ],
                "objective": {"sense": "max", "expression": "U_\\alpha(\\{\\mathbf w_k\\},\\boldsymbol\\theta)"},
                "constraints": [
                    {
                        "id": "bs_sum_power",
                        "relation": "\\sum_{k=1}^K\\|\\mathbf w_k\\|_2^2\\le P_{\\max}",
                        "uses_symbols": ["\\{\\mathbf w_k\\}_{k=1}^K", "P_{\\max}"],
                    },
                    {
                        "id": "ris_unit_modulus",
                        "relation": "|\\theta_n|=1,\\ n=1,\\ldots,N",
                        "uses_symbols": ["\\boldsymbol\\theta", "N"],
                    },
                ],
            }
        )

        self.assertTrue(report["ok"], report["errors"])

    def test_phase21_rejects_mixed_coordinate_convention_in_contract(self) -> None:
        report = validate_phase2_phase1_mathematical_contract_schema(
            {
                "controls": [
                    {
                        "symbol": "\\mathbf q[n]",
                        "meaning": "UAV 3D waypoint at slot n",
                        "domain": "\\mathbb{R}^3",
                    }
                ],
                "parameters": [
                    {
                        "symbol": "\\mathbf w_k",
                        "meaning": "ground IoT device location",
                        "source": "deployment",
                        "domain": "\\mathbb{R}^2",
                    },
                    {"symbol": "H[n]", "meaning": "UAV altitude"},
                ],
                "derived_quantities": [
                    {
                        "symbol": "h_k[n]",
                        "meaning": "air-to-ground channel gain",
                        "definition": "\\beta_0/(\\|\\mathbf q[n]-\\mathbf w_k\\|^2+H[n]^2)",
                        "depends_on": ["\\mathbf q[n]", "\\mathbf w_k", "H[n]"],
                        "used_in": ["rate"],
                    }
                ],
                "objective": {"sense": "max", "expression": "\\sum_k R_k"},
                "constraints": [{"id": "speed", "relation": "\\|\\mathbf q[n+1]-\\mathbf q[n]\\|\\le V_{max}\\Delta"}],
            }
        )

        self.assertFalse(report["ok"])
        self.assertTrue(any("Geometry convention mixes" in item for item in report["errors"]))

    def test_phase21_allows_horizontal_coordinate_plus_altitude_convention(self) -> None:
        report = validate_phase2_phase1_mathematical_contract_schema(
            {
                "controls": [
                    {
                        "symbol": "\\mathbf q[n]",
                        "meaning": "UAV horizontal waypoint at slot n",
                        "domain": "\\mathbb{R}^2",
                    }
                ],
                "parameters": [
                    {
                        "symbol": "\\mathbf w_k",
                        "meaning": "ground IoT device location",
                        "source": "deployment",
                        "domain": "\\mathbb{R}^2",
                    },
                    {"symbol": "H[n]", "meaning": "UAV altitude"},
                ],
                "derived_quantities": [
                    {
                        "symbol": "h_k[n]",
                        "meaning": "air-to-ground channel gain under horizontal coordinates plus altitude",
                        "definition": "\\beta_0/(\\|\\mathbf q[n]-\\mathbf w_k\\|^2+H[n]^2)",
                        "depends_on": ["\\mathbf q[n]", "\\mathbf w_k", "H[n]"],
                        "used_in": ["rate"],
                    }
                ],
                "objective": {"sense": "max", "expression": "\\sum_k R_k"},
                "constraints": [{"id": "speed", "relation": "\\|\\mathbf q[n+1]-\\mathbf q[n]\\|\\le V_{max}\\Delta"}],
            }
        )

        self.assertTrue(report["ok"], report["errors"])

    def test_phase21_allows_decorated_reformulation_auxiliary_symbol(self) -> None:
        report = validate_phase2_phase1_mathematical_contract_schema(
            {
                "controls": [
                    {
                        "symbol": "a_{mk}",
                        "meaning": "binary user-centric clustering variable",
                        "domain": "{0,1}",
                    }
                ],
                "objective": {"sense": "max", "expression": "t"},
                "constraints": [{"id": "cluster_budget", "relation": "\\sum_m a_{mk}\\le L"}],
                "reformulation_only": [
                    {
                        "symbol": "\\tilde a_{mk}",
                        "meaning": "continuous relaxation of the binary clustering variable used only in the algorithmic route",
                    }
                ],
            }
        )

        self.assertTrue(report["ok"], report["errors"])

    def test_phase21_runtime_requires_dynamic_llm_generation(self) -> None:
        source = (SCRIPTS_DIR / "phase_runtime" / "phase21_23_foundation.py").read_text(encoding="utf-8")
        start = source.index("def run_phase2_phase1_llm")
        end = source.index("def run_phase2_phase1_latex_llm")
        body = source[start:end]

        self.assertIn("llm.chat", body)
        self.assertNotIn("phase1_deterministic_fallback", body)
        self.assertNotIn("build_phase2_phase1_swipt_ps_fallback", body)

    def test_phase22_runtime_requires_dynamic_llm_generation(self) -> None:
        source = (SCRIPTS_DIR / "phase_runtime" / "phase21_23_foundation.py").read_text(encoding="utf-8")
        start = source.index("def run_phase2_phase2_llm")
        end = source.index("def run_phase2_phase3_llm")
        body = source[start:end]

        self.assertIn("llm.chat", body)
        self.assertNotIn("phase2_deterministic_fallback", body)
        self.assertNotIn("build_phase2_phase2_swipt_ps_fallback", body)

    def test_phase23_runtime_requires_dynamic_llm_generation(self) -> None:
        source = (SCRIPTS_DIR / "phase_runtime" / "phase21_23_foundation.py").read_text(encoding="utf-8")
        start = source.index("def run_phase2_phase3_llm")
        end = source.index("def run_phase2_phase3_latex_llm")
        body = source[start:end]

        self.assertIn("llm.chat", body)
        self.assertNotIn("phase3_deterministic_fallback", body)
        self.assertNotIn("build_phase2_phase3_swipt_ps_fallback", body)

    def test_phase24_runtime_disables_experiment_fallback_plugins(self) -> None:
        source = (SCRIPTS_DIR / "phase_runtime" / "phase24_plugin_generation.py").read_text(encoding="utf-8")
        start = source.index("def run_phase2_phase24_plugin_llm")
        end = source.index("def repair_phase2_phase24_plugin_llm")
        body = source[start:end]

        self.assertIn("llm.chat", body)
        self.assertNotIn("_phase24_swipt_ps_reference_plugin_fallback", source)
        self.assertNotIn("_phase24_uplink_power_control_reference_plugin_fallback", source)
        self.assertNotIn("_phase24_downlink_reference_plugin_fallback", source)
        self.assertNotIn("WCL_PHASE24_USE_REFERENCE_PLUGIN", body)
        self.assertIn("experiment fallbacks are disabled", body)

    def test_phase21_latex_runtime_uses_semantic_repair_mode(self) -> None:
        source = (SCRIPTS_DIR / "phase_runtime" / "phase21_23_foundation.py").read_text(encoding="utf-8")
        start = source.index("def run_phase2_phase1_latex_llm")
        end = source.index("def run_phase2_phase2_llm")
        body = source[start:end]

        self.assertIn("latex_math_contract_issue_summary", body)
        self.assertIn('repair_mode="semantic_consistency_repair"', body)
        self.assertIn("phase1_latex_semantic_consistency_report.txt", body)

    def test_phase21_rejects_auxiliary_variables_in_original_optimizer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "phase2-1").mkdir()
            handoff = {
                "variables": "beamforming vectors, power-splitting ratios, auxiliary received-power variables",
                "core_constraints": "power budget and received-power consistency",
            }
            system_model = (
                "We consider a downlink model with controllable beamforming vectors and splitting ratios. "
                "The received-power quantities are defined from the signal model and are not independent controls."
            )
            problem = r"""
            The original optimization problem is
            \[
            \max_{\{\mathbf{w}_k,\rho_k,p_k,q_k\}_{k\in\mathcal{K}}}\quad
            \sum_{k\in\mathcal{K}} U_k(q_k)
            \]
            where \(p_k\) and \(q_k\) are auxiliary received-power quantities.
            """
            theory = (
                "The reformulation can introduce auxiliary variables later, but the original optimization problem should keep "
                "the control set separate from derived quantities and algorithmic surrogates."
            )

            with self.assertRaisesRegex(ValueError, "auxiliary/derived quantities"):
                validate_phase2_phase1_contract(
                    run_dir=run_dir,
                    topic="Generic downlink utility maximization",
                    handoff=handoff,
                    system_model_md=system_model,
                    problem_formulation_md=problem,
                    core_theory_package_md=theory,
                )

            valid_problem = r"""
            The original optimization problem is
            \[
            \max_{\{\mathbf{w}_k,\rho_k\}_{k\in\mathcal{K}}}\quad
            \sum_{k\in\mathcal{K}} U_k(q_k(\mathbf{w},\rho_k))
            \]
            subject to \(q_k(\mathbf{w},\rho_k)\in[q_k^{\min},q_k^{\max}]\).
            The auxiliary received-power terms remain deterministic functions of the physical controls.
            """
            report = validate_phase2_phase1_contract(
                run_dir=run_dir,
                topic="Generic downlink utility maximization",
                handoff=handoff,
                system_model_md=system_model,
                problem_formulation_md=valid_problem,
                core_theory_package_md=theory,
            )
            self.assertTrue(report["ok"])

    def test_phase21_contract_gate_rejects_text_coordinate_convention_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "phase2-1").mkdir()
            contract = {
                "controls": [
                    {
                        "symbol": "\\mathbf q[n]",
                        "meaning": "UAV 3D waypoint at slot n",
                        "domain": "\\mathbb{R}^3",
                    }
                ],
                "parameters": [
                    {
                        "symbol": "\\mathbf w_k",
                        "meaning": "ground IoT device location",
                        "domain": "\\mathbb{R}^2",
                    },
                    {"symbol": "H[n]", "meaning": "UAV altitude"},
                ],
                "derived_quantities": [
                    {
                        "symbol": "h_k[n]",
                        "meaning": "channel gain",
                        "definition": "\\beta_0/(\\|\\mathbf q[n]-\\mathbf w_k\\|^2+H[n]^2)",
                        "depends_on": ["\\mathbf q[n]", "\\mathbf w_k", "H[n]"],
                        "used_in": ["rate"],
                    }
                ],
                "objective": {"sense": "max", "expression": "\\sum_k R_k"},
                "constraints": [{"id": "speed", "relation": "\\|\\mathbf q[n+1]-\\mathbf q[n]\\|\\le V_{max}\\Delta"}],
            }

            with self.assertRaisesRegex(ValueError, "Geometry convention mixes"):
                validate_phase2_phase1_contract(
                    run_dir=run_dir,
                    topic="UAV-enabled wireless powered IoT data collection",
                    mathematical_contract=contract,
                    system_model_md=(
                        "The UAV waypoint is \\mathbf q[n]\\in\\mathbb{R}^3 and each ground device has "
                        "\\mathbf w_k\\in\\mathbb{R}^2. The channel is h_k[n]=\\beta_0/(\\|\\mathbf q[n]-\\mathbf w_k\\|^2+H[n]^2). "
                        "The text is long enough to avoid length-only validation failures while preserving the coordinate conflict."
                    ),
                    problem_formulation_md=(
                        "The original problem maximizes sum throughput over the trajectory subject to speed and scheduling constraints. "
                        "Only the physical trajectory and resource allocations are optimized in the original problem."
                    ),
                    core_theory_package_md=(
                        "The theory route should use a dimensionally consistent path-loss expression before applying any trajectory surrogate. "
                        "This fixture verifies that the formulation gate catches the conflict before writing."
                    ),
                )

    def test_mechanism_detector_does_not_match_short_acronyms_inside_words(self) -> None:
        mechanisms = _phase2_count_hard_mechanisms(
            "These nonconvexities arise from interference, rectifier coupling, and harvested energy."
        )

        self.assertIn("swipt_eh", mechanisms)
        self.assertNotIn("ris", mechanisms)

    def test_phase21_rejects_coherent_eh_power_for_independent_streams(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "phase2-1").mkdir()
            system_model = r"""
            A SWIPT downlink transmits independent data symbols s_k and an independent energy symbol s_E.
            The received RF power is P_m^{rf}=|h_m^H \Theta H (\sum_k w_k + w_E)|^2.
            """
            with self.assertRaisesRegex(ValueError, "coherent square"):
                validate_phase2_phase1_contract(
                    run_dir=run_dir,
                    topic="RIS-aided SWIPT with nonlinear energy harvesting",
                    system_model_md=system_model,
                    problem_formulation_md="maximize harvested energy for EH users",
                    core_theory_package_md="nonlinear sigmoid rectifier",
                )

    def test_phase22_rejects_unlifted_nonlinear_eh_qcqp_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "phase2-2").mkdir()
            with self.assertRaisesRegex(ValueError, "convex QCQP"):
                validate_phase2_phase2_contract(
                    run_dir=run_dir,
                    convexity_audit_md="The SWIPT nonlinear EH sigmoid objective is handled by a concave quadratic surrogate.",
                    reformulation_path_md="The resulting beamforming block is a standard convex QCQP.",
                )

    def test_phase22_allows_explicit_nonconcavity_statement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "phase2-2").mkdir()
            report = validate_phase2_phase2_contract(
                run_dir=run_dir,
                convexity_audit_md="The multiuser sum-rate utility is not jointly concave under interference.",
                reformulation_path_md="Use WMMSE with fixed receive filters and MSE weights; the beamforming block is then a convex QCQP under per-antenna constraints.",
            )
            self.assertTrue(report["ok"])

    def test_phase22_rejects_single_gamma_fp_convex_beamformer_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "phase2-2").mkdir()
            with self.assertRaisesRegex(ValueError, "single SINR/gamma auxiliary"):
                validate_phase2_phase2_contract(
                    run_dir=run_dir,
                    convexity_audit_md="For fixed gamma, the w-subproblem is a strictly convex quadratic for multiuser interference.",
                    reformulation_path_md="The beamforming block is a convex QCQP after the gamma update.",
                )

    def test_phase22_records_black_box_safe_counterpart_placeholder_as_advisory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "phase2-2").mkdir()
            report = validate_phase2_phase2_contract(
                run_dir=run_dir,
                convexity_audit_md=(
                    "The distributionally robust chance constraints are handled by the selected ambiguity model."
                ),
                reformulation_path_md=(
                    "For each margin, the deterministic safe conic counterpart is denoted by \\Phi_m\\ge 0."
                ),
            )
            self.assertTrue(report["ok"])
            self.assertTrue(any("Phi-style safe-counterpart placeholder" in item for item in report["warnings"]))

    def test_phase21_records_uncertainty_model_advisory_without_keyword_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "phase2-1").mkdir()
            report = validate_phase2_phase1_contract(
                run_dir=run_dir,
                topic="distributionally robust downlink beamforming",
                system_model_md=(
                    "We consider imperfect CSI and an ambiguity set for each channel in a multiuser wireless link. "
                    "The transmitter selects physical beam controls while each receiver observes a noisy downlink signal. "
                    "The paragraph is intentionally long enough to isolate the uncertainty-model advisory from length checks."
                ),
                problem_formulation_md=(
                    "The original problem minimizes transmit power subject to chance constraints and SINR requirements for all users. "
                    "All optimizer variables are physical controls, while outage and service quantities are derived metrics. "
                    "The uncertainty family is intentionally under-specified in this fixture."
                ),
                core_theory_package_md=(
                    "A safe conic counterpart will be selected later by the theory agent after reading the frozen artifacts. "
                    "This fixture checks that keyword-level robust language records an advisory rather than blocking the formulation. "
                    "The actual prompt must still ask the LLM to close the active mechanism semantically."
                ),
            )
            self.assertTrue(report["ok"])
            self.assertTrue(any("advisory signal" in item for item in report["warnings"]))

    def test_phase23_does_not_reject_keyword_claims_without_contract_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "phase2-3").mkdir()
            report = validate_phase2_phase3_contract(
                run_dir=run_dir,
                algorithm_md="The algorithm uses SDR followed by Gaussian randomization for RIS recovery.",
                convergence_or_complexity_md="The method guarantees convergence to a KKT stationary point.",
                experiment_blueprint_md="Report harvested energy, feasibility, and constraint violation.",
            )
            self.assertTrue(report["ok"])
            self.assertEqual(report["errors"], [])

    def test_phase23_allows_negated_sdr_and_scoped_stationarity_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "phase2-3").mkdir()
            report = validate_phase2_phase3_contract(
                run_dir=run_dir,
                algorithm_md=(
                    "Use a conservative projected SCA update with exact physical acceptance checks. "
                    "No SDR, rank relaxation, Gaussian randomization, or rank-one recovery is used."
                ),
                convergence_or_complexity_md=(
                    "The accepted objective values are nondecreasing by construction. "
                    "This is not a claim of global optimality or KKT stationarity for the original problem; "
                    "a formal projected-stationarity theorem is not claimed without regularity condition proof."
                ),
                experiment_blueprint_md="Report energy efficiency, throughput, feasibility, and constraint violation.",
            )
            self.assertTrue(report["ok"])
            self.assertEqual(report["errors"], [])

    def test_phase23_rejects_algorithm_that_ignores_frozen_contracts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "phase2-1").mkdir()
            (run_dir / "phase2-2").mkdir()
            (run_dir / "phase2-3").mkdir()
            (run_dir / "phase2-1" / "mathematical_contract.json").write_text(
                json.dumps(
                    {
                        "controls": [
                            {"symbol": "q_n", "meaning": "UAV trajectory waypoints"},
                            {"symbol": "p_n", "meaning": "transmit power"},
                        ],
                        "objective": {"sense": "max", "meaning": "energy efficiency in bits per Joule"},
                        "constraints": [{"id": "C1", "meaning": "speed feasibility constraint"}],
                    }
                )
            )
            (run_dir / "phase2-2" / "algorithm_contract.json").write_text(
                json.dumps(
                    {
                        "algorithm_execution_contract": {
                            "update_blocks": [
                                "trajectory_update_under_speed_constraints",
                                "transmit_power_update_under_Pmax",
                            ],
                            "objective_evaluator": "Evaluate energy efficiency as delivered bits divided by total energy.",
                            "constraint_evaluator": "Evaluate speed and power constraint violations.",
                        }
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "frozen controls"):
                validate_phase2_phase3_contract(
                    run_dir=run_dir,
                    algorithm_md="Use a generic numerical optimizer.",
                    convergence_or_complexity_md="The method is empirical and reports runtime.",
                    experiment_blueprint_md="Report feasibility.",
                )

    def test_phase3_2_sanitizer_does_not_double_escape_latex_aliases(self) -> None:
        tex = (
            r"Figure uses $\\lambda_s$ and raw $E_min_mW$ plus text lambda_s."
            "\n\\begin{tablenotes}\\item Table note.\\end{tablenotes}"
        )
        cleaned = sanitize_phase3_2_numerical_results_tex(tex)
        self.assertIn(r"$\lambda_s$", cleaned)
        self.assertIn(r"$E_{\min}$", cleaned)
        self.assertIn(r"$\lambda_s$", cleaned)
        self.assertNotIn(r"$\\lambda_s$", cleaned)
        self.assertNotIn(r"\begin{tablenotes}", cleaned)
        self.assertIn(r"\footnotesize Table note.", cleaned)

    def test_phase3_2_sanitizer_removes_workflow_sweep_language(self) -> None:
        tex = (
            r"Unless otherwise swept, $K=3$. "
            r"Fig.~\ref{fig:a} depicts $\psi$ over the plotted sweep, "
            r"and the caption omits the swept value grid for a non-swept parameter."
        )
        cleaned = sanitize_phase3_2_numerical_results_tex(tex)
        self.assertIn("Unless otherwise stated", cleaned)
        self.assertIn("plotted operating range", cleaned)
        self.assertIn("x-axis value grid", cleaned)
        self.assertIn("fixed parameter", cleaned)
        self.assertNotRegex(cleaned, r"\bsweep|swept\b")

    def test_phase3_2_contamination_allows_generic_user_load_axis(self) -> None:
        tex = (
            r"Fig.~\ref{fig:phase25_figure_2} plots $R_{\mathrm{wsr}}$ "
            r"versus number of users $K$."
        )
        check = analyze_phase3_2_cross_topic_contamination(
            tex,
            "Movable-antenna assisted downlink beamforming and antenna-position optimization for multiuser wireless networks",
        )
        self.assertTrue(check["passed"], check)
        self.assertEqual([], check["hits"])

    def test_phase3_3_internal_term_gate_ignores_figure_file_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            phase_dir = Path(tmp)
            _phase3_3_assert_no_internal_paper_terms(
                phase_dir,
                "numerical_results_section",
                r"\includegraphics[width=\linewidth]{../phase2-5/figures/figure_1.pdf}"
                "\n"
                r"The proposed method improves the weighted sum rate.",
            )
            report = json.loads((phase_dir / "numerical_results_section_internal_terms_report.json").read_text())

        self.assertTrue(report["ok"], report)
        self.assertEqual([], report["hits"])

    def test_phase3_3_blocks_empty_technical_sections_before_assembly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            phase_dir = Path(tmp)
            with self.assertRaisesRegex(ValueError, "Run Phase 3.1 technical drafting"):
                _phase3_3_assert_required_section_not_empty(
                    phase_dir,
                    "system_model_problem_formulation_section",
                    "",
                    min_words=5,
                )
            report = json.loads((phase_dir / "system_model_problem_formulation_section_nonempty_report.json").read_text())

        self.assertFalse(report["ok"])
        self.assertEqual(0, report["word_count"])

    def test_phase23_and_phase3_2_force_top_float_placement(self) -> None:
        phase3 = sanitize_phase3_latex_snippet(
            r"\begin{align}\gamma_k&=1,\label{eq:sinr}\\R_k&=1.\label{eq:rate}\end{align}"
            "\n"
            r"See \eqref{eq:sinr}--\eqref{eq:rate}."
            "\n"
            r"The method preserves the exact Phase~2.1 metrics, and Phase 2.4 and Phase 2.5 experiments verify feasibility."
            "\n"
            r"We preserve the exact system-model metrics and develop a tractable search."
            "\n"
            r"Every KPI is computed from the original Phase 2.1 physical equations."
            "\n"
            r"\begin{algorithm}\caption{Demo}\label{alg:proposed}\begin{algorithmic}[1]\State x\end{algorithmic}\end{algorithm}"
        )
        self.assertIn(r"\begin{algorithm}[!t]", phase3)
        self.assertIn(r"\label{eq:solution_sinr}", phase3)
        self.assertIn(r"\eqref{eq:solution_sinr}", phase3)
        self.assertIn(r"\label{alg:solution_proposed}", phase3)
        self.assertNotIn(r"\label{eq:sinr}", phase3)
        self.assertIn("The method retains the original physical model", phase3)
        self.assertIn("retain the original physical model and develop", phase3)
        self.assertIn("numerical experiments verify feasibility", phase3)
        self.assertNotIn("Phase 2.4", phase3)
        self.assertNotIn("Phase 2.5", phase3)
        self.assertNotIn("preserve the exact system-model metrics", phase3)
        self.assertNotIn("preserves the exact system-model metrics", phase3)
        self.assertNotIn("system-model metrics", phase3)
        self.assertNotIn("physical metrics", phase3)
        self.assertNotIn("original original physical equations", phase3)
        self.assertIn("from the original physical equations", phase3)
        self.assertNotIn("Phase~2.1", phase3)

        tex = "\n".join(
            [
                r"\section{Numerical Results}",
                r"\begin{figure}[h]\caption{A}\end{figure}",
                r"\begin{table}[ht]\caption{B}\end{table}",
            ]
        )
        cleaned = sanitize_phase3_2_numerical_results_tex(tex)
        self.assertIn(r"\begin{figure}[t]", cleaned)
        self.assertNotIn(r"\begin{table}", cleaned)
        self.assertNotIn(r"\begin{table}[ht]", cleaned)

    def test_latex_equation_format_blocks_dense_multi_relation_lines(self) -> None:
        bad = (
            r"\begin{equation}\label{eq:bad}"
            r"x=a, \qquad y=b"
            r"\end{equation}"
            "\n"
            r"\begin{equation}\label{eq:chain}"
            r"z=f(u)=g(v)"
            r"\end{equation}"
            "\n"
            r"\begin{equation}\label{eq:bounds}"
            r"0\leq \rho_k\leq 1"
            r"\end{equation}"
            "\n"
            r"\begin{subequations}\label{eq:channel_uncertainty}"
            r"\begin{align}"
            r"\mathbf h_k&=\widehat{\mathbf h}_k+\mathbf e_k\label{eq:channel_model}\\"
            r"\|\mathbf e_k\|_2&\leq\epsilon_k\label{eq:channel_error_radius}"
            r"\end{align}"
            r"\end{subequations}"
            "\n"
            r"Problem \eqref{eq:p0_obj} is nonconvex."
            "\n"
            r"\begin{align}"
            r"\text{(P0)}\quad\max_x\quad & f(x)\label{eq:p0_obj}\\"
            r"\text{s.t.}\quad & x\leq 1\label{eq:p0_power}"
            r"\end{align}"
            "\n"
            r"\begin{subequations}\label{prob:p0}"
            "\n"
            r"\begin{align}"
            "\n"
            r"\text{(P0)}\quad\max_y\quad & f(y)\label{con:p0_obj}\\"
            "\n"
            r"\text{s.t.}\quad & y\leq 1"
            "\n"
            r"\end{align}"
            "\n"
            r"\end{subequations}"
            "\n"
            r"\begin{align*}"
            r"\text{(P1)}\quad"
            r"\underset{x}{\mathrm{maximize}}\quad & f(x)\\"
            r"\mathrm{subject\ to}\quad & x\leq 1"
            r"\end{align*}"
        )
        report = analyze_latex_equation_line_format(bad)
        self.assertFalse(report["ok"])
        issue_names = {item["issue"] for item in report["issues"]}
        self.assertIn("multiple_primary_relations_on_one_line", issue_names)
        self.assertIn("optimization_problem_labeled_as_equation", issue_names)
        self.assertIn("optimization_problem_referenced_as_equation", issue_names)
        self.assertIn("optimization_problem_constraints_not_numbered", issue_names)
        self.assertIn("optimization_problem_lines_missing_constraint_labels", issue_names)
        self.assertIn("non_optimization_subequations", issue_names)
        self.assertIn("problem_referenced_with_eqref", issue_names)
        self.assertIn("optimization_uses_full_word_maximize_or_minimize", issue_names)
        self.assertIn("optimization_uses_full_subject_to", issue_names)

        good = (
            r"\begin{equation}\label{eq:x}x=a\end{equation}"
            "\n"
            r"\begin{equation}\label{eq:y}y=b\end{equation}"
            "\n"
            r"\begin{equation}\label{eq:rate}"
            r"R_k=B\log_2\left(1+\Gamma_k\right)"
            r"\end{equation}"
            "\n"
            r"\begin{subequations}\label{prob:p0}"
            "\n"
            r"\begin{align}"
            "\n"
            r"\text{(P0)}\quad\max_x\quad & f(x)\label{con:p0_obj}\\"
            "\n"
            r"\text{s.t.}\quad & x\leq 1\label{con:p0_power}"
            "\n"
            r"\end{align}"
            "\n"
            r"\end{subequations}"
            "\n"
            r"problem (P0) is nonconvex."
            "\n"
            r"The power-budget constraint in \eqref{con:p0_power} limits the feasible set."
        )
        self.assertTrue(analyze_latex_equation_line_format(good)["ok"])

    def test_latex_equation_format_allows_numbered_auxiliary_optimization_problem(self) -> None:
        tex = (
            r"\begin{subequations}\label{prob:p2}"
            "\n"
            r"\begin{align}"
            "\n"
            r"\text{(P2)}\quad\max_{\mathbf{\Phi}\succeq\mathbf{0},\,t}\quad & t-\rho\mathrm{tr}(\mathbf{\Phi})\label{obj:p2}\\"
            "\n"
            r"\mathrm{s.t.}\quad & \mathrm{diag}(\mathbf{\Phi})=\mathbf{1}\label{con:p2_unit_diag}"
            "\n"
            r"\end{align}"
            "\n"
            r"\end{subequations}"
        )

        self.assertTrue(analyze_latex_equation_line_format(tex)["ok"], analyze_latex_equation_line_format(tex))

    def test_latex_equation_format_allows_set_builder_uncertainty_sets(self) -> None:
        tex = (
            r"\begin{align}"
            r"\mathcal H_k&=\{\widehat{\mathbf h}_k+\mathbf e_{h,k}:"
            r"\|\mathbf e_{h,k}\|_2\leq\delta_{h,k}\},\quad k\in\mathcal K,"
            r"\label{eq:info_uncertainty_set}\\"
            r"\mathcal G_\ell&=\{\widehat{\mathbf g}_\ell+\mathbf e_{g,\ell}:"
            r"\|\mathbf e_{g,\ell}\|_2\leq\delta_{g,\ell}\},\quad \ell\in\mathcal L,"
            r"\label{eq:energy_uncertainty_set}"
            r"\end{align}"
        )
        self.assertTrue(analyze_latex_equation_line_format(tex)["ok"])

    def test_phase3_1_sanitizer_unwraps_nonproblem_subequations(self) -> None:
        tex = (
            r"\subsection{System Model}"
            "\n"
            r"\begin{subequations}\label{eq:channel_uncertainty}"
            "\n"
            r"\begin{align}"
            "\n"
            r"\mathbf h_k&=\widehat{\mathbf h}_k+\mathbf e_k,\label{eq:channel_model}\\"
            "\n"
            r"\|\mathbf e_k\|_2&\leq\epsilon_k.\label{eq:channel_error_radius}"
            "\n"
            r"\end{align}"
            "\n"
            r"\end{subequations}"
            "\n"
            r"\begin{subequations}\label{prob:p0}"
            "\n"
            r"\begin{align}"
            "\n"
            r"\text{(P0)}\quad\max_x\quad & f(x)\label{obj:p0}\\"
            "\n"
            r"\text{s.t.}\quad & x\leq 1\label{con:p0_power}"
            "\n"
            r"\end{align}"
            "\n"
            r"\end{subequations}"
        )

        cleaned = sanitize_phase3_1_system_problem_snippet(tex)

        self.assertNotIn(r"\begin{subequations}\label{eq:channel_uncertainty}", cleaned)
        self.assertIn(r"\begin{align}", cleaned)
        self.assertIn(r"\label{eq:channel_model}", cleaned)
        self.assertIn(r"\begin{subequations}\label{prob:p0}", cleaned)
        self.assertTrue(analyze_latex_equation_line_format(cleaned)["ok"], cleaned)

    def test_phase3_1_sanitizer_moves_pair_index_range_out_of_constraint_line(self) -> None:
        tex = (
            r"\begin{subequations}\label{prob:p0}"
            "\n"
            r"\begin{align}"
            "\n"
            r"\text{(P0)}\quad\max_x\quad & f(x)\label{obj:p0}\\"
            "\n"
            r"\text{s.t.}\quad & \|p_n-p_m\| \ge d_{\min},\ \ 1\le n<m\le N. \label{con:p0_spacing}"
            "\n"
            r"\end{align}"
            "\n"
            r"\end{subequations}"
        )

        cleaned = sanitize_phase3_1_system_problem_snippet(tex)

        self.assertIn(r"\mathcal{E}\triangleq", cleaned)
        self.assertIn(r"\forall (n,m)\in\mathcal{E}", cleaned)
        self.assertNotIn(r"1\le n<m\le N. \label{con:p0_spacing}", cleaned)
        self.assertTrue(analyze_latex_equation_line_format(cleaned)["ok"], cleaned)

    def test_latex_compile_issue_summary_flags_overfull_boxes(self) -> None:
        tex = "\n".join(
            [
                r"\subsection{Proposed Solution}",
                r"Long display follows.",
                r"\begin{equation}",
                r"e_k=|u_k|^2\left(\rho_k\sum_{j\in\mathcal K}|\mathbf h_k^H\mathbf w_j|^2+\rho_k\sigma_{a,k}^2+\sigma_{d,k}^2\right)-2\operatorname{Re}\{u_k^*\sqrt{\rho_k}\mathbf h_k^H\mathbf w_k\}+1.",
                r"\end{equation}",
            ]
        )
        log = "Overfull \\hbox (53.06142pt too wide) detected at line 4"

        report = analyze_latex_overfull_boxes(log, tex)
        self.assertFalse(report["ok"])
        self.assertEqual(report["issues"][0]["line"], 4)
        self.assertGreater(report["issues"][0]["amount_pt"], 50)
        summary = _extract_latex_issue_summary(log, tex)
        self.assertIn("PDF overfull hbox warnings", summary)
        self.assertIn("line 4", summary)
        self.assertIn("Source context", summary)

    def test_latex_compile_issue_summary_ignores_tiny_overfull_boxes(self) -> None:
        tex = "\n".join([r"\begin{align}", r"x&=y", r"\end{align}"])
        log = "Overfull \\hbox (2.23259pt too wide) detected at line 2"

        report = analyze_latex_overfull_boxes(log, tex)
        self.assertTrue(report["ok"])
        self.assertEqual(report["issues"], [])

    def test_phase3_4_preview_sanitizer_strips_extra_equation_labels(self) -> None:
        tex = (
            r"\subsection{System Model}\label{sec:system_model}"
            "\n"
            r"\begin{equation}\label{eq:problem}"
            r"\begin{aligned}a&=b,\label{eq:a}\\ c&=d.\label{eq:c}\end{aligned}"
            r"\end{equation}"
        )
        cleaned = sanitize_phase3_4_preview_section_tex(tex)
        self.assertIn(r"\subsection{System Model}", cleaned)
        self.assertNotIn(r"\label{sec:system_model}", cleaned)
        self.assertIn(r"\label{eq:problem}", cleaned)
        self.assertNotIn(r"\label{eq:a}", cleaned)
        self.assertNotIn(r"\label{eq:c}", cleaned)

    def test_phase3_4_bibtex_author_normalizer_converts_comma_separated_metadata(self) -> None:
        authors = "F. Liu, Y. Cui, C. Masouros, J. Xu, T. X. Han, Y. C. Eldar, and S. Buzzi"

        normalized = normalize_bibtex_author_list(authors)

        self.assertEqual(
            normalized,
            "F. Liu and Y. Cui and C. Masouros and J. Xu and T. X. Han and Y. C. Eldar and S. Buzzi",
        )

    def test_phase3_4_intro_expander_does_not_insert_deterministic_content(self) -> None:
        intro = "\\section{Introduction}\n" + ("word " * 80) + "The remainder of this letter is organized as follows."
        expanded = ensure_phase3_4_minimum_intro_words(intro, minimum_words=120)
        self.assertEqual(expanded, intro)

    def test_phase3_4_intro_content_quality_blocks_formula_and_result_preview(self) -> None:
        bad = (
            r"\section{Introduction}"
            "\n\nThis letter compares against a fixed power-splitting baseline that sets "
            r"$\rho_k=0.5$ and achieves gains of 95\% in evaluated scenarios."
        )
        report = analyze_phase3_4_introduction_content_quality(bad)
        self.assertFalse(report["ok"])
        self.assertGreater(report["inline_math_count"], 0)
        self.assertTrue(any("numerical-result values" in item for item in report["errors"]))

        good = (
            r"\section{Introduction}"
            "\n\nSimultaneous wireless information and power transfer supports low-power receivers by "
            r"coupling data delivery and energy supply. Prior work has clarified receiver architectures "
            r"and nonlinear harvesting effects, but adaptive splitting and rectifier-aware beamforming "
            r"remain tightly coupled in practical multiuser operation. This letter studies that coupling "
            r"through a rectifier-aware alternating design and positions the evaluation against representative "
            r"fixed-splitting and linear-EH baselines."
            "\n\n"
            r"\begin{itemize}"
            r"\item We formulate a rectifier-aware design for joint beamforming and power splitting."
            r"\item We compare against representative fixed-splitting and linear-EH baselines."
            r"\end{itemize}"
            "\n\nThe remainder of this letter is organized as follows. "
            r"Section~\ref{sec:system_model} presents the system model and problem formulation. "
            r"Section~\ref{sec:proposed_solution} describes the proposed solution. "
            r"Section~\ref{sec:numerical_results} reports the numerical results, and "
            r"Section~\ref{sec:conclusion} concludes the letter."
            "\n\n"
            r"\textit{Notation:} Bold lowercase and uppercase letters denote vectors and matrices, "
            r"respectively. $(\cdot)^H$ denotes Hermitian transpose, and $\|\cdot\|_2$ denotes the Euclidean norm."
        )
        self.assertTrue(analyze_phase3_4_introduction_content_quality(good)["ok"])

    def test_phase3_4_intro_content_quality_blocks_deictic_bottleneck_sentence(self) -> None:
        bad = (
            r"\section{Introduction}"
            "\n\nNear-field communications expose range-angle interference that cannot be captured by "
            "angle-only far-field beams. This letter studies this bottleneck for a narrowband "
            "single-cell downlink where a large uniform linear array serves single-antenna users."
            "\n\n"
            r"\begin{itemize}"
            r"\item We formulate a near-field beamfocusing problem."
            r"\item We develop a WMMSE-based beamfocusing method."
            r"\end{itemize}"
            "\n\nThe remainder of this letter is organized as follows."
            "\n\n"
            r"\textit{Notation:} Bold lowercase and uppercase letters denote vectors and matrices."
        )
        report = analyze_phase3_4_introduction_content_quality(bad)
        self.assertFalse(report["ok"])
        self.assertEqual(report["weak_deictic_study_phrase_count"], 1)
        self.assertTrue(any("weak deictic study sentence" in item for item in report["errors"]))

    def test_phase3_4_intro_content_quality_blocks_first_paragraph_system_setup(self) -> None:
        bad = (
            r"\section{Introduction}"
            "\n\nNear-field communications are promising for large-aperture wireless systems because they can "
            "focus energy over both angle and range. This letter studies this bottleneck for a narrowband "
            "single-cell downlink where a large uniform linear array serves single-antenna users in the "
            "radiative near field with perfect CSI."
            "\n\nPrior work has clarified beam focusing, but practical optimization remains difficult."
            "\n\nThe resulting gap is an algorithm that balances focusing and leakage."
            "\n\nIn this letter, we address this gap. The main contributions are summarized as follows."
            "\n\n"
            r"\begin{itemize}"
            r"\item We formulate a near-field beamfocusing problem."
            r"\item We develop a WMMSE-based beamfocusing method."
            r"\end{itemize}"
            "\n\nThe remainder of this letter is organized as follows."
            "\n\n"
            r"\textit{Notation:} Bold lowercase and uppercase letters denote vectors and matrices."
        )
        report = analyze_phase3_4_introduction_content_quality(bad)
        self.assertFalse(report["ok"])
        self.assertGreater(report["first_paragraph_system_setup_phrase_count"], 0)
        self.assertTrue(any("first paragraph reads like a system-model setup" in item for item in report["errors"]))

    def test_phase3_4_forbidden_terms_do_not_flag_improves(self) -> None:
        report = analyze_phase3_4_forbidden_terms("The evaluated method improves the utility under the considered settings.")
        self.assertTrue(report["ok"])
        self.assertEqual(report["hits"], [])

    def test_phase3_2_gate_blocks_draft_experiment_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            phase25 = run_dir / "phase2-5"
            phase25.mkdir(parents=True)
            (phase25 / "phase25_experiment_summary.json").write_text(
                json.dumps(
                    {
                        "phase25_status": "claim_failure_needs_redesign",
                        "generated_figures_are_draft_only": True,
                        "paper_minimum_ready": False,
                        "figures": [{"figure_id": "figure_1", "paper_ready": False, "draft_or_final": "draft"}],
                    }
                ),
                encoding="utf-8",
            )
            (phase25 / "phase25_verified_registry.json").write_text(
                json.dumps({"status": "verified_experiment_registry", "phase25_status": "claim_failure_needs_redesign"}),
                encoding="utf-8",
            )
            (phase25 / "plot_quality_report.json").write_text(
                json.dumps({"overall_status": "claim_failure_needs_redesign"}),
                encoding="utf-8",
            )

            gate = validate_phase3_2_paper_evidence_gate(run_dir)

            self.assertFalse(gate["ok"])
            self.assertIn("not paper-ready", "\n".join(gate["errors"]))
            self.assertIn("draft", "\n".join(gate["errors"]))

    def test_phase3_2_setup_renders_ofdm_without_eh_ris_leftovers(self) -> None:
        paragraph = render_phase3_2_setup_paragraph(
            {
                "feature_flags": {"ofdm": True, "ris": False, "energy_harvesting": False, "sensing": False},
                "canonical_config": {
                    "system": {"M": 8, "N": 64, "K": 4, "Pmax": 30, "bandwidth_MHz": 20},
                    "constraints": {"R_min": 1.0, "E_min": 5.0},
                    "optimization": {"alpha_comm": 1.0, "alpha_EH": 0.2},
                },
                "methods": {"short": {"proposed": "Proposed", "baseline": "Water-filling"}},
                "figures": [
                    {
                        "x_axis_param": "system.Pmax",
                        "x_axis_latex": "P_{\\max}",
                        "x_values": [20, 25, 30],
                        "seeds_per_point_summary": {"min": 10, "max": 10},
                        "has_error_bars": True,
                    }
                ],
            }
        )

        self.assertIn("BS antennas", paragraph)
        self.assertIn("OFDM subcarriers", paragraph)
        self.assertNotIn("EH users", paragraph)
        self.assertNotIn("RIS elements", paragraph)
        self.assertNotIn("harvester", paragraph.lower())

    def test_phase3_2_setup_keeps_distinct_figure_sweeps_separate(self) -> None:
        paragraph = render_phase3_2_setup_paragraph(
            {
                "canonical_config": {"system": {"K": 4}},
                "methods": {"short": {"proposed": "Proposed", "equal_power_heuristic": "EQ-Power"}},
                "figures": [
                    {
                        "figure_intent": "main_comparison",
                        "x_axis_param": "constraints.gamma_target",
                        "x_axis_latex": "\\gamma",
                        "x_values": [0, 2, 4],
                        "seeds_per_point_summary": {"min": 50, "max": 50},
                    },
                    {
                        "figure_intent": "feasibility_boundary",
                        "x_axis_param": "constraints.gamma_target",
                        "x_axis_latex": "\\gamma",
                        "x_values": [9, 10, 11],
                        "seeds_per_point_summary": {"min": 50, "max": 50},
                    },
                ],
            }
        )

        self.assertIn("50 independent channel realizations", paragraph)
        self.assertNotIn("main performance sweep uses $\\gamma\\in\\{0,2,4\\}$", paragraph)
        self.assertNotIn("feasibility-stress sweep uses $\\gamma\\in\\{9,10,11\\}$", paragraph)
        self.assertNotIn("weighted objective", paragraph)
        self.assertNotIn("$\\gamma\\in\\{0,2,4,9,10,11\\}$", paragraph)

    def test_phase3_2_sweep_enforcer_does_not_merge_distinct_figure_sweeps(self) -> None:
        tex = (
            "The reported sweeps are separated by figure: "
            "main comparison uses $\\gamma\\in\\{0,2,4\\}$, "
            "feasibility boundary uses $\\gamma\\in\\{9,10,11\\}$."
        )
        updated = enforce_phase3_2_axis_value_consistency(
            tex,
            {
                "figures": [
                    {"x_axis_latex": "\\gamma", "x_values": [0, 2, 4]},
                    {"x_axis_latex": "\\gamma", "x_values": [9, 10, 11]},
                ]
            },
        )

        self.assertEqual(updated, tex)

    def test_phase3_5_fallback_review_does_not_create_p0_when_local_gates_pass(self) -> None:
        dimension_scores = {
            name: {"score": 7.1, "brief_reason": "Local deterministic gates passed."}
            for name in [
                "novelty_and_positioning",
                "technical_correctness",
                "theoretical_rigor",
                "algorithm_clarity",
                "experiment_strength",
                "reference_quality",
                "introduction_quality",
                "writing_quality",
                "IEEE_style_and_format",
                "reproducibility_and_consistency",
            ]
        }
        dimension_scores["experiment_strength"]["score"] = 7.8
        plan = _build_phase3_4_revision_plan(
            dimension_scores,
            arxiv_only_entries=[],
            compile_warnings_summary=[],
            forbidden_terms_found=[],
        )

        self.assertEqual(plan["P0"], [])
        self.assertEqual(plan["P1"], [])

    def test_phase3_2_dynamic_table_keeps_optimality_gap_column(self) -> None:
        table = build_phase3_2_dynamic_latex_table(
            "\n".join(
                [
                    "Scenario,Optimality gap to CENT-LP [%],Proposed feasibility,CENT-LP feasibility",
                    "SINR target $\\gamma$ (dB),3.9e-15%,1.0,1.0",
                ]
            )
        )

        self.assertIn(r"Gap[\%]", table)
        self.assertIn("3.90e-15", table)
        self.assertIn(r"\begin{table}[!t]", table)

    def test_phase24_prompts_require_reusable_method_contract(self) -> None:
        validation_prompt = build_phase2_phase24_validation_prompt(
            topic="generic wireless optimization",
            handoff={},
            system_model_md="system",
            problem_formulation_md="problem",
            convexity_audit_md="audit",
            reformulation_path_md="reformulation",
            experiment_blueprint_md="blueprint",
        )
        plugin_prompt = build_phase2_phase24_plugin_prompt(
            topic="generic wireless optimization",
            system_model_md="system",
            problem_formulation_md="problem",
            reformulation_path_md="reformulation",
            algorithm_md="algorithm",
            benchmark_definition_md="benchmark",
            experiment_blueprint_md="blueprint",
            validation_plan_summary="summary",
            problem_data_contract_summary="contract",
        )
        self.assertIn("compared_methods", validation_prompt)
        self.assertIn("implementation_hint", validation_prompt)
        self.assertIn("methods_to_run", validation_prompt)
        self.assertIn("WirelessBenchmarkAgent", validation_prompt)
        self.assertIn("ExperimentDesignAgent", validation_prompt)
        self.assertIn("First-pass experiment-design principle", validation_prompt)
        self.assertIn("Do not choose transmit power as the primary plotted KPI", validation_prompt)
        self.assertIn("If the frozen contract is a performance/utility maximization", validation_prompt)
        self.assertIn("method_solution", plugin_prompt)
        self.assertIn("same true objective", plugin_prompt)
        self.assertIn("Do not call build_model from channel_from_state", plugin_prompt)
        self.assertIn("Phase 2.4 evidence contract", plugin_prompt)
        self.assertIn("Phase 2.4 benchmark/evidence contract", plugin_prompt)
        self.assertIn("Read exact sweep paths", plugin_prompt)
        self.assertIn("must not map all unknown methods to baseline_solution", plugin_prompt)
        self.assertIn("exact execution id", plugin_prompt)
        self.assertIn("real solver path using `cvxpy`", plugin_prompt)
        self.assertIn("The Phase 2.4 execution check rejects candidate-search/proxy code", plugin_prompt)
        self.assertIn("First-pass implementation quality matters", plugin_prompt)
        self.assertIn("used_position_update", plugin_prompt)
        self.assertIn("position_step_norm", plugin_prompt)
        self.assertIn("objective_delta", plugin_prompt)
        self.assertNotIn("mrt_split_baseline", plugin_prompt)

    def test_prompt_renderer_allows_inserted_code_with_prompt_like_tokens(self) -> None:
        prompt = render_prompt_template(
            "phase2_4/plugin_repair.prompt.yaml",
            topic_specific_repair_rules="",
            status="validation_failed",
            validation_error_text="method dispatch failed",
            phase24_execution_contract_md="contract",
            validation_plan_summary="summary",
            current_plugin_code='last_message = f"{solver}:{status}"',
        )

        self.assertIn("Current validation status:", prompt)
        self.assertIn("validation_failed", prompt)
        self.assertIn('last_message = f"{solver}:{status}"', prompt)
        self.assertIn("used_position_update", prompt)
        self.assertIn("position_step_norm", prompt)

    def test_dynamic_topic_guardrail_does_not_inject_stale_mechanisms(self) -> None:
        generic_guardrail = build_wireless_feasibility_guardrail(
            "Generic wireless resource allocation",
            "The paper optimizes a latency-aware scheduling utility with queue stability constraints.",
        )

        self.assertIn("Current detected technical features", generic_guardrail)
        self.assertNotIn("SWIPT", generic_guardrail)
        self.assertNotIn("WMMSE", generic_guardrail)
        self.assertNotIn("RIS", generic_guardrail)
        self.assertNotIn("CRB", generic_guardrail)

        swipt_guardrail = build_wireless_feasibility_guardrail(
            "SWIPT downlink with nonlinear energy harvesting",
            "The system model includes power splitting receivers and a nonlinear rectifier.",
        )
        self.assertIn("SWIPT/EH", swipt_guardrail)
        self.assertIn("RF input power", swipt_guardrail)

        downlink_rate_guardrail = build_wireless_feasibility_guardrail(
            "Multiuser downlink weighted sum-rate maximization",
            "The formulation optimizes beamforming under multiuser SINR coupling.",
        )
        self.assertIn("candidate tools", downlink_rate_guardrail)
        self.assertIn("do not claim", downlink_rate_guardrail)

    def test_phase23_prompt_for_generic_topic_has_no_static_method_bias(self) -> None:
        prompt = build_phase2_phase3_prompt(
            topic="Generic wireless resource allocation with queue stability",
            handoff={"final_title": "Generic Wireless Resource Allocation"},
            mathematical_contract_json='{"controls":[{"symbol":"x","status":"control"}],"derived_quantities":[],"objective":{"sense":"max","expression":"U(x)"},"constraints":[]}',
            system_model_md="A scheduler allocates a generic resource over users and queues.",
            problem_formulation_md="The original problem maximizes a declared utility under queue-stability constraints.",
            core_theory_package_md="No algorithmic auxiliary is part of the original problem.",
            convexity_audit_md="The utility is nonconcave in the scheduling variable.",
            reformulation_path_md="Use the reformulation supported by the current audit.",
        )

        self.assertNotIn("SWIPT", prompt)
        self.assertNotIn("WMMSE", prompt)
        self.assertNotIn("RIS", prompt)
        self.assertNotIn("CRB", prompt)

    def test_phase24_plan_normalizer_uses_declared_methods_only(self) -> None:
        yaml_text = """
problem_family: demo
objective_sense: maximize
paper_evidence_contract:
  compared_methods:
    - id: proposed
      role: proposed
      display_name_short: Proposed
    - id: greedy_baseline
      role: main_baseline
      display_name_short: Greedy
    - id: no_coupling_ablation
      role: mechanism_ablation
      display_name_short: No coupling
    - id: relaxed_bound
      role: upper_bound
      display_name_short: Relaxed bound
  required_result_columns: [method, seed, swept_param, swept_value, scenario_name, rate_mbps, constraint_violation_max]
  figures:
    - id: main
      claim: Main comparison over power budget
      chart_intent: main_comparison
      y_metric: rate_mbps
      required_sweep: power
      methods_to_run: [proposed, greedy_baseline, no_coupling_ablation, relaxed_bound]
    - id: feasibility
      claim: Feasibility under minimum-rate stress
      chart_intent: feasibility_boundary
      y_metric: constraint_violation_max
      required_sweep: qos
      methods_to_run: [proposed, greedy_baseline, no_coupling_ablation, relaxed_bound]
required_outputs:
  scalar_metrics: [rate_mbps, constraint_violation_max]
sweep_definitions:
  - id: power
    variable: system.Pmax
    values: [0.5, 1.0, 1.5, 2.0]
  - id: qos
    variable: constraints.Rmin
    values: [0.1, 0.2, 0.3, 0.4]
"""
        normalized = yaml.safe_load(normalize_phase24_validation_plan_yaml(yaml_text))
        evidence = normalized["paper_evidence_contract"]
        method_ids = [item["id"] for item in evidence["compared_methods"]]
        self.assertEqual(method_ids, ["proposed", "greedy_baseline", "no_coupling_ablation", "relaxed_bound"])
        self.assertEqual(evidence["figures"][0]["methods_to_run"], ["proposed", "greedy_baseline", "no_coupling_ablation"])
        self.assertNotIn("relaxed_bound", evidence["figures"][0]["methods_to_run"])
        self.assertNotIn("relaxed_bound", evidence["figures"][1]["methods_to_run"])
        self.assertNotEqual(evidence["figures"][1]["y_metric"], "constraint_violation_max")
        self.assertNotIn("leh_oris", str(normalized))

    def test_phase24_plan_normalizer_prefers_canonical_sweep_path(self) -> None:
        yaml_text = """
problem_family: demo
objective_sense: maximize
paper_evidence_contract:
  compared_methods:
    - id: proposed
      role: proposed
      display_name_short: Proposed
  required_result_columns: [method, seed, swept_param, swept_value, scenario_name, harvested_energy_mW]
required_outputs:
  scalar_metrics: [harvested_energy_mW]
sweep_definitions:
  - id: eh_threshold
    variable: constraints.E_min
    canonical_path: constraints.E_min_mW
    values: [0.0, 1.0, 2.0]
figure_targets:
  - id: eh_boundary
    claim: EH boundary
    y_metric: harvested_energy_mW
    required_sweep: eh_threshold
    methods_to_run: [proposed]
"""
        normalized = yaml.safe_load(normalize_phase24_validation_plan_yaml(yaml_text))
        figure = normalized["paper_evidence_contract"]["figures"][0]
        self.assertEqual(figure["required_sweep_param"], "constraints.E_min_mW")

    def test_phase24_plan_normalizer_canonicalizes_llm_alias_keys(self) -> None:
        yaml_text = """
problem_family: downlink_beamforming
objective_sense: maximize
research_evidence_contract:
  compared_methods:
    - id: proposed
      role: proposed
      scientific_purpose: Test the proposed optimizer.
      implementation_hint: Run the proposed update.
      fairness_rule: Same channels and constraints.
    - id: fixed_ps_baseline
      role: main_baseline
      scientific_purpose: Disable adaptive splitting.
      implementation_hint: Fix rho and optimize beams.
      fairness_rule: Same channels and constraints.
  required_result_columns: [method, seed, swept_param, swept_value, scenario_name, objective, feasible, sum_rate_bpsHz, harvested_energy_mW]
  figure_candidates:
    - figure_id: fig_power_rate
      claim_id: C1
      chart_intent: main_physical_kpi_comparison
      methods_to_run: [proposed, fixed_ps_baseline]
      sweep_id: sweep_power_budget
      y_metric: sum_rate_bpsHz
sweep_definitions:
  - sweep_id: sweep_power_budget
    path: budgets.Pmax_dBm
    paper_values: [20, 24, 28, 32]
required_outputs:
  tables: []
"""
        normalized = yaml.safe_load(normalize_phase24_validation_plan_yaml(yaml_text))

        self.assertIn("scalar_metrics", normalized["required_outputs"])
        self.assertIn("sum_rate_bpsHz", normalized["required_outputs"]["scalar_metrics"])
        self.assertEqual(normalized["sweep_definitions"][0]["id"], "sweep_power_budget")
        self.assertEqual(normalized["sweep_definitions"][0]["variable"], "budgets.Pmax_dBm")
        figures = normalized["research_evidence_contract"]["figures"]
        self.assertEqual(figures[0]["required_sweep"], "sweep_power_budget")
        self.assertEqual(figures[0]["y_metric"], "sum_rate_bpsHz")

    def test_phase25_sweep_refiner_preserves_contract_and_densifies_runs(self) -> None:
        prompt = build_phase25_sweep_refiner_prompt(
            topic="generic wireless optimization",
            algorithm_md="algorithm",
            benchmark_definition_md="benchmarks",
            experiment_plan_json="{}",
            available_data_summary_json="{}",
            deterministic_paper_sweep_plan_json='{"figures":[{"figure_id":"figure_2","methods_to_run":["proposed","rho_fixed_half"]}]}',
            missing_experiments_md="figure_2 misses rho_fixed_half",
        )
        self.assertIn("Do not change the predeclared KPI", prompt)
        self.assertIn("requires_phase24_design_revision", prompt)
        with tempfile.TemporaryDirectory() as tmp:
            phase25_dir = Path(tmp)
            (phase25_dir / "paper_sweep_plan.json").write_text(
                json.dumps(
                    {
                        "figures": [
                            {
                                "figure_id": "figure_2",
                                "required_sweep_param": "constraints.E_min_mW",
                                "medium_values": [0.0, 0.5, 1.0],
                                "suggested_values": [0.0, 1.0],
                                "suggested_num_seeds": 100,
                                "methods_to_run": ["proposed", "rho_fixed_half"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            normalized = _normalize_phase25_refined_sweep_plan(
                {
                    "status": "needs_more_phase24_runs",
                    "missing_for_figures": [
                        {
                            "figure_id": "figure_2",
                            "required_sweep_param": "constraints.E_min_mW",
                            "suggested_values": [1.0, 2.0, 4.0, 8.0],
                            "replace_existing_values": True,
                            "scout_values": [1.0, 8.0],
                            "medium_values": [1.0, 2.0, 4.0],
                            "medium_num_seeds": 12,
                            "suggested_num_seeds": 30,
                            "methods_to_run": ["proposed", "no_rho"],
                        }
                    ],
                },
                phase25_dir,
            )
        figure = normalized["figures"][0]
        self.assertEqual(figure["required_sweep_param"], "constraints.E_min_mW")
        self.assertEqual(figure["methods_to_run"], ["proposed", "rho_fixed_half"])
        self.assertEqual(figure["suggested_values"], [1.0, 2.0, 4.0, 8.0])
        self.assertEqual(figure["scout_values"], [1.0, 8.0])
        self.assertEqual(figure["medium_values"], [1.0, 2.0, 4.0])
        self.assertEqual(figure["medium_num_seeds"], 50)
        self.assertEqual(figure["suggested_num_seeds"], 100)

    def test_phase24_plan_normalizer_replaces_diagnostic_mechanism_axis(self) -> None:
        yaml_text = """
problem_family: demo
objective_sense: maximize
paper_evidence_contract:
  compared_methods:
    - id: proposed
      role: proposed
      display_name_short: Proposed
    - id: linear_eh
      role: mechanism_ablation
      display_name_short: Linear EH
  required_result_columns: [method, seed, swept_param, swept_value, scenario_name, sca_iterations, true_harvested_energy_mW]
  figures:
    - id: nonlinear_eh_mechanism
      claim: Nonlinear EH sigmoid mechanism
      chart_intent: mechanism_ablation
      y_metric: sca_iterations
      required_metrics: [sca_iterations, true_harvested_energy_mW, max_constraint_violation]
      required_sweep: sigmoid_steepness
      methods_to_run: [proposed, linear_eh]
required_outputs:
  scalar_metrics: [objective, sca_iterations, true_harvested_energy_mW, max_constraint_violation]
sweep_definitions:
  - id: sigmoid_steepness
    variable: EH.steepness_a
    paper_mode:
      values: [0.01, 0.1, 0.3]
  - id: power
    variable: system.Pmax
    values: [0.5, 1.0, 2.0]
"""
        normalized = yaml.safe_load(normalize_phase24_validation_plan_yaml(yaml_text))
        figure = normalized["paper_evidence_contract"]["figures"][0]
        self.assertEqual(figure["y_metric"], "objective")
        self.assertNotIn("sca_iterations", figure["required_metrics"])
        self.assertIn("sca_iterations", normalized["paper_evidence_contract"]["diagnostics"])

    def test_phase24_plan_normalizer_repairs_latex_backslashes_in_double_quotes(self) -> None:
        yaml_text = r"""
problem_family: demo
objective_sense: maximize
paper_evidence_contract:
  compared_methods:
    - id: proposed
      role: proposed
  required_result_columns: [method, seed, swept_param, swept_value, objective, sum_rate_bpsHz]
  figures:
    - id: figure_1
      claim: main
      chart_intent: main_comparison
      y_metric: sum_rate_bpsHz
      required_sweep: power
      methods_to_run: [proposed]
      axis_labels: {x: "$P_{\max}$ (mW)", y: "$R_{\mathrm{sum}}$"}
required_outputs:
  scalar_metrics: [objective, sum_rate_bpsHz]
sweep_definitions:
  - id: power
    variable: system.Pmax
    values: [0.5, 1.0, 2.0]
canonical_config:
  system:
    Pmax: 1.0
"""
        normalized = yaml.safe_load(sanitize_phase24_validation_plan_yaml(yaml_text))
        figure = normalized["paper_evidence_contract"]["figures"][0]
        self.assertEqual(figure["axis_labels"]["x"], "$P_{\\max}$ (mW)")
        self.assertEqual(figure["axis_labels"]["y"], "$R_{\\mathrm{sum}}$")

    def test_phase24_semantic_guardrail_does_not_treat_internal_cache_as_topic_cache(self) -> None:
        self.assertFalse(_phase24_concept_appears("problem._model_cache = {}", "cache"))
        self.assertFalse(_phase24_concept_appears("cached model data for generated solver", "cache"))
        self.assertTrue(_phase24_concept_appears("cache-aided wireless content placement", "cache"))

    def test_phase25_alignment_prefers_executable_contract_chart_type(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "phase2-4").mkdir(parents=True)
            (run_dir / "phase2-4" / "validation_plan.yaml").write_text(
                yaml.safe_dump(
                    {
                        "objective_sense": "maximize",
                        "required_outputs": {"scalar_metrics": ["weighted_sum_rate_bpsHz", "total_runtime_ms"]},
                        "paper_evidence_contract": {
                            "compared_methods": [{"id": "proposed", "role": "proposed"}],
                            "figures": [
                                {
                                    "id": "figure_1",
                                    "claim": "main comparison",
                                    "chart_type": "grouped_bar",
                                    "chart_intent": "main_comparison",
                                    "y_metric": "weighted_sum_rate_bpsHz",
                                    "required_sweep_param": "system.P_m_dBm",
                                    "methods_to_run": ["proposed"],
                                },
                                {
                                    "id": "figure_2",
                                    "claim": "runtime scaling",
                                    "chart_type": "line",
                                    "chart_intent": "mechanism_ablation",
                                    "y_metric": "total_runtime_ms",
                                    "required_sweep_param": "system.M",
                                    "methods_to_run": ["proposed"],
                                },
                            ],
                            "tables": [
                                {
                                    "id": "table_1",
                                    "columns": ["weighted_sum_rate_bpsHz_mean", "total_runtime_ms_mean"],
                                }
                            ],
                        },
                        "figure_targets": [
                            {
                                "id": "figure_1",
                                "claim": "stale raw target",
                                "chart_type": "line",
                                "y_metric": "weighted_sum_rate_bpsHz",
                            }
                        ],
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            aligned = align_phase25_plan_with_phase24_contract(
                {
                    "primary_metric": {"name": "weighted_sum_rate_bpsHz", "higher_is_better": True},
                    "compared_methods": [],
                    "figure_specs": [{"figure_id": "figure_1", "chart_type": "line"}],
                    "table_specs": [],
                },
                run_dir,
            )
        self.assertEqual(aligned["figure_specs"][0]["chart_type"], "grouped_bar")
        columns = aligned["table_specs"][0]["columns"]
        self.assertNotIn("proposed_harvested_energy_mW_mean", columns)
        self.assertIn("proposed_weighted_sum_rate_bpsHz_mean", columns)

    def test_phase25_blocks_runtime_proxy_and_zero_variance_figures(self) -> None:
        import pandas as pd

        with tempfile.TemporaryDirectory() as tmp:
            phase25_dir = Path(tmp)
            df = pd.DataFrame(
                [
                    {
                        "case_id": f"figure_1_system.M_{m}",
                        "case_name": f"figure_1_system.M_{m}",
                        "figure_id": "figure_1",
                        "seed": seed,
                        "swept_param": "system.M",
                        "swept_value": m,
                        "scenario_name": "demo",
                        "method": method,
                        "status": "success",
                        "success": True,
                        "feasible": True,
                        "finite_primary_metric": True,
                        "weighted_sum_rate_bpsHz": 10.0,
                        "total_runtime_ms": float(m) * (3.0 if method == "baseline" else 1.0),
                        "runtime_proxy_used": True,
                    }
                    for m in (8, 16, 32, 64)
                    for seed in range(30)
                    for method in ("proposed", "baseline")
                ]
            )
            plan = {
                "primary_metric": {"name": "weighted_sum_rate_bpsHz", "higher_is_better": True},
                "compared_methods": [
                    {"internal_name": "proposed", "name": "proposed", "role": "proposed"},
                    {"internal_name": "baseline", "name": "baseline", "role": "main_baseline"},
                ],
                "figure_specs": [
                    {
                        "figure_id": "figure_1",
                        "chart_type": "line",
                        "chart_intent": "mechanism_ablation",
                        "methods": ["proposed", "baseline"],
                        "metric": {"name": "total_runtime_ms", "higher_is_better": False},
                        "encoding": {"x": {"field": "swept_value", "sweep_param": "system.M"}},
                        "data_requirements": {"min_points": 4, "min_samples_per_group": 30},
                    }
                ],
            }
            comparison = compute_relative_gain(
                build_per_case_comparison(df, plan),
                primary_metric="weighted_sum_rate_bpsHz",
                higher_is_better=True,
            )
            mc_report = run_monte_carlo_check(df, plan, "weighted_sum_rate_bpsHz", phase25_dir)
            report = check_data_sufficiency(df, comparison, plan, mc_report, quick_mode=False)
        issues = report["figures"][0]["blocking_issues"]
        self.assertIn("runtime_metric_is_proxy_not_measured", issues)
        self.assertIn("zero_variance_across_seeds", issues)
        self.assertFalse(report["figures"][0]["paper_ready"])

    def test_phase24_plan_normalizer_prefers_mechanism_evidence_over_raw_feasibility_line(self) -> None:
        yaml_text = """
problem_family: demo
objective_sense: maximize
paper_evidence_contract:
  compared_methods:
    - id: proposed
      role: proposed
      display_name_short: Proposed
    - id: fixed_phase
      role: main_baseline
      display_name_short: Fixed phase
    - id: no_rho
      role: mechanism_ablation
      display_name_short: No rho
    - id: rho_fixed_half
      role: mechanism_ablation
      display_name_short: Fixed rho
  required_result_columns: [method, seed, swept_param, swept_value, scenario_name, radar_snr_dB, feasible, optimal_rho, harvested_energy_mW, sum_rate_bps_hz]
required_outputs:
  scalar_metrics: [objective, radar_snr_dB, feasible, optimal_rho, harvested_energy_mW, sum_rate_bps_hz]
sweep_definitions:
  - id: lambda_sweep
    variable: optimization.lambda_s
    paper_mode:
      values: [0.1, 1.0, 5.0, 10.0]
  - id: eh_sweep
    variable: constraints.E_min_mW
    paper_mode:
      values: [0.0, 1.0, 2.0, 5.0]
figure_targets:
  - id: radar_main
    claim: Main comparison over sensing weight
    chart_intent: main_comparison
    y_metric: radar_snr_dB
    required_metrics: [radar_snr_dB, sum_rate_bps_hz]
    required_sweep: lambda_sweep
    methods_to_run: [proposed, fixed_phase, no_rho]
  - id: weak_feasibility
    claim: Two-dimensional feasible region heatmap over E_min and power
    chart_intent: feasibility_boundary
    chart_type: heatmap
    y_metric: feasible
    facet_field: system.Pmax
    required_metrics: [feasible, optimal_rho, max_constraint_violation]
    required_sweep: eh_sweep
    methods_to_run: [proposed, no_rho, rho_fixed_half]
  - id: rho_adaptation
    claim: Structural separation adaptation should vary with EH demand
    chart_intent: sensitivity
    y_metric: optimal_rho
    required_metrics: [optimal_rho, harvested_energy_mW, feasible]
    required_sweep: eh_sweep
    methods_to_run: [proposed, rho_fixed_half]
table_target:
  id: table_decomposed
  claim: Decomposed physical metrics
  columns: [method, sum_rate_bps_hz, radar_snr_dB, harvested_energy_mW, optimal_rho, feasible_rate]
"""
        normalized = yaml.safe_load(normalize_phase24_validation_plan_yaml(yaml_text))
        figures = normalized["paper_evidence_contract"]["figures"]

        self.assertEqual(figures[0]["y_metric"], "radar_snr_dB")
        self.assertEqual(figures[1]["y_metric"], "optimal_rho")
        self.assertEqual(figures[1]["required_sweep"], "eh_sweep")
        self.assertIn("rho_fixed_half", figures[1]["methods_to_run"])
        self.assertEqual(normalized["paper_evidence_contract"]["tables"], [])
        self.assertTrue(normalized["paper_evidence_contract"]["tables_optional"])

    def test_phase24_plan_normalizer_keeps_main_abstract_objective_but_preserves_secondary_physical_kpi(self) -> None:
        yaml_text = """
objective_sense: maximize
research_evidence_contract:
  primary_metric:
    name: service_margin_tau
    display_name: minimum normalized service surplus margin tau
    higher_is_better: true
  compared_methods:
    - id: proposed
      role: proposed
      display_name_short: Proposed
    - id: restricted_covariance
      role: main_baseline
      display_name_short: Restricted covariance
  required_result_columns: [method, seed, swept_param, swept_value, scenario_name, objective, feasible, service_margin_tau, min_harvested_dc_mW]
  figures:
    - id: figure_1
      claim: Main comparison over transmit resource
      chart_intent: main_comparison
      y_metric: min_harvested_dc_mW
      required_metrics: [service_margin_tau, min_harvested_dc_mW]
      required_sweep: power_budget_sweep
      methods_to_run: [proposed, restricted_covariance]
    - id: figure_2
      claim: Stress regime over harvested energy requirement
      chart_intent: stress_or_gain
      y_metric: min_harvested_dc_mW
      required_metrics: [service_margin_tau, min_harvested_dc_mW]
      required_sweep: eh_requirement_sweep
      methods_to_run: [proposed, restricted_covariance]
required_outputs:
  scalar_metrics: [min_harvested_dc_mW, service_margin_tau]
sweep_definitions:
  - id: power_budget_sweep
    canonical_path: resources.P_max_W
    scout_values: [0.5, 1.0, 1.5]
  - id: eh_requirement_sweep
    canonical_path: service.energy.P_dc_min_common_mW
    scout_values: [0.03, 0.08, 0.13]
"""
        normalized = yaml.safe_load(normalize_phase24_validation_plan_yaml(yaml_text))
        figures = normalized["research_evidence_contract"]["figures"]

        self.assertEqual(figures[0]["y_metric"], "service_margin_tau")
        self.assertEqual(figures[1]["y_metric"], "min_harvested_dc_mW")
        self.assertIn("harvested", figures[1]["axis_labels"]["y"].lower())


if __name__ == "__main__":
    unittest.main()
