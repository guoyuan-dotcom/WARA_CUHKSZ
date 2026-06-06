from __future__ import annotations

from types import ModuleType

from .flow import Phase2FlowCallbacks


def make_phase2_flow_callbacks(runtime_impl: ModuleType) -> Phase2FlowCallbacks:
    """Build the callback bundle shared by the Phase 2 and Phase 3 runners."""
    return Phase2FlowCallbacks(
        build_pipeline_experiment_design_notes=runtime_impl.build_pipeline_experiment_design_notes,
        build_phase1_handoff=runtime_impl.build_phase1_handoff,
        build_phase3_design_notes=runtime_impl.build_phase3_design_notes,
        build_phase24_design_notes=runtime_impl.build_phase24_design_notes,
        extract_latex_issue_summary=runtime_impl._extract_latex_issue_summary,
        render_phase1_ieee_preview_pdf=runtime_impl.render_phase1_ieee_preview_pdf,
        render_phase3_ieee_preview_pdf=runtime_impl.render_phase3_ieee_preview_pdf,
        render_phase3_1_technical_preview_pdf=runtime_impl.render_phase3_1_technical_preview_pdf,
        repair_phase2_phase1_latex_llm=runtime_impl.repair_phase2_phase1_latex_llm,
        repair_phase2_phase3_latex_llm=runtime_impl.repair_phase2_phase3_latex_llm,
        repair_phase3_1_latex_llm=runtime_impl.repair_phase3_1_latex_llm,
        repair_phase2_phase24_plugin_llm=runtime_impl.repair_phase2_phase24_plugin_llm,
        run_phase3_6_apply_review_fixes_package=runtime_impl.run_phase3_6_apply_review_fixes_package,
        run_phase2_phase1_latex_llm=runtime_impl.run_phase2_phase1_latex_llm,
        run_phase2_phase1_llm=runtime_impl.run_phase2_phase1_llm,
        run_phase2_phase2_llm=runtime_impl.run_phase2_phase2_llm,
        run_phase2_phase3_latex_llm=runtime_impl.run_phase2_phase3_latex_llm,
        run_phase2_phase3_llm=runtime_impl.run_phase2_phase3_llm,
        run_phase3_1_writing_llm=runtime_impl.run_phase3_1_writing_llm,
        run_phase2_phase24_benchmark_llm=runtime_impl.run_phase2_phase24_benchmark_llm,
        run_phase2_phase24_plugin_llm=runtime_impl.run_phase2_phase24_plugin_llm,
        run_phase2_phase24_validation_llm=runtime_impl.run_phase2_phase24_validation_llm,
        run_phase24_paper_sweep_from_plan=runtime_impl.run_phase24_paper_sweep_from_plan,
        run_phase25_wcl_package=runtime_impl.run_phase25_wcl_package,
        run_phase3_2_numerical_results_package=runtime_impl.run_phase3_2_numerical_results_package,
        run_phase3_3_technical_sections_package=runtime_impl.run_phase3_3_technical_sections_package,
        run_phase3_4_introduction_references_package=runtime_impl.run_phase3_4_introduction_references_package,
        run_phase3_5_paper_review_package=runtime_impl.run_phase3_5_paper_review_package,
        phase24_validation_allows_repair=runtime_impl._phase24_validation_allows_repair,
        phase24_validation_error_text=runtime_impl._phase24_validation_error_text,
        validate_phase24_evidence_contract_design=runtime_impl.validate_phase24_evidence_contract_design,
        validate_phase2_phase24_plugin_bundle=runtime_impl.validate_phase2_phase24_plugin_bundle,
        write_phase2_phase24_fixed_harness=runtime_impl.write_phase2_phase24_fixed_harness,
    )
