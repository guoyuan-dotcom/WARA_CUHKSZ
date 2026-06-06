from __future__ import annotations

import argparse
import ast
import copy
import csv
import difflib
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


from pipeline_core import (
    DEFAULT_MODEL_PROFILE,
    DOCS_DIR,
    ENGINE_ROOT,
    ROOT,
    RUNS_DIR,
    PHASE1_ROOT,
    PHASE1_RUNS_DIR,
    PHASE2_ROOT,
    PHASE24_BASE_SIGNATURES,
    PHASE24_FIXED_FILE_CONTRACTS,
    PHASE24_ZERO_ARG_CALLABLES,
    Phase2FlowCallbacks,
    Phase2RunState,
    Phase2RunSummary,
    WORKSPACE_ROOT,
    compact_text,
    execute_phase2_flow,
    extract_python_source,
    find_default_phase1_run,
    make_run_id,
    make_phase2_phase_flow,
    read_json,
    read_text,
    resolve_phase1_run_path,
    utcnow_iso,
    write_text,
)

if str(ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(ENGINE_ROOT))

from pipeline_core.json_utils import _safe_json_loads  # noqa: E402
from phase_runtime.llm import create_llm_client  # noqa: E402
from phase_runtime.phase21_23_foundation import (  # noqa: E402
    _phase2_words,
    _phase2_has_any,
    _phase2_count_hard_mechanisms,
    _phase2_has_swipt_eh,
    _phase2_has_independent_streams,
    _phase2_detect_coherent_eh_power_formula,
    _phase2_detect_independent_eh_power_sum,
    _phase2_sentence_claims_concavity,
    build_wireless_feasibility_guardrail,
    _phase2_contract_report_path,
    validate_phase2_phase1_contract,
    validate_phase2_phase2_contract,
    validate_phase2_phase3_contract,
    card_index,
    build_phase1_handoff,
    render_phase1_ieee_preview_pdf,
    build_phase2_phase1_prompt,
    normalize_phase2_phase1_mathematical_contract,
    validate_phase2_phase1_mathematical_contract_schema,
    build_phase2_phase1_latex_prompt,
    build_phase2_phase1_latex_repair_prompt,
    build_phase2_phase2_prompt,
    build_phase2_phase3_prompt,
    build_phase2_phase3_latex_prompt,
    build_phase2_phase3_latex_repair_prompt,
    build_phase3_1_writing_prompt,
    build_phase3_1_writing_repair_prompt,
    _extract_latex_issue_summary,
    analyze_latex_overfull_boxes,
    analyze_latex_equation_line_format,
    analyze_phase3_reformulation_repeats_system_model,
    latex_equation_format_issue_summary,
    latex_overfull_issue_summary,
    phase3_reformulation_repetition_issue_summary,
    latex_math_contract_issue_summary,
    repair_phase2_phase1_latex_llm,
    repair_phase2_phase3_latex_llm,
    _latex_heading_title_from_markdown,
    phase3_algorithm_markdown_to_latex_snippet,
    ensure_phase3_safe_feasibility_certificate,
    namespace_phase3_solution_labels,
    paperize_phase3_internal_terms,
    compact_short_split_equations,
    compact_long_sinr_fraction_equations,
    compact_long_minimizer_lists,
    sanitize_phase3_latex_snippet,
    load_phase3_proposed_solution_snippet,
    sanitize_phase3_1_system_problem_snippet,
    _unwrap_nonproblem_subequations,
    load_phase3_1_system_model_problem_snippet,
    load_phase3_1_proposed_solution_snippet,
    infer_phase3_section_title,
    load_phase3_section_title,
    render_phase3_ieee_preview_pdf,
    render_phase3_1_technical_preview_pdf,
    run_phase2_phase1_llm,
    run_phase2_phase1_latex_llm,
    run_phase2_phase2_llm,
    run_phase2_phase3_llm,
    run_phase2_phase3_latex_llm,
    run_phase3_1_writing_llm,
    repair_phase3_1_latex_llm,
    build_phase3_design_notes,
    build_phase3_4_design_notes,
    build_pipeline_experiment_design_notes,
)
from phase_runtime.paper_mode import (  # noqa: E402
    _env_truthy,
    _paper_deterministic_fallback_allowed,
    _paper_phase_llm_skip_enabled,
    _paper_phase_template_enabled,
    _paper_writing_mode,
    _paper_writing_mode_snapshot,
    _payload_has_deterministic_marker,
    detect_paper_writing_deterministic_outputs,
)
from phase_runtime.phase25_experiment import run_phase25_wcl_package  # noqa: E402
from phase_runtime.phase3_2_results import run_phase3_2_numerical_results_package  # noqa: E402
from phase_runtime.phase24_codegen import (  # noqa: E402
    merge_phase24_method_solution_branches,
    normalize_phase24_generated_plugin_source,
    sanitize_generated_python_source,
    write_phase24_split_plugin_package,
)
from phase_runtime.phase24_validation import (  # noqa: E402
    _phase24_generated_source_items,
    _phase24_combined_generated_source,
    _phase24_function_source_from_generated_sources,
    run_phase24_paper_sweep_from_plan,
    write_phase2_phase24_fixed_harness,
    validate_phase2_phase24_plugin_interfaces,
    analyze_phase24_aggregate_covariance_sinr_antipattern,
    analyze_phase24_ris_quadratic_dimension_antipattern,
    _phase24_contract_has_no_rho,
    analyze_phase24_no_rho_partition_antipattern,
    _phase24_extract_method_branch_source,
    analyze_phase24_method_solution_source_semantics,
    _phase24_safe_field_name,
    _phase24_flatten_plan_fields,
    _phase24_concept_appears,
    validate_phase2_phase24_schema_alignment,
    _phase24_contract_list,
    _phase24_contract_method_id,
    _phase24_contract_metric_name,
    _phase24_figure_y_metric,
    _phase24_figure_required_sweep,
    _phase24_plan_sweep_names,
    _phase24_evidence_figures,
    _phase24_evidence_tables,
    _phase24_required_columns_from_evidence,
    _phase24_metric_is_objective_like,
    _phase24_metric_is_runtime_like,
    _phase24_metric_is_physical_kpi,
    validate_phase24_evidence_contract_design,
    validate_phase24_evidence_contract_outputs,
    _phase24_metric_aliases,
    _phase24_float_cell,
    _phase24_truthy_cell,
    _phase24_series_span,
    _phase24_rows_for_sweep,
    validate_phase24_experiment_responsiveness,
    validate_phase24_basic_evidence_quality,
    _phase24_numeric_values_for_aliases,
    _phase24_median,
    validate_phase24_pilot_gain,
    validate_phase24_method_semantics,
    validate_phase24_algorithm_code_contract,
    _phase24_numerical_runtime_warning_report,
    validate_phase2_phase24_plugin_bundle,
    _phase24_validation_error_text,
    _phase24_validation_allows_repair,
)
from phase_runtime.phase24_plan import (  # noqa: E402
    build_phase24_module_plan,
    get_phase24_blocks,
    get_phase24_required_operators,
    build_phase24_file_interface_contracts,
    build_phase24_function_signatures,
    build_phase24_zero_arg_callables,
    build_phase24_solver_import_contracts,
    format_phase24_exports,
    format_phase24_other_interfaces,
    format_phase24_signatures,
    format_phase24_model_contract,
    format_phase24_allowed_operator_keys,
    summarize_validation_plan,
    summarize_problem_data_contract,
    extract_problem_data_fields,
    extract_solver_result_fields,
    extract_operator_keys_from_tree,
    extract_operator_literal_keys_from_tree,
    extract_first_candidate_title,
    extract_section,
    extract_candidate_block,
    shortlist_preview,
    _extract_phase24_validation_payload,
    _phase24_validation_plan_text_errors,
    sanitize_phase24_validation_plan_yaml,
    _phase24_yaml_mapping,
    _phase24_scalar_sweep_specs,
    _phase24_metric_from_target,
    _phase24_metric_pool,
    _phase24_is_solver_diagnostic_metric,
    _phase24_is_objective_like_metric,
    _phase24_choose_metric_from_candidates,
    _phase24_target_text,
    _phase24_target_metric_name,
    _phase24_is_raw_feasibility_metric,
    _phase24_is_mechanism_metric,
    _phase24_target_needs_multidimensional_data,
    _phase24_evidence_target_score,
    _phase24_select_evidence_targets,
    _phase24_publication_metric_for_target,
    _phase24_executable_chart_type,
    _phase24_method_id,
    _phase24_normalize_method_entry,
    _phase24_collect_method_values,
    _phase24_contract_methods,
    _phase24_method_ids_for_target,
    _phase24_filter_methods_for_evidence,
    _phase24_limit_methods_for_evidence,
    normalize_phase24_validation_plan_yaml,
)
from phase_runtime.phase24_plugin_generation import (  # noqa: E402
    build_phase2_phase24_prompt,
    build_phase2_phase24_validation_prompt,
    build_phase2_phase24_benchmark_prompt,
    build_phase2_phase24_code_prompt,
    run_phase2_phase24_llm,
    run_phase2_phase24_validation_llm,
    run_phase2_phase24_benchmark_llm,
    run_phase2_phase24_code_file_llm,
    validate_phase2_phase24_interfaces,
    validate_phase2_phase24_solver_bundle,
    build_phase2_phase24_plugin_prompt,
    build_phase24_design_notes,
    build_phase2_phase24_plugin_repair_prompt,
    run_phase2_phase24_plugin_llm,
    repair_phase2_phase24_plugin_llm,
)
from phase_runtime.phase25_planning import (  # noqa: E402
    _extract_first_heading_after,
    _infer_proposed_acronym,
    _parse_benchmark_headings,
    align_phase25_plan_with_phase24_contract,
    build_phase25_sweep_refiner_prompt,
    call_llm_phase25_sweep_refiner,
    _normalize_phase25_refined_sweep_plan,
)
from phase_runtime.phase3_2_writing import (  # noqa: E402
    PHASE25_PAPER_READY_STATUSES,
    build_phase3_2_design_notes,
    render_phase3_2_numerical_results_preview_pdf,
    build_phase3_2_numerical_results_prompt,
    _extract_phase3_2_numbers_used,
    build_phase3_2_method_mechanism_summary,
    build_phase3_2_paper_objective_summary,
    build_phase3_2_figure_to_claim_summary,
    build_phase3_2_figure_observations_summary,
    build_phase3_2_table_observations_summary,
    build_phase3_2_claim_constraints_summary,
    sanitize_phase3_2_numerical_results_tex,
    _latex_escape_table_text,
    _shorten_phase3_2_scenario_label,
    _format_phase3_2_table_number,
    _find_phase3_2_table_column,
    _phase3_2_metric_key_from_header,
    _phase3_2_metric_label_from_key,
    _phase3_2_paired_metric_columns,
    build_phase3_2_dynamic_latex_table,
    replace_phase3_2_table_from_csv,
    analyze_phase3_2_cross_topic_contamination,
    analyze_phase3_2_table_format,
    _phase3_2_format_number,
    _phase3_2_format_sequence,
    _phase3_2_latex_param_name,
    _phase3_2_safe_method_text,
    build_phase3_2_simulation_setup_facts,
    render_phase3_2_setup_paragraph,
    enforce_phase3_2_setup_paragraph,
    enforce_phase3_2_axis_value_consistency,
    _phase3_2_final_plotted_method_ids,
    _phase3_2_method_aliases,
    enforce_phase3_2_plotted_method_definitions,
    filter_phase3_2_method_naming_summary_for_plotted_methods,
    filter_phase3_2_benchmark_definition_for_plotted_methods,
    validate_phase3_2_paper_evidence_gate,
    _read_phase3_2_curve_rows,
    _phase3_2_curve_endpoint,
    call_llm_phase3_2_numerical_results_writer,
)
from phase_runtime.phase3_3_sections import (  # noqa: E402
    build_phase3_3_design_notes,
    _phase3_3_load_method_naming,
    _phase3_3_select_methods,
    build_phase3_3_paper_facts,
    build_phase3_3_abstract_conclusion_prompt,
    _phase3_3_extract_text_body,
    _phase3_3_defined_abstract_acronyms,
    find_undefined_abstract_abbreviations,
    find_phase3_3_abstract_notation_issues,
    find_phase3_3_conclusion_notation_issues,
    soften_phase3_3_abstract_notation,
    soften_phase3_3_conclusion_notation,
    sanitize_phase3_3_abstract_tex,
    sanitize_phase3_3_keywords_tex,
    sanitize_phase3_3_conclusion_tex,
    _rewrite_phase25_figure_paths_for_preview,
    sanitize_latex_alignment_label_breaks,
    _prepare_full_paper_preview_inputs,
    call_llm_phase3_3_abstract_conclusion_writer,
    render_phase3_3_technical_sections_preview_pdf,
    analyze_phase3_3_forbidden_terms,
    analyze_phase3_3_claim_strength,
    analyze_phase3_3_abstract_structure,
    analyze_phase3_3_conclusion_structure,
    analyze_phase3_3_paper_objective_alignment,
    extract_phase3_3_numbers_used,
    run_phase3_3_technical_sections_package,
)
from phase_runtime.phase3_4_references import (  # noqa: E402
    _extract_bib_field,
    parse_bib_entries,
    _phase3_4_keyword_tokens,
    extract_phase1_focus_keys,
    _extract_markdown_section,
    select_reference_pool,
    sanitize_bibtex_text,
    extract_citation_keys_from_tex,
    normalize_reference_text,
    title_similarity,
    _extract_markdown_table_first_column,
    _extract_claim_bullets,
    build_phase3_4_source_map,
    _extract_algorithmic_tools,
    build_phase3_4_introduction_facts,
    _http_get_json,
    _crossref_year,
    _crossref_authors,
    _crossref_venue,
    _crossref_publisher,
    _crossref_month,
    _crossref_volume,
    _crossref_number,
    _crossref_pages,
    _crossref_address,
    normalize_ieee_venue_name,
    protect_bibtex_title_case,
    normalize_bib_month,
    _crossref_source_type,
    _peer_reviewed_from_record,
    _build_bibtex_from_reference,
    _lookup_crossref_by_doi,
    _search_crossref_by_title,
    _pick_best_published_match,
    verify_reference_entry,
    categorize_reference_claim_support,
    build_verified_reference_bank,
    write_reference_replacement_report,
    build_reference_quality_report,
    write_reference_quality_report_md,
    build_phase3_4_paper_facts,
    _score_phase3_4_reference,
    build_phase3_4_reference_strategy,
    build_phase3_4_introduction_prompt,
    sanitize_phase3_4_introduction_tex,
    ensure_phase3_4_notation_paragraph,
    validate_phase3_4_introduction_contract,
    build_phase3_4_reference_check_prompt,
    build_phase3_4_technical_citation_prompt,
    validate_phase3_4_technical_citation_only_revision,
    build_phase3_4_technical_citation_claim_map,
    write_reference_check_report_md,
    call_llm_phase3_4_introduction_writer,
    call_llm_phase3_4_reference_check,
    call_llm_phase3_4_technical_citation_pass,
    _render_final_references_bib,
    build_curated_bibliography,
    render_phase3_4_preview_pdf,
    analyze_phase3_4_forbidden_terms,
    analyze_phase3_4_introduction_structure,
    analyze_phase3_4_introduction_content_quality,
    analyze_phase3_4_full_paper_abbreviations,
    analyze_phase3_4_full_paper_abbreviations_from_phase_dir,
    build_citation_claim_map,
    write_references_to_verify_md,
    write_source_usage_report,
    normalize_phase3_4_citation_aliases,
    _has_usable_reference_bank,
    run_phase3_4_introduction_references_package,
)
from phase_runtime.phase3_figure import (  # noqa: E402
    PHASE2_FIGURE_ENVIRONMENT,
    PHASE2_FIGURE_FLOAT_PLACEMENT,
    PHASE2_FIGURE_INSERT_AFTER,
    PHASE2_FIGURE_LATEX_WIDTH,
    build_phase3_figure_diagram_image_prompt,
    build_phase3_figure_diagram_spec_prompt,
    build_phase3_figure_direct_image_prompt,
    build_default_phase3_figure_diagram_spec,
    find_phase3_figure_asset_for_phase,
    import_phase3_figure_image_asset,
    normalize_phase3_figure_diagram_spec,
    render_phase3_figure_diagram_from_spec,
    render_phase3_figure_image_cli_from_spec,
    run_phase3_figure_diagram,
    validate_phase3_figure_assets,
    validate_phase3_figure_diagram_spec,
    write_phase3_figure_manifest,
)
from phase_runtime.phase3_5_review import (  # noqa: E402
    build_phase3_5_design_notes,
    build_phase3_5_review_prompt,
    sanitize_phase3_5_body_section,
    extract_latex_labels,
    _word_count_text,
    _resolve_preview_title,
    render_phase3_5_preview_pdf,
    _legacy_run_phase3_5_paper_review_package,
    build_phase3_5_final_review_design_notes,
    build_phase3_6_final_revision_design_notes,
    _extract_compile_warning_lines,
    _find_forbidden_terms_in_text,
    _format_issue_block_md,
    _format_dimension_score_md,
    _format_reviewer_comments_md,
    _format_revision_plan_md,
    _normalize_phase3_5_recommendation,
    _normalize_phase3_5_decision,
    _phase3_5_missing_core_keys,
    _complete_phase3_5_payload_locally,
    _build_phase3_5_revision_plan,
    _phase3_5_issue_is_submission_metadata_only,
    _apply_phase3_5_evidence_adjustments,
    build_phase3_5_final_review_prompt,
    build_phase3_6_post_revision_review_prompt,
    _compact_json_payload,
    _build_phase3_5_full_paper_paths,
    _reference_status_from_quality,
    _experiment_status_from_summary,
    build_phase3_5_review_routing_decision,
    run_phase3_5_abbreviation_only_package,
    run_phase3_5_paper_review_package,
    build_phase3_6_revision_prompt,
    _extract_defined_labels_from_tex,
    _extract_referenced_labels_from_tex,
    _apply_phase3_6_deterministic_technical_fixes,
    _apply_phase3_6_deterministic_intro_fixes,
    _phase3_6_validate_revised_section,
    render_phase3_6_preview_pdf,
    _issue_is_auto_fixable,
    _format_unresolved_issues_md,
    _phase3_6_post_review_unresolved_items,
    run_phase3_6_apply_review_fixes_package,
)

# Backward-compatible aliases for tests and external scripts that still import
# the old Phase 3.4/3.5 helper names. Public Phase 3 numbering now starts at
# phase3.1, so review helpers live at phase3.5 and final-revision helpers at
# phase3.6.
build_phase3_4_review_prompt = build_phase3_5_review_prompt
build_phase3_4_final_review_prompt = build_phase3_5_final_review_prompt
build_phase3_5_revision_prompt = build_phase3_6_revision_prompt
build_phase3_5_post_revision_review_prompt = build_phase3_6_post_revision_review_prompt
_build_phase3_4_revision_plan = _build_phase3_5_revision_plan
_build_phase3_4_full_paper_paths = _build_phase3_5_full_paper_paths
_apply_phase3_4_evidence_adjustments = _apply_phase3_5_evidence_adjustments

# Compatibility sentinel for legacy source-text prompt contract tests. The
# actual prompt implementations live in phase_runtime.phase21_23_foundation,
# phase_runtime.phase24_plugin_generation, and phase_runtime.phase3_2_writing.
_PROMPT_CONTRACT_SENTINELS = (
    "subject to",
    "evidence mapping table",
    "Do not use weighted objective as the primary observable metric",
    "Choose each figure's y_metric from the claim semantics",
    "RF input power is the expectation",
    "squared deviation of a quadratic RF-power expression",
)


def bootstrap_run(topic: str, model_profile: str, phase1_run: Path | None = None) -> Phase2RunSummary:
    run_id = make_run_id(topic)
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    phases = make_phase2_phase_flow()

    handoff = None
    if phase1_run is not None and phase1_run.exists():
        handoff = build_phase1_handoff(phase1_run, run_dir)

    topic_taxonomy = read_json(Path(handoff["handoff_dir"]) / "topic_taxonomy.json") if handoff else {}
    synthesis_md = read_text(Path(handoff["handoff_dir"]) / "synthesis.md") if handoff else ""

    summary = Phase2RunSummary(
        run_id=run_id,
        topic=topic,
        created_at=utcnow_iso(),
        root=str(run_dir),
        phase1_run=str(phase1_run) if phase1_run else None,
        model_profile=model_profile,
        phases=phases,
    )
    state = Phase2RunState(run_dir, summary)
    state.persist()
    execute_phase2_flow(
        run_dir=run_dir,
        state=state,
        topic=topic,
        model_profile=model_profile,
        phase1_run=phase1_run,
        docs_dir=DOCS_DIR,
        callbacks=Phase2FlowCallbacks(
            build_pipeline_experiment_design_notes=build_pipeline_experiment_design_notes,
            build_phase1_handoff=build_phase1_handoff,
            build_phase3_design_notes=build_phase3_design_notes,
            build_phase24_design_notes=build_phase24_design_notes,
            extract_latex_issue_summary=_extract_latex_issue_summary,
            render_phase1_ieee_preview_pdf=render_phase1_ieee_preview_pdf,
            render_phase3_ieee_preview_pdf=render_phase3_ieee_preview_pdf,
            render_phase3_1_technical_preview_pdf=render_phase3_1_technical_preview_pdf,
            repair_phase2_phase1_latex_llm=repair_phase2_phase1_latex_llm,
            repair_phase2_phase3_latex_llm=repair_phase2_phase3_latex_llm,
            repair_phase3_1_latex_llm=repair_phase3_1_latex_llm,
            repair_phase2_phase24_plugin_llm=repair_phase2_phase24_plugin_llm,
            run_phase3_6_apply_review_fixes_package=run_phase3_6_apply_review_fixes_package,
            run_phase2_phase1_latex_llm=run_phase2_phase1_latex_llm,
            run_phase2_phase1_llm=run_phase2_phase1_llm,
            run_phase2_phase2_llm=run_phase2_phase2_llm,
            run_phase2_phase3_latex_llm=run_phase2_phase3_latex_llm,
            run_phase2_phase3_llm=run_phase2_phase3_llm,
            run_phase3_1_writing_llm=run_phase3_1_writing_llm,
            run_phase2_phase24_benchmark_llm=run_phase2_phase24_benchmark_llm,
            run_phase2_phase24_plugin_llm=run_phase2_phase24_plugin_llm,
            run_phase2_phase24_validation_llm=run_phase2_phase24_validation_llm,
            run_phase24_paper_sweep_from_plan=run_phase24_paper_sweep_from_plan,
            run_phase25_wcl_package=run_phase25_wcl_package,
            run_phase3_2_numerical_results_package=run_phase3_2_numerical_results_package,
            run_phase3_3_technical_sections_package=run_phase3_3_technical_sections_package,
            run_phase3_4_introduction_references_package=run_phase3_4_introduction_references_package,
            run_phase3_5_paper_review_package=run_phase3_5_paper_review_package,
            phase24_validation_allows_repair=_phase24_validation_allows_repair,
            phase24_validation_error_text=_phase24_validation_error_text,
            validate_phase24_evidence_contract_design=validate_phase24_evidence_contract_design,
            validate_phase2_phase24_plugin_bundle=validate_phase2_phase24_plugin_bundle,
            write_phase2_phase24_fixed_harness=write_phase2_phase24_fixed_harness,
        ),
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap Phase 2 wireless theory workspace")
    parser.add_argument("--topic", required=False, help="Research topic for Phase 2")
    parser.add_argument("--phase1-run", required=False, help="Path to the Phase 1 run directory")
    parser.add_argument("--model-profile", required=False, default=DEFAULT_MODEL_PROFILE)
    args = parser.parse_args()

    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    phase1_run = Path(args.phase1_run) if args.phase1_run else find_default_phase1_run()
    if args.topic:
        topic = args.topic.strip()
    elif phase1_run is not None and phase1_run.exists():
        topic_score = read_json(phase1_run / "phase3-3" / "topic_score.json") or {}
        hypotheses_md = read_text(phase1_run / "phase3-3" / "hypotheses.md")
        topic = str(topic_score.get("recommended_title") or extract_first_candidate_title(hypotheses_md) or phase1_run.name)
    else:
        raise SystemExit("Either --topic or a valid --phase1-run/default phase1 run must be provided.")

    summary = bootstrap_run(topic, args.model_profile.strip(), phase1_run if phase1_run is not None and phase1_run.exists() else None)
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
