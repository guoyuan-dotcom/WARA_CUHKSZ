from __future__ import annotations

from .contracts import AgentContract, ArtifactSpec, GateSpec, validate_agent_contract


CONTENT_AGENT_IDS = (
    "scout_agent",
    "literature_agent",
    "formulation_agent",
    "theory_agent",
    "experiment_agent",
    "validation_agent",
    "analysis_agent",
    "writing_agent",
    "review_agent",
    "repair_agent",
)


def _artifact(
    artifact_id: str,
    path_hint: str,
    *,
    kind: str = "text",
    required: bool = True,
    frozen_required: bool = False,
    description: str = "",
) -> ArtifactSpec:
    return ArtifactSpec(
        id=artifact_id,
        path_hint=path_hint,
        kind=kind,
        required=required,
        frozen_required=frozen_required,
        description=description,
    )


def _gate(gate_id: str, purpose: str, checks: tuple[str, ...], failure_route: str) -> GateSpec:
    return GateSpec(id=gate_id, purpose=purpose, checks=checks, failure_route=failure_route)


def build_default_agent_registry() -> dict[str, AgentContract]:
    """Return the WARA role-agent registry.

    The registry is intentionally independent of low-level runtime numbering.
    `phase_hints` maps each role agent to the phase steps it can serve.
    """

    registry = {
        "scout_agent": AgentContract(
            id="scout_agent",
            role="Select and freeze a WCL-scope research direction with explicit gap, risks, and kill criteria.",
            phase_hints=("phase1.1", "phase1.3", "phase1.4"),
            input_artifacts=(
                _artifact("user_topic", "topic.txt"),
                _artifact("literature_signals", "phase1/evidence_pool.jsonl", kind="jsonl", required=False),
            ),
            output_artifacts=(
                _artifact("candidate_directions", "phase1/candidates.json", kind="json"),
                _artifact("selected_direction", "phase1/selection_decision.json", kind="json"),
                _artifact("handoff_manifest", "phase1/handoff_manifest.json", kind="json"),
            ),
            tools=("llm", "local_json_writer"),
            allowed_actions=(
                "propose wireless research directions",
                "rank novelty and feasibility risks",
                "emit kill criteria",
            ),
            forbidden_actions=(
                "write system-model equations",
                "write solver code",
                "claim numerical performance",
            ),
            gates=(
                _gate(
                    "scout_gate",
                    "Ensure the selected direction is wireless-relevant, modelable, and testable.",
                    ("wireless relevance", "explicit gap", "modelable variables", "testable evidence target"),
                    "repair_agent",
                ),
            ),
        ),
        "literature_agent": AgentContract(
            id="literature_agent",
            role="Ground motivation, related work, and citations in verified literature artifacts.",
            phase_hints=("phase1.2", "phase3.4"),
            input_artifacts=(
                _artifact("selected_direction", "phase1/selection_decision.json", kind="json", required=False),
                _artifact("search_plan", "phase1/search_plan.json", kind="json", required=False),
            ),
            output_artifacts=(
                _artifact("evidence_pack", "phase1/evidence_pack.json", kind="json", required=False),
                _artifact("reference_bank", "phase3-4/verified_reference_bank.json", kind="json"),
                _artifact("citation_claim_map", "phase3-4/citation_claim_map.json", kind="json"),
            ),
            tools=("literature_search", "crossref_lookup", "llm"),
            allowed_actions=(
                "select references from verified sources",
                "map claims to citation keys",
                "insert citations without changing technical content",
            ),
            forbidden_actions=(
                "invent references",
                "change the selected research topic",
                "modify mathematical contracts",
            ),
            gates=(
                _gate(
                    "citation_gate",
                    "Check that cited keys exist and support the cited claims.",
                    ("known bib keys", "claim-reference mapping", "arxiv usage limit"),
                    "repair_agent",
                ),
            ),
        ),
        "formulation_agent": AgentContract(
            id="formulation_agent",
            role="Construct the original system model, variables, objective, constraints, and mathematical contract.",
            phase_hints=("phase2.1",),
            input_artifacts=(
                _artifact("handoff_manifest", "phase1_handoff_manifest.json", kind="json"),
                _artifact("evidence_pack", "input_from_phase1/topic_focused_literature.json", kind="json", required=False),
            ),
            output_artifacts=(
                _artifact("system_model", "phase2-1/system_model.md"),
                _artifact("problem_formulation", "phase2-1/problem_formulation.md"),
                _artifact("mathematical_contract", "phase2-1/mathematical_contract.json", kind="json"),
            ),
            tools=("llm", "json_schema_checker", "latex_checker"),
            allowed_actions=(
                "define physical variables",
                "define derived quantities",
                "formulate the original optimization problem",
            ),
            forbidden_actions=(
                "introduce reformulation-only variables into the original problem",
                "change the selected topic",
                "invent unsupported wireless mechanisms",
            ),
            gates=(
                _gate(
                    "formulation_gate",
                    "Separate controls, parameters, derived quantities, objective, and constraints.",
                    ("schema validity", "variable role consistency", "physical constraint meaning"),
                    "repair_agent",
                ),
            ),
        ),
        "theory_agent": AgentContract(
            id="theory_agent",
            role="Select the tractability route, audit convexity, define reformulation/solution route, algorithm contract, and proof/convergence scope.",
            phase_hints=("phase2.2", "phase2.3"),
            input_artifacts=(
                _artifact("mathematical_contract", "phase2-1/mathematical_contract.json", kind="json", frozen_required=True),
                _artifact("system_model", "phase2-1/system_model.md"),
                _artifact("problem_formulation", "phase2-1/problem_formulation.md"),
            ),
            output_artifacts=(
                _artifact("convexity_audit", "phase2-2/convexity_audit.md"),
                _artifact("reformulation_path", "phase2-2/reformulation_path.md"),
                _artifact("tractability_route_policy", "phase2-2/tractability_route_policy.json", kind="json"),
                _artifact("algorithm_contract", "phase2-2/algorithm_contract.json", kind="json"),
                _artifact("algorithm_description", "phase2-3/algorithm.md"),
            ),
            tools=("llm", "math_consistency_checker"),
            frozen_contracts=("mathematical_contract",),
            allowed_actions=(
                "introduce reformulation-only variables with scope",
                "define algorithm execution requirements",
                "state assumptions for convergence claims",
            ),
            forbidden_actions=(
                "change the original objective",
                "move surrogate variables into the system model",
                "claim global optimality without scoped assumptions",
            ),
            gates=(
                _gate(
                    "theory_gate",
                    "Ensure theory does not rewrite the frozen original problem.",
                    ("contract preservation", "surrogate scope", "claim scope"),
                    "repair_agent",
                ),
            ),
        ),
        "experiment_agent": AgentContract(
            id="experiment_agent",
            role="Generate executable experiment code, benchmark implementations, parameter exploration, validation outputs, and final figures.",
            phase_hints=("phase2.4",),
            input_artifacts=(
                _artifact("mathematical_contract", "phase2-1/mathematical_contract.json", kind="json", frozen_required=True),
                _artifact("algorithm_contract", "phase2-2/algorithm_contract.json", kind="json", frozen_required=True),
                _artifact("algorithm_description", "phase2-3/algorithm.md"),
                _artifact("paper_target", "phase2_summary.json", kind="json"),
            ),
            output_artifacts=(
                _artifact("experiment_plan", "phase2-4/validation_plan.yaml", kind="yaml"),
                _artifact("main_experiment_code", "phase2-4/solver/generated_plugin.py", kind="code"),
                _artifact("validation_results", "phase2-4/solver/outputs/validation_results.csv", kind="csv"),
                _artifact("figure_data", "phase2-5/figures", kind="directory"),
                _artifact("experiment_report", "phase2-5/phase25_experiment_summary.json", kind="json"),
            ),
            tools=("llm", "python", "local_filesystem", "plotting", "csv_reader"),
            frozen_contracts=("mathematical_contract", "algorithm_contract"),
            allowed_actions=(
                "infer paper-level KPIs and sweep knobs from frozen problem semantics",
                "generate executable experiment code",
                "select fair practical benchmarks",
                "run small scout sweeps before expanding point count",
                "drop invalid benchmarks with a recorded reason",
                "publish topic-agnostic figure and claim-evidence metadata for downstream writing",
            ),
            forbidden_actions=(
                "change the mathematical objective",
                "reuse topic-specific metrics or method names from unrelated runs",
                "redefine method ids after validation begins",
                "promote feasibility-only diagnostics as primary paper evidence",
                "change figure claims without rerunning evidence",
            ),
            gates=(
                _gate(
                    "experiment_gate",
                    "Check executable code, finite metrics, sweep consumption, and paper-ready evidence.",
                    (
                        "python import",
                        "required outputs",
                        "finite metrics",
                    "topic-agnostic artifact schema",
                        "same benchmarks across figures",
                        "metadata matches generated figures",
                        "claim-evidence alignment",
                    ),
                    "repair_agent",
                ),
            ),
            notes=(
                "Phase 2.4 owns plan/code/results/figures and should publish evidence metadata for Phase 2.5."
            ),
        ),
        "validation_agent": AgentContract(
            id="validation_agent",
            role="Package Phase 2.4 experiment outputs into verified evidence artifacts for downstream numerical-results writing.",
            phase_hints=("phase2.5",),
            input_artifacts=(
                _artifact("experiment_report", "phase2-5/phase25_experiment_summary.json", kind="json"),
                _artifact("figure_data", "phase2-5/figures", kind="directory"),
            ),
            output_artifacts=(
                _artifact("research_evidence_contract", "phase2-5/research_evidence_contract.yaml", kind="yaml"),
                _artifact("figure_evidence", "phase2-5/figure_evidence.json", kind="json"),
                _artifact("benchmark_definitions", "phase2-5/benchmark_definitions.json", kind="json"),
            ),
            tools=("json_reader", "csv_reader", "local_filesystem"),
            allowed_actions=(
                "read Phase 2.4 outputs",
                "package benchmark definitions",
                "package figure evidence and supported claim scope",
            ),
            forbidden_actions=(
                "regenerate experiment code",
                "select new benchmarks after seeing weak data",
                "change plotted KPI definitions",
            ),
            gates=(
                _gate(
                    "validation_packaging_gate",
                    "Ensure Phase 2.5 packages Phase 2.4 evidence without redesigning the experiment.",
                    ("figure evidence completeness", "benchmark identity consistency", "supported claim scope"),
                    "repair_agent",
                ),
            ),
        ),
        "analysis_agent": AgentContract(
            id="analysis_agent",
            role="Write numerical-results claims and insights from verified evidence without changing experiments.",
            phase_hints=("phase3.2",),
            input_artifacts=(
                _artifact("research_evidence_contract", "phase2-5/research_evidence_contract.yaml", kind="yaml"),
                _artifact("figure_evidence", "phase2-5/figure_evidence.json", kind="json"),
            ),
            output_artifacts=(
                _artifact("numerical_results", "phase3-2/numerical_results_section.tex"),
                _artifact("claim_evidence_map", "phase3-2/claim_evidence_map.json", kind="json"),
            ),
            tools=("llm", "json_reader", "latex_checker"),
            allowed_actions=(
                "summarize trends from verified figures",
                "explain mechanisms and operating regimes",
                "record claim-evidence links",
            ),
            forbidden_actions=(
                "quote unsupported point values",
                "rename benchmarks",
                "invent new metrics or experiments",
            ),
            gates=(
                _gate(
                    "analysis_gate",
                    "Check that each numerical-results claim is supported by verified figure evidence.",
                    ("trend support", "mechanism explanation", "no unsupported numerical overclaiming"),
                    "repair_agent",
                ),
            ),
        ),
        "writing_agent": AgentContract(
            id="writing_agent",
            role="Write paper prose from frozen technical contracts and verified evidence without redefining the research content.",
            phase_hints=("phase3.1", "phase3.3", "phase3.4", "phase3.6"),
            input_artifacts=(
                _artifact("mathematical_contract", "phase2-1/mathematical_contract.json", kind="json", frozen_required=True),
                _artifact("algorithm_contract", "phase2-2/algorithm_contract.json", kind="json", frozen_required=True),
                _artifact("numerical_results", "phase3-2/numerical_results_section.tex"),
                _artifact("reference_bank", "phase3-4/verified_reference_bank.json", kind="json", required=False),
            ),
            output_artifacts=(
                _artifact("technical_draft_sections", "phase3-1/system_model_problem_formulation_ieee_wcl.tex"),
                _artifact("technical_sections", "phase3-3/phase3_3_technical_sections_preview.tex"),
                _artifact("full_paper_preview", "phase3-4/full_paper_preview.pdf", kind="pdf"),
            ),
            tools=("llm", "latex", "pdf_compile"),
            frozen_contracts=("mathematical_contract", "algorithm_contract"),
            allowed_actions=(
                "write IEEE WCL prose from supplied artifacts",
                "summarize trends supported by figures",
                "add citation commands supplied by LiteratureAgent",
            ),
            forbidden_actions=(
                "invent numerical values",
                "change equations or constraints while writing",
                "claim universal superiority",
            ),
            gates=(
                _gate(
                    "writing_gate",
                    "Check paper-native prose, compilation, references, and claim scope.",
                    ("latex compile", "no internal phase/runtime terms", "defined notation", "supported numerical claims"),
                    "repair_agent",
                ),
            ),
        ),
        "review_agent": AgentContract(
            id="review_agent",
            role="Critique the full artifact set for correctness, evidence, citations, and paper readiness.",
            phase_hints=("phase3.5",),
            input_artifacts=(
                _artifact("full_paper_preview", "phase3-4/full_paper_preview.pdf", kind="pdf"),
                _artifact("citation_claim_map", "phase3-4/citation_claim_map.json", kind="json"),
                _artifact("experiment_report", "phase2-5/phase25_experiment_summary.json", kind="json"),
            ),
            output_artifacts=(
                _artifact("review_report", "phase3-5/final_review_report.md"),
                _artifact("critical_issues", "phase3-5/phase3_5_review.json", kind="json"),
            ),
            tools=("llm", "latex_log_reader", "json_reader"),
            allowed_actions=(
                "identify correctness risks",
                "rank paper-blocking issues",
                "recommend bounded repair targets",
            ),
            forbidden_actions=(
                "rewrite artifacts directly",
                "silently change frozen contracts",
            ),
            gates=(
                _gate(
                    "review_gate",
                    "Decide whether a final revision is safe or deeper repair is needed.",
                    ("math consistency", "experiment evidence", "citation adequacy", "WCL style"),
                    "repair_agent",
                ),
            ),
        ),
        "repair_agent": AgentContract(
            id="repair_agent",
            role="Repair one reported issue against one artifact while preserving frozen contracts and recording the diff reason.",
            phase_hints=("phase2.1", "phase2.4", "phase3.1", "phase3.4", "phase3.5", "phase3.6"),
            input_artifacts=(
                _artifact("target_artifact", "<controller-selected>", required=True),
                _artifact("error_report", "<controller-selected>", required=True),
                _artifact("relevant_frozen_contract", "<controller-selected>", required=False, frozen_required=True),
            ),
            output_artifacts=(
                _artifact("repaired_artifact", "<controller-selected>"),
                _artifact("repair_log", "repair_log.md"),
            ),
            tools=("llm", "local_filesystem", "test_runner"),
            allowed_actions=(
                "repair the reported issue only",
                "rerun the relevant gate",
                "write a repair log",
            ),
            forbidden_actions=(
                "rewrite unrelated artifacts",
                "change frozen contracts without controller rollback",
                "hide gate failures",
            ),
            gates=(
                _gate(
                    "repair_gate",
                    "Ensure repair is scoped and the relevant gate now passes.",
                    ("diff scope", "gate rerun", "contract preservation"),
                    "review_agent",
                ),
            ),
        ),
    }

    errors = []
    for contract in registry.values():
        errors.extend(validate_agent_contract(contract))
    if errors:
        raise ValueError("Invalid default agent registry: " + "; ".join(errors))
    return registry


def get_agent_contract(agent_id: str) -> AgentContract:
    registry = build_default_agent_registry()
    key = str(agent_id or "").strip().lower()
    if key == "implementation_agent":
        key = "experiment_agent"
    try:
        return registry[key]
    except KeyError as exc:
        valid = ", ".join(sorted(registry))
        raise KeyError(f"unknown WARA agent {agent_id!r}; valid agents: {valid}") from exc


_PHASE_TO_AGENT_IDS = {
    "phase1.1": ("scout_agent",),
    "phase1.2": ("literature_agent",),
    "phase1.3": ("scout_agent",),
    "phase1.4": ("scout_agent", "literature_agent"),
    "phase2.1": ("formulation_agent",),
    "2.1": ("formulation_agent",),
    "phase2.2": ("theory_agent",),
    "2.2": ("theory_agent",),
    "phase2.3": ("theory_agent",),
    "2.3": ("theory_agent",),
    "phase2.4": ("experiment_agent",),
    "2.4": ("experiment_agent",),
    "phase2.5": ("validation_agent",),
    "2.5": ("validation_agent",),
    "phase3.1": ("writing_agent",),
    "3.1": ("writing_agent",),
    "phase3.2": ("analysis_agent",),
    "3.2": ("analysis_agent",),
    "phase3.3": ("writing_agent",),
    "3.3": ("writing_agent",),
    "phase3.4": ("literature_agent", "writing_agent"),
    "3.4": ("literature_agent", "writing_agent"),
    "phase3.5": ("review_agent",),
    "3.5": ("review_agent",),
    "phase3.6": ("repair_agent", "writing_agent"),
    "3.6": ("repair_agent", "writing_agent"),
}


def phase_to_agent_ids(phase_id: str) -> tuple[str, ...]:
    key = str(phase_id or "").strip().lower()
    if key not in _PHASE_TO_AGENT_IDS:
        valid = ", ".join(sorted(_PHASE_TO_AGENT_IDS))
        raise KeyError(f"unknown WARA phase {phase_id!r}; valid phases: {valid}")
    return _PHASE_TO_AGENT_IDS[key]
