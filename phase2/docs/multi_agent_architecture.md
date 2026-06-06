# WARA Multi-Agent Architecture

WARA should be a controller-orchestrated, artifact-mediated multi-agent research system. The goal is not to create more LLM calls; the goal is to make each role read frozen artifacts, write bounded artifacts, and pass deterministic gates before downstream phases can continue.

## Core Pattern

Controller:
- Owns scheduling, gate evaluation, retry policy, rollback, and stop decisions.
- Does not directly rewrite research content, solver code, or paper prose.
- Maintains phase-specific controller manifests: `phase2_controller_manifest.json` for technical/evidence construction and `phase3_controller_manifest.json` for synthesis/review/repair. The root `controller_manifest.json` is only a compatibility index that points to those ledgers.
- Freezes contracts only after their gates pass.
- Routes failures by type instead of blindly rerunning prompts.

Artifact Workspace:
- Is the only shared state between agents.
- Stores machine-readable handoffs such as `mathematical_contract.json`, `algorithm_contract.json`, `algorithm_execution_contract`, `validation_plan.yaml`, `results.csv`, and gate reports.

Role Agents:
- Read only declared input artifacts.
- Write only declared output artifacts.
- Cannot change frozen contracts unless the controller explicitly reopens an earlier phase.

## Agent Interfaces

ScoutAgent:
- Inputs: user topic, literature search artifacts, novelty criteria.
- Outputs: `candidate_directions.json`, `selected_direction.json`, `handoff_manifest.json`.
- Gate: wireless relevance, explicit gap, modelable variables, testable claims, kill criteria.

LiteratureAgent:
- Inputs: selected direction, search plan.
- Outputs: `evidence_pack.md`, `reference_map.bib`, `related_work_matrix.json`.
- Gate: every key claim maps to a citation; missing baselines are reported.

FormulationAgent:
- Inputs: handoff manifest, evidence pack.
- Outputs: `system_model.md`, `problem_formulation.md`, `mathematical_contract.json`.
- Gate: controls, parameters, derived quantities, objective, and constraints are separated.

TheoryAgent:
- Inputs: frozen mathematical contract and formulation artifacts.
- Outputs: `convexity_audit.md`, `reformulation_path.md`, `algorithm_contract.json`.
- Gate: original problem is not changed; surrogate/reformulation objects are scoped; `algorithm_execution_contract` is present.

ExperimentAgent:
- Inputs: `mathematical_contract.json`, `algorithm_execution_contract`, `validation_plan.yaml`.
- Outputs: `generated_experiment_core.py`, solver package.
- Gate: import, required functions, finite metrics, exact method ids, JSON-normalizable state.

ValidationAgent:
- Inputs: Phase2.4 solver package, figures, raw metrics, and validation logs.
- Outputs: `research_evidence_contract.yaml`, `figure_evidence.json`, `benchmark_definitions.json`.
- Gate: required KPIs, benchmark identities, figure metadata, and supported-claim scope are complete.

AnalysisAgent:
- Inputs: evidence contract and verified figure evidence.
- Outputs: `numerical_results.tex`, `claim_evidence_map.json`.
- Gate: each result paragraph discusses trend, mechanism, and insight without unsupported numerical overclaiming.

ReviewAgent:
- Inputs: all frozen contracts, code, results, and writing artifacts.
- Outputs: `review_report.md`, `critical_issues.json`.
- Gate: math consistency, code consistency, claim-evidence consistency, citation adequacy.

RepairAgent:
- Inputs: one artifact, one error report, and the relevant frozen contract.
- Outputs: repaired artifact and `repair_log.md`.
- Gate: only the reported issue is changed; frozen contracts are not silently rewritten.

## Current Phase Mapping

- Phase 1.1 maps to ScoutAgent for research framing.
- Phase 1.2 maps to LiteratureAgent for evidence grounding.
- Phase 1.3 maps to ScoutAgent for direction-contract selection.
- Phase 1.4 maps to the Controller for handoff materialization.
- Phase 2.1 maps to FormulationAgent.
- Phase 2.2 and Phase 2.3 map to TheoryAgent.
- Phase 2.4 maps to ExperimentAgent.
- Phase 2.5 maps to ValidationAgent and only packages Phase2.4 evidence.
- Phase 3.1 maps to WritingAgent.
- Phase 3.2 maps to AnalysisAgent.
- Phase 3.3 maps to WritingAgent.
- Phase 3.4 maps to LiteratureAgent.
- Phase 3.5 maps to ReviewAgent.
- Phase 3.6 maps to RepairAgent.

## Immediate Design Rules

- Phase 2.4 owns experiment generation, benchmark selection, parameter exploration, execution, and figures.
- Phase 2.5 must not regenerate experiments; it packages Phase2.4 outputs for Phase3.
- `algorithm_execution_contract` is part of `algorithm_contract.json` and is copied into `phase24_execution_contract.json`.
- Numerical evidence should report paper-level KPIs defined by the current mathematical contract, not abstract feasibility diagnostics unless the diagnostic itself is the claim.

## Failure Policy

Gate failure must stop downstream paper generation unless an explicit controller decision records that the artifact is safe to continue. Draft outputs, incomplete sweeps, and unsupported claim mappings are not paper-ready and must not be used for final claims.

## Controller Responsibilities

The controller manages five things:
- State management: record the current artifact path, version, producer, and frozen status.
- Context assembly: give each agent only declared inputs and frozen contracts.
- Quality gates: record deterministic gate results before downstream execution.
- Contract freezing: freeze `mathematical_contract.json`, `algorithm_contract.json`, and `validation_plan.yaml` only after passing gates.
- Failure routing: map error types to bounded repair agents.

Failure routing examples:
- JSON/YAML/schema errors go to `schema_repair_agent`.
- Original-problem or variable inconsistencies go to `formulation_repair_agent`.
- Import, syntax, missing function, or method-id errors go to `implementation_repair_agent`.
- Missing metrics, missing CSV columns, stale sweeps, or non-finite values go to `experiment_code_repair_agent`.
- Unsupported claims or draft-only evidence go to `analysis_or_writing_repair_agent`.

The controller implementation lives in `phase2/scripts/pipeline_core/controller.py` and is wired into `phase2/scripts/pipeline_core/flow.py`. It is intentionally small: it records artifacts, freezes contracts, assembles bounded context, records gate results, and chooses the next repair/stop/continue action. It does not generate research content.

Current Phase2 coverage:
- `phase2.1` records `formulation_agent`, registers `mathematical_contract`, `problem_contract`, and `model_audit`, then freezes the mathematical contract only after `formulation_gate` passes.
- `phase2.2` records `theory_agent`, registers `tractability_route_policy`, `convexity_audit`, `reformulation_path`, and `algorithm_contract`, then freezes the algorithm contract after `algorithm_contract_gate` passes.
- `phase2.3` records the theory claim scope through `theory_gate` and freezes `claim_map` only after the theory audit passes.
- `phase2.4` records `experiment_agent`, freezes `validation_plan`, `phase24_execution_contract`, benchmark plan, and experiment design contract after the design gate, then records implementation validation through `implementation_gate`.
- `phase2.5` records `validation_agent` evidence packaging and blocks downstream writing through `phase25_evidence_gate` when evidence is not paper-ready.
