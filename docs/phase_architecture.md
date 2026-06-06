# WARA Phase Architecture

WARA uses three high-level phases. A phase is a responsibility layer; each numbered entry below is a phase step controlled by the phase-native registry.

## Phase 1: Research Direction and Handoff

Phase 1 owns topic understanding and the downstream handoff. It must not write the final system model, algorithm, experiment code, or paper claims.

| Phase step | Owner | Purpose | Main artifacts |
| --- | --- | --- | --- |
| `phase1.1` | ScoutAgent | Research framing and wireless scope | `phase1-1/research_frame.json`, `phase1-1/research_object.json` |
| `phase1.2` | LiteratureAgent | Evidence grounding and reference requirements | `phase1-2/evidence_pack.json`, `phase1-2/evidence_pack.md` |
| `phase1.3` | ScoutAgent | Direction selection and frozen contracts | `phase1-3/selected_direction.json`, `phase1-3/contract_bundle.json` |
| `phase1.4` | Controller | Export handoff for downstream phases | `phase1-4/phase1_handoff.json`, `phase1_handoff.json` |

## Phase 2: Modeling, Solution, and Experiments

Phase 2 owns the research substance: system model, optimization problem, tractability route, algorithm specification, executable experiment package, and verified paper figures. It stops before paper prose.

| Phase step | Output slot | Owner | Purpose |
| --- | --- | --- | --- |
| `phase2.1` | `phase2-1` | FormulationAgent | Construct the system model and optimization problem |
| `phase2.2` | `phase2-2` | TheoryAgent | Check convexity and choose a reformulation route |
| `phase2.3` | `phase2-3` | TheoryAgent | Specify the algorithm, baseline requirements, validation principles, and experiment plan |
| `phase2.4` | `phase2-4` | ExperimentAgent | Implement and run the experiment code under the validation harness |
| `phase2.5` | `phase2-5` | ValidationAgent | Verify results and promote paper figures |

## Phase 3: Paper Writing, References, Review, and Revision

Phase 3 owns paper expression and quality control. It reads frozen Phase 2 contracts and verified evidence; it must not redesign the model, algorithm, or experiments.

| Phase step | Output slot | Owner | Purpose |
| --- | --- | --- | --- |
| `phase3.1` | `phase3-1` | WritingAgent | Technical sections drafting from frozen mathematical and algorithm contracts |
| `phase3.2` | `phase3-2` | AnalysisAgent | Numerical-results writing from verified evidence |
| `phase3.3` | `phase3-3` | WritingAgent | Technical sections assembly |
| `phase3.4` | `phase3-4` | LiteratureAgent + WritingAgent | Introduction, references, citation curation, full-paper preview |
| `phase3.5` | `phase3-5` | ReviewAgent | Final review and repair routing |
| `phase3.6` | `phase3-6` | RepairAgent + WritingAgent | Scoped final revision and export |

Source layout:
- Phase 3 source assets live in top-level `phase3/`, including `phase3/prompts/` and `phase3/docs/`.
- Runtime output is exposed under `outputs/paper_runs/phase3/<run_id>/`.
- Compatibility artifact directories inside active Phase 2 run folders remain available for pipeline internals, but public run packages are collected under `outputs/paper_runs/`.

## Phase Rules

- New user-facing language should say `Phase 1`, `Phase 2`, and `Phase 3`.
- New writing/review prompts should refer to `phase3.1-phase3.6` for paper-writing steps.
- Runtime artifact slots use phase-native names such as `phase2-4/validation_plan.yaml` and `phase3-2/numerical_results_section.tex`.
- Controller decisions should record phase step ids such as `phase2.4` and `phase3.1`.

## Design Boundary

This split is intentionally not cosmetic. It enforces a research workflow:

- Phase 1 finds and freezes a research direction.
- Phase 2 proves that the research direction can be modeled, solved, and experimentally supported.
- Phase 3 turns the supported research artifact into a paper and checks it.

The most important boundary is between `phase2.5` and `phase3.1`: after Phase 2.5, the experiment evidence is frozen for writing. If Phase 3 finds unsupported claims, the controller should route back to the owner in Phase 2 instead of letting the writing agents invent new evidence.

Phase 3 starts from paper-ready Phase 2.5 evidence by default. Draft, quick-mode, or baseline-only evidence remains a Phase 2 issue: the controller should repair the Phase 2.4 implementation/experiment design or expand Phase 2.5 coverage instead of allowing Phase 3 to write final numerical claims from unsupported figures.
