# WARA Pipeline

WARA is a controller-orchestrated, artifact-mediated multi-agent research pipeline for wireless-optimization papers.

The release runner follows the paper architecture: Phase 1, Phase 2, and
Phase 3 are separate runner/controller stages connected by frozen handoff
artifacts.

## Phase Groups

Phase 1 is the research-direction layer:

- `phase1.1` research_framing
- `phase1.2` evidence_grounding
- `phase1.3` direction_contract
- `phase1.4` wara_handoff

Phase 2 is the modeling, solution, and experiment layer:

- `phase2.1` system_model_problem
- `phase2.2` tractability_route
- `phase2.3` algorithm_design
- `phase2.4` experiment_implementation
- `phase2.5` experiment_evidence_package

Phase 3 is the writing, reference, review, and revision layer:

- `phase3.1` technical_sections_drafting
- `phase3.2` numerical_results_writing
- `phase3.3` technical_sections_assembly
- `phase3.4` introduction_references
- `phase3.5` full_manuscript_review
- `phase3.6` final_revision

## Current Rules

- Phase1 does not choose baselines or write experiments.
- Phase1 research framing and direction selection use WARA-owned ontology, taxonomy, seed-reference, and LLM-client code. External literature search is the only remaining vendor adapter boundary.
- Phase2.4 owns experiment design, benchmark instantiation, implementation, quick validation, and executable package creation.
- Phase2.5 expands the approved Phase2.4 experiment package through scout, medium, and paper-level runs, promotes verified figures, and freezes the supported claim scope.
- Phase2.5 does not pass by default unless the evidence reaches a paper-ready status with paired proposed-baseline curves and a valid evidence audit.
- Phase3 consumes `phase2_to_phase3_handoff.json`; it does not rerun Phase2.
- Frozen mathematical and algorithm contracts are not silently rewritten downstream.

## CLI

```bash
python3 run_wara_pipeline.py --help
python3 run_wara_pipeline.py --topic "integrated sensing communication and powering"
python3 phase1/scripts/run_phase1_pipeline.py --topic "integrated sensing communication and powering"
python3 phase2/scripts/run_phase2_pipeline.py --phase1-handoff /path/to/phase1_handoff.json
python3 phase3/scripts/run_phase3_pipeline.py --phase2-run /path/to/phase2/run
```
