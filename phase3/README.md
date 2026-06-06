# Phase 3 Runtime in WARA

Phase 3 owns paper writing, reference curation, review, and final revision. It consumes frozen Phase 2 contracts and experiment evidence; it must not redesign the system model, algorithm, or experiments.

Phase 3 numbering:
- `3.1` Technical Sections Drafting
- `3.2` Numerical Results Writing
- `3.3` Technical Sections Assembly
- `3.4` Introduction & Reference Curation
- `3.5` Final Review and Repair Routing
- `3.6` Final Revision and Package Export

Phase 3 is launched through `phase3/scripts/run_phase3_pipeline.py` from
`phase2-5/phase2_to_phase3_handoff.json`. Phase 2 runners stop after producing
that handoff; they do not write paper-facing technical sections.

Directory layout:
- `prompts/`: Phase 3 prompt templates and figure-writing prompts.
- `outputs/paper_runs/phase3/`: generated Phase 3 artifact mirrors for completed or active runs.
- `docs/`: Phase 3 design notes.

The public Phase 3 artifacts for each run are collected under `outputs/paper_runs/phase3/<run_id>/`.
