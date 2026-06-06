# Phase 2.4 Experiment Requirements

Phase 2.3 specifies the algorithm route, baseline requirements, validation principles, and claim map. It should not choose final plotting details, concrete sweep grids, Monte Carlo settings, or paper figures.

Phase 2.4 is the experiment-design, implementation, and quick-validation phase step. Before code generation, it must define:

- Claims to support or falsify, including metric direction and failure criteria.
- Compared methods, including the proposed method, mandatory practical baselines, relevant mechanism ablations, and justified oracle/reference diagnostics.
- Canonical configuration fields that the generated solver will consume.
- Sweep definitions with exact dotted paths and quick-validation values.
- Physical KPI metrics, feasibility/violation diagnostics, and actual-used sweep diagnostics.
- Required result columns for all figure evidence targets.
- 2-3 figure evidence candidates; tables are disabled for the current WARA route unless a later phase explicitly re-enables them.
- Quality gates and missing-experiment behavior when quick validation cannot support a research claim.

The corresponding runtime artifacts are:

- `phase2-4/wireless_benchmark_plan.json`
- `phase2-4/experiment_design_contract.json`
- `phase2-4/validation_plan.yaml`
- `phase2-4/phase24_validation_source_contracts.json`

Phase 2.5 may refine rendering and request more runs, but it should not invent new claims, baselines, metrics, or sweep axes after seeing weak data.

Phase 2.4 must run quick validation before Phase 2.5. This run checks that the generated package executes, uses the declared sweep variable, evaluates the proposed method and comparison methods under the same model, and records valid result artifacts. Phase 2.5 then expands the same approved experiment design through scout, medium, and paper-level runs before promoting final figures.
