# Pipeline Experiment Design

## Phase 2.3: derivation and proposed algorithm

Phase 2.3 is responsible for:
- theoretical reformulation
- proposed algorithm design
- convergence or complexity discussion, if available

Phase 2.3 is not responsible for:
- empirical claims
- benchmark or ablation design
- metric selection
- sweep axes or values
- the final 2-3 figures
- chart type decisions
- `paper_sweep_plan.json`
- paper-ready Monte Carlo sample sizes
- final x-axis point counts
- paper-ready sufficiency rules

## Phase 2.4: experiment contract and quick validation

Phase 2.4 is the only phase that designs executable experiments. It translates the frozen math/problem/theory interface into `generated_plugin.py` and runs a deterministic harness in quick mode.

Its job is to define:
- paper claims to test or falsify
- mandatory compared methods, including proposed, practical baselines, ablations, and justified oracle/reference diagnostics
- canonical configuration fields and sweep axes
- required physical KPI columns, feasibility diagnostics, and actual-used sweep diagnostics
- 2-3 figure evidence candidates and no required table target
- missing-experiment behavior when quick validation is not paper-sufficient

It also verifies:
- the plugin can run
- proposed and baseline both return structured outputs
- objectives, constraints, diagnostics, and serialization work
- draft validation results can be produced
- the executable validation plan follows the Phase 2.4 evidence contract rather than an accidental plotting template

Phase 2.4 quick results are not final paper figures.

## Phase 2.5: experiment planner and sufficiency checker

Phase 2.5 combines:
- Phase 2.4 evidence contract
- Phase 2.4 quick results or paper-sweep results requested by Phase 2.5
- the paper target

Phase 2.5 chooses rendering and compact reporting for the Phase 2.4 predeclared evidence:
- 2-3 figures
- no required table in the current WARA route
- chart types
- primary metric display and method labels

Phase 2.5 then checks whether the data are sufficient for a paper-ready package. If not, it generates `paper_sweep_plan.json`.

Phase 2.5 must not invent a new empirical claim, new benchmark story, or new parameter factor after seeing weak data. If the data fail the Phase 2.4 evidence contract, the correct output is a missing-experiment or claim-failure report.

## Paper-sweep executor

The paper-sweep executor runs `paper_sweep_plan.json` after Phase 2.5 requests denser runs.

It:
- reuses the same fixed harness
- reuses the same `generated_plugin.py`
- does not redesign the algorithm
- does not redesign the figures
- writes paper validation csv/json artifacts

## Phase 2.5 rerun

After paper-sweep execution, Phase 2.5 reruns and prefers the paper validation outputs over the quick validation outputs.

Only this rerun may produce paper-ready figures.

## Phase 3.3: technical sections assembly

Phase 3.3 assembles the current technical paper core:
- Phase 2.1 system model and problem formulation
- Phase 2.3 proposed solution
- Phase 3.2 numerical results

Phase 3.3 also generates:
- abstract
- conclusion

Phase 3.3 does not yet complete the full paper. Introduction and references are outside its current scope.

## Phase 3.4: introduction and references

phase3.4_introduction_reference_phase continues from the Phase 3.3 technical package.

It:
- inherits Phase 1 literature sources as a candidate related-work and bibliography pool
- inherits Phase 2 technical sources as the exact problem/method/benchmark source
- inherits Phase 2/3 result sources as the naming and claim-strength source
- verifies and replaces references before building the final bibliography
- writes the introduction from structured source facts rather than a mixed raw prompt
- compiles a bibliography-aware full paper preview

phase3.4_introduction_reference_phase does not redesign algorithms, experiments, or figures.

## Default paper standard

Quick mode:
- at least 20 seeds per point
- 3 to 5 x points for line-style drafts
- draft only

Paper minimum:
- 80 seeds per point
- at least 10 x points for line plots
- minimum acceptance floor

Paper preferred:
- 100 seeds per point
- about 12 x points for line plots
- 5 to 6 representative categories for grouped-bar plots
- default target

High confidence:
- 100 seeds per point
- about 15 x points for line plots

Quick mode never produces paper-ready figures.