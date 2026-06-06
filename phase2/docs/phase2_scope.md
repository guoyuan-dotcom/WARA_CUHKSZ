# Phase 2 Scope

Phase 2 turns the frozen Phase 1 handoff into a concrete wireless optimization study. It builds the system model, formulates the optimization problem, selects a tractable solution route, generates executable experiment code, and verifies numerical results for paper use.

Phase 2 does not perform broad topic exploration, large-scale literature review, or final manuscript writing. Those tasks are handled by Phase 1 and Phase 3.

## Responsibilities

- Build the wireless system model and original optimization formulation.
- Identify convexity, nonconvexity, discrete variables, coupling terms, and other tractability issues.
- Select a solution or reformulation route such as direct convex optimization, conic reformulation, semidefinite relaxation, successive convex approximation, bisection, alternating optimization, mixed-integer optimization, or a scoped heuristic.
- Specify the algorithm, solver requirements, comparison methods, validation principles, and supported claim scope.
- Generate Python experiment code and run it through the fixed validation harness.
- Verify result tables, logs, metrics, methods, and figures before passing a verified result package to Phase 3.

## Output Location

Phase 2 writes run artifacts under:

```text
outputs/paper_runs/phase2/<run_id>/
```

The verified Phase 2 handoff for Phase 3 is materialized after result verification.
