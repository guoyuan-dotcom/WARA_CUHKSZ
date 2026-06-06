from __future__ import annotations

from typing import Any


PHASE_FLOW: tuple[dict[str, Any], ...] = (
    {
        "phase_step": 1,
        "phase_id": "phase2.1",
        "phase": "phase2",
        "phase_name": "Phase 2: Modeling, Solution, and Experiments",
        "name": "2.1 Construct the System Model and Optimization Problem",
    },
    {
        "phase_step": 2,
        "phase_id": "phase2.2",
        "phase": "phase2",
        "phase_name": "Phase 2: Modeling, Solution, and Experiments",
        "name": "2.2 Check Convexity and Choose a Reformulation Route",
    },
    {
        "phase_step": 3,
        "phase_id": "phase2.3",
        "phase": "phase2",
        "phase_name": "Phase 2: Modeling, Solution, and Experiments",
        "name": "2.3 Specify the Algorithm, Baselines, and Experiment Plan",
    },
    {
        "phase_step": 4,
        "phase_id": "phase2.4",
        "phase": "phase2",
        "phase_name": "Phase 2: Modeling, Solution, and Experiments",
        "name": "2.4 Implement and Run the Experiment Code",
    },
    {
        "phase_step": 5,
        "phase_id": "phase2.5",
        "phase": "phase2",
        "phase_name": "Phase 2: Modeling, Solution, and Experiments",
        "name": "2.5 Verify Results and Promote Paper Figures",
    },
    {
        "phase_step": 6,
        "phase_id": "phase3.1",
        "phase": "phase3",
        "phase_name": "Phase 3: Paper Writing, References, Review, and Revision",
        "name": "3.1 Technical Sections Drafting",
    },
    {
        "phase_step": 7,
        "phase_id": "phase3.2",
        "phase": "phase3",
        "phase_name": "Phase 3: Paper Writing, References, Review, and Revision",
        "name": "3.2 Numerical Results Writing",
    },
    {
        "phase_step": 8,
        "phase_id": "phase3.3",
        "phase": "phase3",
        "phase_name": "Phase 3: Paper Writing, References, Review, and Revision",
        "name": "3.3 Technical Sections Assembly",
    },
    {
        "phase_step": 9,
        "phase_id": "phase3.4",
        "phase": "phase3",
        "phase_name": "Phase 3: Paper Writing, References, Review, and Revision",
        "name": "3.4 Introduction & Reference Curation",
    },
    {
        "phase_step": 10,
        "phase_id": "phase3.5",
        "phase": "phase3",
        "phase_name": "Phase 3: Paper Writing, References, Review, and Revision",
        "name": "3.5 Final Review / Pre-submission Review",
    },
    {
        "phase_step": 11,
        "phase_id": "phase3.6",
        "phase": "phase3",
        "phase_name": "Phase 3: Paper Writing, References, Review, and Revision",
        "name": "3.6 Final Revision / Apply Review Fixes",
    },
)


PHASE2_PHASE_FLOW: tuple[tuple[Any, str], ...] = tuple(
    (item["phase_step"], item["name"]) for item in PHASE_FLOW
)


def phase_for_step(phase_step: Any) -> dict[str, Any]:
    key = str(phase_step).strip().lower()
    for item in PHASE_FLOW:
        if str(item["phase_step"]).strip().lower() == key or str(item["phase_id"]).strip().lower() == key:
            return dict(item)
    return {
        "phase_step": phase_step,
        "phase_id": f"phase2.{phase_step}",
        "phase": "phase2",
        "phase_name": "Phase 2: Modeling, Solution, and Experiments",
        "name": f"phase2.{phase_step}",
    }


PHASE24_FIXED_FILE_CONTRACTS: dict[str, dict[str, list[str]]] = {
    "problem_data.py": {
        "classes": ["ProblemData", "SolverResult"],
        "functions": ["make_canonical_problem", "result_to_dict", "save_json", "save_csv"],
    },
    "validation_cases.py": {
        "classes": [],
        "functions": ["load_canonical_case", "make_validation_cases"],
    },
    "proposed_solver.py": {
        "classes": [],
        "functions": ["solve_proposed"],
    },
    "baseline_solver.py": {
        "classes": [],
        "functions": ["solve_baseline"],
    },
    "run_validation.py": {
        "classes": [],
        "functions": ["main"],
    },
}


PHASE24_BASE_SIGNATURES: dict[str, dict[str, list[str]]] = {
    "validation_cases.py": {
        "load_canonical_case": [],
        "make_validation_cases": [],
    },
    "proposed_solver.py": {
        "solve_proposed": ["problem", "seed"],
    },
    "baseline_solver.py": {
        "solve_baseline": ["problem", "seed"],
    },
}


PHASE24_ZERO_ARG_CALLABLES: dict[str, list[str]] = {
    "validation_cases.py": ["load_canonical_case", "make_validation_cases"],
}
