from __future__ import annotations

import argparse
from pathlib import Path


def continue_from_phase25(
    run_dir: Path,
    paper_target: str = "IEEE WCL",
    start_phase: int = 5,
    *,
    model_profile: str = "",
) -> None:
    del paper_target, start_phase, model_profile
    raise RuntimeError(
        "Phase 2 continuation stops at the frozen evidence handoff. "
        f"Use phase3/scripts/run_phase3_pipeline.py --phase2-run {Path(run_dir).resolve()} "
        "so Phase 3 owns technical writing, review, and final package export."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deprecated Phase 2 continuation entrypoint; launch Phase 3 with phase3/scripts/run_phase3_pipeline.py."
    )
    parser.add_argument("run_dir", help="Phase 2 run directory")
    parser.add_argument("--paper-target", default="IEEE WCL")
    parser.add_argument("--start-phase", type=int, default=5)
    parser.add_argument("--model-profile", default="")
    args = parser.parse_args()
    continue_from_phase25(
        Path(args.run_dir),
        paper_target=args.paper_target,
        start_phase=args.start_phase,
        model_profile=args.model_profile,
    )


if __name__ == "__main__":
    main()
