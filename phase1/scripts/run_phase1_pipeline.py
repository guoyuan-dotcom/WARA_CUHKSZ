from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

from wara_phase1_pipeline import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL_PROFILE,
    MODEL_PROFILES,
    make_run_id,
    normalize_model_profile,
    run_wara_phase1,
)


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ROOT.parent
OUTPUTS_ROOT = WORKSPACE_ROOT / "outputs"
PAPER_RUNS_ROOT = OUTPUTS_ROOT / "paper_runs"


def normalize_run_id(raw_run_id: str | None) -> str | None:
    value = str(raw_run_id or "").strip()
    if not value:
        return None
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", value):
        raise SystemExit("--run-id must use only letters, numbers, '.', '_' or '-' and be at most 80 characters.")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description="Run WARA-native Phase 1 controller")
    parser.add_argument("--topic", required=True, help="Research topic")
    parser.add_argument("--model-profile", default=DEFAULT_MODEL_PROFILE, choices=sorted(MODEL_PROFILES))
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--run-id", required=False, help="Optional stable run id, e.g. wara001")
    parser.add_argument(
        "--no-literature-search",
        action="store_true",
        help="Disable Phase 1 external literature search and use local seminal/query-plan evidence only.",
    )
    args = parser.parse_args()

    if args.no_literature_search:
        os.environ["WARA_PHASE1_LITERATURE_SEARCH"] = "0"
    else:
        os.environ.setdefault("WARA_PHASE1_LITERATURE_SEARCH", "1")

    topic = args.topic.strip()
    run_id = normalize_run_id(args.run_id) or make_run_id(topic)
    run_dir = PAPER_RUNS_ROOT / "phase1" / run_id
    result = run_wara_phase1(
        topic=topic,
        run_dir=run_dir,
        model_profile=normalize_model_profile(args.model_profile),
        max_tokens=max(4096, args.max_tokens),
    )
    print(f"Running WARA-native Phase 1 to: {result.run_dir}")
    print(f"Phase 1 handoff: {result.handoff_file}")
    print(f"Selected title: {result.selected_title}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
