from __future__ import annotations

import sys

from run_phase_pipeline import main


def _ensure_phase2_only_default() -> None:
    args = sys.argv[1:]
    if "--stop-phase" not in args:
        sys.argv.extend(["--stop-phase", "2.5"])


if __name__ == "__main__":
    _ensure_phase2_only_default()
    main()
