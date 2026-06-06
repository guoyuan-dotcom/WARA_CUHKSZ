from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import phase_runtime_impl as impl  # noqa: E402
from pipeline_core import (  # noqa: E402
    DEFAULT_MODEL_PROFILE,
    build_claim_map,
    build_experiment_design_contract,
    build_phase24_execution_contract,
    contract_prompt_block,
    read_json,
    read_text,
    select_wireless_benchmark_plan,
    write_json_artifact,
    write_text,
)
from pipeline_core.flow import (  # noqa: E402
    _write_phase2_to_phase3_handoff,
)


def _phase_number(item: dict[str, Any]) -> int | None:
    value = item.get("phase_step", item.get("phase"))
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _set_phase_status(summary: dict[str, Any], phase: int, status: str) -> None:
    for item in summary.get("phases", []):
        if isinstance(item, dict) and _phase_number(item) == phase:
            item["status"] = status
            return


def _write_summary(run_dir: Path, summary: dict[str, Any]) -> None:
    write_text(run_dir / "phase2_summary.json", json.dumps(summary, ensure_ascii=False, indent=2))


def _write_text_if_changed(path: Path, text: str) -> bool:
    if path.exists() and read_text(path) == text:
        return False
    write_text(path, text)
    return True


def _process_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class _RunLock:
    """Prevent concurrent continuation writers for the same run directory."""

    def __init__(self, run_dir: Path, entrypoint: str) -> None:
        self.run_dir = Path(run_dir).resolve()
        self.entrypoint = entrypoint
        self.lock_path = self.run_dir / ".wara_continuation.lock"
        self._acquired = False

    def __enter__(self) -> "_RunLock":
        self.run_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "pid": os.getpid(),
            "entrypoint": self.entrypoint,
            "created_at_epoch": time.time(),
        }
        while True:
            try:
                fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                existing = read_json(self.lock_path) or {}
                existing_pid = int(existing.get("pid") or 0) if isinstance(existing, dict) else 0
                if _process_is_alive(existing_pid):
                    raise RuntimeError(
                        "Another WARA continuation process is already writing this run: "
                        f"pid={existing_pid}, lock={self.lock_path}"
                    )
                try:
                    self.lock_path.unlink()
                except FileNotFoundError:
                    pass
                continue
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
            self._acquired = True
            return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if not self._acquired:
            return
        existing = read_json(self.lock_path) or {}
        if isinstance(existing, dict) and int(existing.get("pid") or 0) == os.getpid():
            try:
                self.lock_path.unlink()
            except FileNotFoundError:
                pass


def _phase24_repair_round_limit() -> int:
    for env_name in ("WARA_PHASE24_REPAIR_ROUNDS", "WCL_PHASE24_REPAIR_ROUNDS"):
        raw_value = os.environ.get(env_name)
        if raw_value is None:
            continue
        try:
            return max(0, int(raw_value))
        except ValueError:
            continue
    return 10


def _phase24_design_repair_round_limit() -> int:
    for env_name in ("WARA_PHASE24_DESIGN_REPAIR_ROUNDS", "WCL_PHASE24_DESIGN_REPAIR_ROUNDS"):
        raw_value = os.environ.get(env_name)
        if raw_value is None:
            continue
        try:
            return max(0, int(raw_value))
        except ValueError:
            continue
    return 10


def _phase3_1_writing_repair_round_limit() -> int:
    for env_name in ("WARA_PHASE3_1_WRITING_REPAIR_ROUNDS", "WCL_PHASE3_1_WRITING_REPAIR_ROUNDS"):
        raw_value = os.environ.get(env_name)
        if raw_value is None:
            continue
        try:
            return max(0, int(raw_value))
        except ValueError:
            continue
    return 3


PAPER_READY_PHASE25_STATUSES = {"paper_minimum_ready", "paper_preferred_ready", "high_confidence_ready"}


def _phase_step_dir_name(step: int) -> str:
    mapping = {
        4: "phase2-4",
        5: "phase2-5",
        6: "phase3-1",
        7: "phase3-2",
        8: "phase3-3",
        9: "phase3-4",
        10: "phase3-5",
        11: "phase3-6",
    }
    return mapping.get(int(step), f"phase-step-{step}")


def _phase25_is_paper_ready(status: str) -> bool:
    return str(status or "").strip() in PAPER_READY_PHASE25_STATUSES


def _phase25_status_from_result(run_dir: Path, result: dict[str, Any] | None = None) -> str:
    summary = read_json(run_dir / "phase2-5" / "phase25_experiment_summary.json") or {}
    for payload in (summary, result or {}):
        if not isinstance(payload, dict):
            continue
        for key in ("phase25_status", "status", "overall_status"):
            value = str(payload.get(key) or "").strip()
            if value:
                return value
    return "unknown"


def _phase25_auto_paper_run_limit() -> int:
    for env_name in ("WARA_PHASE25_AUTO_PAPER_RUNS", "WCL_PHASE25_AUTO_PAPER_RUNS"):
        raw_value = os.environ.get(env_name)
        if raw_value is None:
            continue
        try:
            return max(0, int(raw_value))
        except ValueError:
            continue
    return 10


def _phase25_bounded_budget_continuation_enabled() -> bool:
    raw_value = (
        os.environ.get("WARA_PHASE25_ALLOW_NONPAPER_CONTINUATION")
        or os.environ.get("WCL_ALLOW_DRAFT_PHASE25_CONTINUE")
        or ""
    )
    return str(raw_value).strip().lower() in {"1", "true", "yes"}


def _phase25_auto_paper_sweep_mode() -> str:
    raw_value = os.environ.get("WARA_PHASE25_AUTO_PAPER_MODE") or os.environ.get("WCL_PHASE25_AUTO_PAPER_MODE")
    value = str(raw_value or "auto").strip().lower()
    if value in {"auto", "progressive", "tiered"}:
        return "auto"
    if value in {"medium", "mid", "moderate"}:
        return "medium"
    if value in {"paper", "full", "preferred", "0", "false", "no"}:
        return "paper"
    return "scout"


def _phase25_sweep_mode_for_round(round_index: int) -> str:
    mode = _phase25_auto_paper_sweep_mode()
    if mode != "auto":
        return mode
    if round_index <= 1:
        return "scout"
    if round_index == 2:
        return "medium"
    return "paper"


def _phase25_validation_prefix_for_mode(mode: str) -> str:
    return {
        "scout": "scout_validation",
        "medium": "medium_validation",
        "paper": "paper_validation",
    }.get(mode, "scout_validation")


def _run_phase25_sweep_with_tier(run_dir: Path, mode: str) -> dict[str, Any]:
    quick_sweep = mode in {"scout", "medium"}
    previous_tier = os.environ.get("WARA_PHASE25_SWEEP_TIER")
    os.environ["WARA_PHASE25_SWEEP_TIER"] = mode
    try:
        return impl.run_phase24_paper_sweep_from_plan(run_dir, quick_sweep)
    finally:
        if previous_tier is None:
            os.environ.pop("WARA_PHASE25_SWEEP_TIER", None)
        else:
            os.environ["WARA_PHASE25_SWEEP_TIER"] = previous_tier


def _phase25_should_auto_expand(run_dir: Path, status: str) -> bool:
    return (not _phase25_is_paper_ready(status)) and (run_dir / "phase2-5" / "paper_sweep_plan.json").exists()


def _phase25_float(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if numeric == numeric else default


def _phase25_runtime_snapshot(run_dir: Path) -> dict[str, Any]:
    summary = read_json(run_dir / "phase2-5" / "phase25_experiment_summary.json") or {}
    if not isinstance(summary, dict):
        summary = {}
    preferred_source = str(summary.get("data_source") or summary.get("_phase25_data_source") or "quick_validation").strip()
    valid_sources = ("quick_validation", "scout_validation", "medium_validation", "paper_validation")
    if preferred_source not in set(valid_sources):
        preferred_source = "quick_validation"
    outputs_dir = run_dir / "phase2-4" / "solver" / "outputs"
    def load_values(results_path: Path) -> list[float]:
        values: list[float] = []
        if not results_path.exists():
            return values
        try:
            with results_path.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    raw_value = (
                        row.get("solve_time_sec")
                        or row.get("measured_solve_time_sec")
                        or row.get("runtime_seconds")
                        or row.get("solver_time_sec")
                    )
                    numeric = _phase25_float(raw_value, -1.0)
                    if numeric >= 0.0:
                        values.append(float(numeric))
        except Exception:
            return []
        return values

    candidates: list[dict[str, Any]] = []
    for source in valid_sources:
        results_path = outputs_dir / f"{source}_results.csv"
        values = load_values(results_path)
        if not values:
            continue
        candidates.append(
            {
                "source": source,
                "results_path": results_path,
                "values": values,
                "mtime": results_path.stat().st_mtime,
                "preferred": source == preferred_source,
            }
        )
    if not candidates:
        results_path = outputs_dir / f"{preferred_source}_results.csv"
        values = []
    else:
        candidates.sort(
            key=lambda item: (
                statistics.median(item["values"]),
                item["mtime"],
                1 if item["preferred"] else 0,
            ),
            reverse=True,
        )
        selected = candidates[0]
        preferred = next((item for item in candidates if item["preferred"]), None)
        if preferred and statistics.median(preferred["values"]) >= statistics.median(selected["values"]):
            selected = preferred
        preferred_source = str(selected["source"])
        results_path = Path(selected["results_path"])
        values = list(selected["values"])
    threshold = _phase25_float(
        os.environ.get("WARA_PHASE25_MAX_MEDIAN_SOLVE_TIME_SEC")
        or os.environ.get("WCL_PHASE25_MAX_MEDIAN_SOLVE_TIME_SEC"),
        5.0,
    )
    median_value = statistics.median(values) if values else 0.0
    mean_value = statistics.fmean(values) if values else 0.0
    max_value = max(values) if values else 0.0
    runtime_too_slow = bool(values and median_value > threshold)
    runtime_slow_is_failure = str(
        os.environ.get("WARA_PHASE25_RUNTIME_SLOW_IS_FAILURE")
        or os.environ.get("WCL_PHASE25_RUNTIME_SLOW_IS_FAILURE")
        or ""
    ).strip().lower() in {"1", "true", "yes", "block"}
    return {
        "runtime_data_source": preferred_source,
        "runtime_results_path": str(results_path),
        "runtime_num_rows": len(values),
        "runtime_median_solve_time_sec": median_value,
        "runtime_mean_solve_time_sec": mean_value,
        "runtime_max_solve_time_sec": max_value,
        "runtime_threshold_median_sec": threshold,
        "runtime_too_slow": runtime_too_slow,
        "runtime_slow_is_failure": runtime_slow_is_failure,
        "runtime_failure": bool(runtime_too_slow and runtime_slow_is_failure),
    }


def _phase25_claim_snapshot(run_dir: Path) -> dict[str, Any]:
    summary = read_json(run_dir / "phase2-5" / "phase25_experiment_summary.json") or {}
    if not isinstance(summary, dict):
        summary = {}
    primary = summary.get("primary_claim_check") if isinstance(summary.get("primary_claim_check"), dict) else {}
    strongest = summary.get("strongest_practical_baseline_audit") if isinstance(summary.get("strongest_practical_baseline_audit"), dict) else {}
    snapshot = {
        "phase25_status": str(summary.get("phase25_status") or ""),
        "num_comparable_cases": int(_phase25_float(summary.get("num_comparable_cases"), 0.0)),
        "primary_passes": bool(primary.get("passes", False)),
        "strongest_practical_passes": bool(strongest.get("passes", False)) if strongest else bool(primary.get("passes", False)),
        "proposed_win_rate": max(
            _phase25_float(summary.get("proposed_win_rate")),
            _phase25_float(primary.get("proposed_win_rate")),
            _phase25_float(strongest.get("proposed_win_rate")),
        ),
        "proposed_median_relative_gain": max(
            _phase25_float(summary.get("proposed_median_relative_gain"), -1.0),
            _phase25_float(primary.get("proposed_median_relative_gain"), -1.0),
            _phase25_float(strongest.get("proposed_median_relative_gain"), -1.0),
        ),
        "proposed_mean_relative_gain": max(
            _phase25_float(summary.get("proposed_mean_relative_gain"), -1.0),
            _phase25_float(primary.get("proposed_mean_relative_gain"), -1.0),
            _phase25_float(strongest.get("proposed_mean_relative_gain"), -1.0),
        ),
    }
    snapshot.update(_phase25_runtime_snapshot(run_dir))
    return snapshot


def _phase25_claim_promising(snapshot: dict[str, Any]) -> bool:
    if _phase25_is_paper_ready(str(snapshot.get("phase25_status") or "")):
        return True
    min_win_rate = _phase25_float(os.environ.get("WARA_PHASE25_PROMOTION_MIN_WIN_RATE"), 0.55)
    min_median_gain = _phase25_float(os.environ.get("WARA_PHASE25_PROMOTION_MIN_MEDIAN_GAIN"), 0.0)
    return (
        int(snapshot.get("num_comparable_cases") or 0) > 0
        and (
            (bool(snapshot.get("primary_passes")) and bool(snapshot.get("strongest_practical_passes", True)))
            or (
                _phase25_float(snapshot.get("proposed_win_rate")) >= min_win_rate
                and _phase25_float(snapshot.get("proposed_median_relative_gain"), -1.0) > min_median_gain
            )
        )
    )


def _phase25_can_skip_auto_expansion(status: str, snapshot: dict[str, Any]) -> bool:
    """Phase 2.5 should not skip paper-evidence expansion from quick evidence alone."""

    return False


PHASE25_COVERAGE_ONLY_BLOCKERS = {
    "quick_mode_only",
    "medium_mode_only",
    "too_few_x_points",
    "too_few_categories",
    "too_few_samples_for_box",
    "too_few_iterations",
    "insufficient_heatmap_coverage",
    "too_few_seeds_per_point",
    "too_few_effective_x_points_after_feasibility_filter",
    "too_few_medium_x_points_after_filter",
    "planned_x_points_missing_after_feasibility_filter",
    "low_paired_success_rate_per_x",
}


def _phase25_paper_quality_requires_phase24_design_revision(run_dir: Path) -> dict[str, Any]:
    report = read_json(Path(run_dir) / "phase2-5" / "plot_quality_report.json") or {}
    figures = report.get("figures") if isinstance(report, dict) else []
    if not isinstance(figures, list):
        figures = []
    design_blockers: list[dict[str, Any]] = []
    coverage_blockers: list[dict[str, Any]] = []
    ready_figures: list[dict[str, Any]] = []
    story_keys: list[tuple[str, str, str]] = []
    for item in figures:
        if not isinstance(item, dict) or not item.get("counts_toward_paper_minimum", True):
            continue
        issues = [str(issue) for issue in item.get("blocking_issues", []) if str(issue)]
        noncoverage = sorted({issue for issue in issues if issue not in PHASE25_COVERAGE_ONLY_BLOCKERS})
        figure_summary = {
            "figure_id": item.get("figure_id", ""),
            "x_axis_param": item.get("x_axis_param", ""),
            "required_sweep": item.get("required_sweep", ""),
            "y_metric": item.get("y_metric", ""),
            "num_x_points": item.get("num_x_points"),
            "min_seeds_per_x": item.get("min_seeds_per_x"),
        }
        if noncoverage:
            design_blockers.append(
                {
                    **figure_summary,
                    "blocking_issues": noncoverage,
                }
            )
        elif issues:
            coverage_blockers.append({**figure_summary, "blocking_issues": sorted(set(issues))})
        else:
            ready_figures.append(figure_summary)
        story_keys.append(
            (
                str(item.get("required_sweep") or ""),
                str(item.get("x_axis_param") or ""),
                str(item.get("y_metric") or ""),
            )
        )
    nonempty_story_keys = [key for key in story_keys if any(key)]
    if len(nonempty_story_keys) >= 2 and len(set(nonempty_story_keys)) < len(nonempty_story_keys):
        design_blockers.append(
            {
                "figure_id": "global",
                "blocking_issues": ["duplicate_final_figure_story"],
                "story_keys": [list(key) for key in nonempty_story_keys],
            }
        )
    if not design_blockers:
        return {
            "requires_phase24_design_revision": False,
            "ready_figures": ready_figures,
            "coverage_blockers": coverage_blockers,
        }
    return {
        "requires_phase24_design_revision": True,
        "reason": "paper_quality_has_noncoverage_blockers",
        "design_blockers": design_blockers,
        "ready_figures": ready_figures,
        "coverage_blockers": coverage_blockers,
        "repair_scope": {
            "preserve_figures": ready_figures,
            "redesign_figures": [
                item
                for item in design_blockers
                if str(item.get("figure_id") or "") and str(item.get("figure_id") or "") != "global"
            ],
        },
        "advice": (
            "Do not add more seeds or densify the same x-grid for noncoverage blockers. Preserve paper-ready "
            "figures unless they conflict with frozen contracts, and redesign only failed figures so each final "
            "figure has a responsive mechanism axis, a distinct story, and a metric that varies with the selected "
            "operating-regime parameter."
        ),
    }


def _phase25_selective_figure_repair_lines(quality_revision: dict[str, Any]) -> list[str]:
    if not bool(quality_revision.get("requires_phase24_design_revision")):
        return []
    repair_scope = quality_revision.get("repair_scope") if isinstance(quality_revision.get("repair_scope"), dict) else {}
    ready_figures = repair_scope.get("preserve_figures") if isinstance(repair_scope.get("preserve_figures"), list) else []
    redesign_figures = repair_scope.get("redesign_figures") if isinstance(repair_scope.get("redesign_figures"), list) else []
    lines = [
        "Selective figure repair contract:",
        "- Do not regenerate the whole experiment story if only one final figure failed. Repair at figure granularity.",
    ]
    if ready_figures:
        lines.extend(
            [
                "- Preserve these already paper-ready final figures, including their KPI, method set, and figure story unless a frozen contract makes them invalid:",
                json.dumps(ready_figures, ensure_ascii=False, indent=2)[:4000],
            ]
        )
    if redesign_figures:
        lines.extend(
            [
                "- Redesign or replace only these failed final figures. For each one, pick an active model/operating-regime parameter that should change the objective-equivalent KPI; do not merely add x-points or seeds:",
                json.dumps(redesign_figures, ensure_ascii=False, indent=2)[:4000],
            ]
        )
    lines.extend(
        [
            "- Keep the plotted method set consistent across final figures when scientifically valid; if a benchmark is invalid or degenerate, explain and replace it consistently in both figures.",
            "- If an x-axis repeatedly yields a flat KPI after enough seeds and paper-level x-points, treat that axis as scientifically inactive for this paper and choose a different non-topic-specific mechanism/operating-regime axis from the frozen model.",
        ]
    )
    return lines


def _phase25_experiment_redesign_limit() -> int:
    for env_name in ("WARA_PHASE25_REDESIGN_ROUNDS", "WCL_PHASE25_REDESIGN_ROUNDS"):
        raw_value = os.environ.get(env_name)
        if raw_value is None:
            continue
        try:
            return max(0, int(raw_value))
        except ValueError:
            continue
    return 10


def _phase25_coverage_extension_limit() -> int:
    """Extra bounded paper-sweep passes for densified x-grids.

    These are not scientific redesign rounds. They cover the common case where
    Phase 2.5 has already found a promising figure, then writes a denser
    `paper_sweep_plan.json` after the latest analysis. Without this small
    extension the controller can stop exactly when the next plan would simply
    insert missing in-span x-points.
    """

    for env_name in ("WARA_PHASE25_COVERAGE_EXTENSION_ROUNDS", "WCL_PHASE25_COVERAGE_EXTENSION_ROUNDS"):
        raw_value = os.environ.get(env_name)
        if raw_value is None:
            continue
        try:
            return max(0, int(raw_value))
        except ValueError:
            continue
    return 10


def _phase25_impl_repair_round_limit() -> int:
    for env_name in ("WARA_PHASE25_IMPL_REPAIR_ROUNDS", "WCL_PHASE25_IMPL_REPAIR_ROUNDS"):
        raw_value = os.environ.get(env_name)
        if raw_value is None:
            continue
        try:
            return max(0, int(raw_value))
        except ValueError:
            continue
    return 10


def _phase25_impl_repair_attempt_limit(applied_round_limit: int) -> int:
    for env_name in ("WARA_PHASE25_IMPL_REPAIR_ATTEMPTS", "WCL_PHASE25_IMPL_REPAIR_ATTEMPTS"):
        raw_value = os.environ.get(env_name)
        if raw_value is None:
            continue
        try:
            return max(applied_round_limit, int(raw_value))
        except ValueError:
            continue
    return max(applied_round_limit, 10)


def _phase25_refine_sweep_plan(run_dir: Path, *, topic: str, model_profile: str, reason: str, round_index: int) -> dict[str, Any]:
    phase25_dir = run_dir / "phase2-5"
    write_json_artifact(
        phase25_dir / f"paper_sweep_refiner_request_round{round_index}.json",
        {
            "reason": reason,
            "round": round_index,
            "source": "continue_phase2_from_phase24",
        },
    )
    result = impl.call_llm_phase25_sweep_refiner(
        run_dir=run_dir,
        topic=topic,
        algorithm_md=read_text(run_dir / "phase2-3" / "algorithm.md"),
        benchmark_definition_md=read_text(run_dir / "phase2-4" / "benchmark_plan.md")
        or read_text(run_dir / "phase2-3" / "benchmark_definition.md"),
        model_profile=model_profile,
    )
    write_json_artifact(phase25_dir / f"paper_sweep_refiner_result_round{round_index}.json", result if isinstance(result, dict) else {"result": result})
    return result if isinstance(result, dict) else {}


def _phase25_refiner_requires_phase24_design_revision(refined: dict[str, Any] | None) -> bool:
    if not isinstance(refined, dict):
        return False
    status = str(refined.get("status") or "").strip().lower()
    if status == "requires_phase24_design_revision":
        return True
    notes = refined.get("notes", [])
    notes_text = " ".join(str(item or "") for item in notes) if isinstance(notes, list) else str(notes or "")
    notes_text = notes_text.lower()
    return (
        "requires_phase24_design_revision" in notes_text
        or "phase 2.4 experiment-design repair" in notes_text
        or "phase 2.4 design revision" in notes_text
    )


def _phase25_defer_initial_refiner_redesign_until_sweep(
    run_dir: Path,
    current_status: str,
    refined: dict[str, Any] | None,
) -> bool:
    """Treat initial quick-mode refiner redesign requests as advisory.

    The refiner sees only sparse quick data. If deterministic Phase 2.4 design
    checks passed and a paper_sweep_plan exists, run the tiered scout/medium
    sweep before declaring that the experiment must be redesigned.
    """

    if str(current_status or "").strip() not in {"quick_mode_only", "needs_more_phase24_runs"}:
        return False
    if not _phase25_refiner_requires_phase24_design_revision(refined):
        return False
    if not (Path(run_dir) / "phase2-5" / "paper_sweep_plan.json").exists():
        return False
    design_check = read_json(Path(run_dir) / "phase2-4" / "phase24_evidence_contract_design_check.json") or {}
    if design_check and not bool(design_check.get("ok")):
        return False
    if bool(_phase25_objective_metric_alignment_precheck(Path(run_dir)).get("requires_phase24_design_revision")):
        return False
    return True


def _phase25_semantic_family(text: str) -> str:
    lowered = str(text or "").lower()
    families = {
        "utility": (
            "service level",
            "service-level",
            "minimum normalized",
            "eta",
            "eta_service",
            "eta service",
            "normalized service",
            "service margin",
            "service-margin",
            "normalized surplus",
            "deterministic surplus",
            "worst normalized",
            "utility",
            "tau",
            "\\tau",
        ),
        "energy": ("harvest", "harvested", "energy harvesting", "rf-to-dc", "rectifier", "powering", "dc power"),
        "rate": ("sum-rate", "sum rate", "communication rate", "weighted sum communication rate", "throughput", "spectral efficiency", "bps/hz", "sinr rate", " r_k", "r_k"),
        "sensing": ("sensing", "radar", "crb", "beampattern", "illumination"),
        "secrecy": ("secrecy", "confidential", "eavesdropper"),
        "power": ("transmit power", "sum power", "total power", "power consumption", "resource minimization"),
        "efficiency": ("energy efficiency", "bit/j", "bit per joule"),
        "reliability": ("outage", "reliability", "robust", "chance constraint"),
    }
    scores = {
        family: sum(lowered.count(token) for token in tokens)
        for family, tokens in families.items()
    }
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    if not ordered or ordered[0][1] <= 0:
        return ""
    if len(lowered) < 80:
        return ordered[0][0]
    if len(ordered) > 1 and ordered[0][1] < ordered[1][1] + 2:
        return ""
    return ordered[0][0]


def _phase25_objective_metric_alignment_precheck(run_dir: Path) -> dict[str, Any]:
    phase25_summary = read_json(run_dir / "phase2-5" / "phase25_experiment_summary.json") or {}
    primary_metric = ""
    if isinstance(phase25_summary.get("primary_metric"), dict):
        primary_metric = str(phase25_summary.get("primary_metric", {}).get("name") or "").strip()
    if not primary_metric:
        return {"requires_phase24_design_revision": False}
    math_contract = read_json(run_dir / "phase2-1" / "mathematical_contract.frozen.json") or read_json(run_dir / "phase2-1" / "mathematical_contract.json") or {}
    objective = math_contract.get("objective") if isinstance(math_contract, dict) else {}
    if isinstance(objective, dict):
        objective_bits = [
            str(objective.get("sense") or ""),
            str(objective.get("expression") or ""),
            str(objective.get("meaning") or ""),
            json.dumps(objective.get("terms") or [], ensure_ascii=False),
        ]
    else:
        objective_bits = [str(objective or "")]
    objective_text = "\n".join(objective_bits)
    objective_family = _phase25_semantic_family(objective_text)
    if not objective_family:
        objective_text = "\n".join(
            [
                json.dumps(objective if isinstance(objective, dict) else {}, ensure_ascii=False),
                read_text(run_dir / "phase2-2" / "reformulation_path.md")[:4000],
                read_text(run_dir / "phase2-3" / "algorithm.md")[:4000],
            ]
        )
        objective_family = _phase25_semantic_family(objective_text)
    metric_family = _phase25_semantic_family(primary_metric.replace("_", " "))
    if not objective_family or not metric_family or objective_family == metric_family:
        return {"requires_phase24_design_revision": False}
    # Ambiguous utility objectives may legitimately use a scalar utility alias.
    if any(token in primary_metric.lower() for token in ("objective", "utility", "weighted", "eta_service", "service_level", "service_margin", "normalized_service", "tau")):
        return {"requires_phase24_design_revision": False}
    return {
        "requires_phase24_design_revision": True,
        "reason": "primary_metric_family_mismatch_with_frozen_objective",
        "objective_family": objective_family,
        "primary_metric": primary_metric,
        "primary_metric_family": metric_family,
        "objective_text_excerpt": objective_text[:600],
        "advice": (
            "Phase 2.4 experiment-design repair must regenerate the experiment contract so the main evidence KPI "
            "matches the frozen objective-equivalent physical metric before any scout/medium/paper sweep."
        ),
    }


def _phase25_metric_candidates_for_family(run_dir: Path, family: str) -> list[str]:
    plan_path = Path(run_dir) / "phase2-4" / "validation_plan.yaml"
    try:
        plan = yaml.safe_load(read_text(plan_path)) or {}
    except Exception:
        plan = {}
    evidence = plan.get("research_evidence_contract") if isinstance(plan, dict) else {}
    if not isinstance(evidence, dict):
        evidence = {}
    required = []
    for key in ("required_result_columns",):
        payload = evidence.get(key)
        if isinstance(payload, list):
            required.extend(str(item) for item in payload if isinstance(item, str))
    outputs = plan.get("required_outputs") if isinstance(plan, dict) else {}
    if isinstance(outputs, dict):
        scalars = outputs.get("scalar_metrics")
        if isinstance(scalars, list):
            required.extend(str(item) for item in scalars if isinstance(item, str))
    family_tokens = {
        "utility": ("eta_service", "service_level", "service_margin", "normalized_service", "objective", "utility", "tau", "surplus"),
        "rate": ("weighted_sum_rate", "sum_rate", "throughput", "spectral_efficiency", "bpshz", "bps_hz"),
        "energy": ("harvest", "energy", "eh", "dc_power"),
        "sensing": ("sensing", "radar", "crb", "fim", "beampattern", "illumination"),
        "secrecy": ("secrecy", "confidential", "eavesdrop"),
        "power": ("total_tx_power", "sum_power", "transmit_power", "power_consumption"),
        "efficiency": ("energy_efficiency", "bit_per_joule", "bit/j", "ee_"),
        "reliability": ("outage", "reliability", "success_probability", "violation_probability"),
    }
    tokens = family_tokens.get(str(family or "").strip().lower(), ())
    candidates: list[str] = []
    for name in required:
        normalized = name.lower().replace("/", "_").replace("-", "_")
        normalized_compact = normalized.replace("_", "")
        if any(token in normalized or token.replace("_", "") in normalized_compact for token in tokens):
            if name not in candidates:
                candidates.append(name)
    return candidates[:8]


def _latest_json_artifact(directory: Path, pattern: str) -> dict[str, Any]:
    candidates = [path for path in Path(directory).glob(pattern) if path.is_file()]
    if not candidates:
        return {}
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    payload = read_json(candidates[0]) or {}
    return payload if isinstance(payload, dict) else {}


def _phase25_design_revision_feedback_for_phase24(run_dir: Path) -> str:
    """Summarize prior Phase 2.5 evidence as positive Phase 2.4 experiment-design repair input.

    This intentionally avoids topic-specific fixes. The feedback tells Phase 2.4
    what failed scientifically, while preserving frozen math, objective KPI,
    fair benchmarks, and reproducibility.
    """

    phase25_dir = Path(run_dir) / "phase2-5"
    if not phase25_dir.exists():
        return ""
    quality_gate = read_json(phase25_dir / "phase25_quality_gate.json") or {}
    auto_manifest = read_json(phase25_dir / "phase25_auto_expansion_manifest.json") or {}
    summary = read_json(phase25_dir / "phase25_experiment_summary.json") or {}
    refiner = _latest_json_artifact(phase25_dir, "paper_sweep_refiner_result_round*.json")
    quality_revision = _phase25_paper_quality_requires_phase24_design_revision(run_dir)
    status = str(
        quality_gate.get("phase25_status")
        or auto_manifest.get("final_phase25_status")
        or summary.get("phase25_status")
        or refiner.get("status")
        or ""
    ).strip()
    reason = str(auto_manifest.get("reason") or quality_gate.get("reason") or "").strip()
    refiner_status = str(refiner.get("status") or "").strip()
    trigger_text = " ".join([status, reason, refiner_status]).lower()
    alignment_precheck = _phase25_objective_metric_alignment_precheck(run_dir)
    primary_mismatch = bool(alignment_precheck.get("requires_phase24_design_revision"))
    should_feedback = any(
        token in trigger_text
        for token in (
            "requires_phase24_design_revision",
            "promotion_failed_after_",
            "claim_not_promising",
            "primary_metric_family_mismatch",
        )
    ) or bool(quality_revision.get("requires_phase24_design_revision"))
    if not should_feedback:
        return ""

    primary_metric = ""
    if isinstance(summary.get("primary_metric"), dict):
        primary_metric = str(summary.get("primary_metric", {}).get("name") or "").strip()
    snapshot = {
        "phase25_status": status,
        "auto_expansion_reason": reason,
        "primary_metric": primary_metric,
        "objective_metric_alignment_precheck": alignment_precheck,
        "objective_aligned_metric_candidates": _phase25_metric_candidates_for_family(
            run_dir,
            str(alignment_precheck.get("objective_family") or ""),
        ),
        "num_comparable_cases": summary.get("num_comparable_cases"),
        "proposed_win_rate": summary.get("proposed_win_rate"),
        "proposed_median_relative_gain": summary.get("proposed_median_relative_gain"),
        "primary_claim_check": summary.get("primary_claim_check", {}),
        "strongest_practical_baseline_audit": summary.get("strongest_practical_baseline_audit", {}),
        "latest_refiner_status": refiner_status,
        "paper_quality_design_revision": quality_revision,
    }
    lines = [
        "",
        "",
        "[Phase 2.4 experiment-design feedback from previous Phase 2.5 result verification]",
        "The previous experiment package was not promotable. Use this as design feedback before generating a new validation_plan and code; do not treat it as permission to change the frozen mathematical objective or constraints.",
        "",
        "Controller decision snapshot:",
        json.dumps(snapshot, ensure_ascii=False, indent=2)[:9000],
        "",
        "Redesign rules:",
        *(
            [
                "- Primary KPI mismatch is the root failure. Do not preserve the previous primary y-metric as a final evidence KPI.",
                "- Rebuild research_evidence_contract.primary_metric, objective.metric_name, required scalar metrics, and both final figure y_metrics around the frozen mathematical objective family.",
                "- Use an objective-aligned metric already emitted or naturally computable by the solver. If candidates are listed in the controller snapshot, prefer one of them over inventing a new metric name.",
                "- Keep secondary physical diagnostics only as supporting columns; they must not replace the objective-aligned primary KPI.",
            ]
            if primary_mismatch
            else _phase25_selective_figure_repair_lines(quality_revision)
        ),
        "- Treat previous LLM refiner comments as advisory, not as frozen truth. Follow controller facts, frozen contracts, and current generic policies if a prior refiner note conflicts with them.",
        "- If the issue is only too few x-points in a promising reliable interval, densify within the observed reliable span and keep the same KPI, benchmark set, and figure story.",
        "- If scout/medium results show no positive gain over a valid practical benchmark, do not merely add seeds or delete the benchmark. Redesign Phase 2.4 before code generation: choose an operating regime and x-axis family that directly activates the proposed mechanism and the paper-defined objective-equivalent KPI.",
        "- Re-select or classify benchmarks only for scientific validity, degeneracy, or mechanism mismatch. Do not hide a credible strong benchmark just because it is competitive.",
        "- Keep the first/main y-axis aligned with the frozen paper objective or objective-equivalent KPI. If that metric is an abstract utility, service margin, weighted utility, or tau-style objective, at least one other final figure must expose an interpretable physical/mechanism KPI from the same claim family rather than repeating the abstract objective.",
        "- Make the quick/scout pass cheap through fewer x values, not through noisy Monte Carlo: use enough seeds per selected x value to judge trend and gain.",
        "- Emit concrete schema paths, context_overrides, scout_values, medium_values, and paper suggested_values so Phase 2.5 can execute the tiered plan without inventing a new experiment story.",
    ]
    return "\n".join(lines)


def _phase24_runtime_design_feedback_for_phase24(run_dir: Path) -> str:
    phase24_dir = Path(run_dir) / "phase2-4"
    feedback_paths = sorted(phase24_dir.glob("phase24_runtime_design_feedback_round*.txt"))
    chunks = [read_text(path).strip() for path in feedback_paths if read_text(path).strip()]
    responsiveness = read_json(phase24_dir / "phase24_experiment_responsiveness_check.json") or {}
    if not chunks:
        prior_status = read_json(phase24_dir / "phase24_validation_manifest.json") or {}
        if isinstance(prior_status, dict) and _phase24_should_route_to_runtime_design_repair(prior_status):
            error_path_raw = str(prior_status.get("error_path") or "").strip()
            error_text = read_text(Path(error_path_raw)) if error_path_raw else ""
            chunks.append(
                "\n".join(
                    [
                        "Previous Phase 2.4 quick validation requested experiment-design repair.",
                        "",
                        "Previous validation status:",
                        json.dumps(prior_status, ensure_ascii=False, indent=2),
                        "",
                        "Previous responsiveness check:",
                        json.dumps(responsiveness, ensure_ascii=False, indent=2)[:9000],
                        "",
                        "Previous validation error:",
                        error_text or "(none)",
                        "",
                        "Use this prior failure as active redesign feedback; do not repeat the same nonresponsive figure sweep.",
                    ]
                )
            )
    if not chunks:
        return ""
    axis_directive = _phase24_inactive_axis_directive_for_feedback(
        run_dir=run_dir,
        responsiveness=responsiveness,
        repair_round=_phase24_runtime_design_repair_count(run_dir),
    )
    return "\n\n".join(["", "[Phase 2.4 runtime-design feedback from prior quick validation]", *chunks, axis_directive]).strip()


_PHASE24_INACTIVE_AXIS_PATTERN = re.compile(
    r"(?P<figure_id>figure[\w-]*)[^\n]*?required_sweep=(?P<required_sweep>[A-Za-z0-9_.:-]+)"
    r"[^\n]*?executable path `(?P<x_axis_param>[^`]+)`",
    re.IGNORECASE,
)


def _phase24_unique_axis_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    unique: list[dict[str, Any]] = []
    for record in records:
        key = (
            str(record.get("figure_id") or ""),
            str(record.get("required_sweep") or ""),
            str(record.get("x_axis_param") or ""),
        )
        if key in seen or not any(key):
            continue
        seen.add(key)
        unique.append(record)
    return unique


def _phase24_inactive_axis_records_from_text(text: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for match in _PHASE24_INACTIVE_AXIS_PATTERN.finditer(str(text or "")):
        records.append(
            {
                "figure_id": match.group("figure_id"),
                "required_sweep": match.group("required_sweep"),
                "x_axis_param": match.group("x_axis_param"),
                "source": "runtime_feedback_text",
            }
        )
    return _phase24_unique_axis_records(records)


def _phase24_inactive_axis_records_from_responsiveness(responsiveness: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(responsiveness, dict):
        return []
    records: list[dict[str, Any]] = []
    errors = responsiveness.get("blocking_errors") or responsiveness.get("errors") or []
    if isinstance(errors, list):
        for error in errors:
            records.extend(_phase24_inactive_axis_records_from_text(str(error)))
    checks = responsiveness.get("checks") if isinstance(responsiveness.get("checks"), list) else []
    failed_figures = {str(item.get("figure_id") or "") for item in records if isinstance(item, dict)}
    for check in checks:
        if not isinstance(check, dict):
            continue
        figure_id = str(check.get("figure_id") or "")
        if figure_id and figure_id in failed_figures:
            continue
        if figure_id and failed_figures and figure_id not in failed_figures:
            continue
        relative_span = _phase25_float(check.get("relative_metric_span"), 1.0)
        metric_span = _phase25_float(check.get("metric_span"), 1.0)
        if bool(check.get("sweep_consumption_proven")) and (relative_span <= 1e-6 or metric_span <= 1e-12):
            records.append(
                {
                    "figure_id": figure_id,
                    "required_sweep": check.get("required_sweep", ""),
                    "x_axis_param": str(check.get("actual_used_alias_used") or "").replace("actual_used_", "").replace("_", "."),
                    "metric": check.get("metric", ""),
                    "relative_metric_span": check.get("relative_metric_span"),
                    "source": "responsiveness_check",
                }
            )
    return _phase24_unique_axis_records(records)


def _phase24_ready_figure_records_from_responsiveness(responsiveness: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(responsiveness, dict):
        return []
    failed_ids = {
        str(item.get("figure_id") or "")
        for item in _phase24_inactive_axis_records_from_responsiveness(responsiveness)
        if str(item.get("figure_id") or "")
    }
    ready: list[dict[str, Any]] = []
    checks = responsiveness.get("checks") if isinstance(responsiveness.get("checks"), list) else []
    for check in checks:
        if not isinstance(check, dict):
            continue
        figure_id = str(check.get("figure_id") or "")
        if not figure_id or figure_id in failed_ids:
            continue
        if _phase25_float(check.get("relative_metric_span"), 0.0) <= 1e-6:
            continue
        ready.append(
            {
                "figure_id": figure_id,
                "required_sweep": check.get("required_sweep", ""),
                "metric": check.get("metric", ""),
                "relative_metric_span": check.get("relative_metric_span"),
                "num_x_values": check.get("num_x_values"),
            }
        )
    return _phase24_unique_axis_records(ready)


def _phase24_accumulated_inactive_axis_records(run_dir: Path, responsiveness: dict[str, Any]) -> list[dict[str, Any]]:
    phase24_dir = Path(run_dir) / "phase2-4"
    records = _phase24_inactive_axis_records_from_responsiveness(responsiveness)
    for path in sorted(phase24_dir.glob("phase24_runtime_design_feedback_round*.txt")):
        records.extend(_phase24_inactive_axis_records_from_text(read_text(path)))
    return _phase24_unique_axis_records(records)


def _phase24_inactive_axis_directive_for_feedback(
    *,
    run_dir: Path,
    responsiveness: dict[str, Any],
    repair_round: int,
) -> str:
    current_inactive_records = _phase24_inactive_axis_records_from_responsiveness(responsiveness)
    ready_records = _phase24_ready_figure_records_from_responsiveness(responsiveness)
    ready_sweeps = {str(item.get("required_sweep") or "") for item in ready_records if str(item.get("required_sweep") or "")}
    prior_inactive_records: list[dict[str, Any]] = []
    phase24_dir = Path(run_dir) / "phase2-4"
    for path in sorted(phase24_dir.glob("phase24_runtime_design_feedback_round*.txt")):
        prior_inactive_records.extend(_phase24_inactive_axis_records_from_text(read_text(path)))
    prior_inactive_records = _phase24_unique_axis_records(prior_inactive_records)
    fragile_but_ready_records = [
        item for item in prior_inactive_records if str(item.get("required_sweep") or "") in ready_sweeps
    ]
    hard_forbidden_records = _phase24_unique_axis_records(
        current_inactive_records
        + [
            item
            for item in prior_inactive_records
            if str(item.get("required_sweep") or "") not in ready_sweeps
        ]
    )
    if not hard_forbidden_records and not ready_records and not fragile_but_ready_records:
        return ""
    lines = [
        "",
        "[Controller-enforced figure-axis repair contract]",
    ]
    if hard_forbidden_records:
        lines.extend(
            [
                "The following current failed or still-unresolved final-figure x-axes produced flat/nonresponsive objective KPI after the sweep value was consumed. They are forbidden as final-figure x-axes for the next validation_plan; do not reuse them by only changing context_overrides, point values, or prose.",
                json.dumps(hard_forbidden_records, ensure_ascii=False, indent=2)[:5000],
                "- Replace each failed final figure with a different active model, service-demand, channel-severity, load, power-budget, uncertainty, mobility/deployment, hardware, or algorithm-regime parameter that is present in the frozen contracts/configuration.",
                "- The replacement must keep the frozen objective-equivalent y KPI unless the paper contract explicitly defines a better physical KPI for that figure.",
            ]
        )
    if fragile_but_ready_records:
        lines.extend(
            [
                "These axes failed in an earlier quick validation but are responsive in the latest check. They may be preserved only for the currently ready figure; do not use them to repair another failed figure:",
                json.dumps(fragile_but_ready_records, ensure_ascii=False, indent=2)[:3000],
            ]
        )
    if ready_records:
        lines.extend(
            [
                "The following final figures were responsive in the latest quick validation. Preserve their story and method set unless doing so conflicts with the frozen contracts:",
                json.dumps(ready_records, ensure_ascii=False, indent=2)[:3000],
            ]
        )
    if repair_round >= 2 and hard_forbidden_records:
        lines.append(
            "- Because this is a repeated design-repair round, replacing the inactive axis is mandatory; a plan that keeps the same failed required_sweep/canonical_path is invalid."
        )
    return "\n".join(lines)


def _phase24_runtime_design_repair_count(run_dir: Path) -> int:
    return len(list((Path(run_dir) / "phase2-4").glob("phase24_runtime_design_feedback_round*.txt")))


def _phase24_should_route_to_runtime_design_repair(validation_status: dict[str, Any]) -> bool:
    status = str(validation_status.get("status") or "").strip().lower()
    return bool(validation_status.get("design_repair_recommended")) or status in {
        "experiment_responsiveness_failed",
        "evidence_contract_design_failed",
    }


def _phase24_validation_candidate_score(run_dir: Path, validation_status: dict[str, Any]) -> float:
    phase24_dir = Path(run_dir) / "phase2-4"
    status = str(validation_status.get("status") or "").strip().lower()
    score = 0.0
    if status in {"ok", "passed"}:
        score += 1000.0
    if status in {
        "compile_failed",
        "compile_timeout",
        "interface_failed",
        "missing_outputs",
        "schema_alignment_failed",
        "smoke_failed",
        "smoke_timeout",
        "validation_failed",
        "validation_timeout",
    }:
        score -= 200.0
    for name, weight in [
        ("phase24_codegen_package_check.json", 80.0),
        ("phase24_schema_alignment.json", 80.0),
        ("phase24_evidence_contract_check.json", 100.0),
        ("phase24_method_semantics_check.json", 100.0),
    ]:
        payload = read_json(phase24_dir / name) or {}
        if payload.get("ok") is True:
            score += weight
        elif payload.get("ok") is False:
            score -= weight
    results_path = phase24_dir / "solver" / "outputs" / "validation_results.csv"
    try:
        with results_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except Exception:  # noqa: BLE001
        rows = []
    if not rows:
        return score
    methods = {str(row.get("method") or row.get("method_id") or "").strip().lower() for row in rows}
    figures = {str(row.get("figure_id") or "").strip() for row in rows}
    finite_objective_rows = 0
    finite_physical_rows = 0
    ok_rows = 0
    feasible_rows = 0
    for row in rows:
        if str(row.get("status") or "").strip().lower() == "ok":
            ok_rows += 1
        if str(row.get("feasible") or "").strip().lower() in {"true", "1", "yes"}:
            feasible_rows += 1
        for key in ("objective", "objective_value"):
            try:
                value = float(row.get(key, ""))
            except (TypeError, ValueError):
                continue
            if value == value and value not in {float("inf"), float("-inf")}:
                finite_objective_rows += 1
                break
        for key in ("sum_rate_bpsHz", "sum_rate_bps_hz", "min_user_rate_bpsHz", "rate_bpsHz", "sum_power_W"):
            try:
                value = float(row.get(key, ""))
            except (TypeError, ValueError):
                continue
            if value == value and value not in {float("inf"), float("-inf")}:
                finite_physical_rows += 1
                break
    score += min(len(rows), 100) * 2.0
    score += min(finite_objective_rows, 100) * 3.0
    score += min(finite_physical_rows, 100) * 4.0
    score += min(ok_rows, 100) * 2.0
    score += min(feasible_rows, 100) * 3.0
    if "proposed" in methods:
        score += 80.0
    if len({method for method in methods if method}) >= 2:
        score += 80.0
    if len({figure for figure in figures if figure}) >= 1:
        score += 50.0
    responsiveness = read_json(phase24_dir / "phase24_experiment_responsiveness_check.json") or {}
    responsive_checks = [
        check
        for check in responsiveness.get("checks", [])
        if isinstance(check, dict)
        and float(check.get("relative_metric_span") or 0.0) > 0.01
        and int(check.get("num_x_values") or 0) >= 3
    ]
    if responsive_checks or responsiveness.get("ok") is True:
        score += 100.0
    return score


def _phase24_selected_candidate_can_continue(run_dir: Path, validation_status: dict[str, Any]) -> bool:
    status = str(validation_status.get("status") or "").strip().lower()
    if status == "ok":
        return True
    if status == "pilot_gain_failed":
        return False
    hard_failure_statuses = {
        "compile_failed",
        "compile_timeout",
        "interface_failed",
        "missing_outputs",
        "schema_alignment_failed",
        "smoke_failed",
        "smoke_timeout",
        "validation_failed",
        "validation_timeout",
    }
    if status in hard_failure_statuses:
        return False
    phase24_dir = Path(run_dir) / "phase2-4"
    results_path = phase24_dir / "solver" / "outputs" / "validation_results.csv"
    if not results_path.exists():
        return False
    try:
        with results_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except Exception:  # noqa: BLE001
        return False
    if not rows:
        return False
    methods = {str(row.get("method") or row.get("method_id") or "").strip() for row in rows}
    if "proposed" not in {method.lower() for method in methods} or len({m for m in methods if m}) < 2:
        return False

    sweep_keys: set[tuple[str, str]] = set()
    for row in rows:
        figure_id = str(row.get("figure_id") or "").strip()
        sweep_id = str(row.get("sweep_id") or row.get("sweep_name") or "").strip()
        swept_param = str(row.get("swept_param") or row.get("swept_parameter") or row.get("sweep_parameter") or "").strip()
        swept_value = str(row.get("swept_value") or row.get("sweep_value") or "").strip()
        if figure_id:
            sweep_keys.add(("figure", figure_id))
        if sweep_id and sweep_id.lower() not in {"canonical", "baseline", "default"}:
            sweep_keys.add((sweep_id, swept_value))
        if swept_param and swept_param.lower() not in {"canonical", "baseline", "default"}:
            sweep_keys.add((swept_param, swept_value))
    if len(sweep_keys) < 2:
        return False

    seeds = {str(row.get("seed") or "").strip() for row in rows if str(row.get("seed") or "").strip()}
    required_seed_floor = max(2, min(10, int(os.environ.get("WARA_PHASE24_PILOT_MIN_PAIRED_SEEDS", "20") or 20)))
    if len(seeds) < required_seed_floor:
        return False
    pilot_gain = read_json(phase24_dir / "phase24_pilot_gain_check.json") or {}
    if pilot_gain.get("ok") is not True:
        return False
    finite_objective_rows = 0
    finite_physical_rows = 0
    for row in rows:
        for key in ("objective", "objective_value"):
            try:
                value = float(row.get(key, ""))
            except (TypeError, ValueError):
                continue
            if value == value and abs(value) < float("inf"):
                finite_objective_rows += 1
                break
        for key in ("sum_rate_bpsHz", "sum_rate_bps_hz", "min_user_rate_bpsHz", "rate_bpsHz", "sum_power_W"):
            try:
                value = float(row.get(key, ""))
            except (TypeError, ValueError):
                continue
            if value == value and value not in {float("inf"), float("-inf")}:
                finite_physical_rows += 1
                break
    return finite_objective_rows > 0 and finite_physical_rows > 0


def _phase24_mark_selected_candidate_validation(
    run_dir: Path,
    validation_status: dict[str, Any],
) -> dict[str, Any]:
    phase24_dir = Path(run_dir) / "phase2-4"
    selected_status = dict(validation_status)
    selected_status["original_status"] = validation_status.get("status")
    selected_status["status"] = "selected_after_bounded_repairs"
    selected_status["selected_after_bounded_repairs"] = True
    selected_status["selection_score"] = _phase24_validation_candidate_score(run_dir, validation_status)
    selected_status["reason"] = (
        "Phase 2.4 reached the repair limit, so the controller selected the strongest "
        "executable candidate and forwarded it to Phase 2.5 for evidence packaging. "
        "The final PDF must still be produced by the normal Phase 3 writing and review chain."
    )
    write_text(phase24_dir / "phase24_selected_candidate_note.md", selected_status["reason"])
    write_text(phase24_dir / "phase24_validation_manifest.json", json.dumps(selected_status, ensure_ascii=False, indent=2))
    return selected_status


def _phase24_write_runtime_design_feedback(
    *,
    run_dir: Path,
    validation_status: dict[str, Any],
    repair_round: int,
    repair_round_limit: int,
) -> None:
    phase24_dir = Path(run_dir) / "phase2-4"
    error_path_raw = str(validation_status.get("error_path") or "").strip()
    error_text = read_text(Path(error_path_raw)) if error_path_raw else ""
    responsiveness = read_json(phase24_dir / "phase24_experiment_responsiveness_check.json") or {}
    payload = [
        f"Retry round {repair_round}/{repair_round_limit}.",
        "The generated experiment code executed, but quick validation found an experiment-design problem rather than a syntax/interface bug.",
        "",
        "Validation status:",
        json.dumps(validation_status, ensure_ascii=False, indent=2),
        "",
        "Responsiveness check:",
        json.dumps(responsiveness, ensure_ascii=False, indent=2)[:9000],
        "",
        "Validation error:",
        error_text or "(none)",
        _phase24_inactive_axis_directive_for_feedback(
            run_dir=run_dir,
            responsiveness=responsiveness,
            repair_round=repair_round,
        ),
        "",
        "Repair instructions for the next Phase 2.4 validation_plan:",
        "- Keep the frozen mathematical objective, constraints, method ids, and objective-equivalent primary KPI.",
        "- Do not fabricate variation in code and do not weaken constraints.",
        "- If a required figure has a flat/nonresponsive KPI despite consumed x-values, replace that figure's x-axis or context_overrides with an objective-driver operating regime where the KPI is physically active.",
        "- If the same y-metric is used in both final figures, make the two x-axes probe distinct mechanisms or operating regimes; do not keep a service-threshold sweep that the quick validation already showed is inactive.",
        "- Preserve a fair practical benchmark set across final figures. Do not delete a valid benchmark merely because it is competitive.",
        "- Provide scout_values, medium_values, and paper values for the revised sweep so Phase 2.5 can densify without changing the story.",
    ]
    write_text(phase24_dir / f"phase24_runtime_design_feedback_round{repair_round}.txt", "\n".join(payload))


def _run_phase25_auto_paper_expansion(
    run_dir: Path,
    *,
    paper_target: str,
    topic: str = "",
    model_profile: str = DEFAULT_MODEL_PROFILE,
    initial_result: dict[str, Any],
    initial_status: str,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    """Run the Phase25-requested paper sweep when continuing from Phase 2.4."""

    limit = _phase25_auto_paper_run_limit()
    phase24_dir = run_dir / "phase2-4"
    phase25_dir = run_dir / "phase2-5"
    phase24_dir.mkdir(parents=True, exist_ok=True)
    phase25_dir.mkdir(parents=True, exist_ok=True)
    current_result = initial_result
    current_status = initial_status
    manifest: dict[str, Any] = {
        "enabled": limit > 0,
        "round_limit": limit,
        "sweep_mode": _phase25_auto_paper_sweep_mode(),
        "redesign_round_limit": _phase25_experiment_redesign_limit(),
        "coverage_extension_round_limit": _phase25_coverage_extension_limit(),
        "initial_phase25_status": initial_status,
        "rounds": [],
        "redesigns": [],
        "coverage_extensions": [],
        "final_phase25_status": initial_status,
        "reason": "",
        "source": "continue_phase2_from_phase24",
    }

    if limit <= 0:
        manifest["reason"] = "disabled_by_WARA_PHASE25_AUTO_PAPER_RUNS"
    elif not _phase25_should_auto_expand(run_dir, current_status):
        manifest["reason"] = "paper_sweep_plan_not_needed_or_missing"
    else:
        alignment_precheck = _phase25_objective_metric_alignment_precheck(run_dir)
        if bool(alignment_precheck.get("requires_phase24_design_revision")):
            current_status = "requires_phase24_design_revision"
            manifest["reason"] = str(alignment_precheck.get("reason") or "primary_metric_family_mismatch_with_frozen_objective")
            manifest["redesigns"].append(
                {
                    "round": 0,
                    "trigger": "deterministic_objective_metric_alignment_precheck",
                    "counts_against_redesign_limit": False,
                    "ok": False,
                    "status": current_status,
                    "precheck": alignment_precheck,
                }
            )
            manifest["final_phase25_status"] = current_status
            write_json_artifact(phase25_dir / "phase25_auto_expansion_manifest.json", manifest)
            return current_result, current_status, manifest
        initial_snapshot = _phase25_claim_snapshot(run_dir)
        manifest["initial_claim_snapshot"] = initial_snapshot
        if _phase25_can_skip_auto_expansion(current_status, initial_snapshot):
            manifest["reason"] = "auto_expansion_skipped_by_policy"
            manifest["final_phase25_status"] = current_status
            write_json_artifact(phase25_dir / "phase25_auto_expansion_manifest.json", manifest)
            return current_result, current_status, manifest
        redesign_round_limit = _phase25_experiment_redesign_limit()
        coverage_extension_limit = _phase25_coverage_extension_limit()
        effective_limit = limit + redesign_round_limit + coverage_extension_limit
        manifest["effective_round_limit"] = effective_limit
        redesign_rounds = 0
        coverage_extension_rounds = 0
        try:
            refined = _phase25_refine_sweep_plan(
                run_dir,
                topic=topic,
                model_profile=model_profile,
                reason="initial_quick_results_before_auto_expansion",
                round_index=0,
            )
            manifest["redesigns"].append(
                {
                    "round": 0,
                    "trigger": "initial_quick_results_before_auto_expansion",
                    "counts_against_redesign_limit": False,
                    "ok": bool(refined.get("figures")),
                    "status": refined.get("status", ""),
                }
            )
            if _phase25_refiner_requires_phase24_design_revision(refined):
                if _phase25_defer_initial_refiner_redesign_until_sweep(run_dir, current_status, refined):
                    manifest["redesigns"][-1]["deferred_until_after_tiered_sweep"] = True
                    manifest["redesigns"][-1]["defer_reason"] = (
                        "initial quick data are too sparse; deterministic Phase 2.4 design checks passed, "
                        "so scout/medium evidence should run before redesign routing"
                    )
                else:
                    current_status = "requires_phase24_design_revision"
                    manifest["reason"] = "phase25_refiner_requested_phase24_design_revision_before_auto_expansion"
                    manifest["final_phase25_status"] = current_status
                    write_json_artifact(phase25_dir / "phase25_auto_expansion_manifest.json", manifest)
                    return current_result, current_status, manifest
        except Exception as exc:  # noqa: BLE001
            manifest["redesigns"].append(
                {
                    "round": 0,
                    "trigger": "initial_quick_results_before_auto_expansion",
                    "counts_against_redesign_limit": False,
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
        if bool(initial_snapshot.get("runtime_failure")):
            manifest["reason"] = "implementation_runtime_too_slow_before_auto_expansion"
            manifest["rounds"].append(
                {
                    "round": 0,
                    "mode": "precheck",
                    "phase25_status_after": current_status,
                    "paper_ready_after": False,
                    "claim_snapshot": initial_snapshot,
                }
            )
            manifest["final_phase25_status"] = current_status
            write_json_artifact(phase25_dir / "phase25_auto_expansion_manifest.json", manifest)
            return current_result, current_status, manifest
        next_mode_override = ""
        last_nonpaper_promising = True
        for round_index in range(1, effective_limit + 1):
            if not _phase25_should_auto_expand(run_dir, current_status):
                break
            sweep_mode = next_mode_override or _phase25_sweep_mode_for_round(round_index)
            next_mode_override = ""
            if sweep_mode == "paper" and not last_nonpaper_promising:
                manifest["reason"] = "paper_sweep_skipped_after_unpromising_scout_or_medium"
                break
            try:
                sweep_result = _run_phase25_sweep_with_tier(run_dir, sweep_mode)
                write_json_artifact(phase24_dir / f"phase25_auto_paper_sweep_round{round_index}.json", sweep_result)
            except Exception as exc:  # noqa: BLE001 - preserve continuation state
                sweep_result = {
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "validation_output_prefix": _phase25_validation_prefix_for_mode(sweep_mode),
                }
                write_json_artifact(phase24_dir / f"phase25_auto_paper_sweep_round{round_index}_error.json", sweep_result)
                manifest["rounds"].append(
                    {
                        "round": round_index,
                        "mode": sweep_mode,
                        "phase25_status_after": current_status,
                        "paper_ready_after": False,
                        "sweep_result": sweep_result,
                    }
                )
                manifest["reason"] = f"auto_expansion_failed:{type(exc).__name__}"
                break
            try:
                current_result = impl.run_phase25_wcl_package(run_dir, paper_target=paper_target)
                current_status = _phase25_status_from_result(run_dir, current_result)
            except Exception as exc:  # noqa: BLE001 - preserve continuation state
                reanalysis_error = {
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "phase": "phase25_reanalysis_after_paper_sweep",
                }
                write_json_artifact(phase24_dir / f"phase25_auto_paper_reanalysis_round{round_index}_error.json", reanalysis_error)
                manifest["rounds"].append(
                    {
                        "round": round_index,
                        "mode": sweep_mode,
                        "phase25_status_after": current_status,
                        "paper_ready_after": False,
                        "sweep_result": sweep_result,
                        "reanalysis_error": reanalysis_error,
                    }
                )
                manifest["reason"] = f"auto_reanalysis_failed:{type(exc).__name__}"
                break
            manifest["rounds"].append(
                {
                    "round": round_index,
                    "mode": sweep_mode,
                    "phase25_status_after": current_status,
                    "paper_ready_after": _phase25_is_paper_ready(current_status),
                    "sweep_result": sweep_result,
                    "claim_snapshot": _phase25_claim_snapshot(run_dir),
                }
            )
            if _phase25_is_paper_ready(current_status):
                break
            snapshot = _phase25_claim_snapshot(run_dir)
            if bool(snapshot.get("runtime_failure")):
                manifest["reason"] = f"implementation_runtime_too_slow_after_{sweep_mode}"
                break
            promising = _phase25_claim_promising(snapshot)
            if (
                sweep_mode == "paper"
                and promising
                and str(current_status or "") == "needs_more_phase24_runs"
                and _phase25_should_auto_expand(run_dir, current_status)
            ):
                design_revision = _phase25_paper_quality_requires_phase24_design_revision(run_dir)
                if bool(design_revision.get("requires_phase24_design_revision")):
                    current_status = "requires_phase24_design_revision"
                    manifest["reason"] = str(design_revision.get("reason") or "paper_quality_requires_phase24_design_revision")
                    manifest["redesigns"].append(
                        {
                            "round": round_index,
                            "trigger": "paper_quality_noncoverage_blockers",
                            "counts_against_redesign_limit": False,
                            "ok": False,
                            "status": current_status,
                            "claim_snapshot": snapshot,
                            "quality_revision": design_revision,
                        }
                    )
                    break
                if coverage_extension_rounds < coverage_extension_limit:
                    coverage_extension_rounds += 1
                    manifest["coverage_extensions"].append(
                        {
                            "round": round_index,
                            "trigger": "paper_claim_promising_but_more_coverage_needed",
                            "counts_against_coverage_extension_limit": True,
                            "claim_snapshot": snapshot,
                            "next_mode": "paper",
                        }
                    )
                    next_mode_override = "paper"
                    continue
                manifest["reason"] = "coverage_extension_limit_reached_after_promising_paper_sweep"
                break
            if sweep_mode in {"scout", "medium"}:
                last_nonpaper_promising = promising
            if sweep_mode in {"scout", "medium"} and not promising:
                if redesign_rounds < redesign_round_limit:
                    try:
                        refined = _phase25_refine_sweep_plan(
                            run_dir,
                            topic=topic,
                            model_profile=model_profile,
                            reason=f"{sweep_mode}_claim_not_promising",
                            round_index=round_index,
                        )
                        redesign_rounds += 1
                        manifest["redesigns"].append(
                            {
                                "round": round_index,
                                "trigger": f"{sweep_mode}_claim_not_promising",
                                "counts_against_redesign_limit": True,
                                "ok": bool(refined.get("figures")),
                                "status": refined.get("status", ""),
                                "claim_snapshot": snapshot,
                            }
                        )
                        if _phase25_refiner_requires_phase24_design_revision(refined):
                            current_status = "requires_phase24_design_revision"
                            manifest["reason"] = f"phase25_refiner_requested_phase24_design_revision_after_{sweep_mode}"
                            break
                        next_mode_override = "scout" if sweep_mode == "scout" else "medium"
                        continue
                    except Exception as exc:  # noqa: BLE001
                        manifest["redesigns"].append(
                            {
                                "round": round_index,
                                "trigger": f"{sweep_mode}_claim_not_promising",
                                "counts_against_redesign_limit": True,
                                "ok": False,
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                                "claim_snapshot": snapshot,
                            }
                        )
                current_status = "requires_phase24_design_revision"
                manifest["reason"] = f"requires_phase24_design_revision_after_unpromising_{sweep_mode}"
                break
        if not str(manifest.get("reason") or "").startswith(("auto_expansion_failed", "auto_reanalysis_failed")):
            manifest["reason"] = (
                "paper_ready_after_auto_expansion"
                if _phase25_is_paper_ready(current_status)
                else (manifest.get("reason") or "auto_expansion_limit_reached_or_no_paper_ready_plan")
            )

    manifest["final_phase25_status"] = current_status
    write_json_artifact(phase25_dir / "phase25_auto_expansion_manifest.json", manifest)
    return current_result, current_status, manifest


def _phase25_should_route_to_impl_repair(status: str, auto_expansion: dict[str, Any], run_dir: Path | None = None) -> bool:
    if _phase25_is_paper_ready(status):
        return False
    reason = str((auto_expansion or {}).get("reason") or "")
    rounds = (auto_expansion or {}).get("rounds")
    snapshots = [item.get("claim_snapshot") for item in rounds if isinstance(item, dict) and isinstance(item.get("claim_snapshot"), dict)] if isinstance(rounds, list) else []
    if run_dir is not None:
        summary = read_json(Path(run_dir) / "phase2-5" / "phase25_experiment_summary.json") or {}
        if isinstance(summary, dict) and summary:
            primary = summary.get("primary_claim_check") if isinstance(summary.get("primary_claim_check"), dict) else {}
            strongest = summary.get("strongest_practical_baseline_audit") if isinstance(summary.get("strongest_practical_baseline_audit"), dict) else {}
            snapshots.append(
                {
                    "phase25_status": str(summary.get("phase25_status") or ""),
                    "num_comparable_cases": int(_phase25_float(summary.get("num_comparable_cases"), 0.0)),
                    "primary_passes": bool(primary.get("passes", False)),
                    "strongest_practical_passes": bool(strongest.get("passes", primary.get("passes", False))) if strongest else bool(primary.get("passes", False)),
                    "primary_reason": str(primary.get("reason") or ""),
                    "strongest_reason": str(strongest.get("reason") or ""),
                    "baseline_degenerate": bool(primary.get("baseline_degenerate") or strongest.get("baseline_degenerate")),
                    **_phase25_runtime_snapshot(Path(run_dir)),
                }
            )
    if not reason.startswith("promotion_failed_after_") and str(status or "") != "claim_failure_needs_redesign":
        if not any(
            bool(item.get("baseline_degenerate")) or bool(item.get("runtime_failure"))
            for item in snapshots
            if isinstance(item, dict)
        ):
            return False
    if not snapshots:
        return False
    latest = snapshots[-1]
    return (
        int(latest.get("num_comparable_cases") or 0) > 0
        and (
            bool(latest.get("baseline_degenerate"))
            or bool(latest.get("runtime_failure"))
            or not bool(latest.get("strongest_practical_passes", latest.get("primary_passes", False)))
        )
    )


def _phase25_claim_failure_repair_text(run_dir: Path, auto_expansion: dict[str, Any]) -> str:
    phase25_dir = run_dir / "phase2-5"
    parts = [
        "Phase 2.5 could not promote the experiment to paper-ready after sweep redesign.",
        "Treat this as an implementation-quality repair request, not as permission to weaken the frozen problem, delete benchmarks, or fabricate gains.",
        "Repair generated_experiment_core.py so the proposed method and the practical benchmarks are both faithful, non-degenerate, and fair on the paper-defined KPI.",
        "Allowed repairs: improve proposed-method scaling, line search, projection, surrogate/majorizer step size, initialization, monotone acceptance, truthful diagnostics, practical benchmark initialization/loading, benchmark selection among already-declared practical methods, and computational efficiency without weakening the mathematical model.",
        "If the strongest practical benchmark has near-zero KPI while the proposed method is normal, treat it as a benchmark implementation/selection bug, not as evidence of a huge gain.",
        "If scout or medium sweeps report excessive median solve time, simplify/vectorize the implementation, reduce unnecessary inner loops, cache reusable channel/model quantities, and keep the solver iteration budget paper-practical.",
        "Forbidden repairs: changing the mathematical objective, changing the frozen constraints, relabeling a benchmark as proposed, deleting the strongest practical benchmark without replacing it by another declared practical benchmark, or hard-coding favorable outputs.",
        "",
        "Runtime snapshot:",
        json.dumps(_phase25_runtime_snapshot(run_dir), ensure_ascii=False, indent=2),
        "",
        "Latest Phase 2.5 summary:",
        read_text(phase25_dir / "phase25_experiment_summary.json")[:6000],
        "",
        "Auto-expansion manifest:",
        json.dumps(auto_expansion or {}, ensure_ascii=False, indent=2)[:8000],
        "",
        "Missing/quality notes:",
        read_text(phase25_dir / "missing_experiments.md")[:3000],
    ]
    return "\n".join(parts)


def _repair_phase24_after_phase25_claim_failure(
    *,
    run_dir: Path,
    topic: str,
    model_profile: str,
    phase25_status: str,
    auto_expansion: dict[str, Any],
) -> bool:
    phase24_dir = run_dir / "phase2-4"
    solver_dir = phase24_dir / "solver"
    limit = _phase25_impl_repair_round_limit()
    attempt_limit = _phase25_impl_repair_attempt_limit(limit)
    existing_rounds = sorted(phase24_dir.glob("phase25_claim_failure_repair_round*.json"))
    applied_rounds = [
        path
        for path in existing_rounds
        if bool((read_json(path) or {}).get("applied_to_generated_plugin"))
    ]
    if limit <= 0 or len(applied_rounds) >= limit or len(existing_rounds) >= attempt_limit:
        return False
    repair_round = len(existing_rounds) + 1
    current_plugin_path = solver_dir / "generated_plugin.py"
    original_prompt_path = phase24_dir / "phase24_generated_plugin_prompt.txt"
    if not current_plugin_path.exists() or not original_prompt_path.exists():
        return False
    repair_status = {
        "status": "phase25_claim_failure",
        "phase25_status": phase25_status,
        "auto_expansion_reason": str((auto_expansion or {}).get("reason") or ""),
        "repair_round": repair_round,
        "repair_round_limit": limit,
        "repair_attempt_limit": attempt_limit,
        "attempted_repair_rounds_before": len(existing_rounds),
        "applied_repair_rounds_before": len(applied_rounds),
        "applied_to_generated_plugin": False,
    }
    write_json_artifact(phase24_dir / f"phase25_claim_failure_repair_round{repair_round}.json", repair_status)
    try:
        repaired_adapter = impl.repair_phase2_phase24_plugin_llm(
            run_dir=run_dir,
            topic=topic,
            original_prompt=read_text(original_prompt_path),
            current_plugin_code=read_text(current_plugin_path),
            validation_status=repair_status,
            validation_error_text=_phase25_claim_failure_repair_text(run_dir, auto_expansion),
            model_profile=model_profile,
        )
    except Exception as exc:  # noqa: BLE001 - preserve failed repair attempt in artifacts
        repair_status["error_type"] = type(exc).__name__
        repair_status["error"] = str(exc)
        write_json_artifact(phase24_dir / f"phase25_claim_failure_repair_round{repair_round}.json", repair_status)
        raise
    write_text(phase24_dir / f"phase24_generated_plugin_phase25_claim_repair_round{repair_round}.py", repaired_adapter)
    write_text(current_plugin_path, repaired_adapter)
    repair_status["applied_to_generated_plugin"] = True
    write_json_artifact(phase24_dir / f"phase25_claim_failure_repair_round{repair_round}.json", repair_status)
    validation_status = impl.validate_phase2_phase24_plugin_bundle(run_dir)
    validation_status["repair_attempted"] = True
    validation_status["repair_reason"] = "phase25_claim_failure"
    validation_status["phase25_claim_failure_repair_round"] = repair_round
    write_text(phase24_dir / "phase24_validation_manifest.json", json.dumps(validation_status, ensure_ascii=False, indent=2))
    return validation_status.get("status") == "ok"


def _load_required_mathematical_contract_json(run_dir: Path) -> str:
    """Load the frozen Phase 2.1 math contract before any Phase 2.4 continuation."""

    phase1_dir = run_dir / "phase2-1"
    checked_paths = [
        phase1_dir / "mathematical_contract.frozen.json",
        phase1_dir / "mathematical_contract.json",
    ]
    malformed_errors: list[str] = []
    for path in checked_paths:
        raw = read_text(path).strip()
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            malformed_errors.append(f"{path.name}: {exc}")
            continue
        if not isinstance(payload, dict):
            malformed_errors.append(f"{path.name}: top-level value is not an object")
            continue

        controls = payload.get("controls")
        objective = payload.get("objective")
        constraints = payload.get("constraints")
        if controls and objective and constraints:
            return json.dumps(payload, ensure_ascii=False, indent=2)
        malformed_errors.append(
            f"{path.name}: missing required controls/objective/constraints fields"
        )

    detail = "; ".join(malformed_errors) if malformed_errors else "no contract file with content was found"
    raise ValueError(
        "Phase 2.4 continuation requires a non-empty Phase 2.1 mathematical_contract. "
        "Refusing to run phase2.4 LLM prompts with an empty math interface. "
        f"Checked: {', '.join(str(path) for path in checked_paths)}. Details: {detail}"
    )


def _write_phase24_source_contracts(
    *,
    phase24_dir: Path,
    mathematical_contract_json: str,
    phase24_execution_contract: dict[str, Any],
    wireless_benchmark_plan: dict[str, Any],
    experiment_design_contract: dict[str, Any],
    claim_map: dict[str, Any],
) -> None:
    try:
        mathematical_contract = json.loads(mathematical_contract_json or "{}")
    except json.JSONDecodeError:
        mathematical_contract = {}
    write_json_artifact(
        phase24_dir / "phase24_validation_source_contracts.json",
        {
            "mathematical_contract": mathematical_contract,
            "phase24_execution_contract": phase24_execution_contract,
            "wireless_benchmark_plan": wireless_benchmark_plan,
            "experiment_design_contract": experiment_design_contract,
            "claim_map": claim_map,
        },
    )


def continue_from_phase24(
    run_dir: Path,
    paper_target: str = "IEEE WCL",
    reuse_validation_plan: bool = False,
    repair_existing_plugin: bool = False,
    stop_phase: int = 5,
    model_profile_override: str | None = None,
) -> None:
    run_dir = Path(run_dir).resolve()
    summary = read_json(run_dir / "phase2_summary.json") or {}
    if not summary:
        raise FileNotFoundError(f"phase2_summary.json not found under {run_dir}")
    if stop_phase < 4 or stop_phase > 5:
        raise ValueError(
            "Phase 2 continuation can only run Phase 2.4 and Phase 2.5. "
            "Use phase3/scripts/run_phase3_pipeline.py for Phase 3 writing and review."
        )

    topic = str(summary.get("topic") or run_dir.name)
    model_profile = str(model_profile_override or summary.get("model_profile") or DEFAULT_MODEL_PROFILE)
    if model_profile_override:
        summary["model_profile"] = model_profile
        _write_summary(run_dir, summary)
    phase1_run = Path(str(summary.get("phase1_run") or "")) if summary.get("phase1_run") else None
    handoff = read_json(run_dir / "phase1_handoff_manifest.json") or {}
    if not handoff and phase1_run and phase1_run.exists():
        handoff = impl.build_phase1_handoff(phase1_run, run_dir)
        write_text(run_dir / "phase1_handoff_manifest.json", json.dumps(handoff, ensure_ascii=False, indent=2))

    phase1_outputs = {
        "mathematical_contract_json": _load_required_mathematical_contract_json(run_dir),
        "system_model_md": read_text(run_dir / "phase2-1" / "system_model.md"),
        "problem_formulation_md": read_text(run_dir / "phase2-1" / "problem_formulation.md"),
        "core_theory_package_md": read_text(run_dir / "phase2-1" / "core_theory_package.md"),
    }
    mathematical_contract_json = str(phase1_outputs["mathematical_contract_json"])
    phase2_outputs = {
        "convexity_audit_md": read_text(run_dir / "phase2-2" / "convexity_audit.md"),
        "reformulation_path_md": read_text(run_dir / "phase2-2" / "reformulation_path.md"),
    }
    phase3_outputs = {
        "algorithm_md": read_text(run_dir / "phase2-3" / "algorithm.md"),
        "convergence_or_complexity_md": read_text(run_dir / "phase2-3" / "convergence_or_complexity.md"),
    }

    try:
        for completed_phase in range(1, 4):
            _set_phase_status(summary, completed_phase, "done")
        _set_phase_status(summary, 4, "running")
        for phase in range(5, 12):
            _set_phase_status(summary, phase, "ready")
        _write_summary(run_dir, summary)

        phase24_dir = run_dir / "phase2-4"
        solver_dir = phase24_dir / "solver"
        solver_dir.mkdir(parents=True, exist_ok=True)
        existing_validation_plan = phase24_dir / "validation_plan.yaml"
        problem_contract = read_json(run_dir / "phase2-1" / "problem_contract.json") or {}
        algorithm_contract = read_json(run_dir / "phase2-2" / "algorithm_contract.json") or {}
        claim_map = read_json(run_dir / "phase2-3" / "claim_map.json") or {}
        if problem_contract and algorithm_contract:
            refreshed_claim_map = build_claim_map(
                topic=topic,
                problem_contract=problem_contract,
                algorithm_contract=algorithm_contract,
                algorithm_md=read_text(run_dir / "phase2-3" / "algorithm.md"),
                convergence_or_complexity_md=read_text(run_dir / "phase2-3" / "convergence_or_complexity.md"),
                experiment_blueprint_md=read_text(run_dir / "phase2-3" / "experiment_blueprint.md"),
            )
            existing_metric = ""
            refreshed_metric = ""
            if isinstance(claim_map.get("claims"), list) and claim_map.get("claims"):
                existing_metric = str((claim_map.get("claims") or [{}])[0].get("metric") or "").strip()
            if isinstance(refreshed_claim_map.get("claims"), list) and refreshed_claim_map.get("claims"):
                refreshed_metric = str((refreshed_claim_map.get("claims") or [{}])[0].get("metric") or "").strip()
            if refreshed_metric and refreshed_metric != existing_metric:
                write_json_artifact(run_dir / "phase2-3" / "claim_map.previous_before_objective_alignment.json", claim_map)
                claim_map = refreshed_claim_map
                write_json_artifact(run_dir / "phase2-3" / "claim_map.json", claim_map)
        wireless_benchmark_plan = read_json(phase24_dir / "wireless_benchmark_plan.json") or {}
        experiment_design_contract = read_json(phase24_dir / "experiment_design_contract.json") or {}
        if not wireless_benchmark_plan and problem_contract and algorithm_contract and claim_map:
            wireless_benchmark_plan = select_wireless_benchmark_plan(
                topic=topic,
                problem_contract=problem_contract,
                algorithm_contract=algorithm_contract,
                claim_map=claim_map,
            )
            write_json_artifact(phase24_dir / "wireless_benchmark_plan.json", wireless_benchmark_plan)
        if problem_contract and wireless_benchmark_plan and claim_map:
            refreshed_experiment_design_contract = build_experiment_design_contract(
                problem_contract=problem_contract,
                benchmark_plan=wireless_benchmark_plan,
                claim_map=claim_map,
            )
            existing_figure_metric = ""
            refreshed_figure_metric = ""
            if isinstance(experiment_design_contract.get("figure_contracts"), list) and experiment_design_contract.get("figure_contracts"):
                existing_figure_metric = str((experiment_design_contract.get("figure_contracts") or [{}])[0].get("y_metric") or "").strip()
            if isinstance(refreshed_experiment_design_contract.get("figure_contracts"), list) and refreshed_experiment_design_contract.get("figure_contracts"):
                refreshed_figure_metric = str((refreshed_experiment_design_contract.get("figure_contracts") or [{}])[0].get("y_metric") or "").strip()
            if (not experiment_design_contract) or (refreshed_figure_metric and refreshed_figure_metric != existing_figure_metric):
                if experiment_design_contract:
                    write_json_artifact(phase24_dir / "experiment_design_contract.previous_before_objective_alignment.json", experiment_design_contract)
                experiment_design_contract = refreshed_experiment_design_contract
                write_json_artifact(phase24_dir / "experiment_design_contract.json", experiment_design_contract)
        phase24_evidence_contract_summary = ""
        benchmark_definition_for_phase24 = ""
        if wireless_benchmark_plan and experiment_design_contract:
            phase24_evidence_contract_summary = (
                "[Phase 2.4-owned evidence contract]\n"
                "Use the structured Phase 2.4 contracts below as "
                "the source of truth for claims, compared methods, sweeps, metrics, figure targets, and table targets.\n"
                + contract_prompt_block(
                    benchmark_plan=wireless_benchmark_plan,
                    experiment_design_contract=experiment_design_contract,
                )
            )
            benchmark_definition_for_phase24 = (
                "[Phase 2.4 WirelessBenchmarkAgent benchmark contract]\n"
                + json.dumps(wireless_benchmark_plan, ensure_ascii=False, indent=2)
            )
        phase24_redesign_feedback = ""
        if not reuse_validation_plan and not repair_existing_plugin:
            phase24_redesign_feedback = _phase25_design_revision_feedback_for_phase24(run_dir)
            phase24_runtime_feedback = _phase24_runtime_design_feedback_for_phase24(run_dir)
            if bool(_phase25_objective_metric_alignment_precheck(run_dir).get("requires_phase24_design_revision")):
                # Runtime flat-axis feedback is stale when it was measured on a
                # primary KPI that the controller has just rejected as
                # objective-family mismatched. The next Phase 2.4 design prompt should
                # first rebuild the objective-aligned KPI and only then test
                # active axes for that new metric.
                phase24_runtime_feedback = ""
            if phase24_redesign_feedback.strip():
                write_text(phase24_dir / "phase24_design_feedback_from_phase25.txt", phase24_redesign_feedback)
            if phase24_runtime_feedback.strip():
                write_text(phase24_dir / "phase24_runtime_design_feedback_active.txt", phase24_runtime_feedback)
            # Put the immediate runtime-design contract first so prompt compaction
            # cannot bury controller-forbidden axes behind long Phase 2.5 snapshots.
            phase24_redesign_feedback = phase24_runtime_feedback + phase24_redesign_feedback
        if repair_existing_plugin:
            if not existing_validation_plan.exists():
                raise FileNotFoundError(f"Cannot repair existing plugin without {existing_validation_plan}")
            phase24_validation_yaml = impl.normalize_phase24_validation_plan_yaml(read_text(existing_validation_plan))
            phase24_benchmark = {
                "benchmark_plan_md": read_text(phase24_dir / "benchmark_plan.md"),
                "solver_readme_md": read_text(solver_dir / "README.md"),
            }
        else:
            if reuse_validation_plan and existing_validation_plan.exists():
                phase24_validation_yaml = impl.normalize_phase24_validation_plan_yaml(read_text(existing_validation_plan))
            else:
                phase24_validation_yaml = impl.run_phase2_phase24_validation_llm(
                    run_dir=run_dir,
                    topic=topic,
                    handoff=handoff or {},
                    mathematical_contract_json=mathematical_contract_json,
                    system_model_md=phase1_outputs["system_model_md"],
                    problem_formulation_md=phase1_outputs["problem_formulation_md"],
                    convexity_audit_md=phase2_outputs["convexity_audit_md"],
                    reformulation_path_md=phase2_outputs["reformulation_path_md"],
                    experiment_blueprint_md=phase24_evidence_contract_summary + phase24_redesign_feedback,
                    model_profile=model_profile,
                )
            if reuse_validation_plan and (phase24_dir / "benchmark_plan.md").exists() and (solver_dir / "README.md").exists():
                phase24_benchmark = {
                    "benchmark_plan_md": read_text(phase24_dir / "benchmark_plan.md"),
                    "solver_readme_md": read_text(solver_dir / "README.md"),
                }
            else:
                phase24_benchmark = impl.run_phase2_phase24_benchmark_llm(
                    run_dir=run_dir,
                    topic=topic,
                    handoff=handoff or {},
                    mathematical_contract_json=mathematical_contract_json,
                    system_model_md=phase1_outputs["system_model_md"],
                    problem_formulation_md=phase1_outputs["problem_formulation_md"],
                    benchmark_definition_md=benchmark_definition_for_phase24 + phase24_redesign_feedback,
                    model_profile=model_profile,
                )
        _write_text_if_changed(phase24_dir / "validation_plan.yaml", phase24_validation_yaml)
        _write_text_if_changed(solver_dir / "validation_plan.yaml", phase24_validation_yaml)
        if problem_contract and algorithm_contract and wireless_benchmark_plan:
            validation_plan_payload = impl._phase24_yaml_mapping(phase24_validation_yaml)
            if not isinstance(validation_plan_payload, dict):
                raise ValueError("Phase 2.4 validation plan must parse to a mapping before code generation")
            phase24_execution_contract = build_phase24_execution_contract(
                validation_plan=validation_plan_payload,
                problem_contract=problem_contract,
                algorithm_contract=algorithm_contract,
                benchmark_plan=wireless_benchmark_plan,
            )
            write_json_artifact(phase24_dir / "phase24_execution_contract.json", phase24_execution_contract)
        phase24_execution_contract = read_json(phase24_dir / "phase24_execution_contract.json") or {}
        if phase24_execution_contract:
            _write_phase24_source_contracts(
                phase24_dir=phase24_dir,
                mathematical_contract_json=mathematical_contract_json,
                phase24_execution_contract=phase24_execution_contract,
                wireless_benchmark_plan=wireless_benchmark_plan,
                experiment_design_contract=experiment_design_contract,
                claim_map=claim_map,
            )
        evidence_design_status = impl.validate_phase24_evidence_contract_design(run_dir)
        write_text(
            phase24_dir / "phase24_evidence_contract_design_check.json",
            json.dumps(evidence_design_status, ensure_ascii=False, indent=2),
        )
        if (not evidence_design_status.get("ok")) and (not reuse_validation_plan) and (not repair_existing_plugin):
            design_repair_round_limit = _phase24_design_repair_round_limit()
            for design_round in range(1, design_repair_round_limit + 1):
                previous_errors = "\n".join(str(item) for item in evidence_design_status.get("errors", []))
                previous_warnings = "\n".join(str(item) for item in evidence_design_status.get("warnings", []))
                design_feedback = (
                    "\n\n[Phase 2.4 design-gate retry feedback]\n"
                    f"Retry round {design_round}/{design_repair_round_limit}. The previous validation plan failed the experiment-design gate.\n"
                    f"Errors:\n{previous_errors or '(none)'}\n"
                    f"Warnings:\n{previous_warnings or '(none)'}\n\n"
                    "Revise the validation plan instead of bypassing the gate. Preserve the frozen mathematical model, "
                    "objective semantics, method ids, and benchmark fairness. If the objective is scalarized or weighted "
                    "(for example utility_U_alpha), do not make every final figure use that objective-like y_metric; use "
                    "decomposed physical KPIs tied to the claim, such as sum_rate_bpsHz, min_user_rate_bpsHz, sensing_snr_dB, "
                    "sensing_snr_linear, CRB/MSE, harvested power, reliability, or energy efficiency when those columns are available. "
                    "Every final figure must have a paper-facing axis_labels.x and axis_labels.y with concise physical meaning plus "
                    "public notation, a chart_choice_rationale, an expected_trend/trend_hypothesis, and an active_regime_note. "
                    "Do not use internal schema paths, bare symbols, feasibility diagnostics, or transmit-power minimization unless "
                    "those are the frozen objective or claim-critical KPI."
                )
                write_text(phase24_dir / f"phase24_design_retry_feedback_round{design_round}.txt", design_feedback)
                phase24_validation_yaml = impl.run_phase2_phase24_validation_llm(
                    run_dir=run_dir,
                    topic=topic,
                    handoff=handoff or {},
                    mathematical_contract_json=mathematical_contract_json,
                    system_model_md=phase1_outputs["system_model_md"],
                    problem_formulation_md=phase1_outputs["problem_formulation_md"],
                    convexity_audit_md=phase2_outputs["convexity_audit_md"],
                    reformulation_path_md=phase2_outputs["reformulation_path_md"],
                    experiment_blueprint_md=phase24_evidence_contract_summary + phase24_redesign_feedback + design_feedback,
                    model_profile=model_profile,
                )
                write_text(phase24_dir / f"validation_plan_design_retry_round{design_round}.yaml", phase24_validation_yaml)
                write_text(phase24_dir / "validation_plan.yaml", phase24_validation_yaml)
                write_text(solver_dir / "validation_plan.yaml", phase24_validation_yaml)
                if problem_contract and algorithm_contract and wireless_benchmark_plan:
                    validation_plan_payload = impl._phase24_yaml_mapping(phase24_validation_yaml)
                    if not isinstance(validation_plan_payload, dict):
                        raise ValueError("Phase 2.4 validation plan must parse to a mapping before code generation")
                    phase24_execution_contract = build_phase24_execution_contract(
                        validation_plan=validation_plan_payload,
                        problem_contract=problem_contract,
                        algorithm_contract=algorithm_contract,
                        benchmark_plan=wireless_benchmark_plan,
                    )
                    write_json_artifact(phase24_dir / "phase24_execution_contract.json", phase24_execution_contract)
                phase24_execution_contract = read_json(phase24_dir / "phase24_execution_contract.json") or {}
                if phase24_execution_contract:
                    _write_phase24_source_contracts(
                        phase24_dir=phase24_dir,
                        mathematical_contract_json=mathematical_contract_json,
                        phase24_execution_contract=phase24_execution_contract,
                        wireless_benchmark_plan=wireless_benchmark_plan,
                        experiment_design_contract=experiment_design_contract,
                        claim_map=claim_map,
                    )
                evidence_design_status = impl.validate_phase24_evidence_contract_design(run_dir)
                evidence_design_status["design_retry_round"] = design_round
                evidence_design_status["design_repair_round_limit"] = design_repair_round_limit
                write_text(
                    phase24_dir / "phase24_evidence_contract_design_check.json",
                    json.dumps(evidence_design_status, ensure_ascii=False, indent=2),
                )
                write_text(
                    phase24_dir / f"phase24_evidence_contract_design_check_round{design_round}.json",
                    json.dumps(evidence_design_status, ensure_ascii=False, indent=2),
                )
                if evidence_design_status.get("ok"):
                    break
        if not evidence_design_status.get("ok"):
            error_path = phase24_dir / "phase24_validation_error.txt"
            write_text(error_path, "[research_evidence_contract_design]\n" + "\n".join(evidence_design_status.get("errors", [])))
            write_text(
                phase24_dir / "phase24_validation_manifest.json",
                json.dumps(
                    {
                        "status": "evidence_contract_design_failed",
                        "returncode": 1,
                        "error_path": str(error_path),
                        "repair_attempted": int(evidence_design_status.get("design_repair_round_limit", 0) or 0) > 0,
                        "design_repair_rounds": int(evidence_design_status.get("design_retry_round", 0) or 0),
                        "design_repair_round_limit": int(evidence_design_status.get("design_repair_round_limit", 0) or 0),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            )
            return {
                "run_dir": str(run_dir),
                "status": "blocked",
                "blocked_phase": 4,
                "reason": "phase2.4 validation_plan.yaml failed evidence-contract design validation before code generation.",
                "error_path": str(error_path),
            }
        write_text(phase24_dir / "benchmark_plan.md", phase24_benchmark["benchmark_plan_md"])
        write_text(solver_dir / "README.md", phase24_benchmark["solver_readme_md"])
        write_text(phase24_dir / "phase24_design_notes.md", impl.build_phase24_design_notes())
        write_text(
            phase24_dir / "phase24_harness_manifest.json",
            json.dumps(
                {
                    "mode": "fixed_harness_plugin",
                    "fixed_files": ["problem_data.py", "validation_cases.py", "run_validation.py"],
                    "generated_file": "generated_plugin.py",
                    "plugin_exports": ["build_model", "initial_state", "proposed_step", "baseline_solution", "evaluate_state"],
                },
                ensure_ascii=False,
                indent=2,
            ),
        )

        if not repair_existing_plugin:
            for stale in solver_dir.glob("*.py"):
                stale.unlink()
        impl.write_phase2_phase24_fixed_harness(run_dir)
        if repair_existing_plugin:
            generated_plugin_path = solver_dir / "generated_plugin.py"
            if not generated_plugin_path.exists():
                raise FileNotFoundError(f"Cannot repair existing plugin without {generated_plugin_path}")
            generated_core_path = solver_dir / "generated_experiment_core.py"
            if generated_core_path.exists():
                normalized_core = impl.normalize_phase24_generated_plugin_source(read_text(generated_core_path))
                _write_text_if_changed(generated_core_path, normalized_core)
                _write_text_if_changed(phase24_dir / "phase24_generated_experiment_core.py", normalized_core)
            generated_plugin = read_text(generated_plugin_path)
        else:
            generated_plugin = impl.run_phase2_phase24_plugin_llm(
                run_dir=run_dir,
                topic=topic,
                mathematical_contract_json=mathematical_contract_json,
                system_model_md=phase1_outputs["system_model_md"],
                problem_formulation_md=phase1_outputs["problem_formulation_md"],
                reformulation_path_md=phase2_outputs["reformulation_path_md"],
                algorithm_md=phase3_outputs["algorithm_md"],
                benchmark_definition_md=benchmark_definition_for_phase24 + phase24_redesign_feedback,
                experiment_blueprint_md=phase24_evidence_contract_summary + phase24_redesign_feedback,
                model_profile=model_profile,
            )
            write_text(solver_dir / "generated_plugin.py", generated_plugin)

        validation_status = impl.validate_phase2_phase24_plugin_bundle(run_dir)
        validation_status["repair_attempted"] = False
        repair_rounds = 0
        original_prompt = read_text(phase24_dir / "phase24_generated_plugin_prompt.txt")
        current_plugin_code = generated_plugin
        repair_round_limit = _phase24_repair_round_limit()

        phase24_candidates_dir = phase24_dir / "bounded_repair_candidates"
        phase24_candidates_dir.mkdir(parents=True, exist_ok=True)
        phase24_candidate_records: list[dict[str, Any]] = []

        def _record_phase24_candidate(label: str, plugin_code: str, status: dict[str, Any]) -> None:
            score = _phase24_validation_candidate_score(run_dir, status)
            plugin_path = phase24_candidates_dir / f"{label}.py"
            status_path = phase24_candidates_dir / f"{label}.json"
            record = {
                "label": label,
                "score": score,
                "status": status.get("status"),
                "plugin_path": str(plugin_path),
                "status_path": str(status_path),
            }
            write_text(plugin_path, plugin_code)
            write_text(status_path, json.dumps({**status, "selection_score": score}, ensure_ascii=False, indent=2))
            phase24_candidate_records.append(record)

        _record_phase24_candidate("initial", current_plugin_code, validation_status)
        while impl._phase24_validation_allows_repair(validation_status) and repair_rounds < repair_round_limit:
            validation_error_text = impl._phase24_validation_error_text(run_dir, validation_status)
            try:
                current_plugin_code = impl.repair_phase2_phase24_plugin_llm(
                    run_dir=run_dir,
                    topic=topic,
                    original_prompt=original_prompt,
                    current_plugin_code=current_plugin_code,
                    validation_status=validation_status,
                    validation_error_text=validation_error_text,
                    model_profile=model_profile,
                )
            except Exception as exc:  # noqa: BLE001 - select from already-validated candidates below.
                validation_status["repair_exception_type"] = type(exc).__name__
                validation_status["repair_exception"] = str(exc)
                break
            repair_rounds += 1
            write_text(solver_dir / "generated_plugin.py", current_plugin_code)
            write_text(phase24_dir / f"phase24_generated_plugin_repaired_round{repair_rounds}.py", current_plugin_code)
            validation_status = impl.validate_phase2_phase24_plugin_bundle(run_dir)
            validation_status["repair_attempted"] = True
            validation_status["repair_rounds"] = repair_rounds
            validation_status["repair_round_limit"] = repair_round_limit
            _record_phase24_candidate(f"repair_round_{repair_rounds}", current_plugin_code, validation_status)

        validation_status.setdefault("repair_round_limit", repair_round_limit)
        write_text(phase24_dir / "phase24_validation_manifest.json", json.dumps(validation_status, ensure_ascii=False, indent=2))
        if validation_status.get("status") != "ok":
            if phase24_candidate_records:
                selected_record = max(phase24_candidate_records, key=lambda item: float(item.get("score") or 0.0))
                selected_plugin = read_text(Path(str(selected_record.get("plugin_path"))))
                if selected_plugin and selected_plugin != current_plugin_code:
                    current_plugin_code = selected_plugin
                    write_text(solver_dir / "generated_plugin.py", current_plugin_code)
                    validation_status = impl.validate_phase2_phase24_plugin_bundle(run_dir)
                    validation_status["repair_attempted"] = repair_rounds > 0
                    validation_status["repair_rounds"] = repair_rounds
                    validation_status["repair_round_limit"] = repair_round_limit
                validation_status["selected_candidate"] = selected_record
                write_text(
                    phase24_dir / "phase24_selected_candidate.json",
                    json.dumps({"selected": selected_record, "candidates": phase24_candidate_records}, ensure_ascii=False, indent=2),
                )
            runtime_design_repair_round_limit = _phase24_design_repair_round_limit()
            runtime_design_repair_count = _phase24_runtime_design_repair_count(run_dir)
            if (
                (not reuse_validation_plan)
                and (not repair_existing_plugin)
                and _phase24_should_route_to_runtime_design_repair(validation_status)
                and runtime_design_repair_count < runtime_design_repair_round_limit
            ):
                runtime_design_repair_round = runtime_design_repair_count + 1
                _phase24_write_runtime_design_feedback(
                    run_dir=run_dir,
                    validation_status=validation_status,
                    repair_round=runtime_design_repair_round,
                    repair_round_limit=runtime_design_repair_round_limit,
                )
                return continue_from_phase24(
                    run_dir=run_dir,
                    paper_target=paper_target,
                    reuse_validation_plan=False,
                    repair_existing_plugin=False,
                    stop_phase=stop_phase,
                    model_profile_override=model_profile,
                )
            if _phase24_selected_candidate_can_continue(run_dir, validation_status):
                validation_status = _phase24_mark_selected_candidate_validation(run_dir, validation_status)
                write_text(
                    phase24_dir / "phase24_selected_candidate_status.json",
                    json.dumps(validation_status, ensure_ascii=False, indent=2),
                )
            else:
                _set_phase_status(summary, 4, "blocked")
                _write_summary(run_dir, summary)
                raise RuntimeError(f"phase2.4 validation did not pass: {validation_status}")

        _set_phase_status(summary, 4, "done")
        _write_summary(run_dir, summary)

        runners = {
            5: impl.run_phase25_wcl_package,
        }
        for phase, runner in runners.items():
            if phase > stop_phase:
                _set_phase_status(summary, phase, "ready")
                _write_summary(run_dir, summary)
                continue
            _set_phase_status(summary, phase, "running")
            _write_summary(run_dir, summary)
            phase_result = runner(run_dir, paper_target=paper_target)
            if phase == 5:
                phase25_status = _phase25_status_from_result(
                    run_dir,
                    phase_result if isinstance(phase_result, dict) else {},
                )
                phase_result, phase25_status, _phase25_auto_expansion = _run_phase25_auto_paper_expansion(
                    run_dir,
                    paper_target=paper_target,
                    topic=topic,
                    model_profile=model_profile,
                    initial_result=phase_result if isinstance(phase_result, dict) else {},
                    initial_status=phase25_status,
                )
                if (
                    not _phase25_is_paper_ready(phase25_status)
                    and _phase25_should_route_to_impl_repair(phase25_status, _phase25_auto_expansion, run_dir)
                    and _repair_phase24_after_phase25_claim_failure(
                        run_dir=run_dir,
                        topic=topic,
                        model_profile=model_profile,
                        phase25_status=phase25_status,
                        auto_expansion=_phase25_auto_expansion,
                    )
                ):
                    phase_result = impl.run_phase25_wcl_package(run_dir, paper_target=paper_target)
                    phase25_status = _phase25_status_from_result(
                        run_dir,
                        phase_result if isinstance(phase_result, dict) else {},
                    )
                    phase_result, phase25_status, _phase25_auto_expansion = _run_phase25_auto_paper_expansion(
                        run_dir,
                        paper_target=paper_target,
                        topic=topic,
                        model_profile=model_profile,
                        initial_result=phase_result if isinstance(phase_result, dict) else {},
                        initial_status=phase25_status,
                    )
                allow_draft_continue = str(os.environ.get("WCL_ALLOW_DRAFT_PHASE25_CONTINUE", "")).strip().lower() in {
                    "1",
                    "true",
                    "yes",
                }
                bounded_budget_continuation = _phase25_bounded_budget_continuation_enabled()
                if not _phase25_is_paper_ready(phase25_status) and not allow_draft_continue and not bounded_budget_continuation:
                    write_json_artifact(
                        run_dir / "phase2-5" / "phase25_quality_gate.json",
                        {
                            "status": "blocked",
                            "phase25_status": phase25_status,
                            "auto_expansion": _phase25_auto_expansion,
                            "allowed_statuses": sorted(PAPER_READY_PHASE25_STATUSES),
                            "reason": "phase2.5 evidence is not paper-ready; stop before drafting final paper sections.",
                        },
                    )
                    _set_phase_status(summary, phase, "blocked")
                    _write_summary(run_dir, summary)
                    raise RuntimeError(f"phase2.5 experiment package is not paper-ready: {phase25_status}")
                write_json_artifact(
                    run_dir / "phase2-5" / "phase25_quality_gate.json",
                    {
                        "status": "passed",
                        "phase25_status": phase25_status,
                        "auto_expansion": _phase25_auto_expansion,
                        "allowed_statuses": sorted(PAPER_READY_PHASE25_STATUSES),
                        "paper_ready": _phase25_is_paper_ready(phase25_status),
                        "bounded_budget_continuation": bounded_budget_continuation
                        and not _phase25_is_paper_ready(phase25_status),
                        "draft_continuation": allow_draft_continue and not _phase25_is_paper_ready(phase25_status),
                        "reason": "phase2.5 produced paper-ready experimental evidence."
                        if _phase25_is_paper_ready(phase25_status)
                        else (
                            "Phase 2.5 completed its bounded expansion budget; the controller is continuing "
                            "with the selected experimental evidence package and preserving the evidence-scope "
                            "metadata for Phase 3."
                            if bounded_budget_continuation
                            else "Draft phase2.5 continuation was explicitly allowed by environment."
                        ),
                    },
                )
                _write_phase2_to_phase3_handoff(
                    run_dir,
                    phase25_gate_ok=True,
                    phase25_status=phase25_status,
                    evidence_audit=read_json(run_dir / "phase2-5" / "evidence_audit.json") or {},
                    phase25_auto_expansion=_phase25_auto_expansion,
                )
            _set_phase_status(summary, phase, "done")
            _write_summary(run_dir, summary)
            if phase == stop_phase:
                break
    except Exception as exc:
        current = next(
            (
                phase_num
                for item in summary.get("phases", [])
                if isinstance(item, dict) and item.get("status") == "running"
                for phase_num in [_phase_number(item)]
                if phase_num is not None
            ),
            None,
        )
        if current is None:
            current = next(
                (
                    phase_num
                    for item in summary.get("phases", [])
                    if isinstance(item, dict)
                    and item.get("status") == "blocked"
                    for phase_num in [_phase_number(item)]
                    if phase_num is not None
                    and phase_num >= 5
                ),
                None,
            )
        if current is None:
            current = next(
                (
                    phase_num
                    for item in summary.get("phases", [])
                    if isinstance(item, dict) and item.get("status") == "blocked"
                    for phase_num in [_phase_number(item)]
                    if phase_num is not None
                ),
                4,
            )
        _set_phase_status(summary, current, "blocked")
        _write_summary(run_dir, summary)
        error_path = run_dir / _phase_step_dir_name(current) / f"phase_step_{current}_continue_error.txt"
        write_text(error_path, "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Continue Phase 2 from phase2.4 through the frozen Phase 2.5 handoff.")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--paper-target", default="IEEE WCL")
    parser.add_argument("--reuse-validation-plan", action="store_true", help="Reuse and normalize the existing phase2.4 validation_plan.yaml instead of regenerating it.")
    parser.add_argument("--repair-existing-plugin", action="store_true", help="Validate, repair, and continue from the existing generated_plugin.py without regenerating the phase2.4 plan or plugin first.")
    parser.add_argument("--stop-phase", type=int, default=5, help="Stop after this numeric Phase 2 step; valid values are 4 or 5.")
    parser.add_argument("--model-profile", default="", help="Override the model profile recorded in phase2_summary.json for this continuation run.")
    args = parser.parse_args()
    with _RunLock(Path(args.run_dir).resolve(), "continue_phase2_from_phase24"):
        continue_from_phase24(
            args.run_dir,
            paper_target=args.paper_target,
            reuse_validation_plan=args.reuse_validation_plan,
            repair_existing_plugin=args.repair_existing_plugin,
            stop_phase=args.stop_phase,
            model_profile_override=args.model_profile or None,
        )


if __name__ == "__main__":
    main()
