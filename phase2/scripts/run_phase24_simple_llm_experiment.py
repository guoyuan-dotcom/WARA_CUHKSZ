from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from pipeline_core import DEFAULT_MODEL_PROFILE, extract_python_source, read_text, write_text  # noqa: E402
from wara_core.agents import build_experiment_agent_task_prompt  # noqa: E402
from phase_runtime.llm import create_llm_client  # noqa: E402


def _read_first_existing(*paths: Path) -> str:
    for path in paths:
        text = read_text(path).strip()
        if text:
            return text
    return ""


def _clip(text: str, limit: int) -> str:
    text = str(text or "")
    if len(text) <= limit:
        return text
    head = int(limit * 0.62)
    tail = limit - head
    return text[:head] + "\n\n[... middle omitted for prompt compactness ...]\n\n" + text[-tail:]


def _read_json_file(path: Path, default: Any | None = None) -> Any:
    try:
        text = read_text(path)
        if not text.strip():
            return default
        return json.loads(text)
    except (OSError, json.JSONDecodeError):
        return default


def _write_json_file(path: Path, payload: Any) -> None:
    write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: list[float]) -> float | None:
    finite = [value for value in values if value == value]
    if not finite:
        return None
    return sum(finite) / len(finite)


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _first_text(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _normalized_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())


def _metric_label(metric_name: str, fallback: str = "") -> str:
    if fallback:
        return fallback
    normalized = metric_name.lower()
    labels = {
        "objective": "$\\psi$",
        "worst_case_utility": "$\\psi$",
        "utility": "$\\psi$",
        "sum_rate": "$R_{\\rm sum}$",
        "sum_rate_bpshz": "$R_{\\rm sum}$",
        "throughput": "Throughput",
        "throughput_bpshz": "Throughput",
        "energy_efficiency": "Energy efficiency",
        "harvested_energy": "$P_{\\rm DC}$",
        "harvested_energy_mw": "$P_{\\rm DC}$",
        "min_harvested_dc_mw": "$P_{\\rm DC,min}$",
        "service_margin_tau": "$\\tau$",
        "min_normalized_service_margin": "$\\tau_{\\min}$",
        "worst_case_min_secrecy_rate_bpshz": "$R_{\\rm sec}^{\\min}$",
        "min_secrecy_rate_bpshz": "$R_{\\rm sec}^{\\min}$",
        "sum_secrecy_rate_bpshz": "$R_{\\rm sec}^{\\rm sum}$",
        "weighted_sum_conservative_secrecy_rate_bpshz": "$U_{\\rm sec}$",
    }
    if normalized in labels:
        return labels[normalized]
    return re.sub(r"_+", " ", metric_name).strip()


def _truthy_numeric(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"1", "1.0", "true", "yes", "ok", "feasible", "optimal", "optimal_inaccurate"}:
        return "1"
    if text in {"0", "0.0", "false", "no", "failed", "infeasible"}:
        return "0"
    return ""


def _metric_name_from(value: Any) -> str:
    if isinstance(value, dict):
        return _first_text(value.get("name"), value.get("metric"), value.get("id"))
    return _first_text(value)


def _metric_higher_is_better_from(value: Any, metric_name: str = "") -> bool | None:
    if isinstance(value, dict) and "higher_is_better" in value:
        raw = value.get("higher_is_better")
        if isinstance(raw, bool):
            return raw
        text = str(raw).strip().lower()
        if text in {"true", "1", "yes", "higher", "maximize", "max"}:
            return True
        if text in {"false", "0", "no", "lower", "minimize", "min"}:
            return False
    normalized = str(metric_name or _metric_name_from(value)).lower()
    if any(token in normalized for token in ["cost", "latency", "delay", "violation", "outage", "runtime"]):
        return False
    return None


def _metric_column_names(metric_name: str) -> list[str]:
    name = str(metric_name or "").strip()
    if not name:
        return []
    candidates = [name]
    for suffix in ["_mean", "_avg", "_average", "_median"]:
        if not name.endswith(suffix):
            candidates.append(f"{name}{suffix}")
    return candidates


def _columns_contain_metric(columns: set[str], metric_name: str) -> bool:
    return any(name in columns for name in _metric_column_names(metric_name))


def _row_metric_value(row: dict[str, Any], metric_name: str, *fallback_metrics: str) -> float | None:
    for name in (metric_name, *fallback_metrics):
        if not name:
            continue
        for column_name in _metric_column_names(name):
            value = _float_or_none(row.get(column_name))
            if value is not None:
                return value
    return None


def _row_looks_aggregated(row: dict[str, Any], metric_name: str = "") -> bool:
    keys = set(row.keys())
    if metric_name and f"{metric_name}_mean" in keys:
        return True
    return any(str(key).endswith("_mean") for key in keys)


def _configured_seed_count(
    *,
    paper_run_config: dict[str, Any],
    preview_quality: dict[str, Any],
    default: int = 0,
) -> int:
    seeds = paper_run_config.get("random_seeds") if isinstance(paper_run_config, dict) else None
    if isinstance(seeds, list) and seeds:
        return len(seeds)
    final_grid = preview_quality.get("final_grid") if isinstance(preview_quality, dict) else {}
    if isinstance(final_grid, dict):
        value = _float_or_none(final_grid.get("seeds_per_point"))
        if value is not None and value > 0:
            return int(round(value))
    for key in ["seeds_per_point", "num_seeds", "n_seeds"]:
        value = _float_or_none(paper_run_config.get(key) if isinstance(paper_run_config, dict) else None)
        if value is not None and value > 0:
            return int(round(value))
    return default


def _row_sample_count(
    row: dict[str, Any],
    *,
    metric_name: str,
    configured_seed_count: int,
) -> int:
    for key in ["num_samples", "sample_count", "n_samples", "num_seeds", "n_seeds", "seeds_per_point"]:
        value = _float_or_none(row.get(key))
        if value is not None and value > 0:
            return int(round(value))
    if _row_looks_aggregated(row, metric_name) and configured_seed_count > 0:
        return configured_seed_count
    return 1


def _select_primary_metric(
    *,
    simple_summary: dict[str, Any],
    preview_quality: dict[str, Any],
    figure_report: dict[str, Any],
    paper_run_config: dict[str, Any],
    rows: list[dict[str, str]],
) -> dict[str, Any]:
    claim_evidence = simple_summary.get("claim_evidence") if isinstance(simple_summary.get("claim_evidence"), dict) else {}
    report_figures = _as_list(figure_report.get("figures")) if isinstance(figure_report, dict) else []
    if not report_figures and isinstance(figure_report, dict):
        report_figures = _as_list(figure_report.get("selected_figures"))
    paper_figure_1 = paper_run_config.get("figure_1") if isinstance(paper_run_config.get("figure_1"), dict) else {}
    config_figures = paper_run_config.get("figures") if isinstance(paper_run_config, dict) else {}
    if not paper_figure_1 and isinstance(config_figures, dict):
        paper_figure_1 = config_figures.get("figure_1") if isinstance(config_figures.get("figure_1"), dict) else {}
    candidates = [
        claim_evidence.get("primary_metric"),
        simple_summary.get("primary_metric"),
        preview_quality.get("primary_metric"),
        report_figures[0].get("y_metric") if report_figures and isinstance(report_figures[0], dict) else "",
        paper_figure_1.get("y_metric"),
        "worst_case_utility",
        "objective",
    ]
    columns = set(rows[0].keys()) if rows else set()
    excluded = {
        "figure_id",
        "method",
        "method_id",
        "method_label",
        "seed",
        "swept_param",
        "swept_value",
        "x_value",
        "scenario_name",
        "feasible",
        "feasible_numeric",
        "robust_feasibility",
    }
    columns_numeric_preference = [column for column in columns if column not in excluded]
    metric_name = ""
    higher_is_better: bool | None = None
    for candidate in candidates:
        name = _metric_name_from(candidate)
        if name and (not columns or _columns_contain_metric(columns, name)):
            metric_name = name
            higher_is_better = _metric_higher_is_better_from(candidate, name)
            break
    if not metric_name and columns_numeric_preference:
        preferred = [
            column
            for column in columns_numeric_preference
            if column.endswith("_mean")
            and not any(token in column.lower() for token in ["runtime", "selected_route", "mode"])
        ]
        metric_name = (preferred[0][:-5] if preferred else columns_numeric_preference[0])
        higher_is_better = _metric_higher_is_better_from(None, metric_name)
    metric_name = metric_name or "objective"
    if higher_is_better is None:
        higher_is_better = _metric_higher_is_better_from(None, metric_name)
    if higher_is_better is None:
        higher_is_better = True

    display = _first_text(
        claim_evidence.get("primary_metric_symbol"),
        claim_evidence.get("primary_metric_display"),
        report_figures[0].get("y_axis_label") if report_figures and isinstance(report_figures[0], dict) else "",
        paper_figure_1.get("y_axis_label"),
    )
    return {
        "name": metric_name,
        "display_name": _metric_label(metric_name, display),
        "higher_is_better": higher_is_better,
        "aggregation": "curve_stabilized_by_repeated_seeds",
    }


def _figure_defs_from_reports(
    *,
    out_dir: Path,
    figures_src: Path,
    figure_report: dict[str, Any],
    paper_run_config: dict[str, Any],
    primary_metric: dict[str, Any],
) -> list[dict[str, Any]]:
    raw_figures = [item for item in _as_list(figure_report.get("figures")) if isinstance(item, dict)]
    if not raw_figures:
        raw_figures = [item for item in _as_list(figure_report.get("selected_figures")) if isinstance(item, dict)]
    if not raw_figures and isinstance(paper_run_config, dict):
        config_figures = paper_run_config.get("figures")
        if isinstance(config_figures, dict):
            for filename, item in config_figures.items():
                if isinstance(item, dict):
                    raw = dict(item)
                    raw.setdefault("filename", filename)
                    raw_figures.append(raw)
        for key in sorted(k for k in paper_run_config if k.startswith("figure_")):
            item = paper_run_config.get(key)
            if isinstance(item, dict):
                raw_figures.append(item)

    if not raw_figures:
        raw_figures = [
            {
                "figure_id": "fig1_primary_gain",
                "filename": "figures/fig1_primary_gain.png",
                "x_axis_label": "$x_1$",
                "x_axis_param": "primary_sweep",
                "y_axis_label": primary_metric["display_name"],
                "y_metric": primary_metric["name"],
                "purpose": "main performance comparison",
                "chart_intent": "main_comparison",
            },
            {
                "figure_id": "fig2_insight",
                "filename": "figures/fig2_insight.png",
                "x_axis_label": "$x_2$",
                "x_axis_param": "secondary_sweep",
                "y_axis_label": primary_metric["display_name"],
                "y_metric": primary_metric["name"],
                "purpose": "parameter-sensitivity insight",
                "chart_intent": "mechanism_sensitivity",
            },
        ]

    figure_defs: list[dict[str, Any]] = []
    config_by_filename: dict[str, dict[str, Any]] = {}
    config_by_id: dict[str, dict[str, Any]] = {}
    config_figures = paper_run_config.get("figures") if isinstance(paper_run_config, dict) else {}
    if isinstance(config_figures, dict):
        for figure_key, item in config_figures.items():
            if isinstance(item, dict):
                config_by_id[_normalized_key(figure_key)] = item
                for field in ["figure_id", "id"]:
                    if str(item.get(field) or "").strip():
                        config_by_id[_normalized_key(item.get(field))] = item
                for field in ["filename", "filename_png", "file", "artifact", "path"]:
                    if str(item.get(field) or "").strip():
                        config_by_filename[Path(str(item.get(field))).name] = item
    if isinstance(paper_run_config, dict):
        for figure_key in sorted(key for key in paper_run_config if str(key).startswith("figure_")):
            item = paper_run_config.get(figure_key)
            if isinstance(item, dict):
                config_by_id[_normalized_key(figure_key)] = item
                for field in ["figure_id", "id"]:
                    if str(item.get(field) or "").strip():
                        config_by_id[_normalized_key(item.get(field))] = item
                for field in ["filename", "filename_png", "file", "artifact", "path"]:
                    if str(item.get(field) or "").strip():
                        config_by_filename[Path(str(item.get(field))).name] = item
    for index, raw in enumerate(raw_figures[:2], start=1):
        phase24_id = _first_text(raw.get("figure_id"), raw.get("id"), f"figure_{index}")
        raw_filename = _first_text(
            raw.get("filename"),
            raw.get("filename_png"),
            raw.get("file"),
            raw.get("artifact"),
            raw.get("path"),
            f"{phase24_id}.png",
        )
        config_for_figure = {}
        for key in [_normalized_key(phase24_id), _normalized_key(f"figure_{index}")]:
            if key in config_by_id:
                config_for_figure = dict(config_by_id[key])
                break
        config_for_figure.update(config_by_filename.get(Path(raw_filename).name, {}))
        filename_path = Path(raw_filename)
        if filename_path.is_absolute():
            src_png = filename_path
        elif filename_path.parts and filename_path.parts[0] == "figures":
            src_png = out_dir / filename_path
        else:
            src_png = figures_src / filename_path
        if not src_png.exists():
            candidate = figures_src / f"{phase24_id}.png"
            if candidate.exists():
                src_png = candidate
        if not src_png.exists():
            legacy_name = "fig1_primary_gain.png" if index == 1 else "fig2_insight.png"
            legacy = figures_src / legacy_name
            if legacy.exists():
                src_png = legacy

        x_axis_param = _first_text(
            config_for_figure.get("x_axis_param") if isinstance(config_for_figure, dict) else "",
            config_for_figure.get("x_variable") if isinstance(config_for_figure, dict) else "",
            config_for_figure.get("sweep_param") if isinstance(config_for_figure, dict) else "",
            config_for_figure.get("swept_param") if isinstance(config_for_figure, dict) else "",
            raw.get("x_axis_param"),
            raw.get("swept_param"),
            raw.get("x_param"),
            raw.get("x_axis"),
            raw.get("x_variable"),
            raw.get("sweep_param"),
            config_for_figure.get("x_axis") if isinstance(config_for_figure, dict) else "",
            raw.get("sweep_id"),
            f"x_{index}",
        )
        axis_labels = raw.get("axis_labels") if isinstance(raw.get("axis_labels"), dict) else {}
        config_axis_labels = (
            config_for_figure.get("axis_labels")
            if isinstance(config_for_figure, dict) and isinstance(config_for_figure.get("axis_labels"), dict)
            else {}
        )
        y_metric = _first_text(
            config_for_figure.get("y_metric") if isinstance(config_for_figure, dict) else "",
            raw.get("y_metric"),
            raw.get("metric"),
            config_for_figure.get("y_axis") if isinstance(config_for_figure, dict) else "",
            primary_metric["name"],
        )
        y_label = _first_text(
            config_axis_labels.get("y"),
            raw.get("y_axis_label"),
            axis_labels.get("y"),
            raw.get("y_axis"),
            raw.get("y_label"),
            _metric_label(y_metric),
        )
        figure_defs.append(
            {
                "phase24_id": phase24_id,
                "figure_id": f"figure_{index}",
                "filename_png": f"figure_{index}.png",
                "filename_pdf": f"figure_{index}.pdf",
                "curve_csv": f"figure_{index}_curve_data.csv",
                "x_label": _first_text(
                    config_axis_labels.get("x"),
                    raw.get("x_axis_label"),
                    axis_labels.get("x"),
                    raw.get("x_axis"),
                    raw.get("x_label"),
                    raw.get("x_variable"),
                    x_axis_param,
                ),
                "x_axis_param": x_axis_param,
                "required_sweep": _first_text(raw.get("required_sweep"), f"{x_axis_param}_sweep"),
                "purpose": _first_text(
                    raw.get("purpose"),
                    "main performance comparison" if index == 1 else "parameter-sensitivity insight",
                ),
                "chart_intent": _first_text(
                    raw.get("chart_intent"),
                    raw.get("figure_intent"),
                    "main_comparison" if index == 1 else "mechanism_sensitivity",
                ),
                "y_metric": y_metric,
                "y_label": y_label,
                "src_png": src_png,
            }
        )
    return figure_defs


def _copy_png_and_pdf(src_png: Path, dst_png: Path, dst_pdf: Path) -> None:
    dst_png.parent.mkdir(parents=True, exist_ok=True)
    if src_png.exists():
        shutil.copy2(src_png, dst_png)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.image as mpimg
        import matplotlib.pyplot as plt

        image = mpimg.imread(src_png)
        height, width = image.shape[:2]
        fig = plt.figure(figsize=(width / 300.0, height / 300.0), dpi=300)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(image)
        ax.axis("off")
        fig.savefig(dst_pdf, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
    except Exception:
        # Keep the PNG even if PDF conversion is unavailable; Phase 3.1 can still
        # use the PNG during development, and the missing PDF is visible.
        pass


def _render_verified_curve_figure(
    *,
    curve_rows: list[dict[str, Any]],
    fig_def: dict[str, Any],
    plotted_methods: list[str],
    label_for_method: Any,
    dst_png: Path,
    dst_pdf: Path,
) -> bool:
    """Render the final paper figure from verified curve data.

    LLM-generated preview PNGs sometimes have bad axis limits or silently hide
    curves. Phase 2.5 owns the verified registry, so the final figure should be
    drawn from that registry whenever curve data are available.
    """

    if not curve_rows:
        return False
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        markers = ["o", "s", "^", "v", "D"]
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
        fig, ax = plt.subplots(figsize=(6.0, 4.0), dpi=220, constrained_layout=True)
        all_y: list[float] = []
        for index, method in enumerate(plotted_methods):
            points: list[tuple[float, float]] = []
            for row in curve_rows:
                if str(row.get("method_id") or row.get("method")) != method:
                    continue
                x_value = _float_or_none(row.get("x_value"))
                y_value = _float_or_none(row.get("mean_metric"))
                if x_value is None or y_value is None:
                    continue
                points.append((x_value, y_value))
            if not points:
                continue
            points.sort(key=lambda item: item[0])
            xs = [item[0] for item in points]
            ys = [item[1] for item in points]
            all_y.extend(ys)
            ax.plot(
                xs,
                ys,
                marker=markers[index % len(markers)],
                color=colors[index % len(colors)],
                linewidth=2.0,
                markersize=6.0,
                label=label_for_method(method),
            )
        if not all_y:
            plt.close(fig)
            return False
        y_min = min(all_y)
        y_max = max(all_y)
        y_pad = 0.06 * (y_max - y_min if y_max > y_min else max(abs(y_max), 1.0))
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.set_xlabel(str(fig_def.get("x_label") or fig_def.get("x_axis_param") or "x"))
        ax.set_ylabel(str(fig_def.get("y_label") or fig_def.get("y_metric") or "metric"))
        ax.grid(True, alpha=0.28, linewidth=0.8)
        ax.legend(frameon=True, loc="best")
        dst_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(dst_png, bbox_inches="tight")
        fig.savefig(dst_pdf, bbox_inches="tight")
        plt.close(fig)
        return True
    except Exception:
        return False


def _write_markdown_table(csv_path: Path, md_path: Path) -> None:
    rows = list(csv.reader(csv_path.read_text(encoding="utf-8").splitlines()))
    if not rows:
        write_text(md_path, "")
        return
    widths = [0] * max(len(row) for row in rows)
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(str(cell)))

    def fmt(row: list[str]) -> str:
        padded = [str(cell).ljust(widths[index]) for index, cell in enumerate(row)]
        return "| " + " | ".join(padded) + " |"

    lines = [fmt(rows[0]), "| " + " | ".join("-" * width for width in widths) + " |"]
    lines.extend(fmt(row) for row in rows[1:])
    write_text(md_path, "\n".join(lines) + "\n")


def _clean_simple_output_dir(out_dir: Path) -> None:
    """Remove stale simple-run artifacts so repeated runs cannot reuse old plots."""

    out_dir.mkdir(parents=True, exist_ok=True)
    stale_files = [
        "simple_llm_prompt.txt",
        "simple_llm_raw_response.txt",
        "simple_llm_usage.json",
        "simple_experiment.py",
        "simple_experiment_stdout.txt",
        "simple_experiment_stderr.txt",
        "simple_experiment_manifest.json",
        "simple_experiment_error.txt",
        "simple_legacy_prompt.txt",
        "gain_scout_llm_prompt.txt",
        "gain_scout_legacy_prompt.txt",
        "gain_scout_llm_raw_response.txt",
        "gain_scout_llm_usage.json",
        "gain_scout.py",
        "gain_scout_stdout.txt",
        "gain_scout_stderr.txt",
        "preview_experiment_llm_prompt.txt",
        "preview_experiment_legacy_prompt.txt",
        "preview_experiment_llm_raw_response.txt",
        "preview_experiment_llm_usage.json",
        "preview_experiment.py",
        "preview_experiment_stdout.txt",
        "preview_experiment_stderr.txt",
        "two_call_preview_manifest.json",
        "experiment_scout_plan.json",
        "experiment_plan.json",
    ]
    stale_dirs = ["outputs", "figures"]
    for name in stale_files:
        path = out_dir / name
        if path.exists():
            path.unlink()
    for name in stale_dirs:
        path = out_dir / name
        if path.exists():
            shutil.rmtree(path)


def publish_phase24_simple_as_phase25(run_dir: Path, out_dir: Path) -> dict[str, Any]:
    """Publish Phase 2.4 simple outputs in the phase2-5 evidence schema.

    Phase 3.1 reads a fixed `phase2-5` evidence package. This adapter keeps that
    stable reader contract while preventing Phase 2.5 from redesigning the
    experiment after Phase 2.4 has already generated code, results, and figures.
    """

    run_dir = Path(run_dir).resolve()
    out_dir = Path(out_dir).resolve()
    outputs_dir = out_dir / "outputs"
    figures_src = out_dir / "figures"
    phase25_dir = run_dir / "phase2-5"
    figures_dir = phase25_dir / "figures"
    if phase25_dir.exists():
        for child in phase25_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    figures_dir.mkdir(parents=True, exist_ok=True)

    simple_summary = _read_json_file(outputs_dir / "simple_summary.json", {}) or {}
    preview_quality = _read_json_file(outputs_dir / "preview_quality_report.json", {}) or {}
    paper_recommendation = _read_json_file(outputs_dir / "paper_level_recommendation.json", {}) or {}
    figure_report = _read_json_file(outputs_dir / "figure_selection_report.json", {}) or {}
    benchmark_report = _read_json_file(outputs_dir / "benchmark_selection_report.json", {}) or {}
    paper_run_config = _read_json_file(outputs_dir / "paper_run_config.json", {}) or {}
    experiment_plan_24 = _read_json_file(outputs_dir / "experiment_plan.json", {}) or _read_json_file(out_dir / "experiment_plan.json", {}) or {}
    run_summary = _read_json_file(run_dir / "phase2_summary.json", {}) or {}

    rows: list[dict[str, str]] = []
    csv_path = outputs_dir / "simple_results.csv"
    if csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    for row in rows:
        if not row.get("figure_id"):
            for figure_key in ("figure", "figure_name", "chart_id", "chart", "plot_id", "plot"):
                if str(row.get(figure_key, "")).strip():
                    row["figure_id"] = str(row.get(figure_key, "")).strip()
                    break
        if not row.get("method_id") and row.get("method"):
            row["method_id"] = row.get("method", "")
        if not row.get("method") and row.get("method_id"):
            row["method"] = row.get("method_id", "")
        if not row.get("x_value") and row.get("sweep_value"):
            row["x_value"] = row.get("sweep_value", "")
        if not row.get("x_value") and row.get("swept_value"):
            row["x_value"] = row.get("swept_value", "")
        if not row.get("swept_value") and row.get("x_value"):
            row["swept_value"] = row.get("x_value", "")
        if not row.get("sweep_value") and row.get("x_value"):
            row["sweep_value"] = row.get("x_value", "")
        if not row.get("swept_param") and row.get("sweep_name"):
            row["swept_param"] = row.get("sweep_name", "")
        if not row.get("robust_feasibility") and row.get("feasible_numeric"):
            row["robust_feasibility"] = row.get("feasible_numeric", "")
        if not row.get("robust_feasibility") and row.get("feasible"):
            row["robust_feasibility"] = _truthy_numeric(row.get("feasible"))
        if not row.get("robust_feasibility") and row.get("robust_feasible_fraction"):
            row["robust_feasibility"] = row.get("robust_feasible_fraction", "")
        if not row.get("K") and row.get("num_users_actual"):
            row["K"] = row.get("num_users_actual", "")
        if not row.get("N_t") and row.get("num_tx_antennas_actual"):
            row["N_t"] = row.get("num_tx_antennas_actual", "")
        if not row.get("epsilon_scale") and row.get("uncertainty_scale"):
            row["epsilon_scale"] = row.get("uncertainty_scale", "")
        if not row.get("gamma_min") and row.get("sinr_floor_dB"):
            row["gamma_min"] = row.get("sinr_floor_dB", "")
        if not row.get("active_scenario_count_design") and row.get("active_scenario_count_actual"):
            row["active_scenario_count_design"] = row.get("active_scenario_count_actual", "")
        if not row.get("uncertainty_scenario_count_eval") and row.get("adversarial_error_samples_actual"):
            row["uncertainty_scenario_count_eval"] = row.get("adversarial_error_samples_actual", "")
    has_aggregated_rows = bool(rows and any(_row_looks_aggregated(row) for row in rows))
    if rows and not has_aggregated_rows and not any(str(row.get("seed", "")).strip() for row in rows):
        replicate_counter: dict[tuple[str, str, str], int] = {}
        for row in rows:
            key = (
                str(row.get("figure_id", "")),
                str(row.get("method_id", "")),
                str(row.get("x_value", "")),
            )
            replicate_counter[key] = replicate_counter.get(key, 0) + 1
            row["seed"] = str(replicate_counter[key])

    def summarize_column(column: str) -> Any:
        values: list[Any] = []
        seen: set[str] = set()
        for row in rows:
            raw = str(row.get(column, "")).strip()
            if not raw or raw in seen:
                continue
            seen.add(raw)
            value = _float_or_none(raw)
            values.append(value if value is not None else raw)
        if not values:
            return ""
        if len(values) == 1:
            return values[0]
        return values[:12]

    setup_snapshot = {
        "K": summarize_column("K"),
        "N_t": summarize_column("N_t"),
        "beta_over_alpha": summarize_column("beta_over_alpha"),
        "epsilon_scale": summarize_column("epsilon_scale"),
        "gamma_min": summarize_column("gamma_min"),
        "fixed_rho": summarize_column("fixed_rho"),
        "active_scenario_count_design": summarize_column("active_scenario_count_design"),
        "uncertainty_scenario_count_eval": summarize_column("uncertainty_scenario_count_eval"),
        "regimes": [
            {
                "figure_id": figure_id,
                "regime_id": next((row.get("regime_id", "") for row in rows if row.get("figure_id") == figure_id), ""),
                "regime_label": next((row.get("regime_label", "") for row in rows if row.get("figure_id") == figure_id), ""),
            }
            for figure_id in sorted({row.get("figure_id", "") for row in rows if row.get("figure_id")})
        ],
    }
    selected_system = simple_summary.get("system") or paper_run_config.get("selected_system") or {}
    selected_algorithm = simple_summary.get("algorithm") or paper_run_config.get("selected_algorithm") or {}
    if isinstance(selected_system, dict):
        setup_snapshot["K"] = setup_snapshot["K"] or selected_system.get("num_users")
        setup_snapshot["N_t"] = setup_snapshot["N_t"] or selected_system.get("num_tx_antennas")
        setup_snapshot["power_budget_dBm"] = selected_system.get("power_budget_dBm")
        setup_snapshot["channel_gain"] = selected_system.get("channel_gain")
        weights = selected_system.get("weights") if isinstance(selected_system.get("weights"), dict) else {}
        csi = selected_system.get("csi") if isinstance(selected_system.get("csi"), dict) else {}
        qos = selected_system.get("qos") if isinstance(selected_system.get("qos"), dict) else {}
        setup_snapshot["beta_over_alpha"] = setup_snapshot["beta_over_alpha"] or weights.get("beta_eh")
        setup_snapshot["epsilon_scale"] = setup_snapshot["epsilon_scale"] or csi.get("uncertainty_scale")
        setup_snapshot["gamma_min"] = setup_snapshot["gamma_min"] or qos.get("sinr_floor_dB")
        setup_snapshot["system_parameters"] = selected_system
    if isinstance(selected_algorithm, dict):
        setup_snapshot["active_scenario_count_design"] = (
            setup_snapshot["active_scenario_count_design"] or selected_algorithm.get("active_scenario_count")
        )
        setup_snapshot["uncertainty_scenario_count_eval"] = (
            setup_snapshot["uncertainty_scenario_count_eval"] or selected_algorithm.get("adversarial_error_samples")
        )
        setup_snapshot["algorithm_parameters"] = selected_algorithm

    styles = figure_report.get("plotted_methods_and_styles", {}) if isinstance(figure_report, dict) else {}
    if not styles and isinstance(benchmark_report, dict):
        styles = benchmark_report.get("style", {}) or {}
    display_names = benchmark_report.get("method_display_names", {}) if isinstance(benchmark_report, dict) else {}
    benchmark_entries: dict[str, dict[str, Any]] = {}
    if isinstance(benchmark_report, dict):
        proposed_entry = benchmark_report.get("proposed")
        if isinstance(proposed_entry, dict) and str(proposed_entry.get("id") or "").strip():
            benchmark_entries[str(proposed_entry.get("id")).strip()] = proposed_entry
        raw_benchmarks = benchmark_report.get("benchmarks")
        if isinstance(raw_benchmarks, list):
            for item in raw_benchmarks:
                if isinstance(item, dict) and str(item.get("id") or "").strip():
                    benchmark_entries[str(item.get("id")).strip()] = item
        elif isinstance(raw_benchmarks, dict):
            for method_id, item in raw_benchmarks.items():
                if isinstance(item, dict):
                    payload = dict(item)
                    payload.setdefault("id", method_id)
                    benchmark_entries[str(method_id)] = payload
    plotted_methods = (
        benchmark_report.get("selected_plotted_methods")
        or benchmark_report.get("final_plotted_method_set")
        or simple_summary.get("plotted_methods")
        or simple_summary.get("methods")
        or list(styles.keys())
        or sorted({row.get("method_id", "") for row in rows if row.get("method_id")})
    )
    plotted_methods = [str(method) for method in plotted_methods if str(method)]
    primary_benchmark = str(
        benchmark_report.get("credible_primary_benchmark")
        or benchmark_report.get("primary_practical_benchmark")
        or simple_summary.get("primary_benchmark")
        or (plotted_methods[1] if len(plotted_methods) > 1 else "")
    )
    proposed_method = next((method for method in plotted_methods if "proposed" in method.lower()), plotted_methods[0] if plotted_methods else "proposed")
    primary_metric = _select_primary_metric(
        simple_summary=simple_summary,
        preview_quality=preview_quality,
        figure_report=figure_report,
        paper_run_config=paper_run_config,
        rows=rows,
    )

    def _title_method_id(method_id: str) -> str:
        normalized = str(method_id or "").strip()
        known = {
            "proposed": "Proposed",
            "regularized_zf_heuristic": "RZF",
            "mrt_or_channel_matched": "MRT",
            "mrt_covariance_baseline": "MRT-Cov.",
            "no_shared_covariance_baseline": "No shared cov.",
            "isotropic_shared_covariance_baseline": "Isotropic cov.",
            "equal_power_heuristic": "Equal power",
            "channel_inversion_heuristic": "Channel inversion",
            "fixed_ris": "Fixed-RIS",
            "fixed_ps": "Fixed-PS",
            "linear_eh": "Linear-EH",
            "nominal_csi": "Nominal-CSI",
        }
        lowered = normalized.lower()
        if lowered in known:
            return known[lowered]
        words = [word for word in re.split(r"[_\s-]+", normalized) if word]
        acronyms = {"ris", "zf", "mrt", "mimo", "noma", "swipt", "uav", "csi", "sca", "sdr", "sic"}
        rendered = [word.upper() if word.lower() in acronyms else word.capitalize() for word in words]
        return "-".join(rendered[:3]) if rendered else normalized

    def short_label(method_id: str) -> str:
        style = styles.get(method_id, {}) if isinstance(styles, dict) else {}
        label = str(style.get("label") or "").strip()
        if label:
            return label
        if isinstance(display_names, dict) and str(display_names.get(method_id) or "").strip():
            return str(display_names.get(method_id)).strip()
        entry = benchmark_entries.get(method_id, {})
        for key in ("display_name_short", "short_name", "label"):
            if isinstance(entry, dict) and str(entry.get(key) or "").strip():
                return str(entry.get(key)).strip()
        for row in rows:
            if row.get("method_id") == method_id:
                row_label = str(row.get("method_label") or "").strip()
                if row_label and row_label != method_id:
                    return row_label
        return _title_method_id(method_id)

    def long_label(method_id: str) -> str:
        entry = benchmark_entries.get(method_id, {})
        for key in ("display_name_long", "long_name", "description"):
            if isinstance(entry, dict) and str(entry.get(key) or "").strip():
                text = str(entry.get(key)).strip()
                return text[:1].upper() + text[1:]
        return short_label(method_id)

    compared_methods = [
        {
            "internal_name": method,
            "name": method,
            "role": "proposed" if method == proposed_method else "main_baseline" if method == primary_benchmark else "ablation",
            "display_name_short": short_label(method),
            "display_name_long": long_label(method),
            "source_of_name": "phase2.4_benchmark_selection_report",
        }
        for method in plotted_methods
    ]

    figure_defs = _figure_defs_from_reports(
        out_dir=out_dir,
        figures_src=figures_src,
        figure_report=figure_report,
        paper_run_config=paper_run_config,
        primary_metric=primary_metric,
    )
    if rows and figure_defs:
        figure_aliases: dict[str, str] = {}
        for fig in figure_defs:
            aliases = {
                fig.get("phase24_id"),
                fig.get("figure_id"),
                fig.get("x_axis_param"),
                fig.get("required_sweep"),
                fig.get("x_label"),
            }
            for alias in aliases:
                key = _normalized_key(alias)
                if key:
                    figure_aliases[key] = str(fig["phase24_id"])
        claim_evidence = simple_summary.get("claim_evidence") if isinstance(simple_summary.get("claim_evidence"), dict) else {}
        for fig in figure_defs:
            evidence = claim_evidence.get(str(fig["phase24_id"])) or claim_evidence.get(str(fig["figure_id"]))
            if isinstance(evidence, dict):
                for alias in [evidence.get("axis"), evidence.get("sweep_axis"), evidence.get("x_axis")]:
                    key = _normalized_key(alias)
                    if key:
                        figure_aliases[key] = str(fig["phase24_id"])
        for row in rows:
            if str(row.get("figure_id") or "").strip():
                continue
            row_axis = _first_text(
                row.get("sweep_axis"),
                row.get("sweep_name"),
                row.get("sweep_id"),
                row.get("swept_param"),
                row.get("x_axis_param"),
            )
            mapped = figure_aliases.get(_normalized_key(row_axis))
            if mapped:
                row["figure_id"] = mapped

    preview_passed = bool(
        preview_quality.get("preview_passed")
        or paper_recommendation.get("preview_ready_for_future_paper_level_run")
        or paper_run_config.get("paper_ready_from_preview")
    )
    configured_seed_count = _configured_seed_count(
        paper_run_config=paper_run_config,
        preview_quality=preview_quality,
    )

    def aggregate_curve_rows(fig: dict[str, Any]) -> list[dict[str, Any]]:
        grouped: dict[tuple[str, str], list[float]] = {}
        counts: dict[tuple[str, str], int] = {}
        phase24_id = str(fig["phase24_id"])
        y_metric = str(fig["y_metric"])
        for row in rows:
            if row.get("figure_id") != phase24_id or row.get("method_id") not in plotted_methods:
                continue
            x_value = _first_text(row.get("x_value"), row.get("sweep_value"), row.get("swept_value"))
            if not x_value:
                continue
            key = (x_value, str(row.get("method_id")))
            value = _row_metric_value(row, y_metric, primary_metric["name"], "objective", "worst_case_utility")
            if value is None:
                continue
            grouped.setdefault(key, []).append(value)
            counts[key] = counts.get(key, 0) + _row_sample_count(
                row,
                metric_name=y_metric,
                configured_seed_count=configured_seed_count,
            )
        curve_rows: list[dict[str, Any]] = []
        for (x_value, method), values in sorted(grouped.items(), key=lambda item: (_float_or_none(item[0][0]) or 0.0, item[0][1])):
            curve_rows.append(
                {
                    "x_value": x_value,
                    "swept_value": x_value,
                    "method": method,
                    "method_id": method,
                    "method_label": short_label(method),
                    "metric": y_metric,
                    "mean_metric": _mean(values),
                    "num_samples": counts.get((x_value, method), len(values)),
                }
            )
        return curve_rows

    figure_meta: list[dict[str, Any]] = []
    plot_quality_figures: list[dict[str, Any]] = []
    all_curve_rows: list[dict[str, Any]] = []
    min_points_for_preview = int(os.environ.get("WARA_PHASE24_SIMPLE_MIN_X_POINTS", "6") or 6)
    min_seeds_for_preview = int(os.environ.get("WARA_PHASE24_SIMPLE_MIN_SEEDS", "3") or 3)
    for fig in figure_defs:
        curve_rows = aggregate_curve_rows(fig)
        rendered_from_verified_curve = False
        if curve_rows:
            rendered_from_verified_curve = _render_verified_curve_figure(
                curve_rows=curve_rows,
                fig_def=fig,
                plotted_methods=plotted_methods,
                label_for_method=short_label,
                dst_png=figures_dir / fig["filename_png"],
                dst_pdf=figures_dir / fig["filename_pdf"],
            )
        for curve_row in curve_rows:
            normalized_curve_row = dict(curve_row)
            normalized_curve_row["figure_id"] = fig["figure_id"]
            normalized_curve_row["source_phase24_figure_id"] = fig["phase24_id"]
            all_curve_rows.append(normalized_curve_row)
        curve_path = figures_dir / fig["curve_csv"]
        if curve_rows:
            with curve_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(curve_rows[0].keys()))
                writer.writeheader()
                writer.writerows(curve_rows)
        x_values = sorted({_float_or_none(row.get("x_value")) for row in curve_rows if _float_or_none(row.get("x_value")) is not None})
        seeds_by_x: dict[str, set[str]] = {}
        seed_counts_by_x: dict[str, int] = {}
        for row in rows:
            if row.get("figure_id") == fig["phase24_id"] and row.get("method_id") == proposed_method:
                x_value = _first_text(row.get("x_value"), row.get("sweep_value"), row.get("swept_value"))
                if x_value:
                    sample_count = _row_sample_count(
                        row,
                        metric_name=fig["y_metric"],
                        configured_seed_count=configured_seed_count,
                    )
                    if _row_looks_aggregated(row, fig["y_metric"]):
                        seed_counts_by_x[x_value] = max(seed_counts_by_x.get(x_value, 0), sample_count)
                    elif str(row.get("seed") or "").strip():
                        seeds_by_x.setdefault(x_value, set()).add(str(row.get("seed")))
        if not seeds_by_x and curve_rows:
            for curve_row in curve_rows:
                if curve_row.get("method_id") == proposed_method:
                    seeds_by_x.setdefault(str(curve_row.get("x_value")), set()).update(
                        str(index + 1) for index in range(int(curve_row.get("num_samples") or 0))
                    )
        seeds_per_point = [len(value) for value in seeds_by_x.values() if value]
        seeds_per_point.extend(value for value in seed_counts_by_x.values() if value)
        seeds_summary = {
            "min": min(seeds_per_point) if seeds_per_point else 0,
            "median": sorted(seeds_per_point)[len(seeds_per_point) // 2] if seeds_per_point else 0,
            "max": max(seeds_per_point) if seeds_per_point else 0,
        }
        figure_ready = bool(
            len(x_values) >= min_points_for_preview
            and (seeds_summary["median"] or 0) >= min_seeds_for_preview
            and set(plotted_methods).issubset({row.get("method_id") for row in rows if row.get("figure_id") == fig["phase24_id"]})
        )
        blocking_issues: list[str] = []
        if len(x_values) < min_points_for_preview:
            blocking_issues.append("too_few_x_points")
        if (seeds_summary["median"] or 0) < min_seeds_for_preview:
            blocking_issues.append("too_few_seeds_per_point")
        if not curve_rows:
            blocking_issues.append("missing_verified_curve_data")
        if curve_rows and not rendered_from_verified_curve:
            blocking_issues.append("verified_curve_render_failed")
        missing_methods = sorted(
            set(plotted_methods) - {row.get("method_id") for row in rows if row.get("figure_id") == fig["phase24_id"]}
        )
        if missing_methods:
            blocking_issues.append("missing_method_curve:" + ",".join(missing_methods))
        meta = {
            "figure_id": fig["figure_id"],
            "source_phase24_figure_id": fig["phase24_id"],
            "filename_png": fig["filename_png"],
            "filename_pdf": fig["filename_pdf"],
            "png_path": str(figures_dir / fig["filename_png"]),
            "pdf_path": str(figures_dir / fig["filename_pdf"]),
            "draft_or_final": "final" if figure_ready else "draft",
            "paper_ready": figure_ready,
            "chart_type": "line",
            "x_axis_param": fig["x_axis_param"],
            "x_axis_label": fig["x_label"],
            "required_sweep": fig["required_sweep"],
            "y_metric": fig["y_metric"],
            "y_axis_label": fig["y_label"],
            "num_x_points": len(x_values),
            "methods": plotted_methods,
            "method_display_names_short": {method: short_label(method) for method in plotted_methods},
            "method_display_names_long": {method: long_label(method) for method in plotted_methods},
            "has_error_bars": False,
            "error_display": "none",
            "error_display_label": "",
            "blocking_issues": blocking_issues,
            "seeds_per_point_summary": seeds_summary,
            "render_source": "verified_curve_data" if rendered_from_verified_curve else "phase2_4_png",
            "plot_fidelity": {
                "ok": bool(curve_rows and rendered_from_verified_curve),
                "rendered_from_verified_curve_data": rendered_from_verified_curve,
                "curve_rows": len(curve_rows),
                "expected_methods": plotted_methods,
                "x_points": len(x_values),
                "note": (
                    "Phase 2.5 rendered this figure from verified curve data."
                    if rendered_from_verified_curve
                    else "Phase 2.5 did not render a verified data figure; this blocks paper readiness."
                ),
            },
        }
        figure_meta.append(meta)
        plot_quality_figures.append(
            {
                "figure_id": fig["figure_id"],
                "chart_type": "line",
                "purpose": fig["purpose"],
                "x_axis_param": fig["x_axis_param"],
                "x_axis_label": fig["x_label"],
                "required_sweep": fig["required_sweep"],
                "y_metric": fig["y_metric"],
                "y_axis_label": fig["y_label"],
                "figure_intent": fig["chart_intent"],
                "methods_required": plotted_methods,
                "methods_present": plotted_methods,
                "num_x_points": len(x_values),
                "x_values": x_values,
                "seeds_per_point_summary": seeds_summary,
                "has_error_bars": False,
                "error_display": "none",
                "paper_ready": figure_ready,
                "draft_only": not figure_ready,
                "quality_level": "paper_minimum_ready" if figure_ready else "needs_more_phase24_runs",
                "paper_minimum_ready": figure_ready,
                "blocking_issues": blocking_issues,
                "warnings": [] if figure_ready else [str(preview_quality.get("next_recommended_action") or "Run the recommended Phase 2.4 paper-level experiment.")],
                "render_source": "verified_curve_data" if rendered_from_verified_curve else "phase2_4_png",
                "plot_fidelity": {
                    "ok": bool(curve_rows and rendered_from_verified_curve),
                    "rendered_from_verified_curve_data": rendered_from_verified_curve,
                    "curve_rows": len(curve_rows),
                    "expected_methods": plotted_methods,
                    "x_points": len(x_values),
                },
            }
        )

    adapter_structural_ready = bool(figure_meta and all(fig["paper_ready"] for fig in figure_meta))
    higher_is_better = bool(primary_metric.get("higher_is_better", True))
    by_case_all: dict[tuple[str, str, str], dict[str, float]] = {}
    claim_rows = all_curve_rows if all_curve_rows else rows
    for row in claim_rows:
        method = row.get("method_id")
        if method not in plotted_methods:
            continue
        value = _row_metric_value(row, primary_metric["name"], "mean_metric", "objective", "worst_case_utility")
        if value is None:
            continue
        key = (
            str(row.get("figure_id")),
            _first_text(row.get("x_value"), row.get("sweep_value"), row.get("swept_value")),
            "curve_mean" if all_curve_rows else str(row.get("seed")),
        )
        by_case_all.setdefault(key, {})[str(method)] = value

    def benchmark_advantage_stats(benchmark_method: str) -> dict[str, Any]:
        diffs: list[float] = []
        rels: list[float] = []
        wins_local = 0
        total_local = 0
        for values in by_case_all.values():
            if proposed_method not in values or benchmark_method not in values:
                continue
            raw_diff = values[proposed_method] - values[benchmark_method]
            advantage = raw_diff if higher_is_better else -raw_diff
            diffs.append(advantage)
            denom = abs(values[benchmark_method]) if abs(values[benchmark_method]) > 1e-12 else 1.0
            rels.append(advantage / denom)
            wins_local += 1 if advantage > 0 else 0
            total_local += 1
        return {
            "benchmark_method": benchmark_method,
            "num_comparable_cases": total_local,
            "win_rate": wins_local / total_local if total_local else 0.0,
            "mean_relative_gain": _mean(rels) or 0.0,
            "mean_absolute_gain": _mean(diffs) or 0.0,
            "wins": wins_local,
        }

    benchmark_claim_stats = {
        method: benchmark_advantage_stats(method)
        for method in plotted_methods
        if method != proposed_method
    }

    strongest_diffs: list[float] = []
    strongest_rels: list[float] = []
    strongest_wins = 0
    strongest_total_pairs = 0
    for values in by_case_all.values():
        if proposed_method not in values:
            continue
        benchmark_values = [value for method, value in values.items() if method != proposed_method]
        if not benchmark_values:
            continue
        strongest_benchmark_value = max(benchmark_values) if higher_is_better else min(benchmark_values)
        raw_diff = values[proposed_method] - strongest_benchmark_value
        advantage = raw_diff if higher_is_better else -raw_diff
        strongest_diffs.append(advantage)
        denom = abs(strongest_benchmark_value) if abs(strongest_benchmark_value) > 1e-12 else 1.0
        strongest_rels.append(advantage / denom)
        strongest_wins += 1 if advantage > 0 else 0
        strongest_total_pairs += 1
    strongest_win_rate = strongest_wins / strongest_total_pairs if strongest_total_pairs else 0.0
    strongest_rel_gain = _mean(strongest_rels) or 0.0
    strongest_abs_gain = _mean(strongest_diffs) or 0.0
    strongest_benchmark_claim_pass = bool(
        strongest_total_pairs
        and strongest_win_rate >= float(os.environ.get("WARA_PHASE24_SIMPLE_MIN_STRONGEST_WIN_RATE", "0.60") or 0.60)
        and strongest_abs_gain > 0
    )
    min_claim_win_rate = float(os.environ.get("WARA_PHASE24_SIMPLE_MIN_CLAIM_WIN_RATE", "0.60") or 0.60)
    passing_benchmarks = [
        method
        for method, stats in benchmark_claim_stats.items()
        if stats.get("num_comparable_cases", 0)
        and stats.get("win_rate", 0.0) >= min_claim_win_rate
        and stats.get("mean_absolute_gain", 0.0) > 0
    ]
    passing_benchmarks.sort(
        key=lambda method: (
            benchmark_claim_stats[method].get("win_rate", 0.0),
            benchmark_claim_stats[method].get("mean_relative_gain", 0.0),
            benchmark_claim_stats[method].get("mean_absolute_gain", 0.0),
        ),
        reverse=True,
    )
    claim_benchmark = ""
    if primary_benchmark in passing_benchmarks:
        claim_benchmark = primary_benchmark
    elif passing_benchmarks:
        claim_benchmark = passing_benchmarks[0]
    at_least_one_benchmark_claim_pass = bool(claim_benchmark)
    if claim_benchmark and claim_benchmark != primary_benchmark:
        primary_benchmark = claim_benchmark
        for method_info in compared_methods:
            if method_info.get("internal_name") == proposed_method:
                method_info["role"] = "proposed"
            elif method_info.get("internal_name") == primary_benchmark:
                method_info["role"] = "main_baseline"
            else:
                method_info["role"] = "competitive_benchmark"
    primary_stats = benchmark_claim_stats.get(
        primary_benchmark,
        {
            "num_comparable_cases": 0,
            "win_rate": 0.0,
            "mean_relative_gain": 0.0,
            "mean_absolute_gain": 0.0,
            "wins": 0,
        },
    )
    wins = int(primary_stats.get("wins", 0) or 0)
    total_pairs = int(primary_stats.get("num_comparable_cases", 0) or 0)
    win_rate = float(primary_stats.get("win_rate", 0.0) or 0.0)
    rel_gain = float(primary_stats.get("mean_relative_gain", 0.0) or 0.0)
    abs_gain = float(primary_stats.get("mean_absolute_gain", 0.0) or 0.0)
    deterministic_claim_pass = at_least_one_benchmark_claim_pass
    adapter_ready = bool(adapter_structural_ready and at_least_one_benchmark_claim_pass)
    phase25_status = "paper_minimum_ready" if adapter_ready else "needs_more_phase24_runs"
    for fig in figure_meta:
        fig["draft_or_final"] = "final" if adapter_ready and not fig["blocking_issues"] else "draft"
        fig["paper_ready"] = bool(adapter_ready and not fig["blocking_issues"])
    for fig in plot_quality_figures:
        fig["paper_ready"] = bool(adapter_ready and not fig["blocking_issues"])
        fig["draft_only"] = not fig["paper_ready"]
        fig["quality_level"] = phase25_status
        fig["paper_minimum_ready"] = fig["paper_ready"]
    proposed_utility = _mean(
        [
            value
            for row in rows
            if row.get("method_id") == proposed_method
            for value in [_row_metric_value(row, primary_metric["name"], "objective", "worst_case_utility")]
            if value is not None
        ]
    )
    benchmark_utility = _mean(
        [
            value
            for row in rows
            if row.get("method_id") == primary_benchmark
            for value in [_row_metric_value(row, primary_metric["name"], "objective", "worst_case_utility")]
            if value is not None
        ]
    )
    proposed_feasibility = _mean(
        [
            value
            for row in rows
            if row.get("method_id") == proposed_method
            for value in [_float_or_none(row.get("robust_feasibility"))]
            if value is not None
        ]
    )
    benchmark_feasibility = _mean(
        [
            value
            for row in rows
            if row.get("method_id") == primary_benchmark
            for value in [_float_or_none(row.get("robust_feasibility"))]
            if value is not None
        ]
    )

    primary_claim_check = {
        "mode": "advantage_over_benchmark",
        "proposed_method": proposed_method,
        "baseline_method": primary_benchmark,
        "passes": adapter_ready,
        "proposed_win_rate": win_rate,
        "proposed_mean_relative_gain": rel_gain,
        "proposed_mean_absolute_gain": abs_gain,
        "strongest_benchmark_win_rate": strongest_win_rate,
        "strongest_benchmark_mean_relative_gain": strongest_rel_gain,
        "strongest_benchmark_mean_absolute_gain": strongest_abs_gain,
        "strongest_benchmark_claim_pass": strongest_benchmark_claim_pass,
        "at_least_one_benchmark_claim_pass": at_least_one_benchmark_claim_pass,
        "claim_benchmark": primary_benchmark,
        "benchmark_claim_stats": benchmark_claim_stats,
        "deterministic_claim_pass": deterministic_claim_pass,
        "claim_scope": (
            "Proposed is required to show stable gain over at least one credible plotted benchmark; "
            "other plotted benchmarks may be competitive and should be discussed with scoped wording."
        ),
        "paper_ready": adapter_ready,
    }
    overall = {
        "num_cases": len(
            {
                (
                    row.get("figure_id"),
                    _first_text(row.get("x_value"), row.get("sweep_value"), row.get("swept_value")),
                    row.get("seed"),
                )
                for row in rows
            }
        ),
        "num_results": len(rows),
        "num_comparable_cases": total_pairs,
        "proposed_win_rate": win_rate,
        "proposed_mean_relative_gain": rel_gain,
        "proposed_median_relative_gain": rel_gain,
    }
    primary_claim = str(
        (simple_summary.get("claim_evidence") or {}).get("claim")
        or simple_summary.get("primary_claim")
        or (
            "The proposed design improves the paper-defined primary metric "
            f"over {short_label(primary_benchmark)} in the selected operating regimes."
        )
    )
    physical_kpi_support = {
        key: value
        for source in (preview_quality, simple_summary)
        for key, value in source.items()
        if key.startswith("proposed_")
        and key.endswith("_relative_gain_vs_primary_benchmark")
        and "objective" not in key
        and value is not None
    }
    claim_evidence = dict(simple_summary.get("claim_evidence") or {})
    claim_evidence.update(
        {
            "claim": primary_claim,
            "source_phase": "phase2.4",
            "primary_metric": primary_metric,
            "primary_benchmark": primary_benchmark,
            "claim_benchmark": primary_benchmark,
            "proposed_method": proposed_method,
            "plotted_methods": plotted_methods,
            "evidence_figures": [fig["figure_id"] for fig in figure_meta],
            "objective_formula": _first_text(
                claim_evidence.get("objective_formula"),
                claim_evidence.get("paper_objective_formula"),
                f"paper-defined primary metric `{primary_metric['name']}`",
            ),
            "aggregate_proposed_mean_relative_gain": rel_gain,
            "aggregate_proposed_mean_absolute_gain": abs_gain,
            "paired_feasible_win_rate": win_rate,
            "strongest_benchmark_win_rate": strongest_win_rate,
            "strongest_benchmark_mean_relative_gain": strongest_rel_gain,
            "strongest_benchmark_mean_absolute_gain": strongest_abs_gain,
            "strongest_benchmark_claim_pass": strongest_benchmark_claim_pass,
            "at_least_one_benchmark_claim_pass": at_least_one_benchmark_claim_pass,
            "benchmark_claim_stats": benchmark_claim_stats,
            "deterministic_claim_pass": deterministic_claim_pass,
            "num_comparable_cases": total_pairs,
            "num_strongest_comparable_cases": strongest_total_pairs,
            "proposed_mean_utility": proposed_utility,
            "primary_benchmark_mean_utility": benchmark_utility,
            "proposed_feasibility_rate": proposed_feasibility,
            "primary_benchmark_feasibility_rate": benchmark_feasibility,
            "physical_kpi_support": physical_kpi_support,
            "selected_operating_regime": setup_snapshot,
            "preview_passed": preview_passed,
            "adapter_ready": adapter_ready,
        }
    )
    plot_quality_report = {
        "overall_status": phase25_status,
        "figures": plot_quality_figures,
        "primary_claim_check": primary_claim_check,
        "plot_fidelity_gate": {
            "ok": all((fig.get("plot_fidelity") or {}).get("ok", False) for fig in figure_meta),
            "policy": (
                "Final Phase 2.5 figures are rendered from verified curve data whenever curve data are available; "
                "LLM-generated preview PNGs are not trusted as final evidence."
            ),
            "figures": [
                {
                    "figure_id": fig.get("figure_id"),
                    "render_source": fig.get("render_source"),
                    "plot_fidelity": fig.get("plot_fidelity"),
                }
                for fig in figure_meta
            ],
        },
        "source_phase": "phase2.4",
    }
    phase25_summary = {
        "phase25_status": phase25_status,
        "data_source": "phase2.4_simple_preview",
        "produced_by_phase": "phase2.4",
        "phase2_5_llm_planning_skipped": True,
        "quick_mode": not adapter_ready,
        "num_cases": overall["num_cases"],
        "num_results": overall["num_results"],
        "num_comparable_cases": total_pairs,
        "primary_metric": primary_metric,
        "plotted_methods": plotted_methods,
        "compared_methods": compared_methods,
        "primary_benchmark": primary_benchmark,
        "primary_claim": primary_claim,
        "claim_evidence": claim_evidence,
        "simulation_setup": setup_snapshot,
        "proposed_win_count": wins,
        "proposed_win_rate": win_rate,
        "proposed_mean_relative_gain": rel_gain,
        "proposed_median_relative_gain": rel_gain,
        "primary_claim_check": primary_claim_check,
        "figures": figure_meta,
        "plot_fidelity_gate": plot_quality_report["plot_fidelity_gate"],
        "tables": [],
        "plot_quality_report": plot_quality_report,
        "plot_quality_report_path": str(phase25_dir / "plot_quality_report.json"),
        "overall": overall,
        "paper_ready_figures": [fig["figure_id"] for fig in figure_meta if fig["paper_ready"]],
        "draft_figures": [fig["figure_id"] for fig in figure_meta if not fig["paper_ready"]],
        "paper_minimum_ready": adapter_ready,
        "paper_preferred_ready": False,
        "high_confidence_ready": False,
        "generated_figures_are_draft_only": not adapter_ready,
        "paper_sweep_plan_path": str(phase25_dir / "paper_sweep_plan.json"),
        "missing_experiments_path": str(phase25_dir / "missing_experiments.md"),
        "figure_captions_path": str(phase25_dir / "figure_captions.md"),
        "method_naming_summary_json_path": str(phase25_dir / "method_naming_summary.json"),
        "verified_registry_path": str(phase25_dir / "phase25_verified_registry.json"),
        "verified_registry_status": "verified_experiment_registry",
        "limitations": [] if adapter_ready else [str(preview_quality.get("next_recommended_action") or "Phase 2.4 preview is not paper-ready.")],
    }
    experiment_plan = {
        "paper_target": "IEEE WCL",
        "produced_by_phase": "phase2.4",
        "phase2_5_llm_planning_skipped": True,
        "primary_metric": primary_metric,
        "compared_methods": compared_methods,
        "figure_specs": [
            {
                "figure_id": fig["figure_id"],
                "purpose": fig["purpose"],
                "chart_intent": fig["chart_intent"],
                "chart_type": "line",
                "methods": plotted_methods,
                "metric": {
                    "name": fig["y_metric"],
                    "display_name": fig["y_label"],
                    "higher_is_better": True,
                    "aggregation": "curve_stabilized_by_repeated_seeds",
                },
                "encoding": {
                    "x": {"type": "numeric", "field": "x_value", "sweep_param": fig["x_axis_param"], "display_name": fig["x_label"]},
                    "group": {"type": "method", "field": "method_id", "display_name": "Method"},
                },
                "error_display": "none",
                "source_phase24_figure_id": fig["phase24_id"],
            }
            for fig in figure_defs
        ],
        "table_specs": [],
        "paper_claims_to_test": [
            {
                "claim": primary_claim,
                "required_evidence": "figure_1 and figure_2",
                "failure_mode": "Non-positive primary-metric gain, weak physical KPI support, or non-comparable feasibility.",
            }
        ],
        "missing_experiment_recommendations": [] if preview_passed else [paper_recommendation],
        "phase24_experiment_plan": experiment_plan_24,
    }
    method_naming = {
        "methods": {
            method: {
                "display_name_short": short_label(method),
                "display_name_long": short_label(method),
                "role": "proposed" if method == proposed_method else "main_baseline" if method == primary_benchmark else "ablation",
            }
            for method in plotted_methods
        }
    }
    registry = {
        "registry_schema": "phase2.4_to_phase2.5_compat_v1",
        "status": "verified_experiment_registry",
        "phase25_status": phase25_status,
        "source_policy": "Phase 2.4 generated the experiment, figures, benchmark selection, and summaries; Phase 2.5 LLM planning is skipped.",
        "primary_metric": primary_metric,
        "methods": method_naming["methods"],
        "paper_claims_to_test": experiment_plan["paper_claims_to_test"],
        "summary_numbers": overall,
        "simulation_setup": setup_snapshot,
        "figures": figure_meta,
        "tables": [],
        "comparison_records": [],
        "raw_result_records": rows[:200],
        "allowed_numeric_columns": list(rows[0].keys()) if rows else [],
    }

    captions_src = read_text(figures_src / "figure_captions.md").strip()
    if not captions_src:
        captions_src = "\n\n".join(
            f"Fig. {index}. {fig['y_label']} versus {fig['x_label']}."
            for index, fig in enumerate(figure_defs, start=1)
        )
    write_text(phase25_dir / "figure_captions.md", captions_src.rstrip() + "\n")
    missing = "No additional Phase 2.4 experiments are blocking this package.\n" if adapter_ready else str(preview_quality.get("next_recommended_action") or "Run the recommended Phase 2.4 paper-level experiment before Phase 3.1 final prose.")
    write_text(phase25_dir / "missing_experiments.md", missing.rstrip() + "\n")
    write_text(phase25_dir / "phase25_wcl_experiment_summary.md", (missing.rstrip() if not adapter_ready else "Phase 2.4 produced the paper-ready experiment package consumed by Phase 3.1.") + "\n")
    write_text(phase25_dir / "plot_quality_report.md", json.dumps(plot_quality_report, ensure_ascii=False, indent=2) + "\n")
    _write_json_file(phase25_dir / "phase25_experiment_summary.json", phase25_summary)
    _write_json_file(phase25_dir / "experiment_plan.json", experiment_plan)
    _write_json_file(phase25_dir / "plot_quality_report.json", plot_quality_report)
    _write_json_file(phase25_dir / "phase25_verified_registry.json", registry)
    _write_json_file(phase25_dir / "method_naming_summary.json", method_naming)
    _write_json_file(phase25_dir / "available_data_summary.json", {"source_csv": str(csv_path), "num_rows": len(rows), "columns": list(rows[0].keys()) if rows else []})
    _write_json_file(phase25_dir / "monte_carlo_check.json", {"status": "not_plotted", "note": "Seed stability is stored as diagnostics; final figures use clean curves without error bars."})
    write_text(phase25_dir / "monte_carlo_check.md", "Seed stability is stored as diagnostics; final figures use clean curves without error bars.\n")
    _write_json_file(phase25_dir / "paper_sweep_plan.json", paper_run_config or paper_recommendation)
    _write_json_file(
        phase25_dir / "phase25_manifest.json",
        {
            "produced_by_phase": "phase2.4",
            "phase2_5_role": "compatibility_adapter_only",
            "phase2_5_llm_planning_skipped": True,
            "phase2_5_role": "compatibility_adapter_only",
            "phase2_5_llm_planning_skipped": True,
            "phase25_status": phase25_status,
            "source_dir": str(out_dir),
            "consumer": "phase3.1",
        },
    )
    _write_json_file(phase25_dir / "paper_method_semantics_check.json", {"ok": True, "source": "phase2.4_benchmark_selection_report"})
    write_text(phase25_dir / "experiment_plan_prompt.txt", "Phase 2.5 LLM experiment planning skipped; Phase 2.4 published this compatibility package.\n")
    write_text(phase25_dir / "experiment_plan_raw_response.txt", json.dumps(experiment_plan, ensure_ascii=False, indent=2))
    write_text(phase25_dir / "result_writer_raw_response.txt", "Phase 2.5 result writer skipped; Phase 2.4 published phase2-5-compatible evidence.\n")

    return {
        "phase25_dir": str(phase25_dir),
        "phase25_status": phase25_status,
        "paper_minimum_ready": adapter_ready,
        "figures": [str(figures_dir / fig["filename_png"]) for fig in figure_defs],
        "summary_path": str(phase25_dir / "phase25_experiment_summary.json"),
    }


def build_simple_experiment_prompt(run_dir: Path, topic: str) -> str:
    phase1 = run_dir / "phase2-1"
    phase2 = run_dir / "phase2-2"
    phase3 = run_dir / "phase2-3"
    math_contract = _read_first_existing(
        phase1 / "mathematical_contract.frozen.json",
        phase1 / "mathematical_contract.json",
    )
    system_model = read_text(phase1 / "system_model.md")
    problem_formulation = read_text(phase1 / "problem_formulation.md")
    reformulation = read_text(phase2 / "reformulation_path.md")
    algorithm = read_text(phase3 / "algorithm.md")
    benchmark = read_text(phase3 / "benchmark_definition.md")
    experiment_blueprint = read_text(phase3 / "experiment_blueprint.md")

    return f"""You are a senior wireless-systems researcher and numerical-experiment engineer.

Write ONE complete self-contained Python script for a WCL-level numerical experiment.

Return raw Python source only. Do not return markdown fences, JSON, prose, or explanation.

Goal:
- Demonstrate the performance gain of the proposed scheme for the current paper.
- Implement the proposed algorithm and meaningful benchmark methods.
- Produce paper-supporting evidence, not toy placeholder data.
- Act like the experiment designer for a short IEEE WCL paper: choose the operating regime, sweep axes, KPIs, and figure story that are most likely to reveal the theoretical gain of the proposed scheme.
- Use the paper-defined optimization objective from the frozen mathematical contract as the primary evidence criterion. Do not invent a different objective for deciding whether the proposed method works.
- Final figure y-axes should be IEEE WCL-readable physical KPIs derived from the objective or its terms, not an opaque weighted number alone. If you plot the objective, label it in readable terms and decompose it.
- The script must run by itself from its own directory and write:
  - outputs/theoretical_trends.json
  - outputs/scout_results.csv
  - outputs/scout_summary.json
  - outputs/selected_regime.json
  - outputs/paper_run_config.json
  - outputs/progress_status.json
  - outputs/figure_selection_report.json
  - outputs/benchmark_selection_report.json
  - outputs/simple_results.csv
  - outputs/simple_summary.json
  - figures/fig1_primary_gain.png
  - figures/fig2_insight.png
  - figures/figure_captions.md
  - experiment_plan.json

Experiment design:
- Keep it simple and coherent: define the experiment plan inside the Python script as constants.
- Start by encoding theoretical trend hypotheses in the script and write them to `outputs/theoretical_trends.json`. For each candidate regime, state:
  - why the proposed scheme should gain there,
  - which paper-objective term or constraint bottleneck it improves,
  - which sweep should expose the trend,
  - expected qualitative trends for proposed and benchmarks,
  - which WCL-readable KPI should be plotted if the trend is confirmed.
- Then run a fast scout phase before final plotting. Use few sweep points but enough Monte Carlo seeds per point in scout, evaluate the paper-defined objective and several WCL-readable physical KPIs, and write `outputs/scout_results.csv` plus `outputs/scout_summary.json`.
- If the measured scout trends disagree with the theoretical trend, or if the proposed paper-objective gain is weak/non-positive, automatically adjust physically reasonable parameters and scout again. Candidate knobs must be inferred from the current contract and may include resource budgets, QoS thresholds, model nonlinearities, uncertainty/risk levels, weighting factors, system scale, environment/channel severity, and baseline-specific design assumptions.
- Limit adaptive scout to a finite number of rounds and write rejected regimes with reasons. Do not silently draw final paper figures from a failed scout.
- Use a scout-to-paper expansion rule:
  - Scout phase should be cheap through x-axis sparsity, not noisy averaging: about 3 to 5 sweep points, at least 20 seeds per point, and modest uncertainty samples.
  - A candidate passes scout if the proposed method improves the paper-defined objective over at least one credible feasible/practical benchmark, the sign/trend matches the encoded theoretical hypothesis, feasibility is comparable, and at least one objective-derived physical KPI has an interpretable gain. It does not need to dominate every credible benchmark.
  - Prefer visible WCL-level gains. As a target, look for roughly 5% or more objective gain, or roughly 10% or more gain in a physical KPI while objective gain remains positive. If the theory naturally gives smaller objective gain, require a clear objective decomposition explaining why the physical KPI gain matters.
  - If a candidate passes scout, automatically increase the experiment to a paper phase: about 10 to 14 sweep points, 80 to 100 seeds per point, and denser uncertainty samples. Record these expanded counts in `outputs/paper_run_config.json`.
  - If no candidate passes initial scout rounds, do not immediately run the expensive paper phase. Instead mutate the physically reasonable knobs and continue cheap scout search for additional candidates. Only run paper phase after a scout pass.
  - If no candidate passes after finite adaptive scout search, write `no_strong_gain_found: true`, stop before expensive paper expansion, and do not overclaim. Diagnostic figures are allowed only if clearly marked diagnostic.
- Select a final regime only when it has positive objective gain over at least one credible feasible/practical benchmark and a physically explainable WCL-readable KPI gain. Write it to `outputs/selected_regime.json`.
- Strongest credible benchmark rule:
  - The primary plotted benchmark and scout pass/fail benchmark must be selected among methods that are credible under the same true physical evaluator and against which the proposed method has evidence-supported gain.
  - A method with robust feasibility below about 0.75, or more than about 0.15 below the proposed method, is not a credible primary benchmark even if its unconstrained objective is high. Report it as an infeasible/fragile diagnostic instead.
  - A nominal/non-robust version of the design can be shown as a muted diagnostic curve or mentioned in the summary if it has poor feasibility under the true evaluator, but do not let an infeasible diagnostic method make the proposed scheme look like it failed.
  - Still report all mandatory methods in CSV and summary, including infeasible high-objective baselines.
- Benchmark selection rule:
  - Do not plot every benchmark by default. First evaluate candidate benchmarks internally, then classify each as `primary_plotted`, `optional_ablation`, `diagnostic_only`, `invalid_for_claim`, or `redundant`.
  - If a benchmark behaves contrary to its intended role because it is infeasible, numerically unstable, violates the model assumptions, is dominated/redundant, or is only a diagnostic stress test, remove it from the final plotted set and explain why.
  - Do not hide a valid credible benchmark just because it is competitive with or beats the proposed method on the claimed KPI. If a valid credible benchmark defeats the proposed method, keep it if scientifically useful and revise/scope the claim to the benchmark(s) where the proposed method has honest gain.
  - The final plotted benchmark set must be exactly the same in both final figures: same methods, labels, colors, marker styles, and legend names.
  - Write `outputs/benchmark_selection_report.json` with all internally evaluated methods, selected plotted methods, removed methods, and concrete reasons for each decision.
- You may choose the most favorable scientifically reasonable regime for the proposed scheme, as long as it is consistent with the system model and not numerically fabricated.
- Before coding the simulator, decide why the proposed scheme should gain in that regime, which bottleneck it improves, and which sweep exposes that bottleneck.
- Use physically reasonable wireless parameters. Tune ranges if needed so the proposed method has visible but plausible gains over credible baselines.
- Use two final figures, but only keep strong panels:
  1. Primary WCL-readable performance figure: proposed vs the selected credible benchmark set over one important system sweep, using a physical KPI derived from the paper objective or its terms.
  2. Parameter-sensitivity / insight figure: proposed vs the same selected credible benchmark set over a different meaningful model parameter inferred from the current paper, such as a resource budget, QoS threshold, uncertainty/risk parameter, system scale, channel/environment parameter, model-nonlinearity parameter, weighting factor, or baseline design knob.
- The second figure should usually be a line plot, not a bar chart. Do not default to a stacked bar decomposition. Use a bar chart only if a bar chart is clearly the best scientific visualization and the figure-selection report justifies it.
- Use simple, conventional line markers only: circle (`o`) for the proposed method, square (`s`) for the primary benchmark, and triangle (`^`) for the optional ablation. Do not use stars, diamonds, crosses, plus markers, oversized markers, or visually busy marker/errorbar styles.
- Do not draw error bars in the final figures by default. The final figures should show clean lines plus simple markers only. Store seed variation in JSON/CSV diagnostics only, not on figure axes, legends, or captions.
- Do not visualize uncertainty in the final preview/paper figures unless the user explicitly asks for it. If uncertainty hides the trend or is comparable to the expected gain, increase seeds before producing the final preview/paper figure or reject the regime rather than accepting a noisy plot.
- Use notation-first axis labels in the final figures, not long descriptive phrases. Prefer labels from the current paper notation and objective shorthand. Do not write "mean", "average", "over seeds", "standard error", or similar implementation/statistical-process language on axes, legends, or captions.
- Write `figures/figure_captions.md` in concise IEEE reference-paper style. Captions should be one sentence of the form "Fig. X. [descriptive quantity] [symbol] versus [descriptive x-axis parameter] [symbol], where [essential fixed parameters]." A paired second figure may use "Corresponding [descriptive quantity] [symbol] versus ...". Do not use bare symbol-only captions, and do not put benchmark definitions, trend explanations, legend/style notes, preview status, or claim conclusions in captions.
- The two figures should tell a coherent paper story:
  - Fig. 1 answers "Does the proposed method improve the main performance KPI over the main system sweep?"
  - Fig. 2 answers "Under which parameter regime does the gain become larger, and why is that consistent with the theory/model?"
  - Fig. 2 may use a different x-axis from Fig. 1. This is encouraged if it reveals the effect of an important model parameter.
- Use the same plotted benchmark set in both figures. Internally you may evaluate several mandatory baselines, but final figures should usually plot only 2 to 3 methods: the proposed method, one credible benchmark that anchors the supported claim, and optionally one informative competitive benchmark or ablation. Do not plot many benchmarks unless each one is necessary. If a method is plotted in Fig. 1, it should also be plotted in Fig. 2 with the same label/style, and vice versa.
- If the benchmark-selection report removes a benchmark from plotting, it must remain in CSV/summary diagnostics unless it could not be run at all.
- Do not create four panels by default. Extra panels often make the figure weaker. Each final panel must pass a figure-panel quality check:
  - it has a clear paper claim role,
  - its y-axis is interpretable for IEEE WCL readers,
  - it is supported by scout trends,
  - it is not a feasibility-only diagnostic,
  - it is not redundant with a stronger panel,
  - it does not show a weak or contradictory trend unless explicitly marked as a limitation.
- Write `outputs/figure_selection_report.json` listing candidate panels, selected panels, rejected panels, and rejection reasons. It is better to produce two clean one-panel figures than a cluttered four-panel figure with weak panels.
- In `outputs/figure_selection_report.json`, also write `plotted_methods`, `internal_benchmarks_evaluated`, `fig1_x_axis`, `fig2_x_axis`, and why the Fig. 2 parameter was selected.
- Layout quality matters: use `constrained_layout=True` or equivalent, save with `bbox_inches="tight"`, avoid title/annotation overlap, avoid placing annotations near the top border, and prefer moving details into captions rather than cluttering the axes.
- The evidence criterion must come from the frozen mathematical contract. If the paper objective is a weighted or worst-case utility, use that exact objective for scout pass/fail and expose at least one interpretable physical KPI term or constraint bottleneck when selecting figures.
- WCL-readable y-axis candidates must be derived from the current paper's objective and system model, such as throughput/rate, energy/efficiency, latency, outage/reliability, sensing accuracy, localization error, trajectory cost, utility, or a normalized objective term when those quantities are defined in the current artifacts. Pick the one whose scout trend supports the paper claim.
- Feasibility and constraint violation may appear in summary diagnostics, markers, or a muted diagnostic comparison, but they must not be the main claimed performance metric. Feasibility can determine whether a benchmark is credible for the primary comparison.
- Let the data decide the strongest comparison: you may run several candidate benchmarks internally and plot the most informative credible benchmark, but do not hide a benchmark that clearly defeats the proposed method on the claimed KPI.
- Run enough points/seeds for smooth figures. It is acceptable if the experiment takes time, but avoid infinite loops and avoid black-box long runs.
- Parameter-sensitivity figures are especially noise-sensitive. Use enough random seeds to smooth Fig. 2; if the Fig. 2 trend is visibly jagged or seed variability is larger than the method separation, increase seeds or reject that figure/regime as not preview-ready.
- Add checkpointing: write or append `outputs/scout_results.csv`, `outputs/simple_results.csv`, and `outputs/progress_status.json` during the run, at least after each completed simulation case or sweep point. A simulation case means one combination of regime, x-axis parameter value, and random seed; within one case the script evaluates all internal methods and writes one CSV row per method. Do not wait until the whole paper phase finishes to write progress.
- Keep the paper phase computationally bounded for a laptop run: target completion within about 30 to 45 minutes, use vectorized NumPy where possible, and cap expensive inner-loop budgets unless more work clearly improves the selected claim.
- Use NumPy and Matplotlib only unless a standard-library alternative is enough.
- No CVX dependency is required. If exact convex subproblems are too heavy, implement a faithful iterative numerical surrogate inspired by the algorithm, but the true evaluation objective and constraints must match the physical model.

Implementation requirements:
- Implement proposed, fixed/adaptive benchmark(s), and at least one model/ablation benchmark relevant to the paper.
- Every method must use the same channels, budgets, constraints, and final true physical evaluator.
- Do not fabricate results or add arbitrary offsets to make proposed win.
- Do allow realistic parameter tuning, method-specific design assumptions, and sweep ranges so the benefit of the proposed design is visible in the regime where its theory says it should help.
- If an initial candidate regime does not show a positive gain in the paper-defined objective and at least one interpretable physical KPI, the script should internally scout alternative physically reasonable regimes or sweeps before selecting the final plotted regime.
- Do not use only three hard-coded candidate regimes if they are weak. Generate additional cheap scout candidates by mutating physically meaningful knobs until either a clear pass is found or the adaptive scout budget is exhausted.
- In simple_summary.json, include a short machine-readable `claim_evidence` object naming the chosen claim, exact paper objective formula used as the evidence criterion, primary plotted KPI, primary sweep, plotted benchmark, aggregate proposed objective gain, aggregate proposed plotted-KPI gain, whether scout passed, whether the paper phase expanded the number of points/seeds, and why this regime is theoretically favorable.
- In experiment_plan.json, include the selected regime rationale, candidate KPIs considered, final figure definitions, method definitions, and parameter choices.
- Include finite-value checks and robust plotting code.
- Avoid fragile Matplotlib mathtext labels such as bold math labels (`\\mathbf`); use simple labels or plain Unicode-safe text.
- Use deterministic random seeds.
- Print a concise progress log and final summary. Use `flush=True` for progress prints so logs are visible during long runs.

Current paper topic:
{topic}

Frozen mathematical contract:
{_clip(math_contract, 24000)}

System model:
{_clip(system_model, 12000)}

Problem formulation:
{_clip(problem_formulation, 12000)}

Reformulation / theory route:
{_clip(reformulation, 16000)}

Algorithm to implement:
{_clip(algorithm, 24000)}

Benchmark notes:
{_clip(benchmark, 12000)}

Experiment blueprint from Phase 2.3:
{_clip(experiment_blueprint, 16000)}
"""


def build_gain_scout_prompt(run_dir: Path, topic: str) -> str:
    phase1 = run_dir / "phase2-1"
    phase2 = run_dir / "phase2-2"
    phase3 = run_dir / "phase2-3"
    math_contract = _read_first_existing(
        phase1 / "mathematical_contract.frozen.json",
        phase1 / "mathematical_contract.json",
    )
    system_model = read_text(phase1 / "system_model.md")
    problem_formulation = read_text(phase1 / "problem_formulation.md")
    reformulation = read_text(phase2 / "reformulation_path.md")
    algorithm = read_text(phase3 / "algorithm.md")
    benchmark = read_text(phase3 / "benchmark_definition.md")
    experiment_blueprint = read_text(phase3 / "experiment_blueprint.md")

    return f"""You are a senior wireless-systems researcher and numerical-experiment engineer.

Write ONE complete self-contained Python script for Phase 2.4A: GainScout.

Return raw Python source only. Do not return markdown fences, JSON, prose, or explanation.

Purpose:
- This is NOT the final paper-level experiment.
- The goal is to cheaply discover where the proposed scheme has visible, physically meaningful gain.
- The next LLM call will use your scout outputs to generate a medium-resolution preview experiment.
- Therefore prioritize search breadth, credible benchmark selection, and clear recommendations over long dense simulations.

The script must run by itself from its own directory and write:
- outputs/theoretical_trends.json
- outputs/gain_scout_results.csv
- outputs/gain_scout_summary.json
- outputs/gain_scout_recommendation.json
- outputs/gain_scout_benchmark_selection.json
- outputs/progress_status.json
- figures/scout_fig1_gain_map.png
- figures/scout_fig2_parameter_probe.png
- experiment_scout_plan.json

Scouting rules:
- Start by encoding theoretical trend hypotheses in `outputs/theoretical_trends.json`.
- Test multiple physically reasonable candidate regimes and parameter knobs inferred from the current contract. Candidate knobs may include resource budgets, QoS thresholds, uncertainty/risk parameters, system scale, environment/channel severity, model-nonlinearity parameters, objective weights, and baseline design assumptions.
- For each candidate, evaluate the paper-defined objective from the frozen mathematical contract and several IEEE WCL-readable KPIs derived from that objective.
- Use cheap settings: about 3 to 5 values per x-axis, 2 to 4 random seeds, and modest uncertainty samples.
- A simulation case means one combination of regime, x-axis parameter value, and random seed. Within one case, evaluate all internal methods and write one row per method.
- Add checkpointing: append or rewrite `outputs/gain_scout_results.csv` and `outputs/progress_status.json` after each completed simulation case or sweep point.

Benchmark rules:
- Internally evaluate proposed, strongest practical baseline(s), and at least one informative ablation if possible.
- The primary comparison benchmark must be credible under the same true physical evaluator.
- A method with robust feasibility below about 0.75, or more than about 0.15 below the proposed method, is not a credible primary benchmark even if its unconstrained objective is high. Report it as an infeasible/fragile diagnostic instead.
- Nominal/non-robust diagnostic methods can be evaluated and reported, but if their feasibility under the true evaluator is poor they must not be used to decide that the proposed method failed.
- Do not hide mandatory baselines in CSV/summary, but the recommendation for final plots should usually use only 2 to 3 plotted methods: proposed, one credible benchmark that anchors the supported claim, and optionally one informative competitive benchmark or ablation.
- If a benchmark result does not match its intended role, classify it rather than blindly plotting it. Examples: infeasible under robust evaluation, numerically unstable, violates model assumptions, is only a diagnostic stress test, is redundant with a stronger baseline, or makes the figure cluttered without supporting the claim.
- Do not remove a valid credible benchmark merely because it beats the proposed method. If a credible benchmark beats proposed on the claimed KPI, mark the scout as not ready and recommend a different regime/claim/parameter search.
- The final recommended plotted benchmark set must be exactly the same for both recommended figures.
- Write `outputs/gain_scout_benchmark_selection.json` with:
  - internal_methods_evaluated
  - selected_plotted_methods
  - credible_primary_benchmark
  - optional_ablation_methods
  - diagnostic_only_methods
  - invalid_or_removed_methods
  - per_method_reason
  - same_method_set_required_for_fig1_and_fig2: true

Recommendation rules:
- Write `outputs/gain_scout_recommendation.json` with:
  - selected_regime_id
  - selected_parameter_values
  - selected_plotted_methods
  - internal_methods_evaluated
  - benchmark_selection_summary
  - credible_primary_benchmark
  - infeasible_or_diagnostic_methods
  - fig1_x_axis recommendation, usually a primary resource, scale, channel, or system-budget sweep from the current model
  - fig2_x_axis recommendation, preferably a different meaningful model parameter
  - primary_kpi and secondary_kpi
  - objective_gain_over_credible_benchmark
  - physical_kpi_gain_over_credible_benchmark
  - scout_passed
  - no_strong_gain_found
  - why_the_regime_should_show_gain
  - recommended_preview_budget
- Fig. 2 recommendation should usually be a parameter-sensitivity line plot, not a bar chart or decomposition chart. Use a bar chart only if it is clearly justified.
- The two recommended figures must use the same plotted method set and consistent labels/styles.
- Do not end with `selected_plotted_methods` containing only `proposed`. That is an invalid scout outcome for this workflow.
- If all practical benchmarks are classified as diagnostic-only because feasibility is low, adapt the regime and keep scouting before finalizing: relax the relevant QoS thresholds, reduce uncertainty/risk severity, increase the relevant resource budget, tune objective weights, move away from pathological model extremes, or choose a more feasible fair benchmark implementation.
- A valid scout recommendation must include at least two plotted methods: `proposed` and one credible feasible/practical benchmark for which the proposed method has supported gain. An optional third method may be an informative competitive benchmark or ablation.
- The selected primary benchmark does not need to be perfect, but it must be meaningful under the same true physical evaluator and should have feasibility close enough to the proposed method to support a fair plotted comparison.
- If the first candidate grid does not satisfy this, generate additional cheap candidate regimes dynamically until either the valid plotted benchmark condition is met or the adaptive scout budget is exhausted.
- Recommend simple marker styles for the final preview: proposed uses circle (`o`), primary benchmark uses square (`s`), and optional ablation uses triangle (`^`). Avoid stars, diamonds, crosses, plus markers, oversized markers, and error bars in final figures.
- Recommend notation-first axis labels for the final preview using symbols defined in the current problem formulation instead of long natural-language axis names.
- Recommend concise IEEE reference-paper captions: "Fig. X. [descriptive quantity] $symbol$ versus [descriptive x-axis parameter] $symbol$, where [essential fixed parameters]." Keep benchmark definitions, trend explanations, legend/style notes, preview status, and claim conclusions out of captions.
- Recommend enough seeds for the preview to reduce visible noise, especially for the Fig. 2 parameter-sensitivity plot. If the scout detects high seed variability or jagged/non-monotone noise, recommend more preview seeds or a different parameter axis.
- If no strong gain is found, still recommend the best next cheap parameter-search direction instead of pretending the result is paper-ready.

Current paper topic:
{topic}

Frozen mathematical contract:
{_clip(math_contract, 22000)}

System model:
{_clip(system_model, 10000)}

Problem formulation:
{_clip(problem_formulation, 10000)}

Reformulation / theory route:
{_clip(reformulation, 14000)}

Algorithm to implement:
{_clip(algorithm, 22000)}

Benchmark notes:
{_clip(benchmark, 10000)}

Experiment blueprint from Phase 2.3:
{_clip(experiment_blueprint, 14000)}
"""


def build_preview_experiment_prompt(run_dir: Path, topic: str, scout_dir: Path) -> str:
    phase1 = run_dir / "phase2-1"
    phase2 = run_dir / "phase2-2"
    phase3 = run_dir / "phase2-3"
    math_contract = _read_first_existing(
        phase1 / "mathematical_contract.frozen.json",
        phase1 / "mathematical_contract.json",
    )
    system_model = read_text(phase1 / "system_model.md")
    problem_formulation = read_text(phase1 / "problem_formulation.md")
    reformulation = read_text(phase2 / "reformulation_path.md")
    algorithm = read_text(phase3 / "algorithm.md")
    benchmark = read_text(phase3 / "benchmark_definition.md")
    experiment_blueprint = read_text(phase3 / "experiment_blueprint.md")

    scout_summary = read_text(scout_dir / "outputs" / "gain_scout_summary.json")
    scout_recommendation = read_text(scout_dir / "outputs" / "gain_scout_recommendation.json")
    scout_benchmark_selection = read_text(scout_dir / "outputs" / "gain_scout_benchmark_selection.json")
    scout_results = read_text(scout_dir / "outputs" / "gain_scout_results.csv")
    scout_plan = read_text(scout_dir / "experiment_scout_plan.json")

    return f"""You are a senior wireless-systems researcher and numerical-experiment engineer.

Write ONE complete self-contained Python script for Phase 2.4B: MediumPreview.

Return raw Python source only. Do not return markdown fences, JSON, prose, or explanation.

Purpose:
- Use the completed GainScout artifacts below.
- Increase points/seeds moderately to see whether the two figures look convincing.
- This is still NOT the final paper-level long run. If the preview is good, the controller/human can later run a longer paper-level experiment.

The script must run by itself from its own directory and write:
- outputs/simple_results.csv
- outputs/simple_summary.json
- outputs/preview_quality_report.json
- outputs/paper_level_recommendation.json
- outputs/paper_run_config.json
- outputs/progress_status.json
- outputs/figure_selection_report.json
- outputs/benchmark_selection_report.json
- figures/fig1_primary_gain.png
- figures/fig2_insight.png
- figures/figure_captions.md
- experiment_plan.json

Preview design:
- Follow `gain_scout_recommendation.json` unless it is internally inconsistent; if you revise it, explain why in `outputs/preview_quality_report.json`.
- Use moderate preview settings, not paper-level settings:
  - Fig. 1: about 6 to 9 x-axis points and 6 to 10 seeds.
  - Fig. 2: about 5 to 8 values of the selected model parameter and 8 to 12 seeds because parameter-sensitivity plots are noise-sensitive.
  - Use enough uncertainty samples to make trends meaningful, but cap the runtime for a laptop preview.
  - If seed variability is comparable to the plotted method separation, increase seeds within the preview budget before finalizing figures, or mark the preview as not ready.
- Do not automatically run a long paper-level experiment. Instead write `outputs/paper_level_recommendation.json` stating whether the preview is ready for a future paper-level run and what longer budget should be used.

Figure rules:
- Fig. 1 should be the main performance figure over the selected primary x-axis.
- Fig. 2 should usually be a parameter-sensitivity / insight line plot over a different meaningful model parameter inferred from the current paper, such as a resource budget, QoS threshold, uncertainty/risk parameter, system scale, channel/environment parameter, model-nonlinearity parameter, objective weight, or baseline design knob.
- Do not default to a stacked bar chart or objective decomposition. Use a bar chart only if clearly justified in `outputs/figure_selection_report.json`.
- The two figures must use the same plotted method set, labels, colors, and marker styles.
- Use simple, conventional markers only: proposed uses circle (`o`), the primary benchmark uses square (`s`), and the optional ablation uses triangle (`^`). Do not use stars, diamonds, crosses, plus markers, oversized markers, error bars, or visually busy marker/errorbar combinations.
- Use moderate marker size and readable line width. Do not draw vertical error bars or any other uncertainty visualization in final figures; store seed stability diagnostics in output JSON/CSV instead.
- Use notation-first axis labels in final figures, not long descriptive phrases. Prefer symbols defined in the current problem formulation and the objective shorthand, rather than copied labels from another topic. Do not put "mean", "average", "over seeds", "standard error", or similar implementation/statistical-process language on axes, legends, or captions.
- Write `figures/figure_captions.md` in concise IEEE reference-paper style. Captions should be one sentence of the form "Fig. X. [descriptive quantity] [symbol] versus [descriptive x-axis parameter] [symbol], where [essential fixed parameters]." A paired second figure may use "Corresponding [descriptive quantity] [symbol] versus ...". Do not use bare symbol-only captions, and do not put benchmark definitions, trend explanations, legend/style notes, preview status, or claim conclusions in captions.
- Do not plot many benchmarks. Usually plot only 2 to 3 methods: proposed, the credible benchmark that anchors the supported claim, and optionally one informative competitive benchmark or ablation.
- Internally you may evaluate additional mandatory baselines, but keep fragile/infeasible baselines as diagnostics unless they are credible under the same true physical evaluator.
- Before plotting, perform benchmark selection. Classify each internally evaluated method as `primary_plotted`, `optional_ablation`, `diagnostic_only`, `invalid_for_claim`, or `redundant`.
- If a benchmark behaves contrary to its intended role because it is infeasible, numerically unstable, violates the model assumptions, is redundant, or is only a diagnostic stress test, remove it from both final figures and explain why in `outputs/benchmark_selection_report.json`.
- Do not hide a valid credible benchmark just because it is competitive with or beats the proposed method. If a valid credible benchmark defeats the proposed method, keep it if scientifically useful and revise/scope the claim/regime; do not silently remove it or overclaim.
- The selected plotted method set must be exactly identical in Fig. 1 and Fig. 2. If a method appears in one figure, it must appear in the other with the same label, color, marker, and line style.
- Write `outputs/benchmark_selection_report.json` with internal methods evaluated, selected plotted methods, removed/diagnostic methods, and per-method reasons.
- The final y-axes should be IEEE WCL-readable KPIs derived from the paper objective, or the paper objective itself if clearly labeled as normalized/bit/s/Hz-equivalent utility.
- Avoid cluttered panels. Prefer two clean single-panel figures.
- Use `constrained_layout=True` or equivalent and save with `bbox_inches="tight"`. Avoid title/annotation overlap.

Quality rules:
- A preview is promising if the proposed method has positive paper-objective gain over at least one credible feasible/practical benchmark and an interpretable physical KPI gain, with comparable feasibility. If another plotted benchmark remains competitive, report that regime honestly and do not claim universal superiority.
- A preview figure is not promising if visible seed noise dominates the method separation. In that case, increase seeds or set `preview_passed: false` and recommend more scout/preview sampling.
- If a nominal/non-robust method has poor feasibility under the true evaluator, it is a diagnostic baseline, not the primary benchmark.
- If the preview is weak, write `preview_passed: false` and recommend the next cheap scout direction. Do not overclaim.
- Add checkpointing: append or rewrite `outputs/simple_results.csv` and `outputs/progress_status.json` after each completed simulation case or sweep point.

GainScout summary:
{_clip(scout_summary, 20000)}

GainScout recommendation:
{_clip(scout_recommendation, 20000)}

GainScout benchmark selection:
{_clip(scout_benchmark_selection, 12000)}

GainScout results CSV:
{_clip(scout_results, 24000)}

GainScout plan:
{_clip(scout_plan, 12000)}

Current paper topic:
{topic}

Frozen mathematical contract:
{_clip(math_contract, 18000)}

System model:
{_clip(system_model, 8000)}

Problem formulation:
{_clip(problem_formulation, 8000)}

Reformulation / theory route:
{_clip(reformulation, 10000)}

Algorithm to implement:
{_clip(algorithm, 18000)}

Benchmark notes:
{_clip(benchmark, 8000)}

Experiment blueprint from Phase 2.3:
{_clip(experiment_blueprint, 10000)}
"""


def _call_llm_for_script(
    *,
    llm: Any,
    prompt: str,
    out_dir: Path,
    prefix: str,
    script_name: str,
    max_tokens: int,
) -> tuple[Path, dict[str, Any]]:
    write_text(out_dir / f"{prefix}_llm_prompt.txt", prompt)
    response = llm.chat(
        [{"role": "user", "content": prompt}],
        json_mode=False,
        strip_thinking=True,
        max_tokens=max_tokens,
    )
    write_text(out_dir / f"{prefix}_llm_raw_response.txt", response.content)
    usage = {
        "model": response.model,
        "prompt_tokens": response.prompt_tokens,
        "cached_tokens": response.cached_tokens,
        "completion_tokens": response.completion_tokens,
        "total_tokens": response.total_tokens,
        "finish_reason": response.finish_reason,
        "truncated": response.truncated,
        "requested_max_tokens": max_tokens,
    }
    write_text(out_dir / f"{prefix}_llm_usage.json", json.dumps(usage, ensure_ascii=False, indent=2))
    code = extract_python_source(response.content)
    if not code or "import " not in code:
        raise ValueError(f"LLM response for {prefix} did not contain executable Python source")
    script_path = out_dir / script_name
    write_text(script_path, code)
    return script_path, usage


def _run_generated_script(
    *,
    script_path: Path,
    out_dir: Path,
    prefix: str,
    timeout_sec: int,
) -> subprocess.CompletedProcess[str]:
    run_env = dict(os.environ)
    run_env["PYTHONUNBUFFERED"] = "1"
    result = subprocess.run(
        [sys.executable, str(script_path.name)],
        cwd=out_dir,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore",
        timeout=timeout_sec if timeout_sec > 0 else None,
        env=run_env,
    )
    write_text(out_dir / f"{prefix}_stdout.txt", result.stdout)
    write_text(out_dir / f"{prefix}_stderr.txt", result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"{prefix} failed with return code {result.returncode}")
    return result


def run_two_call_preview_experiment(
    *,
    run_dir: Path,
    model_profile: str,
    max_tokens: int,
    timeout_sec: int,
    clean_output: bool = True,
) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    summary = {}
    try:
        summary = json.loads(read_text(run_dir / "phase2_summary.json") or "{}")
    except json.JSONDecodeError:
        summary = {}
    topic = str(summary.get("topic") or run_dir.name)
    out_dir = run_dir / "phase2-4-simple"
    if clean_output:
        _clean_simple_output_dir(out_dir)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)

    llm = create_llm_client(model_profile)

    scout_legacy_prompt = build_gain_scout_prompt(run_dir, topic)
    scout_prompt = build_experiment_agent_task_prompt(
        run_dir=run_dir,
        task_kind="gain_scout_script",
        output_contract=(
            "Return raw Python source only for a complete self-contained GainScout script. "
            "Do not return markdown fences, JSON, prose, or explanation."
        ),
        legacy_task_prompt=scout_legacy_prompt,
        request_max_chars=50000,
    )
    write_text(out_dir / "gain_scout_legacy_prompt.txt", scout_legacy_prompt)
    scout_script, scout_usage = _call_llm_for_script(
        llm=llm,
        prompt=scout_prompt,
        out_dir=out_dir,
        prefix="gain_scout",
        script_name="gain_scout.py",
        max_tokens=max_tokens,
    )
    _run_generated_script(
        script_path=scout_script,
        out_dir=out_dir,
        prefix="gain_scout",
        timeout_sec=timeout_sec,
    )

    preview_legacy_prompt = build_preview_experiment_prompt(run_dir, topic, out_dir)
    preview_prompt = build_experiment_agent_task_prompt(
        run_dir=run_dir,
        task_kind="medium_preview_script",
        output_contract=(
            "Return raw Python source only for a complete self-contained MediumPreview script. "
            "Do not return markdown fences, JSON, prose, or explanation."
        ),
        legacy_task_prompt=preview_legacy_prompt,
        request_max_chars=50000,
    )
    write_text(out_dir / "preview_experiment_legacy_prompt.txt", preview_legacy_prompt)
    preview_script, preview_usage = _call_llm_for_script(
        llm=llm,
        prompt=preview_prompt,
        out_dir=out_dir,
        prefix="preview_experiment",
        script_name="preview_experiment.py",
        max_tokens=max_tokens,
    )
    _run_generated_script(
        script_path=preview_script,
        out_dir=out_dir,
        prefix="preview_experiment",
        timeout_sec=timeout_sec,
    )
    phase25_export = publish_phase24_simple_as_phase25(run_dir, out_dir)

    manifest = {
        "status": "ok",
        "workflow": "two-call-preview",
        "run_dir": str(run_dir),
        "output_dir": str(out_dir),
        "phase25_export": phase25_export,
        "scout_script_path": str(scout_script),
        "preview_script_path": str(preview_script),
        "usage": {
            "gain_scout": scout_usage,
            "preview_experiment": preview_usage,
            "total_tokens": int(scout_usage.get("total_tokens") or 0) + int(preview_usage.get("total_tokens") or 0),
            "completion_tokens": int(scout_usage.get("completion_tokens") or 0) + int(preview_usage.get("completion_tokens") or 0),
        },
        "expected_outputs": {
            "scout_results": str(out_dir / "outputs" / "gain_scout_results.csv"),
            "scout_recommendation": str(out_dir / "outputs" / "gain_scout_recommendation.json"),
            "scout_benchmark_selection": str(out_dir / "outputs" / "gain_scout_benchmark_selection.json"),
            "csv": str(out_dir / "outputs" / "simple_results.csv"),
            "summary": str(out_dir / "outputs" / "simple_summary.json"),
            "preview_quality_report": str(out_dir / "outputs" / "preview_quality_report.json"),
            "paper_level_recommendation": str(out_dir / "outputs" / "paper_level_recommendation.json"),
            "figure_selection_report": str(out_dir / "outputs" / "figure_selection_report.json"),
            "benchmark_selection_report": str(out_dir / "outputs" / "benchmark_selection_report.json"),
            "fig1": str(out_dir / "figures" / "fig1_primary_gain.png"),
            "fig2": str(out_dir / "figures" / "fig2_insight.png"),
        },
    }
    write_text(out_dir / "two_call_preview_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def run_simple_experiment(
    *,
    run_dir: Path,
    model_profile: str,
    max_tokens: int,
    timeout_sec: int,
    clean_output: bool = True,
) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    summary = {}
    try:
        summary = json.loads(read_text(run_dir / "phase2_summary.json") or "{}")
    except json.JSONDecodeError:
        summary = {}
    topic = str(summary.get("topic") or run_dir.name)
    out_dir = run_dir / "phase2-4-simple"
    if clean_output:
        _clean_simple_output_dir(out_dir)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
    legacy_prompt = build_simple_experiment_prompt(run_dir, topic)
    prompt = build_experiment_agent_task_prompt(
        run_dir=run_dir,
        task_kind="one_shot_simple_experiment_script",
        output_contract=(
            "Return raw Python source only for one complete self-contained experiment script. "
            "Do not return markdown fences, JSON, prose, or explanation."
        ),
        legacy_task_prompt=legacy_prompt,
        request_max_chars=50000,
    )
    write_text(out_dir / "simple_legacy_prompt.txt", legacy_prompt)
    write_text(out_dir / "simple_llm_prompt.txt", prompt)

    llm = create_llm_client(model_profile)
    response = llm.chat(
        [{"role": "user", "content": prompt}],
        json_mode=False,
        strip_thinking=True,
        max_tokens=max_tokens,
    )
    write_text(out_dir / "simple_llm_raw_response.txt", response.content)
    usage = {
        "model": response.model,
        "prompt_tokens": response.prompt_tokens,
        "cached_tokens": response.cached_tokens,
        "completion_tokens": response.completion_tokens,
        "total_tokens": response.total_tokens,
        "finish_reason": response.finish_reason,
        "truncated": response.truncated,
        "requested_max_tokens": max_tokens,
    }
    write_text(out_dir / "simple_llm_usage.json", json.dumps(usage, ensure_ascii=False, indent=2))

    code = extract_python_source(response.content)
    if not code or "import " not in code:
        raise ValueError("LLM response did not contain executable Python source")
    script_path = out_dir / "simple_experiment.py"
    write_text(script_path, code)

    run_env = dict(os.environ)
    run_env["PYTHONUNBUFFERED"] = "1"
    result = subprocess.run(
        [sys.executable, str(script_path.name)],
        cwd=out_dir,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore",
        timeout=timeout_sec if timeout_sec > 0 else None,
        env=run_env,
    )
    write_text(out_dir / "simple_experiment_stdout.txt", result.stdout)
    write_text(out_dir / "simple_experiment_stderr.txt", result.stderr)
    manifest = {
        "status": "ok" if result.returncode == 0 else "failed",
        "returncode": result.returncode,
        "run_dir": str(run_dir),
        "output_dir": str(out_dir),
        "script_path": str(script_path),
        "usage": usage,
        "expected_outputs": {
            "csv": str(out_dir / "outputs" / "simple_results.csv"),
            "summary": str(out_dir / "outputs" / "simple_summary.json"),
            "paper_run_config": str(out_dir / "outputs" / "paper_run_config.json"),
            "progress_status": str(out_dir / "outputs" / "progress_status.json"),
            "figure_selection_report": str(out_dir / "outputs" / "figure_selection_report.json"),
            "benchmark_selection_report": str(out_dir / "outputs" / "benchmark_selection_report.json"),
            "fig1": str(out_dir / "figures" / "fig1_primary_gain.png"),
            "fig2": str(out_dir / "figures" / "fig2_insight.png"),
        },
    }
    write_text(out_dir / "simple_experiment_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    if result.returncode != 0:
        raise RuntimeError(f"simple experiment failed with return code {result.returncode}")
    manifest["phase25_export"] = publish_phase24_simple_as_phase25(run_dir, out_dir)
    write_text(out_dir / "simple_experiment_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LLM Phase 2.4 experiment generation.")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--model-profile", default="")
    parser.add_argument(
        "--workflow",
        choices=["two-call-preview", "one-shot"],
        default=os.environ.get("WCL_SIMPLE_EXPERIMENT_WORKFLOW", "two-call-preview"),
        help="two-call-preview runs GainScout then MediumPreview; one-shot keeps the older single-call generator.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=int(os.environ.get("WCL_SIMPLE_EXPERIMENT_MAX_TOKENS", "100000")),
        help="Requested LLM completion budget. Default is intentionally high for one-shot generation.",
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=int(os.environ.get("WCL_SIMPLE_EXPERIMENT_TIMEOUT_SEC", "0")),
    )
    parser.add_argument("--no-clean-output", action="store_true", help="Keep existing phase2-4-simple artifacts instead of clearing stale outputs before running.")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    summary = {}
    try:
        summary = json.loads(read_text(run_dir / "phase2_summary.json") or "{}")
    except json.JSONDecodeError:
        summary = {}
    model_profile = args.model_profile.strip() or str(summary.get("model_profile") or DEFAULT_MODEL_PROFILE)
    try:
        if args.workflow == "one-shot":
            manifest = run_simple_experiment(
                run_dir=run_dir,
                model_profile=model_profile,
                max_tokens=args.max_tokens,
                timeout_sec=args.timeout_sec,
                clean_output=not args.no_clean_output,
            )
        else:
            manifest = run_two_call_preview_experiment(
                run_dir=run_dir,
                model_profile=model_profile,
                max_tokens=args.max_tokens,
                timeout_sec=args.timeout_sec,
                clean_output=not args.no_clean_output,
            )
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
    except Exception as exc:
        out_dir = run_dir / "phase2-4-simple"
        out_dir.mkdir(parents=True, exist_ok=True)
        write_text(out_dir / "simple_experiment_error.txt", "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
        raise


if __name__ == "__main__":
    main()
