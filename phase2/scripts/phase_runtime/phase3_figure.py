from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from pipeline_core import DEFAULT_MODEL_PROFILE, read_json, read_text, write_text
from phase_runtime.prompt_templates import render_prompt_template


PHASE2_FIGURE_ENVIRONMENT = "figure*"
PHASE2_FIGURE_FLOAT_PLACEMENT = "!t"
PHASE2_FIGURE_LATEX_WIDTH = r"0.7\linewidth"
PHASE2_FIGURE_INSERT_AFTER = "Introduction"


def _safe_json_loads(text: str, fallback: Any) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return fallback


def build_phase3_figure_diagram_spec_prompt(
    *,
    topic: str,
    current_paper_brief_json: str,
    system_model_md: str,
    problem_formulation_md: str,
    proposed_solution_md: str,
    numerical_evidence_json: str,
) -> str:
    return render_prompt_template(
        "phase3_figure/diagram_spec.prompt.yaml",
        topic=topic,
        current_paper_brief_json=current_paper_brief_json,
        system_model_md=system_model_md,
        problem_formulation_md=problem_formulation_md,
        proposed_solution_md=proposed_solution_md,
        numerical_evidence_json=numerical_evidence_json,
    )


def build_phase3_figure_diagram_image_prompt(*, diagram_spec_json: str) -> str:
    return render_prompt_template(
        "phase3_figure/diagram_image.prompt.yaml",
        diagram_spec_json=diagram_spec_json,
    )


def _safe_ascii_id(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", str(value or "").strip().lower()).strip("_")
    if not cleaned:
        cleaned = fallback
    if not cleaned:
        return ""
    if cleaned[0].isdigit():
        cleaned = f"n_{cleaned}"
    return cleaned[:48]


def _short_label(value: str, fallback: str, max_words: int = 4) -> str:
    label = re.sub(r"\s+", " ", str(value or "").strip()) or fallback
    words = label.split()
    if len(words) > max_words:
        label = " ".join(words[:max_words])
    return label[:42].strip()


def build_default_phase3_figure_diagram_spec(
    *,
    topic: str,
    system_model_md: str,
    problem_formulation_md: str,
    proposed_solution_md: str,
    numerical_evidence_json: str = "",
) -> dict[str, Any]:
    """Create a contract-neutral fallback when the image-planning LLM is unavailable.

    The fallback intentionally avoids topic-family special cases. If a run needs
    a domain-specific conceptual diagram, the LLM planner must derive it from
    the supplied contracts and the generated spec must pass validation.
    """
    return {
        "diagram_type": "optimization_structure",
        "figure_title_short": "Problem-to-method flow",
        "caption_seed": "Conceptual overview of the formulation, solution method, and evaluated metrics.",
        "nodes": [
            {"id": "paper_scenario", "label": "Wireless Scenario", "role": "environment", "group": "Model"},
            {"id": "decision_variables", "label": "Decision Variables", "role": "variable", "group": "Model"},
            {"id": "objective", "label": "Objective", "role": "objective", "group": "Problem"},
            {"id": "constraints", "label": "Constraints", "role": "constraint", "group": "Problem"},
            {"id": "solver", "label": "Proposed Solver", "role": "algorithm_step", "group": "Method"},
            {"id": "metrics", "label": "Evaluation Metrics", "role": "metric", "group": "Evaluation"},
            {"id": "benchmark", "label": "Benchmark", "role": "metric", "group": "Evaluation"},
        ],
        "edges": [
            {"from": "paper_scenario", "to": "decision_variables", "label": "", "style": "directed"},
            {"from": "decision_variables", "to": "objective", "label": "", "style": "directed"},
            {"from": "decision_variables", "to": "constraints", "label": "", "style": "directed"},
            {"from": "objective", "to": "solver", "label": "", "style": "dependency"},
            {"from": "constraints", "to": "solver", "label": "", "style": "dependency"},
            {"from": "solver", "to": "metrics", "label": "", "style": "directed"},
            {"from": "benchmark", "to": "metrics", "label": "contrast", "style": "dependency"},
        ],
        "containers": [
            {"id": "problem_box", "label": "Optimization problem", "contains": ["objective", "constraints"], "style": "dashed"}
        ],
        "branches": [{"from": "decision_variables", "label": "", "to": ["objective", "constraints"]}],
        "outputs": [{"id": "metrics", "label": "Evaluation Metrics", "source": "solver"}],
        "paper_placement": _default_phase3_figure_paper_placement(),
        "visual_priorities": [
            "show problem-to-method dependency",
            "separate objective and constraints",
            "make evaluation contrast visible",
            "avoid equations inside the figure",
        ],
        "banned_additions": ["unmentioned physical entities", "unmentioned datasets", "unmentioned hardware blocks"],
    }


def _default_phase3_figure_paper_placement() -> dict[str, str]:
    return {
        "insert_after": PHASE2_FIGURE_INSERT_AFTER,
        "figure_environment": PHASE2_FIGURE_ENVIRONMENT,
        "float_placement": PHASE2_FIGURE_FLOAT_PLACEMENT,
        "latex_width": PHASE2_FIGURE_LATEX_WIDTH,
    }


def normalize_phase3_figure_paper_placement(raw: Any) -> dict[str, str]:
    placement = dict(raw) if isinstance(raw, dict) else {}
    environment = str(placement.get("figure_environment") or PHASE2_FIGURE_ENVIRONMENT).strip()
    if environment not in {"figure", "figure*"}:
        environment = PHASE2_FIGURE_ENVIRONMENT
    float_placement = str(placement.get("float_placement") or PHASE2_FIGURE_FLOAT_PLACEMENT).strip()
    if not re.fullmatch(r"[!htbpH]+", float_placement):
        float_placement = PHASE2_FIGURE_FLOAT_PLACEMENT
    latex_width = str(placement.get("latex_width") or PHASE2_FIGURE_LATEX_WIDTH).strip()
    if latex_width != PHASE2_FIGURE_LATEX_WIDTH:
        latex_width = PHASE2_FIGURE_LATEX_WIDTH
    insert_after = str(placement.get("insert_after") or PHASE2_FIGURE_INSERT_AFTER).strip()
    if insert_after.lower() != PHASE2_FIGURE_INSERT_AFTER.lower():
        insert_after = PHASE2_FIGURE_INSERT_AFTER
    return {
        "insert_after": insert_after,
        "figure_environment": environment,
        "float_placement": float_placement,
        "latex_width": latex_width,
    }


def normalize_phase3_figure_diagram_spec(spec: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(spec or {})
    normalized["diagram_type"] = str(normalized.get("diagram_type") or "optimization_structure").strip()
    normalized["figure_title_short"] = _short_label(
        str(normalized.get("figure_title_short") or ""),
        "Conceptual diagram",
        max_words=8,
    )
    normalized["caption_seed"] = re.sub(
        r"\s+",
        " ",
        str(normalized.get("caption_seed") or "Conceptual overview of the proposed framework.").strip(),
    )

    nodes: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(normalized.get("nodes") or []):
        if not isinstance(raw, dict):
            continue
        node_id = _safe_ascii_id(str(raw.get("id") or ""), f"node_{index + 1}")
        if node_id in seen:
            node_id = _safe_ascii_id(f"{node_id}_{index + 1}", f"node_{index + 1}")
        seen.add(node_id)
        nodes.append(
            {
                "id": node_id,
                "label": _short_label(str(raw.get("label") or ""), node_id.replace("_", " ").title()),
                "role": str(raw.get("role") or "other").strip() or "other",
                "group": _short_label(str(raw.get("group") or ""), "", max_words=5),
            }
        )
    normalized["nodes"] = nodes
    node_ids = {node["id"] for node in nodes}

    edges: list[dict[str, str]] = []
    for raw in normalized.get("edges") or []:
        if not isinstance(raw, dict):
            continue
        src = _safe_ascii_id(str(raw.get("from") or ""), "")
        dst = _safe_ascii_id(str(raw.get("to") or ""), "")
        if src not in node_ids or dst not in node_ids or src == dst:
            continue
        edges.append(
            {
                "from": src,
                "to": dst,
                "label": _short_label(str(raw.get("label") or ""), "", max_words=3),
                "style": str(raw.get("style") or "directed").strip() or "directed",
            }
        )
    normalized["edges"] = edges

    containers: list[dict[str, Any]] = []
    for index, raw in enumerate(normalized.get("containers") or []):
        if not isinstance(raw, dict):
            continue
        contains = [_safe_ascii_id(str(item), "") for item in raw.get("contains") or []]
        contains = [item for item in contains if item in node_ids]
        if not contains:
            continue
        containers.append(
            {
                "id": _safe_ascii_id(str(raw.get("id") or ""), f"container_{index + 1}"),
                "label": _short_label(str(raw.get("label") or ""), "Group", max_words=5),
                "contains": contains,
                "style": "dashed" if str(raw.get("style") or "").strip().lower() == "dashed" else "solid",
            }
        )
    normalized["containers"] = containers

    branches: list[dict[str, Any]] = []
    for raw in normalized.get("branches") or []:
        if not isinstance(raw, dict):
            continue
        src = _safe_ascii_id(str(raw.get("from") or ""), "")
        destinations = [_safe_ascii_id(str(item), "") for item in raw.get("to") or []]
        destinations = [item for item in destinations if item in node_ids]
        if src in node_ids and destinations:
            branches.append({"from": src, "label": _short_label(str(raw.get("label") or ""), ""), "to": destinations})
    normalized["branches"] = branches

    outputs: list[dict[str, str]] = []
    for index, raw in enumerate(normalized.get("outputs") or []):
        if not isinstance(raw, dict):
            continue
        source = _safe_ascii_id(str(raw.get("source") or ""), "")
        output_id = _safe_ascii_id(str(raw.get("id") or ""), f"output_{index + 1}")
        if output_id in node_ids or source in node_ids:
            outputs.append(
                {
                    "id": output_id,
                    "label": _short_label(str(raw.get("label") or ""), output_id.replace("_", " ").title()),
                    "source": source,
                }
            )
    normalized["outputs"] = outputs
    normalized["paper_placement"] = normalize_phase3_figure_paper_placement(normalized.get("paper_placement"))
    normalized["visual_priorities"] = [
        _short_label(str(item), "", max_words=8)
        for item in normalized.get("visual_priorities") or []
        if str(item).strip()
    ][:5]
    normalized["banned_additions"] = [
        _short_label(str(item), "", max_words=8)
        for item in normalized.get("banned_additions") or []
        if str(item).strip()
    ][:8]
    return normalized


def validate_phase3_figure_diagram_spec(spec: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    nodes = spec.get("nodes") if isinstance(spec, dict) else None
    if not isinstance(nodes, list) or len(nodes) < 3:
        errors.append("diagram spec must contain at least 3 nodes")
        nodes = []
    if len(nodes) > 14:
        warnings.append("diagram spec has more than 14 nodes; rendering may be crowded")
    node_ids = [str(node.get("id", "")) for node in nodes if isinstance(node, dict)]
    duplicate_ids = sorted({node_id for node_id in node_ids if node_ids.count(node_id) > 1})
    if duplicate_ids:
        errors.append("duplicate node ids: " + ", ".join(duplicate_ids))
    known = set(node_ids)
    for edge in spec.get("edges") or []:
        if not isinstance(edge, dict):
            errors.append("edge entries must be objects")
            continue
        if edge.get("from") not in known or edge.get("to") not in known:
            errors.append(f"edge references unknown node: {edge}")
    forbidden_label_pattern = re.compile(r"[=<>]|\\frac|\\sum|\\int|\\max|\\min")
    for node in nodes:
        if isinstance(node, dict) and forbidden_label_pattern.search(str(node.get("label") or "")):
            errors.append(f"node label appears equation-like: {node.get('id')}")
    placement = normalize_phase3_figure_paper_placement(spec.get("paper_placement") if isinstance(spec, dict) else {})
    if placement["figure_environment"] != PHASE2_FIGURE_ENVIRONMENT:
        errors.append("paper_placement.figure_environment must be figure*")
    if placement["float_placement"] != PHASE2_FIGURE_FLOAT_PLACEMENT:
        errors.append("paper_placement.float_placement must be !t")
    if placement["latex_width"] != PHASE2_FIGURE_LATEX_WIDTH:
        errors.append(r"paper_placement.latex_width must be 0.7\linewidth")
    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "node_count": len(nodes),
        "edge_count": len(spec.get("edges") or []),
    }


def build_phase3_figure_direct_image_prompt(spec: dict[str, Any]) -> str:
    spec = normalize_phase3_figure_diagram_spec(spec)
    nodes = spec.get("nodes") or []
    edges = spec.get("edges") or []
    containers = spec.get("containers") or []
    node_lines = [
        f"- {node['id']}: visible label \"{node['label']}\"; role {node.get('role', 'other')}"
        for node in nodes
    ]
    edge_lines = []
    label_by_id = {str(node["id"]): str(node["label"]) for node in nodes}
    for edge in edges:
        src = label_by_id.get(str(edge.get("from")), str(edge.get("from")))
        dst = label_by_id.get(str(edge.get("to")), str(edge.get("to")))
        label = _image_safe_edge_label(str(edge.get("label") or "").strip())
        edge_label = f"; edge label \"{label}\"" if label else ""
        edge_lines.append(f"- \"{src}\" -> \"{dst}\"; style {edge.get('style', 'directed')}{edge_label}")
    container_lines = []
    for container in containers:
        contained = [label_by_id.get(str(item), str(item)) for item in container.get("contains") or []]
        container_lines.append(
            f"- {container.get('style', 'solid')} rounded container \"{container.get('label', 'Group')}\" contains: "
            + ", ".join(f"\"{item}\"" for item in contained)
        )
    placement = normalize_phase3_figure_paper_placement(spec.get("paper_placement"))
    visual_priorities = spec.get("visual_priorities") or []
    banned_additions = spec.get("banned_additions") or []
    return render_prompt_template(
        "phase3_figure/direct_image.prompt.yaml",
        diagram_type=str(spec.get("diagram_type", "system_architecture")),
        visible_nodes_text="\n".join(node_lines) if node_lines else "- none",
        directed_links_text="\n".join(edge_lines) if edge_lines else "- none",
        containers_text="\n".join(container_lines) if container_lines else "- none",
        visual_priorities_text="\n".join(f"- {item}" for item in visual_priorities) if visual_priorities else "- clean conceptual overview",
        banned_additions_text="\n".join(f"- {item}" for item in banned_additions) if banned_additions else "- none",
        caption_seed=str(spec.get("caption_seed", "Conceptual overview of the proposed framework.")),
        figure_environment=placement["figure_environment"],
        float_placement=placement["float_placement"],
        latex_width=placement["latex_width"],
        insert_after=placement["insert_after"],
    )


def _image_safe_edge_label(label: str) -> str:
    normalized = str(label or "").strip()
    compact = normalized.lower().replace(" ", "").replace("_", "").replace("-", "").replace("−", "")
    if compact in {"wk", "rho", "rhok", "1rho", "ρ", "ρk", "1ρ"}:
        return ""
    return normalized


def render_phase3_figure_image_cli_from_spec(
    spec: dict[str, Any],
    output_dir: Path,
    *,
    figure_id: str = "conceptual_diagram",
    model: str = "gpt-image-2",
    size: str = "1536x1024",
    quality: str = "high",
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    prompt = build_phase3_figure_direct_image_prompt(spec)
    prompt_path = output_dir / "diagram_image_prompt.txt"
    write_text(prompt_path, prompt)
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set; cannot call the image backend")
    cli_path = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))) / "skills/.system/imagegen/scripts/image_gen.py"
    if not cli_path.exists():
        raise RuntimeError(f"image generation CLI not found: {cli_path}")
    png_path = figures_dir / f"{figure_id}.png"
    image_python = os.environ.get("WARA_IMAGEGEN_PYTHON") or sys.executable or "python3"
    cmd = [
        image_python,
        str(cli_path),
        "generate",
        "--model",
        model,
        "--prompt-file",
        str(prompt_path),
        "--size",
        size,
        "--quality",
        quality,
        "--output-format",
        "png",
        "--out",
        str(png_path),
        "--force",
        "--no-augment",
    ]
    result = subprocess.run(cmd, cwd=output_dir, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    write_text(output_dir / "image_backend_stdout.txt", result.stdout)
    write_text(output_dir / "image_backend_stderr.txt", result.stderr)
    if result.returncode != 0:
        raise RuntimeError("image backend failed: " + (result.stderr.strip() or result.stdout.strip()))
    return {
        "png_path": str(png_path),
        "prompt_path": str(prompt_path),
        "backend": "gpt_image_cli",
        "model": model,
        "size": size,
        "quality": quality,
    }


def import_phase3_figure_image_asset(
    *,
    run_dir: Path,
    source_image_path: Path,
    spec: dict[str, Any] | None = None,
    backend: str = "image_model_manual",
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    phase_dir = run_dir / "phase3-figure"
    phase_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = phase_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    source_image_path = Path(source_image_path)
    if not source_image_path.exists():
        raise FileNotFoundError(source_image_path)
    suffix = source_image_path.suffix.lower() or ".png"
    target = figures_dir / f"conceptual_diagram{suffix}"
    if source_image_path.resolve() != target.resolve():
        shutil.copyfile(source_image_path, target)
    if spec is None:
        spec = read_json(phase_dir / "diagram_spec.json") or {}
    spec = normalize_phase3_figure_diagram_spec(spec)
    write_text(phase_dir / "diagram_spec.json", json.dumps(spec, ensure_ascii=False, indent=2))
    asset_report = validate_phase3_figure_assets(target, target)
    critic_report = {
        "ok": bool(asset_report.get("ok")),
        "fallback_spec_used": False,
        "spec_validation": validate_phase3_figure_diagram_spec(spec),
        "asset_validation": asset_report,
        "render_result": {"png_path": str(target), "backend": backend},
    }
    write_text(phase_dir / "critic_report.json", json.dumps(critic_report, ensure_ascii=False, indent=2))
    return write_phase3_figure_manifest(
        phase_dir=phase_dir,
        spec=spec,
        asset_path=target,
        preview_path=target,
        backend=backend,
        ok=critic_report["ok"],
    )


def _role_color(role: str) -> tuple[str, str]:
    role_key = role.lower()
    if role_key in {"entity", "terminal", "environment"}:
        return "#DCEEFF", "#5B8FC7"
    if role_key in {"processing", "algorithm_step"}:
        return "#E4F6E7", "#5FA875"
    if role_key in {"objective", "variable", "constraint"}:
        return "#FFF1C9", "#C79A35"
    if role_key in {"metric", "data"}:
        return "#F3E8FF", "#9A74B5"
    return "#F2F3F5", "#8A9199"


def _auto_positions(spec: dict[str, Any]) -> dict[str, tuple[float, float]]:
    nodes = spec.get("nodes") or []
    edges = [
        edge
        for edge in spec.get("edges") or []
        if isinstance(edge, dict) and str(edge.get("style") or "").lower() != "feedback"
    ]
    node_ids = [str(node.get("id")) for node in nodes if isinstance(node, dict)]
    levels = {node_id: 0 for node_id in node_ids}
    for _ in range(max(1, len(node_ids))):
        changed = False
        for edge in edges:
            src = str(edge.get("from"))
            dst = str(edge.get("to"))
            if src not in levels or dst not in levels:
                continue
            next_level = min(levels[src] + 1, 5)
            if levels[dst] < next_level:
                levels[dst] = next_level
                changed = True
        if not changed:
            break

    grouped: dict[int, list[str]] = {}
    for node_id, level in levels.items():
        grouped.setdefault(level, []).append(node_id)
    ordered_levels = sorted(grouped)
    positions: dict[str, tuple[float, float]] = {}
    if len(ordered_levels) == 1:
        ordered_levels = [ordered_levels[0]]
    for column_index, level in enumerate(ordered_levels):
        x = 0.08 + (0.84 * column_index / max(1, len(ordered_levels) - 1))
        ids = grouped[level]
        for row_index, node_id in enumerate(ids):
            if len(ids) == 1:
                y = 0.52
            else:
                y = 0.80 - (0.56 * row_index / max(1, len(ids) - 1))
            positions[node_id] = (x, y)
    return positions


def _node_dimensions(label: str) -> tuple[float, float]:
    plot_label = _wrap_node_label(label)
    max_line_len = max(len(line) for line in plot_label.splitlines())
    width = min(0.19, max(0.095, 0.052 + 0.0064 * max_line_len))
    height = 0.108 if "\n" in plot_label else 0.085
    return width, height


def _wrap_node_label(label: str) -> str:
    label = str(label or "").strip()
    words = label.split()
    if len(label) <= 16 or len(words) < 2:
        return label
    split_at = max(1, len(words) // 2)
    return " ".join(words[:split_at]) + "\n" + " ".join(words[split_at:])


def render_phase3_figure_diagram_from_spec(
    spec: dict[str, Any],
    output_dir: Path,
    *,
    figure_id: str = "conceptual_diagram",
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    spec = normalize_phase3_figure_diagram_spec(spec)
    validation = validate_phase3_figure_diagram_spec(spec)
    if not validation["ok"]:
        raise ValueError("invalid diagram spec: " + "; ".join(validation["errors"]))

    nodes = spec.get("nodes") or []
    node_by_id = {str(node["id"]): node for node in nodes}
    node_ids = set(node_by_id)
    positions = _auto_positions(spec)
    fig_width = 7.1 if len(nodes) <= 10 else 7.8
    fig_height = 3.25 if len(nodes) <= 10 else 3.7
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor("white")

    for container in spec.get("containers") or []:
        contained = [item for item in container.get("contains") or [] if item in positions]
        if not contained:
            continue
        xs = [positions[item][0] for item in contained]
        ys = [positions[item][1] for item in contained]
        min_x, max_x = max(0.02, min(xs) - 0.08), min(0.98, max(xs) + 0.08)
        min_y, max_y = max(0.04, min(ys) - 0.12), min(0.92, max(ys) + 0.12)
        linestyle = "--" if container.get("style") == "dashed" else "-"
        patch = FancyBboxPatch(
            (min_x, min_y),
            max_x - min_x,
            max_y - min_y,
            boxstyle="round,pad=0.012,rounding_size=0.025",
            linewidth=1.1,
            linestyle=linestyle,
            edgecolor="#B8C0C8",
            facecolor="#FAFBFC",
            alpha=0.75,
            zorder=0,
        )
        ax.add_patch(patch)
        label = str(container.get("label") or "")
        if label:
            ax.text(min_x + 0.014, max_y - 0.026, label, fontsize=7.4, color="#59636E", va="top", zorder=1)

    def draw_edge(edge: dict[str, Any]) -> None:
        src = str(edge.get("from"))
        dst = str(edge.get("to"))
        if src not in positions or dst not in positions:
            return
        style = str(edge.get("style") or "directed").lower()
        src_xy = positions[src]
        dst_xy = positions[dst]
        color = "#58728A"
        linestyle = "-"
        connectionstyle = "arc3,rad=0.0"
        if style == "wireless":
            color = "#4C91D9"
            linestyle = "--"
            connectionstyle = "arc3,rad=0.17"
        elif style == "feedback":
            color = "#A06A4D"
            linestyle = ":"
            connectionstyle = "arc3,rad=-0.24"
        elif style in {"dependency", "branch"}:
            color = "#6E8F64" if style == "dependency" else "#777777"
            connectionstyle = "arc3,rad=0.08"
        ax.annotate(
            "",
            xy=dst_xy,
            xytext=src_xy,
            arrowprops={
                "arrowstyle": "-|>",
                "lw": 1.25,
                "color": color,
                "linestyle": linestyle,
                "shrinkA": 22,
                "shrinkB": 22,
                "mutation_scale": 11,
                "connectionstyle": connectionstyle,
            },
            zorder=2,
        )
        label = str(edge.get("label") or "").strip()
        if label:
            mx = (src_xy[0] + dst_xy[0]) / 2
            my = (src_xy[1] + dst_xy[1]) / 2
            if style == "wireless":
                my += 0.055
            elif style == "feedback":
                my += 0.035
            elif abs(src_xy[1] - dst_xy[1]) < 0.03:
                my += 0.052
            ax.text(
                mx,
                my,
                label,
                fontsize=7.2,
                color=color,
                ha="center",
                va="center",
                bbox={"boxstyle": "round,pad=0.16", "facecolor": "white", "edgecolor": "none", "alpha": 0.86},
                zorder=5,
            )

    for edge in spec.get("edges") or []:
        draw_edge(edge)

    for node in nodes:
        node_id = str(node["id"])
        x, y = positions.get(node_id, (0.5, 0.5))
        label = str(node.get("label") or node_id)
        plot_label = _wrap_node_label(label)
        width, height = _node_dimensions(label)
        fill, stroke = _role_color(str(node.get("role") or "other"))
        patch = FancyBboxPatch(
            (x - width / 2, y - height / 2),
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            linewidth=1.25,
            edgecolor=stroke,
            facecolor=fill,
            zorder=4,
        )
        ax.add_patch(patch)
        ax.text(
            x,
            y,
            plot_label,
            fontsize=7.6 if len(label) <= 16 else 6.8,
            color="#25313B",
            ha="center",
            va="center",
            linespacing=0.9,
            zorder=6,
        )

    png_path = figures_dir / f"{figure_id}.png"
    pdf_path = figures_dir / f"{figure_id}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.08, facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.08, facecolor="white")
    plt.close(fig)
    return {
        "png_path": str(png_path),
        "pdf_path": str(pdf_path),
        "validation": validation,
        "node_count": len(nodes),
        "edge_count": len(spec.get("edges") or []),
    }


def validate_phase3_figure_assets(png_path: Path, pdf_path: Path) -> dict[str, Any]:
    report: dict[str, Any] = {"ok": True, "errors": [], "warnings": []}
    if not png_path.exists() or png_path.stat().st_size < 1024:
        report["ok"] = False
        report["errors"].append("PNG asset is missing or too small")
    if pdf_path != png_path and (not pdf_path.exists() or pdf_path.stat().st_size < 1024):
        report["ok"] = False
        report["errors"].append("PDF asset is missing or too small")
    try:
        from PIL import Image, ImageStat

        with Image.open(png_path) as image:
            width, height = image.size
            report["png_size"] = [width, height]
            if width < 900 or height < 300:
                report["warnings"].append("PNG dimensions are small for a paper figure")
            grayscale = image.convert("L")
            stat = ImageStat.Stat(grayscale)
            if stat.stddev and stat.stddev[0] < 3.0:
                report["ok"] = False
                report["errors"].append("PNG appears nearly blank")
    except Exception as exc:  # pragma: no cover - defensive validation path
        report["warnings"].append(f"could not inspect PNG pixels: {exc}")
    return report


def write_phase3_figure_manifest(
    *,
    phase_dir: Path,
    spec: dict[str, Any],
    asset_path: Path,
    preview_path: Path,
    backend: str,
    ok: bool,
) -> dict[str, Any]:
    phase_dir = Path(phase_dir)
    caption = str(spec.get("caption_seed") or "Conceptual overview of the proposed framework.").strip()
    label = "fig:conceptual_diagram"
    placement = normalize_phase3_figure_paper_placement(spec.get("paper_placement") if isinstance(spec, dict) else {})
    manifest = {
        "phase_name": "phase3.figure_conceptual_diagram",
        "ok": bool(ok),
        "diagram_type": spec.get("diagram_type"),
        "paper_placement": placement,
        "figure_count": 1,
        "primary_asset": str(Path(asset_path).relative_to(phase_dir)),
        "preview_asset": str(Path(preview_path).relative_to(phase_dir)),
        "figures": [
            {
                "id": "conceptual_diagram",
                "label": label,
                "caption": caption,
                "path": str(Path(asset_path).relative_to(phase_dir)),
                "preview_path": str(Path(preview_path).relative_to(phase_dir)),
                "absolute_path": str(Path(asset_path).resolve()),
                "paper_section": "Introduction",
                "placement": "after_introduction",
                "insert_after": placement["insert_after"],
                "figure_environment": placement["figure_environment"],
                "float_placement": placement["float_placement"],
                "width": placement["latex_width"],
                "backend": backend,
                "source_spec": "diagram_spec.json",
            }
        ],
    }
    write_text(phase_dir / "figure_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def _load_phase3_figure_inputs(run_dir: Path) -> dict[str, str]:
    summary_payload = read_json(run_dir / "phase2_summary.json") or {}
    phase3_4_facts = read_json(run_dir / "phase3-4" / "introduction_facts.json") or {}
    phase3_2_manifest = read_json(run_dir / "phase3-2" / "phase3_2_manifest.json") or {}
    current_paper_brief = phase3_4_facts.get("current_paper_brief", {}) if isinstance(phase3_4_facts, dict) else {}
    if not current_paper_brief:
        current_paper_brief = {"topic": summary_payload.get("topic", run_dir.name)}
    return {
        "topic": str(summary_payload.get("topic", run_dir.name)),
        "current_paper_brief_json": json.dumps(current_paper_brief, ensure_ascii=False, indent=2),
        "system_model_md": read_text(run_dir / "phase2-1" / "system_model.md"),
        "problem_formulation_md": read_text(run_dir / "phase2-1" / "problem_formulation.md"),
        "proposed_solution_md": read_text(run_dir / "phase2-3" / "algorithm.md"),
        "numerical_evidence_json": json.dumps(phase3_2_manifest, ensure_ascii=False, indent=2),
    }


def run_phase3_figure_diagram(
    run_dir: Path,
    *,
    model_profile: str | None = None,
    use_llm: bool = False,
    renderer_backend: str = "image",
    allow_structured_fallback: bool = False,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    phase_dir = run_dir / "phase3-figure"
    phase_dir.mkdir(parents=True, exist_ok=True)
    inputs = _load_phase3_figure_inputs(run_dir)
    prompt = build_phase3_figure_diagram_spec_prompt(**inputs)
    write_text(phase_dir / "diagram_spec_prompt.txt", prompt)

    raw_response = ""
    spec: dict[str, Any] | None = None
    if use_llm:
        try:
            from phase_runtime.llm import create_llm_client

            summary_payload = read_json(run_dir / "phase2_summary.json") or {}
            profile = model_profile or str(summary_payload.get("model_profile") or DEFAULT_MODEL_PROFILE)
            llm = create_llm_client(profile)
            response = llm.chat(
                [{"role": "user", "content": prompt}],
                json_mode=True,
                thinking={"type": "enabled"} if profile == "kimi-k2.6-thinking" else None,
                max_tokens=2500,
            )
            raw_response = response.content
            parsed = _safe_json_loads(raw_response, {})
            if isinstance(parsed, dict):
                spec = parsed
        except Exception as exc:
            raw_response = json.dumps({"llm_error": str(exc)}, ensure_ascii=False, indent=2)
    write_text(phase_dir / "diagram_spec_raw_response.txt", raw_response)

    default_spec_used = False
    if not isinstance(spec, dict) or not spec:
        if use_llm:
            raise RuntimeError("Phase 3 figure planner did not return a usable diagram specification.")
        default_spec_used = True
        spec = build_default_phase3_figure_diagram_spec(
            topic=inputs["topic"],
            system_model_md=inputs["system_model_md"],
            problem_formulation_md=inputs["problem_formulation_md"],
            proposed_solution_md=inputs["proposed_solution_md"],
            numerical_evidence_json=inputs["numerical_evidence_json"],
        )
    spec = normalize_phase3_figure_diagram_spec(spec)
    spec_report = validate_phase3_figure_diagram_spec(spec)
    if not spec_report["ok"]:
        if use_llm:
            raise RuntimeError("Phase 3 figure planner returned an invalid diagram specification.")
        default_spec_used = True
        spec = normalize_phase3_figure_diagram_spec(
            build_default_phase3_figure_diagram_spec(
                topic=inputs["topic"],
                system_model_md=inputs["system_model_md"],
                problem_formulation_md=inputs["problem_formulation_md"],
                proposed_solution_md=inputs["proposed_solution_md"],
                numerical_evidence_json=inputs["numerical_evidence_json"],
            )
        )
        spec_report = validate_phase3_figure_diagram_spec(spec)

    write_text(phase_dir / "diagram_spec.json", json.dumps(spec, ensure_ascii=False, indent=2))
    write_text(phase_dir / "diagram_spec_validation.json", json.dumps(spec_report, ensure_ascii=False, indent=2))
    image_prompt = build_phase3_figure_diagram_image_prompt(
        diagram_spec_json=json.dumps(spec, ensure_ascii=False, indent=2)
    )
    write_text(phase_dir / "diagram_image_prompt.txt", image_prompt)

    try:
        if renderer_backend == "structured":
            render_result = render_phase3_figure_diagram_from_spec(spec, phase_dir)
            primary_path = Path(render_result["pdf_path"])
            preview_path = Path(render_result["png_path"])
            backend = "structured_matplotlib"
        elif renderer_backend == "image":
            render_result = render_phase3_figure_image_cli_from_spec(spec, phase_dir)
            primary_path = Path(render_result["png_path"])
            preview_path = Path(render_result["png_path"])
            backend = str(render_result.get("backend") or "image")
        else:
            raise ValueError(f"unknown renderer_backend: {renderer_backend}")
    except Exception as exc:
        write_text(phase_dir / "image_backend_error.txt", str(exc))
        if not allow_structured_fallback:
            raise
        render_result = render_phase3_figure_diagram_from_spec(spec, phase_dir)
        primary_path = Path(render_result["pdf_path"])
        preview_path = Path(render_result["png_path"])
        backend = "structured_matplotlib"

    asset_report = validate_phase3_figure_assets(preview_path, primary_path)
    critic_report = {
        "ok": bool(spec_report.get("ok")) and bool(asset_report.get("ok")),
        "default_spec_used": default_spec_used,
        "spec_validation": spec_report,
        "asset_validation": asset_report,
        "render_result": render_result,
    }
    write_text(phase_dir / "critic_report.json", json.dumps(critic_report, ensure_ascii=False, indent=2))

    return write_phase3_figure_manifest(
        phase_dir=phase_dir,
        spec=spec,
        asset_path=primary_path,
        preview_path=preview_path,
        backend=backend,
        ok=critic_report["ok"],
    )


def find_phase3_figure_asset_for_phase(phase_dir: Path) -> dict[str, Any] | None:
    phase_dir = Path(phase_dir)
    candidate_manifests = [
        phase_dir / "figure_manifest.json",
        phase_dir.parent / "phase3-figure" / "figure_manifest.json",
    ]
    for manifest_path in candidate_manifests:
        manifest = read_json(manifest_path)
        if not isinstance(manifest, dict) or not manifest.get("ok", False):
            continue
        figures = manifest.get("figures") if isinstance(manifest.get("figures"), list) else []
        figure = next((item for item in figures if isinstance(item, dict)), None)
        if not figure:
            continue
        manifest_placement = manifest.get("paper_placement") if isinstance(manifest.get("paper_placement"), dict) else {}
        preferred = str(manifest.get("primary_asset") or figure.get("path") or "").strip()
        fallback = str(manifest.get("preview_asset") or figure.get("preview_path") or "").strip()
        for rel_path in [preferred, fallback]:
            if not rel_path:
                continue
            asset_path = Path(rel_path)
            if not asset_path.is_absolute():
                asset_path = manifest_path.parent / asset_path
            if asset_path.exists():
                return {
                    "source_path": asset_path,
                    "caption": str(figure.get("caption") or "").strip(),
                    "label": str(figure.get("label") or "fig:conceptual_diagram").strip(),
                    "figure_environment": str(
                        figure.get("figure_environment")
                        or manifest_placement.get("figure_environment")
                        or PHASE2_FIGURE_ENVIRONMENT
                    ),
                    "float_placement": str(
                        figure.get("float_placement")
                        or manifest_placement.get("float_placement")
                        or PHASE2_FIGURE_FLOAT_PLACEMENT
                    ),
                    "width": str(
                        figure.get("width")
                        or manifest_placement.get("latex_width")
                        or PHASE2_FIGURE_LATEX_WIDTH
                    ),
                    "insert_after": str(
                        figure.get("insert_after")
                        or manifest_placement.get("insert_after")
                        or PHASE2_FIGURE_INSERT_AFTER
                    ),
                    "manifest_path": manifest_path,
                    "backend": str(figure.get("backend") or ""),
                }
    return None
