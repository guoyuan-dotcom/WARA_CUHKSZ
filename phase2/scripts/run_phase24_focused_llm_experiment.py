from __future__ import annotations

import argparse
import json
import os
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
from run_phase24_simple_llm_experiment import _clean_simple_output_dir, _clip, publish_phase24_simple_as_phase25  # noqa: E402
from phase_runtime.llm import create_llm_client  # noqa: E402


MATPLOTLIB_LABEL_SANITIZER = r'''
# WARA runtime guard: matplotlib mathtext supports only a LaTeX subset.
# Generated IEEE labels may use paper-LaTeX commands such as \underline,
# which are valid in TeX but can crash PNG generation. Sanitize labels at
# render time without changing the simulation data or paper notation.
try:
    import re as _wara_re
    import matplotlib.text as _wara_mtext

    _wara_original_set_text = _wara_mtext.Text.set_text

    def _wara_sanitize_mathtext_label(_wara_text):
        if not isinstance(_wara_text, str):
            return _wara_text
        _wara_out = _wara_text
        _wara_out = _wara_re.sub(r"\\underline\s*\{([^{}]*)\}", r"\1", _wara_out)
        _wara_out = _wara_re.sub(r"\\underline\s+([A-Za-z])", r"\1", _wara_out)
        _wara_out = _wara_re.sub(r"\\operatorname\s*\{([^{}]*)\}", r"\\mathrm{\1}", _wara_out)
        _wara_out = _wara_re.sub(r"\\text\s*\{([^{}]*)\}", r"\\mathrm{\1}", _wara_out)
        return _wara_out

    def _wara_set_text(self, s):
        return _wara_original_set_text(self, _wara_sanitize_mathtext_label(s))

    _wara_mtext.Text.set_text = _wara_set_text
except Exception:
    pass
'''


def _inject_matplotlib_label_sanitizer(code: str) -> str:
    if "WARA runtime guard: matplotlib mathtext" in code:
        return code
    lines = code.splitlines()
    insert_at = 0
    while insert_at < len(lines) and (
        lines[insert_at].startswith("#!") or "coding" in lines[insert_at][:40]
    ):
        insert_at += 1
    while insert_at < len(lines) and lines[insert_at].startswith("from __future__ import "):
        insert_at += 1
    sanitized_lines = lines[:insert_at] + [MATPLOTLIB_LABEL_SANITIZER.strip(), ""] + lines[insert_at:]
    return "\n".join(sanitized_lines) + ("\n" if code.endswith("\n") else "")


def _read_first_existing(*paths: Path) -> str:
    for path in paths:
        text = read_text(path).strip()
        if text:
            return text
    return ""


def _focused_prompt(run_dir: Path, topic: str) -> str:
    phase1 = run_dir / "phase2-1"
    phase2 = run_dir / "phase2-2"
    phase3 = run_dir / "phase2-3"
    phase4 = run_dir / "phase2-4"
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
    validation_plan = _read_first_existing(phase4 / "validation_plan.yaml", phase4 / "validation_plan_candidate_primary.yaml")
    wireless_benchmark_plan = read_text(phase4 / "wireless_benchmark_plan.json")
    experiment_design_contract = read_text(phase4 / "experiment_design_contract.json")

    return f"""You are a senior IEEE WCL wireless-systems simulation engineer.

Write ONE complete self-contained Python script for a focused Phase 2.4 preview.
Return raw Python source only. Do not return markdown fences, JSON, or prose.

The script must run from its own directory and write exactly these final artifacts:
- outputs/simple_results.csv
- outputs/simple_summary.json
- outputs/preview_quality_report.json
- outputs/paper_level_recommendation.json
- outputs/paper_run_config.json
- outputs/figure_selection_report.json
- outputs/benchmark_selection_report.json
- outputs/progress_status.json
- figures/fig1_primary_gain.png
- figures/fig2_insight.png
- figures/figure_captions.md
- experiment_plan.json

Design rules:
- Implement a compact topic-faithful simulator, not a stale template from another topic.
- Use proposed plus at least one credible practical benchmark in both figures. The plotted method set must be identical across both figures.
- Fig. 1: main performance/resource sweep. Fig. 2: different parameter-sensitivity or operating-regime sweep.
- Use the paper objective or a paper-defined physical KPI as y-axis. Do not use feasibility, violation, runtime, or convergence as final y-axis.
- Run a cheap internal scout over several physically meaningful regimes before final plotting. Vary the relevant stress knobs for this topic, such as array size, deployment geometry, uncertainty/noise level, resource budget, mobility/service load, hardware impairment, or density, and select a regime where the proposed algorithm's extra optimized controls or robust adaptation are expected to matter.
- If the first scout shows the proposed method underperforming all credible benchmarks, do not simply finalize that weak regime. First check whether the proposed implementation accidentally omitted an optimized control, used a weaker update rule than the artifact algorithm, used unfair parameters, or chose a regime where the benchmark is theoretically expected to be competitive. Repair the implementation/regime generically and rerun the scout inside the same script.
- Final plots should use about 7-9 x values for Fig. 1, 6-8 x values for Fig. 2, and repeated seeds to reduce noise. A preview may pass when the proposed method has stable, explainable gain over at least one credible retained benchmark on the primary KPI in most x-points of both figures. The proposed method does not need to dominate every plotted benchmark.
- Do not fabricate gains, add arbitrary offsets, or remove a valid strong benchmark only because it is strong. If no honest gain regime exists after the internal repair/scout, still generate all files with preview_passed=false and explain the concrete next scout direction.
- Use notation-first axis labels and simple markers only: proposed circle, primary claim benchmark square, optional competitive/ablation benchmark triangle. Do not draw error bars.
- If a retained benchmark is competitive with or better than the proposed method in part of the range, keep it if it is scientifically meaningful and scope the claim to the benchmark(s) that the proposed method actually improves. Explain the competitive benchmark's operating regime in JSON summaries rather than forcing a universal-superiority claim.
- Use Matplotlib-compatible mathtext in axis labels. Avoid unsupported TeX commands such as \\underline, \\text, \\operatorname, \\bm, or custom macros.
- Captions must be one-sentence IEEE style: "Fig. X. [quantity] [symbol] versus [parameter] [symbol], where [fixed parameters]."
- Keep all JSON/CSV/figure metadata consistent.
- Include in simple_summary.json: plotted_methods, primary_claim, preview_passed, selected_operating_regime, and claim_evidence.

Current paper topic:
{topic}

Experiment design contract:
{_clip(experiment_design_contract, 3500)}

Validation plan:
{_clip(validation_plan, 4500)}

Wireless benchmark plan:
{_clip(wireless_benchmark_plan, 3500)}

Frozen mathematical contract:
{_clip(math_contract, 4500)}

System model:
{_clip(system_model, 2200)}

Problem formulation:
{_clip(problem_formulation, 2600)}

Reformulation/theory route:
{_clip(reformulation, 2600)}

Algorithm to implement:
{_clip(algorithm, 4500)}

Benchmark notes:
{_clip(benchmark, 2200)}

Experiment blueprint:
{_clip(experiment_blueprint, 2600)}
"""


def run_focused_experiment(
    *,
    run_dir: Path,
    model_profile: str,
    max_tokens: int,
    timeout_sec: int,
    clean_output: bool = True,
) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    summary: dict[str, Any] = {}
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

    legacy_prompt = _focused_prompt(run_dir, topic)
    prompt = legacy_prompt
    write_text(out_dir / "focused_legacy_prompt.txt", legacy_prompt)
    write_text(out_dir / "focused_llm_prompt.txt", prompt)

    llm = create_llm_client(model_profile)
    if str(model_profile or "").strip().lower().startswith("openai-"):
        setattr(llm.config, "reasoning_effort", os.environ.get("WARA_FOCUSED_OPENAI_REASONING_EFFORT", "none"))
    response = llm.chat(
        [{"role": "user", "content": prompt}],
        json_mode=False,
        strip_thinking=True,
        max_tokens=max_tokens,
    )
    write_text(out_dir / "focused_llm_raw_response.txt", response.content)
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
    write_text(out_dir / "focused_llm_usage.json", json.dumps(usage, ensure_ascii=False, indent=2))

    code = extract_python_source(response.content)
    if not code or "import " not in code:
        raise ValueError("Focused LLM response did not contain executable Python source")
    code = _inject_matplotlib_label_sanitizer(code)
    script_path = out_dir / "focused_experiment.py"
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
    write_text(out_dir / "focused_experiment_stdout.txt", result.stdout)
    write_text(out_dir / "focused_experiment_stderr.txt", result.stderr)
    manifest = {
        "status": "ok" if result.returncode == 0 else "failed",
        "returncode": result.returncode,
        "run_dir": str(run_dir),
        "output_dir": str(out_dir),
        "script_path": str(script_path),
        "usage": usage,
    }
    write_text(out_dir / "focused_experiment_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    if result.returncode != 0:
        raise RuntimeError(f"focused experiment failed with return code {result.returncode}")
    manifest["phase25_export"] = publish_phase24_simple_as_phase25(run_dir, out_dir)
    write_text(out_dir / "focused_experiment_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run focused LLM Phase 2.4 experiment generation.")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--model-profile", default="")
    parser.add_argument("--max-tokens", type=int, default=int(os.environ.get("WCL_FOCUSED_EXPERIMENT_MAX_TOKENS", "40000")))
    parser.add_argument("--timeout-sec", type=int, default=int(os.environ.get("WCL_FOCUSED_EXPERIMENT_TIMEOUT_SEC", "0")))
    parser.add_argument("--no-clean-output", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    summary: dict[str, Any] = {}
    try:
        summary = json.loads(read_text(run_dir / "phase2_summary.json") or "{}")
    except json.JSONDecodeError:
        summary = {}
    model_profile = args.model_profile.strip() or str(summary.get("model_profile") or DEFAULT_MODEL_PROFILE)
    try:
        manifest = run_focused_experiment(
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
        write_text(out_dir / "focused_experiment_error.txt", "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
        raise


if __name__ == "__main__":
    main()
