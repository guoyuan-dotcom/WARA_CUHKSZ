from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Iterable

from wara_core.llm.profiles import DEFAULT_MODEL_PROFILE


ROOT = Path(__file__).resolve().parent
REQUIREMENTS_FILE = ROOT / "requirements.txt"
PHASE1_ROOT = ROOT / "phase1"
PHASE2_ROOT = ROOT / "phase2"
PHASE3_ROOT = ROOT / "phase3"
PHASE1_SCRIPT = PHASE1_ROOT / "run_phase1_pipeline.ps1"
PHASE2_SCRIPT = PHASE2_ROOT / "run_phase2_pipeline.ps1"
PHASE3_SCRIPT = PHASE3_ROOT / "run_phase3_pipeline.ps1"
PHASE1_NATIVE_SCRIPT = PHASE1_ROOT / "scripts" / "run_phase1_pipeline.py"
PHASE2_NATIVE_SCRIPT = PHASE2_ROOT / "scripts" / "run_phase2_pipeline.py"
PHASE3_NATIVE_SCRIPT = PHASE3_ROOT / "scripts" / "run_phase3_pipeline.py"
OUTPUTS_ROOT = ROOT / "outputs"
PAPER_RUNS_ROOT = OUTPUTS_ROOT / "paper_runs"
SHARED_RUNS_DIR = PAPER_RUNS_ROOT / "shared"
PHASE1_RUNS_DIR = PAPER_RUNS_ROOT / "phase1"
PHASE2_RUNS_DIR = PAPER_RUNS_ROOT / "phase2"
PHASE1_BLOCKING_RECOMMENDATIONS = {"revise", "reject", "failed", "fail", "not_ready"}
RUNTIME_MODULE_CHECKS = {
    "PyYAML": "yaml",
    "numpy": "numpy",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "tabulate": "tabulate",
    "scipy": "scipy",
    "cvxpy": "cvxpy",
    "requests": "requests",
    "beautifulsoup4": "bs4",
    "PyPDF2": "PyPDF2",
    "arxiv": "arxiv",
}

PHASE1_PHASE_NAMES = {
    1: "1.1 Identify the Wireless System, Variables, and Metrics",
    2: "1.2 Search Related Wireless Literature",
    3: "1.3 Generate and Select a Research Problem",
    4: "1.4 Freeze and Export the Selected Problem",
}
PHASE1_PHASE_AGENTS = {
    1: "ScoutAgent",
    2: "LiteratureAgent",
    3: "ScoutAgent",
    4: "LiteratureAgent + Controller",
}
PHASE1_MAX_PHASE_STEP = 4

PHASE2_PHASE_NAMES = {
    1: "2.1 Construct the System Model and Optimization Problem",
    2: "2.2 Check Convexity and Choose a Reformulation Route",
    3: "2.3 Specify the Algorithm, Baselines, and Experiment Plan",
    4: "2.4 Implement and Run the Experiment Code",
    5: "2.5 Verify Results and Promote Paper Figures",
    6: "3.1 Draft Problem Formulation and Method Sections",
    7: "3.2 Draft Numerical Results and Supported Claims",
    8: "3.3 Assemble Technical Sections",
    9: "3.4 Construct the Full Manuscript Draft",
    10: "3.5 Review the Full Manuscript",
    11: "3.6 Revise and Export the Final Paper Package",
}
PHASE2_PHASE_AGENTS = {
    1: "FormulationAgent",
    2: "TheoryAgent",
    3: "TheoryAgent",
    4: "ExperimentAgent",
    5: "ValidationAgent",
    6: "WritingAgent",
    7: "AnalysisAgent",
    8: "WritingAgent",
    9: "LiteratureAgent + WritingAgent",
    10: "ReviewAgent",
    11: "RepairAgent + WritingAgent",
}

RAW_LEGACY_METADATA_RE = re.compile(r"^\s{2,}(Run ID|Topic|Output|Mode|From|To):", re.IGNORECASE)
RAW_STDOUT_PREFIX_RE = re.compile(r"^\[(STDOUT|STDERR)\]\s*", re.IGNORECASE)


def normalize_run_id(raw_run_id: str | None) -> str | None:
    value = str(raw_run_id or "").strip()
    if not value:
        return None
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", value):
        raise SystemExit("--run-id must use only letters, numbers, '.', '_' or '-' and be at most 80 characters.")
    return value


def make_connected_run_id(topic: str | None) -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    raw_topic = str(topic or "wara-run").strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "-", raw_topic).strip("-")[:32] or "wara-run"
    suffix = hashlib.sha256(raw_topic.encode("utf-8")).hexdigest()[:6]
    return f"wara-{stamp}-{slug}-{suffix}"


def _normalize_subprocess_line(line: str) -> str:
    return RAW_STDOUT_PREFIX_RE.sub("", str(line or "")).strip()


def _should_surface_subprocess_line(line: str) -> bool:
    text = _normalize_subprocess_line(line)
    if not text:
        return False
    if text.startswith("[WARA]"):
        return True
    if RAW_LEGACY_METADATA_RE.match(text):
        return False
    lowered = text.lower()
    noisy_prefixes = (
        "running phase 1.1-1.4 to:",
        "running phase 2.1-3.5 to:",
    )
    if lowered.startswith(noisy_prefixes):
        return False
    error_markers = (
        "traceback",
        "error",
        "exception",
        "failed",
        "fatal",
    )
    return any(marker in lowered for marker in error_markers)


def looks_like_phase1_run(path: Path) -> bool:
    return looks_like_wara_phase1_run(path)


def looks_like_wara_phase1_run(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "phase1-4" / "phase1_handoff.json").exists()
        and (path / "phase1_controller_manifest.json").exists()
    )


def phase1_handoff_for_run(path: Path) -> Path | None:
    candidates = [
        SHARED_RUNS_DIR / "phase1_tail" / path.name / "phase1_handoff.json",
        ROOT / "wara_runs" / "phase1_tail" / path.name / "phase1_handoff.json",
        path / "phase1_handoff.json",
        path / "phase1-4" / "phase1_handoff.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def read_json_file(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _normalize_gate_value(value: object) -> str:
    return str(value or "").strip().lower().replace(" ", "_").replace("-", "_")


def validate_phase1_quality_gate(phase1_run: Path) -> None:
    if os.environ.get("WARA_ALLOW_WEAK_PHASE1", "").strip() == "1":
        log_progress("[WARA] Phase 1 quality gate bypassed by WARA_ALLOW_WEAK_PHASE1=1")
        return

    handoff_file = phase1_handoff_for_run(phase1_run)
    if handoff_file is not None:
        handoff = read_json_file(handoff_file)
        selected = handoff.get("selected_candidate") if isinstance(handoff, dict) else {}
        missing = [
            key
            for key in [
                "selected_candidate",
                "problem_contract_seed",
                "novelty_contract",
                "proof_contract",
                "validation_contract",
                "kill_criteria",
            ]
            if not handoff.get(key)
        ]
        if missing:
            raise SystemExit("[WARA] WARA Phase 1 handoff gate blocked Phase 2: missing " + ", ".join(missing))
        if not isinstance(selected, dict) or not str(selected.get("title") or "").strip():
            raise SystemExit("[WARA] WARA Phase 1 handoff gate blocked Phase 2: selected_candidate.title is missing")
        source_dir = handoff_file.parent
        topic_focused_literature = read_json_file(source_dir / "topic_focused_literature.json")
        topic_focused_references = (source_dir / "topic_focused_references.bib").read_text(encoding="utf-8", errors="ignore") if (source_dir / "topic_focused_references.bib").exists() else ""
        minimum_reference_target = int(os.environ.get("WARA_PHASE1_REFERENCE_MIN", "12") or 12)
        reference_count = max(
            len(topic_focused_literature.get("references", [])) if isinstance(topic_focused_literature, dict) and isinstance(topic_focused_literature.get("references"), list) else 0,
            len(re.findall(r"@\w+\s*\{", topic_focused_references or "")),
        )
        if reference_count < minimum_reference_target:
            raise SystemExit(
                "[WARA] WARA Phase 1 reference gate blocked Phase 2: "
                f"{reference_count} references < hard target {minimum_reference_target}. "
                "Rerun Phase 1 with external literature search enabled."
            )
        log_progress(f"[WARA] WARA Phase 1 handoff gate passed: references={reference_count}, title={selected.get('title')}")
        return

    raise SystemExit(
        "[WARA] Phase 1 run is missing a phase-native handoff. "
        "Rerun Phase 1 so Phase 2 can consume phase1-4/phase1_handoff.json."
    )


def list_phase1_runs() -> list[Path]:
    if not PHASE1_RUNS_DIR.exists():
        return []
    runs = [path for path in PHASE1_RUNS_DIR.iterdir() if looks_like_phase1_run(path)]
    runs.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    return runs


def list_phase1_run_dirs() -> list[Path]:
    if not PHASE1_RUNS_DIR.exists():
        return []
    runs = [
        path
        for path in PHASE1_RUNS_DIR.iterdir()
        if path.is_dir()
    ]
    runs.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    return runs


def list_phase2_runs() -> list[Path]:
    if not PHASE2_RUNS_DIR.exists():
        return []
    runs = [path for path in PHASE2_RUNS_DIR.iterdir() if path.is_dir() and (path / "phase2_summary.json").exists()]
    runs.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    return runs


def latest_phase1_run() -> Path | None:
    runs = list_phase1_runs()
    return runs[0] if runs else None


def read_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="ignore").lstrip("\ufeff"))
    except json.JSONDecodeError:
        return None


def _hidden_popen_kwargs() -> dict:
    kwargs: dict = {}
    if os.name == "nt":
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = 0
        kwargs["startupinfo"] = startupinfo
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return kwargs


def _python_candidates() -> list[list[str]]:
    candidates: list[list[str]] = []
    configured_python = str(os.environ.get("WARA_PYTHON") or "").strip()
    if configured_python:
        candidates.append([configured_python])
    venv_python = ROOT / ".venv" / "bin" / "python"
    if venv_python.exists():
        candidates.append([str(venv_python)])
    python3 = shutil.which("python3")
    if python3:
        candidates.append([python3])
    for common_python in (
        "/usr/bin/python3",
        "/Library/Developer/CommandLineTools/usr/bin/python3",
    ):
        if Path(common_python).exists():
            candidates.append([common_python])
    candidates.append([sys.executable])
    unique: list[list[str]] = []
    seen: set[str] = set()
    for command in candidates:
        key = " ".join(command)
        if key not in seen:
            unique.append(command)
            seen.add(key)
    return unique


def _missing_runtime_modules(command: list[str]) -> list[str]:
    check_code = (
        "import importlib.util, json\n"
        f"mods = {json.dumps(RUNTIME_MODULE_CHECKS, sort_keys=True)}\n"
        "missing = [name for name, module in mods.items() if importlib.util.find_spec(module) is None]\n"
        "print(json.dumps(missing))\n"
    )
    try:
        result = subprocess.run(
            [*command, "-B", "-c", check_code],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=20,
            check=False,
        )
    except Exception:
        return list(RUNTIME_MODULE_CHECKS)
    try:
        payload = json.loads(result.stdout.strip() or "[]")
    except json.JSONDecodeError:
        return list(RUNTIME_MODULE_CHECKS)
    return [str(item) for item in payload] if isinstance(payload, list) else list(RUNTIME_MODULE_CHECKS)


def _python_command() -> list[str]:
    fallback = _python_candidates()[0]
    for command in _python_candidates():
        if not _missing_runtime_modules(command):
            return command
    return fallback


def _require_runtime_dependencies() -> None:
    selected = _python_command()
    missing = _missing_runtime_modules(selected)
    if missing:
        raise SystemExit(
            "No usable Python runtime found for WARA numerical phases. "
            f"Selected Python: {' '.join(selected)}. "
            f"Missing modules: {', '.join(missing)}. "
            "Run `python3 scripts/start_wara_ui.py --host 127.0.0.1 --port 8765` "
            "to create/use .venv, or set WARA_PYTHON to a Python executable with requirements installed."
        )


def log_progress(message: str) -> None:
    print(message, flush=True)


def _phase_log_line(phase_name: str, agent: str, status: str) -> str:
    return f"[WARA] {phase_name} | Agent: {agent} | {status}"


def _phase1_completed_step_from_artifacts(run_dir: Path) -> int:
    completed = 0
    for step in range(1, PHASE1_MAX_PHASE_STEP + 1):
        decision = read_json(run_dir / f"phase1-{step}" / "decision.json")
        if not isinstance(decision, dict):
            break
        status = str(decision.get("status") or "").strip().lower()
        action = str(decision.get("decision") or decision.get("recommendation") or "").strip().lower()
        if status in {"done", "completed", "complete", "success"} and action not in PHASE1_BLOCKING_RECOMMENDATIONS:
            completed = step
            continue
        break
    return completed


class Phase1ProgressMonitor:
    def __init__(self, before: set[Path]) -> None:
        self.before = {path.resolve() for path in before}
        self.run_dir: Path | None = None
        self.last_completed = 0
        self.last_announced_started = 0

    def tick(self) -> None:
        if self.run_dir is None:
            current = [path.resolve() for path in list_phase1_run_dirs()]
            new_runs = [path for path in current if path not in self.before]
            if new_runs:
                self.run_dir = new_runs[0]
                log_progress(f"[WARA] Phase 1 run detected: {self.run_dir.name}")
                self.last_announced_started = 1
                log_progress(_phase_log_line(PHASE1_PHASE_NAMES.get(1, "1.1 Research Framing"), PHASE1_PHASE_AGENTS.get(1, "ScoutAgent"), "started"))
        if self.run_dir is None:
            return
        checkpoint = read_json(self.run_dir / "checkpoint.json")
        if isinstance(checkpoint, dict):
            try:
                completed = int(checkpoint.get("last_completed_phase") or 0)
            except (TypeError, ValueError):
                completed = 0
        else:
            completed = _phase1_completed_step_from_artifacts(self.run_dir)
        while self.last_completed < completed:
            self.last_completed += 1
            phase_name = PHASE1_PHASE_NAMES.get(self.last_completed, f"phase1.{self.last_completed}")
            agent = PHASE1_PHASE_AGENTS.get(self.last_completed, "Controller")
            log_progress(_phase_log_line(phase_name, agent, "completed"))
        next_phase_step = completed + 1
        if next_phase_step <= PHASE1_MAX_PHASE_STEP and self.last_announced_started < next_phase_step:
            self.last_announced_started = next_phase_step
            phase_name = PHASE1_PHASE_NAMES.get(next_phase_step, f"phase1.{next_phase_step}")
            agent = PHASE1_PHASE_AGENTS.get(next_phase_step, "Controller")
            log_progress(_phase_log_line(phase_name, agent, "started"))


class Phase2ProgressMonitor:
    def __init__(self, before: set[Path]) -> None:
        self.before = {path.resolve() for path in before}
        self.run_dir: Path | None = None
        self.phase_status: dict[str, str] = {}

    def tick(self) -> None:
        if self.run_dir is None:
            current = [path.resolve() for path in list_phase2_runs()]
            new_runs = [path for path in current if path not in self.before]
            if new_runs:
                self.run_dir = new_runs[0]
                log_progress(f"[WARA] Phase run detected: {self.run_dir.name}")
        if self.run_dir is None:
            return
        summary = read_json(self.run_dir / "phase2_summary.json")
        if not isinstance(summary, dict):
            return
        for item in summary.get("phases", []):
            if not isinstance(item, dict):
                continue
            phase_step = str(item.get("phase_step") or "").strip()
            status = str(item.get("status") or "").strip().lower()
            if not phase_step or not status:
                continue
            prev = self.phase_status.get(phase_step)
            if prev == status:
                continue
            self.phase_status[phase_step] = status
            try:
                step_key = int(phase_step)
            except ValueError:
                step_key = 0
            phase_name = PHASE2_PHASE_NAMES.get(step_key, item.get("name") or item.get("phase_id") or f"phase2.{phase_step}")
            agent = PHASE2_PHASE_AGENTS.get(step_key, "Controller")
            if status == "running":
                log_progress(_phase_log_line(phase_name, agent, "started"))
            elif status == "done":
                log_progress(_phase_log_line(phase_name, agent, "completed"))
            elif status == "failed":
                log_progress(_phase_log_line(phase_name, agent, "failed"))


def _stream_subprocess(
    command: list[str],
    *,
    cwd: Path,
    env: dict | None = None,
    monitor: object | None = None,
) -> int:
    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="ignore",
        env=env,
        bufsize=1,
        **_hidden_popen_kwargs(),
    )

    def _reader() -> None:
        assert process.stdout is not None
        for raw_line in process.stdout:
            line = raw_line.rstrip()
            if _should_surface_subprocess_line(line):
                print(_normalize_subprocess_line(line), flush=True)

    reader = threading.Thread(target=_reader, daemon=True, name="wara-stream-reader")
    reader.start()
    try:
        while True:
            if monitor is not None and hasattr(monitor, "tick"):
                try:
                    monitor.tick()
                except Exception as exc:
                    log_progress(f"[WARA] monitor warning: {exc}")
            code = process.poll()
            if code is not None:
                break
            time.sleep(1.0)
    finally:
        if process.stdout is not None:
            process.stdout.close()
        reader.join(timeout=3.0)
    return int(process.returncode if process.returncode is not None else 124)


def run_powershell(
    script: Path,
    args: Iterable[str],
    monitor: object | None = None,
    extra_env: dict[str, str] | None = None,
) -> None:
    powershell_exe = shutil.which("powershell") or shutil.which("pwsh")
    if not powershell_exe:
        raise FileNotFoundError("powershell")
    cmd = [
        powershell_exe,
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(script),
        *list(args),
    ]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    if extra_env:
        env.update(extra_env)
    returncode = _stream_subprocess(cmd, cwd=ROOT, env=env, monitor=monitor)
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, cmd)


def run_python(
    script: Path,
    args: Iterable[str],
    monitor: object | None = None,
    extra_env: dict[str, str] | None = None,
) -> None:
    cmd = [*_python_command(), str(script), *list(args)]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    if extra_env:
        env.update(extra_env)
    returncode = _stream_subprocess(cmd, cwd=ROOT, env=env, monitor=monitor)
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, cmd)


def ensure_phase1_run(topic: str | None, explicit_phase1_run: Path | None, skip_phase1: bool, model_profile: str, run_id: str | None) -> Path:
    if explicit_phase1_run is not None:
        if not looks_like_phase1_run(explicit_phase1_run):
            raise SystemExit(f"Invalid phase1 run: {explicit_phase1_run}")
        validate_phase1_quality_gate(explicit_phase1_run)
        log_progress(f"[WARA] Reusing selected Phase 1 run: {explicit_phase1_run.name}")
        return explicit_phase1_run
    if skip_phase1:
        existing = latest_phase1_run()
        if existing is None:
            raise SystemExit(f"--skip-phase1 was set but no valid Phase 1 run exists under {PHASE1_RUNS_DIR}.")
        validate_phase1_quality_gate(existing)
        log_progress(f"[WARA] Skipping Phase 1 and reusing latest scouting run: {existing.name}")
        return existing
    if not topic:
        raise SystemExit("A topic is required unless --phase1-run or --skip-phase1 is used.")

    log_progress("[WARA] Phase 1 started: research framing / evidence grounding / direction handoff")
    log_progress(f"[WARA] Topic: {topic}")
    log_progress("[WARA] Phase 1 mode: feasibility-first gate with direct research-direction handoff")
    started = time.time()
    before_paths = set(list_phase1_run_dirs())
    monitor = Phase1ProgressMonitor(before_paths)
    if shutil.which("powershell") or shutil.which("pwsh"):
        phase1_args = ["-Topic", topic, "-ModelProfile", model_profile.strip()]
        if run_id:
            phase1_args.extend(["-RunId", run_id])
        run_powershell(
            PHASE1_SCRIPT,
            phase1_args,
            monitor=monitor,
            extra_env={"WARA_PHASE1_LITERATURE_SEARCH": os.environ.get("WARA_PHASE1_LITERATURE_SEARCH", "1")},
        )
    else:
        log_progress("[WARA] PowerShell not found; using native Python Phase 1 runner")
        phase1_args = ["--topic", topic, "--model-profile", model_profile.strip()]
        if run_id:
            phase1_args.extend(["--run-id", run_id])
        run_python(
            PHASE1_NATIVE_SCRIPT,
            phase1_args,
            monitor=monitor,
            extra_env={"WARA_PHASE1_LITERATURE_SEARCH": os.environ.get("WARA_PHASE1_LITERATURE_SEARCH", "1")},
        )
    after_runs = list_phase1_run_dirs()
    new_runs = [
        path
        for path in after_runs
        if path.resolve() not in {item.resolve() for item in before_paths}
    ]
    chosen = None
    if new_runs:
        candidate = sorted(new_runs, key=lambda path: path.stat().st_mtime, reverse=True)[0]
        if looks_like_phase1_run(candidate):
            chosen = candidate
        else:
            decision = read_json_file(candidate / "phase1-4" / "decision.json") or {}
            error = decision.get("error") or "missing phase1-4 handoff artifacts"
            raise SystemExit(
                f"Phase 1 produced a new run but it is not valid for Phase 2 handoff: "
                f"{candidate.name}. Error: {error}"
            )
    if chosen is None and run_id:
        requested = PHASE1_RUNS_DIR / run_id
        if requested.is_dir():
            if looks_like_phase1_run(requested):
                chosen = requested
            else:
                raise SystemExit(f"Requested Phase 1 run is not valid for Phase 2 handoff: {requested.name}")
    if chosen is None:
        chosen = latest_phase1_run()
    if chosen is None:
        raise SystemExit("Phase 1 completed but no valid Phase 1 run was found.")
    validate_phase1_quality_gate(chosen)
    elapsed = time.time() - started
    log_progress(f"[WARA] Phase 1 completed in {elapsed:.1f}s -> {chosen.name}")
    return chosen


def _resolve_phase2_run_after_subprocess(before_paths: set[Path], run_id: str | None) -> Path:
    if run_id:
        requested = PHASE2_RUNS_DIR / run_id
        if requested.exists():
            return requested
    after_paths = set(list_phase2_runs())
    created = [path for path in after_paths - before_paths if path.exists()]
    created.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    if created:
        return created[0]
    latest = list_phase2_runs()
    if latest:
        return latest[0]
    raise SystemExit("Phase 2 completed but no Phase 2 run directory was found.")


def run_phase2(phase1_run: Path, model_profile: str, topic: str | None, run_id: str | None) -> Path:
    handoff_file = phase1_handoff_for_run(phase1_run)
    phase2_args = ["-Phase1Run", str(phase1_run), "-ModelProfile", model_profile.strip()]
    native_phase2_args = ["--model-profile", model_profile.strip()]
    if handoff_file is not None:
        native_phase2_args.extend(["--phase1-handoff", str(handoff_file)])
    else:
        native_phase2_args.extend(["--phase1-run", str(phase1_run)])
    if topic:
        phase2_args.extend(["-Topic", topic.strip()])
        native_phase2_args.extend(["--topic", topic.strip()])
    if run_id:
        phase2_args.extend(["-RunId", run_id])
        native_phase2_args.extend(["--run-id", run_id])
    before_paths = set(list_phase2_runs())
    log_progress(f"[WARA] Phase 2 started using Phase 1 run: {phase1_run.name}")
    monitor = Phase2ProgressMonitor(before_paths)
    phase2_env: dict[str, str] = {}
    if shutil.which("powershell") or shutil.which("pwsh"):
        run_powershell(PHASE2_SCRIPT, phase2_args, monitor=monitor, extra_env=phase2_env)
    else:
        log_progress("[WARA] PowerShell not found; using native Python Phase 2 runner")
        run_python(PHASE2_NATIVE_SCRIPT, native_phase2_args, monitor=monitor, extra_env=phase2_env)
    phase2_run = _resolve_phase2_run_after_subprocess(before_paths, run_id)
    log_progress(f"[WARA] Phase 2 completed -> {phase2_run.name}")
    return phase2_run


def run_phase3(phase2_run: Path, model_profile: str, topic: str | None, run_id: str | None) -> None:
    phase3_args = ["-Phase2Run", str(phase2_run)]
    native_phase3_args = ["--phase2-run", str(phase2_run)]
    if topic:
        phase3_args.extend(["-Topic", topic.strip()])
        native_phase3_args.extend(["--topic", topic.strip()])
    if model_profile:
        phase3_args.extend(["-ModelProfile", model_profile.strip()])
        native_phase3_args.extend(["--model-profile", model_profile.strip()])
    if run_id:
        phase3_args.extend(["-RunId", run_id])
        native_phase3_args.extend(["--run-id", run_id])
    before_paths = set(list_phase2_runs())
    log_progress(f"[WARA] Phase 3 started using Phase 2 run: {phase2_run.name}")
    monitor = Phase2ProgressMonitor(before_paths)
    phase3_env: dict[str, str] = {}
    if shutil.which("powershell") or shutil.which("pwsh"):
        run_powershell(PHASE3_SCRIPT, phase3_args, monitor=monitor, extra_env=phase3_env)
    else:
        log_progress("[WARA] PowerShell not found; using native Python Phase 3 runner")
        run_python(PHASE3_NATIVE_SCRIPT, native_phase3_args, monitor=monitor, extra_env=phase3_env)
    log_progress("[WARA] Phase 3 completed")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the connected WARA Phase 1 -> Phase 2 -> Phase 3 pipeline")
    parser.add_argument("--topic", required=False, help="Research topic passed to Phase 1, Phase 2, and Phase 3")
    parser.add_argument("--phase1-run", required=False, help="Reuse an existing Phase 1 run directory")
    parser.add_argument("--skip-phase1", action="store_true", help="Skip Phase 1 and reuse the latest valid Phase 1 run")
    parser.add_argument("--model-profile", default=DEFAULT_MODEL_PROFILE, help="Phase 1/2/3 model profile")
    parser.add_argument("--run-id", required=False, help="Optional stable run id shared by Phase 1, Phase 2, and Phase 3, e.g. wara001")
    args = parser.parse_args()

    explicit_phase1_run = Path(args.phase1_run).resolve() if args.phase1_run else None
    requested_run_id = normalize_run_id(args.run_id) or make_connected_run_id(args.topic)
    log_progress("[WARA] Three-runner pipeline started")
    log_progress(f"[WARA] Workspace root: {ROOT}")
    log_progress(f"[WARA] Model profile: {args.model_profile.strip()}")
    log_progress(f"[WARA] Run id: {requested_run_id}")
    _require_runtime_dependencies()
    phase1_run = ensure_phase1_run(args.topic.strip() if args.topic else None, explicit_phase1_run, args.skip_phase1, args.model_profile, requested_run_id)
    phase2_run = run_phase2(phase1_run, args.model_profile, args.topic, requested_run_id)
    run_phase3(phase2_run, args.model_profile, args.topic, requested_run_id)

    result = {
        "workspace_root": str(ROOT),
        "phase1_run": str(phase1_run),
        "phase2_run": str(phase2_run),
        "phase2_script": str(PHASE2_SCRIPT),
        "phase3_script": str(PHASE3_SCRIPT),
        "run_id": requested_run_id,
    }
    log_progress("[WARA] Three-runner pipeline finished")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
