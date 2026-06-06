from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from email import policy
from email.parser import BytesParser
from hashlib import sha1
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wara_core.llm.profiles import (  # noqa: E402
    DEFAULT_MODEL_PROFILE,
    get_model_profiles,
    normalize_model_profile,
    profile_id_for_model,
)

APP_DIR = ROOT / "app"
REQUIREMENTS_FILE = ROOT / "requirements.txt"
RUNTIME_DIR = ROOT / "runtime"
UI_RUNS_DIR = RUNTIME_DIR / "ui_runs"
PDF_PREVIEW_DIR = RUNTIME_DIR / "pdf_previews"
PDF_REVIEW_DIR = RUNTIME_DIR / "pdf_reviews"
OUTPUTS_ROOT = ROOT / "outputs" / "paper_runs"
PHASE1_RUNS_DIR = OUTPUTS_ROOT / "phase1"
PHASE2_RUNS_DIR = OUTPUTS_ROOT / "phase2"
FINAL_PAPERS_DIR = OUTPUTS_ROOT / "final_papers"
HISTORY_INDEX = OUTPUTS_ROOT / "history_index.json"
DISPLAY_INDEX = OUTPUTS_ROOT / "run_display_index.json"
PIPELINE_SCRIPT = ROOT / "run_wara_pipeline.py"

UI_RUNS_DIR.mkdir(parents=True, exist_ok=True)
PDF_PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
PDF_REVIEW_DIR.mkdir(parents=True, exist_ok=True)

_active_lock = threading.Lock()
_active_run: dict | None = None

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

REVIEW_MODULE_CHECKS = {
    "PyYAML": "yaml",
    "PyPDF2": "PyPDF2",
    "openai": "openai",
}

DEFAULT_CNY_PER_USD = 7.25

PHASE_ACTIVITY = {
    "1.1": {
        "phase": "Subphase 1.1",
        "title": "Identify the Wireless System, Variables, and Metrics",
        "agent": "ScoutAgent",
        "task": "Frames the wireless scenario, controllable variables, objective, constraints, and metrics.",
    },
    "1.2": {
        "phase": "Subphase 1.2",
        "title": "Search Related Wireless Literature",
        "agent": "LiteratureAgent",
        "task": "Builds topic-specific queries and collects prior models, metrics, baselines, and references.",
    },
    "1.3": {
        "phase": "Subphase 1.3",
        "title": "Generate and Select a Research Problem",
        "agent": "ScoutAgent",
        "task": "Generates candidate wireless optimization problems and selects the direction for technical development.",
    },
    "1.4": {
        "phase": "Subphase 1.4",
        "title": "Freeze and Export the Selected Problem",
        "agent": "LiteratureAgent + Controller",
        "task": "Verifies selected-direction grounding and freezes the problem handoff for Phase 2.",
    },
    "2.1": {
        "phase": "Subphase 2.1",
        "title": "Construct the System Model and Optimization Problem",
        "agent": "FormulationAgent",
        "task": "Writes the wireless system model, variables, objective, constraints, and mathematical contract.",
    },
    "2.2": {
        "phase": "Subphase 2.2",
        "title": "Check Convexity and Choose a Reformulation Route",
        "agent": "TheoryAgent",
        "task": "Diagnoses tractability and selects the exact solver, reformulation, approximation, or heuristic route.",
    },
    "2.3": {
        "phase": "Subphase 2.3",
        "title": "Specify the Algorithm, Baselines, and Experiment Plan",
        "agent": "TheoryAgent",
        "task": "Turns the solution route into an implementable algorithm and validation principle.",
    },
    "2.4": {
        "phase": "Subphase 2.4",
        "title": "Implement and Run the Experiment Code",
        "agent": "ExperimentAgent",
        "task": "Generates experiment design/code and runs quick validation through the fixed harness.",
    },
    "2.5": {
        "phase": "Subphase 2.5",
        "title": "Verify Results and Promote Paper Figures",
        "agent": "ValidationAgent",
        "task": "Checks result tables, logs, metrics, and candidate figures before promoting paper-ready outputs.",
    },
    "3.1": {
        "phase": "Subphase 3.1",
        "title": "Draft Problem Formulation and Method Sections",
        "agent": "WritingAgent",
        "task": "Writes the system-model, optimization-problem, reformulation, and method sections in LaTeX.",
    },
    "3.2": {
        "phase": "Subphase 3.2",
        "title": "Draft Numerical Results and Supported Claims",
        "agent": "AnalysisAgent",
        "task": "Interprets verified figures and maps numerical observations to supported claims.",
    },
    "3.3": {
        "phase": "Subphase 3.3",
        "title": "Assemble Technical Sections",
        "agent": "WritingAgent",
        "task": "Assembles the technical sections and writes the abstract, keywords, and conclusion.",
    },
    "3.4": {
        "phase": "Subphase 3.4",
        "title": "Construct the Full Manuscript Draft",
        "agent": "LiteratureAgent + WritingAgent",
        "task": "Builds the introduction, verified citations, bibliography layer, and full manuscript preview.",
    },
    "3.5": {
        "phase": "Subphase 3.5",
        "title": "Review the Full Manuscript",
        "agent": "ReviewAgent",
        "task": "Reviews equations, algorithm text, figures, citations, abbreviations, and claim consistency.",
    },
    "3.6": {
        "phase": "Subphase 3.6",
        "title": "Revise and Export the Final Paper Package",
        "agent": "RepairAgent + WritingAgent",
        "task": "Applies routed manuscript fixes and exports the final paper package after the quality gate.",
    },
}


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_text(path: Path, default: str = "") -> str:
    if not path.exists():
        return default
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    try:
        return json.loads(_read_text(path).lstrip("\ufeff"))
    except json.JSONDecodeError:
        return None


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _tail_text(path: Path, max_lines: int = 240) -> str:
    lines = _read_text(path).splitlines()
    return "\n".join(lines[-max_lines:]) if lines else ""


def _parse_iso(raw: str | None) -> datetime | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def _format_duration(seconds: float | int | None) -> str:
    if seconds is None:
        return "-"
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def _float_env(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, "") or default)
    except ValueError:
        return default
    return value if value > 0 else default


def _profile_id_from_usage_record(record: dict) -> str:
    explicit = str(record.get("model_profile") or "").strip()
    if explicit:
        return normalize_model_profile(explicit)
    return profile_id_for_model(str(record.get("model") or ""))


def _usage_record_cost(record: dict) -> dict | None:
    profile_id = _profile_id_from_usage_record(record)
    profile = get_model_profiles().get(profile_id, {})
    pricing = profile.get("pricing") if isinstance(profile, dict) else None
    if not isinstance(pricing, dict):
        return None
    prompt = int(record.get("prompt_tokens") or 0)
    cached = int(record.get("cached_tokens") or 0)
    completion = int(record.get("completion_tokens") or 0)
    if prompt <= 0 and cached <= 0 and completion <= 0:
        return None
    uncached = max(prompt - cached, 0)
    amount = (
        uncached * float(pricing.get("input_per_m") or 0)
        + max(cached, 0) * float(pricing.get("cached_input_per_m") or 0)
        + max(completion, 0) * float(pricing.get("output_per_m") or 0)
    ) / 1_000_000
    return {
        "amount": amount,
        "currency": str(pricing.get("currency") or "").upper(),
        "profile_id": profile_id,
    }


def _currency_symbol(currency: str) -> str:
    return {"USD": "$"}.get(str(currency or "").upper(), f"{str(currency or '').upper()} ")


def _format_cost_amount(amount: float | None, currency: str) -> str:
    if amount is None:
        return "-"
    return f"{_currency_symbol(currency)}{amount:.2f}"


def _format_cost_totals(cost_by_currency: dict[str, float]) -> str:
    if not cost_by_currency:
        return "-"
    parts = []
    for currency in sorted(cost_by_currency):
        parts.append(_format_cost_amount(cost_by_currency[currency], currency))
    return " + ".join(parts)


def _to_usd_amount(amount: float, currency: str) -> float:
    normalized = str(currency or "").upper()
    if normalized == "USD":
        return amount
    if normalized == "CNY":
        return amount / _float_env("WARA_CNY_PER_USD", DEFAULT_CNY_PER_USD)
    return 0.0


def _estimate_tokens_from_files(run_dir: Path) -> int:
    total = 0
    for path in run_dir.rglob("*"):
        if not path.is_file():
            continue
        name = path.name.lower()
        if "prompt" not in name and "raw_response" not in name:
            continue
        total += max(1, round(len(_read_text(path).encode("utf-8", errors="ignore")) / 4))
    return total


def _int_from_record(record: dict, key: str) -> int:
    try:
        return int(record.get(key) or 0)
    except (TypeError, ValueError):
        return 0


def _usage_records_from_usage_log(run_id: str) -> list[dict]:
    path = UI_RUNS_DIR / run_id / "usage.jsonl"
    if not path.exists():
        return []
    records: list[dict] = []
    for line in _read_text(path).splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            records.append(item)
    return records


def _usage_records_from_recorded_metadata(run_id: str) -> list[dict]:
    records: list[dict] = []
    phase1_summary = _read_json(PHASE1_RUNS_DIR / run_id / "pipeline_summary.json")
    if isinstance(phase1_summary, dict):
        for item in phase1_summary.get("agent_trace", []):
            if isinstance(item, dict) and int(item.get("total_tokens") or 0) > 0:
                records.append(
                    {
                        "source": "phase1_pipeline_summary",
                        "model": item.get("model") or phase1_summary.get("model_profile") or "",
                        "total_tokens": int(item.get("total_tokens") or 0),
                    }
                )

    phase2_dir = PHASE2_RUNS_DIR / run_id
    for path in sorted(phase2_dir.rglob("*")) if phase2_dir.exists() else []:
        if not path.is_file():
            continue
        name = path.name.lower()
        if not (name.endswith("_llm_usage.json") or name.endswith("raw_response_metadata.json")):
            continue
        payload = _read_json(path)
        if isinstance(payload, dict) and int(payload.get("total_tokens") or 0) > 0:
            item = dict(payload)
            item.setdefault("source", str(path.relative_to(phase2_dir)))
            records.append(item)
    return records


def _summarize_usage(run_id: str) -> dict:
    records = _usage_records_from_usage_log(run_id)
    source = "usage_log" if records else "recorded_metadata"
    if not records:
        records = _usage_records_from_recorded_metadata(run_id)
    prompt = sum(_int_from_record(item, "prompt_tokens") for item in records)
    cached = sum(_int_from_record(item, "cached_tokens") for item in records)
    completion = sum(_int_from_record(item, "completion_tokens") for item in records)
    total = sum(_int_from_record(item, "total_tokens") for item in records)
    if total <= 0:
        total = prompt + completion
    costs = [_usage_record_cost(item) for item in records]
    known_costs = [cost for cost in costs if isinstance(cost, dict)]
    cost_by_currency: dict[str, float] = {}
    cost_usd_total = 0.0
    priced_profiles: dict[str, int] = {}
    for cost in known_costs:
        currency = str(cost.get("currency") or "").upper()
        if not currency:
            continue
        amount = float(cost.get("amount") or 0)
        cost_by_currency[currency] = cost_by_currency.get(currency, 0.0) + amount
        cost_usd_total += _to_usd_amount(amount, currency)
        profile_id = str(cost.get("profile_id") or "").strip()
        if profile_id:
            priced_profiles[profile_id] = priced_profiles.get(profile_id, 0) + 1
    model_names = sorted({str(item.get("model") or item.get("model_profile") or "").strip() for item in records if str(item.get("model") or item.get("model_profile") or "").strip()})
    return {
        "usage_source": source if records else "none",
        "usage_records": len(records),
        "prompt_tokens": prompt,
        "cached_tokens": cached,
        "completion_tokens": completion,
        "total_tokens": total,
        "cost_by_currency": {currency: round(amount, 4) for currency, amount in sorted(cost_by_currency.items())},
        "cost_display": _format_cost_amount(cost_usd_total, "USD") if cost_by_currency else "-",
        "cost_display_usd": _format_cost_amount(cost_usd_total, "USD") if cost_by_currency else "-",
        "cost_usd_total": round(cost_usd_total, 4) if cost_by_currency else None,
        "native_cost_by_currency": {currency: round(amount, 4) for currency, amount in sorted(cost_by_currency.items())},
        "native_cost_display": _format_cost_totals(cost_by_currency),
        "cost_cny": round(cost_by_currency.get("CNY", 0.0), 2) if "CNY" in cost_by_currency else None,
        "cost_usd": round(cost_usd_total, 4) if cost_by_currency else None,
        "cost_currency": "USD" if cost_by_currency else "",
        "priced_records": len(known_costs),
        "priced_profiles": priced_profiles,
        "models": model_names,
    }


def _load_dotenv() -> dict[str, str]:
    env = os.environ.copy()
    dotenv = ROOT / ".env"
    if not dotenv.exists():
        return env
    for raw in _read_text(dotenv).splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in env:
            env[key] = value
    return env


def _read_dotenv_values() -> dict[str, str]:
    dotenv = ROOT / ".env"
    values: dict[str, str] = {}
    if not dotenv.exists():
        return values
    for raw in _read_text(dotenv).splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def _refresh_profile_env() -> None:
    for key, value in _read_dotenv_values().items():
        os.environ.setdefault(key, value)


def _write_dotenv_updates(updates: dict[str, str]) -> None:
    dotenv = ROOT / ".env"
    existing_lines = _read_text(dotenv).splitlines() if dotenv.exists() else []
    seen: set[str] = set()
    output: list[str] = []
    for raw_line in existing_lines:
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            output.append(raw_line)
            continue
        key = stripped.split("=", 1)[0].strip()
        if key in updates:
            output.append(f"{key}={updates[key]}")
            seen.add(key)
        else:
            output.append(raw_line)
    if not output:
        output.extend(
            [
                "# Local WARA settings. Do not commit this file.",
                "",
            ]
        )
    for key, value in updates.items():
        if key not in seen:
            output.append(f"{key}={value}")
    dotenv.write_text("\n".join(output).rstrip() + "\n", encoding="utf-8")
    os.environ.update(updates)


def _provider_api_status(env: dict[str, str]) -> dict[str, dict]:
    return {
        "kimi": {
            "label": "Kimi / Moonshot",
            "api_key_env": "KIMI_API_KEY",
            "fallback_api_key_env": "MOONSHOT_API_KEY",
            "api_key_set": bool(env.get("KIMI_API_KEY") or env.get("MOONSHOT_API_KEY")),
            "base_url_env": "KIMI_BASE_URL",
            "base_url": env.get("KIMI_BASE_URL") or env.get("MOONSHOT_BASE_URL") or "https://api.moonshot.cn/v1",
        },
        "openai": {
            "label": "OpenAI",
            "api_key_env": "OPENAI_API_KEY",
            "api_key_set": bool(env.get("OPENAI_API_KEY")),
            "base_url_env": "OPENAI_BASE_URL",
            "base_url": env.get("OPENAI_BASE_URL") or "https://api.openai.com/v1",
        },
        "deepseek": {
            "label": "DeepSeek",
            "api_key_env": "DEEPSEEK_API_KEY",
            "api_key_set": bool(env.get("DEEPSEEK_API_KEY")),
            "base_url_env": "DEEPSEEK_BASE_URL",
            "base_url": env.get("DEEPSEEK_BASE_URL") or "https://api.deepseek.com",
        },
    }


def _provider_id_for_profile(profile: dict) -> str:
    api_key_env = str(profile.get("api_key_env") or "").upper()
    primary_model = str(profile.get("primary_model") or "").lower()
    if "DEEPSEEK" in api_key_env or "deepseek" in primary_model:
        return "deepseek"
    if "OPENAI" in api_key_env or primary_model.startswith("gpt-"):
        return "openai"
    if "KIMI" in api_key_env or "MOONSHOT" in api_key_env or "kimi" in primary_model:
        return "kimi"
    return ""


def _model_config_payload() -> dict:
    _refresh_profile_env()
    env = _load_dotenv()
    profiles = get_model_profiles()
    profile_items = []
    for profile_id, profile in profiles.items():
        api_key_env = str(profile.get("api_key_env") or "")
        provider_id = _provider_id_for_profile(profile)
        profile_items.append(
            {
                "id": profile_id,
                "label": profile.get("label") or profile_id,
                "provider_id": provider_id,
                "provider": profile.get("provider") or "",
                "primary_model": profile.get("primary_model") or "",
                "reviewer_model": profile.get("reviewer_model") or profile.get("primary_model") or "",
                "api_key_env": api_key_env,
                "api_key_set": bool(env.get(api_key_env)),
                "base_url": profile.get("base_url") or "",
                "pricing": profile.get("pricing") or {},
            }
        )
    return {
        "default_model_profile": DEFAULT_MODEL_PROFILE,
        "selected_model_profile": normalize_model_profile(env.get("WARA_MODEL_PROFILE") or DEFAULT_MODEL_PROFILE),
        "profiles": profile_items,
        "providers": _provider_api_status(env),
    }


def _save_model_config(payload: dict) -> dict:
    updates: dict[str, str] = {}
    requested_profile = str(payload.get("model_profile") or payload.get("selected_model_profile") or "").strip()
    normalized_profile = normalize_model_profile(requested_profile) if requested_profile else ""
    if requested_profile:
        updates["WARA_MODEL_PROFILE"] = normalized_profile
    if normalized_profile:
        profile = get_model_profiles().get(normalized_profile, {})
        provider_id = _provider_id_for_profile(profile)
        provider_mapping = {
            "kimi": ("KIMI_API_KEY", "KIMI_BASE_URL"),
            "openai": ("OPENAI_API_KEY", "OPENAI_BASE_URL"),
            "deepseek": ("DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL"),
        }
        key_env, base_env = provider_mapping.get(provider_id, ("", ""))
        api_key = str(payload.get("api_key") or "").strip()
        base_url = str(payload.get("base_url") or "").strip()
        if api_key and key_env:
            updates[key_env] = api_key
        if base_url and base_env:
            updates[base_env] = base_url
    providers = payload.get("providers")
    if isinstance(providers, dict):
        mapping = {
            "kimi": ("KIMI_API_KEY", "KIMI_BASE_URL"),
            "openai": ("OPENAI_API_KEY", "OPENAI_BASE_URL"),
            "deepseek": ("DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL"),
        }
        for provider_id, (key_env, base_env) in mapping.items():
            item = providers.get(provider_id)
            if not isinstance(item, dict):
                continue
            api_key = str(item.get("api_key") or "").strip()
            base_url = str(item.get("base_url") or "").strip()
            if api_key:
                updates[key_env] = api_key
            if base_url:
                updates[base_env] = base_url
    if updates:
        _write_dotenv_updates(updates)
    return _model_config_payload()


def _require_profile_api_key(model_profile: str) -> None:
    _refresh_profile_env()
    profile_id = normalize_model_profile(model_profile)
    profile = get_model_profiles().get(profile_id, {})
    api_key_env = str(profile.get("api_key_env") or "").strip()
    env = _load_dotenv()
    if api_key_env and not env.get(api_key_env):
        raise ValueError(f"missing {api_key_env} for model profile {profile_id}")


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


def _missing_modules(command: list[str], modules: dict[str, str]) -> list[str]:
    check_code = (
        "import importlib.util, json\n"
        f"mods = {json.dumps(modules, sort_keys=True)}\n"
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
        return list(modules)
    try:
        payload = json.loads(result.stdout.strip() or "[]")
    except json.JSONDecodeError:
        return list(modules)
    return [str(item) for item in payload] if isinstance(payload, list) else list(modules)


def _missing_runtime_modules(command: list[str]) -> list[str]:
    return _missing_modules(command, RUNTIME_MODULE_CHECKS)


def _missing_review_modules(command: list[str]) -> list[str]:
    return _missing_modules(command, REVIEW_MODULE_CHECKS)


def _python_command() -> list[str]:
    fallback = _python_candidates()[0]
    for command in _python_candidates():
        if not _missing_runtime_modules(command):
            return command
    return fallback


def _review_python_command() -> list[str]:
    fallback = _python_candidates()[0]
    for command in _python_candidates():
        if not _missing_review_modules(command):
            return command
    return fallback


def _runtime_environment_report() -> dict:
    candidates = []
    for command in _python_candidates():
        candidates.append(
            {
                "command": command,
                "missing_modules": _missing_runtime_modules(command),
            }
        )
    selected = _python_command()
    selected_missing = _missing_runtime_modules(selected)
    return {
        "selected_python": selected,
        "selected_missing_modules": selected_missing,
        "candidates": candidates,
        "requirements_file": str(REQUIREMENTS_FILE),
        "setup_command": "python3 scripts/start_wara_ui.py --host 127.0.0.1 --port 8765",
    }


def _require_runtime_dependencies() -> None:
    report = _runtime_environment_report()
    missing = report.get("selected_missing_modules") or []
    if missing:
        raise RuntimeError(
            "No usable Python runtime found for WARA numerical phases. "
            f"Missing modules: {', '.join(str(item) for item in missing)}. "
            "Run `python3 scripts/start_wara_ui.py --host 127.0.0.1 --port 8765` "
            "to create/use .venv, or set WARA_PYTHON to a Python executable with requirements installed."
        )


def _require_review_dependencies() -> None:
    command = _review_python_command()
    missing = _missing_review_modules(command)
    if missing:
        raise RuntimeError(
            "No usable Python runtime found for the PDF review agent. "
            f"Missing modules: {', '.join(str(item) for item in missing)}. "
            "Run `python3 scripts/start_wara_ui.py --host 127.0.0.1 --port 8765` "
            "to create/use .venv, or set WARA_PYTHON to a Python executable with requirements installed."
        )


def _sanitize_run_id(topic: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    stem = re.sub(r"[^A-Za-z0-9]+", "-", topic.strip().lower()).strip("-")
    stem = stem[:32] or "wara-run"
    return f"ui-{stamp}-{stem}"


def _phase1_handoff_path(run_dir: Path) -> Path | None:
    for candidate in [
        run_dir / "phase1-4" / "phase1_handoff.json",
        OUTPUTS_ROOT / "shared" / "phase1_tail" / run_dir.name / "phase1_handoff.json",
    ]:
        if candidate.exists():
            return candidate
    return None


def _topic_from_phase1(run_dir: Path) -> str:
    handoff = _read_json(_phase1_handoff_path(run_dir) or Path("__missing__"))
    if isinstance(handoff, dict):
        selected = handoff.get("selected_candidate")
        if isinstance(selected, dict) and selected.get("title"):
            return str(selected.get("title"))
        if handoff.get("source_topic"):
            return str(handoff.get("source_topic"))
    intake = _read_json(run_dir / "phase1-1" / "topic_intake.json")
    if isinstance(intake, dict):
        return str(intake.get("topic") or intake.get("source_topic") or run_dir.name)
    return run_dir.name


def _list_phase1_runs() -> list[Path]:
    if not PHASE1_RUNS_DIR.exists():
        return []
    runs = [path for path in PHASE1_RUNS_DIR.iterdir() if path.is_dir()]
    runs.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    return runs


def _list_phase2_runs() -> list[Path]:
    candidates: dict[str, Path] = {}
    if PHASE2_RUNS_DIR.exists():
        for path in PHASE2_RUNS_DIR.iterdir():
            if path.is_dir():
                candidates[path.name] = path
    if FINAL_PAPERS_DIR.exists():
        for path in FINAL_PAPERS_DIR.iterdir():
            if path.is_dir() and path.name not in candidates:
                candidates[path.name] = path
    history_payload = _read_json(HISTORY_INDEX)
    if isinstance(history_payload, dict) and isinstance(history_payload.get("runs"), list):
        ordered: list[Path] = []
        seen: set[str] = set()
        for item in history_payload.get("runs", []):
            run_id = str(item.get("run_id") if isinstance(item, dict) else item or "").strip()
            path = candidates.get(run_id)
            if run_id and path and run_id not in seen:
                ordered.append(path)
                seen.add(run_id)
        live_runs = [
            path
            for run_id, path in candidates.items()
            if run_id not in seen and run_id.startswith("ui-")
        ]
        live_runs.sort(key=lambda item: item.stat().st_mtime, reverse=True)
        return live_runs + ordered
    runs = list(candidates.values())
    runs.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    return runs


def _history_run_ids() -> list[str]:
    payload = _read_json(HISTORY_INDEX)
    if not isinstance(payload, dict) or not isinstance(payload.get("runs"), list):
        return []
    run_ids: list[str] = []
    for item in payload.get("runs", []):
        run_id = str(item.get("run_id") if isinstance(item, dict) else item or "").strip()
        if run_id and run_id not in run_ids:
            run_ids.append(run_id)
    return run_ids


def _numbered_run_universe() -> list[str]:
    candidates: dict[str, Path] = {}
    for root in [UI_RUNS_DIR, PHASE2_RUNS_DIR, FINAL_PAPERS_DIR]:
        if not root.exists():
            continue
        for path in root.iterdir():
            if path.is_dir() and path.name.startswith("ui-"):
                candidates[path.name] = path
    ordered = _history_run_ids()
    remaining = [item for item in candidates.items() if item[0] not in ordered]
    remaining.sort(key=lambda item: item[1].stat().st_mtime)
    ordered.extend(run_id for run_id, _ in remaining)
    return ordered


def _display_index_payload() -> dict:
    payload = _read_json(DISPLAY_INDEX)
    if isinstance(payload, dict) and isinstance(payload.get("numbers"), dict):
        numbers = {
            str(run_id): int(number)
            for run_id, number in payload.get("numbers", {}).items()
            if str(run_id).strip()
        }
    else:
        numbers = {}

    original_numbers = dict(numbers)
    universe = _numbered_run_universe()
    keep = set(universe)
    numbers = {run_id: number for run_id, number in numbers.items() if run_id in keep}
    changed = numbers != original_numbers
    has_gaps = sorted(numbers.values()) != list(range(1, len(numbers) + 1))
    if changed or has_gaps:
        numbers = {}
    next_number = max(numbers.values(), default=0) + 1
    for run_id in universe:
        if run_id not in numbers:
            numbers[run_id] = next_number
            next_number += 1
            changed = True

    result = {"version": 1, "next_number": next_number, "numbers": numbers}
    if changed:
        _write_json(DISPLAY_INDEX, result)
    return result


def _display_id_for_run(run_id: str) -> str:
    payload = _display_index_payload()
    numbers = payload.get("numbers") if isinstance(payload, dict) else {}
    if isinstance(numbers, dict) and run_id in numbers:
        try:
            return f"WARA{int(numbers[run_id]):03d}"
        except (TypeError, ValueError):
            pass
    next_number = int(payload.get("next_number") or 1) if isinstance(payload, dict) else 1
    numbers = dict(numbers or {})
    numbers[run_id] = next_number
    _write_json(DISPLAY_INDEX, {"version": 1, "next_number": next_number + 1, "numbers": numbers})
    return f"WARA{next_number:03d}"


def _final_package_dir(run_id: str) -> Path:
    return FINAL_PAPERS_DIR / run_id


def _phase2_run_dir(run_id: str) -> Path:
    return PHASE2_RUNS_DIR / run_id


def _quality_payload(run_id: str, phase2_run: Path) -> dict:
    for path in [
        _final_package_dir(run_id) / "post_revision_full_paper_quality_gate.json",
        phase2_run / "phase3-6" / "post_revision_full_paper_quality_gate.json",
    ]:
        payload = _read_json(path)
        if isinstance(payload, dict):
            return payload
    return {}


def _preview_pdf(run_id: str, phase2_run: Path) -> Path | None:
    for path in [
        _final_package_dir(run_id) / "paper.pdf",
        phase2_run / "phase3-6" / "revised_full_paper_preview.pdf",
        phase2_run / "phase3-6" / "paper.pdf",
    ]:
        if path.exists():
            return path
    return None


def _phase_status(summary: dict) -> str:
    phases = summary.get("phases") if isinstance(summary.get("phases"), list) else []
    statuses = [str(item.get("status") or "").lower() for item in phases if isinstance(item, dict)]
    legacy_gate_stop = bytes([98, 108, 111, 99, 107, 101, 100]).decode()
    if any(status == "running" for status in statuses):
        return "running"
    if any(status in {"failed", legacy_gate_stop, "error", "needs_review", "needs review"} for status in statuses):
        return "needs_review"
    legacy_completion = bytes([98, 101, 115, 116, 95, 101, 102, 102, 111, 114, 116, 95, 100, 111, 110, 101]).decode()
    if statuses and statuses[-1] in {"done", legacy_completion}:
        return "completed"
    return "idle"


def _public_status(raw_status: object) -> str:
    value = str(raw_status or "idle").strip().lower().replace("-", "_").replace(" ", "_")
    legacy_completion = bytes([98, 101, 115, 116, 95, 101, 102, 102, 111, 114, 116, 95, 100, 111, 110, 101]).decode()
    legacy_gate_stop = bytes([98, 108, 111, 99, 107, 101, 100]).decode()
    if value in {"done", "completed", "complete", "success", "succeeded", legacy_completion}:
        return "completed"
    if value in {"running", "active", "in_progress", "in_progress"}:
        return "running"
    if value in {"failed", legacy_gate_stop, "error", "exception"}:
        return "needs_review"
    if value in {"ready", "waiting"}:
        return "ready"
    return value or "idle"


def _sanitize_public_text(text: str) -> str:
    sanitized = text
    legacy_completion = bytes([98, 101, 115, 116, 95, 101, 102, 102, 111, 114, 116, 95, 100, 111, 110, 101]).decode()
    legacy_repair_label = bytes([98, 101, 115, 116, 95, 101, 102, 102, 111, 114, 116]).decode()
    legacy_gate_stop = bytes([98, 108, 111, 99, 107, 101, 100]).decode()
    replacements = [
        (legacy_completion, "completed"),
        (legacy_repair_label.replace("_", "[-_ ]"), "bounded-repair"),
        (rf"\b{legacy_gate_stop}\b", "needs review"),
    ]
    for pattern, replacement in replacements:
        sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)
    return sanitized


def _public_phases(summary: dict) -> list[dict]:
    phases = summary.get("phases") if isinstance(summary.get("phases"), list) else []
    public: list[dict] = []
    for item in phases:
        if not isinstance(item, dict):
            continue
        phase = dict(item)
        phase["status"] = _public_status(phase.get("status"))
        public.append(phase)
    return public


def _current_phase(summary: dict) -> str:
    phases = summary.get("phases") if isinstance(summary.get("phases"), list) else []
    running = next((item for item in phases if isinstance(item, dict) and str(item.get("status")).lower() == "running"), None)
    if running:
        return str(running.get("name") or running.get("phase_id") or "running")
    needs_review = next((item for item in phases if isinstance(item, dict) and _public_status(item.get("status")) == "needs_review"), None)
    if needs_review:
        return str(needs_review.get("name") or needs_review.get("phase_id") or "needs review")
    done = [item for item in phases if isinstance(item, dict) and _public_status(item.get("status")) == "completed"]
    if done:
        return str(done[-1].get("name") or done[-1].get("phase_id") or "done")
    return "Waiting"


def _build_telemetry(run_id: str, run_dir: Path, started_at: str | None = None) -> dict:
    start = _parse_iso(started_at)
    file_mtimes = []
    if run_dir.exists():
        file_mtimes = [path.stat().st_mtime for path in run_dir.rglob("*") if path.exists()]
    if start is None and file_mtimes:
        start = datetime.fromtimestamp(min(file_mtimes), timezone.utc)
    elif start is None and run_dir.exists():
        start = datetime.fromtimestamp(run_dir.stat().st_mtime, timezone.utc)
    end = datetime.now(timezone.utc)
    if file_mtimes:
        end = datetime.fromtimestamp(max(file_mtimes), timezone.utc)
    runtime = max((end - start).total_seconds(), 0) if start else None
    usage = _summarize_usage(run_id)
    return {
        "runtime_seconds": runtime,
        "runtime_hms": _format_duration(runtime),
        **usage,
    }


def _build_phase2_report(run_id: str) -> dict:
    phase2_run = _phase2_run_dir(run_id)
    final_dir = _final_package_dir(run_id)
    summary = _read_json(phase2_run / "phase2_summary.json")
    if not isinstance(summary, dict):
        summary = _read_json(final_dir / "phase2_summary.json")
    if not isinstance(summary, dict):
        summary = {"phases": []}
    quality = _quality_payload(run_id, phase2_run)
    pdf = _preview_pdf(run_id, phase2_run)
    topic = str(summary.get("topic") or run_id)
    handoff = _read_json(phase2_run / "phase1_handoff_manifest.json")
    if isinstance(handoff, dict) and handoff.get("final_title"):
        topic = str(handoff.get("final_title"))
    status = _phase_status(summary)
    model_profile = normalize_model_profile(str(summary.get("model_profile") or os.environ.get("WARA_MODEL_PROFILE") or DEFAULT_MODEL_PROFILE))
    if pdf and final_dir.exists() and not phase2_run.exists():
        status = "completed"
    if pdf and status == "idle":
        status = "completed"
    review_status = "clear" if quality.get("ok") is True else ("needs review" if quality.get("ok") is False else None)
    return {
        "run_id": run_id,
        "path": str(phase2_run if phase2_run.exists() else final_dir),
        "topic": topic,
        "status": _public_status(status),
        "model_profile": model_profile,
        "phases": _public_phases(summary),
        "current_phase": _current_phase(summary),
        "preview_pdf": str(pdf) if pdf else None,
        "preview_pdf_mtime": str(pdf.stat().st_mtime_ns) if pdf else "",
        "quality_ok": quality.get("ok") if "ok" in quality else None,
        "review_status": review_status,
        "references_total": quality.get("reference_count") or _count_bib_entries(final_dir / "references.bib"),
        "paper_ready_figure_count": quality.get("paper_ready_figure_count"),
        "phase25_status": quality.get("phase25_status"),
        "telemetry": _build_telemetry(run_id, phase2_run if phase2_run.exists() else final_dir),
    }


def _count_bib_entries(path: Path) -> int:
    return len(re.findall(r"@\w+\s*\{", _read_text(path)))


def _build_phase1_report(run_id: str) -> dict:
    run_dir = PHASE1_RUNS_DIR / run_id
    return {
        "run_id": run_id,
        "path": str(run_dir),
        "topic": _topic_from_phase1(run_dir),
        "usable": _phase1_handoff_path(run_dir) is not None,
        "status": "completed" if _phase1_handoff_path(run_dir) else "partial",
    }


def _combined_report(run_id: str | None = None) -> dict:
    if run_id is None:
        runs = _list_phase2_runs()
        run_id = runs[0].name if runs else None
    phase2 = _build_phase2_report(run_id) if run_id else {}
    phase1_path = PHASE1_RUNS_DIR / run_id if run_id else None
    phase1 = _build_phase1_report(run_id) if phase1_path and phase1_path.exists() else None
    return {
        "run_id": run_id,
        "workspace_root": str(ROOT),
        "topic": phase2.get("topic") or (phase1 or {}).get("topic") or "WARA Research Workspace",
        "status": _public_status(phase2.get("status") or "idle"),
        "phase1_report": phase1,
        "phase2_report": phase2,
        "telemetry": phase2.get("telemetry", {}),
    }


def _phase1_summaries() -> list[dict]:
    return [_build_phase1_report(path.name) for path in _list_phase1_runs()]


def _phase2_summaries() -> list[dict]:
    summaries = []
    for path in _list_phase2_runs():
        report = _build_phase2_report(path.name)
        summaries.append({
            "run_id": path.name,
            "display_id": _display_id_for_run(path.name),
            "topic": report.get("topic"),
            "status": report.get("status"),
            "current_phase": report.get("current_phase"),
            "preview_pdf": report.get("preview_pdf"),
        })
    return summaries


def _active_snapshot() -> dict:
    with _active_lock:
        active = dict(_active_run or {})
    if not active:
        return {"status": "idle", "report": _combined_report()}
    process = active.get("process")
    if process and process.poll() is not None and active.get("status") == "running":
        active["status"] = "completed" if process.returncode == 0 else "needs_review"
        with _active_lock:
            if _active_run:
                _active_run.update(active)
    report = _combined_report(active.get("run_id"))
    report["status"] = "running" if active.get("status") == "running" else report.get("status")
    if active.get("status") == "needs_review":
        report["status"] = "needs_review"
    report["telemetry"] = _build_telemetry(active.get("run_id", ""), _phase2_run_dir(active.get("run_id", "")), active.get("started_at"))
    activity = _current_activity(active.get("run_id"))
    if _public_status(active.get("status", "idle")) == "stopped":
        activity = dict(activity)
        activity["status"] = "stopped"
    return {
        "status": _public_status(active.get("status", "idle")),
        "run_id": active.get("run_id"),
        "activity": activity,
        "report": report,
    }


def _phase_key_from_item(item: dict) -> str:
    phase_id = str(item.get("phase_id") or "").strip().lower().replace("_", ".").replace("-", ".")
    match = re.search(r"phase\s*([123])\s*\.?\s*([1-6])", phase_id)
    if match:
        return f"{match.group(1)}.{match.group(2)}"
    try:
        step = int(item.get("phase_step") or 0)
    except (TypeError, ValueError):
        step = 0
    if 1 <= step <= 5:
        return f"2.{step}"
    if 6 <= step <= 11:
        return f"3.{step - 5}"
    return ""


def _activity_payload(phase_key: str, status: str = "running") -> dict:
    metadata = PHASE_ACTIVITY.get(phase_key, {})
    if not metadata:
        return {
            "phase_key": phase_key,
            "phase": "WARA",
            "title": "Waiting for activity",
            "agent": "Controller",
            "task": "Coordinates the next pipeline step.",
            "status": _public_status(status),
        }
    payload = dict(metadata)
    payload["phase_key"] = phase_key
    payload["status"] = _public_status(status)
    return payload


def _phase1_activity(run_id: str) -> dict | None:
    run_dir = PHASE1_RUNS_DIR / run_id
    if not run_dir.exists():
        return None
    checkpoint = _read_json(run_dir / "checkpoint.json")
    if isinstance(checkpoint, dict):
        try:
            completed = int(checkpoint.get("last_completed_phase") or 0)
        except (TypeError, ValueError):
            completed = 0
        next_step = min(max(completed + 1, 1), 4)
        status = "completed" if completed >= 4 else "running"
        return _activity_payload(f"1.{next_step}", status)
    completed = 0
    for step in range(1, 5):
        decision = _read_json(run_dir / f"phase1-{step}" / "decision.json")
        if not isinstance(decision, dict):
            break
        status = str(decision.get("status") or "").strip().lower()
        action = str(decision.get("decision") or decision.get("recommendation") or "").strip().lower()
        if status in {"done", "completed", "complete", "success"} and action not in {"revise", "reject", "failed", "fail", "not_ready"}:
            completed = step
            continue
        return _activity_payload(f"1.{step}", "needs_review")
    if completed > 0:
        next_step = min(completed + 1, 4)
        status = "completed" if completed >= 4 else "running"
        return _activity_payload(f"1.{next_step}", status)
    if _phase1_handoff_path(run_dir):
        return _activity_payload("1.4", "completed")
    return _activity_payload("1.1", "running")


def _summary_activity(summary: dict) -> dict | None:
    phases = summary.get("phases") if isinstance(summary.get("phases"), list) else []
    normalized = [item for item in phases if isinstance(item, dict)]
    for wanted_status in ("running", "failed", "error", "needs_review"):
        for item in normalized:
            status = _public_status(item.get("status"))
            if status == "needs_review" and wanted_status not in {"failed", "error", "needs_review"}:
                continue
            if status == wanted_status or (wanted_status in {"failed", "error"} and status == "needs_review"):
                phase_key = _phase_key_from_item(item)
                if phase_key:
                    return _activity_payload(phase_key, status)
    completed = [
        item
        for item in normalized
        if _public_status(item.get("status")) == "completed"
    ]
    if completed:
        phase_key = _phase_key_from_item(completed[-1])
        if phase_key:
            return _activity_payload(phase_key, "completed")
    return None


def _activity_from_log(text: str) -> dict | None:
    lowered = text.lower()
    has_failure = any(marker in lowered for marker in ("traceback", "modulenotfounderror", "returned non-zero", "failed", "error"))
    if "phase 3 started" in lowered:
        return _activity_payload("3.1", "needs_review" if has_failure else "running")
    if "phase 2 started" in lowered:
        return _activity_payload("2.1", "needs_review" if has_failure else "running")
    if "phase 1 started" in lowered and not re.search(r"\b1\.[2-4]\b", lowered):
        return _activity_payload("1.1", "needs_review" if has_failure else "running")
    matches = list(re.finditer(r"\b([123])\.(\d)\b", lowered))
    for match in reversed(matches):
        phase_key = f"{match.group(1)}.{match.group(2)}"
        if phase_key in PHASE_ACTIVITY:
            line_start = lowered.rfind("\n", 0, match.start()) + 1
            line_end = lowered.find("\n", match.end())
            line = lowered[line_start: line_end if line_end >= 0 else len(lowered)]
            status = "completed" if "completed" in line else ("needs_review" if has_failure or "failed" in line or "error" in line else "running")
            return _activity_payload(phase_key, status)
    return None


def _current_activity(run_id: str | None) -> dict:
    if not run_id:
        return _activity_payload("", "idle")
    phase2 = _phase2_run_dir(run_id)
    summary = _read_json(phase2 / "phase2_summary.json")
    if not isinstance(summary, dict):
        summary = _read_json(_final_package_dir(run_id) / "phase2_summary.json")
    if isinstance(summary, dict):
        activity = _summary_activity(summary)
        if activity:
            return activity
    ui_log = UI_RUNS_DIR / run_id / "stdout.log"
    log_activity = _activity_from_log(_tail_text(ui_log, max_lines=240)) if ui_log.exists() else None
    if log_activity and str(log_activity.get("phase_key") or "").startswith(("2.", "3.")):
        return log_activity
    phase1_activity = _phase1_activity(run_id)
    if phase1_activity:
        return phase1_activity
    return log_activity or _activity_payload("", "idle")


def _activity_header(activity: dict) -> str:
    if not activity or _public_status(activity.get("status")) == "idle":
        return ""
    return "\n".join(
        [
            "[WARA] Current activity",
            f"[WARA] Phase: {activity.get('phase')} - {activity.get('title')}",
            f"[WARA] Agent: {activity.get('agent')}",
            f"[WARA] Task: {activity.get('task')}",
            f"[WARA] Status: {activity.get('status')}",
        ]
    )


def _log_for_run(run_id: str | None) -> str:
    if not run_id:
        return ""
    activity = _current_activity(run_id)
    header = _activity_header(activity)
    ui_log = UI_RUNS_DIR / run_id / "stdout.log"
    if ui_log.exists():
        raw_log = _sanitize_public_text(_tail_text(ui_log))
        return "\n\n".join(part for part in [header, raw_log] if part)
    phase2 = _phase2_run_dir(run_id)
    summary = _read_json(phase2 / "phase2_summary.json")
    if not isinstance(summary, dict):
        summary = _read_json(_final_package_dir(run_id) / "phase2_summary.json")
    if isinstance(summary, dict):
        lines = [header, f"[WARA] run: {run_id}"] if header else [f"[WARA] run: {run_id}"]
        for item in summary.get("phases", []):
            if isinstance(item, dict):
                status = _public_status(item.get("status"))
                phase_key = _phase_key_from_item(item)
                metadata = PHASE_ACTIVITY.get(phase_key, {})
                agent = metadata.get("agent") or "Controller"
                name = metadata.get("title") or item.get("name") or item.get("phase_id")
                lines.append(f"[WARA] {item.get('phase_id')} {status} - {name} ({agent})")
        return _sanitize_public_text("\n".join(lines))
    return header


def _safe_artifact_path(raw_path: str) -> Path:
    path = Path(str(raw_path or "")).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    resolved = path.resolve()
    try:
        resolved.relative_to(ROOT)
    except ValueError as exc:
        raise PermissionError(f"path is outside WARA workspace: {resolved}") from exc
    if not resolved.exists():
        raise FileNotFoundError(str(resolved))
    return resolved


def _open_path(path: Path) -> None:
    if sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    elif os.name == "nt":
        os.startfile(str(path))  # type: ignore[attr-defined]
    else:
        subprocess.Popen(["xdg-open", str(path)])


def _pdf_preview_path(pdf_path: Path) -> Path:
    existing_preview = pdf_path.with_name("paper_preview.png")
    if existing_preview.exists():
        return existing_preview

    cache_key = sha1(f"{pdf_path}:{pdf_path.stat().st_mtime_ns}".encode("utf-8")).hexdigest()[:16]
    cache_dir = PDF_PREVIEW_DIR / cache_key
    preview_path = cache_dir / "preview.png"
    if preview_path.exists():
        return preview_path

    qlmanage = shutil.which("qlmanage")
    if not qlmanage:
        raise RuntimeError("PDF preview requires qlmanage on macOS")

    cache_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [qlmanage, "-t", "-s", "1100", "-o", str(cache_dir), str(pdf_path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )
    generated = cache_dir / f"{pdf_path.name}.png"
    if not generated.exists():
        raise FileNotFoundError(f"preview was not generated for {pdf_path}")
    generated.replace(preview_path)
    return preview_path


def _parse_multipart_body(headers, body: bytes) -> tuple[dict[str, str], dict[str, dict[str, bytes | str]]]:
    content_type = headers.get("Content-Type", "")
    if "multipart/form-data" not in content_type:
        raise ValueError("expected multipart/form-data")
    message = BytesParser(policy=policy.default).parsebytes(
        f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8") + body
    )
    fields: dict[str, str] = {}
    files: dict[str, dict[str, bytes | str]] = {}
    for part in message.iter_parts():
        if part.get_content_disposition() != "form-data":
            continue
        name = part.get_param("name", header="content-disposition") or ""
        if not name:
            continue
        filename = part.get_filename()
        payload = part.get_payload(decode=True) or b""
        if filename:
            files[name] = {
                "filename": str(filename),
                "content_type": str(part.get_content_type() or "application/octet-stream"),
                "data": payload,
            }
        else:
            fields[name] = payload.decode("utf-8", errors="ignore")
    return fields, files


def _safe_review_paper_id(raw: str, fallback: str = "paper") -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(raw or "").strip()).strip("-._")
    return (stem[:80] or fallback).strip("-._") or fallback


def _run_pdf_review(fields: dict[str, str], files: dict[str, dict[str, bytes | str]]) -> dict:
    upload = files.get("pdf")
    if not upload:
        raise ValueError("missing PDF upload")
    filename = str(upload.get("filename") or "paper.pdf")
    data = upload.get("data")
    if not isinstance(data, bytes) or not data:
        raise ValueError("uploaded PDF is empty")
    if not filename.lower().endswith(".pdf") or data[:5] != b"%PDF-":
        raise ValueError("uploaded file must be a PDF")
    if len(data) > 50 * 1024 * 1024:
        raise ValueError("PDF is too large; maximum size is 50 MB")

    model_profile = normalize_model_profile(str(fields.get("model_profile") or os.environ.get("WARA_MODEL_PROFILE") or DEFAULT_MODEL_PROFILE))
    _require_profile_api_key(model_profile)
    _require_review_dependencies()

    requested_id = fields.get("paper_id") or Path(filename).stem
    paper_id = _safe_review_paper_id(requested_id)
    digest = sha1(data).hexdigest()[:10]
    review_id = f"review-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{digest}"
    review_dir = PDF_REVIEW_DIR / review_id
    review_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = review_dir / "paper.pdf"
    out_path = review_dir / "review.json"
    csv_path = review_dir / "review.csv"
    log_path = review_dir / "review.log"
    pdf_path.write_bytes(data)

    max_tokens_raw = str(fields.get("max_tokens") or "12000").strip()
    try:
        max_tokens = max(1000, min(50000, int(max_tokens_raw)))
    except ValueError:
        max_tokens = 12000

    command = [
        *_review_python_command(),
        str(ROOT / "evaluation" / "llm_paper_review_agent.py"),
        "--pdf",
        str(pdf_path),
        "--paper-id",
        paper_id,
        "--out",
        str(out_path),
        "--csv",
        str(csv_path),
        "--model-profile",
        model_profile,
        "--max-tokens",
        str(max_tokens),
    ]
    env = _load_dotenv()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), str(ROOT / "phase2" / "scripts")] + ([existing_pythonpath] if existing_pythonpath else [])
    )
    timeout = int(os.environ.get("WARA_REVIEW_TIMEOUT_SEC", "900") or 900)
    started_at = _utcnow_iso()
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="ignore",
        timeout=timeout,
        check=False,
    )
    finished_at = _utcnow_iso()
    log_path.write_text(
        "COMMAND: " + " ".join(command) + "\n\nSTDOUT:\n" + result.stdout + "\n\nSTDERR:\n" + result.stderr,
        encoding="utf-8",
    )
    if result.returncode != 0:
        tail = "\n".join((result.stderr or result.stdout or "").splitlines()[-20:])
        raise RuntimeError(tail or f"review agent failed with code {result.returncode}")

    payload = _read_json(out_path)
    if not isinstance(payload, dict):
        raise RuntimeError("review agent did not produce a JSON report")
    reviews = payload.get("reviews")
    if not isinstance(reviews, dict) or not reviews:
        raise RuntimeError("review JSON does not contain scoring reviews")
    for review in reviews.values():
        if isinstance(review, dict):
            review["review_id"] = review_id
            review["started_at"] = started_at
            review["finished_at"] = finished_at
            review["model_profile"] = model_profile
    payload["review_id"] = review_id
    payload["started_at"] = started_at
    payload["finished_at"] = finished_at
    payload["model_profile"] = model_profile
    payload["reviews"] = reviews
    _write_json(out_path, payload)
    return {
        "ok": True,
        "review_id": review_id,
        "paper_id": paper_id,
        "model_profile": model_profile,
        "research_validity_score": payload.get("research_validity_score"),
        "optimization_maturity_score": payload.get("optimization_maturity_score"),
        "reviews": reviews,
        "rubric_profiles": payload.get("rubric_profiles") or {},
        "review_json": str(out_path),
        "review_csv": str(csv_path),
        "log_path": str(log_path),
    }


def _stream_process(process: subprocess.Popen, log_path: Path) -> None:
    assert process.stdout is not None
    with log_path.open("a", encoding="utf-8", errors="ignore") as handle:
        for line in process.stdout:
            handle.write(line)
            handle.flush()
    code = process.wait()
    with _active_lock:
        if _active_run and _active_run.get("process") is process:
            _active_run["status"] = "completed" if code == 0 else "needs_review"
            _active_run["returncode"] = code
            _active_run["finished_at"] = _utcnow_iso()
    manifest = log_path.parent / "ui_run_manifest.json"
    payload = _read_json(manifest)
    if isinstance(payload, dict):
        payload.update({"status": "completed" if code == 0 else "needs_review", "returncode": code, "finished_at": _utcnow_iso()})
        _write_json(manifest, payload)


def _start_run(payload: dict) -> dict:
    global _active_run

    topic = str(payload.get("topic") or "").strip()
    model_profile = normalize_model_profile(str(payload.get("model_profile") or os.environ.get("WARA_MODEL_PROFILE") or DEFAULT_MODEL_PROFILE).strip())
    phase1_run = str(payload.get("phase1_run") or "").strip()
    skip_phase1 = bool(payload.get("skip_phase1"))
    if not topic and not phase1_run and not skip_phase1:
        raise ValueError("missing topic")
    _require_profile_api_key(model_profile)
    _require_runtime_dependencies()
    with _active_lock:
        if _active_run and _active_run.get("status") == "running":
            raise RuntimeError("another WARA run is already running")

    run_id = _sanitize_run_id(topic or Path(phase1_run).name or "resume")
    run_dir = UI_RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    _display_id_for_run(run_id)
    log_path = run_dir / "stdout.log"
    usage_log = run_dir / "usage.jsonl"

    cmd = [*_python_command(), str(PIPELINE_SCRIPT), "--model-profile", model_profile, "--run-id", run_id]
    if phase1_run:
        cmd.extend(["--phase1-run", phase1_run])
    elif skip_phase1:
        cmd.append("--skip-phase1")
    if topic:
        cmd.extend(["--topic", topic])

    env = _load_dotenv()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("WARA_USAGE_LOG", str(usage_log))
    process = subprocess.Popen(
        cmd,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="ignore",
        bufsize=1,
    )
    active = {
        "run_id": run_id,
        "topic": topic,
        "model_profile": model_profile,
        "status": "running",
        "started_at": _utcnow_iso(),
        "command": cmd,
        "log_path": str(log_path),
        "process": process,
    }
    _write_json(run_dir / "ui_run_manifest.json", {key: value for key, value in active.items() if key != "process"})
    with _active_lock:
        _active_run = active
    threading.Thread(target=_stream_process, args=(process, log_path), daemon=True).start()
    time.sleep(0.2)
    return _active_snapshot()


def _stop_run() -> dict:
    with _active_lock:
        active = _active_run
    process = active.get("process") if active else None
    if process and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            process.kill()
    with _active_lock:
        if _active_run:
            _active_run["status"] = "stopped"
            _active_run["finished_at"] = _utcnow_iso()
    return {"ok": True, "status": "stopped"}


class WaraHandler(SimpleHTTPRequestHandler):
    def translate_path(self, path: str) -> str:
        parsed = urlparse(path)
        path_only = parsed.path.rstrip("/") or "/"
        aliases = {
            "/": "index.html",
            "/console": "index.html",
            "/review": "review.html",
            "/manuscript-review": "review.html",
        }
        rel = aliases.get(path_only, parsed.path.lstrip("/") or "index.html")
        return str((APP_DIR / rel).resolve())

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def _send_json(self, payload: dict | list, status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length).decode("utf-8") if length else "{}"
        payload = json.loads(raw or "{}")
        return payload if isinstance(payload, dict) else {}

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        try:
            if parsed.path == "/api/overview":
                self._send_json(_combined_report())
                return
            if parsed.path == "/api/status":
                self._send_json(_active_snapshot())
                return
            if parsed.path == "/api/model-config":
                self._send_json(_model_config_payload())
                return
            if parsed.path == "/api/runtime-env":
                self._send_json(_runtime_environment_report())
                return
            if parsed.path == "/api/phase1-runs":
                self._send_json({"runs": _phase1_summaries()})
                return
            if parsed.path == "/api/phase2-runs":
                self._send_json({"runs": _phase2_summaries()})
                return
            if parsed.path.startswith("/api/runs/"):
                self._send_json(_combined_report(parsed.path.rsplit("/", 1)[-1]))
                return
            if parsed.path == "/api/logs":
                status = _active_snapshot()
                run_id = (query.get("run_id") or [status.get("run_id") or ""])[0]
                if not run_id:
                    overview = _combined_report()
                    run_id = str(overview.get("run_id") or "")
                self._send_json({"run_id": run_id, "activity": _current_activity(run_id), "combined": _log_for_run(run_id)})
                return
            if parsed.path == "/api/artifact":
                raw_path = (query.get("path") or [""])[0]
                path = _safe_artifact_path(raw_path)
                data = path.read_bytes()
                ctype = "application/pdf" if path.suffix.lower() == ".pdf" else "application/octet-stream"
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)
                return
            if parsed.path == "/api/pdf-preview":
                raw_path = (query.get("path") or [""])[0]
                path = _safe_artifact_path(raw_path)
                if path.suffix.lower() != ".pdf":
                    raise ValueError("preview source must be a PDF")
                preview_path = _pdf_preview_path(path)
                data = preview_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)
                return
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=500)
            return
        super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/start":
                self._send_json(_start_run(self._read_json_body()), status=202)
                return
            if parsed.path == "/api/stop":
                self._send_json(_stop_run())
                return
            if parsed.path == "/api/model-config":
                self._send_json(_save_model_config(self._read_json_body()))
                return
            if parsed.path == "/api/review-pdf":
                length = int(self.headers.get("Content-Length", "0") or 0)
                if length <= 0:
                    raise ValueError("empty request body")
                fields, files = _parse_multipart_body(self.headers, self.rfile.read(length))
                self._send_json(_run_pdf_review(fields, files))
                return
            if parsed.path == "/api/open-path":
                payload = self._read_json_body()
                path = _safe_artifact_path(str(payload.get("path") or ""))
                _open_path(path)
                self._send_json({"ok": True})
                return
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=400)
            return
        except RuntimeError as exc:
            self._send_json({"error": str(exc)}, status=409)
            return
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=500)
            return
        self.send_error(404)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Serve the WARA web console")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.environ.get("WARA_UI_PORT", "8765")))
    args = parser.parse_args()

    if not APP_DIR.exists():
        raise SystemExit(f"missing app directory: {APP_DIR}")
    server = ThreadingHTTPServer((args.host, args.port), WaraHandler)
    print(f"[WARA UI] http://{args.host}:{args.port}", flush=True)
    print(f"[WARA UI] workspace: {ROOT}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
