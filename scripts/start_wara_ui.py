from __future__ import annotations

import argparse
import os
import subprocess
import sys
import venv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS_FILE = ROOT / "requirements.txt"
VENV_DIR = ROOT / ".venv"
VENV_PYTHON = VENV_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
SERVE_SCRIPT = ROOT / "scripts" / "serve_wara.py"

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


def _run(command: list[str], *, cwd: Path = ROOT) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def _missing_modules(python: Path) -> list[str]:
    code = (
        "import importlib.util, json\n"
        f"mods = {RUNTIME_MODULE_CHECKS!r}\n"
        "missing = [name for name, module in mods.items() if importlib.util.find_spec(module) is None]\n"
        "print(json.dumps(missing))\n"
    )
    result = subprocess.run(
        [str(python), "-B", "-c", code],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        errors="ignore",
        check=False,
    )
    import json

    try:
        payload = json.loads(result.stdout.strip() or "[]")
    except json.JSONDecodeError:
        return list(RUNTIME_MODULE_CHECKS)
    return [str(item) for item in payload] if isinstance(payload, list) else list(RUNTIME_MODULE_CHECKS)


def ensure_environment(force_install: bool = False) -> Path:
    if not VENV_PYTHON.exists():
        print(f"[WARA setup] Creating virtual environment at {VENV_DIR}", flush=True)
        venv.EnvBuilder(with_pip=True).create(VENV_DIR)
    missing = _missing_modules(VENV_PYTHON)
    if force_install or missing:
        if missing:
            print(f"[WARA setup] Installing missing runtime modules: {', '.join(missing)}", flush=True)
        else:
            print("[WARA setup] Reinstalling runtime requirements", flush=True)
        _run([str(VENV_PYTHON), "-m", "pip", "install", "--upgrade", "pip"])
        _run([str(VENV_PYTHON), "-m", "pip", "install", "-r", str(REQUIREMENTS_FILE)])
        missing = _missing_modules(VENV_PYTHON)
    if missing:
        raise SystemExit(
            "WARA runtime setup is incomplete. Missing modules after installation: "
            + ", ".join(missing)
        )
    return VENV_PYTHON


def main() -> None:
    parser = argparse.ArgumentParser(description="Create/use the WARA virtual environment and start the web console.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.environ.get("WARA_UI_PORT", "8765")))
    parser.add_argument("--install", action="store_true", help="Force reinstall requirements before starting the UI.")
    args = parser.parse_args()

    python = ensure_environment(force_install=args.install)
    os.environ["WARA_PYTHON"] = str(python)
    os.execv(str(python), [str(python), "-B", str(SERVE_SCRIPT), "--host", args.host, "--port", str(args.port)])


if __name__ == "__main__":
    main()
