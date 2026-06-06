from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wara_core.agents import ExperimentAgent, RoleAgent, build_default_agent_registry, phase_to_agent_ids  # noqa: E402


def _print_json(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def list_agents() -> None:
    registry = build_default_agent_registry()
    _print_json({agent_id: contract.to_dict() for agent_id, contract in registry.items()})


def show_phase(phase_id: str) -> None:
    registry = build_default_agent_registry()
    agent_ids = phase_to_agent_ids(phase_id)
    _print_json(
        {
            "phase_id": phase_id,
            "agent_ids": list(agent_ids),
            "agents": [registry[agent_id].to_dict() for agent_id in agent_ids],
        }
    )


def bootstrap_experiment(run_dir: Path) -> None:
    agent = ExperimentAgent(run_dir)
    snapshot = agent.bootstrap()
    request_path = agent.write_request_payload()
    _print_json(
        {
            "agent_id": agent.id,
            "snapshot": snapshot.to_dict(),
            "request_path": str(request_path),
            "workspace_manifest_path": str(run_dir / "agent_workspace_manifest.json"),
        }
    )


def bootstrap_agent(agent_id: str, run_dir: Path) -> None:
    if str(agent_id).strip().lower() in {"experiment_agent", "implementation_agent"}:
        bootstrap_experiment(run_dir)
        return
    agent = RoleAgent(run_dir, agent_id)
    snapshot = agent.bootstrap()
    request_path = agent.write_request_payload()
    _print_json(
        {
            "agent_id": agent.id,
            "snapshot": snapshot.to_dict(),
            "request_path": str(request_path),
            "workspace_manifest_path": str(run_dir / "agent_workspace_manifest.json"),
        }
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="WARA agent runtime helper")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list-agents", help="Print the WARA agent registry")
    phase_parser = subparsers.add_parser("phase", help="Show agents mapped to a phase id")
    phase_parser.add_argument("phase_id", help="Example: phase2.4, phase3.3, phase1.3")
    bootstrap_parser = subparsers.add_parser("bootstrap-experiment", help="Bootstrap ExperimentAgent artifacts for a run")
    bootstrap_parser.add_argument("run_dir", type=Path)
    bootstrap_agent_parser = subparsers.add_parser("bootstrap-agent", help="Bootstrap any WARA role agent for a run")
    bootstrap_agent_parser.add_argument("agent_id", help="Example: formulation_agent, theory_agent, writing_agent")
    bootstrap_agent_parser.add_argument("run_dir", type=Path)

    args = parser.parse_args(argv)
    if args.command == "list-agents":
        list_agents()
    elif args.command == "phase":
        show_phase(args.phase_id)
    elif args.command == "bootstrap-experiment":
        bootstrap_experiment(args.run_dir)
    elif args.command == "bootstrap-agent":
        bootstrap_agent(args.agent_id, args.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
