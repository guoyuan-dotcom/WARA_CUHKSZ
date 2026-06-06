from __future__ import annotations

import csv
import copy
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


def _safe_name(value: str) -> str:
    name = re.sub(r"\W+", "_", str(value)).strip("_")
    if not name:
        return "field"
    if name[0].isdigit():
        name = f"field_{name}"
    return name


def _flatten(prefix: str, value: Any, out: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            safe_key = _safe_name(str(key))
            next_prefix = f"{prefix}_{safe_key}" if prefix else safe_key
            _flatten(next_prefix, child, out)
        return
    out[prefix] = value


def _to_array_if_numeric(value: Any) -> Any:
    if isinstance(value, list):
        converted = [_to_array_if_numeric(item) for item in value]
        try:
            array = np.asarray(converted)
            if array.dtype.kind in {"i", "u", "f", "c", "b"}:
                return array
        except Exception:
            pass
        return converted
    if isinstance(value, dict):
        return {str(k): _to_array_if_numeric(v) for k, v in value.items()}
    return value


@dataclass
class ProblemData:
    fields: dict[str, Any]
    case_name: str = "canonical"
    case_id: str = "canonical"
    swept_param: str = "canonical"
    swept_value: Any = 0.0
    scenario_name: str = "canonical"
    validation_plan: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.fields = _to_array_if_numeric(copy.deepcopy(self.fields))
        flattened: dict[str, Any] = {}
        _flatten("", self.fields, flattened)
        self.flat_fields = flattened

    def __getattr__(self, name: str) -> Any:
        if name in self.fields:
            return self.fields[name]
        if name in self.flat_fields:
            return self.flat_fields[name]
        raise AttributeError(name)

    def get(self, path: str, default: Any = None) -> Any:
        if str(path) in self.__dict__:
            return self.__dict__[str(path)]
        current: Any = self.fields
        for part in str(path).replace("/", ".").split("."):
            if not part:
                continue
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                flat_name = _safe_name(str(path).replace(".", "_").replace("/", "_"))
                return self.flat_fields.get(flat_name, default)
        return current

    def clone_with(self, *, case_name: str, case_id: str, swept_param: str, swept_value: Any, scenario_name: str, updates: dict[str, Any] | None = None) -> "ProblemData":
        fields = copy.deepcopy(self.fields)
        for key, value in (updates or {}).items():
            _set_path(fields, key, value)
        return ProblemData(
            fields=fields,
            case_name=case_name,
            case_id=case_id,
            swept_param=swept_param,
            swept_value=swept_value,
            scenario_name=scenario_name,
            validation_plan=copy.deepcopy(self.validation_plan),
        )


@dataclass
class SolverResult:
    method: str
    status: str
    objective: float
    feasible: bool
    iterations: int
    solve_time_sec: float
    message: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    trace_objective: list[float] = field(default_factory=list)


def _set_path(mapping: dict[str, Any], path: str, value: Any) -> None:
    parts = [part for part in str(path).replace("/", ".").split(".") if part]
    if not parts:
        return
    current = mapping
    for part in parts[:-1]:
        current = current.setdefault(part, {})
        if not isinstance(current, dict):
            return
    current[parts[-1]] = value


def make_canonical_problem(validation_plan: dict[str, Any] | None = None, case_name: str = "canonical") -> ProblemData:
    plan = validation_plan if isinstance(validation_plan, dict) else {}
    config = plan.get("canonical_config", {}) if isinstance(plan.get("canonical_config", {}), dict) else {}
    return ProblemData(
        fields=copy.deepcopy(config),
        case_name=str(case_name),
        case_id="canonical",
        swept_param="canonical",
        swept_value=0.0,
        scenario_name=str(config.get("scenario", config.get("scenario_name", "canonical"))),
        validation_plan=copy.deepcopy(plan),
    )


def _serialize_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return [{"real": float(np.real(x)), "imag": float(np.imag(x))} for x in value.reshape(-1)]
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, dict):
        return {str(k): _serialize_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_serialize_value(v) for v in value]
    return value


def result_to_dict(result: SolverResult) -> dict[str, Any]:
    raw = asdict(result)
    return {str(k): _serialize_value(v) for k, v in raw.items()}


def save_json(obj: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_serialize_value(obj), handle, indent=2, ensure_ascii=False)


def save_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _serialize_value(v) for k, v in row.items()})
