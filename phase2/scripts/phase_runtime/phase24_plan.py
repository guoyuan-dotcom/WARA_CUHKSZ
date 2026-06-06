from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

import yaml

from pipeline_core import (
    PHASE24_BASE_SIGNATURES,
    PHASE24_FIXED_FILE_CONTRACTS,
    PHASE24_ZERO_ARG_CALLABLES,
    compact_text,
)


def build_phase24_module_plan(topic: str, algorithm_md: str, reformulation_path_md: str) -> dict[str, Any]:
    text = f"{topic}\n{algorithm_md}\n{reformulation_path_md}".lower()
    primary_semantic = "beamforming_update" if "beamform" in text or "precoder" in text else "primary_update"
    secondary_semantic = "position_update" if "position" in text or "antenna" in text else "secondary_update"
    model_semantic = "channel_and_objective_ops" if "channel" in text or "near-field" in text or "near field" in text else "model_ops"
    required_operators = [
        {
            "name": "channel_from_state",
            "role": "compute physical/channel model from current state",
            "signature": "channel_from_state(problem: ProblemData, state: dict) -> dict",
        },
        {
            "name": "evaluate_state",
            "role": "compute objective/constraints/metrics from current state",
            "signature": "evaluate_state(problem: ProblemData, model: dict, state: dict) -> dict",
        },
        {
            "name": "project_state",
            "role": "repair/project state into feasible domain",
            "signature": "project_state(problem: ProblemData, state: dict) -> dict",
        },
    ]
    return {
        "fixed_files": [
            "problem_data.py",
            "validation_cases.py",
            "proposed_solver.py",
            "baseline_solver.py",
            "run_validation.py",
        ],
        "variable_files": [
            "model_ops.py",
            "proposed_block_a.py",
            "proposed_block_b.py",
        ],
        "model_files": [
            {
                "file": "model_ops.py",
                "semantic_name": model_semantic,
                "required_exports": ["build_model", "evaluate_solution"],
            }
        ],
        "model_contract": {
            "model_top_level_keys": ["state_init", "operators", "metadata"],
            "required_operators": required_operators,
        },
        "blocks": [
            {
                "file": "proposed_block_a.py",
                "export": "run_primary_update",
                "role": "primary_update",
                "semantic_name": primary_semantic,
                "required_exports": ["run_primary_update"],
                "allowed_operator_keys": [item["name"] for item in required_operators],
            },
            {
                "file": "proposed_block_b.py",
                "export": "run_secondary_update",
                "role": "secondary_update",
                "semantic_name": secondary_semantic,
                "required_exports": ["run_secondary_update"],
                "allowed_operator_keys": [item["name"] for item in required_operators],
            },
        ],
    }


def get_phase24_blocks(module_plan: dict[str, Any]) -> list[dict[str, Any]]:
    blocks = module_plan.get("blocks")
    if isinstance(blocks, list):
        return blocks
    legacy = module_plan.get("proposed_blocks")
    if isinstance(legacy, list):
        return legacy
    return []


def get_phase24_required_operators(module_plan: dict[str, Any]) -> list[dict[str, str]]:
    model_contract = module_plan.get("model_contract", {})
    operators = model_contract.get("required_operators")
    if isinstance(operators, list):
        return [item for item in operators if isinstance(item, dict) and item.get("name")]
    return []


def build_phase24_file_interface_contracts(module_plan: dict[str, Any]) -> dict[str, dict[str, list[str]]]:
    contracts = {name: {"classes": list(spec.get("classes", [])), "functions": list(spec.get("functions", []))} for name, spec in PHASE24_FIXED_FILE_CONTRACTS.items()}
    for spec in module_plan.get("model_files", []):
        contracts[spec["file"]] = {"classes": [], "functions": list(spec.get("required_exports", []))}
    for spec in get_phase24_blocks(module_plan):
        contracts[spec["file"]] = {"classes": [], "functions": list(spec.get("required_exports", []))}
    return contracts


def build_phase24_function_signatures(module_plan: dict[str, Any]) -> dict[str, dict[str, list[str]]]:
    signatures = {name: {fn: list(args) for fn, args in mapping.items()} for name, mapping in PHASE24_BASE_SIGNATURES.items()}
    for spec in module_plan.get("model_files", []):
        file_name = spec["file"]
        signatures[file_name] = {
            "build_model": ["problem", "seed"],
            "evaluate_solution": ["problem", "model", "solution"],
        }
    for spec in get_phase24_blocks(module_plan):
        file_name = spec["file"]
        exports = spec.get("required_exports", [])
        if not exports:
            continue
        fn_name = exports[0]
        signatures[file_name] = {
            fn_name: ["problem", "model", "state"],
        }
    return signatures


def build_phase24_zero_arg_callables(module_plan: dict[str, Any]) -> dict[str, list[str]]:
    _ = module_plan
    return {name: list(values) for name, values in PHASE24_ZERO_ARG_CALLABLES.items()}


def build_phase24_solver_import_contracts(module_plan: dict[str, Any]) -> dict[str, list[str]]:
    contracts = {
        "problem_data": ["ProblemData", "SolverResult", "result_to_dict", "save_json", "save_csv"],
        "validation_cases": ["load_canonical_case", "make_validation_cases"],
        "proposed_solver": ["solve_proposed"],
        "baseline_solver": ["solve_baseline"],
    }
    for spec in module_plan.get("model_files", []):
        contracts[spec["file"].replace(".py", "")] = list(spec.get("required_exports", []))
    for spec in get_phase24_blocks(module_plan):
        contracts[spec["file"].replace(".py", "")] = list(spec.get("required_exports", []))
    return contracts


def format_phase24_exports(file_target: str, file_interface_contracts: dict[str, dict[str, list[str]]]) -> str:
    contract = file_interface_contracts[file_target]
    lines: list[str] = []
    for cls_name in contract.get("classes", []):
        lines.append(f"- class {cls_name}")
    for fn_name in contract.get("functions", []):
        lines.append(f"- def {fn_name}(...)")
    return "\n".join(lines)


def format_phase24_other_interfaces(file_target: str, file_interface_contracts: dict[str, dict[str, list[str]]]) -> str:
    rows: list[str] = []
    for name, contract in file_interface_contracts.items():
        if name == file_target:
            continue
        module_name = name.replace(".py", "")
        exports: list[str] = []
        exports.extend(contract.get("classes", []))
        exports.extend(contract.get("functions", []))
        rows.append(f"- {module_name}: {', '.join(exports)}")
    return "\n".join(rows)


def format_phase24_signatures(file_target: str, file_function_signatures: dict[str, dict[str, list[str]]]) -> str:
    signatures = file_function_signatures.get(file_target, {})
    if not signatures:
        return "- No explicit signature constraints."
    lines = []
    for fn_name, args in signatures.items():
        lines.append(f"- {fn_name}({', '.join(args)})")
    return "\n".join(lines)


def format_phase24_model_contract(module_plan: dict[str, Any]) -> str:
    model_contract = module_plan.get("model_contract", {})
    top_keys = model_contract.get("model_top_level_keys", ["state_init", "operators", "metadata"])
    lines = [f"- model_top_level_keys: {top_keys}"]
    for spec in get_phase24_required_operators(module_plan):
        lines.append(f"- operator {spec['name']}: {spec.get('signature', '')} | role: {spec.get('role', '')}")
    return "\n".join(lines)


def format_phase24_allowed_operator_keys(file_target: str, module_plan: dict[str, Any]) -> str:
    if file_target == "model_ops.py":
        return ", ".join(spec["name"] for spec in get_phase24_required_operators(module_plan)) or "(none)"
    if file_target in {"proposed_solver.py", "baseline_solver.py"}:
        return ", ".join(spec["name"] for spec in get_phase24_required_operators(module_plan)) or "(none)"
    for spec in get_phase24_blocks(module_plan):
        if spec.get("file") == file_target:
            return ", ".join(spec.get("allowed_operator_keys", [])) or "(none)"
    return "(none)"


def summarize_validation_plan(yaml_text: str) -> str:
    yaml_text = (yaml_text or "").strip()
    if not yaml_text:
        return "- validation_plan.yaml not available"
    try:
        data = yaml.safe_load(yaml_text) or {}
    except Exception:
        return compact_text(yaml_text, 2000)
    lines: list[str] = []
    if isinstance(data, dict):
        top_keys = list(data.keys())
        lines.append(f"- top_level_keys: {top_keys}")
        canonical = data.get("canonical_config")
        if isinstance(canonical, dict):
            for subkey in ("system", "geometry", "weights", "algorithm"):
                sub = canonical.get(subkey)
                if isinstance(sub, dict):
                    lines.append(f"- canonical_config.{subkey}: {list(sub.keys())}")
        sweeps = data.get("sweep_definitions")
        if isinstance(sweeps, dict):
            lines.append(f"- sweep_definitions: {list(sweeps.keys())}")
            for name, spec in list(sweeps.items())[:8]:
                if isinstance(spec, dict):
                    lines.append(
                        f"  - {name}: variable={spec.get('variable', spec.get('target', ''))}, "
                        f"canonical_path={spec.get('canonical_path', '')}, values={spec.get('values', spec.get('quick_mode', ''))}"
                    )
        elif isinstance(sweeps, list):
            lines.append(
                "- sweep_definitions: "
                + ", ".join(str(item.get("id") or item.get("name") or idx) for idx, item in enumerate(sweeps) if isinstance(item, dict))
            )
            for idx, spec in enumerate(sweeps[:8]):
                if isinstance(spec, dict):
                    lines.append(
                        f"  - {spec.get('id', spec.get('name', idx))}: variable={spec.get('variable', spec.get('target', ''))}, "
                        f"canonical_path={spec.get('canonical_path', '')}, values={spec.get('values', spec.get('quick_mode', ''))}"
                    )
        req = data.get("required_outputs")
        if isinstance(req, dict):
            lines.append(f"- required_outputs: {list(req.keys())}")
        evidence = data.get("research_evidence_contract")
        evidence_name = "research_evidence_contract"
        if not isinstance(evidence, dict):
            evidence = data.get("paper_evidence_contract")
            evidence_name = "paper_evidence_contract"
        if isinstance(evidence, dict):
            figures = evidence.get("figures", [])
            tables = evidence.get("tables", [])
            compared_methods = evidence.get("compared_methods", [])
            compared_method_ids: list[str] = []
            if isinstance(compared_methods, list):
                for item in compared_methods:
                    if isinstance(item, dict):
                        method_id = str(item.get("id") or item.get("internal_name") or item.get("name") or "").strip()
                    else:
                        method_id = str(item or "").strip()
                    if method_id:
                        compared_method_ids.append(method_id)
            if compared_method_ids:
                lines.append(f"- {evidence_name}.compared_method_ids: {compared_method_ids}")
            active_method_ids: list[str] = []
            if isinstance(figures, list):
                lines.append(
                    f"- {evidence_name}.figures: "
                    + ", ".join(
                        f"{item.get('id', item.get('figure_id', 'figure'))}:"
                        f"methods={item.get('methods_to_run', [])}"
                        for item in figures
                        if isinstance(item, dict)
                    )
                )
                for item in figures:
                    if not isinstance(item, dict):
                        continue
                    methods = item.get("methods_to_run", [])
                    if isinstance(methods, list):
                        for method in methods:
                            if isinstance(method, dict):
                                method_id = str(method.get("id") or method.get("internal_name") or method.get("name") or "").strip()
                            else:
                                method_id = str(method or "").strip()
                            if method_id and method_id not in active_method_ids:
                                active_method_ids.append(method_id)
            if isinstance(tables, list):
                lines.append(
                    f"- {evidence_name}.tables: "
                    + ", ".join(
                        f"{item.get('id', item.get('table_id', 'table'))}:{item.get('row_granularity', item.get('group_by', 'unknown'))}"
                        for item in tables
                        if isinstance(item, dict)
                    )
                )
            if active_method_ids:
                lines.append(f"- exact executable method ids required by active figures: {active_method_ids}")
            required_cols = evidence.get("required_result_columns", [])
            if isinstance(required_cols, list):
                lines.append(f"- {evidence_name}.required_result_columns: {required_cols}")
    lines.append("- raw_excerpt:")
    lines.append(compact_text(yaml_text, 1800))
    return "\n".join(lines)


def summarize_problem_data_contract(problem_data_text: str) -> str:
    problem_data_text = (problem_data_text or "").strip()
    if not problem_data_text:
        return "- problem_data.py not available yet"
    try:
        tree = ast.parse(problem_data_text)
    except SyntaxError:
        return compact_text(problem_data_text, 1800)
    lines: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "ProblemData":
            field_names: list[str] = []
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    field_names.append(item.target.id)
            lines.append(f"- ProblemData fields: {field_names}")
        if isinstance(node, ast.ClassDef) and node.name == "SolverResult":
            field_names: list[str] = []
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    field_names.append(item.target.id)
            lines.append(f"- SolverResult fields: {field_names}")
        if isinstance(node, ast.FunctionDef) and node.name == "make_canonical_problem":
            arg_names = [arg.arg for arg in node.args.args]
            lines.append(f"- make_canonical_problem signature: {arg_names}")
    return "\n".join(lines) if lines else "- Could not summarize problem_data contract"


def extract_problem_data_fields(problem_data_text: str) -> list[str]:
    problem_data_text = (problem_data_text or "").strip()
    if not problem_data_text:
        return []
    try:
        tree = ast.parse(problem_data_text)
    except SyntaxError:
        return []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "ProblemData":
            fields: list[str] = []
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    fields.append(item.target.id)
            return fields
    return []


def extract_solver_result_fields(problem_data_text: str) -> list[str]:
    problem_data_text = (problem_data_text or "").strip()
    if not problem_data_text:
        return []
    try:
        tree = ast.parse(problem_data_text)
    except SyntaxError:
        return []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "SolverResult":
            fields: list[str] = []
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    fields.append(item.target.id)
            return fields
    return []


def extract_operator_keys_from_tree(tree: ast.AST) -> set[str]:
    keys: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Subscript)
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == "model"
        ):
            outer = node.slice
            inner = node.value.slice
            if (
                isinstance(inner, ast.Constant)
                and inner.value == "operators"
                and isinstance(outer, ast.Constant)
                and isinstance(outer.value, str)
            ):
                keys.add(outer.value)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get":
            if len(node.args) >= 1 and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                key_name = node.args[0].value
                owner = node.func.value
                if (
                    isinstance(owner, ast.Call)
                    and isinstance(owner.func, ast.Attribute)
                    and owner.func.attr == "get"
                    and isinstance(owner.func.value, ast.Name)
                    and owner.func.value.id == "model"
                    and len(owner.args) >= 1
                    and isinstance(owner.args[0], ast.Constant)
                    and owner.args[0].value == "operators"
                ):
                    keys.add(key_name)
    return keys


def extract_operator_literal_keys_from_tree(tree: ast.AST) -> set[str]:
    keys: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
            if any(isinstance(target, ast.Name) and target.id == "operators" for target in node.targets):
                for key in node.value.keys:
                    if isinstance(key, ast.Constant) and isinstance(key.value, str):
                        keys.add(key.value)
    return keys


def extract_first_candidate_title(hypotheses_md: str) -> str:
    match = re.search(r"^## Candidate 1:\s*(.+?)\s*$", hypotheses_md, flags=re.MULTILINE)
    if match:
        return match.group(1).strip()
    return ""


def extract_section(block: str, heading: str) -> str:
    pattern = rf"\*\*{re.escape(heading)}\*\*\s*(.*?)(?=\n\*\*[^*]+\*\*|\Z)"
    match = re.search(pattern, block, flags=re.DOTALL)
    if not match:
        return ""
    return match.group(1).strip()


def extract_candidate_block(hypotheses_md: str) -> str:
    match = re.search(r"(^## Candidate 1:.*?)(?=^## Candidate 2:|\Z)", hypotheses_md, flags=re.MULTILINE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return hypotheses_md.strip()


def shortlist_preview(shortlist_path: Path, limit: int = 8) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not shortlist_path.exists():
        return rows
    with shortlist_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
            if len(rows) >= limit:
                break
    return rows


def _extract_phase24_validation_payload(response_text: str) -> dict[str, Any]:
    """Recover validation YAML when a model emits an invalid JSON wrapper."""
    text = str(response_text or "").strip()
    if not text:
        return {}
    if "validation_plan_yaml" in text:
        match = re.search(r'["\']validation_plan_yaml["\']\s*:\s*["\']', text)
        if match:
            yaml_text = text[match.end():]
            yaml_text = re.sub(r'["\']\s*}\s*$', "", yaml_text, flags=re.S).strip()
            yaml_text = yaml_text.replace("\\n", "\n").replace('\\"', '"')
            return {"validation_plan_yaml": yaml_text}
    if text.startswith("problem_family:") or "\npaper_evidence_contract:" in text or "\nresearch_evidence_contract:" in text:
        return {"validation_plan_yaml": text}
    return {}


def _phase24_validation_plan_text_errors(yaml_text: str) -> list[str]:
    """Return hard errors for the executable Phase 2.4 validation plan text."""
    text = (yaml_text or "").strip()
    if not text:
        return ["validation_plan_yaml is empty"]
    try:
        plan = yaml.safe_load(text) or {}
    except Exception as exc:
        return [f"validation_plan_yaml is not valid YAML: {exc}"]
    if not isinstance(plan, dict):
        return ["validation_plan_yaml did not parse to a mapping"]

    errors: list[str] = []
    evidence = plan.get("research_evidence_contract")
    evidence_name = "research_evidence_contract"
    if not isinstance(evidence, dict) or not evidence:
        evidence = plan.get("paper_evidence_contract")
        evidence_name = "paper_evidence_contract"
    if not isinstance(evidence, dict) or not evidence:
        errors.append("research_evidence_contract is missing or not a mapping")
    else:
        methods = evidence.get("compared_methods")
        if not isinstance(methods, list) or not methods:
            errors.append(f"{evidence_name}.compared_methods is missing or empty")
        figures = evidence.get("figures")
        if not isinstance(figures, list) or not figures:
            top_figures = plan.get("figure_targets") or plan.get("figures")
            if not isinstance(top_figures, list) or not top_figures:
                errors.append(f"{evidence_name}.figures or figure targets are missing")
        required_columns = evidence.get("required_result_columns")
        if not isinstance(required_columns, list) or not {"method", "seed"}.issubset({str(x) for x in required_columns}):
            errors.append(f"{evidence_name}.required_result_columns must include method and seed")
    if not isinstance(plan.get("canonical_config"), dict):
        errors.append("canonical_config is missing or not a mapping")
    sweeps = plan.get("sweep_definitions")
    if not isinstance(sweeps, list) or not sweeps:
        errors.append("sweep_definitions is missing or empty")
    required_outputs = plan.get("required_outputs")
    if not isinstance(required_outputs, dict) or not isinstance(required_outputs.get("scalar_metrics"), list) or not required_outputs.get("scalar_metrics"):
        errors.append("required_outputs.scalar_metrics is missing or empty")
    return errors


def _phase24_quote_latex_double_quoted_scalars(text: str) -> str:
    """Convert YAML double-quoted LaTeX scalars to single quotes.

    YAML treats backslashes inside double quotes as escape prefixes, so labels
    such as "$P_{\\max}$" can fail on unknown escapes like "\\m". Single-quoted
    scalars preserve LaTeX backslashes literally.
    """

    def replace(match: re.Match[str]) -> str:
        value = match.group(1)
        if "\\" not in value:
            return match.group(0)
        return "'" + value.replace("'", "''") + "'"

    return re.sub(r'"([^"\n]*(?:\\.[^"\n]*)*)"', replace, text)


def sanitize_phase24_validation_plan_yaml(yaml_text: str) -> str:
    """Repair common LLM YAML scalar formatting without changing the plan semantics."""
    text = (yaml_text or "").strip()
    if not text:
        return text
    try:
        yaml.safe_load(text)
        return text
    except Exception:
        pass

    text = _phase24_quote_latex_double_quoted_scalars(text)
    try:
        yaml.safe_load(text)
        return text
    except Exception:
        pass

    def quote_flow_sequence_scalars(line: str) -> str:
        match = re.match(r"^(\s*[A-Za-z_][\w.-]*:\s*)\[(.*)\](\s*)$", line)
        if not match:
            return line
        prefix, raw_items, suffix = match.groups()
        repaired_items: list[str] = []
        for item in raw_items.split(","):
            value = item.strip()
            if not value:
                continue
            lowered = value.lower()
            if (
                value.startswith(('"', "'"))
                or re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?", value, flags=re.IGNORECASE)
                or lowered in {"true", "false", "null", "none"}
                or re.fullmatch(r"[A-Za-z_][\w.-]*", value)
            ):
                repaired_items.append(value)
            else:
                repaired_items.append(json.dumps(value, ensure_ascii=False))
        return f"{prefix}[{', '.join(repaired_items)}]{suffix}"

    text = "\n".join(quote_flow_sequence_scalars(line) for line in text.splitlines())
    try:
        yaml.safe_load(text)
        return text
    except Exception:
        pass

    repaired_lines: list[str] = []
    for line in text.splitlines():
        key_value = re.match(r"^(\s*[A-Za-z_][\w.-]*:\s+)(.+?)\s*$", line)
        if key_value:
            prefix, value_text = key_value.groups()
            stripped_value = value_text.strip()
            if (
                ": " in stripped_value
                and not stripped_value.startswith(('"', "'", "|", ">", "{", "["))
            ):
                repaired_lines.append(prefix + json.dumps(stripped_value, ensure_ascii=False))
                continue
        match = re.match(r"^(\s*)-\s+(.+?)\s*$", line)
        if not match:
            repaired_lines.append(line)
            continue
        indent, item_text = match.groups()
        stripped = item_text.strip()
        if (
            ": " in stripped
            and not stripped.startswith(('"', "'", "|", ">", "{", "["))
            and not re.fullmatch(r"[A-Za-z_][\w.-]*:.*", stripped)
        ):
            repaired_lines.append(f"{indent}- >-")
            repaired_lines.append(f"{indent}  {stripped}")
        else:
            repaired_lines.append(line)

    repaired = "\n".join(repaired_lines).strip()
    try:
        yaml.safe_load(repaired)
        return repaired
    except Exception:
        return text


def _phase24_yaml_mapping(yaml_text: str) -> dict[str, Any] | None:
    text = sanitize_phase24_validation_plan_yaml(yaml_text)
    try:
        payload = yaml.safe_load(text) or {}
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _phase24_scalar_sweep_specs(plan: dict[str, Any]) -> list[dict[str, Any]]:
    raw_sweeps = plan.get("sweep_definitions", [])
    if not isinstance(raw_sweeps, list):
        raw_sweeps = []
    specs: list[dict[str, Any]] = []
    for idx, sweep in enumerate(raw_sweeps):
        if not isinstance(sweep, dict):
            continue
        raw_variable = str(sweep.get("variable") or sweep.get("target") or sweep.get("path") or "").strip()
        canonical_path = str(sweep.get("canonical_path") or sweep.get("path") or "").strip()
        variable = canonical_path or raw_variable
        values = sweep.get(
            "paper_mode",
            sweep.get("paper_mode_values", sweep.get("paper_values", sweep.get("values", []))),
        )
        if isinstance(values, dict):
            values = values.get("values", values.get("grid", values.get("suggested_values", [])))
        if not isinstance(values, list) or not values:
            quick_mode = sweep.get("quick_mode")
            if isinstance(quick_mode, dict):
                values = quick_mode.get("values", [])
            elif isinstance(quick_mode, list):
                values = quick_mode
        if not isinstance(values, list) or not values:
            values = sweep.get("quick_mode_values", sweep.get("quick_values", []))
        if not isinstance(values, list) or not values:
            values = sweep.get("scout_values", sweep.get("medium_values", sweep.get("suggested_values", [])))
        if values == "all_values":
            values = sweep.get("values", [])
        if not variable or not isinstance(values, list) or not values:
            continue
        scalar_values = []
        for value in values:
            if isinstance(value, (list, tuple, dict)):
                continue
            scalar_values.append(value)
        if len(scalar_values) < 2:
            continue
        specs.append(
            {
                "id": str(sweep.get("id") or sweep.get("name") or f"sweep_{idx + 1}"),
                "variable": variable,
                "raw_variable": raw_variable,
                "canonical_path": canonical_path,
                "values": scalar_values,
                "description": str(sweep.get("description") or sweep.get("note") or ""),
            }
        )
    return specs


def _phase24_metric_name_from_spec(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("name", "metric", "id", "column", "y_metric"):
            text = str(value.get(key) or "").strip()
            if text:
                return text
        return ""
    text = str(value or "").strip()
    if not text:
        return ""
    if text.startswith("{") and text.endswith("}"):
        try:
            parsed = ast.literal_eval(text)
        except Exception:
            parsed = None
        if isinstance(parsed, dict):
            return _phase24_metric_name_from_spec(parsed)
    return text


def _phase24_metric_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        values = [values] if values not in (None, "") else []
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        metric = _phase24_metric_name_from_spec(value)
        if metric and metric not in seen:
            result.append(metric)
            seen.add(metric)
    return result


def _phase24_metric_from_target(target: dict[str, Any], fallback_metrics: list[str], fallback_index: int) -> str:
    for key in ("y_metric", "metric", "primary_metric"):
        value = _phase24_metric_name_from_spec(target.get(key))
        if value:
            return value
    required_metrics = target.get("required_metrics")
    if isinstance(required_metrics, list):
        for value in required_metrics:
            metric = _phase24_metric_name_from_spec(value)
            if (
                metric
                and metric not in {"method", "seed", "swept_value", "scenario_name", "feasible"}
                and not _phase24_is_solver_diagnostic_metric(metric)
                and "violation" not in metric.lower()
            ):
                return metric
    preferred = [
        "weighted_sum_rate_bpsHz",
        "sum_rate_bpsHz",
        "min_user_rate_bpsHz",
        "rate_fairness_jain_index",
        "SINR_dB",
        "SNR_dB",
        "rate_bpsHz",
        "R_c_bpsHz",
        "spectral_efficiency",
        "sensing_gain",
        "sensing_metric",
        "sensing_beampattern_gain",
        "crb_trace",
        "tr_CRB",
        "harvested_power_mW",
        "P_EH_mW",
        "eh_total_mW",
        "objective_value",
        "objective",
    ]
    for metric in preferred:
        if metric in fallback_metrics:
            return metric
    if fallback_metrics:
        return fallback_metrics[min(fallback_index, len(fallback_metrics) - 1)]
    return "objective"


def _phase24_metric_pool(target: dict[str, Any], fallback_metrics: list[str]) -> list[str]:
    pool: list[str] = []
    for key in ("y_metric", "metric", "primary_metric"):
        value = _phase24_metric_name_from_spec(target.get(key))
        if value:
            pool.append(value)
    required_metrics = target.get("required_metrics")
    if isinstance(required_metrics, list):
        pool.extend(_phase24_metric_name_from_spec(item) for item in required_metrics)
    pool.extend(_phase24_metric_name_from_spec(item) for item in fallback_metrics)
    ignored = {"", "method", "seed", "swept_value", "swept_param", "scenario_name", "case_id"}
    cleaned: list[str] = []
    for item in pool:
        if item in ignored or item in cleaned:
            continue
        cleaned.append(item)
    return cleaned


def _phase24_contextual_metric_override(requested: str, candidates: list[str], context_text: str) -> str:
    context = str(context_text or "").lower()
    requested_lower = str(requested or "").lower()

    def choose(preferred: list[str]) -> str:
        return _phase24_choose_metric_from_candidates(candidates, preferred)

    rate_context = any(token in context for token in ("rate", "throughput", "spectral efficiency", "sinr", "secrecy"))
    rate_requested = any(token in requested_lower for token in ("rate", "throughput", "bps", "sinr", "secrecy"))
    sensing_context = any(token in context for token in ("sensing", "radar", "crb", "beampattern", "illumination", "target"))
    sensing_requested = any(token in requested_lower for token in ("sensing", "radar", "crb", "beampattern", "illumination"))
    energy_context = any(
        token in context
        for token in (
            "harvest",
            "harvested",
            "energy",
            "eh",
            "rectifier",
            "rf-to-dc",
            "rf to dc",
            "powering",
        )
    )
    energy_requested = any(token in requested_lower for token in ("harvest", "energy", "eh", "dc", "rf"))
    if (rate_context and rate_requested) or (sensing_context and sensing_requested) or (energy_context and energy_requested):
        return ""
    if energy_context and not energy_requested:
        replacement = choose(
            [
                "min_harvested_dc_mW",
                "true_harvested_energy_mW",
                "harvested_energy_mW",
                "harvested_power_mW",
                "P_EH_mW",
                "eh_total_mW",
                "t_star_mW",
                "min_rf_input_mW",
            ]
        )
        if replacement:
            return replacement

    if sensing_context and not sensing_requested:
        replacement = choose(
            [
                "sensing_illumination_mW",
                "sensing_margin_mW",
                "radar_SNR_dB",
                "radar_snr",
                "sensing_metric",
                "sensing_gain",
                "crb_trace",
                "tr_CRB",
            ]
        )
        if replacement:
            return replacement

    if rate_context and not rate_requested:
        replacement = choose(
            [
                "weighted_sum_rate_bpsHz",
                "sum_rate_bpsHz",
                "min_user_rate_bpsHz",
                "worst_case_min_secrecy_rate_bpsHz",
                "R_sec_min_bpsHz",
                "R_sec_sum_bpsHz",
                "rate_bpsHz",
                "R_c_bpsHz",
                "spectral_efficiency",
            ]
        )
        if replacement:
            return replacement

    return ""


def _phase24_is_solver_diagnostic_metric(metric: str) -> bool:
    lowered = str(metric or "").strip().lower()
    if not lowered:
        return False
    diagnostic_tokens = (
        "iteration",
        "iter",
        "runtime",
        "solve_time",
        "solver_time",
        "time_sec",
        "surrogate",
        "rank",
        "lambda",
        "active",
        "sparsity",
        "eigen",
        "status",
        "gap",
        "residual",
        "feasible",
        "feasibility",
        "violation",
    )
    return any(token in lowered for token in diagnostic_tokens)


def _phase24_is_objective_like_metric(metric: str) -> bool:
    text = str(metric or "").strip().lower().replace("-", "_")
    if not text:
        return False
    exact_objective_names = {
        "objective",
        "objective_value",
        "weighted_objective",
        "utility",
        "weighted_utility",
        "eta_service_level",
        "service_level",
        "normalized_service_level",
        "achieved_tau",
        "tau_star",
        "service_margin",
        "service_margin_tau",
        "normalized_service_margin",
        "min_normalized_service_margin",
        "achieved_margin",
    }
    if text in exact_objective_names:
        return True
    if any(token in text for token in ("objective", "utility")) and not any(
        token in text for token in ("violation", "gap", "residual")
    ):
        return True
    return bool(re.search(r"(^|_)tau(_|$)", text)) or (
        "service" in text and "margin" in text and not any(token in text for token in ("sinr", "sensing", "eh", "energy", "harvest"))
    ) or (
        "service" in text and "level" in text and not any(token in text for token in ("sinr", "sensing", "eh", "energy", "harvest"))
    )


def _phase24_metric_display_label(metric: str, primary_metric_payload: Any = None) -> str:
    """Return a paper-facing fallback label when a normalized metric changes."""

    metric_text = str(metric or "").strip()
    metric_lower = metric_text.lower().replace("-", "_")
    if isinstance(primary_metric_payload, dict) and metric_text == str(primary_metric_payload.get("name") or "").strip():
        display = str(primary_metric_payload.get("display_name") or "").strip()
        if display:
            return display
    labels = {
        "service_margin_tau": r"worst normalized service surplus $\tau$",
        "normalized_service_margin": r"normalized service surplus $\tau$",
        "min_normalized_service_margin": r"minimum normalized service surplus",
        "eta_service_level": r"minimum normalized service level $\eta$",
        "normalized_service_level": r"normalized service level $\eta$",
        "sum_rate_bpshz": r"sum rate $R_{\rm sum}$ (bps/Hz)",
        "weighted_sum_rate_bpshz": r"weighted sum rate $R_{\rm wsr}$ (bps/Hz)",
        "min_user_rate_bpshz": r"minimum user rate $R_{\min}$ (bps/Hz)",
        "min_harvested_dc_mw": r"minimum harvested DC power $P_{\rm dc}^{\min}$ (mW)",
        "harvested_energy_mw": r"harvested energy $E_{\rm h}$ (mW)",
        "true_harvested_energy_mw": r"harvested energy $E_{\rm h}$ (mW)",
        "radar_snr_db": r"radar SNR $\Gamma_{\rm r}$ (dB)",
        "sensing_metric": r"sensing quality",
        "sensing_gain": r"sensing gain",
        "shared_power_fraction": r"shared-power fraction",
        "optimal_rho": r"power-splitting ratio $\rho$",
        "rho": r"power-splitting ratio $\rho$",
        "total_power_w": r"total transmit power $P_{\rm tx}$ (W)",
        "sum_power_w": r"total transmit power $P_{\rm tx}$ (W)",
        "p_tx_mw": r"transmit power $P_{\rm tx}$ (mW)",
    }
    return labels.get(metric_lower, metric_text.replace("_", " "))


def _phase24_axis_label_matches_metric(axis_label: str, metric: str) -> bool:
    label = str(axis_label or "").lower().replace("-", "_")
    metric_lower = str(metric or "").lower().replace("-", "_")
    if not label or not metric_lower:
        return False
    families = [
        (("eta", "tau", "service", "level", "surplus", "utility"), ("eta_service", "service_level", "tau", "service_margin", "normalized_service", "objective", "utility")),
        (("harvest", "energy", "dc", "eh"), ("harvest", "energy", "dc", "eh")),
        (("rate", "throughput", "bps", "sinr"), ("rate", "throughput", "bps", "sinr")),
        (("sensing", "radar", "snr", "crb"), ("sensing", "radar", "snr", "crb")),
        (("power", "p_tx", "transmit"), ("power", "p_tx", "total_power", "sum_power")),
        (("rho", "split", "fraction", "allocation"), ("rho", "fraction", "allocation", "shared_power")),
    ]
    for label_tokens, metric_tokens in families:
        if any(token in label for token in label_tokens) and any(token in metric_lower for token in metric_tokens):
            return True
    compact_label = re.sub(r"[^a-z0-9]+", "", label)
    compact_metric = re.sub(r"[^a-z0-9]+", "", metric_lower)
    return bool(compact_metric and compact_metric in compact_label)


def _phase24_choose_metric_from_candidates(candidates: list[str], preferred: list[str]) -> str:
    candidate_set = {str(item): str(item) for item in candidates if str(item).strip()}
    lower_to_metric = {str(item).lower(): str(item) for item in candidates if str(item).strip()}
    for metric in preferred:
        if metric in candidate_set:
            return candidate_set[metric]
        if metric.lower() in lower_to_metric:
            return lower_to_metric[metric.lower()]
    for metric in candidates:
        if not _phase24_is_solver_diagnostic_metric(metric) and not _phase24_is_objective_like_metric(metric):
            return metric
    for metric in candidates:
        if metric and _phase24_is_objective_like_metric(metric):
            return metric
    return "objective"


def _phase24_target_text(target: dict[str, Any]) -> str:
    return " ".join(
        str(target.get(key) or "")
        for key in (
            "id",
            "claim",
            "chart_intent",
            "intent",
            "evidence_rationale",
            "why_alternatives_are_weaker",
            "x_field",
            "y_metric",
            "required_sweep",
            "required_sweep_param",
        )
    ).lower()


def _phase24_target_metric_name(target: dict[str, Any]) -> str:
    for key in ("y_metric", "metric", "primary_metric"):
        value = target.get(key)
        if isinstance(value, dict):
            value = value.get("name")
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _phase24_is_raw_feasibility_metric(metric: str) -> bool:
    return str(metric or "").strip().lower() in {"feasible", "success", "status"}


def _phase24_is_mechanism_metric(metric: str) -> bool:
    lowered = str(metric or "").strip().lower()
    return any(
        token in lowered
        for token in (
            "rho",
            "partition",
            "separation",
            "harvest",
            "energy",
            "snr",
            "rate",
            "sinr",
            "crb",
            "rank",
            "eigenvalue",
            "constraint_violation",
            "violation",
        )
    )


def _phase24_is_constraint_threshold_self_metric(metric: str, text: str) -> bool:
    metric_lower = str(metric or "").strip().lower()
    text_lower = str(text or "").strip().lower()
    threshold_tokens = (
        "min",
        "minimum",
        "threshold",
        "requirement",
        "q_min",
        "qmin",
        "e_min",
        "emin",
        "r_min",
        "rmin",
        "s_min",
        "smin",
        "gamma",
        "sinr target",
        "target",
    )
    has_threshold_sweep = any(token in text_lower for token in threshold_tokens)
    if not has_threshold_sweep:
        return False
    metric_groups = (
        (("harvest", "energy", "dc", "eh"), ("q_min", "qmin", "e_min", "emin", "energy", "harvest", "dc", "eh")),
        (("rate", "throughput", "sinr"), ("r_min", "rmin", "gamma", "sinr", "rate")),
        (("sensing", "radar", "crb"), ("s_min", "smin", "sensing", "radar", "crb")),
    )
    for metric_tokens, sweep_tokens in metric_groups:
        if any(token in metric_lower for token in metric_tokens) and any(token in text_lower for token in sweep_tokens):
            return True
    return False


def _phase24_is_raw_saturating_mechanism_metric(metric: str, text: str) -> bool:
    """Detect raw mechanism-output KPIs that can go flat under shape-parameter sweeps."""
    metric_lower = str(metric or "").strip().lower()
    text_lower = str(text or "").strip().lower()
    if not metric_lower:
        return False
    if _phase24_is_objective_like_metric(metric_lower) or any(
        token in metric_lower for token in ("utility", "efficiency", "rate", "throughput", "sinr", "snr", "crb")
    ):
        return False
    raw_output_metric = any(
        token in metric_lower
        for token in (
            "harvest",
            "p_dc",
            "dc_power",
            "eh_total",
            "eh_power",
            "p_eh",
            "energy_mw",
            "power_mw",
        )
    )
    mechanism_shape_sweep = any(
        token in text_lower
        for token in (
            "rectifier",
            "sigmoid",
            "logistic",
            "saturation",
            "steepness",
            "turn_on",
            "turn-on",
            "model-shape",
            "shape parameter",
            "conversion curve",
            "nonlinear harvesting",
            "nonlinear eh",
            "rf-to-dc",
            "rf to dc",
        )
    )
    return raw_output_metric and mechanism_shape_sweep


def _phase24_target_needs_multidimensional_data(target: dict[str, Any]) -> bool:
    text = _phase24_target_text(target)
    chart_type = str(target.get("chart_type") or "").strip().lower()
    if chart_type in {"heatmap", "contour", "surface"}:
        return True
    return any(token in text for token in ("2d", "two-dimensional", "two parameter", "two-parameter", "heatmap", "contour"))


def _phase24_evidence_target_score(target: dict[str, Any], *, slot: int) -> tuple[int, int]:
    """Rank predeclared evidence targets without using topic-specific names.

    Slot 0 should usually be the main physical-performance comparison. Slot 1
    should complement it with an ablation, mechanism, robustness, or operating
    regime diagnostic. This prevents the old "first two targets" behavior from
    promoting a convenient but weak 0/1 feasibility line over a stronger
    mechanism figure declared later in the evidence contract.
    """
    text = _phase24_target_text(target)
    metric = _phase24_target_metric_name(target)
    metric_lower = metric.lower()
    score = 0
    if any(token in text for token in ("main_comparison", "primary", "main comparison")):
        score += 16 if slot == 0 else 4
    if any(token in text for token in ("mechanism", "ablation", "sensitivity", "robustness", "structural", "adaptation", "rho")):
        score += 18 if slot == 1 else 5
    if any(token in text for token in ("rho", "structural", "partition", "separation", "adaptation")) and metric_lower in {"optimal_rho", "rho"}:
        score += 8 if slot == 1 else 3
    if _phase24_is_constraint_threshold_self_metric(metric, text):
        score -= 35
    if _phase24_is_raw_saturating_mechanism_metric(metric, text):
        score -= 12
    if any(token in text for token in ("scalability", "complexity")):
        score += 6 if slot == 1 else 1
    if "convergence" in text or "fixed-point rate" in text or "fixed point rate" in text:
        score += 14 if slot == 1 else 2
        if any(token in metric_lower for token in ("error", "gap", "residual", "trajectory")):
            score += 8
    if _phase24_is_mechanism_metric(metric):
        score += 10
    if _phase24_is_solver_diagnostic_metric(metric) or "violation" in metric_lower:
        score -= 8
    if _phase24_is_objective_like_metric(metric):
        score -= 10
    if _phase24_is_raw_feasibility_metric(metric):
        score -= 12
        if any(token in text for token in ("boundary", "operating region", "feasible region")):
            score += 4
    if _phase24_target_needs_multidimensional_data(target):
        score -= 10
    if any(token in metric_lower for token in ("runtime", "time", "iteration", "solver")):
        score -= 4
        if not any(token in text for token in ("scalability", "complexity", "wall-clock", "wall clock", "convergence")):
            score -= 10
    # Stable tie-breaker: keep earlier evidence-contract order when scores match.
    return score, -int(target.get("_evidence_order", 0))


def _phase24_select_evidence_targets(raw_targets: list[Any], limit: int = 2) -> list[dict[str, Any]]:
    targets = [dict(item, _evidence_order=idx) for idx, item in enumerate(raw_targets) if isinstance(item, dict)]
    if not targets:
        return []
    selected: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    used_sweeps: set[str] = set()
    for slot in range(limit):
        candidates = [target for target in targets if str(target.get("id") or target.get("figure_id") or target.get("_evidence_order")) not in used_ids]
        if not candidates:
            break
        # Rank by scientific evidence value, not by artificial sweep diversity.
        # A convergence or feasibility-boundary figure may legitimately use the
        # same operating sweep as the main comparison.
        pool = candidates
        best = max(pool, key=lambda target: _phase24_evidence_target_score(target, slot=slot))
        target_id = str(best.get("id") or best.get("figure_id") or best.get("_evidence_order"))
        selected.append({k: v for k, v in best.items() if k != "_evidence_order"})
        used_ids.add(target_id)
        sweep_key = str(best.get("required_sweep") or best.get("required_sweep_param") or best.get("x_field") or "")
        if sweep_key:
            used_sweeps.add(sweep_key)
    return selected


def _phase24_publication_metric_for_target(
    target: dict[str, Any],
    fallback_metrics: list[str],
    fallback_index: int,
    sweep_spec: dict[str, Any],
    primary_metric: str = "",
) -> str:
    """Prefer physical KPIs over solver diagnostics for paper evidence figures."""
    requested = _phase24_metric_from_target(target, fallback_metrics, fallback_index)
    candidates = _phase24_metric_pool(target, fallback_metrics)
    chart_intent = str(target.get("chart_intent") or target.get("intent") or "").lower()
    requested_lower = str(requested or "").lower()
    claim_text_for_primary = _phase24_target_text(target)
    primary_metric = str(primary_metric or "").strip()
    primary_is_usable = (
        bool(primary_metric)
        and not _phase24_is_solver_diagnostic_metric(primary_metric)
        and not _phase24_is_raw_feasibility_metric(primary_metric)
    )
    mechanism_specific_target = any(
        token in f"{chart_intent} {claim_text_for_primary}"
        for token in (
            "mechanism",
            "ablation",
            "structural",
            "adaptation",
            "rho",
            "convergence",
            "complexity",
            "runtime",
        )
    )
    primary_is_objective_like = _phase24_is_objective_like_metric(primary_metric)
    if primary_is_usable and not mechanism_specific_target:
        requested_is_weak = (
            _phase24_is_solver_diagnostic_metric(requested)
            or _phase24_is_raw_feasibility_metric(requested)
            or _phase24_is_constraint_threshold_self_metric(requested, claim_text_for_primary)
            or _phase24_is_raw_saturating_mechanism_metric(requested, claim_text_for_primary)
        )
        if fallback_index == 0:
            return primary_metric
        if not primary_is_objective_like:
            if (
                any(token in chart_intent for token in ("main", "comparison", "stress", "gain", "tradeoff", "sensitivity", "robust"))
                or requested_is_weak
            ):
                return primary_metric
    requested_mechanism_specific = any(
        token in requested_lower
        for token in (
            "rho",
            "partition",
            "separation",
            "active_count",
            "selected",
            "placement",
            "position",
        )
    )
    if mechanism_specific_target and requested_mechanism_specific and not (
        _phase24_is_solver_diagnostic_metric(requested)
        or _phase24_is_raw_feasibility_metric(requested)
        or _phase24_is_constraint_threshold_self_metric(requested, claim_text_for_primary)
    ):
        return requested
    axis_metric_context = " ".join(
        [
            str((target.get("axis_labels") or {}).get("y") if isinstance(target.get("axis_labels"), dict) else ""),
            str(target.get("y_axis_label") or ""),
            str(target.get("y_display_name") or ""),
        ]
    ).lower()
    axis_override = _phase24_contextual_metric_override(requested, candidates, axis_metric_context)
    if axis_override:
        return axis_override
    if axis_metric_context and (
        (
            any(token in axis_metric_context for token in ("rate", "throughput", "bps", "sinr", "secrecy"))
            and any(token in requested_lower for token in ("rate", "throughput", "bps", "sinr", "secrecy"))
        )
        or (
            any(token in axis_metric_context for token in ("harvest", "energy", "eh", "dc", "rf"))
            and any(token in requested_lower for token in ("harvest", "energy", "eh", "dc", "rf"))
        )
        or (
            any(token in axis_metric_context for token in ("sensing", "radar", "crb", "beampattern", "illumination"))
            and any(token in requested_lower for token in ("sensing", "radar", "crb", "beampattern", "illumination"))
        )
    ):
        return requested
    diagnostic_requested = _phase24_is_solver_diagnostic_metric(requested) or "violation" in requested_lower
    claim_text = " ".join(
        [
            str(target.get("claim") or ""),
            str(target.get("evidence_rationale") or ""),
            str(target.get("why_alternatives_are_weaker") or ""),
            str(target.get("intended_insight") or ""),
            str(target.get("primary_message") or ""),
            str(target.get("trend_hypothesis") or ""),
            str(target.get("expected_trend") or ""),
            str(target.get("active_regime_note") or ""),
            str(target.get("required_sweep") or ""),
            str(sweep_spec.get("id") or ""),
            str(sweep_spec.get("variable") or ""),
            str(sweep_spec.get("canonical_path") or ""),
            str(sweep_spec.get("description") or ""),
            str((target.get("axis_labels") or {}).get("x") if isinstance(target.get("axis_labels"), dict) else ""),
            str((target.get("axis_labels") or {}).get("y") if isinstance(target.get("axis_labels"), dict) else ""),
            str(target.get("y_axis_label") or ""),
            str(target.get("y_display_name") or ""),
        ]
    ).lower()
    contextual_override = "" if diagnostic_requested else _phase24_contextual_metric_override(requested, candidates, claim_text)
    if contextual_override:
        return contextual_override
    objective_requested = _phase24_is_objective_like_metric(requested)
    threshold_self_metric_requested = _phase24_is_constraint_threshold_self_metric(requested, claim_text)
    saturating_mechanism_metric_requested = _phase24_is_raw_saturating_mechanism_metric(requested, claim_text)
    mechanism_like = any(
        token in chart_intent
        for token in ("mechanism", "sensitivity", "robustness", "feasibility", "scalability", "structural")
    )
    if _phase24_is_raw_feasibility_metric(requested) and any(
        token in claim_text for token in ("rho", "structural", "partition", "separation", "adaptation")
    ):
        preferred = [
            "optimal_rho",
            "rho",
            "M_eh",
            "M_rx",
            "true_harvested_energy_mW",
            "harvested_energy_mW",
        ]
        replacement = _phase24_choose_metric_from_candidates(candidates, preferred)
        if replacement and not _phase24_is_raw_feasibility_metric(replacement):
            return replacement
    if (
        not threshold_self_metric_requested
        and not saturating_mechanism_metric_requested
        and not diagnostic_requested
        and not (objective_requested and mechanism_like)
    ):
        return requested

    preferred: list[str] = []
    if saturating_mechanism_metric_requested:
        preferred.extend(
            [
                "objective",
                "objective_value",
                "weighted_sum_rate_bpsHz",
                "sum_rate_bpsHz",
                "sensing_quality",
                "rate_energy_utility",
                "utility",
            ]
        )
    if any(token in claim_text for token in ("eh", "energy", "harvest", "sigmoid", "nonlinear", "powering", "rectifier", "saturation")):
        preferred.extend(
            [
                "objective",
                "objective_value",
                "min_harvested_dc_mW",
                "t_star_mW",
                "true_harvested_energy_mW",
                "harvested_energy_mW",
                "P_EH_mW",
                "eh_total_mW",
                "harvested_power_mW",
                "P_in_actual_mW",
                "sensing_quality",
                "weighted_sum_rate_bpsHz",
                "sum_rate_bpsHz",
            ]
        )
    if any(token in claim_text for token in ("radar", "sensing", "snr", "crb")):
        preferred.extend(["radar_SNR_dB", "radar_snr", "sensing_metric", "sensing_gain", "crb_trace", "tr_CRB"])
    if any(token in claim_text for token in ("rate", "communication", "sinr", "throughput", "sum-rate", "sum rate")):
        preferred.extend(
            [
                "weighted_sum_rate_bpsHz",
                "sum_rate_bpsHz",
                "min_user_rate_bpsHz",
                "max_user_rate_bpsHz",
                "rate_fairness_jain_index",
                "rate_bpsHz",
                "R_c_bpsHz",
                "R_c",
                "min_sinr",
                "spectral_efficiency",
            ]
        )
    if any(token in claim_text for token in ("rho", "structural", "partition", "separation")):
        preferred.extend(
            [
                "weighted_sum_rate_bpsHz",
                "sum_rate_bpsHz",
                "min_user_rate_bpsHz",
                "lambda_star_active_count",
                "optimal_rho",
                "rho",
                "M_eh",
                "M_rx",
            ]
        )
    if any(token in claim_text for token in ("constraint", "feasible", "feasibility", "violation", "outage")):
        preferred.extend(
            [
                "weighted_sum_rate_bpsHz",
                "sum_rate_bpsHz",
                "min_user_rate_bpsHz",
                "rate_fairness_jain_index",
                "radar_SNR_dB",
                "sensing_metric",
                "true_harvested_energy_mW",
                "harvested_energy_mW",
                "P_tx_mW",
                "sum_power_mW",
                "sum_power_W",
            ]
        )
    preferred.extend(
        [
            "radar_SNR_dB",
            "weighted_sum_rate_bpsHz",
            "sum_rate_bpsHz",
            "min_user_rate_bpsHz",
            "rate_fairness_jain_index",
            "true_harvested_energy_mW",
            "harvested_energy_mW",
            "lambda_star_active_count",
            "optimal_rho",
            "rho",
        ]
    )
    if primary_is_objective_like and fallback_index > 0:
        preferred = [metric for metric in preferred if not _phase24_is_objective_like_metric(metric)]
    return _phase24_choose_metric_from_candidates(candidates, preferred)


def _phase24_executable_chart_type(raw_chart_type: str, sweep_values: list[Any], claim: str) -> str:
    chart_type = str(raw_chart_type or "").strip().lower()
    numeric_values = 0
    for value in sweep_values:
        try:
            float(value)
            numeric_values += 1
        except Exception:
            pass
    claim_lower = claim.lower()
    ordered_numeric_sweep = numeric_values >= 3 and numeric_values == len(sweep_values)
    categorical_claim = any(token in claim_lower for token in ["category", "categorical", "ablation set", "method family"])
    if chart_type in {"heatmap", "convergence"}:
        return "line" if numeric_values >= 3 else "grouped_bar"
    if chart_type in {"scatter", "scatter_trend", "scatter_with_trend", "line"}:
        scatter_claim = any(
            token in claim_lower
            for token in [
                "tradeoff",
                "pareto",
                "regime",
                "operating regime",
                "stochastic",
                "monte carlo",
                "random",
                "noisy",
                "gain profile",
            ]
        )
        if scatter_claim:
            return "scatter_trend" if ordered_numeric_sweep else "scatter"
        return "line"
    if chart_type in {"box", "grouped_bar", "bar", "categorical_summary", "ablation_bar"}:
        if chart_type != "box" and ordered_numeric_sweep and not categorical_claim:
            return "line"
        return chart_type
    return "line" if numeric_values >= 3 else "grouped_bar"


def _phase24_method_id(value: Any, fallback: str) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", str(value or "")).strip("_").lower()
    return text or fallback


def _phase24_normalize_method_entry(item: Any, fallback_id: str, default_role: str) -> dict[str, Any]:
    if isinstance(item, dict):
        raw_id = item.get("id") or item.get("internal_name") or item.get("method_id") or item.get("name") or fallback_id
        method_id = _phase24_method_id(raw_id, fallback_id)
        short = str(item.get("display_name_short") or item.get("short_name") or item.get("name") or method_id).strip()
        long = str(item.get("display_name_long") or item.get("description") or item.get("name") or short).strip()
        role = str(item.get("role") or default_role).strip() or default_role
        role_lower = role.lower()
        mandatory_status = str(
            item.get("mandatory_status")
            or item.get("method_status")
            or item.get("execution_status")
            or item.get("status")
            or ("mandatory" if role_lower in {"proposed", "main_baseline"} else "")
        ).strip()
        raw_id_text = str(raw_id or "").strip().lower()
        if role_lower == "proposed" and raw_id_text in {"", "proposed", "proposal", "main"}:
            method_id = "proposed"
        return {
            "id": method_id,
            "internal_name": method_id,
            "name": method_id,
            "role": role,
            "mandatory_status": mandatory_status,
            "display_name_short": short,
            "display_name_long": long,
            "scientific_purpose": str(item.get("scientific_purpose") or item.get("purpose") or item.get("claim") or "").strip(),
            "implementation_hint": str(item.get("implementation_hint") or item.get("implementation") or item.get("solver_hint") or "").strip(),
            "fairness_rule": str(item.get("fairness_rule") or item.get("fairness") or "").strip(),
        }
    label = str(item or fallback_id).strip() or fallback_id
    method_id = _phase24_method_id(label, fallback_id)
    return {
        "id": method_id,
        "internal_name": method_id,
        "name": method_id,
        "role": default_role,
        "mandatory_status": "mandatory" if default_role == "proposed" else "",
        "display_name_short": label,
        "display_name_long": label,
        "scientific_purpose": "",
        "implementation_hint": "",
        "fairness_rule": "",
    }


def _phase24_collect_method_values(container: dict[str, Any], key: str) -> list[Any]:
    values = container.get(key, [])
    if isinstance(values, list):
        return values
    if isinstance(values, dict):
        collected: list[Any] = []
        for name, value in values.items():
            if isinstance(value, dict):
                merged = dict(value)
                merged.setdefault("id", name)
                collected.append(merged)
            else:
                collected.append({"id": name, "display_name_short": str(value), "display_name_long": str(value)})
        return collected
    return []


def _phase24_contract_methods(evidence: dict[str, Any], plan: dict[str, Any]) -> list[dict[str, Any]]:
    methods_by_id: dict[str, dict[str, Any]] = {}

    def add(item: Any, fallback_id: str, role: str) -> None:
        method = _phase24_normalize_method_entry(item, fallback_id, role)
        method_id = method["id"]
        if method_id not in methods_by_id:
            methods_by_id[method_id] = method
            return
        existing = methods_by_id[method_id]
        for key, value in method.items():
            if value and not existing.get(key):
                existing[key] = value

    add({"id": "proposed", "display_name_short": "Proposed", "display_name_long": "Proposed optimization method"}, "proposed", "proposed")

    method_sections = [
        ("compared_methods", "comparison"),
        ("benchmark_methods", "benchmark"),
        ("baseline_methods", "benchmark"),
        ("heuristic_methods", "heuristic"),
        ("ablation_methods", "mechanism_ablation"),
        ("upper_bound_methods", "upper_bound"),
        ("oracle_methods", "upper_bound"),
        ("methods", "comparison"),
    ]
    for container in (evidence, plan):
        if not isinstance(container, dict):
            continue
        for key, role in method_sections:
            for idx, item in enumerate(_phase24_collect_method_values(container, key)):
                add(item, f"{role}_{idx + 1}", role)

    figures = evidence.get("figures", [])
    if not isinstance(figures, list):
        figures = plan.get("figure_targets", [])
    if isinstance(figures, list):
        for fig_idx, figure in enumerate(figures):
            if not isinstance(figure, dict):
                continue
            for key in ("methods_to_run", "methods", "methods_required", "series", "curves"):
                values = figure.get(key, [])
                if not isinstance(values, list):
                    continue
                for idx, item in enumerate(values):
                    add(item, f"figure_{fig_idx + 1}_method_{idx + 1}", "figure_method")

    if not any(method_id != "proposed" for method_id in methods_by_id):
        add(
            {
                "id": "baseline",
                "display_name_short": "Baseline",
                "display_name_long": "Fixed-harness fair benchmark slot",
                "scientific_purpose": (
                    "Provides the fixed-harness comparison entry only when no concrete Phase 2.4 benchmark was declared."
                ),
                "implementation_hint": (
                    "Implement baseline_solution as the strongest available feasible heuristic from the benchmark definition."
                ),
                "fairness_rule": "Use the same channels, budgets, constraints, and evaluation metrics as the proposed method.",
                "mandatory_status": "optional_diagnostic",
            },
            "baseline",
            "main_baseline",
        )

    return list(methods_by_id.values())


def _phase24_method_ids_for_target(target: dict[str, Any], methods: list[dict[str, Any]]) -> list[str]:
    aliases: dict[str, str] = {}
    for method in methods:
        method_id = str(method.get("id") or method.get("internal_name") or method.get("name") or "").strip()
        if not method_id:
            continue
        for key in ("id", "internal_name", "name", "display_name_short", "display_name_long"):
            value = str(method.get(key) or "").strip().lower()
            if value:
                aliases[value] = method_id
    selected: list[str] = []
    for key in ("methods_to_run", "methods", "methods_required", "series", "curves"):
        values = target.get(key, [])
        if not isinstance(values, list):
            continue
        for item in values:
            raw = item.get("id") or item.get("internal_name") or item.get("name") if isinstance(item, dict) else item
            normalized = _phase24_method_id(raw, "")
            method_id = aliases.get(str(raw or "").strip().lower()) or aliases.get(normalized) or normalized
            if method_id and method_id not in selected:
                selected.append(method_id)
    if selected:
        return selected
    return [str(method.get("id") or method.get("internal_name") or method.get("name")) for method in methods if str(method.get("id") or method.get("internal_name") or method.get("name", "")).strip()]


def _phase24_filter_methods_for_evidence(methods: list[dict[str, Any]], method_ids: list[str], metric: str, chart_intent: str, claim: str) -> list[str]:
    method_by_id = {str(method.get("id") or method.get("internal_name") or method.get("name")): method for method in methods}
    evidence_text = " ".join([metric, chart_intent, claim]).lower()
    explicit_reference_claim = any(
        token in evidence_text
        for token in ("optimality gap", "optimality-gap", "upper bound", "upper-bound", "oracle", "relaxation", "centralized reference")
    )
    exclude_upper_bound = not explicit_reference_claim or any(
        token in evidence_text for token in ("violation", "feasible", "feasibility", "outage", "constraint", "runtime")
    )
    if not exclude_upper_bound:
        filtered = list(method_ids)
    else:
        filtered = []
        for method_id in method_ids:
            role = str(method_by_id.get(method_id, {}).get("role", "")).lower()
            if any(token in role for token in ("upper", "oracle", "relax")):
                continue
            filtered.append(method_id)
        filtered = filtered or method_ids
    return _phase24_limit_methods_for_evidence(methods, filtered, metric, chart_intent, claim)


def _phase24_limit_methods_for_evidence(methods: list[dict[str, Any]], method_ids: list[str], metric: str, chart_intent: str, claim: str) -> list[str]:
    """Keep figures focused while preserving Phase 2.4 mandatory practical baselines."""
    unique = [method for method in dict.fromkeys(str(item).strip() for item in method_ids if str(item).strip()) if method]
    if not unique:
        return unique
    if "proposed" not in unique:
        unique.insert(0, "proposed")
    method_by_id = {str(method.get("id") or method.get("internal_name") or method.get("name")): method for method in methods}
    for method in methods:
        method_id = str(method.get("id") or method.get("internal_name") or method.get("name") or "").strip()
        if not method_id or method_id in unique:
            continue
        role = str(method.get("role", "")).lower()
        status = str(method.get("mandatory_status", "")).lower()
        if "mandatory" in status and not any(token in role for token in ("upper", "oracle", "relax")):
            unique.append(method_id)
    intent_text = " ".join([metric, chart_intent, claim]).lower()
    if any(token in intent_text for token in ("main", "comparison", "optimal", "sum_power", "sum power", "objective")):
        has_practical = any(
            any(token in str(method_by_id.get(method, {}).get("role", "")).lower() for token in ("heuristic", "benchmark", "ablation"))
            for method in unique
            if method != "proposed"
        )
        if not has_practical:
            for method in methods:
                method_id = str(method.get("id") or method.get("internal_name") or method.get("name") or "").strip()
                role = str(method.get("role", "")).lower()
                if method_id and method_id not in unique and any(token in role for token in ("heuristic", "benchmark", "ablation")):
                    unique.append(method_id)
                    break
    mechanism_intent = any(token in intent_text for token in ("mechanism", "ablation", "convergence", "sensitivity", "robustness"))
    if any(token in intent_text for token in ("mechanism", "feasible", "feasibility", "constraint", "ablation", "boundary", "convergence")):
        max_methods = 5
        role_priority = {
            "mandatory": -1,
            "mechanism_ablation": 0,
            "model_ablation": 0,
            "main_baseline": 1,
            "heuristic": 2,
            "comparison": 3,
        }
    else:
        # Run the mandatory practical baseline family first; Phase 2.5 can later
        # choose the clearest subset for the WCL figure.
        max_methods = 5
        role_priority = {
            "mandatory": -1,
            "heuristic": 0,
            "benchmark": 0,
            "main_baseline": 0,
            "mechanism_ablation": 1,
            "model_ablation": 1,
            "comparison": 3,
        }
        has_mechanism = any(
            any(token in str(method_by_id.get(method, {}).get("role", "")).lower() for token in ("mechanism_ablation", "model_ablation"))
            for method in unique
        )
        if mechanism_intent and not has_mechanism:
            for method in methods:
                method_id = str(method.get("id") or method.get("internal_name") or method.get("name") or "").strip()
                role = str(method.get("role", "")).lower()
                if method_id and method_id not in unique and any(token in role for token in ("mechanism_ablation", "model_ablation")):
                    unique.append(method_id)
                    break
    proposed = [method for method in unique if method == "proposed"]
    others = [method for method in unique if method != "proposed"]

    def sort_key(method_id: str) -> tuple[int, int]:
        role = str(method_by_id.get(method_id, {}).get("role", "")).lower()
        status = str(method_by_id.get(method_id, {}).get("mandatory_status", "")).lower()
        priority = 4
        if "mandatory" in status:
            priority = -1
        for token, value in role_priority.items():
            if token in role:
                priority = min(priority, value)
                break
        return priority, unique.index(method_id)

    selected = proposed[:1] + sorted(others, key=sort_key)[: max(0, max_methods - len(proposed[:1]))]
    return selected or unique[:max_methods]


def _phase24_normalize_semantic_forbidden_concepts(plan: dict[str, Any]) -> None:
    """Keep semantic forbidden-concepts conditional on mechanisms actually present."""
    guardrails = plan.get("semantic_guardrails", {})
    if not isinstance(guardrails, dict):
        guardrails = {}
    forbidden_concepts = [
        str(item).strip()
        for item in guardrails.get("forbidden_concepts", [])
        if str(item).strip()
    ]
    plan_text = yaml.safe_dump(plan, sort_keys=False, allow_unicode=True).lower()
    concept_groups = [
        (
            ["energy_harvesting", "harvested", "SWIPT", "sigmoid_EH", "EH_receiver"],
            [
                "energy harvesting",
                "swipt",
                "wireless power transfer",
                "wireless powered",
                "wpt",
                "harvest",
                "energy causality",
                "battery",
                "eh_",
                "p_eh",
            ],
        ),
        (["radar", "CRB", "FIM", "beam_pattern", "sensing_beam"], ["radar", "crb", "fisher", "fim", "sensing"]),
        (["RIS", "IRS", "STAR_RIS", "reconfigurable_intelligent"], ["ris", "irs", "reflecting surface", "reconfigurable intelligent"]),
    ]
    for concepts, activation_terms in concept_groups:
        concept_set = {concept.lower() for concept in concepts}
        is_active = any(term.lower() in plan_text for term in activation_terms)
        if is_active:
            forbidden_concepts = [concept for concept in forbidden_concepts if concept.lower() not in concept_set]
            continue
        existing = {item.lower() for item in forbidden_concepts}
        for concept in concepts:
            if concept.lower() not in existing:
                forbidden_concepts.append(concept)
                existing.add(concept.lower())
    if forbidden_concepts:
        guardrails["forbidden_concepts"] = forbidden_concepts
    else:
        guardrails.pop("forbidden_concepts", None)
    if guardrails:
        plan["semantic_guardrails"] = guardrails
    else:
        plan.pop("semantic_guardrails", None)


def _phase24_repair_practical_baseline_feasibility(plan: dict[str, Any], evidence: dict[str, Any]) -> None:
    """Keep practical baselines inside frozen association/sparsity constraints.

    LLM-generated validation plans sometimes describe a practical RZF/ZF
    benchmark as "full association" even when the frozen problem includes a
    hard per-user association sparsity limit such as L_max. That makes the
    benchmark infeasible and breaks paired-gain validation. Full-cooperation
    methods can still exist, but only as oracle/reference diagnostics.
    """

    canonical = plan.get("canonical_config")
    if not isinstance(canonical, dict):
        return
    association = canonical.get("association")
    if not isinstance(association, dict):
        association = {}
    lmax = (
        association.get("L_max")
        or association.get("Lmax")
        or association.get("max_association")
        or canonical.get("L_max")
        or canonical.get("Lmax")
    )
    if lmax in {None, ""}:
        return

    methods = evidence.get("compared_methods")
    if not isinstance(methods, list):
        return

    for method in methods:
        if not isinstance(method, dict):
            continue
        method_id = str(method.get("id") or method.get("internal_name") or method.get("name") or "").lower()
        display = str(method.get("display_name_long") or method.get("display_name_short") or "").lower()
        role = str(method.get("role") or "").lower()
        combined = " ".join(
            str(method.get(key) or "").lower()
            for key in ("id", "name", "display_name_short", "display_name_long", "scientific_purpose", "implementation_hint", "fairness_rule")
        )
        full_assoc = any(
            token in combined
            for token in (
                "full association",
                "full-association",
                "all ap",
                "all aps",
                "all access point",
                "serve all users",
                "all users at every ap",
                "no sparsity",
                "without sparsity",
            )
        )
        is_zf_family = any(token in method_id or token in display or token in combined for token in ("zf", "zero-forcing", "zero forcing"))
        is_reference = any(token in role or token in method_id or token in display for token in ("oracle", "reference", "upper_bound", "upper-bound", "full_cooperation"))

        if full_assoc and is_reference:
            method["role"] = "reference_oracle"
            method["mandatory_status"] = str(method.get("mandatory_status") or "optional_diagnostic")
            method["fairness_rule"] = (
                f"Diagnostic only under the association limit L_max={lmax}; not used as the main practical benchmark because full association relaxes the frozen sparsity constraint."
            )
            continue

        if full_assoc and is_zf_family and not is_reference:
            method["implementation_hint"] = (
                f"Use fixed top-L_max AP-user association with L_max={lmax} from channel strength, then compute regularized-ZF or matched-filter directions and fair power loading under the same per-AP power and association constraints."
            )
            method["fairness_rule"] = (
                f"Same channels, same per-AP power budgets, same noise floor, same user set, and the same association sparsity limit L_max={lmax}; do not use full association as the practical plotted benchmark."
            )
            method["scientific_purpose"] = str(method.get("scientific_purpose") or "").replace("full association", f"fixed top-L_max association with L_max={lmax}")


def normalize_phase24_validation_plan_yaml(yaml_text: str) -> str:
    """Make Phase 2.4 experiment contracts executable by the fixed harness.

    The LLM may propose ambitious paper evidence such as multi-axis Pareto
    surfaces or convergence traces. Those are useful as aspirations, but the
    The Phase 2.4/2.5 harness executes scalar sweeps over methods declared by the paper
    evidence contract. This normalizer preserves the scientific claims while
    adding an explicit executable contract that downstream code can run and
    audit without topic-specific method names.
    """
    text = sanitize_phase24_validation_plan_yaml(yaml_text)
    plan = _phase24_yaml_mapping(text)
    if not isinstance(plan, dict):
        return text
    _phase24_normalize_semantic_forbidden_concepts(plan)

    raw_sweeps = plan.get("sweep_definitions")
    if isinstance(raw_sweeps, dict):
        normalized_sweeps = []
        for idx, (name, sweep) in enumerate(raw_sweeps.items()):
            if not isinstance(sweep, dict):
                continue
            item = dict(sweep)
            item.setdefault("id", str(name) or f"sweep_{idx + 1}")
            normalized_sweeps.append(item)
        plan["sweep_definitions"] = normalized_sweeps
        raw_sweeps = normalized_sweeps
    if isinstance(raw_sweeps, list):
        for idx, sweep in enumerate(raw_sweeps):
            if not isinstance(sweep, dict):
                continue
            if not str(sweep.get("id") or sweep.get("name") or "").strip():
                sweep["id"] = str(sweep.get("sweep_id") or f"sweep_{idx + 1}").strip()
            if not str(sweep.get("variable") or sweep.get("canonical_path") or "").strip() and str(sweep.get("path") or "").strip():
                sweep["variable"] = str(sweep.get("path") or "").strip()

    scalar_sweeps = _phase24_scalar_sweep_specs(plan)
    if not scalar_sweeps:
        return yaml.safe_dump(plan, sort_keys=False, allow_unicode=True)

    required_outputs = plan.get("required_outputs", {})
    scalar_metrics = []
    if isinstance(required_outputs, dict) and isinstance(required_outputs.get("scalar_metrics"), list):
        scalar_metrics = [
            metric
            for metric in (_phase24_metric_name_from_spec(item) for item in required_outputs.get("scalar_metrics", []))
            if metric
        ]
    required_columns = ["method", "seed", "swept_param", "swept_value", "scenario_name", "objective", "feasible"]
    evidence = plan.get("research_evidence_contract", {})
    if not isinstance(evidence, dict) or not evidence:
        evidence = plan.get("paper_evidence_contract", {})
    if not isinstance(evidence, dict):
        evidence = {}
    primary_metric_payload = evidence.get("primary_metric")
    if isinstance(primary_metric_payload, dict):
        primary_metric_name = str(primary_metric_payload.get("name") or primary_metric_payload.get("metric") or "").strip()
    else:
        primary_metric_name = str(primary_metric_payload or "").strip()
    if primary_metric_name and primary_metric_name not in scalar_metrics:
        scalar_metrics.insert(0, primary_metric_name)
    if not isinstance(evidence.get("figures"), list):
        candidate_figures = evidence.get("figure_candidates")
        if not isinstance(candidate_figures, list):
            candidate_figures = plan.get("figure_candidates")
        if isinstance(candidate_figures, list) and candidate_figures:
            evidence["figures"] = candidate_figures
    if not scalar_metrics:
        ignored_columns = {
            "method",
            "seed",
            "swept_param",
            "swept_value",
            "scenario_name",
            "case_id",
            "objective",
            "feasible",
            "status",
            "message",
            "converged",
            "iterations",
            "solve_time_s",
            "solve_time_sec",
            "runtime_s",
            "runtime_sec",
            "unavailable_reason",
        }
        metric_candidates: list[str] = []
        for figure in evidence.get("figures", []):
            if isinstance(figure, dict):
                metric = _phase24_metric_name_from_spec(
                    figure.get("y_metric") or figure.get("metric") or figure.get("primary_metric")
                )
                if metric:
                    metric_candidates.append(metric)
        for column in evidence.get("required_result_columns", []):
            metric = _phase24_metric_name_from_spec(column)
            if metric and metric not in ignored_columns:
                metric_candidates.append(metric)
        seen_metrics: set[str] = set()
        scalar_metrics = []
        for metric in metric_candidates:
            if metric not in seen_metrics:
                scalar_metrics.append(metric)
                seen_metrics.add(metric)
        if scalar_metrics:
            if not isinstance(required_outputs, dict):
                required_outputs = {}
            required_outputs["scalar_metrics"] = scalar_metrics
            plan["required_outputs"] = required_outputs
    diagnostic_scalar_metrics = [metric for metric in scalar_metrics if _phase24_is_solver_diagnostic_metric(metric)]
    evidence.setdefault("diagnostics", [])
    if isinstance(evidence.get("diagnostics"), list):
        evidence["diagnostics"] = sorted(_phase24_metric_list([*evidence.get("diagnostics", []), *diagnostic_scalar_metrics]))
    figure_scalar_metrics = [metric for metric in scalar_metrics if not _phase24_is_solver_diagnostic_metric(metric)]
    if not figure_scalar_metrics:
        figure_scalar_metrics = [metric for metric in scalar_metrics if metric not in {"method", "seed", "swept_value", "scenario_name"}]
    if isinstance(required_outputs, dict):
        required_outputs["scalar_metrics"] = figure_scalar_metrics
        diagnostics = required_outputs.get("diagnostics")
        if not isinstance(diagnostics, list):
            diagnostics = []
        required_outputs["diagnostics"] = sorted(_phase24_metric_list([*diagnostics, *diagnostic_scalar_metrics]))
        plan["required_outputs"] = required_outputs
    _phase24_repair_practical_baseline_feasibility(plan, evidence)
    contract_methods = _phase24_contract_methods(evidence, plan)
    contract_method_ids = [
        str(method.get("id") or method.get("internal_name") or method.get("name")).strip()
        for method in contract_methods
        if str(method.get("id") or method.get("internal_name") or method.get("name", "")).strip()
    ]
    for value in evidence.get("required_result_columns", []):
        column = _phase24_metric_name_from_spec(value)
        if column and not _phase24_is_solver_diagnostic_metric(column) and column not in required_columns:
            required_columns.append(column)
    for metric in figure_scalar_metrics:
        if metric and not _phase24_is_solver_diagnostic_metric(metric) and metric not in required_columns:
            required_columns.append(metric)

    raw_targets: list[Any] = []
    figure_targets = plan.get("figure_targets")
    if isinstance(figure_targets, list):
        raw_targets.extend(figure_targets)
    top_level_figures = plan.get("figures")
    if isinstance(top_level_figures, list):
        raw_targets.extend(top_level_figures)
    evidence_figures = evidence.get("figures")
    if isinstance(evidence_figures, list):
        raw_targets.extend(evidence_figures)
    raw_targets = _phase24_select_evidence_targets(raw_targets, limit=2) or raw_targets

    normalized_figures: list[dict[str, Any]] = []
    used_sweeps: set[str] = set()
    target_count = max(2, min(3, len(raw_targets) if raw_targets else len(scalar_sweeps)))
    for idx in range(target_count):
        target = raw_targets[idx] if idx < len(raw_targets) and isinstance(raw_targets[idx], dict) else {}
        requested_sweep = target.get("required_sweep", "")
        requested_ids = [str(item) for item in requested_sweep] if isinstance(requested_sweep, list) else [str(requested_sweep)]
        requested_ids = [item.strip() for item in requested_ids if item.strip()]
        chosen = next(
            (
                spec
                for spec in scalar_sweeps
                if any(
                    requested in {str(spec.get("id")), str(spec.get("variable")), str(spec.get("canonical_path")), str(spec.get("raw_variable"))}
                    for requested in requested_ids
                )
            ),
            None,
        )
        if chosen is None:
            chosen = next((spec for spec in scalar_sweeps if spec["id"] not in used_sweeps), scalar_sweeps[min(idx, len(scalar_sweeps) - 1)])
        used_sweeps.add(str(chosen["id"]))
        claim = str(target.get("claim") or evidence.get("required_result", {}).get("statement") if isinstance(evidence.get("required_result"), dict) else target.get("claim") or chosen.get("description") or "Claim-focused evidence")
        metric = _phase24_publication_metric_for_target(
            target,
            scalar_metrics,
            idx,
            chosen,
            primary_metric=primary_metric_name,
        )
        chart_type = _phase24_executable_chart_type(str(target.get("chart_type", "")), list(chosen.get("values", [])), claim)
        figure_id = str(target.get("id") or f"figure_{idx + 1}")
        if not figure_id.startswith("figure_"):
            figure_id = f"figure_{idx + 1}"
        chart_intent = str(target.get("chart_intent") or target.get("intent") or "main_comparison")
        if chart_intent == "convergence" and metric.lower() in {"power_error_norm", "final_relative_error", "error_norm"}:
            target_required_metrics = [
                value
                for value in (
                    _phase24_metric_name_from_spec(item)
                    for item in (target.get("required_metrics", []) if isinstance(target.get("required_metrics"), list) else [])
                )
                if value
            ]
            replacement = _phase24_choose_metric_from_candidates(
                figure_scalar_metrics + target_required_metrics,
                ["spectral_radius_F", "rho_F", "convergence_iter", "iteration_count"],
            )
            if replacement:
                metric = replacement
        methods_to_run = _phase24_filter_methods_for_evidence(
            contract_methods,
            _phase24_method_ids_for_target(target, contract_methods),
            metric,
            chart_intent,
            claim,
        )
        axis_labels = target.get("axis_labels") if isinstance(target.get("axis_labels"), dict) else {}
        if not axis_labels:
            x_axis_label = str(target.get("x_axis_label") or target.get("x_display_name") or "").strip()
            y_axis_label = str(target.get("y_axis_label") or target.get("y_display_name") or "").strip()
            if x_axis_label or y_axis_label:
                axis_labels = {}
                if x_axis_label:
                    axis_labels["x"] = x_axis_label
                if y_axis_label:
                    axis_labels["y"] = y_axis_label
        else:
            axis_labels = dict(axis_labels)
        y_axis_label = str(axis_labels.get("y") or "").strip()
        if not y_axis_label or not _phase24_axis_label_matches_metric(y_axis_label, metric):
            axis_labels["y"] = _phase24_metric_display_label(metric, primary_metric_payload)
        chart_choice_rationale = str(
            target.get("chart_choice_rationale")
            or target.get("evidence_rationale")
            or target.get("evidence_rule")
            or ""
        ).strip()
        if not chart_choice_rationale:
            chart_choice_rationale = (
                "The evidence is mapped to a scalar sweep that the Phase 2.4/2.5 harness can execute reproducibly; "
                "multi-sweep surfaces or convergence traces are deferred until the solver emits the required raw data."
            )
        trend_hypothesis = str(target.get("trend_hypothesis") or "").strip()
        expected_trend = str(target.get("expected_trend") or trend_hypothesis).strip()
        if not trend_hypothesis and expected_trend:
            trend_hypothesis = expected_trend
        normalized_figures.append(
            {
                "id": figure_id,
                "claim": claim,
                "chart_intent": chart_intent,
                "chart_type": chart_type,
                "chart_choice_rationale": chart_choice_rationale,
                "why_alternatives_are_weaker": (
                    str(target.get("why_alternatives_are_weaker") or "").strip()
                    or
                    "A chart requiring unrun cross-sweeps, per-iteration logs, or undeclared method functions would "
                    "overstate the available evidence."
                ),
                "intended_insight": str(target.get("intended_insight") or target.get("primary_message") or "").strip(),
                "trend_hypothesis": trend_hypothesis,
                "expected_trend": expected_trend,
                "active_regime_note": str(target.get("active_regime_note") or "").strip(),
                "axis_labels": axis_labels,
                "caption": str(target.get("caption") or "").strip(),
                "caption_context": str(target.get("caption_context") or target.get("fixed_parameter_caption") or "").strip(),
                "x_field": "swept_value",
                "y_metric": metric,
                "group_field": "method",
                "facet_field": None,
                "required_metrics": sorted({metric, "objective", "feasible", *figure_scalar_metrics[:8]}),
                "required_sweep": str(chosen["id"]),
                "required_sweep_param": str(chosen["variable"]),
                "suggested_values": list(chosen["values"])[:14],
                "minimum_paper_points": max(min(max(len(chosen["values"]), 10), 14), 10),
                "minimum_paper_seeds": 50,
                "methods_to_run": methods_to_run or contract_method_ids or ["proposed"],
                "final_display_policy": "proposed_plus_one_best_practical_benchmark",
            }
        )

    raw_tables = evidence.get("tables")
    if not isinstance(raw_tables, list) or not raw_tables:
        raw_tables = plan.get("table_target")
    if isinstance(raw_tables, dict):
        raw_tables = [raw_tables]
    if not isinstance(raw_tables, list) or not raw_tables:
        raw_tables = [{}]
    table = raw_tables[0] if isinstance(raw_tables[0], dict) else {}
    evidence["contract_mode"] = "executable_single_sweep"
    evidence["compared_methods"] = contract_methods
    evidence["execution_limitations"] = [
        "Phase 2.4/2.5 runs every method declared in research_evidence_contract.compared_methods when generated_plugin.py exposes method_solution for non-default methods.",
        "Paper-mode sweeps must be scalar paths under canonical_config.",
        "Convergence or multi-dimensional Pareto claims require explicit per-iteration or cross-sweep data before plotting.",
    ]
    evidence["figures"] = normalized_figures
    evidence["tables"] = []
    evidence["tables_optional"] = True
    evidence["final_display_policy"] = "proposed_plus_one_best_practical_benchmark"
    evidence["paper_sweep_policy"] = {
        "line_or_scatter_preferred_points": 14,
        "categorical_preferred_categories": 6,
        "preferred_seeds_per_point": 100,
        "publication_figures": "2_to_3",
        "tables": "disabled",
        "noisy_numeric_sweep_display": "If a stochastic numeric sweep shows a clear average separation but local point-to-point wiggles, prefer scatter_trend or scatter over connecting noisy means with a solid line.",
    }
    evidence["required_result_columns"] = required_columns
    evidence["forbidden_defaults"] = [
        "always use line for figure_1",
        "always use grouped_bar for figure_2",
        "always use scenario-averaged table",
        "plot convergence without per-iteration rows",
        "claim a method comparison when the method is absent from compared_methods or generated_plugin.py cannot execute it",
    ]
    plan["research_evidence_contract"] = evidence
    # Backward-compatible alias for existing harness/Phase 2.5 readers.
    plan["paper_evidence_contract"] = evidence
    _phase24_normalize_semantic_forbidden_concepts(plan)
    return yaml.safe_dump(plan, sort_keys=False, allow_unicode=True)
