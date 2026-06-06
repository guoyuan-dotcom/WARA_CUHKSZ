from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path

from pipeline_core import write_text


PHASE24_SPLIT_ADAPTER_VERSION = "phase24-split-adapter-2026-05-17-v1"


def sanitize_generated_python_source(source: str) -> str:
    """Apply tiny syntax-only repairs to LLM Python output."""
    text = str(source or "")
    for _ in range(3):
        try:
            ast.parse(text)
            return text
        except SyntaxError as exc:
            if "unmatched ')'" not in str(exc.msg) or not exc.lineno:
                return text
            lines = text.splitlines()
            line_index = int(exc.lineno) - 1
            if line_index < 0 or line_index >= len(lines):
                return text
            line = lines[line_index]
            fixed: str | None = None
            for idx in reversed([pos for pos, ch in enumerate(line) if ch == ")"]):
                candidate_lines = list(lines)
                candidate_lines[line_index] = line[:idx] + line[idx + 1 :]
                candidate = "\n".join(candidate_lines) + ("\n" if text.endswith("\n") else "")
                try:
                    ast.parse(candidate)
                    fixed = candidate
                    break
                except SyntaxError:
                    continue
            if fixed is None:
                return text
            text = fixed
    return text


def _phase24_method_solution_source(source: str) -> str:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ""
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "method_solution":
            return ast.get_source_segment(source, node) or ""
    return ""


def _phase24_top_level_function_source(source: str, function_name: str) -> str:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ""
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
            return ast.get_source_segment(source, node) or ""
    return ""


def _phase24_top_level_function_names(source: str) -> set[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _phase24_called_top_level_functions(function_source: str, current_names: set[str]) -> set[str]:
    """Return top-level helpers called by a function body.

    LLM repair responses sometimes keep a harness-facing export such as
    `baseline_solution` but drop the private helper it calls. When we preserve a
    previous export, preserve its local helper dependency closure too; otherwise
    a repair can pass syntax checks and still fail at runtime with NameError.
    """

    try:
        tree = ast.parse(function_source)
    except SyntaxError:
        return set()
    called: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            name = str(node.func.id)
            if name in current_names:
                called.add(name)
    return called


def _phase24_function_accepts_parameter(source: str, function_name: str, parameter_name: str) -> bool:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name != function_name:
            continue
        if any(arg.arg == parameter_name for arg in node.args.args + node.args.kwonlyargs):
            return True
        return node.args.kwarg is not None
    return False


def _phase24_top_level_function_node(source: str, function_name: str) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
            return node
    return None


def _phase24_positional_arg_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    return [arg.arg for arg in list(node.args.posonlyargs) + list(node.args.args)]


def _phase24_replace_function_signature(
    source: str,
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    new_signature: str,
    prelude_lines: list[str] | None = None,
) -> str:
    """Replace a top-level function signature while preserving its body.

    This is intentionally narrow. It fixes interface drift such as
    `method_solution(method, problem=None, model=None, seed=0)` without touching
    the generated algorithm logic inside the body.
    """

    lines = source.splitlines()
    start_idx = int(node.lineno) - 1
    if start_idx < 0 or start_idx >= len(lines):
        return source
    if not node.body:
        return source
    end_idx = int(node.body[0].lineno) - 2
    if end_idx < start_idx:
        end_idx = start_idx
    indent = re.match(r"^(\s*)", lines[start_idx]).group(1)
    body_line = lines[int(node.body[0].lineno) - 1]
    body_indent = re.match(r"^(\s*)", body_line).group(1)
    replacement = [indent + new_signature]
    replacement.extend(body_indent + line for line in (prelude_lines or []))
    new_lines = lines[:start_idx] + replacement + lines[end_idx + 1 :]
    suffix = "\n" if source.endswith("\n") else ""
    candidate = "\n".join(new_lines) + suffix
    try:
        ast.parse(candidate)
    except SyntaxError:
        return source
    return candidate


def _phase24_add_kwargs_to_function_signature(
    source: str,
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> str:
    if node.args.kwarg is not None:
        return source
    lines = source.splitlines()
    start_idx = int(node.lineno) - 1
    if start_idx < 0 or start_idx >= len(lines) or not node.body:
        return source
    end_idx = int(node.body[0].lineno) - 2
    if end_idx < start_idx:
        end_idx = start_idx
    signature_block = "\n".join(lines[start_idx : end_idx + 1])
    if "**kwargs" in signature_block:
        return source
    if re.search(r"\(\s*\)", signature_block):
        updated_block = re.sub(r"\(\s*\)(\s*(?:->\s*[^:]+)?\s*):", r"(**kwargs)\1:", signature_block, count=1)
    else:
        updated_block = re.sub(r"\)(\s*(?:->\s*[^:]+)?\s*):", r", **kwargs)\1:", signature_block, count=1)
    if updated_block == signature_block:
        return source
    new_lines = lines[:start_idx] + updated_block.splitlines() + lines[end_idx + 1 :]
    suffix = "\n" if source.endswith("\n") else ""
    candidate = "\n".join(new_lines) + suffix
    try:
        ast.parse(candidate)
    except SyntaxError:
        return source
    return candidate


def _phase24_add_kwargs_for_internal_keyword_drift(source: str) -> str:
    """Let local helper definitions tolerate extra generated keyword options.

    A recurring Phase 2.4 repair failure is that the LLM changes one helper call
    but forgets to update the helper signature, e.g.,
    `_solve_power_sca(..., enforce_floor=True)`. Adding `**kwargs` to the local
    helper is safer than another LLM repair round and preserves the helper body.
    """

    text = source
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return text
    function_nodes = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    accepted_args = {
        name: {arg.arg for arg in node.args.args + node.args.kwonlyargs}
        for name, node in function_nodes.items()
    }
    needs_kwargs: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        callee = node.func.id
        target = function_nodes.get(callee)
        if target is None or target.args.kwarg is not None:
            continue
        for keyword in node.keywords:
            if keyword.arg and keyword.arg not in accepted_args.get(callee, set()):
                needs_kwargs.add(callee)
                break
    for name in sorted(needs_kwargs, key=lambda item: int(function_nodes[item].lineno), reverse=True):
        text = _phase24_add_kwargs_to_function_signature(text, function_nodes[name])
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return source
        function_nodes = {
            node.name: node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
    return text


def _phase24_add_missing_training_scenarios_cache(source: str) -> str:
    if "model[\"training_scenarios\"]" not in source or "_make_scenarios_for_relay" not in source:
        return source
    if re.search(r"model\s*\[\s*['\"]training_scenarios['\"]\s*\]\s*=", source):
        return source
    insertion = (
        "\n"
        "    if \"training_scenarios\" not in model:\n"
        "        model[\"training_scenarios\"] = [\n"
        "            _make_scenarios_for_relay(\n"
        "                model,\n"
        "                m,\n"
        "                int(model.get(\"training_scenarios_Ntr\", 12)),\n"
        "                int(seed) + 401 + 37 * int(m),\n"
        "                adversarial=True,\n"
        "            )\n"
        "            for m in range(int(model.get(\"relay_count_M\", 0)))\n"
        "        ]\n"
    )
    text = re.sub(r"\n(\s*)return model(\s*\n)", insertion + r"\1return model\2", source, count=1)
    try:
        ast.parse(text)
    except SyntaxError:
        return source
    return text


def _phase24_add_model_dimension_aliases(source: str) -> str:
    if "model[\"M\"]" not in source or "relay_count_M" not in source:
        return source
    if re.search(r"['\"]M['\"]\s*:", source) or re.search(r"model\s*\[\s*['\"]M['\"]\s*\]\s*=", source):
        return source
    insertion = (
        "\n"
        "    if \"relay_count_M\" in model and \"M\" not in model:\n"
        "        model[\"M\"] = int(model.get(\"relay_count_M\", 0))\n"
    )
    text = re.sub(r"\n(\s*)return model(\s*\n)", insertion + r"\1return model\2", source, count=1)
    try:
        ast.parse(text)
    except SyntaxError:
        return source
    return text


def _phase24_normalize_solver_tuple_dict_return(source: str) -> str:
    if "_solve_power_sca" not in source or "sol[\"ok\"]" not in source:
        return source
    text = re.sub(
        r"return\s+p_cur\s*,\s*\{\s*"
        r"\"used_power_sca_update\"\s*:\s*bool\(used_solver\)\s*,\s*"
        r"\"power_sca_status\"\s*:\s*status\s*,\s*"
        r"\"power_sca_objective\"\s*:\s*float\(last_value\)\s*if\s*np\.isfinite\(last_value\)\s*else\s*None\s*,?\s*"
        r"\}",
        (
            "return {\n"
            "        \"ok\": bool(used_solver and status not in {\"failed\", \"error\", \"infeasible\"}),\n"
            "        \"p\": p_cur,\n"
            "        \"status\": status,\n"
            "        \"used_power_sca_update\": bool(used_solver),\n"
            "        \"power_sca_status\": status,\n"
            "        \"power_sca_objective\": float(last_value) if np.isfinite(last_value) else None,\n"
            "    }"
        ),
        source,
        count=1,
        flags=re.DOTALL,
    )
    try:
        ast.parse(text)
    except SyntaxError:
        return source
    return text


def _phase24_add_selected_beams_to_relay_state(source: str) -> str:
    if '["beams"]' not in source or '"relays": relays' not in source or '"beams": relays[selected]["beams"]' in source:
        return source
    text = source.replace(
        '"relays": relays,\n        "selected_relay": int(selected),',
        '"relays": relays,\n        "selected_relay": int(selected),\n        "beams": relays[selected]["beams"],',
        1,
    )
    try:
        ast.parse(text)
    except SyntaxError:
        return source
    return text


def _phase24_normalize_make_state_argument_order(source: str) -> str:
    if "def _make_state(method: str, iteration: int, phi:" not in source:
        return source
    if not re.search(r"_make_state\(\s*(?:method|['\"][^'\"]+['\"])\s*,\s*(?:phi|np\.array|np\.asarray)", source):
        return source
    text = source.replace(
        "def _make_state(method: str, iteration: int, phi: np.ndarray, p: np.ndarray, W: np.ndarray, extra: Dict[str, Any] = None)",
        "def _make_state(method: str, phi: np.ndarray, p: np.ndarray, W: np.ndarray, iteration: int, extra: Dict[str, Any] = None)",
        1,
    )
    try:
        ast.parse(text)
    except SyntaxError:
        return source
    return text


def _phase24_sanitize_psd_eigendecomposition_blocks(source: str) -> str:
    """Stabilize generic SDP/SDR post-processing before randomization.

    CVX solvers can return numerically dirty PSD matrices even when a status is
    acceptable. LLM-generated SDR code often feeds those values directly into an
    eigendecomposition and Gaussian randomization, which creates NaN/Inf matrix
    products. Keep the solver route intact, but sanitize the matrix factors used
    only for recovery/randomization.
    """

    if "np.linalg.eigh" not in source and "cp.trace" not in source and "_complex_normal" not in source:
        return source
    text = source

    def _insert_hermitian_matrix_sanitizers(current: str) -> str:
        lines = current.splitlines()
        out: list[str] = []
        for idx, line in enumerate(lines):
            out.append(line)
            match = re.match(
                r"^(?P<indent>[ \t]*)(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*0\.5\s*\*\s*\(\s*(?P=name)\s*\+\s*(?P=name)\.conj\(\)\.T\s*\)\s*$",
                line,
            )
            if not match:
                continue
            name = match.group("name")
            if f"np.linalg.eigh({name})" not in current and f"cp.trace({name} @" not in current:
                continue
            next_lines = "\n".join(lines[idx + 1 : idx + 4])
            if f"{name} = np.nan_to_num({name}" in next_lines:
                continue
            indent = match.group("indent")
            out.append(f"{indent}{name} = np.nan_to_num({name}, nan=0.0, posinf=0.0, neginf=0.0)")
            out.append(
                f"{indent}{name} = np.clip(np.real({name}), -1.0e6, 1.0e6) + "
                f"1j * np.clip(np.imag({name}), -1.0e6, 1.0e6)"
            )
        suffix = "\n" if current.endswith("\n") else ""
        return "\n".join(out) + suffix

    text = re.sub(
        r"(?P<indent>^[ \t]*)(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*np\.asarray\((?P<expr>[^\\n]+?\.value)\s*,\s*dtype\s*=\s*complex\s*\)\s*$",
        (
            r"\g<indent>\g<name> = np.asarray(\g<expr>, dtype=complex)\n"
            r"\g<indent>\g<name> = np.nan_to_num(\g<name>, nan=0.0, posinf=0.0, neginf=0.0)\n"
            r"\g<indent>\g<name> = np.clip(np.real(\g<name>), -1.0e6, 1.0e6) + 1j * np.clip(np.imag(\g<name>), -1.0e6, 1.0e6)"
        ),
        text,
        flags=re.MULTILINE,
    )
    text = _insert_hermitian_matrix_sanitizers(text)
    text = re.sub(
        r"cp\.trace\(\s*(?P<matrix>[A-Za-z_][A-Za-z0-9_]*)\s*@\s*(?P<variable>[A-Za-z_][A-Za-z0-9_]*)\s*\)",
        r"cp.sum(cp.multiply(\g<matrix>.T, \g<variable>))",
        text,
    )
    text = re.sub(
        r"(?P<indent>^[ \t]*)(?P<vals>[A-Za-z_][A-Za-z0-9_]*)\s*,\s*(?P<vecs>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*np\.linalg\.eigh\((?P<matrix>[A-Za-z_][A-Za-z0-9_]*)\)\s*$",
        (
            r"\g<indent>try:\n"
            r"\g<indent>    \g<vals>, \g<vecs> = np.linalg.eigh(\g<matrix>)\n"
            r"\g<indent>except np.linalg.LinAlgError:\n"
            r"\g<indent>    \g<vals> = np.ones(\g<matrix>.shape[0], dtype=float)\n"
            r"\g<indent>    \g<vecs> = np.eye(\g<matrix>.shape[0], dtype=complex)"
        ),
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r"(?P<indent>^[ \t]*)(?P<vals>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*np\.maximum\(\s*np\.real\(\s*(?P=vals)\s*\)\s*,\s*0\.0\s*\)\s*$",
        (
            r"\g<indent>\g<vals> = np.nan_to_num(np.real(\g<vals>), nan=0.0, posinf=0.0, neginf=0.0)\n"
            r"\g<indent>\g<vals> = np.clip(\g<vals>, 0.0, 1.0e6)"
        ),
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r"(?P<indent>^[ \t]*)(?P<lhs>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?P<vecs>[A-Za-z_][A-Za-z0-9_]*)\s*@\s*np\.diag\(\s*np\.sqrt\(\s*np\.maximum\(\s*(?P<vals>[A-Za-z_][A-Za-z0-9_]*)\s*,\s*0\.0\s*\)\s*\)\s*\)\s*$",
        (
            r"\g<indent>safe_vals = np.nan_to_num(np.real(\g<vals>), nan=0.0, posinf=0.0, neginf=0.0)\n"
            r"\g<indent>safe_vals = np.clip(safe_vals, 0.0, 1.0e6)\n"
            r"\g<indent>safe_vecs = np.nan_to_num(\g<vecs>, nan=0.0, posinf=0.0, neginf=0.0)\n"
            r"\g<indent>safe_vecs = np.clip(np.real(safe_vecs), -1.0e3, 1.0e3) + 1j * np.clip(np.imag(safe_vecs), -1.0e3, 1.0e3)\n"
            r"\g<indent>\g<lhs> = safe_vecs * np.sqrt(safe_vals)[None, :]"
        ),
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r"(?P<indent>^[ \t]*)(?P<lhs>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*safe_vecs\s*@\s*np\.diag\(\s*np\.sqrt\(\s*safe_vals\s*\)\s*\)\s*$",
        r"\g<indent>\g<lhs> = safe_vecs * np.sqrt(safe_vals)[None, :]",
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r"(?P<indent>^[ \t]*)(?P<lhs>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?P<matrix>[A-Za-z_][A-Za-z0-9_]*)\s*@\s*(?P<sample_func>_complex_normal\([^\n]+\))\s*$",
        (
            r"\g<indent>random_vec = \g<sample_func>\n"
            r"\g<indent>random_vec = np.nan_to_num(random_vec, nan=0.0, posinf=0.0, neginf=0.0)\n"
            r"\g<indent>random_vec = np.clip(np.real(random_vec), -1.0e3, 1.0e3) + 1j * np.clip(np.imag(random_vec), -1.0e3, 1.0e3)\n"
            r"\g<indent>\g<lhs> = np.sum(\g<matrix> * random_vec[None, :], axis=1)\n"
            r"\g<indent>\g<lhs> = np.nan_to_num(\g<lhs>, nan=0.0, posinf=0.0, neginf=0.0)"
        ),
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r"(?P<line>^(?P<indent>[ \t]*)safe_vecs\s*=\s*np\.nan_to_num\([^\n]+\)\s*$\n)(?!(?P=indent)safe_vecs\s*=\s*np\.clip)",
        (
            r"\g<line>"
            r"\g<indent>safe_vecs = np.clip(np.real(safe_vecs), -1.0e3, 1.0e3) + "
            r"1j * np.clip(np.imag(safe_vecs), -1.0e3, 1.0e3)\n"
        ),
        text,
        flags=re.MULTILINE,
    )
    try:
        ast.parse(text)
    except SyntaxError:
        return source
    return text


def _phase24_normalize_nan_to_num_complex_fill_values(source: str) -> str:
    """Avoid NumPy dtype crashes from complex fill scalars on real arrays.

    Generated wireless code often mixes real layout variables with complex
    channels/beamformers. NumPy accepts real fill values for complex arrays, but
    rejects complex fill values for real arrays. Replacing literal `a + 0.0j`
    keyword fills by their real part is therefore the safe, topic-agnostic
    choice for `np.nan_to_num(...)`.
    """

    if "nan_to_num" not in source or "j" not in source:
        return source

    def _fix_call(match: re.Match[str]) -> str:
        call = match.group(0)
        call = re.sub(
            r"\b(nan|posinf|neginf)\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*\+\s*0(?:\.0)?j",
            r"\1=\2",
            call,
        )
        call = re.sub(
            r"\b(nan|posinf|neginf)\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*-\s*0(?:\.0)?j",
            r"\1=\2",
            call,
        )
        call = re.sub(
            r"\b(nan|posinf|neginf)\s*=\s*complex\(\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*,\s*0(?:\.0)?\s*\)",
            r"\1=\2",
            call,
        )
        return call

    text = re.sub(r"np\.nan_to_num\([^\n]*\)", _fix_call, source)
    try:
        ast.parse(text)
    except SyntaxError:
        return source
    return text


def _phase24_normalize_harness_export_signatures(source: str) -> str:
    """Normalize recurring LLM signature drift for fixed Phase 2.4 exports."""

    text = source
    proposed_node = _phase24_top_level_function_node(text, "proposed_step")
    if proposed_node is not None:
        args = _phase24_positional_arg_names(proposed_node)
        if args[:3] == ["problem", "model", "state"] and len(args) == 3 and proposed_node.args.vararg is None:
            text = _phase24_replace_function_signature(
                text,
                proposed_node,
                "def proposed_step(problem, model, state, iteration=0):",
            )
        elif args[:3] == ["problem", "model", "state"] and (len(args) < 4 or args[3] != "iteration"):
            text = _phase24_replace_function_signature(
                text,
                proposed_node,
                "def proposed_step(problem, model, state, iteration=0, **kwargs):",
                ["seed = kwargs.get(\"seed\", iteration)"],
            )
        elif len(args) >= 2 and args[0] == "state" and args[1] == "model":
            text = _phase24_replace_function_signature(
                text,
                proposed_node,
                "def proposed_step(problem, model, state, iteration=0, **kwargs):",
                ["seed = kwargs.get(\"seed\", iteration)"],
            )

    evaluate_node = _phase24_top_level_function_node(text, "evaluate_state")
    if evaluate_node is not None:
        args = _phase24_positional_arg_names(evaluate_node)
        if args[:3] != ["problem", "model", "state"] and len(args) >= 2 and args[0] == "state" and args[1] == "model":
            text = _phase24_replace_function_signature(
                text,
                evaluate_node,
                "def evaluate_state(problem, model, state, **kwargs):",
                [
                    "seed = kwargs.get(",
                    "    \"seed\",",
                    "    state.get(\"seed\", getattr(problem, \"realization_id\", 0)) if isinstance(state, dict) else getattr(problem, \"realization_id\", 0),",
                    ")",
                ],
            )

    method_node = _phase24_top_level_function_node(text, "method_solution")
    if method_node is not None:
        args = _phase24_positional_arg_names(method_node)
        method_aliases = {"method", "method_id", "method_name"}
        if len(args) >= 3 and args[0] in method_aliases and "problem" in args[1:3] and "model" in args[1:3]:
            method_arg = args[0]
            text = _phase24_replace_function_signature(
                text,
                method_node,
                f"def method_solution(problem, model, {method_arg}, seed=0, **kwargs):",
                ["kwargs.setdefault(\"problem\", problem)"],
            )
        elif len(args) >= 2 and args[0] in method_aliases and args[1] == "model":
            method_arg = args[0]
            text = _phase24_replace_function_signature(
                text,
                method_node,
                f"def method_solution(problem, model, {method_arg}, seed=0, **kwargs):",
                ["kwargs.setdefault(\"problem\", problem)"],
            )
    return text


def preserve_phase24_required_exports(current_source: str, repaired_source: str) -> str:
    """Keep fixed harness exports when a repair response accidentally drops them."""

    required = [
        "build_model",
        "initial_state",
        "proposed_step",
        "baseline_solution",
        "evaluate_state",
        "method_solution",
    ]
    current_names = _phase24_top_level_function_names(current_source)
    repaired_names = _phase24_top_level_function_names(repaired_source)
    missing_sources: list[str] = []
    missing_names: list[str] = []
    seen: set[str] = set()

    def enqueue(function_name: str) -> None:
        if function_name in seen or function_name in repaired_names or function_name not in current_names:
            return
        seen.add(function_name)
        missing_names.append(function_name)

    for function_name in required:
        enqueue(function_name)
    for function_name in missing_names:
        current_function = _phase24_top_level_function_source(current_source, function_name)
        if not current_function:
            continue
        missing_sources.append(current_function)
        for helper_name in sorted(_phase24_called_top_level_functions(current_function, current_names)):
            enqueue(helper_name)
    if not missing_sources:
        return repaired_source
    merged = repaired_source.rstrip() + "\n\n\n# Preserved fixed harness exports from the previous passing implementation.\n" + "\n\n\n".join(missing_sources) + "\n"
    try:
        ast.parse(merged)
    except SyntaxError:
        return repaired_source
    return merged


def _phase24_method_branch_map(method_source: str) -> dict[str, str]:
    lines = method_source.splitlines()
    starts: list[tuple[int, str, str]] = []
    branch_re = re.compile(r"^(?P<indent>\s*)(?:if|elif)\s+method\s*==\s*['\"](?P<method>[^'\"]+)['\"]\s*:")
    for idx, line in enumerate(lines):
        match = branch_re.match(line)
        if match:
            starts.append((idx, match.group("method"), match.group("indent")))
    branches: dict[str, str] = {}
    for pos, (start_idx, method_id, indent) in enumerate(starts):
        end_idx = len(lines)
        for next_idx in range(start_idx + 1, len(lines)):
            line = lines[next_idx]
            if re.match(rf"^{re.escape(indent)}(?:elif\s+method\s*==\s*['\"]|else\s*:)", line):
                end_idx = next_idx
                break
        if pos + 1 < len(starts):
            end_idx = min(end_idx, starts[pos + 1][0])
        block = "\n".join(lines[start_idx:end_idx]).rstrip()
        if block:
            branches[method_id] = block
    return branches


def merge_phase24_method_solution_branches(current_source: str, repaired_source: str) -> str:
    """Preserve working method_solution branches when a repair response drops them."""
    current_method = _phase24_method_solution_source(current_source)
    repaired_method = _phase24_method_solution_source(repaired_source)
    if not current_method or not repaired_method:
        return repaired_source
    current_branches = _phase24_method_branch_map(current_method)
    repaired_branches = _phase24_method_branch_map(repaired_method)
    missing = [
        method_id
        for method_id in current_branches
        if method_id not in repaired_branches and method_id not in {"proposed", "baseline"}
    ]
    if not missing:
        return repaired_source

    repaired_lines = repaired_method.splitlines()
    branch_indent = "    "
    for line in repaired_lines:
        match = re.match(r"^(\s*)(?:if|elif)\s+method\s*==", line)
        if match:
            branch_indent = match.group(1)
            break
    insert_at = len(repaired_lines)
    for idx, line in enumerate(repaired_lines):
        if re.match(rf"^{re.escape(branch_indent)}else\s*:", line):
            insert_at = idx
            break
    insert_blocks: list[str] = []
    for method_id in missing:
        block = current_branches[method_id]
        block_lines = block.splitlines()
        if block_lines:
            block_lines[0] = re.sub(r"^(\s*)if\s+", r"\1elif ", block_lines[0], count=1)
        insert_blocks.extend(block_lines)
        insert_blocks.append("")
    merged_lines = repaired_lines[:insert_at] + insert_blocks + repaired_lines[insert_at:]
    merged_method = "\n".join(merged_lines).rstrip() + "\n"
    merged_source = repaired_source.replace(repaired_method, merged_method, 1)
    try:
        ast.parse(merged_source)
    except SyntaxError:
        return repaired_source
    return merged_source


def normalize_phase24_generated_plugin_source(source: str) -> str:
    """Apply deterministic, topic-agnostic fixes for recurring Phase 2.4 math bugs."""
    text = sanitize_generated_python_source(source)
    text = _phase24_normalize_nan_to_num_complex_fill_values(text)
    text = _phase24_normalize_harness_export_signatures(text)
    text = _phase24_add_kwargs_for_internal_keyword_drift(text)
    text = _phase24_normalize_solver_tuple_dict_return(text)
    text = _phase24_add_model_dimension_aliases(text)
    text = _phase24_add_missing_training_scenarios_cache(text)
    text = _phase24_add_selected_beams_to_relay_state(text)
    text = _phase24_normalize_make_state_argument_order(text)
    text = _phase24_sanitize_psd_eigendecomposition_blocks(text)
    text = re.sub(
        r"_herm\(\s*A_radar\s*\)\.T\s*@\s*A_radar\b",
        "_herm(A_radar) @ A_radar",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"(?P<lhs>\bM_eh\s*=\s*)max\(\s*1\s*,\s*int\(\s*np\.floor\(\s*rho\s*\*\s*M\s*\)\s*\)\s*\)",
        r"\g<lhs>(0 if rho <= 0 else max(1, int(np.floor(rho * M))))",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"(?P<lhs>\bM_eh\s*=\s*)max\(\s*1\s*,\s*int\(\s*floor\(\s*rho\s*\*\s*M\s*\)\s*\)\s*\)",
        r"\g<lhs>(0 if rho <= 0 else max(1, int(floor(rho * M))))",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"(?P<lhs>\bM_eh\s*=\s*)max\(\s*1\s*,\s*int\(\s*rho\s*\*\s*M\s*\)\s*\)",
        r"\g<lhs>(0 if rho <= 0 else max(1, int(rho * M)))",
        text,
        flags=re.IGNORECASE,
    )
    if "_run_fixed_trajectory" in text and "_fixed_traj_method(" in text:
        text = re.sub(
            r"_fixed_traj_method\(\s*problem\s*,\s*model\s*,\s*seed\s*\)",
            r"_run_fixed_trajectory(problem, model, initial_state(problem, model, seed=seed))",
            text,
        )
    if "_run_nearest_greedy" in text and "_nearest_user_method(" in text:
        text = re.sub(
            r"_nearest_user_method\(\s*problem\s*,\s*model\s*,\s*seed\s*\)",
            r"_run_nearest_greedy(problem, model, initial_state(problem, model, seed=seed))",
            text,
        )
    if "_run_proposed" in text and "_no_propulsion_method(" in text:
        text = re.sub(
            r"_no_propulsion_method\(\s*problem\s*,\s*model\s*,\s*seed\s*\)",
            r"_run_proposed(problem, model, initial_state(problem, model, seed=seed), ignore_propulsion=True)",
            text,
        )
    if "_run_proposed" in text and "_proposed_method(" in text:
        text = re.sub(
            r"_proposed_method\(\s*problem\s*,\s*model\s*,\s*seed\s*\)",
            r"_run_proposed(problem, model, initial_state(problem, model, seed=seed), ignore_propulsion=False)",
            text,
        )
    proposed_step_source = _phase24_top_level_function_source(text, "proposed_step")
    if (
        "_run_proposed" in text
        and proposed_step_source
        and re.search(r"_block_[A-Za-z0-9_]+", proposed_step_source)
    ):
        replacement = (
            "def proposed_step(problem: Any, model: Dict[str, Any], state: Dict[str, Any], iteration: int) -> Dict[str, Any]:\n"
            "    \"\"\"Harness-facing proposed update; delegates to the generated proposed routine.\"\"\"\n"
            "    return _run_proposed(problem, model, state, ignore_propulsion=False)\n"
        )
        text = text.replace(proposed_step_source, replacement, 1)
    if "_run_proposed" in text and not _phase24_function_accepts_parameter(text, "_run_proposed", "ignore_propulsion"):
        text = re.sub(
            r"_run_proposed\(\s*problem\s*,\s*model\s*,\s*state\s*,\s*ignore_propulsion\s*=\s*False\s*\)",
            "_run_proposed(problem, model, state)",
            text,
        )
        text = re.sub(
            r"_run_proposed\(\s*problem\s*,\s*model\s*,\s*state\s*,\s*ignore_propulsion\s*=\s*True\s*\)",
            "_run_proposed(problem, model, state)",
            text,
        )
        text = re.sub(
            r"_run_proposed\((?P<args>.*?)\s*,\s*ignore_propulsion\s*=\s*(?:False|True)\s*\)",
            r"_run_proposed(\g<args>)",
            text,
            flags=re.DOTALL,
        )
    defined_functions = _phase24_top_level_function_names(text)
    compatibility_wrappers: list[str] = []
    if "def solve_method" in text or "def solve(" in text:
        if "_run_fixed_trajectory" in text and "_run_fixed_trajectory" not in defined_functions:
            compatibility_wrappers.append(
                "def _run_fixed_trajectory(problem, model, state):\n"
                "    \"\"\"Compatibility wrapper for LLM-generated method_solution branches.\"\"\"\n"
                "    try:\n"
                "        return solve_method(problem, model=model, method=\"fixed_trajectory_baseline\", seed=0, initial=state)\n"
                "    except TypeError:\n"
                "        return solve_method(problem, model=model, method=\"fixed_trajectory_baseline\", seed=0)\n"
            )
        if "_run_nearest_greedy" in text and "_run_nearest_greedy" not in defined_functions:
            compatibility_wrappers.append(
                "def _run_nearest_greedy(problem, model, state):\n"
                "    \"\"\"Compatibility wrapper for LLM-generated method_solution branches.\"\"\"\n"
                "    try:\n"
                "        return solve_method(problem, model=model, method=\"nearest_user_greedy\", seed=0, initial=state)\n"
                "    except TypeError:\n"
                "        return solve_method(problem, model=model, method=\"nearest_user_greedy\", seed=0)\n"
            )
        if "_run_proposed" in text and "_run_proposed" not in defined_functions:
            compatibility_wrappers.append(
                "def _run_proposed(problem, model, state, ignore_propulsion=False):\n"
                "    \"\"\"Compatibility wrapper for LLM-generated proposed/no-propulsion branches.\"\"\"\n"
                "    method = \"no_propulsion_awareness\" if ignore_propulsion else \"proposed\"\n"
                "    try:\n"
                "        return solve_method(problem, model=model, method=method, seed=0, initial=state)\n"
                "    except TypeError:\n"
                "        return solve_method(problem, model=model, method=method, seed=0)\n"
            )
    if compatibility_wrappers:
        dispatcher = (
            "def _phase24_dispatch_compat(problem, model, method, state):\n"
            "    \"\"\"Route drifting internal helper names through the generated solver entrypoint.\"\"\"\n"
            "    target = globals().get(\"solve_method\") or globals().get(\"solve\")\n"
            "    if target is None:\n"
            "        raise NameError(\"generated core has no solve_method or solve dispatcher\")\n"
            "    try:\n"
            "        return target(problem, model=model, method=method, seed=0, initial=state)\n"
            "    except TypeError:\n"
            "        try:\n"
            "            return target(problem, model=model, method=method, seed=0)\n"
            "        except TypeError:\n"
            "            return target(problem, method=method, seed=0)\n"
        )
        normalized_wrappers = [
            re.sub(
                r"try:\n\s+return solve_method\((?:.|\n)*?except TypeError:\n\s+return solve_method\((?:.|\n)*?\n",
                "",
                wrapper,
            )
            for wrapper in compatibility_wrappers
        ]
        for idx, wrapper in enumerate(normalized_wrappers):
            if wrapper.startswith("def _run_fixed_trajectory"):
                normalized_wrappers[idx] = (
                    "def _run_fixed_trajectory(problem, model, state):\n"
                    "    \"\"\"Compatibility wrapper for LLM-generated method_solution branches.\"\"\"\n"
                    "    return _phase24_dispatch_compat(problem, model, \"fixed_trajectory_baseline\", state)\n"
                )
            elif wrapper.startswith("def _run_nearest_greedy"):
                normalized_wrappers[idx] = (
                    "def _run_nearest_greedy(problem, model, state):\n"
                    "    \"\"\"Compatibility wrapper for LLM-generated method_solution branches.\"\"\"\n"
                    "    return _phase24_dispatch_compat(problem, model, \"nearest_user_greedy\", state)\n"
                )
            elif wrapper.startswith("def _run_proposed"):
                normalized_wrappers[idx] = (
                    "def _run_proposed(problem, model, state, ignore_propulsion=False):\n"
                    "    \"\"\"Compatibility wrapper for LLM-generated proposed/no-propulsion branches.\"\"\"\n"
                    "    method = \"no_propulsion_awareness\" if ignore_propulsion else \"proposed\"\n"
                    "    return _phase24_dispatch_compat(problem, model, method, state)\n"
                )
        merged = (
            text.rstrip()
            + "\n\n\n# Deterministic compatibility wrappers for non-contracted LLM helper names.\n"
            + dispatcher
            + "\n\n"
            + "\n\n".join(normalized_wrappers)
            + "\n"
        )
        try:
            ast.parse(merged)
            text = merged
        except SyntaxError:
            pass
    return text


def build_phase24_split_plugin_adapter() -> str:
    """Return the deterministic adapter used by the fixed Phase 2.4 harness."""
    return '''
"""Thin Phase 2.4 adapter.

The generated experiment implementation lives in generated_experiment_core.py.
Keeping this adapter small lets the harness exports remain stable while the
algorithm, method, and metric logic can be audited as a separate generated file.
"""

import inspect
import math
import os

import numpy as np

import generated_experiment_core as _core


PHASE24_SPLIT_ADAPTER_VERSION = "__PHASE24_SPLIT_ADAPTER_VERSION__"


def _call_core(func, *args, **kwargs):
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return func(*args, **kwargs)
    parameters = signature.parameters
    accepts_var_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values())
    if accepts_var_kwargs:
        return func(*args, **kwargs)
    positional_names = [
        name
        for name, param in parameters.items()
        if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ][: len(args)]
    filtered = {key: value for key, value in kwargs.items() if key in parameters and key not in positional_names}
    return func(*args, **filtered)


def _as_float_or_none(value):
    try:
        if isinstance(value, complex):
            value = abs(value) if abs(value.imag) > 1e-8 else value.real
        number = float(value)
    except Exception:
        return None
    if number != number or math.isinf(number):
        return None
    return number


def _finite_number(value, default=0.0):
    number = _as_float_or_none(value)
    if number is None:
        return float(default)
    return float(number)


def _sanitize_array(value):
    arr = np.asarray(value)
    if np.iscomplexobj(arr):
        real = np.nan_to_num(np.real(arr), nan=0.0, posinf=0.0, neginf=0.0)
        imag = np.nan_to_num(np.imag(arr), nan=0.0, posinf=0.0, neginf=0.0)
        real = np.clip(real, -1.0e12, 1.0e12)
        imag = np.clip(imag, -1.0e12, 1.0e12)
        return real + 1j * imag
    clean = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        return np.clip(clean, -1.0e12, 1.0e12)
    except Exception:
        return clean


def _sanitize_value(value):
    if isinstance(value, dict):
        return {key: _sanitize_value(val) for key, val in value.items()}
    if isinstance(value, tuple):
        return tuple(_sanitize_value(item) for item in value)
    if isinstance(value, list):
        return [_sanitize_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _sanitize_array(value)
    if isinstance(value, np.generic):
        return _sanitize_value(value.item())
    if isinstance(value, complex):
        return complex(_finite_number(value.real), _finite_number(value.imag))
    if isinstance(value, float):
        return _finite_number(value)
    return value


def _sanitize_state(state, *, fallback_method="unknown"):
    state = _sanitize_value(state)
    if not isinstance(state, dict):
        state = {"raw_state": state}
    state.setdefault("method", fallback_method)
    return state


def _runtime_error_state(method, exc, previous_state=None):
    state = _sanitize_state(previous_state or {}, fallback_method=str(method or "runtime_error"))
    state["method"] = str(method or state.get("method") or "runtime_error")
    state["feasible"] = False
    state["constraint_violation"] = max(float(state.get("constraint_violation") or 0.0), 1.0e6)
    state["phase24_runtime_error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    return state


def _quick_cap_int(env_name, default):
    try:
        return max(1, int(os.environ.get(env_name, str(default))))
    except Exception:
        return int(default)


def _is_quick_validation_mode():
    tier = str(os.environ.get("WARA_PHASE25_SWEEP_TIER") or os.environ.get("WCL_PHASE25_SWEEP_TIER") or "").strip().lower()
    if tier in {"scout", "medium", "paper"}:
        return False
    mode = str(os.environ.get("WARA_RUN_MODE") or "").strip().lower()
    if mode in {"scout_validation", "medium_validation", "paper_validation"}:
        return False
    return True


def _phase25_sweep_iteration_cap():
    tier = str(os.environ.get("WARA_PHASE25_SWEEP_TIER") or os.environ.get("WCL_PHASE25_SWEEP_TIER") or "").strip().lower()
    if tier not in {"scout", "medium", "paper"}:
        return None
    raw_value = os.environ.get("WARA_PHASE25_MAX_ITERATIONS") or os.environ.get("WCL_PHASE25_MAX_ITERATIONS")
    if raw_value is None:
        return None
    try:
        return max(1, int(raw_value))
    except Exception:
        return None


def _positive_int(value, default):
    try:
        return max(1, int(value))
    except Exception:
        return int(default)


def _normalize_metrics(metrics):
    metrics = dict(metrics) if isinstance(metrics, dict) else {}
    sanitized_nonfinite = False
    normalized_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, (dict, list, tuple, np.ndarray, np.generic, complex, float, int)):
            cleaned = _sanitize_value(value)
            if isinstance(cleaned, complex):
                cleaned = abs(cleaned) if abs(cleaned.imag) > 1e-8 else cleaned.real
            if isinstance(cleaned, np.ndarray):
                cleaned = cleaned.tolist()
            if isinstance(cleaned, (float, int)):
                original_number = _as_float_or_none(value)
                if original_number is None:
                    sanitized_nonfinite = True
                cleaned = float(cleaned)
            normalized_metrics[key] = cleaned
        else:
            normalized_metrics[key] = value
    metrics = normalized_metrics
    if "constraint_violation_max" not in metrics and "max_constraint_violation" in metrics:
        metrics["constraint_violation_max"] = metrics["max_constraint_violation"]
    if "constraint_violation" not in metrics:
        if "constraint_violation_max" in metrics:
            metrics["constraint_violation"] = metrics["constraint_violation_max"]
        else:
            violation_candidates = []
            for key, value in list(metrics.items()):
                lowered = str(key).lower()
                numeric = _as_float_or_none(value)
                if numeric is None:
                    continue
                if "violation" in lowered or "residual" in lowered:
                    violation_candidates.append(abs(numeric))
                elif "margin" in lowered:
                    violation_candidates.append(max(0.0, -numeric))
            if violation_candidates:
                metrics["constraint_violation"] = max(violation_candidates)
            else:
                metrics["constraint_violation"] = 0.0 if bool(metrics.get("feasible", False)) else 1.0
    for key in list(metrics.keys()):
        lowered = str(key).lower()
        if any(token in lowered for token in ("objective", "rate", "throughput", "energy", "power", "utility", "sinr", "fairness", "efficiency", "violation", "margin")):
            if isinstance(metrics.get(key), (int, float, complex, np.generic)):
                original = _as_float_or_none(metrics.get(key))
                if original is None:
                    sanitized_nonfinite = True
                    metrics[key] = 0.0
                else:
                    metrics[key] = float(original)
    if sanitized_nonfinite:
        metrics["sanitized_nonfinite"] = True
        metrics["feasible"] = False
        metrics["constraint_violation"] = max(float(metrics.get("constraint_violation") or 0.0), 1.0e6)
    if "constraint_violation_max" not in metrics:
        metrics["constraint_violation_max"] = metrics["constraint_violation"]
    if "max_constraint_violation" not in metrics:
        metrics["max_constraint_violation"] = metrics["constraint_violation_max"]
    return metrics


def _normalize_model(raw_model):
    model = dict(raw_model) if isinstance(raw_model, dict) else {}
    metadata = model.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    algorithm = model.get("algorithm")
    if isinstance(algorithm, dict):
        algorithm = dict(algorithm)
        model["algorithm"] = algorithm
    phase25_iter_cap = _phase25_sweep_iteration_cap()
    if phase25_iter_cap is not None:
        metadata["max_iterations"] = min(
            _positive_int(metadata.get("max_iterations", phase25_iter_cap), phase25_iter_cap),
            phase25_iter_cap,
        )
        if isinstance(algorithm, dict):
            algorithm["max_iterations"] = min(
                _positive_int(algorithm.get("max_iterations", phase25_iter_cap), phase25_iter_cap),
                phase25_iter_cap,
            )
    elif _is_quick_validation_mode():
        quick_iter_cap = _quick_cap_int("WARA_PHASE24_QUICK_MAX_ITERATIONS", 2)
        metadata["max_iterations"] = min(_positive_int(metadata.get("max_iterations", quick_iter_cap), quick_iter_cap), quick_iter_cap)
        if isinstance(algorithm, dict):
            algorithm["max_iterations"] = min(_positive_int(algorithm.get("max_iterations", quick_iter_cap), quick_iter_cap), quick_iter_cap)
    else:
        if "max_iterations" in metadata:
            metadata["max_iterations"] = _positive_int(metadata.get("max_iterations"), 8)
        elif isinstance(algorithm, dict) and "max_iterations" in algorithm:
            metadata["max_iterations"] = _positive_int(algorithm.get("max_iterations"), 8)
        if isinstance(algorithm, dict) and "max_iterations" in algorithm:
            algorithm["max_iterations"] = _positive_int(algorithm.get("max_iterations"), metadata.get("max_iterations", 8))
    # The LLM sometimes returns operator callables with drifting signatures.
    # Keep the public harness contract deterministic and route through the
    # adapter, which already delegates to top-level generated-core functions.
    operators = {
        "channel_from_state": channel_from_state,
        "project_state": project_state,
        "evaluate_state": evaluate_state,
    }
    state_init = model.get("state_init")
    if not isinstance(state_init, dict):
        state_init = {}
    model["state_init"] = state_init
    model["operators"] = operators
    model["metadata"] = metadata
    return model


def build_model(problem, seed=0):
    return _normalize_model(_core.build_model(problem, seed=seed))


def initial_state(problem, model, seed=0):
    try:
        return _sanitize_state(_core.initial_state(problem, model, seed=seed), fallback_method="initial")
    except Exception as exc:
        return _runtime_error_state("initial", exc, {"seed": seed})


def proposed_step(problem, model, state, iteration):
    clean_state = _sanitize_state(state, fallback_method="proposed")
    try:
        return _sanitize_state(_core.proposed_step(problem, model, clean_state, iteration), fallback_method="proposed")
    except Exception as exc:
        failed = _runtime_error_state("proposed", exc, clean_state)
        failed["failed_iteration"] = int(iteration) if isinstance(iteration, int) else iteration
        return failed


def baseline_solution(problem, model, seed=0):
    try:
        return _sanitize_state(
            _call_core(
                _core.baseline_solution,
                problem,
                model,
                seed=seed,
                method="baseline",
                method_id="baseline",
                method_name="baseline",
            ),
            fallback_method="baseline",
        )
    except Exception as exc:
        return _runtime_error_state("baseline", exc, {"seed": seed})


def evaluate_state(problem, model, state):
    clean_state = _sanitize_state(state)
    try:
        metrics = _core.evaluate_state(problem, model, clean_state)
    except Exception as exc:
        metrics = {
            "objective": 0.0,
            "feasible": False,
            "constraint_violation": 1.0e6,
            "phase24_evaluate_error": f"{type(exc).__name__}: {str(exc)[:300]}",
        }
    metrics = _normalize_metrics(metrics)
    if clean_state.get("phase24_runtime_error"):
        metrics["feasible"] = False
        metrics["constraint_violation"] = max(float(metrics.get("constraint_violation") or 0.0), 1.0e6)
        metrics["phase24_runtime_error"] = clean_state.get("phase24_runtime_error")
    return metrics


def channel_from_state(problem, state):
    if hasattr(_core, "channel_from_state"):
        try:
            return _sanitize_value(_core.channel_from_state(problem, _sanitize_state(state)))
        except Exception:
            return {}
    cached = getattr(problem, "_model_cache", None)
    if isinstance(cached, dict):
        operator = (cached.get("operators") or {}).get("channel_from_state")
        if callable(operator) and operator is not channel_from_state:
            return operator(problem, state)
    return {}


def project_state(problem, state):
    if hasattr(_core, "project_state"):
        try:
            return _sanitize_state(_core.project_state(problem, _sanitize_state(state)))
        except Exception:
            return _sanitize_state(state)
    cached = getattr(problem, "_model_cache", None)
    if isinstance(cached, dict):
        operator = (cached.get("operators") or {}).get("project_state")
        if callable(operator) and operator is not project_state:
            return operator(problem, state)
    return dict(state)


def method_solution(problem, model, method, seed=0):
    if hasattr(_core, "method_solution"):
        try:
            return _sanitize_state(
                _call_core(
                    _core.method_solution,
                    problem,
                    model,
                    method,
                    seed=seed,
                    method_id=method,
                    method_name=method,
                ),
                fallback_method=str(method or "method"),
            )
        except Exception as exc:
            return _runtime_error_state(str(method or "method"), exc, {"seed": seed})
    method_key = str(method or "").strip().lower()
    if method_key == "proposed":
        state = initial_state(problem, model, seed=seed)
        for iteration in range(int((model.get("metadata") or {}).get("max_iterations", 1))):
            state = proposed_step(problem, model, state, iteration)
        state["method"] = "proposed"
        return _sanitize_state(state, fallback_method="proposed")
    if method_key == "baseline":
        return baseline_solution(problem, model, seed=seed)
    raise ValueError(f"generated_experiment_core.py does not implement method_solution for method={method!r}")
'''.strip().replace("__PHASE24_SPLIT_ADAPTER_VERSION__", PHASE24_SPLIT_ADAPTER_VERSION) + "\n"


def write_phase24_split_plugin_package(phase_dir: Path, solver_dir: Path, core_source: str) -> str:
    """Persist split Phase 2.4 code and return the generated_plugin.py adapter source."""
    solver_dir = Path(solver_dir)
    phase_dir = Path(phase_dir)
    core_code = normalize_phase24_generated_plugin_source(core_source)
    core_hash = hashlib.sha256(core_code.encode("utf-8")).hexdigest()
    write_text(solver_dir / "generated_experiment_core.py", core_code)
    adapter = build_phase24_split_plugin_adapter()
    write_text(solver_dir / "generated_plugin.py", adapter)
    manifest = {
        "mode": "split_generated_core_with_deterministic_adapter",
        "codegen_version": PHASE24_SPLIT_ADAPTER_VERSION,
        "generated_experiment_core_sha256": core_hash,
        "llm_authored_files": ["generated_experiment_core.py"],
        "deterministic_adapter": "generated_plugin.py",
        "reason": (
            "Phase 2.4 separates algorithm/metric implementation from harness exports so experiment code "
            "can be audited and repaired without regenerating runner, schema, or serialization logic."
        ),
    }
    write_text(phase_dir / "phase24_split_code_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    return adapter
