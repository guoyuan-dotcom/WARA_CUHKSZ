"""Minimal YAML shim for constrained local runtimes.

This is intentionally small and supports the subset of YAML used by the
wireless Phase 11/12 pipeline artifacts and configs:

- mappings and sequences by indentation
- quoted/plain scalars
- block scalars via ``|`` and ``>``
- ``safe_load`` / ``safe_dump`` / ``dump`` / ``YAMLError``

It is not a full YAML 1.2 implementation and should only be activated via
``PYTHONPATH`` in the restricted fallback runtime.
"""

from __future__ import annotations

import io
import json
import re
from typing import Any

__all__ = ["YAMLError", "safe_load", "safe_dump", "dump"]


class YAMLError(ValueError):
    """Raised when the minimal parser cannot understand the input."""


def safe_load(stream: Any) -> Any:
    text = _coerce_text(stream)
    parser = _Parser(text)
    return parser.parse()


def safe_dump(
    data: Any,
    stream: Any | None = None,
    default_flow_style: bool | None = None,
    allow_unicode: bool = True,
    sort_keys: bool = False,
    width: int | None = None,
) -> str | None:
    del default_flow_style, width
    text = _dump_node(data, 0, allow_unicode=allow_unicode, sort_keys=sort_keys)
    if not text.endswith("\n"):
        text += "\n"
    if stream is None:
        return text
    if hasattr(stream, "write"):
        stream.write(text)
        return None
    raise TypeError("stream must be file-like when provided")


def dump(*args: Any, **kwargs: Any) -> str | None:
    return safe_dump(*args, **kwargs)


def _coerce_text(stream: Any) -> str:
    if hasattr(stream, "read"):
        payload = stream.read()
        if isinstance(payload, bytes):
            return payload.decode("utf-8")
        return str(payload)
    if isinstance(stream, bytes):
        return stream.decode("utf-8")
    return str(stream)


class _Parser:
    def __init__(self, text: str) -> None:
        self.lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    def parse(self) -> Any:
        idx = self._skip_ignored(0)
        if idx >= len(self.lines):
            return None
        node, idx = self._parse_block(idx, self._indent_of(idx))
        idx = self._skip_ignored(idx)
        if idx < len(self.lines):
            raise YAMLError(f"Unexpected trailing content at line {idx + 1}")
        return node

    def _parse_block(self, idx: int, indent: int) -> tuple[Any, int]:
        idx = self._skip_ignored(idx)
        if idx >= len(self.lines):
            return {}, idx
        stripped = self.lines[idx].lstrip(" ")
        if stripped.startswith("- "):
            return self._parse_sequence(idx, indent)
        return self._parse_mapping(idx, indent)

    def _parse_sequence(self, idx: int, indent: int) -> tuple[list[Any], int]:
        items: list[Any] = []
        while idx < len(self.lines):
            idx = self._skip_ignored(idx)
            if idx >= len(self.lines):
                break
            line = self.lines[idx]
            cur_indent = self._indent_of(idx)
            if cur_indent < indent:
                break
            if cur_indent != indent:
                raise YAMLError(f"Bad sequence indentation at line {idx + 1}")
            stripped = line[cur_indent:]
            if not stripped.startswith("- "):
                break
            rest = stripped[2:]
            idx += 1
            if not rest.strip():
                item, idx = self._parse_block(idx, indent + 2)
                items.append(item)
                continue
            if self._looks_like_mapping_entry(rest):
                item, idx = self._parse_inline_mapping(rest, idx, indent + 2)
                items.append(item)
                continue
            value, idx = self._parse_value(rest, idx, indent)
            items.append(value)
        return items, idx

    def _parse_mapping(self, idx: int, indent: int) -> tuple[dict[str, Any], int]:
        result: dict[str, Any] = {}
        while idx < len(self.lines):
            idx = self._skip_ignored(idx)
            if idx >= len(self.lines):
                break
            line = self.lines[idx]
            cur_indent = self._indent_of(idx)
            if cur_indent < indent:
                break
            if cur_indent != indent:
                raise YAMLError(f"Bad mapping indentation at line {idx + 1}")
            stripped = line[cur_indent:]
            if stripped.startswith("- "):
                break
            key, rest = self._split_key_value(stripped, idx)
            idx += 1
            if not rest.strip():
                next_idx = self._skip_ignored(idx)
                if next_idx >= len(self.lines):
                    result[key] = None
                    idx = next_idx
                    continue
                next_indent = self._indent_of(next_idx)
                next_stripped = self.lines[next_idx].lstrip(" ")
                if next_indent <= indent and not next_stripped.startswith("- "):
                    result[key] = None
                    idx = next_idx
                    continue
                block_indent = next_indent if next_indent > indent else indent
                value, idx = self._parse_block(next_idx, block_indent)
                result[key] = value
                continue
            value, idx = self._parse_value(rest, idx, indent)
            result[key] = value
        return result, idx

    def _parse_inline_mapping(
        self, first_rest: str, idx: int, indent: int
    ) -> tuple[dict[str, Any], int]:
        key, rest = self._split_key_value(first_rest, idx - 1)
        if rest.strip():
            value, idx = self._parse_value(rest, idx, indent - 2)
            result = {key: value}
        else:
            next_idx = self._skip_ignored(idx)
            if next_idx >= len(self.lines):
                result = {key: None}
                idx = next_idx
            else:
                next_indent = self._indent_of(next_idx)
                next_stripped = self.lines[next_idx].lstrip(" ")
                if next_indent <= indent - 2 and not next_stripped.startswith("- "):
                    result = {key: None}
                    idx = next_idx
                else:
                    block_indent = next_indent if next_indent > indent - 2 else indent - 2
                    value, idx = self._parse_block(next_idx, block_indent)
                    result = {key: value}
        while idx < len(self.lines):
            idx = self._skip_ignored(idx)
            if idx >= len(self.lines):
                break
            cur_indent = self._indent_of(idx)
            if cur_indent < indent:
                break
            if cur_indent != indent:
                raise YAMLError(f"Bad inline mapping indentation at line {idx + 1}")
            stripped = self.lines[idx][cur_indent:]
            if stripped.startswith("- "):
                break
            key, rest = self._split_key_value(stripped, idx)
            idx += 1
            if not rest.strip():
                next_idx = self._skip_ignored(idx)
                if next_idx >= len(self.lines):
                    result[key] = None
                    idx = next_idx
                    continue
                next_indent = self._indent_of(next_idx)
                next_stripped = self.lines[next_idx].lstrip(" ")
                if next_indent <= indent and not next_stripped.startswith("- "):
                    result[key] = None
                    idx = next_idx
                    continue
                block_indent = next_indent if next_indent > indent else indent
                value, idx = self._parse_block(next_idx, block_indent)
                result[key] = value
                continue
            value, idx = self._parse_value(rest, idx, indent)
            result[key] = value
        return result, idx

    def _parse_value(self, rest: str, idx: int, parent_indent: int) -> tuple[Any, int]:
        token = rest.strip()
        if token.startswith('"'):
            return self._parse_double_quoted(token, idx)
        if token.startswith("'"):
            return self._parse_single_quoted(token, idx)
        if token in {"|", "|-", "|+"}:
            return self._parse_block_scalar(idx, parent_indent, folded=False)
        if token in {">", ">-", ">+"}:
            return self._parse_block_scalar(idx, parent_indent, folded=True)
        return self._parse_plain_scalar(token, idx, parent_indent)

    def _parse_double_quoted(self, token: str, idx: int) -> tuple[str, int]:
        content = token[1:]
        while True:
            close_pos = self._find_unescaped_quote(content)
            if close_pos >= 0:
                payload = content[:close_pos]
                payload = re.sub(r"\\([ \t])", r"\1", payload)
                payload = re.sub(
                    r"\\x([0-9A-Fa-f]{2})",
                    lambda m: chr(int(m.group(1), 16)),
                    payload,
                )
                return json.loads(f'"{payload}"'), idx
            if idx >= len(self.lines):
                raise YAMLError("Unterminated double-quoted scalar")
            nxt = self.lines[idx].lstrip(" ")
            idx += 1
            if content.endswith("\\"):
                content = content[:-1] + nxt
            else:
                content += " " + nxt

    def _parse_single_quoted(self, token: str, idx: int) -> tuple[str, int]:
        content = token[1:]
        while True:
            pos = content.find("'")
            if pos >= 0:
                return content[:pos].replace("''", "'"), idx
            if idx >= len(self.lines):
                raise YAMLError("Unterminated single-quoted scalar")
            nxt = self.lines[idx].lstrip(" ")
            idx += 1
            content += " " + nxt

    def _parse_block_scalar(
        self, idx: int, parent_indent: int, *, folded: bool
    ) -> tuple[str, int]:
        collected: list[str] = []
        next_idx = idx
        child_indent: int | None = None
        while next_idx < len(self.lines):
            line = self.lines[next_idx]
            cur_indent = len(line) - len(line.lstrip(" "))
            if not line.strip():
                collected.append("")
                next_idx += 1
                continue
            if cur_indent <= parent_indent:
                break
            if child_indent is None:
                child_indent = cur_indent
            collected.append(line[child_indent:])
            next_idx += 1
        if folded:
            pieces: list[str] = []
            for chunk in collected:
                if not chunk:
                    pieces.append("\n")
                elif not pieces or pieces[-1].endswith("\n"):
                    pieces.append(chunk)
                else:
                    pieces.append(" " + chunk)
            return "".join(pieces).rstrip("\n"), next_idx
        return "\n".join(collected).rstrip("\n"), next_idx

    def _parse_plain_scalar(
        self, token: str, idx: int, parent_indent: int
    ) -> tuple[Any, int]:
        parts = [token]
        next_idx = idx
        while next_idx < len(self.lines):
            probe = self._skip_ignored(next_idx)
            if probe >= len(self.lines):
                next_idx = probe
                break
            line = self.lines[probe]
            cur_indent = self._indent_of(probe)
            if cur_indent <= parent_indent:
                next_idx = probe
                break
            stripped = line.lstrip(" ")
            if stripped.startswith("- ") and cur_indent == parent_indent:
                next_idx = probe
                break
            parts.append(stripped)
            next_idx = probe + 1
        combined = " ".join(part.strip() for part in parts if part.strip())
        return _parse_plain_scalar(combined), next_idx

    def _split_key_value(self, stripped: str, idx: int) -> tuple[str, str]:
        in_single = False
        in_double = False
        for pos, ch in enumerate(stripped):
            if ch == "'" and not in_double:
                in_single = not in_single
            elif ch == '"' and not in_single and (pos == 0 or stripped[pos - 1] != "\\"):
                in_double = not in_double
            elif ch == ":" and not in_single and not in_double:
                if pos + 1 == len(stripped) or stripped[pos + 1] in {" ", "\t"}:
                    return stripped[:pos].strip(), stripped[pos + 1 :]
        raise YAMLError(f"Missing ':' in mapping entry at line {idx + 1}")

    def _looks_like_mapping_entry(self, text: str) -> bool:
        try:
            self._split_key_value(text, 0)
            return True
        except YAMLError:
            return False

    def _skip_ignored(self, idx: int) -> int:
        while idx < len(self.lines):
            stripped = self.lines[idx].strip()
            if not stripped or stripped.startswith("#"):
                idx += 1
                continue
            return idx
        return idx

    def _indent_of(self, idx: int) -> int:
        line = self.lines[idx]
        return len(line) - len(line.lstrip(" "))

    @staticmethod
    def _find_unescaped_quote(content: str) -> int:
        escaped = False
        for pos, ch in enumerate(content):
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == '"':
                return pos
        return -1


def _parse_plain_scalar(token: str) -> Any:
    if token in {"null", "Null", "NULL", "~", ""}:
        return None
    if token in {"true", "True", "TRUE"}:
        return True
    if token in {"false", "False", "FALSE"}:
        return False
    if token.startswith("[") or token.startswith("{"):
        try:
            return json.loads(token)
        except Exception:
            return token
    try:
        if token.startswith("0") and token not in {"0", "0.0"} and not token.startswith("0."):
            raise ValueError
        return int(token)
    except Exception:
        pass
    try:
        return float(token)
    except Exception:
        return token


def _dump_node(
    value: Any,
    indent: int,
    *,
    allow_unicode: bool,
    sort_keys: bool,
) -> str:
    pad = " " * indent
    if isinstance(value, dict):
        items = value.items()
        if sort_keys:
            items = sorted(items, key=lambda kv: str(kv[0]))
        lines: list[str] = []
        for key, item in items:
            key_s = str(key)
            if _is_scalar(item):
                lines.append(f"{pad}{key_s}: {_dump_scalar(item, allow_unicode)}")
            else:
                lines.append(f"{pad}{key_s}:")
                lines.append(_dump_node(item, indent + 2, allow_unicode=allow_unicode, sort_keys=sort_keys))
        return "\n".join(lines) if lines else f"{pad}{{}}"
    if isinstance(value, list):
        if not value:
            return f"{pad}[]"
        lines = []
        for item in value:
            if _is_scalar(item):
                lines.append(f"{pad}- {_dump_scalar(item, allow_unicode)}")
            else:
                lines.append(f"{pad}-")
                lines.append(_dump_node(item, indent + 2, allow_unicode=allow_unicode, sort_keys=sort_keys))
        return "\n".join(lines)
    return f"{pad}{_dump_scalar(value, allow_unicode)}"


def _is_scalar(value: Any) -> bool:
    return not isinstance(value, (dict, list))


def _dump_scalar(value: Any, allow_unicode: bool) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if "\n" in text:
        body = "\n".join(f"  {line}" for line in text.splitlines())
        return f"|\n{body}"
    if text == "" or any(ch in text for ch in ":#[]{}&*!|>'\"%@`"):
        return json.dumps(text, ensure_ascii=not allow_unicode)
    return text
