from __future__ import annotations

import json
import logging
import os
import random
import re
import signal
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any


logger = logging.getLogger(__name__)

_NEW_TOKEN_PARAM_MODELS = frozenset(
    {
        "o3",
        "o3-mini",
        "o4-mini",
        "gpt-5",
        "gpt-5.1",
        "gpt-5.2",
        "gpt-5.3",
        "gpt-5.4",
        "gpt-5.5",
    }
)
_NO_TEMPERATURE_PREFIXES = frozenset({"o3", "o3-mini", "o4-mini", "gpt-5"})
_OMIT_TEMPERATURE_PREFIXES = frozenset({"kimi-k2.5", "kimi-k2.6"})
_KIMI_THINKING_PREFIXES = frozenset({"kimi-k2.5", "kimi-k2.6", "kimi-k2-thinking", "kimi-k2-thinking-turbo"})
_MAX_BACKOFF_SEC = 900
_DEFAULT_TIMEOUT_SEC = 900
_DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)


@dataclass
class LLMResponse:
    content: str
    model: str
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    finish_reason: str = ""
    truncated: bool = False
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMConfig:
    base_url: str
    api_key: str
    wire_api: str = "chat_completions"
    primary_model: str = "gpt-5.5"
    fallback_models: list[str] = field(default_factory=list)
    max_tokens: int = 4096
    temperature: float = 0.7
    max_retries: int = 3
    retry_base_delay: float = 2.0
    retry_max_delay: float = _MAX_BACKOFF_SEC
    # 0 can still be used explicitly through WARA_LLM_TIMEOUT_SEC=0 for
    # debugging, but production WARA runs should not wait forever on a
    # provider connection.
    timeout_sec: int = _DEFAULT_TIMEOUT_SEC
    reasoning_effort: str = ""
    user_agent: str = _DEFAULT_USER_AGENT
    extra_headers: dict[str, str] = field(default_factory=dict)
    fallback_url: str = ""
    fallback_api_key: str = ""


@contextmanager
def _temporary_dns_override(hostname: str, ip_address: str):
    if not hostname or not ip_address:
        yield
        return
    original = socket.getaddrinfo

    def patched(host, port, family=0, type=0, proto=0, flags=0):
        if str(host).lower() == hostname.lower():
            host = ip_address
        return original(host, port, family, type, proto, flags)

    socket.getaddrinfo = patched
    try:
        yield
    finally:
        socket.getaddrinfo = original


def _resolve_timeout_sec(configured_timeout: int | float | str | None) -> int:
    raw_env = os.environ.get("WARA_LLM_TIMEOUT_SEC", "").strip()
    raw_value: Any = raw_env if raw_env else configured_timeout
    try:
        value = int(float(raw_value))
    except (TypeError, ValueError):
        value = _DEFAULT_TIMEOUT_SEC
    if raw_env and value <= 0:
        return 0
    return value if value > 0 else _DEFAULT_TIMEOUT_SEC


@contextmanager
def _request_deadline(timeout_sec: int | None):
    """Apply a wall-clock deadline for main-thread provider calls."""
    if not timeout_sec or timeout_sec <= 0:
        yield
        return
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    if not hasattr(signal, "SIGALRM") or not hasattr(signal, "setitimer"):
        yield
        return
    if signal.getitimer(signal.ITIMER_REAL)[0] > 0:
        yield
        return

    old_handler = signal.getsignal(signal.SIGALRM)

    def _timeout_handler(signum, frame):  # noqa: ARG001
        raise TimeoutError(f"LLM request exceeded timeout_sec={timeout_sec}")

    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, float(timeout_sec))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


class LLMClient:
    """WARA-native OpenAI-compatible chat client used by Phase1 and Phase2."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.config.timeout_sec = _resolve_timeout_sec(config.timeout_sec)
        self._model_chain = [config.primary_model]

    @classmethod
    def from_rc_config(cls, rc_config: Any) -> "LLMClient":
        llm = getattr(rc_config, "llm", rc_config)
        api_key_env = str(getattr(llm, "api_key_env", "OPENAI_API_KEY") or "OPENAI_API_KEY")
        api_key = str(getattr(llm, "api_key", "") or os.environ.get(api_key_env, "") or "")
        base_url = str(getattr(llm, "base_url", "") or "https://api.openai.com/v1")
        return cls(
            LLMConfig(
                base_url=base_url,
                api_key=api_key,
                wire_api=str(getattr(llm, "wire_api", "chat_completions") or "chat_completions"),
                primary_model=str(getattr(llm, "primary_model", "") or "gpt-5.5"),
                fallback_models=list(getattr(llm, "fallback_models", []) or []),
                max_tokens=int(getattr(llm, "max_tokens", 4096) or 4096),
                temperature=float(getattr(llm, "temperature", 0.7) or 0.7),
                max_retries=int(getattr(llm, "max_retries", 3) or 3),
                retry_base_delay=float(getattr(llm, "retry_base_delay", 2.0) or 2.0),
                retry_max_delay=float(getattr(llm, "retry_max_delay", _MAX_BACKOFF_SEC) or _MAX_BACKOFF_SEC),
                timeout_sec=int(getattr(llm, "timeout_sec", 0) or 0),
                reasoning_effort=str(getattr(llm, "reasoning_effort", "") or ""),
            )
        )

    @staticmethod
    def _normalize_wire_api(wire_api: str) -> str:
        normalized = (wire_api or "").strip().lower().replace("-", "_")
        if normalized in {"", "chat/completions", "chat_completions"}:
            return "chat_completions"
        if normalized == "responses":
            return "responses"
        return normalized

    def _endpoint_path(self) -> str:
        return "/responses" if self._normalize_wire_api(self.config.wire_api) == "responses" else "/chat/completions"

    def _endpoint_url(self, base_url: str) -> str:
        return f"{base_url.rstrip('/')}{self._endpoint_path()}"

    @staticmethod
    def _has_prefix(model: str, prefixes: set[str] | frozenset[str]) -> bool:
        return any(model.startswith(prefix) for prefix in prefixes)

    def _uses_openai_native_token_params(self) -> bool:
        parsed = urllib.parse.urlparse(self.config.base_url)
        hostname = (parsed.hostname or "").lower()
        return hostname == "api.openai.com"

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        json_mode: bool = False,
        system: str | None = None,
        strip_thinking: bool = False,
        thinking: dict[str, Any] | None = None,
    ) -> LLMResponse:
        if system:
            messages = [{"role": "system", "content": system}] + list(messages)
        models = [model] if model else list(self._model_chain)
        max_tok = int(max_tokens or self.config.max_tokens)
        temp = self.config.temperature if temperature is None else float(temperature)
        last_error: Exception | None = None
        for candidate in models:
            if not candidate:
                continue
            try:
                response = self._call_with_retry(candidate, messages, max_tok, temp, json_mode, thinking)
                if strip_thinking:
                    response = LLMResponse(
                        content=_strip_thinking_tags(response.content),
                        model=response.model,
                        prompt_tokens=response.prompt_tokens,
                        cached_tokens=response.cached_tokens,
                        completion_tokens=response.completion_tokens,
                        total_tokens=response.total_tokens,
                        finish_reason=response.finish_reason,
                        truncated=response.truncated,
                        raw=response.raw,
                    )
                self._persist_usage(response)
                return response
            except Exception as exc:  # noqa: BLE001
                if len(models) > 1:
                    logger.warning("Model %s failed: %s. Trying next.", candidate, exc)
                else:
                    logger.warning("Model %s failed: %s.", candidate, exc)
                print(f"[WARA] LLM model {candidate} failed: {exc}", flush=True)
                last_error = exc
        raise RuntimeError(f"All models failed. Last error: {last_error}") from last_error

    def preflight(self) -> tuple[bool, str]:
        try:
            self.chat([{"role": "user", "content": "ping"}], max_tokens=64, temperature=0)
            return True, f"OK - model {self.config.primary_model} responding"
        except urllib.error.HTTPError as exc:
            return False, f"HTTP {exc.code}"
        except Exception as exc:  # noqa: BLE001
            return False, str(exc)

    def _call_with_retry(
        self,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        json_mode: bool,
        thinking: dict[str, Any] | None,
    ) -> LLMResponse:
        for attempt in range(max(1, self.config.max_retries)):
            try:
                return self._raw_call(model, messages, max_tokens, temperature, json_mode, thinking)
            except urllib.error.HTTPError as exc:
                body = _safe_read_http_body(exc)
                if exc.code == 400 and not _is_transient_error(body):
                    raise
                if exc.code not in {400, 429, 500, 502, 503, 504, 529}:
                    raise
                if attempt >= self.config.max_retries - 1:
                    raise
                self._sleep_before_retry(attempt, model, exc.code, exc.headers.get("Retry-After") if exc.headers else None)
            except (urllib.error.URLError, TimeoutError, OSError):
                if attempt >= self.config.max_retries - 1:
                    raise
                self._sleep_before_retry(attempt, model, 0)
        raise RuntimeError(f"LLM call failed after {self.config.max_retries} retries for model {model}")

    def _sleep_before_retry(self, attempt: int, model: str, status: int, retry_after: str | None = None) -> None:
        delay = min(self.config.retry_base_delay * (2**attempt), self.config.retry_max_delay)
        if retry_after:
            try:
                delay = max(delay, min(float(retry_after), self.config.retry_max_delay))
            except ValueError:
                pass
        delay += random.uniform(0, delay * 0.25)
        message = (
            f"[WARA] LLM transient error for {model} "
            f"(status={status}); retry {attempt + 2}/{self.config.max_retries} in {delay:.1f}s"
        )
        print(message, flush=True)
        logger.info(message)
        time.sleep(delay)

    def _raw_call(
        self,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        json_mode: bool,
        thinking: dict[str, Any] | None,
    ) -> LLMResponse:
        body = self._build_body(model, [dict(message) for message in messages], max_tokens, temperature, json_mode, thinking)
        stream_chat = bool(getattr(self.config, "stream", False)) and self._normalize_wire_api(self.config.wire_api) == "chat_completions"
        if stream_chat:
            body["stream"] = True
        payload = json.dumps(body).encode("utf-8")
        url = self._endpoint_url(self.config.base_url)
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "User-Agent": self.config.user_agent,
        }
        headers.update(self.config.extra_headers)
        request = urllib.request.Request(url, data=payload, headers=headers)
        parsed = urllib.parse.urlparse(url)
        dns_override = os.environ.get("OPENAI_API_RESOLVE_IP", "").strip() if parsed.hostname == "api.openai.com" else ""
        timeout = self.config.timeout_sec if self.config.timeout_sec > 0 else None
        endpoint_label = f"{parsed.netloc}{parsed.path}"
        started_at = time.time()
        effective_output_tokens = body.get("max_completion_tokens", body.get("max_tokens", max_tokens))
        print(
            "[WARA] LLM request start "
            f"model={model} endpoint={endpoint_label} requested_max_tokens={max_tokens} "
            f"effective_output_tokens={effective_output_tokens} timeout_sec={timeout or 0}",
            flush=True,
        )
        try:
            with _temporary_dns_override(parsed.hostname or "", dns_override):
                with _request_deadline(timeout):
                    with urllib.request.urlopen(request, timeout=timeout) as response:
                        if stream_chat:
                            parsed_response = self._parse_chat_stream_response(response, model)
                            print(
                                "[WARA] LLM request complete "
                                f"model={model} elapsed_sec={time.time() - started_at:.1f}",
                                flush=True,
                            )
                            return parsed_response
                        data = json.loads(response.read())
        except (urllib.error.URLError, OSError):
            raise
        finally:
            if "data" not in locals():
                logger.debug("LLM request did not complete for model %s", model)
        if not isinstance(data, dict):
            raise ValueError(f"Malformed API response: expected object, got {type(data).__name__}")
        if data.get("error"):
            raise _http_error_from_api_error(data["error"])
        if self._normalize_wire_api(self.config.wire_api) == "responses":
            parsed_response = self._parse_responses_response(data, model)
        else:
            parsed_response = self._parse_chat_response(data, model)
        print(
            "[WARA] LLM request complete "
            f"model={model} elapsed_sec={time.time() - started_at:.1f}",
            flush=True,
        )
        return parsed_response

    @staticmethod
    def _parse_chat_stream_response(response: Any, model: str) -> LLMResponse:
        chunks: list[str] = []
        non_sse_lines: list[str] = []
        finish_reason = ""
        usage: dict[str, Any] = {}
        response_model = model
        for raw_line in response:
            try:
                line = raw_line.decode("utf-8", errors="ignore").strip()
            except AttributeError:
                line = str(raw_line).strip()
            if not line:
                continue
            if not line.startswith("data:"):
                non_sse_lines.append(line)
                continue
            data_text = line[len("data:") :].strip()
            if data_text == "[DONE]":
                break
            try:
                data = json.loads(data_text)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict) and data.get("error"):
                raise _http_error_from_api_error(data["error"])
            if isinstance(data.get("usage"), dict):
                usage = data["usage"]
            if data.get("model"):
                response_model = str(data.get("model") or model)
            choices = data.get("choices")
            if not isinstance(choices, list) or not choices:
                continue
            choice = choices[0] if isinstance(choices[0], dict) else {}
            if choice.get("finish_reason"):
                finish_reason = str(choice.get("finish_reason") or "")
            delta = choice.get("delta", {}) if isinstance(choice.get("delta"), dict) else {}
            content = delta.get("content")
            if isinstance(content, list):
                chunks.extend(str(item.get("text", "")) for item in content if isinstance(item, dict))
            elif content:
                chunks.append(str(content))
            message = choice.get("message", {}) if isinstance(choice.get("message"), dict) else {}
            message_content = message.get("content")
            if message_content and not chunks:
                chunks.append(str(message_content))
        if not chunks and non_sse_lines:
            fallback_text = "\n".join(non_sse_lines).strip()
            try:
                data = json.loads(fallback_text)
            except json.JSONDecodeError:
                data = {}
            if isinstance(data, dict) and data:
                if data.get("error"):
                    raise _http_error_from_api_error(data["error"])
                return LLMClient._parse_chat_response(data, model)
        return LLMResponse(
            content="".join(chunks),
            model=response_model,
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            cached_tokens=int(usage.get("cached_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
            total_tokens=int(usage.get("total_tokens", 0) or 0),
            finish_reason=finish_reason,
            truncated=finish_reason == "length",
            raw={"stream": True},
        )

    def _build_body(
        self,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        json_mode: bool,
        thinking: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if self._normalize_wire_api(self.config.wire_api) == "responses":
            body: dict[str, Any] = {
                "model": model,
                "input": [
                    {
                        "role": str(message.get("role", "user") or "user"),
                        "content": [{"type": "input_text", "text": str(message.get("content", "") or "")}],
                    }
                    for message in messages
                ],
                "max_output_tokens": max_tokens,
            }
            if not self._has_prefix(model, _NO_TEMPERATURE_PREFIXES):
                body["temperature"] = temperature
            return body

        body = {"model": model, "messages": messages}
        if not self._has_prefix(model, _NO_TEMPERATURE_PREFIXES) and not self._has_prefix(model, _OMIT_TEMPERATURE_PREFIXES):
            body["temperature"] = temperature
        if self._has_prefix(model, _NEW_TOKEN_PARAM_MODELS) and self._uses_openai_native_token_params():
            reasoning_effort = _openai_reasoning_effort(getattr(self.config, "reasoning_effort", ""))
            if reasoning_effort:
                body["reasoning_effort"] = reasoning_effort
        effective_thinking_type = ""
        if isinstance(thinking, dict) and thinking:
            body["thinking"] = thinking
            effective_thinking_type = str(thinking.get("type") or "").lower()
        elif self._has_prefix(model, {"kimi-k2.5", "kimi-k2.6"}):
            body["thinking"] = {"type": "disabled"}
            effective_thinking_type = "disabled"

        if self._has_prefix(model, _NEW_TOKEN_PARAM_MODELS) and self._uses_openai_native_token_params():
            token_floor = _int_env("WARA_OPENAI_NATIVE_MAX_COMPLETION_TOKEN_FLOOR", 12000)
            token_cap = _int_env("WARA_OPENAI_NATIVE_MAX_COMPLETION_TOKEN_CAP", 0)
            requested_tokens = max(max_tokens, token_floor)
            body["max_completion_tokens"] = min(requested_tokens, token_cap) if token_cap > 0 else requested_tokens
        elif self._has_prefix(model, _KIMI_THINKING_PREFIXES) and effective_thinking_type == "enabled":
            body["max_tokens"] = max(max_tokens, int(os.environ.get("WARA_KIMI_THINKING_MAX_TOKENS", "8192")))
        else:
            body["max_tokens"] = max_tokens

        if json_mode:
            if _model_uses_prompt_json_instruction(model):
                hint = "You MUST respond with valid JSON only. Do not include text outside the JSON object."
                if messages and messages[0].get("role") == "system":
                    messages[0]["content"] = hint + "\n\n" + str(messages[0].get("content") or "")
                else:
                    messages.insert(0, {"role": "system", "content": hint})
            else:
                body["response_format"] = {"type": "json_object"}
        return body

    @staticmethod
    def _parse_chat_response(data: dict[str, Any], model: str) -> LLMResponse:
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError(f"Malformed API response: missing choices. Got: {data}")
        choice = choices[0] if isinstance(choices[0], dict) else {}
        usage = data.get("usage", {}) if isinstance(data.get("usage"), dict) else {}
        message = choice.get("message", {}) if isinstance(choice.get("message"), dict) else {}
        content = message.get("content") or ""
        if isinstance(content, list):
            content = "\n".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
        finish_reason = str(choice.get("finish_reason") or "")
        return LLMResponse(
            content=str(content or ""),
            model=str(data.get("model") or model),
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            cached_tokens=int(usage.get("cached_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
            total_tokens=int(usage.get("total_tokens", 0) or 0),
            finish_reason=finish_reason,
            truncated=finish_reason == "length",
            raw=data,
        )

    @staticmethod
    def _parse_responses_response(data: dict[str, Any], model: str) -> LLMResponse:
        chunks: list[str] = []
        for item in data.get("output", []) or []:
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            for content_item in item.get("content", []) or []:
                if isinstance(content_item, dict) and content_item.get("type") == "output_text":
                    chunks.append(str(content_item.get("text") or ""))
        usage = data.get("usage", {}) if isinstance(data.get("usage"), dict) else {}
        prompt_tokens = int(usage.get("input_tokens", 0) or 0)
        completion_tokens = int(usage.get("output_tokens", 0) or 0)
        total_tokens = int(usage.get("total_tokens", prompt_tokens + completion_tokens) or 0)
        details = usage.get("input_tokens_details", {}) if isinstance(usage.get("input_tokens_details"), dict) else {}
        finish_reason = str(data.get("status") or "")
        incomplete = data.get("incomplete_details", {}) if isinstance(data.get("incomplete_details"), dict) else {}
        reason = str(incomplete.get("reason") or "")
        if reason:
            finish_reason = reason
        return LLMResponse(
            content="".join(chunks),
            model=str(data.get("model") or model),
            prompt_tokens=prompt_tokens,
            cached_tokens=int(details.get("cached_tokens", usage.get("cached_tokens", 0)) or 0),
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            finish_reason=finish_reason,
            truncated=finish_reason in {"length", "max_output_tokens", "content_filter"},
            raw=data,
        )

    def _persist_usage(self, response: LLMResponse) -> None:
        usage_log = str(os.environ.get("WARA_USAGE_LOG", "")).strip()
        if not usage_log:
            return
        try:
            os.makedirs(os.path.dirname(usage_log), exist_ok=True)
            with open(usage_log, "a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "ts": time.time(),
                            "model": response.model,
                            "prompt_tokens": response.prompt_tokens,
                            "cached_tokens": response.cached_tokens,
                            "completion_tokens": response.completion_tokens,
                            "total_tokens": response.total_tokens,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to persist usage record: %s", exc)


def _strip_thinking_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", str(text or ""), flags=re.DOTALL | re.IGNORECASE).strip()


def _int_env(name: str, default: int) -> int:
    raw = str(os.environ.get(name, "")).strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _openai_reasoning_effort(config_value: str) -> str:
    value = str(os.environ.get("WARA_OPENAI_REASONING_EFFORT") or config_value or "").strip().lower()
    if value in {"none", "low", "medium", "high", "xhigh"}:
        return value
    return ""


def _model_uses_prompt_json_instruction(model: str) -> bool:
    lowered = model.lower()
    return lowered.startswith(
        (
            "deepseek",
            "moonshot",
            "kimi",
            "kimi-k2",
        )
    )


def _safe_read_http_body(exc: urllib.error.HTTPError) -> str:
    try:
        return exc.read().decode("utf-8", errors="ignore")
    except Exception:  # noqa: BLE001
        return ""


def _is_transient_error(body: str) -> bool:
    lowered = body.lower()
    return any(
        token in lowered
        for token in (
            "rate limit",
            "ratelimit",
            "overloaded",
            "temporarily",
            "capacity",
            "throttl",
            "too many",
            "retry",
            "saturated",
            "later",
            "upstream",
        )
    )


def _http_error_from_api_error(error: Any) -> urllib.error.HTTPError:
    message = str(error.get("message", error)) if isinstance(error, dict) else str(error)
    import io

    return urllib.error.HTTPError("", 500, message, None, io.BytesIO(message.encode("utf-8")))


__all__ = ["LLMClient", "LLMConfig", "LLMResponse"]
