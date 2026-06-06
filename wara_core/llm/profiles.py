from __future__ import annotations

import os
from typing import Any


DEFAULT_MODEL_PROFILE = "kimi-k2.6-no-thinking"
DEFAULT_CNY_PER_USD = 7.25


def _kimi_api_key_env() -> str:
    if "KIMI_API_KEY" not in os.environ and "MOONSHOT_API_KEY" in os.environ:
        return "MOONSHOT_API_KEY"
    return "KIMI_API_KEY"


def _float_env(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, "") or default)
    except ValueError:
        return default
    return value if value > 0 else default


def _usd_pricing_from_cny(cached_input: float, input_price: float, output: float) -> dict[str, float | str]:
    cny_per_usd = _float_env("WARA_CNY_PER_USD", DEFAULT_CNY_PER_USD)
    return {
        "currency": "USD",
        "cached_input_per_m": round(cached_input / cny_per_usd, 4),
        "input_per_m": round(input_price / cny_per_usd, 4),
        "output_per_m": round(output / cny_per_usd, 4),
        "converted_from_currency": "CNY",
        "cny_per_usd": cny_per_usd,
    }


def get_model_profiles() -> dict[str, dict[str, Any]]:
    """Return the WARA-wide model profile registry.

    The registry is shared by Phase 1, Phase 2, and Phase 3. Environment values
    are read when this function is called so local `.env` overrides are honored.
    """

    kimi_base_url = os.environ.get("KIMI_BASE_URL") or os.environ.get("MOONSHOT_BASE_URL") or "https://api.moonshot.cn/v1"
    deepseek_base_url = os.environ.get("DEEPSEEK_BASE_URL") or "https://api.deepseek.com"
    deepseek_chat_model = os.environ.get("DEEPSEEK_CHAT_MODEL") or "deepseek-chat"
    deepseek_reasoner_model = os.environ.get("DEEPSEEK_REASONER_MODEL") or "deepseek-reasoner"
    openai_base_url = os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    kimi_api_key_env = _kimi_api_key_env()
    kimi_uses_usd = "moonshot.ai" in kimi_base_url.lower()
    kimi_k26_pricing = (
        {"currency": "USD", "cached_input_per_m": 0.16, "input_per_m": 0.95, "output_per_m": 4.00}
        if kimi_uses_usd
        else _usd_pricing_from_cny(cached_input=1.10, input_price=6.50, output=27.00)
    )
    kimi_k25_pricing = (
        {"currency": "USD", "cached_input_per_m": 0.10, "input_per_m": 0.60, "output_per_m": 3.00}
        if kimi_uses_usd
        else _usd_pricing_from_cny(cached_input=0.70, input_price=4.00, output=21.00)
    )

    return {
        "kimi-k2.6-no-thinking": {
            "label": "Kimi 2.6 No Thinking",
            "provider": "openai-compatible",
            "base_url": kimi_base_url,
            "wire_api": "chat_completions",
            "api_key_env": kimi_api_key_env,
            "primary_model": "kimi-k2.6",
            "fallback_models": [],
            "reviewer_model": "kimi-k2.6",
            "reviewer_fallback_models": [],
            "reviewer_thinking_mode": "disabled",
            "pricing": {**kimi_k26_pricing, "unit": "per_1m_tokens", "source": "Moonshot/Kimi platform"},
        },
        "kimi-k2.6-thinking": {
            "label": "Kimi 2.6 Thinking",
            "provider": "openai-compatible",
            "base_url": kimi_base_url,
            "wire_api": "chat_completions",
            "api_key_env": kimi_api_key_env,
            "primary_model": "kimi-k2.6",
            "fallback_models": [],
            "reviewer_model": "kimi-k2.6",
            "reviewer_fallback_models": [],
            "reviewer_thinking_mode": "enabled",
            "pricing": {**kimi_k26_pricing, "unit": "per_1m_tokens", "source": "Moonshot/Kimi platform"},
        },
        "kimi-k2.5": {
            "label": "Kimi 2.5",
            "provider": "openai-compatible",
            "base_url": kimi_base_url,
            "wire_api": "chat_completions",
            "api_key_env": kimi_api_key_env,
            "primary_model": "kimi-k2.5",
            "fallback_models": [],
            "reviewer_model": "kimi-k2.5",
            "reviewer_fallback_models": [],
            "reviewer_thinking_mode": "disabled",
            "pricing": {**kimi_k25_pricing, "unit": "per_1m_tokens", "source": "Moonshot/Kimi platform"},
        },
        "openai-gpt-5.5": {
            "label": "OpenAI GPT-5.5",
            "provider": "openai",
            "base_url": openai_base_url,
            "wire_api": "chat_completions",
            "api_key_env": "OPENAI_API_KEY",
            "primary_model": "gpt-5.5",
            "fallback_models": [],
            "reviewer_model": "gpt-5.5",
            "reviewer_fallback_models": [],
            "reviewer_thinking_mode": "disabled",
            "pricing": {
                "currency": "USD",
                "cached_input_per_m": 0.25,
                "input_per_m": 2.50,
                "output_per_m": 15.00,
                "unit": "per_1m_tokens",
                "source": "OpenAI API pricing, standard short-context rate",
            },
        },
        "openai-gpt-5.3-codex": {
            "label": "OpenAI GPT-5.3-Codex",
            "provider": "openai",
            "base_url": openai_base_url,
            "wire_api": "chat_completions",
            "api_key_env": "OPENAI_API_KEY",
            "primary_model": "gpt-5.3-codex",
            "fallback_models": [],
            "reviewer_model": "gpt-5.3-codex",
            "reviewer_fallback_models": [],
            "reviewer_thinking_mode": "disabled",
            "pricing": {
                "currency": "USD",
                "cached_input_per_m": 0.175,
                "input_per_m": 1.75,
                "output_per_m": 14.00,
                "unit": "per_1m_tokens",
                "source": "OpenAI API pricing, Codex rate",
            },
        },
        "deepseek-chat": {
            "label": "DeepSeek Chat",
            "provider": "openai-compatible",
            "base_url": deepseek_base_url,
            "wire_api": "chat_completions",
            "api_key_env": "DEEPSEEK_API_KEY",
            "primary_model": deepseek_chat_model,
            "fallback_models": [],
            "reviewer_model": deepseek_chat_model,
            "reviewer_fallback_models": [],
            "reviewer_thinking_mode": "disabled",
            "pricing": {
                "currency": "USD",
                "cached_input_per_m": 0.07,
                "input_per_m": 0.27,
                "output_per_m": 1.10,
                "unit": "per_1m_tokens",
                "source": "DeepSeek API pricing",
            },
        },
        "deepseek-reasoner": {
            "label": "DeepSeek Reasoner",
            "provider": "openai-compatible",
            "base_url": deepseek_base_url,
            "wire_api": "chat_completions",
            "api_key_env": "DEEPSEEK_API_KEY",
            "primary_model": deepseek_reasoner_model,
            "fallback_models": [],
            "reviewer_model": deepseek_reasoner_model,
            "reviewer_fallback_models": [],
            "reviewer_thinking_mode": "disabled",
            "pricing": {
                "currency": "USD",
                "cached_input_per_m": 0.14,
                "input_per_m": 0.55,
                "output_per_m": 2.19,
                "unit": "per_1m_tokens",
                "source": "DeepSeek API pricing",
            },
        },
    }


MODEL_PROFILES = get_model_profiles()


def normalize_model_profile(model_profile: str | None) -> str:
    """Normalize UI/runtime model profile ids to the active WARA profile id."""

    requested = str(model_profile or "").strip()
    profiles = get_model_profiles()
    if requested in profiles:
        return requested
    if requested in {"kimi", "kimi-k2.6", "k2.6", "moonshot-kimi"}:
        return "kimi-k2.6-no-thinking"
    if requested in {"openai", "gpt-5.5", "gpt5.5", "openai-gpt"}:
        return "openai-gpt-5.5"
    if requested in {"codex", "openai-codex", "gpt-5.3-codex", "gpt5.3-codex"}:
        return "openai-gpt-5.3-codex"
    if requested in {"deepseek", "deepseek-chat", "deepseek-v3"}:
        return "deepseek-chat"
    if requested in {"deepseek-reasoner", "deepseek-r1", "deepseek-reasoning"}:
        return "deepseek-reasoner"
    return DEFAULT_MODEL_PROFILE


def get_model_profile(model_profile: str | None) -> dict[str, Any]:
    profile_name = normalize_model_profile(model_profile)
    return get_model_profiles()[profile_name]


def profile_id_for_model(model: str | None) -> str:
    """Map a provider model id in usage logs back to a WARA profile id."""

    value = str(model or "").strip().lower()
    if not value:
        return DEFAULT_MODEL_PROFILE
    if "kimi-k2.6" in value:
        return "kimi-k2.6-no-thinking"
    if "kimi-k2.5" in value:
        return "kimi-k2.5"
    if "gpt-5.3-codex" in value:
        return "openai-gpt-5.3-codex"
    if "gpt-5.5" in value:
        return "openai-gpt-5.5"
    if "deepseek-reasoner" in value:
        return "deepseek-reasoner"
    if "deepseek-chat" in value or "deepseek-v3" in value:
        return "deepseek-chat"
    return normalize_model_profile(value)
