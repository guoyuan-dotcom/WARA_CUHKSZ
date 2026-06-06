from __future__ import annotations

import os
import sys
from typing import Any

import yaml

from pipeline_core import DEFAULT_MODEL_PROFILE, ENGINE_ROOT, MODEL_PROFILES, WARA_LLM_CONFIG_TEMPLATE, WORKSPACE_ROOT, normalize_model_profile

if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))
if str(ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(ENGINE_ROOT))

from wara_core.llm.client import LLMClient  # noqa: E402


def create_llm_client(model_profile: str) -> LLMClient:
    profile_name = normalize_model_profile(model_profile)
    profile = MODEL_PROFILES.get(profile_name, MODEL_PROFILES[DEFAULT_MODEL_PROFILE])
    template = yaml.safe_load(WARA_LLM_CONFIG_TEMPLATE.read_text(encoding="utf-8"))
    template.setdefault("llm", {})
    for key in ["provider", "base_url", "wire_api", "api_key_env"]:
        if profile.get(key):
            template["llm"][key] = profile[key]
    template["llm"]["fallback_models"] = []
    template["llm"]["reviewer_fallback_models"] = []
    if profile.get("api_key_env"):
        template["llm"]["api_key"] = ""
    template["llm"]["primary_model"] = profile["primary_model"]
    template["llm"]["reviewer_model"] = profile["reviewer_model"]
    template["llm"]["reviewer_thinking_mode"] = profile["reviewer_thinking_mode"]
    env_overrides = {
        "max_retries": "WARA_LLM_MAX_RETRIES",
        "retry_base_delay": "WARA_LLM_RETRY_BASE_DELAY",
        "retry_max_delay": "WARA_LLM_RETRY_MAX_DELAY",
        "timeout_sec": "WARA_LLM_TIMEOUT_SEC",
    }
    for config_key, env_key in env_overrides.items():
        raw_value = os.environ.get(env_key, "").strip()
        if not raw_value:
            continue
        try:
            if config_key in {"retry_base_delay", "retry_max_delay"}:
                template["llm"][config_key] = float(raw_value)
            else:
                template["llm"][config_key] = int(raw_value)
        except ValueError:
            continue

    class _Obj:
        pass

    rc = _Obj()
    rc.llm = _Obj()
    for key, value in template["llm"].items():
        setattr(rc.llm, key, value)
    return LLMClient.from_rc_config(rc)
