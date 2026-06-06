from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from wara_core.llm.client import LLMClient, LLMConfig


class LLMClientTokenBudgetTests(unittest.TestCase):
    def test_client_uses_bounded_timeout_by_default(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            client = LLMClient(
                LLMConfig(
                    base_url="https://api.openai.com/v1",
                    api_key="test",
                    primary_model="gpt-5.5",
                    timeout_sec=0,
                )
            )
        self.assertEqual(client.config.timeout_sec, 900)

    def test_client_honors_timeout_env_override(self) -> None:
        with patch.dict(os.environ, {"WARA_LLM_TIMEOUT_SEC": "123"}, clear=True):
            client = LLMClient(
                LLMConfig(
                    base_url="https://api.openai.com/v1",
                    api_key="test",
                    primary_model="gpt-5.5",
                    timeout_sec=900,
                )
            )
        self.assertEqual(client.config.timeout_sec, 123)

    def test_openai_gpt55_completion_budget_uses_configurable_floor_not_32k_default(self) -> None:
        client = LLMClient(
            LLMConfig(
                base_url="https://api.openai.com/v1",
                api_key="test",
                primary_model="gpt-5.5",
            )
        )
        with patch.dict(os.environ, {}, clear=False):
            body = client._build_body(
                "gpt-5.5",
                [{"role": "user", "content": "write code"}],
                max_tokens=6000,
                temperature=0.7,
                json_mode=False,
                thinking=None,
            )
        self.assertEqual(body["max_completion_tokens"], 12000)

    def test_openai_gpt55_completion_budget_honors_cap(self) -> None:
        client = LLMClient(
            LLMConfig(
                base_url="https://api.openai.com/v1",
                api_key="test",
                primary_model="gpt-5.5",
            )
        )
        with patch.dict(
            os.environ,
            {
                "WARA_OPENAI_NATIVE_MAX_COMPLETION_TOKEN_FLOOR": "16000",
                "WARA_OPENAI_NATIVE_MAX_COMPLETION_TOKEN_CAP": "10000",
            },
        ):
            body = client._build_body(
                "gpt-5.5",
                [{"role": "user", "content": "write code"}],
                max_tokens=6000,
                temperature=0.7,
                json_mode=False,
                thinking=None,
            )
        self.assertEqual(body["max_completion_tokens"], 10000)

    def test_openai_gpt55_reasoning_effort_is_serialized_when_configured(self) -> None:
        client = LLMClient(
            LLMConfig(
                base_url="https://api.openai.com/v1",
                api_key="test",
                primary_model="gpt-5.5",
                reasoning_effort="low",
            )
        )
        body = client._build_body(
            "gpt-5.5",
            [{"role": "user", "content": "write code"}],
            max_tokens=6000,
            temperature=0.7,
            json_mode=False,
            thinking=None,
        )
        self.assertEqual(body["reasoning_effort"], "low")


if __name__ == "__main__":
    unittest.main()
