from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import pr_agent.algo.ai_handlers.litellm_ai_handler as litellm_handler
from pr_agent.algo.ai_handlers.litellm_ai_handler import LiteLLMAIHandler


def _make_settings(enable_streaming_models=None):
    return type("Settings", (), {
        "config": type("Config", (), {
            "reasoning_effort": None,
            "ai_timeout": 30,
            "custom_reasoning_model": False,
            "max_model_tokens": 32000,
            "verbosity_level": 0,
            "seed": -1,
            "enable_streaming_models": enable_streaming_models or [],
            "get": lambda self, key, default=None: (
                self.enable_streaming_models if key == "enable_streaming_models" else default
            ),
        })(),
        "litellm": type("LiteLLM", (), {
            "get": lambda self, key, default=None: default,
        })(),
        "get": lambda self, key, default=None: default,
    })()


def _mock_response():
    mock = MagicMock()
    mock.__getitem__ = lambda self, key: {
        "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]
    }[key]
    mock.dict.return_value = {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}
    return mock


class TestStreamingModelConfiguration:
    @pytest.mark.asyncio
    async def test_default_streaming_model_enables_stream(self, monkeypatch):
        monkeypatch.setattr(litellm_handler, "get_settings", lambda: _make_settings())

        with patch("pr_agent.algo.ai_handlers.litellm_ai_handler.acompletion",
                   new_callable=AsyncMock) as mock_call, \
                patch("pr_agent.algo.ai_handlers.litellm_ai_handler._handle_streaming_response",
                      new_callable=AsyncMock,
                      return_value=("ok", "stop")):
            mock_call.return_value = object()

            handler = LiteLLMAIHandler()
            await handler.chat_completion(model="openai/qwq-plus", system="sys", user="usr")

        assert mock_call.call_args[1]["stream"] is True

    @pytest.mark.asyncio
    async def test_configured_streaming_model_enables_stream(self, monkeypatch):
        monkeypatch.setattr(
            litellm_handler,
            "get_settings",
            lambda: _make_settings(enable_streaming_models=["openrouter/my-stream-model"]),
        )

        with patch("pr_agent.algo.ai_handlers.litellm_ai_handler.acompletion",
                   new_callable=AsyncMock) as mock_call, \
                patch("pr_agent.algo.ai_handlers.litellm_ai_handler._handle_streaming_response",
                      new_callable=AsyncMock,
                      return_value=("ok", "stop")):
            mock_call.return_value = object()

            handler = LiteLLMAIHandler()
            await handler.chat_completion(model="openrouter/my-stream-model", system="sys", user="usr")

        assert mock_call.call_args[1]["stream"] is True

    @pytest.mark.asyncio
    async def test_unconfigured_model_remains_non_streaming(self, monkeypatch):
        monkeypatch.setattr(
            litellm_handler,
            "get_settings",
            lambda: _make_settings(enable_streaming_models=["openrouter/my-stream-model"]),
        )

        with patch("pr_agent.algo.ai_handlers.litellm_ai_handler.acompletion",
                   new_callable=AsyncMock) as mock_call:
            mock_call.return_value = _mock_response()

            handler = LiteLLMAIHandler()
            await handler.chat_completion(model="openrouter/normal-model", system="sys", user="usr")

        assert "stream" not in mock_call.call_args[1]
