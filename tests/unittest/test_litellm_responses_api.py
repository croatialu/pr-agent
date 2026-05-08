from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import litellm
import openai
import pytest

import pr_agent.algo.ai_handlers.litellm_ai_handler as litellm_handler
from pr_agent.algo.ai_handlers.litellm_ai_handler import LiteLLMAIHandler


def _make_settings(use_responses_api_models=None, enable_streaming_models=None, openai_key=None):
    return type("Settings", (), {
        "config": type("Config", (), {
            "reasoning_effort": "medium",
            "ai_timeout": 30,
            "custom_reasoning_model": False,
            "max_model_tokens": 32000,
            "verbosity_level": 0,
            "seed": -1,
            "use_responses_api_models": use_responses_api_models or [],
            "enable_streaming_models": enable_streaming_models or [],
            "get": lambda self, key, default=None: getattr(self, key, default),
        })(),
        "litellm": type("LiteLLM", (), {
            "get": lambda self, key, default=None: default,
        })(),
        "openai": type("OpenAI", (), {
            "key": openai_key,
        })(),
        "get": lambda self, key, default=None: openai_key if key == "OPENAI.KEY" and openai_key else default,
    })()


def _mock_chat_completion_response():
    mock = MagicMock()
    mock.__getitem__ = lambda self, key: {
        "choices": [{"message": {"content": "chat ok"}, "finish_reason": "stop"}]
    }[key]
    mock.dict.return_value = {"choices": [{"message": {"content": "chat ok"}, "finish_reason": "stop"}]}
    return mock


class _AsyncStream:
    def __init__(self, events):
        self._events = events

    def __aiter__(self):
        self._iter = iter(self._events)
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


class TestResponsesApiModels:
    @pytest.mark.asyncio
    async def test_configured_responses_model_uses_responses_api_streaming(self, monkeypatch):
        monkeypatch.setattr(
            litellm_handler,
            "get_settings",
            lambda: _make_settings(
                use_responses_api_models=["openai/gpt-5.5"],
                enable_streaming_models=["openai/gpt-5.5"],
            ),
        )
        stream = _AsyncStream([
            SimpleNamespace(type="response.output_text.delta", delta="hello"),
            SimpleNamespace(type="response.output_text.delta", delta=" world"),
            SimpleNamespace(type="response.completed", response=SimpleNamespace(status="completed")),
        ])

        with patch("pr_agent.algo.ai_handlers.litellm_ai_handler.AsyncOpenAI") as mock_client_cls, \
                patch("pr_agent.algo.ai_handlers.litellm_ai_handler.acompletion",
                      new_callable=AsyncMock) as mock_acompletion:
            mock_client = MagicMock()
            mock_client.responses.create = AsyncMock(return_value=stream)
            mock_client_cls.return_value = mock_client

            handler = LiteLLMAIHandler()
            response, finish_reason = await handler.chat_completion(
                model="openai/gpt-5.5",
                system="sys",
                user="usr",
            )

        assert response == "hello world"
        assert finish_reason == "stop"
        mock_acompletion.assert_not_called()
        mock_client.responses.create.assert_awaited_once()
        call_kwargs = mock_client.responses.create.call_args[1]
        assert call_kwargs["model"] == "gpt-5.5"
        assert call_kwargs["instructions"] == "sys"
        assert call_kwargs["input"] == [{"role": "user", "content": "usr"}]
        assert call_kwargs["stream"] is True
        assert call_kwargs["reasoning"] == {"effort": "medium"}
        assert "temperature" not in call_kwargs

    @pytest.mark.asyncio
    async def test_unconfigured_model_keeps_chat_completion_path(self, monkeypatch):
        monkeypatch.setattr(
            litellm_handler,
            "get_settings",
            lambda: _make_settings(use_responses_api_models=["openai/gpt-5.5"]),
        )

        with patch("pr_agent.algo.ai_handlers.litellm_ai_handler.AsyncOpenAI") as mock_client_cls, \
                patch("pr_agent.algo.ai_handlers.litellm_ai_handler.acompletion",
                      new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.return_value = _mock_chat_completion_response()

            handler = LiteLLMAIHandler()
            response, finish_reason = await handler.chat_completion(
                model="openai/kimi-for-coding",
                system="sys",
                user="usr",
            )

        assert response == "chat ok"
        assert finish_reason == "stop"
        mock_client_cls.assert_not_called()
        mock_acompletion.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_responses_model_config_matches_without_provider_prefix(self, monkeypatch):
        monkeypatch.setattr(
            litellm_handler,
            "get_settings",
            lambda: _make_settings(
                use_responses_api_models=["gpt-5.5"],
                enable_streaming_models=["gpt-5.5"],
            ),
        )
        stream = _AsyncStream([
            SimpleNamespace(type="response.output_text.delta", delta="ok"),
            SimpleNamespace(type="response.completed", response=SimpleNamespace(status="completed")),
        ])

        with patch("pr_agent.algo.ai_handlers.litellm_ai_handler.AsyncOpenAI") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.responses.create = AsyncMock(return_value=stream)
            mock_client_cls.return_value = mock_client

            handler = LiteLLMAIHandler()
            response, finish_reason = await handler.chat_completion(
                model="openai/gpt-5.5",
                system="sys",
                user="usr",
            )

        assert response == "ok"
        assert finish_reason == "stop"
        call_kwargs = mock_client.responses.create.call_args[1]
        assert call_kwargs["model"] == "gpt-5.5"
        assert call_kwargs["stream"] is True

    @pytest.mark.asyncio
    async def test_responses_api_client_uses_configured_openai_key(self, monkeypatch):
        monkeypatch.setattr(
            litellm_handler,
            "get_settings",
            lambda: _make_settings(
                use_responses_api_models=["openai/gpt-5.5"],
                openai_key="test-openai-key",
            ),
        )

        with patch("pr_agent.algo.ai_handlers.litellm_ai_handler.AsyncOpenAI") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.responses.create = AsyncMock(return_value=SimpleNamespace(output_text="ok", status="completed"))
            mock_client_cls.return_value = mock_client

            handler = LiteLLMAIHandler()
            await handler.chat_completion(
                model="openai/gpt-5.5",
                system="sys",
                user="usr",
            )

        assert mock_client_cls.call_args[1]["api_key"] == "test-openai-key"

    @pytest.mark.asyncio
    async def test_responses_api_client_does_not_use_generic_litellm_api_key(self, monkeypatch):
        monkeypatch.setattr(litellm, "openai_key", None)
        monkeypatch.setattr(litellm, "api_key", "test-generic-provider-key")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setattr(
            litellm_handler,
            "get_settings",
            lambda: _make_settings(use_responses_api_models=["openai/gpt-5.5"]),
        )

        with patch("pr_agent.algo.ai_handlers.litellm_ai_handler.AsyncOpenAI") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.responses.create = AsyncMock(return_value=SimpleNamespace(output_text="ok", status="completed"))
            mock_client_cls.return_value = mock_client

            handler = LiteLLMAIHandler()
            await handler.chat_completion(
                model="openai/gpt-5.5",
                system="sys",
                user="usr",
            )

        assert "api_key" not in mock_client_cls.call_args[1]

    @pytest.mark.asyncio
    async def test_responses_api_forwards_extra_headers(self, monkeypatch):
        monkeypatch.setattr(
            litellm_handler,
            "get_settings",
            lambda: _make_settings(use_responses_api_models=["openai/gpt-5.5"]),
        )

        with patch("pr_agent.algo.ai_handlers.litellm_ai_handler.AsyncOpenAI") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.responses.create = AsyncMock(return_value=SimpleNamespace(output_text="ok", status="completed"))
            mock_client_cls.return_value = mock_client

            handler = LiteLLMAIHandler()
            await handler._get_completion(
                model="openai/gpt-5.5",
                messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": "usr"}],
                timeout=30,
                extra_headers={"x-test": "1"},
            )

        call_kwargs = mock_client.responses.create.call_args[1]
        assert call_kwargs["extra_headers"] == {"x-test": "1"}

    @pytest.mark.asyncio
    async def test_responses_api_rejects_non_openai_provider_prefix(self, monkeypatch):
        monkeypatch.setattr(
            litellm_handler,
            "get_settings",
            lambda: _make_settings(use_responses_api_models=["azure/gpt-5.5"]),
        )

        handler = LiteLLMAIHandler()
        with pytest.raises(ValueError, match="Responses API is only supported for OpenAI models"):
            await handler._get_completion(
                model="azure/gpt-5.5",
                messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": "usr"}],
                timeout=30,
            )

    @pytest.mark.asyncio
    async def test_responses_api_model_can_run_without_streaming(self, monkeypatch):
        monkeypatch.setattr(
            litellm_handler,
            "get_settings",
            lambda: _make_settings(use_responses_api_models=["openai/gpt-5.5"]),
        )

        with patch("pr_agent.algo.ai_handlers.litellm_ai_handler.AsyncOpenAI") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.responses.create = AsyncMock(return_value=SimpleNamespace(output_text="ok", status="completed"))
            mock_client_cls.return_value = mock_client

            handler = LiteLLMAIHandler()
            response, finish_reason = await handler.chat_completion(
                model="openai/gpt-5.5",
                system="sys",
                user="usr",
            )

        assert response == "ok"
        assert finish_reason == "stop"
        call_kwargs = mock_client.responses.create.call_args[1]
        assert call_kwargs["model"] == "gpt-5.5"
        assert call_kwargs["stream"] is False

    @pytest.mark.asyncio
    async def test_responses_stream_incomplete_raises_api_error(self, monkeypatch):
        monkeypatch.setattr(
            litellm_handler,
            "get_settings",
            lambda: _make_settings(
                use_responses_api_models=["openai/gpt-5.5"],
                enable_streaming_models=["openai/gpt-5.5"],
            ),
        )
        stream = _AsyncStream([
            SimpleNamespace(type="response.output_text.delta", delta="partial"),
            SimpleNamespace(
                type="response.incomplete",
                response=SimpleNamespace(incomplete_details={"reason": "max_output_tokens"}),
            ),
        ])

        with patch("pr_agent.algo.ai_handlers.litellm_ai_handler.AsyncOpenAI") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.responses.create = AsyncMock(return_value=stream)
            mock_client_cls.return_value = mock_client

            handler = LiteLLMAIHandler()
            with pytest.raises(openai.APIError, match="Responses API stream incomplete"):
                await handler.chat_completion(
                    model="openai/gpt-5.5",
                    system="sys",
                    user="usr",
                )
