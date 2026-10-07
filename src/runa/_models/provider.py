"""provider.py: `ModelProvider`, routing a model name to one of two backends by its prefix."""

import os
from dataclasses import dataclass

import httpx2 as httpx
from anthropic import AsyncAnthropic

from runa._loop import LoopCache
from runa._models.anthropic import AnthropicModel
from runa._models.chat_completions import OpenAICompatibleModel
from runa._models.interface import Model
from runa.exceptions import UserError

DEFAULT_MODEL = "gpt-5.4-nano"


@dataclass(frozen=True)
class _Endpoint:
    """One chat-completions-shaped provider: where it lives and which env var holds its key."""

    prefix: str
    base_url: str
    api_key_env: str


_OPENAI = _Endpoint("gpt", "https://api.openai.com/v1/", "OPENAI_API_KEY")
_ENDPOINTS: tuple[_Endpoint, ...] = (
    _Endpoint(
        "gemini", "https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY"
    ),
    _Endpoint("llama", "https://api.llama.com/compat/v1/", "LLAMA_API_KEY"),
    _Endpoint("deepseek", "https://api.deepseek.com/v1/", "DEEPSEEK_API_KEY"),
    _Endpoint(
        "qwen", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/", "DASHSCOPE_API_KEY"
    ),
)


def _client_for(endpoint: _Endpoint) -> httpx.AsyncClient:
    """An HTTP client pointed at `endpoint`, carrying the key its env var holds.

    Raises `UserError` when that variable is unset, which is why the client is built lazily: an
    app that never asks for a `gemini-*` model should not need `GEMINI_API_KEY`.
    """
    api_key = os.environ.get(endpoint.api_key_env)
    if api_key is None:
        raise UserError(
            f"{endpoint.api_key_env} is not set. Set it to use a {endpoint.prefix}-* model."
        )
    return httpx.AsyncClient(
        base_url=endpoint.base_url,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=600.0,
    )


class ModelProvider:
    """Routes a model name to one of two backends by its prefix.

    A name starting with `claude` goes through `AnthropicModel`; everything else (`gpt-*`,
    `gemini-*`, `llama-*`, `deepseek-*`, `qwen-*`, or an unrecognized/bare name) goes through
    `OpenAICompatibleModel` against that provider's own chat-completions endpoint.
    """

    def __init__(self) -> None:
        """Start with no clients; each is created lazily, on first use, and then reused.

        A client's connections are bound to the event loop running when it first makes a request,
        so each is held per loop by a `LoopCache`: `Agent.run_sync` opens a fresh loop per call,
        and a client left over from a closed one would crash the next call trying to reuse it.
        `get_model` does no I/O and is called at `Agent` construction time, with no loop running
        at all, which `LoopCache` treats as a client the first loop to use it adopts.
        """
        self._http_clients: LoopCache[str, httpx.AsyncClient] = LoopCache()
        self._anthropic_clients: LoopCache[str, AsyncAnthropic] = LoopCache()

    def get_model(self, model_name: str | None) -> Model:
        """Return the `Model` for `model_name` (or Runa's own default, if `None`)."""
        name = model_name or DEFAULT_MODEL
        lower = name.lower()

        if lower.startswith("claude"):
            return AnthropicModel(name, self._get_anthropic_client())

        endpoint = next((e for e in _ENDPOINTS if lower.startswith(e.prefix)), _OPENAI)
        return OpenAICompatibleModel(name, self._get_http_client(endpoint))

    def _get_http_client(self, endpoint: _Endpoint) -> httpx.AsyncClient:
        return self._http_clients.get(endpoint.prefix, lambda: _client_for(endpoint))

    def _get_anthropic_client(self) -> AsyncAnthropic:
        return self._anthropic_clients.get("claude", AsyncAnthropic)


__all__ = ["DEFAULT_MODEL", "ModelProvider"]
