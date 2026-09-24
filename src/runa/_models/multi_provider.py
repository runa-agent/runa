"""multi_provider.py: `ModelProvider`, routing a model name to one of two backends by its prefix."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass

import httpx2 as httpx
from anthropic import AsyncAnthropic

from runa._models.anthropic import AnthropicModel
from runa._models.interface import Model
from runa._models.openai_chatcompletions import OpenAICompatibleModel
from runa.exceptions import UserError

DEFAULT_MODEL = "gpt-5.4-nano"


@dataclass(frozen=True)
class _Backend:
    """One chat-completions-shaped provider: where it lives and which env var holds its key."""

    prefix: str
    base_url: str
    api_key_env: str


_OPENAI = _Backend("gpt", "https://api.openai.com/v1/", "OPENAI_API_KEY")
_BACKENDS: tuple[_Backend, ...] = (
    _Backend(
        "gemini", "https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY"
    ),
    _Backend("llama", "https://api.llama.com/compat/v1/", "LLAMA_API_KEY"),
    _Backend("deepseek", "https://api.deepseek.com/v1/", "DEEPSEEK_API_KEY"),
    _Backend(
        "qwen", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/", "DASHSCOPE_API_KEY"
    ),
)


class ModelProvider:
    """Routes a model name to one of two backends by its prefix.

    A name starting with `claude` goes through `AnthropicModel`; everything else (`gpt-*`,
    `gemini-*`, `llama-*`, `deepseek-*`, `qwen-*`, or an unrecognized/bare name) goes through
    `OpenAICompatibleModel` against that provider's own chat-completions endpoint.
    """

    def __init__(self) -> None:
        """Start with no clients; each is created lazily, on first use, and then reused.

        A client's connections are bound to the event loop running when it first makes a
        request, so each is cached alongside that loop; `run_sync` opens a fresh loop per
        call (`asyncio.run`), and a client left over from a now-closed loop would crash the
        next call trying to reuse it. A loop mismatch discards the stale client and builds a
        new one instead. `get_model` itself has no async requirement (it does no I/O), so
        this also has to tolerate being called with no running loop at all, e.g. at `Agent`
        construction time; the loop is then simply left unrecorded until first real use.
        """
        self._http_clients: dict[
            str, tuple[asyncio.AbstractEventLoop | None, httpx.AsyncClient]
        ] = {}
        self._anthropic_client: tuple[asyncio.AbstractEventLoop | None, AsyncAnthropic] | None = (
            None
        )

    def get_model(self, model_name: str | None) -> Model:
        """Return the `Model` for `model_name` (or Runa's own default, if `None`)."""
        name = model_name or DEFAULT_MODEL
        lower = name.lower()

        if lower.startswith("claude"):
            return AnthropicModel(name, self._get_anthropic_client())

        backend = next((b for b in _BACKENDS if lower.startswith(b.prefix)), _OPENAI)
        return OpenAICompatibleModel(name, self._get_http_client(backend))

    @staticmethod
    def _running_loop() -> asyncio.AbstractEventLoop | None:
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            return None

    def _get_http_client(self, backend: _Backend) -> httpx.AsyncClient:
        loop = self._running_loop()
        cached = self._http_clients.get(backend.prefix)
        if cached is not None and (loop is None or cached[0] is None or cached[0] is loop):
            return cached[1]
        api_key = os.environ.get(backend.api_key_env)
        if api_key is None:
            raise UserError(
                f"{backend.api_key_env} is not set. Set it to use a {backend.prefix}-* model."
            )
        client = httpx.AsyncClient(
            base_url=backend.base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=600.0,
        )
        self._http_clients[backend.prefix] = (loop, client)
        return client

    def _get_anthropic_client(self) -> AsyncAnthropic:
        loop = self._running_loop()
        cached = self._anthropic_client
        if cached is None or (loop is not None and cached[0] is not None and cached[0] is not loop):
            cached = (loop, AsyncAnthropic())
            self._anthropic_client = cached
        return cached[1]


__all__ = ["DEFAULT_MODEL", "ModelProvider"]
