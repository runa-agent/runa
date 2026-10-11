"""provider.py: `ModelProvider`, routing a model name to one of two backends by its prefix."""

import difflib
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
class Endpoint:
    """One chat-completions-shaped provider: whose names are its, where it lives, where its key is.

    `prefixes` is every model-name prefix the provider answers to. Usually one, but OpenAI's
    reasoning and codex families aren't spelled `gpt-*`, so it takes several; the first is the
    canonical one, naming the provider in an error message and keying its client.
    """

    prefixes: tuple[str, ...]
    base_url: str
    api_key_env: str

    @property
    def name(self) -> str:
        """What to call this provider when talking to a developer about it."""
        return self.prefixes[0]


_ANTHROPIC_PREFIX = "claude"

# Named, unlike its siblings, because `runa.embeddings` reads it too: OpenAI is the one provider
# here that also serves embeddings, and its address and key variable are written only here.
OPENAI = Endpoint(
    ("gpt", "o1", "o3", "o4", "chatgpt", "codex"), "https://api.openai.com/v1/", "OPENAI_API_KEY"
)

_ENDPOINTS: tuple[Endpoint, ...] = (
    OPENAI,
    Endpoint(
        ("gemini",), "https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY"
    ),
    Endpoint(("llama",), "https://api.llama.com/compat/v1/", "LLAMA_API_KEY"),
    Endpoint(("deepseek",), "https://api.deepseek.com/v1/", "DEEPSEEK_API_KEY"),
    Endpoint(
        ("qwen",), "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/", "DASHSCOPE_API_KEY"
    ),
)
_PREFIXES: tuple[str, ...] = (_ANTHROPIC_PREFIX, *(p for e in _ENDPOINTS for p in e.prefixes))


def check_model_name(name: str) -> None:
    """Raise `UserError` unless `name`'s prefix is one Runa routes to a provider.

    The set is closed, and that is the point. A model name is the only thing that picks the
    vendor, the price, the capability and the credential, so a prefix that matches nothing can't
    be answered with a default: `"claud-sonnet-5"` would send an app's prompts to a different
    company than its author believes, under a different key, and report the resulting 404 as
    model misbehaviour mid-run. Runa already refuses to let a misspelled *setting* name
    (`modell = ...`) configure nothing silently (`agent._reject_unknown_settings`); the typo one
    field over, on the value that selects the provider, gets the same answer.

    `Agent.__init__` calls this at construction, through `_checked_model`, so the common case
    fails before a run and names the attribute. `get_model` calls it too, being the seam a
    `RunConfig`'s own provider can be reached through without an `Agent`. For an endpoint Runa
    doesn't name -- a gateway, a proxy, something self-hosted -- the sanctioned shape is a `Model`
    instance rather than a bare name, which `_checked_model` passes through untouched.
    """
    if name.lower().startswith(_PREFIXES):
        return
    raise UserError(
        f"{name!r} starts with no prefix Runa routes to a provider, so it names no model."
        f"{_prefix_hint(name)} For an endpoint Runa doesn't name, pass a Model instance "
        f"instead of a name."
    )


def _prefix_hint(name: str) -> str:
    """Point a rejected name at the prefix it most likely misspelled, or list them all."""
    close = difflib.get_close_matches(name.lower().split("-")[0], _PREFIXES, n=1)
    if close:
        return f" Did you mean a {close[0]}-* model?"
    return f" Model names start with one of: {', '.join(f'{p}-*' for p in _PREFIXES)}."


def _client_for(endpoint: Endpoint) -> httpx.AsyncClient:
    """An HTTP client pointed at `endpoint`, carrying the key its env var holds.

    Raises `UserError` when that variable is unset, which is why the client is built lazily: an
    app that never asks for a `gemini-*` model should not need `GEMINI_API_KEY`.
    """
    api_key = os.environ.get(endpoint.api_key_env)
    if api_key is None:
        raise UserError(
            f"{endpoint.api_key_env} is not set. Set it to use a {endpoint.name}-* model."
        )
    return httpx.AsyncClient(
        base_url=endpoint.base_url,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=600.0,
    )


class ModelProvider:
    """Routes a model name to one of two backends by its prefix.

    A name starting with `claude` goes through `AnthropicModel`; the other prefixes (`gpt-*` and
    OpenAI's other families, `gemini-*`, `llama-*`, `deepseek-*`, `qwen-*`) go through
    `OpenAICompatibleModel` against that provider's own chat-completions endpoint. A name
    matching none of them is a `UserError`, not a default (see `check_model_name`).
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
        name = DEFAULT_MODEL if model_name is None else model_name
        check_model_name(name)
        lower = name.lower()

        if lower.startswith(_ANTHROPIC_PREFIX):
            return AnthropicModel(name, self._get_anthropic_client())

        # Total: `_PREFIXES` is these endpoints' prefixes plus Anthropic's, both handled above.
        endpoint = next(e for e in _ENDPOINTS if lower.startswith(e.prefixes))
        return OpenAICompatibleModel(name, self._get_http_client(endpoint))

    def _get_http_client(self, endpoint: Endpoint) -> httpx.AsyncClient:
        return self._http_clients.get(endpoint.name, lambda: _client_for(endpoint))

    def _get_anthropic_client(self) -> AsyncAnthropic:
        return self._anthropic_clients.get(_ANTHROPIC_PREFIX, AsyncAnthropic)


__all__ = ["DEFAULT_MODEL", "OPENAI", "Endpoint", "ModelProvider", "check_model_name"]
