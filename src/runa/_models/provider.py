"""provider.py: `ModelProvider`, routing a model name to one of two backends by its prefix."""

import difflib
from dataclasses import dataclass
from typing import TYPE_CHECKING

from runa._loop import LoopCache
from runa._models.interface import Model
from runa.exceptions import UserError

if TYPE_CHECKING:
    from anthropic import AsyncAnthropic

DEFAULT_MODEL = "gpt-5.4-nano"


@dataclass(frozen=True)
class Endpoint:
    """One chat-completions-shaped provider: whose names are its, where it lives, where its key is.

    `prefixes` is every model-name prefix the provider answers to. Usually one, but OpenAI's
    reasoning and codex families aren't spelled `gpt-*`, so it takes several.
    """

    prefixes: tuple[str, ...]
    base_url: str
    api_key_env: str


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

    The set is closed: a model name picks the vendor and the credential, so a prefix that
    matches nothing has no safe default (`"claud-sonnet-5"` must not reach another vendor under
    another key). `Agent.__init__` and `get_model` both call this. An endpoint Runa doesn't name
    is reached by passing a `Model` instance instead of a name.
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


class ModelProvider:
    """Routes a model name to one of two backends by its prefix.

    A name starting with `claude` goes through `AnthropicModel`; the other prefixes (`gpt-*` and
    OpenAI's other families, `gemini-*`, `llama-*`, `deepseek-*`, `qwen-*`) go through
    `OpenAICompatibleModel` against that provider's own chat-completions endpoint. A name
    matching none of them is a `UserError`, not a default (see `check_model_name`).

    Each backend is imported inside the branch that routes to it, so importing `runa` loads no
    vendor SDK.
    """

    def __init__(self) -> None:
        """Start with no Anthropic client; it is created lazily, on first use, and then reused.

        A client's connections are bound to the event loop that first uses it, so it is held per
        loop by a `LoopCache`. Every other provider's client is `runa._models.transport`'s, resolved
        by `OpenAICompatibleModel` on each request.
        """
        self._anthropic_clients: LoopCache[str, AsyncAnthropic] = LoopCache()

    def get_model(self, model_name: str | None) -> Model:
        """Return the `Model` for `model_name` (or Runa's own default, if `None`)."""
        name = DEFAULT_MODEL if model_name is None else model_name
        check_model_name(name)
        lower = name.lower()

        if lower.startswith(_ANTHROPIC_PREFIX):
            from runa._models.anthropic import AnthropicModel

            return AnthropicModel(name, self._get_anthropic_client())

        from runa._models.chat_completions import OpenAICompatibleModel

        # Total: `_PREFIXES` is these endpoints' prefixes plus Anthropic's, both handled above.
        endpoint = next(e for e in _ENDPOINTS if lower.startswith(e.prefixes))
        return OpenAICompatibleModel(
            name, base_url=endpoint.base_url, api_key_env=endpoint.api_key_env
        )

    def _get_anthropic_client(self) -> AsyncAnthropic:
        from anthropic import AsyncAnthropic

        return self._anthropic_clients.get(_ANTHROPIC_PREFIX, AsyncAnthropic)


__all__ = ["DEFAULT_MODEL", "OPENAI", "Endpoint", "ModelProvider", "check_model_name"]
