"""`runa._models`: Runa's model provider and its two backends.

`ModelProvider.get_model` maps a model name's prefix to a backend. A name starting with `claude`
goes through `AnthropicModel`, talking to Anthropic's own SDK directly, Anthropic's Messages API
isn't chat-completions-shaped, so it's the one backend that needs real translation. The other
prefixes (`gpt-*`, `gemini-*`, `llama-*`, `deepseek-*`, `qwen-*`) go through
`OpenAICompatibleModel`, which speaks the chat-completions wire format that OpenAI, Gemini, Llama,
DeepSeek, and Qwen all share, over plain HTTP via `httpx2`, not the `openai` package. The set of
prefixes is closed: a name matching none of them is a `UserError` (`check_model_name`), not an
unnamed provider's model quietly resolved against OpenAI's endpoint.

Conversation items are plain chat-completions-shaped message dicts everywhere in Runa (see
`runa._items`, which owns that shape and every reader of it); `AnthropicModel` is the only place
that ever converts away from it. Every callable the model is offered arrives as a `ToolSchema`
(a name, a description, a JSON schema for its arguments), which belongs to neither provider's wire
format; whether a run treats a call as a tool, a delegate or a handoff is not this package's
concern.

A turn is one `ModelRequest`, so adding something a request carries doesn't re-thread a parameter
through every backend. Split by concern: `interface` (the `Model` protocol, `ModelRequest`,
`StreamDelta`, and what both backends share), `chat_completions` (the chat-completions backend),
`anthropic` (the Claude backend), `provider` (`ModelProvider`, routing a name to one of the
two), and `transport` (the HTTP client and retry policy that every request not made through
Anthropic's SDK goes through, `runa.embeddings` included).

The two backends are resolved one name at a time, the way `runa`'s own facade resolves the
public surface: a package's `__init__` runs before any of its submodules, so naming them here
eagerly would make the Anthropic SDK -- most of the cost of importing a provider -- a load-time
dependency of everything that reads `DEFAULT_MODEL` or `check_model_name`, which
is `runa.agent` and so every entry point. `ModelProvider.get_model` imports the backend it
routes to; nothing else needs either.
"""

from typing import TYPE_CHECKING

from runa._models.interface import Model, ModelRequest, StreamDelta, ToolSchema
from runa._models.provider import DEFAULT_MODEL, OPENAI, ModelProvider, check_model_name

if TYPE_CHECKING:
    from runa._models.anthropic import AnthropicModel
    from runa._models.chat_completions import OpenAICompatibleModel

_BACKENDS = {
    "AnthropicModel": "runa._models.anthropic",
    "OpenAICompatibleModel": "runa._models.chat_completions",
}
"""The lazily resolved backends, mapped to the module that defines each one: every backend is
imported by whoever routes to it."""

__all__ = [
    "DEFAULT_MODEL",
    "OPENAI",
    "AnthropicModel",
    "Model",
    "ModelProvider",
    "ModelRequest",
    "OpenAICompatibleModel",
    "StreamDelta",
    "ToolSchema",
    "check_model_name",
]


def __getattr__(name: str) -> object:
    """Resolve a backend's name, importing only the module that defines it."""
    from importlib import import_module

    try:
        module = _BACKENDS[name]
    except KeyError:
        raise AttributeError(f"module 'runa._models' has no attribute {name!r}") from None
    value = getattr(import_module(module), name)
    globals()[name] = value
    return value
