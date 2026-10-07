"""`runa._models`: Runa's own model provider, two backends, no `openai-agents`, no `openai` SDK.

`ModelProvider.get_model` maps a model name's prefix to a backend. A name starting with `claude`
goes through `AnthropicModel`, talking to Anthropic's own SDK directly, Anthropic's Messages API
isn't chat-completions-shaped, so it's the one backend that needs real translation. Everything else
(`gpt-*`, `gemini-*`, `llama-*`, `deepseek-*`, `qwen-*`, or an unrecognized/bare name) goes through
`OpenAICompatibleModel`, which speaks the chat-completions wire format that OpenAI, Gemini, Llama,
DeepSeek, and Qwen all share, over plain HTTP via `httpx2`, not the `openai` package.

Conversation items are plain chat-completions-shaped message dicts everywhere in Runa (see
`runa._items`, which owns that shape and every reader of it); `AnthropicModel` is the only place
that ever converts away from it. Tools/handoffs are read structurally here
(`.name`/`.description`/`.params_json_schema` for a tool, `.tool_name`/`.tool_description` for a
handoff) rather than importing their concrete types, so this package has no dependency on
`runa.tool`/`runa.handoff`; both arrive at a backend as a `ToolSchema`, which belongs to neither
provider's wire format.

A turn is one `ModelRequest`, so adding something a request carries doesn't re-thread a parameter
through every backend. Split by concern: `interface` (the `Model` protocol, `ModelRequest`,
`StreamDelta`, and what both backends share), `chat_completions` (the chat-completions backend),
`anthropic` (the Claude backend), and `provider` (`ModelProvider`, routing a name to one of the
two).
"""

from runa._models.anthropic import AnthropicModel
from runa._models.chat_completions import OpenAICompatibleModel
from runa._models.interface import Model, ModelRequest, StreamDelta
from runa._models.provider import DEFAULT_MODEL, ModelProvider

__all__ = [
    "DEFAULT_MODEL",
    "AnthropicModel",
    "Model",
    "ModelProvider",
    "ModelRequest",
    "OpenAICompatibleModel",
    "StreamDelta",
]
