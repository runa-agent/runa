"""`runa._models`: Runa's own model provider, two backends, no `openai-agents`, no `openai` SDK.

`ModelProvider.get_model` maps a model name's prefix to a backend. A name starting with `claude`
goes through `AnthropicModel`, talking to Anthropic's own SDK directly, Anthropic's Messages API
isn't chat-completions-shaped, so it's the one backend that needs real translation. Everything else
(`gpt-*`, `gemini-*`, `llama-*`, `deepseek-*`, `qwen-*`, or an unrecognized/bare name) goes through
`OpenAICompatibleModel`, which speaks the chat-completions wire format that OpenAI, Gemini, Llama,
DeepSeek, and Qwen all share, over plain HTTP via `httpx2`, not the `openai` package.

Conversation items are plain chat-completions-shaped message dicts everywhere in Runa (see
`runa._types.TResponseInputItem`); `AnthropicModel` is the only place that ever converts away from
that shape. Tools/handoffs are read structurally here (`.name`/`.description`/`.params_json_schema`
for a tool, `.tool_name`/`.tool_description` for a handoff) rather than importing their concrete
types, so this package has no dependency on `runa.tool`/`runa.handoff`.

Split by concern: `interface` (the `Model` protocol, `StreamDelta`, shared wire-format helpers),
`openai_chatcompletions` (the chat-completions backend), `anthropic` (the Claude backend), and
`multi_provider` (`ModelProvider`, routing a model name to one of the two).
"""

from runa._models.anthropic import AnthropicModel
from runa._models.interface import Model, StreamDelta
from runa._models.multi_provider import DEFAULT_MODEL, ModelProvider
from runa._models.openai_chatcompletions import OpenAICompatibleModel

__all__ = [
    "DEFAULT_MODEL",
    "AnthropicModel",
    "Model",
    "ModelProvider",
    "OpenAICompatibleModel",
    "StreamDelta",
]
