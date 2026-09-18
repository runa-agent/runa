# Model Providers

`model` is a plain string on your `Agent`:

```python
class SupportAgent(Agent):
    name = "support_agent"
    model = "claude-sonnet-5"
```

No client to construct, no provider to configure globally. Runa resolves the string to a
provider by its prefix, and reads the matching API key from the environment, typically via
`.env`, loaded by your app's `main.py`.

| Prefix | Provider | Env Var |
|---|---|---|
| `claude` | Anthropic | `ANTHROPIC_API_KEY` |
| `gpt` | OpenAI | `OPENAI_API_KEY` |
| `gemini` | Google | `GEMINI_API_KEY` |
| `llama` | Meta | `LLAMA_API_KEY` |
| `deepseek` | DeepSeek | `DEEPSEEK_API_KEY` |
| `qwen` | Alibaba | `DASHSCOPE_API_KEY` |

An unrecognized or bare model name falls back to the OpenAI-compatible backend. The default,
when `model` isn't set, is `"gpt-5.4-nano"`.

Every provider except Anthropic speaks the same chat-completions wire format, so they share one
backend. Anthropic's Messages API is shaped differently and gets its own.

## Model Settings

Tune sampling per agent with `model_settings`. `ModelSettings` currently lives at
`runa._types`, not yet re-exported from `runa` itself:

```python
from runa import Agent
from runa._types import ModelSettings


class SupportAgent(Agent):
    name = "support_agent"
    model_settings = ModelSettings(temperature=0.2, max_tokens=500)
```

## Images

`Agent.run`/`run_sync`/`run_streamed` also take a list of strings instead of plain text, for a
multimodal message. Each string is auto-detected by extension: an image (a URL, a `data:image/...`
URI, or a local file path, base64-encoded automatically) or plain text otherwise.

```python
agent.run_sync(["What's in this image?", "https://example.com/cat.png"])
agent.run_sync(["What's in this image?", "photo.jpg"])  # a local file works too
```

For a string the extension heuristic can't classify (a signed URL with no file extension, say),
build the part explicitly with `runa.content` and mix it into the list:

```python
from runa import content

agent.run_sync(["What's in this image?", content.image("https://example.com/img?id=42")])
```

Content parts are OpenAI's own `image_url`/`text` shape, so every OpenAI-compatible provider (OpenAI,
Gemini, Llama, DeepSeek, Qwen) takes it as-is; Anthropic gets its own translation to Claude's
`image` content block.

## Mixing Providers

Because `model` is per-agent, a single application can freely mix providers across agents: a
fast, cheap model for routing, a stronger one for the agent that actually answers.

```python
class Router(Agent):
    name = "router"
    model = "gpt-5.4-nano"
    subagents = [SupportAgent.handoff]


class SupportAgent(Agent):
    name = "support_agent"
    model = "claude-opus-5"
```

## Example

```python
--8<--"examples/10_model/multi_provider.py"
```

```python
--8<--"examples/10_model/model_settings.py"
```

More in [`examples/10_model/`](https://github.com/Benybrahim/runa/tree/main/examples/10_model).
