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
| `gpt`, `o1`, `o3`, `o4`, `chatgpt`, `codex` | OpenAI | `OPENAI_API_KEY` |
| `gemini` | Google | `GEMINI_API_KEY` |
| `llama` | Meta | `LLAMA_API_KEY` |
| `deepseek` | DeepSeek | `DEEPSEEK_API_KEY` |
| `qwen` | Alibaba | `DASHSCOPE_API_KEY` |

Those keys cover an agent's turns, which is all `model` decides. An agent that also sets `memory`
or `knowledge` needs `OPENAI_API_KEY` on top of its own provider's key: embedding is the one call
Runa makes that `model` doesn't route (see
[Where embeddings go](deployment.md#where-embeddings-go)).

That list is the whole list. A name matching none of it raises `UserError` when the agent is
constructed, naming the prefix it probably meant, because the alternative is worse than an error:
`model = "claud-sonnet-5"` would otherwise send your prompts to a different company under a
different key, and report the resulting 404 as model misbehaviour mid-run. The default, when
`model` isn't set, is `"gpt-5.4-nano"`.

```pycon
>>> SupportAgent(model="claud-sonnet-5")
UserError: 'claud-sonnet-5' starts with no prefix Runa routes to a provider, so it names no
model. Did you mean a claude-* model? For an endpoint Runa doesn't name, pass a Model instance
instead of a name.
```

For a gateway, a proxy, or something self-hosted (an endpoint Runa has no name for), pass a
`Model` instead of a name. `model` takes any object with the two methods a backend has, so
Runa's own chat-completions backend works, pointed at your `base_url`:

```python
import httpx2 as httpx
from runa._models import OpenAICompatibleModel


class SupportAgent(Agent):
    name = "support_agent"
    model = OpenAICompatibleModel(
        "mixtral-8x7b",
        httpx.AsyncClient(base_url="https://gateway.internal/v1/", headers={...}),
    )
```

This is the sharp knife, and it is deliberately spelled differently from a name: a bare name
can't tell "a model Runa has never heard of" and "I mistyped" apart, so the first one has to say
so out loud.

Every provider except Anthropic speaks the same chat-completions wire format, so they share one
backend. Anthropic's Messages API is shaped differently and gets its own.

## Model Settings

Tune sampling per agent with `model_settings`:

```python
from runa import Agent, ModelSettings


class SupportAgent(Agent):
    name = "support_agent"
    model_settings = ModelSettings(temperature=0.2, max_tokens=500)
```

`max_tokens` here caps the length of one response. To cap what a whole run may spend, use
`Agent.max_tokens`; see [Bounding a run](agents.md#bounding-a-run).

## Reasoning Effort

How hard a reasoning model thinks before answering is `reasoning_effort`, one of three levels:

```python
class ResearchAgent(Agent):
    name = "research_agent"
    model = "claude-opus-5"
    model_settings = ModelSettings(reasoning_effort="high")
```

`low`/`medium`/`high` is the vocabulary both backends share, so the setting keeps meaning the
same thing when `model` changes vendor: a chat-completions provider is sent `reasoning_effort`,
Claude is sent adaptive thinking at that effort. Unset, each provider's own default stands.

Asking a model that can't reason for effort is the provider's error, not a setting Runa drops on
the floor -- a misspelled `model` is loud for the same reason. A provider with a deeper ladder
than three levels is reached by passing a `Model`.

What the thinking cost is on `run.usage.output_tokens_details.reasoning_tokens`, whichever
backend answered. The reasoning itself stays in the conversation: Claude returns its thinking as
content blocks carrying a signature, and those ride along in history, a session and a serialized
`RunState`, so every later turn of a tool-calling loop hands the model its own reasoning back
instead of paying to derive it again. They're the one part of an item no other provider is ever
sent.

## Retries

Both backends retry connection errors and `408`/`409`/`429`/`5xx` twice, with jittered backoff
and honoring a `retry-after` header. Tune it per agent, or turn it off:

```python
class SupportAgent(Agent):
    model_settings = ModelSettings(max_retries=5)  # 0 disables retrying outright
```

## Images

`Agent.run`/`run_sync`/`run_streamed` also take a list instead of plain text, for a multimodal
message. Each item is auto-detected: an image URL or `data:image/...` URI becomes an image, a
`Path` is read off disk and base64-encoded automatically, and any other string is plain text.

```python
from pathlib import Path

agent.run_sync(["What's in this image?", "https://example.com/cat.png"])
agent.run_sync(["What's in this image?", Path("photo.jpg")])  # a local file, read for you
```

A local file is a `Path`, never a bare string: the list often carries a user's own words, and a
string that happens to end in `.png` would otherwise turn their message into a read of your
filesystem. A bare string naming a local image raises `ValueError` and names both spellings that
say the path is yours, `Path(...)` and `content.image(...)`.

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

## Retries

Every provider retries the same failures the same way: connection errors, 408, 409, 429 and 5xx
are retried twice with exponential backoff, honoring a `retry-after` header. A request that still
fails ends the run with `status="error"`, never an uncaught HTTP exception.

## Example

```python
--8<--"examples/10_model/multi_provider.py"
```

```python
--8<--"examples/10_model/model_settings.py"
```

More in [`examples/10_model/`](https://github.com/Benybrahim/runa/tree/main/examples/10_model).
