# Agents

An `Agent` is a Python class. Its class attributes are its entire configuration. There is no
separate config object to build.

```python
from runa import Agent


class SupportAgent(Agent):
    name = "support_agent"
    instructions = "You help customers troubleshoot their orders."
    model = "claude-sonnet-5"
```

## Attributes

| Attribute | Default | Meaning |
|---|---|---|
| `name` | required | The agent's identity. Traces, handoffs, and `runa chat` all key off it. |
| `instructions` | `None` | A system prompt string, or a `(context) -> str` callable. |
| `model` | `"gpt-5.4-nano"` | Which model to call. See [Model Providers](models.md). |
| `model_settings` | `ModelSettings()` | Temperature, max tokens, and other sampling options. |
| `tools` | `[]` | `@tool`-decorated functions. See [Tools](tools.md). |
| `subagents` | `[]` | Other agents to hand off or delegate to. See [Subagents](subagents.md). |
| `guardrails` | `[]` | Input/output checks. See [Guardrails](guardrails.md). |
| `mcp` / `mcp_servers` | `[]` | MCP servers whose tools this agent can call. See [MCP Servers](mcp.md). |
| `memory` | `None` | Long-term memory across conversations. See [Memory](memory.md). |
| `knowledge` | `None` | Retrieval from application documents. See [Knowledge](knowledge.md). |
| `output_type` | `None` | A dataclass, Pydantic model or `TypedDict` the final output is parsed into. |
| `max_turns` | `10` | How many model calls one run may make before it errors. |
| `hooks` | `None` | An `AgentHooks` instance scoped to this agent. See [Tracing and Hooks](tracing.md). |
| `compact` | `False` | Keep long-running history from growing without bound. See below. |

`name` is the only required attribute. Everything else has a sane default, in the spirit of
convention over configuration.

## Instructions as a Function

`instructions` can be a plain string, or a function of one argument, the `context` passed to
`run`/`run_sync`. It is resolved fresh on every call:

```python
from dataclasses import dataclass


@dataclass
class Context:
    user_name: str


def instructions(context: Context) -> str:
    return f"You are a friendly assistant helping {context.user_name}."


class Assistant(Agent):
    name = "assistant"
    instructions = instructions


Assistant().run_sync("Hi", context=Context(user_name="Ada"))
```

`context` is never sent to the model. It is plumbed through to `instructions`, tools, and
guardrails as is, so it is the place to put per-run data (a user id, a tenant, a request-scoped
client) without reaching for global state.

## Running an Agent

```python
agent = SupportAgent()
run = agent.run_sync("My order hasn't arrived.")
print(run.output)
```

`run_sync`/`run`/`run_streamed` all append the turn to `agent.history`, so the next call on the
same instance continues the conversation. Pass `session=` instead to persist history to
`runa.db`. See [Sessions and Chat](sessions.md).

`run`/`run_sync` return a `Run`:

* `run.output`: the final output (an `output_type` instance, when set), or `None` unless completed
* `run.status`: `"completed"`, `"paused"` (a tool call awaits [approval](approval.md)) or `"error"`
* `run.interruptions`: the calls awaiting approval, when `status == "paused"`
* `run.error`: the error message, when `status == "error"`
* `run.usage`: this call's token usage
* `run.trace`: the full span tree for this call. See [Tracing and Hooks](tracing.md).
* `run.input_guardrail_results` (and `output_`, `tool_input_`, `tool_output_`): every guardrail
  that ran. See [Guardrails](guardrails.md#audit-trail).

A tripped guardrail or a runtime error (`MaxTurnsExceeded`, a model error) is caught and reported
as `status="error"` instead of being raised. `agent.history` is left unchanged, since the turn
never completed.

`run_streamed` instead yields `StreamEvent`s as the model responds. It is the same run as `run`:
guardrails, hooks, tracing, memory, approvals and `session=` all apply. Once the stream ends,
`stream.run` holds the same `Run` that `run` would have returned, errors included, and
`agent.history` is updated:

```python
stream = agent.run_streamed("My order hasn't arrived.")
async for event in stream:
    ...
print(stream.run.output)
```

## Structured Output

Set `output_type` and `run.output` is an instance of it. The model is asked for the matching
JSON schema (on Claude, OpenAI and every chat-completions provider), and an answer that still
doesn't fit comes back as `status="error"`:

```python
@dataclass
class Ticket:
    category: str
    urgent: bool


class TriageAgent(Agent):
    name = "triage_agent"
    output_type = Ticket


run = TriageAgent().run_sync("My card was charged twice!")
run.output.urgent  # True
```

## Usage

```python
agent.last_usage  # tokens used by the most recent call
agent.usage  # accumulated across every call on this instance
```

Both are populated whether or not the run succeeded, used a `session`, or passed `hooks`.

## Keeping History From Growing

A long-running chat can push `history`/`session` past a model's context window. Set `compact`
to trim it automatically:

```python
class SupportAgent(Agent):
    name = "support_agent"
    compact = True
```

`compact = True` uses Runa's default compactor: once history passes a token threshold, it keeps
only the most recent exchange. It is a rolling window, not a summary. Pass your own
`Compactor` (any `(items, usage_tokens) -> items | None` callable) for a different strategy, such
as an LLM-written summary. `compact = False`, the default, turns this off.

## Visualizing an Agent

`agent.graph` renders the agent, its tools, and its subagents as a Graphviz diagram: inline in
Jupyter, or `.render(path)` to save it, or `.source` for the raw DOT text. A delegate is drawn
with a dotted edge, a handoff with a dashed one. Rendering an actual image needs the system `dot`
binary.

## Example

```python
--8<--"examples/01_agent/basic_agent.py"
```

```python
--8<--"examples/01_agent/dynamic_instructions.py"
```

More in [`examples/01_agent/`](https://github.com/Benybrahim/runa/tree/main/examples/01_agent).
