# Tracing and Hooks

## Tracing

Every `run`/`run_sync`/`run_streamed` call is traced automatically. There is no separate setup
step. A `Trace` is a hierarchical tree of `Span`s: agent, LLM call, tool call, handoff,
guardrail.

```python
run = agent.run_sync("...")
print(run.trace)  # human-readable span tree
```

Traces are persisted to `runa.db` by default. Inspect them from the CLI:

```bash
runa traces list           # most recent traces
runa traces errors         # most recent traces that errored
runa traces show TRACE_ID  # one trace's full span tree
```

Or visually, with `runa ui`. See [CLI Reference](cli.md#runa-ui).

### Privacy Policy

`observe()` configures what tracing captures, globally or for a block:

```python
from runa import observe

observe(capture_inputs=False)  # applies immediately, stays applied

with observe(redact=["password", "ssn"]):
    agent.run_sync(...)  # redacted within this block only
```

Options: `capture_inputs`/`capture_outputs` (whether to keep them at all), `redact` (a list of
dict keys to scrub) or a custom `redactor` callable, and `max_input_bytes`/`max_output_bytes`/
`max_tool_result_bytes` to truncate what's kept.

### Custom Exporters

By default, traces go to `runa.db` via `SQLiteExporter`. `add_exporter` adds one more exporter
alongside whatever's active, without disabling the default:

```python
from runa import add_exporter, ConsoleExporter

add_exporter(ConsoleExporter())  # runa.db still gets written to, console output is now added
```

`observe(exporter=...)` is a full override instead, since it's also how `with observe(...):`
swaps exporters for one block's duration:

```python
from runa import ConsoleExporter, SQLiteExporter, observe

observe(exporter=[SQLiteExporter(), ConsoleExporter()])  # replaces the exporter list entirely
```

Write your own by implementing `TraceExporter`'s single method, `export(self, trace: Trace) -> None`.

### Langfuse

Install the `runa[langfuse]` extra and set `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` (a `.env`
file works, same as `OPENAI_API_KEY` and friends), and every trace starts going to that Langfuse
project too, `runa.db` included -- no code change:

```bash
uv add "runa-ai[langfuse]"
echo "LANGFUSE_PUBLIC_KEY=pk-lf-..." >> .env
echo "LANGFUSE_SECRET_KEY=sk-lf-..." >> .env
```

Set `LANGFUSE_HOST` too for a non-default Langfuse region or a self-hosted instance. Without
`LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` set, or without the `langfuse` extra installed,
tracing behaves exactly as if `LangfuseExporter` didn't exist -- SQLite only.

To point at a Langfuse project without using the environment (e.g. a second project, or one
picked at runtime), construct `LangfuseExporter` directly instead and add it with `add_exporter`:

```python
from runa import add_exporter
from runa.tracing.langfuse import LangfuseExporter

add_exporter(LangfuseExporter(public_key="pk-lf-...", secret_key="sk-lf-..."))
```

`LangfuseExporter` posts plain OpenTelemetry spans straight to Langfuse's OTLP endpoint rather
than using the `langfuse` package: Runa exports a trace's whole span tree at once, after the run
finishes, and Langfuse's own SDK can only backdate an observation's end time, not its start time
-- it would otherwise record every span as starting at export time with an end time already in
the past.

### Example

```python
--8<--"examples/14_tracing/inspect_trace.py"
```

```python
--8<--"examples/14_tracing/custom_exporter.py"
```

## Hooks

Tracing is unconditional. Hooks are optional lifecycle callbacks for your own logic: logging,
metrics, side effects.

`RunHooks` is passed per-call and fires for every agent involved in a run, including subagents:

```python
from runa import RunHooks


class MyHooks(RunHooks):
    async def on_tool_end(self, context, agent, tool, result):
        print(f"{tool.name} -> {result!r}")


agent.run_sync("...", hooks=MyHooks())
```

`RunHooks` fires `on_agent_start`, `on_agent_end`, `on_handoff`, `on_tool_start`, `on_tool_end`,
`on_llm_start`, and `on_llm_end`. Every method is a no-op unless overridden.

`AgentHooks` is scoped to a single `Agent` subclass instead, via its `hooks` class attribute. It
fires only for that agent, not for the whole run:

```python
class SupportAgent(Agent):
    name = "support_agent"
    hooks = MyAgentHooks()
```

`AgentHooks` fires `on_start`, `on_end`, `on_handoff`, `on_tool_start`, `on_tool_end`,
`on_llm_start`, and `on_llm_end`.

`LoggingRunHooks`/`LoggingAgentHooks` are the framework's defaults, logging each event through
the standard `logging` module under the `"runa"` logger name. Don't subclass them to add
behavior; subclass `RunHooks`/`AgentHooks` directly and pass your own instance instead.

### Example

```python
--8<--"examples/11_hooks/run_hooks.py"
```

```python
--8<--"examples/11_hooks/agent_hooks.py"
```
