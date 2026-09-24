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

### Grouping Runs

Every `Agent.run()` inside a `tracing.trace` block gets the block's trace id as its `group_id`,
including runs started with `asyncio.gather`. Each run still has its own trace:

```python
from runa import tracing

with tracing.trace("nightly-batch"):
    for ticket in tickets:
        SupportAgent().run_sync(ticket)
```

Outside a block, a session-backed run is grouped by its session id.

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

Install the `runa-ai[langfuse]` extra and set `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` (a `.env`
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

More in [`examples/14_tracing/`](https://github.com/Benybrahim/runa/tree/main/examples/14_tracing).

Every trace above -- SQLite, Langfuse, or a custom exporter -- is also browsable visually with
`runa ui`. See [CLI Reference](cli.md#runa-ui).

## Traces Across Replicas

The default exporter writes to this process's `db/runa.db`, which is per-process by design. Three
replicas keep three disjoint histories and a dashboard that shows one of them. Set one variable
and traces (along with sessions, memory, knowledge and eval history) go to a shared Postgres:

```bash
uv add "runa-ai[postgres]"
export RUNA_POSTGRES_DSN=postgresql://user:password@host:5432/runa
```

No code changes: `list_traces(...)`, `runa traces`, `runa ui` and the exporter all follow that
variable, so a trace written by one replica is readable from any of them.

## Retention

Traces and spans are append-only, so a long-lived deployment's database grows until the disk
does. `runa prune` is the retention pass:

```bash
runa prune --older-than 30 --dry-run   # what would go
runa prune --older-than 30             # traces, sessions and eval runs older than 30 days
```

There is no background thread doing this on its own, on purpose: when to delete your data is your
decision, not the framework's. See [Deployment](deployment.md#retention).

## Logs and the Privacy Policy

The privacy policy above governs the default log output too, not just spans. `LoggingRunHooks`
(the default `hooks`) logs at INFO without ever carrying content: an agent's answer and a tool's
result are user data, and a production app running at INFO should not be writing them to stdout.
Content is logged at DEBUG only, and passes through the same `redact`/`redactor`/`capture_outputs`
settings on its way:

```python
observe(capture_outputs=False)  # no outputs in traces, and none in the DEBUG log lines either
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

More in [`examples/11_hooks/`](https://github.com/Benybrahim/runa/tree/main/examples/11_hooks).
