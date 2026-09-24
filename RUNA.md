# RUNA.md

How an application is meant to use each Runa primitive. Runa is
opinionated: for every primitive below there is exactly one sanctioned
shape, not a menu of equivalent options. Where the rule is enforced in
code (a raised error, not just a docs recommendation), that's noted:
breaking it isn't a style nit, it's a `TypeError` at runtime.

14 primitives:

1. [Agent](#1-agent) 2. [Tool](#2-tool) 3. [Guardrail](#3-guardrail)
4. [Approval](#4-human-approval) 5. [Subagent](#5-subagent-handoffdelegate)
6. [Session](#6-session) 7. [Memory](#7-memory) 8. [Knowledge](#8-knowledge)
9. [MCP Server](#9-mcp-server) 10. [Model](#10-model) 11. [Hooks](#11-hooks)
12. [Test](#12-test) 13. [Eval](#13-eval-casedataset) 14. [Tracing](#14-tracing)

## 1. Agent

**Always a subclass, never instantiated directly.**

```python
class SupportAgent(Agent):
    name = "support_agent"
    model = "claude-sonnet-5"
    tools = [...]
    guardrails = [...]
```

`instructions` is loaded by default from `app/prompts/support_agent.md`.

**An agent is run only through its own methods**: `run`, `run_sync` or `run_streamed`, each
returning (or ending with) a `Run`: `.output`, `.status` (`"completed"`, `"paused"` or
`"error"`), `.interruptions`, `.trace`, `.usage`, `.error`, and the guardrail audit trail. Errors never raise out of a run;
they come back as `status="error"`. Two more class attributes cover what a run needs:

* `output_type`: a dataclass, Pydantic model or `TypedDict` the final answer is parsed into.
  The model is asked for that JSON schema (on every provider) and a mismatch is an error `Run`.
* `max_turns` (default 10): how many model calls one run may make.

## 2. Tool

**Always a plain function wrapped in `@tool`**

```python
@tool
def get_weather(city: str) -> str:
    """Return the current weather for a city.
    
    city: city name
    """
```

## 3. Guardrail

**Always a plain function wrapped in `@guardrail`**, living in `app/guardrails/`
(`runa generate guardrail`). Can be used both for agents and tools.

For:
- input: `guardrail.input` or `guardrail.i`
- output: `guardrail.output` or `guardrail.o`

```python
@guardrail
def block_empty(x: str) -> bool:
    """Trip when the user sends an empty message."""
    return not x.strip()


# agent guardrail
class MyAgent(Agent):
    guardrails = [block_empty.input]


# tool guardrail
@tool(guardrails=[block_empty.input])
def get_weather(city: str) -> str: ...
```

**IMPORTANT**: Guardrails are applied in list order, the first one to trip stops the rest.

## 4. Human Approval

Tool use can need human approval:
```python
@tool(needs_approval=True)
def issue_refund(amount: float) -> str: ...
```

A function wrapped in `@approval` can also be used, to check if human approval is needed or no

```python
@approval
def large_refund(amount: float) -> bool:
    """Refunds of $50 or more need a human to sign off."""
    return amount >= 50


@tool(needs_approval=large_refund)
def issue_refund(amount: float) -> str: ...
```

**Pausing and resuming.** A call that needs approval doesn't run: it pauses the run, returned
as `status="paused"` with `run.interruptions`. Resolve each one against `run.to_state()`, then
resume by passing that `RunState` back into `run`/`run_sync`/`run_streamed` in place of a
message. A session-backed run saves nothing while paused: pass the same `session=` when
resuming, and the whole turn is saved once the run finishes.

```python
run = agent.run_sync("issue a $75 refund")
while run.status == "paused":
    state = run.to_state()
    for item in run.interruptions:
        state.approve(item)  # or state.reject(item, rejection_message="not authorized")
    run = agent.run_sync(state)
```

`approve`/`reject` also take `always=True`: the decision sticks for every future call to that
tool (by name, not just this one call id), such as an operator answering "always" instead of "yes" in
`runa chat`'s `[y/N/a]` prompt, or the same choice made programmatically. A rejection can carry
a custom `rejection_message`, fed back to the model instead of the default text; with
`always=True` it's reused for every later rejected call to that tool too. A `call_id` that
already executed once can't be submitted again: resuming the same `RunState` twice raises
`DuplicateToolCallError` rather than silently re-running the tool.

`run_streamed` pauses the same way: once the stream ends, its `.run` is the paused `Run`,
resumed with `agent.run_streamed(state)` (or `run`/`run_sync`).

**Durability.** `RunState` survives a process restart: `state.to_json()`/`.to_string()`
serialize it (as a plain dict, or a JSON string); `RunState.from_json(agent, blob)`/
`.from_string(agent, blob)` rebuild it, given a fresh instance of the agent the run started
with (used to re-resolve the current agent and each pending tool by name; a live `Agent`
instance and a tool's closure can't round-trip through JSON themselves). A `context` that was a
dataclass comes back as a plain dict, not its original class; the run's guardrail-result audit
trail (see below) and trace spans aren't included in the serialized blob: spans are already
durably persisted separately (see [Tracing](#14-tracing)). An unrecognized `schema_version`
raises `UserError` rather than resuming from a blob a different, incompatible version of Runa
produced.

**Guardrail audit trail.** Every guardrail that ran (tripped or not, a delegate's included) is
listed on the `Run`, whatever its status, as `input_guardrail_results`/`.output_guardrail_results`/
`.tool_input_guardrail_results`/`.tool_output_guardrail_results`, not just whichever one
stopped the run. A paused `RunState` carries the same four; each is also a span in `run.trace`.

## 5. Subagent (handoff/delegate)

```python
class MyAgent(Agent):
    subagents = [MyAgent2.handoff, MyAgent3.delegate, Agent4]
```

* `.handoff/.h`: the subagent takes over the run entirely.
* `.delegate/.d`: the subagent does the task and returns its result to the
  main agent, like a tool call: the main agent stays in control.
* `None`: main agent will choose automatically to handoff or delegate

A `.delegate` call shares the caller's [approval](#4-human-approval) ledger and usage
accounting: a sticky (`always=True`) decision on the caller's side already covers a matching
tool the delegate calls. A delegate that pauses for approval pauses its caller too: the
delegate's calls appear in the caller's `run.interruptions`, and resuming the caller resumes the
delegate where it stopped (also across `to_json`/`from_json`).

**Subagents are for when the model picks the order.** When your code picks it (always A, then
B and C together), write a plain `async` function. Don't build a workflow class:

```python
async def onboard(ticket: str) -> Run:
    triage = await TriageAgent().run(ticket)
    billing, docs = await asyncio.gather(
        BillingAgent().run(triage.output), DocsAgent().run(triage.output)
    )
    return await SummaryAgent().run(f"{billing.output}\n{docs.output}")
```

Parallelism the model asks for is already handled: delegates are tools, and tool calls from
one model message run concurrently unless `ModelSettings(parallel_tool_calls=False)`.


## 6. Session

**Default to no session (`agent.history`, in-memory, instance-scoped).
Reach for `SQLiteSession` only when a conversation must survive past the
`Agent` instance**: a new process, a different request, a resumed CLI
chat.

```python
session = SQLiteSession("user-42")
agent.run_sync("...", session=session)
```

With a `session`, only the new message is ever passed to `run`/`run_sync`,
prior turns come back from `runa.db` automatically, and `agent.history`
is left untouched. Don't mix the two: pick session-backed or
in-memory per agent instance, not both for the same conversation.

**`message` is one turn, never a transcript** (enforced: a list of
`{"role": ..., "content": ...}` messages raises `TypeError`). Load an
earlier conversation through the seam the chosen mode already has:
`agent.history = [...]` in memory, `await session.add_items([...])`
before the first run with a session.
`SQLiteSession` is the only session implementation Runa ships; a custom
store subclasses `SessionABC`'s four methods, nothing less.

## 7. Memory

**Opt in via `Agent(memory=...)`, one of `"auto"`, `"llm"`, a `Memory(...)`
instance, or `None` (the default).**

```python
class SupportAgent(Agent):
    name = "support_agent"
    memory = "auto"
```

* `"auto"`: the run retrieves relevant memories before each run and
  persists new ones after, with no manual `.search`/`.remember` calls.
* `"llm"`: the model gets a `search_memory` tool and decides itself when
  to call it; nothing is written automatically.
* `None`: the agent behaves exactly as if `runa.memory` didn't exist.

`"llm"` always uses a default `Memory()`; pass your own instance for
`"auto"` mode if you need a non-default `db_path`/`model`/`store`.
`Memory` is durable, semantic, `user_id`-scoped facts backed by the same
`db/runa.db` (`sqlite-vec`) `SQLiteSession` uses: `remember`/`search`/
`forget`, nothing lower. This is long-term memory *across* conversations;
for one conversation's own turns, see [Session](#6-session).

## 8. Knowledge

**Opt in via `Agent(knowledge=...)`, one of `"auto"`, `"llm"`, a `Knowledge(...)` instance, or
`None` (the default) -- the same four shapes as [Memory](#7-memory).**

```python
class SupportAgent(Agent):
    name = "support_agent"
    knowledge = "auto"
```

* `"auto"` (or a `Knowledge(...)` instance): the run searches it before every turn, injecting
  matches as a labeled block -- no manual `.search` calls.
* `"llm"`: the model gets a `search_knowledge` tool and decides itself when to call it.
* `None`: the agent behaves exactly as if `runa.knowledge` didn't exist.

`"llm"` always uses a default `Knowledge()`; pass your own instance for `"auto"` mode if you need
a non-default `directory`/`db_path`/`model`/`store`. `Knowledge()` means `app/knowledge/`,
`db/runa.db` (`sqlite-vec`), OpenAI's `text-embedding-3-small` -- discovery, chunking, embeddings,
and vector storage are entirely internal; put Markdown/PDF/text/CSV files under `app/knowledge/`
and no manual `.ingest()` call is needed either (`.search` ingests lazily on first use). Pass
`Knowledge("some/other/path")` for a non-default source directory.

`Knowledge` is application-scoped, not `user_id`-scoped, and its source of truth is a directory
of files, not calls to `.remember` -- do not conflate it with [Memory](#7-memory):

```text
Session   = conversation history
Memory    = durable user/agent facts
Knowledge = application/domain information
```

## 9. MCP Server

**Always built as `MCPServer(...).http(...)` or `MCPServer(...).stdio(...)`,
listed in `mcp=`/`mcp_servers=`, never `MCPServerStdio(...)`/
`MCPServerStreamableHttp(...)` directly** unless you're the one
implementing a third transport.

```python
files = MCPServer(name="files").stdio("npx", ["-y", "@modelcontextprotocol/server-filesystem", "."])
search = MCPServer(name="search").http("https://example.com/mcp")

mcp = [files, search]
```

An MCP server's tools are exposed to the model exactly like `@tool`
functions. Never branch application code on whether a tool came from MCP
or from `@tool`; the framework already erases that distinction. The
connection opens lazily and lives for the agent's whole lifetime, so
there's no explicit `.connect()`/`.close()` for user code to call in the
common case.

## 10. Model

**Always a plain string on `model`, never a constructed client.** The
string's prefix (`claude-`, `gpt-`, `gemini-`, `llama-`, `deepseek-`,
`qwen-`) picks the provider and its API key env var; nothing is
configured globally.

```python
model = "claude-sonnet-5"
```

Mixing providers across agents in one app (a cheap router, a stronger
answerer) is expected, not an edge case: `model` is per-agent by design.
Tune sampling with `model_settings = ModelSettings(...)`, never
provider-specific kwargs on `model` itself.

## 11. Hooks

**Override only the lifecycle methods you need; every other method stays
a no-op.** Two scopes, chosen by what the callback should see, not by
which one is "newer":

* `RunHooks`, passed as `Agent.run(hooks=...)`, fires for every agent in
  that run, including ones reached by handoff/delegate. Use this for
  cross-cutting concerns (metrics, a single audit log for the whole run).
* `AgentHooks`, assigned to an `Agent` subclass's `hooks` attribute,
  fires only for that one agent. Use this for a concern that belongs to
  one agent's identity, not the run as a whole.

Don't subclass `LoggingRunHooks`/`LoggingAgentHooks` to add behavior;
subclass `RunHooks`/`AgentHooks` directly and pass your own: the
`Logging*` classes are the framework's default, not a base to build on.

## 12. Test

**A test is a bare `test_*` function using plain `assert`, no pytest, no
custom assertion helpers.** `runa test` is its own small runner, not a
pytest wrapper, specifically so a generated app needs no test framework
as a dependency.

```python
def test_answers_politely():
    run = SupportAgent().run_sync("Hi")
    assert run.status == "completed"
```

`async def test_*` is awaited automatically, so don't wrap async tests in
`asyncio.run` yourself.

## 13. Eval (Case/Dataset)

**An eval is `evals/<agent_name>.jsonl`, one `Case` per line**; the
filename is the Agent's declared `name`, so `runa eval` resolves the agent
from it with no registration. `runa generate agent` creates it alongside
every agent. Reach for a Python module under `evals/`
(module-level `agent` and `dataset`) only when cases or the agent need
code; a `.jsonl` sharing that module's stem is its data, not a second eval.

```json
{"input": "Where's my order #4821?", "expected": "Asks for or looks up the order status"}
"Hi there"
```

Grow a dataset from real failures, not invented ones: `runa eval --add
TRACE_ID` (or "Add to evals" in `runa ui`) appends a traced run's input.
Each `runa eval` compares against the previous run and flags regressions.

Only `Case.input` is required; add `expected`/`expected_tool`/`context`
only for the specific grading signal each enables, don't fill in fields
a case doesn't need "for completeness." The judge model defaults to the
agent's own `model`; override it with `judge=` only when a cheaper/
different model should grade instead of the agent's own.

## 14. Tracing

**Never configured, never opted into: every `Agent.run`/`run_sync` is
traced automatically**, and `Run.trace` is always populated. Reach for
manual `trace`/`span` (`runa.tracing`) only to group work: a workflow, a
batch job, a pre/post-processing step. An `Agent.run()` inside a `trace`
block still produces its own `Trace`, stamped with the block's id as
`group_id`. There is no `group_id=` argument on `run`.

```python
with tracing.trace("nightly-batch") as t:
    with tracing.span("step-1", t):
        ...
```

Add a `TraceExporter` (`ConsoleExporter`, `SQLiteExporter`, or your own)
to change *where* traces go; never change *whether* they're captured.
There is no flag for that.

---

When a change would let a user reach the same result through a second
shape (a new `Handoff(...)` call site, a hand-built tool schema, a
`RunHooks` subclass that overrides `Logging*`), that's a sign to close the
second path, not to document it alongside the first.
