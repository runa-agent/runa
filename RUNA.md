# RUNA.md

How an application is meant to use each Runa primitive. Runa is
opinionated: for every primitive below there is exactly one sanctioned
shape, not a menu of equivalent options. Where the rule is enforced in
code (a raised error, not just a docs recommendation), that's noted:
breaking it isn't a style nit, it's an error at runtime: a `UserError` for
any mistake in a declared value, and a `TypeError` only where Python itself
raises one from a signature (an unknown keyword argument).

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

**The settings are a closed set.** A name that isn't one of them configures nothing, so it's an
error rather than an agent that quietly runs on the default `model`. Enforced in code for the
declaration and the override alike, each the way its own call site allows: a class attribute is a
`UserError` naming the setting it most likely meant, and a constructor keyword is a `TypeError`
from the signature itself, every setting being a keyword-only, annotated parameter of
`__init__` -- so a type checker and an editor see them too. Only data is checked in a class body:
methods, properties and `_`-prefixed attributes are the subclass's own business, as are a mixin's
attributes. Passing a setting beats declaring it, and passing `None` means `None`
(`SupportAgent(memory=None)` turns a declared `memory = "auto"` off for that instance).

**A run reads the settings as resolved when the agent was built.** Reassigning one on an instance
(`agent.model = ...`) resolves it again, so it reaches the next run and a bad value is a
`UserError` at that line. The list settings are stored as tuples, so they change by
reassignment, never in place.

**An agent is run only through its own methods**: `run`, `run_sync` or `run_streamed`, each
returning (or ending with) a `Run`: `.output`, `.status` (`"completed"`, `"paused"` or
`"error"`), `.interruptions`, `.trace`, `.usage`, `.error`, `.tool_calls`, and the guardrail audit trail. Errors never raise out of a run;
they come back as `status="error"`. The boundary decides that, not the exception's type: a
tripped guardrail, a model error and a bug in an app's own `Hooks`/`Session`/`Compactor` alike
(the last wrapped in `UnexpectedError`, its traceback logged), each with the trace around the
failure. The two that still raise are `asyncio.CancelledError`, because a caller cancelling a run
asked for it to stop rather than to fail, and `OperatorError`, which isn't a run's failure at all.
Four more class attributes cover what a run needs:

* `output_type`: a dataclass, Pydantic model or `TypedDict` the final answer is parsed into.
  The model is asked for that JSON schema (on every provider) and a mismatch is an error `Run`.
* `max_turns` (default 10): how many model calls one run may make.
* `max_tokens` (default `None`): how many tokens one run may spend, in total. Not the same
  knob as `ModelSettings(max_tokens=...)`, which caps one response's length.
* `timeout` (default `None`): how many wall-clock seconds one run may take.

A run's `.usage`, `max_tokens` and trace count every call made on its behalf, not just the turn
loop's: [memory](#7-memory) and [knowledge](#8-knowledge) embeddings (a lazy corpus ingest
included) and memory's post-answer extraction call are each an `"llm"` span with its usage. The
budget is checked after each of the loop's model calls, so the extraction call, made once the
answer is in, is counted but never fails a finished run.

**One instance runs one conversation at a time.** An `Agent` holds `self.history`, so two
overlapping session-less runs on the same instance raise `UserError` rather than interleaving.
Pass a `session` per run, or use an `Agent` per run. Enforced in code.

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
`always=True` it's reused for every later rejected call to that tool too. **A pause resumes
once**: resuming it a second time, from any copy of it in any process, is an error `Run`
(`RunAlreadyResumed`) rather than a tool run twice. Enforced in code. A state carries the `context`
and `user_id` of the run it paused, so passing either alongside one raises `UserError` instead of
being dropped: pass them to the first run.

`run_streamed` pauses the same way: once the stream ends, its `.run` is the paused `Run`,
resumed with `agent.run_streamed(state)` (or `run`/`run_sync`).

**Durability.** A pause is kept in `runa.db` the moment it happens, under `run.run_id`, so it
survives a process restart with nothing to save by hand. Another process (a worker, a later
request, a Slack bot) gets it back with `await RunState.load(agent, run_id)` (`load_sync` beside
`run_sync`), given a fresh instance of the agent the run started with, decides it, and resumes
it the same way. That is the one cross-process shape: there is no blob to hand out, because a
copy of a pause is a second chance to run it. `load` returns `None` for a pause that doesn't
exist or already resumed, and spends nothing: the resumed run claims the pause, atomically, so of
two processes resuming one pause exactly one runs it.

```python
state = RunState.load_sync(SupportAgent(), run_id)
for item in state.pending:
    state.approve(item)
run = SupportAgent().run_sync(state)
```

Everything the run carries comes back: its conversation, `context` and user, usage, the tool
calls it already made, the model calls already spent against its `max_turns`, and the time
already spent against its `timeout` (time paused doesn't count). After a handoff the run is
still the starting agent's: its ceilings, memory, knowledge and compaction stay in force. Loading resolves no tools, so it never connects to an agent's MCP servers: the resumed
run looks each pending call's tool up itself. A `context` that was a dataclass comes back as a
plain dict, not its original class; the run's guardrail-result audit trail (see below) and trace
spans aren't stored with the pause: spans are already durably persisted separately (see
[Tracing](#14-tracing)). A pause stored by an incompatible version of Runa raises `UserError`
on `load` rather than resuming. A pause nobody resumes stays in the `paused_runs` table until an
operator prunes it by `created_at`.

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
delegate where it stopped (also from `RunState.load`). A delegate's run is part of its
caller's: the run's [hooks](#11-hooks) fire for it, and its spans are in the caller's trace, under
the delegate call. A streamed caller shows that call and its output, not the delegate's tokens.
The call is one of the caller's `run.tool_calls`, the delegate's own calls as its `.children`.

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
Pass `session="<id>"` only when a conversation must survive past the
`Agent` instance**: a new process, a different request, a resumed CLI
chat.

```python
agent.run_sync("...", session="user-42")
```

**The id is the application's; the backend is `runa.db`'s.** A string is
resolved through `db.session(...)`, so one call persists to a local
file, a shared Postgres, or nothing at all, according to
`RUNA_DATABASE_URL` -- a backend is never named at the call site. Pass a
`Session` instead only when a run needs a specific store:
`db.at("sqlite:///other/runa.db").session(id)` to point at one specific file.
`db.at(url)` is the one escape hatch every concern shares (`.cache()`,
`.traces()`, ...), so no store has a backend-named class of its own. A
session is where turns live, never who the run is for: that's
`user_id=` on the run (see [Memory](#7-memory)).

With a `session`, only the new message is ever passed to `run`/`run_sync`,
prior turns come back from `runa.db` automatically, and `agent.history`
is left untouched. Don't mix the two: pick session-backed or
in-memory per agent instance, not both for the same conversation.

**`message` is one turn, never a transcript** (enforced: a list of
`{"role": ..., "content": ...}` messages raises `TypeError`). Load an
earlier conversation through the seam the chosen mode already has:
`agent.history = [...]` in memory, `await session.add_items([...])`
before the first run with a session.
`db.session` is how every shipped implementation is reached, so which one
a deployment gets is never a call site's decision; a custom store
subclasses `Session`'s three abstract methods (`get_items`, `add_items`,
`set_items`), the whole of what a run calls.

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
`"auto"` mode if you need a non-default `model`/`store`.
`Memory` is durable, semantic, `user_id`-scoped facts backed by the same
`db/runa.db` (`sqlite-vec`) sessions use: `remember`/`search`/
`forget`, nothing lower. This is long-term memory *across* conversations;
for one conversation's own turns, see [Session](#6-session).

**Who a run is for is `user_id=` on `run`/`run_sync`/`run_streamed`**,
beside `context=`, session or not. It's `RunContext.user_id` to the run's
tools and delegates and the scope of every memory read and write the run
makes. A run passing none shares one user-less scope with every other such
run, so an agent serving many users passes it on every turn. A paused
`RunState` carries it: passing `user_id=` alongside one raises `UserError`.

**Embedding is the one call an agent's `model` doesn't route.** Memory and
[Knowledge](#8-knowledge) send text to OpenAI (`OPENAI_API_KEY`) whatever
that model is -- it's the only provider Runa routes chat to that also
serves embeddings -- so a `claude-*` agent with `memory` set needs that
key too, and a run says so rather than degrading. A deployment that must
embed elsewhere (Azure, a gateway, something self-hosted) sets
`RUNA_EMBEDDINGS_BASE_URL`, plus `RUNA_EMBEDDINGS_API_KEY` when that
endpoint's token differs. It's an environment variable, not an argument,
for the reason `RUNA_DATABASE_URL` is: `memory = "auto"` names no
instance to pass one to, and where a deployment's embeddings come from is
not a call site's decision. `model=` on `Memory`/`Knowledge` still names
the embedding model, which is the application's decision.

**A source that doesn't answer is skipped; anything else ends the run.**
A connection error, a timeout, or a 408/409/429/5xx still failing after
the retries is a `ServiceUnavailable`: the run goes on without that
memory or knowledge, with the failure on its span and in the log. A
refused key is a `UserError`, and like any other exception it ends the
run, because an empty block reads like nothing relevant. A custom
`memory=`/`knowledge=` object raises `runa.exceptions.ServiceUnavailable`
to be skipped the same way. Enforced in code.

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
a non-default `directory`/`model`/`store`. `Knowledge()` means `app/knowledge/`,
`db/runa.db` (`sqlite-vec`), OpenAI's `text-embedding-3-small` -- discovery, chunking, embeddings,
and vector storage are entirely internal; put Markdown/PDF/text/CSV files under `app/knowledge/`
and no manual `.ingest()` call is needed either: `.search` ingests whenever the store doesn't
already hold the corpus it would write, so a first run ingests, an edited file re-ingests, and
nothing else does -- not a second process, and not the next request's `Agent`. `ingest()` asks
that same question, which is why it's one verb rather than a cheap check beside a forced rebuild:
idempotent, so `runa serve` calls it for every declared agent at startup and a deploy's first
request doesn't pay to embed the corpus. Pass
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
listed in `mcp=`.** `MCPServer` is the only exported name:
the per-transport classes it returns aren't a second call site, and a third
transport is written by subclassing `_MCPServerBase`, not by constructing
them. Enforced in code, in the ordinary Python way: every parameter of
`MCPServer`/`.http`/`.stdio` is named, so a misspelled or unsupported
option is a `TypeError` at the call site rather than a server that silently
ignores it.

```python
files = MCPServer(name="files").stdio("npx", ["-y", "@modelcontextprotocol/server-filesystem", "."])
search = MCPServer(name="search").http("https://example.com/mcp")

mcp = [files, search]
```

An MCP server's tools are exposed to the model exactly like `@tool`
functions. Never branch application code on whether a tool came from MCP
or from `@tool`; the framework already erases that distinction -- including
the two policies a tool carries. A remote tool has no decorator to declare
them on, so the server does, for every tool it exposes:
`MCPServer(name=..., needs_approval=..., guardrails=[...])` takes the same
`bool`/[`@approval`](#4-human-approval) predicate and the same
[guardrail](#3-guardrail) list `@tool` does. Per server, because trust is a
property of the server rather than of the transport it's reached over.

**A name is a key.** Every tool, delegate and handoff an agent offers has a
name no other one has, and every agent in one graph has a name of its own:
the model calls a tool by name, a sticky approval is keyed by it, and a
resumed run finds its agent by it. Enforced in code, a duplicate is a
`UserError`. An MCP tool is offered as `<server>__<tool>` (`files__read_file`),
so two servers exposing the same tool stay two tools, told apart by their
`name=`. The connection opens lazily and lives for the agent's whole lifetime, so
there's no explicit `.connect()`/`.close()` for user code to call in the
common case.

**An agent's whole lifetime means the process.** `await agent.run(...)`
on an app's own loop and `run_sync` (every call of which shares one
background loop) each keep one connection open across turns, and a
`.stdio` server's subprocess is shut down when the process exits.

## 10. Model

**Always a plain string on `model`, never a constructed client.** The
string's prefix (`claude-`, `gpt-`, `gemini-`, `llama-`, `deepseek-`,
`qwen-`, plus OpenAI's non-`gpt` families) picks the provider and its API
key env var; nothing is configured globally.

```python
model = "claude-sonnet-5"
```

**The prefix set is closed**, the way `Agent`'s settings are: a name
matching none of it is a `UserError` at construction naming the prefix it
most likely meant, not a default. Enforced in code, because the failure it
replaces is the worse one, `model = "claud-sonnet-5"` sending an app's
prompts to a different vendor under a different key and surfacing as a 404
attributed to model misbehaviour. An endpoint Runa doesn't name (a
gateway, a proxy, something self-hosted) is reached by passing a `Model`
instead of a name, which is the one sanctioned way to say "not one of
these" out loud.

Mixing providers across agents in one app (a cheap router, a stronger
answerer) is expected, not an edge case: `model` is per-agent by design.
Tune sampling with `model_settings = ModelSettings(...)`, never
provider-specific kwargs on `model` itself.

## 11. Hooks

**Always one `Hooks` subclass, overriding only the lifecycle methods you
need; every other method stays a no-op.** There is one class and one set
of event names. Scope is where you put the instance, not a second class
to pick:

* Passed as `Agent.run(hooks=...)`, it fires for every agent in that run,
  including ones reached by handoff/delegate. Use this for cross-cutting
  concerns (metrics, a single audit log for the whole run).
* Assigned to an `Agent` subclass's `hooks` attribute, it fires only for
  that one agent. Use this for a concern that belongs to one agent's
  identity, not the run as a whole.

Both scopes fire for the same event, with the same arguments, the run's
first. `on_handoff` is the one event an agent-scoped instance doesn't see
for every turn of its own agent: it fires on the target's hooks, not the
sender's, and is told `(from_agent, to_agent)` either way.

Don't subclass `LoggingHooks` to add behavior; subclass `Hooks` directly
and pass your own: `LoggingHooks` is the framework's default, not a base
to build on.

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

Only `Case.input` is required; `Case.context` is the run's `context=`, for an
agent that needs one. Add `expected`/`expected_tool`/`passages`
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

Add a `TraceExporter` (`ConsoleExporter`, `StoreExporter(db.at(url).traces())`, or your own)
to change *where* traces go; never change *whether* they're captured.
There is no flag for that.

---

When a change would let a user reach the same result through a second
shape (a new `Handoff(...)` call site, a hand-built tool schema, a
`Hooks` subclass that overrides `LoggingHooks`), that's a sign to close the
second path, not to document it alongside the first.

"One shape" applies to declarations as well as primitives. Every list
setting (`tools`, `subagents`, `guardrails`, `mcp`) is a list, never a
dict grouping its entries, and each setting has one name. A shorthand alias
(`.h`/`.d`, `.i`/`.o`) is a shorter spelling of the same binding and
produces the same object, so it doesn't count as a second shape. A bare
entry isn't one either: it means something of its own (auto for a
subagent, both sides for a guardrail).
