# Human Approval

Some tool calls are sensitive enough that a person should sign off before they run: deleting a
record, sending an email, spending money. Gate a tool with `needs_approval`:

```python
from runa import Agent, approval, tool


@approval
def large_refund(amount: float) -> bool:
    """Refunds of $50 or more need a human to sign off."""
    return amount >= 50


@tool(needs_approval=large_refund)
def issue_refund(amount: float) -> str:
    """Refund the customer."""
    ...
```

`@approval` wraps a predicate the same way `@guardrail` does. Its parameters are looked up by
name from the tool call's parsed arguments; name one `ctx` or `call_id` to receive the run
context or call id instead. `True` means what the parameter is called: this call needs approval.
That is the same sense as a guardrail's tripwire, just gated on a person instead of enforced
automatically.

Pass `needs_approval=True` instead of a predicate to always require approval, with no condition.

A tool coming from an [MCP server](mcp.md) is gated the same way, on the server rather than on a
decorator it doesn't have: `MCPServer(name="files", needs_approval=writes_outside_tmp)` applies
to every tool that server exposes.

`runa chat` prompts interactively for any call that needs approval:

```
approve issue_refund({"amount": 120})? [y/N]
```

Rejecting a call skips the tool entirely. The model sees "rejected by the operator" instead of a
result.

## `needs_approval` vs. a Guardrail

Both are `predicate(args) -> bool`, and a tripped guardrail and a call that needs approval both
start from the same `True`. What happens next is different:

| | `guardrails=[fn]` (tool guardrail) | `needs_approval=fn` |
|---|---|---|
| Who makes the final call | The predicate itself | A human, later |
| On `True` | Raises a tripwire, the run ends in an error | Pauses the run, returns an `Interruption` |
| Can be overridden | No | Yes, the operator can approve anyway |

Use a **guardrail** for calls that must never go through, no matter who is asking. The predicate
is the whole decision. Use **`needs_approval`** for calls that are fine with a person's sign-off.
The predicate only decides whether to stop and ask. The actual yes or no comes from whoever
resolves the pending call.

## Pausing and Resuming

A call that needs approval does not run. It pauses the run instead: the `Run` comes back with
`status="paused"` and the pending calls in `run.interruptions`. Resolve each one against
`run.to_state()`, then resume by passing that `RunState` back into `run`/`run_sync` in place of
a message:

```python
run = agent.run_sync("issue a $75 refund")
while run.status == "paused":
    state = run.to_state()
    for item in run.interruptions:
        state.approve(item)  # or state.reject(item, rejection_message="not authorized")
    run = agent.run_sync(state)
```

`approve`/`reject` also take `always=True`: the decision then sticks for every future call to
that tool, matched by name rather than just this one call id. This is what an operator answering
"always" instead of "yes" to `runa chat`'s `[y/N/a]` prompt does under the hood. A rejection can
carry a custom `rejection_message`, fed back to the model instead of the default text; with
`always=True`, that message is reused for every later rejected call to the same tool too.

A pause resumes once. Resuming it a second time, from the same `RunState` or from another
process's copy of it, comes back as an error `Run` (`RunAlreadyResumed`) rather than running the
tool again.

`run_streamed` pauses the same way. Once the stream ends, its `.run` is the paused `Run`;
resolve it and pass the state back to continue, streaming again:

```python
stream = agent.run_streamed("issue a $75 refund")
async for event in stream:
    ...
state = stream.run.to_state()
for item in stream.run.interruptions:
    state.approve(item)
async for event in agent.run_streamed(state):
    ...
```

A [delegate](subagents.md) that calls an approval-gated tool pauses its caller the same way: the
delegate's calls show up in the caller's `run.interruptions`, and resuming the caller resumes the
delegate right where it stopped.

## Durability

A pause survives a process restart with nothing to save by hand: it is kept in `runa.db` the
moment it happens, under `run.run_id`. Another process gets it back with
`await RunState.load(agent, run_id)`, or `RunState.load_sync` beside `run_sync`, given a fresh
instance of the agent the run started with:

```python
state = RunState.load_sync(SupportAgent(), run_id)
for item in state.pending:
    state.approve(item)
run = SupportAgent().run_sync(state)
```

`load` returns `None` for a pause that doesn't exist or already resumed. It spends nothing and
resolves no tools, so it never connects to an agent's MCP servers: the resumed run claims the
pause, atomically, so of two workers resuming the same pause exactly one runs it. There is no
blob to hand a client instead, since every copy of a pause would be another chance to run it.
Over HTTP this is already written: `runa serve` reports the `run_id`, and posting it back with a
decision per interruption resumes it (see [Deployment](deployment.md#resuming-a-paused-run)).

Everything the run carries comes back: its conversation and session, `context` and user, its usage, the
tool calls it already made and the time it has already spent against its `timeout`. A `context`
that was a dataclass comes back as a plain dict, not its original class. The run's
[guardrail audit trail](guardrails.md#audit-trail) and trace spans are not stored with the pause. Spans are
already persisted separately (see [Tracing and Hooks](tracing.md)). A pause stored by an
incompatible version of Runa raises `UserError` on `load` rather than resuming.

## Example

```python
--8<--"examples/04_approval/needs_approval.py"
```

```python
--8<--"examples/04_approval/durable_resume.py"
```

More in [`examples/04_approval/`](https://github.com/Benybrahim/runa/tree/main/examples/04_approval).
