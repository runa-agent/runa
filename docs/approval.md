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

A `call_id` that already ran once cannot be submitted again. Resuming the same `RunState` twice
raises `DuplicateToolCallError` rather than silently re-running the tool.

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

`RunState` survives a process restart. `state.to_json()`/`.to_string()` serialize it, as a plain
dict or a JSON string. `RunState.from_json(agent, blob)`/`.from_string(agent, blob)` rebuild it,
given a fresh instance of the agent the run started with, used to re-resolve the current agent
and each pending tool by name. A live `Agent` instance and a tool's closure cannot round-trip
through JSON themselves.

A `context` that was a dataclass comes back as a plain dict, not its original class. The run's
guardrail audit trail (below) and trace spans are not included in the serialized blob. Spans are
already persisted separately (see [Tracing and Hooks](tracing.md)). An unrecognized
`schema_version` raises `UserError` rather than resuming from a blob a different, incompatible
version of Runa produced.

## Guardrail Audit Trail

Every guardrail that ran during a run, tripped or not, is recorded on
`result.input_guardrail_results`, `.output_guardrail_results`, `.tool_input_guardrail_results`,
and `.tool_output_guardrail_results` (and the same four on a paused `RunState`, reflecting only
what ran before the pause), not just whichever one stopped the run.

## Example

```python
--8<--"examples/04_approval/needs_approval.py"
```

```python
--8<--"examples/04_approval/durable_resume.py"
```

More in [`examples/04_approval/`](https://github.com/Benybrahim/runa/tree/main/examples/04_approval).
