# Guardrails

A guardrail is a function `(value) -> bool`. Return `True` and the run trips. Return `False` and
it continues.

```python
from runa import guardrail


@guardrail
def block_empty(input: str) -> bool:
    """Trip when the user sends an empty message."""
    return not input.strip()
```

Bind it to a side with `.input` or `.output`, and list it on an agent:

```python
class SupportAgent(Agent):
    name = "support_agent"
    instructions = "..."
    guardrails = [block_empty.input]
```

* On `.input`, the predicate sees the latest user message as plain text.
* On `.output`, it sees the agent's final output.
* `.i` and `.o` are shorthand for `.input` and `.output`, the exact same binding, just shorter to
  type. There is no other way to bind a guardrail.

Listed bare, with no `.input`/`.output`, a guardrail is wired to both sides:

```python
guardrails = [block_empty.input, contains_pii]  # contains_pii checks input and output
```

Group them explicitly instead, if you would rather be specific:

```python
guardrails = {"input": [block_empty], "output": [contains_pii]}
```

A tripped guardrail stops the run before the model call (input side), or before the caller sees
the output (output side). `run`/`run_sync` come back with `status="error"` instead of raising.

## Guardrails on Tools

The same `@guardrail` predicate works against a [tool's](tools.md) arguments or return value, via
`@tool(guardrails=[...])`:

```python
@guardrail
def no_args(args: dict) -> bool:
    """Trip if this tool is somehow called with arguments."""
    return bool(args)


@tool(guardrails=[no_args.input])
def now() -> str:
    """Return the current time."""
    ...
```

On a tool, `.input` sees the call's parsed arguments as a `dict`. `.output` sees the tool's raw
return value. A guardrail that only observes, for logging or metrics, without ever tripping just
always returns `False`.

## Sync and Async Guardrails

A guardrail predicate can be a plain `def` or `async def`. Pick whichever reads better for what
the predicate checks; a sync `def` never blocks the run: it runs off the event loop, in a worker
thread (`asyncio.to_thread` under the hood), so a sync predicate that happens to do blocking I/O
(a moderation API call, ...) never stalls other concurrent runs or tool calls. An `async def`
predicate is awaited directly instead, which avoids that thread-dispatch overhead for a predicate
that's already genuinely async. Neither choice can break the run either way; it's purely a style
call.

## Human Approval

Some tool calls should not run without a person saying yes. That is a different mechanism from a
guardrail: a guardrail's predicate is the final verdict, while `needs_approval`'s predicate only
decides whether to stop and ask a human. See [Human Approval](approval.md).

## Example

```python
--8<--"examples/03_guardrail/input_output_guardrails.py"
```

```python
--8<--"examples/03_guardrail/tool_guardrails.py"
```

More in [`examples/03_guardrail/`](https://github.com/Benybrahim/runa/tree/main/examples/03_guardrail).
