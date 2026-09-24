# Subagents

An agent can bring in another agent to help, in one of two ways: **handoff** (transfer control
entirely) or **delegate** (call it and get an answer back). Both are declared with `subagents`.

```python
from runa import Agent


class Researcher(Agent):
    name = "researcher"
    instructions = "You research topics thoroughly and report back findings."


class Translator(Agent):
    name = "translator"
    instructions = "You translate text into French."


class Assistant(Agent):
    name = "assistant"
    instructions = "You help the user, bringing in specialists as needed."
    subagents = [Researcher.handoff, Translator.delegate]
```

## Handoff

`SomeAgent.handoff` transfers the whole conversation to `SomeAgent`. From that point on, it is
the one talking to the user. Use this when a subagent should fully take over, for example
"route this support ticket to billing." `.h` is shorthand for `.handoff`, the exact same
binding, just shorter to type. There is no other way to bind a handoff.

## Delegate

`SomeAgent.delegate` wires `SomeAgent` in as a callable tool instead. It does the task and
returns its result to the calling agent, like a tool call. The calling agent stays in control: it
gets the output back and decides what to do next. Use this when you want a subroutine, not a
handoff, for example "ask the researcher, then summarize their answer yourself." `.d` is
shorthand for `.delegate`, the same as `.h`/`.handoff`.

A bare entry in `subagents`, with no `.handoff`/`.delegate`, is wired as **both**: a handoff and
a delegate tool at once, so the model can either call it for a quick answer or transfer the whole
conversation to it:

```python
subagents = [Researcher, Translator.delegate]  # Researcher: both, Translator: delegate only
```

A `.delegate` call shares the caller's [approval](approval.md) ledger and usage accounting: a
sticky (`always=True`) decision on the caller's side already covers a matching tool the delegate
calls. A delegate that pauses on an approval pauses its caller too: the delegate's pending calls
appear in the caller's `run.interruptions`, and resuming the caller's `RunState` resumes the
delegate where it stopped, even after a `to_json`/`from_json` round trip.

## Parallel Delegates

Delegates are tools, so when the model calls several in one message, they run concurrently.
Set `model_settings = ModelSettings(parallel_tool_calls=False)` on the calling agent to run them
one by one instead.

## Naming a Delegate's Tool

A delegated agent is exposed as a tool named after it by default. Override the name or
description it is given:

```python
subagents = [Researcher.delegate(tool_name="research", tool_description="Research a topic.")]
```

## Grouping by Mode

For a longer list, group entries under `"handoff"`, `"delegate"`, or `"auto"` instead of
repeating `.handoff`/`.delegate` on each:

```python
subagents = {
    "handoff": [Billing, Returns],
    "delegate": [Researcher, Translator],
}
```

Entries under `"auto"` pass through as given, whether bare (wired as both) or already
`.handoff`/`.delegate`-bound, for mixing modes within one dict without forcing every entry the
same way.

## Workflows in Code

Subagents are for when the model picks the order. When your code picks it, write a plain
`async` function. Wrap it in `tracing.trace` so its runs are grouped:

```python
import asyncio

from runa import tracing


async def onboard(ticket: str) -> str:
    triage = await TriageAgent().run(ticket)
    billing, docs = await asyncio.gather(
        BillingAgent().run(triage.output), DocsAgent().run(triage.output)
    )
    summary = await SummaryAgent().run(f"{billing.output}\n{docs.output}")
    return summary.output


with tracing.trace("onboard"):
    asyncio.run(onboard("..."))
```

Each run keeps its own trace, with the block's trace id as its `group_id`.

## Example

```python
--8<--"examples/05_subagent/handoff.py"
```

```python
--8<--"examples/05_subagent/delegate.py"
```

More in [`examples/05_subagent/`](https://github.com/Benybrahim/runa/tree/main/examples/05_subagent).
