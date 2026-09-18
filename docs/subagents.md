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
calls. There is one known limitation: a delegate call that pauses on a non-sticky approval is not
surfaced back to the caller as an interruption. It comes back as a plain `None` result instead,
since a single delegate call has no pause/resume state of its own. Cover approval-gated tools
reachable from a delegate with a sticky decision ahead of time if the delegate might call them.

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

## Example

```python
--8<--"examples/05_subagent/handoff.py"
```

```python
--8<--"examples/05_subagent/delegate.py"
```
