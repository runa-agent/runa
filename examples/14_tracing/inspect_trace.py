"""Every run is traced automatically: `run.trace`'s span tree, with no setup step.

See RUNA.md #14 and docs/tracing.md.

Traces are persisted to `runa.db` by default -- inspect them later with `runa traces list`/
`show TRACE_ID`, or `runa ui`.

Run it:

    uv run python examples/14_tracing/inspect_trace.py
"""

from datetime import datetime

from runa import Agent, tool


@tool
def current_time() -> str:
    """Return the current local time as an ISO 8601 string."""
    return datetime.now().isoformat()


class GreeterAgent(Agent):
    """Greets the user, and tells them the time if asked."""

    name = "greeter_agent"
    instructions = "You greet the user, and tell them the time if asked."
    tools = [current_time]


run = GreeterAgent().run_sync("Hi! What time is it?")
print(run.trace)  # human-readable span tree: agent, LLM, tool spans

assert run.trace is not None  # every run is traced; `Trace | None` only covers a type that skips it
for span in run.trace.spans:
    print(f"- {span.type} span: {span.name!r} -> {span.output!r}")
