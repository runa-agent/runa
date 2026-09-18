"""The same `@guardrail` predicate, checked against a tool's arguments or return value.

See RUNA.md #3 and docs/guardrails.md ("Guardrails on Tools").

On a tool, `.input` sees the call's parsed arguments as a `dict`. `.output` sees the tool's raw
return value. A guardrail that only observes, without ever tripping, just always returns `False`.

Run it:

    uv run python examples/03_guardrail/tool_guardrails.py
"""

from datetime import datetime

from runa import Agent, guardrail, tool


@guardrail
def no_args(args: dict) -> bool:
    """Trip if `now` is somehow called with arguments."""
    return bool(args)


@guardrail
def block_long(output: str) -> bool:
    """Trip when the tool's return value runs longer than 200 characters."""
    return len(output) > 200


@guardrail
def log_call(value: object) -> bool:
    """Print the value seen on either side of the call, but never trip."""
    print(f"now(): {value!r}")
    return False


@tool(guardrails=[no_args.input, block_long.output, log_call])
def now() -> str:
    """Return the current local time as an ISO 8601 string."""
    return datetime.now().isoformat()


class ClockAgent(Agent):
    """Tells the user the current time."""

    name = "clock_agent"
    instructions = "You tell the user the current time when asked."
    tools = [now]


run = ClockAgent().run_sync("What time is it right now?")
print(run.output)
