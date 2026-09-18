"""The smallest useful Runa agent: one `Agent`, one `@tool`, one call.

Run it:

    uv run python examples/00_quickstart/hello.py

Uses Runa's default model (`gpt-5.4-nano`), so it runs with just `OPENAI_API_KEY` set -- change
`model` below (and set the matching key) to run it on a different provider (see
`examples/10_model/`). Every other numbered folder here covers one more primitive, in the same
order as RUNA.md.
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
    instructions = "You greet the user warmly, and tell them the time if asked."
    tools = [current_time]


run = GreeterAgent().run_sync("Hi! What time is it?")
print(run.output)
