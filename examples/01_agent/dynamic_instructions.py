"""`instructions` as a function of `context`, resolved fresh on every run.

See RUNA.md #1 and docs/agents.md ("Instructions as a Function").

`context` is never sent to the model -- it's plumbed through to `instructions`, tools, and
guardrails as is, so it's the place to put per-run data (a user id, a tenant) without reaching
for global state.

Run it:

    uv run python examples/01_agent/dynamic_instructions.py
"""

from dataclasses import dataclass

from runa import Agent


@dataclass
class Context:
    """Per-run data threaded into `instructions` below."""

    user_name: str


def instructions(context: Context) -> str:
    """Greet the user by name, resolved fresh from `Agent.run`'s `context=` each call."""
    return f"You are a friendly assistant helping {context.user_name}."


class Assistant(Agent):
    """An assistant whose instructions are built per-run instead of fixed at class time."""

    name = "assistant"
    instructions = instructions


run = Assistant().run_sync("Hi, who am I talking to?", context=Context(user_name="Ada"))
print(run.output)
