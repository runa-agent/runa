"""`@guardrail`: a predicate checked against an agent's input or output.

See RUNA.md #3 and docs/guardrails.md.

Run it:

    uv run python examples/03_guardrail/input_output_guardrails.py
"""

import re

from runa import Agent, guardrail

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


@guardrail
def block_empty(input: str) -> bool:
    """Trip when the user sends an empty message."""
    return not input.strip()


@guardrail
def contains_pii(text: str) -> bool:
    """Trip when the text contains an email address. Checks both input and output below."""
    return bool(_EMAIL.search(text))


class SupportAgent(Agent):
    """Refuses empty messages, and refuses to discuss or echo an email address."""

    name = "support_agent"
    instructions = "You are a helpful support assistant. Never ask for or repeat an email address."
    guardrails = [block_empty.input, contains_pii]  # bare = wired to both input and output


agent = SupportAgent()

run = agent.run_sync("   ")
print(f"empty input: status={run.status!r} error={run.error!r}")

run = agent.run_sync("My email is ada@example.com, can you note it down?")
print(f"pii input: status={run.status!r} error={run.error!r}")
