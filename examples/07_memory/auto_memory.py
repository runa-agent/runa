"""`memory = "auto"`: the framework retrieves and stores memories for you.

See RUNA.md #7 and docs/memory.md.

Before each run, Runa searches memory for anything relevant to the user's message. After the
run, it asks the model what's durably worth keeping and stores it -- no manual `remember`/
`search` calls. Memory is scoped by the `user_id` passed to `run`, session or not.

Run it:

    uv run python examples/07_memory/auto_memory.py
"""

from runa import Agent


class SupportAgent(Agent):
    """Remembers durable facts about the user across separate conversations."""

    name = "support_agent"
    instructions = "You are a helpful support assistant."
    memory = "auto"


SupportAgent().run_sync(
    "I only speak Japanese, please reply in Japanese from now on.", user_id="user-42"
)

# ... a new conversation later, a fresh agent with no history, same user ...
run = SupportAgent().run_sync("What's my order status?", user_id="user-42")
print(run.output)  # sees "user speaks Japanese", retrieved automatically
