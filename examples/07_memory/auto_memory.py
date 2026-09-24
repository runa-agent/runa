"""`memory = "auto"`: the framework retrieves and stores memories for you.

See RUNA.md #7 and docs/memory.md.

Before each run, Runa searches memory for anything relevant to the user's message. After the
run, it asks the model what's durably worth keeping and stores it -- no manual `remember`/
`search` calls. Memory is scoped by `user_id`, set on the `session` passed to `run`.

Run it:

    uv run python examples/07_memory/auto_memory.py
"""

from runa import Agent, SQLiteSession


class SupportAgent(Agent):
    """Remembers durable facts about the user across separate conversations."""

    name = "support_agent"
    instructions = "You are a helpful support assistant."
    memory = "auto"


agent = SupportAgent()
session = SQLiteSession("chat-1", user_id="user-42")

agent.run_sync("I only speak Japanese, please reply in Japanese from now on.", session=session)

# ... a new conversation later, a different session, same user ...
later_session = SQLiteSession("chat-2", user_id="user-42")
run = agent.run_sync("What's my order status?", session=later_session)
print(run.output)  # sees "user speaks Japanese", retrieved automatically
