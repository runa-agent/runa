"""`model_settings`: tuning sampling per agent, never with provider-specific kwargs on `model`.

See RUNA.md #10 and docs/models.md ("Model Settings").

Run it:

    uv run python examples/10_model/model_settings.py
"""

from runa import Agent
from runa._types import ModelSettings


class SupportAgent(Agent):
    """A low-temperature, short-answer support agent."""

    name = "support_agent"
    instructions = "You are a helpful support assistant. Answer in one short sentence."
    model_settings = ModelSettings(temperature=0.2, max_tokens=60)


run = SupportAgent().run_sync("What's the capital of France?")
print(run.output)
