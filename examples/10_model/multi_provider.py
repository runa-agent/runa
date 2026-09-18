"""Mixing model providers across agents: a cheap router, a stronger answerer.

See RUNA.md #10 and docs/models.md.

`model` is a plain string, resolved to a provider by its prefix (`claude-`, `gpt-`, `gemini-`,
`llama-`, `deepseek-`, `qwen-`); no client to construct, no provider configured globally.

Run it (needs both `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` set):

    uv run python examples/10_model/multi_provider.py
"""

from runa import Agent


class SupportAgent(Agent):
    """The agent that actually answers -- a stronger model, since it does the real work."""

    name = "support_agent"
    model = "claude-opus-5"
    instructions = "You are a helpful, thorough support assistant."


class Router(Agent):
    """A fast, cheap model whose only job is routing to the right specialist."""

    name = "router"
    model = "gpt-5.4-nano"
    instructions = "Route every message to the support agent."
    subagents = [SupportAgent.handoff]


run = Router().run_sync("My order hasn't arrived.")
print(run.output)
