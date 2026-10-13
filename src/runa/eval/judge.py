"""eval/judge.py: the judge model `eval/evaluation/semantic.py`'s metrics grade with.

`ask()` runs a prompt through a bare `runa.Agent`, so semantic metrics grade with whatever model
an app already talks to instead of requiring a separate client or API key, and through the same
one door as everything else: the provider, the `RunConfig` and the tracing are the Agent's, not a
second copy assembled here. Reading the JSON object out of a reply isn't here either: that's
`runa._items.parsed_json_object`, shared with every other caller that asks a model for JSON.
"""

from dataclasses import dataclass
from typing import Protocol

from runa.agent import Agent
from runa.exceptions import ModelBehaviorError


class _JudgeAgent(Agent):
    """A bare, tool-less agent for asking a judge model one prompt at a time."""

    name = "Judge"


class JudgeModel(Protocol):
    """The shape `eval/evaluation/semantic.py`'s metrics need from a judge: `ask()` a prompt.

    `Judge` satisfies this structurally, with no explicit inheritance needed; so does a
    lightweight stand-in built for testing the metrics themselves.
    """

    async def ask(self, prompt: str) -> str:
        """Send `prompt` to the judge model, returning its reply."""
        ...


@dataclass
class Judge:
    """A model reachable through Runa's own provider, asked one prompt at a time."""

    model: str

    async def ask(self, prompt: str) -> str:
        """Send `prompt` to `self.model` through a bare, tool-less `Agent`.

        A judge run that didn't complete raises rather than returning its empty `output`:
        `Metric.evaluate` turns a raised exception into an `ERROR` result carrying the reason,
        which is the difference between "the judge model is unreachable" and a `NoneType` error
        three frames later.
        """
        judge_agent = _JudgeAgent(model=self.model, tools=[])
        run = await judge_agent.run(prompt)
        if run.status != "completed":
            raise ModelBehaviorError(f"judge model {self.model!r} did not answer: {run.error}")
        return run.output


def judge_model(model: str) -> Judge:
    """Build the judge `evaluate_semantic()` should grade with."""
    return Judge(model)
