"""eval/judge.py: the judge model `eval/evaluation/semantic.py`'s metrics grade with.

`ask()` runs a prompt through the same `runa._models.ModelProvider` every `runa.Agent` uses (see
`runa.agent`), so semantic metrics grade with whatever model an app already talks to instead of
requiring a separate client or API key. `extract_json()` pulls a JSON object out of a judge's
reply, tolerating the odd trailing comma a model sometimes emits.
"""

import json
import re
from dataclasses import dataclass
from typing import Any, Protocol

from runa.agent import _MODEL_PROVIDER, Agent
from runa.run_config import RunConfig
from runa.runner import Runner

_TRAILING_COMMA = re.compile(r",\s*([\]}])")


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
        """Send `prompt` to `self.model` through a bare, tool-less `Agent`."""
        judge_agent = _JudgeAgent(model=self.model, tools=[])
        result = await Runner.run(
            judge_agent, prompt, run_config=RunConfig(model_provider=_MODEL_PROVIDER)
        )
        return result.final_output


def judge_model(model: str) -> Judge:
    """Build the judge `evaluate_semantic()` should grade with."""
    return Judge(model)


def extract_json(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of a judge's reply.

    Judges wrap their JSON in prose or markdown fences more often than not, and occasionally
    leave a trailing comma before a closing `]`/`}`; this tolerates both rather than demanding a
    clean `json.loads`.
    """
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"judge reply contained no JSON object: {text!r}")
    candidate = text[start:end]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return json.loads(_TRAILING_COMMA.sub(r"\1", candidate))
