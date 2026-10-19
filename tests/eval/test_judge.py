"""Tests for `runa.eval.judge`: the judge model semantic metrics grade with."""

import asyncio
from typing import Any

import pytest
from helpers import finished_run

from runa.agent import Agent
from runa.eval.judge import judge_model
from runa.exceptions import ModelBehaviorError
from runa.run import Run


def test_judge_model_asks_through_the_named_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """`judge_model(...).ask(...)` runs the prompt through a bare `Agent` using that model."""
    captured: dict[str, Any] = {}

    async def fake_run(self: Agent, message: Any, *args: Any, **kwargs: Any) -> Run:
        captured["model"] = self.model
        captured["input"] = message
        return finished_run("PASS")

    monkeypatch.setattr(Agent, "run", fake_run)

    output = asyncio.run(judge_model("gpt-5.4-nano").ask("grade this"))

    assert output == "PASS"
    assert captured["model"] == "gpt-5.4-nano"
    assert captured["input"] == "grade this"


def test_judge_model_raises_when_the_run_did_not_complete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A judge run reported as `status="error"` raises, rather than returning its empty output."""

    async def fake_run(self: Agent, message: Any, *args: Any, **kwargs: Any) -> Run:
        return finished_run(None, status="error", error="no API key")

    monkeypatch.setattr(Agent, "run", fake_run)

    with pytest.raises(ModelBehaviorError, match="no API key"):
        asyncio.run(judge_model("gpt-5.4-nano").ask("grade this"))
