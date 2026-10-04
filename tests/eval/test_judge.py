"""Tests for `runa.eval.judge`: the judge model semantic metrics grade with."""

import asyncio
from typing import Any

import pytest

from runa._types import Usage
from runa.agent import Agent
from runa.eval.judge import extract_json, judge_model
from runa.exceptions import ModelBehaviorError
from runa.run import Run


def test_judge_model_asks_through_the_named_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """`judge_model(...).ask(...)` runs the prompt through a bare `Agent` using that model."""
    captured: dict[str, Any] = {}

    async def fake_run(self: Agent, message: Any, *args: Any, **kwargs: Any) -> Run:
        captured["model"] = self.model
        captured["input"] = message
        return Run(output="PASS", trace=None, usage=Usage())

    monkeypatch.setattr(Agent, "run", fake_run)

    output = asyncio.run(judge_model("gpt-5.4-nano").ask("grade this"))

    assert output == "PASS"
    assert captured["model"] == "gpt-5.4-nano"
    assert captured["input"] == "grade this"


def test_extract_json_parses_a_clean_object() -> None:
    """A reply that's already valid JSON parses straight through."""
    assert extract_json('{"score": 0.8}') == {"score": 0.8}


def test_extract_json_ignores_surrounding_prose() -> None:
    """A judge that wraps its JSON in commentary or markdown fences still parses."""
    reply = 'Sure, here is my verdict:\n```json\n{"verdict": "yes", "reason": "matches"}\n```'
    assert extract_json(reply) == {"verdict": "yes", "reason": "matches"}


def test_extract_json_repairs_a_trailing_comma() -> None:
    """A trailing comma before a closing bracket is stripped and retried, not just rejected."""
    reply = '{"statements": ["a", "b",], "extra": "x",}'
    assert extract_json(reply) == {"statements": ["a", "b"], "extra": "x"}


def test_extract_json_raises_when_no_object_is_present() -> None:
    """A reply with no `{...}` at all raises rather than silently returning garbage."""
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_extract_json_raises_on_unrepairable_json() -> None:
    """Malformed JSON that isn't just a trailing comma still raises."""
    with pytest.raises(ValueError):
        extract_json('{"score": }')


def test_judge_model_raises_when_the_run_did_not_complete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A judge run reported as `status="error"` raises, rather than returning its empty output."""

    async def fake_run(self: Agent, message: Any, *args: Any, **kwargs: Any) -> Run:
        return Run(output=None, trace=None, usage=Usage(), status="error", error="no API key")

    monkeypatch.setattr(Agent, "run", fake_run)

    with pytest.raises(ModelBehaviorError, match="no API key"):
        asyncio.run(judge_model("gpt-5.4-nano").ask("grade this"))
