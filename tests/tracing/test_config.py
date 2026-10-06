"""Tests for `runa.tracing.observe`: the privacy/size policy and its `with`/bare-call forms."""

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from runa import db
from runa._types import ModelResponse, ModelSettings, Usage
from runa.run_config import RunConfig
from runa.run_internal.run_loop import _run_async
from runa.tool import tool
from runa.tracing import ConsoleExporter, Trace, config, observe


class _TextModel:
    def __init__(self, text: str) -> None:
        self._text = text

    async def get_response(self, *args: Any, **kwargs: Any) -> ModelResponse:  # noqa: ANN002, ANN003
        return ModelResponse(
            output=[{"role": "assistant", "content": self._text, "tool_calls": None}],
            usage=Usage(input_tokens=1, output_tokens=1, total_tokens=2, requests=1),
        )


def _tool_call_then_text(name: str, arguments: str, text: str) -> list[ModelResponse]:
    return [
        ModelResponse(
            output=[
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": name, "arguments": arguments},
                        }
                    ],
                }
            ],
            usage=Usage(input_tokens=1, output_tokens=1, total_tokens=2, requests=1),
        ),
        ModelResponse(
            output=[{"role": "assistant", "content": text, "tool_calls": None}],
            usage=Usage(input_tokens=1, output_tokens=1, total_tokens=2, requests=1),
        ),
    ]


class _ScriptedModel:
    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = list(responses)

    async def get_response(self, *args: Any, **kwargs: Any) -> ModelResponse:  # noqa: ANN002, ANN003
        return self._responses.pop(0)


def _agent_with_tool() -> Any:
    @tool
    def now(x: int) -> str:
        """Return a fixed time, ignoring `x`."""
        return "ok"

    return SimpleNamespace(
        name="TestAgent",
        instructions="hi",
        model=_ScriptedModel(_tool_call_then_text("now", '{"x": 1}', "done")),
        tools=[now],
        handoffs=[],
        input_guardrails=[],
        output_guardrails=[],
        output_type=None,
        mcp_servers=[],
        memory=None,
        knowledge=None,
        compact=False,
        model_settings=ModelSettings(),
    )


def test_observe_bare_call_leaves_the_setting_applied() -> None:
    """A plain `observe(capture_inputs=False)` call (no `with`) applies immediately and sticks."""
    observe(capture_inputs=False)
    try:
        assert config.capture_inputs() is False
    finally:
        observe(capture_inputs=True)


def test_observe_context_manager_restores_the_previous_setting_on_exit() -> None:
    """`with observe(...):` restores whatever policy was active before it, once the block ends."""
    assert config.capture_inputs() is True

    with observe(capture_inputs=False):
        assert config.capture_inputs() is False

    assert config.capture_inputs() is True


def test_default_exporters_is_the_store_only_without_langfuse_env_vars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no `LANGFUSE_*` env vars set, the default list is one exporter over the store."""
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.delenv("RUNA_DATABASE_URL", raising=False)

    exporters = config._default_exporters()

    assert [type(e).__name__ for e in exporters] == ["StoreExporter"]
    assert type(exporters[0].store).__name__ == "SQLiteTraceStore"  # type: ignore[attr-defined]


def test_default_exporters_adds_langfuse_once_its_env_vars_are_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Setting `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` is enough on its own, no code change.

    This is the whole point of checking them in `_default_exporters`: a Langfuse project should
    start receiving traces the moment its keys are in the environment, matching how tracing
    itself needs no separate setup step.
    """
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-env")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-env")
    monkeypatch.delenv("RUNA_DATABASE_URL", raising=False)

    names = [type(e).__name__ for e in config._default_exporters()]

    assert names == ["StoreExporter", "LangfuseExporter"]


def test_the_local_exporter_follows_the_database_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """Traces must land in the same database sessions do, or a deployment's story is split."""
    pytest.importorskip("asyncpg")
    monkeypatch.setenv("RUNA_DATABASE_URL", "postgresql://runa:runa@localhost:5432/runa")

    exporter = config._local_exporter()

    assert type(exporter.store).__name__ == "PostgresTraceStore"  # type: ignore[attr-defined]


def test_the_default_exporters_are_resolved_on_first_use_not_at_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A generated `main.py` calls `load_dotenv()` after importing runa, so import is too early.

    Resolving at import meant an app whose `.env` set `RUNA_DATABASE_URL` got a local store
    frozen in before that file had run, writing traces to a file while its sessions went to
    Postgres.
    """
    pytest.importorskip("asyncpg")
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.setattr(config._config, "exporters", None)  # as it is at import
    monkeypatch.setenv("RUNA_DATABASE_URL", "postgresql://runa:runa@localhost:5432/runa")

    exporters = config.exporters()

    assert [type(e).__name__ for e in exporters] == ["StoreExporter"]
    assert type(exporters[0].store).__name__ == "PostgresTraceStore"  # type: ignore[attr-defined]


def test_the_exporter_writes_to_the_store_it_was_resolved_with(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One decision per exporter, not one per write.

    `SQLiteExporter.export` used to call a module-level `save_trace` that asked `runa.db` for the
    backend all over again, so the store a deployment had installed and the store its traces
    landed in were two separate answers to the same question. Changing the environment after an
    exporter exists must not move its traces.
    """
    monkeypatch.setenv("RUNA_DATABASE_URL", "memory://")
    exporter = config._local_exporter()

    monkeypatch.setenv("RUNA_DATABASE_URL", "postgresql://nowhere:5432/none")
    exporter.export(Trace(id="t1", name="A", start_time=0.0))

    monkeypatch.setenv("RUNA_DATABASE_URL", "memory://")
    assert [trace.id for trace in db.traces().list()] == ["t1"]


def test_add_exporter_appends_without_replacing_the_active_ones() -> None:
    """`add_exporter` keeps whatever was active (the default store exporter) and adds to it.

    Unlike `observe(exporter=...)`, which fully replaces the list -- the right tool for a
    `with observe(...):` block that swaps exporters temporarily, but a footgun for the common
    case of wanting one more exporter without silently losing the default.
    """
    before = list(config.exporters())
    extra = ConsoleExporter()

    config.add_exporter(extra)
    try:
        assert config.exporters() == [*before, extra]
    finally:
        observe(exporter=before)


def test_capture_inputs_false_means_tool_span_input_is_not_recorded() -> None:
    """`observe(capture_inputs=False)` means a captured tool span's `input` stays `None`."""
    agent = _agent_with_tool()
    with observe(capture_inputs=False):
        result = asyncio.run(_run_async(agent, "hi", run_config=RunConfig(workflow_name="T")))

    assert result.trace is not None
    tool_span = next(span for span in result.trace.spans if span.type == "tool")
    assert tool_span.input is None
    assert tool_span.output == "ok"


def test_redact_scrubs_matching_keys_from_dict_shaped_output() -> None:
    """`redact=["email"]` replaces that key's value with `"[REDACTED]"` in dict-shaped output."""

    @tool
    def lookup() -> dict[str, Any]:
        """Return a dict containing an email."""
        return {"email": "a@b.com", "id": 1}

    agent: Any = SimpleNamespace(
        name="TestAgent",
        instructions="hi",
        model=_ScriptedModel(_tool_call_then_text("lookup", "{}", "done")),
        tools=[lookup],
        handoffs=[],
        input_guardrails=[],
        output_guardrails=[],
        output_type=None,
        mcp_servers=[],
        memory=None,
        knowledge=None,
        compact=False,
        model_settings=ModelSettings(),
    )

    with observe(redact=["email"]):
        result = asyncio.run(_run_async(agent, "hi", run_config=RunConfig(workflow_name="T")))

    assert result.trace is not None
    tool_span = next(span for span in result.trace.spans if span.type == "tool")
    assert tool_span.output["email"] == "[REDACTED]"
    assert tool_span.output["id"] == 1


def test_max_input_bytes_truncates_a_long_tool_argument() -> None:
    """`max_input_bytes` truncates a tool span's input once it exceeds the configured size."""

    @tool
    def now(x: str) -> str:
        """Return a fixed time, ignoring `x`."""
        return "ok"

    long_arg = "x" * 100
    agent: Any = SimpleNamespace(
        name="TestAgent",
        instructions="hi",
        model=_ScriptedModel(_tool_call_then_text("now", f'{{"x": "{long_arg}"}}', "done")),
        tools=[now],
        handoffs=[],
        input_guardrails=[],
        output_guardrails=[],
        output_type=None,
        mcp_servers=[],
        memory=None,
        knowledge=None,
        compact=False,
        model_settings=ModelSettings(),
    )

    with observe(max_input_bytes=10):
        result = asyncio.run(_run_async(agent, "hi", run_config=RunConfig(workflow_name="T")))

    assert result.trace is not None
    tool_span = next(span for span in result.trace.spans if span.type == "tool")
    assert len(tool_span.input) < 100
    assert tool_span.input.endswith("[truncated]")
