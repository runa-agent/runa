"""Tests for `runa.cli.eval`: `run_project_evals`."""

from pathlib import Path
from typing import Any

import pytest

from runa.cli._project import NotARunaProject
from runa.cli.chat import AgentNotFound
from runa.cli.eval import (
    InvalidEvalModule,
    TraceHasNoInput,
    add_trace_to_evals,
    run_project_evals,
)
from runa.cli.generate import generate_agent
from runa.cli.new import scaffold_project
from runa.cli.traces import TraceNotFound
from runa.eval import Case, Dataset
from runa.tracing import Span, Trace
from runa.tracing.storage import save_trace


def _write_evaluation(project_dir: Path, filename: str, source: str) -> None:
    (project_dir / "evals" / filename).write_text(source)


def _eval_module_source(class_name: str, agent_name: str) -> str:
    return (
        "from runa import Agent, Case\n\n\n"
        f"class {class_name}(Agent):\n"
        f"    name = {agent_name!r}\n\n\n"
        f"agent = {class_name}()\n"
        "dataset: list[Case] = [Case(input='hi', expected='hi')]\n"
    )


def _patch_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_run(agent: Any, input: Any, **kwargs: Any) -> Any:
        from dataclasses import dataclass, field

        from runa.tracing import Trace

        @dataclass
        class _FakeResult:
            final_output: Any = input
            new_items: list[Any] = field(default_factory=list)
            trace: Trace = field(
                default_factory=lambda: Trace(id="t", name="t", start_time=0.0, spans=[])
            )

        return _FakeResult()

    async def fake_evaluate_semantic(*args: Any, **kwargs: Any) -> list[Any]:
        return []

    monkeypatch.setattr("runa.eval.tracing.adapter.Runner.run", staticmethod(fake_run))
    monkeypatch.setattr("runa.eval.evaluate.evaluate_semantic", fake_evaluate_semantic)
    monkeypatch.setattr("runa.eval.evaluate.save_report", lambda report: 1)
    monkeypatch.setattr("runa.eval.evaluate.load_baseline", lambda agent_name: None)


def test_run_project_evals_raises_outside_a_runa_project(tmp_path: Path) -> None:
    """`run_project_evals` refuses to run where `evals/` doesn't exist."""
    with pytest.raises(NotARunaProject):
        run_project_evals(tmp_path)


def test_run_project_evals_raises_for_a_module_missing_agent_or_dataset(tmp_path: Path) -> None:
    """A module under `evals/` that doesn't declare `agent`/`dataset` is rejected."""
    project_dir = scaffold_project("demo", root=tmp_path)
    _write_evaluation(project_dir, "broken_eval.py", "agent = None\n")

    with pytest.raises(InvalidEvalModule):
        run_project_evals(project_dir)


def test_run_project_evals_evaluates_every_declared_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every module's `agent`/`dataset` is run through `agent.evaluate()`."""
    project_dir = scaffold_project("demo", root=tmp_path)
    _write_evaluation(
        project_dir, "support_eval.py", _eval_module_source("_Placeholder", "support")
    )
    _patch_runner(monkeypatch)

    reports = run_project_evals(project_dir)

    assert len(reports) == 1
    assert len(reports[0].cases) == 1
    assert reports[0].cases[0].passed


def test_run_project_evals_filters_to_one_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`agent_name` runs only the `evals/` module whose `agent` declares that `name`."""
    project_dir = scaffold_project("demo", root=tmp_path)
    _write_evaluation(project_dir, "support_eval.py", _eval_module_source("_Support", "support"))
    _write_evaluation(project_dir, "billing_eval.py", _eval_module_source("_Billing", "billing"))
    _patch_runner(monkeypatch)

    reports = run_project_evals(project_dir, "billing")

    assert len(reports) == 1


def test_run_project_evals_resolves_a_jsonl_dataset_s_agent_from_its_filename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`evals/<agent_name>.jsonl` alone is an eval: its filename names the Agent to grade."""
    project_dir = scaffold_project("demo", root=tmp_path)
    generate_agent("SupportAgent", root=project_dir, instructions="Help.")
    _write_evaluation(project_dir, "support_agent.jsonl", '{"input": "hi"}\n{"input": "yo"}\n')
    _patch_runner(monkeypatch)

    reports = run_project_evals(project_dir, "support_agent")

    assert [report.agent_name for report in reports] == ["SupportAgent"]
    assert [case.case.input for case in reports[0].cases] == ["hi", "yo"]


def test_run_project_evals_leaves_a_module_s_own_jsonl_to_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A `.jsonl` sharing a module's stem is that module's data, not a second eval."""
    project_dir = scaffold_project("demo", root=tmp_path)
    _write_evaluation(project_dir, "support_eval.py", _eval_module_source("_Support", "support"))
    _write_evaluation(project_dir, "support_eval.jsonl", '{"input": "hi"}\n')
    _patch_runner(monkeypatch)

    reports = run_project_evals(project_dir)

    assert len(reports) == 1


def test_run_project_evals_raises_for_a_jsonl_naming_no_agent(tmp_path: Path) -> None:
    """A `.jsonl` whose filename matches no declared Agent `name` is rejected."""
    project_dir = scaffold_project("demo", root=tmp_path)
    _write_evaluation(project_dir, "ghost_agent.jsonl", '{"input": "hi"}\n')

    with pytest.raises(AgentNotFound):
        run_project_evals(project_dir)


def test_run_project_evals_raises_for_an_unknown_agent_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An `agent_name` matching no `evals/` module's `agent` is rejected."""
    project_dir = scaffold_project("demo", root=tmp_path)
    _write_evaluation(project_dir, "support_eval.py", _eval_module_source("_Support", "support"))
    _patch_runner(monkeypatch)

    with pytest.raises(AgentNotFound):
        run_project_evals(project_dir, "nonexistent")


def _save_agent_trace(project_dir: Path, trace_id: str, input: str | None) -> None:
    trace = Trace(id=trace_id, name="SupportAgent", start_time=0.0, end_time=1.0)
    trace.spans = [
        Span(
            id=f"{trace_id}_root",
            trace_id=trace_id,
            parent_id=None,
            name="support_agent",
            type="agent",
            start_time=0.0,
            end_time=1.0,
            input=input,
        )
    ]
    save_trace(trace, db_path=project_dir / "db" / "runa.db")


def test_add_trace_to_evals_appends_the_run_s_input_to_its_agent_s_dataset(
    tmp_path: Path,
) -> None:
    """The trace's input lands as a new last line of `evals/<agent>.jsonl`, linked by trace id."""
    project_dir = scaffold_project("demo", root=tmp_path)
    eval_file = project_dir / "evals" / "support_agent.jsonl"
    eval_file.write_text('"an existing case"')  # no trailing newline
    _save_agent_trace(project_dir, "t1", "Where is my order?")

    assert add_trace_to_evals("t1", root=project_dir, expected="Looks it up") == eval_file

    assert [case.input for case in Dataset.from_jsonl(eval_file)] == [
        "an existing case",
        "Where is my order?",
    ]
    assert Dataset.from_jsonl(eval_file)[1] == Case(
        input="Where is my order?", expected="Looks it up", metadata={"trace_id": "t1"}
    )


def test_add_trace_to_evals_rejects_an_unknown_trace_or_one_without_input(
    tmp_path: Path,
) -> None:
    """Nothing is written for a missing trace, or one that recorded no user input."""
    project_dir = scaffold_project("demo", root=tmp_path)
    _save_agent_trace(project_dir, "t1", None)

    with pytest.raises(TraceNotFound):
        add_trace_to_evals("nope", root=project_dir)
    with pytest.raises(TraceHasNoInput):
        add_trace_to_evals("t1", root=project_dir)
    assert not (project_dir / "evals" / "support_agent.jsonl").exists()
