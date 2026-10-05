"""Tests for `runa.eval.corpus`: `add_trace_to_evals`."""

from pathlib import Path

import pytest

from runa import db
from runa.cli.new import scaffold_project
from runa.eval import Case, Dataset
from runa.eval.corpus import CaseAlreadyInEvals, TraceHasNoInput, add_trace_to_evals
from runa.tracing import Span, Trace, TraceNotFound


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
    db.traces(project_dir).save(trace)


def test_add_trace_to_evals_appends_the_run_s_input_to_its_agent_s_dataset(
    tmp_path: Path,
) -> None:
    """The trace's input is appended once to `evals/<agent>.jsonl`, linked by trace id."""
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
    with pytest.raises(CaseAlreadyInEvals):
        add_trace_to_evals("t1", root=project_dir)
    assert len(Dataset.from_jsonl(eval_file)) == 2


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
