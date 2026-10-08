"""eval/corpus.py: the `evals/` directory as data -- read a case, add a case.

A dataset grows from real failures, not invented ones, so "this answer was wrong" has to become
a line of `evals/<agent_name>.jsonl` without anyone hand-editing the file. Two surfaces offer
that -- `runa eval --add TRACE_ID` and "Add to evals" on `runa ui`'s trace page -- and both ask
the same three questions: what did this run start with, is it already a case, where does it go.

Running those datasets is the other half, and it lives in `cli/eval.py` with the command that
asks for it. This module only reads and writes the files.
"""

import json
from pathlib import Path

from runa import db
from runa.eval.dataset import Dataset
from runa.exceptions import OperatorError
from runa.project import NotARunaProject
from runa.tracing import Trace, TraceNotFound


class TraceHasNoInput(OperatorError):
    """Raised when a trace recorded no user input to replay as an eval case."""


class CaseAlreadyInEvals(OperatorError):
    """Raised when a trace's input is already a case in its agent's `evals/` dataset."""


def require_evals_dir(root: Path) -> Path:
    """Return `root/evals`, raising `NotARunaProject` if it doesn't exist."""
    evals_dir = root / "evals"
    if not evals_dir.is_dir():
        raise NotARunaProject(
            f"{evals_dir} does not exist, run this from inside a Runa "
            "project created with `runa new`"
        )
    return evals_dir


def traced_input(trace: Trace) -> tuple[str, str] | None:
    """The `(agent_name, user input)` a trace's run started with, `None` if it recorded none."""
    agent_span = next(
        (span for span in trace.spans if span.parent_id is None and span.type == "agent"), None
    )
    if agent_span is None or not isinstance(agent_span.input, str) or not agent_span.input:
        return None
    return agent_span.name, agent_span.input


def has_case(eval_file: Path, input: str) -> bool:
    """Whether `eval_file` already holds a case with this `input`."""
    return eval_file.exists() and any(case.input == input for case in Dataset.from_jsonl(eval_file))


def add_trace_to_evals(trace_id: str, *, root: Path, expected: str | None = None) -> Path:
    """Append the traced run's input as a new case to `evals/<agent_name>.jsonl`.

    The agent is the trace's root agent span, the one the run started with. `expected` says what a
    good answer would have been; without it the case still grades task completion and relevance.
    The case's `metadata` keeps `trace_id`, so a failing case links back to the run it came from.
    An input already in the file raises `CaseAlreadyInEvals` rather than adding it twice.
    """
    evals_dir = require_evals_dir(root)
    trace = db.traces().get(trace_id)
    if trace is None:
        raise TraceNotFound(f"no trace found with id {trace_id!r}")
    traced = traced_input(trace)
    if traced is None:
        raise TraceHasNoInput(f"trace {trace_id!r} recorded no user input to add as a case")
    agent_name, input = traced

    case: dict[str, object] = {"input": input}
    if expected:
        case["expected"] = expected
    case["metadata"] = {"trace_id": trace.id}

    eval_file = evals_dir / f"{agent_name}.jsonl"
    if has_case(eval_file, input):
        raise CaseAlreadyInEvals(f"{input!r} is already a case in {eval_file}")
    existing = eval_file.read_text() if eval_file.exists() else ""
    separator = "\n" if existing and not existing.endswith("\n") else ""
    eval_file.write_text(f"{existing}{separator}{json.dumps(case, ensure_ascii=False)}\n")
    return eval_file


__all__ = [
    "CaseAlreadyInEvals",
    "TraceHasNoInput",
    "add_trace_to_evals",
    "has_case",
    "require_evals_dir",
    "traced_input",
]
