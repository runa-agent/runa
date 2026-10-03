"""cli/eval.py: `runa eval`, run evals/ datasets.

A thin loop that hands each `evals/` dataset to `agent.evaluate()`, the same code path
production evaluation runs through, not a parallel CLI-only harness. By convention
`evals/<agent_name>.jsonl` is a whole eval on its own: the filename is the Agent's declared
`name`, each line a `Case`. A Python module (`evals/*.py` declaring module-level `agent` and
`dataset`) is the escape hatch, and owns any `.jsonl` sharing its stem. `add_trace_to_evals()`
turns a real run into a new line of that file, from "this answer was wrong" to a case guarding it.
"""

import asyncio
import importlib
import json
from collections.abc import Iterable
from pathlib import Path

from runa.agent import Agent
from runa.cli._project import NotARunaProject, loaded_app, require_agents_dir, resolve_db_path
from runa.cli.chat import AgentNotFound, find_agent_class
from runa.cli.traces import TraceNotFound
from runa.eval import Case, Dataset, Report
from runa.tracing import Trace, get_trace


class InvalidEvalModule(Exception):
    """Raised when an `evals/` module doesn't declare `agent` and `dataset`."""


class TraceHasNoInput(Exception):
    """Raised when a trace recorded no user input to replay as an eval case."""


class CaseAlreadyInEvals(Exception):
    """Raised when a trace's input is already a case in its agent's `evals/` dataset."""


def _require_evals_dir(root: Path) -> Path:
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
    evals_dir = _require_evals_dir(root)
    trace = get_trace(trace_id, db_path=resolve_db_path(root))
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


def _load_module(eval_file: Path) -> tuple[Agent, Iterable[Case]]:
    module = importlib.import_module(f"evals.{eval_file.stem}")
    agent = getattr(module, "agent", None)
    dataset = getattr(module, "dataset", None)
    if agent is None or dataset is None:
        raise InvalidEvalModule(f"{eval_file} must define module-level `agent` and `dataset`")
    return agent, dataset


def run_project_evals(root: Path, agent_name: str | None = None) -> list[Report]:
    """Evaluate every `evals/` dataset against its agent.

    `agent_name`, when given, filters this down to the dataset(s) whose agent declares that
    `name` (the same identity `runa chat <name>` takes) instead of running the whole `evals/`
    directory.
    """
    evals_dir = _require_evals_dir(root)

    with loaded_app(root):
        modules = [
            _load_module(eval_file)
            for eval_file in sorted(evals_dir.glob("*.py"))
            if eval_file.stem != "__init__"
        ]
        module_stems = {path.stem for path in evals_dir.glob("*.py")}
        jsonl_files = [
            path for path in sorted(evals_dir.glob("*.jsonl")) if path.stem not in module_stems
        ]
        if jsonl_files:
            agents_dir = require_agents_dir(root)
            for path in jsonl_files:
                try:
                    agent_cls = find_agent_class(path.stem, agents_dir=agents_dir)
                except AgentNotFound as exc:
                    raise AgentNotFound(f"{path}: {exc}, rename it to an Agent's `name`") from exc
                modules.append((agent_cls(), Dataset.from_jsonl(path)))

        if agent_name is not None:
            modules = [(agent, dataset) for agent, dataset in modules if agent.name == agent_name]
            if not modules:
                raise AgentNotFound(f"no evals/ dataset found for Agent named {agent_name!r}")

        async def _run_all() -> list[Report]:
            return [await agent.evaluate(dataset) for agent, dataset in modules]

        return asyncio.run(_run_all())
