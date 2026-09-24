"""cli/eval.py: `runa eval`, run evals/ datasets.

A thin loop that hands each `evals/` dataset to `agent.evaluate()`, the same code path
production evaluation runs through, not a parallel CLI-only harness. By convention
`evals/<agent_name>.jsonl` is a whole eval on its own: the filename is the Agent's declared
`name`, each line a `Case`. A Python module (`evals/*.py` declaring module-level `agent` and
`dataset`) is the escape hatch, and owns any `.jsonl` sharing its stem.
"""

import asyncio
import importlib
from collections.abc import Iterable
from pathlib import Path

from runa.agent import Agent
from runa.cli._project import NotARunaProject, loaded_app, require_agents_dir
from runa.cli.chat import AgentNotFound, find_agent_class
from runa.eval import Case, Dataset, Report


class InvalidEvalModule(Exception):
    """Raised when an `evals/` module doesn't declare `agent` and `dataset`."""


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
    evals_dir = root / "evals"
    if not evals_dir.is_dir():
        raise NotARunaProject(
            f"{evals_dir} does not exist, run this from inside a Runa "
            "project created with `runa new`"
        )

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
