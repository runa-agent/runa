"""cli/test.py: `runa test`, run tests/ test functions.

Complements `runa eval` (cli/eval.py): tests verify invariants with plain
`assert` statements against a result, evals measure behavior with
`expect(result).to_...()`. Import every `tests/` module and run its
`test_*` functions, catching `AssertionError` instead of crashing so a full
report comes back in one pass.

Deliberately not a pytest wrapper: `runa` doesn't add pytest as a runtime
dependency just so a generated app can run its own tests, matching
`run_evals()`'s choice not to depend on an external harness either.
"""

import importlib
import inspect
from collections.abc import Awaitable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from runa import _loop
from runa.project import NotARunaProject, loaded_app


@dataclass
class TestResult:
    """The outcome of running one `test_*` function."""

    name: str
    passed: bool
    error: str | None = None


async def _await(awaitable: Awaitable[Any]) -> Any:
    """Await `awaitable` inside a real coroutine, which is what `runa._loop.run_sync` takes.

    An `async def` test returns a coroutine, but a test may equally return any awaitable, and
    the task it runs as is built from a coroutine alone.
    """
    return await awaitable


def run_project_tests(root: Path) -> list[TestResult]:
    """Import every `tests/` module and run its `test_*` functions.

    `root` is the project under test, and the `loaded_app` block below says so for its database
    too: a test that drives a real run writes that project's `db/runa.db`, not whichever
    directory this was called from.
    """
    tests_dir = root / "tests"
    if not tests_dir.is_dir():
        raise NotARunaProject(
            f"{tests_dir} does not exist, run this from inside a Runa "
            "project created with `runa new`"
        )

    with loaded_app(root):
        results: list[TestResult] = []
        for test_file in sorted(tests_dir.glob("*.py")):
            if test_file.stem == "__init__":
                continue
            module = importlib.import_module(f"tests.{test_file.stem}")
            for attr_name, attr in inspect.getmembers(module, inspect.isfunction):
                if not attr_name.startswith("test_"):
                    continue
                name = f"{test_file.stem}.{attr_name}"
                try:
                    outcome = attr()
                    if inspect.isawaitable(outcome):
                        _loop.run_sync(_await(outcome))
                except AssertionError as exc:
                    results.append(TestResult(name=name, passed=False, error=str(exc)))
                else:
                    results.append(TestResult(name=name, passed=True))
        return results
