"""Tests for `runa.cli.test`: `run_project_tests`."""

from pathlib import Path

import pytest

from runa.cli.new import scaffold_project
from runa.cli.test import run_project_tests
from runa.project import NotARunaProject


def _write_test_module(project_dir: Path, filename: str, source: str) -> None:
    (project_dir / "tests" / filename).write_text(source)


def test_run_project_tests_raises_outside_a_runa_project(tmp_path: Path) -> None:
    """`run_project_tests` refuses to run where `tests/` doesn't exist."""
    with pytest.raises(NotARunaProject):
        run_project_tests(tmp_path)


def test_run_project_tests_reports_pass_and_fail(tmp_path: Path) -> None:
    """A passing and a failing `test_*` function are both reported, without stopping early."""
    project_dir = scaffold_project("demo", root=tmp_path)
    # Named to avoid pytest's own assertion-rewrite hook, which would otherwise intercept this
    # dynamically-imported module too (it matches pytest's default `test_*.py` pattern) and
    # rewrite `assert False, "nope"` into something richer than the plain message asserted below.
    _write_test_module(
        project_dir,
        "smoke_cases.py",
        "def test_ok():\n    assert True\n\n\ndef test_bad():\n    assert False, 'nope'\n",
    )

    results = run_project_tests(project_dir)

    by_name = {r.name: r for r in results}
    assert by_name["smoke_cases.test_ok"].passed
    assert not by_name["smoke_cases.test_bad"].passed
    assert by_name["smoke_cases.test_bad"].error == "nope"


def test_run_project_tests_awaits_async_test_functions(tmp_path: Path) -> None:
    """An `async def test_*` function is awaited, not skipped as a coroutine object."""
    project_dir = scaffold_project("demo", root=tmp_path)
    _write_test_module(
        project_dir,
        "smoke_async.py",
        "async def test_ok():\n    assert True\n",
    )

    results = run_project_tests(project_dir)

    assert results[0].passed
