"""Tests for `runa.exceptions`: the two roots, and that every failure declares which one it is.

`cli/main.py` used to decide which failures were the operator's by listing thirteen types
imported from nine modules. The classification now lives at each declaration, which is what
makes it checkable here: `test_no_unclassified_exception_types` is the assertion that list
could never be, since it covers the surfaces the CLI doesn't reach.
"""

import ast
from pathlib import Path

import pytest

from runa.cli.main import main
from runa.exceptions import OperatorError, RunaError

_SRC = Path(__file__).resolve().parents[1] / "src" / "runa"

_UNCLASSIFIED_BY_DESIGN = {
    # The two roots themselves.
    "OperatorError",
    "RunaError",
    # A control-flow signal, not a failure: `handoff.py` raises it to unwind a paused delegate.
    "DelegatePaused",
    # The operator typed nothing wrong, their `main.py` is broken, and it earns the message
    # that says so rather than the one every operator error shares. See `runa.project`.
    "AppLoadError",
}


def _classes_subclassing_exception_directly() -> dict[str, Path]:
    """Every `class X(Exception)` declared under `src/runa/`, by name.

    Read with `ast` rather than by importing: a module behind an optional extra (`web/`,
    `serve.py`) would otherwise go unchecked on a plain install, and those are exactly the
    surfaces whose errors the CLI never catches.
    """
    found: dict[str, Path] = {}
    for path in sorted(_SRC.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.ClassDef):
                continue
            if any(isinstance(b, ast.Name) and b.id == "Exception" for b in node.bases):
                found[node.name] = path
    return found


def test_no_unclassified_exception_types() -> None:
    """Nothing subclasses `Exception` directly without being one of the documented few.

    A new failure is either a run failure (`RunaError`) or the operator's input
    (`OperatorError`), and saying which is the declaring module's job -- it is the one that
    knows. Subclassing `Exception` instead means no surface handles it: the CLI prints a
    traceback for a typo and `runa ui` returns a 500.
    """
    unclassified = _classes_subclassing_exception_directly()

    unexpected = {
        name: str(path.relative_to(_SRC))
        for name, path in unclassified.items()
        if name not in _UNCLASSIFIED_BY_DESIGN
    }
    assert not unexpected, (
        f"these subclass Exception directly: {unexpected}. Subclass OperatorError (the "
        "operator's input) or RunaError (a run failure), or add it to "
        "_UNCLASSIFIED_BY_DESIGN with the reason."
    )


def test_operator_error_is_not_a_run_failure() -> None:
    """`OperatorError` is a sibling of `RunaError`, so `Agent.run` never swallows one.

    A run catches `RunaError` and turns it into `Run(status="error")`. A mistyped session id
    is not a run failure, and must not come back as one.
    """
    assert not issubclass(OperatorError, RunaError)
    assert not issubclass(RunaError, OperatorError)


def test_a_new_operator_error_gets_the_clean_cli_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An `OperatorError` subclass `cli/main.py` has never heard of still exits 1 cleanly.

    The point of the base: declaring the classification is enough, with nothing to remember
    to add to `main`.
    """

    class _BrandNewOperatorError(OperatorError):
        """A type declared after `cli/main.py` was last edited."""

    def _raise(*_args: object, **_kwargs: object) -> Path:
        raise _BrandNewOperatorError("that name is taken")

    monkeypatch.setattr("runa.cli.main.scaffold_project", _raise)

    exit_code = main(["new", "demo"], cwd=tmp_path)

    assert exit_code == 1
    assert capsys.readouterr().err == "error: that name is taken\n"
