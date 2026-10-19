"""Tests that the prose in `src/runa/` names only what its own module defines, or is public.

A private name is private so it can be renamed with one edit. A docstring or comment that cites
another module's private name in backticks makes that rename a search across the codebase, and
the citation rots silently when the search misses it.
"""

import ast
import io
import re
import tokenize
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src" / "runa"

_CITATION = re.compile(r"`([A-Za-z_][\w./]*)`")


def _modules() -> set[str]:
    """Every module under `src/runa/` as a dotted path relative to `runa`, packages included."""
    found: set[str] = set()
    for path in _SRC.rglob("*.py"):
        parts = list(path.relative_to(_SRC).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        for end in range(1, len(parts) + 1):
            found.add(".".join(parts[:end]))
    return found


def _defined_in(tree: ast.Module) -> set[str]:
    """Every name a module binds itself: functions, classes, assignments, arguments, attributes."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store):
            names.add(node.attr)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
    return names


def _prose(source: str, tree: ast.Module) -> list[tuple[int, str]]:
    """Every comment and docstring line of a module, with its line number."""
    lines: list[tuple[int, str]] = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT:
            lines.append((token.start[0], token.string))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        docstring = ast.get_docstring(node, clean=False)
        if docstring is None:
            continue
        first = node.body[0].lineno
        lines.extend((first + offset, line) for offset, line in enumerate(docstring.splitlines()))
    return lines


def _foreign_private(
    citation: str, here: str, package: str, modules: set[str], own: set[str]
) -> bool:
    """Whether `citation` names a private attribute some other module defines.

    Leading segments that spell a module (`runa._loop`, `run_internal/run_loop`, a sibling's
    `_html.py`) are a path, not a symbol: a private module is the package's internal API. The
    segments after them are symbols, and a private one is foreign unless the path leads back
    here or, with no path at all, this module defines it.
    """
    segments = [s for s in re.split(r"[./]", citation.removesuffix(".py")) if s]
    if segments[:1] == ["runa"]:
        segments = segments[1:]
    elif package and f"{package}.{segments[0]}" in modules:
        segments = [*package.split("."), *segments]
    module = ""
    while segments and ".".join([*filter(None, [module]), segments[0]]) in modules:
        module = ".".join([*filter(None, [module]), segments.pop(0)])
    private = [s for s in segments if s.startswith("_") and not s.startswith("__")]
    if not private:
        return False
    if module:
        return module != here
    return any(name not in own for name in private)


def test_prose_never_cites_another_modules_private_names() -> None:
    """No comment or docstring backticks a `_name` that only another module defines."""
    modules = _modules()
    offenders: list[str] = []
    for path in sorted(_SRC.rglob("*.py")):
        source = path.read_text()
        tree = ast.parse(source)
        parts = list(path.relative_to(_SRC).with_suffix("").parts)
        package = ".".join(parts[:-1])
        here = package if parts[-1] == "__init__" else ".".join(parts)
        own = _defined_in(tree)
        for line_number, line in _prose(source, tree):
            for citation in _CITATION.findall(line):
                if _foreign_private(citation, here, package, modules, own):
                    offenders.append(f"{path.relative_to(_SRC)}:{line_number}: `{citation}`")
    assert not offenders, "prose cites another module's private name:\n" + "\n".join(offenders)
