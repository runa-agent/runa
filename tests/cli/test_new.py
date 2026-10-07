"""Tests for `runa.cli.new`: `scaffold_project`."""

from pathlib import Path

import pytest

from runa.cli.new import ProjectAlreadyExists, scaffold_project


def test_scaffold_project_creates_the_conventional_app_layout(tmp_path: Path) -> None:
    """`scaffold_project` creates every `app/` subdir plus the project's own top-level files."""
    project_dir = scaffold_project("demo", root=tmp_path)

    assert project_dir == tmp_path / "demo"
    for subdir in ("agents", "guardrails", "prompts", "tools"):
        assert (project_dir / "app" / subdir / "__init__.py").is_file()
    for subdir in ("tests", "evals", "config"):
        assert (project_dir / subdir / "__init__.py").is_file()
    assert (project_dir / "db").is_dir()
    assert (project_dir / "docs").is_dir()
    for name in ("pyproject.toml", "main.py", "asgi.py", "Dockerfile", ".gitignore", ".env"):
        assert (project_dir / name).is_file()


def test_scaffold_project_raises_if_the_directory_already_exists(tmp_path: Path) -> None:
    """`scaffold_project` refuses to overwrite an existing directory."""
    (tmp_path / "demo").mkdir()

    with pytest.raises(ProjectAlreadyExists):
        scaffold_project("demo", root=tmp_path)


def test_scaffold_project_without_a_name_scaffolds_root_in_place(tmp_path: Path) -> None:
    """Omitting `name` scaffolds `root` itself instead of a subdirectory."""
    project_dir = scaffold_project(None, root=tmp_path)

    assert project_dir == tmp_path
    assert (project_dir / "app" / "agents" / "__init__.py").is_file()
    assert (project_dir / "main.py").is_file()
    assert f'name = "{tmp_path.name}"' in (project_dir / "pyproject.toml").read_text()


def test_scaffold_project_without_a_name_raises_if_root_already_has_an_app(
    tmp_path: Path,
) -> None:
    """Omitting `name` still refuses to clobber an existing Runa app in `root`."""
    (tmp_path / "main.py").write_text("")

    with pytest.raises(ProjectAlreadyExists):
        scaffold_project(None, root=tmp_path)


def test_main_py_only_loads_dotenv_with_no_configure_step(tmp_path: Path) -> None:
    """The generated `main.py` has no `configure()`/provider wiring left over from the old CLI."""
    project_dir = scaffold_project("demo", root=tmp_path)

    main_py = (project_dir / "main.py").read_text()

    assert "load_dotenv()" in main_py
    assert "configure(" not in main_py


def test_asgi_py_builds_the_app_from_the_public_factory(tmp_path: Path) -> None:
    """The generated `asgi.py` reaches the server the documented way, and authenticated.

    `tests/serve` proves the file actually serves; this pins the shape, so a scaffold that drifts
    into a private import or an open-by-default server fails here.
    """
    project_dir = scaffold_project("demo", root=tmp_path)

    asgi_py = (project_dir / "asgi.py").read_text()

    assert "from runa.serve import create_app, resolve_api_key" in asgi_py
    assert "resolve_api_key(no_auth=False)" in asgi_py
