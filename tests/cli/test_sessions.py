"""Tests for `runa.cli.sessions`: formatting over whatever `runa.db.sessions(root)` resolved.

The store's own behaviour -- ordering, timestamps, agent matching, `SessionNotFound` -- is
`tests/test_session_store.py`'s, run against every backend. What's left here is this module's
only job: turning a listing and a transcript into lines.
"""

import asyncio
from pathlib import Path

import pytest

from runa import db
from runa.cli.new import scaffold_project
from runa.cli.sessions import SessionNotFound, list_sessions, show_session


def _add_history(root: Path, session_id: str) -> None:
    session = db.session(session_id, root=root)
    asyncio.run(session.add_items([{"role": "user", "content": "hello"}]))


def test_list_sessions_reports_none_when_there_is_no_history(tmp_path: Path) -> None:
    """`list_sessions` reports no sessions when the project has no history yet."""
    project_dir = scaffold_project("demo", root=tmp_path)

    assert list_sessions(root=project_dir) == "no sessions found"


def test_list_sessions_lists_a_session_with_history(tmp_path: Path) -> None:
    """A session with history appears in the listing, with its timestamp."""
    project_dir = scaffold_project("demo", root=tmp_path)
    _add_history(project_dir, "SupportAgent")

    output = list_sessions(root=project_dir)

    assert output.startswith("SupportAgent  ")


def test_list_sessions_reads_the_project_at_root(tmp_path: Path) -> None:
    """`root` names the project being read, and its `db/runa.db` is the one opened.

    The convention that resolves to lives in `runa.db.sqlite_path(root)` now; this is the test
    that it still holds from a caller that only ever names a directory.
    """
    other = scaffold_project("other", root=tmp_path)
    _add_history(other, "SupportAgent")
    empty = scaffold_project("empty", root=tmp_path)

    assert "SupportAgent" in list_sessions(root=other)
    assert list_sessions(root=empty) == "no sessions found"


def test_show_session_raises_for_an_unknown_session(tmp_path: Path) -> None:
    """`show_session` raises `SessionNotFound` for a session id with no history."""
    project_dir = scaffold_project("demo", root=tmp_path)

    with pytest.raises(SessionNotFound):
        show_session("nope", root=project_dir)


def test_show_session_renders_history(tmp_path: Path) -> None:
    """`show_session` renders each message as `<timestamp>  <role>: <text>`."""
    project_dir = scaffold_project("demo", root=tmp_path)
    _add_history(project_dir, "SupportAgent")

    output = show_session("SupportAgent", root=project_dir)

    assert output.startswith("session SupportAgent\n\n")
    assert output.rstrip().endswith("user: hello")
