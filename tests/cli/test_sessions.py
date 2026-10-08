"""Tests for `runa.cli.sessions`: formatting over whatever `runa.db.sessions()` resolved.

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
    db.use_project(root)
    session = db.session(session_id)
    asyncio.run(session.add_items([{"role": "user", "content": "hello"}]))


def test_list_sessions_reports_none_when_there_is_no_history(tmp_path: Path) -> None:
    """`list_sessions` reports no sessions when the project has no history yet."""
    db.use_project(scaffold_project("demo", root=tmp_path))

    assert list_sessions() == "no sessions found"


def test_list_sessions_lists_a_session_with_history(tmp_path: Path) -> None:
    """A session with history appears in the listing, with its timestamp."""
    _add_history(scaffold_project("demo", root=tmp_path), "SupportAgent")

    output = list_sessions()

    assert output.startswith("SupportAgent  ")


def test_list_sessions_reads_the_project_in_use(tmp_path: Path) -> None:
    """The project `db.use_project` named is the one whose `db/runa.db` gets opened.

    The convention that resolves to lives in `runa.db` now, applied to every concern at once;
    this is the test that it still holds from a caller that names no directory at all.
    """
    other = scaffold_project("other", root=tmp_path)
    _add_history(other, "SupportAgent")
    empty = scaffold_project("empty", root=tmp_path)

    db.use_project(other)
    assert "SupportAgent" in list_sessions()

    db.use_project(empty)
    assert list_sessions() == "no sessions found"


def test_show_session_raises_for_an_unknown_session(tmp_path: Path) -> None:
    """`show_session` raises `SessionNotFound` for a session id with no history."""
    db.use_project(scaffold_project("demo", root=tmp_path))

    with pytest.raises(SessionNotFound):
        show_session("nope")


def test_show_session_renders_history(tmp_path: Path) -> None:
    """`show_session` renders each message as `<timestamp>  <role>: <text>`."""
    _add_history(scaffold_project("demo", root=tmp_path), "SupportAgent")

    output = show_session("SupportAgent")

    assert output.startswith("session SupportAgent\n\n")
    assert output.rstrip().endswith("user: hello")
