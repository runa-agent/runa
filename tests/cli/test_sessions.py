"""Tests for session history reads: `runa.cli.sessions`' formatting over `session.storage`."""

import asyncio
from pathlib import Path

import pytest

from runa.cli._project import resolve_db_path
from runa.cli.new import scaffold_project
from runa.cli.sessions import list_sessions, show_session
from runa.session import SQLiteSession
from runa.session.storage import (
    SessionNotFound,
    session_messages,
    session_rows,
    sessions_for_agent,
)


def _add_history(db_path: Path, session_id: str) -> None:
    session = SQLiteSession(session_id, db_path=db_path)
    asyncio.run(session.add_items([{"role": "user", "content": "hello"}]))


def test_list_sessions_reports_none_when_runa_db_is_empty(tmp_path: Path) -> None:
    """`list_sessions` reports no sessions when `runa.db` has no history yet."""
    project_dir = scaffold_project("demo", root=tmp_path)

    assert list_sessions(root=project_dir) == "no sessions found"


def test_list_sessions_lists_a_session_with_history(tmp_path: Path) -> None:
    """A session with history appears in the listing."""
    project_dir = scaffold_project("demo", root=tmp_path)
    _add_history(project_dir / "db" / "runa.db", "SupportAgent")

    assert "SupportAgent" in list_sessions(root=project_dir)


def test_sessions_for_agent_matches_prefixed_and_exact_ids(tmp_path: Path) -> None:
    """`sessions_for_agent` finds both `Support-<suffix>` chats and a bare `Support` one.

    It ignores sessions belonging to a different agent.
    """
    project_dir = scaffold_project("demo", root=tmp_path)
    db_path = project_dir / "db" / "runa.db"
    _add_history(db_path, "Support-20260101-000000-aaaa")
    _add_history(db_path, "Support")
    _add_history(db_path, "OtherAgent-20260101-000000-bbbb")

    sessions = sessions_for_agent("Support", db_path=resolve_db_path(project_dir))

    ids = [session_id for session_id, _ in sessions]
    assert set(ids) == {"Support-20260101-000000-aaaa", "Support"}


def test_sessions_for_agent_reports_none_when_there_is_no_history(tmp_path: Path) -> None:
    """`sessions_for_agent` returns an empty list when `runa.db` has no matching session."""
    project_dir = scaffold_project("demo", root=tmp_path)

    assert sessions_for_agent("Support", db_path=resolve_db_path(project_dir)) == []


def test_show_session_raises_for_an_unknown_session(tmp_path: Path) -> None:
    """`show_session` raises `SessionNotFound` for a session id with no history."""
    project_dir = scaffold_project("demo", root=tmp_path)

    with pytest.raises(SessionNotFound):
        show_session("nope", root=project_dir)


def test_show_session_renders_history(tmp_path: Path) -> None:
    """`show_session` renders the session's messages."""
    project_dir = scaffold_project("demo", root=tmp_path)
    _add_history(project_dir / "db" / "runa.db", "SupportAgent")

    output = show_session("SupportAgent", root=project_dir)

    assert "hello" in output


def test_session_rows_reports_no_rows_when_runa_db_is_empty(tmp_path: Path) -> None:
    """`session_rows` returns an empty list when `runa.db` has no history yet."""
    project_dir = scaffold_project("demo", root=tmp_path)

    assert session_rows(db_path=resolve_db_path(project_dir)) == []


def test_session_rows_returns_id_and_updated_at_pairs(tmp_path: Path) -> None:
    """`session_rows` returns every session as an `(id, updated_at)` pair."""
    project_dir = scaffold_project("demo", root=tmp_path)
    _add_history(project_dir / "db" / "runa.db", "SupportAgent")

    rows = session_rows(db_path=resolve_db_path(project_dir))

    assert [session_id for session_id, _ in rows] == ["SupportAgent"]


def test_session_messages_raises_for_an_unknown_session(tmp_path: Path) -> None:
    """`session_messages` raises `SessionNotFound` for a session id with no history."""
    project_dir = scaffold_project("demo", root=tmp_path)

    with pytest.raises(SessionNotFound):
        session_messages("nope", db_path=resolve_db_path(project_dir))


def test_session_messages_returns_role_and_text_per_message(tmp_path: Path) -> None:
    """`session_messages` splits each message into its `role` and `text`."""
    project_dir = scaffold_project("demo", root=tmp_path)
    _add_history(project_dir / "db" / "runa.db", "SupportAgent")

    messages = session_messages("SupportAgent", db_path=resolve_db_path(project_dir))

    assert messages == [{"created_at": messages[0]["created_at"], "role": "user", "text": "hello"}]
