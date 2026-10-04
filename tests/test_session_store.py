"""The `SessionStore` contract, run against every adapter that needs no server.

One set of assertions, parametrized over the backends `runa.db` can resolve without a live
Postgres. The write side comes from `db.session(...)` and the read side from `db.sessions(...)`,
both resolved the same way an app's would be, which is the pairing these tests are really about:
a run appends through one and `runa chat --show` reads it back through the other.

`tests/test_postgres.py` runs the same contract against the Postgres pair.
"""

import asyncio
from pathlib import Path

import pytest

from runa import db
from runa.session.store import SessionNotFound, SessionStore


@pytest.fixture(params=["sqlite", "ephemeral"])
def root(request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A project root whose store is the parametrized backend.

    The tests take a root rather than a store because both sides of the session backend are
    resolved from it, which is what a CLI command does with the project it is run in.
    """
    if request.param == "ephemeral":
        monkeypatch.setenv(db.DATABASE_URL_ENV, "memory://")
    else:
        monkeypatch.delenv(db.DATABASE_URL_ENV, raising=False)
    return tmp_path


def _store(root: Path) -> SessionStore:
    return db.sessions(root)


def _add_history(root: Path, session_id: str, text: str = "hello") -> None:
    session = db.session(session_id, root=root)
    asyncio.run(session.add_items([{"role": "user", "content": text}]))


def test_listing_is_empty_before_anything_is_written(root: Path) -> None:
    """A deployment with no history lists nothing, rather than failing on a missing table."""
    assert _store(root).listing() == []


def test_listing_returns_each_session_with_its_timestamp(root: Path) -> None:
    """Every written session appears once, with an `updated_at`."""
    _add_history(root, "SupportAgent")

    sessions = _store(root).listing()

    assert [session.id for session in sessions] == ["SupportAgent"]
    assert sessions[0].updated_at


def test_listing_renders_updated_at_the_same_way_in_every_backend(root: Path) -> None:
    """`updated_at` is `"YYYY-MM-DD HH:MM:SS"`, with no offset, whichever store wrote it.

    The Postgres side used to hand back a `TIMESTAMPTZ`'s offset-bearing ISO string where SQLite's
    text had none, so the same session read differently depending on the deployment.
    """
    _add_history(root, "SupportAgent")

    updated_at = _store(root).listing()[0].updated_at

    assert len(updated_at) == len("2026-10-04 10:08:03")
    assert updated_at[4] == updated_at[7] == "-"
    assert updated_at[10] == " "
    assert "+" not in updated_at


def test_listing_by_agent_matches_prefixed_and_exact_ids(root: Path) -> None:
    """`listing(agent=...)` finds both `Support-<suffix>` chats and a bare `Support` one."""
    _add_history(root, "Support-20260101-000000-aaaa")
    _add_history(root, "Support")
    _add_history(root, "OtherAgent-20260101-000000-bbbb")

    sessions = _store(root).listing(agent="Support")

    assert {session.id for session in sessions} == {"Support-20260101-000000-aaaa", "Support"}


def test_listing_by_agent_does_not_treat_underscores_as_wildcards(root: Path) -> None:
    """An agent named `a_b` doesn't match a session of `axb`, in any backend."""
    _add_history(root, "axb-1")

    assert _store(root).listing(agent="a_b") == []


def test_listing_by_agent_is_empty_when_there_is_no_history(root: Path) -> None:
    """`listing(agent=...)` returns an empty list when nothing matches."""
    assert _store(root).listing(agent="Support") == []


def test_listing_breaks_updated_at_ties_by_session_id_descending(root: Path) -> None:
    """Two sessions written in the same second list in one order, not the backend's own.

    SQLite broke this tie on `rowid` and Postgres on `session_id`, so `runa chat --list` and the
    Sessions page disagreed between a local deployment and a shared one.
    """
    for session_id in ("Support-a", "Support-c", "Support-b"):
        _add_history(root, session_id)

    sessions = _store(root).listing(agent="Support")

    assert [session.id for session in sessions] == ["Support-c", "Support-b", "Support-a"]


def test_messages_raises_for_an_unknown_session(root: Path) -> None:
    """`messages` raises `SessionNotFound` rather than returning an empty transcript."""
    with pytest.raises(SessionNotFound):
        _store(root).messages("nope")


def test_messages_splits_each_message_into_role_and_text(root: Path) -> None:
    """`messages` flattens a stored item into the `(role, text)` a transcript shows."""
    _add_history(root, "SupportAgent", text="hello")

    messages = _store(root).messages("SupportAgent")

    assert [(message.role, message.text) for message in messages] == [("user", "hello")]
    assert messages[0].created_at


def test_messages_are_oldest_first(root: Path) -> None:
    """A transcript reads in the order it was written."""
    session = db.session("SupportAgent", root=root)
    asyncio.run(
        session.add_items(
            [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "second"},
            ]
        )
    )

    messages = _store(root).messages("SupportAgent")

    assert [message.text for message in messages] == ["first", "second"]


def test_messages_flattens_a_list_shaped_content(root: Path) -> None:
    """A content list of text parts is joined, the shape a model's own reply is stored in."""
    session = db.session("SupportAgent", root=root)
    asyncio.run(
        session.add_items([{"role": "assistant", "content": [{"text": "one "}, {"text": "two"}]}])
    )

    assert _store(root).messages("SupportAgent")[0].text == "one two"
