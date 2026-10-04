"""The session contract, both sides of it, run against every adapter that needs no server.

The checks live in `tests/contracts/session.py` and are driven here over the backends `runa.db` can
resolve without a live Postgres. Both halves come from a project root, resolved the way an app's
would be -- the write side through `db.session(...)`, the read side through `db.sessions(...)` --
which is the pairing these tests are really about: a run appends through one and
`runa chat --show` reads it back through the other.

`tests/test_postgres.py` runs the same checks against the Postgres pair.

The tie-break on `updated_at` is asserted here rather than in the contract, because only a
second-resolution clock produces the tie at all; see `test_listing_breaks_updated_at_ties_...`.
"""

import asyncio
from pathlib import Path

import pytest
from contracts.session import CONTRACT, Check, SessionPair

from runa import db


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


def _pair(root: Path, tag: str = "SupportAgent") -> SessionPair:
    """Both sides of `root`'s session backend, as the contract expects them."""
    return SessionPair(
        write=lambda session_id: db.session(session_id, root=root), read=db.sessions(root), tag=tag
    )


@pytest.mark.parametrize("check", CONTRACT, ids=lambda check: check.__name__)
def test_session_contract(root: Path, check: Check) -> None:
    """Every local backend answers the session contract the same way, on both sides."""
    asyncio.run(check(_pair(root)))


def test_listing_is_empty_before_anything_is_written(root: Path) -> None:
    """A deployment with no history lists nothing, rather than failing on a missing table.

    Outside the contract because it is an assertion about an untouched deployment, which a shared
    database running the same checks is not.
    """
    assert db.sessions(root).listing() == []


def test_listing_breaks_updated_at_ties_by_session_id_descending(root: Path) -> None:
    """Two sessions written in the same second list in one order, not the backend's own.

    SQLite broke this tie on `rowid` and Postgres on `session_id`, so `runa chat --list` and the
    Sessions page disagreed between a local deployment and a shared one. Asserted over the
    backends whose clock has second resolution, which is what makes the tie reproducible: Postgres
    timestamps three consecutive writes distinctly and orders them by time alone.
    """
    pair = _pair(root)
    for suffix in ("-a", "-c", "-b"):
        asyncio.run(pair.session(suffix).add_items([{"role": "user", "content": "hi"}]))

    listed = pair.read.listing(agent=pair.tag)

    assert [summary.id for summary in listed] == [pair.id("-c"), pair.id("-b"), pair.id("-a")]
