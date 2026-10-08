"""The session contract, both sides of it, run against every adapter that needs no server.

The checks live in `tests/contracts/session.py` and are driven here over the backends `runa.db` can
resolve without a live Postgres. Both halves are resolved the way an app's would be -- the write
side through `db.session(...)`, the read side through `db.sessions(...)`, over one project this
fixture points `runa.db` at -- which is the pairing these tests are really about: a run appends
through one and `runa chat --show` reads it back through the other.

`tests/test_postgres.py` runs the same checks against the Postgres pair.

The tie-break on `updated_at` is asserted here rather than in the contract, because only a
second-resolution clock produces the tie at all; see `test_listing_breaks_updated_at_ties_...`.
"""

import asyncio
from pathlib import Path

import pytest
from contracts.session import CONTRACT, Check, SessionPair

from runa import db
from runa.db.schema import POSTGRES, SQLITE, Dialect
from runa.session.store import agent_filter


@pytest.fixture(params=["sqlite", "ephemeral"])
def project(
    request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Point `runa.db` at a project whose store is the parametrized backend.

    The tests name no path because neither side of the session backend takes one: a CLI command
    tells `runa.db` which project it is running in, once, and both halves follow.
    """
    if request.param == "ephemeral":
        monkeypatch.setenv(db.DATABASE_URL_ENV, "memory://")
    else:
        monkeypatch.delenv(db.DATABASE_URL_ENV, raising=False)
    db.use_project(tmp_path)


def _pair(tag: str = "SupportAgent") -> SessionPair:
    """Both sides of the active project's session backend, as the contract expects them."""
    return SessionPair(write=db.session, read=db.sessions(), tag=tag)


@pytest.mark.parametrize("check", CONTRACT, ids=lambda check: check.__name__)
def test_session_contract(project: None, check: Check) -> None:
    """Every local backend answers the session contract the same way, on both sides."""
    asyncio.run(check(_pair()))


def test_listing_is_empty_before_anything_is_written(project: None) -> None:
    """A deployment with no history lists nothing, rather than failing on a missing table.

    Outside the contract because it is an assertion about an untouched deployment, which a shared
    database running the same checks is not.
    """
    assert db.sessions().listing() == []


def test_listing_breaks_updated_at_ties_by_session_id_descending(project: None) -> None:
    """Two sessions written in the same second list in one order, not the backend's own.

    SQLite broke this tie on `rowid` and Postgres on `session_id`, so `runa chat --list` and the
    Sessions page disagreed between a local deployment and a shared one. Asserted over the
    backends whose clock has second resolution, which is what makes the tie reproducible: Postgres
    timestamps three consecutive writes distinctly and orders them by time alone.
    """
    pair = _pair()
    for suffix in ("-a", "-c", "-b"):
        asyncio.run(pair.session(suffix).add_items([{"role": "user", "content": "hi"}]))

    listed = pair.read.listing(agent=pair.tag)

    assert [summary.id for summary in listed] == [pair.id("-c"), pair.id("-b"), pair.id("-a")]


@pytest.mark.parametrize("dialect", [SQLITE, POSTGRES], ids=lambda dialect: dialect.name)
def test_agent_filter_carries_its_escape_clause_in_every_dialect(dialect: Dialect) -> None:
    """The escaped pattern and the `ESCAPE` clause that gives it meaning come back together.

    Both are one value because an adapter handed the pattern alone can bind it without the clause,
    which is how the Postgres listing was written: correct only because a backslash happens to be
    its default escape character. Asserted per dialect rather than per adapter, since the live
    Postgres half of the contract is `tests/test_postgres.py`.
    """
    where, params = agent_filter("a_b", dialect)

    assert "ESCAPE '\\'" in where
    assert params == ("a_b", "a\\_b-%")


@pytest.mark.parametrize("dialect", [SQLITE, POSTGRES], ids=lambda dialect: dialect.name)
def test_agent_filter_is_no_clause_at_all_for_no_agent(dialect: Dialect) -> None:
    """`agent=None` filters nothing, so a listing composes one string either way."""
    assert agent_filter(None, dialect) == ("", ())
