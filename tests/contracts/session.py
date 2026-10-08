"""contracts/session.py: the one session contract, both sides of it, for every backend.

A session has two halves in Runa: `Session`, which a run appends to, and `SessionStore`, which
`runa sessions`, `runa chat --list/--show` and the Sessions page read it back through. They are
only useful as a pair -- a run writes through one and an operator reads through the other -- so
one contract covers both, and a check gets a `SessionPair` rather than a single store.

Every check is a coroutine: the write side is async in every backend, while the read side is sync
and simply called inside it. Each check takes its ids from `SessionPair.id`, which prefixes them
with a tag unique to that check, so the same checks hold against a live Postgres whose
`agent_sessions` table already holds other runs' rows.

The one rule deliberately left out is how ties on `updated_at` break. Every backend orders a
listing by `updated_at` descending and breaks ties on session id descending, but only a
second-resolution clock (SQLite's `CURRENT_TIMESTAMP`, and the ephemeral store, which matches it)
produces the tie at all; Postgres timestamps three writes distinctly. It is asserted in
`tests/test_session_store.py`, against the backends that can actually produce it.
"""

from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any

import pytest

from runa.session import Session
from runa.session.store import SessionNotFound, SessionStore


@dataclass(frozen=True)
class SessionPair:
    """One deployment's session backend: the write side, the read side, and a unique id prefix.

    `write` is a factory rather than a session because most checks need more than one session id,
    and `runa.db.session(...)` is how an app gets each of them.
    """

    write: Callable[[str], Session]
    read: SessionStore
    tag: str

    def id(self, suffix: str = "") -> str:
        """One session id of this check's own, unique to it."""
        return f"{self.tag}{suffix}"

    def session(self, suffix: str = "") -> Session:
        """The write side for one of this check's session ids."""
        return self.write(self.id(suffix))


Check = Callable[[SessionPair], Coroutine[Any, Any, None]]


async def _write(pair: SessionPair, suffix: str, *texts: str) -> Session:
    """One session with one user message per text, through the write side."""
    session = pair.session(suffix)
    await session.add_items([{"role": "user", "content": text} for text in texts])
    return session


async def check_add_items_then_get_items_round_trips_in_order(pair: SessionPair) -> None:
    """Items come back oldest-first, matching the order they were added in."""
    session = pair.session()
    await session.add_items([{"role": "user", "content": "hi"}])
    await session.add_items([{"role": "assistant", "content": "hello"}])

    assert await session.get_items() == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]


async def check_get_items_with_limit_returns_the_latest_n_in_order(pair: SessionPair) -> None:
    """`limit` returns the most recent items, still oldest-first."""
    session = await _write(pair, "", "0", "1", "2")

    items = await session.get_items(limit=2)

    assert [item["content"] for item in items] == ["1", "2"]


async def check_pop_item_removes_and_returns_the_most_recent_item(pair: SessionPair) -> None:
    """`pop_item` removes the last item added and returns it."""
    session = await _write(pair, "", "first", "second")

    popped = await session.pop_item()

    assert popped == {"role": "user", "content": "second"}
    assert await session.get_items() == [{"role": "user", "content": "first"}]


async def check_pop_item_on_an_empty_session_returns_none(pair: SessionPair) -> None:
    """Popping from a session with no history returns `None`, not an error."""
    assert await pair.session().pop_item() is None


async def check_set_items_replaces_the_entire_history(pair: SessionPair) -> None:
    """`set_items` drops whatever was stored and writes `items` in its place."""
    session = await _write(pair, "", "old", "older")
    await session.set_items([{"role": "user", "content": "new"}])

    assert await session.get_items() == [{"role": "user", "content": "new"}]


async def check_clear_session_removes_everything_and_can_be_written_to_again(
    pair: SessionPair,
) -> None:
    """A cleared session has no items left, and is still usable afterward."""
    session = await _write(pair, "", "hi")
    await session.clear_session()
    after_clear = await session.get_items()

    await session.add_items([{"role": "user", "content": "again"}])

    assert after_clear == []
    assert await session.get_items() == [{"role": "user", "content": "again"}]


async def check_sessions_are_isolated_by_session_id(pair: SessionPair) -> None:
    """Items added under one session id don't leak into another sharing the same store."""
    first = await _write(pair, "-a", "from a")
    second = await _write(pair, "-b", "from b")

    assert await first.get_items() == [{"role": "user", "content": "from a"}]
    assert await second.get_items() == [{"role": "user", "content": "from b"}]


async def check_listing_returns_each_written_session_once(pair: SessionPair) -> None:
    """Every written session appears in the listing exactly once, with an `updated_at`."""
    await _write(pair, "-a", "hi")
    await _write(pair, "-b", "hi")

    listed = pair.read.listing(agent=pair.tag)

    assert sorted(summary.id for summary in listed) == [pair.id("-a"), pair.id("-b")]
    assert all(summary.updated_at for summary in listed)


async def check_listing_is_empty_for_an_agent_with_no_history(pair: SessionPair) -> None:
    """`listing(agent=...)` returns an empty list rather than failing when nothing matches."""
    assert pair.read.listing(agent=pair.id("-never-chatted")) == []


async def check_listing_renders_updated_at_the_same_way_in_every_backend(
    pair: SessionPair,
) -> None:
    """`updated_at` is `"YYYY-MM-DD HH:MM:SS"`, with no offset, whichever store wrote it.

    The Postgres side used to hand back a `TIMESTAMPTZ`'s offset-bearing ISO string where SQLite's
    text had none, so the same session read differently depending on the deployment.
    """
    await _write(pair, "", "hi")

    updated_at = pair.read.listing(agent=pair.tag)[0].updated_at

    assert len(updated_at) == len("2026-10-04 10:08:03")
    assert updated_at[4] == updated_at[7] == "-"
    assert updated_at[10] == " "
    assert "+" not in updated_at


async def check_listing_by_agent_matches_prefixed_and_exact_ids(pair: SessionPair) -> None:
    """`listing(agent=...)` finds both `Agent-<suffix>` chats and a bare `Agent` one."""
    await _write(pair, "-20260101-000000-aaaa", "hi")
    await _write(pair, "", "hi")
    await _write(pair, "z-20260101-000000-bbbb", "hi")

    listed = pair.read.listing(agent=pair.tag)

    assert {summary.id for summary in listed} == {pair.id("-20260101-000000-aaaa"), pair.id()}


async def check_listing_by_agent_does_not_treat_underscores_as_wildcards(
    pair: SessionPair,
) -> None:
    """An agent named `a_b` doesn't match a session of `axb`, in any backend."""
    await _write(pair, "xb-1", "hi")

    assert pair.read.listing(agent=pair.id("_b")) == []


async def check_messages_raises_for_an_unknown_session(pair: SessionPair) -> None:
    """`messages` raises `SessionNotFound` rather than returning an empty transcript.

    "No such session" and "a session with nothing in it" are different answers, and
    `runa chat --show` prints different things for them.
    """
    with pytest.raises(SessionNotFound):
        pair.read.messages(pair.id("-never-written"))


async def check_messages_splits_each_message_into_role_and_text(pair: SessionPair) -> None:
    """`messages` flattens a stored item into the `(role, text)` a transcript shows."""
    await _write(pair, "", "hello")

    messages = pair.read.messages(pair.id())

    assert [(message.role, message.text) for message in messages] == [("user", "hello")]
    assert messages[0].created_at


async def check_messages_are_oldest_first(pair: SessionPair) -> None:
    """A transcript reads in the order it was written."""
    session = pair.session()
    await session.add_items(
        [{"role": "user", "content": "first"}, {"role": "assistant", "content": "second"}]
    )

    messages = pair.read.messages(pair.id())

    assert [message.text for message in messages] == ["first", "second"]


async def check_messages_flattens_a_list_shaped_content(pair: SessionPair) -> None:
    """A content list of text parts is joined, the shape a model's own reply is stored in."""
    session = pair.session()
    await session.add_items([{"role": "assistant", "content": [{"text": "one "}, {"text": "two"}]}])

    assert pair.read.messages(pair.id())[0].text == "one two"


CONTRACT: list[Check] = [
    check_add_items_then_get_items_round_trips_in_order,
    check_get_items_with_limit_returns_the_latest_n_in_order,
    check_pop_item_removes_and_returns_the_most_recent_item,
    check_pop_item_on_an_empty_session_returns_none,
    check_set_items_replaces_the_entire_history,
    check_clear_session_removes_everything_and_can_be_written_to_again,
    check_sessions_are_isolated_by_session_id,
    check_listing_returns_each_written_session_once,
    check_listing_is_empty_for_an_agent_with_no_history,
    check_listing_renders_updated_at_the_same_way_in_every_backend,
    check_listing_by_agent_matches_prefixed_and_exact_ids,
    check_listing_by_agent_does_not_treat_underscores_as_wildcards,
    check_messages_raises_for_an_unknown_session,
    check_messages_splits_each_message_into_role_and_text,
    check_messages_are_oldest_first,
    check_messages_flattens_a_list_shaped_content,
]
