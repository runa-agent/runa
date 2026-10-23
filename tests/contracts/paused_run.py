"""contracts/paused_run.py: the one `PausedRunStore` contract, so every backend is held to it.

What is worth testing is what `runa serve` relies on: a saved state comes back unchanged under
its own id, only through the agent it paused on, and a claim succeeds exactly once.

Each check takes the store and one agent name it is free to write under. `test_paused_runs.py`
drives these over the local backends on every `make test`; `test_postgres.py` drives them over
`PostgresPausedRunStore` when a live database is reachable.
"""

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any

from runa.run_state.store import PausedRunStore

Check = Callable[[PausedRunStore, str], Coroutine[Any, Any, None]]

_STATE = {"schema_version": 8, "run": {"pending": [{"call_id": "call_1"}]}}


async def check_load_on_unknown_id_returns_none(store: PausedRunStore, agent: str) -> None:
    """An id nothing was saved under is a miss, not an error."""
    assert await store.load(agent, "nope") is None


async def check_save_then_load_round_trips_the_state(store: PausedRunStore, agent: str) -> None:
    """A saved state comes back unchanged under the id `save` returned."""
    run_id = await store.save(agent, _STATE)

    assert await store.load(agent, run_id) == _STATE


async def check_every_save_gets_its_own_id(store: PausedRunStore, agent: str) -> None:
    """Saving the same state twice is two pauses, not one overwritten."""
    first = await store.save(agent, _STATE)
    second = await store.save(agent, _STATE)

    assert first != second


async def check_a_pause_is_only_found_through_its_agent(store: PausedRunStore, agent: str) -> None:
    """Another agent's route neither reads nor claims it."""
    run_id = await store.save(agent, _STATE)

    assert await store.load(f"{agent}-other", run_id) is None
    assert await store.claim(f"{agent}-other", run_id) is False
    assert await store.load(agent, run_id) == _STATE


async def check_a_claim_succeeds_once(store: PausedRunStore, agent: str) -> None:
    """The first claim takes the pause; a second finds it gone, and so does a load."""
    run_id = await store.save(agent, _STATE)

    assert await store.claim(agent, run_id) is True
    assert await store.claim(agent, run_id) is False
    assert await store.load(agent, run_id) is None


async def check_concurrent_claims_have_one_winner(store: PausedRunStore, agent: str) -> None:
    """Two requests resuming one pause at once: exactly one of them gets to run it."""
    run_id = await store.save(agent, _STATE)

    claims = await asyncio.gather(*(store.claim(agent, run_id) for _ in range(5)))

    assert sorted(claims) == [False, False, False, False, True]


CONTRACT: list[Check] = [
    check_load_on_unknown_id_returns_none,
    check_save_then_load_round_trips_the_state,
    check_every_save_gets_its_own_id,
    check_a_pause_is_only_found_through_its_agent,
    check_a_claim_succeeds_once,
    check_concurrent_claims_have_one_winner,
]
