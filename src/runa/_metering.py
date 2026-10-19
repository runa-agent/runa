"""_metering.py: charging a model call made below a run to the run it was made for.

Every model call a run pays for is wrapped in `metered`, whether the run's loop makes it or code
the run reaches without handing it anything (the embedding request inside `Memory.search`, say).
Inside a run it is charged to that run, its usage counted and a span put in its trace under the
innermost span open; outside one (a `Knowledge.ingest()` at `runa serve` startup) it is charged
to nothing.
"""

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from contextvars import ContextVar
from typing import Any, Protocol

from runa._types import Usage

Charge = Callable[[Usage], None]


class Meter(Protocol):
    """Where the calls made inside a `charging` block are charged."""

    def call(self, name: str, *, input: Any = None) -> AbstractContextManager[Charge]:
        """A `with` block around one call to the model `name`; the block charges what it spent."""
        ...


_current: ContextVar[Meter | None] = ContextVar("runa_meter", default=None)


@contextmanager
def charging(meter: Meter) -> Iterator[None]:
    """Charge each `metered` call inside the block, in this task or one it starts, to `meter`."""
    token = _current.set(meter)
    try:
        yield
    finally:
        _current.reset(token)


@contextmanager
def metered(name: str, *, input: Any = None) -> Iterator[Charge]:
    """A `with` block around one call to the model `name`, handing it what to charge `Usage` to."""
    meter = _current.get()
    if meter is None:
        yield _uncharged
        return
    with meter.call(name, input=input) as charge:
        yield charge


def _uncharged(usage: Usage) -> None:
    """Outside a run, a call's usage belongs to no one."""


__all__ = ["Charge", "Meter", "charging", "metered"]
