"""runner.py: `Runner`, Runa's own agent loop, replaces `agents.Runner`."""

from __future__ import annotations

import asyncio
from typing import Any

from runa._types import RunContextWrapper, TResponseInputItem
from runa.lifecycle import RunHooks
from runa.result import RunResult, RunResultStreaming
from runa.run_config import RunConfig
from runa.run_internal.run_loop import _run_async
from runa.run_state import RunState
from runa.session import SessionABC


class Runner:
    """Runs an `Agent` for one turn: `run`/`run_sync` (final output) or `run_streamed` (events)."""

    @staticmethod
    async def run(
        agent: Any,
        input: str | list[TResponseInputItem] | RunState,
        *,
        context: Any = None,
        hooks: RunHooks[Any] | None = None,
        run_config: RunConfig | None = None,
        session: SessionABC | None = None,
        _context_wrapper: RunContextWrapper[Any] | None = None,
    ) -> RunResult:
        """Run `agent` on `input` (or resume a paused `RunState`) and return the final result.

        `_context_wrapper` is internal: used by `agent_as_tool`'s nested delegate calls to pass
        a forked `RunContextWrapper` through instead of building a fresh one from `context`; not
        meant to be passed directly.
        """
        return await _run_async(
            agent,
            input,
            context=context,
            hooks=hooks,
            run_config=run_config,
            session=session,
            _context_wrapper=_context_wrapper,
        )

    @staticmethod
    def run_sync(
        agent: Any,
        input: str | list[TResponseInputItem] | RunState,
        *,
        context: Any = None,
        hooks: RunHooks[Any] | None = None,
        run_config: RunConfig | None = None,
        session: SessionABC | None = None,
    ) -> RunResult:
        """Synchronous `run`, for callers not already inside an event loop."""
        return asyncio.run(
            Runner.run(
                agent, input, context=context, hooks=hooks, run_config=run_config, session=session
            )
        )

    @staticmethod
    def run_streamed(
        agent: Any,
        input: str | list[TResponseInputItem] | RunState,
        *,
        context: Any = None,
        hooks: RunHooks[Any] | None = None,
        run_config: RunConfig | None = None,
        session: SessionABC | None = None,
    ) -> RunResultStreaming:
        """Run `agent` on `input` (or resume a paused `RunState`), streaming `StreamEvent`s.

        The same run as `run`: a tool call needing approval ends the stream with
        `interruptions`, resumed by passing the resolved `to_state()` back in.
        """
        context_wrapper = (
            input.context_wrapper
            if isinstance(input, RunState)
            else RunContextWrapper(context=context)
        )
        return RunResultStreaming(
            lambda emit: _run_async(
                agent,
                input,
                hooks=hooks,
                run_config=run_config,
                session=session,
                _context_wrapper=context_wrapper,
                emit=emit,
            ),
            context_wrapper,
        )


__all__ = ["Runner"]
