"""serve.py: `runa serve`, this app's agents over HTTP.

The missing half of a deployment. Runa could always *build* an agent, but putting one in front of
users meant writing the web layer yourself, and every app wrote the same one: resolve the agent,
build it per request, pass a session, stream or don't, turn a `Run` into JSON. That is convention,
not application code, so it lives here.

Three decisions this makes for you, each the one a production deployment wants:

* **An `Agent` is built per request.** An instance carries the conversation it is running (see
  `Agent.run`), so a module-level agent shared across requests would interleave users' histories.
  Constructing one is cheap; the alternative is a data leak.
* **Conversation state is a `session`, never `self.history`.** A `session_id` in the request body
  is the whole continuity mechanism, which is also what makes more than one replica possible.
* **Authentication is on unless you turn it off.** A bearer token from `RUNA_API_KEY`, checked on
  every route but `/health`. An agent endpoint spends money per call, so open-by-default is the
  wrong default, and `--no-auth` is one flag away for local use.

`fastapi` is imported lazily by `cli/serve.py`, so a plain install still runs every other command;
only `runa serve` itself needs the `serve` extra.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from runa.agent import Agent
from runa.cli._project import iter_agent_classes, loaded_app, require_agents_dir
from runa.lifecycle import logger
from runa.run import Run
from runa.session import SQLiteSession


class RunRequest(BaseModel):
    """One turn for an agent to run."""

    message: str = Field(description="The user's message, one turn, never a transcript.")
    session_id: str | None = Field(
        default=None,
        description="Continue (or start) this conversation. Omitted, the turn is stateless.",
    )
    user_id: str | None = Field(
        default=None, description="Scopes an agent's memory to one user, when it has memory."
    )


def _jsonable(value: Any) -> Any:
    """Render an agent's output for JSON, including a dataclass `output_type`."""
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return str(value)


def _run_payload(run: Run, agent_name: str) -> dict[str, Any]:
    """The wire shape of a finished (or paused, or failed) `Run`.

    `status` is the `Run`'s own, so a caller distinguishes "the agent answered" from "it needs an
    approval" from "it failed" without parsing prose. `trace_id` is what makes a production
    incident debuggable: it is the handle for `runa traces show`.
    """
    return {
        "agent": agent_name,
        "status": run.status,
        "output": _jsonable(run.output),
        "error": run.error,
        "trace_id": run.trace.id if run.trace else None,
        "usage": {
            "input_tokens": run.usage.input_tokens,
            "output_tokens": run.usage.output_tokens,
            "total_tokens": run.usage.total_tokens,
            "requests": run.usage.requests,
        },
        "interruptions": [
            {
                "tool_name": getattr(item, "tool_name", None),
                "call_id": getattr(item, "call_id", None),
            }
            for item in run.interruptions
        ],
    }


def create_app(root: Path, *, api_key: str | None) -> FastAPI:
    """Build the `runa serve` app for the project at `root`.

    `api_key` is the bearer token every route but `/health` requires; `None` disables the check
    entirely (`runa serve --no-auth`). `create_app` takes it as a parameter rather than reading
    the environment itself, the same way every `cli/*.py` command takes `root` instead of
    assuming `cwd`, so a test (or an app embedding this) can be explicit.
    """
    agents_dir = require_agents_dir(root)

    def _agent_classes() -> dict[str, type[Agent]]:
        """Resolve the app's agents by declared `name`, importing `main.py` first.

        Done per request rather than once at startup so an edit to `app/agents/` is picked up by
        a reloading server, and so a broken agent module fails that one request instead of
        preventing the process from starting at all.
        """
        with loaded_app(root):
            return {
                getattr(cls, "name", cls.__name__): cls for cls in iter_agent_classes(agents_dir)
            }

    async def _authenticate(request: Request) -> None:
        """Require `Authorization: Bearer <RUNA_API_KEY>` unless the server was started open."""
        if api_key is None:
            return
        header = request.headers.get("authorization", "")
        scheme, _, token = header.partition(" ")
        if scheme.lower() != "bearer" or not token:
            raise HTTPException(status_code=401, detail="expected an Authorization: Bearer token")
        if not _constant_time_equal(token, api_key):
            raise HTTPException(status_code=403, detail="invalid API key")

    app = FastAPI(title="runa serve", description="This Runa app's agents, over HTTP.")

    @app.get("/health")
    def health() -> dict[str, str]:
        """Liveness, unauthenticated: a load balancer cannot hold a bearer token."""
        return {"status": "ok"}

    # Everything else hangs off a router that carries the auth dependency, rather than the app
    # carrying it: an app-level dependency also guards `/health`, which would lock out exactly
    # the caller that cannot authenticate.
    api = APIRouter(dependencies=[Depends(_authenticate)])

    @api.get("/agents")
    def agents() -> dict[str, list[str]]:
        """Every agent this app serves, by the `name` each one declares."""
        return {"agents": sorted(_agent_classes())}

    def _build(agent_name: str) -> Agent:
        """A fresh `Agent` for one request; the module docstring says why it is per request."""
        classes = _agent_classes()
        agent_cls = classes.get(agent_name)
        if agent_cls is None:
            raise HTTPException(status_code=404, detail=f"no agent named {agent_name!r}")
        return agent_cls()

    @api.post("/agents/{agent_name}/runs")
    async def run_agent(agent_name: str, body: RunRequest) -> dict[str, Any]:
        """Run one turn and return the whole `Run`."""
        agent = _build(agent_name)
        session = (
            SQLiteSession(body.session_id, root / "db" / "runa.db", user_id=body.user_id)
            if body.session_id
            else None
        )
        run = await agent.run(body.message, session=session)
        return _run_payload(run, agent_name)

    @api.post("/agents/{agent_name}/runs/stream")
    async def stream_agent(agent_name: str, body: RunRequest) -> StreamingResponse:
        """Run one turn, streaming tokens as server-sent events, then the finished `Run`.

        Each `data:` line is one JSON object: `{"type": "token", "text": ...}` as the answer is
        produced, then a single `{"type": "run", ...}` carrying the same payload the non-streaming
        route returns, so a client gets the trace id and usage either way.
        """
        agent = _build(agent_name)
        session = (
            SQLiteSession(body.session_id, root / "db" / "runa.db", user_id=body.user_id)
            if body.session_id
            else None
        )

        async def events() -> AsyncIterator[str]:
            stream = agent.run_streamed(body.message, session=session)
            try:
                async for event in stream:
                    text = _token_text(event)
                    if text:
                        yield f"data: {json.dumps({'type': 'token', 'text': text})}\n\n"
            except Exception as exc:  # noqa: BLE001 -- the client must not be left hanging
                # The response status went out with the first byte, so an error here cannot
                # become a 500. Silently truncating the stream would leave a client unable to
                # tell a finished answer from a dropped one, so the failure is itself an event.
                logger.exception("streamed run failed for agent %s", agent_name)
                error = {"type": "error", "error": f"{type(exc).__name__}: {exc}"}
                yield f"data: {json.dumps(error)}\n\n"
            else:
                if stream.run is not None:
                    payload = {"type": "run", **_run_payload(stream.run, agent_name)}
                    yield f"data: {json.dumps(payload)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    app.include_router(api)
    return app


def _token_text(event: Any) -> str | None:
    """The incremental text on a `RawResponsesStreamEvent`, if it carries any.

    Its `data` is a `StreamDelta` (see `runa.stream_events`), so the text is an attribute. The
    other two event kinds carry completed items rather than fragments and contribute nothing
    here; the finished `Run` at the end of the stream already reports what they produced.
    """
    text = getattr(getattr(event, "data", None), "text", None)
    return text if isinstance(text, str) and text else None


def _constant_time_equal(left: str, right: str) -> bool:
    """Compare two tokens without leaking their common prefix through timing."""
    import hmac

    return hmac.compare_digest(left, right)


__all__ = ["RunRequest", "create_app"]
