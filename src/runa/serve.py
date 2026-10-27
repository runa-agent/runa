"""serve.py: `runa serve`, this app's agents over HTTP.

What a deployment gets:

* **An `Agent` is built per request, from a class resolved once at startup.** The app is
  imported once, by `create_app`, so picking up an edit to `app/agents/` or `app/prompts/` takes
  a restart. Every declared agent's `knowledge=` corpus is ingested at startup too.
* **Conversation state is a `session`, never `self.history`.** A `session_id` in the request body
  is the whole continuity mechanism, and `runa.db` decides where that history lives
  (`RUNA_DATABASE_URL`), so more than one replica needs no code change.
* **Authentication is on unless you turn it off.** A bearer token from `RUNA_API_KEY`, checked on
  every route but `/health`; `--no-auth` turns it off for local use.
* **A paused run stays on the server, and resumes through the route it paused on.** A pause is
  kept in `runa.db` by the run itself and the response carries only its `run_id`; the request
  body carries that `run_id` plus the `decisions` a human made, in place of `message`, so the
  approval ledger never leaves the server. Resuming follows `RunState`'s rules.

Only serving needs the `serve` extra; `cli/serve.py` imports this module lazily. `create_app` is
public, for an `asgi.py` that brings its own server or mounts these routes inside a larger one.
One Runa app per process: a second `create_app` for a different project is a `UserError` (see
`project.use`).
"""

import asyncio
import json
import os
from collections.abc import AsyncIterator, Iterable, Iterator
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Any, Self

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, model_validator

from runa import project
from runa._log import logger
from runa.agent import Agent, warm
from runa.exceptions import OperatorError, UserError
from runa.run import Run
from runa.run_state import RunState
from runa.stream_events import TextDelta
from runa.tracing.export import flush_exports

API_KEY_ENV = "RUNA_API_KEY"


class MissingAPIKey(OperatorError):
    """Raised when a server is built with neither `RUNA_API_KEY` nor an explicit `no_auth`."""


class Decision(BaseModel):
    """What a human decided about one `Interruption` of a paused run.

    `call_id` is the one the paused payload reported, and the only handle a client needs: the
    tool, its arguments and the agent it belongs to are the server's to look back up off the
    `state`. `always` and `rejection_message` are `RunState.approve`/`.reject`'s own keywords,
    named the same here on purpose -- the wire shape of an approval is the Python one.
    """

    call_id: str = Field(description="The interruption being decided, by the id it reported.")
    approve: bool = Field(description="True runs the paused call, False skips it.")
    always: bool = Field(
        default=False,
        description="Apply this decision to every future call to that tool, not just this one.",
    )
    rejection_message: str | None = Field(
        default=None, description="Fed back to the model instead of the default rejection text."
    )


class RunRequest(BaseModel):
    """One turn for an agent to run: a new message, or a paused run and the decisions resolving it.

    Exactly one of `message`/`run_id` -- the two stand for the same argument of `Agent.run`, which
    takes a message or the `RunState` of a paused run, so a body carrying both (or neither) is a
    `422` rather than a guess about which one was meant.
    """

    message: str | None = Field(
        default=None, description="The user's message, one turn, never a transcript."
    )
    run_id: str | None = Field(
        default=None,
        description="A paused run's `run_id`, sent back in place of a message to resume it.",
    )
    decisions: list[Decision] = Field(
        default_factory=list,
        description="Approvals and rejections for the paused run's interruptions.",
    )
    session_id: str | None = Field(
        default=None,
        description=(
            "Continue (or start) this conversation. Omitted, the turn is stateless. A paused "
            "run resumes in the conversation it paused in, so it is never sent beside `run_id`."
        ),
    )
    user_id: str | None = Field(
        default=None,
        description="Who a new turn is for: scopes an agent's memory to one user, session or not.",
    )

    @model_validator(mode="after")
    def _exactly_one_turn(self) -> Self:
        """Reject a body that is neither a turn nor a resume, or tries to be both."""
        if (self.message is None) == (self.run_id is None):
            raise ValueError("send either a `message` to run or the `run_id` of a paused run")
        if self.decisions and self.run_id is None:
            raise ValueError("`decisions` resolve a paused run, so they need its `run_id` too")
        return self


@contextmanager
def _refused_as_422() -> Iterator[None]:
    """Answer a turn `Agent.run` refuses to start with a `422`, its message as the detail.

    Which turns those are is `Agent.run`'s to say; this maps its `UserError` and nothing else.
    """
    try:
        yield
    except UserError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _run_payload(run: Run, agent_name: str) -> dict[str, Any]:
    """`Run.to_payload()`, plus what HTTP knows and the run doesn't: which agent ran it."""
    return {"agent": agent_name, **run.to_payload()}


async def _turn(agent: Agent, agent_name: str, body: RunRequest) -> str | RunState:
    """What this request asks `agent` to do: run a message, or finish the paused run it names.

    The two are one argument to `Agent.run`, so they are one argument here too, and the
    forward and the inverse direction stay the same route (see this module's docstring).

    A `run_id` this agent has no pause under -- unknown, or already resumed -- is a `404`. A pause
    that no longer rebuilds (a schema version this Runa doesn't know, an agent renamed since it
    paused) is a `409`, and a decision for a call the pause doesn't have is a `422`; none of them
    spends the pause. A request that loses a race to resume it gets back an `"error"` run.
    An `Interruption` left undecided keeps pausing the run, which the next response reports the
    same way as the first: the decisions a client sends are the ones a human has actually made,
    and nothing here invents the rest.
    """
    if body.run_id is None:
        # `RunRequest` has already rejected a body with neither; this is for the type checker.
        return body.message or ""
    try:
        state = await RunState.load(agent, body.run_id)
    except UserError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if state is None:
        raise HTTPException(
            status_code=404,
            detail=f"{agent_name!r} has no paused run {body.run_id!r}, or it was already resumed",
        )

    pending = {item.call_id: item for item in state.pending}
    for decision in body.decisions:
        interruption = pending.get(decision.call_id)
        if interruption is None:
            raise HTTPException(
                status_code=422,
                detail=f"this run has no interruption awaiting {decision.call_id!r}",
            )
        if decision.approve:
            state.approve(interruption, always=decision.always)
        else:
            state.reject(
                interruption, always=decision.always, rejection_message=decision.rejection_message
            )
    return state


def resolve_api_key(*, no_auth: bool) -> str | None:
    """The token a server should require, read from `RUNA_API_KEY`, or `None` to serve open.

    Refuses to start unauthenticated by accident: an agent endpoint costs money per call, so
    "nobody set the variable" has to be an error rather than a silently open door. `no_auth=True`
    (`runa serve --no-auth`, or the argument in a generated `asgi.py`) makes the same choice
    explicit and is then perfectly fine for local use, or behind something that already
    authenticates.

    `create_app` takes the resolved token rather than calling this itself, so the environment is
    read in exactly one place a deployment can see, and a test can pass a token outright.
    """
    if no_auth:
        return None
    api_key = os.environ.get(API_KEY_ENV)
    if not api_key:
        raise MissingAPIKey(
            f"{API_KEY_ENV} is not set. Set it to the token clients must send as "
            "`Authorization: Bearer <token>`, or pass --no-auth to serve without authentication."
        )
    return api_key


def create_app(root: Path, *, api_key: str | None) -> FastAPI:
    """Build the app serving the agents of the project at `root`.

    `api_key` is the bearer token every route but `/health` requires; `None` disables the check
    entirely (`runa serve --no-auth`). `create_app` takes it as a parameter rather than reading
    the environment itself, the same way this takes `root` instead of assuming `cwd`, so a test
    (or an app embedding this) can be explicit: `resolve_api_key` is the conventional way to fill
    it in.

    `root` also goes to `project.use`, which is how this server says it reads that app: its state
    resolves against `root` and its `main.py` is the one imported, so an embedded app served from
    outside its own directory keeps everything together -- a session route and the memory an
    agent writes during that same request resolve to one file, not two.
    """
    project.use(root)
    agents_dir = project.require_agents_dir(root)

    # The app's agents by declared `name`, resolved here and kept: a server resolves them once, at
    # startup, not once per request (see this module's docstring). Reading them is also what
    # imports the app, so a broken agent module stops the process from starting rather than
    # failing one route, which is the failure a deployment wants -- found by the thing that rolls
    # out, not by a user.
    agent_classes = {cls.name: cls for cls in project.iter_agent_classes(agents_dir)}

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

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        """Ingest knowledge before the first request; write every queued trace after the last."""
        await _ingest_knowledge(agent_classes.values())
        yield
        await asyncio.to_thread(flush_exports)

    app = FastAPI(
        title="runa serve",
        description="This Runa app's agents, over HTTP.",
        lifespan=lifespan,
    )

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
        return {"agents": sorted(agent_classes)}

    def _build(agent_name: str) -> Agent:
        """A fresh `Agent` for one request; the module docstring says why it is per request."""
        agent_cls = agent_classes.get(agent_name)
        if agent_cls is None:
            raise HTTPException(status_code=404, detail=f"no agent named {agent_name!r}")
        return agent_cls()

    @api.post("/agents/{agent_name}/runs")
    async def run_agent(agent_name: str, body: RunRequest) -> dict[str, Any]:
        """Run one turn -- or finish a paused one -- and return the whole `Run`.

        A `Run` that paused for approval comes back with `status="paused"`, its `interruptions`
        and the `run_id` that resolves them; posting that `run_id` back with a `decisions` list
        resumes it here, in the session it paused in.
        """
        agent = _build(agent_name)
        turn = await _turn(agent, agent_name, body)
        with _refused_as_422():
            run = await agent.run(turn, session=body.session_id, user_id=body.user_id)
        return _run_payload(run, agent_name)

    @api.post("/agents/{agent_name}/runs/stream")
    async def stream_agent(agent_name: str, body: RunRequest) -> StreamingResponse:
        """Run one turn, streaming tokens as server-sent events, then the finished `Run`.

        Each `data:` line is one JSON object: `{"type": "token", "text": ...}` as the answer is
        produced, then a single `{"type": "run", ...}` carrying the same payload the non-streaming
        route returns, so a client gets the trace id and usage either way -- including the `run_id`
        of a run that paused, which this route takes back the same way the other one does. That
        body is resolved, and the stream started, before the response starts, so a pause that can't
        be resumed or a turn `Agent.run_streamed` refuses is still a `4xx` rather than an error
        event in a stream that already answered `200`.

        A run that fails needs nothing special here: `Agent.run_streamed` reports every failure
        on the `Run` the stream ends with (see `Agent.run`), so `{"type": "run"}` with
        `status="error"` is one wire shape for a turn that answered and one that didn't.
        """
        agent = _build(agent_name)
        turn = await _turn(agent, agent_name, body)
        with _refused_as_422():
            stream = agent.run_streamed(turn, session=body.session_id, user_id=body.user_id)

        async def events() -> AsyncIterator[str]:
            try:
                async for event in stream:
                    if isinstance(event, TextDelta):
                        token = {"type": "token", "text": event.text}
                        yield f"data: {json.dumps(token)}\n\n"
                if stream.run is not None:
                    payload = {"type": "run", **_run_payload(stream.run, agent_name)}
                    yield f"data: {json.dumps(payload)}\n\n"
            except Exception:  # noqa: BLE001 -- the client must not be left hanging
                # The response status went out with the first byte, so an error here cannot
                # become a 500. Silently truncating the stream would leave a client unable to
                # tell a finished answer from a dropped one, so the failure is itself an event.
                # A failed *run* doesn't come through here (it arrives as the `run` event above);
                # this is the serving code itself breaking, so the event says only that much:
                # its type and message can name internals a caller has no business reading.
                logger.exception("streaming the run failed for agent %s", agent_name)
                error = {"type": "error", "error": "the run failed"}
                yield f"data: {json.dumps(error)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    app.include_router(api)
    return app


async def _ingest_knowledge(agent_classes: Iterable[type[Agent]]) -> None:
    """Ingest every declared agent's knowledge corpus, before any request can need it.

    The same choice `create_app` already makes by resolving the agent classes at startup, applied
    to the other startup-shaped cost an app can carry: reading a directory of documents and
    embedding every chunk of it. Left to `Knowledge.search`'s own lazy ingest, that lands inside
    whichever user's turn arrives first after a rollout -- metered, corpus-sized, and attributed
    to a `retrieval` span rather than to the deploy that caused it. A startup that fails on a
    missing embeddings credential is the same trade: found by the thing that rolls out.

    What each class warms is `warm`'s; a class no agent could be built from fails startup here
    rather than its first request. Agents sharing one corpus cost nothing extra: `ingest` asks
    the store whether it already holds what it would write, so the second one pays a `stat` per
    file and a version read.
    """
    for agent_cls in agent_classes:
        chunks = await warm(agent_cls)
        if chunks:
            # Only when it actually wrote: zero is the usual answer (a corpus the store already
            # held), and a line per agent per boot saying nothing happened is noise.
            logger.info("ingested %d knowledge chunks for %s", chunks, agent_cls.name)


def _constant_time_equal(left: str, right: str) -> bool:
    """Compare two tokens without leaking their common prefix through timing."""
    import hmac

    return hmac.compare_digest(left, right)


__all__ = [
    "API_KEY_ENV",
    "Decision",
    "MissingAPIKey",
    "RunRequest",
    "create_app",
    "resolve_api_key",
]
