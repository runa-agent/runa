"""serve.py: `runa serve`, this app's agents over HTTP.

The missing half of a deployment. Runa could always *build* an agent, but putting one in front of
users meant writing the web layer yourself, and every app wrote the same one: resolve the agent,
build it per request, pass a session, stream or don't, hand a paused run's state back once a
human has decided, turn a `Run` into JSON. That is convention, not application code, so it lives
here.

Four decisions this makes for you, each the one a production deployment wants:

* **An `Agent` is built per request, from a class resolved once at startup.** An instance carries
  the conversation it is running (see `Agent.run`), so a module-level agent shared across requests
  would interleave users' histories. Constructing one is cheap; the alternative is a data leak.
  The *class* is the other half of that choice: importing the app is what is not cheap, and doing
  it per request would re-run `main.py` and re-initialize whatever the app keeps at module level --
  a client, a pool, a cache -- underneath runs already holding the old one. So `create_app`
  imports the app once, and a server picks up an edit to `app/agents/` by restarting (which is
  what `uvicorn asgi:app --reload` already does, properly, from the outside).
* **Conversation state is a `session`, never `self.history`.** A `session_id` in the request body
  is the whole continuity mechanism, and `runa.db` decides where that history lives: the local
  `db/runa.db` by default, the shared Postgres once `RUNA_DATABASE_URL` is set, which is what
  makes more than one replica possible. Serving behind replicas is therefore one variable, not a
  code change, and nothing here names a backend.
* **Authentication is on unless you turn it off.** A bearer token from `RUNA_API_KEY`, checked on
  every route but `/health`. An agent endpoint spends money per call, so open-by-default is the
  wrong default, and `--no-auth` is one flag away for local use.
* **A paused run resumes through the route it paused on.** `Agent.run` takes a `RunState` in
  place of a message, so the request body does literally that: `state` plus the `decisions` a
  human made, instead of `message`. Reporting a pause without accepting one back would leave
  every app to re-derive the five steps `RunState.from_json` exists for; a second pair of
  `/resume` endpoints would leave a streaming client unable to resume streaming. The blob a
  client sends back is a capability -- it carries the run's context and its approval ledger, and
  nothing here records that one was already spent -- which is the other reason `RUNA_API_KEY` is
  a server-to-server credential rather than something to hand a browser.

`runa serve` is the short way to run this app, and `cli/serve.py` imports this module lazily so a
plain install still runs every other command; only serving needs the `serve` extra. The app object
itself is public, though: `runa new` scaffolds an `asgi.py` that builds it, for the deployment that
brings its own server or mounts these routes inside a larger one. One Runa app per process, in
that case: a second `create_app` for a different project is a `UserError` rather than two apps
quietly sharing the last one's imports and database (see `project.use`).
"""

import json
import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Self

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, model_validator

from runa import db, project
from runa._log import logger
from runa.agent import Agent
from runa.exceptions import OperatorError, UserError
from runa.run import Run
from runa.run_state import RunState

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

    Exactly one of `message`/`state` -- the two are the same argument of `Agent.run`, which takes
    a message or the `RunState` of a paused run, so a body carrying both (or neither) is a `422`
    rather than a guess about which one was meant.
    """

    message: str | None = Field(
        default=None, description="The user's message, one turn, never a transcript."
    )
    state: dict[str, Any] | None = Field(
        default=None,
        description="A paused run's `state`, sent back in place of a message to resume it.",
    )
    decisions: list[Decision] = Field(
        default_factory=list,
        description="Approvals and rejections for the paused `state`'s interruptions.",
    )
    session_id: str | None = Field(
        default=None,
        description="Continue (or start) this conversation. Omitted, the turn is stateless.",
    )
    user_id: str | None = Field(
        default=None, description="Scopes an agent's memory to one user, when it has memory."
    )

    @model_validator(mode="after")
    def _exactly_one_turn(self) -> Self:
        """Reject a body that is neither a turn nor a resume, or tries to be both."""
        if (self.message is None) == (self.state is None):
            raise ValueError("send either a `message` to run or the `state` of a paused run")
        if self.decisions and self.state is None:
            raise ValueError("`decisions` resolve a paused run, so they need its `state` too")
        return self


def _run_payload(run: Run, agent_name: str) -> dict[str, Any]:
    """A `Run` on the wire, plus the one thing HTTP knows and the run doesn't: which agent ran it.

    `Run.to_payload()` owns the rest. Re-listing a `Run`'s fields here is what let this module
    report an `Interruption.tool_name` that never existed, so the mapping lives next to the type
    it maps and a transport adds only what is genuinely its own.
    """
    return {"agent": agent_name, **run.to_payload()}


def _turn(agent: Agent, body: RunRequest) -> str | RunState:
    """What this request asks `agent` to do: run a message, or finish the paused run it carries.

    The two are one argument to `Agent.run`, so they are one argument here too, and the
    forward and the inverse direction stay the same route (see this module's docstring).

    A blob that no longer rebuilds -- a schema version this Runa doesn't know, an agent renamed
    since it paused -- is the client's `state`, so its `UserError` is a `422` naming what was
    wrong, not a `500`. An `Interruption` left undecided keeps pausing the run, which the next
    response reports the same way as the first: the decisions a client sends are the ones a human
    has actually made, and nothing here invents the rest.
    """
    if body.state is None:
        # `RunRequest` has already rejected a body with neither; this is for the type checker.
        return body.message or ""
    try:
        state = RunState.from_json(agent, body.state)
    except UserError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

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
    agent_classes = {
        getattr(cls, "name", cls.__name__): cls for cls in project.iter_agent_classes(agents_dir)
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
        and the `state` that resolves them; posting that `state` back with a `decisions` list
        (and the same `session_id`) resumes it here.
        """
        agent = _build(agent_name)
        session = db.session(body.session_id, user_id=body.user_id) if body.session_id else None
        run = await agent.run(_turn(agent, body), session=session)
        return _run_payload(run, agent_name)

    @api.post("/agents/{agent_name}/runs/stream")
    async def stream_agent(agent_name: str, body: RunRequest) -> StreamingResponse:
        """Run one turn, streaming tokens as server-sent events, then the finished `Run`.

        Each `data:` line is one JSON object: `{"type": "token", "text": ...}` as the answer is
        produced, then a single `{"type": "run", ...}` carrying the same payload the non-streaming
        route returns, so a client gets the trace id and usage either way -- including the `state`
        of a run that paused, which this route takes back the same way the other one does. That
        body is resolved before the response starts, so an unreadable `state` is still a `422`
        rather than an error event in a stream that already answered `200`.

        A run that fails needs nothing special here: `Agent.run_streamed` reports every failure
        on the `Run` the stream ends with (see `Agent.run`), so `{"type": "run"}` with
        `status="error"` is one wire shape for a turn that answered and one that didn't.
        """
        agent = _build(agent_name)
        session = db.session(body.session_id, user_id=body.user_id) if body.session_id else None
        turn = _turn(agent, body)

        async def events() -> AsyncIterator[str]:
            stream = agent.run_streamed(turn, session=session)
            try:
                async for event in stream:
                    text = _token_text(event)
                    if text:
                        yield f"data: {json.dumps({'type': 'token', 'text': text})}\n\n"
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


__all__ = [
    "API_KEY_ENV",
    "Decision",
    "MissingAPIKey",
    "RunRequest",
    "create_app",
    "resolve_api_key",
]
