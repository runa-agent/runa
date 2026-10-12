"""Tests for `runa.serve`: the HTTP surface `runa serve` puts in front of this app's agents.

Covers the four decisions the module makes on a deployment's behalf -- an agent per request,
sessions rather than `self.history`, authentication on by default, and a paused run resumed
through the route it paused on -- plus the wire shape a client actually depends on.
"""

import asyncio
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from runa import db
from runa.cli.generate import generate_agent
from runa.cli.new import scaffold_project
from runa.project import iter_agent_classes, loaded_app
from runa.run_state import RunState
from runa.serve import MissingAPIKey, create_app, resolve_api_key

_STUB_MODEL = '''
from collections.abc import AsyncIterator
from typing import Any

from runa._models import StreamDelta
from runa._types import ModelResponse, Usage


class StubModel:
    """Answers with the number of history items it was handed, so leakage is observable."""

    async def get_response(self, request: Any, **kwargs: Any) -> ModelResponse:
        items = request.input
        return ModelResponse(
            output=[{"role": "assistant", "content": f"items={len(items)}", "tool_calls": None}],
            usage=Usage(input_tokens=3, output_tokens=4, total_tokens=7, requests=1),
        )

    async def stream_response(self, request: Any, **kwargs: Any) -> AsyncIterator[StreamDelta]:
        items = request.input
        yield StreamDelta(text=f"items={len(items)}")
        yield StreamDelta(
            usage=Usage(input_tokens=3, output_tokens=4, total_tokens=7, requests=1)
        )
'''


_PAUSING_AGENT = '''
"""An agent that pauses on an approval-gated tool, to pin the paused wire shape."""

from collections.abc import AsyncIterator
from typing import Any

from runa import Agent, tool
from runa._models import StreamDelta
from runa._types import ModelResponse, Usage


@tool(needs_approval=True)
def issue_refund(amount: float) -> str:
    """Refund `amount` dollars.

    amount: dollars to refund
    """
    return f"refunded {amount}"


class ScriptedModel:
    """Asks for one approval-gated refund, then reports back whatever the refund call answered.

    Stateless, deliberately: `runa serve` builds a fresh agent per request, so a resumed run
    gets a new model instance and a script popped off a list would ask for the refund all over
    again. What is in the input is the only thing either turn depends on.
    """

    _REFUND_CALL = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "issue_refund", "arguments": '{"amount": 75}'},
            }
        ],
    }

    def _message(self, request: Any) -> dict[str, Any]:
        """Ask for the refund, or answer with the result (or refusal) already in the input."""
        settled = [item for item in request.input if item.get("role") == "tool"]
        if not settled:
            return self._REFUND_CALL
        return {"role": "assistant", "content": settled[-1]["content"], "tool_calls": None}

    async def get_response(self, request: Any, **kwargs: Any) -> ModelResponse:
        return ModelResponse(
            output=[self._message(request)],
            usage=Usage(input_tokens=1, output_tokens=1, total_tokens=2, requests=1),
        )

    async def stream_response(self, request: Any, **kwargs: Any) -> AsyncIterator[StreamDelta]:
        message = self._message(request)
        for call in message["tool_calls"] or []:
            yield StreamDelta(
                tool_call_index=0,
                tool_call_id=call["id"],
                tool_call_name=call["function"]["name"],
                tool_call_arguments=call["function"]["arguments"],
            )
        if message["content"] is not None:
            yield StreamDelta(text=message["content"])
        yield StreamDelta(
            usage=Usage(input_tokens=1, output_tokens=1, total_tokens=2, requests=1)
        )


class RefundAgent(Agent):
    name = "refund_agent"
    instructions = "Refund when asked."
    model = ScriptedModel()
    tools = [issue_refund]
'''


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A scaffolded project whose one agent answers from a stub model, never a real provider."""
    project_dir = scaffold_project("demo", root=tmp_path)
    generate_agent("SupportAgent", root=project_dir, model="gpt-5.4-nano")
    (project_dir / "app" / "stub_model.py").write_text(_STUB_MODEL)

    agent_file = project_dir / "app" / "agents" / "support_agent.py"
    source = agent_file.read_text()
    source = source.replace(
        "from runa import Agent",
        "from runa import Agent\n\nfrom app.stub_model import StubModel",
    )
    source = source.replace('model = "gpt-5.4-nano"', "model = StubModel()")
    agent_file.write_text(source)
    return project_dir


@pytest.fixture
def client(project: Path) -> TestClient:
    """An authenticated client; the token is the one the app was built with."""
    return TestClient(create_app(project, api_key="secret-token"))


@pytest.fixture
def pausing_client(project: Path) -> TestClient:
    """A client for a project whose second agent pauses on an approval-gated refund."""
    (project / "app" / "agents" / "refund_agent.py").write_text(_PAUSING_AGENT)
    return TestClient(create_app(project, api_key="secret-token"))


def _auth(token: str = "secret-token") -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_create_app_points_the_db_at_its_own_project(project: Path) -> None:
    """An app served from outside its own directory keeps all of its state in one file.

    `create_app(root)` is public API -- `runa new` scaffolds an `asgi.py` that calls it for a
    deployment bringing its own server -- and `root` is the only thing it is told about the
    project. A session route resolving to `root` while the memory an agent writes during that
    same request resolved to the server's cwd is the split `runa.db` exists to prevent, so the
    assertion covers one concern from each side of that old divide.
    """
    db.use_project(None)

    create_app(project, api_key="secret-token")

    assert db.sessions().db_path == project / "db" / "runa.db"  # type: ignore[attr-defined]
    assert db.memory_store(dimensions=4).db_path == project / "db" / "runa.db"  # type: ignore[attr-defined]


def _paused_body(client: TestClient) -> dict[str, Any]:
    """Run the refund agent far enough to pause, and return the payload it answered with."""
    response = client.post(
        "/agents/refund_agent/runs", json={"message": "refund $75"}, headers=_auth()
    )

    assert response.status_code == 200
    body: dict[str, Any] = response.json()
    assert body["status"] == "paused"
    return body


def test_health_needs_no_token(client: TestClient) -> None:
    """A load balancer cannot hold a bearer token, so liveness has to be open."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_a_request_without_a_token_is_rejected(client: TestClient) -> None:
    """Authentication is on by default: an agent endpoint spends money per call."""
    assert client.get("/agents").status_code == 401


def test_a_request_with_the_wrong_token_is_rejected(client: TestClient) -> None:
    """A wrong token is 403, distinct from 401's "you sent none"."""
    assert client.get("/agents", headers=_auth("nope")).status_code == 403


def test_a_malformed_authorization_header_is_rejected(client: TestClient) -> None:
    """Basic auth, or a bare token, is not a bearer token."""
    response = client.get("/agents", headers={"Authorization": "Basic secret-token"})

    assert response.status_code == 401


def test_agents_are_listed_by_declared_name(client: TestClient) -> None:
    """The `name` an agent declares is the identity an operator and a client both use."""
    response = client.get("/agents", headers=_auth())

    assert response.status_code == 200
    assert response.json() == {"agents": ["support_agent"]}


def test_the_app_is_imported_once_not_once_per_request(project: Path) -> None:
    """Serving imports the app at startup and keeps it: module-level state survives a request.

    `main.py` and everything under `app/` used to be dropped from `sys.modules` and imported
    again on every request, so whatever an app kept at module level -- an HTTP client, a
    connection pool, a cache -- was rebuilt under runs already holding the old one, and the
    event loop paid for the imports. The line appended to `main.py` here stands in for all of
    that: one import, however many requests.
    """
    main_file = project / "main.py"
    main_file.write_text(
        main_file.read_text()
        + '\nwith open(__file__ + ".imports", "a") as _f:\n    _f.write("x")\n'
    )

    client = TestClient(create_app(project, api_key="secret-token"))
    for _ in range(3):
        run = client.post("/agents/support_agent/runs", json={"message": "hi"}, headers=_auth())
        assert run.status_code == 200
    assert client.get("/agents", headers=_auth()).status_code == 200

    assert Path(f"{main_file}.imports").read_text() == "x"


def test_a_run_returns_output_status_usage_and_a_trace_id(client: TestClient) -> None:
    """The wire shape a client depends on, including the handle for `runa traces show`."""
    response = client.post("/agents/support_agent/runs", json={"message": "hello"}, headers=_auth())

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["output"] == "items=1"
    assert body["usage"]["total_tokens"] == 7
    assert body["trace_id"]


def test_a_paused_run_names_the_tool_awaiting_approval(pausing_client: TestClient) -> None:
    """The fields a human decision needs: which tool, with which arguments, on whose behalf.

    This is the shape `Interruption` actually has. `serve` used to name the tool `tool_name`,
    which no interruption has ever carried, so every paused run reported `null` here.
    """
    interruptions = _paused_body(pausing_client)["interruptions"]

    assert len(interruptions) == 1
    assert interruptions[0]["name"] == "issue_refund"
    assert interruptions[0]["call_id"] == "call_1"
    assert interruptions[0]["agent"] == "refund_agent"
    assert json.loads(interruptions[0]["arguments"]) == {"amount": 75}


def test_a_paused_run_carries_the_state_that_resolves_it(
    pausing_client: TestClient, project: Path
) -> None:
    """A status of `"paused"` is useless without the blob that resumes it, so it travels too.

    Rebuilt here the way the other side of an HTTP boundary would: `RunState.from_json` against a
    fresh instance of the agent that started the run.
    """
    state_json = _paused_body(pausing_client)["state"]

    with loaded_app(project):
        classes = {
            getattr(cls, "name", None): cls
            for cls in iter_agent_classes(project / "app" / "agents")
        }
        state = RunState.from_json(classes["refund_agent"](), state_json)

    assert [item.name for item in state.pending] == ["issue_refund"]


def _stream_run(client: TestClient, body: dict[str, Any]) -> dict[str, Any]:
    """Stream a turn for the refund agent and return the `run` event that ends it."""
    with client.stream(
        "POST", "/agents/refund_agent/runs/stream", json=body, headers=_auth()
    ) as response:
        assert response.status_code == 200
        events = [
            json.loads(line.removeprefix("data: "))
            for line in response.iter_lines()
            if line.startswith("data: ") and not line.endswith("[DONE]")
        ]

    runs = [event for event in events if event.get("type") == "run"]
    assert len(runs) == 1
    return runs[0]


def _resume(
    client: TestClient, paused: dict[str, Any], decisions: list[dict[str, Any]]
) -> dict[str, Any]:
    """Hand a paused payload's `state` back with `decisions`, the way a client resolves a pause."""
    response = client.post(
        "/agents/refund_agent/runs",
        json={"state": paused["state"], "decisions": decisions},
        headers=_auth(),
    )

    assert response.status_code == 200
    body: dict[str, Any] = response.json()
    return body


def test_an_approved_pause_resumes_through_the_route_it_paused_on(
    pausing_client: TestClient,
) -> None:
    """The inverse of a paused response: the `state` it carried, plus what the human decided.

    `serve` used to report a pause and accept nothing back, which left every deploying app to
    re-derive `RunState.from_json`, the decisions and the re-render for itself.
    """
    paused = _paused_body(pausing_client)

    body = _resume(pausing_client, paused, [{"call_id": "call_1", "approve": True}])

    assert body["status"] == "completed"
    assert body["output"] == "refunded 75"
    assert body["interruptions"] == []
    assert body["state"] is None


def test_a_rejected_pause_tells_the_model_why(pausing_client: TestClient) -> None:
    """`approve: false` skips the call, and `rejection_message` is what the model is told."""
    paused = _paused_body(pausing_client)

    body = _resume(
        pausing_client,
        paused,
        [{"call_id": "call_1", "approve": False, "rejection_message": "not authorized"}],
    )

    assert body["status"] == "completed"
    assert body["output"] == "not authorized"


def test_an_undecided_interruption_pauses_again(pausing_client: TestClient) -> None:
    """Nothing is invented for a call nobody decided: the run comes back paused on it."""
    paused = _paused_body(pausing_client)

    body = _resume(pausing_client, paused, [])

    assert body["status"] == "paused"
    assert [item["call_id"] for item in body["interruptions"]] == ["call_1"]


def test_a_streamed_run_pauses_and_resumes_on_its_own_route(pausing_client: TestClient) -> None:
    """A streaming client resumes streaming: the state travels on the route it came from.

    Which is why a resume is the same route taking a different body rather than a second pair of
    endpoints -- those would strand exactly this client.
    """
    paused = _stream_run(pausing_client, {"message": "refund $75"})
    assert paused["status"] == "paused"

    resumed = _stream_run(
        pausing_client,
        {"state": paused["state"], "decisions": [{"call_id": "call_1", "approve": True}]},
    )

    assert resumed["status"] == "completed"
    assert resumed["output"] == "refunded 75"


def test_a_resume_carries_the_session_the_run_paused_on(pausing_client: TestClient) -> None:
    """A session-backed run saves nothing while paused: the whole turn lands on resume.

    Which is why `session_id` belongs on a resume at all -- the same one the pause started with,
    exactly as `Agent.run` wants it.
    """
    first = pausing_client.post(
        "/agents/refund_agent/runs",
        json={"message": "refund $75", "session_id": "conv-refund"},
        headers=_auth(),
    ).json()
    assert first["status"] == "paused"
    assert asyncio.run(db.session("conv-refund").get_items()) == []

    resumed = pausing_client.post(
        "/agents/refund_agent/runs",
        json={
            "state": first["state"],
            "decisions": [{"call_id": "call_1", "approve": True}],
            "session_id": "conv-refund",
        },
        headers=_auth(),
    ).json()

    assert resumed["status"] == "completed"
    stored = asyncio.run(db.session("conv-refund").get_items())
    assert stored[0]["content"] == "refund $75"
    assert stored[-1]["content"] == "refunded 75"


def test_a_decision_for_a_call_this_pause_does_not_have_is_a_422(
    pausing_client: TestClient,
) -> None:
    """Dropping it silently would pause the run again with nothing saying why."""
    paused = _paused_body(pausing_client)

    response = pausing_client.post(
        "/agents/refund_agent/runs",
        json={"state": paused["state"], "decisions": [{"call_id": "call_9", "approve": True}]},
        headers=_auth(),
    )

    assert response.status_code == 422
    assert "call_9" in json.dumps(response.json())


def test_an_unreadable_state_is_a_422(pausing_client: TestClient) -> None:
    """The blob is the client's, so a version this Runa cannot read is its error, not a 500."""
    paused = _paused_body(pausing_client)
    stale = {**paused["state"], "schema_version": 99}

    response = pausing_client.post(
        "/agents/refund_agent/runs", json={"state": stale}, headers=_auth()
    )

    assert response.status_code == 422


def test_a_message_and_a_state_together_are_a_422(pausing_client: TestClient) -> None:
    """They are the same argument of `Agent.run`, so a body carrying both is not a guess."""
    paused = _paused_body(pausing_client)

    response = pausing_client.post(
        "/agents/refund_agent/runs",
        json={"message": "refund $75", "state": paused["state"]},
        headers=_auth(),
    )

    assert response.status_code == 422


def test_decisions_without_a_state_are_a_422(client: TestClient) -> None:
    """A decision resolves an interruption of a specific paused run, or it resolves nothing."""
    response = client.post(
        "/agents/support_agent/runs",
        json={"message": "hi", "decisions": [{"call_id": "call_1", "approve": True}]},
        headers=_auth(),
    )

    assert response.status_code == 422


def test_a_completed_run_carries_no_state(client: TestClient) -> None:
    """Only a paused run has something to resume, so `state` is `None` for every other status."""
    response = client.post("/agents/support_agent/runs", json={"message": "hi"}, headers=_auth())

    assert response.json()["state"] is None


def test_an_unknown_agent_is_404(client: TestClient) -> None:
    """A typo in the path is an operator error, not a 500."""
    response = client.post("/agents/nope/runs", json={"message": "hi"}, headers=_auth())

    assert response.status_code == 404


def test_a_missing_message_is_a_422(client: TestClient) -> None:
    """The request body is validated before an agent is ever built."""
    response = client.post("/agents/support_agent/runs", json={}, headers=_auth())

    assert response.status_code == 422


def test_concurrent_stateless_requests_do_not_share_history(client: TestClient) -> None:
    """The per-request agent contract, observed end to end.

    The stub answers with how many items it was handed. If the server reused one `Agent`
    instance, the second request would see the first conversation's turns.
    """
    first = client.post("/agents/support_agent/runs", json={"message": "one"}, headers=_auth())
    second = client.post("/agents/support_agent/runs", json={"message": "two"}, headers=_auth())

    assert first.json()["output"] == "items=1"
    assert second.json()["output"] == "items=1"


def test_a_session_id_carries_the_conversation_forward(client: TestClient) -> None:
    """Continuity is a `session_id`, which is also what lets a second replica serve the turn."""
    body: dict[str, Any] = {"message": "one", "session_id": "conv-1"}
    client.post("/agents/support_agent/runs", json=body, headers=_auth())
    second = client.post(
        "/agents/support_agent/runs",
        json={"message": "two", "session_id": "conv-1"},
        headers=_auth(),
    )

    assert second.json()["output"] == "items=3"  # user, assistant, user


def test_separate_sessions_stay_separate(client: TestClient) -> None:
    """Two users on one server must not see each other's turns."""
    client.post(
        "/agents/support_agent/runs",
        json={"message": "one", "session_id": "user-a"},
        headers=_auth(),
    )
    other = client.post(
        "/agents/support_agent/runs",
        json={"message": "one", "session_id": "user-b"},
        headers=_auth(),
    )

    assert other.json()["output"] == "items=1"


def test_a_session_is_persisted_to_the_projects_database(client: TestClient, project: Path) -> None:
    """Sessions land in the app's own `db/runa.db`, where `runa ui` and `runa chat` find them."""
    client.post(
        "/agents/support_agent/runs",
        json={"message": "one", "session_id": "conv-x"},
        headers=_auth(),
    )

    assert (project / "db" / "runa.db").exists()


def test_no_auth_serves_every_route_openly(project: Path) -> None:
    """`--no-auth` is a real escape hatch for local use, not a soft warning."""
    open_client = TestClient(create_app(project, api_key=None))

    assert open_client.get("/agents").status_code == 200


def test_resolve_api_key_refuses_to_start_unauthenticated_by_accident(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unset variable must be an error, not a silently open server."""
    monkeypatch.delenv("RUNA_API_KEY", raising=False)

    with pytest.raises(MissingAPIKey, match="RUNA_API_KEY is not set"):
        resolve_api_key(no_auth=False)


def test_resolve_api_key_reads_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """The token comes from `RUNA_API_KEY`, the one variable a deployment has to set."""
    monkeypatch.setenv("RUNA_API_KEY", "from-env")

    assert resolve_api_key(no_auth=False) == "from-env"


def test_resolve_api_key_returns_none_for_no_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    """`--no-auth` wins over whatever is in the environment."""
    monkeypatch.setenv("RUNA_API_KEY", "ignored")

    assert resolve_api_key(no_auth=True) is None


def test_the_scaffolded_asgi_module_is_a_working_app(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`runa new`'s `asgi.py` really serves the app, so the embedding path is not just documented.

    Loaded the way a server loads it (`uvicorn asgi:app`), which also pins that its `app` is built
    at import time and takes its token from the environment, not from `runa serve`'s flags.
    """
    monkeypatch.setenv("RUNA_API_KEY", "from-env")

    spec = importlib.util.spec_from_file_location("demo_asgi", project / "asgi.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    client = TestClient(module.app)
    assert client.get("/health").status_code == 200
    assert client.get("/agents").status_code == 401
    response = client.get("/agents", headers={"Authorization": "Bearer from-env"})
    assert response.json() == {"agents": ["support_agent"]}


def test_streaming_ends_with_the_finished_run(client: TestClient) -> None:
    """A streaming client still gets the trace id and usage, in a final `run` event."""
    with client.stream(
        "POST",
        "/agents/support_agent/runs/stream",
        json={"message": "hello"},
        headers=_auth(),
    ) as response:
        assert response.status_code == 200
        payloads = [
            json.loads(line.removeprefix("data: "))
            for line in response.iter_lines()
            if line.startswith("data: ") and not line.endswith("[DONE]")
        ]

    run_events = [p for p in payloads if p.get("type") == "run"]
    assert len(run_events) == 1
    assert run_events[0]["status"] == "completed"
    assert run_events[0]["trace_id"]


def test_streaming_actually_streams_tokens(client: TestClient) -> None:
    """Tokens must arrive as `token` events; only emitting the final run is not streaming."""
    with client.stream(
        "POST",
        "/agents/support_agent/runs/stream",
        json={"message": "hello"},
        headers=_auth(),
    ) as response:
        payloads = [
            json.loads(line.removeprefix("data: "))
            for line in response.iter_lines()
            if line.startswith("data: ") and not line.endswith("[DONE]")
        ]

    tokens = [p["text"] for p in payloads if p.get("type") == "token"]
    assert "".join(tokens) == "items=1"


def test_a_mid_stream_failure_is_reported_as_an_event(project: Path) -> None:
    """A stream cannot become a 500: the status went out with the first byte.

    Truncating silently would leave a client unable to tell a finished answer from a dropped
    connection, so an unexpected failure has to arrive as an `error` event instead. What it says
    is deliberately generic: the exception's type and message belong in the server log, not in a
    caller's hands (CWE-209).
    """
    broken = project / "app" / "stub_model.py"
    broken.write_text(
        broken.read_text().replace(
            "    async def stream_response",
            "    async def _disabled_stream_response",
        )
    )
    client = TestClient(create_app(project, api_key="secret-token"))

    with client.stream(
        "POST",
        "/agents/support_agent/runs/stream",
        json={"message": "hello"},
        headers=_auth(),
    ) as response:
        payloads = [
            json.loads(line.removeprefix("data: "))
            for line in response.iter_lines()
            if line.startswith("data: ") and not line.endswith("[DONE]")
        ]

    errors = [p for p in payloads if p.get("type") == "error"]
    assert errors
    assert errors[0]["error"] == "the run failed"
