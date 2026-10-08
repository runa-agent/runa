"""Tests for `runa.serve`: the HTTP surface `runa serve` puts in front of this app's agents.

Covers the three decisions the module makes on a deployment's behalf -- an agent per request,
sessions rather than `self.history`, and authentication on by default -- plus the wire shape a
client actually depends on.
"""

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

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

from typing import Any

from runa import Agent, tool
from runa._types import ModelResponse, Usage


@tool(needs_approval=True)
def issue_refund(amount: float) -> str:
    """Refund `amount` dollars.

    amount: dollars to refund
    """
    return f"refunded {amount}"


class ScriptedModel:
    """Asks for one approval-gated refund, then answers once it has run."""

    def __init__(self) -> None:
        self._messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "issue_refund", "arguments": '{"amount": 75}'},
                    }
                ],
            },
            {"role": "assistant", "content": "refunded 75.0", "tool_calls": None},
        ]

    async def get_response(self, request: Any, **kwargs: Any) -> ModelResponse:
        return ModelResponse(
            output=[self._messages.pop(0)],
            usage=Usage(input_tokens=1, output_tokens=1, total_tokens=2, requests=1),
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
