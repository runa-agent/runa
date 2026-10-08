"""Tests for `runa.web.app`: every `runa ui` route, over a scaffolded project."""

import asyncio
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from helpers import finished_run

from runa import db
from runa.cli.generate import generate_agent
from runa.cli.new import scaffold_project
from runa.eval.case import Case
from runa.eval.evaluation.core import EvaluationResult, Status
from runa.eval.report import CaseReport, Report
from runa.exceptions import OperatorError
from runa.tracing.spans import Span
from runa.tracing.traces import Trace
from runa.web.app import _trace_url, create_app


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A scaffolded project with one agent, one session, two traces, and one eval run."""
    project_dir = scaffold_project("demo", root=tmp_path)
    generate_agent("SupportAgent", root=project_dir)

    db.use_project(project_dir)
    session = db.session("support_agent-1")
    asyncio.run(session.add_items([{"role": "user", "content": "hi there"}]))

    trace = Trace(
        id="trace_1",
        name="support_agent",
        start_time=0.0,
        end_time=1.0,
        session_id="support_agent-1",
    )
    trace.spans = [
        Span(
            id="s1",
            trace_id="trace_1",
            parent_id=None,
            name="turn",
            type="agent",
            start_time=0.0,
            end_time=1.0,
            status="error",
            error="boom",
            input="Where is my order?",
        ),
        Span(
            id="s2",
            trace_id="trace_1",
            parent_id="s1",
            name="transfer_to_billing_agent",
            type="handoff",
            start_time=0.0,
            end_time=1.0,
        ),
    ]
    db.traces().save(trace)

    trace2 = Trace(
        id="trace_2",
        name="support_agent",
        start_time=2.0,
        end_time=2.5,
        session_id="support_agent-1",
    )
    trace2.spans = [
        Span(
            id="s3",
            trace_id="trace_2",
            parent_id=None,
            name="turn",
            type="agent",
            start_time=2.0,
            end_time=2.5,
        )
    ]
    db.traces().save(trace2)

    db.evals().save(
        Report(
            agent_name="support_agent",
            cases=[
                CaseReport(
                    index=0,
                    case=Case(input="hi", expected="hi"),
                    run=finished_run("hi"),
                    results=[
                        EvaluationResult(
                            metric="task_completion", status=Status.PASS, reason="ok", score=1.0
                        )
                    ],
                )
            ],
        )
    )
    return project_dir


@pytest.fixture
def client(project: Path) -> TestClient:
    """A `TestClient` for `create_app(project)`."""
    return TestClient(create_app(project))


def test_create_app_points_the_db_at_its_own_project(project: Path) -> None:
    """`create_app(root)` is the whole configuration an embedded dashboard gets.

    The `project` fixture sets the project itself to seed data, so clearing it first is what makes
    this an assertion about `create_app` rather than about the fixture. `create_app` is public
    API (`runa new` scaffolds an `asgi.py` around its `serve` counterpart), and the root it is
    handed has to reach every concern: a dashboard reading sessions out of `root` while the agents
    it serves wrote memory to the cwd is the split `runa.db` exists to prevent.
    """
    db.use_project(None)

    TestClient(create_app(project))

    assert db.sqlite_path() == project / "db" / "runa.db"
    assert db.memory_store(dimensions=4).db_path == project / "db" / "runa.db"  # type: ignore[attr-defined]


def test_index_redirects_to_agents(client: TestClient) -> None:
    """`/` redirects to `/agents`, the dashboard's landing page."""
    response = client.get("/", follow_redirects=False)

    assert response.status_code in (302, 307)
    assert response.headers["location"] == "/agents"


def test_agents_page_lists_declared_agents(client: TestClient) -> None:
    """`/agents` shows every declared Agent subclass."""
    response = client.get("/agents")

    assert response.status_code == 200
    assert "support_agent" in response.text


def test_agents_page_reports_a_clean_error_outside_a_runa_project(tmp_path: Path) -> None:
    """`/agents` returns 400 (not a 500 traceback) when `root` isn't a Runa project."""
    response = TestClient(create_app(tmp_path)).get("/agents")

    assert response.status_code == 400


def test_sessions_list_and_detail(client: TestClient) -> None:
    """`/sessions` lists the session; `/sessions/{id}` renders its messages."""
    listing = client.get("/sessions")
    assert "support_agent-1" in listing.text

    detail = client.get("/sessions/support_agent-1")
    assert detail.status_code == 200
    assert "hi there" in detail.text


def test_session_detail_404s_for_an_unknown_session(client: TestClient) -> None:
    """`/sessions/{id}` returns 404 for a session id with no history."""
    assert client.get("/sessions/nope").status_code == 404


def test_session_detail_lists_its_traces(client: TestClient) -> None:
    """`/sessions/{id}` shows the traces produced by that session's turns, spans expanded."""
    detail = client.get("/sessions/support_agent-1")

    assert detail.status_code == 200
    assert 'class="trace-card"' in detail.text
    assert "support_agent" in detail.text
    assert "boom" in detail.text  # the trace's error span, shown inline with no expand click


def test_trace_detail(client: TestClient) -> None:
    """`/traces/{id}` is still reachable directly (no nav tab), showing the trace's spans."""
    detail = client.get("/traces/trace_1")

    assert detail.status_code == 200
    assert "boom" in detail.text
    assert "support_agent-1" in detail.text


def test_session_detail_labels_trace_cards_by_turn_not_agent_name(client: TestClient) -> None:
    """Trace cards read "turn 1", "turn 2", ... in order, not the (redundant) agent name."""
    detail = client.get("/sessions/support_agent-1")

    turn1 = detail.text.index("turn 1")
    turn2 = detail.text.index("turn 2")
    assert turn1 < turn2

    card_section = detail.text[detail.text.index('class="trace-card"') :]
    assert "support_agent" not in card_section.split("</summary>")[0]


def test_trace_detail_strips_the_handoff_tool_name_prefix(client: TestClient) -> None:
    """A handoff span shows the target agent's name, not the full `transfer_to_...` tool name."""
    detail = client.get("/traces/trace_1")

    assert "billing_agent" in detail.text
    assert "transfer_to_billing_agent" not in detail.text


def test_trace_detail_shows_a_divider_after_a_handoff(client: TestClient) -> None:
    """A `class="handoff-divider"` row marks who took over right after the handoff span."""
    detail = client.get("/traces/trace_1")

    assert 'class="handoff-divider">Handoff &middot; billing_agent<' in detail.text


def test_trace_detail_adds_its_input_to_the_agent_s_evals(
    client: TestClient, project: Path
) -> None:
    """The "Add to evals" form appends the trace's input to `evals/<agent>.jsonl`."""
    assert 'action="/traces/trace_1/eval"' in client.get("/traces/trace_1").text

    response = client.post("/traces/trace_1/eval", data={"expected": "Looks up the order"})
    client.post("/traces/trace_1/eval", data={"expected": "a double submit"})

    assert response.status_code == 200
    assert "In <code>evals/turn.jsonl</code>" in response.text
    assert "Add to evals" not in response.text
    lines = (project / "evals" / "turn.jsonl").read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == {
        "input": "Where is my order?",
        "expected": "Looks up the order",
        "metadata": {"trace_id": "trace_1"},
    }


def test_trace_add_to_evals_never_redirects_off_site(client: TestClient) -> None:
    """A trace id crafted to look like a host stays one path segment of this app (CWE-601)."""
    response = client.post("//traces/evil.example.com/eval", follow_redirects=False)

    assert response.status_code == 404
    assert "evil.example.com" not in response.headers.get("location", "")

    assert _trace_url("/evil.example.com") == "/traces/%2Fevil.example.com"
    assert _trace_url("\\\\evil.example.com") == "/traces/%5C%5Cevil.example.com"


def test_trace_detail_hides_add_to_evals_without_a_recorded_input(client: TestClient) -> None:
    """A trace whose root agent span recorded no input has nothing to add."""
    assert "Add to evals" not in client.get("/traces/trace_2").text


def test_evaluation_detail_links_cases_to_traces_and_flags_regressions(
    client: TestClient, project: Path
) -> None:
    """A newer run failing a case the fixture's run passed marks it regressed, with a trace link."""
    failing = CaseReport(
        index=0,
        case=Case(input="hi"),
        run=finished_run("no", trace=Trace(id="trace_1", name="support_agent", start_time=0.0)),
        results=[EvaluationResult(metric="task_completion", status=Status.FAIL, reason="bad")],
    )
    run_id = db.evals().save(Report(agent_name="support_agent", cases=[failing]))

    detail = client.get(f"/evaluations/{run_id}").text

    assert "last run 1/1 passed" in detail
    assert ">regressed<" in detail
    assert 'href="/traces/trace_1"' in detail


def test_trace_detail_404s_for_an_unknown_trace(client: TestClient) -> None:
    """`/traces/{id}` returns 404 for a trace id the store has no record of."""
    assert client.get("/traces/nope").status_code == 404


def test_traces_has_no_nav_tab(client: TestClient) -> None:
    """There's no standalone `/traces` list route or nav entry; sessions cover that."""
    assert client.get("/traces").status_code == 404
    assert "Traces" not in client.get("/agents").text


def test_evaluations_list_and_detail(client: TestClient) -> None:
    """`/evaluations` lists the run; `/evaluations/{id}` shows its per-case results."""
    listing = client.get("/evaluations")
    assert "support_agent" in listing.text

    detail = client.get("/evaluations/1")
    assert detail.status_code == 200
    assert "task_completion" in detail.text


def test_evaluation_detail_404s_for_an_unknown_run(client: TestClient) -> None:
    """`/evaluations/{id}` returns 404 for an `eval_runs.id` that doesn't exist."""
    assert client.get("/evaluations/999").status_code == 404


def test_routes_keep_their_own_status_over_the_operator_error_fallback(
    client: TestClient,
) -> None:
    """A route that names its own type still answers with that status, not the fallback 400.

    `SessionNotFound`/`TraceNotFound`/`EvalRunNotFound` are `OperatorError`s, so a single
    handler would have flattened these three 404s into 400. Unlike `cli/main.py`, this app has
    more than one right answer for an operator error, so the specific routes stay specific.
    """
    assert client.get("/sessions/nope").status_code == 404
    assert client.get("/traces/nope").status_code == 404
    assert client.get("/evaluations/999").status_code == 404


def test_an_unhandled_operator_error_renders_a_400_not_a_500(project: Path) -> None:
    """An `OperatorError` no route catches renders the error page instead of crashing.

    The floor under the specific handlers: a new operator error reaching this app from a
    surface that predates its route is a bad request, not a server fault.
    """

    class _BrandNewOperatorError(OperatorError):
        """A type `web/app.py` has never heard of."""

    def _raise(**_kwargs: object) -> str:
        raise _BrandNewOperatorError("no such thing")

    app = create_app(project)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("runa.web.app.agents_page.render", _raise)
        response = TestClient(app, raise_server_exceptions=False).get("/agents")

    assert response.status_code == 400
    assert "Error" in response.text
