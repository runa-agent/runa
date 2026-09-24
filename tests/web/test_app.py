"""Tests for `runa.web.app`: every `runa ui` route, over a scaffolded project."""

import asyncio
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from runa.cli.generate import generate_agent
from runa.cli.new import scaffold_project
from runa.eval.case import Case
from runa.eval.evaluation.core import EvaluationResult, Status
from runa.eval.report import CaseReport, Report
from runa.eval.storage import save_report
from runa.eval.tracing.adapter import AgentRun
from runa.session import SQLiteSession
from runa.tracing.spans import Span
from runa.tracing.storage import save_trace
from runa.tracing.traces import Trace
from runa.web.app import create_app


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A scaffolded project with one agent, one session, two traces, and one eval run."""
    project_dir = scaffold_project("demo", root=tmp_path)
    generate_agent("SupportAgent", root=project_dir)
    db_path = project_dir / "db" / "runa.db"

    session = SQLiteSession("support_agent-1", db_path=db_path)
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
    save_trace(trace, db_path=db_path)

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
    save_trace(trace2, db_path=db_path)

    save_report(
        Report(
            agent_name="support_agent",
            cases=[
                CaseReport(
                    index=0,
                    case=Case(input="hi", expected="hi"),
                    run=AgentRun(input="hi", final_output="hi"),
                    results=[
                        EvaluationResult(
                            metric="task_completion", status=Status.PASS, reason="ok", score=1.0
                        )
                    ],
                )
            ],
        ),
        db_path=db_path,
    )
    return project_dir


@pytest.fixture
def client(project: Path) -> TestClient:
    """A `TestClient` for `create_app(project)`."""
    return TestClient(create_app(project))


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

    assert response.status_code == 200
    assert "Added to <code>evals/turn.jsonl</code>" in response.text
    line = (project / "evals" / "turn.jsonl").read_text().splitlines()[-1]
    assert json.loads(line) == {
        "input": "Where is my order?",
        "expected": "Looks up the order",
        "metadata": {"trace_id": "trace_1"},
    }


def test_trace_detail_hides_add_to_evals_without_a_recorded_input(client: TestClient) -> None:
    """A trace whose root agent span recorded no input has nothing to add."""
    assert "Add to evals" not in client.get("/traces/trace_2").text


def test_trace_detail_404s_for_an_unknown_trace(client: TestClient) -> None:
    """`/traces/{id}` returns 404 for a trace id `db/runa.db` has no record of."""
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
