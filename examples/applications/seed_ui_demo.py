"""Seed a scaffolded project with realistic sample data for trying `runa ui`.

Complements `tour.py`: that one exercises every primitive through a real model call (needs
`OPENAI_API_KEY`, costs tokens, and its output varies run to run); this one fakes the data
directly, so it's free, instant, and exactly reproducible, good for iterating on the dashboard
itself rather than the agent behavior behind it.

Covers every page `runa ui` has: two agents with real `subagents` wiring (a handoff and a
delegate), a multi-turn session, an errored run, a handoff transferring control mid-turn, a
delegate call (and its own separate, session-less trace), and an eval run with a pass and a fail.

A trace's spans and a turn's chat messages are written in the same order `run_loop.py` actually
produces them (trace saved, then its messages persisted, with a short sleep between turns --
SQLite's `CURRENT_TIMESTAMP` only has 1-second resolution), so the session page's merge-by-
timestamp timeline sorts them the way it would for a real run.

Run it:

    make ui-demo

which seeds `ui_demo/` (gitignored, safe to delete and re-run) and launches `runa ui` over it.
"""

import asyncio
import shutil
import time
from pathlib import Path

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

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "ui_demo"
shutil.rmtree(OUT_DIR, ignore_errors=True)

project = scaffold_project("ui_demo", root=REPO_ROOT)
generate_agent("SupportAgent", root=project, model="gpt-5.4-nano")
generate_agent("BillingAgent", root=project, model="gpt-5.4-nano")
generate_agent("PolicyResearcherAgent", root=project, model="gpt-5.4-nano")
db_path = project / "db" / "runa.db"

# Wire the real subagent relationships (`generate_agent` has no --subagent flag; you write these
# by hand, same as tools/guardrails) so the Agents page's "Subagents" chip actually shows them --
# the handoff/delegate traces below only fake what a *run* looks like, this is what makes the
# underlying agent code match that story.
(project / "app" / "agents" / "policy_researcher_agent.py").write_text(
    "from runa import Agent\n"
    "\n"
    "\n"
    "class PolicyResearcherAgent(Agent):\n"
    '    name = "policy_researcher"\n'
    '    model = "gpt-5.4-nano"\n'
)
(project / "app" / "agents" / "support_agent.py").write_text(
    "from app.agents.billing_agent import BillingAgent\n"
    "from app.agents.policy_researcher_agent import PolicyResearcherAgent\n"
    "from runa import Agent\n"
    "\n"
    "\n"
    "class SupportAgent(Agent):\n"
    '    name = "support_agent"\n'
    '    model = "gpt-5.4-nano"\n'
    "    subagents = [BillingAgent.handoff, PolicyResearcherAgent.delegate]\n"
)

session1 = SQLiteSession("support_agent-1", db_path=db_path)
session2 = SQLiteSession("billing_agent-1", db_path=db_path)


async def add(session: SQLiteSession, role: str, content: str) -> None:
    """Append one message to `session`'s history."""
    await session.add_items([{"role": role, "content": content}])


def now() -> float:
    """Real `time.time()`, not an arithmetic offset -- a span's "duration" is real elapsed time."""
    return time.time()


# Turn 1, session 1: a clean multi-span run (agent -> llm -> tool -> llm)
root_start = now()
llm1_start = now()
time.sleep(0.3)
llm1_end = now()
tool1_start = now()
time.sleep(0.2)
tool1_end = now()
llm2_start = now()
time.sleep(0.3)
llm2_end = now()
root_end = now()

trace_ok = Trace(
    id="trace_ok",
    name="support_agent",
    start_time=root_start,
    end_time=root_end,
    session_id="support_agent-1",
)
trace_ok.spans = [
    Span(
        id="root",
        trace_id="trace_ok",
        parent_id=None,
        name="support_agent",
        type="agent",
        start_time=root_start,
        end_time=root_end,
    ),
    Span(
        id="llm1",
        trace_id="trace_ok",
        parent_id="root",
        name="gpt-5.4-nano",
        type="llm",
        start_time=llm1_start,
        end_time=llm1_end,
        input={"messages": [{"role": "user", "content": "my invoice looks wrong"}]},
        output={"content": "let me look up your invoice", "usage": {"total_tokens": 412}},
    ),
    Span(
        id="tool1",
        trace_id="trace_ok",
        parent_id="root",
        name="lookup_invoice",
        type="tool",
        start_time=tool1_start,
        end_time=tool1_end,
        input={"invoice_id": "INV-2201"},
        output={"status": "found", "amount": 42.0},
    ),
    Span(
        id="llm2",
        trace_id="trace_ok",
        parent_id="root",
        name="gpt-5.4-nano",
        type="llm",
        start_time=llm2_start,
        end_time=llm2_end,
        output={"content": "your invoice total is $42.00", "usage": {"total_tokens": 268}},
    ),
]
save_trace(trace_ok, db_path=db_path)
asyncio.run(add(session1, "user", "my invoice looks wrong"))
asyncio.run(add(session1, "assistant", "your invoice total is $42.00"))
time.sleep(1.1)

# Turn 2, session 1: a short follow-up, showing one session accruing multiple traces
root3_start = now()
time.sleep(0.6)
root3_end = now()
trace_ok2 = Trace(
    id="trace_ok2",
    name="support_agent",
    start_time=root3_start,
    end_time=root3_end,
    session_id="support_agent-1",
)
trace_ok2.spans = [
    Span(
        id="root3",
        trace_id="trace_ok2",
        parent_id=None,
        name="support_agent",
        type="agent",
        start_time=root3_start,
        end_time=root3_end,
    ),
    Span(
        id="llm4",
        trace_id="trace_ok2",
        parent_id="root3",
        name="gpt-5.4-nano",
        type="llm",
        start_time=root3_start,
        end_time=root3_end,
        output={"content": "anything else I can help with?", "usage": {"total_tokens": 140}},
    ),
]
save_trace(trace_ok2, db_path=db_path)
asyncio.run(add(session1, "user", "nope that's all, thanks"))
asyncio.run(add(session1, "assistant", "anything else I can help with?"))
time.sleep(1.1)

# Turn 1, session 2: an errored run, with a nested guardrail failure
root2_start = now()
llm3_start = now()
time.sleep(0.4)
llm3_end = now()
guard1_start = now()
time.sleep(0.3)
guard1_end = now()
root2_end = now()
trace_err = Trace(
    id="trace_err",
    name="billing_agent",
    start_time=root2_start,
    end_time=root2_end,
    session_id="billing_agent-1",
)
trace_err.spans = [
    Span(
        id="root2",
        trace_id="trace_err",
        parent_id=None,
        name="billing_agent",
        type="agent",
        start_time=root2_start,
        end_time=root2_end,
        status="error",
    ),
    Span(
        id="llm3",
        trace_id="trace_err",
        parent_id="root2",
        name="gpt-5.4-nano",
        type="llm",
        start_time=llm3_start,
        end_time=llm3_end,
        output={"content": "sure, here's the refund", "usage": {"total_tokens": 190}},
    ),
    Span(
        id="guard1",
        trace_id="trace_err",
        parent_id="root2",
        name="no_unauthorized_refunds",
        type="guardrail",
        start_time=guard1_start,
        end_time=guard1_end,
        status="error",
        error="refund of $500 exceeds the $100 auto-approval limit",
    ),
]
save_trace(trace_err, db_path=db_path)
asyncio.run(add(session2, "user", "can you refund me $500?"))
asyncio.run(add(session2, "assistant", "sure, here's the refund"))
time.sleep(1.1)

# Turn 1, session 3: a HANDOFF -- support_agent transfers the whole conversation to
# billing_agent mid-turn. One trace, flat: the root "agent" span keeps its original name
# (support_agent, since that's who started the turn) and every span, before and after the
# switch, stays a sibling under it -- `run_loop.py` never creates a second root span for the
# agent that took over, it just keeps appending to the same one.
session3 = SQLiteSession("support_agent-3", db_path=db_path)
root4_start = now()
llm5_start = now()
time.sleep(0.3)
llm5_end = now()
handoff_start = now()
handoff_end = now()
llm6_start = now()
time.sleep(0.4)
llm6_end = now()
root4_end = now()
trace_handoff = Trace(
    id="trace_handoff",
    name="support_agent",
    start_time=root4_start,
    end_time=root4_end,
    session_id="support_agent-3",
)
trace_handoff.spans = [
    Span(
        id="root4",
        trace_id="trace_handoff",
        parent_id=None,
        name="support_agent",
        type="agent",
        start_time=root4_start,
        end_time=root4_end,
    ),
    Span(
        id="llm5",
        trace_id="trace_handoff",
        parent_id="root4",
        name="gpt-5.4-nano",
        type="llm",
        start_time=llm5_start,
        end_time=llm5_end,
        input={"messages": [{"role": "user", "content": "cancel my subscription and refund me"}]},
    ),
    Span(
        id="handoff1",
        trace_id="trace_handoff",
        parent_id="root4",
        name="transfer_to_billing_agent",
        type="handoff",
        start_time=handoff_start,
        end_time=handoff_end,
    ),
    Span(
        id="llm6",
        trace_id="trace_handoff",
        parent_id="root4",
        name="gpt-5.4-nano",
        type="llm",
        start_time=llm6_start,
        end_time=llm6_end,
        output={
            "content": "done, your subscription is cancelled and the refund is on its way",
            "usage": {"total_tokens": 205},
        },
    ),
]
save_trace(trace_handoff, db_path=db_path)
asyncio.run(add(session3, "user", "cancel my subscription and refund me"))
asyncio.run(
    add(session3, "assistant", "done, your subscription is cancelled and the refund is on its way")
)
time.sleep(1.1)

# Turn 1, session 4: a DELEGATE -- support_agent calls policy_researcher as a subroutine.
# Unlike a handoff, this is TWO separate traces, not one: `agent_as_tool` (runa/handoff.py)
# runs the delegate via a plain `agent.run()` call, which always starts its own independent
# `Trace`. The caller's trace shows it as its own `type="delegate"` span (distinct from a plain
# tool call, since FunctionTool.is_delegate=True), but the delegate's own execution only shows up
# as its own trace, with no session (a delegate call is never given `session=`) and, today,
# nothing in the outer trace's "delegate" span linking to it -- you'd only find it by knowing its
# id.
session4 = SQLiteSession("support_agent-4", db_path=db_path)
root5_start = now()
llm7_start = now()
time.sleep(0.3)
llm7_end = now()
delegate_start = now()

# The delegate's own, separate, session-less trace, produced by that nested agent.run()
inner_root_start = now()
inner_llm_start = now()
time.sleep(0.4)
inner_llm_end = now()
inner_root_end = now()
trace_delegate_inner = Trace(
    id="trace_delegate_inner",
    name="policy_researcher",
    start_time=inner_root_start,
    end_time=inner_root_end,
)
trace_delegate_inner.spans = [
    Span(
        id="inner_root",
        trace_id="trace_delegate_inner",
        parent_id=None,
        name="policy_researcher",
        type="agent",
        start_time=inner_root_start,
        end_time=inner_root_end,
    ),
    Span(
        id="inner_llm",
        trace_id="trace_delegate_inner",
        parent_id="inner_root",
        name="gpt-5.4-nano",
        type="llm",
        start_time=inner_llm_start,
        end_time=inner_llm_end,
        input={"messages": [{"role": "user", "content": "what is the late delivery policy?"}]},
        output={
            "content": "orders more than 5 business days late qualify for a partial refund",
            "usage": {"total_tokens": 96},
        },
    ),
]
save_trace(trace_delegate_inner, db_path=db_path)

delegate_end = now()
llm8_start = now()
time.sleep(0.3)
llm8_end = now()
root5_end = now()
trace_delegate_outer = Trace(
    id="trace_delegate_outer",
    name="support_agent",
    start_time=root5_start,
    end_time=root5_end,
    session_id="support_agent-4",
)
trace_delegate_outer.spans = [
    Span(
        id="root5",
        trace_id="trace_delegate_outer",
        parent_id=None,
        name="support_agent",
        type="agent",
        start_time=root5_start,
        end_time=root5_end,
    ),
    Span(
        id="llm7",
        trace_id="trace_delegate_outer",
        parent_id="root5",
        name="gpt-5.4-nano",
        type="llm",
        start_time=llm7_start,
        end_time=llm7_end,
        input={"messages": [{"role": "user", "content": "what's your late delivery policy?"}]},
    ),
    Span(
        id="delegate1",
        trace_id="trace_delegate_outer",
        parent_id="root5",
        name="policy_researcher",
        type="delegate",
        start_time=delegate_start,
        end_time=delegate_end,
        input={"input": "what is the late delivery policy?"},
        output="orders more than 5 business days late qualify for a partial refund",
    ),
    Span(
        id="llm8",
        trace_id="trace_delegate_outer",
        parent_id="root5",
        name="gpt-5.4-nano",
        type="llm",
        start_time=llm8_start,
        end_time=llm8_end,
        output={
            "content": "orders more than 5 business days late qualify for a partial refund",
            "usage": {"total_tokens": 142},
        },
    ),
]
save_trace(trace_delegate_outer, db_path=db_path)
asyncio.run(add(session4, "user", "what's your late delivery policy?"))
asyncio.run(
    add(
        session4,
        "assistant",
        "orders more than 5 business days late qualify for a partial refund",
    )
)

save_report(
    Report(
        agent_name="support_agent",
        cases=[
            CaseReport(
                index=0,
                case=Case(input="my invoice looks wrong", expected="looks up the invoice"),
                run=AgentRun(input="my invoice looks wrong", final_output="your total is $42.00"),
                results=[
                    EvaluationResult(
                        metric="task_completion", status=Status.PASS, reason="resolved", score=1.0
                    )
                ],
            ),
            CaseReport(
                index=1,
                case=Case(input="refund me $500", expected="declines politely"),
                run=AgentRun(input="refund me $500", final_output="sure, here's the refund"),
                results=[
                    EvaluationResult(
                        metric="task_completion",
                        status=Status.FAIL,
                        reason="approved a refund above the auto-approval limit",
                        score=0.0,
                    )
                ],
            ),
        ],
    ),
    db_path=db_path,
)

print(f"seeded {project}")
