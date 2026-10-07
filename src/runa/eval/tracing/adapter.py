"""eval/tracing/adapter.py: run one `Case` through an `Agent` and normalize the result.

`Run.trace` is carried through as the run's observability record, for `runa ui` and `runa traces`
to display. The graded evidence is not read off it: `AgentRun.tool_calls` is the turn loop's own
`ToolCall` record, because a `"tool"` span's input/output passes through the tracing privacy
policy (`runa.tracing.observe`) first, and a team that stops capturing inputs for PII reasons
would otherwise be grading its agent on arguments the judge never sees.

A case runs through `Agent.run`, the same door an application's own call uses, so an agent is
evaluated with the wiring it actually ships with: its guardrails, its memory and knowledge, its
`max_turns`/`max_tokens`/`timeout`. Each case gets a `_fresh()` copy of the agent for the same
reason `agent_as_tool` does: a dataset's cases are independent, and `evaluate_agent` runs up to
`concurrency` of them at once, so sharing one instance would let one case's conversation leak
into the next.
"""

import time
from dataclasses import dataclass, field

from runa.agent import Agent
from runa.eval.case import Case
from runa.tool import ToolCall
from runa.tracing import Trace

_EMPTY_TRACE = Trace(id="", name="", start_time=0.0, end_time=0.0, spans=[], metadata={})


@dataclass
class AgentRun:
    """A `Case`'s input run through an `Agent`, normalized for evaluation."""

    input: str
    final_output: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    error: str | None = None
    latency: float = 0.0
    trace: Trace = field(default_factory=lambda: _EMPTY_TRACE)


async def run_agent_for_eval(agent: Agent, case: Case) -> AgentRun:
    """Run `case.input` through `agent` and capture an `AgentRun`.

    A run that fails (a guardrail tripwire, `MaxTurnsExceeded`, ...) is captured as an `AgentRun`
    with `error` set rather than propagating, so a bad case doesn't stop the rest of a dataset
    from evaluating. `Agent.run` already reports those as `status="error"`, which is what this
    reads instead of catching `RunaError` a second time.
    """
    start = time.monotonic()
    run = await agent._fresh().run(case.input)
    latency = time.monotonic() - start
    trace = run.trace if run.trace is not None else _EMPTY_TRACE

    if run.status == "error":
        return AgentRun(
            input=case.input, final_output=None, error=run.error, latency=latency, trace=trace
        )

    return AgentRun(
        input=case.input,
        final_output=run.output,
        tool_calls=list(run._tool_calls),
        latency=latency,
        trace=trace,
    )
