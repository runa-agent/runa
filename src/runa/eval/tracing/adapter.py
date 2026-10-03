"""eval/tracing/adapter.py: run one `Case` through an `Agent` and normalize the result.

`Runner.run()`'s own `RunResult.trace` is used directly: evaluation and observability read the
same `Trace`/`Span` data instead of two parallel execution-history models: `AgentRun.tool_calls`
is derived straight from `AgentRun.trace.spans`.
"""

import time
from dataclasses import dataclass, field

from runa.agent import _MODEL_PROVIDER, Agent
from runa.eval.case import Case
from runa.exceptions import RunaError
from runa.run_config import RunConfig
from runa.runner import Runner
from runa.tracing import Trace

_EMPTY_TRACE = Trace(id="", name="", start_time=0.0, end_time=0.0, spans=[], metadata={})


@dataclass
class ToolCallRecord:
    """One tool call an agent made during a run, paired with its output."""

    name: str
    arguments: str
    output: str | None = None


@dataclass
class AgentRun:
    """A `Case`'s input run through an `Agent`, normalized for evaluation."""

    input: str
    final_output: str | None
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    error: str | None = None
    latency: float = 0.0
    trace: Trace = field(default_factory=lambda: _EMPTY_TRACE)


async def run_agent_for_eval(agent: Agent, case: Case) -> AgentRun:
    """Run `case.input` through `agent` and capture an `AgentRun`.

    A run that raises (a guardrail tripwire, `MaxTurnsExceeded`, ...) is
    captured as an `AgentRun` with `error` set rather than propagating, so a
    bad case doesn't stop the rest of a dataset from evaluating.
    """
    start = time.monotonic()
    run_config = RunConfig(model_provider=_MODEL_PROVIDER, workflow_name=type(agent).__name__)
    try:
        result = await Runner.run(agent, case.input, run_config=run_config)
    except RunaError as exc:
        latency = time.monotonic() - start
        trace = (
            exc.run_data.trace if exc.run_data and exc.run_data.trace is not None else _EMPTY_TRACE
        )
        return AgentRun(
            input=case.input, final_output=None, error=str(exc), latency=latency, trace=trace
        )
    latency = time.monotonic() - start

    tool_calls = [
        ToolCallRecord(
            name=span.name,
            arguments=str(span.input) if span.input is not None else "",
            output=str(span.output) if span.output is not None else None,
        )
        for span in result.trace.spans
        if span.type == "tool"
    ]
    return AgentRun(
        input=case.input,
        final_output=result.final_output,
        tool_calls=tool_calls,
        latency=latency,
        trace=result.trace,
    )
