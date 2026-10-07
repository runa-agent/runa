# ADR-0003: eval evidence is recorded by the run, not read off its trace

- Status: accepted
- Date: 2026-10-07

## Context

`eval/tracing/adapter.py` derived `AgentRun.tool_calls` from `Run.trace.spans`,
on the stated grounds that evaluation and observability should read one
execution history rather than two parallel models. The model they shared was the
wrong one. `run_internal/spans.py` applies the tracing privacy policy
(`runa.tracing.observe`) as each span opens and closes, writing the filtered
value onto `span.input`/`span.output`, so the in-memory `Trace` is filtered too,
not just an exported copy. `eval/evaluation/semantic.py` then formatted those
values straight into the judge prompt.

An observability setting therefore decided what an eval graded, with no error and
nothing in the report to say the evidence was degraded:

- `observe(capture_inputs=False)` -- set for exactly the PII reason the knob
  exists -- gave the judge every tool call with empty arguments;
- `observe(capture_outputs=False)` gave it calls with no result;
- `max_tool_result_bytes` needed no configuration at all: it defaults to 32,000,
  so any tool returning more than that was graded against a truncated string.

`runa eval` compares each run against the previous one, so the drift would have
been reported as a regression in the agent.

## Decision

The turn loop records what it ran. `_run_tool_call` already holds the tool's
name, the raw arguments string and the raw result at the point it closes the
span, so it appends a `ToolCall` (`runa/tool.py`) to `_Run.tool_calls`, and
`_finish` carries that list onto `Run._tool_calls`. `run_agent_for_eval` reads
it; `AgentRun.trace` stays, as the display copy for `runa ui` and
`runa traces`.

`ToolCall` is frozen, holds three strings, and is private on `Run`: `trace` is
still how a caller inspects a run, and a public `run.tool_calls` would be a
second shape for a question the trace already answers.

A delegate call is not recorded in its caller's list. A `.delegate` runs a whole
nested `Agent.run()` with its own trace, and records its own calls there; the
caller's list is what the caller's own tools did. This matches what the span
filter (`span.type == "tool"`, which excludes `"delegate"`) already selected.

A paused `RunState` carries the calls that ran before the pause, so a resumed
run reports the whole turn. It is not serialized by `to_json`, for the same
reason trace spans aren't: a restored run reports what it did after being
restored.

The rejected alternative was to stop filtering the in-memory `Trace` and apply
the policy in the exporters instead. It keeps one execution model, and it was
rejected because the policy is also the rule for what this process *holds*, not
only for what it ships out: an app that turns off input capture to avoid keeping
PII in memory would still be keeping it, and every exporter -- including a
caller's own -- would have to remember to apply the policy itself.

## Consequences

- An eval score no longer moves when a tracing setting does. The default
  32,000-byte tool-result truncation can't silently weaken `task_completion` or
  `faithfulness` grading, and `tool_correctness` sees the real arguments.
- There is one more small representation of a run's tool calls, which is the
  honest cost. It is three strings per executed call, bounded by the tool results
  the run already holds in `items`, and it has one writer and one reader.
- A future review will see `ToolCall` beside the `"tool"` spans and propose
  collapsing them. That is this ADR's subject: they answer different questions,
  one gradeable and one displayable, and the display one is governed by a knob
  the graded one must not be.
- `ToolCallRecord` is gone from `runa.eval.tracing`; `ToolCall` replaces it.
