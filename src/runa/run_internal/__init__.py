"""`runa.run_internal`: the turn loop behind `Agent.run`, and its execution-time helpers.

One turn: call the model, then either return its text (subject to output guardrails) or execute
whatever tools it called (subject to tool guardrails and `needs_approval`) and loop. A handoff is
just a tool call whose name matches a registered `Handoff`; calling it switches `current_agent` for
the rest of the run. Tracing spans (`runa.tracing.Span`/`Trace`) are emitted directly as the loop
runs, so this is the one and only tracer.

Nothing here is public API, and there is no public class in front of it either: `Agent.run` calls
the turn loop directly, so there is no second way to run an agent. What a caller does touch lives at
the top level: `Run`/ `RunStream`, `RunState`, `Interruption`, and the stream-event types
(`runa.run`, `runa.run_state`, `runa.stream_events`). This package holds only execution-time detail:
`run_loop` (the turn loop itself), `guardrails`, `tool_execution`, `streaming`, `active_run` (the
one value the loop passes down), `run_config` (`RunConfig`, what `Agent` settles before the loop
starts -- no public signature takes one, so it belongs here and not beside `RunState`),
`agent_shape` (what more than one of them reads off an Agent), and `spans` (tracing span helpers).
"""
