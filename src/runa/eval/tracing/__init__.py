"""`runa.eval.tracing`: turn a `Runner.run()` result into an `AgentRun`."""

from __future__ import annotations

from runa.eval.tracing.adapter import AgentRun, ToolCallRecord, run_agent_for_eval

__all__ = ["AgentRun", "ToolCallRecord", "run_agent_for_eval"]
