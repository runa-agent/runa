"""`runa.eval.tracing`: turn an `Agent.run()` result into an `AgentRun`."""

from runa.eval.tracing.adapter import AgentRun, run_agent_for_eval

__all__ = ["AgentRun", "run_agent_for_eval"]
