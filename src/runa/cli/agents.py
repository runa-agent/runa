"""cli/agents.py: introspect this app's declared `Agent` subclasses under `app/agents/`.

Reads class attributes only (`name`, `model`, `tools`, `guardrails`, `subagents`, `memory`,
`knowledge`) -- no `Agent` is ever instantiated, so listing agents never triggers memory/
knowledge setup, prompt-file creation, or any API call.
"""

from dataclasses import dataclass, field
from pathlib import Path

from runa.agent import Agent, Subagent, _flatten_subagents
from runa.cli._project import iter_agent_classes, loaded_app, require_agents_dir


@dataclass
class AgentInfo:
    """One declared `Agent` subclass, summarized for display."""

    name: str
    class_name: str
    model: str
    tools: list[str] = field(default_factory=list)
    guardrails: list[str] = field(default_factory=list)
    subagents: list[str] = field(default_factory=list)
    memory: str = "off"
    knowledge: str = "off"


def _tool_name(tool: object) -> str:
    return getattr(tool, "name", None) or getattr(tool, "__name__", None) or str(tool)


def _guardrail_name(guardrail: object) -> str:
    name = getattr(guardrail, "name", None)
    if name:
        return str(name)
    function = getattr(guardrail, "guardrail_function", None)
    return getattr(function, "__name__", None) or str(guardrail)


def _subagent_label(subagent: Subagent | type[Agent]) -> str:
    """Label one flattened subagent entry, `Subagent`-wrapped or bare (auto mode).

    `_flatten_subagents` only wraps a bare `Agent` subclass in a `Subagent` when a dict-shaped
    `subagents` gave it an explicit `.handoff`/`.delegate` mode; unwrapped means "auto".
    """
    if isinstance(subagent, Subagent):
        agent_name = getattr(subagent.agent, "name", subagent.agent.__name__)
        return f"{agent_name} ({subagent.mode})"
    agent_name = getattr(subagent, "name", getattr(subagent, "__name__", str(subagent)))
    return f"{agent_name} (auto)"


def _retrieval_label(setting: object) -> str:
    if setting is None:
        return "off"
    return setting if isinstance(setting, str) else "on"


def _describe(agent_cls: type[Agent]) -> AgentInfo:
    subagents_raw = getattr(agent_cls, "subagents", [])
    return AgentInfo(
        name=getattr(agent_cls, "name", agent_cls.__name__),
        class_name=agent_cls.__name__,
        model=getattr(agent_cls, "model", Agent.model),
        tools=[_tool_name(tool) for tool in getattr(agent_cls, "tools", [])],
        guardrails=[_guardrail_name(g) for g in getattr(agent_cls, "guardrails", [])],
        subagents=[_subagent_label(sub) for sub in _flatten_subagents(subagents_raw)],
        memory=_retrieval_label(getattr(agent_cls, "memory", None)),
        knowledge=_retrieval_label(getattr(agent_cls, "knowledge", None)),
    )


def list_agents(*, root: Path) -> list[AgentInfo]:
    """Return every `Agent` subclass declared under `root/app/agents/`, alphabetically by name."""
    agents_dir = require_agents_dir(root)
    with loaded_app(root):
        infos = [_describe(agent_cls) for agent_cls in iter_agent_classes(agents_dir)]
    return sorted(infos, key=lambda info: info.name)


__all__ = ["AgentInfo", "list_agents"]
