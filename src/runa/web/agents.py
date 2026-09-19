"""web/agents.py: the Agents page -- every declared `Agent` subclass, read-only.

All data comes from `runa.cli.agents.list_agents`; this module only turns `AgentInfo`s into HTML.
"""

from pathlib import Path

from runa.cli.agents import AgentInfo, list_agents
from runa.web._html import chip, chips, empty_hint, escape, page


def _card(info: AgentInfo) -> str:
    rows = [
        ("Model", chip(info.model, "accent")),
        ("Tools", chips(info.tools)),
        ("Guardrails", chips(info.guardrails)),
        ("Subagents", chips(info.subagents)),
        ("Memory", chip(info.memory, "ok" if info.memory != "off" else "default")),
        ("Knowledge", chip(info.knowledge, "ok" if info.knowledge != "off" else "default")),
    ]
    fields = "".join(
        f'<div class="field-label">{label}</div>{value_html}' for label, value_html in rows
    )
    return f"""<div class="card">
  <h2>{escape(info.name)}<span style="color:var(--muted); font-weight:400"> · \
{escape(info.class_name)}</span></h2>
  {fields}
</div>"""


def render(*, root: Path) -> str:
    """Render `/agents`: one card per `Agent` subclass declared under `app/agents/`."""
    infos = list_agents(root=root)
    body = (
        "".join(_card(info) for info in infos)
        if infos
        else empty_hint(
            "no agents found under app/agents/, run", "runa generate agent MyAgent --model ..."
        )
    )
    return page(
        title="Agents",
        active="Agents",
        body=f'<h1>Agents</h1><p class="subtitle">Every Agent subclass declared in this app.'
        f"</p>{body}",
    )
