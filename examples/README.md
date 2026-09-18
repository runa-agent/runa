# Examples

Runnable code for every primitive in [RUNA.md](../RUNA.md), one folder per primitive, numbered
to match. Every script runs standalone:

```bash
uv run python examples/<folder>/<file>.py
```

| Folder | Primitive |
|---|---|
| [00_quickstart](00_quickstart) | The smallest useful agent |
| [01_agent](01_agent) | [Agent](../RUNA.md#1-agent) |
| [02_tool](02_tool) | [Tool](../RUNA.md#2-tool) |
| [03_guardrail](03_guardrail) | [Guardrail](../RUNA.md#3-guardrail) |
| [04_approval](04_approval) | [Human Approval](../RUNA.md#4-human-approval) |
| [05_subagent](05_subagent) | [Subagent](../RUNA.md#5-subagent-handoffdelegate) |
| [06_session](06_session) | [Session](../RUNA.md#6-session) |
| [07_memory](07_memory) | [Memory](../RUNA.md#7-memory) |
| [08_knowledge](08_knowledge) | [Knowledge](../RUNA.md#8-knowledge) |
| [09_mcp](09_mcp) | [MCP Server](../RUNA.md#9-mcp-server) |
| [10_model](10_model) | [Model](../RUNA.md#10-model) |
| [11_hooks](11_hooks) | [Hooks](../RUNA.md#11-hooks) |
| [12_test](12_test) | [Test](../RUNA.md#12-test) |
| [13_eval](13_eval) | [Eval (Case/Dataset)](../RUNA.md#13-eval-casedataset) |
| [14_tracing](14_tracing) | [Tracing](../RUNA.md#14-tracing) |
| [applications](applications) | Full apps, combining several primitives |

Each folder has its own `README.md`. For the same examples rendered alongside the prose that
explains each primitive, see the [docs](https://benybrahim.github.io/runa/) site instead.
