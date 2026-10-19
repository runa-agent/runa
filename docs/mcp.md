# MCP Servers

An [MCP](https://modelcontextprotocol.io) server exposes tools over a standard protocol, instead
of a Python function you write yourself. Runa speaks it over two transports:

```python
from runa import Agent, MCPServer

files = MCPServer(name="files").stdio("npx", ["-y", "@modelcontextprotocol/server-filesystem", "."])
search = MCPServer(name="search").http("https://example.com/mcp")


class Assistant(Agent):
    name = "assistant"
    instructions = "..."
    mcp = [files, search]
```

The server's tools are listed once, and cached, the first time the agent needs them, then
exposed to the model exactly like an ordinary `@tool` function. The model can't tell the
difference. The connection itself opens lazily and stays open for the agent's whole lifetime,
not reopened every turn.

Each tool is offered under its server's name, as `<server>__<tool>`: the filesystem server's
`read_file` above is `files__read_file`. A tool's own name is the server's choice, not yours, so
this is what keeps two servers that both expose `delete` from shadowing each other, or sharing
one `always=True` approval. Give each server its own `name=`: two tools an agent offers under one
name are a `UserError`.

## Approval and Guardrails

Exactly like an ordinary tool includes the two things a tool carries besides its schema: whether
a call needs a human, and the guardrails its arguments and result are checked against. A remote
tool has no `@tool` decorator to declare them on, so the server declares them for every tool it
exposes:

```python
from runa import MCPServer, approval, guardrail


@approval
def writes_outside_tmp(path: str) -> bool:
    """Writing anywhere but /tmp needs a human to sign off."""
    return not path.startswith("/tmp/")


@guardrail
def no_path_traversal(args: dict) -> bool:
    """Trip when a call tries to escape the directory the server was given."""
    return ".." in str(args.get("path", ""))


files = MCPServer(
    name="files",
    needs_approval=writes_outside_tmp,
    guardrails=[no_path_traversal.input],
).stdio("npx", ["-y", "@modelcontextprotocol/server-filesystem", "."])
```

`needs_approval` takes a `bool` or an `@approval` predicate, and `guardrails` takes the same list
`@tool(guardrails=[...])` does, bound with `.input`/`.output` against a call's arguments and its
result. Both are given to `MCPServer(...)` rather than to `.stdio`/`.http`, because trust is a
property of the server -- someone else's process, with its own filesystem, shell or API reach --
not of the transport it's reached over. A predicate's parameters are looked up by name from the
remote tool's arguments, so it is written against that server's schema.

See [Human Approval](approval.md) and [Guardrails](guardrails.md).

## Example

```python
--8<--"examples/09_mcp/stdio_server.py"
```

More in [`examples/09_mcp/`](https://github.com/Benybrahim/runa/tree/main/examples/09_mcp).
