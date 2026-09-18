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

`mcp=` is sugar for `mcp_servers=`. Pass either.

MCP tools go through the same `needs_approval` gate as any other tool. See
[Human Approval](approval.md).

## Example

```python
--8<--"examples/09_mcp/stdio_server.py"
```
