"""`MCPServer`: exposing an external MCP server's tools to an agent, just like `@tool` functions.

See RUNA.md #9 and docs/mcp.md.

Needs Node (`npx`) available on `PATH` -- this spins up the reference filesystem MCP server over
stdio, scoped to the current directory. `.http(url)` connects to a remote server the same way,
just swap the transport. The connection opens lazily and stays open for the agent's whole
lifetime, not reopened every turn.

Run it:

    uv run python examples/09_mcp/stdio_server.py
"""

from runa import Agent, MCPServer

files = MCPServer(name="files").stdio("npx", ["-y", "@modelcontextprotocol/server-filesystem", "."])


class FileAgent(Agent):
    """Answers questions about files in the current directory, via the MCP filesystem server."""

    name = "file_agent"
    instructions = "You answer questions about files in the current directory."
    mcp = [files]


run = FileAgent().run_sync("What files are in this directory?")
print(run.output)
