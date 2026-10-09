"""mcp.py: `MCPServer(...).http(...)`/`.stdio(...)`, Runa's own bridge to the official MCP SDK.

A server's tools are listed once (cached from then on) and exposed as ordinary `FunctionTool`s.
`run_internal` never needs to know a tool came from an MCP server rather than a `@tool` function.
The connection itself is opened lazily, on first use, and kept open for as long as it can be
reused: an MCP server is meant to be a persistent connection, not something reopened every turn.

As long as it *can* be is one event loop. A `ClientSession`'s streams -- a subprocess's pipes
under `.stdio`, an HTTP connection under `.http` -- belong to the loop that was running when they
were opened, and do not survive it being closed, which is the lifetime `runa._loop.LoopCache`
exists to state and which `db/pool.py`'s pools, `cache/redis.py`'s client and `ModelProvider`'s
HTTP clients already hold themselves to. A session is the fourth such resource, so it is held the
same way: per loop, built on first use there. `await agent.run(...)` on an app's own loop gets
the persistent connection the paragraph above promises; `Agent.run_sync` opens a loop per call,
so each turn connects again, and a `.stdio` server spawns its subprocess again with it. That cost
is the reason `run_sync` closes these connections itself (see `Agent.run_sync`) instead of
leaving a process per turn behind, and the reason a long-lived app should prefer `run`.
"""

from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any

from mcp import ClientSession, StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp_types import TextContent

from runa._items import parsed_arguments
from runa._loop import LoopCache
from runa._types import RunContextWrapper
from runa.tool import FunctionTool


@dataclass
class _Connection:
    """One loop's live session with a server, the tools it listed, and how to shut it down.

    Session and tool list are one value because the tools are the session's: a `FunctionTool`
    here is only a schema plus a closure that calls back through `connect()`, so keeping the two
    together is what stops a list of tools from outliving the session it was read from.
    """

    stack: AsyncExitStack
    session: ClientSession
    tools: list[FunctionTool]


class _MCPServerBase:
    """Shared connect/list/call machinery for one MCP server; `.http`/`.stdio` pick a transport."""

    def __init__(self, *, name: str | None = None, address: str) -> None:
        """Store the server's display `name` and the `address` its connections are keyed by.

        `address` is where this server is reached -- a URL, or a command line -- and is the key
        its per-loop connection is held under, the way `cache/redis.py` keys its client by the
        Redis URL. Nothing connects here; the first `connect`/`list_tools` does.
        """
        self.name = name or "mcp"
        self._address = address
        self._connections: LoopCache[str, _Connection] = LoopCache()

    def _transport(self) -> Any:
        raise NotImplementedError

    async def connect(self) -> _Connection:
        """This server's connection on the *current* event loop, opening one if it has none.

        The one door to the session: a connection left over from a closed loop would hang or
        raise "Event loop is closed" rather than reconnect, so it is never reached from here.
        Two coroutines racing on one loop open one connection between them, not two.
        """
        return await self._connections.aget(self._address, self._open)

    async def _open(self) -> _Connection:
        """Open a session on the current loop and list its tools, as one unit.

        Listing here rather than on demand costs nothing -- resolving an agent's shape lists
        every server's tools at the start of each run anyway -- and means a `_Connection` is
        never half-built: a caller that has one has the tools that session exposes.
        """
        stack = AsyncExitStack()
        read, write = await stack.enter_async_context(self._transport())
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        listed = await session.list_tools()
        tools = [self._as_function_tool(tool) for tool in listed.tools]
        return _Connection(stack=stack, session=session, tools=tools)

    async def close(self) -> None:
        """Close this loop's connection to the server, if it has one, and forget its tools.

        An app that runs on one loop never needs this, same as `db/pool.py`'s `close_pool`: the
        connection lives as long as the loop does, which is as long as the process. A caller that
        *owns* a loop it is about to close does, since a `.stdio` server's subprocess would
        otherwise be left running until the dropped session is garbage collected --
        `Agent.run_sync` is that caller, once per call.
        """
        connection = self._connections.pop(self._address)
        if connection is not None:
            await connection.stack.aclose()

    async def list_tools(self) -> list[FunctionTool]:
        """Connect if needed, and return this server's tools as `FunctionTool`s (cached)."""
        return (await self.connect()).tools

    def _as_function_tool(self, tool: Any) -> FunctionTool:
        async def on_invoke_tool(ctx: RunContextWrapper, arguments_json: str, call_id: str) -> Any:
            session = (await self.connect()).session
            args = parsed_arguments(arguments_json)
            result = await session.call_tool(tool.name, args)
            text = "".join(block.text for block in result.content if isinstance(block, TextContent))
            if result.is_error:
                return f"error: {text or 'the tool call failed'}"
            return text or result.structured_content

        return FunctionTool(
            name=tool.name,
            description=tool.description or "",
            params_json_schema=tool.input_schema,
            on_invoke_tool=on_invoke_tool,
        )


class MCPServerStdio(_MCPServerBase):
    """An MCP server run as a local subprocess, spoken to over stdin/stdout."""

    def __init__(
        self,
        command: str,
        args: list[str] | None = None,
        *,
        name: str | None = None,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> None:
        """Store the command to spawn; nothing runs until the first `connect`/`list_tools`."""
        super().__init__(name=name, address=" ".join([command, *(args or [])]))
        self._params = StdioServerParameters(command=command, args=args or [], env=env, cwd=cwd)

    def _transport(self) -> Any:
        return stdio_client(self._params)


class MCPServerStreamableHttp(_MCPServerBase):
    """An MCP server reachable over streamable HTTP."""

    def __init__(self, url: str, *, name: str | None = None) -> None:
        """Store the server's `url`; nothing connects until the first `connect`/`list_tools`."""
        super().__init__(name=name, address=url)
        self._url = url

    def _transport(self) -> Any:
        return streamable_http_client(self._url)


class MCPServer:
    """The one way to build a server: shared config here, finalized by picking a transport.

    What's given here (currently just `name`) is common to every transport; `http`/`stdio` take
    the params specific to that transport, a URL vs. a command to spawn. Every parameter is
    spelled out on all three, rather than collected as `**kwargs` and forwarded, so that a
    misspelled or unsupported option is a `TypeError` naming it at the call site -- the way a
    mistyped `Agent` attribute is a `UserError` naming it at construction -- instead of a server
    that quietly ignores it and misbehaves on connect.

    The two transport classes are deliberately absent from `__all__`: `MCPServer` is the whole
    public surface, and a third transport is written by subclassing `_MCPServerBase`, not by
    reaching for `MCPServerStdio` directly.
    """

    def __init__(self, *, name: str | None = None) -> None:
        """Hold config shared by every transport, applied when `http`/`stdio` is called."""
        self._name = name

    def http(self, url: str) -> MCPServerStreamableHttp:
        """Connect over streamable HTTP."""
        return MCPServerStreamableHttp(url, name=self._name)

    def stdio(
        self,
        command: str,
        args: list[str] | None = None,
        *,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> MCPServerStdio:
        """Spawn a local process over stdio."""
        return MCPServerStdio(command, args, name=self._name, env=env, cwd=cwd)


__all__ = ["MCPServer"]
