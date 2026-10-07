"""mcp.py: `MCPServer(...).http(...)`/`.stdio(...)`, Runa's own bridge to the official MCP SDK.

A server's tools are listed once (cached from then on) and exposed as ordinary `FunctionTool`s.
`run_internal` never needs to know a tool came from an MCP server rather than a `@tool` function.
The connection itself is opened lazily, on first use, and kept open across an `Agent`'s whole
lifetime (not per `Agent.run()` call): an MCP server is meant to be a persistent, reusable
connection, not something reopened every turn.
"""

from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession, StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp_types import TextContent

from runa._items import parsed_arguments
from runa._types import RunContextWrapper
from runa.tool import FunctionTool


class _MCPServerBase:
    """Shared connect/list/call machinery for one MCP server; `.http`/`.stdio` pick a transport."""

    def __init__(self, *, name: str | None = None) -> None:
        """Store the server's display `name`; the connection itself opens lazily."""
        self.name = name or "mcp"
        self._session: ClientSession | None = None
        self._stack: AsyncExitStack | None = None
        self._tools: list[FunctionTool] | None = None

    def _transport(self) -> Any:
        raise NotImplementedError

    async def connect(self) -> None:
        """Open the connection and initialize the MCP session, if not already open."""
        if self._session is not None:
            return
        stack = AsyncExitStack()
        read, write = await stack.enter_async_context(self._transport())
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        self._stack = stack
        self._session = session

    async def close(self) -> None:
        """Close the connection, if open, and forget any cached tool list."""
        if self._stack is not None:
            await self._stack.aclose()
        self._session = None
        self._stack = None
        self._tools = None

    async def list_tools(self) -> list[FunctionTool]:
        """Connect if needed, and return this server's tools as `FunctionTool`s (cached)."""
        if self._tools is not None:
            return self._tools
        await self.connect()
        assert self._session is not None
        result = await self._session.list_tools()
        self._tools = [self._as_function_tool(tool) for tool in result.tools]
        return self._tools

    def _as_function_tool(self, tool: Any) -> FunctionTool:
        async def on_invoke_tool(ctx: RunContextWrapper, arguments_json: str, call_id: str) -> Any:
            await self.connect()
            assert self._session is not None
            args = parsed_arguments(arguments_json)
            result = await self._session.call_tool(tool.name, args)
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
        super().__init__(name=name)
        self._params = StdioServerParameters(command=command, args=args or [], env=env, cwd=cwd)

    def _transport(self) -> Any:
        return stdio_client(self._params)


class MCPServerStreamableHttp(_MCPServerBase):
    """An MCP server reachable over streamable HTTP."""

    def __init__(self, url: str, *, name: str | None = None) -> None:
        """Store the server's `url`; nothing connects until the first `connect`/`list_tools`."""
        super().__init__(name=name)
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
