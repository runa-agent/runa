"""Tests for `runa.mcp`: the bridge from an MCP server's tools to `FunctionTool`."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any, cast

import pytest
from mcp_types import CallToolResult, ListToolsResult, TextContent, Tool

from runa import mcp
from runa._types import RunContextWrapper
from runa.mcp import (
    MCPServer,
    MCPServerStdio,
    MCPServerStreamableHttp,
    _Connection,
    _MCPServerBase,
)


class _FakeSession:
    """A stand-in for `mcp.ClientSession`: scripted `list_tools`/`call_tool` results."""

    def __init__(self, tools: list[Tool], call_results: dict[str, CallToolResult]) -> None:
        self._tools = tools
        self._call_results = call_results
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def list_tools(self) -> ListToolsResult:
        return ListToolsResult(tools=self._tools)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> CallToolResult:
        self.calls.append((name, arguments))
        return self._call_results[name]


class _FakeServer(_MCPServerBase):
    """An `_MCPServerBase` opening onto a `_FakeSession` instead of a real transport.

    `_open` is what a transport would supply, so overriding it leaves everything this file is
    testing -- the per-loop caching in `connect`, the tool mapping, `close` -- as written.
    """

    def __init__(self, session: _FakeSession) -> None:
        super().__init__(name="fake", address="fake://server")
        self._fake_session = session
        self.opens = 0
        self.closed = 0

    async def _open(self) -> _Connection:
        self.opens += 1
        listed = await self._fake_session.list_tools()
        stack = AsyncExitStack()
        await stack.enter_async_context(_count_close(self))
        return _Connection(
            stack=stack,
            session=cast(Any, self._fake_session),
            tools=[self._as_function_tool(tool) for tool in listed.tools],
        )


@asynccontextmanager
async def _count_close(server: _FakeServer) -> AsyncIterator[None]:
    """Stand in for a transport, recording that its connection was actually shut down."""
    try:
        yield
    finally:
        server.closed += 1


def test_list_tools_maps_mcp_tools_to_function_tools() -> None:
    """Each MCP `Tool`'s name/description/schema becomes a `FunctionTool`'s."""
    tool = Tool(
        name="add",
        description="Add two numbers.",
        input_schema={"type": "object", "properties": {"a": {"type": "integer"}}},
    )
    server = _FakeServer(_FakeSession([tool], {}))

    tools = asyncio.run(server.list_tools())

    assert len(tools) == 1
    assert tools[0].name == "add"
    assert tools[0].description == "Add two numbers."
    assert tools[0].params_json_schema == tool.input_schema


def test_list_tools_caches_after_the_first_call() -> None:
    """A second `list_tools()` call reuses the cached result instead of asking the server again."""
    session = _FakeSession([Tool(name="add", input_schema={"type": "object"})], {})
    server = _FakeServer(session)

    async def twice() -> tuple[list[Any], list[Any]]:
        return await server.list_tools(), await server.list_tools()

    first, second = asyncio.run(twice())

    assert first is second
    assert server.opens == 1


def test_a_second_event_loop_gets_its_own_connection() -> None:
    """A session cannot outlive the loop it was opened on, so a new loop opens a new one.

    `Agent.run_sync` is an `asyncio.run` per call, so this is the shape of every `runa chat`
    turn after the first: reusing turn 1's session here is a hang or "Event loop is closed".
    """
    session = _FakeSession([Tool(name="add", input_schema={"type": "object"})], {})
    server = _FakeServer(session)

    first = asyncio.run(server.list_tools())
    second = asyncio.run(server.list_tools())

    assert server.opens == 2
    assert first is not second


def test_close_shuts_the_connection_down_and_reconnects_on_demand() -> None:
    """`close()` closes the transport and forgets the tools; the next call opens a fresh one."""
    session = _FakeSession([Tool(name="add", input_schema={"type": "object"})], {})
    server = _FakeServer(session)

    async def use_close_and_use_again() -> None:
        await server.list_tools()
        await server.close()
        assert server.closed == 1
        await server.list_tools()

    asyncio.run(use_close_and_use_again())

    assert server.opens == 2


def test_close_is_a_no_op_when_this_loop_never_connected() -> None:
    """Closing a server that was never used on this loop does nothing, rather than failing."""
    server = _FakeServer(_FakeSession([], {}))

    asyncio.run(server.close())

    assert (server.opens, server.closed) == (0, 0)


def test_on_invoke_tool_extracts_text_from_a_successful_call() -> None:
    """A successful `call_tool` result's text content becomes the tool's return value."""
    tool = Tool(name="add", input_schema={"type": "object"})
    result = CallToolResult(content=[TextContent(type="text", text="3")])
    session = _FakeSession([tool], {"add": result})
    server = _FakeServer(session)

    (function_tool,) = asyncio.run(server.list_tools())

    async def call() -> Any:
        return await function_tool.on_invoke_tool(
            RunContextWrapper(context=None), '{"a": 1, "b": 2}', "call_1"
        )

    output = asyncio.run(call())

    assert output == "3"
    assert session.calls == [("add", {"a": 1, "b": 2})]


def test_on_invoke_tool_surfaces_an_error_result() -> None:
    """A `CallToolResult` with `is_error=True` surfaces as an error string, not a crash."""
    tool = Tool(name="fail", input_schema={"type": "object"})
    result = CallToolResult(content=[TextContent(type="text", text="boom")], is_error=True)
    session = _FakeSession([tool], {"fail": result})
    server = _FakeServer(session)

    (function_tool,) = asyncio.run(server.list_tools())

    async def call() -> Any:
        return await function_tool.on_invoke_tool(RunContextWrapper(context=None), "{}", "call_1")

    output = asyncio.run(call())

    assert output == "error: boom"


def test_on_invoke_tool_falls_back_to_structured_content() -> None:
    """A tool call with no text content returns `structured_content` instead."""
    tool = Tool(name="lookup", input_schema={"type": "object"})
    result = CallToolResult(content=[], structured_content={"id": 1})
    session = _FakeSession([tool], {"lookup": result})
    server = _FakeServer(session)

    (function_tool,) = asyncio.run(server.list_tools())

    async def call() -> Any:
        return await function_tool.on_invoke_tool(RunContextWrapper(context=None), "{}", "call_1")

    output = asyncio.run(call())

    assert output == {"id": 1}


def test_mcp_server_stdio_builds_from_the_shared_config() -> None:
    """`MCPServer(name=...).stdio(...)` forwards shared kwargs into `MCPServerStdio`."""
    server = MCPServer(name="fs").stdio("npx", ["-y", "server-filesystem"])

    assert isinstance(server, MCPServerStdio)
    assert server.name == "fs"


def test_mcp_server_http_builds_from_the_shared_config() -> None:
    """`MCPServer(name=...).http(...)` carries the shared name onto the HTTP transport."""
    server = MCPServer(name="search").http("https://example.com/mcp")

    assert isinstance(server, MCPServerStreamableHttp)
    assert server.name == "search"


def test_stdio_passes_its_own_params_to_the_subprocess() -> None:
    """`env`/`cwd` are `stdio`'s own params, reaching the spawn parameters."""
    server = MCPServer().stdio("npx", ["serve"], env={"TOKEN": "x"}, cwd="/tmp")

    assert server._params.env == {"TOKEN": "x"}
    assert server._params.cwd == "/tmp"


def test_an_unknown_option_is_rejected_at_the_call_site() -> None:
    """A misspelled option fails where it was written, instead of being silently dropped."""
    with pytest.raises(TypeError, match="timeuot"):
        MCPServer(name="files", timeuot=30)  # pyright: ignore[reportCallIssue]

    with pytest.raises(TypeError, match="timeuot"):
        MCPServer(name="files").stdio("npx", timeuot=30)  # pyright: ignore[reportCallIssue]

    with pytest.raises(TypeError, match="timeuot"):
        MCPServer(name="search").http("https://example.com/mcp", timeuot=30)  # pyright: ignore[reportCallIssue]


def test_mcp_server_is_the_only_exported_name() -> None:
    """The builder is the whole public surface; the transports aren't a second call site."""
    assert mcp.__all__ == ["MCPServer"]
