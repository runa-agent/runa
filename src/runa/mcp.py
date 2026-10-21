"""mcp.py: `MCPServer(...).http(...)`/`.stdio(...)`, Runa's own bridge to the official MCP SDK.

A server's tools are listed once (cached from then on) and exposed as ordinary `FunctionTool`s.
`run_internal` never needs to know a tool came from an MCP server rather than a `@tool` function.
The connection itself is opened lazily, on first use, and kept open for as long as it can be
reused: an MCP server is meant to be a persistent connection, not something reopened every turn.

Exposed as ordinary `FunctionTool`s includes the two policies a tool carries: `needs_approval`
and tool guardrails are the whole of what an agent may say about a tool it offers, so they are
asked of the server (`MCPServer(needs_approval=..., guardrails=[...])`) and stamped onto every
tool it builds. The server is the grain because trust is a property of the server -- someone
else's process, with its own filesystem, shell or API reach -- and a remote tool has no decorator
of its own to declare them on.

As long as it *can* be is one event loop. A `ClientSession`'s streams -- a subprocess's pipes
under `.stdio`, an HTTP connection under `.http` -- belong to the loop that was running when they
were opened, and do not survive it being closed, which is the lifetime `runa._loop.LoopCache`
exists to state and which `db/pool.py`'s pools, `cache/redis.py`'s client and `ModelProvider`'s
HTTP clients already hold themselves to. A session is the fourth such resource, so it is held the
same way: per loop, built on first use there. An app's own loop and the background loop every
`Agent.run_sync` shares both live as long as the process, so either gets the persistent
connection the paragraph above promises. The cache is built with a `close`, so the background
loop shuts every server it connected -- a subagent's included -- down before the interpreter
exits rather than leaving a `.stdio` subprocess behind.
"""

import re
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any

from mcp import ClientSession, StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp_types import TextContent

from runa._items import parsed_arguments
from runa._loop import LoopCache
from runa._types import RunContext
from runa.guardrail import BoundGuardrail, GuardrailsList, Phase, flatten_guardrails
from runa.tool import FunctionTool, _NeedsApproval


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

    async def close(self) -> None:
        """Shut the session and its transport down: a `.stdio` server's subprocess with it."""
        await self.stack.aclose()


class _MCPServerBase:
    """Shared connect/list/call machinery for one MCP server; `.http`/`.stdio` pick a transport."""

    def __init__(
        self,
        *,
        name: str | None = None,
        address: str,
        needs_approval: _NeedsApproval = False,
        guardrails: GuardrailsList | None = None,
    ) -> None:
        """Store the server's `name`, the `address` its connections are keyed by, and its policy.

        `name` prefixes every tool the server lists, so it is the part of a tool's name the app
        controls: two servers exposing the same tool are told apart by naming them differently.

        `address` is where this server is reached -- a URL, or a command line -- and is the key
        its per-loop connection is held under, the way `cache/redis.py` keys its client by the
        Redis URL. `needs_approval`/`guardrails` are what every tool this server lists carries,
        flattened once here rather than per tool: a `FunctionTool`'s guardrails are only read, so
        one mapping is shared by all of them. Nothing connects here; the first
        `connect`/`list_tools` does.
        """
        self.name = name or "mcp"
        self._tool_prefix = re.sub(r"[^a-zA-Z0-9_-]+", "_", self.name).strip("_") or "mcp"
        self._address = address
        self._needs_approval = needs_approval
        self._guardrails: dict[Phase, list[BoundGuardrail]] = flatten_guardrails(
            guardrails or [], tool=True
        )
        self._connections: LoopCache[str, _Connection] = LoopCache(close=_Connection.close)

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
        connection lives as long as the loop does, which is as long as the process. Nor does a
        synchronous caller: the connections `Agent.run_sync` opened are closed at exit.
        """
        connection = self._connections.pop(self._address)
        if connection is not None:
            await connection.close()

    async def list_tools(self) -> list[FunctionTool]:
        """Connect if needed, and return this server's tools as `FunctionTool`s (cached)."""
        return (await self.connect()).tools

    def _as_function_tool(self, tool: Any) -> FunctionTool:
        """One listed MCP tool as a `FunctionTool`: its schema, a call back through `connect`.

        Named `<server>__<tool>`: the tool's own name is the server's choice, not the app's, so
        two servers' `delete` stay two tools -- and two sticky approvals -- rather than one. The
        server's approval and guardrail policy comes along, so the gates `run_internal` already
        runs for every tool call have something to read here too.
        """

        async def on_invoke_tool(ctx: RunContext, arguments_json: str, call_id: str) -> Any:
            session = (await self.connect()).session
            args = parsed_arguments(arguments_json)
            result = await session.call_tool(tool.name, args)
            text = "".join(block.text for block in result.content if isinstance(block, TextContent))
            if result.is_error:
                return f"error: {text or 'the tool call failed'}"
            return text or result.structured_content

        return FunctionTool(
            name=f"{self._tool_prefix}__{tool.name}",
            description=tool.description or "",
            params_json_schema=tool.input_schema,
            on_invoke_tool=on_invoke_tool,
            guardrails=self._guardrails,
            needs_approval=self._needs_approval,
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
        needs_approval: _NeedsApproval = False,
        guardrails: GuardrailsList | None = None,
    ) -> None:
        """Store the command to spawn; nothing runs until the first `connect`/`list_tools`."""
        super().__init__(
            name=name,
            address=" ".join([command, *(args or [])]),
            needs_approval=needs_approval,
            guardrails=guardrails,
        )
        self._params = StdioServerParameters(command=command, args=args or [], env=env, cwd=cwd)

    def _transport(self) -> Any:
        return stdio_client(self._params)


class MCPServerStreamableHttp(_MCPServerBase):
    """An MCP server reachable over streamable HTTP."""

    def __init__(
        self,
        url: str,
        *,
        name: str | None = None,
        needs_approval: _NeedsApproval = False,
        guardrails: GuardrailsList | None = None,
    ) -> None:
        """Store the server's `url`; nothing connects until the first `connect`/`list_tools`."""
        super().__init__(
            name=name, address=url, needs_approval=needs_approval, guardrails=guardrails
        )
        self._url = url

    def _transport(self) -> Any:
        return streamable_http_client(self._url)


class MCPServer:
    """The one way to build a server: shared config here, finalized by picking a transport.

    What's given here -- the server's name, and the approval/guardrail policy its tools carry --
    is common to every transport; `http`/`stdio` take the params specific to that transport, a
    URL vs. a command to spawn. Every parameter is
    spelled out on all three, rather than collected as `**kwargs` and forwarded, so that a
    misspelled or unsupported option is a `TypeError` naming it at the call site -- the way a
    mistyped `Agent` attribute is a `UserError` naming it at construction -- instead of a server
    that quietly ignores it and misbehaves on connect.

    The two transport classes are deliberately absent from `__all__`: `MCPServer` is the whole
    public surface, and a third transport is written by subclassing `_MCPServerBase`, not by
    reaching for `MCPServerStdio` directly.
    """

    def __init__(
        self,
        *,
        name: str | None = None,
        needs_approval: _NeedsApproval = False,
        guardrails: GuardrailsList | None = None,
    ) -> None:
        """Hold config shared by every transport, applied when `http`/`stdio` is called.

        `needs_approval` (a `bool`, or an `@approval` predicate over the remote tool's arguments)
        and `guardrails` (`@guardrail` predicates, bound with `.input`/`.output` against a call's
        arguments and its result) apply to every tool this server exposes -- `@tool`'s two
        parameters, asked of the server because that's where a remote tool's trust is decided.
        """
        self._name = name
        self._needs_approval = needs_approval
        self._guardrails = guardrails

    def http(self, url: str) -> MCPServerStreamableHttp:
        """Connect over streamable HTTP."""
        return MCPServerStreamableHttp(
            url,
            name=self._name,
            needs_approval=self._needs_approval,
            guardrails=self._guardrails,
        )

    def stdio(
        self,
        command: str,
        args: list[str] | None = None,
        *,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> MCPServerStdio:
        """Spawn a local process over stdio."""
        return MCPServerStdio(
            command,
            args,
            name=self._name,
            env=env,
            cwd=cwd,
            needs_approval=self._needs_approval,
            guardrails=self._guardrails,
        )


__all__ = ["MCPServer"]
