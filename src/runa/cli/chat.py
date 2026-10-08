"""cli/chat.py: `runa chat`, talk to an Agent in a loop from argv.

Each invocation starts a fresh session by default, so a chat doesn't silently keep piling onto
the same conversation. `--continue`/`--resume` pick up a past one instead, keyed by session id
over whichever store `runa.db` resolves (the same history `runa chat --list`/`--show` reads, see
`cli/sessions.py`).

A line is one turn; a triple-quote line opens a multi-line block that the next one closes,
so a pasted document or stack trace is sent as one message instead of one turn per line.
"""

from datetime import datetime
from pathlib import Path
from uuid import uuid4

from runa import db
from runa.agent import Agent
from runa.project import find_agent_class, loaded_app, require_agents_dir
from runa.session import Session

_BLOCK = '"""'


def _new_session_id(agent_name: str) -> str:
    """A fresh id for a new chat: sortable by start time, unique enough for interactive use."""
    return f"{agent_name}-{datetime.now():%Y%m%d-%H%M%S}-{uuid4().hex[:4]}"


def _pick_session(agent_name: str, *, root: Path) -> str:
    """Prompt the operator to choose one of `agent_name`'s past sessions, newest first.

    Falls back to starting a new session when there's no history to pick from.
    """
    sessions = db.sessions(root).listing(agent=agent_name)
    if not sessions:
        print(f"no previous session for {agent_name!r}, starting a new one")
        return _new_session_id(agent_name)

    print(f"previous sessions for {agent_name}:")
    for index, session in enumerate(sessions, start=1):
        print(f"  {index}. {session.id}  ({session.updated_at})")
    choice = input(f"resume which? [1-{len(sessions)}, default 1] ").strip()
    try:
        index = int(choice) if choice else 1
    except ValueError:
        index = 1
    index = min(max(index, 1), len(sessions))
    return sessions[index - 1].id


def _resolve_session_id(
    agent_name: str,
    *,
    root: Path,
    session_id: str | None,
    continue_last: bool,
    resume: str | None,
) -> str:
    """Pick the session id for this chat.

    An explicit `--session` wins, then `--continue` (the most recent past session), then
    `--resume` (a given id, or a picker over past ones when none is given), and otherwise a
    fresh session every time.
    """
    if session_id is not None:
        return session_id
    if continue_last:
        sessions = db.sessions(root).listing(agent=agent_name)
        if sessions:
            return sessions[0].id
        print(f"no previous session for {agent_name!r}, starting a new one")
        return _new_session_id(agent_name)
    if resume is not None:
        return resume or _pick_session(agent_name, root=root)
    return _new_session_id(agent_name)


def _read_message() -> str:
    """Read one turn: a single line, or every line between a pair of triple-quote lines.

    Raises `EOFError` on Ctrl-D, including inside an unclosed block.
    """
    line = input("> ").strip()
    if line != _BLOCK:
        return line
    lines: list[str] = []
    while (line := input("... ")).strip() != _BLOCK:
        lines.append(line)
    return "\n".join(lines).strip()


def _ask(prompt: str) -> str:
    """`input()`, but `""` once stdin is exhausted, e.g. after a piped message."""
    try:
        return input(prompt).strip().lower()
    except EOFError:
        print()
        return ""


def _run_turn(agent: Agent, message: str, session: Session) -> None:
    """Send `message`, prompt for any approvals it pauses on, and print the final reply."""
    run = agent.run_sync(message, session=session)
    while run.status == "paused":
        state = run.to_state()
        for item in run.interruptions:
            answer = _ask(f"approve {item.name}({item.arguments})? [y/N/a] ")
            if answer in {"a", "always"}:
                state.approve(item, always=True)
            elif answer in {"y", "yes"}:
                state.approve(item)
            else:
                state.reject(item)
        run = agent.run_sync(state, session=session)

    print(run.output if run.status == "completed" else f"error: {run.error}")


def run_agent_repl(
    agent_name: str,
    *,
    root: Path,
    session_id: str | None = None,
    continue_last: bool = False,
    resume: str | None = None,
    message: str | None = None,
) -> None:
    """Chat with the named Agent in a loop, over one session.

    The app is loaded and the Agent instantiated once for the whole session, so turns share
    the in-process object instead of round-tripping through `db/runa.db` on every call. A pending
    approval is resolved right here by prompting the operator, since there's someone to ask.

    With `message` (what `runa chat` reads from piped stdin), send just that one turn and return
    instead of looping. Approvals then get rejected, since stdin is already used up.
    """
    agents_dir = require_agents_dir(root)

    with loaded_app(root):
        agent_cls = find_agent_class(agent_name, agents_dir=agents_dir)
        agent = agent_cls()
        resolved_session_id = _resolve_session_id(
            agent.name,
            root=root,
            session_id=session_id,
            continue_last=continue_last,
            resume=resume,
        )
        session = db.session(resolved_session_id, root=root)

        if message is not None:
            if message.strip():
                _run_turn(agent, message.strip(), session)
            return

        print(f"chatting with {agent.name} (session {resolved_session_id!r})")
        print(f"type 'exit' or Ctrl-D to quit, {_BLOCK} to start and end a multi-line message\n")

        while True:
            try:
                user_input = _read_message()
            except EOFError, KeyboardInterrupt:
                print()
                return
            if not user_input:
                continue
            if user_input in {"exit", "quit"}:
                return
            _run_turn(agent, user_input, session)
