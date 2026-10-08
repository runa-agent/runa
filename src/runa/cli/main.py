"""cli/main.py: the `runa` command-line entry point.

`new` and `generate` are scaffolding: they write files following the app/
convention and never touch the runtime. `chat`, `eval`, and `test` do
touch it, but only by calling existing library functions
(`agent.run_sync()`, `agent.evaluate()`, `run_project_tests()`) against
the app in `cwd`; no logic lives here that doesn't already exist elsewhere.
"""

import argparse
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from runa.cli.chat import run_agent_repl
from runa.cli.eval import run_project_evals
from runa.cli.generate import (
    generate_agent,
    generate_evaluation,
    generate_guardrail,
    generate_prompt,
    generate_tool,
)
from runa.cli.new import scaffold_project
from runa.cli.serve import serve_agents
from runa.cli.sessions import list_sessions, show_session
from runa.cli.test import run_project_tests
from runa.cli.traces import list_errors_cli, list_traces_cli, show_trace
from runa.cli.ui import serve_ui
from runa.eval.corpus import add_trace_to_evals
from runa.exceptions import OperatorError
from runa.project import AppLoadError

_EXTRA_FOR_MODULE = {
    "fastapi": "serve",
    "uvicorn": "serve",
    "asyncpg": "postgres",
    "pgvector": "postgres",
    "redis": "redis",
}
"""Which optional extra provides a module, so a missing one is an instruction, not a traceback.

`runa serve` and `runa ui` both import their web stack lazily, which keeps a plain install
working for every other command but means the failure surfaces only when the command is run.
That failure is an operator error, and this module's contract is that those get one clean line.

`serve` and `ui` ship the same two packages, so a missing `fastapi`/`uvicorn` could be either;
the command being run decides which one to name, since telling a `runa ui` user to install
`serve` would be advice that happens to work while explaining nothing.
"""


def _split_names(value: str | None) -> list[str]:
    """Split a `--tool`/`--guardrail` value ("a, b,c") into trimmed, non-empty names."""
    return [part.strip() for part in value.split(",") if part.strip()] if value else []


def _relative(path: Path, root: Path) -> str:
    """Name a generated file the way a developer would type it: relative to the app root."""
    return str(path.relative_to(root))


def _runa_version() -> str:
    try:
        return version("runa-ai")
    except PackageNotFoundError:
        return "unknown"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="runa", description="Scaffold, run, and evaluate Runa agents."
    )
    parser.add_argument("--version", action="version", version=f"runa {_runa_version()}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    new_parser = subparsers.add_parser("new", help="Scaffold a new Runa application")
    new_parser.add_argument(
        "name", nargs="?", default=None, help="omit to scaffold the current directory in place"
    )

    generate_parser = subparsers.add_parser(
        "generate", help="Generate scaffolding inside a Runa application"
    )
    generate_subparsers = generate_parser.add_subparsers(dest="kind", required=True)
    agent_parser = generate_subparsers.add_parser("agent", help="Generate a new Agent")
    agent_parser.add_argument("name", help="UpperCamelCase, ending in 'Agent', e.g. SupportAgent")
    agent_parser.add_argument(
        "--model", required=True, help="e.g. gpt-5.4-nano, claude-... (required)"
    )
    agent_parser.add_argument("--instructions", default=None, help="inline instructions string")
    agent_parser.add_argument(
        "--tool",
        default=None,
        help="comma-separated tool names, e.g. search_web,research:fetch_page",
    )
    agent_parser.add_argument("--guardrail", default=None, help="comma-separated guardrail names")
    agent_parser.add_argument("--memory", choices=["auto", "llm"], default=None)
    agent_parser.add_argument("--knowledge", choices=["auto", "llm"], default=None)
    agent_parser.add_argument("--compact", action="store_true")
    tool_parser = generate_subparsers.add_parser("tool", help="Generate a new @tool function")
    tool_parser.add_argument(
        "name",
        help="e.g. search_web (app/tools/core.py) or research:search_web (app/tools/research.py)",
    )
    tool_parser.add_argument(
        "--description", default=None, help="one-line docstring describing what it does"
    )
    generate_subparsers.add_parser(
        "guardrail", help="Generate a new @guardrail function"
    ).add_argument("name")
    generate_subparsers.add_parser("prompt", help="Generate a new app/prompts/ file").add_argument(
        "name"
    )
    evaluation_parser = generate_subparsers.add_parser(
        "evaluation", help="Generate a new evals/<agent>.jsonl dataset"
    )
    evaluation_parser.add_argument("name", help="the agent's snake_case name, e.g. support_agent")

    chat_parser = subparsers.add_parser("chat", help="Chat with an Agent, or inspect past sessions")
    chat_parser.add_argument(
        "agent_name",
        nargs="?",
        default=None,
        help="the Agent's declared `name`, e.g. support_agent",
    )
    session_group = chat_parser.add_mutually_exclusive_group()
    session_group.add_argument(
        "--session",
        default=None,
        help="pin an exact conversation id to persist to/resume in runa.db",
    )
    session_group.add_argument(
        "--continue",
        "-c",
        dest="continue_",
        action="store_true",
        help="resume this Agent's most recent session instead of starting a new one",
    )
    session_group.add_argument(
        "--resume",
        nargs="?",
        const="",
        default=None,
        metavar="SESSION_ID",
        help="resume SESSION_ID, or without one, pick from a list of past sessions",
    )
    chat_parser.add_argument(
        "--list", action="store_true", help="List every session in runa.db instead of chatting"
    )
    chat_parser.add_argument(
        "--show",
        metavar="SESSION_ID",
        default=None,
        help="Show one session's history instead of chatting",
    )

    eval_parser = subparsers.add_parser("eval", help="Run this app's evals/ cases")
    eval_parser.add_argument(
        "agent_name",
        nargs="?",
        default=None,
        help="only run this Agent's evals/ dataset (its declared `name`, e.g. support_agent)",
    )
    eval_parser.add_argument(
        "--add",
        metavar="TRACE_ID",
        default=None,
        help="add a traced run's input as a case to its agent's evals/ dataset, instead of running",
    )
    eval_parser.add_argument(
        "--expected",
        default=None,
        help="with --add: what a good answer would have said",
    )
    subparsers.add_parser("test", help="Run this app's tests/ test functions")

    traces_parser = subparsers.add_parser("traces", help="Inspect this app's traces in runa.db")
    traces_subparsers = traces_parser.add_subparsers(dest="traces_action", required=True)

    traces_subparsers.add_parser("list", help="List the most recent traces")
    traces_subparsers.add_parser("errors", help="List the most recent traces that errored")

    traces_show_parser = traces_subparsers.add_parser("show", help="Show a trace's span tree")
    traces_show_parser.add_argument("trace_id")

    serve_parser = subparsers.add_parser(
        "serve", help="Serve this app's agents over HTTP (needs the `serve` extra)"
    )
    serve_parser.add_argument(
        "--host", default="127.0.0.1", help="use 0.0.0.0 in a container (default: 127.0.0.1)"
    )
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument(
        "--no-auth",
        action="store_true",
        help="serve without a bearer token; otherwise RUNA_API_KEY is required",
    )
    serve_parser.add_argument(
        "--workers", type=int, default=1, help="uvicorn worker processes (default: 1)"
    )

    ui_parser = subparsers.add_parser(
        "ui", help="Serve a local dashboard over this app's db/runa.db (needs the `ui` extra)"
    )
    ui_parser.add_argument("--host", default="127.0.0.1")
    ui_parser.add_argument("--port", type=int, default=8765)

    return parser


def main(argv: list[str] | None = None, *, cwd: Path | None = None) -> int:
    """Parse argv and dispatch.

    Operator-input errors (a mistyped session id or tool_call_id, running outside a Runa app
    directory) become a clean message on stderr and exit code 1 instead of a raw Python
    traceback. Anything else (a real bug, in Runa or in the app's own code) still propagates
    with its full traceback, since swallowing that would hide the thing a developer actually
    needs to see.

    Which failures are the operator's is not decided here: each one subclasses `OperatorError`
    where it is raised, in the module that already knows whose fault it is, so a new command's
    new error needs nothing added to this function. The other two clauses are the failures that
    earn a more specific message than `error: {exc}`, not a second copy of that decision.
    """
    cwd = cwd or Path.cwd()
    args = _build_parser().parse_args(argv)

    try:
        return _dispatch(args, cwd)
    except ModuleNotFoundError as exc:
        if exc.name == "main":
            print(f"error: no main.py found in {cwd}, is this a Runa app?", file=sys.stderr)
            return 1
        extra = _EXTRA_FOR_MODULE.get(exc.name or "")
        if extra is None:
            raise
        if extra == "serve" and args.command == "ui":
            extra = "ui"
        print(
            f"error: `runa {args.command}` needs the `{extra}` extra, which a plain install "
            f'does not include.\n  uv add "runa-ai[{extra}]"',
            file=sys.stderr,
        )
        return 1
    except OperatorError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except AppLoadError as exc:
        print(
            f"error: failed to load {cwd / 'main.py'}: {exc}\n"
            "(run `python main.py` directly to see the full traceback)",
            file=sys.stderr,
        )
        return 1


def _dispatch(args: argparse.Namespace, cwd: Path) -> int:
    if args.command == "new":
        project_dir = scaffold_project(args.name, root=cwd)
        print(f"created {project_dir}")
        cd_step = f"  cd {project_dir.name}\n" if args.name else ""
        print(
            f"\nnext steps:\n{cd_step}"
            "  put your OPENAI_API_KEY in .env   # or whichever model your agents use\n"
            "  runa generate agent MyAgent --model gpt-5.4-nano\n"
            "  runa chat my_agent"
        )
        return 0

    if args.command == "generate" and args.kind == "agent":
        agent = generate_agent(
            args.name,
            root=cwd,
            model=args.model,
            instructions=args.instructions,
            tools=_split_names(args.tool),
            guardrails=_split_names(args.guardrail),
            memory=args.memory,
            knowledge=args.knowledge,
            compact=args.compact,
        )
        print(f"created {agent.file}")
        prompt_step = f"write {_relative(agent.prompt, cwd)}, " if agent.prompt else ""
        print(
            f"\nnext: {prompt_step}"
            f"add eval cases to {_relative(agent.dataset, cwd)}, "
            "add tools with\n  runa generate tool <name>\n"
            "then chat with it:\n"
            f"  runa chat {agent.name}\n"
            "or call it from your own code:\n"
            f"  {agent.import_line}\n"
            f"  {agent.symbol}().run_sync('...')"
        )
        return 0

    if args.command == "generate" and args.kind == "tool":
        generated_tool = generate_tool(args.name, root=cwd, description=args.description)
        print(f"created {generated_tool.file}")
        print(
            "\nnext: implement it, then declare it on an Agent, e.g.\n"
            f"  {generated_tool.import_line}\n\n"
            "  class MyAgent(Agent):\n"
            '      name = "my_agent"\n'
            '      model = "gpt-5.4-nano"\n'
            f"      tools = [{generated_tool.symbol}]"
        )
        return 0

    if args.command == "generate" and args.kind == "guardrail":
        generated_guardrail = generate_guardrail(args.name, root=cwd)
        print(f"created {generated_guardrail.file}")
        print(
            "\nnext: implement it, then bind it to an Agent or @tool, e.g.\n"
            f"  {generated_guardrail.import_line}\n"
            f"  guardrails = [{generated_guardrail.symbol}.input]"
        )
        return 0

    if args.command == "generate" and args.kind == "prompt":
        print(f"created {generate_prompt(args.name, root=cwd).file}")
        print("\nnext: write the prompt, then load it from an Agent's `instructions`")
        return 0

    if args.command == "generate" and args.kind == "evaluation":
        print(f"created {generate_evaluation(args.name, root=cwd).file}")
        print(
            "\nnext: add one case per line, e.g. "
            '{"input": "...", "expected": "..."}, then\n'
            "  runa eval"
        )
        return 0

    if args.command == "chat":
        if args.list:
            print(list_sessions(root=cwd))
            return 0
        if args.show is not None:
            print(show_session(args.show, root=cwd))
            return 0
        if args.agent_name is None:
            print("error: runa chat needs an agent name, or --list/--show", file=sys.stderr)
            return 1
        run_agent_repl(
            args.agent_name,
            root=cwd,
            session_id=args.session,
            continue_last=args.continue_,
            resume=args.resume,
            message=None if sys.stdin.isatty() else sys.stdin.read(),
        )
        return 0

    if args.command == "eval" and args.add is not None:
        eval_file = add_trace_to_evals(args.add, root=cwd, expected=args.expected)
        print(f"added trace {args.add} to {eval_file}")
        if args.expected is None:
            print('\nnext: give it an "expected" answer in that file, then\n  runa eval')
        return 0

    if args.command == "eval":
        reports = run_project_evals(cwd, args.agent_name)
        for report in reports:
            print(report)
            print()
        total = sum(len(report.cases) for report in reports)
        failed = sum(len(report.failed) for report in reports)
        regressed = sum(len(report.regressions) for report in reports)
        print(
            f"{total - failed}/{total} passed" + (f", {regressed} regressed" if regressed else "")
        )
        return 1 if failed else 0

    if args.command == "test":
        results = run_project_tests(cwd)
        for result in results:
            status = "PASS" if result.passed else f"FAIL: {result.error}"
            print(f"{result.name}: {status}")
        failed = sum(1 for result in results if not result.passed)
        print(f"\n{len(results) - failed}/{len(results)} passed")
        return 1 if failed else 0

    if args.command == "serve":
        serve_agents(
            cwd,
            host=args.host,
            port=args.port,
            no_auth=args.no_auth,
            workers=args.workers,
        )
        return 0

    if args.command == "ui":
        serve_ui(cwd, host=args.host, port=args.port)
        return 0

    if args.traces_action == "list":
        print(list_traces_cli(root=cwd))
    elif args.traces_action == "show":
        print(show_trace(args.trace_id, root=cwd))
    else:
        print(list_errors_cli(root=cwd))
    return 0
