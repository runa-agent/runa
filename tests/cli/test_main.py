"""Tests for `runa.cli.main`: argv parsing, dispatch, and clean error reporting."""

import io
from pathlib import Path

import pytest

from runa.cli.main import main
from runa.cli.new import scaffold_project


class _Stdin(io.StringIO):
    """A `sys.stdin` stand-in that reports whether it's a terminal."""

    def __init__(self, text: str = "", *, tty: bool) -> None:
        super().__init__(text)
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


@pytest.fixture(autouse=True)
def _interactive_stdin(monkeypatch: pytest.MonkeyPatch) -> None:
    """Behave as if run from a terminal; pytest's own stdin is neither a tty nor readable."""
    monkeypatch.setattr("sys.stdin", _Stdin(tty=True))


def test_new_scaffolds_a_project_and_prints_next_steps(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`runa new demo` scaffolds the project and prints next-step guidance."""
    exit_code = main(["new", "demo"], cwd=tmp_path)

    assert exit_code == 0
    assert (tmp_path / "demo" / "main.py").is_file()
    assert "created" in capsys.readouterr().out


def test_new_without_a_name_scaffolds_cwd_and_omits_the_cd_step(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`runa new` with no argument scaffolds `cwd` in place, no `cd` step in next steps."""
    exit_code = main(["new"], cwd=tmp_path)

    assert exit_code == 0
    assert (tmp_path / "main.py").is_file()
    out = capsys.readouterr().out
    assert f"created {tmp_path}" in out
    assert "cd " not in out


def test_new_reports_an_existing_directory_as_a_clean_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`runa new` on an existing directory prints one clean line and exits 1."""
    (tmp_path / "demo").mkdir()

    exit_code = main(["new", "demo"], cwd=tmp_path)

    assert exit_code == 1
    assert "already exists" in capsys.readouterr().err


def test_generate_agent_dispatches_to_generate_agent(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`runa generate agent SupportAgent --model ...` writes the agent file under the given cwd."""
    project_dir = scaffold_project("demo", root=tmp_path)

    exit_code = main(
        ["generate", "agent", "SupportAgent", "--model", "gpt-5.4-nano"], cwd=project_dir
    )

    assert exit_code == 0
    assert (project_dir / "app" / "agents" / "support_agent.py").is_file()


def test_generate_agent_without_model_is_a_clean_argparse_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--model` is required; omitting it exits non-zero via argparse, not a traceback."""
    project_dir = scaffold_project("demo", root=tmp_path)

    with pytest.raises(SystemExit) as excinfo:
        main(["generate", "agent", "SupportAgent"], cwd=project_dir)

    assert excinfo.value.code != 0


def test_generate_agent_rejects_a_class_name_not_ending_in_agent(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A class name that doesn't end in `Agent` is a clean error, not a traceback."""
    project_dir = scaffold_project("demo", root=tmp_path)

    exit_code = main(["generate", "agent", "Support", "--model", "gpt-5.4-nano"], cwd=project_dir)

    assert exit_code == 1
    assert "error:" in capsys.readouterr().err


def test_generate_agent_flags_dispatch_model_instructions_and_tools(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--model`/`--instructions`/`--tool a,b` land in the generated file via one dispatch."""
    project_dir = scaffold_project("demo", root=tmp_path)
    (project_dir / "app" / "tools" / "search_web.py").write_text("def search_web() -> str: ...\n")

    exit_code = main(
        [
            "generate",
            "agent",
            "SupportAgent",
            "--model",
            "gpt-5.4-nano",
            "--instructions",
            "Help users",
            "--tool",
            "search_web",
            "--compact",
        ],
        cwd=project_dir,
    )

    assert exit_code == 0
    content = (project_dir / "app" / "agents" / "support_agent.py").read_text()
    assert 'model = "gpt-5.4-nano"' in content
    assert "tools = [search_web]" in content
    assert "compact = True" in content


def test_generate_agent_next_steps_point_to_the_prompt_file_and_generate_tool(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Without `--instructions`, next-steps names the prompt stub and `runa generate tool`."""
    project_dir = scaffold_project("demo", root=tmp_path)

    exit_code = main(
        ["generate", "agent", "SupportAgent", "--model", "gpt-5.4-nano"], cwd=project_dir
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "app/prompts/support_agent.md" in out
    assert "runa generate tool <name>" in out
    assert "instructions" not in out


def test_generate_agent_next_steps_omit_the_prompt_file_with_explicit_instructions(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """With `--instructions`, next-steps doesn't mention the prompt stub (none was written)."""
    project_dir = scaffold_project("demo", root=tmp_path)

    exit_code = main(
        [
            "generate",
            "agent",
            "SupportAgent",
            "--model",
            "gpt-5.4-nano",
            "--instructions",
            "Help",
        ],
        cwd=project_dir,
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "app/prompts" not in out
    assert "runa generate tool <name>" in out


def test_generate_tool_dispatches_to_generate_tool(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`runa generate tool search_web --description ...` writes the docstring in."""
    project_dir = scaffold_project("demo", root=tmp_path)

    exit_code = main(
        [
            "generate",
            "tool",
            "search_web",
            "--description",
            "Search the web and return a summary.",
        ],
        cwd=project_dir,
    )

    assert exit_code == 0
    content = (project_dir / "app" / "tools" / "core.py").read_text()
    assert '"""Search the web and return a summary."""' in content


def test_generate_tool_without_description_falls_back_to_a_todo_stub(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`runa generate tool ...` with no `--description` keeps the old TODO stub."""
    project_dir = scaffold_project("demo", root=tmp_path)

    exit_code = main(["generate", "tool", "search_web"], cwd=project_dir)

    assert exit_code == 0
    content = (project_dir / "app" / "tools" / "core.py").read_text()
    assert '"""TODO: describe what this tool does."""' in content


def test_generate_tool_with_a_module_prefix_writes_and_prints_the_function_name(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`research:search_web` writes `app/tools/research.py` and prints the func name."""
    project_dir = scaffold_project("demo", root=tmp_path)

    exit_code = main(["generate", "tool", "research:search_web"], cwd=project_dir)

    assert exit_code == 0
    assert (project_dir / "app" / "tools" / "research.py").is_file()
    out = capsys.readouterr().out
    assert "from app.tools.research import search_web" in out
    assert "tools = [search_web]" in out


def test_generate_guardrail_dispatches_to_generate_guardrail(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`runa generate guardrail BlockEmpty` writes the guardrail file under the given cwd."""
    project_dir = scaffold_project("demo", root=tmp_path)

    exit_code = main(["generate", "guardrail", "BlockEmpty"], cwd=project_dir)

    assert exit_code == 0
    assert (project_dir / "app" / "guardrails" / "block_empty.py").is_file()


def test_chat_reports_agent_not_found_as_a_clean_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`runa chat` on an unknown Agent name prints one clean line and exits 1, not a traceback."""
    project_dir = scaffold_project("demo", root=tmp_path)

    exit_code = main(["chat", "Nope"], cwd=project_dir)

    assert exit_code == 1
    assert "no Agent named 'Nope'" in capsys.readouterr().err


def test_chat_dispatches_to_run_agent_repl(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`runa chat Support` calls `run_agent_repl`, asking for a fresh session by default."""
    project_dir = scaffold_project("demo", root=tmp_path)
    calls: list[tuple[str, Path, str | None, bool, str | None]] = []

    def fake_repl(
        name: str,
        *,
        root: Path,
        session_id: str | None = None,
        continue_last: bool = False,
        resume: str | None = None,
        message: str | None = None,
    ) -> None:
        calls.append((name, root, session_id, continue_last, resume))

    monkeypatch.setattr("runa.cli.main.run_agent_repl", fake_repl)

    exit_code = main(["chat", "Support"], cwd=project_dir)

    assert exit_code == 0
    assert calls == [("Support", project_dir, None, False, None)]


def test_chat_continue_flag_dispatches_continue_last(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`runa chat Support --continue` (and its `-c` alias) asks to resume the last session."""
    project_dir = scaffold_project("demo", root=tmp_path)
    calls: list[bool] = []

    def fake_repl(
        name: str,
        *,
        root: Path,
        session_id: str | None = None,
        continue_last: bool = False,
        resume: str | None = None,
        message: str | None = None,
    ) -> None:
        calls.append(continue_last)

    monkeypatch.setattr("runa.cli.main.run_agent_repl", fake_repl)

    assert main(["chat", "Support", "--continue"], cwd=project_dir) == 0
    assert main(["chat", "Support", "-c"], cwd=project_dir) == 0
    assert calls == [True, True]


def test_chat_resume_flag_dispatches_the_given_or_empty_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--resume ID` passes the id through; bare `--resume` passes `""` to trigger a picker."""
    project_dir = scaffold_project("demo", root=tmp_path)
    calls: list[str | None] = []

    def fake_repl(
        name: str,
        *,
        root: Path,
        session_id: str | None = None,
        continue_last: bool = False,
        resume: str | None = None,
        message: str | None = None,
    ) -> None:
        calls.append(resume)

    monkeypatch.setattr("runa.cli.main.run_agent_repl", fake_repl)

    assert main(["chat", "Support", "--resume", "Support-old"], cwd=project_dir) == 0
    assert main(["chat", "Support", "--resume"], cwd=project_dir) == 0
    assert calls == ["Support-old", ""]


def test_chat_sends_piped_stdin_as_one_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`cat doc.md | runa chat Support` passes all of stdin as a single `message`."""
    project_dir = scaffold_project("demo", root=tmp_path)
    monkeypatch.setattr("sys.stdin", _Stdin("line one\nline two\n", tty=False))
    calls: list[str | None] = []

    def fake_repl(name: str, *, root: Path, message: str | None = None, **kwargs: object) -> None:
        calls.append(message)

    monkeypatch.setattr("runa.cli.main.run_agent_repl", fake_repl)

    assert main(["chat", "Support"], cwd=project_dir) == 0
    assert calls == ["line one\nline two\n"]


def test_chat_with_no_name_or_flags_reports_a_clean_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`runa chat` with no Agent name and no `--list`/`--show` prints a clean error."""
    project_dir = scaffold_project("demo", root=tmp_path)

    exit_code = main(["chat"], cwd=project_dir)

    assert exit_code == 1
    assert "needs an agent name" in capsys.readouterr().err


def test_missing_main_py_reports_a_clean_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Running a command against a directory with no `main.py` prints one clean line."""
    (tmp_path / "tests").mkdir(parents=True)

    exit_code = main(["test"], cwd=tmp_path)

    assert exit_code == 1
    assert "no main.py found" in capsys.readouterr().err


def test_eval_reports_pass_fail_counts_and_exit_code(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`runa eval` against an empty `evals/` reports 0/0 passed and exits 0."""
    project_dir = scaffold_project("demo", root=tmp_path)

    exit_code = main(["eval"], cwd=project_dir)

    assert exit_code == 0
    assert "0/0 passed" in capsys.readouterr().out


def test_eval_with_agent_name_reports_a_clean_error_when_unmatched(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`runa eval AGENT_NAME` against an unmatched name exits 1 with a clean message."""
    project_dir = scaffold_project("demo", root=tmp_path)

    exit_code = main(["eval", "nonexistent"], cwd=project_dir)

    assert exit_code == 1
    assert "nonexistent" in capsys.readouterr().err


def test_test_reports_a_failing_test_with_exit_code_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`runa test` exits 1 and reports the failure when a `test_*` function fails."""
    project_dir = scaffold_project("demo", root=tmp_path)
    (project_dir / "tests" / "test_smoke.py").write_text(
        "def test_bad():\n    assert False, 'nope'\n"
    )

    exit_code = main(["test"], cwd=project_dir)

    out = capsys.readouterr().out
    assert exit_code == 1
    assert "FAIL: nope" in out
    assert "0/1 passed" in out


def test_chat_list_reports_no_sessions(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """`runa chat --list` against a fresh project reports no sessions."""
    project_dir = scaffold_project("demo", root=tmp_path)

    exit_code = main(["chat", "--list"], cwd=project_dir)

    assert exit_code == 0
    assert "no sessions found" in capsys.readouterr().out


def test_chat_show_reports_unknown_session_as_a_clean_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`runa chat --show` on an unknown session id prints a clean error."""
    project_dir = scaffold_project("demo", root=tmp_path)

    exit_code = main(["chat", "--show", "nope"], cwd=project_dir)

    assert exit_code == 1
    assert "no session found" in capsys.readouterr().err
