"""A Runa test: a bare `test_*` function with a plain `assert`, no pytest.

See RUNA.md #12 and docs/evaluation.md.

`runa test` is its own small runner, specifically so a generated app needs no test framework as
a dependency -- this file lives in `examples/`, not `tests/`, purely to keep it runnable and
readable standalone; a real app's own tests live under `tests/` and run with `runa test`.
`async def test_*` is awaited automatically, no `asyncio.run` wrapper needed.
"""

from runa import Agent


class GreeterAgent(Agent):
    """Greets the user warmly."""

    name = "greeter_agent"
    instructions = "You greet the user warmly, in one short sentence."


def test_answers_politely() -> None:
    """A completed run should produce some output."""
    run = GreeterAgent().run_sync("Hi")
    assert run.status == "completed"
    assert run.output


async def test_answers_politely_async() -> None:
    """The same check, using the async entry point instead of `run_sync`."""
    run = await GreeterAgent().run("Hi")
    assert run.status == "completed"
    assert run.output
