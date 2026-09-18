"""A custom `TraceExporter`, added alongside the default `SQLiteExporter`.

See RUNA.md #14 and docs/tracing.md ("Custom Exporters").

`add_exporter` adds one more exporter without disabling the default; `observe(exporter=...)`
would replace the list entirely instead. Writing your own is one method: `export(trace) -> None`.

Run it:

    uv run python examples/14_tracing/custom_exporter.py
"""

from runa import Agent, add_exporter


class SpanCountExporter:
    """Prints how many spans each trace recorded, without touching where else it's exported.

    `TraceExporter` is a `Protocol`: matching its single `export` method is enough, no
    inheritance required.
    """

    def export(self, trace: object) -> None:
        """Print the trace's name and span count."""
        print(f"  [exporter] trace {trace.name!r}: {len(trace.spans)} spans")  # type: ignore[attr-defined]


add_exporter(SpanCountExporter())  # runa.db still gets written to; this is now added alongside it


class GreeterAgent(Agent):
    """Greets the user warmly."""

    name = "greeter_agent"
    instructions = "You greet the user warmly, in one short sentence."


run = GreeterAgent().run_sync("Hi")
print(run.output)
