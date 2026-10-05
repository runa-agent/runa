"""Tests for `runa.tracing.Trace`: `duration`, `status`, `errors`, `walk`, and `__str__`."""

from runa.tracing import Span, Trace


def _span(id: str, parent_id: str | None, type: str = "custom", **kwargs: object) -> Span:
    return Span(
        id=id,
        trace_id="t1",
        parent_id=parent_id,
        name=id,
        type=type,  # type: ignore[arg-type]
        start_time=0.0,
        end_time=1.0,
        **kwargs,  # type: ignore[arg-type]
    )


def test_trace_duration_is_the_gap_between_start_and_end() -> None:
    """A finished trace's `duration` is `end_time - start_time`."""
    trace = Trace(id="t1", name="Agent", start_time=1.0, end_time=3.5)

    assert trace.duration == 2.5


def test_trace_duration_is_none_while_open() -> None:
    """A trace with no `end_time` yet has no `duration`."""
    trace = Trace(id="t1", name="Agent", start_time=1.0)

    assert trace.duration is None


def test_trace_status_is_error_if_any_span_errored() -> None:
    """`Trace.status` is `"error"` when at least one span errored, `"ok"` otherwise."""
    ok_trace = Trace(id="t1", name="Agent", start_time=0.0, end_time=1.0, spans=[_span("s1", None)])
    assert ok_trace.status == "ok"

    failing_span = _span("s2", None, status="error", error="boom")
    error_trace = Trace(id="t1", name="Agent", start_time=0.0, end_time=1.0, spans=[failing_span])
    assert error_trace.status == "error"


def test_trace_errors_lists_only_the_failing_spans() -> None:
    """`Trace.errors` returns just the spans whose `status` is `"error"`."""
    ok_span = _span("s1", None)
    failing_span = _span("s2", None, status="error", error="boom")
    trace = Trace(
        id="t1", name="Agent", start_time=0.0, end_time=1.0, spans=[ok_span, failing_span]
    )

    assert trace.errors == [failing_span]


def test_trace_str_renders_an_empty_trace_as_just_the_header() -> None:
    """A trace with no spans still renders a header line."""
    trace = Trace(id="t1", name="Agent", start_time=0.0, end_time=1.0)

    assert str(trace) == "Trace Agent [1.00s] ✓"


def test_trace_str_renders_nested_spans_as_a_tree() -> None:
    """`__str__` nests spans under their `parent_id`, labeling each by its `type`."""
    agent = _span("agent1", None, type="agent")
    llm = _span("llm1", "agent1", type="llm")
    tool = _span("tool1", "agent1", type="tool")
    trace = Trace(
        id="t1", name="SupportAgent", start_time=0.0, end_time=1.0, spans=[agent, llm, tool]
    )

    rendered = str(trace)

    assert rendered.startswith("Trace SupportAgent [1.00s] ✓")
    assert "Agent agent1" in rendered
    assert "LLM llm1" in rendered
    assert "Tool tool1" in rendered
    # both llm1 and tool1 are indented under agent1, not at the root
    lines = rendered.splitlines()
    agent_line_index = next(i for i, line in enumerate(lines) if "Agent agent1" in line)
    for name in ("LLM llm1", "Tool tool1"):
        index, line = next((i, line) for i, line in enumerate(lines) if name in line)
        assert index > agent_line_index
        assert line.strip() != line  # indented, not flush left


def test_trace_str_shows_error_glyph_and_message_for_a_failing_span() -> None:
    """A failing span renders with the `✗` glyph and its error message."""
    failing = _span("boom1", None, status="error", error="tool exploded")
    trace = Trace(id="t1", name="Agent", start_time=0.0, end_time=1.0, spans=[failing])

    rendered = str(trace)

    assert "✗" in rendered
    assert "tool exploded" in rendered


def test_walk_nests_spans_under_their_parent_in_start_order() -> None:
    """`walk` recovers the tree from the flat `spans` list, ordering siblings by start time."""
    root = _span("agent1", None, type="agent")
    second = _span("llm2", "agent1", type="llm")
    second.start_time = 2.0
    first = _span("llm1", "agent1", type="llm")
    first.start_time = 1.0
    trace = Trace(id="t1", name="A", start_time=0.0, end_time=1.0, spans=[root, second, first])

    (row,) = trace.walk()

    assert row.span is root
    assert [kid.span.id for kid in row.children] == ["llm1", "llm2"]


def test_walk_labels_each_row_the_way_a_person_reads_it() -> None:
    """A row carries the type's label, the formatted duration, and no tokens off an llm span."""
    trace = Trace(
        id="t1", name="A", start_time=0.0, end_time=1.0, spans=[_span("call", None, type="tool")]
    )

    (row,) = trace.walk()

    assert (row.label, row.name, row.duration, row.tokens) == ("Tool", "call", "1.00s", None)


def test_walk_strips_the_handoff_tool_name_prefix() -> None:
    """A handoff row is named for the target agent, not the `transfer_to_` tool the model calls.

    The prefix is the one the model sees; `span.name` still carries it, so a renderer that wants
    the raw tool name can still reach it.
    """
    span = _span("transfer_to_billing_agent", None, type="handoff")
    trace = Trace(id="t1", name="A", start_time=0.0, end_time=1.0, spans=[span])

    (row,) = trace.walk()

    assert row.name == "billing_agent"
    assert row.span.name == "transfer_to_billing_agent"


def test_only_a_handoff_row_hands_off() -> None:
    """`hands_off` marks the boundary after which siblings belong to the agent handed to."""
    handoff = _span("transfer_to_billing_agent", None, type="handoff")
    tool = _span("lookup", None, type="tool")
    trace = Trace(id="t1", name="A", start_time=0.0, end_time=1.0, spans=[handoff, tool])

    assert [row.hands_off for row in trace.walk()] == [True, False]


def test_trace_str_names_a_handoff_for_its_target_and_marks_the_takeover() -> None:
    """The text tree knows the same two handoff facts the HTML one does.

    Both render from `walk`, so `runa traces show` cannot drift back into printing the raw
    `transfer_to_` tool name or leaving the takeover boundary invisible.
    """
    handoff = _span("transfer_to_billing_agent", "agent1", type="handoff")
    trace = Trace(
        id="t1",
        name="A",
        start_time=0.0,
        end_time=1.0,
        spans=[_span("agent1", None, type="agent"), handoff],
    )

    rendered = str(trace)

    assert "transfer_to_billing_agent" not in rendered
    assert "Handoff billing_agent" in rendered
    assert "→ billing_agent takes over" in rendered
