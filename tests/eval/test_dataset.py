"""Tests for `runa.eval.dataset`: `Dataset`."""

from pathlib import Path

from runa.eval.case import Case
from runa.eval.dataset import Dataset


def test_dataset_wraps_a_plain_list_of_cases() -> None:
    """A `Dataset` iterates its cases in order and reports their count."""
    cases = [Case(input="a"), Case(input="b")]

    dataset = Dataset(cases)

    assert len(dataset) == 2
    assert [case.input for case in dataset] == ["a", "b"]
    assert dataset[0].input == "a"


def test_dataset_from_jsonl_loads_one_case_per_line(tmp_path: Path) -> None:
    """`from_jsonl` builds one `Case` per non-blank line, keyed by field name."""
    jsonl_path = tmp_path / "cases.jsonl"
    jsonl_path.write_text(
        '{"input": "What is the refund policy?", "expected": "30 days"}\n'
        "\n"
        '{"input": "Cancel order 123", "expected_tool": "cancel_order"}\n'
    )

    dataset = Dataset.from_jsonl(jsonl_path)

    assert len(dataset) == 2
    assert dataset[0] == Case(input="What is the refund policy?", expected="30 days")
    assert dataset[1] == Case(input="Cancel order 123", expected_tool="cancel_order")


def test_dataset_from_jsonl_reads_a_bare_string_as_an_input_only_case(tmp_path: Path) -> None:
    """A line that's just a JSON string is shorthand for `{"input": ...}`."""
    jsonl_path = tmp_path / "cases.jsonl"
    jsonl_path.write_text('"Where is my order?"\n{"input": "Hi", "expected": "A greeting"}\n')

    dataset = Dataset.from_jsonl(jsonl_path)

    assert list(dataset) == [
        Case(input="Where is my order?"),
        Case(input="Hi", expected="A greeting"),
    ]
