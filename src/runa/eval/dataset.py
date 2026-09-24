"""eval/dataset.py: `Dataset`, an ordered collection of `Case`.

A plain `list[Case]` already satisfies everything `evaluate_agent()` needs
(it just iterates), so `Dataset` only exists for the one thing a bare list
can't do: load itself from a JSONL file.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from runa.eval.case import Case


class Dataset:
    """An ordered, iterable collection of `Case`."""

    def __init__(self, cases: Iterable[Case]) -> None:
        """Wrap `cases` in a list, preserving order."""
        self._cases = list(cases)

    def __iter__(self) -> Iterator[Case]:
        """Iterate over cases in order."""
        return iter(self._cases)

    def __len__(self) -> int:
        """Return the number of cases."""
        return len(self._cases)

    def __getitem__(self, index: int) -> Case:
        """Return the case at `index`."""
        return self._cases[index]

    @classmethod
    def from_jsonl(cls, path: str | Path) -> Dataset:
        """Load one `Case` per non-blank line of a JSONL file.

        Each line's JSON object is passed as keyword arguments to `Case`, so
        a line only needs the fields it uses, e.g. `{"input": "...",
        "expected": "..."}`. A bare JSON string is shorthand for an input-only
        case: `"..."` is `{"input": "..."}`.
        """
        lines = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        return cls(Case(line) if isinstance(line, str) else Case(**line) for line in lines)
