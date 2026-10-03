"""eval/case.py: `Case`, one eval input plus what "good" looks like for it.

Deliberately minimal: `input` is required, everything else is an optional
signal that `evaluate_agent()` (see `eval/evaluate.py`) uses to decide which
metrics apply. Adding `expected`, `expected_tool`, or `context` never
requires touching an existing `Case`.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class Case:
    """One eval case: an input, plus optional evidence of what a good run looks like.

    `expected` enables answer-correctness grading; `expected_tool` enables
    the deterministic "was it called" (tool-correctness) check; `context`
    (retrieval passages the agent should be grounded in) enables
    faithfulness grading. Any combination is valid, including none of them,
    in which case only task completion and answer relevance run.
    """

    input: str
    expected: str | None = None
    expected_tool: str | None = None
    context: list[str] | None = None
    metadata: dict[str, Any] | None = None
