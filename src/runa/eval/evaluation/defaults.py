"""eval/evaluation/defaults.py: default pass/fail thresholds, per metric.

Implementation defaults, not promises of universal correctness. Overridable
via `agent.evaluate(dataset, threshold=...)` (applies to every metric) or
`thresholds={...}` (applies to just the named ones).
"""

DEFAULT_THRESHOLDS: dict[str, float] = {
    "task_completion": 0.90,
    "answer_correctness": 0.85,
    "answer_relevance": 0.85,
    "faithfulness": 0.85,
    "tool_correctness": 0.90,
}
