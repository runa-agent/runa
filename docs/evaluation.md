# Evaluating Agents

Runa distinguishes two kinds of checks. **Tests** verify invariants with a plain `assert`.
**Evals** grade behavior, including with a judge model, against a dataset of cases.

`evals/` holds datasets of `Case`s, graded against an agent:

```bash
runa generate evaluation support_agent
```

```python
# evals/support_agent_eval.py
from runa import Case

from app.agents import SupportAgent

agent = SupportAgent()

dataset = [
    Case(
        input="Where's my order #4821?",
        expected="Asks for or looks up the order status",
    ),
]
```

```bash
runa eval
```

A module must declare module-level `agent` and `dataset`. `runa eval` imports every module under
`evals/` and calls `agent.evaluate(dataset)` on each, the same path production evaluation runs
through.

Pass the Agent's declared `name` to run just that one's module, for example
`runa eval support_agent`.

## What a `Case` Can Carry

Only `input` is required. Everything else is optional evidence that decides which metrics run:

| Field | Enables |
|---|---|
| `expected` | Answer-correctness grading |
| `expected_tool` | A deterministic "was it called" tool-correctness check |
| `context` | Faithfulness grading against retrieval passages |
| `metadata` | Arbitrary data carried through to the report |

With none of them, task completion and answer relevance still run. Every case runs to completion
even if an earlier one errors, and the finished `Report` is persisted to `runa.db`.

## Loading Cases From a File

Write a handful of cases inline. Once a dataset grows, or is exported from real runs, keep it in a
JSONL file next to the module, one `Case` per line:

```json
{"input": "Where's my order #4821?", "expected": "Asks for or looks up the order status"}
{"input": "Cancel order A100", "expected_tool": "cancel_order"}
```

```python
# evals/support_agent_eval.py
from pathlib import Path

from runa import Dataset

from app.agents import SupportAgent

agent = SupportAgent()

dataset = Dataset.from_jsonl(Path(__file__).with_suffix(".jsonl"))
```

Each line's keys are `Case` fields, so a line only carries the ones it needs. Any other format
works too, since `dataset` is just an iterable of `Case`:

```python
import csv

with open(Path(__file__).with_suffix(".csv")) as f:
    dataset = [Case(**row) for row in csv.DictReader(f)]
```

## Choosing a Judge

Semantic metrics (task completion, answer correctness, answer relevance, faithfulness, tool
correctness) are graded by a judge model, by default the agent's own `model`:

```python
report = await agent.evaluate(dataset, judge="gpt-5.4")
```

Override a pass threshold globally or per metric:

```python
await agent.evaluate(dataset, threshold=0.8)
await agent.evaluate(dataset, thresholds={"faithfulness": 0.9})
```

## Example

```python
--8<--"examples/13_eval/case_dataset.py"
```

The same dataset, loaded from JSONL:

```python
--8<--"examples/13_eval/jsonl_dataset.py"
```

More in [`examples/13_eval/`](https://github.com/Benybrahim/runa/tree/main/examples/13_eval).

Tests are a separate, deterministic check, a bare `test_*` function with a plain `assert`:

```python
--8<--"examples/12_test/test_example.py"
```

More in [`examples/12_test/`](https://github.com/Benybrahim/runa/tree/main/examples/12_test).
