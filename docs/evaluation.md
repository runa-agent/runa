# Evaluating Agents

Runa distinguishes two kinds of checks. **Tests** verify invariants with a plain `assert`.
**Evals** grade behavior, including with a judge model, against a dataset of cases.

An eval is one file, `evals/<agent_name>.jsonl`, with one case per line:

```bash
runa generate evaluation support_agent
```

```json
{"input": "Where's my order #4821?", "expected": "Asks for or looks up the order status"}
{"input": "Cancel order A100", "expected_tool": "cancel_order"}
```

```bash
runa eval
```

The filename is the Agent's declared `name`, the same one `runa chat` takes, so there is nothing
to register. Each line's keys are `Case` fields, and a line only carries the ones it needs.
`runa eval` calls `agent.evaluate(dataset)` for every file, the same path production evaluation
runs through.

Pass the Agent's `name` to run just its dataset, for example `runa eval support_agent`.

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

## Building Cases in Python

When cases need code, for example a configured agent or generated inputs, write a module instead.
It declares module-level `agent` and `dataset`, any iterable of `Case`:

```python
# evals/support_agent_eval.py
from runa import Case

from app.agents import SupportAgent

agent = SupportAgent(model="gpt-5.4-nano")

dataset = [Case(input=f"Where's my order #{n}?") for n in range(4800, 4810)]
```

A module can still keep its cases in a file, and a `.jsonl` sharing its stem belongs to it rather
than running on its own:

```python
dataset = Dataset.from_jsonl(Path(__file__).with_suffix(".jsonl"))
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
