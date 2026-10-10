# Hooks

RUNA.md #11, [docs/tracing.md](../../docs/tracing.md#hooks).

One `Hooks` class, two scopes, chosen by where the instance goes:

* **`run_hooks.py`** -- passed per-call to `run`, firing for every agent in the run.
* **`agent_hooks.py`** -- assigned to an `Agent` subclass's `hooks`, firing for that agent only.
