# Memory

RUNA.md #7, [docs/memory.md](../../docs/memory.md).

* **`auto_memory.py`** -- `memory = "auto"`: retrieved and stored automatically, scoped by `user_id`.
* **`llm_memory.py`** -- `memory = "llm"`: the model decides when to call `search_memory`.
