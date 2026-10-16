"""_log.py: `logger`, the one `logging.Logger` Runa's own code writes to.

A module of its own because logging a warning should not cost an import of the `Hooks`
subsystem, nor risk a cycle: `runa.lifecycle` reads `runa.tracing`'s privacy policy, and
`runa.tracing` logs failed exports.

So this file imports nothing from `runa`, deliberately, the way `paused_delegates.py` does: a
leaf every layer can name at the top of the file, which is what keeps the runtime import graph
a DAG.

Handlers, levels and formatting are the application's, never Runa's: a library that calls
`basicConfig` has configured somebody else's program. An app that wants Runa's lines picks them
out by name -- `logging.getLogger("runa")`.
"""

import logging

logger = logging.getLogger("runa")

__all__ = ["logger"]
