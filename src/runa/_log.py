"""_log.py: `logger`, the one `logging.Logger` Runa's own code writes to.

Imports nothing from `runa`, so every layer can import it without a cycle. Handlers, levels and
formatting are the application's: an app that wants Runa's lines picks them out with
`logging.getLogger("runa")`.
"""

import logging

logger = logging.getLogger("runa")

__all__ = ["logger"]
