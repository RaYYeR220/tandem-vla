"""The dashboard's backend: a single-process async server over the frozen module contracts
in ``docs/INTERFACES.md``.

Nothing in here reaches into ``sim``, ``control``, ``gate``, ``eval``, ``planner``, ``voice``,
``bench``, ``perception`` or ``policy`` except through the public functions/classes those
packages already export. This package only wires them together and speaks SSE/JSON at the
edge.
"""

from .app import create_app

__all__ = ["create_app"]
