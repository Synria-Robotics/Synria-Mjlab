"""Synria mjlab task package — registers all tasks via entry point."""

from synria_mjlab.locomotion import getup as _getup  # noqa: F401
from synria_mjlab.locomotion import velocity as _velocity  # noqa: F401
from synria_mjlab.manipulation import handover as _handover  # noqa: F401
from synria_mjlab.manipulation import lift as _lift  # noqa: F401
from synria_mjlab.manipulation import peg_insertion as _peg  # noqa: F401
from synria_mjlab.manipulation import reach as _reach  # noqa: F401
