"""Compatibility import surface for all EvidenceFlow domain contracts.

New code may import focused modules; infrastructure adapters can use this stable
aggregator without depending on the domain package's internal file layout.
"""

from app.domain import *  # noqa: F403
from app.domain import __all__ as __all__
