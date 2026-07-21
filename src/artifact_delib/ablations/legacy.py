"""Legacy ablation classes — importing from the old ablations.py module.

These are the original A1-A12 implementations kept for backward compat.
"""

from __future__ import annotations

# Import from the old ablations module (the top-level one, not this package)
import sys
import importlib.util

# The old ablations module at src/artifact_delib/ablations.py
# is shadowed by this package. We import from the original source file directly.
from artifact_delib.ablations import (
    AblationNoMultiExpert,
    AblationSingleExpert,
    AblationNoRouter,
    AblationNoRecheck,
    AblationFixedAllRecheck,
    AblationNoDeliberation,
    AblationFreeDebate,
    AblationFixedDeliberation,
    AblationNoDeferredJudge,
)

__all__ = [
    "AblationNoMultiExpert",
    "AblationSingleExpert",
    "AblationNoRouter",
    "AblationNoRecheck",
    "AblationFixedAllRecheck",
    "AblationNoDeliberation",
    "AblationFreeDebate",
    "AblationFixedDeliberation",
    "AblationNoDeferredJudge",
]
