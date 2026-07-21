"""Backward-compat re-exports for the old baselines module.

Previously:
  from artifact_delib.baselines import DirectVLMBaseline

Still works — it redirects to the new package.
"""
from __future__ import annotations

import warnings as _warnings

from artifact_delib.baselines.legacy import (
    FixedFullBaseline as FixedFullBaseline,
    FixedMultiExpertBaseline as FixedMultiExpertBaseline,
    GenericMADBaseline as GenericMADBaseline,
)
from artifact_delib.baselines.direct_vlm import DirectVLMBaseline as DirectVLMBaseline
from artifact_delib.baselines.self_consistency import SelfConsistencyBaseline as SelfConsistencyBaseline
from artifact_delib.baselines.multi_agent_debate import MultiAgentDebateBaseline as MultiAgentDebateBaseline
from artifact_delib.baselines.registry import get_baseline, list_baselines, register_baseline

__all__ = [
    "DirectVLMBaseline",
    "FixedMultiExpertBaseline",
    "FixedFullBaseline",
    "GenericMADBaseline",
    "SelfConsistencyBaseline",
    "MultiAgentDebateBaseline",
    "get_baseline",
    "list_baselines",
    "register_baseline",
]
