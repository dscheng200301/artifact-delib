"""ArtifactDelib baselines package.

External Baselines (paper table):
  - clip_zero_shot       — CLIP zero-shot image classification
  - dinov2_knn           — DINOv2 k-NN with distance-weighted voting
  - blip2_zero_shot      — BLIP-2 zero-shot image-to-text identification
  - direct_single_vlm    — Single VLM call, no expert decomposition
  - self_consistency_vlm — N independent VLM samples + aggregation
  - multi_agent_debate   — N-agent free-form debate with rounds

Ours:
  - artifact_delib_rule  — Full pipeline with RuleRouter
  - artifact_delib_mlp   — Full pipeline with MLPRouter (if trained)

Internal / Legacy (not in external baseline table):
  - FixedMultiExpert     — always call all 5 experts, no routing
  - FixedFull            — always run all experts + rechecks + deliberation
  - GenericMAD           — generic N-agent free debate (internal diagnostic)
"""

from artifact_delib.baselines.registry import (
    BASELINE_REGISTRY,
    get_baseline,
    list_baselines,
    register_baseline,
)

# ── Re-export new-style baseline classes ──
from artifact_delib.baselines.direct_vlm import DirectVLMBaseline
from artifact_delib.baselines.self_consistency import SelfConsistencyBaseline
from artifact_delib.baselines.multi_agent_debate import MultiAgentDebateBaseline
from artifact_delib.baselines.legacy import (
    FixedMultiExpertBaseline,
    FixedFullBaseline,
    GenericMADBaseline,
)

__all__ = [
    "BASELINE_REGISTRY",
    "get_baseline",
    "list_baselines",
    "register_baseline",
    "DirectVLMBaseline",
    "SelfConsistencyBaseline",
    "MultiAgentDebateBaseline",
    "FixedMultiExpertBaseline",
    "FixedFullBaseline",
    "GenericMADBaseline",
]
