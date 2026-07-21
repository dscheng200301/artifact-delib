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
  - fixed_multi_expert   — always call all 5 experts, no routing
  - fixed_full           — always run all experts + rechecks + deliberation
  - generic_mad          — generic N-agent free debate (internal diagnostic)
"""

from artifact_delib.baselines.registry import (
    BASELINE_REGISTRY,
    CAT_EXTERNAL,
    CAT_LEGACY,
    CAT_OURS,
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

# ── Auto-register all known baselines (once per import) ──
_AUTO_REGISTERED = False

if not _AUTO_REGISTERED:
    _AUTO_REGISTERED = True

    # ── External ──────────────────────────────────────────────

    register_baseline(
        "direct_single_vlm", DirectVLMBaseline,
        category=CAT_EXTERNAL,
        description="Single VLM call, no expert decomposition",
    )

    register_baseline(
        "self_consistency_vlm", SelfConsistencyBaseline,
        category=CAT_EXTERNAL,
        description="N independent VLM samples + majority voting aggregation",
    )

    register_baseline(
        "multi_agent_debate", MultiAgentDebateBaseline,
        category=CAT_EXTERNAL,
        description="N-agent free-form debate with multiple rounds",
    )

    # Torch-dependent baselines — skip registration if dependencies missing
    try:
        from artifact_delib.baselines.clip_zero_shot import ClipZeroShotBaseline  # noqa: F811
        register_baseline(
            "clip_zero_shot", ClipZeroShotBaseline,
            category=CAT_EXTERNAL,
            description="CLIP zero-shot image classification",
            requires_download=True,
        )
    except ImportError:
        pass

    try:
        from artifact_delib.baselines.dinov2_knn import Dinov2KNNBaseline  # noqa: F811
        register_baseline(
            "dinov2_knn", Dinov2KNNBaseline,
            category=CAT_EXTERNAL,
            description="DINOv2 k-NN with distance-weighted voting",
            requires_fit=True,
            requires_download=True,
        )
    except ImportError:
        pass

    try:
        from artifact_delib.baselines.blip2_zero_shot import Blip2ZeroShotBaseline  # noqa: F811
        register_baseline(
            "blip2_zero_shot", Blip2ZeroShotBaseline,
            category=CAT_EXTERNAL,
            description="BLIP-2 zero-shot image-to-text identification",
            requires_download=True,
        )
    except ImportError:
        pass

    # ── Ours ──────────────────────────────────────────────────

    try:
        from artifact_delib.pipeline.artifact_delib_pipeline import ArtifactDelibPipeline  # noqa: F811
        register_baseline(
            "artifact_delib_rule", ArtifactDelibPipeline,
            category=CAT_OURS,
            description="Full pipeline with RuleRouter",
        )
    except ImportError:
        pass

    try:
        from artifact_delib.router.learned_router import MLPRouter  # noqa: F811
        register_baseline(
            "artifact_delib_mlp", MLPRouter,
            category=CAT_OURS,
            description="Full pipeline with MLPRouter (requires training)",
            requires_fit=True,
        )
    except ImportError:
        pass

    # ── Legacy ────────────────────────────────────────────────

    register_baseline(
        "fixed_multi_expert", FixedMultiExpertBaseline,
        category=CAT_LEGACY,
        description="Always call all 5 experts, no routing",
    )

    register_baseline(
        "fixed_full", FixedFullBaseline,
        category=CAT_LEGACY,
        description="Always run all experts + rechecks + deliberation",
    )

    register_baseline(
        "generic_mad", GenericMADBaseline,
        category=CAT_LEGACY,
        description="Generic N-agent free debate (internal diagnostic)",
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
