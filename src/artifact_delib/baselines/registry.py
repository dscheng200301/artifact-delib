"""Baseline registry — map method names to classes.

External baselines (paper):
  clip_zero_shot, dinov2_knn, blip2_zero_shot,
  direct_single_vlm, self_consistency_vlm, multi_agent_debate

Ours:
  artifact_delib_rule, artifact_delib_mlp

Internal / legacy (not in paper table):
  fixed_multi_expert, fixed_full, generic_mad
"""

from __future__ import annotations

import warnings
from typing import Any

# ── Registry storage ──
_BASELINE_REGISTRY: dict[str, type] = {}
_BASELINE_META: dict[str, dict[str, Any]] = {}

# ── Category tags ──
CAT_EXTERNAL = "external"
CAT_OURS = "ours"
CAT_LEGACY = "legacy"
CAT_ABLATION = "ablation"

RESERVED_NAMES = frozenset([
    "clip_zero_shot",
    "dinov2_knn",
    "blip2_zero_shot",
    "direct_single_vlm",
    "self_consistency_vlm",
    "multi_agent_debate",
    "artifact_delib_rule",
    "artifact_delib_mlp",
    "fixed_multi_expert",
    "fixed_full",
    "generic_mad",
])


def register_baseline(
    name: str,
    cls: type,
    *,
    category: str = CAT_EXTERNAL,
    description: str = "",
    requires_fit: bool = False,
    requires_download: bool = False,
) -> None:
    """Register a baseline class under a canonical name.

    Args:
        name: Canonical name (kebab-case).
        cls: The class to register.
        category: One of CAT_EXTERNAL, CAT_OURS, CAT_LEGACY, CAT_ABLATION.
        description: Short human-readable description.
        requires_fit: True if the baseline needs a build_index() call.
        requires_download: True if it downloads model weights automatically.
    """
    _BASELINE_REGISTRY[name] = cls
    _BASELINE_META[name] = {
        "name": name,
        "class": cls,
        "category": category,
        "description": description,
        "requires_fit": requires_fit,
        "requires_download": requires_download,
    }


def get_baseline(name: str, **kwargs: Any) -> Any:
    """Instantiate a baseline by name.

    Args:
        name: Canonical baseline name.
        **kwargs: Passed to the class constructor.

    Returns:
        An instance of the registered baseline class.

    Raises:
        KeyError: If the name is not registered.
    """
    if name not in _BASELINE_REGISTRY:
        candidates = [k for k in _BASELINE_REGISTRY if name in k]
        msg = (
            f"Unknown baseline {name!r}. "
            f"Registered: {list(_BASELINE_REGISTRY)}"
        )
        if candidates:
            msg += f" Did you mean {candidates[0]!r}?"
        raise KeyError(msg)

    cls = _BASELINE_REGISTRY[name]
    # Check optional dependencies before instantiation
    _check_optional_deps(name)
    return cls(**kwargs)


def list_baselines(
    category: str | None = None,
) -> dict[str, dict[str, Any]]:
    """List registered baselines, optionally filtered by category.

    Args:
        category: If given, only return baselines in this category.

    Returns:
        Dict of {name: meta}.
    """
    if category:
        return {
            k: v for k, v in _BASELINE_META.items()
            if v["category"] == category
        }
    return dict(_BASELINE_META)


def _check_optional_deps(name: str) -> None:
    """Emit a clear warning if optional dependencies are missing."""
    torch_deps = {"clip_zero_shot", "dinov2_knn", "blip2_zero_shot"}
    if name in torch_deps:
        try:
            import torch  # noqa: F401
        except ImportError:
            warnings.warn(
                f"Baseline {name!r} requires PyTorch. "
                "Install it with: pip install torch torchvision",
                ImportWarning,
                stacklevel=2,
            )


# ── Module-level convenience ──
BASELINE_REGISTRY = _BASELINE_REGISTRY

# ── Auto-registration (lazy, runs once) ──
_AUTO_REGISTERED = False


def _auto_register() -> None:
    """Populate the registry with all known baselines.

    Called lazily when list_baselines() or get_baseline() is first used.
    Torch-dependent baselines are registered if their classes can be imported;
    if torch is not installed, they are silently skipped.
    """
    global _AUTO_REGISTERED
    if _AUTO_REGISTERED:
        return
    _AUTO_REGISTERED = True

    # External
    from artifact_delib.baselines.direct_vlm import DirectVLMBaseline
    register_baseline(
        "direct_single_vlm", DirectVLMBaseline,
        category=CAT_EXTERNAL,
        description="Single VLM call, no expert decomposition",
    )

    from artifact_delib.baselines.self_consistency import SelfConsistencyBaseline
    register_baseline(
        "self_consistency_vlm", SelfConsistencyBaseline,
        category=CAT_EXTERNAL,
        description="N independent VLM samples + majority voting aggregation",
    )

    from artifact_delib.baselines.multi_agent_debate import MultiAgentDebateBaseline
    register_baseline(
        "multi_agent_debate", MultiAgentDebateBaseline,
        category=CAT_EXTERNAL,
        description="N-agent free-form debate with multiple rounds",
    )

    # Torch-dependent baselines — skip registration if import fails
    try:
        from artifact_delib.baselines.clip_zero_shot import ClipZeroShotBaseline
        register_baseline(
            "clip_zero_shot", ClipZeroShotBaseline,
            category=CAT_EXTERNAL,
            description="CLIP zero-shot image classification",
            requires_download=True,
        )
    except ImportError:
        pass

    try:
        from artifact_delib.baselines.dinov2_knn import Dinov2KNNBaseline
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
        from artifact_delib.baselines.blip2_zero_shot import Blip2ZeroShotBaseline
        register_baseline(
            "blip2_zero_shot", Blip2ZeroShotBaseline,
            category=CAT_EXTERNAL,
            description="BLIP-2 zero-shot image-to-text identification",
            requires_download=True,
        )
    except ImportError:
        pass

    # Ours
    try:
        from artifact_delib.pipeline.artifact_delib_pipeline import ArtifactDelibPipeline
        register_baseline(
            "artifact_delib_rule", ArtifactDelibPipeline,
            category=CAT_OURS,
            description="Full pipeline with RuleRouter",
        )
    except ImportError:
        pass

    try:
        from artifact_delib.router.learned_router import MLPRouter
        register_baseline(
            "artifact_delib_mlp", MLPRouter,
            category=CAT_OURS,
            description="Full pipeline with MLPRouter (requires training)",
            requires_fit=True,
        )
    except ImportError:
        pass

    # Legacy
    from artifact_delib.baselines.legacy import (
        FixedMultiExpertBaseline, FixedFullBaseline, GenericMADBaseline,
    )
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


# Override list_baselines and get_baseline to auto-register first
_original_list = list_baselines
_original_get = get_baseline


def list_baselines(category: str | None = None) -> dict[str, dict[str, Any]]:
    _auto_register()
    return _original_list(category)


def get_baseline(name: str, **kwargs: Any) -> Any:
    _auto_register()
    return _original_get(name, **kwargs)


__all__ = [
    "BASELINE_REGISTRY",
    "CAT_EXTERNAL",
    "CAT_OURS",
    "CAT_LEGACY",
    "CAT_ABLATION",
    "get_baseline",
    "list_baselines",
    "register_baseline",
]
