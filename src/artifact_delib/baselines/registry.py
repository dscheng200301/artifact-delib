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
