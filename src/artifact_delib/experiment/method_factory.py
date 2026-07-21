"""Method factory — instantiate baseline/pipeline methods by name.

Maps method names from config to their registered classes.
"""

from __future__ import annotations

import logging
from typing import Any

from artifact_delib.api.base import ModelClient
from artifact_delib.baselines.registry import get_baseline, list_baselines

logger = logging.getLogger(__name__)


def create_method(
    name: str,
    client: ModelClient,
    model_name: str = "default",
    **kwargs: Any,
) -> Any:
    """Instantiate a method by name.

    Args:
        name: Canonical method name from config (e.g. "direct_single_vlm").
        client: ModelClient instance for API calls.
        model_name: Model name to pass to the method.
        **kwargs: Additional keyword arguments passed to the constructor.

    Returns:
        An instance conforming to BaselineProtocol.

    Raises:
        KeyError: If the method name is not registered.
    """
    instance = get_baseline(
        name,
        client=client,
        model_name=model_name,
        **kwargs,
    )
    return instance


def list_available_methods() -> dict[str, dict[str, Any]]:
    """List all registered methods with metadata.

    Returns:
        Dict of {method_name: metadata_dict}.
    """
    return list_baselines()


def validate_method_name(name: str) -> bool:
    """Check if a method name is registered.

    Args:
        name: Method name to check.

    Returns:
        True if the method exists in the registry.
    """
    return name in list_baselines()