"""Core ablation experiments for ArtifactDelib.

Each ablation removes or replaces exactly one component of the full pipeline.

A1: w/o Expert Specialization     — 5 agents with same prompt vs 5 specialized
A2: w/o Disagreement Analysis     — Margin-only routing, no semantic analysis
A3: w/o Dynamic Routing           — Fixed full path for all samples
A4: Random Recheck                — Random expert instead of targeted
A5: w/o Controlled Deliberation   — Skip deliberation, go to Judge directly
A6: Free Debate                   — Unstructured debate vs controlled hypothesis-critic
A7: w/o Critic                    — Hypothesis A+B only, no CriticAgent
A8: Early Judge                   — Judge before recheck/deliberation
"""

from artifact_delib.ablations.no_expert_specialization import AblationNoExpertSpecialization
from artifact_delib.ablations.no_disagreement_analysis import AblationNoDisagreementAnalysis
from artifact_delib.ablations.no_dynamic_routing import AblationNoDynamicRouting
from artifact_delib.ablations.random_recheck import AblationRandomRecheck
from artifact_delib.ablations.no_controlled_deliberation import AblationNoControlledDeliberation
from artifact_delib.ablations.free_debate import AblationFreeDebate
from artifact_delib.ablations.no_critic import AblationNoCritic
from artifact_delib.ablations.early_judge import AblationEarlyJudge

# ── Legacy ablation imports (from src/artifact_delib/ablations.py) ──
# Because this package shadows the old module file, we import the old file
# using importlib to make them available.

import importlib
import sys

_old_name = "artifact_delib.ablations_old"
if _old_name not in sys.modules:
    # Load the old ablations module under an alias
    from importlib.machinery import SourceFileLoader
    import os
    _path = os.path.join(
        os.path.dirname(__file__), "..", "ablations.py"
    )
    if os.path.exists(_path):
        _spec = importlib.util.spec_from_file_location(_old_name, _path)
        if _spec and _spec.loader:
            _mod = importlib.util.module_from_spec(_spec)
            sys.modules[_old_name] = _mod
            _spec.loader.exec_module(_mod)

            # Re-export legacy classes with Legacy prefix (to avoid conflicts)
            # Old classes that have new equivalents are NOT re-exported
            LEGACY_MAP = {
                "AblationNoMultiExpert": "LegacyAblationNoMultiExpert",
                "AblationSingleExpert": "LegacyAblationSingleExpert",
                "AblationNoRouter": "LegacyAblationNoRouter",
                "AblationNoRecheck": "LegacyAblationNoRecheck",
                "AblationFixedAllRecheck": "LegacyAblationFixedAllRecheck",
                "AblationNoDeliberation": "LegacyAblationNoDeliberation",
                "AblationFreeDebate": "LegacyAblationFreeDebate",
                "AblationFixedDeliberation": "LegacyAblationFixedDeliberation",
                "AblationNoDeferredJudge": "LegacyAblationNoDeferredJudge",
            }
            for _old_name, _legacy_name in LEGACY_MAP.items():
                if hasattr(_mod, _old_name):
                    globals()[_legacy_name] = getattr(_mod, _old_name)

__all__ = [
    "AblationNoExpertSpecialization",
    "AblationNoDisagreementAnalysis",
    "AblationNoDynamicRouting",
    "AblationRandomRecheck",
    "AblationNoControlledDeliberation",
    "AblationFreeDebate",
    "AblationNoCritic",
    "AblationEarlyJudge",
    # Legacy (prefixed to avoid conflict)
    "LegacyAblationNoMultiExpert",
    "LegacyAblationSingleExpert",
    "LegacyAblationNoRouter",
    "LegacyAblationNoRecheck",
    "LegacyAblationFixedAllRecheck",
    "LegacyAblationNoDeliberation",
    "LegacyAblationFreeDebate",
    "LegacyAblationFixedDeliberation",
    "LegacyAblationNoDeferredJudge",
]
