"""Controlled Expert Deliberation — hypothesis-level negotiation (Innovation 3).

When targeted recheck rounds are exhausted but candidates remain indistinguishable,
HypothesisAgent A argues for Top-1, HypothesisAgent B for Top-2, under Critic oversight.
Max 2 rounds. Stops on REVISE, ABSTAIN, or no new information.
"""

from artifact_delib.agents.deliberation.critic_agent import CriticAgent
from artifact_delib.agents.deliberation.deliberation_manager import DeliberationManager
from artifact_delib.agents.deliberation.hypothesis_agent import HypothesisAgent

__all__ = ["HypothesisAgent", "CriticAgent", "DeliberationManager"]
