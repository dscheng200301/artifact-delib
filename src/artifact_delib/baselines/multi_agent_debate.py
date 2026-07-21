"""Multi-Agent Debate Baseline (external baseline).

Image → 4 Independent General Agents → 2 Debate Rounds → Final Judge

Each agent sees the original image. Round 1: independent analysis. Round 2: agents
see each other's opinions and can revise. A final judge makes the identification.

This is a strong external baseline that does NOT use ArtifactDelib's:
  - Specialized expert decomposition
  - Top-K candidate generation
  - Disagreement analysis
  - Targeted recheck
  - Controlled deliberation
"""

from __future__ import annotations

import uuid
from pathlib import Path

from artifact_delib.api.base import ModelClient
from artifact_delib.api.schemas import ModelRequest, TokenUsage
from artifact_delib.schemas import (
    CandidateSet,
    ExpertReport,
    FinalIdentification,
    PipelineResult,
    SummarizedReport,
    VisualPerceptionReport,
)
from artifact_delib.agents.artifact_judge import ArtifactJudge

DEFAULT_N_AGENTS = 4
DEFAULT_N_ROUNDS = 2


class MultiAgentDebateBaseline:
    """Image → N agents → M debate rounds → Judge.

    Pure multi-agent debate without controlled deliberation.
    This is an external baseline, NOT an ablation of ArtifactDelib.
    """

    name = "multi_agent_debate"

    def __init__(
        self,
        client: ModelClient,
        model_name: str = "default",
        n_agents: int = DEFAULT_N_AGENTS,
        n_rounds: int = DEFAULT_N_ROUNDS,
    ) -> None:
        self.client = client
        self.model_name = model_name
        self.n_agents = max(2, n_agents)
        self.n_rounds = max(1, n_rounds)

    def run(
        self,
        image_path: Path,
        sample_id: str = "unknown",
    ) -> PipelineResult:
        """Run multi-agent debate with independent rounds."""
        import base64

        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        total_calls = 0
        total_input = 0
        total_output = 0

        # ── Round 1: Independent analysis (agents can't see each other) ──
        opinions: list[str] = []
        for i in range(self.n_agents):
            opinion, usage = self._call_agent(encoded, agent_no=i + 1, prior_opinions=())
            opinions.append(opinion)
            total_calls += 1
            total_input += usage.input_tokens
            total_output += usage.output_tokens

        # ── Subsequent rounds: agents see prior opinions ──
        for rnd in range(1, self.n_rounds):
            new_opinions: list[str] = []
            for i in range(self.n_agents):
                # All prior round opinions are shared (but not hidden chain-of-thought)
                opinion, usage = self._call_agent(
                    encoded,
                    agent_no=i + 1,
                    prior_opinions=tuple(opinions),
                )
                new_opinions.append(opinion)
                total_calls += 1
                total_input += usage.input_tokens
                total_output += usage.output_tokens
            opinions = new_opinions

        # ── Final Judge ──
        judge = ArtifactJudge(self.client, self.model_name)
        debate_summary = SummarizedReport(
            content="多智能体辩论结果：\n" + "\n".join(
                f"Agent {i+1}: {op[:120]}..."
                for i, op in enumerate(opinions)
            ),
            usage=TokenUsage(),
        )

        # The final judge call uses same judge as ArtifactDelib
        final = judge.adjudicate(
            image_path=image_path,
            visual_report=VisualPerceptionReport(content="", usage=TokenUsage()),
            summarized_report=debate_summary,
            candidates=CandidateSet(candidates=()),
            expert_reports=(),
        )
        total_calls += 1
        total_input += final.usage.input_tokens
        total_output += final.usage.output_tokens

        total_usage = TokenUsage(input_tokens=total_input, output_tokens=total_output)
        return PipelineResult(
            sample_id=sample_id,
            final_identification=final,
            visual_perception_report=VisualPerceptionReport(content="", usage=TokenUsage()),
            expert_reports=(),
            summarized_report=debate_summary,
            initial_candidates=CandidateSet(candidates=()),
            total_usage=total_usage,
            total_api_calls=total_calls,
            status="COMPLETED",
        )

    def _call_agent(
        self,
        encoded: str,
        agent_no: int,
        prior_opinions: tuple[str, ...],
    ) -> tuple[str, TokenUsage]:
        """Call one debate agent — returns (opinion_text, usage)."""
        prior_text = ""
        if prior_opinions:
            # Share only the public reasoning, not hidden chain-of-thought
            prior_text = "\n其他专家的已发表意见：\n" + "\n".join(
                f"  Agent {i+1}: {op[:200]}"
                for i, op in enumerate(prior_opinions)
            )

        response = self.client.generate(ModelRequest(
            request_id=str(uuid.uuid4()),
            model=self.model_name,
            system_prompt=(
                f"你是古代文物鉴定专家 #{agent_no}。请根据图片独立判断文物身份。\n"
                "你可以看到其他专家的已发表意见，但请基于你自己的分析做出判断。\n"
                "输出一段清晰的自然语言识别结论，包括大类、具体类型和可能的年代。"
            ),
            user_prompt=f"请识别这件文物。{prior_text}",
            image_base64=f"data:image/png;base64,{encoded}",
            max_output_tokens=256,
            temperature=0.3,
            prompt_name="multi_agent_debate",
        ))
        return response.content.strip(), response.usage
