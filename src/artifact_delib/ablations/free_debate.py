"""A6: Free Debate instead of Controlled Deliberation.

Replaces HypothesisAgent/CriticAgent with free-form debate among 2 agents.
This is an ABLATION — the rest of the ArtifactDelib pipeline is preserved.

Overrides _run_deliberation to use free debate instead of controlled deliberation.
The debate output is passed to the Judge as deliberation context.
"""

from __future__ import annotations

import uuid

from artifact_delib.pipeline.artifact_delib_pipeline import ArtifactDelibPipeline
from artifact_delib.pipeline.state import PipelineState
from artifact_delib.api.schemas import ModelRequest, TokenUsage
from artifact_delib.schemas import (
    CandidateSet,
    DeliberationResult,
    DeliberationRound,
    ExpertReport,
    SummarizedReport,
)


class AblationFreeDebate(ArtifactDelibPipeline):
    """A6: Free debate replaces controlled deliberation."""
    name = "ablation_free_debate"  # noqa: A003 — same as parent class attribute pattern

    def __init__(self, *args, n_agents: int = 2, n_rounds: int = 2, **kwargs):
        super().__init__(*args, **kwargs)
        self._n_agents = n_agents
        self._n_rounds = n_rounds

    def _run_deliberation(self, state: PipelineState) -> None:
        """Replace controlled deliberation with free-form debate."""
        import base64

        encoded = base64.b64encode(state.image_path.read_bytes()).decode("ascii")
        cands = state.current_candidates or state.initial_candidates
        if cands is None:
            cands = CandidateSet(candidates=())
        summary_text = (
            state.summarized_report.content[:300]
            if state.summarized_report
            else ""
        )

        # ── Round 1: Independent opinions ──
        opinions: list[str] = []
        for i in range(self._n_agents):
            response = self.client.generate(ModelRequest(
                request_id=str(uuid.uuid4()),
                model=self.model_name,
                system_prompt=(
                    f"你是考古文物鉴定专家 #{i+1}。请基于已有的专家分析报告，"
                    "独立判断这件文物的具体身份。"
                ),
                user_prompt=(
                    f"专家综合分析：{summary_text}\n"
                    f"主要候选：{cands.candidates[0].text if cands.candidates else '未知'}\n"
                    "请给出你的最终鉴定结论。"
                ),
                image_base64=f"data:image/png;base64,{encoded}",
                max_output_tokens=256,
                temperature=0.3,
                prompt_name="free_debate",
            ))
            opinions.append(response.content.strip())
            state.accounting.record_call(
                f"free_debate_agent_{i+1}_round_1",
                response.usage,
            )

        # ── Round 2: See others' opinions ──
        for rnd in range(1, self._n_rounds):
            new_opinions: list[str] = []
            for i in range(self._n_agents):
                prior = "\n".join(
                    f"Agent {j+1}: {op[:150]}"
                    for j, op in enumerate(opinions)
                )
                response = self.client.generate(ModelRequest(
                    request_id=str(uuid.uuid4()),
                    model=self.model_name,
                    system_prompt=(
                        f"你是专家 #{i+1}。请考虑其他专家的意见后，"
                        "给出你的最终判断。你可以坚持或修正你的观点。"
                    ),
                    user_prompt=f"其他专家意见：\n{prior}\n你的最终判断：",
                    image_base64=f"data:image/png;base64,{encoded}",
                    max_output_tokens=256,
                    temperature=0.3,
                    prompt_name="free_debate",
                ))
                new_opinions.append(response.content.strip())
                state.accounting.record_call(
                    f"free_debate_agent_{i+1}_round_{rnd+1}",
                    response.usage,
                )
            opinions = new_opinions

        # ── Pack into DeliberationResult so Judge can consume it ──
        debate_summary = "Free debate opinions:\n" + "\n".join(
            f"Agent {i+1}: {op[:150]}" for i, op in enumerate(opinions)
        )

        rounds = tuple(
            DeliberationRound(
                round_no=r,
                candidate_a_opinion=opinions[0][:200] if len(opinions) > 0 else "",
                candidate_b_opinion=opinions[1][:200] if len(opinions) > 1 else "",
                candidate_a_decision="MAINTAIN",
                candidate_b_decision="MAINTAIN",
                critic_feedback="(free debate — no critic)",
            )
            for r in range(1, self._n_rounds + 1)
        )

        state.deliberation_result = DeliberationResult(
            rounds=rounds,
            stop_reason="free_debate_max_rounds",
            summary=debate_summary,
            usage=state.accounting.to_token_usage(),
        )
