"""Baselines for ArtifactDelib paper evaluation.

B1: DirectVLM          — Image → Single VLM → Final Answer
B2: FixedMultiExpert   — Image → VP → 5 Experts → Summarizer → Judge (no routing)
B3: GenericMADebate    — Multi-Agent Debate: N agents → free debate → Judge
B4: FixedFull          — Always run all experts + all rechecks + 2 deliberation rounds
B5: ArtifactDelib-Rule — Full pipeline with RuleRouter (== ArtifactDelibPipeline)
"""

from __future__ import annotations

import uuid
from pathlib import Path

from artifact_delib.api.base import ModelClient
from artifact_delib.api.schemas import ModelRequest, TokenUsage

from artifact_delib.agents.artifact_judge import ArtifactJudge
from artifact_delib.agents.candidate_generator import CandidateGenerator
from artifact_delib.agents.deliberation.critic_agent import CriticAgent
from artifact_delib.agents.deliberation.deliberation_manager import DeliberationManager
from artifact_delib.agents.deliberation.hypothesis_agent import HypothesisAgent
from artifact_delib.agents.disagreement_analyzer import DisagreementAnalyzer
from artifact_delib.agents.experts.glyph_expert import GlyphExpert
from artifact_delib.agents.experts.local_detail_expert import LocalDetailExpert
from artifact_delib.agents.experts.material_craft_expert import MaterialCraftExpert
from artifact_delib.agents.experts.shape_expert import ShapeExpert
from artifact_delib.agents.experts.style_expert import StyleExpert
from artifact_delib.agents.report_summarizer import ReportSummarizer
from artifact_delib.agents.targeted_recheck import TargetedExpertRecheck
from artifact_delib.agents.visual_perception_agent import VisualPerceptionAgent
from artifact_delib.constants import MAX_DELIBERATION_ROUNDS
from artifact_delib.pipeline.artifact_delib_pipeline import ArtifactDelibPipeline
from artifact_delib.schemas import (
    CandidateSet,
    ExpertReport,
    FinalIdentification,
    PipelineResult,
    RouteDecision,
    SummarizedReport,
    VisualPerceptionReport,
)

# ═══════════════════════════════════════════════════════════════
#  Shared helpers
# ═══════════════════════════════════════════════════════════════

def _make_experts(client: ModelClient, model_name: str):
    return (
        ShapeExpert(client, model_name),
        StyleExpert(client, model_name),
        GlyphExpert(client, model_name),
        MaterialCraftExpert(client, model_name),
        LocalDetailExpert(client, model_name),
    )


# ═══════════════════════════════════════════════════════════════
#  B1: Direct VLM — single call, no experts, no routing
# ═══════════════════════════════════════════════════════════════

class DirectVLMBaseline:
    """B1: Image → Single VLM → Final Answer.

    One model call. Proves whether multi-expert decomposition improves accuracy.
    """

    def __init__(self, client: ModelClient, model_name: str = "default") -> None:
        self.client = client
        self.model_name = model_name

    def run(self, image_path: Path, sample_id: str = "unknown") -> PipelineResult:
        import base64
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        response = self.client.generate(ModelRequest(
            request_id=str(uuid.uuid4()),
            model=self.model_name,
            system_prompt=(
                "你是古代文物鉴定专家。请直接识别这张文物图片中的器物。\n\n"
                "请输出一段自然语言描述，包括：\n"
                "- 文物大类\n- 具体类型\n- 可能的年代或朝代\n- 可能的材质和工艺\n\n"
                "不要输出JSON。直接输出一段流畅的自然语言。"
            ),
            user_prompt="请直接识别这件文物。",
            image_base64=f"data:image/png;base64,{encoded}",
            max_output_tokens=512,
            temperature=0.0,
            prompt_name="direct_vlm",
        ))
        final = FinalIdentification(content=response.content.strip(), usage=response.usage)
        return PipelineResult(
            sample_id=sample_id, final_identification=final,
            visual_perception_report=VisualPerceptionReport(content="", usage=TokenUsage()),
            expert_reports=(), summarized_report=SummarizedReport(content="", usage=TokenUsage()),
            initial_candidates=CandidateSet(candidates=()),
            total_usage=response.usage, total_api_calls=1, status="COMPLETED",
        )


# ═══════════════════════════════════════════════════════════════
#  B2: Fixed Multi-Expert — all experts, no routing/recheck
# ═══════════════════════════════════════════════════════════════

class FixedMultiExpertBaseline:
    """B2: Image → VP → 5 Experts → Summarizer → Judge.

    Same 5 experts every sample, no dynamic routing.
    Proves whether dynamic routing reduces unnecessary expert calls.
    """

    def __init__(self, client: ModelClient, model_name: str = "default") -> None:
        self.client = client
        self.model_name = model_name
        self.vp = VisualPerceptionAgent(client, model_name)
        self.experts = _make_experts(client, model_name)
        self.summarizer = ReportSummarizer(client, model_name)
        self.judge = ArtifactJudge(client, model_name)

    def run(self, image_path: Path, sample_id: str = "unknown") -> PipelineResult:
        total_calls = 0

        # VP
        vp = self.vp.analyze(image_path)
        total_calls += 1

        # All 5 experts
        reports = []
        for exp in self.experts:
            reports.append(exp.analyze(image_path))
            total_calls += 1
        et = tuple(reports)

        # Summarize
        summary = self.summarizer.summarize(vp, et)
        total_calls += 1

        # Candidate generator (for comparison fairness)
        cg = CandidateGenerator(self.client, self.model_name)
        candidates = cg.generate(summary)
        total_calls += 1

        # Judge
        final = self.judge.adjudicate(image_path, vp, summary, candidates, et)
        total_calls += 1

        return PipelineResult(
            sample_id=sample_id, final_identification=final,
            visual_perception_report=vp, expert_reports=et,
            summarized_report=summary, initial_candidates=candidates,
            total_usage=TokenUsage(), total_api_calls=total_calls, status="COMPLETED",
        )


# ═══════════════════════════════════════════════════════════════
#  B3: Generic Multi-Agent Debate — free debate, not controlled
# ═══════════════════════════════════════════════════════════════

class GenericMADBaseline:
    """B3: N agents independently judge → free debate N rounds → Judge.

    No controlled deliberation. Agents debate freely without hypothesis assignment.
    Proves whether controlled deliberation (Innovation 3) is better than free debate.
    """

    N_AGENTS = 4
    N_ROUNDS = 2

    def __init__(self, client: ModelClient, model_name: str = "default",
                 n_agents: int = 4, n_rounds: int = 2) -> None:
        self.client = client
        self.model_name = model_name
        self.n_agents = n_agents
        self.n_rounds = n_rounds

    def run(self, image_path: Path, sample_id: str = "unknown") -> PipelineResult:
        total_calls = 0

        # Round 1: N agents independently judge
        opinions: list[str] = []
        for i in range(self.n_agents):
            opinion, usage = self._call_agent(image_path, i + 1, prior_opinions=())
            opinions.append(opinion)
            total_calls += 1

        # Debate rounds
        for rnd in range(1, self.n_rounds):
            new_opinions: list[str] = []
            for i in range(self.n_agents):
                opinion, usage = self._call_agent(
                    image_path, i + 1, prior_opinions=tuple(opinions[-self.n_agents:]),
                )
                new_opinions.append(opinion)
                total_calls += 1
            opinions = new_opinions

        # Final judge
        judge = ArtifactJudge(self.client, self.model_name)
        summary = SummarizedReport(
            content="多位专家自由辩论结果：\n" + "\n".join(
                f"Agent {i+1}: {op[:80]}..." for i, op in enumerate(opinions)
            ),
        )
        candidates = CandidateSet(candidates=())
        final = judge.adjudicate(
            image_path, VisualPerceptionReport(content="", usage=TokenUsage()),
            summary, candidates,
        )
        total_calls += 1

        return PipelineResult(
            sample_id=sample_id, final_identification=final,
            visual_perception_report=VisualPerceptionReport(content="", usage=TokenUsage()),
            expert_reports=(), summarized_report=summary,
            initial_candidates=CandidateSet(candidates=()),
            total_usage=TokenUsage(), total_api_calls=total_calls, status="COMPLETED",
        )

    def _call_agent(self, image_path: Path, agent_no: int,
                    prior_opinions: tuple[str, ...]) -> tuple[str, TokenUsage]:
        import base64
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")

        prior_text = ""
        if prior_opinions:
            prior_text = "其他专家的已发表意见：\n" + "\n".join(
                f"  Agent {i+1}: {op[:100]}" for i, op in enumerate(prior_opinions)
            )

        response = self.client.generate(ModelRequest(
            request_id=str(uuid.uuid4()),
            model=self.model_name,
            system_prompt=(
                f"你是古代文物鉴定专家 #{agent_no}。请根据图片独立判断文物身份。\n"
                "先查看其他专家的意见（如果有），然后给出你的判断。\n"
                "输出一段自然语言识别结果。不要输出JSON。"
            ),
            user_prompt=f"请识别这件文物。{prior_text}",
            image_base64=f"data:image/png;base64,{encoded}",
            max_output_tokens=256,
            temperature=0.3,  # slight variance for debate
            prompt_name="generic_mad",
        ))
        return response.content.strip(), response.usage


# ═══════════════════════════════════════════════════════════════
#  B4: FixedFull — always run everything unconditionally
# ═══════════════════════════════════════════════════════════════

class FixedFullBaseline:
    """B4: Always run all experts + all rechecks + 2 deliberation rounds.

    No conditional routing. Every sample runs the maximum possible path.
    Proves whether dynamic routing reduces cost.
    """

    def __init__(self, client: ModelClient, model_name: str = "default") -> None:
        self.client = client
        self.model_name = model_name
        self.vp = VisualPerceptionAgent(client, model_name)
        se, st, g, m, l = _make_experts(client, model_name)
        self.shape_expert = se
        self.style_expert = st
        self.glyph_expert = g
        self.material_expert = m
        self.local_detail_expert = l
        self.summarizer = ReportSummarizer(client, model_name)
        self.candidate_gen = CandidateGenerator(client, model_name)
        self.disagreement_analyzer = DisagreementAnalyzer(client, model_name)
        self.recheck_coordinator = TargetedExpertRecheck(
            shape_expert=se, style_expert=st, glyph_expert=g,
            material_expert=m, local_detail_expert=l,
        )
        self.deliberation_manager = DeliberationManager(
            HypothesisAgent(client, model_name),
            CriticAgent(client, model_name),
            MAX_DELIBERATION_ROUNDS,
        )
        self.judge = ArtifactJudge(client, model_name)

    def run(self, image_path: Path, sample_id: str = "unknown") -> PipelineResult:
        total_calls = 0

        # Step 1: VP
        vp = self.vp.analyze(image_path)
        total_calls += 1

        # Step 2: All 5 experts
        reports = []
        for exp in [self.shape_expert, self.style_expert, self.glyph_expert,
                     self.material_expert, self.local_detail_expert]:
            reports.append(exp.analyze(image_path))
            total_calls += 1
        et = tuple(reports)

        # Step 3: Summarize
        summary = self.summarizer.summarize(vp, et)
        total_calls += 1

        # Step 4: Candidates
        candidates = self.candidate_gen.generate(summary)
        total_calls += 1

        # Step 5: Disagreement
        disagreement = self.disagreement_analyzer.analyze(candidates, summary)
        total_calls += 1

        # Step 6: Run ALL rechecks unconditionally
        recheck_records = []
        for action in ("SHAPE_RECHECK", "STYLE_RECHECK", "GLYPH_RECHECK",
                       "MATERIAL_RECHECK", "LOCAL_DETAIL_RECHECK"):
            route = RouteDecision(action=action, reason="fixed_full", recheck_count=0)
            record = self.recheck_coordinator.execute(
                image_path, route, candidates, disagreement, et,
                recheck_history=tuple(recheck_records), round_no=len(recheck_records) + 1,
            )
            recheck_records.append(record)
            total_calls += 1

        # Re-summarize after all rechecks
        expert_reports = []
        for exp_name in ["shape", "style", "glyph", "material", "local_detail"]:
            matching = [r for r in recheck_records if r.expert_name == exp_name or True]
            # Use last recheck for each expert
            last = next((r for r in reversed(recheck_records)
                         if _name_match(r.expert_name, exp_name)), None)
            if last:
                expert_reports.append(ExpertReport(
                    expert_name=exp_name, content=last.new_content,
                    usage=last.usage,
                ))
            else:
                for r in et:
                    if r.expert_name == exp_name:
                        expert_reports.append(r)
                        break
        et_updated = tuple(expert_reports)
        summary = self.summarizer.summarize(vp, et_updated)
        total_calls += 1

        # Step 7: Fixed 2 deliberation rounds
        deliberation_result = self.deliberation_manager.deliberate(
            candidates, summary, et_updated,
            recheck_reports=tuple(
                ExpertReport(expert_name=r.expert_name, content=r.new_content, usage=r.usage)
                for r in recheck_records
            ),
        )
        total_calls += len(deliberation_result.rounds) * 3

        # Step 8: Judge
        final = self.judge.adjudicate(
            image_path, vp, summary, candidates, et_updated,
            recheck_reports=tuple(
                ExpertReport(expert_name=r.expert_name,
                             content=r.new_content, usage=r.usage)
                for r in recheck_records
            ),
            deliberation_result=deliberation_result,
        )
        total_calls += 1

        return PipelineResult(
            sample_id=sample_id, final_identification=final,
            visual_perception_report=vp, expert_reports=et_updated,
            summarized_report=summary, initial_candidates=candidates,
            disagreement_analysis=disagreement,
            route_decisions=tuple(
                RouteDecision(action=a, reason="fixed_full", recheck_count=0)
                for a in ("SHAPE_RECHECK", "STYLE_RECHECK", "GLYPH_RECHECK",
                          "MATERIAL_RECHECK", "LOCAL_DETAIL_RECHECK")
            ),
            recheck_reports=tuple(
                ExpertReport(expert_name=r.expert_name,
                             content=r.new_content, usage=r.usage)
                for r in recheck_records
            ),
            recheck_records=tuple(recheck_records),
            deliberation_result=deliberation_result,
            total_usage=TokenUsage(), total_api_calls=total_calls, status="COMPLETED",
        )


def _name_match(record_name: str, expert_name: str) -> bool:
    """Match Chinese expert names to English internal names."""
    name_map = {"器形": "shape", "纹饰风格": "style", "铭文款识": "glyph",
                "材质工艺": "material", "局部细节": "local_detail"}
    return name_map.get(record_name) == expert_name
