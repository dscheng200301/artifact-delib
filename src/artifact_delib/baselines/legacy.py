"""Legacy / internal diagnostic baselines — NOT in the external baseline table.

These are kept for backward compatibility and internal analysis:
  - FixedMultiExpertBaseline — always run all 5 experts, no routing
  - FixedFullBaseline       — always run all experts + rechecks + deliberation
  - GenericMADBaseline      — generic N-agent free debate (not controlled deliberation)

These are NOT external baselines. They are internal diagnostics to validate
specific design decisions within the ArtifactDelib framework.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from artifact_delib.api.base import ModelClient
from artifact_delib.api.schemas import ModelRequest, TokenUsage
from artifact_delib.constants import MAX_DELIBERATION_ROUNDS
from artifact_delib.schemas import (
    CandidateSet,
    ExpertReport,
    FinalIdentification,
    PipelineResult,
    RouteDecision,
    SummarizedReport,
    VisualPerceptionReport,
)

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


def _make_experts(client: ModelClient, model_name: str):
    return (
        ShapeExpert(client, model_name),
        StyleExpert(client, model_name),
        GlyphExpert(client, model_name),
        MaterialCraftExpert(client, model_name),
        LocalDetailExpert(client, model_name),
    )


def _name_match(record_name: str, expert_name: str) -> bool:
    """Match Chinese expert names to English internal names."""
    name_map = {
        "器形": "shape",
        "纹饰风格": "style",
        "铭文款识": "glyph",
        "材质工艺": "material",
        "局部细节": "local_detail",
    }
    return name_map.get(record_name) == expert_name


# ═══════════════════════════════════════════════════════════════════
#  FixedMultiExpertBaseline  (legacy diagnostic)
# ═══════════════════════════════════════════════════════════════════

class FixedMultiExpertBaseline:
    """LEGACY: Image → VP → 5 Experts → Summarizer → Judge (no routing).

    Previously called B2. Now an INTERNAL DIAGNOSTIC, not an external baseline.
    Proves that dynamic routing reduces unnecessary expert calls.

    .. deprecated::
        This is kept for backward compatibility. New baselines should
        use the new package structure.
    """

    name = "fixed_multi_expert"

    def __init__(
        self,
        client: ModelClient,
        model_name: str = "default",
    ) -> None:
        self.client = client
        self.model_name = model_name
        self.vp = VisualPerceptionAgent(client, model_name)
        self.experts = _make_experts(client, model_name)
        self.summarizer = ReportSummarizer(client, model_name)
        self.judge = ArtifactJudge(client, model_name)

    def run(
        self,
        image_path: Path,
        sample_id: str = "unknown",
    ) -> PipelineResult:
        total_calls = 0

        vp = self.vp.analyze(image_path)
        total_calls += 1

        reports = []
        for exp in self.experts:
            reports.append(exp.analyze(image_path))
            total_calls += 1
        et = tuple(reports)

        summary = self.summarizer.summarize(vp, et)
        total_calls += 1

        cg = CandidateGenerator(self.client, self.model_name)
        candidates = cg.generate(summary)
        total_calls += 1

        final = self.judge.adjudicate(image_path, vp, summary, candidates, et)
        total_calls += 1

        return PipelineResult(
            sample_id=sample_id,
            final_identification=final,
            visual_perception_report=vp,
            expert_reports=et,
            summarized_report=summary,
            initial_candidates=candidates,
            total_usage=TokenUsage(),
            total_api_calls=total_calls,
            status="COMPLETED",
        )


# ═══════════════════════════════════════════════════════════════════
#  GenericMADBaseline  (legacy diagnostic / internal)
# ═══════════════════════════════════════════════════════════════════

class GenericMADBaseline:
    """LEGACY: N independent agents → free debate → Judge.

    Previously called B3. Now an INTERNAL DIAGNOSTIC.
    Use MultiAgentDebateBaseline for the external baseline instead.

    .. deprecated::
        Use MultiAgentDebateBaseline from artifact_delib.baselines instead.
    """

    name = "generic_mad"
    N_AGENTS = 4
    N_ROUNDS = 2

    def __init__(
        self,
        client: ModelClient,
        model_name: str = "default",
        n_agents: int = 4,
        n_rounds: int = 2,
    ) -> None:
        self.client = client
        self.model_name = model_name
        self.n_agents = n_agents
        self.n_rounds = n_rounds

    def run(
        self,
        image_path: Path,
        sample_id: str = "unknown",
    ) -> PipelineResult:
        total_calls = 0

        opinions: list[str] = []
        for i in range(self.n_agents):
            opinion, usage = self._call_agent(image_path, i + 1, prior_opinions=())
            opinions.append(opinion)
            total_calls += 1

        for rnd in range(1, self.n_rounds):
            new_opinions: list[str] = []
            for i in range(self.n_agents):
                opinion, usage = self._call_agent(
                    image_path, i + 1,
                    prior_opinions=tuple(opinions[-self.n_agents:]),
                )
                new_opinions.append(opinion)
                total_calls += 1
            opinions = new_opinions

        judge = ArtifactJudge(self.client, self.model_name)
        summary = SummarizedReport(
            content="多位专家自由辩论结果：\n" + "\n".join(
                f"Agent {i+1}: {op[:80]}..." for i, op in enumerate(opinions)
            ),
        )
        candidates = CandidateSet(candidates=())
        final = judge.adjudicate(
            image_path,
            VisualPerceptionReport(content="", usage=TokenUsage()),
            summary, candidates,
        )
        total_calls += 1

        return PipelineResult(
            sample_id=sample_id,
            final_identification=final,
            visual_perception_report=VisualPerceptionReport(content="", usage=TokenUsage()),
            expert_reports=(),
            summarized_report=summary,
            initial_candidates=CandidateSet(candidates=()),
            total_usage=TokenUsage(),
            total_api_calls=total_calls,
            status="COMPLETED",
        )

    def _call_agent(
        self,
        image_path: Path,
        agent_no: int,
        prior_opinions: tuple[str, ...],
    ) -> tuple[str, TokenUsage]:
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
            temperature=0.3,
            prompt_name="generic_mad",
        ))
        return response.content.strip(), response.usage


# ═══════════════════════════════════════════════════════════════════
#  FixedFullBaseline  (legacy diagnostic / oracle upper bound)
# ═══════════════════════════════════════════════════════════════════

class FixedFullBaseline:
    """LEGACY: Always run all experts + all rechecks + 2 deliberation rounds.

    Previously called B4. Now an INTERNAL DIAGNOSTIC — demonstrates the
    cost upper bound when no adaptive routing is used.

    .. deprecated::
        This is kept for backward compatibility.
    """

    name = "fixed_full"

    def __init__(
        self,
        client: ModelClient,
        model_name: str = "default",
    ) -> None:
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

    def run(
        self,
        image_path: Path,
        sample_id: str = "unknown",
    ) -> PipelineResult:
        total_calls = 0

        vp = self.vp.analyze(image_path)
        total_calls += 1

        reports = []
        for exp in [self.shape_expert, self.style_expert, self.glyph_expert,
                     self.material_expert, self.local_detail_expert]:
            reports.append(exp.analyze(image_path))
            total_calls += 1
        et = tuple(reports)

        summary = self.summarizer.summarize(vp, et)
        total_calls += 1

        candidates = self.candidate_gen.generate(summary)
        total_calls += 1

        disagreement = self.disagreement_analyzer.analyze(candidates, summary)
        total_calls += 1

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

        expert_reports = []
        for exp_name in ["shape", "style", "glyph", "material", "local_detail"]:
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

        deliberation_result = self.deliberation_manager.deliberate(
            candidates, summary, et_updated,
            recheck_reports=tuple(
                ExpertReport(expert_name=r.expert_name, content=r.new_content, usage=r.usage)
                for r in recheck_records
            ),
        )
        total_calls += len(deliberation_result.rounds) * 3

        final = self.judge.adjudicate(
            image_path, vp, summary, candidates, et_updated,
            recheck_reports=tuple(
                ExpertReport(expert_name=r.expert_name, content=r.new_content, usage=r.usage)
                for r in recheck_records
            ),
            deliberation_result=deliberation_result,
        )
        total_calls += 1

        return PipelineResult(
            sample_id=sample_id,
            final_identification=final,
            visual_perception_report=vp,
            expert_reports=et_updated,
            summarized_report=summary,
            initial_candidates=candidates,
            disagreement_analysis=disagreement,
            route_decisions=tuple(
                RouteDecision(action=a, reason="fixed_full", recheck_count=0)
                for a in ("SHAPE_RECHECK", "STYLE_RECHECK", "GLYPH_RECHECK",
                          "MATERIAL_RECHECK", "LOCAL_DETAIL_RECHECK")
            ),
            recheck_reports=tuple(
                ExpertReport(expert_name=r.expert_name, content=r.new_content, usage=r.usage)
                for r in recheck_records
            ),
            recheck_records=tuple(recheck_records),
            deliberation_result=deliberation_result,
            total_usage=TokenUsage(),
            total_api_calls=total_calls,
            status="COMPLETED",
        )
