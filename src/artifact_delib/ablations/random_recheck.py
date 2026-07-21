"""A4: Random Recheck — replace targeted expert selection with random choice.

Same recheck trigger count as the full pipeline, but the expert to recheck
is chosen randomly instead of based on disagreement analysis.

Requires running with at least 3 different random seeds and reporting
mean and std deviation.
"""

from __future__ import annotations

import random
from pathlib import Path

from artifact_delib.constants import MAX_RECHECK_ROUNDS
from artifact_delib.pipeline.artifact_delib_pipeline import (
    ArtifactDelibPipeline,
    _ACTION_TO_EXPERT_NAME,
)
from artifact_delib.schemas import ExpertReport, PipelineResult, RouteDecision

_RECHECK_ACTIONS = list(_ACTION_TO_EXPERT_NAME.keys())


class AblationRandomRecheck(ArtifactDelibPipeline):
    """A4: Random targeted recheck — random expert instead of disagreement-driven.

    Same number of rechecks as full pipeline would do, but which expert
    to recheck is chosen uniformly at random from available (un-done) experts.
    """

    name = "ablation_random_recheck"

    def __init__(
        self,
        client,
        model_name: str = "default",
        top_k: int = 3,
        max_recheck_rounds: int = 2,
        random_seed: int = 42,
    ) -> None:
        super().__init__(client, model_name, top_k, max_recheck_rounds)
        self._rng = random.Random(random_seed)
        self.random_seed = random_seed

    def run(
        self,
        image_path: Path,
        sample_id: str = "unknown",
    ) -> PipelineResult:
        """Run pipeline with random recheck selection."""
        total_calls = 0

        # Standard steps: VP, experts, summarizer, candidates, disagreement
        vp = self.visual_perception.analyze(image_path)
        total_calls += 1

        reports = []
        for expert in [
            self.shape_expert, self.style_expert, self.glyph_expert,
            self.material_expert, self.local_detail_expert,
        ]:
            reports.append(expert.analyze(image_path))
            total_calls += 1
        et = tuple(reports)

        summary = self.summarizer.summarize(vp, et)
        total_calls += 1

        candidates = self.candidate_generator.generate(summary)
        total_calls += 1

        disagreement = self.disagreement_analyzer.analyze(candidates, summary)
        total_calls += 1

        # Random recheck loop (instead of rule-based)
        route_decisions: list[RouteDecision] = []
        recheck_records = []
        deliberation_result = None
        recheck_count = 0
        deliberation_count = 0
        completed_rechecks: set[str] = set()
        initial_candidates = candidates

        while True:
            # Normal router for first decision, then random for rechecks
            if recheck_count == 0:
                route = self.router.route(
                    disagreement=disagreement,
                    candidates=candidates,
                    recheck_count=recheck_count,
                    deliberation_count=deliberation_count,
                    completed_rechecks=tuple(completed_rechecks),
                )
            else:
                # Randomly choose an unused recheck action
                available = [a for a in _RECHECK_ACTIONS if a not in completed_rechecks]
                if not available or recheck_count >= self.max_recheck_rounds:
                    # After max rounds, use normal routing
                    route = self.router.route(
                        disagreement=disagreement,
                        candidates=candidates,
                        recheck_count=recheck_count,
                        deliberation_count=deliberation_count,
                        completed_rechecks=tuple(completed_rechecks),
                    )
                else:
                    chosen = self._rng.choice(available)
                    route = RouteDecision(
                        action=chosen,
                        reason=f"random:{self.random_seed}:{chosen}",
                        recheck_count=recheck_count,
                    )

            route_decisions.append(route)

            if route.action == "FAST":
                break

            if route.action == "DELIBERATION":
                deliberation_count += 1
                deliberation_result = self.deliberation_manager.deliberate(
                    candidates=candidates,
                    summarized_report=summary,
                    expert_reports=et,
                    recheck_reports=tuple(
                        ExpertReport(expert_name=r.expert_name, content=r.new_content, usage=r.usage)
                        for r in recheck_records
                    ),
                )
                for _ in deliberation_result.rounds:
                    total_calls += 3
                break

            if route.action in _ACTION_TO_EXPERT_NAME:
                record = self.recheck_coordinator.execute(
                    image_path=image_path,
                    route=route,
                    candidates=candidates,
                    disagreement=disagreement,
                    current_reports=et,
                    recheck_history=tuple(recheck_records),
                    round_no=recheck_count + 1,
                )
                total_calls += 1
                recheck_records.append(record)
                recheck_count += 1
                completed_rechecks.add(route.action)

                # Re-summarize, re-candidate, re-disagreement
                expert_name = _ACTION_TO_EXPERT_NAME[route.action]
                et = list(et)
                for i, r in enumerate(et):
                    if r.expert_name == expert_name:
                        et[i] = ExpertReport(
                            expert_name=expert_name,
                            content=record.new_content,
                            usage=record.usage,
                        )
                        break
                et = tuple(et)

                summary = self.summarizer.summarize(vp, et)
                total_calls += 1
                candidates = self.candidate_generator.generate(summary)
                total_calls += 1
                disagreement = self.disagreement_analyzer.analyze(candidates, summary)
                total_calls += 1
                continue

            break

        # Judge
        final = self.judge.adjudicate(
            image_path=image_path,
            visual_report=vp,
            summarized_report=summary,
            candidates=candidates,
            expert_reports=et,
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
            expert_reports=et,
            summarized_report=summary,
            initial_candidates=initial_candidates,
            disagreement_analysis=disagreement,
            route_decisions=tuple(route_decisions),
            recheck_reports=tuple(
                ExpertReport(expert_name=r.expert_name, content=r.new_content, usage=r.usage)
                for r in recheck_records
            ),
            recheck_records=tuple(recheck_records),
            deliberation_result=deliberation_result,
            total_usage=final.usage,
            total_api_calls=total_calls,
            status="COMPLETED",
        )
