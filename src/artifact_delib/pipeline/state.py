"""PipelineState and RunAccounting for pluggable pipeline stages.

PipelineState holds all intermediate results, so ablation variants can
override individual stages rather than reimplementing the entire run().

RunAccounting provides unified, auditable API call / token / latency / cost
tracking across all pipeline methods and external baselines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from artifact_delib.api.schemas import TokenUsage
from artifact_delib.schemas import (
    CandidateSet,
    DeliberationResult,
    DisagreementAnalysis,
    ExpertReport,
    FinalIdentification,
    PipelineResult,
    RecheckRecord,
    RouteDecision,
    SummarizedReport,
    VisualPerceptionReport,
)


# ═══════════════════════════════════════════════════════════════
#  RunAccounting — unified token/call/latency/cost tracker
# ═══════════════════════════════════════════════════════════════

@dataclass
class RunAccounting:
    """Unified accounting for one sample run.

    Every model call records its usage here. Cache hits increment
    logical_call_count but NOT remote_call_count.
    """

    logical_call_count: int = 0
    remote_call_count: int = 0
    cache_hits: int = 0

    input_tokens: int = 0
    output_tokens: int = 0

    total_latency_ms: float = 0.0

    estimated_cost_usd: float = 0.0

    per_agent_calls: dict[str, int] = field(default_factory=dict)
    per_agent_tokens: dict[str, int] = field(default_factory=dict)

    # Track individual call records for detailed audit
    call_records: list[dict] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def record_call(
        self,
        agent_name: str,
        usage: TokenUsage,
        latency_ms: float | None = None,
        cost_usd: float = 0.0,
        cache_hit: bool = False,
    ) -> None:
        """Record one logical model call.

        Latency is auto-detected from usage.total_latency_ms if not explicitly
        provided. Cost is estimated at $0.002/1K input + $0.008/1K output
        if not explicitly provided.
        """
        self.logical_call_count += 1
        if cache_hit:
            self.cache_hits += 1
        else:
            self.remote_call_count += 1

        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens

        # Use explicit latency, fall back to usage.total_latency_ms
        actual_latency = latency_ms if latency_ms is not None else usage.total_latency_ms
        self.total_latency_ms += actual_latency

        # Auto-estimate cost if not provided
        if cost_usd == 0.0:
            cost_usd = (
                usage.input_tokens * 0.002 / 1000
                + usage.output_tokens * 0.008 / 1000
            )
        self.estimated_cost_usd += cost_usd

        self.per_agent_calls[agent_name] = (
            self.per_agent_calls.get(agent_name, 0) + 1
        )
        self.per_agent_tokens[agent_name] = (
            self.per_agent_tokens.get(agent_name, 0) + usage.total_tokens
        )

        self.call_records.append({
            "agent": agent_name,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "latency_ms": latency_ms,
            "cost_usd": cost_usd,
            "cache_hit": cache_hit,
        })

    def to_token_usage(self) -> TokenUsage:
        return TokenUsage(
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
        )

    def merge(self, other: RunAccounting) -> None:
        """Merge another accounting into this one."""
        self.logical_call_count += other.logical_call_count
        self.remote_call_count += other.remote_call_count
        self.cache_hits += other.cache_hits
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.total_latency_ms += other.total_latency_ms
        self.estimated_cost_usd += other.estimated_cost_usd
        for k, v in other.per_agent_calls.items():
            self.per_agent_calls[k] = self.per_agent_calls.get(k, 0) + v
        for k, v in other.per_agent_tokens.items():
            self.per_agent_tokens[k] = self.per_agent_tokens.get(k, 0) + v
        self.call_records.extend(other.call_records)


# ═══════════════════════════════════════════════════════════════
#  PipelineState — holds all intermediate results
# ═══════════════════════════════════════════════════════════════

@dataclass
class PipelineState:
    """Mutable container for all intermediate pipeline state.

    Each pipeline stage reads from and writes to this state.
    Ablation variants override individual stages to change behavior.
    """

    image_path: Path
    sample_id: str = "unknown"

    # Accounting
    accounting: RunAccounting = field(default_factory=RunAccounting)

    # Step 1: Visual Perception
    visual_report: VisualPerceptionReport | None = None

    # Step 2: Expert Reports
    expert_reports: list[ExpertReport] = field(default_factory=list)

    # Step 3: Summarized Report
    summarized_report: SummarizedReport | None = None

    # Step 4: Candidate Generation
    initial_candidates: CandidateSet | None = None
    current_candidates: CandidateSet | None = None

    # Step 5: Disagreement Analysis
    disagreement: DisagreementAnalysis | None = None

    # Routing loop
    route_decisions: list[RouteDecision] = field(default_factory=list)
    recheck_records: list[RecheckRecord] = field(default_factory=list)
    recheck_count: int = 0
    deliberation_count: int = 0
    completed_rechecks: set[str] = field(default_factory=set)

    # Preliminary judgment (for early-judge ablation)
    preliminary_judgment: FinalIdentification | None = None

    # Deliberation
    deliberation_result: DeliberationResult | None = None

    # Final
    final_identification: FinalIdentification | None = None

    def to_pipeline_result(self) -> PipelineResult:
        """Convert state to a PipelineResult."""
        return PipelineResult(
            sample_id=self.sample_id,
            final_identification=self.final_identification
            or FinalIdentification(content="", usage=TokenUsage()),
            visual_perception_report=self.visual_report
            or VisualPerceptionReport(content="", usage=TokenUsage()),
            expert_reports=tuple(self.expert_reports),
            summarized_report=self.summarized_report
            or SummarizedReport(content="", usage=TokenUsage()),
            initial_candidates=self.initial_candidates
            or CandidateSet(candidates=()),
            disagreement_analysis=self.disagreement,
            route_decisions=tuple(self.route_decisions),
            recheck_reports=tuple(
                ExpertReport(
                    expert_name=r.expert_name,
                    content=r.new_content,
                    usage=r.usage,
                )
                for r in self.recheck_records
            ),
            updated_candidates=(
                self.current_candidates if len(self.route_decisions) > 1 else None
            ),
            recheck_records=tuple(self.recheck_records),
            deliberation_result=self.deliberation_result,
            total_usage=self.accounting.to_token_usage(),
            total_api_calls=self.accounting.logical_call_count,
            status="COMPLETED",
        )
