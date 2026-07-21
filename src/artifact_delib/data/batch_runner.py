"""Batch runner — run any pipeline/baseline/ablation on an entire dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from artifact_delib.evaluation.experiment_logger import ExperimentLogger
from artifact_delib.evaluation.metrics import ArtifactMetrics, SampleEvaluation
from artifact_delib.evaluation.prediction_parser import ParsedIdentification
from artifact_delib.schemas import ArtifactSample, PipelineResult


class Runnable(Protocol):
    def run(self, image_path: Path, sample_id: str) -> PipelineResult:
        """Run one method on one sample."""
        ...


class BatchRunner:
    """Run any pipeline or baseline on a list of ArtifactSample.

    Supports:
    - Resumable execution (skips already-completed samples)
    - JSONL output
    - Experiment logging
    """

    def __init__(
        self,
        method: Runnable,
        output_root: Path,
        experiment_id: str,
        method_name: str = "unknown",
        config: dict | None = None,
    ) -> None:
        self.method = method
        self.output_root = output_root
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.experiment_id = experiment_id
        self.method_name = method_name
        self.config = config or {}
        self.logger = ExperimentLogger(output_root)

    def run(self, samples: list[ArtifactSample]) -> list[PipelineResult]:
        """Run the method on all samples, with resume support."""
        pred_path = self.output_root / "predictions.jsonl"
        existing: dict[str, PipelineResult] = {}
        if pred_path.exists():
            for line in pred_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    data = json.loads(line)
                    existing[data.get("sample_id", "")] = data

        self.logger.start_experiment(self.experiment_id, self.method_name, self.config)

        results: list[PipelineResult] = []
        n_skipped = 0

        with pred_path.open("a", encoding="utf-8") as f:
            for i, sample in enumerate(samples):
                if sample.sample_id in existing:
                    n_skipped += 1
                    continue

                result = self.method.run(sample.image_path, sample.sample_id)
                results.append(result)

                # Write JSONL row
                f.write(json.dumps({
                    "sample_id": result.sample_id,
                    "final_prediction": result.final_identification.content,
                    "api_calls": result.total_api_calls,
                    "tokens": result.total_usage.total_tokens,
                    "status": result.status,
                    "candidates": [
                        {"text": c.text, "confidence": c.confidence}
                        for c in result.initial_candidates.candidates
                    ],
                    "route_decisions": [
                        {"action": rd.action, "reason": rd.reason}
                        for rd in result.route_decisions
                    ],
                    "deliberation_rounds": len(
                        result.deliberation_result.rounds
                    ) if result.deliberation_result else 0,
                }, ensure_ascii=False) + "\n")

                if (i + n_skipped + 1) % 10 == 0:
                    print(f"  [{i + n_skipped + 1}/{len(samples)}] {sample.sample_id} "
                          f"→ {result.total_api_calls} calls")

        if n_skipped > 0:
            print(f"  Skipped {n_skipped} already-completed samples")

        return results

    def evaluate(
        self,
        results: list[PipelineResult],
        samples: list[ArtifactSample],
    ) -> dict:
        """Evaluate results against gold labels."""
        metrics = ArtifactMetrics()
        sample_map = {s.sample_id: s for s in samples}

        evals = []
        for r in results:
            gold = sample_map.get(r.sample_id)
            if gold is None:
                continue
            # Collect top-K candidate texts for Top-5 accuracy
            candidate_texts = [
                c.text for c in r.initial_candidates.candidates
            ] if r.initial_candidates and r.initial_candidates.candidates else None
            e = metrics.evaluate_sample(
                sample_id=r.sample_id,
                final_text=r.final_identification.content,
                gold_category=gold.category,
                gold_type=gold.fine_grained_type,
                gold_period=gold.period,
                gold_material=gold.material,
                candidate_texts=candidate_texts,
            )
            evals.append(e)

        agg = metrics.compute_metrics(
            evals,
            token_counts=[r.total_usage.total_tokens for r in results],
            api_calls=[r.total_api_calls for r in results],
            latencies_ms=[
                r.total_usage.total_latency_ms for r in results
            ],
            recheck_counts=[len(r.recheck_records) for r in results],
            deliberation_rounds=[
                len(r.deliberation_result.rounds) if r.deliberation_result else 0
                for r in results
            ],
            triggered_recheck=sum(
                1 for r in results if len(r.recheck_records) > 0
            ),
            triggered_deliberation=sum(
                1 for r in results
                if r.deliberation_result and len(r.deliberation_result.rounds) > 0
            ),
        )

        metrics_dict = {
            "n_samples": agg.n_samples,
            "top1_type_accuracy": agg.top1_type_accuracy,
            "top1_category_accuracy": agg.top1_category_accuracy,
            "top1_period_accuracy": agg.top1_period_accuracy,
            "top1_material_accuracy": agg.top1_material_accuracy,
            "top1_joint_accuracy": agg.top1_joint_accuracy,
            "top5_type_accuracy": agg.top5_type_accuracy,
            "top5_period_accuracy": agg.top5_period_accuracy,
            "top5_joint_accuracy": agg.top5_joint_accuracy,
            "macro_f1_type": agg.macro_f1_type,
            "macro_f1_category": agg.macro_f1_category,
            "macro_f1_period": agg.macro_f1_period,
            "macro_f1_material": agg.macro_f1_material,
            "micro_f1_type": agg.micro_f1_type,
            "micro_f1_category": agg.micro_f1_category,
            "micro_f1_period": agg.micro_f1_period,
            "micro_f1_material": agg.micro_f1_material,
            "parse_failure_rate": agg.parse_failure_rate,
            "average_tokens": agg.average_tokens,
            "average_api_calls": agg.average_api_calls,
            "p50_latency_ms": agg.p50_latency_ms,
            "p95_latency_ms": agg.p95_latency_ms,
            "average_latency_ms": agg.average_latency_ms,
            "correction_rate": agg.correction_rate,
            "harm_rate": agg.harm_rate,
            "no_change_rate": agg.no_change_rate,
            "deliberation_trigger_rate": agg.deliberation_trigger_rate,
            "recheck_trigger_rate": agg.recheck_trigger_rate,
            "avg_rechecks": agg.avg_rechecks,
            "avg_deliberation_rounds": agg.avg_deliberation_rounds,
        }
        self.logger.complete_experiment(metrics_dict, n_samples=len(results))
        return metrics_dict
