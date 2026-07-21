"""Batch runner — run any pipeline/baseline/ablation on an entire dataset.

Supports:
- Resumable execution (skips already-completed samples, merges results)
- Config hash validation (refuses to resume with different config)
- JSONL output
- Experiment logging
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Protocol

from artifact_delib.evaluation.experiment_logger import ExperimentLogger
from artifact_delib.evaluation.metrics import ArtifactMetrics, SampleEvaluation
from artifact_delib.evaluation.prediction_parser import ParsedIdentification
from artifact_delib.schemas import ArtifactSample, PipelineResult

logger = logging.getLogger(__name__)


class Runnable(Protocol):
    def run(self, image_path: Path, sample_id: str) -> PipelineResult:
        """Run one method on one sample."""
        ...


def _compute_config_hash(config: dict | None) -> str:
    """Compute a hash of the experiment config for resume validation.

    Includes: method, model, temperature, max_tokens, prompt version, etc.
    """
    if config is None:
        return ""
    # Normalize: sort keys, exclude output dir and timestamps
    normalized = {}
    for k, v in sorted(config.items()):
        if k in ("output", "output_dir", "timestamp", "experiment"):
            continue
        normalized[k] = v
    raw = json.dumps(normalized, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class ResumeResult:
    """Deserialized resume state from a previous run."""

    def __init__(self, results: list[dict[str, Any]], config_hash: str) -> None:
        self.results = results
        self.config_hash = config_hash


class BatchRunner:
    """Run any pipeline or baseline on a list of ArtifactSample.

    Supports:
    - Resumable execution (skips already-completed samples)
    - Config hash validation (refuses to resume with different config)
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
        self.config_hash = _compute_config_hash(self.config)
        self.logger = ExperimentLogger(output_root)

    def run(self, samples: list[ArtifactSample]) -> list[PipelineResult]:
        """Run the method on all samples, with resume support.

        Returns ALL results (existing + new), in the requested sample order.
        """
        pred_path = self.output_root / "predictions.jsonl"

        # ── Load existing results ──
        existing: dict[str, dict[str, Any]] = {}
        if pred_path.exists():
            config_hash_path = self.output_root / ".config_hash"
            if config_hash_path.exists():
                old_hash = config_hash_path.read_text(encoding="utf-8").strip()
                if old_hash and old_hash != self.config_hash:
                    raise RuntimeError(
                        f"Config hash mismatch: existing={old_hash}, "
                        f"current={self.config_hash}. "
                        "Refusing to resume with different configuration."
                    )

            for line in pred_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    data = json.loads(line)
                    existing[data.get("sample_id", "")] = data
            logger.info("Loaded %d existing results from %s", len(existing), pred_path)

        self.logger.start_experiment(self.experiment_id, self.method_name, self.config)

        # ── Run missing samples ──
        new_results: list[PipelineResult] = []
        n_skipped = 0
        n_run = 0

        # Use a temporary file for atomic writes
        tmp_path = self.output_root / "predictions.jsonl.tmp"
        with tmp_path.open("a", encoding="utf-8") as f:
            for i, sample in enumerate(samples):
                if sample.sample_id in existing:
                    n_skipped += 1
                    continue

                try:
                    result = self.method.run(sample.image_path, sample.sample_id)
                    new_results.append(result)
                    n_run += 1

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

                    if (i + 1) % 10 == 0:
                        print(f"  [{i + 1}/{len(samples)}] {sample.sample_id} "
                              f"→ {result.total_api_calls} calls")
                except Exception as exc:
                    logger.error("Failed to run sample %s: %s", sample.sample_id, exc)
                    # Write a failure record
                    f.write(json.dumps({
                        "sample_id": sample.sample_id,
                        "final_prediction": "",
                        "api_calls": 0,
                        "tokens": 0,
                        "status": f"ERROR: {exc}",
                        "candidates": [],
                        "route_decisions": [],
                        "deliberation_rounds": 0,
                    }, ensure_ascii=False) + "\n")

        # ── Merge existing + new, write final consolidated file ──
        all_rows: list[dict[str, Any]] = list(existing.values())

        # Re-read the tmp file to get new rows
        if tmp_path.exists():
            for line in tmp_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    data = json.loads(line)
                    sid = data.get("sample_id", "")
                    if sid not in existing:
                        all_rows.append(data)
                        existing[sid] = data  # dedup for subsequent reads

        # Sort by requested sample order
        sample_order = {s.sample_id: i for i, s in enumerate(samples)}
        all_rows.sort(key=lambda r: sample_order.get(r.get("sample_id", ""), 9999))

        # Atomic write: write to tmp, then rename
        final_path = self.output_root / "predictions.jsonl"
        tmp_final = self.output_root / "predictions.jsonl.final.tmp"
        with tmp_final.open("w", encoding="utf-8") as f:
            for row in all_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        tmp_final.replace(final_path)

        # Clean up tmp files
        if tmp_path.exists():
            tmp_path.unlink()
        if tmp_final.exists():
            tmp_final.unlink()

        # Save config hash
        (self.output_root / ".config_hash").write_text(
            self.config_hash, encoding="utf-8"
        )

        # ── Build PipelineResult list from ALL rows ──
        # For existing results, we reconstruct minimal PipelineResult objects
        all_results: list[PipelineResult] = new_results[:]

        # Add placeholder results for skipped samples
        for row in all_rows:
            sid = row.get("sample_id", "")
            if not any(r.sample_id == sid for r in all_results):
                # Create a minimal PipelineResult from the JSON row
                from artifact_delib.api.schemas import TokenUsage
                from artifact_delib.schemas import (
                    CandidateSet,
                    FinalIdentification,
                    PipelineResult,
                    VisualPerceptionReport,
                    SummarizedReport,
                )

                # We lost the full result for skipped samples, but we can
                # reconstruct a minimal one for metrics evaluation
                all_results.append(PipelineResult(
                    sample_id=sid,
                    final_identification=FinalIdentification(
                        content=row.get("final_prediction", ""),
                        usage=TokenUsage(),
                    ),
                    visual_perception_report=VisualPerceptionReport(
                        content="", usage=TokenUsage(),
                    ),
                    expert_reports=(),
                    summarized_report=SummarizedReport(
                        content="", usage=TokenUsage(),
                    ),
                    initial_candidates=CandidateSet(candidates=()),
                    total_api_calls=row.get("api_calls", 0),
                    status=row.get("status", "COMPLETED"),
                ))

        # Re-sort to match input order
        all_results.sort(key=lambda r: sample_order.get(r.sample_id, 9999))

        if n_skipped > 0:
            print(f"  Skipped {n_skipped} already-completed samples")
        if n_run > 0:
            print(f"  Ran {n_run} new samples")

        print(f"  Total: {len(all_results)} results")

        return all_results

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