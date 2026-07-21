"""Result writer — save experiment outputs to disk.

Output structure::

    results/<experiment_name>/
    ├── config_snapshot.yaml
    ├── predictions.jsonl
    ├── raw_responses.jsonl
    ├── api_calls.jsonl
    ├── metrics.json
    ├── routing_stats.json
    ├── failures.jsonl
    └── experiment_summary.md
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from artifact_delib.schemas import PipelineResult

logger = logging.getLogger(__name__)


def write_config_snapshot(
    output_dir: Path,
    config_dict: dict,
) -> None:
    """Write a YAML snapshot of the experiment config."""
    import yaml

    path = output_dir / "config_snapshot.yaml"
    with path.open("w", encoding="utf-8") as f:
        yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    logger.debug("Config snapshot written to %s", path)


def write_predictions(
    output_dir: Path,
    results: list[PipelineResult],
    gold_labels: dict[str, dict[str, str | None]],
) -> None:
    """Write predictions.jsonl with sample-level predictions and gold labels."""
    path = output_dir / "predictions.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for r in results:
            gold = gold_labels.get(r.sample_id, {})
            record = {
                "sample_id": r.sample_id,
                "method": "",  # filled by caller
                "prediction": r.final_identification.content,
                "parsed_category": None,
                "parsed_type": None,
                "parsed_period": None,
                "parsed_material": None,
                "gold_category": gold.get("category"),
                "gold_type": gold.get("fine_grained_type"),
                "gold_period": gold.get("period"),
                "gold_material": gold.get("material"),
                "api_calls": r.total_api_calls,
                "total_tokens": r.total_usage.total_tokens,
                "status": r.status,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.debug("Predictions written to %s", path)


def write_raw_responses(
    output_dir: Path,
    results: list[PipelineResult],
) -> None:
    """Write raw_responses.jsonl with full pipeline provenance."""
    path = output_dir / "raw_responses.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for r in results:
            record = {
                "sample_id": r.sample_id,
                "final_identification": r.final_identification.content,
                "visual_perception": r.visual_perception_report.content,
                "expert_reports": [
                    {"name": e.expert_name, "content": e.content}
                    for e in r.expert_reports
                ],
                "summarized_report": r.summarized_report.content,
                "initial_candidates": [
                    {"text": c.text, "confidence": c.confidence}
                    for c in r.initial_candidates.candidates
                ],
                "route_decisions": [
                    {"action": rd.action, "reason": rd.reason}
                    for rd in r.route_decisions
                ],
                "recheck_records": [
                    {
                        "round": rc.round_no,
                        "expert": rc.expert_name,
                        "context": rc.context_query,
                        "new_content": rc.new_content,
                    }
                    for rc in r.recheck_records
                ],
                "deliberation_rounds": (
                    len(r.deliberation_result.rounds)
                    if r.deliberation_result
                    else 0
                ),
                "status": r.status,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.debug("Raw responses written to %s", path)


def write_api_calls_log(
    output_dir: Path,
    results: list[PipelineResult],
    method_name: str,
) -> None:
    """Write api_calls.jsonl with per-call accounting details."""
    path = output_dir / "api_calls.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for r in results:
            # Extract accounting from call_records if available
            call_records = getattr(r, "_call_records", None) or []
            if call_records:
                for rec in call_records:
                    record = {
                        "sample_id": r.sample_id,
                        "method": method_name,
                        "agent": rec.get("agent", "unknown"),
                        "input_tokens": rec.get("input_tokens", 0),
                        "output_tokens": rec.get("output_tokens", 0),
                        "latency_ms": rec.get("latency_ms", None),
                        "cost_usd": rec.get("cost_usd", 0.0),
                        "cache_hit": rec.get("cache_hit", False),
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            else:
                # Fallback: aggregate totals
                record = {
                    "sample_id": r.sample_id,
                    "method": method_name,
                    "agent": "aggregate",
                    "input_tokens": r.total_usage.input_tokens,
                    "output_tokens": r.total_usage.output_tokens,
                    "latency_ms": r.total_usage.total_latency_ms,
                    "cost_usd": None,
                    "cache_hit": None,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.debug("API calls log written to %s", path)


def write_metrics(
    output_dir: Path,
    metrics_dict: dict,
) -> None:
    """Write metrics.json with aggregate evaluation metrics."""
    path = output_dir / "metrics.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2, ensure_ascii=False)
    logger.debug("Metrics written to %s", path)


def write_routing_stats(
    output_dir: Path,
    results: list[PipelineResult],
) -> None:
    """Write routing_stats.json with route distribution."""
    from collections import Counter

    route_counts: Counter[str] = Counter()
    recheck_counts: list[int] = []
    deliberation_counts: list[int] = []

    for r in results:
        for rd in r.route_decisions:
            route_counts[rd.action] += 1
        recheck_counts.append(len(r.recheck_records))
        deliberation_counts.append(
            len(r.deliberation_result.rounds) if r.deliberation_result else 0
        )

    total = len(results) if results else 1
    stats = {
        "total_samples": len(results) if results else 0,
        "route_distribution": dict(route_counts),
        "route_percentages": {
            k: round(v / total * 100, 1) for k, v in route_counts.items()
        },
        "recheck_stats": {
            "total": sum(recheck_counts),
            "mean": sum(recheck_counts) / total if recheck_counts else 0,
            "samples_with_recheck": sum(1 for c in recheck_counts if c > 0),
        },
        "deliberation_stats": {
            "total_rounds": sum(deliberation_counts),
            "mean": sum(deliberation_counts) / total if deliberation_counts else 0,
            "samples_with_deliberation": sum(1 for c in deliberation_counts if c > 0),
        },
    }

    path = output_dir / "routing_stats.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    logger.debug("Routing stats written to %s", path)


def write_failures(
    output_dir: Path,
    results: list[PipelineResult],
) -> None:
    """Write failures.jsonl with non-COMPLETED results."""
    path = output_dir / "failures.jsonl"
    failures = [r for r in results if r.status != "COMPLETED"]
    if not failures:
        path.write_text("[]\n", encoding="utf-8")
        return

    with path.open("w", encoding="utf-8") as f:
        for r in failures:
            record = {
                "sample_id": r.sample_id,
                "status": r.status,
                "final_identification": r.final_identification.content,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.debug("Failures written to %s", path)


def write_experiment_summary(
    output_dir: Path,
    experiment_name: str,
    method_name: str,
    config: dict,
    metrics: dict,
    n_samples: int,
    elapsed_seconds: float,
    leakage_summary: str | None = None,
) -> None:
    """Write an experiment_summary.md with human-readable results."""
    path = output_dir / "experiment_summary.md"

    lines = [
        f"# Experiment Summary: {experiment_name}",
        f"",
        f"- **Method:** {method_name}",
        f"- **Samples:** {n_samples}",
        f"- **Duration:** {elapsed_seconds:.1f}s ({elapsed_seconds / 60:.1f} min)",
        f"- **Timestamp:** {datetime.now().isoformat()}",
        f"",
    ]

    if leakage_summary:
        lines.append(f"- **Leakage:** {leakage_summary}")
        lines.append("")

    if metrics:
        lines.append("## Metrics")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        for key, value in sorted(metrics.items()):
            if isinstance(value, float):
                lines.append(f"| {key} | {value:.4f} |")
            elif value is None:
                lines.append(f"| {key} | N/A |")
            else:
                lines.append(f"| {key} | {value} |")
        lines.append("")

    lines.append("## Configuration")
    lines.append("")
    lines.append("```yaml")
    import yaml
    lines.append(yaml.dump(config, default_flow_style=False, allow_unicode=True))
    lines.append("```")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    logger.debug("Experiment summary written to %s", path)


def write_all_results(
    output_dir: Path,
    experiment_name: str,
    method_name: str,
    config_dict: dict,
    results: list[PipelineResult],
    gold_labels: dict[str, dict[str, str | None]],
    metrics: dict,
    elapsed_seconds: float,
    leakage_summary: str | None = None,
) -> None:
    """Write all experiment output files.

    Args:
        output_dir: Root output directory for the experiment.
        experiment_name: Name of the experiment.
        method_name: Name of the method that was run.
        config_dict: Experiment config as a dict.
        results: Pipeline results for all samples.
        gold_labels: Gold labels keyed by sample_id.
        metrics: Aggregate metrics dict.
        elapsed_seconds: Wall-clock time for the experiment.
        leakage_summary: Optional leakage check summary string.
    """
    write_config_snapshot(output_dir, config_dict)
    write_predictions(output_dir, results, gold_labels)
    write_raw_responses(output_dir, results)
    write_api_calls_log(output_dir, results, method_name)
    write_metrics(output_dir, metrics)
    write_routing_stats(output_dir, results)
    write_failures(output_dir, results)
    write_experiment_summary(
        output_dir=output_dir,
        experiment_name=experiment_name,
        method_name=method_name,
        config=config_dict,
        metrics=metrics,
        n_samples=len(results),
        elapsed_seconds=elapsed_seconds,
        leakage_summary=leakage_summary,
    )
    print(f"\n  Results written to: {output_dir}")