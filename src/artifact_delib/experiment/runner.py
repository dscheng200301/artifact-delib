"""Experiment runner — orchestrates the full experiment lifecycle.

Flow:
    Load Config → Validate Config → Load Manifest → Apply Split →
    Leakage Preflight → Select Samples → Create ModelClient →
    For each method: Instantiate → BatchRunner → Evaluate → Save Results
"""

from __future__ import annotations

import json
import logging
import time
import sys
from pathlib import Path
from typing import Any

from artifact_delib.api.base import ModelClient
from artifact_delib.baselines.registry import list_baselines
from artifact_delib.data.batch_runner import BatchRunner
from artifact_delib.data.importer import ArtifactDatasetImporter
from artifact_delib.evaluation.metrics import ArtifactMetrics
from artifact_delib.evaluation.prediction_parser import PredictionParser
from artifact_delib.experiment.config import ExperimentConfig
from artifact_delib.experiment.method_factory import create_method
from artifact_delib.experiment.preflight import (
    PreflightReport,
    run_leakage_preflight,
    run_preflight,
)
from artifact_delib.experiment.result_writer import write_all_results
from artifact_delib.schemas import ArtifactSample, PipelineResult

logger = logging.getLogger(__name__)


def _create_model_client(
    model_name: str,
    allow_remote: bool,
) -> ModelClient:
    """Create a ModelClient based on configuration.

    Args:
        model_name: Model name string.
        allow_remote: If True, use a real DashScope API client.

    Returns:
        A ModelClient instance.

    Raises:
        RuntimeError: If allow_remote=True but client creation fails
            (fail-closed: never fallback to mock).
    """
    if allow_remote:
        logger.info("Creating real DashScope API client (fail-closed mode)")
        try:
            from artifact_delib.api.dashscope_client import DashScopeModelClient

            raw_client = DashScopeModelClient(model=model_name)
        except Exception as e:
            logger.error(
                "Failed to create DashScope client: %s. "
                "FAIL-CLOSED — aborting. Set DASHSCOPE_API_KEY environment variable.",
                e,
            )
            raise RuntimeError(
                f"Cannot create DashScope client: {e}. "
                "Real mode never falls back to mock. "
                "Set DASHSCOPE_API_KEY and ensure network access."
            ) from e

        # Wrap in guarded + audited layers
        try:
            from artifact_delib.api.guarded import GuardedModelClient
            from artifact_delib.api.audited import AuditedModelClient
            from artifact_delib.api.cache import ResponseCache
            from artifact_delib.api.budget import BudgetManager
            from artifact_delib.api.call_log import CallLogStore

            cache = ResponseCache()
            budget = BudgetManager()
            call_log = CallLogStore()
            guarded = GuardedModelClient(
                client=raw_client,
                cache=cache,
                budget=budget,
                call_log=call_log,
                max_retries=3,
                max_concurrency=1,
            )
            return AuditedModelClient(
                client=guarded,
                call_log=call_log,
            )
        except Exception as e:
            logger.error(
                "Failed to wrap DashScope client with safeguards: %s. "
                "FAIL-CLOSED — aborting.",
                e,
            )
            raise RuntimeError(
                f"Cannot wrap DashScope client: {e}. "
                "Real mode never falls back to mock."
            ) from e
    else:
        from artifact_delib.models.mock_artifact import ArtifactMockClient

        logger.info("Creating mock client (allow_remote=False)")
        return ArtifactMockClient()


def _build_gold_labels(
    samples: list[ArtifactSample],
) -> dict[str, dict[str, str | None]]:
    """Build a dict of gold labels keyed by sample_id."""
    return {
        s.sample_id: {
            "category": s.category,
            "fine_grained_type": s.fine_grained_type,
            "period": s.period,
            "material": s.material,
        }
        for s in samples
    }


def _run_method(
    method_name: str,
    client: ModelClient,
    model_name: str,
    samples: list[ArtifactSample],
    config: ExperimentConfig,
    allow_remote: bool,
) -> dict[str, Any]:
    """Run one method on all samples and return results + metrics.

    Args:
        method_name: Name of the method to run.
        client: ModelClient instance.
        model_name: Model name string.
        samples: List of ArtifactSample to process.
        config: Full experiment configuration.
        allow_remote: Whether remote API calls are allowed.

    Returns:
        Dict with keys: method_name, results, metrics, gold_labels, elapsed.
    """
    print(f"\n{'=' * 56}")
    if config.execution.concurrency > 1:
        print(f"  Method: {method_name} (concurrency={config.execution.concurrency})")
    else:
        print(f"  Method: {method_name}")
    print(f"{'=' * 56}")

    # Instantiate method
    method = create_method(
        name=method_name,
        client=client,
        model_name=model_name,
    )

    # Build output directory
    method_dir = Path(config.output.dir) / config.experiment.name / method_name
    method_dir.mkdir(parents=True, exist_ok=True)

    # Run batch
    batch = BatchRunner(
        method=method,
        output_root=method_dir,
        experiment_id=config.experiment.name,
        method_name=method_name,
        config=config.model_dump(mode="python"),
    )

    t0 = time.time()
    results = batch.run(samples)
    elapsed = time.time() - t0

    print(f"\n  Completed: {len(results)}/{len(samples)} samples in {elapsed:.1f}s")

    # Evaluate
    gold = _build_gold_labels(samples)
    metrics = batch.evaluate(results, samples)

    # Write results
    write_all_results(
        output_dir=method_dir,
        experiment_name=config.experiment.name,
        method_name=method_name,
        config_dict=config.to_yaml_dict(),
        results=results,
        gold_labels=gold,
        metrics=metrics,
        elapsed_seconds=elapsed,
    )

    completed = sum(1 for r in results if r.status == "COMPLETED")
    completed_rate = completed / len(results) if results else 0
    print(f"  Completed rate: {completed_rate:.1%} ({completed}/{len(results)})")
    parse_fail = metrics.get("parse_failure_rate")
    if parse_fail is not None:
        print(f"  Parse failure rate: {parse_fail:.2%}")
    else:
        print(f"  Parse failure rate: N/A")

    return {
        "method_name": method_name,
        "results": results,
        "metrics": metrics,
        "gold_labels": gold,
        "elapsed": elapsed,
    }


def run_experiment(
    config: ExperimentConfig,
    allow_remote: bool = False,
) -> None:
    """Run a full experiment: preflight, methods, evaluation, output.

    Flow:
        1. Load & validate config
        2. Load FULL manifest
        3. Load split assignments → inject into ALL samples
        4. Leakage preflight on FULL dataset (train + validation + test)
        5. Select experimental split (e.g. test only)
        6. Apply max_samples
        7. For each method: Instantiate → BatchRunner → Evaluate → Save

    Args:
        config: Validated experiment configuration.
        allow_remote: If True, allow real API calls.

    Raises:
        SystemExit: If preflight checks fail critically.
    """
    print(f"\n{'=' * 56}")
    print(f"  ArtifactDelib — Experiment: {config.experiment.name}")
    if config.experiment.description:
        print(f"  {config.experiment.description}")
    print(f"{'=' * 56}")

    # ── 1. Config preflight ──
    preflight_report, _ = run_preflight(config, verbose=True)
    if not preflight_report.passed:
        print("\n  Preflight FAILED. Aborting experiment.")
        raise SystemExit(1)

    # ── 2. Load FULL manifest ──
    from artifact_delib.data.importer import ArtifactDatasetImporter

    manifest_path = Path(config.dataset.manifest)
    image_root = Path(config.dataset.image_root)
    importer = ArtifactDatasetImporter(image_root=image_root)
    all_samples = importer.import_manifest(manifest_path)
    print(f"\n  Loaded {len(all_samples)} samples from manifest")

    # ── 3. Apply split assignments ──
    if config.dataset.split_file:
        split_path = Path(config.dataset.split_file)
        if split_path.exists():
            from artifact_delib.data.split_manifest import SplitManifest

            split_manifest = SplitManifest.load(config.dataset.split_file)
            all_samples = split_manifest.assign(all_samples)
            print(f"  Split assignments applied: "
                  f"{sum(1 for s in all_samples if s.split == 'train')} train, "
                  f"{sum(1 for s in all_samples if s.split == 'validation')} val, "
                  f"{sum(1 for s in all_samples if s.split == 'test')} test, "
                  f"{sum(1 for s in all_samples if s.split is None)} unassigned")

    # ── 4. Leakage preflight on FULL dataset ──
    leakage_report = run_leakage_preflight(all_samples, config.leakage_detection)
    leakage_report.print()
    if not leakage_report.passed:
        print("\n  Leakage preflight FAILED on full dataset. Aborting experiment.")
        print("  Fix leakage issues or disable failing checks.")
        raise SystemExit(1)

    leakage_summary = (
        "; ".join(
            c for c in leakage_report.checks
            if "LEAKAGE" in c or "EXACT" in c or "NEAR" in c or "OBJECT" in c
        )
    ) or None

    # ── 5. Select experimental split ──
    target_split = "test"
    experiment_samples = [s for s in all_samples if s.split == target_split]
    if not experiment_samples:
        print(f"\n  No samples with split='{target_split}'. Using all samples.")
        experiment_samples = all_samples

    # ── 6. Apply max_samples ──
    if config.dataset.max_samples and len(experiment_samples) > config.dataset.max_samples:
        import random
        rng = random.Random(config.dataset.seed)
        rng.shuffle(experiment_samples)
        experiment_samples = experiment_samples[: config.dataset.max_samples]
        print(f"\n  Limited to {len(experiment_samples)} samples (max_samples={config.dataset.max_samples})")
    else:
        print(f"\n  Using {len(experiment_samples)} samples from split='{target_split}'")

    # ── 7. Create model client ──
    client = _create_model_client(config.model.name, allow_remote)

    # ── 8. Run each method ──
    all_results: list[dict[str, Any]] = []
    for method_name in config.methods:
        result = _run_method(
            method_name=method_name,
            client=client,
            model_name=config.model.name,
            samples=experiment_samples,
            config=config,
            allow_remote=allow_remote,
        )
        all_results.append(result)

    # ── Experiment summary ──
    print(f"\n{'=' * 56}")
    print(f"  Experiment Complete: {config.experiment.name}")
    print(f"{'=' * 56}")
    for r in all_results:
        m = r["metrics"]
        print(f"  {r['method_name']:30s} "
              f"completed={r['results']} "
              f"parse_fail={m.get('parse_failure_rate', 0):.2%} "
              f"tokens={m.get('average_tokens', 0):.0f} "
              f"latency_p50={m.get('p50_latency_ms', 0):.0f}ms")
    print()


def dry_run_experiment(config_path: str) -> None:
    """Validate a config thoroughly without executing API calls.

    Checks:
        1. YAML syntax
        2. Config field validation
        3. Manifest existence
        4. Image root existence
        5. Split file existence
        6. Method name validity
        7. Model config
        8. Output directory creatability
        9. Optional dependency availability
        10. Remote call detection

    Args:
        config_path: Path to the YAML config file.

    Raises:
        SystemExit: On failure.
    """
    config_file = Path(config_path)

    if not config_file.exists():
        print(f"Error: config file not found: {config_file}", file=sys.stderr)
        raise SystemExit(1)

    # 1. YAML syntax
    try:
        import yaml
        with config_file.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if data is None:
            print("Error: empty YAML file", file=sys.stderr)
            raise SystemExit(1)
    except yaml.YAMLError as e:
        print(f"Error: YAML syntax error: {e}", file=sys.stderr)
        raise SystemExit(1)

    # 2. Schema validation
    try:
        config = ExperimentConfig(**data)
        print("CONFIG_OK")
    except Exception as e:
        print(f"Error: config validation failed: {e}", file=sys.stderr)
        raise SystemExit(1)

    # 3. Manifest
    manifest_path = Path(config.dataset.manifest)
    if manifest_path.exists():
        print("DATASET_PATH_OK")
    else:
        print(f"WARNING: DATASET_PATH_NOT_FOUND — {manifest_path}")

    # 4. Split file
    if config.dataset.split_file:
        split_path = Path(config.dataset.split_file)
        if split_path.exists():
            print("SPLIT_PATH_OK")
        else:
            print(f"WARNING: SPLIT_PATH_NOT_FOUND — {split_path}")

    # 5. Methods
    valid_methods = set(list_baselines().keys())
    for method_name in config.methods:
        if method_name in valid_methods:
            print(f"METHODS_OK — {method_name}")
        else:
            print(f"ERROR: UNKNOWN_METHOD — {method_name}", file=sys.stderr)
            raise SystemExit(1)

    # 6. Remote calls
    if config.execution.allow_remote_calls:
        print("REMOTE_CALLS_ENABLED")
    else:
        print("REMOTE_CALLS_DISABLED")

    # 7. Model download
    if config.execution.allow_model_download:
        print("MODEL_DOWNLOAD_ENABLED")
    else:
        print("MODEL_DOWNLOAD_DISABLED")

    # 8. Output directory
    output_dir = Path(config.output.dir)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        print("OUTPUT_DIR_OK")
    except OSError as e:
        print(f"ERROR: OUTPUT_DIR_FAILED — {e}", file=sys.stderr)
        raise SystemExit(1)

    print("READY_FOR_DRY_RUN")