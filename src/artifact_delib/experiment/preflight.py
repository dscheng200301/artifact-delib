"""Preflight checks — validate dataset, config, and environment before running."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from artifact_delib.data.importer import ArtifactDatasetImporter
from artifact_delib.data.leakage_detector import LeakageDetector
from artifact_delib.experiment.config import ExperimentConfig, LeakageConfig
from artifact_delib.schemas import ArtifactSample

logger = logging.getLogger(__name__)


class PreflightReport:
    """Result of preflight validation."""

    def __init__(self) -> None:
        self.checks: list[str] = []
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def add_ok(self, msg: str) -> None:
        self.checks.append(f"  OK  {msg}")

    def add_error(self, msg: str) -> None:
        self.errors.append(f"  ERR {msg}")
        self.checks.append(f"  ERR {msg}")

    def add_warning(self, msg: str) -> None:
        self.warnings.append(f"  WARN {msg}")
        self.checks.append(f"  WARN {msg}")

    @property
    def passed(self) -> bool:
        return len(self.errors) == 0

    def print(self) -> None:
        print("\n" + "=" * 56)
        print("  Preflight Report")
        print("=" * 56)
        for c in self.checks:
            print(c)
        if self.warnings:
            print(f"\n  Warnings: {len(self.warnings)}")
        if self.errors:
            print(f"\n  Errors: {len(self.errors)}")
        print("=" * 56)


def run_preflight(
    config: ExperimentConfig,
    verbose: bool = False,
) -> tuple[PreflightReport, list[ArtifactSample] | None]:
    """Run all preflight checks for an experiment config.

    Args:
        config: Validated experiment configuration.
        verbose: If True, print detailed output.

    Returns:
        Tuple of (report, samples_or_None). samples is None if preflight fails.
    """
    report = PreflightReport()

    # ── 1. Config syntax ──
    report.add_ok("CONFIG_OK")

    # ── 2. Manifest file ──
    manifest_path = Path(config.dataset.manifest)
    if not manifest_path.exists():
        report.add_error(
            f"DATASET_MANIFEST_NOT_FOUND — {manifest_path}"
        )
    elif not manifest_path.is_file():
        report.add_error(
            f"DATASET_MANIFEST_NOT_A_FILE — {manifest_path}"
        )
    else:
        report.add_ok(f"DATASET_MANIFEST_OK — {manifest_path}")

    # ── 3. Image root ──
    image_root = Path(config.dataset.image_root)
    if not image_root.exists():
        report.add_warning(
            f"IMAGE_ROOT_NOT_FOUND — {image_root} (will be created if needed)"
        )
    elif not image_root.is_dir():
        report.add_error(
            f"IMAGE_ROOT_NOT_A_DIR — {image_root}"
        )
    else:
        report.add_ok(f"IMAGE_ROOT_OK — {image_root}")

    # ── 4. Split file ──
    if config.dataset.split_file:
        split_path = Path(config.dataset.split_file)
        if not split_path.exists():
            report.add_warning(
                f"SPLIT_FILE_NOT_FOUND — {split_path} (will use all samples)"
            )
        else:
            report.add_ok(f"SPLIT_FILE_OK — {split_path}")

    # ── 5. Method names ──
    try:
        from artifact_delib.baselines.registry import get_baseline, list_baselines

        valid_methods = set(list_baselines().keys())
        for method_name in config.methods:
            if method_name not in valid_methods:
                report.add_error(
                    f"UNKNOWN_METHOD — {method_name!r}. "
                    f"Valid: {sorted(valid_methods)}"
                )
            else:
                meta = list_baselines().get(method_name, {})
                flags = []
                if meta.get("requires_fit"):
                    flags.append("requires_fit")
                if meta.get("requires_download"):
                    flags.append("requires_download")
                flag_str = f" [{', '.join(flags)}]" if flags else ""
                report.add_ok(f"METHOD_OK — {method_name}{flag_str}")
    except ImportError as e:
        report.add_warning(f"METHOD_CHECK_SKIPPED — {e}")

    # ── 6. Remote calls check ──
    if config.execution.allow_remote_calls:
        # Check for API keys
        import os

        api_keys = [
            "DASHSCOPE_API_KEY",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "QWEN_API_KEY",
        ]
        found_keys = [k for k in api_keys if os.environ.get(k)]
        if found_keys:
            report.add_ok(
                f"REMOTE_CALLS_ENABLED — keys found: {', '.join(found_keys)}"
            )
        else:
            report.add_warning(
                "REMOTE_CALLS_ENABLED — NO API KEYS FOUND "
                "(set DASHSCOPE_API_KEY, OPENAI_API_KEY, etc.)"
            )
    else:
        report.add_ok("REMOTE_CALLS_DISABLED")

    # ── 7. Model download ──
    if config.execution.allow_model_download:
        report.add_ok("MODEL_DOWNLOAD_ENABLED")
    else:
        report.add_ok("MODEL_DOWNLOAD_DISABLED")

    # ── 8. Output directory ──
    output_dir = Path(config.output.dir)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        report.add_ok(f"OUTPUT_DIR_OK — {output_dir}")
    except OSError as e:
        report.add_error(f"OUTPUT_DIR_CREATE_FAILED — {e}")

    # ── 9. Optional dependencies ──
    vision_methods = {"clip_zero_shot", "dinov2_knn", "blip2_zero_shot"}
    requested_vision = set(config.methods) & vision_methods
    if requested_vision:
        try:
            import torch  # noqa: F401
            report.add_ok("VISION_DEPS_OK — torch available")
        except ImportError:
            report.add_error(
                f"VISION_DEPS_MISSING — methods {requested_vision} require torch. "
                "Install: pip install artifact-delib[vision-baselines]"
            )

    # ── 10. Deliberation-only methods ──
    if "artifact_delib_mlp" in config.methods:
        report.add_warning(
            "ARTIFACT_DELIB_MLP — requires trained MLPRouter weights. "
            "Use artifact_delib_rule for RuleRouter fallback."
        )

    # ── Print summary ──
    if verbose:
        report.print()

    # ── Load samples if manifest exists ──
    samples = None
    if report.passed and manifest_path.exists():
        try:
            importer = ArtifactDatasetImporter(image_root=image_root)
            samples = importer.import_manifest(manifest_path)
            if config.dataset.split_file:
                split_path = Path(config.dataset.split_file)
                if split_path.exists():
                    split_ids = set(
                        line.strip()
                        for line in split_path.read_text(encoding="utf-8").splitlines()
                        if line.strip()
                    )
                    samples = [s for s in samples if s.sample_id in split_ids]
                    report.add_ok(
                        f"SPLIT_APPLIED — {len(samples)} samples in split"
                    )

            if config.dataset.max_samples:
                import random
                rng = random.Random(config.dataset.seed)
                rng.shuffle(samples)
                samples = samples[: config.dataset.max_samples]
                report.add_ok(
                    f"MAX_SAMPLES — limited to {len(samples)}"
                )

            report.add_ok(f"MANIFEST_LOADED — {len(samples)} samples")
        except Exception as e:
            report.add_error(f"MANIFEST_LOAD_FAILED — {e}")

    return report, samples


def run_leakage_preflight(
    samples: list[ArtifactSample],
    leakage_config: LeakageConfig,
) -> PreflightReport:
    """Run leakage detection checks and enforce configured policies.

    Args:
        samples: Loaded dataset samples.
        leakage_config: Leakage detection configuration.

    Returns:
        PreflightReport with leakage check results.
    """
    report = PreflightReport()

    if not leakage_config.run_before_experiment:
        report.add_ok("LEAKAGE_CHECK_SKIPPED")
        return report

    try:
        detector = LeakageDetector()
        leakage_report = detector.run_all(samples)

        # ── Exact duplicates ──
        if leakage_report.exact_duplicates:
            msg = f"{len(leakage_report.exact_duplicates)} exact duplicate group(s)"
            if leakage_config.fail_on_exact_duplicate:
                report.add_error(f"EXACT_DUPLICATE — {msg}")
            else:
                report.add_warning(f"EXACT_DUPLICATE — {msg}")
        else:
            report.add_ok("EXACT_DUPLICATE — none found")

        # ── Near duplicates ──
        if leakage_report.near_duplicates:
            msg = f"{len(leakage_report.near_duplicates)} near-duplicate pair(s)"
            if leakage_config.warn_on_near_duplicate:
                report.add_warning(f"NEAR_DUPLICATE — {msg}")
            else:
                report.add_ok(f"NEAR_DUPLICATE — {msg}")
        else:
            report.add_ok("NEAR_DUPLICATE — none found")

        # ── Filename leakage ──
        if leakage_report.filename_leaks:
            msg = f"{len(leakage_report.filename_leaks)} filename leakage(s)"
            if leakage_config.warn_on_filename_leakage:
                report.add_warning(f"FILENAME_LEAKAGE — {msg}")
            else:
                report.add_ok(f"FILENAME_LEAKAGE — {msg}")
        else:
            report.add_ok("FILENAME_LEAKAGE — none found")

        # ── Corrupt images ──
        if leakage_report.corrupt_images:
            report.add_error(
                f"CORRUPT_IMAGE — {len(leakage_report.corrupt_images)} corrupt image(s)"
            )
        else:
            report.add_ok("CORRUPT_IMAGE — none found")

        # ── Object/group overlap ──
        overlap = _detect_object_overlap(samples)
        if overlap:
            msg = f"{len(overlap)} sample(s) with object-group split leakage"
            if leakage_config.fail_on_object_overlap:
                report.add_error(f"OBJECT_OVERLAP — {msg}")
            else:
                report.add_warning(f"OBJECT_OVERLAP — {msg}")
        else:
            report.add_ok("OBJECT_OVERLAP — none found")

        # ── Label coverage ──
        if leakage_report.label_coverage is not None:
            if leakage_report.label_coverage.is_clean:
                report.add_ok(
                    f"LABEL_COVERAGE — all {leakage_report.label_coverage.test_label_count} "
                    "test labels seen in train"
                )
            else:
                report.add_warning(
                    f"LABEL_COVERAGE — {len(leakage_report.label_coverage.missing_test_labels)} "
                    f"test label(s) never seen in train: "
                    f"{leakage_report.label_coverage.missing_test_labels}"
                )

        report.add_ok(f"LEAKAGE_SUMMARY — {leakage_report.summary}")

    except Exception as e:
        report.add_error(f"LEAKAGE_CHECK_FAILED — {e}")

    return report


def _detect_object_overlap(samples: list[ArtifactSample]) -> list[str]:
    """Detect samples where the same artifact_group_id appears in multiple splits.

    Returns a list of sample_ids that are in overlapping groups.
    """
    from collections import defaultdict

    group_splits: dict[str, set[str]] = defaultdict(set)
    group_samples: dict[str, list[str]] = defaultdict(list)

    for s in samples:
        gid = s.artifact_group_id
        if gid:
            group_splits[gid].add(s.split or "unassigned")
            group_samples[gid].append(s.sample_id)

    overlapping: list[str] = []
    for gid, splits in group_splits.items():
        if len(splits) > 1:
            overlapping.extend(group_samples[gid])

    return overlapping