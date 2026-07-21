"""CLI entry point for ArtifactDelib experiments.

Usage:
    artifact-delib run configs/experiments/pilot_smoke.yaml
    artifact-delib run configs/experiments/pilot_smoke.yaml --dry-run
    artifact-delib run configs/experiments/pilot.yaml --allow-remote
    artifact-delib list
    artifact-delib help
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("artifact_delib.cli")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ArtifactDelib: Dynamic Multi-Expert Deliberation for Artifact Identification",
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="help",
        choices=["run", "list", "help"],
        help="Command: run (experiment), list (baselines), help",
    )
    parser.add_argument(
        "config",
        nargs="?",
        default=None,
        help="Path to experiment YAML config file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Validate config without executing API calls",
    )
    parser.add_argument(
        "--allow-remote",
        action="store_true",
        default=False,
        help="Override allow_remote_calls to true",
    )

    args = parser.parse_args()

    if args.command == "help":
        parser.print_help()
        sys.exit(0)

    if args.command == "list":
        _list_baselines()
        sys.exit(0)

    if args.command == "run":
        if args.config is None:
            print("Error: config file required for 'run' command", file=sys.stderr)
            sys.exit(1)
        _run_experiment(args.config, dry_run=args.dry_run, allow_remote=args.allow_remote)
        sys.exit(0)


def _list_baselines() -> None:
    """List all registered baselines by category."""
    try:
        from artifact_delib.baselines.registry import list_baselines, CAT_EXTERNAL, CAT_OURS, CAT_LEGACY

        print("\nArtifactDelib — Registered Baselines")
        print("=" * 50)

        for cat_name, cat_label in [
            (CAT_EXTERNAL, "External Baselines"),
            (CAT_OURS, "Ours"),
            (CAT_LEGACY, "Legacy / Internal"),
        ]:
            baselines = list_baselines(cat_name)
            if not baselines:
                continue
            print(f"\n  {cat_label}:")
            for name, meta in sorted(baselines.items()):
                desc = meta.get("description", "")
                flags = []
                if meta.get("requires_fit"):
                    flags.append("fit")
                if meta.get("requires_download"):
                    flags.append("download")
                flag_str = f" [{', '.join(flags)}]" if flags else ""
                print(f"    {name:30s} {desc}{flag_str}")

        print()
    except ImportError as e:
        print(f"Error importing baselines: {e}", file=sys.stderr)


def _run_experiment(
    config_path: str,
    dry_run: bool = False,
    allow_remote: bool = False,
) -> None:
    """Run an experiment from a YAML config file.

    Args:
        config_path: Path to the YAML configuration file.
        dry_run: If True, validate config and environment without executing.
        allow_remote: If True, allow real API calls.
    """
    config_file = Path(config_path)
    if not config_file.exists():
        print(f"Error: config file not found: {config_file}", file=sys.stderr)
        sys.exit(1)

    # ── Dry-run mode: validate thoroughly without executing ──
    if dry_run:
        from artifact_delib.experiment.runner import dry_run_experiment

        dry_run_experiment(config_path)
        print()
        print("  To run with real API calls:")
        print(f"    artifact-delib run {config_path} --allow-remote")
        print()
        return

    # ── Load and validate config ──
    from artifact_delib.experiment.config import ExperimentConfig

    try:
        config = ExperimentConfig.from_yaml(config_file)
    except Exception as e:
        print(f"Error: config validation failed: {e}", file=sys.stderr)
        sys.exit(1)

    # ── Check remote call authorization ──
    effective_remote = allow_remote or config.execution.allow_remote_calls

    if not effective_remote:
        print("\n" + "=" * 56)
        print("  Remote calls not allowed.")
        print("=" * 56)
        print()
        print("  This experiment requires API calls but --allow-remote was not set")
        print("  and config.execution.allow_remote_calls is false.")
        print()
        print("  To proceed:")
        print(f"    artifact-delib run {config_path} --allow-remote")
        print()
        print("  WARNING: This will incur API costs.")
        print()
        sys.exit(1)

    # Confirm
    print("\n" + "=" * 56)
    print(f"  Running experiment: {config.experiment.name}")
    print(f"  Methods: {', '.join(config.methods)}")
    print(f"  Model: {config.model.name}")
    print(f"  Max samples: {config.dataset.max_samples or 'all'}")
    print("=" * 56)
    print()
    print("  WARNING: This will make real API calls and incur costs.")
    print("  Press Ctrl+C within 5 seconds to abort...")
    print()

    import time
    time.sleep(5)

    # ── Run ──
    from artifact_delib.experiment.runner import run_experiment

    run_experiment(config, allow_remote=effective_remote)


if __name__ == "__main__":
    main()