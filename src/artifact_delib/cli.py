"""CLI entry point for ArtifactDelib experiments.

Usage:
    artifact-delib run configs/experiments/pilot.yaml
    python -m artifact_delib.cli run configs/experiments/pilot.yaml --help
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

    This is a STUB implementation for dry-run validation.
    Full experiment execution requires explicit API authorization.
    """
    config_file = Path(config_path)
    if not config_file.exists():
        print(f"Error: config file not found: {config_file}", file=sys.stderr)
        sys.exit(1)

    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:
        print("Warning: PyYAML not installed, reading config manually")
        print()

    config_text = config_file.read_text(encoding="utf-8")
    print(f"\nArtifactDelib — Experiment Configuration")
    print("=" * 50)
    print(f"  Config: {config_file}")
    print(f"  Dry-run: {dry_run}")
    print(f"  Allow remote: {allow_remote}")

    if dry_run:
        print("\n  DRY RUN — no API calls will be made.")
        print("  Config validated successfully.")
        print()
        print("  To run with real API calls:")
        print(f"    artifact-delib run {config_path} --allow-remote")
        print()
        return

    if not allow_remote:
        print("\n  ERROR: Remote calls not allowed.", file=sys.stderr)
        print("  Pass --allow-remote to enable API calls.", file=sys.stderr)
        print()
        print("  WARNING: This will incur API costs.", file=sys.stderr)
        sys.exit(1)

    # Full experiment execution — requires explicit authorization
    print("\n  Full experiment execution not yet implemented.")
    print("  Use --dry-run for config validation.")
    print()


if __name__ == "__main__":
    main()