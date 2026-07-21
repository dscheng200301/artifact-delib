#!/usr/bin/env python3
"""Generate result table skeletons for the ArtifactDelib paper.

Table 1: External Baselines
Table 2: Core Ablations
Table 3: Internal Comparison

These scripts emit table HEADERS only — NO fabricated numbers.
Real results must be written by running actual experiments.
"""

import csv
from pathlib import Path

TABLE_DIR = Path(__file__).resolve().parent.parent / "results" / "tables"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def _ensure_dirs() -> None:
    """Create the results directory structure."""
    dirs = [
        "external_baselines/raw",
        "external_baselines/predictions",
        "external_baselines/metrics",
        "external_baselines/logs",
        "core_ablations/raw",
        "core_ablations/predictions",
        "core_ablations/metrics",
        "core_ablations/logs",
        "tables",
        "figures",
    ]
    for d in dirs:
        (RESULTS_DIR / d).mkdir(parents=True, exist_ok=True)


# ═════════════════════════════════════════════════════
#  Table 1: External Baselines
# ═════════════════════════════════════════════════════

TABLE1_HEADER = [
    "Method",
    "Category Acc.",
    "Type Acc.",
    "Period Acc.",
    "Material Acc.",
    "Joint Acc.",
    "Macro-F1",
    "API Calls",
    "Total Tokens",
    "Latency (s)",
    "Cost (est.)",
]

TABLE1_METHODS = [
    "clip_zero_shot",
    "dinov2_knn",
    "blip2_zero_shot",
    "direct_single_vlm",
    "self_consistency_vlm",
    "multi_agent_debate",
    "artifact_delib_rule",
    "artifact_delib_mlp",
]


def generate_table1(output_path: Path | None = None) -> list[str]:
    """Generate Table 1 skeleton (external baselines)."""
    output_path = output_path or TABLE_DIR / "external_baselines.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [TABLE1_HEADER] + [[m] + [""] * (len(TABLE1_HEADER) - 1) for m in TABLE1_METHODS]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"Table 1 written to {output_path}")
    print(f"  Rows: {len(TABLE1_METHODS)} methods")
    print(f"  NOTE: All values are empty. Run experiments to fill them.")
    return rows


# ═════════════════════════════════════════════════════
#  Table 2: Core Ablations
# ═════════════════════════════════════════════════════

TABLE2_HEADER = [
    "Variant",
    "Category Acc.",
    "Type Acc.",
    "Period Acc.",
    "Joint Acc.",
    "Macro-F1",
    "API Calls",
    "Tokens",
    "Latency (s)",
    "Delta vs Full",
]

TABLE2_VARIANTS = [
    "Full ArtifactDelib",
    "w/o Expert Specialization (A1)",
    "w/o Disagreement Analysis (A2)",
    "w/o Dynamic Routing (A3)",
    "Random Recheck (A4)",
    "w/o Controlled Deliberation (A5)",
    "Free Debate (A6)",
    "w/o Critic (A7)",
    "Early Judge (A8)",
]


def generate_table2(output_path: Path | None = None) -> list[str]:
    """Generate Table 2 skeleton (core ablations)."""
    output_path = output_path or TABLE_DIR / "core_ablations.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [TABLE2_HEADER] + [[m] + [""] * (len(TABLE2_HEADER) - 1) for m in TABLE2_VARIANTS]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"Table 2 written to {output_path}")
    print(f"  Rows: {len(TABLE2_VARIANTS)} variants")
    print(f"  NOTE: All values are empty. Run experiments to fill them.")
    return rows


# ═════════════════════════════════════════════════════
#  Table 3: Internal Comparison
# ═════════════════════════════════════════════════════

TABLE3_HEADER = [
    "Method",
    "Category Acc.",
    "Type Acc.",
    "Period Acc.",
    "Material Acc.",
    "Joint Acc.",
    "Macro-F1",
    "API Calls",
    "Total Tokens",
    "Latency (s)",
    "Cost (est.)",
]

TABLE3_METHODS = [
    "FixedMultiExpert",
    "FixedFull",
    "ArtifactDelib-Rule",
    "ArtifactDelib-MLP",
    "OracleRouter",
]


def generate_table3(output_path: Path | None = None) -> list[str]:
    """Generate Table 3 skeleton (internal comparison)."""
    output_path = output_path or TABLE_DIR / "internal_comparison.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [TABLE3_HEADER] + [[m] + [""] * (len(TABLE3_HEADER) - 1) for m in TABLE3_METHODS]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"Table 3 written to {output_path}")
    print(f"  Rows: {len(TABLE3_METHODS)} methods (internal comparison)")
    print(f"  NOTE: All values are empty. Run experiments to fill them.")
    return rows


# ═════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("ArtifactDelib Result Table Generator")
    print("=" * 60)
    print()
    print("WARNING: This script generates table HEADERS only.")
    print("All data cells are EMPTY — no fabricated results.")
    print("Fill them by running real experiments.")
    print()
    _ensure_dirs()
    print(f"Results directory: {RESULTS_DIR}")
    print()
    generate_table1()
    print()
    generate_table2()
    print()
    generate_table3()
    print()
    print("Done. Deferred baselines: AutoGen and CAMEL")
    print("are intentionally not implemented in this phase.")
