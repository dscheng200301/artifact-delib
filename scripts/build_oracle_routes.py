"""Build Oracle Route dataset from Met artifact dataset.

This script:
1. Runs the full pipeline (RuleRouter) on train/val samples
2. Extracts features and scores different route choices
3. Selects the optimal route for each sample
4. Saves training data for the Learned Router
"""

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

from artifact_delib.data.importer import ArtifactDatasetImporter
from artifact_delib.pipeline.artifact_delib_pipeline import ArtifactDelibPipeline
from artifact_delib.models.mock_artifact import ArtifactMockClient
from artifact_delib.router.oracle_route_builder import OracleRouteBuilder
from artifact_delib.router.learned_router import TrainingRecord, route_to_label
from artifact_delib.evaluation.prediction_parser import PredictionParser
from artifact_delib.evaluation.metrics import ArtifactMetrics


def main():
    root = Path("data/artifact")
    manifest = root / "met_artifact_manifest.csv"
    image_root = root / "images"
    output_dir = root / "oracle_routes"
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("BUILDING ORACLE ROUTE DATASET")
    print("=" * 60)

    # Step 1: Import dataset
    print("\n[1/4] Importing dataset...")
    importer = ArtifactDatasetImporter(image_root=image_root)
    all_samples = importer.import_manifest(manifest)
    print(f"  [OK] Imported {len(all_samples)} samples")

    # Step 2: Load train split
    print("\n[2/4] Loading train split...")
    train_file = root / "splits" / "train.txt"
    with open(train_file, "r", encoding="utf-8") as f:
        train_ids = set(line.strip() for line in f if line.strip())
    train_samples = [s for s in all_samples if s.sample_id in train_ids]
    print(f"  [OK] Loaded {len(train_samples)} train samples")

    # Step 3: Initialize pipeline and oracle
    print("\n[3/4] Initializing pipeline and oracle...")
    client = ArtifactMockClient()
    pipeline = ArtifactDelibPipeline(client=client)
    parser = PredictionParser()
    metrics = ArtifactMetrics(parser)
    oracle = OracleRouteBuilder(parser, metrics)
    print(f"  [OK] Pipeline and oracle ready")

    # Step 4: Process samples and build oracle dataset
    print("\n[4/4] Processing samples...")
    records = []
    n_processed = 0

    for i, sample in enumerate(train_samples):
        try:
            # Run pipeline
            result = pipeline.run(sample.image_path, sample.sample_id)

            # Extract features
            features = oracle.extract_features(result)

            # Try different route actions and score them
            outcomes = []
            for route_action in ["FAST", "SHAPE_RECHECK", "STYLE_RECHECK",
                                "GLYPH_RECHECK", "MATERIAL_RECHECK",
                                "LOCAL_DETAIL_RECHECK", "DELIBERATION"]:
                outcome = oracle.score_route(
                    route_action=route_action,
                    final_text=result.final_identification.content,
                    gold_category=sample.category,
                    gold_type=sample.fine_grained_type,
                    gold_period=sample.period,
                    api_calls=result.total_api_calls,
                    tokens=result.total_usage.total_tokens,
                )
                outcomes.append(outcome)

            # Select best route
            best_action, best_outcome = oracle.select_oracle_route(outcomes)

            if best_outcome:
                record = TrainingRecord(
                    features=features,
                    oracle_label=route_to_label(best_action),
                    oracle_route=best_action,
                    sample_id=sample.sample_id,
                )
                records.append(record)

            n_processed += 1
            if (i + 1) % 100 == 0:
                print(f"  [{i+1}/{len(train_samples)}] Processed {n_processed} samples, {len(records)} records")

        except Exception as e:
            print(f"  [WARN] Error processing {sample.sample_id}: {e}")
            continue

    print(f"\n  [OK] Processed {n_processed} samples, created {len(records)} training records")

    # Step 5: Save training data
    print("\n[5/5] Saving training data...")
    import json
    output_file = output_dir / "oracle_training_data.json"

    # Map disagreement_type to numeric code
    disagreement_type_map = {
        "SHAPE": 0,
        "STYLE": 1,
        "GLYPH": 2,
        "MATERIAL": 3,
        "LOCAL_DETAIL": 4,
        "MULTI_FACTOR": 5,
        "UNKNOWN": 5,
    }

    data = {
        "n_samples": len(records),
        "records": [
            {
                "features": r.features if isinstance(r.features, list) else [
                    r.features.top1_confidence,
                    r.features.top2_confidence,
                    r.features.margin,
                    float(disagreement_type_map.get(r.features.disagreement_type, 5)),
                    float(r.features.n_candidates),
                    # Derived features
                    r.features.top1_confidence - r.features.top2_confidence,
                    r.features.top1_confidence * r.features.top2_confidence,
                ],
                "oracle_label": r.oracle_label,
                "oracle_route": r.oracle_route,
                "sample_id": r.sample_id,
            }
            for r in records
        ],
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"  [OK] Saved to {output_file}")

    # Summary
    from collections import Counter
    route_counts = Counter(r.oracle_route for r in records)
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total training records: {len(records)}")
    print("\nRoute distribution:")
    for route, count in sorted(route_counts.items(), key=lambda x: -x[1]):
        print(f"  {route:20s}: {count:4d} ({count/len(records)*100:.1f}%)")

    print("\n[SUCCESS] Oracle Route dataset built!")


if __name__ == "__main__":
    main()
