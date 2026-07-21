"""Run data split and validation on the Met artifact dataset."""

import sys
from pathlib import Path

# Fix Windows encoding
sys.stdout.reconfigure(encoding='utf-8')

from artifact_delib.data.importer import ArtifactDatasetImporter
from artifact_delib.data.splitter import ArtifactDatasetSplitter
from artifact_delib.data.validator import ArtifactDatasetValidator


def main():
    root = Path("data/artifact")
    manifest = root / "met_artifact_manifest.csv"
    image_root = root / "images"

    print("=" * 60)
    print("MET ARTIFACT DATASET - DATA SPLIT AND VALIDATION")
    print("=" * 60)

    # Step 1: Import dataset
    print("\n[1/4] Importing dataset...")
    importer = ArtifactDatasetImporter(image_root=image_root)
    samples = importer.import_manifest(manifest)
    print(f"  [OK] Imported {len(samples)} samples")

    # Step 2: Validate dataset
    print("\n[2/4] Validating dataset...")
    validator = ArtifactDatasetValidator()
    issues = validator.validate(samples)

    if issues["missing_images"]:
        print(f"  [WARN] Missing images: {len(issues['missing_images'])}")
    else:
        print(f"  [OK] All images present")

    if issues["duplicate_ids"]:
        print(f"  [WARN] Duplicate IDs: {len(issues['duplicate_ids'])}")
    else:
        print(f"  [OK] No duplicate IDs")

    # Step 3: Split dataset
    print("\n[3/4] Splitting dataset (70/10/20)...")
    splitter = ArtifactDatasetSplitter(
        train_ratio=0.70,
        validation_ratio=0.10,
        seed=42,
    )
    splits = splitter.split(samples)

    print(f"  [OK] Train: {len(splits['train'])} samples")
    print(f"  [OK] Validation: {len(splits['validation'])} samples")
    print(f"  [OK] Test: {len(splits['test'])} samples")

    # Step 4: Validate splits
    print("\n[4/4] Validating splits...")
    all_split_samples = splits["train"] + splits["validation"] + splits["test"]
    split_issues = validator.validate(all_split_samples)

    if split_issues["group_split_leakage"]:
        print(f"  [WARN] Group leakage detected: {len(split_issues['group_split_leakage'])}")
    else:
        print(f"  [OK] No group leakage between splits")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total samples: {len(samples)}")
    print(f"Train: {len(splits['train'])} ({len(splits['train'])/len(samples)*100:.1f}%)")
    print(f"Validation: {len(splits['validation'])} ({len(splits['validation'])/len(samples)*100:.1f}%)")
    print(f"Test: {len(splits['test'])} ({len(splits['test'])/len(samples)*100:.1f}%)")

    # Category distribution
    from collections import Counter
    train_cats = Counter(s.category for s in splits["train"])
    val_cats = Counter(s.category for s in splits["validation"])
    test_cats = Counter(s.category for s in splits["test"])

    print("\nCategory distribution:")
    all_cats = set(train_cats.keys()) | set(val_cats.keys()) | set(test_cats.keys())
    for cat in sorted(all_cats):
        print(f"  {cat:20s} | Train: {train_cats.get(cat, 0):4d} | Val: {val_cats.get(cat, 0):4d} | Test: {test_cats.get(cat, 0):4d}")

    # Save split info
    print("\n" + "=" * 60)
    print("SAVING SPLIT INFORMATION")
    print("=" * 60)

    output_dir = root / "splits"
    output_dir.mkdir(exist_ok=True)

    for split_name, split_samples in splits.items():
        output_file = output_dir / f"{split_name}.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            for s in split_samples:
                f.write(f"{s.sample_id}\n")
        print(f"  [OK] Saved {split_name}: {output_file}")

    print("\n[SUCCESS] Data split and validation complete!")


if __name__ == "__main__":
    main()
