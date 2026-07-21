"""Train Learned Router (MLP) on Oracle Route dataset.

This script:
1. Loads Oracle Route training data
2. Splits into train/validation
3. Trains MLP router
4. Evaluates accuracy
5. Saves trained model
"""

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

import json
from artifact_delib.router.learned_router import MLPRouter, TrainingRecord, TrainingSet
from artifact_delib.router.oracle_route_builder import route_to_label


def main():
    root = Path("data/artifact")
    oracle_file = root / "oracle_routes" / "oracle_training_data.json"
    model_output = root / "oracle_routes" / "learned_router.json"

    print("=" * 60)
    print("TRAINING LEARNED ROUTER (MLP)")
    print("=" * 60)

    # Step 1: Load training data
    print("\n[1/5] Loading Oracle Route dataset...")
    with open(oracle_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"  [OK] Loaded {data['n_samples']} samples")

    # Step 2: Convert to TrainingSet
    print("\n[2/5] Converting to TrainingSet...")
    records = []
    for r in data["records"]:
        record = TrainingRecord(
            features=r["features"],
            oracle_label=r["oracle_label"],
            oracle_route=r["oracle_route"],
            sample_id=r["sample_id"],
        )
        records.append(record)

    training_set = TrainingSet(records=records)
    print(f"  [OK] Created TrainingSet with {len(training_set)} records")

    # Step 3: Split into train/validation
    print("\n[3/5] Splitting into train/validation (80/20)...")
    n_total = len(training_set)
    n_train = int(n_total * 0.8)
    train_records = training_set.records[:n_train]
    val_records = training_set.records[n_train:]

    train_set = TrainingSet(records=train_records)
    val_set = TrainingSet(records=val_records)

    print(f"  [OK] Train: {len(train_set)} samples")
    print(f"  [OK] Validation: {len(val_set)} samples")

    # Step 4: Train MLP
    print("\n[4/5] Training MLP router...")
    mlp = MLPRouter(learning_rate=0.01, random_seed=42)

    # Train with early stopping
    best_val_acc = 0.0
    patience = 10
    patience_counter = 0

    for epoch in range(100):
        # Train one epoch
        history = mlp.train(train_set, epochs=1, batch_size=32)

        # Evaluate on validation
        val_acc = mlp.accuracy(val_set)

        if (epoch + 1) % 10 == 0:
            train_loss = history[-1] if history else 0.0
            print(f"  Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  [OK] Early stopping at epoch {epoch+1}")
                break

    print(f"  [OK] Training complete. Best validation accuracy: {best_val_acc:.4f}")

    # Step 5: Evaluate and save
    print("\n[5/5] Evaluating and saving model...")

    # Final evaluation
    train_acc = mlp.accuracy(train_set)
    val_acc = mlp.accuracy(val_set)

    print(f"\n  Final Results:")
    print(f"    Train Accuracy: {train_acc:.4f}")
    print(f"    Validation Accuracy: {val_acc:.4f}")

    # Route distribution on validation set
    from collections import Counter
    val_predictions = [mlp.predict(r.features) for r in val_records]
    pred_counts = Counter(val_predictions)

    print(f"\n  Predicted Route Distribution (Validation):")
    for route, count in sorted(pred_counts.items(), key=lambda x: -x[1]):
        print(f"    {route:20s}: {count:4d} ({count/len(val_records)*100:.1f}%)")

    # Save model
    mlp.save(model_output)
    print(f"\n  [OK] Saved trained model to {model_output}")

    print("\n" + "=" * 60)
    print("[SUCCESS] Learned Router training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
