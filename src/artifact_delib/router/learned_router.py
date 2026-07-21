"""Learned Router — MLP that learns to predict the optimal route from features.

Architecture:
- Input: RouteFeatures → 7-dim float vector
- Hidden: 2-layer MLP (32 → 16)
- Output: 7-class softmax (FAST, SHAPE/STYLE/GLYPH/MATERIAL/LOCAL_DETAIL_RECHECK, DELIBERATION)

Training data: OracleRouteDataset (features, oracle_route_label) pairs
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from artifact_delib.router.oracle_route_builder import (
    N_CLASSES,
    N_FEATURES,
    features_to_tensor,
    label_to_route,
    route_to_label,
)
from artifact_delib.schemas import CandidateSet, DisagreementAnalysis, RouteDecision


@dataclass
class TrainingRecord:
    """One training example: features → oracle route."""

    features: list[float]
    oracle_label: int
    oracle_route: str
    sample_id: str = ""


@dataclass
class TrainingSet:
    """A collection of training records."""

    records: list[TrainingRecord] = field(default_factory=list)

    @property
    def X(self) -> list[list[float]]:
        return [r.features for r in self.records]

    @property
    def y(self) -> list[int]:
        return [r.oracle_label for r in self.records]

    def __len__(self) -> int:
        return len(self.records)


class MLPRouter:
    """Simple MLP-based router that predicts the best next action.

    Architecture: Input(N_FEATURES) → FC(32) → ReLU → FC(16) → ReLU → FC(N_CLASSES)
    Training: Cross-entropy loss, SGD optimizer.
    """

    def __init__(self, learning_rate: float = 0.01, random_seed: int = 42) -> None:
        import random
        self.lr = learning_rate
        self.rng = random.Random(random_seed)
        self.trained = False
        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier-like initialization for 3-layer MLP."""
        # Layer 1: N_FEATURES → 32
        scale1 = math.sqrt(2.0 / (N_FEATURES + 32))
        self.w1 = [[self.rng.uniform(-scale1, scale1) for _ in range(N_FEATURES)]
                    for _ in range(32)]
        self.b1 = [0.0] * 32

        # Layer 2: 32 → 16
        scale2 = math.sqrt(2.0 / (32 + 16))
        self.w2 = [[self.rng.uniform(-scale2, scale2) for _ in range(32)]
                    for _ in range(16)]
        self.b2 = [0.0] * 16

        # Layer 3: 16 → N_CLASSES
        scale3 = math.sqrt(2.0 / (16 + N_CLASSES))
        self.w3 = [[self.rng.uniform(-scale3, scale3) for _ in range(16)]
                    for _ in range(N_CLASSES)]
        self.b3 = [0.0] * N_CLASSES

    def forward(self, x: list[float]) -> list[float]:
        """Forward pass: return logits for each class."""
        # Layer 1
        h1 = [max(0.0, sum(x[j] * self.w1[i][j] for j in range(len(x))) + self.b1[i])
              for i in range(32)]
        # Layer 2
        h2 = [max(0.0, sum(h1[j] * self.w2[i][j] for j in range(32)) + self.b2[i])
              for i in range(16)]
        # Layer 3
        logits = [sum(h2[j] * self.w3[i][j] for j in range(16)) + self.b3[i]
                  for i in range(N_CLASSES)]
        return logits

    def predict(self, features: list[float]) -> str:
        """Predict the best route action for given features."""
        logits = self.forward(features)
        pred_label = max(range(N_CLASSES), key=lambda i: logits[i])
        return label_to_route(pred_label)

    def predict_with_confidence(self, features: list[float]) -> tuple[str, float]:
        """Predict route with confidence score."""
        logits = self.forward(features)
        # Softmax
        max_logit = max(logits)
        exp_sum = sum(math.exp(l - max_logit) for l in logits)
        probs = [math.exp(l - max_logit) / exp_sum for l in logits]
        pred_label = max(range(N_CLASSES), key=lambda i: probs[i])
        return label_to_route(pred_label), probs[pred_label]

    def route(
        self,
        disagreement: DisagreementAnalysis | None,
        candidates: CandidateSet,
        recheck_count: int = 0,
        deliberation_count: int = 0,
        completed_rechecks: tuple[str, ...] = (),
    ) -> RouteDecision:
        """Router-compatible interface for use in the pipeline."""
        from artifact_delib.router.oracle_route_builder import RouteFeatures

        hint = disagreement.route_hint if disagreement else "MULTI_FACTOR"
        feats = RouteFeatures(
            top1_confidence=candidates.top1_confidence,
            top2_confidence=candidates.top2_confidence,
            margin=candidates.margin,
            disagreement_type=hint,
            n_candidates=len(candidates.candidates),
        )
        x = features_to_tensor(feats)
        action, confidence = self.predict_with_confidence(x)

        # Don't repeat completed rechecks
        done = set(completed_rechecks)
        if action in done and deliberation_count == 0:
            # Fall through to next best action
            logits = self.forward(x)
            sorted_labels = sorted(range(N_CLASSES), key=lambda i: -logits[i])
            for label in sorted_labels:
                alt_action = label_to_route(label)
                if alt_action not in done and alt_action != "FAST":
                    action = alt_action
                    break
            else:
                action = "FAST"

        reason = f"mlp:{confidence:.2f}" if self.trained else "mlp:untrained"
        return RouteDecision(
            action=action, reason=reason,
            recheck_count=recheck_count,  # type: ignore[arg-type]
            deliberation_count=deliberation_count,
        )

    def train(
        self,
        training_set: TrainingSet,
        epochs: int = 50,
        batch_size: int = 16,
        verbose: bool = False,
    ) -> list[float]:
        """Train the MLP with mini-batch SGD and cross-entropy loss.

        Returns the loss history per epoch.
        """
        n = len(training_set)
        if n == 0:
            return []

        loss_history: list[float] = []
        records = training_set.records

        for epoch in range(epochs):
            # Shuffle
            self.rng.shuffle(records)
            total_loss = 0.0
            n_batches = 0

            for start in range(0, n, batch_size):
                batch = records[start:start + batch_size]
                batch_loss = self._train_batch(
                    [r.features for r in batch], [r.oracle_label for r in batch]
                )
                total_loss += batch_loss
                n_batches += 1

            avg_loss = total_loss / n_batches if n_batches else 0.0
            loss_history.append(avg_loss)

            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                print(f"  epoch {epoch:3d}: loss={avg_loss:.4f}")

        self.trained = True
        return loss_history

    def _train_batch(
        self, X_batch: list[list[float]], y_batch: list[int]
    ) -> float:
        """Compute gradients for one mini-batch and update weights."""
        batch_size = len(X_batch)
        total_loss = 0.0

        # Zero gradients
        dw1 = [[0.0] * N_FEATURES for _ in range(32)]
        db1 = [0.0] * 32
        dw2 = [[0.0] * 32 for _ in range(16)]
        db2 = [0.0] * 16
        dw3 = [[0.0] * 16 for _ in range(N_CLASSES)]
        db3 = [0.0] * N_CLASSES

        for x, y_true in zip(X_batch, y_batch):
            # Forward pass with intermediate activations for backprop
            # Layer 1
            z1 = [sum(x[j] * self.w1[i][j] for j in range(len(x))) + self.b1[i]
                  for i in range(32)]
            a1 = [max(0.0, z) for z in z1]
            # Layer 2
            z2 = [sum(a1[j] * self.w2[i][j] for j in range(32)) + self.b2[i]
                  for i in range(16)]
            a2 = [max(0.0, z) for z in z2]
            # Layer 3
            logits = [sum(a2[j] * self.w3[i][j] for j in range(16)) + self.b3[i]
                      for i in range(N_CLASSES)]

            # Softmax
            max_logit = max(logits)
            exp_logits = [math.exp(l - max_logit) for l in logits]
            exp_sum = sum(exp_logits)
            probs = [e / exp_sum for e in exp_logits]

            # Cross-entropy loss
            loss = -math.log(max(probs[y_true], 1e-10))
            total_loss += loss

            # Backprop from softmax+CE: dL/dlogits = probs - one_hot
            dlogits = [probs[i] - (1.0 if i == y_true else 0.0)
                       for i in range(N_CLASSES)]

            # Layer 3 gradients
            for i in range(N_CLASSES):
                db3[i] += dlogits[i]  # Accumulate
                for j in range(16):
                    dw3[i][j] += dlogits[i] * a2[j]

            # Backprop through ReLU Layer 2
            dz2 = [0.0] * 16
            for j in range(16):
                for i in range(N_CLASSES):
                    dz2[j] += dlogits[i] * self.w3[i][j]
                if z2[j] <= 0:
                    dz2[j] = 0.0

            for i in range(16):
                db2[i] += dz2[i]
                for j in range(32):
                    dw2[i][j] += dz2[i] * a1[j]

            # Backprop through ReLU Layer 1
            dz1 = [0.0] * 32
            for j in range(32):
                for i in range(16):
                    dz1[j] += dz2[i] * self.w2[i][j]
                if z1[j] <= 0:
                    dz1[j] = 0.0

            for i in range(32):
                db1[i] += dz1[i]
                for j in range(N_FEATURES):
                    dw1[i][j] += dz1[i] * x[j]

        # Gradient step
        lr = self.lr / batch_size
        for i in range(32):
            self.b1[i] -= lr * db1[i]
            for j in range(N_FEATURES):
                self.w1[i][j] -= lr * dw1[i][j]
        for i in range(16):
            self.b2[i] -= lr * db2[i]
            for j in range(32):
                self.w2[i][j] -= lr * dw2[i][j]
        for i in range(N_CLASSES):
            self.b3[i] -= lr * db3[i]
            for j in range(16):
                self.w3[i][j] -= lr * dw3[i][j]

        return total_loss / batch_size

    def accuracy(self, training_set: TrainingSet) -> float:
        """Compute training accuracy."""
        if len(training_set) == 0:
            return 1.0
        correct = sum(
            self.predict(r.features) == r.oracle_route
            for r in training_set.records
        )
        return correct / len(training_set)

    def save(self, path: Path) -> None:
        """Save model weights to JSON."""
        import json
        data = {
            "w1": self.w1, "b1": self.b1,
            "w2": self.w2, "b2": self.b2,
            "w3": self.w3, "b3": self.b3,
            "trained": self.trained,
        }
        path.write_text(json.dumps(data), encoding="utf-8")

    def load(self, path: Path) -> None:
        """Load model weights from JSON."""
        import json
        data = json.loads(path.read_text(encoding="utf-8"))
        self.w1 = data["w1"]; self.b1 = data["b1"]
        self.w2 = data["w2"]; self.b2 = data["b2"]
        self.w3 = data["w3"]; self.b3 = data["b3"]
        self.trained = data.get("trained", True)
