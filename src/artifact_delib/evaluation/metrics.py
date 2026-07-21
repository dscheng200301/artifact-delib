"""Metrics for ArtifactDelib evaluation — computed only from parsed predictions vs gold labels.

Metrics computed:
- Top-1 / Top-3 / Top-5 Accuracy (category, type, period, material, joint)
  Top-1 = final prediction; Top-K = gold in top-K candidate texts.
- Per-class Precision / Recall / F1 (category, type, period, material)
- Macro-F1 (averaged across classes)
- Correction Rate (initial wrong → final correct after interaction)
- Harm Rate (initial correct → final wrong after interaction)
- Cost metrics: average tokens, average API calls

These metrics are NEVER used during inference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median

from artifact_delib.evaluation.prediction_parser import (
    ParsedIdentification,
    PredictionParser,
)


@dataclass(frozen=True)
class PerClassMetrics:
    """Precision / Recall / F1 for a single class."""

    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    support: int = 0        # number of true samples for this class


@dataclass(frozen=True)
class EvaluationResult:
    """Aggregate evaluation results for a batch of predictions."""

    n_samples: int

    # Top-1 accuracy (= final prediction)
    top1_category_accuracy: float | None = None
    top1_type_accuracy: float | None = None
    top1_period_accuracy: float | None = None
    top1_material_accuracy: float | None = None
    top1_joint_accuracy: float | None = None

    # Top-K accuracy (K=3,5) — whether gold is in any of top-K candidate texts
    top3_type_accuracy: float | None = None
    top5_type_accuracy: float | None = None
    top3_period_accuracy: float | None = None
    top5_period_accuracy: float | None = None
    top3_joint_accuracy: float | None = None
    top5_joint_accuracy: float | None = None

    # Per-class macro metrics
    macro_f1_type: float | None = None
    macro_f1_period: float | None = None
    macro_f1_category: float | None = None
    macro_f1_material: float | None = None
    macro_precision_type: float | None = None
    macro_recall_type: float | None = None
    macro_precision_period: float | None = None
    macro_recall_period: float | None = None

    # Interaction metrics
    correction_rate: float | None = None
    harm_rate: float | None = None
    no_change_rate: float | None = None

    # Micro-F1 (global precision/recall across all samples)
    micro_f1_type: float | None = None
    micro_f1_period: float | None = None
    micro_f1_category: float | None = None
    micro_f1_material: float | None = None

    # Parse failure rate
    parse_failure_rate: float | None = None

    # Deliberation statistics
    deliberation_trigger_rate: float | None = None
    recheck_trigger_rate: float | None = None
    avg_rechecks: float | None = None
    avg_deliberation_rounds: float | None = None

    # Per-class detail
    per_type: dict[str, PerClassMetrics] | None = None
    per_period: dict[str, PerClassMetrics] | None = None
    per_category: dict[str, PerClassMetrics] | None = None
    per_material: dict[str, PerClassMetrics] | None = None

    # Cost
    average_tokens: float | None = None
    median_tokens: float | None = None
    total_api_calls: int = 0
    average_api_calls: float | None = None
    p50_latency_ms: float | None = None
    p95_latency_ms: float | None = None
    average_latency_ms: float | None = None

    # ── Backward-compatible aliases (legacy field names) ──
    # These map old names → Top-1 accuracy so existing code keeps working.

    @property
    def category_accuracy(self) -> float | None:
        """Legacy alias for top1_category_accuracy."""
        return self.top1_category_accuracy

    @property
    def type_accuracy(self) -> float | None:
        """Legacy alias for top1_type_accuracy."""
        return self.top1_type_accuracy

    @property
    def period_accuracy(self) -> float | None:
        """Legacy alias for top1_period_accuracy."""
        return self.top1_period_accuracy

    @property
    def joint_accuracy(self) -> float | None:
        """Legacy alias for top1_joint_accuracy."""
        return self.top1_joint_accuracy


@dataclass
class SampleEvaluation:
    """Evaluation result for a single sample."""

    sample_id: str

    # Top-1
    category_correct: bool = False
    type_correct: bool = False
    period_correct: bool = False
    material_correct: bool = False
    joint_correct: bool = False

    # Top-K (whether gold is in any of top-K candidate texts)
    type_in_top3: bool = False
    type_in_top5: bool = False
    period_in_top3: bool = False
    period_in_top5: bool = False
    joint_in_top3: bool = False
    joint_in_top5: bool = False

    # For per-class P/R/F1: predicted classes (for precision calculation)
    predicted_category: str | None = None
    predicted_type: str | None = None
    predicted_period: str | None = None
    predicted_material: str | None = None
    gold_category: str | None = None
    gold_type: str | None = None
    gold_period: str | None = None
    gold_material: str | None = None

    # Interaction metrics
    corrected: bool = False
    harmed: bool = False


class ArtifactMetrics:
    """Compute evaluation metrics from pipeline results vs gold labels.

    Computes Top-1 / Top-3 / Top-5 accuracy (academic standard for FGVR),
    per-class precision/recall/F1, macro-F1, and interaction metrics.
    """

    def __init__(self, parser: PredictionParser | None = None) -> None:
        self.parser = parser or PredictionParser()

    def evaluate_sample(
        self,
        sample_id: str,
        final_text: str,
        gold_category: str | None,
        gold_type: str | None,
        gold_period: str | None,
        gold_material: str | None = None,
        initial_text: str | None = None,
        candidate_texts: list[str] | None = None,
    ) -> SampleEvaluation:
        """Evaluate one sample against its gold labels.

        Args:
            sample_id: Unique sample identifier.
            final_text: The final NL identification (Top-1 prediction).
            gold_category / gold_type / gold_period / gold_material: Ground-truth labels.
            initial_text: Optional initial prediction for correction/harm tracking.
            candidate_texts: Optional list of top-K candidate NL texts (sorted by
                confidence, highest first). Used to compute Top-K accuracy.
        """
        pred = self.parser.parse(final_text)

        # ── Top-1 accuracy ──
        cat_correct = (
            pred.category == gold_category
            if pred.category and gold_category
            else False
        )
        type_correct = (
            pred.fine_grained_type == gold_type
            if pred.fine_grained_type and gold_type
            else False
        )
        period_correct = (
            pred.period == gold_period
            if pred.period and gold_period
            else False
        )
        material_correct = (
            pred.material == gold_material
            if pred.material and gold_material
            else False
        )
        joint = type_correct and period_correct

        # ── Top-K accuracy (from candidate texts) ──
        type_in_top3, type_in_top5 = False, False
        period_in_top3, period_in_top5 = False, False
        joint_in_top3, joint_in_top5 = False, False
        if candidate_texts:
            type_in_top3, type_in_top5 = self._check_top_k(
                candidate_texts,
                gold_type,
                extract_fn=lambda text: (self.parser.parse(text).fine_grained_type),
            )
            period_in_top3, period_in_top5 = self._check_top_k(
                candidate_texts,
                gold_period,
                extract_fn=lambda text: (self.parser.parse(text).period),
            )
            joint_in_top3 = type_in_top3 and period_in_top3
            joint_in_top5 = type_in_top5 and period_in_top5

        # ── Correction / harm ──
        corrected = False
        harmed = False
        if initial_text is not None:
            initial_pred = self.parser.parse(initial_text)
            initial_type_ok = (
                initial_pred.fine_grained_type == gold_type
                if initial_pred.fine_grained_type and gold_type
                else False
            )
            corrected = not initial_type_ok and type_correct
            harmed = initial_type_ok and not type_correct

        return SampleEvaluation(
            sample_id=sample_id,
            category_correct=cat_correct,
            type_correct=type_correct,
            period_correct=period_correct,
            material_correct=material_correct,
            joint_correct=joint,
            type_in_top3=type_in_top3,
            type_in_top5=type_in_top5,
            period_in_top3=period_in_top3,
            period_in_top5=period_in_top5,
            joint_in_top3=joint_in_top3,
            joint_in_top5=joint_in_top5,
            predicted_category=pred.category,
            predicted_type=pred.fine_grained_type,
            predicted_period=pred.period,
            predicted_material=pred.material,
            gold_category=gold_category,
            gold_type=gold_type,
            gold_period=gold_period,
            gold_material=gold_material,
            corrected=corrected,
            harmed=harmed,
        )

    @staticmethod
    def _check_top_k(
        candidate_texts: list[str],
        gold: str | None,
        extract_fn,
    ) -> tuple[bool, bool]:
        """Return (in_top3, in_top5) — whether `gold` appears in any of top-K
        candidate texts' parsed fields."""
        if not gold:
            return False, False
        in_top3, in_top5 = False, False
        for i, text in enumerate(candidate_texts[:5]):
            extracted = extract_fn(text)
            if extracted == gold:
                if i < 3:
                    in_top3 = True
                in_top5 = True
        return in_top3, in_top5

    def compute_metrics(
        self,
        evaluations: list[SampleEvaluation],
        token_counts: list[int] | None = None,
        api_calls: list[int] | None = None,
        latencies_ms: list[float] | None = None,
        recheck_counts: list[int] | None = None,
        deliberation_rounds: list[int] | None = None,
        triggered_recheck: int = 0,
        triggered_deliberation: int = 0,
    ) -> EvaluationResult:
        """Aggregate sample evaluations into summary metrics.

        Args:
            evaluations: Per-sample evaluation results.
            token_counts: Total tokens per sample (input+output).
            api_calls: API calls per sample.
            latencies_ms: Total latency in milliseconds per sample.
            recheck_counts: Number of rechecks per sample.
            deliberation_rounds: Number of deliberation rounds per sample.
            triggered_recheck: Count of samples that triggered a recheck.
            triggered_deliberation: Count of samples that triggered deliberation.
        """
        n = len(evaluations)
        if n == 0:
            return EvaluationResult(n_samples=0)

        # ── Top-1 accuracy ──
        top1_cat = sum(e.category_correct for e in evaluations) / n
        top1_type = sum(e.type_correct for e in evaluations) / n
        top1_period = sum(e.period_correct for e in evaluations) / n
        top1_material = sum(e.material_correct for e in evaluations) / n
        top1_joint = sum(e.joint_correct for e in evaluations) / n

        # ── Top-K accuracy ──
        top3_type = sum(e.type_in_top3 for e in evaluations) / n
        top5_type = sum(e.type_in_top5 for e in evaluations) / n
        top3_period = sum(e.period_in_top3 for e in evaluations) / n
        top5_period = sum(e.period_in_top5 for e in evaluations) / n
        top3_joint = sum(e.joint_in_top3 for e in evaluations) / n
        top5_joint = sum(e.joint_in_top5 for e in evaluations) / n

        # ── Correction / harm ──
        total_correctable = sum(1 for e in evaluations if e.corrected or e.harmed)
        corrections = sum(1 for e in evaluations if e.corrected)
        harms = sum(1 for e in evaluations if e.harmed)
        correction_rate = corrections / total_correctable if total_correctable else None
        harm_rate = harms / (n - total_correctable + harms) if n - total_correctable + harms else None

        # ── Per-class Precision / Recall / F1 ──
        per_cat = self._compute_per_class_metrics(
            evaluations,
            pred_fn=lambda e: e.predicted_category,
            gold_fn=lambda e: e.gold_category,
        )
        per_type = self._compute_per_class_metrics(
            evaluations,
            pred_fn=lambda e: e.predicted_type,
            gold_fn=lambda e: e.gold_type,
        )
        per_period = self._compute_per_class_metrics(
            evaluations,
            pred_fn=lambda e: e.predicted_period,
            gold_fn=lambda e: e.gold_period,
        )
        per_material = self._compute_per_class_metrics(
            evaluations,
            pred_fn=lambda e: e.predicted_material,
            gold_fn=lambda e: e.gold_material,
        )

        # ── Macro averages ──
        def _macro(values: dict[str, PerClassMetrics], attr: str) -> float | None:
            if not values:
                return None
            return sum(getattr(v, attr) for v in values.values()) / len(values)

        macro_f1_type = _macro(per_type, "f1")
        macro_f1_period = _macro(per_period, "f1")
        macro_f1_category = _macro(per_cat, "f1")
        macro_f1_material = _macro(per_material, "f1")
        macro_p_type = _macro(per_type, "precision")
        macro_r_type = _macro(per_type, "recall")
        macro_p_period = _macro(per_period, "precision")
        macro_r_period = _macro(per_period, "recall")

        # ── Micro-F1 (global P/R across all samples) ──
        micro_f1_type = self._compute_micro_f1(
            evaluations,
            pred_fn=lambda e: e.predicted_type,
            gold_fn=lambda e: e.gold_type,
        )
        micro_f1_period = self._compute_micro_f1(
            evaluations,
            pred_fn=lambda e: e.predicted_period,
            gold_fn=lambda e: e.gold_period,
        )
        micro_f1_category = self._compute_micro_f1(
            evaluations,
            pred_fn=lambda e: e.predicted_category,
            gold_fn=lambda e: e.gold_category,
        )
        micro_f1_material = self._compute_micro_f1(
            evaluations,
            pred_fn=lambda e: e.predicted_material,
            gold_fn=lambda e: e.gold_material,
        )

        # ── Parse failure rate ──
        parse_failures = sum(
            1 for e in evaluations
            if e.predicted_type is None and e.predicted_category is None
            and e.predicted_period is None and e.predicted_material is None
        )
        parse_failure_rate = parse_failures / n if n else None

        # ── No-change rate ──
        changed = sum(1 for e in evaluations if e.corrected or e.harmed)
        no_change_rate = (n - changed) / n if n else None

        # ── Latency percentiles ──
        avg_lat_ms = sum(latencies_ms) / len(latencies_ms) if latencies_ms else None
        p50_lat_ms = _percentile(latencies_ms, 50) if latencies_ms else None
        p95_lat_ms = _percentile(latencies_ms, 95) if latencies_ms else None

        # ── Deliberation / recheck stats ──
        avg_rechecks = sum(recheck_counts) / len(recheck_counts) if recheck_counts else None
        avg_delib = sum(deliberation_rounds) / len(deliberation_rounds) if deliberation_rounds else None
        recheck_trigger_rate = triggered_recheck / n if n else None
        deliberation_trigger_rate = triggered_deliberation / n if n else None

        # ── Cost ──
        avg_tokens = sum(token_counts) / len(token_counts) if token_counts else None
        med_tokens = float(median(token_counts)) if token_counts else None
        avg_api = sum(api_calls) / len(api_calls) if api_calls else None

        return EvaluationResult(
            n_samples=n,
            top1_category_accuracy=top1_cat,
            top1_type_accuracy=top1_type,
            top1_period_accuracy=top1_period,
            top1_material_accuracy=top1_material,
            top1_joint_accuracy=top1_joint,
            top3_type_accuracy=top3_type,
            top5_type_accuracy=top5_type,
            top3_period_accuracy=top3_period,
            top5_period_accuracy=top5_period,
            top3_joint_accuracy=top3_joint,
            top5_joint_accuracy=top5_joint,
            macro_f1_type=macro_f1_type,
            macro_f1_period=macro_f1_period,
            macro_f1_category=macro_f1_category,
            macro_f1_material=macro_f1_material,
            macro_precision_type=macro_p_type,
            macro_recall_type=macro_r_type,
            macro_precision_period=macro_p_period,
            macro_recall_period=macro_r_period,
            correction_rate=correction_rate,
            harm_rate=harm_rate,
            no_change_rate=no_change_rate,
            micro_f1_type=micro_f1_type,
            micro_f1_period=micro_f1_period,
            micro_f1_category=micro_f1_category,
            micro_f1_material=micro_f1_material,
            parse_failure_rate=parse_failure_rate,
            deliberation_trigger_rate=deliberation_trigger_rate,
            recheck_trigger_rate=recheck_trigger_rate,
            avg_rechecks=avg_rechecks,
            avg_deliberation_rounds=avg_delib,
            per_type=per_type or None,
            per_period=per_period or None,
            per_category=per_cat or None,
            per_material=per_material or None,
            average_tokens=avg_tokens,
            median_tokens=med_tokens,
            total_api_calls=sum(api_calls) if api_calls else 0,
            average_api_calls=avg_api,
            p50_latency_ms=p50_lat_ms,
            p95_latency_ms=p95_lat_ms,
            average_latency_ms=avg_lat_ms,
        )

    @staticmethod
    def _compute_per_class_metrics(
        evaluations: list[SampleEvaluation],
        pred_fn,
        gold_fn,
    ) -> dict[str, PerClassMetrics]:
        """Compute per-class precision / recall / F1 / support.

        A class counts as:
        - TP: predicted as class AND gold is class
        - FP: predicted as class AND gold is NOT class
        - FN: predicted as NOT class AND gold is class
        """
        # Collect all classes that appear as gold or prediction
        all_classes: set[str] = set()
        for e in evaluations:
            p = pred_fn(e)
            g = gold_fn(e)
            if p:
                all_classes.add(p)
            if g:
                all_classes.add(g)

        result: dict[str, PerClassMetrics] = {}
        for cls in all_classes:
            tp = fp = fn = support = 0
            for e in evaluations:
                p = pred_fn(e)
                g = gold_fn(e)
                if g == cls:
                    support += 1
                if p == cls and g == cls:
                    tp += 1
                elif p == cls and g != cls:
                    fp += 1
                elif p != cls and g == cls:
                    fn += 1

            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            result[cls] = PerClassMetrics(
                precision=precision,
                recall=recall,
                f1=f1,
                support=support,
            )
        return result

    @staticmethod
    def _compute_micro_f1(
        evaluations: list[SampleEvaluation],
        pred_fn,
        gold_fn,
    ) -> float | None:
        """Compute micro-F1 from global TP/FP/FN counts.

        Micro-F1 = 2 * micro_P * micro_R / (micro_P + micro_R)
          where micro_P = sum(TP) / sum(TP + FP)
                micro_R = sum(TP) / sum(TP + FN)
        """
        total_tp = total_fp = total_fn = 0
        all_classes: set[str] = set()
        for e in evaluations:
            p = pred_fn(e)
            g = gold_fn(e)
            if p:
                all_classes.add(p)
            if g:
                all_classes.add(g)

        if not all_classes:
            return None

        for cls in all_classes:
            for e in evaluations:
                p = pred_fn(e)
                g = gold_fn(e)
                if p == cls and g == cls:
                    total_tp += 1
                elif p == cls and g != cls:
                    total_fp += 1
                elif p != cls and g == cls:
                    total_fn += 1

        micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
        micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
        if micro_p + micro_r == 0:
            return 0.0
        return 2 * micro_p * micro_r / (micro_p + micro_r)


def _percentile(values: list[float] | None, p: float) -> float | None:
    """Compute the p-th percentile of a list of values."""
    if not values:
        return None
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * p / 100.0
    f = int(k)
    c = k - f
    if f + 1 < len(sorted_vals):
        return sorted_vals[f] + c * (sorted_vals[f + 1] - sorted_vals[f])
    return sorted_vals[f]
