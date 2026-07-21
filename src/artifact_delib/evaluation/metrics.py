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

    # Label coverage — what fraction of samples have valid gold labels
    category_label_coverage: float | None = None
    type_label_coverage: float | None = None
    period_label_coverage: float | None = None
    material_label_coverage: float | None = None
    joint_label_coverage: float | None = None

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
    """Evaluation result for a single sample.

    Correctness fields are bool | None:
    - True: prediction matches gold
    - False: prediction does not match gold
    - None: gold label is missing (no valid comparison)
    """

    sample_id: str

    # Top-1 — None means gold label not available
    category_correct: bool | None = None
    type_correct: bool | None = None
    period_correct: bool | None = None
    material_correct: bool | None = None
    joint_correct: bool | None = None

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

        Missing gold labels (None) are properly handled:
        - If gold is None, the corresponding accuracy comparison is skipped
          (not counted as correct or incorrect)
        - This prevents missing labels from being wrongly counted as prediction errors
        - Use label_coverage metrics to report how many samples have valid labels

        Args:
            sample_id: Unique sample identifier.
            final_text: The final NL identification (Top-1 prediction).
            gold_category / gold_type / gold_period / gold_material: Ground-truth labels.
                None means the label is not available (not an error).
            initial_text: Optional initial prediction for correction/harm tracking.
            candidate_texts: Optional list of top-K candidate NL texts (sorted by
                confidence, highest first). Used to compute Top-K accuracy.
        """
        pred = self.parser.parse(final_text)

        # ── Top-1 accuracy ──
        # When gold is None, the comparison is undefined — set to None
        # to indicate "no valid comparison" rather than "incorrect".
        cat_correct: bool | None = (
            pred.category == gold_category
            if pred.category is not None and gold_category is not None
            else None
        )
        type_correct: bool | None = (
            pred.fine_grained_type == gold_type
            if pred.fine_grained_type is not None and gold_type is not None
            else None
        )
        period_correct: bool | None = (
            pred.period == gold_period
            if pred.period is not None and gold_period is not None
            else None
        )
        material_correct: bool | None = (
            pred.material == gold_material
            if pred.material is not None and gold_material is not None
            else None
        )
        # Joint: only defined when both type AND period have valid gold
        joint: bool | None = (
            True if (type_correct is True and period_correct is True)
            else False if (type_correct is not None and period_correct is not None
                          and type_correct is False or period_correct is False)
            else None
        )
        # Simpler: joint is None if either is None
        if type_correct is None or period_correct is None:
            joint = None
        else:
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
        if initial_text is not None and gold_type is not None:
            initial_pred = self.parser.parse(initial_text)
            initial_type_ok = (
                initial_pred.fine_grained_type == gold_type
                if initial_pred.fine_grained_type is not None and gold_type is not None
                else None
            )
            if initial_type_ok is not None and type_correct is not None:
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

        # ── Top-1 accuracy with valid-only denominators ──
        # Each metric only counts samples where the gold label is available.
        # Missing gold labels (None) are excluded from the denominator.
        def _valid_accuracy(
            evaluations: list[SampleEvaluation],
            attr: str,
        ) -> tuple[float | None, int, int]:
            """Compute accuracy using only samples with valid gold labels.

            Returns:
                (accuracy, valid_count, total_count)
            """
            valid = [e for e in evaluations if getattr(e, attr) is not None]
            if not valid:
                return None, 0, len(evaluations)
            correct = sum(1 for e in valid if getattr(e, attr) is True)
            return correct / len(valid), len(valid), len(evaluations)

        top1_cat, n_cat_valid, n_cat_total = _valid_accuracy(evaluations, "category_correct")
        top1_type, n_type_valid, n_type_total = _valid_accuracy(evaluations, "type_correct")
        top1_period, n_period_valid, n_period_total = _valid_accuracy(evaluations, "period_correct")
        top1_material, n_material_valid, n_material_total = _valid_accuracy(evaluations, "material_correct")

        # Joint: only when both type AND period are valid
        joint_valid = [
            e for e in evaluations
            if e.type_correct is not None and e.period_correct is not None
        ]
        if joint_valid:
            joint_correct = sum(
                1 for e in joint_valid
                if e.type_correct is True and e.period_correct is True
            )
            top1_joint = joint_correct / len(joint_valid)
            n_joint_valid = len(joint_valid)
        else:
            top1_joint = None
            n_joint_valid = 0

        # ── Top-K accuracy (valid-only denominators) ──
        def _valid_topk(
            evaluations: list[SampleEvaluation],
            attr: str,
            gold_attr: str,
        ) -> float | None:
            valid = [e for e in evaluations if getattr(e, gold_attr) is not None]
            if not valid:
                return None
            return sum(getattr(e, attr) for e in valid) / len(valid)

        top3_type = _valid_topk(evaluations, "type_in_top3", "gold_type")
        top5_type = _valid_topk(evaluations, "type_in_top5", "gold_type")
        top3_period = _valid_topk(evaluations, "period_in_top3", "gold_period")
        top5_period = _valid_topk(evaluations, "period_in_top5", "gold_period")
        top3_joint = _valid_topk(evaluations, "joint_in_top3", "gold_type")
        top5_joint = _valid_topk(evaluations, "joint_in_top5", "gold_type")

        # ── Correction / harm ──
        # Only count corrections/harms for samples with valid gold labels
        valid_type_evals = [e for e in evaluations if e.gold_type is not None]
        total_correctable = sum(1 for e in valid_type_evals if e.corrected or e.harmed)
        corrections = sum(1 for e in valid_type_evals if e.corrected)
        harms = sum(1 for e in valid_type_evals if e.harmed)
        n_valid = len(valid_type_evals)
        correction_rate = corrections / total_correctable if total_correctable else None
        harm_rate = harms / (n_valid - total_correctable + harms) if n_valid - total_correctable + harms else None

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

        # ── Label coverage ──
        cat_coverage = n_cat_valid / n if n else None
        type_coverage = n_type_valid / n if n else None
        period_coverage = n_period_valid / n if n else None
        material_coverage = n_material_valid / n if n else None
        joint_coverage = n_joint_valid / n if n else None

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
            category_label_coverage=cat_coverage,
            type_label_coverage=type_coverage,
            period_label_coverage=period_coverage,
            material_label_coverage=material_coverage,
            joint_label_coverage=joint_coverage,
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
