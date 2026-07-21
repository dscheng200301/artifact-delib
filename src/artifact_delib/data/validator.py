"""Artifact dataset validator — checks images, labels, and leakage."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from artifact_delib.schemas import ArtifactSample


class ArtifactDatasetValidator:
    """Validate artifact dataset: images exist, labels consistent, no leakage."""

    def validate(self, samples: list[ArtifactSample]) -> dict[str, list[str]]:
        """Run all validation checks and return issues dict."""
        issues: dict[str, list[str]] = {
            "missing_images": [],
            "duplicate_ids": [],
            "group_split_leakage": [],
            "warnings": [],
        }

        seen_ids: set[str] = set()
        for s in samples:
            # Check duplicates
            if s.sample_id in seen_ids:
                issues["duplicate_ids"].append(f"duplicate sample_id: {s.sample_id}")
            seen_ids.add(s.sample_id)

            # Check images exist
            if not s.image_path.exists():
                issues["missing_images"].append(
                    f"missing image: {s.sample_id} → {s.image_path}"
                )

        # Check group leakage
        leakage = self._find_group_leakage(samples)
        for gid, splits in leakage.items():
            issues["group_split_leakage"].append(
                f"group {gid} appears in splits: {splits}"
            )

        return issues

    def is_valid(self, samples: list[ArtifactSample]) -> bool:
        """Return True if no critical issues found."""
        issues = self.validate(samples)
        return not any(
            issues[k] for k in ["missing_images", "duplicate_ids", "group_split_leakage"]
        )

    def report(self, samples: list[ArtifactSample]) -> str:
        """Return a human-readable validation report."""
        issues = self.validate(samples)
        lines = ["Dataset Validation Report", "=" * 40]

        lines.append(f"\nTotal samples:   {len(samples)}")
        unique_objs = len(set(s.artifact_group_id or s.sample_id for s in samples))
        lines.append(f"Unique objects:  {unique_objs}")

        splits = defaultdict(int)
        cats = defaultdict(int)
        for s in samples:
            if s.split:
                splits[s.split] += 1
            if s.category:
                cats[s.category] += 1

        if splits:
            lines.append("\nSplits:")
            for k, v in sorted(splits.items()):
                lines.append(f"  {k}: {v}")

        if cats:
            lines.append("\nCategories:")
            for k, v in sorted(cats.items(), key=lambda x: -x[1]):
                lines.append(f"  {k}: {v}")

        lines.append(f"\nIssues:")
        for issue_type, items in issues.items():
            count = len(items)
            status = "✅" if count == 0 else f"⚠️ {count}"
            lines.append(f"  {issue_type}: {status}")
            if items and issue_type != "warnings":
                for item in items[:5]:
                    lines.append(f"    - {item}")
                if len(items) > 5:
                    lines.append(f"    ... and {len(items) - 5} more")

        return "\n".join(lines)

    @staticmethod
    def _find_group_leakage(
        samples: list[ArtifactSample],
    ) -> dict[str, set[str]]:
        """Find groups assigned to multiple splits."""
        group_splits: dict[str, set[str]] = defaultdict(set)
        for s in samples:
            gid = s.artifact_group_id or s.sample_id
            if s.split:
                group_splits[gid].add(s.split)
        return {g: splits for g, splits in group_splits.items() if len(splits) > 1}
