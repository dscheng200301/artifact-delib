"""Data leakage detector — exact/visual duplicates, label leakage, image corruption.

Protects model evaluation integrity by detecting leakage before training.
Never reveals test labels to inference code — all checks operate on dataset
metadata only.
"""

from __future__ import annotations

import hashlib
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from artifact_delib.schemas import ArtifactSample

logger = logging.getLogger(__name__)


# ============================================================================
# Return-type dataclasses
# ============================================================================


@dataclass(frozen=True)
class DuplicateGroup:
    """A group of samples whose image files have identical SHA-256 hashes.

    Attributes:
        hash_value: The shared hex digest.
        sample_ids: All sample IDs that share this image.
        representative_path: One of the image paths (for diagnostics).
    """

    hash_value: str
    sample_ids: tuple[str, ...]
    representative_path: str


@dataclass(frozen=True)
class NearDuplicatePair:
    """Two images whose perceptual hashes differ by a small Hamming distance.

    Attributes:
        sample_id_a: First sample ID.
        sample_id_b: Second sample ID.
        hamming_distance: Number of differing bits between aHashes (0--64).
    """

    sample_id_a: str
    sample_id_b: str
    hamming_distance: int


@dataclass(frozen=True)
class FilenameLeakage:
    """A sample whose filename contains category/type/period keywords.

    This is a red flag: models may exploit the filename as a shortcut
    instead of learning visual features.

    Attributes:
        sample_id: The sample with a suspicious filename.
        filename: The filename (stem only, no extension).
        matched_keywords: Which keywords were found in the filename.
    """

    sample_id: str
    filename: str
    matched_keywords: tuple[str, ...]


@dataclass(frozen=True)
class CorruptImage:
    """An image file that cannot be opened or decoded.

    Attributes:
        sample_id: The owning sample.
        image_path: Absolute or relative path to the corrupt file.
        error: Exception message explaining the failure.
    """

    sample_id: str
    image_path: str
    error: str


@dataclass(frozen=True)
class CategoryDistribution:
    """Per-split counts across category, fine-grained type, and period axes.

    Attributes:
        split: Dataset split name (train / validation / test / unassigned).
        total_samples: Number of samples in this split.
        category_counts: ``{category_label: count}``.
        type_counts: ``{fine_grained_type_label: count}``.
        period_counts: ``{period_label: count}``.
    """

    split: str
    total_samples: int
    category_counts: dict[str, int] = field(default_factory=dict)
    type_counts: dict[str, int] = field(default_factory=dict)
    period_counts: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class LabelCoverageResult:
    """Whether every test-set label also appears in the training set.

    Attributes:
        missing_test_labels: Test labels never seen in training.
        train_label_count: Unique labels in train split.
        test_label_count: Unique labels in test split.
        is_clean: ``True`` iff every test label exists in training.
    """

    missing_test_labels: tuple[str, ...]
    train_label_count: int
    test_label_count: int
    is_clean: bool


@dataclass(frozen=True)
class LeakageReport:
    """Aggregated result of all leakage detection checks.

    Attributes:
        exact_duplicates: Groups of byte-identical images.
        near_duplicates: Pairs of perceptually near-identical images.
        filename_leaks: Samples with label keywords in their filename.
        corrupt_images: Images that failed to open.
        category_distributions: Per-split label counts.
        label_coverage: Result of train→test label coverage check
            (``None`` when splits are absent).
        summary: Human-readable one-line summary.
    """

    exact_duplicates: tuple[DuplicateGroup, ...]
    near_duplicates: tuple[NearDuplicatePair, ...]
    filename_leaks: tuple[FilenameLeakage, ...]
    corrupt_images: tuple[CorruptImage, ...]
    category_distributions: tuple[CategoryDistribution, ...]
    label_coverage: LabelCoverageResult | None
    summary: str


# ============================================================================
# Static keyword sets for filename leakage detection
# ============================================================================

# Common artifact type / shape keywords likely to appear in museum filenames.
_TYPE_KEYWORDS: frozenset[str] = frozenset({
    # General vessel forms
    "vase", "jar", "bowl", "plate", "dish", "cup", "bottle", "flask",
    "ewer", "pitcher", "basin", "urn", "pot", "beaker", "goblet",
    "chalice", "amphora", "krater", "hydria", "lekythos", "kylix",
    "oinochoe", "pyxis", "rhyton", "aryballos", "alabastron", "askos",
    "stamnos", "pelike", "dinos", "lebes", "loutrophoros", "psykter",
    # Chinese bronze vessel types
    "ding", "gui", "hu", "zun", "gu", "jue", "jia", "he", "you",
    "pan", "dou", "fu", "yan", "bu", "lei", "pou", "gong", "fangyi",
    "fangding", "li", "xu", "zhi", "ling", "dou",
    # Sculpture & figures
    "figure", "figurine", "statue", "statuette", "bust", "sculpture",
    "relief", "stele", "plaque", "tablet", "fragment",
    # Flat / painting
    "scroll", "painting", "album", "fan", "screen", "hanging",
    "handscroll", "manuscript", "illumination", "folio",
    # Containers & boxes
    "box", "casket", "chest", "coffer", "case",
    # Weapons & tools
    "sword", "blade", "dagger", "knife", "axe", "spear", "halberd",
    "arrowhead", "helmet", "armor", "shield",
    # Mirrors & personal items
    "mirror", "comb", "pendant", "bead", "ornament", "buckle",
    "brooch", "ring", "bracelet", "necklace", "earring", "crown",
    "diadem", "tiara",
    # Textiles
    "textile", "garment", "robe", "tapestry", "embroidery", "carpet",
    "rug", "costume", "dress", "shawl", "mantle", "tunic",
    # Furniture & architectural
    "throne", "chair", "table", "stool", "bed", "cabinet", "stand",
    "pedestal", "column", "capital", "frieze", "architrave",
    # Miscellaneous
    "lid", "cover", "handle", "spout", "rim", "base", "foot",
    "sherd", "shard", "medallion", "coin", "seal", "stamp",
    "incense", "censer", "burner", "lamp", "candlestick",
    "tile", "brick", "mural", "fresco", "mosaic",
})

# Period / dynasty / culture keywords.
_PERIOD_KEYWORDS: frozenset[str] = frozenset({
    # Chinese
    "neolithic", "shang", "zhou", "qin", "han", "sanguo",
    "jin", "sui", "tang", "song", "liao", "xia", "yuan",
    "ming", "qing", "republican",
    "yangshao", "longshan", "liangzhu", "erlitou", "sanxingdui",
    "western zhou", "eastern zhou", "spring and autumn",
    "warring states", "western han", "eastern han",
    "three kingdoms", "northern and southern",
    "northern wei", "southern dynasties",
    "northern song", "southern song",
    "jurchen", "khitan",
    # Japanese
    "jomon", "yayoi", "kofun", "asuka", "nara", "heian",
    "kamakura", "muromachi", "momoyama", "edo", "meiji",
    # Korean
    "goryeo", "joseon", "silla", "baekje", "goguryeo",
    # South / Southeast Asian
    "gupta", "maurya", "kushan", "gandhara", "pala", "chola",
    "vijayanagara", "mughal", "angkor", "khmer", "champa",
    "srivijaya", "majapahit", "sukhothai", "ayutthaya",
    # West / Central Asian
    "sumerian", "akkadian", "babylonian", "assyrian", "hittite",
    "urartian", "phrygian", "lydian", "median",
    "achaemenid", "parthian", "sasanian", "seleucid",
    "elamite", "bactrian", "sogdian",
    # Mediterranean / European
    "minoan", "mycenaean", "cycladic",
    "archaic", "classical", "hellenistic",
    "etruscan", "roman", "byzantine",
    "early christian", "migration period",
    "carolingian", "ottonian", "romanesque", "gothic",
    "renaissance", "baroque", "rococo", "neoclassical",
    # Pre-Columbian Americas
    "olmec", "maya", "aztec", "toltec", "mixtec", "zapotec",
    "inca", "moche", "nazca", "chimu", "chavin", "wai",
    "taino", "muisca", "chimor",
    # Other
    "bronze age", "iron age", "stone age", "chalcolithic",
    "paleolithic", "mesolithic",
    "punic", "phoenician", "carthaginian",
    "nabataean", "palmyrene",
    "coptic", "nubian", "aksumite",
    "benin", "ife", "yoruba", "akan", "dogon", "bamana",
    "islamic", "umayyad", "abbasid", "fatimid", "seljuk",
    "timurid", "safavid", "qajar", "ottoman", "mamluk",
    "northwest coast", "mississippian", "ancestral pueblo",
})

# Broad material / category keywords.
_MATERIAL_KEYWORDS: frozenset[str] = frozenset({
    "ceramic", "porcelain", "pottery", "earthenware", "stoneware",
    "terracotta", "bronze", "brass", "copper", "iron", "steel",
    "gold", "silver", "electrum", "lead", "pewter", "tin",
    "jade", "nephrite", "jadeite", "serpentine",
    "marble", "limestone", "sandstone", "granite", "alabaster",
    "basalt", "diorite", "schist", "soapstone", "steatite",
    "obsidian", "flint", "chert", "quartz", "crystal",
    "lapis", "turquoise", "carnelian", "agate", "amethyst",
    "wood", "bamboo", "lacquer", "lacquerware",
    "ivory", "bone", "horn", "antler", "shell", "coral",
    "glass", "enamel", "cloisonne", "faience",
    "textile", "silk", "cotton", "wool", "linen", "hemp",
    "paper", "ink", "pigment",
    "stone", "clay", "metal", "resin", "plaster", "stucco",
    # Category-like terms
    "sculpture", "painting", "calligraphy", "print",
    "ceramics", "metalwork", "glassware", "textiles",
    "furniture", "armor", "weaponry", "jewelry",
    "manuscript", "book", "codex",
})

# Combined set for fast membership testing.
_ALL_KEYWORDS: frozenset[str] = _TYPE_KEYWORDS | _PERIOD_KEYWORDS | _MATERIAL_KEYWORDS


# ============================================================================
# LeakageDetector
# ============================================================================


class LeakageDetector:
    """Detect six categories of data leakage in artifact datasets.

    This class is entirely stateless — every method is ``@staticmethod``.
    It is safe to call concurrently from multiple threads.

    **Checks performed** (see individual method docstrings for details):

    1. :meth:`detect_exact_duplicates` — SHA-256 byte-identical images.
    2. :meth:`detect_near_duplicates` — perceptual (aHash) near-duplicates.
    3. :meth:`detect_filename_leakage` — label keywords in filenames.
    4. :meth:`detect_corrupt_images` — unreadable / truncated image files.
    5. :meth:`compute_category_distribution` — per-split label histograms.
    6. :meth:`check_label_coverage` — test labels that never appear in train.

    Usage::

        from artifact_delib.data.leakage_detector import LeakageDetector

        samples: list[ArtifactSample] = ...
        report = LeakageDetector.run_all(samples)
        print(report.summary)

        # Or run individual checks:
        dups = LeakageDetector.detect_exact_duplicates(samples)
    """

    # ------------------------------------------------------------------
    # 1. SHA-256 exact duplicate detection
    # ------------------------------------------------------------------

    @staticmethod
    def detect_exact_duplicates(
        samples: list[ArtifactSample],
    ) -> list[DuplicateGroup]:
        """Find samples whose image files are byte-for-byte identical.

        Computes SHA-256 over the full file content.  Only groups with
        >1 member are returned.

        Args:
            samples: Dataset samples with populated ``image_path`` fields.

        Returns:
            List of :class:`DuplicateGroup`, one per distinct hash that
            appears more than once.  Empty list if all images are unique.

        Example::

            dups = LeakageDetector.detect_exact_duplicates(samples)
            for g in dups:
                print(f"Hash {g.hash_value[:12]}... → {list(g.sample_ids)}")
        """
        hash_to_samples: dict[str, list[str]] = defaultdict(list)
        hash_to_path: dict[str, str] = {}

        for s in samples:
            path = s.image_path
            if not path.exists():
                continue
            try:
                file_hash = LeakageDetector._sha256_file(path)
            except (OSError, PermissionError) as exc:
                logger.warning("Cannot hash %s: %s", path, exc)
                continue
            hash_to_samples[file_hash].append(s.sample_id)
            if file_hash not in hash_to_path:
                hash_to_path[file_hash] = str(path)

        return [
            DuplicateGroup(
                hash_value=h,
                sample_ids=tuple(sorted(ids)),
                representative_path=hash_to_path[h],
            )
            for h, ids in hash_to_samples.items()
            if len(ids) > 1
        ]

    # ------------------------------------------------------------------
    # 2. Perceptual hash near-duplicate detection
    # ------------------------------------------------------------------

    @staticmethod
    def detect_near_duplicates(
        samples: list[ArtifactSample],
        threshold: int = 5,
    ) -> list[NearDuplicatePair]:
        """Find near-duplicate images using average-hash (aHash).

        Resizes each image to 8x8 grayscale, computes the mean pixel
        value, and builds a 64-bit hash where each bit indicates whether
        the corresponding pixel is above the mean.

        Pairs whose Hamming distance is ≤ *threshold* (default 5) are
        returned as near-duplicates.

        If Pillow is not installed this method logs a warning and
        returns an empty list.

        Args:
            samples: Dataset samples.
            threshold: Maximum Hamming distance to consider a pair
                near-duplicate (range 0--64, default 5).

        Returns:
            List of :class:`NearDuplicatePair`.  Only pairs where
            ``sample_id_a < sample_id_b`` (lexicographically) are
            included to avoid duplicates.

        Example::

            near = LeakageDetector.detect_near_duplicates(samples, threshold=3)
            for p in near:
                print(f"{p.sample_id_a} ≈ {p.sample_id_b} (dist={p.hamming_distance})")
        """
        try:
            from PIL import Image  # noqa: F811 — lazy import
        except ImportError:
            logger.warning(
                "Pillow not installed — skipping perceptual hash near-duplicate detection."
            )
            return []

        # Compute aHash for every readable image.
        hashes: list[tuple[str, int]] = []  # (sample_id, 64-bit int hash)
        for s in samples:
            if not s.image_path.exists():
                continue
            try:
                ahash_int = LeakageDetector._compute_ahash_int(s.image_path)
            except Exception as exc:
                logger.debug("aHash failed for %s: %s", s.sample_id, exc)
                continue
            hashes.append((s.sample_id, ahash_int))

        # Pairwise comparison — O(n²).  For large datasets consider
        # locality-sensitive hashing; this is fine for < 10k samples.
        pairs: list[NearDuplicatePair] = []
        n = len(hashes)
        for i in range(n):
            id_a, hash_a = hashes[i]
            for j in range(i + 1, n):
                id_b, hash_b = hashes[j]
                dist = LeakageDetector._hamming_distance(hash_a, hash_b)
                if dist <= threshold:
                    # Lexicographic order for determinism.
                    if id_a < id_b:
                        pairs.append(
                            NearDuplicatePair(
                                sample_id_a=id_a,
                                sample_id_b=id_b,
                                hamming_distance=dist,
                            )
                        )
                    else:
                        pairs.append(
                            NearDuplicatePair(
                                sample_id_a=id_b,
                                sample_id_b=id_a,
                                hamming_distance=dist,
                            )
                        )

        return pairs

    # ------------------------------------------------------------------
    # 3. Filename label leakage check
    # ------------------------------------------------------------------

    @staticmethod
    def detect_filename_leakage(
        samples: list[ArtifactSample],
    ) -> list[FilenameLeakage]:
        """Detect filenames that contain category/type/period/material keywords.

        Checks each filename (stem only, lower-cased) against:

        1. A static curated vocabulary of artifact-type, period, and
           material keywords (``_TYPE_KEYWORDS``, ``_PERIOD_KEYWORDS``,
           ``_MATERIAL_KEYWORDS``).
        2. The sample's own label values — if a label string appears
           literally in the filename it is flagged as well.

        Args:
            samples: Dataset samples.

        Returns:
            List of :class:`FilenameLeakage` entries.  Samples whose
            filenames contain **no** keywords are omitted.

        Example::

            leaks = LeakageDetector.detect_filename_leakage(samples)
            for lk in leaks:
                print(f"{lk.sample_id}: '{lk.filename}' matched {lk.matched_keywords}")
        """
        results: list[FilenameLeakage] = []

        # Build a combined set: static keywords + all label values in the dataset.
        dynamic_keywords: set[str] = set()
        for s in samples:
            for attr in (
                "category",
                "fine_grained_type",
                "period",
                "dynasty",
                "material",
                "craft",
                "region",
            ):
                val = getattr(s, attr, None)
                if val and isinstance(val, str):
                    dynamic_keywords.add(val.lower())

        all_keywords = _ALL_KEYWORDS | frozenset(dynamic_keywords)

        for s in samples:
            stem = s.image_path.stem.lower()
            # Tokenize the stem on common delimiters to avoid substring
            # false positives (e.g. "handle" matching "panhandle").
            tokens = set(stem.replace("_", " ").replace("-", " ").replace(".", " ").split())
            # Also check the raw stem for multi-word keywords.
            matched: set[str] = set()

            for kw in all_keywords:
                kw_lower = kw.lower()
                # Multi-word keywords checked against full stem.
                if " " in kw_lower:
                    if kw_lower in stem:
                        matched.add(kw)
                else:
                    # Single-word keywords checked against token set.
                    if kw_lower in tokens:
                        matched.add(kw)

            if matched:
                results.append(
                    FilenameLeakage(
                        sample_id=s.sample_id,
                        filename=s.image_path.stem,
                        matched_keywords=tuple(sorted(matched)),
                    )
                )

        return results

    # ------------------------------------------------------------------
    # 4. Image corruption check
    # ------------------------------------------------------------------

    @staticmethod
    def detect_corrupt_images(
        samples: list[ArtifactSample],
    ) -> list[CorruptImage]:
        """Verify that every sample's image can be opened and decoded.

        Uses Pillow to attempt loading each image.  Checks both
        ``Image.open()`` (header parse) and ``.verify()`` /
        ``.getdata()`` (full decode).

        If Pillow is not installed this method logs a warning and
        returns an empty list.

        Args:
            samples: Dataset samples.

        Returns:
            List of :class:`CorruptImage` — one per file that raised an
            exception during open/decode.  Empty list if all images are
            readable.

        Example::

            bad = LeakageDetector.detect_corrupt_images(samples)
            for c in bad:
                print(f"CORRUPT {c.sample_id}: {c.error}")
        """
        try:
            from PIL import Image, UnidentifiedImageError  # noqa: F811
        except ImportError:
            logger.warning(
                "Pillow not installed — skipping image corruption check."
            )
            return []

        results: list[CorruptImage] = []

        for s in samples:
            path = s.image_path
            if not path.exists():
                results.append(
                    CorruptImage(
                        sample_id=s.sample_id,
                        image_path=str(path),
                        error="File not found",
                    )
                )
                continue

            try:
                with Image.open(path) as img:
                    img.verify()
            except UnidentifiedImageError:
                # verify() may pass for some broken images; try loading pixel data.
                try:
                    with Image.open(path) as img:
                        img.load()
                except Exception as exc2:
                    results.append(
                        CorruptImage(
                            sample_id=s.sample_id,
                            image_path=str(path),
                            error=str(exc2),
                        )
                    )
            except Exception as exc:
                results.append(
                    CorruptImage(
                        sample_id=s.sample_id,
                        image_path=str(path),
                        error=str(exc),
                    )
                )

        return results

    # ------------------------------------------------------------------
    # 5. Category distribution statistics
    # ------------------------------------------------------------------

    @staticmethod
    def compute_category_distribution(
        samples: list[ArtifactSample],
    ) -> list[CategoryDistribution]:
        """Count samples per category, fine-grained type, and period
        within each dataset split.

        Args:
            samples: Dataset samples.

        Returns:
            List of :class:`CategoryDistribution`, one per distinct
            ``split`` value found in the dataset, sorted by split name.

        Example::

            dists = LeakageDetector.compute_category_distribution(samples)
            for d in dists:
                print(f"{d.split}: {d.total_samples} samples, "
                      f"{len(d.category_counts)} categories")
        """
        # Group samples by split.
        split_samples: dict[str, list[ArtifactSample]] = defaultdict(list)
        for s in samples:
            split_key = s.split or "unassigned"
            split_samples[split_key].append(s)

        results: list[CategoryDistribution] = []
        for split_name in sorted(split_samples.keys()):
            group = split_samples[split_name]
            cat_counts: Counter[str] = Counter()
            type_counts: Counter[str] = Counter()
            period_counts: Counter[str] = Counter()

            for s in group:
                if s.category:
                    cat_counts[s.category] += 1
                if s.fine_grained_type:
                    type_counts[s.fine_grained_type] += 1
                if s.period:
                    period_counts[s.period] += 1

            results.append(
                CategoryDistribution(
                    split=split_name,
                    total_samples=len(group),
                    category_counts=dict(cat_counts),
                    type_counts=dict(type_counts),
                    period_counts=dict(period_counts),
                )
            )

        return results

    # ------------------------------------------------------------------
    # 6. Label coverage check
    # ------------------------------------------------------------------

    @staticmethod
    def check_label_coverage(
        samples: list[ArtifactSample],
    ) -> LabelCoverageResult | None:
        """Verify that every test-set category label appears in training.

        This is a critical leakage check: if a test label never appears
        in training the model cannot possibly predict it correctly, yet
        evaluation metrics would misleadingly report it as a failure.

        Labels are taken from the ``category`` field.  If no samples
        have ``split="test"`` or ``split="train"`` this method returns
        ``None``.

        Args:
            samples: Dataset samples with ``split`` and ``category``
                populated.

        Returns:
            :class:`LabelCoverageResult` or ``None`` if splits are absent.

        Example::

            cov = LeakageDetector.check_label_coverage(samples)
            if cov and not cov.is_clean:
                print(f"Missing labels: {cov.missing_test_labels}")
        """
        train_labels: set[str] = set()
        test_labels: set[str] = set()

        for s in samples:
            if not s.category:
                continue
            if s.split == "train":
                train_labels.add(s.category)
            elif s.split == "test":
                test_labels.add(s.category)

        if not test_labels:
            logger.info(
                "No test-split samples found — label coverage check skipped."
            )
            return None

        if not train_labels:
            logger.warning(
                "No train-split samples found — cannot verify label coverage."
            )
            return None

        missing = test_labels - train_labels

        return LabelCoverageResult(
            missing_test_labels=tuple(sorted(missing)),
            train_label_count=len(train_labels),
            test_label_count=len(test_labels),
            is_clean=len(missing) == 0,
        )

    # ------------------------------------------------------------------
    # Aggregated run_all
    # ------------------------------------------------------------------

    @classmethod
    def run_all(cls, samples: list[ArtifactSample]) -> LeakageReport:
        """Run all six leakage checks and return an aggregated report.

        This is the recommended entry point for comprehensive leakage
        detection before training or evaluation.

        Args:
            samples: Full dataset (all splits).

        Returns:
            :class:`LeakageReport` with results from every check.

        Example::

            report = LeakageDetector.run_all(samples)
            if report.exact_duplicates:
                print("WARNING: exact duplicates found!")
            print(report.summary)
        """
        exact = cls.detect_exact_duplicates(samples)
        near = cls.detect_near_duplicates(samples)
        leaks = cls.detect_filename_leakage(samples)
        corrupt = cls.detect_corrupt_images(samples)
        dists = cls.compute_category_distribution(samples)
        coverage = cls.check_label_coverage(samples)

        # Build human-readable summary.
        parts: list[str] = []
        if exact:
            parts.append(f"{len(exact)} exact duplicate group(s)")
        if near:
            parts.append(f"{len(near)} near-duplicate pair(s)")
        if leaks:
            parts.append(f"{len(leaks)} filename leakage(s)")
        if corrupt:
            parts.append(f"{len(corrupt)} corrupt image(s)")

        has_issues = bool(exact or near or leaks or corrupt)
        if coverage is not None and not coverage.is_clean:
            parts.append(
                f"{len(coverage.missing_test_labels)} unseen test label(s)"
            )
            has_issues = True

        summary = (
            f"Leakage checks: {'ISSUES FOUND' if has_issues else 'ALL CLEAN'} — "
            + "; ".join(parts)
            if parts
            else "Leakage checks: ALL CLEAN — no issues detected."
        )

        return LeakageReport(
            exact_duplicates=tuple(exact),
            near_duplicates=tuple(near),
            filename_leaks=tuple(leaks),
            corrupt_images=tuple(corrupt),
            category_distributions=tuple(dists),
            label_coverage=coverage,
            summary=summary,
        )

    # ==================================================================
    # Internal helpers
    # ==================================================================

    @staticmethod
    def _sha256_file(path: Path, chunk_size: int = 1 << 16) -> str:
        """Compute the SHA-256 hex digest of a file.

        Reads in *chunk_size* blocks to keep memory bounded for large
        images (default 64 KiB).
        """
        sha256 = hashlib.sha256()
        with open(path, "rb") as fh:
            while True:
                chunk = fh.read(chunk_size)
                if not chunk:
                    break
                sha256.update(chunk)
        return sha256.hexdigest()

    @staticmethod
    def _compute_ahash_int(path: Path) -> int:
        """Compute the average-hash (aHash) of an image as a 64-bit integer.

        Algorithm:
            1. Open image, convert to grayscale, resize to 8x8.
            2. Compute mean pixel value.
            3. Build 64-bit integer where bit i == 1 iff pixel[i] > mean.

        Raises:
            OSError, ``UnidentifiedImageError``, etc. — propagated to
            caller for graceful handling.
        """
        from PIL import Image  # noqa: F811 — lazy import

        with Image.open(path) as img:
            gray = img.convert("L").resize((8, 8), Image.LANCZOS)
            pixels: list[int] = list(gray.getdata())
            if len(pixels) != 64:  # pragma: no cover — defensive
                raise ValueError(f"Expected 64 pixels, got {len(pixels)}")

        avg = sum(pixels) / 64.0
        bits_int = 0
        for i, p in enumerate(pixels):
            if p > avg:
                bits_int |= 1 << (63 - i)  # MSB first for consistent ordering
        return bits_int

    @staticmethod
    def _hamming_distance(a: int, b: int) -> int:
        """Count the number of set bits in ``a XOR b`` (both are 64-bit).

        Uses ``int.bit_count()`` (Python 3.8+).
        """
        return (a ^ b).bit_count()
