"""Met Museum Open Access concurrent downloader.

v2: Fetches image URLs via the Met Collection API, then downloads images
with asyncio concurrency. Compatible with the GitHub CSV (no direct image URLs).

Flow: CSV filter → API fetch (get image URLs) → concurrent image download
CC0 license — free for research and publication.
"""

from __future__ import annotations

import asyncio
import csv
import json
import ssl
import time
from pathlib import Path

import httpx

from artifact_delib.schemas import ArtifactSample

# ── URLs ──
_MET_CSV_URLS = [
    "https://raw.githubusercontent.com/metmuseum/openaccess/master/MetObjects.csv",
    "https://github.com/metmuseum/openaccess/raw/master/MetObjects.csv",
]
_MET_API_OBJECT = "https://collectionapi.metmuseum.org/public/collection/v1/objects/{}"

# ── Default headers (Met API requires a User-Agent, otherwise returns 403) ──
_DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json,image/*",
}

# ── Chinese culture filter ──
_CHINESE_CULTURES = {
    "chin", "china", "chinese", "chinese|japanese", "manchu",
    "central asia, china", "manchu china",
}

# ── Dynasty mapping ──
_DYNASTY_MAP = [
    ("neolithic period", "新石器时代"), ("liangzhu", "良渚文化"),
    ("shang dynasty", "商"), ("shang", "商"),
    ("zhou dynasty", "周"), ("western zhou", "西周"), ("eastern zhou", "东周"),
    ("spring and autumn", "春秋"), ("warring states", "战国"),
    ("qin dynasty", "秦"), ("qin", "秦"),
    ("han dynasty", "汉"), ("western han", "西汉"), ("eastern han", "东汉"),
    ("three kingdoms", "三国"), ("six dynasties", "六朝"),
    ("jin dynasty", "晋"), ("northern wei", "北魏"),
    ("tang dynasty", "唐"), ("tang", "唐"),
    ("five dynasties", "五代"), ("liao dynasty", "辽"),
    ("song dynasty", "宋"), ("northern song", "北宋"), ("southern song", "南宋"),
    ("jin dynasty", "金"), ("yuan dynasty", "元"), ("yuan", "元"),
    ("ming dynasty", "明"), ("ming", "明"),
    ("qing dynasty", "清"), ("qing", "清"),
]

# ── Classification → category ──
_CLASS_MAP = {
    "ceramics": "瓷器", "bronzes": "青铜器", "jades": "玉器", "jade": "玉器",
    "lacquer": "漆器", "gold": "金银器", "silver": "金银器",
    "sculpture": "雕塑", "paintings": "绘画", "calligraphy": "书法",
    "textiles": "纺织品", "furniture": "家具",
    "arms and armor": "兵器", "metalwork": "金属器",
}

# ── Material → Chinese ──
_MATERIAL_MAP = [
    ("porcelain", "瓷"), ("stoneware", "瓷"), ("ceramic", "陶"),
    ("bronze", "青铜"), ("copper", "铜"), ("jade", "玉"),
    ("nephrite", "玉"), ("jadeite", "玉"), ("lacquer", "漆"),
    ("gold", "金"), ("silver", "银"), ("silk", "丝"),
    ("bamboo", "竹"), ("wood", "木"), ("iron", "铁"),
    ("earthenware", "陶"),
]

# ── Type keywords ──
_TYPE_MAP = [
    ("meiping", "梅瓶"), ("yuhuchun", "玉壶春瓶"), ("vase", "瓶"),
    ("bowl", "碗"), ("dish", "盘"), ("cup", "杯"), ("jar", "罐"),
    ("ewer", "壶"), ("teapot", "茶壶"), ("ding", "鼎"),
    ("gu", "觚"), ("jue", "爵"), ("zun", "尊"),
    ("bi disc", "玉璧"), ("cong", "琮"), ("mirror", "镜"),
    ("incense burner", "香炉"), ("box", "盒"), ("brush", "笔"),
    ("scroll", "卷轴"), ("figure", "人物俑"), ("buddha", "佛像"),
    ("bottle", "瓶"), ("plate", "盘"), ("stem cup", "高足杯"),
    ("pillow", "枕"), ("covered box", "盒"),
]


class MetDownloader:
    """Async concurrent Met Museum artifact downloader.

    Phase 1: Download and filter CSV for Chinese artifacts
    Phase 2: Fetch image URLs from Met Collection API (concurrent, rate-limited)
    Phase 3: Download images (concurrent)

    Usage:
        downloader = MetDownloader(Path("data/artifact"), max_objects=500, concurrency=20)
        samples = await downloader.run()
    """

    def __init__(
        self,
        output_root: Path,
        max_objects: int = 500,
        concurrency: int = 10,
        api_concurrency: int = 5,
        timeout: int = 30,
    ) -> None:
        self.output_root = output_root
        self.max_objects = max_objects
        self.concurrency = concurrency
        self.api_concurrency = api_concurrency
        self.timeout = timeout

        self.images_dir = output_root / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.csv_cache = output_root / "MetObjects.csv"
        self.cache_dir = output_root / "api_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = output_root / "met_artifact_manifest.csv"

        # Stats
        self._start_time = 0.0

    async def run(self) -> list[ArtifactSample]:
        """Full pipeline."""
        self._start_time = time.perf_counter()

        # Phase 1: Get CSV
        csv_path = await self._ensure_csv()

        # Phase 2: Filter + fetch image URLs via API
        objects = await self._fetch_objects(csv_path)

        elapsed = time.perf_counter() - self._start_time
        print(f"\n  Phase 1+2 done: {len(objects)} objects ({elapsed:.1f}s)")

        # Phase 3: Concurrent image download
        samples = await self._download_images(objects)

        # Phase 4: Save manifest
        self._save_manifest(samples)

        elapsed = time.perf_counter() - self._start_time
        print(f"\n  Done: {len(samples)} samples in {elapsed:.1f}s")
        return samples

    async def _ensure_csv(self) -> Path:
        """Download CSV if not cached."""
        if self.csv_cache.exists():
            sz = self.csv_cache.stat().st_size / 1e6
            print(f"  CSV cached: {sz:.0f} MB")
            return self.csv_cache

        print("  Downloading MetObjects.csv (~50MB uncompressed)...")
        async with httpx.AsyncClient(timeout=300, follow_redirects=True, headers=_DEFAULT_HEADERS) as c:
            for url in _MET_CSV_URLS:
                try:
                    r = await c.get(url)
                    if r.status_code == 200 and len(r.content) > 100000:
                        self.csv_cache.write_bytes(r.content)
                        print(f"    Downloaded: {len(r.content)/1e6:.0f} MB")
                        return self.csv_cache
                except Exception as e:
                    print(f"    Failed: {e}")
        raise RuntimeError("Cannot download CSV")

    async def _fetch_objects(self, csv_path: Path) -> list[dict]:
        """Filter CSV + fetch image URLs from API (with local cache).

        Collects all Chinese artifact IDs from CSV, then queries the Met API
        in batches (stopping early once we have enough objects with images).
        """
        # Step A: Collect ALL Chinese artifact object IDs from CSV
        print("  Collecting Chinese artifact IDs from CSV...")
        all_ids = []
        seen = set()
        with csv_path.open(newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                if (row.get("Is Public Domain") or "").strip().lower() != "true":
                    continue
                culture = (row.get("Culture") or "").strip().lower()
                if not self._is_chinese(culture):
                    continue
                oid = (row.get("Object ID") or "").strip()
                if not oid or oid in seen:
                    continue
                seen.add(oid)
                all_ids.append(oid)

        print(f"    Total Chinese artifacts: {len(all_ids)}")

        # Step B: Fetch image URLs from API in batches, stop when we have enough
        print(f"    Fetching image URLs (api_concurrency={self.api_concurrency})...")
        sem = asyncio.Semaphore(self.api_concurrency)
        fetched = 0
        objects = []
        # Process in chunks of 2000 to avoid huge asyncio.gather
        batch_size = 2000

        async def fetch_one(oid: str) -> dict | None:
            nonlocal fetched
            async with sem:
                cache_file = self.cache_dir / f"{oid}.json"
                if cache_file.exists():
                    data = json.loads(cache_file.read_text())
                    if data.get("primaryImage") or data.get("primaryImageSmall"):
                        return data
                    return None

                # Fetch from API with retry + exponential backoff
                for attempt in range(3):
                    try:
                        async with httpx.AsyncClient(
                            timeout=self.timeout, follow_redirects=True,
                            headers=_DEFAULT_HEADERS,
                        ) as c:
                            r = await c.get(_MET_API_OBJECT.format(oid))
                            fetched += 1

                            if r.status_code == 200 and len(r.content) > 100:
                                try:
                                    data = r.json()
                                except Exception:
                                    await asyncio.sleep(0.5)
                                    continue
                                cache_file.write_text(json.dumps(data, ensure_ascii=False))
                                if data.get("primaryImage") or data.get("primaryImageSmall"):
                                    return data
                                return None  # No image, don't retry
                            elif r.status_code == 429:
                                wait = 2.0 * (2 ** attempt)
                                await asyncio.sleep(wait)
                            elif r.status_code == 403:
                                await asyncio.sleep(1.0)
                            else:
                                await asyncio.sleep(0.5)
                    except (httpx.TimeoutException, httpx.ConnectError, httpx.HTTPError):
                        await asyncio.sleep(2.0 * (2 ** attempt))
                return None

        # Process in batches to avoid fetching all 12k+ IDs
        for batch_start in range(0, len(all_ids), batch_size):
            if len(objects) >= self.max_objects:
                break
            batch = all_ids[batch_start:batch_start + batch_size]
            tasks = [fetch_one(oid) for oid in batch]
            results = await asyncio.gather(*tasks)
            for r in results:
                if r is not None:
                    objects.append(r)
                    if len(objects) >= self.max_objects:
                        break

        objects = objects[:self.max_objects]
        print(f"    Got {len(objects)} objects with images (made {fetched} API calls)")
        return objects

    async def _download_images(self, objects: list[dict]) -> list[ArtifactSample]:
        """Download images concurrently and build ArtifactSample list."""
        existing = set(f.name for f in self.images_dir.glob("*.jpg"))
        sem = asyncio.Semaphore(self.concurrency)
        samples = []
        lock = asyncio.Lock()
        downloaded = [0]
        skipped = [0]
        failed = [0]
        last = [0]

        async def download_one(obj: dict) -> ArtifactSample | None:
            async with sem:
                oid = str(obj.get("objectID", ""))
                fname = f"met-{oid}.jpg"
                path = self.images_dir / fname

                # Download if not exists
                if fname not in existing:
                    url = obj.get("primaryImage") or obj.get("primaryImageSmall", "")
                    if not url:
                        skipped[0] += 1
                        return None
                    try:
                        async with httpx.AsyncClient(
                            timeout=self.timeout, follow_redirects=True,
                            headers=_DEFAULT_HEADERS,
                        ) as c:
                            r = await c.get(url)
                            if r.status_code == 200 and len(r.content) > 1000:
                                path.write_bytes(r.content)
                                downloaded[0] += 1
                                existing.add(fname)
                            else:
                                skipped[0] += 1
                                return None
                    except Exception:
                        failed[0] += 1
                        return None
                else:
                    downloaded[0] += 1  # already cached

                # Build ArtifactSample
                culture = (obj.get("culture") or "").strip()
                dynasty = (obj.get("dynasty") or "").strip()
                period, dyn = self._extract_period(obj)

                sample = ArtifactSample(
                    sample_id=f"met-{oid}",
                    image_path=path,
                    category=self._extract_category(obj),
                    fine_grained_type=self._extract_type(obj),
                    period=period,
                    dynasty=dyn,
                    material=self._extract_material(obj),
                    craft=self._extract_craft(obj),
                    region=self._extract_region(obj),
                    source="Met Museum (CC0)",
                    split="unassigned",
                    artifact_group_id=oid,
                )

                # Progress
                async with lock:
                    samples.append(sample)
                    total = downloaded[0] + skipped[0] + failed[0]
                    if total - last[0] >= 50:
                        last[0] = total
                        e = time.perf_counter() - self._start_time
                        print(f"    [{total}/{len(objects)}] "
                              f"ok={downloaded[0]} skip={skipped[0]} "
                              f"fail={failed[0]} ({e:.0f}s)")

                return sample

        tasks = [download_one(obj) for obj in objects]
        await asyncio.gather(*tasks)

        return [s for s in samples if s is not None]

    def _save_manifest(self, samples: list[ArtifactSample]) -> None:
        with self.manifest_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "sample_id", "image_path", "category", "fine_grained_type",
                "period", "dynasty", "material", "craft", "region",
                "artifact_group_id", "split", "source",
            ])
            for s in samples:
                w.writerow([
                    s.sample_id, s.image_path.name,
                    s.category or "", s.fine_grained_type or "",
                    s.period or "", s.dynasty or "",
                    s.material or "", s.craft or "", s.region or "",
                    s.artifact_group_id or "", s.split or "", s.source or "",
                ])

    # ── Helpers ──

    def _is_chinese(self, culture: str) -> bool:
        if not culture:
            return False
        c = culture.strip().lower()
        return any(ch in c for ch in _CHINESE_CULTURES)

    def _extract_category(self, obj: dict) -> str | None:
        cls = (obj.get("classification") or "").strip().lower()
        dept = (obj.get("department") or "").strip().lower()
        combined = f"{dept} {cls}"
        for key, cat in _CLASS_MAP.items():
            if key in combined:
                return cat
        return None

    def _extract_type(self, obj: dict) -> str | None:
        title = (obj.get("title") or "").strip().lower()
        cls = (obj.get("classification") or "").strip().lower()
        oname = (obj.get("objectName") or "").strip().lower()
        combined = f"{title} {cls} {oname}"
        found = []
        for key, cn in _TYPE_MAP:
            if key in combined:
                found.append(key)
        if found:
            longest = max(found, key=len)
            for k, v in _TYPE_MAP:
                if k == longest:
                    return v
        return None

    def _extract_period(self, obj: dict) -> tuple[str | None, str | None]:
        period = (obj.get("period") or "").strip()
        dynasty = (obj.get("dynasty") or "").strip()
        date = (obj.get("objectDate") or "").strip()
        combined = f"{period} {dynasty} {date}".lower()
        for eng, cn in _DYNASTY_MAP:
            if eng in combined:
                p = cn
                if "early" in combined:
                    p = cn + "早期"
                elif "late" in combined:
                    p = cn + "晚期"
                return p, cn
        return None, None

    def _extract_material(self, obj: dict) -> str | None:
        medium = (obj.get("medium") or "").strip().lower()
        for key, cn in _MATERIAL_MAP:
            if key in medium:
                return cn
        return None

    def _extract_craft(self, obj: dict) -> str | None:
        medium = (obj.get("medium") or "").strip().lower()
        if "blue and white" in medium:
            return "青花"
        if "underglaze" in medium:
            return "釉下彩"
        if "glazed" in medium:
            return "施釉"
        if "cast" in medium:
            return "铸造"
        if "carved" in medium:
            return "雕刻"
        if "lacquered" in medium:
            return "髹漆"
        return None

    def _extract_region(self, obj: dict) -> str | None:
        country = (obj.get("country") or "").strip()
        if "china" in country.lower():
            region = (obj.get("region") or "").strip()
            sub = (obj.get("subregion") or "").strip()
            return sub or region or "中国"
        return country or None
