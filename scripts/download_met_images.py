"""Low-concurrency Met image downloader, category-targeted.

Rate limiting strategy (实测 2026-07-20):
- Met API 限流: ~82 请求 / 50 秒 ≈ 1.6 req/s，超过后封锁 ~10-15 秒
- 策略: 每批 70 次请求, 间隔 0.7s (~1.4 req/s), 批间暂停 15s

Optimization (2026-07-20):
- 优先下载 primaryImageSmall（缩略图 50-200KB）而非 primaryImage（原图 1-3MB）
- 文物识别任务用缩略图足够，下载速度提升 5-10x
"""
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import httpx

ROOT = Path("data/artifact")
CSV_PATH = ROOT / "MetObjects.csv"
IMAGES_DIR = ROOT / "images"
API_CACHE = ROOT / "api_cache"
MANIFEST = ROOT / "met_artifact_manifest.csv"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
API_CACHE.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json,image/*",
}

CHINESE_CULTURES = {
    "chin", "china", "chinese", "chinese|japanese", "manchu",
    "central asia, china", "manchu china",
}

# Category targets: (display_name, [CSV Classification values], target_count)
CATEGORY_TARGETS = [
    ("Ceramics",        ["Ceramics", "Ceramics-Porcelain-Export", "Ceramics-Porcelain",
                         "Ceramics-Sculpture", "Ceramics-Containers", "Ceramics-Pottery"], 1500),
    ("Jade",            ["Jade"], 1200),
    ("Paintings",       ["Paintings"], 800),
    ("Metalwork",       ["Metalwork", "Metalwork-Gilt Bronze",
                         "Metalwork-Silver In Combination", "Metalwork-Bronze",
                         "Metalwork-Coins-Inscribed"], 800),
    ("Textiles",        ["Textiles-Embroidered", "Textiles-Woven", "Textiles-Tapestries",
                         "Textiles-Rugs", "Textiles-Velvets", "Textiles-Painted",
                         "Textiles-Painted and Printed", "Textiles-Printed",
                         "Textiles", "Textiles-Costumes", "Textiles-Dyed",
                         "Textiles-Embroidered-Painted and Printed"], 800),
    ("Snuff Bottles",   ["Snuff Bottles"], 400),
    ("Sculpture",       ["Sculpture", "Sculpture-Miniature"], 300),
    ("Tomb Pottery",    ["Tomb Pottery"], 250),
    ("Lacquer",         ["Lacquer"], 200),
    ("Calligraphy",     ["Calligraphy"], 130),
    ("Enamels",         ["Enamels", "Cloisonné"], 20),
]

BATCH_SIZE = 70
BATCH_PAUSE = 15
REQ_DELAY = 0.7
IMG_REQ_DELAY = 0.5


def is_chinese(culture):
    if not culture:
        return False
    return any(ch in culture.strip().lower() for ch in CHINESE_CULTURES)


def load_csv_ids_by_category():
    cls_to_cat = {}
    for cat_name, cls_values, _ in CATEGORY_TARGETS:
        for cv in cls_values:
            cls_to_cat[cv] = cat_name

    buckets = defaultdict(list)
    seen_ids = set()
    with CSV_PATH.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if (row.get("Is Public Domain") or "").strip().lower() != "true":
                continue
            if not is_chinese((row.get("Culture") or "").strip().lower()):
                continue
            oid = (row.get("Object ID") or "").strip()
            if not oid or oid in seen_ids:
                continue
            seen_ids.add(oid)
            cls = (row.get("Classification") or "").strip()
            cat = cls_to_cat.get(cls)
            if cat is None:
                continue
            buckets[cat].append((oid, cls))
    return buckets


def load_cached_objects():
    cached = {}
    for fp in API_CACHE.glob("*.json"):
        try:
            data = json.loads(fp.read_text())
            if data.get("primaryImage") or data.get("primaryImageSmall"):
                cached[str(data.get("objectID", ""))] = data
        except Exception:
            pass
    return cached


def fetch_api_for_category(cat_name, ids_with_cls, target, already_cached, client, state):
    cached_cls = []
    needed_ids = []
    for oid, cls in ids_with_cls:
        if oid in already_cached:
            cached_cls.append((already_cached[oid], cls))
        else:
            needed_ids.append((oid, cls))

    have = len(cached_cls)
    print(f"  [{cat_name}] already cached with images: {have}/{target}", flush=True)
    if have >= target:
        return cached_cls[:target], []

    newly_cached = []
    requests_in_batch = 0
    for oid, cls in needed_ids:
        if have >= target:
            break

        if requests_in_batch > 0 and requests_in_batch % BATCH_SIZE == 0:
            state["batches"] += 1
            elapsed = time.time() - state["t0"]
            print(f"    batch {state['batches']} done | "
                  f"{state['total_cached']} cached total | {elapsed:.0f}s", flush=True)
            if have < target:
                print(f"    pausing {BATCH_PAUSE}s for rate-limit reset...", flush=True)
                time.sleep(BATCH_PAUSE)
            requests_in_batch = 0

        try:
            r = client.get(
                f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{oid}"
            )
            requests_in_batch += 1
            state["api_calls"] += 1

            if r.status_code == 200 and len(r.content) > 100:
                try:
                    data = r.json()
                except Exception:
                    data = None
                if data is not None:
                    cache_file = API_CACHE / f"{oid}.json"
                    cache_file.write_text(json.dumps(data, ensure_ascii=False))
                    if data.get("primaryImage") or data.get("primaryImageSmall"):
                        already_cached[oid] = data
                        newly_cached.append((data, cls))
                        cached_cls.append((data, cls))
                        have += 1
                        state["total_cached"] += 1
            elif r.status_code in (403, 429):
                print(f"    {r.status_code} at API call {state['api_calls']}, "
                      f"pausing {BATCH_PAUSE}s...", flush=True)
                time.sleep(BATCH_PAUSE)
                try:
                    r = client.get(
                        f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{oid}"
                    )
                    requests_in_batch += 1
                    state["api_calls"] += 1
                    if r.status_code == 200 and len(r.content) > 100:
                        try:
                            data = r.json()
                        except Exception:
                            data = None
                        if data is not None:
                            cache_file = API_CACHE / f"{oid}.json"
                            cache_file.write_text(json.dumps(data, ensure_ascii=False))
                            if data.get("primaryImage") or data.get("primaryImageSmall"):
                                already_cached[oid] = data
                                newly_cached.append((data, cls))
                                cached_cls.append((data, cls))
                                have += 1
                                state["total_cached"] += 1
                except Exception:
                    pass
        except Exception:
            time.sleep(2)

        time.sleep(REQ_DELAY)

    return cached_cls[:target], newly_cached


def download_images(all_objects_with_cls, state):
    """Download images for all objects with images. Skips existing files.

    OPTIMIZED: Uses primaryImageSmall (thumbnail 50-200KB) instead of
    primaryImage (full-res 1-3MB) for much faster downloads.
    """
    print(f"\n[2/2] Downloading images ({len(all_objects_with_cls)} objects)...", flush=True)
    downloaded = 0
    skipped = 0
    with httpx.Client(timeout=30, follow_redirects=True, headers=HEADERS) as client:
        requests_in_batch = 0
        for i, (obj, cls) in enumerate(all_objects_with_cls):
            oid = str(obj.get("objectID", ""))
            fname = f"met-{oid}.jpg"
            path = IMAGES_DIR / fname
            if path.exists() and path.stat().st_size > 1000:
                skipped += 1
                downloaded += 1
                continue
            # OPTIMIZATION: prefer primaryImageSmall (thumbnail) over primaryImage (full-res)
            # Thumbnails are 50-200KB vs 1-3MB for full-res, 5-10x faster download
            url = obj.get("primaryImageSmall") or obj.get("primaryImage", "")
            if not url:
                continue

            if requests_in_batch > 0 and requests_in_batch % BATCH_SIZE == 0:
                state["batches"] += 1
                elapsed = time.time() - state["t0"]
                print(f"    img batch {state['batches']} done | "
                      f"{downloaded} imgs | {elapsed:.0f}s", flush=True)
                print(f"    pausing {BATCH_PAUSE}s for rate-limit reset...", flush=True)
                time.sleep(BATCH_PAUSE)
                requests_in_batch = 0

            try:
                r = client.get(url)
                requests_in_batch += 1
                state["img_calls"] += 1
                if r.status_code == 200 and len(r.content) > 1000:
                    path.write_bytes(r.content)
                    downloaded += 1
                    if downloaded % 50 == 0:
                        elapsed = time.time() - state["t0"]
                        print(f"    [{downloaded} imgs] ({elapsed:.0f}s)", flush=True)
                elif r.status_code in (403, 429):
                    print(f"    CDN {r.status_code}, pausing {BATCH_PAUSE}s...", flush=True)
                    time.sleep(BATCH_PAUSE)
                    try:
                        r = client.get(url)
                        requests_in_batch += 1
                        state["img_calls"] += 1
                        if r.status_code == 200 and len(r.content) > 1000:
                            path.write_bytes(r.content)
                            downloaded += 1
                    except Exception:
                        pass
            except Exception:
                pass
            time.sleep(IMG_REQ_DELAY)

    print(f"  Images: {downloaded} downloaded ({skipped} already existed)", flush=True)
    return downloaded


def write_manifest(all_objects_with_cls):
    with MANIFEST.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "image_path", "category", "fine_grained_type",
                     "period", "dynasty", "material", "craft", "region",
                     "artifact_group_id", "split", "source"])
        for obj, _cls in all_objects_with_cls:
            oid = str(obj.get("objectID", ""))
            culture = (obj.get("culture") or "").strip()
            period = (obj.get("period") or "").strip()
            dynasty = (obj.get("dynasty") or "").strip()
            medium = (obj.get("medium") or "").strip()
            classification = (obj.get("classification") or "").strip()
            title = (obj.get("title") or "").strip()
            country = (obj.get("country") or "").strip()
            region_val = (obj.get("region") or "").strip()
            subregion = (obj.get("subregion") or "").strip()

            medium_lower = medium.lower()
            mat = None
            for kw, cn in [("porcelain", "瓷"), ("stoneware", "瓷"), ("ceramic", "陶"),
                           ("bronze", "青铜"), ("copper", "铜"), ("jade", "玉"),
                           ("nephrite", "玉"), ("jadeite", "玉"), ("lacquer", "漆"),
                           ("gold", "金"), ("silver", "银"), ("silk", "丝"),
                           ("bamboo", "竹"), ("wood", "木"), ("iron", "铁")]:
                if kw in medium_lower:
                    mat = cn
                    break

            craft = None
            if "blue and white" in medium_lower: craft = "青花"
            elif "underglaze" in medium_lower: craft = "釉下彩"
            elif "glazed" in medium_lower: craft = "施釉"
            elif "cast" in medium_lower: craft = "铸造"
            elif "carved" in medium_lower: craft = "雕刻"
            elif "lacquered" in medium_lower: craft = "髹漆"

            reg = subregion or region_val or ("中国" if "china" in country.lower() else country)

            w.writerow([
                f"met-{oid}", f"met-{oid}.jpg",
                classification, title, period, dynasty,
                mat or "", craft or "", reg or "",
                oid, "unassigned", "Met Museum (CC0)",
            ])


def main():
    t0 = time.time()
    state = {"t0": t0, "batches": 0, "api_calls": 0, "img_calls": 0, "total_cached": 0}

    print("[1/2] Reading CSV and grouping by category...", flush=True)
    buckets = load_csv_ids_by_category()
    for cat_name, _, target in CATEGORY_TARGETS:
        ids = buckets.get(cat_name, [])
        print(f"  {cat_name:14s} | {len(ids):5d} CSV IDs available | target {target}",
              flush=True)

    print("\n[1.5/2] Loading API cache...", flush=True)
    cached = load_cached_objects()
    state["total_cached"] = len(cached)
    print(f"  {len(cached)} objects with images already cached", flush=True)

    print("\n[1/2] Fetching API data per category...", flush=True)
    category_objects = {}
    with httpx.Client(timeout=30, follow_redirects=True, headers=HEADERS) as client:
        for cat_name, cls_values, target in CATEGORY_TARGETS:
            ids_with_cls = buckets.get(cat_name, [])
            if not ids_with_cls:
                print(f"  [{cat_name}] no IDs in CSV, skipping", flush=True)
                category_objects[cat_name] = []
                continue
            objs, newly = fetch_api_for_category(
                cat_name, ids_with_cls, target, cached, client, state
            )
            category_objects[cat_name] = objs
            elapsed = time.time() - t0
            print(f"  [{cat_name}] got {len(objs)}/{target} with images "
                  f"({elapsed:.0f}s elapsed)", flush=True)

    all_with_cls = []
    for cat_name, _, _ in CATEGORY_TARGETS:
        all_with_cls.extend(category_objects.get(cat_name, []))
    print(f"\nTotal objects to download images for: {len(all_with_cls)}", flush=True)
    download_images(all_with_cls, state)

    write_manifest(all_with_cls)

    print(f"\n{'='*60}", flush=True)
    print("CATEGORY SUMMARY", flush=True)
    total_imgs = 0
    for cat_name, _, target in CATEGORY_TARGETS:
        objs = category_objects.get(cat_name, [])
        imgs_on_disk = sum(
            1 for obj, _ in objs
            if (IMAGES_DIR / f"met-{obj.get('objectID', '')}.jpg").exists()
            and (IMAGES_DIR / f"met-{obj.get('objectID', '')}.jpg").stat().st_size > 1000
        )
        total_imgs += imgs_on_disk
        status = "OK" if imgs_on_disk >= target else "SHORT"
        print(f"  {cat_name:14s} | {imgs_on_disk:5d}/{target} images | {status}",
              flush=True)
    print(f"{'='*60}", flush=True)
    print(f"TOTAL IMAGES: {total_imgs}", flush=True)
    print(f"API calls: {state['api_calls']}, Image calls: {state['img_calls']}",
          flush=True)
    elapsed = time.time() - t0
    print(f"Elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)", flush=True)
    print(f"Manifest: {MANIFEST}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
