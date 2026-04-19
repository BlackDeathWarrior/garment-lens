"""
download.py — GarLens Image Downloader
---------------------------------------
Downloads raw garment images from DuckDuckGo Image Search.
Applies sanity checks: file size, resolution, SHA-256 deduplication.
Output : data/raw_images/*.jpg
         data/raw_catalog.csv
         data/skipped.log
         data/download.log

Resume-safe: re-running this script skips images already at target count
             and never re-downloads an image with a hash already in the CSV.

Dependencies (pip install):
    ddgs  requests  Pillow  tqdm
"""

import csv
import hashlib
import logging
import time
from collections import deque
from io import BytesIO
from pathlib import Path

import requests
from ddgs import DDGS
from ddgs.exceptions import RatelimitException
from PIL import Image
from tqdm import tqdm


# ── Configuration ─────────────────────────────────────────────────────────────
CATEGORIES: dict[str, list[str]] = {
    "TSHIRT": [
        "crew neck t-shirt product photo blue",
        "v-neck t-shirt product photo red",
        "oversized t-shirt streetwear product photo green",
        "graphic t-shirt product photo multicolor",
        "indian cotton t-shirt product photo",
        "japanese minimal t-shirt product photo",
        "latin american casual t-shirt product photo",
    ],
    "SHIRT": [
        "oxford shirt product photo light blue",
        "linen shirt product photo beige",
        "striped formal shirt product photo",
        "batik shirt product photo indonesia",
        "guayabera shirt product photo latin",
        "indian kurta shirt product photo",
        "japanese work shirt product photo",
    ],
    "JEANS": [
        "slim fit jeans product photo dark wash",
        "straight fit jeans product photo light wash",
        "wide leg jeans product photo blue",
        "black jeans product photo",
        "gray jeans product photo",
        "distressed jeans product photo",
        "high waist jeans product photo",
    ],
    "JACKET": [
        "bomber jacket product photo olive",
        "denim jacket product photo blue",
        "leather jacket product photo brown",
        "puffer jacket product photo red",
        "varsity jacket product photo",
        "utility jacket product photo khaki",
        "japanese workwear jacket product photo",
    ],
    "DRESS": [
        "maxi dress product photo floral",
        "midi dress product photo blue",
        "cocktail dress product photo black",
        "summer dress product photo yellow",
        "indian anarkali dress product photo",
        "african print dress product photo",
        "middle eastern modest dress product photo",
    ],
}

# Dataset defaults: 500 total images across 5 categories = 100 per category.
TOTAL_TARGET_IMAGES: int = 500
IMAGES_PER_CATEGORY: int = max(1, TOTAL_TARGET_IMAGES // len(CATEGORIES))
MIN_RESOLUTION: tuple[int, int] = (224, 224)
MIN_FILE_SIZE_KB: float = 10.0
REQUEST_DELAY_S: float = 0.4    # Courtesy delay between downloads
SEARCH_MAX_ATTEMPTS: int = 4
SEARCH_BASE_BACKOFF_S: float = 1.5
REQUEST_TIMEOUT_S: float = 12.0
IMAGE_FETCH_MAX_ATTEMPTS: int = 3
IMAGE_FETCH_BACKOFF_S: float = 0.8

DATA_DIR:    Path = Path("data")
IMAGE_DIR:   Path = DATA_DIR / "raw_images"
CATALOG_CSV: Path = DATA_DIR / "raw_catalog.csv"
SKIP_LOG:    Path = DATA_DIR / "skipped.log"
DOWNLOAD_LOG: Path = DATA_DIR / "download.log"

CSV_FIELDS = ["sku", "filename", "category", "width", "height",
              "size_kb", "source_url", "sha256"]


# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(DOWNLOAD_LOG, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _validate_image(data: bytes) -> tuple[bool, int, int]:
    """
    Returns (is_valid, width, height).
    Fails on: corrupt files, truncated streams, or below MIN_RESOLUTION.
    """
    try:
        img = Image.open(BytesIO(data))
        img.verify()                    # Detects corruption / truncation
        img = Image.open(BytesIO(data)) # Re-open; verify() closes the stream
        w, h = img.size
        return (w >= MIN_RESOLUTION[0] and h >= MIN_RESOLUTION[1]), w, h
    except Exception:
        return False, 0, 0


def _load_catalog(
    csv_path: Path,
) -> tuple[dict[str, int], set[str], set[str], set[str]]:
    """
    Reads an existing catalog CSV to support resuming a partial download.

    Returns:
        category_counts : {category_code: images_already_saved}
        seen_hashes     : set of SHA-256 hashes (global dedup guard)
        seen_skus       : SKU values already present in the catalog
        seen_filenames  : filename values already present in the catalog
    """
    category_counts: dict[str, int] = {k: 0 for k in CATEGORIES}
    seen_hashes: set[str] = set()
    seen_skus: set[str] = set()
    seen_filenames: set[str] = set()

    if not csv_path.exists():
        return category_counts, seen_hashes, seen_skus, seen_filenames

    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cat = row.get("category", "")
            if cat in category_counts:
                category_counts[cat] += 1
            h = row.get("sha256", "")
            if h:
                seen_hashes.add(h)
            sku = row.get("sku", "").strip()
            if sku:
                seen_skus.add(sku)
            filename = row.get("filename", "").strip()
            if filename:
                seen_filenames.add(filename)

    return category_counts, seen_hashes, seen_skus, seen_filenames


# ── Core download logic ───────────────────────────────────────────────────────
def _search_images(query: str, max_results: int) -> list[dict]:
    """Run image search with retries and exponential backoff."""
    for attempt in range(1, SEARCH_MAX_ATTEMPTS + 1):
        try:
            with DDGS() as ddgs:
                return list(ddgs.images(query=query, max_results=max_results))
        except RatelimitException as exc:
            wait_s = SEARCH_BASE_BACKOFF_S * (2 ** (attempt - 1))
            log.warning(
                "DDG rate limit (attempt %d/%d): %s",
                attempt,
                SEARCH_MAX_ATTEMPTS,
                exc,
            )
            if attempt < SEARCH_MAX_ATTEMPTS:
                time.sleep(wait_s)
        except Exception as exc:  # noqa: BLE001
            wait_s = SEARCH_BASE_BACKOFF_S * (2 ** (attempt - 1))
            log.warning(
                "DDG search error (attempt %d/%d): %s",
                attempt,
                SEARCH_MAX_ATTEMPTS,
                exc,
            )
            if attempt < SEARCH_MAX_ATTEMPTS:
                time.sleep(wait_s)
    return []


def _build_diverse_candidates(queries: list[str], needed: int) -> list[dict]:
    """Interleave candidates from multiple query buckets to avoid single-query bias."""
    per_query_max = max(needed * 3, 24)
    query_buckets: list[deque[dict]] = []

    for query in queries:
        query_results = _search_images(query=query, max_results=per_query_max)
        if query_results:
            query_buckets.append(deque(query_results))
        log.info("  Query '%s' returned %d candidates.", query, len(query_results))

    if not query_buckets:
        return []

    merged: list[dict] = []
    seen_urls: set[str] = set()
    limit = needed * 12

    while query_buckets and len(merged) < limit:
        active: list[deque[dict]] = []
        for bucket in query_buckets:
            while bucket:
                candidate = bucket.popleft()
                url = candidate.get("image", "")
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                merged.append(candidate)
                break
            if bucket:
                active.append(bucket)
        query_buckets = active

    return merged


def _fetch_image_bytes(url: str, skip_fh) -> bytes | None:
    """Fetch image bytes with retry/backoff for transient failures."""
    for attempt in range(1, IMAGE_FETCH_MAX_ATTEMPTS + 1):
        try:
            resp = requests.get(
                url,
                timeout=REQUEST_TIMEOUT_S,
                headers={"User-Agent": "Mozilla/5.0"},
            )
        except requests.RequestException as exc:
            if attempt >= IMAGE_FETCH_MAX_ATTEMPTS:
                skip_fh.write(f"NETWORK_ERROR\t{exc}\t{url}\n")
                return None
            time.sleep(IMAGE_FETCH_BACKOFF_S * attempt)
            continue

        if resp.status_code == 200:
            return resp.content

        if resp.status_code in {429, 500, 502, 503, 504} and attempt < IMAGE_FETCH_MAX_ATTEMPTS:
            time.sleep(IMAGE_FETCH_BACKOFF_S * attempt)
            continue

        skip_fh.write(f"HTTP_{resp.status_code}\t{url}\n")
        return None

    return None


def _download_category(
    code: str,
    queries: list[str],
    already_have: int,
    seen_hashes: set[str],
    seen_skus: set[str],
    seen_filenames: set[str],
    writer: "csv.DictWriter",
    skip_fh,
) -> int:
    """
    Downloads images for one garment category.

    Args:
        code         : Category code used in SKU (e.g. "TSHIRT")
        queries      : DDGS image search strings for color/style variety
        already_have : Images already saved for this category (resume offset)
        seen_hashes  : Mutable global hash set; updated in place
        seen_skus    : Mutable SKU set to avoid ID collisions
        seen_filenames : Mutable filename set to avoid filename collisions
        writer       : Open CSV DictWriter to append rows
        skip_fh      : Open file handle for the skip-log

    Returns:
        Number of new images saved in this run.
    """
    needed = IMAGES_PER_CATEGORY - already_have
    if needed <= 0:
        log.info(f"[{code}] Already at target ({IMAGES_PER_CATEGORY}). Skipping.")
        return 0

    log.info(f"[{code}] Need {needed} more image(s). Searching across query variants ...")

    # Query multiple color/style/region prompts and interleave them.
    results = _build_diverse_candidates(queries=queries, needed=needed)

    if not results:
        log.warning(f"[{code}] Search returned no results after retries.")
        skip_fh.write(f"SEARCH_FAILED\t{code}\t{' | '.join(queries)}\n")
        return 0

    saved = 0
    sku_idx = already_have + 1   # Continue numbering from where we left off

    for result in tqdm(results, desc=f"  [{code}]", unit="img"):
        if saved >= needed:
            break

        url = result.get("image", "")
        if not url:
            continue

        while True:
            sku = f"{code}-{sku_idx:04d}"
            filename = f"{sku}.jpg"
            filepath = IMAGE_DIR / filename
            if (
                sku not in seen_skus
                and filename not in seen_filenames
                and not filepath.exists()
            ):
                break
            sku_idx += 1

        # ── Network fetch ──────────────────────────────────────────────────
        data = _fetch_image_bytes(url=url, skip_fh=skip_fh)
        if data is None:
            continue

        size_kb = len(data) / 1024

        # ── Sanity checks ──────────────────────────────────────────────────
        if size_kb < MIN_FILE_SIZE_KB:
            skip_fh.write(f"TOO_SMALL\t{size_kb:.1f}KB\t{url}\n")
            continue

        digest = _sha256(data)
        if digest in seen_hashes:
            skip_fh.write(f"DUPLICATE\t{url}\n")
            continue

        valid, w, h = _validate_image(data)
        if not valid:
            skip_fh.write(f"BAD_IMAGE\t{w}x{h}px\t{url}\n")
            continue

        # ── Persist ────────────────────────────────────────────────────────
        filepath.write_bytes(data)
        seen_hashes.add(digest)
        seen_skus.add(sku)
        seen_filenames.add(filename)

        writer.writerow({
            "sku":        sku,
            "filename":   filename,
            "category":   code,
            "width":      w,
            "height":     h,
            "size_kb":    round(size_kb, 1),
            "source_url": url,
            "sha256":     digest,
        })

        log.debug(f"  Saved {filename} ({w}x{h}, {size_kb:.1f}KB)")
        saved   += 1
        sku_idx += 1
        time.sleep(REQUEST_DELAY_S)

    log.info(
        f"[{code}] Saved {saved} new image(s). "
        f"Running total: {already_have + saved}/{IMAGES_PER_CATEGORY}"
    )
    return saved


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    category_counts, seen_hashes, seen_skus, seen_filenames = _load_catalog(CATALOG_CSV)
    write_header = not CATALOG_CSV.exists()

    target_total = IMAGES_PER_CATEGORY * len(CATEGORIES)
    log.info("=" * 60)
    log.info("GarLens Downloader")
    log.info(f"Target : {IMAGES_PER_CATEGORY} images x {len(CATEGORIES)} "
             f"categories = {target_total} total")
    log.info(f"Output : {IMAGE_DIR.resolve()}")
    log.info("=" * 60)

    total_new = 0

    with (
        open(CATALOG_CSV, "a", newline="", encoding="utf-8") as csv_fh,
        open(SKIP_LOG,    "a", encoding="utf-8")             as skip_fh,
    ):
        writer = csv.DictWriter(csv_fh, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()

        for code, queries in CATEGORIES.items():
            try:
                saved = _download_category(
                    code=code,
                    queries=queries,
                    already_have=category_counts[code],
                    seen_hashes=seen_hashes,
                    seen_skus=seen_skus,
                    seen_filenames=seen_filenames,
                    writer=writer,
                    skip_fh=skip_fh,
                )
            except Exception as exc:  # noqa: BLE001
                saved = 0
                log.exception("[%s] Category failed: %s", code, exc)
                skip_fh.write(f"CATEGORY_ERROR\t{code}\t{type(exc).__name__}:{exc}\n")
            total_new += saved
            csv_fh.flush()  # Write after every category; safe against mid-run crashes

    log.info("=" * 60)
    log.info(f"Download complete. {total_new} new image(s) saved.")
    log.info(f"Catalog : {CATALOG_CSV.resolve()}")
    log.info(f"Images  : {IMAGE_DIR.resolve()}")
    log.info(f"Skipped : {SKIP_LOG.resolve()}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
