"""Ingest raw garment images, clean them, and upsert embeddings to ChromaDB."""

from __future__ import annotations

import argparse
import csv
import logging
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import chromadb
from PIL import Image

from garlens_pipeline import (
    MODEL_NAME,
    analyze_foreground_complexity,
    embed_image_bytes,
    load_model_bundle,
    remove_background,
    resolve_device,
)

DATA_DIR = Path("data")
RAW_CATALOG = DATA_DIR / "raw_catalog.csv"
RAW_IMAGES_DIR = DATA_DIR / "raw_images"
CHROMA_DIR = DATA_DIR / "chroma_db"
CLEANING_REPORT = DATA_DIR / "cleaning_report.csv"
INGEST_LOG = DATA_DIR / "ingest.log"
COLLECTION_NAME = "garments"

CLEANING_FIELDS = [
    "timestamp",
    "sku",
    "status",
    "reason",
    "width",
    "height",
    "aspect_ratio",
    "multi_garment_flag",
    "component_count",
    "multicolor_flag",
    "dominant_color_count",
    "pattern_flag",
    "pattern_score",
    "duration_ms",
]


@dataclass
class IngestStats:
    scanned: int = 0
    skipped_exists: int = 0
    skipped_invalid: int = 0
    skipped_excluded: int = 0
    inserted: int = 0


def setup_logging() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        handlers=[logging.FileHandler(INGEST_LOG), logging.StreamHandler()],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--min-aspect-ratio", type=float, default=0.3)
    parser.add_argument("--max-aspect-ratio", type=float, default=3.5)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--limit", type=int, default=0, help="0 means no limit")
    parser.add_argument(
        "--exclude-skus-file",
        default="data/invalid_skus.txt",
        help="Path to newline-delimited SKU blocklist to exclude from indexing.",
    )
    parser.add_argument(
        "--skip-multi-garment",
        action="store_true",
        help="Skip images flagged as likely containing multiple garments.",
    )
    parser.add_argument(
        "--skip-patterned",
        action="store_true",
        help="Skip images flagged as high pattern/design complexity.",
    )
    return parser.parse_args()


def load_catalog_rows(limit: int = 0) -> list[dict[str, str]]:
    if not RAW_CATALOG.exists():
        raise FileNotFoundError(f"Missing catalog file: {RAW_CATALOG}")

    with RAW_CATALOG.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    return rows if limit <= 0 else rows[:limit]


def ensure_cleaning_report() -> None:
    if CLEANING_REPORT.exists():
        with CLEANING_REPORT.open("r", encoding="utf-8", newline="") as fh:
            existing_header = next(csv.reader(fh), [])
        if existing_header == CLEANING_FIELDS:
            return

        backup_path = DATA_DIR / f"cleaning_report.pre_flags.{int(time.time())}.csv"
        CLEANING_REPORT.replace(backup_path)
        logging.warning(
            "Cleaning report header changed. Previous report moved to: %s",
            backup_path.resolve(),
        )

    with CLEANING_REPORT.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CLEANING_FIELDS)
        writer.writeheader()


def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def append_cleaning_row(writer: csv.DictWriter, **kwargs: Any) -> None:
    payload = {field: kwargs.get(field, "") for field in CLEANING_FIELDS}
    payload["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    writer.writerow(payload)


def image_dimensions(raw_bytes: bytes) -> tuple[bool, int, int]:
    try:
        image = Image.open(BytesIO(raw_bytes))
        image.verify()
        image = Image.open(BytesIO(raw_bytes))
        width, height = image.size
        return True, width, height
    except Exception:
        return False, 0, 0


def safe_float(value: str, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def fetch_existing_ids(collection) -> set[str]:
    try:
        payload = collection.get(include=[])
    except TypeError:
        payload = collection.get()
    return set(payload.get("ids", []))


def row_value(row: dict[str, str], key: str) -> str:
    """Read CSV value by key with tolerance for BOM/quoted headers."""
    value = row.get(key, None)
    if value is not None:
        return value
    for raw_key, raw_value in row.items():
        normalized = raw_key.strip().strip('"').lstrip("\ufeff")
        if normalized == key:
            return raw_value
    return ""


def load_excluded_skus(path_str: str) -> set[str]:
    path = Path(path_str)
    if not path.exists():
        return set()
    excluded: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        sku = line.strip()
        if not sku or sku.startswith("#"):
            continue
        excluded.add(sku)
    return excluded


def main() -> None:
    args = parse_args()
    setup_logging()
    ensure_cleaning_report()

    try:
        rows = load_catalog_rows(limit=args.limit)
    except FileNotFoundError as exc:
        logging.error("%s. Run `python download.py` first.", exc)
        raise SystemExit(1) from exc

    collection = get_collection()
    excluded_skus = load_excluded_skus(args.exclude_skus_file)
    if excluded_skus:
        logging.info("Loaded %d excluded SKU(s) from %s", len(excluded_skus), args.exclude_skus_file)
        # Ensure excluded SKUs are removed from vector index if already present.
        collection.delete(ids=list(excluded_skus))

    existing_ids = fetch_existing_ids(collection)

    device = resolve_device(args.device)
    logging.info("Resolved device: %s", device)
    logging.info("Loading model: %s", args.model_name)
    processor, model = load_model_bundle(model_name=args.model_name, device=device)

    stats = IngestStats()
    upsert_ids: list[str] = []
    upsert_embeddings: list[list[float]] = []
    upsert_metadatas: list[dict[str, Any]] = []

    with CLEANING_REPORT.open("a", encoding="utf-8", newline="") as report_fh:
        report_writer = csv.DictWriter(report_fh, fieldnames=CLEANING_FIELDS)

        for row in rows:
            stats.scanned += 1
            sku = row_value(row, "sku").strip()
            filename = row_value(row, "filename").strip()
            category = row_value(row, "category").strip()
            source_url = row_value(row, "source_url").strip()

            if not sku or not filename:
                stats.skipped_invalid += 1
                append_cleaning_row(
                    report_writer,
                    sku=sku,
                    status="skip",
                    reason="missing_sku_or_filename",
                )
                continue

            if sku in excluded_skus:
                stats.skipped_excluded += 1
                append_cleaning_row(
                    report_writer,
                    sku=sku,
                    status="skip",
                    reason="excluded_by_blocklist",
                )
                continue

            if sku in existing_ids:
                stats.skipped_exists += 1
                append_cleaning_row(
                    report_writer,
                    sku=sku,
                    status="skip",
                    reason="already_indexed",
                )
                continue

            image_path = RAW_IMAGES_DIR / filename
            if not image_path.exists():
                stats.skipped_invalid += 1
                append_cleaning_row(
                    report_writer,
                    sku=sku,
                    status="skip",
                    reason="missing_file",
                )
                continue

            start = time.perf_counter()
            raw_bytes = image_path.read_bytes()
            valid, width, height = image_dimensions(raw_bytes)
            if not valid:
                stats.skipped_invalid += 1
                append_cleaning_row(
                    report_writer,
                    sku=sku,
                    status="skip",
                    reason="invalid_image",
                )
                continue

            aspect_ratio = (width / height) if height else 0.0
            if not (args.min_aspect_ratio <= aspect_ratio <= args.max_aspect_ratio):
                stats.skipped_invalid += 1
                append_cleaning_row(
                    report_writer,
                    sku=sku,
                    status="skip",
                    reason="aspect_ratio_out_of_bounds",
                    width=width,
                    height=height,
                    aspect_ratio=round(aspect_ratio, 4),
                )
                continue

            complexity: dict[str, float | int | bool] | None = None
            try:
                no_bg_bytes = remove_background(raw_bytes)
                complexity = analyze_foreground_complexity(no_bg_bytes)

                if complexity["is_multi_garment"] and args.skip_multi_garment:
                    stats.skipped_invalid += 1
                    append_cleaning_row(
                        report_writer,
                        sku=sku,
                        status="skip",
                        reason="flagged_multi_garment",
                        width=width,
                        height=height,
                        aspect_ratio=round(aspect_ratio, 4),
                        multi_garment_flag=complexity["is_multi_garment"],
                        component_count=complexity["component_count"],
                        multicolor_flag=complexity["is_multicolor"],
                        dominant_color_count=complexity["dominant_color_count"],
                        pattern_flag=complexity["is_patterned"],
                        pattern_score=complexity["pattern_score"],
                    )
                    continue

                if complexity["is_patterned"] and args.skip_patterned:
                    stats.skipped_invalid += 1
                    append_cleaning_row(
                        report_writer,
                        sku=sku,
                        status="skip",
                        reason="flagged_patterned",
                        width=width,
                        height=height,
                        aspect_ratio=round(aspect_ratio, 4),
                        multi_garment_flag=complexity["is_multi_garment"],
                        component_count=complexity["component_count"],
                        multicolor_flag=complexity["is_multicolor"],
                        dominant_color_count=complexity["dominant_color_count"],
                        pattern_flag=complexity["is_patterned"],
                        pattern_score=complexity["pattern_score"],
                    )
                    continue

                embedding = embed_image_bytes(
                    image_bytes=no_bg_bytes,
                    processor=processor,
                    model=model,
                    device=device,
                )
            except Exception as exc:  # noqa: BLE001
                stats.skipped_invalid += 1
                append_cleaning_row(
                    report_writer,
                    sku=sku,
                    status="skip",
                    reason=f"embedding_error:{type(exc).__name__}",
                    width=width,
                    height=height,
                    aspect_ratio=round(aspect_ratio, 4),
                    multi_garment_flag=(
                        complexity["is_multi_garment"] if complexity is not None else ""
                    ),
                    component_count=(
                        complexity["component_count"] if complexity is not None else ""
                    ),
                    multicolor_flag=(
                        complexity["is_multicolor"] if complexity is not None else ""
                    ),
                    dominant_color_count=(
                        complexity["dominant_color_count"] if complexity is not None else ""
                    ),
                    pattern_flag=(
                        complexity["is_patterned"] if complexity is not None else ""
                    ),
                    pattern_score=(
                        complexity["pattern_score"] if complexity is not None else ""
                    ),
                )
                continue

            duration_ms = (time.perf_counter() - start) * 1000.0

            upsert_ids.append(sku)
            upsert_embeddings.append(embedding)
            upsert_metadatas.append(
                {
                    "sku": sku,
                    "filename": filename,
                    "category": category,
                    "width": width,
                    "height": height,
                    "size_kb": safe_float(row_value(row, "size_kb"), fallback=0.0),
                    "source_url": source_url,
                    "multi_garment": bool(complexity["is_multi_garment"]),
                    "component_count": int(complexity["component_count"]),
                    "multicolor": bool(complexity["is_multicolor"]),
                    "dominant_color_count": int(complexity["dominant_color_count"]),
                    "patterned": bool(complexity["is_patterned"]),
                    "pattern_score": float(complexity["pattern_score"]),
                }
            )

            append_cleaning_row(
                report_writer,
                sku=sku,
                status="inserted",
                reason="ok",
                width=width,
                height=height,
                aspect_ratio=round(aspect_ratio, 4),
                multi_garment_flag=complexity["is_multi_garment"],
                component_count=complexity["component_count"],
                multicolor_flag=complexity["is_multicolor"],
                dominant_color_count=complexity["dominant_color_count"],
                pattern_flag=complexity["is_patterned"],
                pattern_score=complexity["pattern_score"],
                duration_ms=round(duration_ms, 2),
            )

            if len(upsert_ids) >= args.batch_size:
                collection.upsert(
                    ids=upsert_ids,
                    embeddings=upsert_embeddings,
                    metadatas=upsert_metadatas,
                )
                stats.inserted += len(upsert_ids)
                existing_ids.update(upsert_ids)
                upsert_ids.clear()
                upsert_embeddings.clear()
                upsert_metadatas.clear()
                report_fh.flush()

        if upsert_ids:
            collection.upsert(
                ids=upsert_ids,
                embeddings=upsert_embeddings,
                metadatas=upsert_metadatas,
            )
            stats.inserted += len(upsert_ids)
            report_fh.flush()

    logging.info("Ingestion complete")
    logging.info("Scanned           : %d", stats.scanned)
    logging.info("Inserted          : %d", stats.inserted)
    logging.info("Skipped (indexed) : %d", stats.skipped_exists)
    logging.info("Skipped (excluded): %d", stats.skipped_excluded)
    logging.info("Skipped (invalid) : %d", stats.skipped_invalid)
    logging.info("Chroma collection count: %d", collection.count())
    logging.info("Cleaning report: %s", CLEANING_REPORT.resolve())


if __name__ == "__main__":
    main()
