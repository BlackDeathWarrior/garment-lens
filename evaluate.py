"""Evaluate retrieval quality (Top-K hit rate) and latency for GarLens."""

from __future__ import annotations

import argparse
import csv
import random
import statistics
import time
from pathlib import Path

from sqlite_compat import ensure_sqlite_compat

ensure_sqlite_compat()

import chromadb

from garlens_pipeline import MODEL_NAME, embed_image_bytes, load_model_bundle, remove_background, resolve_device

DATA_DIR = Path("data")
RAW_CATALOG = DATA_DIR / "raw_catalog.csv"
CHROMA_DIR = DATA_DIR / "chroma_db"
COLLECTION_NAME = "garments"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--sample-size", type=int, default=25)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_catalog() -> list[dict[str, str]]:
    if not RAW_CATALOG.exists():
        raise FileNotFoundError(f"Catalog not found: {RAW_CATALOG}")
    with RAW_CATALOG.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    values_sorted = sorted(values)
    index = (len(values_sorted) - 1) * p
    lo = int(index)
    hi = min(lo + 1, len(values_sorted) - 1)
    frac = index - lo
    return values_sorted[lo] * (1 - frac) + values_sorted[hi] * frac


def fetch_existing_ids(collection) -> set[str]:
    try:
        payload = collection.get(include=[])
    except TypeError:
        payload = collection.get()
    return set(payload.get("ids", []))


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    device = resolve_device(args.device)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    if collection.count() == 0:
        raise RuntimeError("Collection is empty. Run ingest.py first.")

    indexed_ids = fetch_existing_ids(collection)
    rows = [row for row in load_catalog() if row.get("sku", "") in indexed_ids]
    if not rows:
        raise RuntimeError("No indexed catalog rows available for evaluation.")

    sample_size = min(args.sample_size, len(rows))
    sample_rows = random.sample(rows, sample_size)

    processor, model = load_model_bundle(model_name=args.model_name, device=device)

    hits = 0
    latencies_ms: list[float] = []

    for row in sample_rows:
        sku = row["sku"]
        image_path = DATA_DIR / "raw_images" / row["filename"]
        if not image_path.exists():
            continue

        start = time.perf_counter()
        raw_bytes = image_path.read_bytes()
        processed = remove_background(raw_bytes)
        embedding = embed_image_bytes(
            image_bytes=processed,
            processor=processor,
            model=model,
            device=device,
        )
        result = collection.query(
            query_embeddings=[embedding],
            n_results=args.top_k,
            include=["distances"],
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        latencies_ms.append(elapsed_ms)

        retrieved_ids = result.get("ids", [[]])[0]
        if sku in retrieved_ids:
            hits += 1

    executed = len(latencies_ms)
    if executed == 0:
        raise RuntimeError("Evaluation could not run on any valid files.")

    hit_rate = hits / executed
    p50 = statistics.median(latencies_ms)
    p95 = percentile(latencies_ms, 0.95)

    if args.top_k >= 5:
        hit_rate_target = 0.80
    elif args.top_k == 3:
        hit_rate_target = 0.70
    else:
        hit_rate_target = 0.60

    latency_target = 1200.0 if device == "cuda" else 3500.0

    print("=" * 68)
    print("GarLens Evaluation Summary")
    print(f"Device                     : {device}")
    print(f"Model                      : {args.model_name}")
    print(f"Top-K                      : {args.top_k}")
    print(f"Samples executed           : {executed}")
    print(f"Hit Rate@{args.top_k}       : {hit_rate:.2%} (target >= {hit_rate_target:.0%})")
    print(f"Latency p50               : {p50:.1f} ms")
    print(f"Latency p95               : {p95:.1f} ms (target <= {latency_target:.0f} ms)")
    print("=" * 68)

    quality_pass = hit_rate >= hit_rate_target
    latency_pass = p95 <= latency_target
    if quality_pass and latency_pass:
        print("RESULT: PASS")
        raise SystemExit(0)

    print("RESULT: FAIL")
    if not quality_pass:
        print(" - Hit rate below target.")
    if not latency_pass:
        print(" - p95 latency above target.")
    raise SystemExit(2)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}")
        raise SystemExit(1) from exc
