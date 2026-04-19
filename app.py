"""Streamlit UI for GarLens image-to-image garment search."""

from __future__ import annotations

import time
from pathlib import Path

from sqlite_compat import ensure_sqlite_compat

ensure_sqlite_compat()

import chromadb
import streamlit as st
import torch

from garlens_pipeline import MODEL_NAME, embed_image_bytes, load_model_bundle, remove_background, resolve_device

DATA_DIR = Path("data")
CHROMA_DIR = DATA_DIR / "chroma_db"
RAW_IMAGES_DIR = DATA_DIR / "raw_images"
COLLECTION_NAME = "garments"


@st.cache_resource
def load_runtime(model_name: str, device_pref: str):
    device = resolve_device(device_pref)
    processor, model = load_model_bundle(model_name=model_name, device=device)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return device, processor, model, collection


def render_results(ids, metadatas, distances) -> None:
    if not ids:
        st.info("No matches found.")
        return

    rows = []
    for idx, sku in enumerate(ids):
        metadata = metadatas[idx] if idx < len(metadatas) else {}
        distance = distances[idx] if idx < len(distances) else None
        filename = metadata.get("filename", f"{sku}.jpg")
        rows.append(
            {
                "sku": sku,
                "metadata": metadata,
                "distance": distance,
                "filename": filename,
            }
        )

    sort_label = st.selectbox(
        "Sort Results By",
        options=[
            "Relevance (lowest distance)",
            "Relevance (highest distance)",
            "Category (A-Z)",
            "SKU (A-Z)",
            "Pattern Score (high-low)",
            "Pattern Score (low-high)",
            "File Name (A-Z)",
        ],
        index=0,
    )

    if sort_label == "Relevance (lowest distance)":
        rows.sort(key=lambda r: float("inf") if r["distance"] is None else r["distance"])
    elif sort_label == "Relevance (highest distance)":
        rows.sort(key=lambda r: -1.0 if r["distance"] is None else -r["distance"])
    elif sort_label == "Category (A-Z)":
        rows.sort(key=lambda r: str(r["metadata"].get("category", "")))
    elif sort_label == "SKU (A-Z)":
        rows.sort(key=lambda r: str(r["sku"]))
    elif sort_label == "Pattern Score (high-low)":
        rows.sort(
            key=lambda r: -(float(r["metadata"].get("pattern_score", -1.0) or -1.0))
        )
    elif sort_label == "Pattern Score (low-high)":
        rows.sort(key=lambda r: float(r["metadata"].get("pattern_score", 1e9) or 1e9))
    elif sort_label == "File Name (A-Z)":
        rows.sort(key=lambda r: str(r["filename"]))

    for row in rows:
        sku = row["sku"]
        metadata = row["metadata"]
        distance = row["distance"]
        filename = row["filename"]
        image_path = RAW_IMAGES_DIR / filename
        source_url = str(metadata.get("source_url", "") or "").strip()

        cols = st.columns([1, 2])
        with cols[0]:
            if image_path.exists():
                st.image(str(image_path), use_container_width=True)
            elif source_url:
                st.image(source_url, use_container_width=True)
                st.caption("Preview via source URL (local image not present).")
            else:
                st.warning(f"Image not found: {filename}")
        with cols[1]:
            st.markdown(f"**SKU:** `{sku}`")
            st.markdown(f"**Category:** `{metadata.get('category', 'unknown')}`")
            st.markdown(f"**Filename:** `{filename}`")
            if source_url:
                st.markdown(f"**Source URL:** `{source_url}`")
            if "patterned" in metadata:
                st.markdown(f"**Patterned Flag:** `{metadata.get('patterned')}`")
            if "pattern_score" in metadata:
                st.markdown(f"**Pattern Score:** `{metadata.get('pattern_score')}`")
            if "multicolor" in metadata:
                st.markdown(f"**Multicolor Flag:** `{metadata.get('multicolor')}`")
            if distance is not None:
                st.markdown(f"**Cosine Distance:** `{distance:.4f}` (lower is better)")
        st.divider()


def main() -> None:
    st.set_page_config(page_title="GarLens Search", layout="wide")
    st.title("GarLens Visual Search")
    st.caption("Learning-focused, professional-grade local image-to-image search for garments.")

    with st.sidebar:
        st.header("Search Settings")
        top_k = st.slider("Top-K Results", min_value=1, max_value=20, value=5)
        device_pref = st.selectbox(
            "Device",
            options=["auto", "cuda", "cpu"],
            index=0,
            help="Auto uses CUDA when available, otherwise CPU.",
        )
        exclude_patterned = st.checkbox(
            "Exclude Patterned Results",
            value=False,
            help="Filters out items flagged as patterned during ingest.",
        )
        model_name = st.text_input("Model", value=MODEL_NAME)
        st.caption(
            f"PyTorch: `{torch.__version__}` | torch CUDA: `{torch.version.cuda}` | "
            f"cuda available: `{torch.cuda.is_available()}`"
        )
        if device_pref == "cuda" and not torch.cuda.is_available():
            st.warning(
                "CUDA selected but unavailable in this Python env. "
                "Install a CUDA-enabled torch build."
            )
            st.code(
                "python -m pip install --index-url https://download.pytorch.org/whl/cu128 "
                "torch torchvision torchaudio",
                language="bash",
            )
        st.caption("Suggested baseline targets: Top-5 hit rate >= 80%, p95 latency <= 1.2s (CUDA) or <= 3.5s (CPU).")

    try:
        device, processor, model, collection = load_runtime(model_name=model_name, device_pref=device_pref)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to initialize runtime: {exc}")
        st.stop()

    st.success(f"Runtime ready on `{device}`. Indexed items: `{collection.count()}`")
    if not RAW_IMAGES_DIR.exists():
        st.info(
            "Local `data/raw_images` folder is not present. "
            "Results will use `source_url` previews when available."
        )

    if collection.count() == 0:
        st.warning("Collection is empty. Run `python ingest.py` first.")
        st.stop()

    uploaded = st.file_uploader("Upload garment image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        st.info("Upload an image to search similar garments.")
        st.stop()

    query_start = time.perf_counter()
    raw = uploaded.getvalue()
    st.subheader("Query Preview")
    preview_cols = st.columns(2)
    with preview_cols[0]:
        st.markdown("**Uploaded Image**")
        st.image(raw, use_container_width=True)

    try:
        processed = remove_background(raw)
        with preview_cols[1]:
            st.markdown("**Foreground Preview (Background Removed)**")
            st.image(processed, use_container_width=True)

        embedding = embed_image_bytes(
            image_bytes=processed,
            processor=processor,
            model=model,
            device=device,
        )
    except Exception as exc:  # noqa: BLE001
        st.error(f"Preprocessing or embedding failed: {exc}")
        st.stop()

    preprocess_ms = (time.perf_counter() - query_start) * 1000.0

    search_start = time.perf_counter()
    query_kwargs = {}
    if exclude_patterned:
        query_kwargs["where"] = {"patterned": False}
    results = collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
        include=["metadatas", "distances"],
        **query_kwargs,
    )
    search_ms = (time.perf_counter() - search_start) * 1000.0
    total_ms = preprocess_ms + search_ms

    st.subheader("Performance")
    metric_cols = st.columns(3)
    metric_cols[0].metric("Preprocess + Embed", f"{preprocess_ms:.1f} ms")
    metric_cols[1].metric("Vector Query", f"{search_ms:.1f} ms")
    metric_cols[2].metric("Total", f"{total_ms:.1f} ms")

    st.subheader("Matches")
    render_results(
        ids=results.get("ids", [[]])[0],
        metadatas=results.get("metadatas", [[]])[0],
        distances=results.get("distances", [[]])[0],
    )


if __name__ == "__main__":
    main()
