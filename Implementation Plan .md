# GarLens — Visual Search Engine Implementation Plan

## Project Overview
An internal, open-source "Google Lens" style image-to-image search engine for a garment database.  
Users upload a photo of a garment and the system returns visually similar items from the catalog.  
Fully local, free, GPU-accelerated, targeting **100 images across 5 garment categories** for the initial test run (configurable for production scale).

---

## Technical Stack

| Layer | Tool | Why |
|---|---|---|
| Data Collection | `duckduckgo-search` + `requests` | Free, no API key, produces real-world messy data |
| Data Cleaning | `Pillow`, `hashlib` (built-in) | Corruption checks, resolution filter, SHA-256 dedup |
| Background Removal | `rembg` | Isolates garment from mannequins / lifestyle backgrounds |
| Embedding Model | `patrickjohncyh/fashion-clip` (HuggingFace) | Domain-tuned CLIP for garments; runs on CUDA |
| Vector Database | `chromadb` (local, persistent) | File-based, zero cost, cosine similarity built-in |
| UI | `streamlit` | Interactive prototype; no separate API layer needed |

> **No n8n.** Orchestration is handled by plain Python scripts — simpler, easier to debug, no extra service to run.

---

## Environment

### Required Libraries (`requirements.txt`)
```
duckduckgo-search
requests
Pillow
tqdm
rembg
torch torchvision          # Install CUDA build from pytorch.org
transformers
chromadb
streamlit
```

> **PyTorch must be installed separately** with the correct CUDA wheel from pytorch.org. Do not use plain `pip install torch` — it may pull the CPU-only build.

### GPU Note
Your RTX 4070M (8 GB VRAM) is well within requirements:
- `rembg` (U2-Net): ~170 MB VRAM
- FashionCLIP (ViT-B/32): ~600 MB VRAM
- Both run simultaneously with headroom to spare.
- Estimated throughput: ~2–5 s/image → 100 images in under 10 minutes.

---

## Step 1: Download Raw Images (`download.py`) ✅ Complete

Pull raw, uncleaned garment images from DuckDuckGo Image Search. This is the **intentionally messy** source used to practice real data cleaning.

**Search queries** are crafted per category (e.g. `"plain t-shirt apparel product white background"`) but results will include:
- Lifestyle / model shots
- Mannequin photos
- Inconsistent aspect ratios and resolutions
- Watermarked or low-quality thumbnails
- Duplicate images from different URLs

**What the script does:**

1. Searches DDG for each of the 5 garment categories
2. Downloads images and applies immediate sanity checks:
   - **File size** — skips anything under 10 KB (thumbnail artifacts)
   - **Resolution** — skips below 224 × 224 px (unusable for embedding)
   - **Image integrity** — detects corrupt / truncated files via `Pillow.verify()`
   - **SHA-256 deduplication** — skips exact duplicates regardless of filename or URL
3. Assigns a deterministic SKU: `CATEGORY-XXXX` (e.g. `TSHIRT-0001`)
4. Appends a row to `data/raw_catalog.csv` per saved image
5. Logs every skipped file with its reason to `data/skipped.log`

**Output:**
```
data/
  raw_images/       ← TSHIRT-0001.jpg, JEANS-0003.jpg, …
  raw_catalog.csv   ← sku, filename, category, width, height, size_kb, source_url, sha256
  skipped.log       ← reason-coded log of every rejected URL
  download.log      ← timestamped run log
```

**Resume-safe:** Re-running `download.py` reads the existing CSV, picks up the SKU counter where it left off per category, and skips hashes already seen. A mid-run crash loses at most one image.

**Configuration (top of script):**
- `IMAGES_PER_CATEGORY = 20` → 100 images total for test run
- Change to `200` for a 1,000-image production ingestion

---

## Step 2: Ingest, Clean & Embed (`ingest.py`) ← *To be built*

Process the raw downloaded images into searchable vectors.

### 2a — Data Cleaning Pass
Before embedding, each image is inspected:

| Check | Action on Failure |
|---|---|
| Already in ChromaDB | Skip (resume-safe) |
| Corrupted file (post-download) | Log, skip |
| Aspect ratio extreme (< 0.3 or > 3.5) | Log, skip |
| `rembg` failure | Log, skip |

Cleaning insights are written to `data/cleaning_report.csv` for review.

### 2b — Background Removal
Pass each accepted image through `rembg` to strip backgrounds (mannequins, lifestyle clutter). The isolated garment is held in memory — it is **not** written to disk separately, keeping storage clean.

### 2c — Embedding (FashionCLIP on GPU)
Load `patrickjohncyh/fashion-clip` via HuggingFace Transformers onto the CUDA device.  
Process images in batches (batch size configurable, default 16) for throughput.  
Each image produces a **512-dimensional float vector**.

### 2d — Upsert to ChromaDB
Push each vector with its metadata into a persistent ChromaDB collection named `garments`, configured with **cosine similarity**.

Metadata stored per vector:
```
sku | filename | category | width | height | size_kb
```

**Resume-safe:** Query ChromaDB for existing IDs before each batch; skip IDs already present.

---

## Step 3: Build the Search UI (`app.py`) ← *To be built*

A single Streamlit app; no separate API server required.

### UI Flow
1. **Upload** — `st.file_uploader` accepts JPG/PNG from the user's device
2. **Preprocess** — Same pipeline as ingestion: `rembg` → FashionCLIP embedding
3. **Query** — `collection.query(query_embeddings=[vec], n_results=N)` (N is a sidebar slider, default 5)
4. **Display** — Results shown as an image grid with SKU, category, and cosine similarity score

### Key Design Decisions
- Streamlit-only (no FastAPI). The search logic runs in-process — no HTTP overhead for an internal tool.
- FashionCLIP model is loaded once at startup using `@st.cache_resource` — no per-query reload.
- ChromaDB client is also cached. Cold start is ~5–10 s; subsequent queries are near-instant.

---

## Step 4: Local Deployment & Testing

1. **Run:** `streamlit run app.py` (port `8501` by default)
2. **Network sharing:** Find host machine's local IP (`ipconfig`). Team members on the same network can access `http://192.168.x.x:8501`
3. **No authentication** — acceptable trade-off for an internal prototype

### Recommended Test Cases
| Scenario | What it tests |
|---|---|
| Clean studio photo | Baseline — should return near-identical matches |
| Worn by a human (not mannequin) | `rembg` robustness |
| Folded / partially visible garment | Embedding quality under occlusion |
| Wrong category upload | Graceful degradation (low similarity scores expected) |
| Photo taken in poor lighting | Preprocessing resilience |

---

## File Structure (Final)
```
GarLens/
  download.py            ← Step 1: Download & catalog raw images  ✅
  ingest.py              ← Step 2: Clean, embed, store in ChromaDB
  app.py                 ← Step 3: Streamlit search UI
  requirements.txt
  data/
    raw_images/          ← Raw downloaded JPGs
    raw_catalog.csv      ← SKU metadata from download
    cleaning_report.csv  ← Cleaning audit from ingest
    skipped.log
    download.log
    chroma_db/           ← Persistent ChromaDB storage
```

---

## Quality Gates (Learning + Professional Baseline)
- **Retrieval quality:** target `Hit Rate@5 >= 80%` on a sampled local evaluation set.
- **Latency target (end-to-end query):**
  - CUDA path: `p95 <= 1200 ms`
  - CPU fallback path: `p95 <= 3500 ms`
- **Evaluation command:** `python evaluate.py --top-k 5 --sample-size 25 --device auto`

These gates are intentionally practical for a 100-image prototype, and can be tightened as dataset size and model stability improve.

## Runtime Fallback Policy
- Preferred runtime is CUDA when available.
- If CUDA is unavailable, pipeline and UI must run on CPU without code changes.
- Device selection should remain explicit in CLI/UI (`auto`, `cuda`, `cpu`) for reproducible experiments.

## Open Items
- [ ] `ingest.py` — to be built after `download.py` is validated
- [ ] `app.py` — to be built after ChromaDB is populated with test data
- [ ] `requirements.txt` — finalise after confirming CUDA PyTorch wheel version
