# Repository Guidelines

## Project Structure & Module Organization
This repository is a local Python visual-search pipeline.
- `download.py`: main script for image search, validation, deduplication, and catalog logging.
- `ingest.py`: cleans catalog images, removes backgrounds, embeds with FashionCLIP, and upserts to ChromaDB.
- `app.py`: Streamlit UI for image upload and Top-K similarity search.
- `evaluate.py`: benchmark script for Hit Rate@K and latency (p50/p95).
- `garlens_pipeline.py`: shared model/preprocessing utilities with CPU fallback.
- `requirements.txt`: runtime Python dependencies.
- `data/`: generated artifacts (`raw_images/`, `raw_catalog.csv`, `cleaning_report.csv`, logs, `chroma_db/`).
- `venv/`: local virtual environment.

Keep stages split by responsibility.

## Build, Test, and Development Commands
Use Python 3.13+ and a virtual environment.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python download.py
python ingest.py --device auto
python evaluate.py --top-k 5 --sample-size 25
streamlit run app.py
```

- First two commands create/activate an isolated environment.
- `pip install -r requirements.txt` installs runtime dependencies.
- `python download.py` populates `data/raw_images` and `data/raw_catalog.csv`.
- `python ingest.py` builds/updates the ChromaDB index (`data/chroma_db`).
- `python evaluate.py` validates retrieval quality and latency against baseline targets.
- `streamlit run app.py` launches the local search UI.

Quick validation commands:
- `python -m pip check` verifies dependency consistency.
- `python -m compileall download.py ingest.py app.py evaluate.py` catches syntax issues.

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indentation and explicit type hints where practical.
- `snake_case` for functions/variables.
- `UPPER_SNAKE_CASE` for module constants (for example `IMAGES_PER_CATEGORY`).
- Category/SKU prefixes stay uppercase (`TSHIRT-0001` pattern).

Prefer small helper functions for validation and resume logic, and keep logging reason-coded so failures remain diagnosable from `skipped.log`.

## Testing Guidelines
No automated test suite is committed yet. For new logic, add `pytest` tests under `tests/` using `test_*.py` naming.
- Mock network calls (`requests.get`, `DDGS.images`) to keep tests deterministic.
- Cover deduplication, resume behavior, and CSV schema stability.
- Run a manual smoke test with low `IMAGES_PER_CATEGORY` before merging.

## Commit & Pull Request Guidelines
No Git history is available in this workspace snapshot, so use Conventional Commits going forward (for example `feat: add ingest resume guard`, `fix: handle corrupt JPEG headers`).

PRs should include:
- clear scope and rationale,
- behavior changes in downloader output/logging,
- config changes (thresholds, category/query edits),
- linked issue/task when applicable.

## Security & Configuration Tips
Do not commit generated data (`data/raw_images/`, logs) or `venv/`. Keep request delays/timeouts conservative to reduce blocking/rate-limit risk from external image sources.
