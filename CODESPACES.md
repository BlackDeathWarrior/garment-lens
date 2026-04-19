# GarLens Codespaces Setup

This is the fastest way to demo GarLens from any laptop/phone browser without carrying your main machine.

## 1. Create Codespace
1. Open `https://github.com/BlackDeathWarrior/garment-lens`.
2. Click **Code > Codespaces > Create codespace on main**.
3. Wait for `postCreateCommand` to install dependencies.

## 2. Bring demo data bundle
On your current machine (repo root):

```powershell
python scripts/package_demo_data.py
```

This creates a lightweight bundle `artifacts/garlens-demo-data.zip` containing:
- `data/chroma_db/`
- `data/raw_catalog.csv`
- `data/invalid_skus.txt` (if present)

If you want all local images as well (full 500-image bundle):

```powershell
python scripts/package_demo_data.py --include-raw-images --output artifacts/garlens-full-data.zip
```

Upload that zip into Codespaces (drag-drop in VS Code web), then run:

```bash
python scripts/unpack_demo_data.py --zip artifacts/<your-zip-name>.zip
```

## 3. Run the app

```bash
python -m streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

Codespaces will auto-forward port `8501`; open the generated URL and share it.

## Notes
- GPU/CUDA is generally unavailable in Codespaces; app runs on CPU (`--device auto` falls back automatically).
- If `data/raw_images/` is absent, search still works from the vector index. Result previews fall back to `source_url`.
- If your index was built with `--skip-patterned`/`--skip-multi-garment`, those exclusions are preserved in demo behavior.
