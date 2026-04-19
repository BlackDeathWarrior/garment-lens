"""Create a demo bundle for Codespaces usage.

Bundle includes:
- data/chroma_db/
- data/raw_catalog.csv
- data/invalid_skus.txt (if present)
- optionally data/raw_images/ (--include-raw-images)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


REQUIRED_PATHS = [
    Path("data/chroma_db"),
    Path("data/raw_catalog.csv"),
]
OPTIONAL_PATHS = [
    Path("data/invalid_skus.txt"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="artifacts/garlens-demo-data.zip",
        help="Output zip file path.",
    )
    parser.add_argument(
        "--include-raw-images",
        action="store_true",
        help="Include data/raw_images in the zip (larger bundle, full local previews).",
    )
    return parser.parse_args()


def add_path_to_zip(zf: ZipFile, path: Path) -> None:
    if path.is_file():
        zf.write(path, arcname=str(path.as_posix()))
        return
    for file_path in path.rglob("*"):
        if file_path.is_file():
            zf.write(file_path, arcname=str(file_path.as_posix()))


def main() -> None:
    args = parse_args()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    missing = [p for p in REQUIRED_PATHS if not p.exists()]
    if missing:
        missing_csv = ", ".join(str(p) for p in missing)
        raise SystemExit(f"Missing required path(s): {missing_csv}")

    with ZipFile(out_path, mode="w", compression=ZIP_DEFLATED) as zf:
        for req in REQUIRED_PATHS:
            add_path_to_zip(zf, req)
        for opt in OPTIONAL_PATHS:
            if opt.exists():
                add_path_to_zip(zf, opt)
        if args.include_raw_images:
            raw_images = Path("data/raw_images")
            if not raw_images.exists():
                raise SystemExit("Missing required path for full bundle: data/raw_images")
            add_path_to_zip(zf, raw_images)

    mode = "full (with raw_images)" if args.include_raw_images else "lightweight"
    print(f"Demo bundle created ({mode}): {out_path.resolve()}")


if __name__ == "__main__":
    main()
