"""Unpack a demo bundle into repository data paths."""

from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZipFile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zip",
        required=True,
        help="Path to garlens demo zip bundle.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    zip_path = Path(args.zip)
    if not zip_path.exists():
        raise SystemExit(f"Zip file not found: {zip_path}")

    with ZipFile(zip_path, "r") as zf:
        zf.extractall(Path("."))

    print("Demo bundle extracted successfully.")


if __name__ == "__main__":
    main()
