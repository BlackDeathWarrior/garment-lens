"""SQLite compatibility shim for environments with old stdlib sqlite3.

Chroma requires sqlite >= 3.35. In some hosted dev environments (including
some Codespaces bases), Python's bundled sqlite3 can be older.
"""

from __future__ import annotations

import importlib
import sys

MIN_SQLITE_VERSION = (3, 35, 0)


def _parse_version(version_text: str) -> tuple[int, int, int]:
    parts: list[int] = []
    for token in version_text.split("."):
        try:
            parts.append(int(token))
        except ValueError:
            break
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])


def ensure_sqlite_compat() -> None:
    import sqlite3  # noqa: PLC0415

    if _parse_version(sqlite3.sqlite_version) >= MIN_SQLITE_VERSION:
        return

    try:
        pysqlite3 = importlib.import_module("pysqlite3")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "sqlite3 is too old for Chroma (requires >= 3.35). "
            "Install `pysqlite3-binary` and retry."
        ) from exc

    sys.modules["sqlite3"] = pysqlite3
