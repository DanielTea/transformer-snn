"""Recreate the checked PG-19 benchmark excerpts from pinned source objects."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from urllib.request import urlopen


DEFAULT_MANIFEST = (Path(__file__).resolve().parents[1]
                    / "results" / "lm_successor" / "pg19_test_excerpt_manifest.json")


def matches(data: bytes, size: int, sha256: str) -> bool:
    return len(data) == size and hashlib.sha256(data).hexdigest() == sha256


def write_excerpt(destination: Path, data: bytes) -> None:
    # Publish only verified bytes and keep the temporary file on this filesystem.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as output:
            temporary = Path(output.name)
            output.write(data)
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def prepare_book(book: dict, output: Path) -> None:
    name = book["excerpt_file"]
    destination = output / name
    excerpt_size = book["excerpt_bytes"]
    source_size = book["source_bytes"]
    start = book["excerpt_start_raw_byte"]
    if excerpt_size <= 0 or start < 0 or start + excerpt_size > source_size:
        raise ValueError(f"{name}: excerpt falls outside its source")
    if destination.is_file() and matches(destination.read_bytes(), excerpt_size, book["excerpt_sha256"]):
        print(f"Verified {destination}")
        return

    with urlopen(book["source_url"], timeout=30) as response:
        source = response.read(source_size + 1)
    if not matches(source, source_size, book["source_sha256"]):
        raise ValueError(f"{name}: downloaded source size or SHA-256 does not match the manifest")
    excerpt = source[start:start + excerpt_size]
    if not matches(excerpt, excerpt_size, book["excerpt_sha256"]):
        raise ValueError(f"{name}: extracted excerpt does not match the manifest")
    write_excerpt(destination, excerpt)
    print(f"Wrote {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="directory for the excerpt files")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST,
                        help="excerpt manifest (defaults to the checked benchmark manifest)")
    args = parser.parse_args()

    try:
        books = json.loads(args.manifest.read_text(encoding="utf-8"))["books"]
        names = [book["excerpt_file"] for book in books]
        if not names or len(names) != len(set(names)):
            raise ValueError("manifest must list at least one excerpt with unique file names")
        if any(name in ("", ".", "..") or Path(name).name != name for name in names):
            raise ValueError("excerpt names must be plain file names")
        args.output.mkdir(parents=True, exist_ok=True)
        for book in books:
            prepare_book(book, args.output)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"{parser.prog}: {exc}\n")


if __name__ == "__main__":
    main()
