"""Traversal and hashing shared by the release manifest writer and its verifier.

Both scripts must agree exactly on which files are public, in what order they are
recorded, and how each record is built. Keeping that in one module is the only way
to guarantee they cannot drift apart.
"""
from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "PUBLIC_RELEASE_MANIFEST.json"
MANIFEST_SCHEMA = "dat-parkinsons-public-release-manifest-v1"

# Build products and tool state are not part of the release. `__pycache__` in
# particular is created by the documented `python -m compileall` step, so a walk
# that included it would make verification fail on its own instructions.
SKIP_DIRECTORIES = {
    ".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".ipynb_checkpoints", ".virtual_documents", ".idea", ".vscode",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def public_files() -> list[Path]:
    """Every published file, ordered by POSIX relative path.

    Sorting the string rather than the Path keeps the order identical on Windows
    and POSIX: Path comparison is case-folded on Windows and byte-wise elsewhere,
    which would otherwise make a manifest built on one platform unverifiable on
    the other.
    """
    files = [
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and not SKIP_DIRECTORIES.intersection(path.parts)
        and path != MANIFEST
    ]
    return sorted(files, key=lambda path: path.relative_to(ROOT).as_posix())


def record(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": stat.st_size,
        "sha256": sha256(path),
    }
