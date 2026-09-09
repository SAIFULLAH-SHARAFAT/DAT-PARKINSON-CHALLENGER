"""Regenerate the deterministic public release manifest after reviewed edits."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "PUBLIC_RELEASE_MANIFEST.json"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


files = []
for path in sorted(ROOT.rglob("*")):
    if not path.is_file() or ".git" in path.parts or path == MANIFEST:
        continue
    files.append(
        {
            "path": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": digest(path),
        }
    )

payload = {
    "schema": "dat-parkinsons-public-release-manifest-v1",
    "generated_for_release_date": "2026-09-09",
    "self_excluded": True,
    "files": files,
}
MANIFEST.write_text(
    json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8"
)
print(f"wrote {MANIFEST.name} with {len(files)} files")
