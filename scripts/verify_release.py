"""Fail closed when public-release privacy or integrity checks fail."""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "PUBLIC_RELEASE_MANIFEST.json"
FORBIDDEN_SUFFIXES = {
    ".nii", ".dcm", ".dicom", ".npy", ".npz", ".pt", ".pth", ".ckpt",
    ".safetensors", ".pkl", ".pickle", ".joblib", ".h5", ".hdf5",
    ".onnx", ".zip",
}
FORBIDDEN_FILENAMES = {"train_labels.csv", "submission.csv", ".env"}
TEXT_SUFFIXES = {
    ".py", ".md", ".txt", ".json", ".toml", ".yaml", ".yml", ".cff",
    ".gitignore", ".example",
}
TEXT_PATTERNS = {
    "historical user-specific Kaggle mount": re.compile(
        r"/kaggle/input/(?:datasets/)?na" r"hinalam(?:/|$)", re.IGNORECASE
    ),
    "local workspace path": re.compile(r"/(?:workspace|home)/[^\s'\"`]+"),
    "Windows user path": re.compile(r"[A-Za-z]:\\\\Users\\\\[^\\\s]+"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "likely credential assignment": re.compile(
        r"(?im)^\s*(?:api[_-]?key|access[_-]?token|secret|password)\s*=\s*['\"][^'\"]+['\"]"
    ),
}
MAX_FILE_BYTES = 8 * 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def public_files() -> list[Path]:
    files = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or ".git" in path.parts:
            continue
        if path == MANIFEST:
            continue
        files.append(path)
    return files


def main() -> int:
    errors: list[str] = []
    files = public_files()
    for path in files:
        rel = path.relative_to(ROOT).as_posix()
        lower_name = path.name.lower()
        suffixes = {suffix.lower() for suffix in path.suffixes}
        if lower_name in FORBIDDEN_FILENAMES:
            errors.append(f"forbidden filename: {rel}")
        if suffixes & FORBIDDEN_SUFFIXES:
            errors.append(f"forbidden binary/data suffix: {rel}")
        if path.stat().st_size > MAX_FILE_BYTES:
            errors.append(f"unexpected file larger than 8 MiB: {rel}")
        if path.suffix.lower() in TEXT_SUFFIXES or path.name in {
            "LICENSE", "README.md", ".gitignore", "requirements.txt"
        }:
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                errors.append(f"text file is not UTF-8: {rel}")
                continue
            for label, pattern in TEXT_PATTERNS.items():
                if pattern.search(text):
                    errors.append(f"{label}: {rel}")

    if MANIFEST.is_file():
        try:
            manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
            expected = manifest["files"]
        except Exception as exc:
            errors.append(f"invalid release manifest: {type(exc).__name__}")
        else:
            actual = [
                {
                    "path": path.relative_to(ROOT).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
                for path in files
            ]
            if actual != expected:
                errors.append("PUBLIC_RELEASE_MANIFEST.json does not match the working tree")
    else:
        errors.append("PUBLIC_RELEASE_MANIFEST.json is missing")

    if errors:
        print("PUBLIC RELEASE VERIFICATION FAILED", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print(f"PUBLIC RELEASE VERIFICATION PASSED ({len(files)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
