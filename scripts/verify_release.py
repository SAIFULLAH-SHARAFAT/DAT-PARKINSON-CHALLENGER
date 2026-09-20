"""Fail closed when public-release privacy or integrity checks fail."""
from __future__ import annotations

import argparse
import json
import re
import sys

from release_common import (
    MANIFEST,
    MANIFEST_SCHEMA,
    ROOT,
    public_files,
    record,
)

FORBIDDEN_SUFFIXES = {
    # Medical volumes and study containers.
    ".nii", ".dcm", ".dicom", ".ima", ".nrrd", ".mha", ".mhd", ".img", ".hdr",
    # Arrays, caches, checkpoints, and weights.
    ".npy", ".npz", ".pt", ".pth", ".ckpt", ".safetensors", ".pkl", ".pickle",
    ".joblib", ".h5", ".hdf5", ".onnx", ".gguf", ".pb", ".tflite", ".bin",
    ".mat", ".msgpack",
    # Tabular stores: any of these may carry per-case labels, UIDs or predictions.
    ".csv", ".parquet", ".feather", ".arrow", ".db", ".sqlite",
    # Archives.
    ".zip", ".tar", ".tgz", ".gz", ".bz2", ".xz", ".7z", ".rar",
    # Notebooks carry cell output, execution history, and path metadata.
    ".ipynb",
    # Key material.
    ".pem", ".key", ".p12", ".pfx", ".jks", ".ppk",
}
FORBIDDEN_FILENAMES = {
    "train_labels.csv", "submission.csv",
    ".env", ".env.local", ".env.production", ".envrc",
    "kaggle.json", "credentials.json", ".netrc", "_netrc",
    "id_rsa", "id_dsa", "id_ecdsa", "id_ed25519", ".npmrc", ".pypirc",
}
# Any file whose name ends with one of these is rejected regardless of suffix
# parsing, so `prod.env` and `secrets.env` cannot slip through as ".env" files.
FORBIDDEN_NAME_ENDINGS = (".env",)
TEXT_SUFFIXES = {
    ".py", ".md", ".txt", ".json", ".toml", ".yaml", ".yml", ".cff",
    ".example", ".cfg", ".ini", ".sh", ".gitattributes",
}
TEXT_FILENAMES = {"LICENSE", "NOTICE", "Makefile", "Dockerfile", ".gitignore"}

# A placeholder mount is the documented way to describe a private dataset, so it
# must not trip the scan that looks for a real one.
PLACEHOLDER_MOUNTS = re.compile(r"YOUR_[A-Z0-9_]+|<[^>\s]+>|\$\{?[A-Z_][A-Z0-9_]*\}?")

TEXT_PATTERNS = {
    # A concrete user-owned Kaggle dataset mount. Anchored per line, and not
    # tied to one historical username: any non-placeholder slug is rejected.
    "user-specific Kaggle mount": re.compile(
        r"/kaggle/input/(?:datasets/)?(?!\s)([A-Za-z0-9][-\w.]*)", re.MULTILINE
    ),
    # Real Windows paths, single- or double-escaped, either slash, plus UNC.
    "Windows user path": re.compile(
        r"(?:[A-Za-z]:[\/]{1,2}Users[\/]|\{2}[A-Za-z0-9._-]+\[A-Za-z0-9._$-]+)"
    ),
    "local filesystem path": re.compile(
        r"/(?:workspace|home|Users|mnt|media|srv|root|content)/[^\s'\"`)\]}]+"
    ),
    "private key block": re.compile(
        r"-----BEGIN (?:RSA |DSA |EC |OPENSSH |PGP )?PRIVATE KEY(?: BLOCK)?-----"
        r"|PuTTY-" r"User-Key-File"
    ),
    "credential assignment": re.compile(
        r"(?i)(?:api[_-]?key|secret|password|passwd|token|auth)"
        r"\s*[:=]\s*['\"][^'\"\s]{8,}['\"]"
    ),
    "credential in environment assignment": re.compile(
        r"(?i)\b[A-Z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD)\b\s*[:=]\s*['\"][^'\"\s]{8,}['\"]"
    ),
    "credential-shaped literal": re.compile(
        r"\bghp_[A-Za-z0-9]{36}\b|\bgithub_pat_[A-Za-z0-9_]{22,}\b"
        r"|\bsk-(?:ant-)?[A-Za-z0-9_-]{20,}\b|\bAKIA[0-9A-Z]{16}\b"
        r"|\bxox[baprs]-[A-Za-z0-9-]{10,}\b|\bhf_[A-Za-z0-9]{34}\b"
        r"|\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"
    ),
    "temporary directory literal": re.compile(r"(?<![\w.])/t" r"mp/[^\s'\"`)\]}]+"),
    "email address": re.compile(
        r"[A-Za-z0-9._%+-]+@(?!example\.)[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
    ),
}
MAX_FILE_BYTES = 8 * 1024 * 1024
MAX_TEXT_SCAN_BYTES = 4 * 1024 * 1024


def is_text(path) -> bool:
    return path.suffix.lower() in TEXT_SUFFIXES or path.name in TEXT_FILENAMES


def scan_text(path, rel: str, errors: list[str]) -> None:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        errors.append(f"text file is not UTF-8: {rel}")
        return
    for label, pattern in TEXT_PATTERNS.items():
        for match in pattern.finditer(text):
            found = match.group(0)
            if PLACEHOLDER_MOUNTS.search(found):
                continue
            line = text.count("\n", 0, match.start()) + 1
            errors.append(f"{label}: {rel}:{line}: {found[:80]}")
            break


def file_errors(files) -> list[str]:
    """Per-file privacy and hygiene checks, with no manifest comparison.

    Exposed separately so `update_manifest.py` can refuse to bless a tree that
    would fail these checks, instead of laundering a leak into a signed manifest.
    """
    errors: list[str] = []
    for path in files:
        rel = path.relative_to(ROOT).as_posix()
        if path.is_symlink():
            errors.append(f"symlink is not allowed in a release tree: {rel}")
            continue
        lower_name = path.name.lower()
        if lower_name in FORBIDDEN_FILENAMES or lower_name.endswith(FORBIDDEN_NAME_ENDINGS):
            errors.append(f"forbidden filename: {rel}")
        if {suffix.lower() for suffix in path.suffixes} & FORBIDDEN_SUFFIXES:
            errors.append(f"forbidden binary/data suffix: {rel}")
        size = path.stat().st_size
        if size > MAX_FILE_BYTES:
            errors.append(f"unexpected file larger than 8 MiB: {rel}")
            continue
        if is_text(path) and size <= MAX_TEXT_SCAN_BYTES:
            scan_text(path, rel, errors)
    return errors


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--privacy-only",
        action="store_true",
        help=(
            "run the per-file privacy and hygiene checks and skip manifest "
            "comparison. The manifest pins a published snapshot, so it is a "
            "release check: editing a document should not fail an ordinary push, "
            "but leaking a file always should."
        ),
    )
    args = parser.parse_args(argv)

    files = public_files()
    errors = file_errors(files)

    if args.privacy_only:
        if errors:
            print("PRIVACY VERIFICATION FAILED", file=sys.stderr)
            for error in errors:
                print(f"- {error}", file=sys.stderr)
            return 1
        print(f"PRIVACY VERIFICATION PASSED ({len(files)} files, manifest not compared)")
        return 0

    if MANIFEST.is_file():
        try:
            manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
            expected = manifest["files"]
        except Exception as exc:
            errors.append(f"invalid release manifest: {type(exc).__name__}")
        else:
            if manifest.get("schema") != MANIFEST_SCHEMA:
                errors.append(
                    f"manifest schema is {manifest.get('schema')!r}, expected {MANIFEST_SCHEMA!r}"
                )
            if manifest.get("self_excluded") is not True:
                errors.append("manifest does not declare self_excluded: true")
            # Compared as a mapping so the result never depends on the platform's
            # path sort order, and so a mismatch names the files responsible.
            want = {entry["path"]: entry for entry in expected}
            have = {entry["path"]: entry for entry in (record(p) for p in files)}
            for rel in sorted(set(want) - set(have)):
                errors.append(f"manifest lists a file that is not present: {rel}")
            for rel in sorted(set(have) - set(want)):
                errors.append(f"file is present but absent from the manifest: {rel}")
            for rel in sorted(set(want) & set(have)):
                if want[rel] != have[rel]:
                    errors.append(
                        f"content differs from the manifest: {rel} "
                        f"(manifest {want[rel]['bytes']}B/{want[rel]['sha256'][:12]}, "
                        f"actual {have[rel]['bytes']}B/{have[rel]['sha256'][:12]})"
                    )
    else:
        errors.append("PUBLIC_RELEASE_MANIFEST.json is missing")

    if errors:
        print("PUBLIC RELEASE VERIFICATION FAILED", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        if any(error.startswith(("content differs", "file is present", "manifest lists"))
               for error in errors):
            print(
                "\nThe manifest is stale. If those changes are intended, run:\n"
                "    python scripts/update_manifest.py\n"
                "It re-runs the privacy checks first and refuses to write if any fail.",
                file=sys.stderr,
            )
        return 1
    print(f"PUBLIC RELEASE VERIFICATION PASSED ({len(files)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
