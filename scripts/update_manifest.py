"""Regenerate the deterministic public release manifest after reviewed edits.

This refuses to write when the tree would fail the verifier's per-file privacy
checks, so the manifest can never be used to bless a leak. Run
`scripts/verify_release.py` afterwards to confirm the result.
"""
from __future__ import annotations

import argparse
import datetime as _datetime
import json
import sys

from release_common import MANIFEST, MANIFEST_SCHEMA, public_files, record
from verify_release import file_errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release-date",
        default=_datetime.date.today().isoformat(),
        help="value recorded as generated_for_release_date (default: today)",
    )
    parser.add_argument(
        "--allow-failing-checks",
        action="store_true",
        help="write the manifest even if per-file privacy checks fail (not for releases)",
    )
    args = parser.parse_args()

    files = public_files()
    errors = file_errors(files)
    if errors and not args.allow_failing_checks:
        print("REFUSING TO WRITE THE MANIFEST: per-file checks failed", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    payload = {
        "schema": MANIFEST_SCHEMA,
        "generated_for_release_date": args.release_date,
        "self_excluded": True,
        "files": [record(path) for path in files],
    }
    # newline="" disables translation. The manifest pins byte content, so it must
    # be LF on every platform: written with the default translation on Windows it
    # becomes CRLF, and the same tree would then hash differently across machines.
    MANIFEST.write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
        newline="",
    )
    print(f"wrote {MANIFEST.name} with {len(files)} files ({args.release_date})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
