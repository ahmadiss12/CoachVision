"""Write the OpenAPI document to a file, without starting a server.

    python scripts/export_openapi.py            # -> backend/openapi.json
    python scripts/export_openapi.py --check    # fail if the file is stale

The committed document is what the mobile client's types are generated from
(see docs/TYPED_CONTRACT.md). Exporting it offline means CI can regenerate and
diff the client types without booting the app, and it makes a schema change
visible in the pull request that causes it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

DEFAULT_OUT = BACKEND_ROOT / "openapi.json"


def build_document() -> dict:
    # Importing the app constructs Settings, which refuses to load without a
    # usable JWT secret outside development. Nothing here touches the database.
    os.environ.setdefault("ENVIRONMENT", "development")

    from coachvision.main import app  # noqa: PLC0415 - import after sys.path setup

    return app.openapi()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if the file on disk does not match the live app.",
    )
    args = parser.parse_args()

    # sort_keys keeps the diff meaningful: without it, dict ordering churn
    # shows up as a spurious schema change on every export.
    document = json.dumps(build_document(), indent=2, sort_keys=True) + "\n"

    if args.check:
        if not args.out.exists():
            print(f"{args.out} does not exist. Run: python scripts/export_openapi.py", file=sys.stderr)
            return 1
        if args.out.read_text() != document:
            print(
                f"{args.out.name} is out of date with the app.\n"
                "Run: python scripts/export_openapi.py\n"
                "Then regenerate the client types: cd mobile && npm run generate:api",
                file=sys.stderr,
            )
            return 1
        print(f"{args.out.name} is up to date.")
        return 0

    args.out.write_text(document)
    print(f"Wrote {args.out} ({len(document):,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
