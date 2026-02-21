from __future__ import annotations

import argparse
import json
from pathlib import Path

from hpcopt.api.app import app


def main() -> int:
    parser = argparse.ArgumentParser(description="Export current FastAPI OpenAPI spec as baseline.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("schemas/openapi_baseline.json"),
        help="Output baseline JSON path.",
    )
    args = parser.parse_args()

    payload = app.openapi()
    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote OpenAPI baseline: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
