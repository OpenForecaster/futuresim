#!/usr/bin/env python3
"""Check that FSIM_SEARCH_DB points to a usable LanceDB search table."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import lancedb


REQUIRED_FIELDS = {
    "chunk_id",
    "article_id",
    "chunk_index",
    "title",
    "source",
    "date",
    "content",
    "vector",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db-path",
        default=os.environ.get("FSIM_SEARCH_DB", "artifacts/forecast-news-embeddings"),
        help="LanceDB directory. Defaults to FSIM_SEARCH_DB.",
    )
    parser.add_argument("--table", default="articles")
    args = parser.parse_args()

    db_path = Path(args.db_path).expanduser()
    table_path = db_path / f"{args.table}.lance"
    if not table_path.exists():
        print(f"search_ready: NO")
        print(f"missing table path: {table_path}")
        return 1

    db = lancedb.connect(str(db_path))
    table = db.open_table(args.table)
    fields = set(table.schema.names)
    missing = sorted(REQUIRED_FIELDS - fields)

    print(f"db_path: {db_path}")
    print(f"table:   {args.table}")
    print(f"rows:    {table.count_rows():,}")
    print(f"fields:  {', '.join(table.schema.names)}")

    if missing:
        print("search_ready: NO")
        print(f"missing fields: {', '.join(missing)}")
        return 1

    print("search_ready: YES")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
