#!/usr/bin/env python
"""Build LanceDB search indices (stage 2/2 after ingest).

Usage:
  # Build FTS + vector index:
  python scripts/build_lancedb_index.py --build_fts --force

  # Build FTS only (recommended first retry path on large tables):
  python scripts/build_lancedb_index.py --build_fts --skip_vector_index --force

  # Build vector index only:
  python scripts/build_lancedb_index.py --force
"""

import argparse
import hashlib
import os
import shutil
from pathlib import Path

import lancedb


def _is_fts_index(idx_obj) -> bool:
    return "FTS" in str(idx_obj).upper()


def _tantivy_likely_unsupported(db_path: str) -> bool:
    """Heuristic: /is filesystem lacks required lockfile semantics for Tantivy."""
    return db_path.startswith("/is/")


def _build_tantivy_index_external(
    table,
    field_name: str,
    replace: bool,
    tantivy_index_root: str,
) -> tuple[str, str]:
    """
    Build Tantivy index under a lock-capable filesystem and symlink table fts path to it.
    """
    from lancedb.fts import create_index, populate_index

    fts_path, _, _ = table._get_fts_index_path()
    fts_path = Path(fts_path)
    digest = hashlib.sha1(str(fts_path).encode("utf-8")).hexdigest()[:16]
    target = Path(tantivy_index_root) / f"{table.name}_{digest}"

    if replace and target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)

    idx = create_index(str(target), [field_name])
    populate_index(idx, table, [field_name])

    fts_path.parent.mkdir(parents=True, exist_ok=True)
    if fts_path.is_symlink() or fts_path.is_file():
        fts_path.unlink()
    elif fts_path.exists():
        shutil.rmtree(fts_path)
    os.symlink(str(target), str(fts_path))

    return str(target), str(fts_path)


def _has_fts_index(table, indices) -> bool:
    """Detect FTS availability for both native and tantivy-backed indices."""
    if any(_is_fts_index(idx) for idx in indices):
        return True

    # Tantivy-backed FTS may not appear in list_indices() on some versions.
    try:
        table.search("__lancedb_fts_probe_token__").limit(1).to_list()
        return True
    except Exception as e:
        msg = str(e)
        if "Cannot perform full text search unless an INVERTED index" in msg:
            return False
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_path", default="/is/cluster/fast/sgoel/forecasting/news/deduped_articles/lance/Qwen3-Embedding-8B")
    parser.add_argument("--table_name", default="articles")
    parser.add_argument("--num_partitions", type=int, default=256, help="IVF partitions")
    parser.add_argument("--num_sub_vectors", type=int, default=64, help="PQ sub-vectors (must divide vector dim, e.g. 64 for 4096-dim)")
    parser.add_argument("--metric", default="cosine", choices=["cosine", "L2", "dot"])
    parser.add_argument(
        "--build_fts",
        action="store_true",
        help="Build full-text index on 'content' before vector index",
    )
    fts_position_group = parser.add_mutually_exclusive_group()
    fts_position_group.add_argument(
        "--fts_with_position",
        dest="fts_with_position",
        action="store_true",
        help="Enable phrase-position support in FTS (default)",
    )
    fts_position_group.add_argument(
        "--no_fts_with_position",
        dest="fts_with_position",
        action="store_false",
        help="Disable phrase-position support in FTS (lower memory usage)",
    )
    fts_backend_group = parser.add_mutually_exclusive_group()
    fts_backend_group.add_argument(
        "--fts_use_tantivy",
        dest="fts_use_tantivy",
        action="store_true",
        help="Use Tantivy backend for FTS (recommended for large tables)",
    )
    fts_backend_group.add_argument(
        "--fts_use_native",
        dest="fts_use_tantivy",
        action="store_false",
        help="Use native Lance FTS backend",
    )
    parser.add_argument(
        "--skip_vector_index",
        action="store_true",
        help="Skip IVF-PQ vector index creation (FTS-only run)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Non-interactive mode: rebuild existing indices without prompt",
    )
    parser.add_argument(
        "--accelerator",
        default=None,
        choices=["cuda"],
        help="Use GPU accelerator for IVF-PQ vector index creation (e.g. 'cuda')",
    )
    parser.add_argument(
        "--tantivy_index_root",
        default=os.environ.get(
            "TANTIVY_INDEX_ROOT",
            "/lustre/home/sgoel/forecasting/lancedb_tantivy_indices",
        ),
        help=(
            "Root directory for external Tantivy index data when DB path lacks lock support "
            "(default: %(default)s)"
        ),
    )
    parser.set_defaults(fts_with_position=True, fts_use_tantivy=True)
    args = parser.parse_args()
    
    print(f"Connecting to: {args.db_path}")
    db = lancedb.connect(args.db_path)
    table = db.open_table(args.table_name)
    
    row_count = table.count_rows()
    print(f"Table '{args.table_name}' has {row_count:,} rows")
    
    indices = table.list_indices()
    print(f"Existing indices: {indices}")

    if args.build_fts:
        effective_use_tantivy = args.fts_use_tantivy
        use_external_tantivy = False
        if effective_use_tantivy and _tantivy_likely_unsupported(args.db_path):
            use_external_tantivy = True

        backend = "tantivy" if effective_use_tantivy else "native-lance"
        if use_external_tantivy:
            backend = f"tantivy-external({args.tantivy_index_root})"
        has_fts = _has_fts_index(table, indices)
        if has_fts and not args.force:
            print("Detected existing FTS index")
            response = input("FTS index already exists. Rebuild? [y/N]: ")
            if response.lower() != "y":
                print("Skipping FTS rebuild.")
            else:
                print(
                    "\nBuilding FTS index "
                    f"(with_position={args.fts_with_position}, backend={backend})..."
                )
                if use_external_tantivy:
                    target, link = _build_tantivy_index_external(
                        table=table,
                        field_name="content",
                        replace=True,
                        tantivy_index_root=args.tantivy_index_root,
                    )
                    print(f"External Tantivy index path: {target}")
                    print(f"Linked table FTS path: {link}")
                else:
                    table.create_fts_index(
                        "content",
                        with_position=args.fts_with_position,
                        use_tantivy=effective_use_tantivy,
                        replace=True,
                    )
                print("✅ FTS index created successfully")
        else:
            print(
                "\nBuilding FTS index "
                f"(with_position={args.fts_with_position}, backend={backend})..."
            )
            try:
                if use_external_tantivy:
                    target, link = _build_tantivy_index_external(
                        table=table,
                        field_name="content",
                        replace=True,
                        tantivy_index_root=args.tantivy_index_root,
                    )
                    print(f"External Tantivy index path: {target}")
                    print(f"Linked table FTS path: {link}")
                else:
                    table.create_fts_index(
                        "content",
                        with_position=args.fts_with_position,
                        use_tantivy=effective_use_tantivy,
                        replace=True,
                    )
            except ImportError as e:
                if effective_use_tantivy:
                    raise ImportError(
                        "Tantivy backend requested but dependencies are missing. "
                        "Install with: pip install tantivy pylance"
                    ) from e
                raise
            print("✅ FTS index created successfully")
        print(f"FTS available after build: {_has_fts_index(table, table.list_indices())}")

    if args.skip_vector_index:
        print("Skipping vector index (--skip_vector_index)")
        final_indices = table.list_indices()
        print(f"Final indices: {final_indices}")
        print(f"FTS available: {_has_fts_index(table, final_indices)}")
        return

    # Check if vector index already exists (ignore FTS index)
    indices = table.list_indices()
    vector_indices = [idx for idx in indices if not _is_fts_index(idx)]
    if vector_indices and not args.force:
        print(f"Existing vector indices: {vector_indices}")
        response = input("Vector index already exists. Rebuild? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    elif not vector_indices and indices:
        print("Found FTS index (keyword search), but no vector index yet. Proceeding...")
    
    print(f"\nBuilding IVF-PQ index...")
    print(f"  Partitions: {args.num_partitions}")
    print(f"  Sub-vectors: {args.num_sub_vectors}")
    print(f"  Metric: {args.metric}")
    print("This may take 30-60 minutes for 14M+ rows...")
    
    create_kwargs = dict(
        metric=args.metric,
        num_partitions=args.num_partitions,
        num_sub_vectors=args.num_sub_vectors,
        replace=True,
    )
    if args.accelerator:
        create_kwargs["accelerator"] = args.accelerator
        print(f"  Accelerator: {args.accelerator}")
    table.create_index(**create_kwargs)
    
    print("\n✅ Index created successfully!")
    final_indices = table.list_indices()
    print(f"Final indices: {final_indices}")
    print(f"FTS available: {_has_fts_index(table, final_indices)}")
    print("Search queries should now be faster.")

if __name__ == "__main__":
    main()
