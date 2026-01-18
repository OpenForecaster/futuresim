#!/usr/bin/env python
"""Build IVF index on LanceDB for fast vector search.

Run this on a GPU node with sufficient memory (80GB+).
Usage: python scripts/build_lancedb_index.py
"""

import argparse
import lancedb

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_path", default="/is/cluster/fast/sgoel/forecasting/news/deduped_articles/lance/Qwen3-Embedding-8B")
    parser.add_argument("--table_name", default="articles")
    parser.add_argument("--num_partitions", type=int, default=256, help="IVF partitions")
    parser.add_argument("--num_sub_vectors", type=int, default=64, help="PQ sub-vectors (must divide vector dim, e.g. 64 for 4096-dim)")
    parser.add_argument("--metric", default="cosine", choices=["cosine", "L2", "dot"])
    args = parser.parse_args()
    
    print(f"Connecting to: {args.db_path}")
    db = lancedb.connect(args.db_path)
    table = db.open_table(args.table_name)
    
    row_count = table.count_rows()
    print(f"Table '{args.table_name}' has {row_count:,} rows")
    
    # Check if vector index already exists (ignore FTS index)
    indices = table.list_indices()
    vector_indices = [idx for idx in indices if "FTS" not in str(idx)]
    if vector_indices:
        print(f"Existing vector indices: {vector_indices}")
        response = input("Vector index already exists. Rebuild? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    elif indices:
        print(f"Found FTS index (for keyword search), but no vector index yet. Proceeding...")
    
    print(f"\nBuilding IVF-PQ index...")
    print(f"  Partitions: {args.num_partitions}")
    print(f"  Sub-vectors: {args.num_sub_vectors}")
    print(f"  Metric: {args.metric}")
    print("This may take 30-60 minutes for 14M+ rows...")
    
    table.create_index(
        metric=args.metric,
        num_partitions=args.num_partitions,
        num_sub_vectors=args.num_sub_vectors,
        replace=True  # Replace existing index if any
    )
    
    print("\n✅ Index created successfully!")
    print("Search queries should now be 10-50x faster.")

if __name__ == "__main__":
    main()
