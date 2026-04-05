#!/usr/bin/env python3
"""
Deduplicate news articles in JSONL files.

Deduplicates based on title + maintext hash to remove exact duplicates.
Creates a 'deduped/' subdirectory in the input directory.

Usage:
    python deduplicate_news_jsonl.py --jsonl_path /path/to/jsonl --num_workers 16
"""

import argparse
import json
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.news_article_dedup import article_identity_key, merge_article_records


def deduplicate_jsonl_file(file_path: str, output_dir: str) -> int:
    """
    Deduplicate a single JSONL file using stable article identity keys.
    The output file is rewritten atomically on every run.
    
    Args:
        file_path: Path to the JSONL file
        output_dir: Directory to store deduplicated file
        
    Returns:
        Number of duplicates removed
    """
    file_name = os.path.basename(file_path)
    output_path = os.path.join(output_dir, file_name)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    articles_by_key: dict[str, dict] = {}
    ordered_keys: list[str] = []
    duplicates_removed = 0
    anonymous_counter = 0

    with open(file_path, 'r', encoding='utf-8') as input_file:
        lines = input_file.readlines()

    for line in tqdm(lines, desc=f"Processing {file_name}", leave=False, position=1):
        try:
            article = json.loads(line)

            content_key = article_identity_key(article)
            if content_key is None:
                content_key = f"anonymous:{anonymous_counter}"
                anonymous_counter += 1

            if content_key not in articles_by_key:
                articles_by_key[content_key] = article
                ordered_keys.append(content_key)
            else:
                articles_by_key[content_key] = merge_article_records(articles_by_key[content_key], article)
                duplicates_removed += 1
        except json.JSONDecodeError:
            continue

    fd, tmp_path = tempfile.mkstemp(
        dir=os.path.dirname(output_path),
        prefix=f".{file_name}.",
        suffix=".tmp",
        text=True,
    )
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as output_file:
            for content_key in ordered_keys:
                output_file.write(json.dumps(articles_by_key[content_key], ensure_ascii=False) + '\n')
        os.replace(tmp_path, output_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    
    return duplicates_removed


def process_directory(directory_path: str, num_workers: int) -> None:
    """
    Find all JSONL files in a directory and deduplicate them in parallel.
    
    Args:
        directory_path: Directory containing JSONL files
        num_workers: Number of parallel workers
    """
    jsonl_files = list(Path(directory_path).glob('*.jsonl'))
    total_files = len(jsonl_files)
    
    print(f"Found {total_files} JSONL files in {directory_path}")
    
    deduped_dir = os.path.join(directory_path, "deduped")
    os.makedirs(deduped_dir, exist_ok=True)
    print(f"Created output directory: {deduped_dir}")
    
    # Limit workers to avoid I/O bottleneck
    num_workers = min(num_workers, 16)
    
    total_duplicates = 0
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_file = {executor.submit(deduplicate_jsonl_file, str(file_path), deduped_dir): file_path 
                         for file_path in jsonl_files}
        
        for future in tqdm(future_to_file, total=total_files, desc="Deduplicating files"):
            file_path = future_to_file[future]
            try:
                duplicates_removed = future.result()
                total_duplicates += duplicates_removed
                print(f"Processed {file_path.name}: Removed {duplicates_removed} duplicates")
            except Exception as e:
                print(f"Error processing {file_path.name}: {e}")
    
    print(f"Deduplication complete. Total duplicates removed: {total_duplicates}")
    print(f"Deduplicated files saved to: {deduped_dir}")


def main():
    parser = argparse.ArgumentParser(description="Deduplicate news articles in JSONL files")
    parser.add_argument("--jsonl_path", type=str, required=True, 
                        help="Directory containing JSONL files to deduplicate")
    parser.add_argument("--num_workers", type=int, default=16,
                        help="Number of parallel workers (default: 16)")
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.jsonl_path):
        print(f"Error: {args.jsonl_path} is not a valid directory")
        return
    
    process_directory(args.jsonl_path, args.num_workers)


if __name__ == "__main__":
    main()
