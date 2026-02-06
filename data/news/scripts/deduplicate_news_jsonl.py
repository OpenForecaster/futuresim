#!/usr/bin/env python3
"""
Deduplicate news articles in JSONL files.

Deduplicates based on title + maintext hash to remove exact duplicates.
Creates a 'deduped/' subdirectory in the input directory.

Usage:
    python deduplicate_news_jsonl.py --jsonl_path /path/to/jsonl --num_workers 16
"""

import json
import os
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from typing import Set
from tqdm import tqdm
import hashlib
import random


def deduplicate_jsonl_file(file_path: str, output_dir: str) -> int:
    """
    Deduplicate a single JSONL file based on title and maintext using in-memory processing.
    If output file already exists, append new unique articles to it.
    
    Args:
        file_path: Path to the JSONL file
        output_dir: Directory to store deduplicated file
        
    Returns:
        Number of duplicates removed
    """
    file_name = os.path.basename(file_path)
    output_path = os.path.join(output_dir, file_name)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    seen_content: Set[str] = set()
    
    # If output file exists, read it first to build the set of existing content
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as existing_file:
            for line in tqdm(existing_file, desc=f"Reading existing {file_name}", leave=False, position=1):
                try:
                    article = json.loads(line)
                    title = article.get('title', '') or ''
                    maintext = article.get('maintext', '') or ''
                    content_key = hashlib.md5((title + maintext).encode('utf-8')).hexdigest()
                    seen_content.add(content_key)
                except json.JSONDecodeError:
                    continue
    
    # Read new file
    with open(file_path, 'r', encoding='utf-8') as input_file:
        lines = input_file.readlines()
    
    # Process all lines with a progress bar
    new_unique_articles = []
    duplicates_removed = 0
    
    for line in tqdm(lines, desc=f"Processing {file_name}", leave=False, position=1):
        try:
            article = json.loads(line)
            
            title = article.get('title', '') or ''
            maintext = article.get('maintext', '') or ''
            
            # Debug print occasionally
            if random.random() < 0.001:
                print(f"Title: {title[:80]}...")
                print(f"Date: {article.get('date_publish', '')}")
            
            content_key = hashlib.md5((title + maintext).encode('utf-8')).hexdigest()
            
            if content_key not in seen_content:
                seen_content.add(content_key)
                new_unique_articles.append(line)
            else:
                duplicates_removed += 1
                
        except json.JSONDecodeError:
            continue
    
    # Append new unique articles to the output file
    with open(output_path, 'a', encoding='utf-8') as output_file:
        output_file.writelines(new_unique_articles)
    
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
