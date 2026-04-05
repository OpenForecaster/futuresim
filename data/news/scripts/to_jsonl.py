#!/usr/bin/env python3
"""
Convert individual JSON article files to JSONL format.

Takes a directory containing subdirectories (one per domain) with JSON files
and converts each domain's articles into a single JSONL file.

Usage:
    python to_jsonl.py /path/to/articles --output_dir /path/to/jsonl --workers 48
"""

import json
import os
import random
import time
import concurrent.futures
import argparse
import subprocess
from tqdm import tqdm
from pathlib import Path


def log_message(log_file, message):
    """Log a message to both console and log file"""
    print(message, flush=True)
    if log_file:
        with open(log_file, 'a') as f:
            f.write(message + '\n')


def process_directory(base_dir, subdir, output_dir, verify_sample, delete_jsons, log_file):
    """Process a single depth-1 subdirectory of JSON files into a JSONL file"""
    subdir_path = os.path.join(base_dir, subdir)
    jsonl_file = os.path.join(output_dir, f"{subdir}.jsonl")
    
    try:
        cmd = ["find", subdir_path, "-type", "f", "-name", "*.json"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        json_files = result.stdout.strip().split('\n')
        json_files = [f for f in json_files if f]
        json_files.sort()
        
        if not json_files:
            return 0, 0, 0
        
        log_message(log_file, f"Processing {subdir}: found {len(json_files)} JSON files")
        
        processed_count = 0
        verified_count = 0
        doc_ids_to_verify = []
        
        os.makedirs(os.path.dirname(jsonl_file), exist_ok=True)
        
        with open(jsonl_file, 'w', encoding='utf-8') as outfile:
            for file_path in tqdm(json_files, desc=f"Processing {subdir}", leave=False):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        article = json.load(f)
                    
                    doc_id = os.path.splitext(os.path.basename(file_path))[0]
                    article['id'] = doc_id
                    outfile.write(json.dumps(article) + '\n')
                    processed_count += 1
                    
                    if verify_sample > 0 and random.random() < verify_sample:
                        doc_ids_to_verify.append((doc_id, article, file_path))
                
                except Exception as e:
                    log_message(log_file, f"Error processing {file_path}: {str(e)}")
        
        delete_count = 0
        if doc_ids_to_verify:
            verified_count = verify_jsonl(jsonl_file, doc_ids_to_verify, log_file)
            
            if delete_jsons and verified_count == len(doc_ids_to_verify):
                for file_path in json_files:
                    try:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                            delete_count += 1
                    except Exception as e:
                        log_message(log_file, f"Error deleting {file_path}: {str(e)}")
        elif delete_jsons:
            log_message(log_file, f"No files were verified, but deletion was requested")
            for file_path in json_files:
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        delete_count += 1
                except Exception as e:
                    log_message(log_file, f"Error deleting {file_path}: {str(e)}")
        
        return processed_count, verified_count, delete_count
    
    except subprocess.SubprocessError as e:
        log_message(log_file, f"Error finding JSON files in {subdir_path}: {str(e)}")
        return 0, 0, 0
    except Exception as e:
        log_message(log_file, f"Error processing directory {subdir_path}: {str(e)}")
        return 0, 0, 0


def verify_jsonl(jsonl_file, doc_ids_to_verify, log_file):
    """Verify that documents were correctly written to the JSONL file"""
    verified_count = 0
    
    try:
        doc_id_to_line = {}
        
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                try:
                    article = json.loads(line)
                    doc_id = article.get('id')
                    if doc_id:
                        doc_id_to_line[doc_id] = (line_num, article)
                except Exception as e:
                    log_message(log_file, f"Error parsing line {line_num} in {jsonl_file}: {str(e)}")
        
        for doc_id, original_article, file_path in doc_ids_to_verify:
            if doc_id in doc_id_to_line:
                line_num, stored_article = doc_id_to_line[doc_id]
                if stored_article == original_article:
                    verified_count += 1
                else:
                    log_message(log_file, f"Verification failed for {doc_id}: content mismatch")
            else:
                log_message(log_file, f"Verification failed for {doc_id}: not found in JSONL file")
        
        return verified_count
    
    except Exception as e:
        log_message(log_file, f"Error verifying JSONL file {jsonl_file}: {str(e)}")
        return 0


def parallel_convert_to_jsonl(json_dir, output_dir, max_workers=48,
                             verify_sample=0.01, delete_jsons=False,
                             resume_processed_dirs=False):
    """Convert JSON files to JSONL files in parallel"""
    start_time = time.time()
    
    os.makedirs(output_dir, exist_ok=True)
    
    log_file = os.path.join(output_dir, "conversion_log.txt")
    log_message(log_file, f"Starting JSONL conversion from {json_dir}")
    log_message(log_file, f"Output directory: {output_dir}")
    log_message(log_file, f"Workers: {max_workers}, Verify: {verify_sample*100}%, Delete JSONs: {delete_jsons}")
    
    try:
        subdirs = [d for d in os.listdir(json_dir) 
                  if os.path.isdir(os.path.join(json_dir, d))]
    except Exception as e:
        log_message(log_file, f"Error listing subdirectories in {json_dir}: {str(e)}")
        return 0
    
    total_subdirs = len(subdirs)
    log_message(log_file, f"Found {total_subdirs} subdirectories to process")
    
    processed_dirs_path = os.path.join(output_dir, "processed_dirs.txt")
    
    already_processed = set()
    if resume_processed_dirs and os.path.exists(processed_dirs_path):
        with open(processed_dirs_path, 'r') as f:
            already_processed = set(line.strip() for line in f)
        log_message(log_file, f"Resuming conversion. {len(already_processed)} subdirectories already processed.")
    elif os.path.exists(processed_dirs_path):
        log_message(log_file, f"Ignoring previous processed_dirs file and rebuilding JSONL outputs.")
    
    subdirs_to_process = [d for d in subdirs if d not in already_processed]
    
    log_message(log_file, f"Starting parallel conversion with {max_workers} workers...")
    
    total_processed = 0
    total_verified = 0
    total_deleted = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_subdir = {
            executor.submit(
                process_directory, 
                json_dir, 
                subdir,
                output_dir,
                verify_sample, 
                delete_jsons,
                log_file
            ): subdir for subdir in subdirs_to_process
        }
        
        mode = 'a' if resume_processed_dirs else 'w'
        with open(processed_dirs_path, mode) as processed_file:
            for future in tqdm(concurrent.futures.as_completed(future_to_subdir), total=len(subdirs_to_process)):
                subdir = future_to_subdir[future]
                try:
                    processed, verified, delete = future.result()
                    total_processed += processed
                    total_verified += verified
                    total_deleted += delete
                    processed_file.write(subdir + '\n')
                    processed_file.flush()
                    log_message(log_file, f"Completed {subdir}, processed {processed} documents, verified {verified}. Total: {total_processed}")
                except Exception as e:
                    log_message(log_file, f"ERROR in subdirectory {subdir}: {str(e)}")
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    log_message(log_file, f"Conversion complete. Total documents processed: {total_processed}")
    if total_processed > 0:
        log_message(log_file, f"Verification rate: {total_verified/total_processed:.2%}")
    log_message(log_file, f"Elapsed time: {elapsed_time:.2f} seconds")
    log_message(log_file, f"Processing rate: {total_processed/elapsed_time:.2f} documents/second")
    if total_deleted > 0:
        log_message(log_file, f"Deleted {total_deleted} documents")
    
    return total_processed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert JSON files to JSONL files")
    
    parser.add_argument('json_dir', type=str, 
                       help="Directory containing the JSON files to convert")
    
    parser.add_argument('--output_dir', type=str, default=None,
                       help="Output directory for JSONL files (default: parent of json_dir + /jsonl)")
    
    parser.add_argument('--workers', type=int, default=48,
                       help="Number of parallel workers to use (default: 48)")
    
    parser.add_argument('--verify', type=float, default=0.1,
                       help="Fraction of documents to verify (default: 0.1 = 10%%)")
    
    parser.add_argument('--delete', action='store_true',
                       help="Delete JSON files after successful conversion")

    parser.add_argument('--resume_processed_dirs', action='store_true',
                       help="Skip subdirectories already listed in processed_dirs.txt")
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(args.json_dir), "jsonl")
    
    parallel_convert_to_jsonl(
        args.json_dir,
        args.output_dir,
        max_workers=args.workers,
        verify_sample=args.verify,
        delete_jsons=args.delete,
        resume_processed_dirs=args.resume_processed_dirs,
    )
