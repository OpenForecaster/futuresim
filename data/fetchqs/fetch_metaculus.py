#!/usr/bin/env python3
"""
Fetch Metaculus data and cache it locally.
Updates existing cache without overwriting.
"""

import argparse
import sys
import os
from datetime import datetime
from collections import defaultdict

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from data.fetchqs import get_fetcher
from data.fetchqs.base import CachedQuestion

def main():
    parser = argparse.ArgumentParser(description="Fetch Metaculus questions.")
    parser.add_argument("--type", choices=["binary", "mcq"], required=True, 
                       help="Question type to fetch")
    parser.add_argument("--start_date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end_date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--cache_dir", default="/is/cluster/fast/sgoel/forecasting/qs/cache",
                       help="Directory to save parquet cache")
    parser.add_argument("--min_forecasters", type=int, default=10,
                       help="Minimum number of forecasters required (default: 10)")
    
    args = parser.parse_args()
    
    start = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    end = datetime.strptime(args.end_date, "%Y-%m-%d").date()
    
    dataset_name = f"metaculus_{args.type}"
    if args.type == "mcq":
        dataset_name = "metaculus_mcq"
        
    # We instantiate fetcher directly or via get_fetcher? 
    # get_fetcher needs to be updated to accept min_forecasters or we use custom init.
    # Updating get_fetcher is cleaner.
    from data.fetchqs.metaculus import MetaculusFetcher
    fetcher = MetaculusFetcher(type_filter="multiple_choice" if args.type == "mcq" else "binary", 
                              min_forecasters=args.min_forecasters)
    
    # Fetch new questions
    questions = fetcher.fetch_new(start, end)
    
    # Save/Merge to cache
    fetcher.save_to_cache(questions, args.cache_dir)
    
    # Load ALL cached questions to generate full summary
    all_questions = fetcher.load_from_cache(args.cache_dir)
    
    # Generate Monthly Summary
    print("\n" + "="*50)
    print("MONTHLY SUMMARY (All Cached Data)")
    print("="*50)
    
    summary = defaultdict(int)
    for q in all_questions:
        month_key = q.resolution_date.strftime("%Y-%m")
        summary[month_key] += 1
        
    total_qs = 0
    for month in sorted(summary.keys()):
        count = summary[month]
        total_qs += count
        print(f"{month}: {count} questions")
        
    print("-" * 50)
    print(f"Total Available Questions: {total_qs}")
    
    # Show stats for current filter settings
    if args.min_forecasters > 0:
        filtered_count = sum(1 for q in all_questions if q.metadata and q.metadata.get('nr_forecasters', 0) >= args.min_forecasters)
        print(f"Total satisfying min_forecasters>={args.min_forecasters}: {filtered_count}")
        
    print("="*50)

if __name__ == "__main__":
    main()
