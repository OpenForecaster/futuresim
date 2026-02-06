#!/bin/bash
# CCNews Article Extraction
# 
# Downloads and extracts articles from CommonCrawl News dataset using news-please.
# Filters to specific domains and date range.
#
# Usage: ./run_news_crawl.sh <start_date> [end_date]
# Example: ./run_news_crawl.sh 2025-08-01 2026-01-31

set -euo pipefail

# Parse arguments
START_DATE=${1:-2025-08-01}
END_DATE=${2:-}  # Empty means no end date (download up to now)

echo "=== CCNews Download ==="
echo "Start date: $START_DATE"
echo "End date: ${END_DATE:-none (up to latest)}"

# Setup PATH
export PATH="/usr/bin:/bin:/usr/sbin:/sbin:$PATH"
source ~/.bashrc 2>/dev/null || true

# Export dates as environment variables for commoncrawl.py
export NEWS_START_DATE="$START_DATE"
export NEWS_END_DATE="$END_DATE"

# Paths
WARC_DIR="/is/cluster/fast/sgoel/forecasting/news/filtered_cc_warc_2025_2026"
ARTICLES_DIR="/is/cluster/fast/sgoel/forecasting/news/filtered_cc_articles_2025_2026"
REPO_DIR="/home/sgoel/forecast-sim/data/news"

# Create directories
mkdir -p "$WARC_DIR"
mkdir -p "$ARTICLES_DIR"

# Activate environment
source ~/forecast-sim/fsim/bin/activate

# Navigate to news-please
cd "$REPO_DIR/news-please"

echo "Starting CCNews extraction..."
echo "WARC dir: $WARC_DIR"
echo "Articles dir: $ARTICLES_DIR"

# Run commoncrawl script
# Args: warc_dir, articles_dir, delete_warc, num_extractors
python3 -m newsplease.examples.commoncrawl \
    "$WARC_DIR" \
    "$ARTICLES_DIR" \
    delete \
    1

echo "CCNews extraction complete!"

