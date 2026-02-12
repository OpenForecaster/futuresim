#!/bin/bash
set -e

# Setup script for News Pipeline dependencies
# This applies patches to the news-please submodule to enable custom filtering.

# Get repo root
REPO_ROOT="$(git rev-parse --show-toplevel)"
NEWS_PLEASE_DIR="$REPO_ROOT/data/news/news-please"
PATCH_FILE="$REPO_ROOT/data/news/news-please.patch"

echo "=== Setting up News Pipeline Dependencies ==="

# 0. Ensure NLTK data is available (avoid downloads inside HTCondor jobs)
echo "Ensuring NLTK tokenizer data (punkt_tab/punkt) is available..."
export NLTK_DATA="${NLTK_DATA:-$REPO_ROOT/fsim/nltk_data}"
mkdir -p "$NLTK_DATA"
if python3 "$REPO_ROOT/data/news/scripts/ensure_nltk_data.py"; then
    echo "  [SUCCESS] NLTK data ready."
else
    echo "  [WARNING] Could not ensure NLTK data. CCNews crawl may error with missing punkt_tab."
    echo "            Try: source ~/forecast-sim/fsim/bin/activate && python3 $REPO_ROOT/data/news/scripts/ensure_nltk_data.py"
fi

# 1. Initialize submodule if needed
if [ ! -d "$NEWS_PLEASE_DIR/.git" ]; then
    echo "Initializing news-please submodule..."
    git submodule update --init --recursive "$NEWS_PLEASE_DIR"
fi

# 2. Apply patch
echo "Checking patches..."
cd "$NEWS_PLEASE_DIR"

# Check if patch is already applied or file is modified
if ! git diff --quiet; then
    echo "  [INFO] news-please submodule has modified content. Assuming patch is applied or manually modified. Skipping."
else
    # Try to apply patch
    if git apply --check "$PATCH_FILE"; then
        git apply "$PATCH_FILE"
        echo "  [SUCCESS] Patch applied to news-please."
    else
        echo "  [WARNING] Patch failed to apply cleanly. Please check $NEWS_PLEASE_DIR manually."
    fi
fi

echo "=== Setup Complete ==="
