#!/bin/bash
set -e

# Setup script for News Pipeline dependencies
# This applies patches to the news-please submodule to enable custom filtering.

# Get repo root
REPO_ROOT="$(git rev-parse --show-toplevel)"
NEWS_PLEASE_DIR="$REPO_ROOT/data/news/news-please"
PATCH_FILE="$REPO_ROOT/data/news/news-please.patch"

echo "=== Setting up News Pipeline Dependencies ==="

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
