#!/bin/bash
set -e

# Setup script for News Pipeline dependencies
# This applies patches to the news-please submodule to enable custom filtering.

# Get repo root
REPO_ROOT="$(git rev-parse --show-toplevel)"
NEWS_PLEASE_DIR="$REPO_ROOT/data/news/news-please"
PATCH_FILE="$REPO_ROOT/data/news/news-please.patch"
VENV_ACTIVATE="$REPO_ROOT/.venv/bin/activate"

echo "=== Setting up News Pipeline Dependencies ==="

# 0. Activate project venv (required for dependency checks/install)
if [ ! -f "$VENV_ACTIVATE" ]; then
    echo "ERROR: Missing virtualenv at $VENV_ACTIVATE"
    exit 2
fi
source "$VENV_ACTIVATE"

# Prefer pip if available in the active env; fall back to uv pip for uv-managed envs.
if python3 -m pip --version >/dev/null 2>&1; then
    PIP_INSTALL=(python3 -m pip install --quiet)
elif command -v uv >/dev/null 2>&1; then
    PIP_INSTALL=(uv pip install --quiet)
else
    echo "ERROR: Neither pip nor uv is available for dependency installation."
    exit 2
fi

# 1. Ensure language/tokenizer runtime dependency used by newspaper4k exists.
# Missing this causes noisy extraction failures for Bengali pages.
echo "Ensuring Python dependency indic-nlp-library is installed..."
if python3 -c "import indicnlp" >/dev/null 2>&1; then
    echo "  [SUCCESS] indic-nlp-library already available."
else
    if "${PIP_INSTALL[@]}" indic-nlp-library; then
        echo "  [SUCCESS] Installed indic-nlp-library."
    else
        echo "ERROR: Failed to install indic-nlp-library in project venv."
        exit 2
    fi
fi

# 1b. Ensure optional Tantivy FTS dependencies are available.
# We use this backend for more stable phrase-position indexing on very large tables.
echo "Ensuring Tantivy FTS dependencies (tantivy + pylance) are installed..."
if python3 -c "import tantivy, lance" >/dev/null 2>&1; then
    echo "  [SUCCESS] tantivy and pylance already available."
else
    if "${PIP_INSTALL[@]}" tantivy pylance; then
        echo "  [SUCCESS] Installed tantivy and pylance."
    else
        echo "ERROR: Failed to install tantivy/pylance in project venv."
        echo "       Run manually: source ${REPO_ROOT}/.venv/bin/activate && uv pip install tantivy pylance"
        exit 2
    fi
fi

# 0. Ensure NLTK data is available (avoid downloads inside HTCondor jobs)
echo "Ensuring NLTK tokenizer data (punkt_tab/punkt) is available..."
export NLTK_DATA="${NLTK_DATA:-$REPO_ROOT/.venv/nltk_data}"
mkdir -p "$NLTK_DATA"
if python3 "$REPO_ROOT/data/news/scripts/ensure_nltk_data.py"; then
    echo "  [SUCCESS] NLTK data ready."
else
    echo "ERROR: Could not ensure NLTK data. CCNews crawl will fail with missing punkt_tab."
    echo "       Try: source ${REPO_ROOT}/.venv/bin/activate && python3 ${REPO_ROOT}/data/news/scripts/ensure_nltk_data.py"
    exit 2
fi

# 2. Initialize submodule if needed
if [ ! -d "$NEWS_PLEASE_DIR/.git" ]; then
    echo "Initializing news-please submodule..."
    git submodule update --init --recursive "$NEWS_PLEASE_DIR"
fi

# 3. Apply patch
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
