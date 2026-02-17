#!/bin/bash
# Upload local folder to Hugging Face dataset using a staged copy in $HOME.

set -euo pipefail

export PATH="/usr/bin:/bin:/usr/sbin:/sbin:$PATH"
source ~/.bashrc 2>/dev/null || true

VENV_ACTIVATE="${HF_UPLOAD_VENV_ACTIVATE:-$HOME/forecast-sim/fsim/bin/activate}"
if [[ -f "$VENV_ACTIVATE" ]]; then
  # shellcheck disable=SC1090
  source "$VENV_ACTIVATE"
fi

REPO_ID="${HF_UPLOAD_REPO_ID:-${1:-shash42/forecast-news}}"
SOURCE_PATH="${HF_UPLOAD_LOCAL_PATH:-${2:-/lustre/fast/fast/sgoel/forecasting/news/deduped_articles/data}}"
NUM_WORKERS="${HF_UPLOAD_NUM_WORKERS:-8}"
INCLUDE_GLOB="${HF_UPLOAD_INCLUDE_GLOB:-**/*.parquet}"
EXCLUDE_GLOB="${HF_UPLOAD_EXCLUDE_GLOB:-}"
PROGRESS_SECS="${HF_UPLOAD_PROGRESS_SECS:-60}"
STAGE_ROOT="${HF_UPLOAD_STAGE_ROOT:-$HOME/hf_upload_staging}"
STAGE_PATH="${HF_UPLOAD_STAGE_PATH:-}"
VERIFY_REMOTE="${HF_UPLOAD_VERIFY_REMOTE:-1}"
DELETE_STAGE_ON_SUCCESS="${HF_UPLOAD_DELETE_STAGE_ON_SUCCESS:-1}"

if [[ ! -d "$SOURCE_PATH" ]]; then
  echo "ERROR: source path does not exist: $SOURCE_PATH"
  exit 2
fi

if ! command -v hf >/dev/null 2>&1; then
  echo "ERROR: hf CLI not found in PATH."
  exit 2
fi

if ! [[ "$NUM_WORKERS" =~ ^[0-9]+$ ]]; then
  echo "ERROR: HF_UPLOAD_NUM_WORKERS must be an integer (got '$NUM_WORKERS')."
  exit 2
fi

if ! [[ "$PROGRESS_SECS" =~ ^[0-9]+$ ]]; then
  echo "ERROR: HF_UPLOAD_PROGRESS_SECS must be an integer (got '$PROGRESS_SECS')."
  exit 2
fi

if [[ "$VERIFY_REMOTE" != "0" && "$VERIFY_REMOTE" != "1" ]]; then
  echo "ERROR: HF_UPLOAD_VERIFY_REMOTE must be 0 or 1 (got '$VERIFY_REMOTE')."
  exit 2
fi

if [[ "$DELETE_STAGE_ON_SUCCESS" != "0" && "$DELETE_STAGE_ON_SUCCESS" != "1" ]]; then
  echo "ERROR: HF_UPLOAD_DELETE_STAGE_ON_SUCCESS must be 0 or 1 (got '$DELETE_STAGE_ON_SUCCESS')."
  exit 2
fi

count_expected_files() {
  local upload_path="$1"
  local include_glob="$2"
  local exclude_glob="$3"

  python - "$upload_path" "$include_glob" "$exclude_glob" <<'PY'
import fnmatch
import sys
from pathlib import Path

root, include_glob, exclude_glob = sys.argv[1:4]
count = 0
for path in Path(root).rglob("*"):
    if not path.is_file():
        continue
    rel = path.relative_to(root).as_posix()
    if not fnmatch.fnmatch(rel, include_glob):
        continue
    if exclude_glob and fnmatch.fnmatch(rel, exclude_glob):
        continue
    count += 1
print(count)
PY
}

print_progress() {
  local upload_path="$1"
  local expected_total="$2"

  python - "$upload_path" "$expected_total" <<'PY'
import sys
from pathlib import Path

upload_path, expected_total = sys.argv[1], int(sys.argv[2])
meta_root = Path(upload_path) / ".cache" / "huggingface" / "upload"

metadata_files = list(meta_root.rglob("*.metadata")) if meta_root.exists() else []
hashed = 0
uploaded = 0
committed = 0

for mf in metadata_files:
    try:
        lines = mf.read_text().splitlines()
    except Exception:
        continue

    if len(lines) >= 8:
        if len(lines) > 3 and lines[3].strip():
            hashed += 1
        if len(lines) > 6 and lines[6].strip() == "1":
            uploaded += 1
        if len(lines) > 7 and lines[7].strip() == "1":
            committed += 1

print(
    "Progress: "
    f"committed={committed}/{expected_total} "
    f"uploaded={uploaded}/{expected_total} "
    f"hashed={hashed}/{expected_total} "
    f"metadata_files={len(metadata_files)}"
)
PY
}

verify_remote_upload() {
  local repo_id="$1"
  local upload_path="$2"
  local include_glob="$3"
  local exclude_glob="$4"

  python - "$repo_id" "$upload_path" "$include_glob" "$exclude_glob" <<'PY'
import fnmatch
import sys
from pathlib import Path

from huggingface_hub import HfApi

repo_id, upload_path, include_glob, exclude_glob = sys.argv[1:5]
root = Path(upload_path)

expected = []
for path in root.rglob("*"):
    if not path.is_file():
        continue
    rel = path.relative_to(root).as_posix()
    if not fnmatch.fnmatch(rel, include_glob):
        continue
    if exclude_glob and fnmatch.fnmatch(rel, exclude_glob):
        continue
    expected.append(rel)

api = HfApi()
remote = set(api.list_repo_files(repo_id=repo_id, repo_type="dataset"))
missing = [rel for rel in expected if rel not in remote]

print(f"Verification: expected={len(expected)} remote_total={len(remote)}")
if missing:
    print("ERROR: Missing files on Hub (first 20): " + ", ".join(missing[:20]), file=sys.stderr)
    sys.exit(1)
print("Verification successful: all expected files are present on Hub.")
PY
}

STAGE_ROOT="${STAGE_ROOT%/}"
if [[ -z "$STAGE_PATH" ]]; then
  stage_stamp="$(date +%Y%m%d_%H%M%S)"
  sanitized_repo="${REPO_ID//\//__}"
  STAGE_PATH="${STAGE_ROOT}/${sanitized_repo}_${stage_stamp}_$$"
fi

mkdir -p "$STAGE_PATH"
if find "$STAGE_PATH" -mindepth 1 -print -quit | grep -q .; then
  echo "ERROR: stage path already exists and is not empty: $STAGE_PATH"
  exit 2
fi

echo "Staging source folder into home before upload..."
echo "  Stage root:  $STAGE_ROOT"
echo "  Stage path:  $STAGE_PATH"
if command -v rsync >/dev/null 2>&1; then
  rsync -a --exclude '.cache/huggingface/***' "$SOURCE_PATH"/ "$STAGE_PATH"/
else
  cp -a "$SOURCE_PATH"/. "$STAGE_PATH"/
  rm -rf "$STAGE_PATH/.cache/huggingface"
fi
echo "Staging complete."

EXPECTED_TOTAL="$(count_expected_files "$STAGE_PATH" "$INCLUDE_GLOB" "$EXCLUDE_GLOB")"

echo "Starting HF upload job"
echo "  Repo:        $REPO_ID (dataset)"
echo "  Source path: $SOURCE_PATH"
echo "  Upload path: $STAGE_PATH"
echo "  Workers:     $NUM_WORKERS"
echo "  Include:     $INCLUDE_GLOB"
if [[ -n "$EXCLUDE_GLOB" ]]; then
  echo "  Exclude:     $EXCLUDE_GLOB"
fi
echo "  Expected:    $EXPECTED_TOTAL files"
echo "  Progress:    ${PROGRESS_SECS}s"

upload_cmd=(
  hf upload-large-folder "$REPO_ID" "$STAGE_PATH"
  --repo-type dataset
  --num-workers "$NUM_WORKERS"
  --include "$INCLUDE_GLOB"
)

if [[ -n "$EXCLUDE_GLOB" ]]; then
  upload_cmd+=(--exclude "$EXCLUDE_GLOB")
fi

"${upload_cmd[@]}" &
upload_pid=$!

if (( PROGRESS_SECS > 0 )); then
  while kill -0 "$upload_pid" 2>/dev/null; do
    sleep "$PROGRESS_SECS"
    if ! kill -0 "$upload_pid" 2>/dev/null; then
      break
    fi
    print_progress "$STAGE_PATH" "$EXPECTED_TOTAL"
  done
fi

wait "$upload_pid"

print_progress "$STAGE_PATH" "$EXPECTED_TOTAL"

if [[ "$VERIFY_REMOTE" == "1" ]]; then
  echo "Verifying uploaded files against Hub..."
  verify_remote_upload "$REPO_ID" "$STAGE_PATH" "$INCLUDE_GLOB" "$EXCLUDE_GLOB"
fi

if [[ "$DELETE_STAGE_ON_SUCCESS" == "1" ]]; then
  case "$STAGE_PATH" in
    "$STAGE_ROOT"/*)
      echo "Deleting staged copy: $STAGE_PATH"
      rm -rf "$STAGE_PATH"
      ;;
    *)
      echo "WARN: refusing to delete stage path outside stage root: $STAGE_PATH"
      ;;
  esac
fi

echo "HF upload complete."
