# Hugging Face Upload Job

Submit `hf upload-large-folder` as an HTCondor batch job.

The upload wrapper stages a full copy of `local_path` into Condor scratch by default before uploading. This avoids `/fast`/NFS filelock issues during hashing and upload, but it also means `request_disk` must be large enough for the entire staged tree, not just the files matched by `--include`.

## Quick Start

```bash
cd /home/sgoel/forecast-sim
source .venv/bin/activate

python mpi_scripts/hf_upload/submit_hf_upload.py \
  --repo_id shash42/forecast-news \
  --local_path /fast/sgoel/forecasting/news/deduped_articles/data \
  --num_workers 8 \
  --progress_secs 60 \
  --memory 64 \
  --disk 200 \
  --bid 15
```

## What It Does

- Copies `local_path` into a per-job staging folder under Condor scratch by default.
- Uses `HF_UPLOAD_STAGE_ROOT` or `--stage_root` only when you intentionally want a persistent staging location.
- Uploads from that staged copy to avoid source-folder lock issues.
- Prints periodic progress lines like `committed/uploaded/hashed`.
- Verifies expected files exist on the Hub.
- Deletes the staged copy on success (use `--keep_stage` to keep it).

## Common Patterns

Update only `2023-2026` parquet while preserving older years already on the Hub:

```bash
python mpi_scripts/hf_upload/submit_hf_upload.py \
  --repo_id shash42/forecast-news \
  --local_path /fast/sgoel/forecasting/news/deduped_articles/data \
  --include '202[3-6]/**/*.parquet' \
  --num_workers 8 \
  --memory 64 \
  --disk 160 \
  --bid 15
```

Important: `--include` filters which files are uploaded, but the staging copy still includes the entire `local_path`. To preserve remote directory structure, keep `local_path` at the tree root and filter with `--include` rather than pointing `local_path` directly at a year or month subdirectory.

If a very large upload gets stuck retrying the Xet path, force the plain uploader on submit:

```bash
HF_HUB_DISABLE_XET=1 python mpi_scripts/hf_upload/submit_hf_upload.py \
  --repo_id <repo> \
  --local_path <path> \
  ...
```

## Static `.sub` Alternative

```bash
condor_submit_bid 15 mpi_scripts/hf_upload/hf_upload.sub
```
