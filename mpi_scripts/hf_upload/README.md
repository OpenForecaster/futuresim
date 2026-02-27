# Hugging Face Upload Job

Submit `hf upload-large-folder` as an HTCondor batch job.

## Quick Start

```bash
cd /home/sgoel/forecast-sim
source ~/forecast-sim/fsim/bin/activate

python mpi_scripts/hf_upload/submit_hf_upload.py \
  --repo_id shash42/forecast-news \
  --local_path /lustre/fast/fast/sgoel/forecasting/news/deduped_articles/data \
  --num_workers 8 \
  --stage_root /home/sgoel/hf_upload_staging \
  --progress_secs 60 \
  --memory 64 \
  --disk 200 \
  --bid 15
```

## What It Does

- Copies `local_path` into a per-job staging folder under `stage_root`.
- Uploads from that staged copy to avoid source-folder lock issues.
- Prints periodic progress lines like `committed/uploaded/hashed`.
- Verifies expected files exist on the Hub.
- Deletes the staged copy on success (use `--keep_stage` to keep it).

## Static `.sub` Alternative

```bash
condor_submit_bid 15 mpi_scripts/hf_upload/hf_upload.sub
```
