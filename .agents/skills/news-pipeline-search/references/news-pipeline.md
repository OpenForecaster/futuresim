# News Pipeline

## Stage Order

1. crawl CCNews
2. convert JSON to JSONL
3. deduplicate
4. convert to parquet
5. generate embeddings
6. build LanceDB table + scalar index
7. build or refresh FTS

## Shared Setup

```bash
source .venv/bin/activate
./data/news/scripts/setup_news_pipeline.sh
```

The setup script applies the `news-please` patch and installs the extra deps the repo expects for extraction and FTS.

## MPI-Style Pipeline

Launch crawling:

```bash
python data/news/scripts/launch_news_crawl.py \
  --start-date YYYY-MM-DD \
  --end-date YYYY-MM-DD
```

Run staged processing:

```bash
python data/news/run_news_pipeline.py --status
python data/news/run_news_pipeline.py --step jsonl
python data/news/run_news_pipeline.py --step dedup
python data/news/run_news_pipeline.py --step parquet
python data/news/run_news_pipeline.py --step embed
python data/news/run_news_pipeline.py --step lancedb
python data/news/run_news_pipeline.py --step lancedb_index
```

## Direct-Script Path

Use this when you do not want MPI-specific wrappers or hardcoded cluster assumptions.

Suggested layout:

```bash
export NEWS_ROOT=/path/to/news
export WARC_DIR=$NEWS_ROOT/warc
export RAW_ARTICLES_DIR=$NEWS_ROOT/raw_articles
export JSONL_DIR=$NEWS_ROOT/jsonl
export DEDUPED_DIR=$JSONL_DIR/deduped
export PARQUET_DIR=$NEWS_ROOT/deduped_articles
export EMB_DIR=$PARQUET_DIR/embeddings
export LANCE_DIR=$PARQUET_DIR/lance
export MODEL_PATH="${FSIM_EMBEDDING_MODEL}"
```

Representative commands:

```bash
cd data/news/news-please
NEWS_START_DATE=YYYY-MM-DD NEWS_END_DATE=YYYY-MM-DD \
python -m newsplease.examples.commoncrawl \
  "$WARC_DIR" "$RAW_ARTICLES_DIR" delete 1

cd ../..
python data/news/scripts/to_jsonl.py "$RAW_ARTICLES_DIR" \
  --output_dir "$JSONL_DIR" --workers 48 --verify 0.1

python data/news/scripts/deduplicate_news_jsonl.py \
  --jsonl_path "$JSONL_DIR" --num_workers 16

python scripts/convert_jsonl_to_parquet.py \
  --input-dirs "$DEDUPED_DIR" \
  --output-dir "$PARQUET_DIR" \
  --workers 32 \
  --batch-size 128

python scripts/embed_articles.py \
  --start_date YYYY-MM-DD \
  --end_date YYYY-MM-DD \
  --model Qwen3-Embedding-8B \
  --model_path "$MODEL_PATH" \
  --articles_dir "$PARQUET_DIR/data" \
  --output_dir "$EMB_DIR" \
  --chunk_tokens 512 \
  --worker_id 0 --num_workers 1 --resume

python scripts/build_lancedb.py \
  --start_date YYYY-MM-DD \
  --end_date YYYY-MM-DD \
  --model Qwen3-Embedding-8B \
  --articles_dir "$PARQUET_DIR/data" \
  --embeddings_dir "$EMB_DIR" \
  --output_dir "$LANCE_DIR" \
  --skip_fts_index \
  --overwrite

python scripts/build_lancedb_index.py \
  --db_path "$LANCE_DIR/Qwen3-Embedding-8B" \
  --build_fts \
  --fts_with_position \
  --fts_use_tantivy \
  --tantivy_index_root "${FSIM_TANTIVY_INDEX_ROOT}" \
  --force
```

## Rerun And Repair Notes

- `to_jsonl.py` now rebuilds per-domain JSONL outputs by default so reruns stay idempotent. Use `--resume_processed_dirs` only when you intentionally want the older skip-already-seen behavior.
- `deduplicate_news_jsonl.py` now rewrites each deduped JSONL atomically instead of appending into existing outputs.
- `convert_jsonl_to_parquet.py` now always compacts touched day folders back to a single deduplicated shard so reruns stay idempotent.

## Shortcut: Reuse Existing LanceDB

If you already have a prebuilt LanceDB table, skip the expensive embedding and Stage 1 table-build work. Run Stage 2 locally only to create the machine-local search-serving indices.

Public mirrors:

- Canonical parquet corpus: `https://huggingface.co/datasets/shash42/forecast-news`
- Prebuilt LanceDB table: `https://huggingface.co/datasets/shash42/forecast-news-embeddings`
- The `forecast-news-embeddings` dataset name is historical; it currently ships the prebuilt LanceDB table, not raw per-day `embeddings.npz`.

AISA example:

```bash
hf download shash42/forecast-news-embeddings \
  --repo-type dataset \
  --local-dir /mnt/nfs/datasets_ac/news/deduped_articles/lance/Qwen3-Embedding-8B \
  --max-workers 8
```

MPI Stage 2 wrapper:

```bash
BUILD_FTS=1 FTS_WITH_POSITION=1 FTS_USE_TANTIVY=1 BUILD_VECTOR_INDEX=1 \
condor_submit_bid 25 mpi_scripts/build_lancedb/build_index.sub
```

AISA Stage 2 wrapper:

```bash
sbatch aisa_scripts/build_lancedb/build_index_aisa.sh
```

Notes:

- On MPI `/fast` and related shared filesystems, the Stage 2 wrapper automatically places Tantivy data in an external lock-capable directory and links it back into the DB.
- A prebuilt LanceDB table copied from another machine is still useful, but collaborators should normally run Stage 2 locally so the Tantivy sidecar is created on paths valid for their environment.
- This is not a rebuild of the LanceDB table itself. It is a local index/bootstrap step on top of the shipped table.
- For hybrid search, Stage 2 must have built FTS successfully and the runtime must also have access to `FSIM_EMBEDDING_MODEL` for query encoding.
- IVF-PQ is a speed optimization, not a correctness requirement, but collaborators usually want it enabled for interactive hybrid search.
- Add `--accelerator cuda` only when the machine running Stage 2 actually has a usable GPU.
