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
  --skip_vector_index \
  --force
```

## Shortcut: Reuse Existing LanceDB

If you already have a prebuilt LanceDB directory, skip the expensive embedding and table-build stages and rebuild Stage 2 locally for parity with current repo defaults.
