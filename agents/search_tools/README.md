# Search Tools

Agents depend on the `BaseSearchTool` interface in `base.py`, not on LanceDB
itself. The public runner constructs the bundled LanceDB implementation when
`search_db`/`FSIM_SEARCH_DB` is set; leaving it empty runs without retrieval.

Download the public LanceDB artifact with:

```bash
hf download shash42/forecast-news-embeddings \
  --repo-type dataset \
  --local-dir "$FSIM_SEARCH_DB"
```

Custom corpora can use the bundled LanceDB implementation by providing an
`articles` table with `chunk_id`, `article_id`, `chunk_index`, `title`,
`source`, `date`, `content`, and `vector`; `date_publish` and `url` are
optional. Other retrieval backends can be added by implementing `BaseSearchTool`
and returning `SearchResult` objects.
