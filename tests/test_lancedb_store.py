from datetime import date

from agents.search_tools.lancedb.store import LanceDBSearchTool


class _DummySearchBuilder:
    def __init__(self):
        self._rows = []
        self.last_prefilter = None

    def vector(self, _v):
        return self

    def text(self, _q):
        return self

    def where(self, _w, prefilter=True):
        self.last_prefilter = prefilter
        return self

    def limit(self, _n):
        return self

    def to_list(self):
        return self._rows


class _DummyTable:
    def __init__(self):
        self.calls = []

    def search(self, *args, **kwargs):
        # Match both keyword and semantic/hybrid call signatures.
        builder = _DummySearchBuilder()
        self.calls.append({"args": args, "kwargs": kwargs, "builder": builder})
        return builder


class _DummyEmbedOutputs:
    def __init__(self, embedding):
        self.embedding = embedding


class _DummyEmbedItem:
    def __init__(self, embedding):
        self.outputs = _DummyEmbedOutputs(embedding=embedding)


class _DummyEmbedModelEmptyList:
    def embed(self, _texts, use_tqdm=False):
        return []


class _DummyEmbedModelEmptyVector:
    def embed(self, _texts, use_tqdm=False):
        return [_DummyEmbedItem([])]


class _DummySearchBuilderHybridProbe:
    def __init__(self, rows):
        self._rows = rows
        self.last_prefilter = None

    def vector(self, _v):
        return self

    def text(self, _q):
        return self

    def where(self, _w, prefilter=True):
        self.last_prefilter = prefilter
        return self

    def limit(self, _n):
        return self

    def to_list(self):
        return self._rows


class _DummyTableHybridEmptyFtsHit:
    def __init__(self):
        self.calls = []

    def search(self, *args, **kwargs):
        query_type = kwargs.get("query_type")
        builder = None
        if query_type == "hybrid":
            builder = _DummySearchBuilderHybridProbe([])
        elif query_type == "fts":
            builder = _DummySearchBuilderHybridProbe([{"title": "fts hit"}])
        else:
            # semantic path not used in this test
            builder = _DummySearchBuilderHybridProbe([])
        self.calls.append({"args": args, "kwargs": kwargs, "builder": builder})
        return builder


class _DummyEmbedModelGood:
    def embed(self, _texts, use_tqdm=False):
        return [_DummyEmbedItem([0.1, 0.2, 0.3])]


def _mk_tool(embed_model):
    tool = LanceDBSearchTool.__new__(LanceDBSearchTool)
    tool._available = True
    tool._table = _DummyTable()
    tool._embedding_model = embed_model
    tool._model_loaded = True
    tool._model_path = None
    tool._db_path = ""
    tool._db = None
    tool._config = {}
    tool._chunk_tokens = 512
    return tool


def _mk_tool_with_table(embed_model, table):
    tool = LanceDBSearchTool.__new__(LanceDBSearchTool)
    tool._available = True
    tool._table = table
    tool._embedding_model = embed_model
    tool._model_loaded = True
    tool._model_path = None
    tool._db_path = ""
    tool._db = None
    tool._config = {}
    tool._chunk_tokens = 512
    return tool


def test_search_returns_empty_when_embed_returns_no_outputs():
    tool = _mk_tool(_DummyEmbedModelEmptyList())
    out = tool.search("test query", max_results=5, max_date=date(2025, 4, 24), search_type="hybrid")
    assert out == []


def test_search_returns_empty_when_embed_returns_empty_vector():
    tool = _mk_tool(_DummyEmbedModelEmptyVector())
    out = tool.search("test query", max_results=5, max_date=date(2025, 4, 24), search_type="hybrid")
    assert out == []


def test_search_returns_empty_when_hybrid_is_empty_but_fts_has_matches():
    table = _DummyTableHybridEmptyFtsHit()
    tool = _mk_tool_with_table(_DummyEmbedModelGood(), table)
    out = tool.search("test query", max_results=5, max_date=date(2025, 4, 24), search_type="hybrid")
    assert out == []


def test_hybrid_where_uses_prefilter_false_workaround():
    tool = _mk_tool(_DummyEmbedModelGood())
    out = tool.search(
        "test query", max_results=5, min_date=date(2025, 4, 1), max_date=date(2025, 4, 24), search_type="hybrid"
    )
    assert out == []
    assert len(tool._table.calls) == 1
    assert tool._table.calls[0]["kwargs"].get("query_type") == "hybrid"
    assert tool._table.calls[0]["builder"].last_prefilter is False


def test_keyword_where_uses_prefilter_false_default():
    tool = _mk_tool(_DummyEmbedModelGood())
    out = tool.search(
        "test query", max_results=5, min_date=date(2025, 4, 1), max_date=date(2025, 4, 24), search_type="keyword"
    )
    assert out == []
    assert len(tool._table.calls) == 1
    assert tool._table.calls[0]["kwargs"].get("query_type") == "fts"
    assert tool._table.calls[0]["builder"].last_prefilter is False
