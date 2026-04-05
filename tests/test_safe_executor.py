from datetime import date

import pandas as pd

from environment.safe_executor import QueryExecutor


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"qid": "Q1", "score": 0.2, "is_resolved": False},
            {"qid": "Q2", "score": 0.8, "is_resolved": True},
            {"qid": "Q3", "score": 0.5, "is_resolved": False},
        ]
    )


def test_query_executor_allows_in_memory_dataframe_manipulation() -> None:
    executor = QueryExecutor(timeout_seconds=1.0)

    result, error = executor.execute(
        _sample_df(),
        (
            "active = df[df['is_resolved'] == False].copy()\n"
            "active['double_score'] = active['score'] * 2\n"
            "print(active[['qid', 'double_score']].sort_values('double_score'))"
        ),
        current_date=date(2026, 4, 4),
    )

    assert error is None
    assert "Q1" in result
    assert "Q3" in result
    assert "double_score" in result


def test_query_executor_still_allows_head_for_previewing() -> None:
    executor = QueryExecutor(timeout_seconds=1.0)

    result, error = executor.execute(
        _sample_df(),
        "print(df[['qid', 'score']].head(2))",
        current_date=date(2026, 4, 4),
    )

    assert error is None
    assert "Q1" in result
    assert "Q2" in result


def test_query_executor_blocks_pandas_io_escape_hatch() -> None:
    executor = QueryExecutor(timeout_seconds=1.0)

    result, error = executor.execute(
        _sample_df(),
        "print(pd.io.common.os.listdir('.'))",
        current_date=date(2026, 4, 4),
    )

    assert result == ""
    assert error == (
        "UnsafeQueryError: Blocked module access in query_df: pd.io.common.os.listdir"
    )


def test_query_executor_blocks_pandas_readers() -> None:
    executor = QueryExecutor(timeout_seconds=1.0)

    result, error = executor.execute(
        _sample_df(),
        "pd.read_csv('/etc/hosts')",
        current_date=date(2026, 4, 4),
    )

    assert result == ""
    assert error == "UnsafeQueryError: Blocked pandas reader in query_df: pd.read_csv"


def test_query_executor_blocks_output_writes() -> None:
    executor = QueryExecutor(timeout_seconds=1.0)

    result, error = executor.execute(
        _sample_df(),
        "df.to_csv('/tmp/out.csv')",
        current_date=date(2026, 4, 4),
    )

    assert result == ""
    assert error == "UnsafeQueryError: Blocked output method in query_df: to_csv"


def test_query_executor_allows_string_serializers_without_file_output() -> None:
    executor = QueryExecutor(timeout_seconds=1.0)

    result, error = executor.execute(
        _sample_df(),
        "print(df[['qid', 'score']].to_json(orient='records'))",
        current_date=date(2026, 4, 4),
    )

    assert error is None
    assert '"qid":"Q1"' in result
    assert '"score":0.2' in result


def test_query_executor_blocks_private_attribute_access() -> None:
    executor = QueryExecutor(timeout_seconds=1.0)

    result, error = executor.execute(
        _sample_df(),
        "df.__class__",
        current_date=date(2026, 4, 4),
    )

    assert result == ""
    assert error == "UnsafeQueryError: Private attribute access is not allowed in query_df: __class__"


def test_query_executor_allows_safe_df_query() -> None:
    executor = QueryExecutor(timeout_seconds=1.0)

    result, error = executor.execute(
        _sample_df(),
        "print(df.query(\"qid == 'Q2' and is_resolved == True\"))",
        current_date=date(2026, 4, 4),
    )

    assert error is None
    assert "Q2" in result
    assert "0.8" in result


def test_query_executor_blocks_df_query_with_external_reference() -> None:
    executor = QueryExecutor(timeout_seconds=1.0)

    result, error = executor.execute(
        _sample_df(),
        "df.query('@pd.io.common.os.listdir(\".\")')",
        current_date=date(2026, 4, 4),
    )

    assert result == ""
    assert error == (
        "UnsafeQueryError: Blocked df.query call in query_df: only literal query strings "
        "without @ references or custom eval context are allowed"
    )


def test_query_executor_allows_type_name_introspection() -> None:
    executor = QueryExecutor(timeout_seconds=1.0)

    result, error = executor.execute(
        _sample_df(),
        "print(type(df['qid'].iloc[0]).__name__)",
        current_date=date(2026, 4, 4),
    )

    assert error is None
    assert "str" in result


def test_query_executor_allows_safe_datetime_import() -> None:
    executor = QueryExecutor(timeout_seconds=1.0)

    result, error = executor.execute(
        _sample_df(),
        "from datetime import timedelta\nprint(today + timedelta(days=2))",
        current_date=date(2026, 4, 4),
    )

    assert error is None
    assert "2026-04-06" in result


def test_query_executor_allows_import_pandas_as_pd() -> None:
    executor = QueryExecutor(timeout_seconds=1.0)

    result, error = executor.execute(
        _sample_df(),
        "import pandas as pd\nprint(pd.Timestamp('2026-04-04').date())",
        current_date=date(2026, 4, 4),
    )

    assert error is None
    assert "2026-04-04" in result


def test_query_executor_allows_simple_helper_function() -> None:
    executor = QueryExecutor(timeout_seconds=1.0)

    result, error = executor.execute(
        _sample_df(),
        (
            "def label(row):\n"
            "    return f\"{row['qid']}:{row['score']}\"\n"
            "print(df.head(1).apply(label, axis=1).iloc[0])"
        ),
        current_date=date(2026, 4, 4),
    )

    assert error is None
    assert "Q1:0.2" in result


def test_query_executor_blocks_unsafe_imports() -> None:
    executor = QueryExecutor(timeout_seconds=1.0)

    result, error = executor.execute(
        _sample_df(),
        "import os\nprint(os.listdir('.'))",
        current_date=date(2026, 4, 4),
    )

    assert result == ""
    assert error == "UnsafeQueryError: Import not allowed in query_df: os"
