import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def test_convert_parquet_to_jsonl_script(tmp_path):
    input_dir = tmp_path / "data"
    day_dir = input_dir / "2025" / "04" / "23"
    day_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = day_dir / "articles_b0000.parquet"

    table = pa.table(
        {
            "id": ["a1"],
            "title": ["Example title"],
            "source": ["Example source"],
            "date": [date(2025, 4, 23)],
            "date_publish": [date(2025, 4, 22)],
            "date_modify": [None],
            "url": ["https://example.com"],
            "content": ["Example body text"],
            "authors": [["Alice", "Bob"]],
            "description": ["Example description"],
        }
    )
    pq.write_table(table, parquet_path)

    output_dir = tmp_path / "jsonl"
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "convert_parquet_to_jsonl.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--input_dir",
            str(input_dir),
            "--output_dir",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    out_path = output_dir / "2025" / "04" / "23" / "articles_b0000.jsonl"
    assert out_path.exists()
    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["id"] == "a1"
    assert row["date"] == "2025-04-23"
    assert row["date_publish"] == "2025-04-22"
    assert row["authors"] == ["Alice", "Bob"]
    assert "Converted 1 parquet file(s)" in completed.stdout
