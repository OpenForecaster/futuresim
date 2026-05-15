from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from .base import CachedQuestion, DataFetcher


class CustomFetcher(DataFetcher):
    """Load forecast questions from a local CSV, JSONL, JSON, or Parquet file."""

    @property
    def source_name(self) -> str:
        return "custom"

    def fetch_new(self, start_date: date, end_date: date):
        raise NotImplementedError("Custom datasets are loaded from dataset_path.")

    @staticmethod
    def _is_missing(value) -> bool:
        if value is None:
            return True
        if isinstance(value, str):
            return value.strip() == ""
        try:
            missing = pd.isna(value)
        except (TypeError, ValueError):
            return False
        return bool(missing) if not hasattr(missing, "__len__") else False

    @classmethod
    def _first_present(cls, row: dict, names: tuple[str, ...], default=None):
        for name in names:
            value = row.get(name)
            if not cls._is_missing(value):
                return value
        return default

    @staticmethod
    def _parse_date(value) -> Optional[date]:
        if CustomFetcher._is_missing(value):
            return None
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        parsed = pd.to_datetime(value, errors="coerce")
        if pd.isna(parsed):
            return None
        return parsed.date()

    @staticmethod
    def _parse_options(value):
        if CustomFetcher._is_missing(value):
            return None
        if isinstance(value, list):
            return value
        if hasattr(value, "tolist"):
            return value.tolist()
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                return None
            return parsed if isinstance(parsed, list) else None
        return None

    def _read_path(self, path: Path) -> pd.DataFrame:
        if path.is_dir():
            root = path / "data" if (path / "data").is_dir() else path
            candidates = []
            for pattern in (
                f"{self.split}-*.parquet",
                f"{self.split}.parquet",
                f"{self.split}.jsonl",
                f"{self.split}.json",
                f"{self.split}.csv",
            ):
                candidates.extend(sorted(root.glob(pattern)))
            if not candidates:
                raise ValueError(
                    f"No custom dataset files for split '{self.split}' in {path}. "
                    "Pass a file path or use names like test.jsonl / test-00000-of-00001.parquet."
                )
            frames = [self._read_path(candidate) for candidate in candidates]
            return pd.concat(frames, ignore_index=True)

        suffix = path.suffix.lower()
        if suffix == ".parquet":
            return pd.read_parquet(path)
        if suffix == ".csv":
            return pd.read_csv(path)
        if suffix == ".jsonl":
            return pd.read_json(path, lines=True)
        if suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                data = data.get("questions", data.get("data", data.get("rows", data)))
            if isinstance(data, dict):
                data = list(data.values())
            return pd.DataFrame(data)
        raise ValueError(f"Unsupported custom dataset format: {path}")

    def load_from_cache(
        self,
        cache_dir: str,
        resolution_start: date = None,
        resolution_end: date = None,
        min_forecasters: int = 0,
        resolved_only: bool = False,
    ) -> list[CachedQuestion]:
        del cache_dir
        del min_forecasters
        del resolved_only

        if not self.dataset_path:
            raise ValueError("--dataset_path is required when --dataset custom")

        df = self._read_path(Path(self.dataset_path)).copy()
        questions: list[CachedQuestion] = []
        missing = []

        for idx, row_obj in df.iterrows():
            row = row_obj.to_dict()
            qid = self._first_present(row, ("qid", "question_id", "id"))
            title = self._first_present(row, ("title", "question_title", "question"))
            resolution_date = self._parse_date(
                self._first_present(row, ("resolution_date", "close_time", "resolve_time"))
            )
            truth = self._first_present(
                row, ("ground_truth_answer", "ground_truth", "answer", "resolution", "resolved_to")
            )

            if qid in (None, "") or title in (None, "") or resolution_date is None or truth in (None, ""):
                missing.append(str(idx))
                continue
            if resolution_start and resolution_date < resolution_start:
                continue
            if resolution_end and resolution_date > resolution_end:
                continue

            questions.append(
                CachedQuestion(
                    qid=str(qid),
                    title=str(title),
                    background=str(self._first_present(row, ("background",), "")),
                    resolution_criteria=str(
                        self._first_present(row, ("resolution_criteria", "criteria"), "")
                    ),
                    answer_type=str(self._first_present(row, ("answer_type", "type"), "freeform")),
                    resolution_date=resolution_date,
                    ground_truth_answer=str(truth),
                    options=self._parse_options(row.get("options")),
                    source=str(self._first_present(row, ("source",), "custom")),
                    source_split=str(self._first_present(row, ("source_split", "split"), self.split)),
                    prompt=str(self._first_present(row, ("prompt",), "")),
                )
            )

        if missing:
            preview = ", ".join(missing[:5])
            raise ValueError(
                "Custom dataset rows missing required fields "
                "(qid/title/resolution_date/ground_truth_answer): "
                f"{preview}{'...' if len(missing) > 5 else ''}"
            )

        return questions
