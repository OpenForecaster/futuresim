from .base import DataFetcher, CachedQuestion
from datasets import load_dataset, Dataset
from datetime import datetime, date
import os
import glob
import hashlib
from typing import Optional

import pandas as pd

class OpenForesightFetcher(DataFetcher):
    """Fetcher for the standard OpenForesight dataset (HuggingFace/disk)."""
    
    @property
    def source_name(self) -> str:
        return "openforesight"
    
    def fetch_new(self, start_date: date, end_date: date):
        raise NotImplementedError("OpenForesight data is static/pre-downloaded.")
        
    def get_prompt_context(self) -> str:
        return "" # No special context, standard rules apply

    def _load_split_dataframe(self, path: str, split: str, cache_dir: Optional[str]) -> pd.DataFrame:
        """Load one OpenForesight split as a DataFrame."""
        if os.path.isdir(path):
            parquet_files = sorted(glob.glob(os.path.join(path, f"{split}-*.parquet")))
            if not parquet_files:
                raise ValueError(f"No parquet files found for split '{split}' in {path}")
            dfs = [pd.read_parquet(pq_file) for pq_file in parquet_files]
            return pd.concat(dfs, ignore_index=True)

        kwargs = {}
        if cache_dir:
            kwargs["cache_dir"] = cache_dir
        ds = load_dataset(path, split=split, **kwargs)
        if isinstance(ds, Dataset):
            return ds.to_pandas()
        return pd.DataFrame(ds)

    @staticmethod
    def _hash_sample_key(value: str) -> str:
        return hashlib.sha1(str(value).encode("utf-8")).hexdigest()

    def _apply_resolution_filters(
        self,
        df: pd.DataFrame,
        *,
        resolution_start: Optional[date],
        resolution_end: Optional[date],
    ) -> pd.DataFrame:
        if df.empty:
            return df
        if resolution_start:
            df = df[df["resolution_date"] >= resolution_start]
        if resolution_end:
            df = df[df["resolution_date"] <= resolution_end]
        return df

    def _apply_monthly_subsample(
        self,
        df: pd.DataFrame,
        *,
        subsample_per_month: Optional[int],
    ) -> pd.DataFrame:
        if not subsample_per_month or subsample_per_month <= 0 or df.empty:
            return df

        work = df.copy()
        work["_resolution_month"] = pd.to_datetime(work["resolution_date"]).dt.to_period("M").astype(str)
        work["_sample_key"] = work["qid"].astype(str).map(self._hash_sample_key)
        sampled = (
            work.sort_values(["_resolution_month", "_sample_key", "qid"])
            .groupby("_resolution_month", group_keys=False)
            .head(int(subsample_per_month))
            .drop(columns=["_resolution_month", "_sample_key"])
        )
        return sampled

    @staticmethod
    def _schedule_date(value) -> Optional[date]:
        if value in (None, ""):
            return None
        if isinstance(value, date):
            return value
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, str):
            return datetime.strptime(value, "%Y-%m-%d").date()
        raise ValueError(f"Unsupported schedule date: {value!r}")

    def _prepare_split_dataframe(
        self,
        path: str,
        split: str,
        cache_dir: Optional[str],
        *,
        resolution_start: Optional[date],
        resolution_end: Optional[date],
        subsample_per_month: Optional[int] = None,
    ) -> pd.DataFrame:
        df = self._load_split_dataframe(path, split, cache_dir).copy()
        df["resolution_date"] = pd.to_datetime(
            df["resolution_date"], errors="coerce"
        ).dt.date
        df = df.dropna(subset=["resolution_date"])
        df = self._apply_resolution_filters(
            df,
            resolution_start=resolution_start,
            resolution_end=resolution_end,
        )
        df = self._apply_monthly_subsample(
            df,
            subsample_per_month=subsample_per_month,
        )
        df["source_split"] = split
        return df
        
    def load_from_cache(
        self,
        cache_dir: str,
        resolution_start: date = None,
        resolution_end: date = None,
        min_forecasters: int = 0,
        resolved_only: bool = False,
    ) -> list[CachedQuestion]:
        """Load from the dataset_path (HF dataset) instead of parquet cache."""

        del min_forecasters
        del resolved_only

        # Determine path or dataset name
        path = self.dataset_path or "openforesight"
        prepend_train_start = self._schedule_date(self.prepend_train_resolution_start)
        prepend_train_end = self._schedule_date(self.prepend_train_resolution_end)
        main_df = self._prepare_split_dataframe(
            path,
            self.split,
            cache_dir,
            resolution_start=resolution_start,
            resolution_end=resolution_end,
            subsample_per_month=None,
        )

        if prepend_train_start or prepend_train_end:
            print(
                f"Loading OpenForesight data from: {path} "
                f"(split={self.split} with prepended train window)"
            )
            train_df = self._prepare_split_dataframe(
                path=path,
                split="train",
                cache_dir=cache_dir,
                resolution_start=max(
                    [d for d in (resolution_start, prepend_train_start) if d is not None],
                    default=None,
                ),
                resolution_end=min(
                    [d for d in (resolution_end, prepend_train_end) if d is not None],
                    default=None,
                ),
                subsample_per_month=self.subsample_per_month,
            )
            combined_df = pd.concat([train_df, main_df], ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=["qid"], keep="first")
            combined_df = combined_df.sort_values(["resolution_date", "qid"]).reset_index(drop=True)
        else:
            print(f"Loading OpenForesight data from: {path} (split={self.split})")
            combined_df = main_df

        questions = []
        for item in combined_df.to_dict("records"):
            r_date = self._parse_date(item.get("resolution_date"))
            if not r_date:
                continue

            questions.append(CachedQuestion(
                qid=str(item.get("qid")),
                title=item.get("question_title", ""),
                background=item.get("background", ""),
                resolution_criteria=item.get("resolution_criteria", ""),
                answer_type=item.get("answer_type", ""),
                resolution_date=r_date,
                ground_truth_answer=item.get("answer", ""),
                options=None, # OpenForesight (current) doesn't use options column
                source="openforesight",
                source_split=item.get("source_split", self.split) or self.split,
                prompt=item.get("prompt", "") or "",
            ))
            
        return questions

    def _parse_date(self, date_val) -> date:
        if date_val is None:
            return None
        if isinstance(date_val, str):
            try:
                dt = datetime.strptime(date_val, "%Y-%m-%d %H:%M:%S")
                return dt.date()
            except ValueError:
                try:
                    return datetime.strptime(date_val, "%Y-%m-%d").date()
                except ValueError:
                    return None
        elif isinstance(date_val, datetime):
            return date_val.date()
        elif isinstance(date_val, date):
            return date_val
        return None
