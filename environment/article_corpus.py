"""Date-gated article corpus visibility and workspace staging."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable, Literal, Optional


StageMode = Literal["symlink", "hardlink", "copy"]


@dataclass(frozen=True)
class VisibleArticleFile:
    """One source article file visible to an agent on a simulation date."""

    source_path: Path
    relative_path: Path
    article_date: date


@dataclass
class ArticleWorkspace:
    """A local agent-visible article workspace managed by the environment."""

    articles_dir: Path
    mode: StageMode = "symlink"
    search_cutoff_days: int = 0
    freeze_after_start: bool = False
    start_date: Optional[date] = None
    last_staged_date: Optional[date] = None


class ArticleCorpusView:
    """
    Environment-owned view of the dated article corpus.

    The corpus is expected to be laid out as ``YYYY/MM/DD/articles.jsonl``.
    This class decides which files are visible for a simulation date and can
    materialize those files into local workspaces without exposing the full
    source corpus.
    """

    def __init__(self, articles_base: str | Path, *, default_start_date: Optional[date] = None):
        self.articles_base: Optional[Path] = Path(articles_base).expanduser() if articles_base else None
        self.default_start_date = default_start_date

    @property
    def is_available(self) -> bool:
        return self.articles_base is not None and self.articles_base.exists()

    @staticmethod
    def _dated_relative_path(d: date) -> Path:
        return Path(f"{d.year}") / f"{d.month:02d}" / f"{d.day:02d}" / "articles.jsonl"

    def source_path_for_date(self, d: date) -> Path:
        if self.articles_base is None:
            raise RuntimeError("No article corpus is configured.")
        return self.articles_base / self._dated_relative_path(d)

    def effective_max_date(
        self,
        current_date: date,
        *,
        start_date: Optional[date] = None,
        search_cutoff_days: int = 0,
        freeze_after_start: bool = False,
    ) -> date:
        start = start_date or self.default_start_date or current_date
        effective_current = start if freeze_after_start else current_date
        return effective_current - timedelta(days=max(0, int(search_cutoff_days or 0)))

    def visible_dates(
        self,
        current_date: date,
        *,
        start_date: Optional[date] = None,
        search_cutoff_days: int = 0,
        freeze_after_start: bool = False,
        since_date: Optional[date] = None,
    ) -> Iterable[date]:
        start = start_date or self.default_start_date or current_date
        max_date = self.effective_max_date(
            current_date,
            start_date=start,
            search_cutoff_days=search_cutoff_days,
            freeze_after_start=freeze_after_start,
        )
        if since_date is not None:
            start = max(start, since_date + timedelta(days=1))
        if max_date < start:
            return []

        days = []
        d = start
        while d <= max_date:
            days.append(d)
            d += timedelta(days=1)
        return days

    def visible_files(
        self,
        current_date: date,
        *,
        start_date: Optional[date] = None,
        search_cutoff_days: int = 0,
        freeze_after_start: bool = False,
        since_date: Optional[date] = None,
    ) -> list[VisibleArticleFile]:
        files: list[VisibleArticleFile] = []
        for d in self.visible_dates(
            current_date,
            start_date=start_date,
            search_cutoff_days=search_cutoff_days,
            freeze_after_start=freeze_after_start,
            since_date=since_date,
        ):
            rel = self._dated_relative_path(d)
            if self.articles_base is None:
                continue
            src = self.articles_base / rel
            if src.exists():
                files.append(VisibleArticleFile(src, rel, d))
        return files

    def stage_local(
        self,
        articles_dir: str | Path,
        current_date: date,
        *,
        mode: StageMode = "symlink",
        start_date: Optional[date] = None,
        search_cutoff_days: int = 0,
        freeze_after_start: bool = False,
        since_date: Optional[date] = None,
    ) -> Optional[date]:
        """
        Materialize visible files into ``articles_dir``.

        Returns the effective max staged date, or ``None`` when no corpus is
        configured. The directory shape under ``articles_dir`` mirrors the
        source corpus.
        """
        if not self.is_available:
            return None

        if mode not in ("symlink", "hardlink", "copy"):
            raise ValueError(f"Unknown article stage mode: {mode!r}")

        dst_root = Path(articles_dir)
        dst_root.mkdir(parents=True, exist_ok=True)
        max_date = self.effective_max_date(
            current_date,
            start_date=start_date,
            search_cutoff_days=search_cutoff_days,
            freeze_after_start=freeze_after_start,
        )

        for item in self.visible_files(
            current_date,
            start_date=start_date,
            search_cutoff_days=search_cutoff_days,
            freeze_after_start=freeze_after_start,
            since_date=since_date,
        ):
            dst = dst_root / item.relative_path
            dst.parent.mkdir(parents=True, exist_ok=True)
            if freeze_after_start:
                self._copy_filtered_jsonl(item.source_path, dst, cutoff=max_date)
            elif not dst.exists():
                self._materialize_file(item.source_path, dst, mode=mode)

        return max_date

    @staticmethod
    def _materialize_file(src: Path, dst: Path, *, mode: StageMode) -> None:
        if mode == "symlink":
            dst.symlink_to(src)
            return
        if mode == "hardlink":
            try:
                os.link(src, dst)
            except OSError as exc:
                if exc.errno == 18:  # EXDEV
                    raise RuntimeError(
                        "hardlink article staging requires source and destination on "
                        f"the same filesystem ({src} -> {dst}). Use mode='copy' "
                        "or move one of the paths."
                    ) from exc
                raise
            return
        shutil.copy2(src, dst)

    @staticmethod
    def _copy_filtered_jsonl(src: Path, dst: Path, *, cutoff: date) -> None:
        cutoff_iso = cutoff.isoformat()
        tmp = dst.with_suffix(".jsonl.tmp")
        with open(src) as inp, open(tmp, "w") as out:
            for line in inp:
                try:
                    article = json.loads(line)
                except json.JSONDecodeError:
                    continue
                article_date = str(article.get("date") or "")[:10]
                publish_date = str(article.get("date_publish") or "")[:10]
                if article_date and article_date > cutoff_iso:
                    continue
                if publish_date and publish_date > cutoff_iso:
                    continue
                out.write(line)
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        tmp.replace(dst)
