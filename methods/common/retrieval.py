"""Shared retrieval helpers for BM25 and MedCPT."""

from __future__ import annotations

import hashlib
import json
import mmap
import os
import struct
import tempfile
import threading
from array import array
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_DIR = Path(os.getenv("MEDRAG_CORPORA_DIR", PROJECT_ROOT / "corpora"))
DEFAULT_CORPORA = ("pubmed", "guidelines", "textbooks", "statpearls", "wikipedia")


def resolve_path(path: str | Path) -> Path:
    """Resolve project-relative or absolute paths consistently."""
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj.resolve()
    return (PROJECT_ROOT / path_obj).resolve()


def env_int(name: str, default: int) -> int:
    """Read one integer env var without failing on malformed values."""
    try:
        return int(str(os.getenv(name, str(default))).strip() or default)
    except ValueError:
        return default


def parse_corpus_list(
    value: str | Sequence[str] | None,
    *,
    default: Sequence[str] = DEFAULT_CORPORA,
) -> List[str]:
    """Parse a comma-separated corpus list while preserving order."""
    if value is None:
        return [str(item) for item in default]
    if isinstance(value, str):
        parts = value.split(",")
    else:
        parts = list(value)
    corpora = [str(item).strip() for item in parts if str(item).strip()]
    return corpora or [str(item) for item in default]


class _JsonlLineStore:
    """Random-access JSONL reader backed by a compact byte-offset cache."""

    def __init__(self, path: Path, *, cache_dir: Path) -> None:
        self.path = path.resolve()
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.offset_path = self._offset_path()
        self._ensure_offsets()
        self._offset_file = self.offset_path.open("rb")
        self._offset_mmap = mmap.mmap(self._offset_file.fileno(), 0, access=mmap.ACCESS_READ)
        self._data_file = self.path.open("rb")
        self._lock = threading.Lock()
        self.num_rows = len(self._offset_mmap) // 8

    def _offset_path(self) -> Path:
        stat = self.path.stat()
        key = hashlib.sha1(str(self.path).encode("utf-8")).hexdigest()[:16]
        return self.cache_dir / f"{key}.{stat.st_size}.{int(stat.st_mtime)}.u64"

    def _ensure_offsets(self) -> None:
        if self.offset_path.exists() and self.offset_path.stat().st_size > 0:
            return
        tmp_fd, tmp_name = tempfile.mkstemp(
            prefix=f"{self.offset_path.name}.",
            suffix=".tmp",
            dir=str(self.cache_dir),
        )
        try:
            with open(tmp_fd, "wb", closefd=True) as out, self.path.open("rb") as src:
                offsets = array("Q")
                offset = 0
                for line in src:
                    offsets.append(offset)
                    offset += len(line)
                    if len(offsets) >= 1_000_000:
                        offsets.tofile(out)
                        offsets = array("Q")
                if offsets:
                    offsets.tofile(out)
            Path(tmp_name).replace(self.offset_path)
        except Exception:
            Path(tmp_name).unlink(missing_ok=True)
            raise

    def get(self, idx: int) -> Dict[str, Any]:
        if idx < 0 or idx >= self.num_rows:
            raise IndexError(f"JSONL row out of bounds: idx={idx}, rows={self.num_rows}")
        offset = struct.unpack_from("<Q", self._offset_mmap, idx * 8)[0]
        with self._lock:
            self._data_file.seek(offset)
            line = self._data_file.readline()
        if not line:
            raise IndexError(f"Missing JSONL row idx={idx}: {self.path}")
        return json.loads(line)

    def close(self) -> None:
        for handle in (
            getattr(self, "_offset_mmap", None),
            getattr(self, "_offset_file", None),
            getattr(self, "_data_file", None),
        ):
            if handle is None:
                continue
            try:
                handle.close()
            except Exception:
                pass


class JsonlLineStoreCache:
    """LRU cache of random-access JSONL readers."""

    def __init__(self, cache_dir: Path, *, max_open: int | None = None, env_name: str = "RETRIEVAL_STORE_CACHE_SIZE") -> None:
        self.cache_dir = cache_dir
        self.max_open = max(1, int(max_open or env_int(env_name, 64)))
        self._stores: "OrderedDict[Path, _JsonlLineStore]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, path: Path, idx: int) -> Dict[str, Any]:
        path = path.resolve()
        with self._lock:
            store = self._stores.pop(path, None)
            if store is None:
                store = _JsonlLineStore(path, cache_dir=self.cache_dir)
            self._stores[path] = store
            while len(self._stores) > self.max_open:
                _, old_store = self._stores.popitem(last=False)
                old_store.close()
            return store.get(idx)


def summarize_retrieved_docs(snippets: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep only the retrieval fields we want in canonical traces."""
    rows: List[Dict[str, Any]] = []
    for item in snippets:
        row = {
            "id": item.get("id"),
            "corpus": item.get("corpus"),
            "title": item.get("title", ""),
            "score": float(item.get("score", 0.0)),
            "content": item.get("content", ""),
        }
        if item.get("PMID") is not None:
            row["PMID"] = item.get("PMID")
        rows.append(row)
    return rows
