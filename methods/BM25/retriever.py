"""Minimal BM25 retriever over the external MedRAG corpus layout."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from methods.common.retrieval import (
    DEFAULT_DB_DIR,
    PROJECT_ROOT,
    JsonlLineStoreCache,
    env_int,
    parse_corpus_list,
    resolve_path,
    summarize_retrieved_docs,
)


CHUNK_OFFSET_CACHE_DIR = PROJECT_ROOT / "temp" / "retrieval_chunk_offsets"
DEFAULT_BM25_CORPORA = ("pubmed", "textbooks", "statpearls", "wikipedia")


def _prepare_query(text: str) -> str:
    query = " ".join(str(text or "").split())
    max_terms = env_int("BM25_MAX_QUERY_TERMS", 0)
    if max_terms > 0:
        query = " ".join(query.split()[:max_terms])
    max_chars = env_int("BM25_MAX_QUERY_CHARS", 0)
    if max_chars > 0 and len(query) > max_chars:
        query = query[:max_chars]
    return query


class BM25Retriever:
    """BM25 retrieval over one or more MedRAG corpora."""

    def __init__(
        self,
        *,
        db_dir: str | Path = DEFAULT_DB_DIR,
        corpora: Sequence[str] | None = None,
    ) -> None:
        try:
            from pyserini.pyclass import autoclass
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Missing BM25 runtime dependencies. Install pyserini and pyjnius, and load a Java module before running BM25."
            ) from exc

        self.db_dir = resolve_path(db_dir)
        self.corpora = parse_corpus_list(corpora, default=DEFAULT_BM25_CORPORA)
        self._searchers: Dict[str, Any] = {}
        self._chunk_cache = JsonlLineStoreCache(
            CHUNK_OFFSET_CACHE_DIR,
            env_name="BM25_CHUNK_STORE_CACHE_SIZE",
        )
        self._simple_searcher_cls = autoclass("io.anserini.search.SimpleSearcher")

        for corpus in self.corpora:
            index_dir = self.db_dir / corpus / "index" / "bm25"
            if not index_dir.exists():
                raise FileNotFoundError(f"Missing BM25 index for corpus={corpus}: {index_dir}")
            self._searchers[corpus] = self._simple_searcher_cls(str(index_dir))

    def _load_jsonl_line(self, corpus: str, source: str, index: int) -> Dict[str, Any]:
        path = self.db_dir / corpus / "chunk" / f"{source}.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        item = self._chunk_cache.get(path, index)
        item.setdefault("id", f"{source}_{index}")
        item.setdefault("title", "")
        item.setdefault("content", item.get("contents", ""))
        item["corpus"] = corpus
        return item

    def _doc_from_hit(self, corpus: str, hit: Any) -> Dict[str, Any]:
        docid = str(hit.docid)
        source, index_text = docid.rsplit("_", 1)
        item = self._load_jsonl_line(corpus, source, int(index_text))
        item["score"] = float(hit.score)
        return item

    def retrieve(self, query: str, *, k: int) -> Tuple[List[Dict[str, Any]], List[float]]:
        merged: Dict[str, Dict[str, Any]] = {}
        search_query = _prepare_query(query)
        per_corpus_k = env_int("BM25_PER_CORPUS_K", 0)
        if per_corpus_k <= 0:
            per_corpus_k = max(int(k) * 4, 32)

        for corpus, searcher in self._searchers.items():
            hits = searcher.search(search_query, k=per_corpus_k)
            for hit in hits:
                try:
                    item = self._doc_from_hit(corpus, hit)
                except (FileNotFoundError, IndexError, ValueError):
                    continue
                key = f"{corpus}:{item['id']}"
                prev = merged.get(key)
                if prev is None or float(item["score"]) > float(prev["score"]):
                    merged[key] = item

        ranked = sorted(merged.values(), key=lambda row: float(row.get("score", 0.0)), reverse=True)
        top_items = ranked[: int(k)]
        scores = [float(item.get("score", 0.0)) for item in top_items]
        return top_items, scores
