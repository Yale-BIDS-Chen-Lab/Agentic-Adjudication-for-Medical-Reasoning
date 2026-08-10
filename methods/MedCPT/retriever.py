"""Minimal MedCPT dense retriever over the external MedRAG corpus layout."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from methods.common.retrieval import (
    DEFAULT_DB_DIR,
    PROJECT_ROOT,
    JsonlLineStoreCache,
    parse_corpus_list,
    resolve_path,
)


INDEX_ORG = "ncbi"
INDEX_LEAF = "MedCPT-Article-Encoder"
QUERY_ENCODER = "ncbi/MedCPT-Query-Encoder"
DEFAULT_MEDCPT_CORPORA = ("pubmed", "textbooks", "statpearls", "wikipedia")
CHUNK_OFFSET_CACHE_DIR = PROJECT_ROOT / "temp" / "retrieval_chunk_offsets"
METADATA_OFFSET_CACHE_DIR = PROJECT_ROOT / "temp" / "medcpt_metadata_offsets"


class _HFQueryEncoder:
    def __init__(self, model_name: str, *, device: str = "cpu") -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.device = device
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        )
        self.model.to(device)
        self.model.eval()

    def encode(self, texts: Sequence[str]) -> Any:
        max_length = getattr(self.tokenizer, "model_max_length", 512) or 512
        if max_length > 512 or max_length <= 0:
            max_length = 512
        inputs = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self._torch.no_grad():
            outputs = self.model(**inputs)
            embeds = outputs.last_hidden_state[:, 0]
        return embeds.detach().cpu().numpy().astype("float32")


class MedCPTRetriever:
    """MedCPT dense retrieval over one or more MedRAG corpora."""

    def __init__(
        self,
        *,
        db_dir: str | Path = DEFAULT_DB_DIR,
        corpora: Sequence[str] | None = None,
        device: str = "cpu",
    ) -> None:
        try:
            import faiss
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Missing MedCPT runtime dependency 'faiss'. Install faiss-cpu before running MedCPT."
            ) from exc

        self.db_dir = resolve_path(db_dir)
        self.corpora = parse_corpus_list(corpora, default=DEFAULT_MEDCPT_CORPORA)
        self._faiss = faiss
        self._indices: Dict[str, Any] = {}
        self._metadata_paths: Dict[str, Path] = {}
        self._metadata_cache = JsonlLineStoreCache(
            METADATA_OFFSET_CACHE_DIR,
            env_name="MEDCPT_METADATA_STORE_CACHE_SIZE",
        )
        self._chunk_cache = JsonlLineStoreCache(
            CHUNK_OFFSET_CACHE_DIR,
            env_name="MEDCPT_CHUNK_STORE_CACHE_SIZE",
        )

        for corpus in self.corpora:
            index_dir = self.db_dir / corpus / "index" / INDEX_ORG / INDEX_LEAF
            faiss_path = index_dir / "faiss.index"
            meta_path = index_dir / "metadatas.jsonl"
            if not faiss_path.exists() or not meta_path.exists():
                raise FileNotFoundError(f"Missing MedCPT index for corpus={corpus}: {index_dir}")
            self._indices[corpus] = self._faiss.read_index(str(faiss_path))
            self._metadata_paths[corpus] = meta_path

        self.encoder = _HFQueryEncoder(QUERY_ENCODER, device=device)

    def _load_chunk(self, corpus: str, source: str, index: int) -> Dict[str, Any]:
        path = self.db_dir / corpus / "chunk" / f"{source}.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        item = self._chunk_cache.get(path, index)
        item.setdefault("id", f"{source}_{index}")
        item.setdefault("title", "")
        item.setdefault("content", item.get("contents", ""))
        item["corpus"] = corpus
        return item

    def retrieve(self, query: str, *, k: int) -> Tuple[List[Dict[str, Any]], List[float]]:
        merged: Dict[str, Dict[str, Any]] = {}
        vector = self.encoder.encode([str(query)])

        for corpus, index in self._indices.items():
            scores, indices = index.search(vector, int(k))
            meta_path = self._metadata_paths[corpus]
            for score, idx in zip(scores[0].tolist(), indices[0].tolist()):
                if idx < 0:
                    continue
                try:
                    meta = self._metadata_cache.get(meta_path, int(idx))
                    item = self._load_chunk(corpus, str(meta["source"]), int(meta["index"]))
                except (FileNotFoundError, IndexError, KeyError, ValueError):
                    continue
                item["score"] = float(score)
                key = f"{corpus}:{item['id']}"
                prev = merged.get(key)
                if prev is None or float(item["score"]) > float(prev["score"]):
                    merged[key] = item

        ranked = sorted(merged.values(), key=lambda row: float(row.get("score", 0.0)), reverse=True)
        top_items = ranked[: int(k)]
        scores = [float(item.get("score", 0.0)) for item in top_items]
        return top_items, scores
