"""Vector store (ChromaDB) + BM25 hybrid retrieval with local ONNX embeddings (fastembed)."""
import hashlib
import logging
import os
import re
import threading
from dataclasses import dataclass

import chromadb
from chromadb.config import Settings as ChromaSettings
from rank_bm25 import BM25Okapi

from .config import Settings
from .ingest import Page

log = logging.getLogger(__name__)
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def _tok(s: str) -> list[str]:
    return re.findall(r"\w+", s.lower())


@dataclass
class Hit:
    id: str
    text: str
    url: str
    title: str
    section: str
    page_type: str
    similarity: float | None = None


class KnowledgeStore:
    def __init__(self, settings: Settings):
        self.s = settings
        self._embedder = None
        self._lock = threading.RLock()
        path = settings.data_dir / "chroma"
        path.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(path), settings=ChromaSettings(anonymized_telemetry=False))
        self.col = client.get_or_create_collection("placidway", metadata={"hnsw:space": "cosine"})
        self._bm25, self._docs = None, []
        self._rebuild_bm25()

    @property
    def embedder(self):
        if self._embedder is None:
            from fastembed import TextEmbedding

            self._embedder = TextEmbedding(self.s.embed_model, cache_dir=os.getenv("FASTEMBED_CACHE_PATH"))
        return self._embedder

    def _embed(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self.embedder.embed(texts)]

    def count(self) -> int:
        return self.col.count()

    def upsert_page(self, page: Page) -> int:
        docs = [f"{page.title} - {c.section}\n{c.text}" if c.section else f"{page.title}\n{c.text}" for c in page.chunks]
        if not docs:
            return 0
        key = hashlib.sha1(page.url.encode()).hexdigest()[:10]
        embeddings = self._embed(docs)
        with self._lock:
            self.col.delete(where={"url": page.url})
            self.col.add(
                ids=[f"{key}-{i}" for i in range(len(docs))],
                documents=docs,
                embeddings=embeddings,
                metadatas=[
                    {"url": page.url, "title": page.title, "section": c.section, "page_type": page.page_type}
                    for c in page.chunks
                ],
            )
            self._rebuild_bm25()
        return len(docs)

    def delete_url(self, url: str):
        with self._lock:
            self.col.delete(where={"url": url})
            self._rebuild_bm25()

    def _rebuild_bm25(self):
        data = self.col.get(include=["documents", "metadatas"])
        self._docs = [
            Hit(i, d, m["url"], m["title"], m.get("section", ""), m.get("page_type", ""))
            for i, d, m in zip(data["ids"], data["documents"], data["metadatas"])
        ]
        self._bm25 = BM25Okapi([_tok(h.text) for h in self._docs]) if self._docs else None

    def retrieve(self, query: str, k: int | None = None) -> tuple[list[Hit], float]:
        """Hybrid (vector + BM25) search fused with Reciprocal Rank Fusion.

        Returns (hits, best_vector_similarity); the similarity feeds the grounding gate.
        """
        k = k or self.s.top_k
        n = self.col.count()
        if n == 0:
            return [], 0.0
        qe = self._embed([QUERY_PREFIX + query])[0]
        with self._lock:
            res = self.col.query(
                query_embeddings=[qe], n_results=min(k * 2, n), include=["documents", "metadatas", "distances"]
            )
            by_id: dict[str, Hit] = {}
            scores: dict[str, float] = {}
            best = 0.0
            for rank, (i, d, m, dist) in enumerate(
                zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0])
            ):
                sim = 1.0 - dist
                best = max(best, sim)
                by_id[i] = Hit(i, d, m["url"], m["title"], m.get("section", ""), m.get("page_type", ""), sim)
                scores[i] = scores.get(i, 0) + 1 / (60 + rank)
            if self._bm25:
                bm = self._bm25.get_scores(_tok(query))
                order = [o for o in sorted(range(len(bm)), key=lambda x: bm[x], reverse=True)[: k * 2] if bm[o] > 0]
                for rank, idx in enumerate(order):
                    h = self._docs[idx]
                    by_id.setdefault(h.id, h)
                    scores[h.id] = scores.get(h.id, 0) + 1 / (60 + rank)
        ranked = sorted(scores, key=scores.get, reverse=True)[:k]
        return [by_id[i] for i in ranked], best
