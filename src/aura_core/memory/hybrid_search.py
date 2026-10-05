"""In-memory hybrid search: Okapi BM25 + dense vectors, no database.

A small, dependency-free engine for corpora that fit in RAM (tests, notebooks,
prototypes). For persistent storage and RRF fusion over SQLite, see
`SqliteVecStore.search_hybrid`.

Scoring here is a weighted sum of max-normalised BM25 and cosine similarity,
not RRF. The two engines are kept side by side on purpose so they can be
compared on the same questions.
"""

import math
import re
from collections import Counter
from dataclasses import dataclass, field

from aura_core.memory.embed_engine import Embedder


@dataclass
class HybridSearchResult:
  id: str
  content: str
  bm25_score: float
  vector_score: float
  combined_score: float
  metadata: dict = field(default_factory=dict)


class BM25:
  """Okapi BM25 over a list of (doc_id, text)."""

  def __init__(self, k1: float = 1.5, b: float = 0.75):
    self.k1 = k1
    self.b = b
    self.doc_ids: list[str] = []
    self.doc_contents: list[str] = []
    self._tf: list[Counter] = []
    self.doc_lengths: list[int] = []
    self.avgdl = 0.0
    self.df: dict[str, int] = {}
    self.idf: dict[str, float] = {}

  @staticmethod
  def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zàâäéèêëïîôùûüç0-9]+", text.lower())

  def fit(self, documents: list[tuple[str, str]]) -> None:
    self.doc_ids, self.doc_contents, self._tf, self.doc_lengths = [], [], [], []
    self.df = {}
    for doc_id, content in documents:
      tokens = self.tokenize(content)
      self.doc_ids.append(doc_id)
      self.doc_contents.append(content)
      self._tf.append(Counter(tokens))
      self.doc_lengths.append(len(tokens))
      for term in set(tokens):
        self.df[term] = self.df.get(term, 0) + 1
    n = len(self.doc_ids)
    self.avgdl = sum(self.doc_lengths) / n if n else 0.0
    # Lucene-style IDF with +1 smoothing: never negative.
    self.idf = {t: math.log((n - df + 0.5) / (df + 0.5) + 1) for t, df in self.df.items()}

  def score(self, query: str, doc_idx: int) -> float:
    tf = self._tf[doc_idx]
    doc_len = self.doc_lengths[doc_idx]
    total = 0.0
    for term in self.tokenize(query):
      freq = tf.get(term)
      if not freq:
        continue
      norm = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
      total += self.idf.get(term, 0.0) * (freq * (self.k1 + 1)) / norm
    return total

  def search(self, query: str, top_k: int = 10) -> list[tuple[str, str, float]]:
    scored = [(self.doc_ids[i], self.doc_contents[i], self.score(query, i)) for i in range(len(self.doc_ids))]
    scored = [s for s in scored if s[2] > 0]
    scored.sort(key=lambda s: s[2], reverse=True)
    return scored[:top_k]


def _cosine(a: list[float], b: list[float]) -> float:
  dot = sum(x * y for x, y in zip(a, b, strict=True))
  na = math.sqrt(sum(x * x for x in a))
  nb = math.sqrt(sum(y * y for y in b))
  return dot / (na * nb) if na and nb else 0.0


class HybridSearchEngine:
  """Weighted BM25 + cosine. Without an embedder it is BM25 only."""

  def __init__(self, embedder: Embedder | None = None, bm25_weight: float = 0.4, vector_weight: float = 0.6):
    self.embedder = embedder
    self.bm25_weight = bm25_weight
    self.vector_weight = vector_weight
    self.bm25 = BM25()
    self.documents: dict[str, str] = {}
    self.metadata: dict[str, dict] = {}
    self.vectors: dict[str, list[float]] = {}

  def index_batch(self, documents: list[tuple[str, str, dict | None]]) -> int:
    """Index (doc_id, content, metadata) tuples and rebuild the BM25 statistics."""
    for doc_id, content, metadata in documents:
      self.documents[doc_id] = content
      self.metadata[doc_id] = metadata or {}
    if self.embedder is not None and documents:
      for (doc_id, _, _), vec in zip(documents, self.embedder.embed([d[1] for d in documents]), strict=True):
        self.vectors[doc_id] = vec
    self.bm25.fit(list(self.documents.items()))
    return len(documents)

  def search(self, query: str, top_k: int = 10, min_score: float = 0.0) -> list[HybridSearchResult]:
    bm25_scores = {doc_id: s for doc_id, _, s in self.bm25.search(query, top_k=top_k * 2)}
    vector_scores: dict[str, float] = {}
    if self.embedder is not None and self.vectors:
      qvec = self.embedder.embed([query])[0]
      sims = sorted(((i, _cosine(qvec, v)) for i, v in self.vectors.items()), key=lambda x: x[1], reverse=True)
      vector_scores = {i: s for i, s in sims[: top_k * 2] if s > 0}
    bm25_max = max(bm25_scores.values(), default=0.0)
    vector_max = max(vector_scores.values(), default=0.0)
    results = []
    for doc_id in bm25_scores.keys() | vector_scores.keys():
      b = bm25_scores.get(doc_id, 0.0) / bm25_max if bm25_max > 0 else 0.0
      v = vector_scores.get(doc_id, 0.0) / vector_max if vector_max > 0 else 0.0
      combined = self.bm25_weight * b + self.vector_weight * v
      if combined >= min_score:
        results.append(HybridSearchResult(doc_id, self.documents[doc_id], b, v, combined, self.metadata[doc_id]))
    results.sort(key=lambda r: r.combined_score, reverse=True)
    return results[:top_k]
