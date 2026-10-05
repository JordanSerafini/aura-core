"""Optional second stage: cross-encoder reranking of the fused candidates.

The hybrid fusion orders by RANKS and is blind to the fine meaning of a
passage. A cross-encoder reads (question, passage) together and scores their
relevance, which can fix false positives left at ranks 5-20.

Rule: wire this into a pipeline only after the benchmark (`aura_core.eval`)
shows a real gain on your own questions. Without a gain, leave it unused.

The import is lazy (sentence-transformers and torch are slow to import), the
model is cached per process, and the function is fail-safe: if the model cannot
load, candidates come back unchanged (truncated to `top_k`) instead of breaking
retrieval.

  AURA_CORE_RERANK_MODEL   model name (default: BAAI/bge-reranker-v2-m3)
"""

import os
from functools import lru_cache

MAX_LENGTH = 512
_MAX_CHARS = 2000  # longer passages are truncated by the model anyway


def model_name() -> str:
  return os.getenv("AURA_CORE_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")


@lru_cache(maxsize=1)
def _model():
  from sentence_transformers import CrossEncoder

  return CrossEncoder(model_name(), max_length=MAX_LENGTH)


def available() -> bool:
  """True when the dependency is installed (the weights may still need a download)."""
  import importlib.util

  return importlib.util.find_spec("sentence_transformers") is not None


def rerank(query: str, candidates: list[dict], top_k: int = 5, text_key: str = "content") -> list[dict]:
  """Reorder `candidates` by decreasing relevance; adds a `rerank_score` key to each."""
  if not candidates:
    return []
  try:
    pairs = [(query, (c.get(text_key) or "")[:_MAX_CHARS]) for c in candidates]
    scores = _model().predict(pairs)
  except Exception:
    return candidates[:top_k]
  for c, s in zip(candidates, scores, strict=False):
    c["rerank_score"] = float(s)
  return sorted(candidates, key=lambda c: c.get("rerank_score", 0.0), reverse=True)[:top_k]
