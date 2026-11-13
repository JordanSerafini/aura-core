"""Reciprocal Rank Fusion, shared by the store (`search_hybrid`) and the ambient recall (`retrieval`).

score(item) = sum, over the rankings it appears in, of 1 / (RRF_K + rank), with rank counted from 1 (Cormack
et al., 2009). Only ranks matter, so lists scored on different scales (cosine distance, bm25) never need
calibrating. One definition on purpose: the two copies it replaces disagreed by one on the rank.
"""

from collections.abc import Hashable, Iterable

RRF_K = 60


def rrf_term(rank: int, rrf_k: int = RRF_K) -> float:
  """What one list adds to the score of an item it ranks at `rank` (1 = best)."""
  return 1.0 / (rrf_k + rank)


def rrf_scores(*rankings: Iterable[Hashable], rrf_k: int = RRF_K) -> dict:
  """Fused score of every key of `rankings` (each a sequence of keys, best first), keys in first-seen order."""
  scores: dict = {}
  for ranking in rankings:
    for rank, key in enumerate(ranking, start=1):
      scores[key] = scores.get(key, 0.0) + rrf_term(rank, rrf_k)
  return scores
