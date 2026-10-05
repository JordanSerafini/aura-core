"""Embedding backends.

Two implementations share the same tiny interface (`dim`, `embed(texts)`):

- `SentenceTransformerEmbedder`: a real model, loaded lazily. Needs the `st`
  extra and, on first use, network access to download the weights (set
  `AURA_CORE_OFFLINE=1` to forbid downloads and use only the local cache).
- `HashingEmbedder`: a deterministic bag-of-words hashing trick. It is NOT
  semantic. It exists so the vector code path can be tested and demonstrated
  with no download, and so tests are reproducible.

Environment:
  AURA_CORE_EMBED_MODEL   model name (default: sentence-transformers/all-MiniLM-L6-v2)
  AURA_CORE_EMBED_DEVICE  force a device, e.g. "cpu"
  AURA_CORE_OFFLINE       "1" to use the local model cache only
"""

import hashlib
import math
import os
import re
import threading
from typing import Protocol

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_DIM = 384


class Embedder(Protocol):
  dim: int

  def embed(self, texts: list[str]) -> list[list[float]]: ...


def _truthy(name: str) -> bool:
  return os.getenv(name, "").strip().lower() in {"1", "true", "yes"}


class SentenceTransformerEmbedder:
  """Lazy sentence-transformers wrapper with a CPU fallback on CUDA OOM."""

  def __init__(self, model_name: str | None = None):
    self.model_name = model_name or os.getenv("AURA_CORE_EMBED_MODEL", DEFAULT_MODEL)
    self._model = None
    self._lock = threading.Lock()
    self._dim: int | None = None

  def _load(self, device: str | None = None):
    try:
      from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - depends on the environment
      raise RuntimeError(
        "sentence-transformers is not installed; use `pip install aura-core[st]` "
        "or the lexical-only mode"
      ) from exc
    return SentenceTransformer(
      self.model_name,
      local_files_only=_truthy("AURA_CORE_OFFLINE"),
      device=device,
    )

  def _get_model(self):
    if self._model is None:
      with self._lock:
        if self._model is None:
          forced = os.getenv("AURA_CORE_EMBED_DEVICE", "").strip() or None
          try:
            self._model = self._load(forced)
          except Exception as exc:
            if forced or "out of memory" not in str(exc).lower():
              raise
            self._model = self._load("cpu")
    return self._model

  @property
  def dim(self) -> int:
    if self._dim is None:
      model = self._get_model()
      getter = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
      self._dim = int(getter())
    return self._dim

  def embed(self, texts: list[str]) -> list[list[float]]:
    if not texts:
      return []
    vectors = self._get_model().encode(
      texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
    )
    return [v.tolist() for v in vectors]


class HashingEmbedder:
  """Deterministic hashed bag-of-words vectors, L2-normalised. Not semantic."""

  def __init__(self, dim: int = 256):
    self.dim = dim

  def embed(self, texts: list[str]) -> list[list[float]]:
    out = []
    for text in texts:
      vec = [0.0] * self.dim
      for token in re.findall(r"\w+", text.lower()):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        n = int.from_bytes(digest, "big")
        vec[n % self.dim] += 1.0 if (n >> 63) & 1 else -1.0
      norm = math.sqrt(sum(v * v for v in vec)) or 1.0
      out.append([v / norm for v in vec])
    return out


def get_embedder(kind: str | None = None) -> Embedder:
  """`kind`: "st" (sentence-transformers) or "hashing". Default: "st"."""
  kind = (kind or os.getenv("AURA_CORE_EMBEDDER", "st")).lower()
  if kind in {"st", "sentence-transformers"}:
    return SentenceTransformerEmbedder()
  if kind == "hashing":
    return HashingEmbedder()
  raise ValueError(f"unknown embedder kind: {kind!r}")
