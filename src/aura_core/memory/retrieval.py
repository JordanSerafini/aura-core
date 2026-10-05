"""Ambient recall: pick 2-3 short memory extracts relevant to a prompt.

Designed for injecting context into an agent prompt, where a wrong or
tangential extract is worse than none. Hence the abstentions:

- fewer than 2 content-bearing words in the prompt: no subject, return nothing;
- an explicit technical identifier (like `ABC-12345`) that the corpus has never
  seen: return nothing rather than the nearest lookalike;
- a chunk found by the keyword side ONLY must cover at least 2 distinct query
  words (a single shared word is a coincidence, not a memory); chunks found by
  the vector side are exempt;
- each chunk is given at most once per session, and at most one extract per
  source file, so three extracts are three different pieces of information.

Both rankings (BM25 and vector k-NN) are fused by Reciprocal Rank Fusion.
`embedder` is a callable `str -> list[float]`; without it, keyword side only.
"""

import re
import sqlite3
import struct
import time
from collections.abc import Callable, Iterable

STOPWORDS = frozenset("""
avec dans pour sans mais donc alors cette votre notre leur sont etre avoir fait
faire faut tout tous toute toutes plus moins bien aussi comme quand ainsi entre
vers chez sous apres avant encore jamais toujours peut veux veut mettre prends
prend continue continuer please thanks thank merci puis donne moi le la les des
une aux ces son ses mes tes nos vos ton fais
this that with from have what when will would could should about there then
which where while your just into over also than them they their been were does
""".split())

DEFAULT_TYPES = ("topic", "note")
RRF_K = 60


def _tokens(prompt: str, limit: int = 8) -> list[str]:
  # Min length 3, not 4: 3-letter acronyms are the most discriminating terms
  # (highest IDF) in a technical corpus and a threshold of 4 would drop them all.
  seen, out = set(), []
  for w in re.findall(r"[a-zA-ZÀ-ſ0-9_-]{3,}", prompt.lower()):
    if w in STOPWORDS or w in seen:
      continue
    seen.add(w)
    out.append(w)
    if len(out) >= limit:
      break
  return out


def _in_clause(types: Iterable[str]) -> tuple[str, list[str]]:
  types = list(types)
  return "(" + ",".join("?" * len(types)) + ")", types


def semantic_rows(db, prompt: str, embedder: Callable[[str], list[float]] | None,
                  types: Iterable[str], timing: dict, limit: int = 6) -> list[tuple]:
  """Top chunks by vector similarity, `[]` when the vector side is unavailable."""
  timing.update(embed_ms=0.0, knn_ms=0.0)
  if embedder is None:
    return []
  t0 = time.perf_counter()
  try:
    import sqlite_vec

    vec = embedder(prompt)
    t1 = time.perf_counter()
    timing["embed_ms"] = (t1 - t0) * 1000
    marks, params = _in_clause(types)
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=0.8)
    try:
      conn.enable_load_extension(True)
      sqlite_vec.load(conn)
      conn.enable_load_extension(False)
      blob = struct.pack(f"{len(vec)}f", *vec)
      # The type filter must restrict the CANDIDATES before the k-NN; applied
      # after the join it would return nothing whenever the nearest neighbours
      # are of another type.
      rows = conn.execute(
        "SELECT c.rowid, c.type, c.source, substr(c.content,1,200) "
        "FROM vec_chunks v JOIN chunks c ON c.rowid = v.rowid "
        "WHERE v.embedding MATCH ? AND k = ? "
        f"AND v.rowid IN (SELECT rowid FROM chunks WHERE type IN {marks}) "
        "ORDER BY v.distance",
        (blob, limit, *params),
      ).fetchall()
      timing["knn_ms"] = (time.perf_counter() - t1) * 1000
      return rows
    finally:
      conn.close()
  except Exception:
    timing["embed_fallback"] = True
    return []


def prompt_selection(prompt: str, session_id: str, *, db, embedder: Callable[[str], list[float]] | None = None,
                     types: Iterable[str] = DEFAULT_TYPES, seen: dict | None = None,
                     timing: dict | None = None) -> list[tuple]:
  """Return up to 3 `(rowid, type, source, extract)` tuples, or `[]` to abstain."""
  timing = timing if timing is not None else {}
  seen = seen if seen is not None else {}
  types = tuple(types)
  tokens = _tokens(prompt)
  if len(tokens) < 2:
    return []
  marks, type_params = _in_clause(types)

  ids = re.findall(r"\b[a-zA-Z]+-\d{4,}\b", prompt)
  if ids:
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=0.5) as conn:
      for identifier in ids:
        found = conn.execute(
          f"SELECT 1 FROM chunks WHERE type IN {marks} AND instr(lower(content), lower(?)) > 0 LIMIT 1",
          (*type_params, identifier),
        ).fetchone()
        if not found:
          timing["abstention"] = "unknown_identifier"
          return []

  query = " OR ".join(f'"{t}"' for t in tokens)
  t_fts = time.perf_counter()
  try:
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=0.8)
    try:
      fts_rows = conn.execute(
        "SELECT c.rowid, c.type, c.source, snippet(chunks_fts, 0, '', '', '...', 22) "
        "FROM chunks_fts JOIN chunks c ON c.rowid = chunks_fts.rowid "
        f"WHERE chunks_fts MATCH ? AND c.type IN {marks} "
        "ORDER BY bm25(chunks_fts) LIMIT 6",
        (query, *type_params),
      ).fetchall()
      # How many DISTINCT query words does each keyword candidate really contain?
      covers: dict[int, int] = {}
      ids_ = [r[0] for r in fts_rows]
      if ids_:
        ph = ",".join("?" * len(ids_))
        for rid, content in conn.execute(f"SELECT rowid, lower(content) FROM chunks WHERE rowid IN ({ph})", ids_):
          words = set(re.findall(r"[a-z0-9à-ÿ_-]{3,}", content or ""))
          covers[rid] = sum(1 for t in tokens if t in words)
    finally:
      conn.close()
  except sqlite3.Error:
    return []
  finally:
    timing["fts_ms"] = (time.perf_counter() - t_fts) * 1000

  sem_rows = semantic_rows(db, prompt, embedder, types, timing, limit=6)
  sem_ids = {r[0] for r in sem_rows}
  scores: dict[int, float] = {}
  meta: dict[int, tuple] = {}
  for rank, row in enumerate(fts_rows):
    scores[row[0]] = scores.get(row[0], 0.0) + 1.0 / (RRF_K + rank)
    meta[row[0]] = row
  for rank, row in enumerate(sem_rows):
    scores[row[0]] = scores.get(row[0], 0.0) + 1.0 / (RRF_K + rank)
    meta.setdefault(row[0], row)
  ordered = [meta[rid] for rid, _ in sorted(scores.items(), key=lambda kv: -kv[1])]

  if len(seen) > 80:  # naive purge: the process may live for weeks
    seen.clear()
  injected = seen.setdefault(session_id or "?", set())
  picked, seen_sources = [], set()
  for rowid, ctype, source, extract in ordered:
    if rowid in injected or source in seen_sources:
      continue
    if rowid not in sem_ids and covers.get(rowid, 0) < 2:
      continue
    extract = " ".join((extract or "").split())[:170]
    if not extract:
      continue
    seen_sources.add(source)
    injected.add(rowid)
    picked.append((rowid, ctype, source, extract))
    if len(picked) >= 3:
      break
  return picked
