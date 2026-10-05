"""Single-file vector + full-text store on SQLite.

Schema:
  chunks(id, source, type, content, mtime, content_hash, metadata JSON)
  vec_chunks(rowid, embedding FLOAT[dim])      sqlite-vec virtual table
  chunks_fts(content)                          FTS5, external-content on chunks
  sources_meta(source, type, last_indexed, content_hash, chunk_count)
  store_meta(key, value)                       holds the embedding dimension

Embeddings are optional per chunk: a chunk without a vector is still found by
the BM25 (FTS5) side, which is what the lexical-only mode relies on.

`search_hybrid` fuses the vector ranking and the BM25 ranking with Reciprocal
Rank Fusion, then applies small recency and importance priors.
"""

import hashlib
import json
import math
import re
import sqlite3
import struct
import time
from datetime import datetime
from pathlib import Path

import sqlite_vec

from aura_core.config import default_memory_db
from aura_core.memory.embed_engine import DEFAULT_DIM, Embedder
from aura_core.memory.secret_redact import mask_env_values, redact


def _serialize_vec(vec: list[float]) -> bytes:
  return struct.pack(f"{len(vec)}f", *vec)


def _hash_content(content: str) -> str:
  return hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()


def _types(filter_type) -> list[str]:
  if not filter_type:
    return []
  return [filter_type] if isinstance(filter_type, str) else list(filter_type)


class SqliteVecStore:
  """Vector + BM25 store in one SQLite file."""

  def __init__(self, db_path: Path | str | None = None, *, dim: int | None = None,
               read_only: bool = False, redact_secrets: bool = True):
    self.db_path = Path(db_path) if db_path else default_memory_db()
    self.read_only = read_only
    self.redact_secrets = redact_secrets
    self._conn: sqlite3.Connection | None = None
    self._requested_dim = dim
    self.dim = dim or DEFAULT_DIM
    if not read_only:
      self.db_path.parent.mkdir(parents=True, exist_ok=True)
      self._init_schema()
    else:
      self.dim = self._stored_dim() or self.dim

  # -- connection and schema ------------------------------------------------

  def _connect(self) -> sqlite3.Connection:
    if self._conn is None:
      if self.read_only:
        conn = sqlite3.connect(self.db_path.resolve().as_uri() + "?mode=ro", uri=True, timeout=10)
        conn.execute("PRAGMA query_only = ON")
      else:
        conn = sqlite3.connect(str(self.db_path), timeout=10)
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
      conn.execute("PRAGMA busy_timeout = 10000")
      conn.enable_load_extension(True)
      sqlite_vec.load(conn)
      conn.enable_load_extension(False)
      conn.row_factory = sqlite3.Row
      self._conn = conn
    return self._conn

  def _stored_dim(self) -> int | None:
    try:
      row = self._connect().execute("SELECT value FROM store_meta WHERE key='dim'").fetchone()
    except sqlite3.OperationalError:
      return None
    return int(row["value"]) if row else None

  def _init_schema(self) -> None:
    conn = self._connect()
    conn.executescript("""
      CREATE TABLE IF NOT EXISTS store_meta (key TEXT PRIMARY KEY, value TEXT);
      CREATE TABLE IF NOT EXISTS chunks (
        id TEXT PRIMARY KEY,
        source TEXT NOT NULL,
        type TEXT NOT NULL,
        content TEXT NOT NULL,
        mtime INTEGER,
        content_hash TEXT,
        metadata TEXT,
        created_at REAL DEFAULT (strftime('%s','now'))
      );
      CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source);
      CREATE INDEX IF NOT EXISTS idx_chunks_type ON chunks(type);
      CREATE TABLE IF NOT EXISTS sources_meta (
        source TEXT PRIMARY KEY,
        type TEXT NOT NULL,
        last_indexed REAL,
        content_hash TEXT,
        chunk_count INTEGER DEFAULT 0
      );
    """)
    stored = self._stored_dim()
    if stored is not None:
      if self._requested_dim is not None and self._requested_dim != stored:
        raise ValueError(
          f"{self.db_path} was created with dim={stored}, requested dim={self._requested_dim}; "
          "use a new database file after changing the embedding model"
        )
      self.dim = stored
    else:
      conn.execute("INSERT INTO store_meta(key, value) VALUES('dim', ?)", (str(self.dim),))
    conn.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(embedding FLOAT[{self.dim}])")
    conn.commit()
    self._init_fts()

  def _init_fts(self) -> None:
    """FTS5 BM25 as an external-content index on `chunks` (idempotent).

    The index is kept in sync by triggers; `chunks` stays the source of truth.
    Do not trust `count(*) FROM chunks_fts` to know whether the index is
    populated: on an external-content table it reads the content table and
    returns its total even when the index is empty. A marker row is used instead.
    """
    conn = self._connect()
    conn.executescript("""
      CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
        content, content='chunks', content_rowid='rowid',
        tokenize='unicode61 remove_diacritics 2'
      );
      CREATE TRIGGER IF NOT EXISTS chunks_fts_ai AFTER INSERT ON chunks BEGIN
        INSERT INTO chunks_fts(rowid, content) VALUES (new.rowid, new.content);
      END;
      CREATE TRIGGER IF NOT EXISTS chunks_fts_ad AFTER DELETE ON chunks BEGIN
        INSERT INTO chunks_fts(chunks_fts, rowid, content) VALUES('delete', old.rowid, old.content);
      END;
      CREATE TRIGGER IF NOT EXISTS chunks_fts_au AFTER UPDATE ON chunks BEGIN
        INSERT INTO chunks_fts(chunks_fts, rowid, content) VALUES('delete', old.rowid, old.content);
        INSERT INTO chunks_fts(rowid, content) VALUES (new.rowid, new.content);
      END;
      CREATE TABLE IF NOT EXISTS fts_state (done INTEGER DEFAULT 0);
    """)
    conn.commit()
    done = conn.execute("SELECT count(*) AS n FROM fts_state WHERE done = 1").fetchone()["n"]
    chunks_n = conn.execute("SELECT count(*) AS n FROM chunks").fetchone()["n"]
    if not done and chunks_n > 0:
      conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
      conn.execute("INSERT INTO fts_state (done) VALUES (1)")
      conn.commit()

  # -- writes ---------------------------------------------------------------

  def get_source_hash(self, source: str) -> str | None:
    row = self._connect().execute(
      "SELECT content_hash FROM sources_meta WHERE source = ?", (source,)
    ).fetchone()
    return row["content_hash"] if row else None

  def is_indexed(self, source: str, content_hash: str) -> bool:
    return self.get_source_hash(source) == content_hash

  def delete_by_source(self, source: str) -> int:
    """Delete every chunk (and vector) of a source. Returns the number of chunks removed."""
    conn = self._connect()
    rowids = [r["rowid"] for r in conn.execute("SELECT rowid FROM chunks WHERE source = ?", (source,))]
    if rowids:
      conn.execute(f"DELETE FROM vec_chunks WHERE rowid IN ({','.join('?' * len(rowids))})", rowids)
      conn.execute("DELETE FROM chunks WHERE source = ?", (source,))
    conn.execute("DELETE FROM sources_meta WHERE source = ?", (source,))
    conn.commit()
    return len(rowids)

  def _clean(self, text: str) -> str:
    if not self.redact_secrets:
      return text
    text, _ = mask_env_values(text)
    return redact(text)

  def upsert(self, chunks: list[dict]) -> int:
    """Insert or replace chunks.

    Each chunk dict: id, source, type, content (required); embedding
    (optional list[float] of length `dim`), mtime, metadata, content_hash.
    Content and metadata are passed through the secret masker before writing,
    which also feeds the FTS index. The caller's embedding is kept as given.
    """
    if not chunks:
      return 0
    conn = self._connect()
    n = 0
    owns_transaction = not conn.in_transaction
    try:
      if owns_transaction:
        conn.execute("BEGIN IMMEDIATE")
      for c in chunks:
        missing = [k for k in ("id", "source", "type", "content") if k not in c]
        if missing:
          raise ValueError(f"incomplete chunk, missing keys: {', '.join(missing)}")
        emb = c.get("embedding")
        if emb is not None and (len(emb) != self.dim or not all(math.isfinite(v) for v in emb)):
          raise ValueError(f"invalid embedding for {c['id']}: {len(emb)} dimensions, {self.dim} expected")
        content = self._clean(c["content"])
        content_hash = c.get("content_hash") or _hash_content(content)
        meta_json = self._clean(json.dumps(c.get("metadata", {}), default=str, ensure_ascii=False))
        # RETURNING is required: cursor.lastrowid can still hold the rowid of a
        # previous INSERT when ON CONFLICT takes the UPDATE branch.
        row = conn.execute(
          """INSERT INTO chunks (id, source, type, content, mtime, content_hash, metadata)
             VALUES (?, ?, ?, ?, ?, ?, ?)
             ON CONFLICT(id) DO UPDATE SET
               source = excluded.source, type = excluded.type, content = excluded.content,
               mtime = excluded.mtime, content_hash = excluded.content_hash,
               metadata = excluded.metadata
             RETURNING rowid""",
          (c["id"], c["source"], c["type"], content, c.get("mtime"), content_hash, meta_json),
        ).fetchone()
        conn.execute("DELETE FROM vec_chunks WHERE rowid = ?", (row["rowid"],))
        if emb is not None:
          conn.execute(
            "INSERT INTO vec_chunks(rowid, embedding) VALUES (?, ?)",
            (row["rowid"], _serialize_vec(emb)),
          )
        n += 1
      if owns_transaction:
        conn.commit()
    except Exception:
      if owns_transaction:
        conn.rollback()
      raise
    return n

  def add_documents(self, docs: list[dict], embedder: Embedder | None = None,
                    *, type_: str = "topic", source_prefix: str = "doc") -> int:
    """Convenience: index `{id, content[, metadata]}` dicts, embedding them if an embedder is given."""
    chunks = []
    for d in docs:
      chunks.append({
        "id": d["id"],
        "source": d.get("source", f"{source_prefix}/{d['id']}"),
        "type": d.get("type", type_),
        "content": d["content"],
        "metadata": d.get("metadata", {}),
        "mtime": d.get("mtime"),
      })
    if embedder is not None:
      vectors = embedder.embed([c["content"] for c in chunks])
      for c, v in zip(chunks, vectors, strict=True):
        c["embedding"] = v
    return self.upsert(chunks)

  def replace_source(self, source: str, type_: str, content_hash: str, chunks: list[dict]) -> int:
    """Atomically replace the vectors, text and metadata of one source, keeping fact provenance."""
    if not chunks or any(c.get("source") != source for c in chunks):
      raise ValueError("replacement must contain chunks of exactly one source")
    if len({c["id"] for c in chunks}) != len(chunks):
      raise ValueError("duplicate replacement chunk IDs")
    from aura_core.memory.fact_provenance import capture_source, ensure_schema, reconcile_source

    conn = self._connect()
    ensure_schema(conn)
    try:
      conn.execute("BEGIN IMMEDIATE")
      capture_source(conn, source)
      conn.execute("DELETE FROM vec_chunks WHERE rowid IN (SELECT rowid FROM chunks WHERE source=?)", (source,))
      conn.execute("DELETE FROM chunks WHERE source=?", (source,))
      n = self.upsert(chunks)
      conn.execute(
        """INSERT INTO sources_meta(source,type,last_indexed,content_hash,chunk_count)
           VALUES (?,?,?,?,?) ON CONFLICT(source) DO UPDATE SET
           type=excluded.type,last_indexed=excluded.last_indexed,
           content_hash=excluded.content_hash,chunk_count=excluded.chunk_count""",
        (source, type_, time.time(), content_hash, n),
      )
      reconcile_source(conn, source)
      conn.commit()
      return n
    except BaseException:
      conn.rollback()
      raise

  def mark_source_indexed(self, source: str, type_: str, content_hash: str, chunk_count: int) -> None:
    conn = self._connect()
    conn.execute(
      """INSERT INTO sources_meta (source, type, last_indexed, content_hash, chunk_count)
         VALUES (?, ?, ?, ?, ?)
         ON CONFLICT(source) DO UPDATE SET type = excluded.type,
           last_indexed = excluded.last_indexed, content_hash = excluded.content_hash,
           chunk_count = excluded.chunk_count""",
      (source, type_, time.time(), content_hash, chunk_count),
    )
    conn.commit()

  # -- reads ----------------------------------------------------------------

  @staticmethod
  def _meta(raw: str | None) -> dict:
    try:
      return json.loads(raw) if raw else {}
    except ValueError:
      return {}

  def search(self, query_embedding: list[float], k: int = 5, filter_type=None,
             filter_source: str | None = None, filter_id_prefix: str | None = None,
             filter_rowids: list[int] | None = None) -> list[dict]:
    """k-NN vector search. Returns dicts {id, source, type, content, mtime, distance, metadata}.

    Filters are applied to the CANDIDATE set before the k-NN is computed:
    filtering after the join would return nothing whenever the global nearest
    neighbours belong to another type.
    """
    where, params = [], []
    types = _types(filter_type)
    if types:
      where.append("type IN (" + ",".join("?" for _ in types) + ")")
      params.extend(types)
    if filter_source:
      where.append("source = ?")
      params.append(filter_source)
    if filter_id_prefix:
      where.append("id LIKE ?")
      params.append(filter_id_prefix + "%")
    if filter_rowids is not None:
      if not filter_rowids:
        return []
      where.append("rowid IN (" + ",".join("?" for _ in filter_rowids) + ")")
      params.extend(filter_rowids)
    candidate = ("AND v.rowid IN (SELECT rowid FROM chunks WHERE " + " AND ".join(where) + ")") if where else ""
    sql = f"""
      SELECT c.id, c.source, c.type, c.content, c.mtime, c.metadata, v.distance
      FROM vec_chunks v JOIN chunks c ON c.rowid = v.rowid
      WHERE v.embedding MATCH ? AND k = ? {candidate}
      ORDER BY v.distance
    """
    rows = self._connect().execute(sql, [_serialize_vec(query_embedding), k, *params]).fetchall()
    return [
      {"id": r["id"], "source": r["source"], "type": r["type"], "content": r["content"],
       "mtime": r["mtime"], "distance": r["distance"], "metadata": self._meta(r["metadata"])}
      for r in rows
    ]

  @staticmethod
  def fts_query(query: str) -> str | None:
    """Natural-language query to a tolerant FTS5 MATCH: quoted terms joined by OR.

    Bare FTS5 words are AND-ed, which is far too strict for questions.
    Returns None when no usable term remains.
    """
    terms = re.findall(r"\w{2,}", query, flags=re.UNICODE)
    if not terms:
      return None
    return " OR ".join('"' + t.replace('"', '""') + '"' for t in terms)

  def search_fts(self, query: str, k: int = 5, filter_type=None, filter_source: str | None = None,
                 filter_rowids: list[int] | None = None) -> list[dict]:
    """BM25 full-text search. `score` is the bm25 value: lower is better."""
    match = self.fts_query(query)
    if match is None:
      return []
    where, params = ["chunks_fts MATCH ?"], [match]
    types = _types(filter_type)
    if types:
      where.append("c.type IN (" + ",".join("?" for _ in types) + ")")
      params.extend(types)
    if filter_source:
      where.append("c.source = ?")
      params.append(filter_source)
    if filter_rowids is not None:
      if not filter_rowids:
        return []
      where.append("c.rowid IN (" + ",".join("?" for _ in filter_rowids) + ")")
      params.extend(filter_rowids)
    sql = f"""
      SELECT c.id, c.source, c.type, c.content, c.mtime, c.metadata, bm25(chunks_fts) AS score
      FROM chunks_fts JOIN chunks c ON c.rowid = chunks_fts.rowid
      WHERE {" AND ".join(where)}
      ORDER BY score LIMIT ?
    """
    try:
      rows = self._connect().execute(sql, [*params, k]).fetchall()
    except sqlite3.OperationalError:
      return []
    return [
      {"id": r["id"], "source": r["source"], "type": r["type"], "content": r["content"],
       "mtime": r["mtime"], "score": r["score"], "metadata": self._meta(r["metadata"])}
      for r in rows
    ]

  def _dated_session_rowids(self, requested_date: str) -> list[int]:
    from aura_core.memory.session_dates import session_header_date

    rows = self._connect().execute(
      "SELECT rowid,type,substr(content,1,512) AS preview FROM chunks "
      "WHERE type IN ('session','episode') AND instr(substr(content,1,512),?)>0",
      (requested_date,),
    ).fetchall()
    return [r["rowid"] for r in rows if session_header_date(r["type"], r["preview"]) == requested_date]

  def search_hybrid(self, query_embedding: list[float] | None, query_text: str, k: int = 5,
                    filter_type=None, filter_source: str | None = None, pool: int = 20,
                    rrf_k: int = 60, recency_weight: float = 0.15,
                    importance_weight: float = 0.05) -> list[dict]:
    """Vector + BM25 fusion by Reciprocal Rank Fusion.

    RRF: score(doc) = sum over lists of 1 / (rrf_k + rank). Only ranks matter,
    so heterogeneous scales (cosine distance vs bm25) never need calibrating.
    `pool` is how deep each list is read before fusion. With
    `query_embedding=None` this degrades to BM25 only.

    Returns dicts {id, source, type, content, metadata, rrf_score, vec_rank,
    fts_rank, recency_score, importance_score, final_score}, best first.
    """
    vec_results = (
      self.search(query_embedding, k=pool, filter_type=filter_type, filter_source=filter_source)
      if query_embedding is not None else []
    )
    fts_results = self.search_fts(query_text, k=pool, filter_type=filter_type, filter_source=filter_source)
    fused: dict[str, dict] = {}

    def _fuse(results: list[dict], rank_key: str, target: dict) -> None:
      for rank, r in enumerate(results):
        e = target.setdefault(r["id"], {
          "id": r["id"], "source": r["source"], "type": r["type"], "content": r["content"],
          "mtime": r.get("mtime"), "metadata": r["metadata"],
          "rrf_score": 0.0, "vec_rank": None, "fts_rank": None,
        })
        e["rrf_score"] += 1.0 / (rrf_k + rank + 1)
        e[rank_key] = rank + 1

    _fuse(vec_results, "vec_rank", fused)
    _fuse(fts_results, "fts_rank", fused)

    # An explicit date asked for a session constrains the selection BEFORE the
    # top-k: re-ranking the global pool cannot recover a summary that is absent from it.
    from aura_core.memory.session_dates import requested_session_date

    types = set(_types(filter_type))
    requested_date = requested_session_date(query_text) if not types or types & {"session", "episode"} else None
    if requested_date:
      rowids = self._dated_session_rowids(requested_date)
      if rowids:
        dated: dict[str, dict] = {}
        if query_embedding is not None:
          _fuse(self.search(query_embedding, k=pool, filter_type=filter_type,
                            filter_source=filter_source, filter_rowids=rowids), "vec_rank", dated)
        _fuse(self.search_fts(query_text, k=pool, filter_type=filter_type,
                              filter_source=filter_source, filter_rowids=rowids), "fts_rank", dated)
        for item in dated.values():
          item["session_date_match"] = requested_date
        fused.update(dated)

    temporal_query = bool(re.search(
      r"\b(recent|latest|current|now|today|yesterday|actuel|maintenant|aujourd'hui|hier)\b",
      query_text, flags=re.IGNORECASE))
    effective_recency_weight = recency_weight * (2 if temporal_query else 1)
    max_rank_score = 1.0 / (rrf_k + 1)
    now_ts = time.time()
    for entry in fused.values():
      mtime = entry.get("mtime")
      if not mtime:
        try:
          mtime = datetime.fromisoformat(str(entry["metadata"].get("timestamp", ""))).timestamp()
        except (TypeError, ValueError):
          mtime = None
      age_days = max(0.0, (now_ts - float(mtime)) / 86400) if mtime else 3650.0
      entry["recency_score"] = math.exp(-age_days / 90)
      try:
        importance = float(entry["metadata"].get("importance", 0.5))
      except (TypeError, ValueError):
        importance = 0.5
      entry["importance_score"] = max(0.0, min(1.0, importance))
      entry["final_score"] = (
        entry["rrf_score"]
        + max_rank_score * effective_recency_weight * entry["recency_score"]
        + max_rank_score * importance_weight * entry["importance_score"]
      )
    ranked = sorted(fused.values(), key=lambda e: (bool(e.get("session_date_match")), e["final_score"]),
                    reverse=True)
    return ranked[:k]

  # -- introspection --------------------------------------------------------

  def stats(self) -> dict:
    conn = self._connect()
    total = conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
    by_type = {r["type"]: r["n"] for r in conn.execute("SELECT type, COUNT(*) AS n FROM chunks GROUP BY type")}
    sources = conn.execute("SELECT COUNT(*) AS n FROM sources_meta").fetchone()["n"]
    size = self.db_path.stat().st_size if self.db_path.exists() else 0
    return {"db_path": str(self.db_path), "db_size_bytes": size, "dim": self.dim,
            "total_chunks": total, "total_sources": sources, "by_type": by_type}

  def health(self) -> dict:
    """Check store invariants without loading any embedding model.

    `vectors` may be lower than `chunks` (chunks indexed lexically only); more
    vectors than chunks means orphaned vectors, which is the real anomaly.
    """
    conn = self._connect()
    chunks = conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
    vectors = conn.execute("SELECT COUNT(*) AS n FROM vec_chunks").fetchone()["n"]
    quick_check = conn.execute("PRAGMA quick_check").fetchone()[0]
    orphans = vectors - min(vectors, chunks)
    tolerance = max(2, chunks // 1000)
    return {"ok": quick_check == "ok" and orphans <= tolerance, "quick_check": quick_check,
            "chunks": chunks, "vectors": vectors, "orphan_vectors": orphans,
            "chunks_without_vector": max(0, chunks - vectors)}

  def count(self, type_: str | None = None) -> int:
    conn = self._connect()
    if type_:
      return conn.execute("SELECT COUNT(*) AS n FROM chunks WHERE type = ?", (type_,)).fetchone()["n"]
    return conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]

  def close(self) -> None:
    if self._conn is not None:
      self._conn.close()
      self._conn = None
