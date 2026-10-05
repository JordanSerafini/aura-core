"""Durable evidence for extracted facts, independent of mutable chunk IDs.

Context: a separate extractor (not included here) can keep a `fact_ledger`
table (id, source, subject, object, chunk_id, superseded_by). When a source
document is re-chunked its chunk IDs change; this module keeps the evidence
text of each fact and re-links facts to the new chunks.

Every function is a no-op when `fact_ledger` does not exist, so the store works
without it. No autonomous truth promotion: ambiguous links are flagged
`needs_review`, never resolved by guessing.

`absent_since` records when a fact's OBJECT stopped appearing in its source
note. `needs_review` mixes three cases (subject reworded, several candidate
chunks, value gone); only the last one means the note no longer states the
fact, so it is measured separately and compared after compaction (accents,
case, punctuation and spaces removed) so "98.4%" and "98,4 %" are the same
object. When the measurement is impossible (no `chunks` table) the answer is
"unknown", never "absent".
"""

import hashlib
import re
import sqlite3
import time
import unicodedata
from pathlib import Path


def ensure_schema(conn: sqlite3.Connection) -> None:
  conn.execute("""CREATE TABLE IF NOT EXISTS fact_provenance(
    fact_id TEXT PRIMARY KEY, source TEXT NOT NULL, evidence TEXT,
    evidence_sha256 TEXT, source_sha256 TEXT, status TEXT NOT NULL,
    superseded_fact_id TEXT, updated_at REAL NOT NULL
  )""")
  cols = {r[1] for r in conn.execute("PRAGMA table_info(fact_provenance)")}
  if "absent_since" not in cols:
    conn.execute("ALTER TABLE fact_provenance ADD COLUMN absent_since REAL")
  # DDL here must not commit an enclosing replacement transaction.


def _has_facts(conn: sqlite3.Connection) -> bool:
  return bool(conn.execute("SELECT 1 FROM sqlite_master WHERE name='fact_ledger'").fetchone())


def _has_chunks(conn: sqlite3.Connection) -> bool:
  return bool(conn.execute("SELECT 1 FROM sqlite_master WHERE name='chunks'").fetchone())


def capture_source(conn: sqlite3.Connection, source: str) -> None:
  """Snapshot the evidence text of a source's facts BEFORE its chunks are replaced."""
  if not _has_facts(conn) or not _has_chunks(conn):
    return
  rows = conn.execute(
    """SELECT f.id, f.chunk_id, f.superseded_by, c.content
       FROM fact_ledger f LEFT JOIN chunks c ON c.id=f.chunk_id WHERE f.source=?""",
    (source,),
  ).fetchall()
  for fid, _cid, successor, content in rows:
    old = conn.execute("SELECT evidence FROM fact_provenance WHERE fact_id=?", (fid,)).fetchone()
    if old and old[0]:
      continue
    successor_id = None
    if successor:
      found = conn.execute("SELECT id FROM fact_ledger WHERE chunk_id=? LIMIT 1", (successor,)).fetchone()
      successor_id = found[0] if found else None
    digest = hashlib.sha256(content.encode()).hexdigest() if content else None
    conn.execute(
      """INSERT INTO fact_provenance(fact_id,source,evidence,evidence_sha256,
           source_sha256,status,superseded_fact_id,updated_at) VALUES(?,?,?,?,?,?,?,?)
         ON CONFLICT(fact_id) DO UPDATE SET evidence=excluded.evidence,
           evidence_sha256=excluded.evidence_sha256, superseded_fact_id=excluded.superseded_fact_id,
           status=excluded.status, updated_at=excluded.updated_at""",
      (fid, source, content, digest, None, "linked" if content else "needs_review", successor_id, time.time()),
    )


def reconcile_source(conn: sqlite3.Connection, source: str) -> None:
  """Re-link facts to the new chunks of `source`. A fact links only on an unambiguous match."""
  if not _has_facts(conn):
    return
  columns = {r[1] for r in conn.execute("PRAGMA table_info(fact_ledger)")}
  if "object" not in columns:
    return
  chunks = conn.execute("SELECT id,content FROM chunks WHERE source=?", (source,)).fetchall()
  facts = conn.execute("SELECT id,object,subject FROM fact_ledger WHERE source=?", (source,)).fetchall()
  for fid, obj, subject in facts:
    # Literal object AND subject must both appear; ambiguity goes to review.
    matches = [
      (cid, text) for cid, text in chunks
      if str(obj).strip() and str(obj).casefold() in text.casefold()
      and str(subject).casefold() in text.casefold()
    ]
    if len(matches) == 1:
      status = "linked"
      conn.execute("UPDATE fact_ledger SET chunk_id=? WHERE id=?", (matches[0][0], fid))
    else:
      status = "needs_review" if chunks or Path(source).is_file() else "source_missing"
    conn.execute("UPDATE fact_provenance SET status=?,updated_at=? WHERE fact_id=?", (status, time.time(), fid))
  refresh_absence(conn, [source])


def _compact(text: str) -> str:
  """Lowercase, no accents, punctuation or spaces: "98,4 %" == "98.4%"."""
  text = unicodedata.normalize("NFKD", text or "")
  text = "".join(c for c in text if not unicodedata.combining(c))
  return re.sub(r"[\W_]+", "", text.lower())


_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
_WORD = re.compile(r"[^\W\d_]{4,}")


def object_present(obj: str, corpus: str) -> bool:
  """Does the note (`corpus`, already passed through `_compact`) still carry the value?

  Deliberately lenient: concluding "absent" wrongly would close a correct fact.
  Present if the compacted value appears as-is; else, if it contains numbers,
  if ALL its numbers appear; else, if all its words of 4+ letters appear.
  """
  needle = _compact(obj)
  if not needle or needle in corpus:
    return True
  nums = [_compact(n) for n in _NUMBER.findall(obj or "")]
  if nums:
    return all(n in corpus for n in nums)
  words = [_compact(w) for w in _WORD.findall(obj or "")]
  return bool(words) and all(w in corpus for w in words)


def refresh_absence(conn: sqlite3.Connection, sources: list[str] | None = None) -> set[str]:
  """Update `absent_since`; return the fact IDs whose absence state changed.

  `absent_since` is the start of the note version that no longer carries the
  value: the largest `mtime` among its chunks (capped at now), or now if the
  note is gone. Once set it is never moved, otherwise a note touched every day
  would postpone closing the fact forever.
  """
  if not _has_facts(conn) or not _has_chunks(conn):
    return set()  # not measurable: unknown, never "absent"
  ensure_schema(conn)
  if sources is None:
    sources = [r[0] for r in conn.execute("SELECT DISTINCT source FROM fact_ledger")]
  now = time.time()
  changed: set[str] = set()
  for source in sources:
    rows = conn.execute("SELECT content, mtime FROM chunks WHERE source=?", (source,)).fetchall()
    if rows:
      corpus = _compact(" ".join(r[0] or "" for r in rows))
      since = min(now, max(float(r[1] or 0) for r in rows) or now)
    elif Path(source).is_file():
      try:
        corpus = _compact(Path(source).read_text(encoding="utf-8", errors="ignore"))
      except OSError:  # unreadable note: not measured
        continue
      since = min(now, Path(source).stat().st_mtime)
    else:
      corpus, since = "", now  # note gone: nothing carries the value any more
    facts = conn.execute(
      "SELECT f.id, f.object, p.fact_id, p.absent_since FROM fact_ledger f "
      "LEFT JOIN fact_provenance p ON p.fact_id=f.id WHERE f.source=?",
      (source,),
    ).fetchall()
    for fid, obj, has_prov, absent_since in facts:
      absent = not object_present(str(obj), corpus)
      if absent == (absent_since is not None):
        continue
      if not has_prov:
        conn.execute(
          "INSERT OR IGNORE INTO fact_provenance(fact_id,source,status,updated_at) VALUES(?,?,?,?)",
          (fid, source, "needs_review", now),
        )
      conn.execute(
        "UPDATE fact_provenance SET absent_since=?, updated_at=? WHERE fact_id=?",
        (since if absent else None, now, fid),
      )
      changed.add(fid)
  return changed
