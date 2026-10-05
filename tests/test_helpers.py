import sqlite3

from aura_core.memory import fact_provenance as fp
from aura_core.memory.embed_engine import HashingEmbedder
from aura_core.memory.hybrid_search import BM25, HybridSearchEngine
from aura_core.memory.memory_types import (
  Episode,
  KnowledgeTriple,
  MemoryScore,
  Skill,
  calculate_recency_score,
)
from aura_core.memory.reranker import rerank
from aura_core.memory.session_dates import requested_session_date, session_header_date


# -- session dates ------------------------------------------------------------
def test_requested_session_date_variants():
  assert requested_session_date("the session of 2026-03-14") == "2026-03-14"
  assert requested_session_date("session du 14 mars 2026") == "2026-03-14"
  assert requested_session_date("session on March 14th, 2026") == "2026-03-14"


def test_requested_session_date_refuses_ambiguity():
  assert requested_session_date("2026-03-14") is None  # not about a session
  assert requested_session_date("sessions on 2026-03-14 and 2026-03-15") is None
  assert requested_session_date("session after 2026-03-14") is None
  assert requested_session_date("session of 2026-02-31") is None  # invalid calendar date


def test_session_header_date_ignores_body_dates():
  body = "Session abc123 - 2026-03-14 - review\nmentions 2025-01-01"
  assert session_header_date("session", body) == "2026-03-14"
  assert session_header_date("episode", "[session/x]\n" + body) == "2026-03-14"
  assert session_header_date("topic", body) is None


# -- BM25 and in-memory hybrid ---------------------------------------------------
def test_bm25_ranks_rare_terms_higher():
  bm = BM25()
  bm.fit([("a", "ferry ferry harbour"), ("b", "harbour crane maintenance"), ("c", "ferry schedule")])
  top = bm.search("crane", top_k=3)
  assert [t[0] for t in top] == ["b"]
  assert bm.search("zzz-unknown") == []


def test_in_memory_hybrid_bm25_only_and_with_vectors():
  docs = [("a", "ferry leaves at dawn", None), ("b", "lifejackets under benches", None)]
  lexical = HybridSearchEngine()
  lexical.index_batch(docs)
  assert lexical.search("lifejackets", top_k=1)[0].id == "b"
  dense = HybridSearchEngine(embedder=HashingEmbedder(64))
  dense.index_batch(docs)
  best = dense.search("ferry dawn", top_k=1)[0]
  assert best.id == "a" and best.vector_score > 0 and best.bm25_score > 0


# -- types ------------------------------------------------------------------------
def test_types_roundtrip_and_ids():
  ep = Episode(id="", timestamp="2026-01-15T10:00:00", context="c", action="a", outcome="o",
               thought_process="t", entities=["x"])
  assert ep.id.startswith("ep_") and Episode.from_dict(ep.to_dict()) == ep
  sk = Skill(id="", name="n", description="d", pattern="p", trigger_conditions=[], action_template="t")
  assert sk.id.startswith("sk_") and Skill.from_dict(sk.to_dict()) == sk
  kt = KnowledgeTriple(id="", subject="Python", predicate="is_a", object="language")
  assert kt.id.startswith("kg_") and kt.to_text() == "Python is_a language"
  assert KnowledgeTriple.from_dict(kt.to_dict()) == kt


def test_scoring_and_recency():
  s = MemoryScore(similarity=1.0, importance=1.0, recency=1.0, access_frequency=1.0)
  assert abs(s.combined_score - 1.0) < 1e-9
  from datetime import datetime

  now = datetime(2026, 1, 31)
  assert calculate_recency_score("2026-01-31T00:00:00", 30, now=now) == 1.0
  assert 0.36 < calculate_recency_score("2026-01-01T00:00:00", 30, now=now) < 0.38  # e^-1
  assert calculate_recency_score("not a date") == 0.5


# -- reranker fail-safe ------------------------------------------------------------
def test_rerank_empty_and_failsafe(monkeypatch):
  assert rerank("q", []) == []
  from aura_core.memory import reranker

  def boom():
    raise RuntimeError("no model")

  monkeypatch.setattr(reranker, "_model", boom)
  cands = [{"content": "a"}, {"content": "b"}, {"content": "c"}]
  assert rerank("q", cands, top_k=2) == cands[:2]


# -- fact provenance ---------------------------------------------------------------
def _db():
  conn = sqlite3.connect(":memory:")
  conn.executescript("""
    CREATE TABLE chunks(id TEXT PRIMARY KEY, source TEXT, content TEXT, mtime INTEGER);
    CREATE TABLE fact_ledger(id TEXT PRIMARY KEY, source TEXT, subject TEXT, object TEXT,
                             chunk_id TEXT, superseded_by TEXT);
  """)
  fp.ensure_schema(conn)
  return conn


def test_provenance_is_noop_without_ledger():
  conn = sqlite3.connect(":memory:")
  fp.capture_source(conn, "s")
  fp.reconcile_source(conn, "s")
  assert fp.refresh_absence(conn) == set()


def test_object_present_is_lenient_about_format():
  assert fp.object_present("98,4 %", fp._compact("uptime was 98.4% last week"))
  assert fp.object_present("88% full (462 GB free)", fp._compact("88 % used, 462 GB free"))
  assert not fp.object_present("failed", fp._compact("everything is working"))


def test_absence_is_detected_and_not_moved():
  conn = _db()
  conn.execute("INSERT INTO chunks VALUES('c1','note','The pump is running at 40 psi',100)")
  conn.execute("INSERT INTO fact_ledger VALUES('f1','note','pump','40 psi','c1',NULL)")
  assert fp.refresh_absence(conn, ["note"]) == set()  # value present: nothing to flag
  conn.execute("UPDATE chunks SET content='The pump was replaced', mtime=200 WHERE id='c1'")
  assert fp.refresh_absence(conn, ["note"]) == {"f1"}
  first = conn.execute("SELECT absent_since FROM fact_provenance WHERE fact_id='f1'").fetchone()[0]
  assert first == 200.0
  conn.execute("UPDATE chunks SET mtime=300 WHERE id='c1'")
  assert fp.refresh_absence(conn, ["note"]) == set()  # already flagged: not re-dated
  assert conn.execute("SELECT absent_since FROM fact_provenance WHERE fact_id='f1'").fetchone()[0] == first


def test_reconcile_links_only_unambiguous_matches():
  conn = _db()
  conn.execute("INSERT INTO chunks VALUES('c1','note','The pump runs at 40 psi',1)")
  conn.execute("INSERT INTO chunks VALUES('c2','note','The valve sticks',1)")
  conn.execute("INSERT INTO fact_ledger VALUES('f1','note','pump','40 psi','old',NULL)")
  fp.capture_source(conn, "note")
  fp.reconcile_source(conn, "note")
  assert conn.execute("SELECT chunk_id FROM fact_ledger WHERE id='f1'").fetchone()[0] == "c1"
  conn.execute("INSERT INTO chunks VALUES('c3','note','pump 40 psi again',1)")
  fp.reconcile_source(conn, "note")
  status = conn.execute("SELECT status FROM fact_provenance WHERE fact_id='f1'").fetchone()[0]
  assert status == "needs_review"  # two candidate chunks: ambiguity is flagged, not guessed
