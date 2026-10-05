import pytest

from aura_core.memory.embed_engine import HashingEmbedder
from aura_core.memory.sqlite_vec_store import SqliteVecStore

EMB = HashingEmbedder(dim=64)


@pytest.fixture
def store(tmp_path):
  s = SqliteVecStore(tmp_path / "m.db", dim=EMB.dim)
  yield s
  s.close()


def _docs():
  return [
    {"id": "a", "content": "The ferry leaves the harbour at dawn"},
    {"id": "b", "content": "Invoice ABC-12345 was paid by bank transfer"},
    {"id": "c", "content": "Lifejackets are stored under the benches"},
  ]


def test_upsert_and_count(store):
  assert store.add_documents(_docs(), EMB) == 3
  assert store.count() == 3
  assert store.health()["ok"]


def test_upsert_is_idempotent(store):
  store.add_documents(_docs(), EMB)
  store.add_documents(_docs(), EMB)
  assert store.count() == 3
  assert store.health()["vectors"] == 3


def test_fts_finds_exact_identifier(store):
  store.add_documents(_docs(), EMB)
  hits = store.search_fts("ABC-12345", k=3)
  assert [h["id"] for h in hits][:1] == ["b"]


def test_vector_search_returns_nearest(store):
  store.add_documents(_docs(), EMB)
  hits = store.search(EMB.embed(["lifejackets stored under benches"])[0], k=1)
  assert hits[0]["id"] == "c"


def test_hybrid_rank_fields(store):
  store.add_documents(_docs(), EMB)
  q = "where are lifejackets"
  hits = store.search_hybrid(EMB.embed([q])[0], q, k=3)
  assert hits[0]["id"] == "c"
  assert hits[0]["fts_rank"] == 1 and hits[0]["vec_rank"] is not None
  assert all("final_score" in h for h in hits)


def test_hybrid_surfaces_what_vectors_miss(store):
  """An exact identifier is found by BM25 even when the vector side is blind to it."""
  store.add_documents(_docs(), EMB)
  blind = [0.0] * 63 + [1.0]  # a query vector unrelated to any document
  hits = store.search_hybrid(blind, "invoice ABC-12345", k=1)
  assert hits[0]["id"] == "b"
  assert hits[0]["fts_rank"] == 1


def test_lexical_only_mode_without_vectors(store):
  store.add_documents(_docs(), embedder=None)
  assert [h["id"] for h in store.search_hybrid(None, "ferry dawn", k=1)] == ["a"]
  health = store.health()
  assert health["ok"] and health["chunks_without_vector"] == 3


def test_type_filter_applies_before_knn(store):
  store.upsert([
    {"id": "t1", "source": "s1", "type": "topic", "content": "alpha", "embedding": EMB.embed(["alpha"])[0]},
    {"id": "n1", "source": "s2", "type": "note", "content": "alpha beta", "embedding": EMB.embed(["alpha beta"])[0]},
  ])
  hits = store.search(EMB.embed(["alpha beta"])[0], k=1, filter_type="topic")
  assert [h["id"] for h in hits] == ["t1"]


def test_dimension_mismatch_is_rejected(store):
  with pytest.raises(ValueError, match="invalid embedding"):
    store.upsert([{"id": "x", "source": "s", "type": "topic", "content": "x", "embedding": [0.1, 0.2]}])


def test_reopening_with_another_dim_fails(tmp_path):
  SqliteVecStore(tmp_path / "d.db", dim=64).close()
  with pytest.raises(ValueError, match="created with dim=64"):
    SqliteVecStore(tmp_path / "d.db", dim=128)
  reopened = SqliteVecStore(tmp_path / "d.db")  # no dim given: adopts the stored one
  assert reopened.dim == 64
  reopened.close()


def test_delete_by_source(store):
  store.add_documents(_docs(), EMB)
  assert store.delete_by_source("doc/a") == 1
  assert store.count() == 2
  assert store.health()["ok"]
  assert store.search_fts("dawn", k=3) == []  # FTS index followed the delete


def test_replace_source_is_atomic_and_replaces(store):
  def chunk(i, text):
    return {"id": f"s:{i}", "source": "s", "type": "topic", "content": text, "embedding": EMB.embed([text])[0]}

  store.replace_source("s", "topic", "h1", [chunk(0, "first version"), chunk(1, "more text")])
  store.replace_source("s", "topic", "h2", [chunk(0, "second version")])
  assert store.count() == 1
  assert store.get_source_hash("s") == "h2" and store.is_indexed("s", "h2")
  assert store.search_fts("first", k=3) == []
  with pytest.raises(ValueError):
    store.replace_source("s", "topic", "h3", [{**chunk(0, "x"), "source": "other"}])


def test_secrets_are_masked_before_writing(store):
  secret = "sk-" + "a1B2c3D4e5F6g7H8i9J0"
  store.upsert([{"id": "z", "source": "s", "type": "topic", "content": f"deploy with key {secret} today"}])
  row = store.search_fts("deploy", k=1)[0]
  assert secret not in row["content"]
  assert store.search_fts("a1B2c3D4e5F6g7H8i9J0", k=1) == []  # not in the FTS index either


def test_read_only_store(tmp_path):
  w = SqliteVecStore(tmp_path / "r.db", dim=EMB.dim)
  w.add_documents(_docs(), EMB)
  w.close()
  r = SqliteVecStore(tmp_path / "r.db", read_only=True)
  assert r.count() == 3 and r.dim == EMB.dim
  r.close()


def test_session_date_constrains_selection(store):
  def chunk(i, day):
    text = f"Session {i}abc - {day} - weekly review\nDiscussed the ferry timetable"
    return {"id": f"sess{i}", "source": f"s{i}", "type": "session", "content": text,
            "embedding": EMB.embed([text])[0]}

  store.upsert([chunk(i, f"2026-03-{10 + i}") for i in range(1, 6)])
  q = "what did we discuss in the session of 2026-03-13"
  hits = store.search_hybrid(EMB.embed([q])[0], q, k=1, filter_type="session")
  assert hits[0]["session_date_match"] == "2026-03-13"
  assert hits[0]["id"] == "sess3"
