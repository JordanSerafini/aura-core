import json

import pytest

from aura_core.memory.embed_engine import HashingEmbedder
from aura_core.memory.secret_redact import MARK
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


def _source_chunk(i, text):
  return {"id": f"s:{i}", "source": "s", "type": "topic", "content": text, "embedding": EMB.embed([text])[0]}


def test_replace_source_is_atomic_and_replaces(store):
  chunk = _source_chunk
  store.replace_source("s", "topic", "h1", [chunk(0, "first version"), chunk(1, "more text")])
  store.replace_source("s", "topic", "h2", [chunk(0, "second version")])
  assert store.count() == 1
  assert store.get_source_hash("s") == "h2" and store.is_indexed("s", "h2")
  assert store.search_fts("first", k=3) == []
  with pytest.raises(ValueError):
    store.replace_source("s", "topic", "h3", [{**chunk(0, "x"), "source": "other"}])


def test_replace_source_rolls_back_when_a_late_step_fails(store, monkeypatch):
  # The ValueError above fires before the transaction opens. Fail inside it instead:
  # neither the chunks nor sources_meta may keep the half-done replacement.
  from aura_core.memory import fact_provenance

  store.replace_source("s", "topic", "h1", [_source_chunk(0, "first version"), _source_chunk(1, "more text")])

  def boom(conn, source):
    raise RuntimeError("reconcile failed")

  monkeypatch.setattr(fact_provenance, "reconcile_source", boom)
  with pytest.raises(RuntimeError):
    store.replace_source("s", "topic", "h2", [_source_chunk(0, "second version")])
  assert store.count() == 2
  assert store.get_source_hash("s") == "h1"


class CountingEmbedder:
  def __init__(self, fail=False):
    self.dim, self.texts, self.fail = EMB.dim, [], fail

  def embed(self, texts):
    if self.fail:
      raise RuntimeError("model unavailable")
    self.texts.extend(texts)
    return EMB.embed(texts)


def _bare_chunk(i, text):
  return {"id": f"s:{i}", "source": "s", "type": "topic", "content": text}


def test_replace_source_only_embeds_the_changed_chunks(store):
  emb = CountingEmbedder()
  store.replace_source("s", "topic", "h1", [_bare_chunk(0, "monday: backups ran"), _bare_chunk(1, "tuesday: quiet")],
                       embedder=emb)
  assert emb.texts == ["monday: backups ran", "tuesday: quiet"]
  kept = store.vectors_by_hash("s")
  assert len(kept) == 2
  # The journal grows: one new chunk, one rewritten, one unchanged.
  emb.texts.clear()
  store.replace_source("s", "topic", "h2", [_bare_chunk(0, "monday: backups ran"),
                                            _bare_chunk(1, "tuesday: disk full"),
                                            _bare_chunk(2, "wednesday: fixed")], embedder=emb)
  assert emb.texts == ["tuesday: disk full", "wednesday: fixed"]
  after = store.vectors_by_hash("s")
  (same,) = set(kept) & set(after)  # only monday's text is in both versions
  assert len(after) == 3 and after[same] == kept[same]
  hit = store.search(EMB.embed(["monday: backups ran"])[0], k=1)[0]
  assert hit["id"] == "s:0" and hit["distance"] < 1e-6  # the reused vector is the original one
  assert store.health()["ok"] and store.health()["vectors"] == 3


def test_replace_source_with_a_failing_embedder_changes_nothing(store):
  store.replace_source("s", "topic", "h1", [_bare_chunk(0, "first version")], embedder=CountingEmbedder())
  with pytest.raises(RuntimeError):
    store.replace_source("s", "topic", "h2", [_bare_chunk(0, "second version")], embedder=CountingEmbedder(fail=True))
  assert store.get_source_hash("s") == "h1"
  assert [h["id"] for h in store.search_fts("first", k=3)] == ["s:0"]


def test_secrets_are_masked_before_writing(store):
  secret = "sk-" + "a1B2c3D4e5F6g7H8i9J0"
  store.upsert([{"id": "z", "source": "s", "type": "topic", "content": f"deploy with key {secret} today"}])
  row = store.search_fts("deploy", k=1)[0]
  assert secret not in row["content"]
  assert store.search_fts("a1B2c3D4e5F6g7H8i9J0", k=1) == []  # not in the FTS index either


def _fake(prefix: str, body: str) -> str:
  # Built at runtime so no secret-shaped literal sits in the repository.
  return prefix + body


def _stored_metadata(store, chunk_id):
  return store._connect().execute("SELECT metadata FROM chunks WHERE id = ?", (chunk_id,)).fetchone()[0]


def test_metadata_is_masked_value_by_value(store, tmp_path):
  # json.dumps escapes quotes and turns {"api_key": v} into '"api_key": "v"': masked after that,
  # `password="..."` and `name: value` are no longer recognised, while the same text in `content` is.
  pw = _fake("Xq7pL2", "wzR9kT4m")
  token = _fake("ghp_", "A1b2C3d4E5f6G7h8I9j0K1l2")
  other = _fake("ghp_", "Z9y8X7w6V5u4T3s2R1q0P9o8")

  class Opaque:  # stringified by json.dumps(default=str)
    def __str__(self):
      return f"line 1\npassword={pw}"

  meta = {"cmd": f'export DB_PASSWORD="{pw}"', "note": f"line 1\npassword={pw}", "api_key": pw,
          "nested": {"steps": [f"use {token} once", "then rotate"]}, "count": 3, "label": "plain words"}
  store.upsert([{"id": "m", "source": "s", "type": "topic", "content": "deploy notes",
                 "metadata": {**meta, "object": Opaque(), other: "a key is masked too"}}])
  raw = _stored_metadata(store, "m")
  assert pw not in raw and token not in raw and other not in raw
  kept = json.loads(raw)
  assert kept["api_key"] == MARK and kept["cmd"].startswith("export DB_PASSWORD=")
  assert (kept["count"], kept["label"], kept["nested"]["steps"][1]) == (3, "plain words", "then rotate")
  plain = SqliteVecStore(tmp_path / "plain.db", dim=EMB.dim, redact_secrets=False)
  plain.upsert([{"id": "m", "source": "s", "type": "topic", "content": "deploy notes", "metadata": meta}])
  assert json.loads(_stored_metadata(plain, "m")) == meta  # masking switched off: stored as given
  plain.close()


def test_ordinary_metadata_fields_stay_readable(store):
  # A field is masked as `name=value` only when its name holds a credential. In a record, `key`, `sort_key`
  # or `auth` are ordinary settings: the same words in prose (`auth=...`) are still masked in `content`.
  pw = _fake("Xq7pL2", "wzR9kT4m")
  ordinary = {"key": "notes/backup-policy.md", "sort_key": "title_asc", "auth": "required",
              "token_count": "125000", "token_type": "refresh", "password_policy": "rotate yearly",
              "passage": "chapter two", "max_tokens": 4096}
  secret = {"password": pw, "DB_PASSWORD": pw, "apiKey": pw, "client_secret": pw, "access-token": pw,
            "privateKey": pw, "clientSecret": pw, "accessToken": pw, "aws_secret_access_key": pw, "credentials": pw}
  store.upsert([{"id": "m", "source": "s", "type": "topic", "content": "deploy notes",
                 "metadata": {**ordinary, **secret}}])
  kept = json.loads(_stored_metadata(store, "m"))
  assert {k: kept[k] for k in ordinary} == ordinary
  assert {k: kept[k] for k in secret} == dict.fromkeys(secret, MARK)


def test_a_credential_field_is_masked_whatever_its_value(store):
  # Under a credential name the value is the secret, short, with spaces or starting with `$`. Only a whole
  # reference, a placeholder or an unset marker stays readable.
  pw = _fake("Xq7pL2", "wzR9kT4m")
  secret = {"password": "1234", "PGPASSWORD": pw, "SSHPASS": pw, "db_pass": pw, "pw": pw,
            "passphrase": "correct horse battery staple", "Authorization": _fake("Basic ", "dXNlcjpwYXNz"),
            "api_key": _fake("$uper", "$ecret99"), "token": "a b"}
  kept = {"token_ref": "x", "access_token": "${API_TOKEN}", "db_password": " $DB_PASSWORD\n", "secret": "None",
          "client_secret": "<your secret>", "api_key_hint": "rotate yearly", "password_hint": "", "boarding": "gate 4"}
  store.upsert([{"id": "m", "source": "s", "type": "topic", "content": "deploy notes",
                 "metadata": {**secret, **kept}}])
  stored = json.loads(_stored_metadata(store, "m"))
  assert {k: stored[k] for k in secret} == dict.fromkeys(secret, MARK)
  assert {k: stored[k] for k in kept} == kept


def test_the_embedder_only_sees_masked_text(store):
  token = _fake("ghp_", "A1b2C3d4E5f6G7h8I9j0K1l2")
  pw = _fake("Xq7pL2", "wzR9kT4m")
  emb = CountingEmbedder()
  store.add_documents([{"id": "d", "content": f"rotate the CI token {token} on monday"}], emb)
  store.replace_source("s", "topic", "h1", [_bare_chunk(0, f"password={pw} for the backup box")], embedder=emb)
  assert len(emb.texts) == 2 and all(MARK in text for text in emb.texts)
  assert not any(token in text or pw in text for text in emb.texts)
  # The vector stored is the one of the text stored, so a search on the masked text finds it exactly.
  stored = store.search_fts("rotate", k=1)[0]["content"]
  hit = store.search(EMB.embed([stored])[0], k=1)[0]
  assert hit["id"] == "d" and hit["distance"] < 1e-6


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


def test_mark_source_indexed_without_chunks(store):
  store.mark_source_indexed("notes/a.md", "note", "h1", 0)
  assert store.is_indexed("notes/a.md", "h1") and not store.is_indexed("notes/a.md", "h2")
  store.mark_source_indexed("notes/a.md", "note", "h2", 3)
  assert store.get_source_hash("notes/a.md") == "h2"
