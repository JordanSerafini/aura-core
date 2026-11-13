import inspect
import sqlite3

import pytest

from aura_core.memory.embed_engine import HashingEmbedder
from aura_core.memory.retrieval import (
  EMBED_MAX_CHARS,
  conversation_context,
  embed_text,
  prompt_selection,
  semantic_rows,
)
from aura_core.memory.sqlite_vec_store import SqliteVecStore

EMB = HashingEmbedder(dim=64)


@pytest.fixture
def db(tmp_path):
  path = tmp_path / "r.db"
  store = SqliteVecStore(path, dim=EMB.dim)
  store.add_documents([
    {"id": "1", "content": "Lifejackets are stored under the benches on the main deck", "source": "safety.md"},
    {"id": "2", "content": "Lifejackets inspection happens every six months by the safety officer",
     "source": "safety.md"},
    {"id": "3", "content": "Ferry timetable: weekday departures at dawn and noon", "source": "timetable.md"},
    {"id": "4", "content": "Incident INC-4471 was a hydraulic hose failure", "source": "incidents.md"},
    {"id": "5", "content": "Radio uses channel twelve", "source": "radio.md", "type": "vault"},
  ], EMB, type_="topic")
  store.close()
  return path


def test_picks_relevant_extract(db):
  picked = prompt_selection("where are the lifejackets stored on deck", "s1", db=db)
  assert picked and picked[0][2] == "safety.md"


def test_one_extract_per_source(db):
  picked = prompt_selection("lifejackets stored inspection safety officer benches", "s1", db=db)
  assert len({p[2] for p in picked}) == len(picked)


def test_nothing_is_repeated_within_a_session(db):
  seen = {}
  first = prompt_selection("ferry timetable weekday departures", "s1", db=db, seen=seen)
  again = prompt_selection("ferry timetable weekday departures", "s1", db=db, seen=seen)
  other = prompt_selection("ferry timetable weekday departures", "s2", db=db, seen=seen)
  assert first and not again and other


def test_purge_keeps_current_session(db):
  seen = {f"old{i}": set() for i in range(80)}
  first = prompt_selection("ferry timetable weekday departures", "s1", db=db, seen=seen)
  again = prompt_selection("ferry timetable weekday departures", "s1", db=db, seen=seen)
  assert first and not again


def test_abstains_without_subject(db):
  assert prompt_selection("ok continue", "s1", db=db) == []


def test_abstains_on_unknown_identifier(db):
  timing = {}
  assert prompt_selection("what happened in INC-9999 hydraulic hose", "s1", db=db, timing=timing) == []
  assert timing["abstention"] == "unknown_identifier"
  assert prompt_selection("what happened in INC-4471 hydraulic hose", "s1", db=db)


def test_single_shared_word_is_not_enough(db):
  # keyword side only: "lifejackets" alone appears in the corpus, but the prompt shares no 2nd word
  assert prompt_selection("lifejackets zeppelin quasar", "s1", db=db) == []


def test_vector_side_rescues_a_chunk_and_types_are_respected(db):
  picked = prompt_selection("radio channel twelve", "s1", db=db, embedder=lambda t: EMB.embed([t])[0])
  assert all(p[1] in {"topic", "note"} for p in picked)  # the 'vault' chunk is out of scope by default
  picked = prompt_selection("radio channel twelve", "s2", db=db, types=("vault",),
                            embedder=lambda t: EMB.embed([t])[0])
  assert picked and picked[0][1] == "vault"


def test_accented_words_count_toward_coverage(tmp_path):
  # "Écluse" (accented first letter, which SQLite's lower() leaves alone) and "cœur" (outside the old à-ÿ range).
  path = tmp_path / "accents.db"
  store = SqliteVecStore(path, dim=EMB.dim)
  store.add_documents([
    {"id": "1", "content": "Écluse du cœur système : vanne manuelle au niveau deux", "source": "ecluse.md"},
    {"id": "2", "content": "Horaires du bac entre les deux rives", "source": "bac.md"},
  ], EMB, type_="topic")
  store.close()
  picked = prompt_selection("écluse cœur", "s1", db=path)
  assert picked and picked[0][2] == "ecluse.md"


def _store(tmp_path, docs):
  path = tmp_path / "x.db"
  store = SqliteVecStore(path, dim=EMB.dim)
  store.add_documents(docs, EMB, type_="topic")
  store.close()
  return path


def test_at_most_two_extracts_by_default(tmp_path):
  path = _store(tmp_path, [
    {"id": str(i), "content": f"Lifejackets inspection report number {i} by the safety officer",
     "source": f"report{i}.md"} for i in range(5)])
  assert len(prompt_selection("lifejackets inspection safety", "s", db=path, embedder=EMB)) == 2


def test_index_files_old_notes_and_journals_need_both_sides(tmp_path):
  import time
  path = _store(tmp_path, [
    {"id": "1", "content": "Lifejackets inspection links list", "source": "MEMORY.md"},
    {"id": "2", "content": "Lifejackets inspection done last spring", "source": "old.md",
     "mtime": time.time() - 200 * 86400},
    {"id": "3", "content": "Lifejackets inspection noted in the journal", "source": "2026-01-02.md"},
  ])
  # keyword side only (no embedder): none of them qualifies
  assert prompt_selection("lifejackets inspection", "s", db=path) == []


def test_pinned_notes_do_not_age(tmp_path):
  import time
  path = _store(tmp_path, [
    {"id": "1", "content": "Lifejackets inspection must never be skipped before a crossing", "source": "rules.md",
     "mtime": time.time() - 400 * 86400},
    {"id": "2", "content": "Lifejackets inspection noted in the journal", "source": "2025-01-02.md",
     "mtime": time.time() - 400 * 86400},
  ])
  assert prompt_selection("lifejackets inspection", "a", db=path) == []
  picked = prompt_selection("lifejackets inspection", "b", db=path, pinned=frozenset({"rules.md", "2025-01-02.md"}))
  assert [p[2] for p in picked] == ["rules.md"]  # pinning exempts from age, not from the journal rule


def test_short_prompt_takes_its_subject_from_the_previous_reply(db):
  ctx = conversation_context("where is it at?", "The ferry timetable export is running. [SUGGEST: ok]")
  assert "timetable" in ctx and "SUGGEST" not in ctx
  assert conversation_context(" ".join(["word"] * 13), "timetable") == ""
  timing = {}
  prompt_selection("where is the export at", "s", db=db, embedder=EMB, context=ctx, timing=timing)
  assert timing["embed_ms"] >= 0


def _fake(prefix: str, body: str) -> str:
  # Built at runtime so no secret-shaped literal sits in the repository.
  return prefix + body


def _leaks(secret: str, text: str, n: int = 5) -> bool:
  return any(secret[i:i + n] in text for i in range(len(secret) - n + 1))


def test_text_is_masked_before_it_is_cut():
  # A secret cut in two is no longer recognised by the patterns: what is left of it goes through.
  body = "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8"
  token = _fake("ghp_", body)
  head = conversation_context("and that one?", "w" * 290 + " " + token + " " + "word " * 100)
  tail = conversation_context("and that one?", "word " * 100 + token + " " + "w" * 280)
  pw = _fake("Xq7pL2", "wzR9kT4m")
  capped = embed_text("x" * 335 + " password=" + pw)  # the cut leaves 5 characters of the value
  assert not _leaks(body, head) and not _leaks(body, tail) and not _leaks(pw, capped)
  assert len(capped) == EMBED_MAX_CHARS


def test_the_prompt_reaches_the_embedder_masked(db):
  sent = []

  def spy(text):
    sent.append(text)
    return EMB.embed([text])[0]

  token = _fake("ghp_", "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8")
  prompt_selection(f"where are the lifejackets stored, token {token}", "s", db=db, embedder=spy,
                   context=f"previous reply with password={_fake('Xq7pL2', 'wzR9kT4m')}")
  assert sent and token not in sent[0] and "wzR9kT4m" not in sent[0] and "lifejackets" in sent[0]


def test_text_sent_to_the_embedder_is_capped():
  assert embed_text("short prompt") == "short prompt"
  assert embed_text("x" * 1000) == "x" * EMBED_MAX_CHARS
  reply = "start of the reply " + "y" * 900 + " end of the reply"
  text = embed_text("where is it at", reply)
  assert len(text) <= EMBED_MAX_CHARS and text.startswith("where is it at\n\nstart of the reply")
  assert text.endswith("end of the reply") and " ... " in text
  assert embed_text("z" * 330, reply) == "z" * 330  # less than 40 characters left: no context at all


def test_a_long_prompt_reaches_the_embedder_capped(db):
  sent = []

  def spy(text):
    sent.append(text)
    return EMB.embed([text])[0]

  prompt = "where are the lifejackets stored " + "and also " * 200
  prompt_selection(prompt, "s", db=db, embedder=spy, context="previous reply about the ferry")
  assert sent and len(sent[0]) <= EMBED_MAX_CHARS and sent[0].startswith("where are the lifejackets")


def test_a_prompt_without_accents_counts_accented_words(tmp_path):
  # FTS folds diacritics ("ecluse" finds "Écluse"), the coverage count compared exact words: 0 covered words,
  # and the extract the keyword side had found was dropped. Folded on both sides, either way round.
  path = _store(tmp_path, [
    {"id": "1", "content": "Écluse amont : vanne manuelle réglée au niveau deux", "source": "amont.md"},
    {"id": "2", "content": "Ecluse aval : pompe de relevage hors service", "source": "aval.md"},
    {"id": "3", "content": "Horaires du bac entre les deux rives", "source": "bac.md"},
  ])
  picked = prompt_selection("ecluse reglee", "s", db=path)
  assert picked and picked[0][2] == "amont.md"
  picked = prompt_selection("écluse relevage", "s", db=path)
  assert picked and picked[0][2] == "aval.md"


def test_a_missing_database_abstains_even_with_an_identifier(tmp_path):
  # The identifier check sat outside the `try`: a missing database raised OperationalError into the caller.
  missing = tmp_path / "absent.db"
  for embedder in (None, EMB):
    assert prompt_selection("what happened in INC-4471 hydraulic hose", "s", db=missing, embedder=embedder) == []
  assert not missing.exists()  # opened read only: nothing created


def test_a_database_path_with_uri_characters_is_read(tmp_path):
  # f"file:{db}?mode=ro" took `#` for a fragment and `?` for a query: the path was cut there, SQLite opened
  # (read-write, the mode being cut off too) another file, and both sides returned [] in silence.
  docs = [{"id": "1", "content": "Lifejackets are stored under the benches on the main deck", "source": "a.md"},
          {"id": "2", "content": "Ferry timetable: weekday departures at dawn and noon", "source": "b.md"}]
  for sub in ("with#hash", "with?mark", "with%25percent", "with space"):
    (tmp_path / sub).mkdir()
    path = _store(tmp_path / sub, docs)
    assert prompt_selection("where are the lifejackets stored on deck", "s", db=path), sub
    assert prompt_selection("what happened in INC-4471 lifejackets deck", "s", db=path) == [], sub
    assert semantic_rows(path, "lifejackets stored", lambda t: EMB.embed([t])[0], ("topic",), {}), sub
  assert sorted(p.name for p in tmp_path.iterdir()) == sorted(("with#hash", "with?mark", "with%25percent",
                                                                "with space"))


def test_every_connection_is_closed(db, monkeypatch):
  # `with sqlite3.connect(...) as conn` commits or rolls back, it does not close: two connections per call leaked.
  opened = []
  real = sqlite3.connect

  def spy(*args, **kwargs):
    conn = real(*args, **kwargs)
    opened.append(conn)
    return conn

  monkeypatch.setattr(sqlite3, "connect", spy)
  assert prompt_selection("what happened in INC-4471 hydraulic hose", "s", db=db, embedder=EMB)
  assert prompt_selection("ferry timetable weekday departures", "s", db=db)
  assert len(opened) >= 3
  for conn in opened:
    with pytest.raises(sqlite3.ProgrammingError):
      conn.execute("SELECT 1")


def test_the_vector_side_is_the_store_search_and_rrf_is_shared(db):
  # retrieval had its own copy of the k-NN (sqlite_vec, struct.pack, the SQL) and of RRF, with ranks shifted by
  # one against the store's search_hybrid.
  from aura_core.memory import fusion, retrieval, sqlite_vec_store

  source = inspect.getsource(retrieval)
  assert "struct" not in source and "import sqlite_vec\n" not in source and "MATCH ? AND k" not in source
  assert retrieval.rrf_scores is fusion.rrf_scores and sqlite_vec_store.rrf_term is fusion.rrf_term
  assert fusion.rrf_term(1) == 1 / (fusion.RRF_K + 1)
  vec = EMB.embed(["lifejackets stored on deck"])[0]
  store = SqliteVecStore(db, read_only=True)
  try:
    expected = [(h["rowid"], h["type"], h["source"], h["content"][:200])
                for h in store.search(vec, k=6, filter_type=["topic", "note"])]
  finally:
    store.close()
  timing = {}
  assert semantic_rows(db, "lifejackets stored on deck", lambda t: EMB.embed([t])[0], ("topic", "note"),
                       timing) == expected and "embed_fallback" not in timing
  # An Embedder object is accepted too: it used to raise inside, and the vector side was dropped in silence.
  assert semantic_rows(db, "lifejackets stored on deck", EMB, ("topic", "note"), timing) == expected
  assert semantic_rows(db, "lifejackets stored on deck", EMB, (), {}) == []  # no type asked, nothing returned
