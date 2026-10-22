import pytest

from aura_core.memory.embed_engine import HashingEmbedder
from aura_core.memory.retrieval import conversation_context, prompt_selection
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
  # "Écluse" (accent initial, lower() de SQLite ne le touche pas) et "cœur" (hors de l'ancienne plage à-ÿ).
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


def test_short_prompt_takes_its_subject_from_the_previous_reply(db):
  ctx = conversation_context("where is it at?", "The ferry timetable export is running. [SUGGEST: ok]")
  assert "timetable" in ctx and "SUGGEST" not in ctx
  assert conversation_context(" ".join(["word"] * 13), "timetable") == ""
  timing = {}
  prompt_selection("where is the export at", "s", db=db, embedder=EMB, context=ctx, timing=timing)
  assert timing["embed_ms"] >= 0
