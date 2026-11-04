import pytest

from aura_core.memory.embed_engine import HashingEmbedder
from aura_core.memory.retrieval import EMBED_MAX_CHARS, conversation_context, embed_text, prompt_selection
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
