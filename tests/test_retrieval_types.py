"""A single type given as a string means that type, as it does for `SqliteVecStore.search(filter_type=...)`."""
import pytest

from aura_core.memory.embed_engine import HashingEmbedder
from aura_core.memory.retrieval import prompt_selection, semantic_rows
from aura_core.memory.sqlite_vec_store import SqliteVecStore

EMB = HashingEmbedder(dim=64)
PROMPT = "where are the lifejackets stored on deck"


@pytest.fixture
def db(tmp_path):
  path = tmp_path / "t.db"
  store = SqliteVecStore(path, dim=EMB.dim)
  store.add_documents([
    {"id": "1", "content": "Lifejackets are stored under the benches on the main deck", "source": "safety.md"},
    {"id": "2", "content": "Lifejackets inspection happens every six months by the safety officer",
     "source": "safety.md"},
  ], EMB, type_="topic")
  store.close()
  return path


def test_prompt_selection_string_type_matches_tuple(db):
  as_tuple = prompt_selection(PROMPT, "s1", db=db, embedder=EMB, types=("topic",))
  as_str = prompt_selection(PROMPT, "s2", db=db, embedder=EMB, types="topic")
  assert as_tuple
  assert as_str == as_tuple


def test_semantic_rows_string_type_matches_tuple(db):
  embed = lambda t: EMB.embed([t])[0]  # noqa: E731
  as_tuple = semantic_rows(db, PROMPT, embed, ("topic",), {})
  as_str = semantic_rows(db, PROMPT, embed, "topic", {})
  assert as_tuple
  assert as_str == as_tuple
