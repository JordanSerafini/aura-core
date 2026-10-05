import json

import pytest

from aura_core.eval import bench
from aura_core.memory.embed_engine import HashingEmbedder


@pytest.fixture(scope="module")
def data():
  return bench.load_bundled()


def test_bundled_ground_truth_is_consistent(data):
  corpus, truth = data
  assert bench.validate(corpus, truth) == []
  assert len(corpus["documents"]) >= 30 and len(truth["questions"]) >= 20
  assert len({d["id"] for d in corpus["documents"]}) == len(corpus["documents"])
  assert len({q["id"] for q in truth["questions"]}) == len(truth["questions"])
  assert truth["version"] >= 1


def test_validation_catches_a_bad_gold_id(data):
  corpus, truth = data
  bad = {**truth, "questions": truth["questions"] + [{"id": "qx", "question": "?", "relevant": ["nope"], "kind": "k"}]}
  assert any("unknown document" in p for p in bench.validate(corpus, bad))
  with pytest.raises(ValueError):
    bench.run(corpus, bad)


def test_metric_arithmetic():
  s = bench.score([1, 2, None, 5])
  assert s["recall@3"] == 0.5 and s["found"] == 3
  assert abs(s["mrr"] - (1 + 0.5 + 0 + 0.2) / 4) < 1e-9
  assert bench.first_gold_rank(["a", "b", "c"], ["c"]) == 3
  assert bench.first_gold_rank(["a"], ["z"]) is None


def test_lexical_mode_runs_end_to_end(data):
  corpus, truth = data
  res = bench.run(corpus, truth, None)
  assert list(res) == ["fts"]
  # Regression floor, a little under the value measured when the set was written.
  assert res["fts"]["recall@3"] >= 0.70 and res["fts"]["mrr"] >= 0.65
  # BM25 should nail questions that share words with the answer.
  assert res["fts"]["per_kind"]["keyword"]["recall@3"] == 1.0
  assert res["fts"]["per_kind"]["identifier"]["recall@3"] == 1.0


def test_hashing_embedder_exercises_all_vector_paths(data):
  corpus, truth = data
  res = bench.run(corpus, truth, HashingEmbedder(256))
  assert set(res) == {"fts", "semantic", "hybrid"}
  for r in res.values():
    assert 0.0 <= r["recall@3"] <= 1.0 and 0.0 <= r["mrr"] <= 1.0
  # Fusion should not lose the exact-match questions the lexical side gets right.
  assert res["hybrid"]["per_kind"]["identifier"]["recall@3"] == 1.0


def test_cli_json_output(capsys):
  assert bench.main(["--embedder", "lexical", "--json"]) == 0
  payload = json.loads(capsys.readouterr().out)
  assert payload["embedder"] == "lexical" and "fts" in payload["results"]
  assert payload["questions"] == len(bench.load_bundled()[1]["questions"])
