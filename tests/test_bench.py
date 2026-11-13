import json
import sys
import types

import pytest

from aura_core.eval import bench
from aura_core.memory import reranker
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


def test_empty_ground_truth_is_refused_not_divided_by_zero(data):
  corpus, truth = data
  empty = {**truth, "questions": []}
  assert any("no question" in p for p in bench.validate(corpus, empty))
  with pytest.raises(ValueError):
    bench.run(corpus, empty)


def test_rerank_without_vectors_is_reported(capsys):
  bench.main(["--embedder", "lexical", "--rerank", "--json"])
  assert "--rerank ignored" in capsys.readouterr().err


# -- rerank path: never the real cross-encoder ------------------------------------------------------------------


class _WordOverlapModel:
  """Stands in for the cross-encoder: scores a pair by the words the passage shares with the question."""

  def __init__(self):
    self.calls = 0

  def predict(self, pairs):
    self.calls += 1
    return [float(len(set(q.lower().split()) & set(p.lower().split()))) for q, p in pairs]


def _model_that_fails(loads):
  def load():
    loads.append(1)
    raise OSError("no cached weights for the cross-encoder")
  return load


def test_rerank_is_unavailable_without_the_dependency(data, monkeypatch):
  corpus, truth = data
  loads = []
  monkeypatch.setattr(reranker, "available", lambda: False)
  monkeypatch.setattr(reranker, "_model", _model_that_fails(loads))
  res = bench.run(corpus, truth, HashingEmbedder(256), rerank=True)
  assert set(res["rerank"]) == {"unavailable"} and "not installed" in res["rerank"]["unavailable"]
  assert loads == []  # not even tried
  assert 0.0 <= res["hybrid"]["recall@3"] <= 1.0  # the other paths are still scored


def test_rerank_is_unavailable_when_the_model_does_not_load(data, monkeypatch):
  # The fail-safe reranker returned the hybrid candidates unchanged, and the bench counted them as reranked.
  corpus, truth = data
  monkeypatch.setattr(reranker, "available", lambda: True)
  monkeypatch.setattr(reranker, "_model", _model_that_fails([]))
  res = bench.run(corpus, truth, HashingEmbedder(256), rerank=True)
  assert set(res["rerank"]) == {"unavailable"}
  assert "OSError" in res["rerank"]["unavailable"] and "no cached weights" in res["rerank"]["unavailable"]
  assert {"fts", "semantic", "hybrid"} <= set(res) and all("recall@3" in res[p] for p in ("fts", "semantic", "hybrid"))


def test_rerank_path_is_scored_with_a_working_model(data, monkeypatch):
  corpus, truth = data
  model = _WordOverlapModel()
  monkeypatch.setattr(reranker, "available", lambda: True)
  monkeypatch.setattr(reranker, "_model", lambda: model)
  calls, real = [], reranker.rerank

  def spy(*args, **kwargs):
    out = real(*args, **kwargs)
    calls.append((kwargs, out))
    return out

  monkeypatch.setattr(reranker, "rerank", spy)
  res = bench.run(corpus, truth, HashingEmbedder(256), rerank=True)
  assert "unavailable" not in res["rerank"] and 0.0 <= res["rerank"]["recall@3"] <= 1.0
  assert len(calls) == len(truth["questions"]) == model.calls
  assert all(kwargs.get("strict") is True for kwargs, _ in calls)
  assert all(out and all("rerank_score" in c for c in out) for _, out in calls)


def test_reranker_strict_mode_raises_instead_of_returning_unranked(monkeypatch):
  monkeypatch.setattr(reranker, "_model", _model_that_fails([]))
  cands = [{"content": "a"}, {"content": "b"}, {"content": "c"}]
  assert reranker.rerank("q", cands, top_k=2, strict=False) == cands[:2]
  assert all("rerank_score" not in c for c in cands)
  with pytest.raises(OSError):
    reranker.rerank("q", cands, top_k=2, strict=True)

  class OneScore:  # fewer scores than candidates would leave some of them unranked
    def predict(self, pairs):
      return [1.0]

  monkeypatch.setattr(reranker, "_model", lambda: OneScore())
  with pytest.raises(ValueError):
    reranker.rerank("q", cands, top_k=2, strict=True)


def test_cross_encoder_uses_the_local_cache_only_when_offline(monkeypatch):
  seen = {}

  class FakeCrossEncoder:
    def __init__(self, name, **kwargs):
      seen.update(kwargs, name=name)

  fake = types.ModuleType("sentence_transformers")
  fake.CrossEncoder = FakeCrossEncoder
  monkeypatch.setitem(sys.modules, "sentence_transformers", fake)
  monkeypatch.setenv("AURA_CORE_OFFLINE", "1")
  reranker._model.__wrapped__()  # past the per-process cache
  assert seen["local_files_only"] is True and seen["name"] == reranker.model_name()
  monkeypatch.setenv("AURA_CORE_OFFLINE", "0")
  reranker._model.__wrapped__()
  assert seen["local_files_only"] is False


def test_cli_reports_an_unavailable_rerank_and_exits_1(monkeypatch, capsys):
  monkeypatch.setattr(reranker, "available", lambda: True)
  monkeypatch.setattr(reranker, "_model", _model_that_fails([]))
  assert bench.main(["--embedder", "hashing", "--rerank", "--json"]) == 1
  captured = capsys.readouterr()
  results = json.loads(captured.out)["results"]
  assert set(results["rerank"]) == {"unavailable"} and "recall@3" in results["hybrid"]
  assert "warning" in captured.err and "rerank" in captured.err and "no cached weights" in captured.err
  # Text mode: one line that says so, and no rerank row in the per-kind table or the misses.
  assert bench.main(["--embedder", "hashing", "--rerank", "--misses"]) == 1
  lines = capsys.readouterr().out.splitlines()
  assert [line for line in lines if line.startswith("rerank")] == [
    f"rerank     unavailable: {results['rerank']['unavailable']}"]
  assert bench.main(["--embedder", "hashing"]) == 0  # no --rerank asked, nothing missing


def test_rerank_without_vectors_exits_1(capsys):
  assert bench.main(["--embedder", "lexical", "--rerank", "--json"]) == 1
  assert "--rerank ignored" in capsys.readouterr().err
