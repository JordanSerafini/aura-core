"""Retrieval benchmark: recall@k and MRR, per retrieval path.

A question set with KNOWN answers (gold document ids, versioned in
`data/ground_truth.json`) is run against a corpus; each path is scored on
whether the gold document shows up in the top-k.

Paths:
  fts        BM25 only (SQLite FTS5). Needs no model, no download.
  semantic   vector k-NN only.
  hybrid     BM25 + vector fused by RRF (`SqliteVecStore.search_hybrid`).
  rerank     hybrid candidates re-ordered by a cross-encoder. Opt-in
             (`--rerank`), downloads a model. If the cross-encoder cannot
             load or score, the path is reported unavailable, with no
             figures, and the command exits with code 1.

Embedders (`--embedder`):
  lexical    no embedder: only the fts path runs. Always works offline.
  hashing    deterministic hashed bag-of-words. NOT semantic; it only
             exercises the vector code path with no download.
  st         sentence-transformers (`pip install -e ".[st]"` from a checkout),
             downloads weights on first use.
  auto       (default) `st` if installed and loadable, else `lexical`.

Metrics: recall@3 = share of questions whose gold document is in the top 3.
MRR = mean of 1/rank of the first gold document, looking at the top 10
(0 when absent). The bundled corpus is synthetic and tiny: the numbers say
whether the pipeline works, not how it would do on your data. Use
`--corpus` and `--truth` to score your own.
"""

import argparse
import hashlib
import json
import sys
import tempfile
from importlib import resources
from pathlib import Path

from aura_core.memory.embed_engine import HashingEmbedder, SentenceTransformerEmbedder
from aura_core.memory.sqlite_vec_store import SqliteVecStore

TOP_K = 10
RECALL_K = 3  # the "recall@3" label and key are hard-wired in the output
POOL = 20


def load_bundled() -> tuple[dict, dict]:
  data = resources.files("aura_core.eval") / "data"
  return (json.loads((data / "corpus.json").read_text(encoding="utf-8")),
          json.loads((data / "ground_truth.json").read_text(encoding="utf-8")))


def validate(corpus: dict, truth: dict) -> list[str]:
  """Return problems in the ground truth (an unknown gold id makes a question unanswerable)."""
  ids = {d["id"] for d in corpus["documents"]}
  problems = []
  if not truth.get("questions"):
    problems.append("ground truth has no question: nothing to score")  # score() would divide by zero
  for q in truth.get("questions", []):
    if not q["relevant"]:
      problems.append(f"{q['id']}: no relevant document")
    problems += [f"{q['id']}: unknown document {r}" for r in q["relevant"] if r not in ids]
  return problems


def first_gold_rank(ranked_ids: list[str], gold: list[str]) -> int | None:
  for i, doc_id in enumerate(ranked_ids[:TOP_K], 1):
    if doc_id in gold:
      return i
  return None


def score(ranks: list[int | None]) -> dict:
  n = len(ranks)
  return {
    f"recall@{RECALL_K}": sum(1 for r in ranks if r is not None and r <= RECALL_K) / n,
    "mrr": sum(1.0 / r for r in ranks if r) / n,
    "found": sum(1 for r in ranks if r is not None),
  }


def pick_embedder(kind: str):
  """Return (embedder_or_None, label). `auto` falls back to lexical with a notice."""
  if kind == "lexical":
    return None, "lexical"
  if kind == "hashing":
    return HashingEmbedder(), "hashing"
  embedder = SentenceTransformerEmbedder()
  try:
    embedder.dim  # noqa: B018 - forces the model to load
  except Exception as exc:
    if kind == "st":
      raise
    print(f"[bench] sentence-transformers unavailable ({type(exc).__name__}); running lexical only",
          file=sys.stderr)
    return None, "lexical"
  return embedder, f"st:{embedder.model_name}"


class _RerankFailed(Exception):
  """The cross-encoder did not load or score: the rerank path gets no figures."""


def _short_reason(exc: BaseException) -> str:
  lines = str(exc).strip().splitlines()
  return f"{type(exc).__name__}: {lines[0][:200]}" if lines else type(exc).__name__


def _score_path(fn, questions: list[dict], qvecs: list) -> dict:
  ranks = [first_gold_rank([r["id"] for r in fn(q["question"], v)], q["relevant"])
           for q, v in zip(questions, qvecs, strict=True)]
  per_kind = {}
  for kind in sorted({q["kind"] for q in questions}):
    sub = [r for r, q in zip(ranks, questions, strict=True) if q["kind"] == kind]
    per_kind[kind] = {"n": len(sub), **score(sub)}
  return {**score(ranks), "per_kind": per_kind, "ranks": {q["id"]: r for q, r in zip(questions, ranks, strict=True)}}


def _score_rerank(store: SqliteVecStore, questions: list[dict], qvecs: list) -> dict:
  """The rerank path, or {"unavailable": reason}.

  The reranker is called with `strict=True`: its fail-safe answer, the hybrid
  candidates unchanged, would otherwise be scored as a reranked list.
  """
  from aura_core.memory import reranker

  if not reranker.available():
    return {"unavailable": "sentence-transformers is not installed (pip install -e \".[st]\")"}

  def fn(q, v):
    candidates = store.search_hybrid(v, q, k=POOL, pool=POOL)
    try:
      return reranker.rerank(q, candidates, top_k=TOP_K, strict=True)
    except Exception as exc:
      raise _RerankFailed(f"{reranker.model_name()} did not load or score ({_short_reason(exc)})") from exc

  try:
    return _score_path(fn, questions, qvecs)
  except _RerankFailed as exc:
    return {"unavailable": str(exc)}


def run(corpus: dict, truth: dict, embedder=None, *, rerank: bool = False) -> dict:
  """Score every available path. Returns {path: {metrics, per_kind, ranks}}.

  The rerank path needs an embedder. When the cross-encoder cannot load or
  score, it is {"rerank": {"unavailable": reason}}, never figures.
  """
  problems = validate(corpus, truth)
  if problems:
    raise ValueError("invalid ground truth: " + "; ".join(problems))
  questions = truth["questions"]
  docs = [{"id": d["id"], "content": d["text"]} for d in corpus["documents"]]

  with tempfile.TemporaryDirectory() as tmp:
    store = SqliteVecStore(Path(tmp) / "bench.db", dim=embedder.dim if embedder else None)
    try:
      store.add_documents(docs, embedder, type_="topic", source_prefix="synthetic")
      qvecs = embedder.embed([q["question"] for q in questions]) if embedder else [None] * len(questions)
      paths = {"fts": lambda q, v: store.search_fts(q, k=TOP_K)}
      if embedder:
        paths["semantic"] = lambda q, v: store.search(v, k=TOP_K)
        paths["hybrid"] = lambda q, v: store.search_hybrid(v, q, k=TOP_K, pool=POOL)
      out = {name: _score_path(fn, questions, qvecs) for name, fn in paths.items()}
      if embedder and rerank:
        out["rerank"] = _score_rerank(store, questions, qvecs)
      return out
    finally:
      store.close()


def main(argv: list[str] | None = None) -> int:
  ap = argparse.ArgumentParser(prog="aura-core-eval", description=__doc__.split("\n\n")[0])
  ap.add_argument("--embedder", choices=["auto", "lexical", "hashing", "st"], default="auto")
  ap.add_argument("--rerank", action="store_true",
                  help="also score the cross-encoder path (downloads a model; exit code 1 if it cannot be scored)")
  ap.add_argument("--corpus", help="corpus JSON (default: bundled synthetic corpus)")
  ap.add_argument("--truth", help="ground truth JSON (default: bundled)")
  ap.add_argument("--json", action="store_true", help="machine-readable output")
  ap.add_argument("--misses", action="store_true", help="list questions each path fails at recall@3")
  args = ap.parse_args(argv)

  corpus, truth = load_bundled()
  if args.corpus:
    corpus = json.loads(Path(args.corpus).read_text(encoding="utf-8"))
  if args.truth:
    truth = json.loads(Path(args.truth).read_text(encoding="utf-8"))
  embedder, label = pick_embedder(args.embedder)
  if args.rerank and embedder is None:
    # The cross-encoder reranks the hybrid pool, which needs vectors: say so instead of dropping the path silently.
    print(f"warning: --rerank ignored, the {label} embedder has no vectors (use --embedder hashing or st)",
          file=sys.stderr)
  results = run(corpus, truth, embedder, rerank=args.rerank)
  for name, r in results.items():
    if "unavailable" in r:
      print(f"warning: no figures for the {name} path, unavailable: {r['unavailable']}", file=sys.stderr)

  digest = hashlib.sha256(json.dumps(truth, sort_keys=True).encode()).hexdigest()[:10]
  header = {"embedder": label, "ground_truth_version": truth.get("version"), "ground_truth_sha": digest,
            "documents": len(corpus["documents"]), "questions": len(truth["questions"])}
  if args.json:
    print(json.dumps({**header, "results": results}, indent=2))
  else:
    _print_text(header, results, truth, misses=args.misses)
  # Asked for and not produced is a failure, so a script or CI run does not read the missing line as a pass.
  reranked = "rerank" in results and "unavailable" not in results["rerank"]
  return 1 if args.rerank and not reranked else 0


def _print_text(header: dict, results: dict, truth: dict, *, misses: bool) -> None:
  print(f"corpus: {header['documents']} docs | ground truth v{header['ground_truth_version']} "
        f"({header['questions']} questions, sha {header['ground_truth_sha']}) | embedder: {header['embedder']}")
  print(f"{'path':<10} {'recall@3':>9} {'MRR':>7} {'found@10':>9}")
  for name, r in results.items():
    if "unavailable" in r:
      print(f"{name:<10} unavailable: {r['unavailable']}")
    else:
      print(f"{name:<10} {r['recall@3']:>9.3f} {r['mrr']:>7.3f} {r['found']:>6}/{header['questions']}")
  scored = {name: r for name, r in results.items() if "unavailable" not in r}  # fts always is
  kinds = next(iter(scored.values()))["per_kind"]
  print("\nrecall@3 by question kind:")
  recall_key = f"recall@{RECALL_K}"
  titles = [f"{k} (n={v['n']})" for k, v in kinds.items()]
  print(f"{'path':<10} " + " ".join(f"{t:>18}" for t in titles))
  for name, r in scored.items():
    print(f"{name:<10} " + " ".join(f"{r['per_kind'][k][recall_key]:>18.3f}" for k in kinds))
  if misses:
    by_id = {q["id"]: q for q in truth["questions"]}
    for name, r in scored.items():
      missed = [qid for qid, rank in r["ranks"].items() if rank is None or rank > RECALL_K]
      print(f"\n{name}: {len(missed)} question(s) outside the top {RECALL_K}")
      for qid in missed:
        print(f"  {qid} [{by_id[qid]['kind']}] rank={r['ranks'][qid]}  {by_id[qid]['question']}")


if __name__ == "__main__":
  sys.exit(main())
