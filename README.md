# aura-core

Three small, self-contained pieces extracted from AURA, my personal AI orchestrator (a few hundred Python files on one Linux box, private):

- **Hybrid memory store**: SQLite + [sqlite-vec](https://github.com/asg017/sqlite-vec) for vectors, FTS5 for BM25, fused with Reciprocal Rank Fusion. Optional cross-encoder reranker.
- **Retrieval benchmark**: recall@3 and MRR per retrieval path (BM25 only, vector only, hybrid, reranked), against a versioned ground truth.
- **Markdown-driven scheduler**: tasks live in a plain `HEARTBEAT.md`, parsed into APScheduler jobs, with run history in SQLite.

This repo is a clean extraction, not the whole system. It has no history from the original project, no personal data and no network calls except an optional model download.

## Why it exists

The benchmark is the part I care about most. In the original project it was used to test retrieval ideas that looked obvious on paper. Several did not beat the existing setup once measured, and were dropped. The lesson I kept: write the questions and the answers down first, then change the retrieval.

Figures from the private instance (not reproducible from this repo): about 9.4k indexed chunks, hybrid BM25 + vector retrieval, 65 recurring tasks scheduled from one markdown file.

## Layout

```
src/aura_core/
  memory/      sqlite_vec_store.py   store: chunks, FTS5, vec0, RRF fusion
               embed_engine.py       embedders (sentence-transformers, hashing)
               reranker.py           optional cross-encoder
               retrieval.py          ambient recall: 2-3 short extracts for a prompt, abstains when unsure
               secret_redact.py      masks credential-looking values before indexing
  eval/        bench.py              the benchmark
               data/                 synthetic corpus (30 docs) + ground truth (23 questions)
  heartbeat/   parser.py, models.py  HEARTBEAT.md grammar
               runner.py, cli.py     APScheduler runner + history
examples/HEARTBEAT.md                harmless sample task file
tests/                               65 tests
```

## Install

```
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"          # core + pytest
pip install -e ".[st]"           # optional: sentence-transformers (downloads a model)
pytest
```

State (databases) goes to `$AURA_CORE_HOME`, default `~/.aura_core`.

## Memory store

```python
from aura_core.memory.embed_engine import HashingEmbedder
from aura_core.memory.sqlite_vec_store import SqliteVecStore

emb = HashingEmbedder()                      # not semantic, see "Limits"
store = SqliteVecStore("demo.db", dim=256)
store.add_documents(
  [{"id": "a", "content": "Backups run every Sunday at 02:00 and are encrypted."},
   {"id": "b", "content": "The scheduler reads its tasks from a markdown file."}],
  embedder=emb)

q = "when do backups run"
for hit in store.search_hybrid(emb.embed([q])[0], q, k=2):
  print(hit["id"], hit["rrf_score"])
```

With `query_embedding=None`, `search_hybrid` degrades to BM25 only.

## Benchmark

```
aura-core-eval --embedder lexical      # BM25 only, offline, no model
aura-core-eval --embedder hashing      # also exercises the vector path, offline
aura-core-eval --embedder st --rerank  # real embeddings + cross-encoder (downloads models)
aura-core-eval --corpus my_corpus.json --truth my_truth.json --misses
```

Result of `--embedder hashing` on the bundled synthetic data (what I ran here):

```
path        recall@3     MRR  found@10
fts            0.739   0.729     19/23
semantic       0.391   0.408     18/23
hybrid         0.652   0.531     19/23
```

Read this carefully: the hashing embedder is a bag of hashed words, not a language model, so the vector path is weak by construction and hybrid scores below BM25 here. The run proves the pipeline works end to end. It says nothing about how a real embedding model would do. I did not run the `st` and `--rerank` paths for this README, so there is no number for them. The bundled corpus is tiny and synthetic; score your own documents with `--corpus` and `--truth`.

## Scheduler

```
aura-core-heartbeat --file examples/HEARTBEAT.md list      # parsed tasks and warnings
aura-core-heartbeat --file examples/HEARTBEAT.md once      # run everything once
aura-core-heartbeat --file examples/HEARTBEAT.md run       # start the scheduler
aura-core-heartbeat history
```

Grammar (`## Every 5 minutes`, `## Daily at 08:30`, `## Weekly on friday at 17:45`, then `- [ ] \`command\` -- [local] [timeout:30] [depends:id] description`) is documented at the top of `heartbeat/parser.py`. A bad line is reported, never fatal: the rest of the file keeps running. Tasks marked `[tokens]` are skipped unless `--allow-tokens` is given. The runner executes shell commands only. There is no LLM call, no notification channel.

## Limits

- Tested on Python 3.12 only (declared `>=3.10`). SQLite built with FTS5 and extension loading (sqlite-vec requirement).
- The hashing embedder is for tests and offline runs only.
- Single process, single writer. No auth, not meant to be exposed on a network.
- `secret_redact` is pattern based. It reduces the chance of indexing a credential, it does not guarantee it.
- Extracted from a larger personal project: some design choices only make sense in that context.

## Beyond this repo

Four other parts of the private system are documented as short case studies in [`docs/`](docs/README.md), with diagrams, design decisions, measured figures and failures:

- [Proactive signals](docs/proactive.md): suggestions and bubbles that speak first, and how they learn to stop.
- [The nightly loop](docs/night-loop.md): autonomous overnight sessions with a review step that does not use a model.
- [The notes vault](docs/vault.md): Markdown notes that scripts and sessions keep current.
- [Phone app, watch app and desktop mascot](docs/mobile-and-mascot.md): three clients on one server.

The code for these parts stays private. Nothing in `docs/` can be run from this repository, and every figure there is from the origin private instance with its measurement date.

## License

MIT, see `LICENSE`.
