"""Ambient recall: pick up to 2 short memory extracts relevant to a prompt.

Designed for injecting context into an agent prompt, where a wrong or
tangential extract is worse than none. Hence the abstentions:

- fewer than 2 content-bearing words in the prompt: no subject, return nothing;
- an explicit technical identifier (like `ABC-12345`) that the corpus has never
  seen: return nothing rather than the nearest lookalike;
- a chunk found by the keyword side ONLY must cover at least 2 distinct query
  words (a single shared word is a coincidence, not a memory); chunks found by
  the vector side are exempt;
- each chunk is given at most once per session, and at most one extract per
  source file, so two extracts are two different pieces of information;
- index files (a list of links, like a `MEMORY.md`) are never injected, and
  dated journals or notes older than `old_days` only when BOTH sides found them.
  Files named in `pinned` do not age: a standing rule written months ago ("never
  restart X during a backup") is as valid as on its first day.

Measured on 60 real prompts hand-graded extract by extract: 37 % relevant with
3 extracts; the 3rd was relevant 25 % of the time, notes older than 60 days
10 %, index files 0 %, a chunk found by both sides 71 %. With the rules above:
45 %. Filtering alone gains little (a mediocre extract replaces another one):
the number injected is what matters.

A short prompt ("where is the export at?") rarely carries its subject; the
previous assistant reply usually does. `conversation_context()` builds that
context and `prompt_selection(context=...)` adds it to the vector side only, so
an explicit subject in the prompt still wins through the keyword side. The text
sent to the embedder is capped (`embed_text`): its cost grows with its length.
Both functions mask secrets before they cut, never after: what is left of a
token cut in two has no recognisable shape.

Both rankings (BM25 and the store's vector k-NN, `SqliteVecStore.search`) are
fused by Reciprocal Rank Fusion (`fusion`, the same as `search_hybrid`).
`embedder` is a callable `str -> list[float]` or an `Embedder`; without it,
keyword side only.
Every query runs on read-only connections that are closed before returning,
and an unreadable database (missing, locked, not a store) means abstaining,
never an exception.
"""

import contextlib
import re
import sqlite3
import time
import unicodedata
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from aura_core.memory.fusion import RRF_K, rrf_scores
from aura_core.memory.secret_redact import clean

if TYPE_CHECKING:
  from aura_core.memory.embed_engine import Embedder

STOPWORDS = frozenset("""
avec dans pour sans mais donc alors cette votre notre leur sont etre avoir fait
faire faut tout tous toute toutes plus moins bien aussi comme quand ainsi entre
vers chez sous apres avant encore jamais toujours peut veux veut mettre prends
prend continue continuer please thanks thank merci puis donne moi le la les des
une aux ces son ses mes tes nos vos ton fais
this that with from have what when will would could should about there then
which where while your just into over also than them they their been were does
""".split())

DEFAULT_TYPES = ("topic", "note")
__all__ = ["RRF_K", "conversation_context", "embed_text", "prompt_selection", "semantic_rows"]
MAX_INJECT = 2
INDEX_FILES = frozenset({"MEMORY.md"})
OLD_DAYS = 90
JOURNAL_RE = re.compile(r"\d{4}-\d\d-\d\d(?:-weekly)?\.md$")
CONTEXT_MAX_WORDS = 12
CONTEXT_CHARS = 300
_SUGGEST = re.compile(r"\[(?:SUGGEST|PROPOSE)[^\]]*\]")
_IDENTIFIER = re.compile(r"\b[a-zA-Z]+-\d{4,}\b")
FTS_LIMIT = 6
VECTOR_LIMIT = 6
DB_TIMEOUT = 0.8  # seconds; the recall hook of the original project waits 0.65 s for the whole selection


# Embedding time grows with the length of the text (about 0.8 ms per character on CPU for the model of
# the original project, whose recall hook waits 0.65 s at most). Whole prompts made the vector side miss
# that deadline on 59 % of calls over two days. Capped at 350 characters, a replay of 57 real prompts went
# from 42 misses to 0, and the top 3 of the retrieval benchmark did not change on any of its 91 questions.
EMBED_MAX_CHARS = 350


def embed_text(prompt: str, context: str = "", max_chars: int = EMBED_MAX_CHARS) -> str:
  """Text for the vector side: start of the prompt, then the context (head and tail) in the room left."""
  prompt, context = clean(prompt), clean(context)
  head = prompt[:max_chars]
  room = max_chars - len(head) - 2
  if not context or room < 40:
    return head
  if len(context) > room:
    half = (room - 5) // 2
    context = f"{context[:half]} ... {context[-half:]}"
  return f"{head}\n\n{context}"


def conversation_context(prompt: str, previous_reply: str) -> str:
  """Head and tail of the previous assistant reply when the prompt is short; '' otherwise.

  Hand-graded on the 26 prompts of 12 words or fewer: relevant extracts 16/34 ->
  20/33 on the 18 whose selection changes, prompts with a useful extract 11 -> 15.
  """
  if not previous_reply or len(prompt.split()) > CONTEXT_MAX_WORDS:
    return ""
  text = " ".join(_SUGGEST.sub(" ", clean(previous_reply)).split())
  if not text:
    return ""
  if len(text) <= 2 * CONTEXT_CHARS:
    return text
  return f"{text[:CONTEXT_CHARS]} ... {text[-CONTEXT_CHARS:]}"


# One word pattern for the prompt and for the coverage count: two different ranges, plus SQLite's ASCII-only
# lower(), made "Œuvre"/"Écluse" count as 0 covered words and dropped good extracts (review of 18/10).
_WORD = re.compile(r"[a-zA-ZÀ-ſ0-9_-]{3,}")


def _tokens(prompt: str, limit: int = 8) -> list[str]:
  # Min length 3, not 4: 3-letter acronyms are the most discriminating terms
  # (highest IDF) in a technical corpus and a threshold of 4 would drop them all.
  seen, out = set(), []
  for w in _WORD.findall(prompt.lower()):
    if w in STOPWORDS or w in seen:
      continue
    seen.add(w)
    out.append(w)
    if len(out) >= limit:
      break
  return out


def _fold(text: str) -> str:
  """Lower case without diacritics, as the FTS5 index compares (`remove_diacritics`): "Écluse" == "ecluse"."""
  return "".join(c for c in unicodedata.normalize("NFKD", text.lower()) if not unicodedata.combining(c))


def _as_types(types: str | Iterable[str]) -> tuple[str, ...]:
  """One type given as a string is that type, as in `SqliteVecStore.search(filter_type=...)`, not its letters."""
  return (types,) if isinstance(types, str) else tuple(types)


def _in_clause(types: Iterable[str]) -> tuple[str, list[str]]:
  types = list(types)
  return "(" + ",".join("?" * len(types)) + ")", types


def _connect_ro(db, timeout: float = DB_TIMEOUT) -> sqlite3.Connection:
  """Read-only connection. `as_uri()` escapes `#`, `?` and `%`: in an f-string `file:` URI they cut the path,
  SQLite then opened another file (read-write, the mode being cut off too) and every query found nothing."""
  return sqlite3.connect(Path(db).resolve().as_uri() + "?mode=ro", uri=True, timeout=timeout)


def semantic_rows(db, prompt: str, embedder: "Callable[[str], list[float]] | Embedder | None",
                  types: Iterable[str], timing: dict, limit: int = VECTOR_LIMIT) -> list[tuple]:
  """Top `(rowid, type, source, extract)` by vector similarity (`SqliteVecStore.search`, which filters the
  types BEFORE the k-NN), `[]` when the vector side is unavailable."""
  timing.update(embed_ms=0.0, knn_ms=0.0)
  types = list(_as_types(types))
  if embedder is None or not types or db is None:
    return []
  t0 = time.perf_counter()
  try:
    from aura_core.memory.sqlite_vec_store import SqliteVecStore  # imported late: the keyword side needs none of it

    # An `Embedder` (the store's interface) is accepted too: passed where a callable was expected, it raised
    # here and the vector side was dropped in silence.
    vec = embedder.embed([prompt])[0] if hasattr(embedder, "embed") else embedder(prompt)
    t1 = time.perf_counter()
    timing["embed_ms"] = (t1 - t0) * 1000
    with contextlib.closing(SqliteVecStore(db, read_only=True, timeout=DB_TIMEOUT)) as store:
      hits = store.search(vec, k=limit, filter_type=types)
    timing["knn_ms"] = (time.perf_counter() - t1) * 1000
  except Exception:
    timing["embed_fallback"] = True
    return []
  return [(h["rowid"], h["type"], h["source"], (h["content"] or "")[:200]) for h in hits]


def _unknown_identifier(conn: sqlite3.Connection, prompt: str, marks: str, type_params: list[str]) -> bool:
  """True when the prompt names an identifier (`ABC-12345`) that no chunk of these types contains."""
  for identifier in _IDENTIFIER.findall(prompt):
    found = conn.execute(
      f"SELECT 1 FROM chunks WHERE type IN {marks} AND instr(lower(content), lower(?)) > 0 LIMIT 1",
      (*type_params, identifier),
    ).fetchone()
    if not found:
      return True
  return False


def _keyword_rows(conn: sqlite3.Connection, tokens: list[str], marks: str,
                  type_params: list[str]) -> tuple[list[tuple], dict[int, int]]:
  """BM25 candidates, and how many DISTINCT query words each one really contains.

  Words are compared folded, as the index matches them: a prompt typed without accents ("ecluse") finds
  accented content through FTS, and must count it as covered too, or the extract is dropped.
  """
  query = " OR ".join(f'"{t}"' for t in tokens)
  rows = conn.execute(
    "SELECT c.rowid, c.type, c.source, snippet(chunks_fts, 0, '', '', '...', 22) "
    "FROM chunks_fts JOIN chunks c ON c.rowid = chunks_fts.rowid "
    f"WHERE chunks_fts MATCH ? AND c.type IN {marks} "
    f"ORDER BY bm25(chunks_fts) LIMIT {FTS_LIMIT}",
    (query, *type_params),
  ).fetchall()
  covers: dict[int, int] = {}
  if rows:
    wanted = {_fold(t) for t in tokens}
    ph = ",".join("?" * len(rows))
    for rid, content in conn.execute(f"SELECT rowid, content FROM chunks WHERE rowid IN ({ph})",
                                     [r[0] for r in rows]):
      covers[rid] = len(wanted & set(_WORD.findall(_fold(content or ""))))
  return rows, covers


def _mtimes(conn: sqlite3.Connection, rowids: list[int]) -> dict[int, float]:
  if not rowids:
    return {}
  try:
    ph = ",".join("?" * len(rowids))
    return dict(conn.execute(f"SELECT rowid, mtime FROM chunks WHERE rowid IN ({ph})", rowids))
  except sqlite3.Error:
    return {}


def _session_injected(seen: dict, session_id: str) -> set:
  """Rowids already given in this session. The process may live for weeks: old sessions are forgotten, never
  the current one."""
  key = session_id or "?"
  if len(seen) > 80:
    current = seen.get(key)
    seen.clear()
    if current is not None:
      seen[key] = current
  return seen.setdefault(key, set())


class _Evidence(NamedTuple):
  sem_ids: set
  covers: dict
  mtimes: dict


class _Rules(NamedTuple):
  index_files: frozenset
  old_days: float
  pinned: frozenset


def _qualifies(rowid: int, source: str, ev: _Evidence, rules: _Rules) -> bool:
  """The abstention rules of the module docstring, for one candidate."""
  by_keywords = ev.covers.get(rowid, 0) >= 2
  if rowid not in ev.sem_ids and not by_keywords:
    return False
  name = str(source).rsplit("/", 1)[-1]
  if name in rules.index_files:
    return False
  if rowid in ev.sem_ids and by_keywords:  # found by both sides: no age or journal rule
    return True
  mt = ev.mtimes.get(rowid)
  old = bool(mt) and (time.time() - float(mt)) / 86400 > rules.old_days and name not in rules.pinned
  return not (old or JOURNAL_RE.search(name))


def _pick(ordered: list[tuple], ev: _Evidence, rules: _Rules, injected: set, limit: int) -> list[tuple]:
  picked, seen_sources = [], set()
  for rowid, ctype, source, extract in ordered:
    if rowid in injected or source in seen_sources or not _qualifies(rowid, source, ev, rules):
      continue
    extract = " ".join((extract or "").split())[:170]
    if not extract:
      continue
    seen_sources.add(source)
    injected.add(rowid)
    picked.append((rowid, ctype, source, extract))
    if len(picked) >= limit:
      break
  return picked


def prompt_selection(prompt: str, session_id: str, *, db,
                     embedder: "Callable[[str], list[float]] | Embedder | None" = None,
                     types: Iterable[str] = DEFAULT_TYPES, seen: dict | None = None,
                     timing: dict | None = None, limit: int = MAX_INJECT,
                     context: str = "", index_files: frozenset = INDEX_FILES,
                     old_days: float = OLD_DAYS, pinned: frozenset = frozenset()) -> list[tuple]:
  """Return up to `limit` `(rowid, type, source, extract)` tuples, or `[]` to abstain."""
  timing = timing if timing is not None else {}
  seen = seen if seen is not None else {}
  types = _as_types(types)
  tokens = _tokens(prompt)
  if len(tokens) < 2 or db is None:  # no database to read (None used to open a file named "None"): abstain
    return []
  marks, type_params = _in_clause(types)
  t_fts = time.perf_counter()
  try:
    with contextlib.closing(_connect_ro(db)) as conn:
      if _unknown_identifier(conn, prompt, marks, type_params):
        timing["abstention"] = "unknown_identifier"
        return []
      t_fts = time.perf_counter()
      try:
        fts_rows, covers = _keyword_rows(conn, tokens, marks, type_params)
      finally:
        timing["fts_ms"] = (time.perf_counter() - t_fts) * 1000
      sem_rows = semantic_rows(db, embed_text(prompt, context), embedder, types, timing)
      scores = rrf_scores([r[0] for r in fts_rows], [r[0] for r in sem_rows])
      mtimes = _mtimes(conn, list(scores))
  except sqlite3.Error:
    if "abstention" not in timing:  # an unreadable database is timed like a failed keyword search, as before
      timing.setdefault("fts_ms", (time.perf_counter() - t_fts) * 1000)
    return []
  rows = {r[0]: r for r in sem_rows} | {r[0]: r for r in fts_rows}  # the keyword snippet beats the vector head
  ordered = [rows[rid] for rid, _ in sorted(scores.items(), key=lambda kv: -kv[1])]
  ev = _Evidence({r[0] for r in sem_rows}, covers, mtimes)
  return _pick(ordered, ev, _Rules(index_files, old_days, pinned), _session_injected(seen, session_id), limit)
