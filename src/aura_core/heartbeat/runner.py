"""Run the tasks declared in a HEARTBEAT.md with APScheduler, keeping history in SQLite.

Deliberately small. Tasks are shell commands; each run is recorded in a
`task_runs` table (status, exit code, duration, output tail). There is no
notification channel and no LLM call anywhere in this module.

Security: the file is trusted configuration. Every task line is executed with
`/bin/sh -c`. Never point this at a HEARTBEAT.md you did not write.

Statuses recorded per run:
  ok                  exit code 0
  failed              non-zero exit code
  timeout             killed after `[timeout:N]` seconds (whole process group)
  skipped_dependency  `[depends:X]` and X has not completed successfully (last run)
  skipped_tokens      task is marked `[tokens]` and tokens were not allowed
"""

import os
import signal
import sqlite3
import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.schedulers.base import BaseScheduler
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from aura_core.config import default_heartbeat_db
from aura_core.heartbeat.models import ScheduledTask, command_stem
from aura_core.heartbeat.parser import ParsedTasks, parse_heartbeat

OUTPUT_TAIL_CHARS = 2000
RELOAD_JOB_ID = "__reload__"


@dataclass
class RunResult:
  task_id: str
  status: str
  exit_code: int | None
  duration_s: float
  output_tail: str


class RunHistory:
  """SQLite history of task runs. One connection per call: safe across scheduler threads."""

  def __init__(self, db_path: Path | str | None = None):
    self.db_path = Path(db_path) if db_path else default_heartbeat_db()
    self.db_path.parent.mkdir(parents=True, exist_ok=True)
    with self._connect() as conn:
      conn.execute("""CREATE TABLE IF NOT EXISTS task_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        task_id TEXT NOT NULL, stem TEXT NOT NULL, command TEXT NOT NULL,
        started_at TEXT NOT NULL, duration_s REAL NOT NULL,
        status TEXT NOT NULL, exit_code INTEGER, output_tail TEXT
      )""")
      conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_task ON task_runs(task_id, id)")

  @contextmanager
  def _connect(self):
    """Open, commit and CLOSE (sqlite3's own context manager does not close)."""
    conn = sqlite3.connect(self.db_path, timeout=10)
    try:
      conn.execute("PRAGMA journal_mode = WAL")
      with conn:
        yield conn
    finally:
      conn.close()

  def record(self, task: ScheduledTask, started_at: datetime, result: RunResult) -> None:
    with self._connect() as conn:
      conn.execute(
        "INSERT INTO task_runs(task_id, stem, command, started_at, duration_s, status, exit_code, output_tail) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (task.task_id, command_stem(task.command), task.command, started_at.isoformat(timespec="seconds"),
         result.duration_s, result.status, result.exit_code, result.output_tail),
      )

  def last_status(self, task_ref: str) -> str | None:
    """Status of the most recent run of the task whose id OR stem is `task_ref`."""
    with self._connect() as conn:
      row = conn.execute(
        "SELECT status FROM task_runs WHERE (task_id = ? OR stem = ?) AND status NOT LIKE 'skipped%' "
        "ORDER BY id DESC LIMIT 1", (task_ref, task_ref)).fetchone()
    return row[0] if row else None

  def recent(self, limit: int = 20) -> list[dict]:
    with self._connect() as conn:
      conn.row_factory = sqlite3.Row
      rows = conn.execute("SELECT * FROM task_runs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


def _kill_group(proc: subprocess.Popen) -> None:
  try:
    os.killpg(proc.pid, signal.SIGKILL)
  except (ProcessLookupError, PermissionError):
    proc.kill()


def execute_task(task: ScheduledTask, history: RunHistory, *, allow_tokens: bool = False) -> RunResult:
  """Run one task now, record it, return the result. Never raises for task failures."""
  started = datetime.now()
  t0 = time.monotonic()

  def finish(status: str, code: int | None, output: str) -> RunResult:
    result = RunResult(task.task_id, status, code, round(time.monotonic() - t0, 3),
                       output[-OUTPUT_TAIL_CHARS:])
    history.record(task, started, result)
    return result

  if task.uses_tokens and not allow_tokens:
    return finish("skipped_tokens", None, "task is marked [tokens]; run with --allow-tokens to include it")
  if task.depends_on and history.last_status(task.depends_on) != "ok":
    return finish("skipped_dependency", None, f"dependency {task.depends_on!r} has not completed successfully")

  proc = subprocess.Popen(
    task.command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    start_new_session=True,  # own process group, so a timeout kills children too
  )
  try:
    output, _ = proc.communicate(timeout=task.timeout)
  except subprocess.TimeoutExpired:
    _kill_group(proc)
    output, _ = proc.communicate()
    return finish("timeout", None, (output or "") + f"\n[killed after {task.timeout}s]")
  return finish("ok" if proc.returncode == 0 else "failed", proc.returncode, output or "")


def run_once(tasks: list[ScheduledTask], history: RunHistory, *, allow_tokens: bool = False) -> list[RunResult]:
  """Run every task once, in file order (so `[depends:...]` can be satisfied by earlier tasks)."""
  return [execute_task(t, history, allow_tokens=allow_tokens) for t in tasks]


def build_trigger(task: ScheduledTask):
  if task.schedule_type == "interval":
    return IntervalTrigger(seconds=task.interval_seconds)
  if task.schedule_type == "daily":
    return CronTrigger(hour=task.daily_hour, minute=task.daily_minute)
  if task.schedule_type == "weekly":
    return CronTrigger(day_of_week=task.weekly_day, hour=task.daily_hour, minute=task.daily_minute)
  raise ValueError(f"unsupported schedule type: {task.schedule_type}")


def sync_jobs(scheduler: BaseScheduler, tasks: list[ScheduledTask], history: RunHistory, *,
              allow_tokens: bool = False, run_on_start: bool = False) -> tuple[int, int]:
  """Make the scheduler's jobs match `tasks`. Returns (jobs_added_or_updated, jobs_removed)."""
  wanted = {t.task_id: t for t in tasks}
  removed = 0
  for job in scheduler.get_jobs():
    if job.id != RELOAD_JOB_ID and job.id not in wanted:
      job.remove()
      removed += 1
  for task in wanted.values():
    extra = {"next_run_time": datetime.now() + timedelta(seconds=1)} if run_on_start else {}
    scheduler.add_job(
      execute_task, trigger=build_trigger(task), args=[task, history], kwargs={"allow_tokens": allow_tokens},
      id=task.task_id, name=task.command, replace_existing=True, **extra,
    )
  return len(wanted), removed


class HeartbeatRunner:
  """Blocking scheduler that follows edits to the HEARTBEAT.md file (checked by mtime)."""

  def __init__(self, heartbeat_file: Path | str, history: RunHistory | None = None, *,
               allow_tokens: bool = False, run_on_start: bool = False, reload_seconds: int = 30,
               max_workers: int = 4):
    self.file = Path(heartbeat_file)
    self.history = history or RunHistory()
    self.allow_tokens = allow_tokens
    self.run_on_start = run_on_start
    self.reload_seconds = reload_seconds
    self.scheduler = BlockingScheduler(
      executors={"default": ThreadPoolExecutor(max_workers)},
      job_defaults={"coalesce": True, "max_instances": 1, "misfire_grace_time": 60},
    )
    self._mtime: float | None = None
    self._lock = threading.Lock()

  def reload(self, *, first: bool = False) -> ParsedTasks | None:
    """Re-read the file if it changed. Returns the parsed tasks when a reload happened."""
    with self._lock:
      try:
        mtime = self.file.stat().st_mtime
      except OSError:
        return None
      if mtime == self._mtime:
        return None
      self._mtime = mtime
      tasks = parse_heartbeat(self.file)
      for message in tasks.errors:
        print(f"[heartbeat] warning: {message}", flush=True)
      # A file that parses to zero tasks because of errors must not wipe working jobs.
      if not tasks and tasks.errors and self.scheduler.get_jobs():
        return tasks
      sync_jobs(self.scheduler, tasks, self.history, allow_tokens=self.allow_tokens,
                run_on_start=self.run_on_start and first)
      print(f"[heartbeat] {len(tasks)} task(s) scheduled from {self.file}", flush=True)
      return tasks

  def run(self) -> None:
    self.reload(first=True)
    self.scheduler.add_job(self.reload, IntervalTrigger(seconds=self.reload_seconds), id=RELOAD_JOB_ID)
    try:
      self.scheduler.start()
    except (KeyboardInterrupt, SystemExit):
      pass
