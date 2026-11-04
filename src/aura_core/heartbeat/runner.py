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
  skipped_repeated_hour  clocks went back and this daily/weekly slot already ran
                      during the first pass of the repeated hour
  skipped_missed      the run would have started later than its `[catchup:N]` window
  skipped_overlap     the previous run of the same task was still going
  skipped_adaptive    `[adaptive:N]`: the output has not changed for a while,
                      the task is running at a fraction of its cadence
"""

import contextlib
import os
import signal
import sqlite3
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from apscheduler.events import EVENT_JOB_MAX_INSTANCES, EVENT_JOB_MISSED
from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.schedulers.base import BaseScheduler
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from aura_core.config import default_heartbeat_db
from aura_core.heartbeat.adaptive import multiplier, next_state
from aura_core.heartbeat.models import ScheduledTask, catch_up_seconds, command_stem
from aura_core.heartbeat.parser import ParsedTasks, parse_heartbeat

OUTPUT_TAIL_CHARS = 2000
KILL_DRAIN_SECONDS = 5
RELOAD_JOB_ID = "__reload__"
REPEATED_HOUR_WINDOW = timedelta(minutes=70)
SKIPPED_RUN_EVENTS = EVENT_JOB_MISSED | EVENT_JOB_MAX_INSTANCES


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
      conn.execute("""CREATE TABLE IF NOT EXISTS task_adaptive (
        task_id TEXT PRIMARY KEY, fingerprint TEXT, unchanged INTEGER NOT NULL, skip_until REAL NOT NULL
      )""")

  @contextlib.contextmanager
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

  def ran_since(self, task_id: str, since: datetime) -> bool:
    """Did a real (not skipped) run of this task start at or after `since` (naive wall-clock time)?"""
    with self._connect() as conn:
      row = conn.execute(
        "SELECT 1 FROM task_runs WHERE task_id = ? AND started_at >= ? AND status NOT LIKE 'skipped%' LIMIT 1",
        (task_id, since.isoformat(timespec="seconds"))).fetchone()
    return row is not None

  def adaptive_state(self, task_id: str) -> tuple[str | None, int, float] | None:
    """(output fingerprint, repeats in a row, epoch time until which scheduled runs are skipped)."""
    with self._connect() as conn:
      row = conn.execute("SELECT fingerprint, unchanged, skip_until FROM task_adaptive WHERE task_id = ?",
                         (task_id,)).fetchone()
    return tuple(row) if row else None

  def save_adaptive(self, task_id: str, fingerprint: str | None, unchanged: int, skip_until: float) -> None:
    with self._connect() as conn:
      conn.execute("INSERT INTO task_adaptive(task_id, fingerprint, unchanged, skip_until) VALUES (?,?,?,?) "
                   "ON CONFLICT(task_id) DO UPDATE SET fingerprint = excluded.fingerprint, "
                   "unchanged = excluded.unchanged, skip_until = excluded.skip_until",
                   (task_id, fingerprint, unchanged, skip_until))

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


def _drain_after_kill(proc: subprocess.Popen) -> str:
  """Output left in the pipe after the group kill, read for KILL_DRAIN_SECONDS at most.

  A process started with `setsid` (a daemon, a detached helper) is not in the
  killed group and keeps the pipe open for as long as it lives. Reading up to
  end of file would block this worker thread with it, and with
  `max_instances=1` every later run of the task would be lost without a trace.
  """
  try:
    output, _ = proc.communicate(timeout=KILL_DRAIN_SECONDS)
    return output or ""
  except subprocess.TimeoutExpired as exc:
    partial = exc.output or b""  # raw bytes, even in text mode
    with contextlib.suppress(subprocess.TimeoutExpired):
      proc.wait(timeout=KILL_DRAIN_SECONDS)
    if proc.stdout:
      proc.stdout.close()
    return partial.decode(errors="replace") if isinstance(partial, bytes) else partial


def in_repeated_hour(now_ts: float | None = None) -> bool:
  """True during the hour that the switch back to winter time makes happen twice.

  APScheduler 3 (checked on 3.11.3) computes cron fire times on the wall clock:
  a task due daily at 02:30 fires at 02:30 summer time, then again at 02:30
  winter time one hour later. Only during that second pass was the UTC offset
  one real hour ago larger than it is now. `is not None` on purpose: after the
  switch the offset is zero in London or Lisbon, and zero is falsy.
  """
  now_ts = time.time() if now_ts is None else now_ts
  before = datetime.fromtimestamp(now_ts - 3600).astimezone().utcoffset()
  current = datetime.fromtimestamp(now_ts).astimezone().utcoffset()
  return before is not None and current is not None and before > current


def execute_task(task: ScheduledTask, history: RunHistory, *, allow_tokens: bool = False,
                 scheduled: bool = False) -> RunResult:
  """Run one task now, record it, return the result. Never raises for task failures.

  `scheduled` is set when the scheduler fires the task. Only then is a daily or
  weekly slot that already ran in the first pass of the repeated hour skipped
  (a backup run twice fails on the archive it has just written). A manual run
  is never skipped for that reason.
  """
  started = datetime.now()
  t0 = time.monotonic()

  def finish(status: str, code: int | None, output: str) -> RunResult:
    result = RunResult(task.task_id, status, code, round(time.monotonic() - t0, 3),
                       output[-OUTPUT_TAIL_CHARS:])
    history.record(task, started, result)
    if task.adaptive and not status.startswith("skipped"):
      history.save_adaptive(task.task_id, *next_state(task, adaptive, status == "ok", output, time.time()))
    return result

  if task.uses_tokens and not allow_tokens:
    return finish("skipped_tokens", None, "task is marked [tokens]; run with --allow-tokens to include it")
  adaptive = history.adaptive_state(task.task_id) if task.adaptive else None
  if scheduled and adaptive and time.time() < adaptive[2]:
    until = datetime.fromtimestamp(adaptive[2]).strftime("%Y-%m-%d %H:%M")
    return finish("skipped_adaptive", None, f"same output {adaptive[1]} times in a row: running every "
                  f"x{multiplier(adaptive[1], task.adaptive)} periods, next real run after {until}")
  if task.depends_on and history.last_status(task.depends_on) != "ok":
    return finish("skipped_dependency", None, f"dependency {task.depends_on!r} has not completed successfully")
  # Run history uses naive local times: during the second pass, the first one looks like it just happened.
  if (scheduled and task.schedule_type in ("daily", "weekly") and in_repeated_hour()
      and history.ran_since(task.task_id, started - REPEATED_HOUR_WINDOW)):
    return finish("skipped_repeated_hour", None, "clocks went back: this slot already ran an hour ago")

  proc = subprocess.Popen(
    task.command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    start_new_session=True,  # own process group, so a timeout kills children too
  )
  try:
    output, _ = proc.communicate(timeout=task.timeout)
  except subprocess.TimeoutExpired:
    _kill_group(proc)
    output = _drain_after_kill(proc)
    return finish("timeout", None, output + f"\n[killed after {task.timeout}s]")
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


def record_skipped_runs(scheduler: BaseScheduler, history: RunHistory):
  """Listener that writes into the history the runs APScheduler drops on its own.

  Without it they leave no trace at all: a run that would start later than its
  catch-up window, and a run that would overlap the previous one, still going
  (`max_instances=1`). Attach it with `scheduler.add_listener(..., SKIPPED_RUN_EVENTS)`.
  """
  def record(event) -> None:
    if not event.code & SKIPPED_RUN_EVENTS:
      return
    job = scheduler.get_job(event.job_id)
    if job is None or not job.args:  # gone since, or the reload job (no task argument)
      return
    task = job.args[0]
    if event.code == EVENT_JOB_MISSED:
      due = event.scheduled_run_time.astimezone().strftime("%Y-%m-%d %H:%M")
      result = RunResult(task.task_id, "skipped_missed", None, 0.0,
                         f"run due at {due} would have started more than {job.misfire_grace_time}s late "
                         "(machine asleep or every worker busy); [catchup:N] widens the window")
    else:
      result = RunResult(task.task_id, "skipped_overlap", None, 0.0, "the previous run is still going")
    history.record(task, datetime.now(), result)
  return record


def sync_jobs(scheduler: BaseScheduler, tasks: list[ScheduledTask], history: RunHistory, *,
              allow_tokens: bool = False, run_on_start: bool = False) -> tuple[int, int]:
  """Make the scheduler's jobs match `tasks`. Returns (jobs_added_or_updated, jobs_removed)."""
  wanted = {t.task_id: t for t in tasks}
  removed = 0
  for job in scheduler.get_jobs():
    if job.id != RELOAD_JOB_ID and job.id not in wanted:
      job.remove()
      removed += 1
  changed = 0
  for task in wanted.values():
    # An unchanged task keeps its job: replacing it restarted its IntervalTrigger, so an hourly task in a file
    # edited every 30 minutes never ran (review of 18/10).
    job = scheduler.get_job(task.task_id)
    if (job is not None and not run_on_start and list(job.args[:1]) == [task]
        and job.kwargs.get("allow_tokens") == allow_tokens):
      continue
    extra = {"next_run_time": datetime.now() + timedelta(seconds=1)} if run_on_start else {}
    scheduler.add_job(
      execute_task, trigger=build_trigger(task), args=[task, history],
      kwargs={"allow_tokens": allow_tokens, "scheduled": True},
      id=task.task_id, name=task.command, replace_existing=True,
      misfire_grace_time=catch_up_seconds(task), **extra,
    )
    changed += 1
  return changed, removed


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
    self.scheduler.add_listener(record_skipped_runs(self.scheduler, self.history), SKIPPED_RUN_EVENTS)
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
