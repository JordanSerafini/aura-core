import shutil
import time
from datetime import datetime, timedelta, timezone

import pytest
from apscheduler.events import (
  EVENT_JOB_MAX_INSTANCES,
  EVENT_JOB_MISSED,
  JobExecutionEvent,
  JobSubmissionEvent,
)
from apscheduler.executors.base import run_job
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from aura_core.heartbeat import runner as runner_module
from aura_core.heartbeat.cli import main as cli_main
from aura_core.heartbeat.parser import parse_text
from aura_core.heartbeat.runner import (
  RELOAD_JOB_ID,
  SKIPPED_RUN_EVENTS,
  HeartbeatRunner,
  RunHistory,
  execute_task,
  in_repeated_hour,
  record_skipped_runs,
  run_once,
  sync_jobs,
)


@pytest.fixture
def history(tmp_path):
  return RunHistory(tmp_path / "hb.db")


def one(text):
  return parse_text("## Every 1 hour\n" + text)[0]


def test_success_and_failure_are_recorded(history):
  ok = execute_task(one("- [ ] `echo hello` -- [local] x"), history)
  assert (ok.status, ok.exit_code, ok.output_tail.strip()) == ("ok", 0, "hello")
  bad = execute_task(one("- [ ] `sh -c 'echo boom >&2; exit 3'` -- x"), history)
  assert (bad.status, bad.exit_code) == ("failed", 3) and "boom" in bad.output_tail
  rows = history.recent()
  assert [r["status"] for r in rows] == ["failed", "ok"]


def test_timeout_kills_the_whole_process_group(history):
  t0 = time.monotonic()
  task = one("- [ ] `sh -c 'sleep 30 & sleep 30'` -- [timeout:1] slow")
  result = execute_task(task, history)
  assert result.status == "timeout" and time.monotonic() - t0 < 10


@pytest.mark.skipif(shutil.which("setsid") is None, reason="needs setsid")
def test_timeout_does_not_wait_for_a_process_that_left_the_group(history, monkeypatch):
  # `setsid` puts the grandchild in a new session: the group kill misses it, and it still holds the
  # output pipe. Reading that pipe to the end would block the worker thread for as long as it lives.
  monkeypatch.setattr(runner_module, "KILL_DRAIN_SECONDS", 0.5)
  t0 = time.monotonic()
  task = one("- [ ] `sh -c 'echo started; setsid sleep 8 & sleep 30'` -- [timeout:1] escapes")
  result = execute_task(task, history)
  assert result.status == "timeout" and time.monotonic() - t0 < 4
  assert "started" in result.output_tail and "[killed after 1s]" in result.output_tail


def test_dependency_gate(history):
  tasks = parse_text("""## Every 1 hour
- [ ] `echo first` -- [id:first] a
- [ ] `echo second` -- [depends:first] b
- [ ] `false` -- [id:broken] c
- [ ] `echo third` -- [depends:broken] d
""")
  statuses = [r.status for r in run_once(tasks, history)]
  assert statuses == ["ok", "ok", "failed", "skipped_dependency"]
  # a dependency that never ran is not satisfied either
  lonely = parse_text("## Every 1 hour\n- [ ] `echo x` -- [depends:ghost] d\n")
  assert run_once(lonely, history)[0].status == "skipped_dependency"


def test_dependency_can_be_referenced_by_stem(history):
  tasks = parse_text("## Every 1 hour\n- [ ] `printf base` -- a\n- [ ] `echo next` -- [depends:printf] b\n")
  assert [r.status for r in run_once(tasks, history)] == ["ok", "ok"]
  # same stem on both sides is refused at parse time as a self-dependency
  assert parse_text("## Every 1 hour\n- [ ] `echo base` -- a\n- [ ] `echo next` -- [depends:echo] b\n").errors


def test_token_tasks_are_skipped_unless_allowed(history):
  task = one("- [ ] `echo llm` -- [tokens] would call a model")
  assert execute_task(task, history).status == "skipped_tokens"
  assert execute_task(task, history, allow_tokens=True).status == "ok"
  # a skipped run must not count as the dependency's last real status
  assert history.last_status(task.task_id) == "ok"


@pytest.fixture
def local_tz(monkeypatch):
  def use(name):
    monkeypatch.setenv("TZ", name)
    time.tzset()
  yield use
  monkeypatch.undo()
  time.tzset()


def _utc(*args) -> float:
  return datetime(*args, tzinfo=timezone.utc).timestamp()


def test_in_repeated_hour_is_only_the_second_pass(local_tz):
  # Paris, 2026-10-25: 03:00 summer time becomes 02:00 winter time at 01:00 UTC.
  local_tz("Europe/Paris")
  assert not in_repeated_hour(_utc(2026, 10, 25, 0, 30))  # 02:30, first pass
  assert in_repeated_hour(_utc(2026, 10, 25, 1, 30))      # 02:30, second pass
  assert not in_repeated_hour(_utc(2026, 10, 25, 2, 0))   # 03:00, back to normal
  assert not in_repeated_hour(_utc(2026, 3, 29, 1, 30))   # spring forward skips an hour, repeats none
  assert not in_repeated_hour(_utc(2026, 7, 1, 12, 0))
  # London lands on UTC+0: a zero offset must still count.
  local_tz("Europe/London")
  assert in_repeated_hour(_utc(2026, 10, 25, 1, 30))
  local_tz("UTC")
  assert not in_repeated_hour(_utc(2026, 10, 25, 1, 30))


def test_daily_slot_is_not_replayed_in_the_repeated_hour(history, monkeypatch):
  daily = parse_text("## Daily at 02:30\n- [ ] `echo backup` -- [local] x\n")[0]
  hourly = one("- [ ] `echo ping` -- [local] y")
  monkeypatch.setattr(runner_module, "in_repeated_hour", lambda now_ts=None: True)
  assert execute_task(daily, history, scheduled=True).status == "ok"  # nothing ran yet: first pass
  assert execute_task(daily, history, scheduled=True).status == "skipped_repeated_hour"
  assert history.last_status(daily.task_id) == "ok"  # the skip is not the dependency status
  assert execute_task(daily, history).status == "ok"  # a manual run is never skipped
  assert [execute_task(hourly, history, scheduled=True).status for _ in range(2)] == ["ok", "ok"]
  monkeypatch.setattr(runner_module, "in_repeated_hour", lambda now_ts=None: False)
  assert execute_task(daily, history, scheduled=True).status == "ok"


def test_sync_jobs_marks_runs_as_scheduled(history):
  sched = BackgroundScheduler()
  sched.start(paused=True)
  sync_jobs(sched, parse_text("## Daily at 02:30\n- [ ] `echo a` -- x\n"), history)
  assert sched.get_jobs()[0].kwargs == {"allow_tokens": False, "scheduled": True}
  sched.shutdown(wait=False)


def test_sync_jobs_builds_the_right_triggers_and_prunes(history):
  sched = BackgroundScheduler()
  sched.start(paused=True)  # jobs added to a never-started scheduler are only "pending" and are not de-duplicated
  tasks = parse_text("""## Every 5 minutes
- [ ] `echo a` -- x
## Daily at 08:30
- [ ] `echo b` -- x
## Weekly on friday at 17:45
- [ ] `echo c` -- x
""")
  assert sync_jobs(sched, tasks, history) == (3, 0)
  jobs = {j.name: j for j in sched.get_jobs()}
  assert isinstance(jobs["echo a"].trigger, IntervalTrigger) and jobs["echo a"].trigger.interval_length == 300
  assert isinstance(jobs["echo b"].trigger, CronTrigger) and isinstance(jobs["echo c"].trigger, CronTrigger)
  assert "day_of_week='4'" in str(jobs["echo c"].trigger)
  assert sync_jobs(sched, tasks[:1], history) == (0, 2)  # file edited: two tasks removed, the kept one untouched
  assert len(sched.get_jobs()) == 1
  sched.shutdown(wait=False)


def test_sync_jobs_keeps_the_schedule_of_unchanged_tasks(history):
  sched = BackgroundScheduler()
  sched.start(paused=True)
  text = "## Every 1 hour\n- [ ] `echo a` -- x\n"
  sync_jobs(sched, parse_text(text), history)
  job = sched.get_jobs()[0]
  first = job.next_run_time
  time.sleep(1.1)
  # The file is edited elsewhere (new task): the hourly job must not be pushed back.
  assert sync_jobs(sched, parse_text(text + "- [ ] `echo b` -- y\n"), history) == (1, 0)
  assert sched.get_job(job.id).next_run_time == first
  # A real change of the task (new interval) does replace it.
  assert sync_jobs(sched, parse_text("## Every 2 hours\n- [ ] `echo a` -- x\n"), history)[0] == 1
  sched.shutdown(wait=False)


def test_a_daily_task_moved_to_another_hour_fires_at_the_new_hour(history):
  # Keeping the stored next run time is right for an interval (its phase), wrong for a cron: a task moved
  # from 23:45 to 22:30 would still fire at 23:45 that evening, in the window it was moved out of.
  sched = BackgroundScheduler()
  sched.start(paused=True)
  sync_jobs(sched, parse_text("## Daily at 23:45\n- [ ] `echo suite` -- x\n"), history)
  assert sync_jobs(sched, parse_text("## Daily at 22:30\n- [ ] `echo suite` -- x\n"), history) == (1, 0)
  job = sched.get_jobs()[0]
  assert (job.next_run_time.hour, job.next_run_time.minute) == (22, 30)
  sched.shutdown(wait=False)


def test_sync_jobs_gives_each_job_its_catch_up_window(history):
  sched = BackgroundScheduler()
  sched.start(paused=True)
  sync_jobs(sched, parse_text("""## Every 5 minutes
- [ ] `echo a` -- x
- [ ] `echo b` -- [catchup:30] y
## Daily at 08:30
- [ ] `echo c` -- z
"""), history)
  assert {j.name: j.misfire_grace_time for j in sched.get_jobs()} == {"echo a": 600, "echo b": 30, "echo c": 10800}
  sched.shutdown(wait=False)


def test_a_run_later_than_its_window_is_recorded_not_lost(history):
  # A run that starts 30 minutes late (machine asleep, every worker busy). APScheduler's own executor code
  # decides: missed with the default window of an interval task, run with [catchup:3600].
  sched = BackgroundScheduler()
  sched.start(paused=True)
  sync_jobs(sched, parse_text("## Every 1 hour\n- [ ] `echo late` -- [local] x\n"
                              "- [ ] `echo patient` -- [local] [catchup:3600] y\n"), history)
  record = record_skipped_runs(sched, history)
  late = datetime.now(timezone.utc) - timedelta(minutes=30)
  for job in sched.get_jobs():
    for event in run_job(job, "default", [late], "test"):
      record(event)
  sched.shutdown(wait=False)
  assert sorted(r["status"] for r in history.recent()) == ["ok", "skipped_missed"]  # an executed run adds no row
  rows = {r["command"]: r for r in history.recent()}
  assert rows["echo late"]["status"] == "skipped_missed" and "[catchup:N]" in rows["echo late"]["output_tail"]
  assert rows["echo patient"]["status"] == "ok"
  assert history.last_status(rows["echo late"]["task_id"]) is None  # a skip is never a dependency status


def test_an_overlapping_run_is_recorded(history):
  sched = BackgroundScheduler()
  sched.start(paused=True)
  tasks = parse_text("## Every 1 minute\n- [ ] `echo slow` -- x\n")
  sync_jobs(sched, tasks, history)
  sched.add_job(lambda: None, IntervalTrigger(seconds=30), id=RELOAD_JOB_ID)
  record = record_skipped_runs(sched, history)
  now = datetime.now(timezone.utc)
  record(JobSubmissionEvent(EVENT_JOB_MAX_INSTANCES, tasks[0].task_id, "default", [now]))
  record(JobSubmissionEvent(EVENT_JOB_MAX_INSTANCES, RELOAD_JOB_ID, "default", [now]))  # not a task
  record(JobExecutionEvent(EVENT_JOB_MISSED, "removed-meanwhile", "default", now))
  sched.shutdown(wait=False)
  assert [r["status"] for r in history.recent()] == ["skipped_overlap"]


def test_the_running_scheduler_records_a_missed_run(tmp_path, history):
  assert any(mask & EVENT_JOB_MISSED for _, mask in HeartbeatRunner(tmp_path / "HB.md", history).scheduler._listeners)
  sched = BackgroundScheduler()
  sched.add_listener(record_skipped_runs(sched, history), SKIPPED_RUN_EVENTS)
  sched.start(paused=True)
  sync_jobs(sched, parse_text("## Every 1 hour\n- [ ] `echo asleep` -- [local] x\n"), history)
  # Woken up 90 minutes after the job was due: the last due run is 30 minutes late, past the 600 s default.
  sched.get_jobs()[0].modify(next_run_time=datetime.now(timezone.utc) - timedelta(minutes=90))
  sched.resume()
  try:
    deadline = time.time() + 10
    while time.time() < deadline and not history.recent():
      time.sleep(0.1)
  finally:
    sched.shutdown(wait=True)
  assert [r["status"] for r in history.recent()] == ["skipped_missed"]


def test_runner_follows_file_edits(tmp_path, history):
  f = tmp_path / "HEARTBEAT.md"
  f.write_text("## Every 1 hour\n- [ ] `echo a` -- x\n")
  runner = HeartbeatRunner(f, history)
  runner.scheduler = BackgroundScheduler()
  runner.scheduler.start(paused=True)
  assert len(runner.reload(first=True)) == 1
  assert runner.reload() is None  # unchanged file: nothing to do
  f.write_text("## Every 1 hour\n- [ ] `echo a` -- x\n- [ ] `echo b` -- y\n")
  import os

  os.utime(f, (time.time() + 5, time.time() + 5))
  assert len(runner.reload()) == 2 and len(runner.scheduler.get_jobs()) == 2
  # a broken edit that parses to nothing must not wipe the working jobs
  f.write_text("## Every banana\n- [ ] `echo a` -- x\n")
  os.utime(f, (time.time() + 10, time.time() + 10))
  runner.reload()
  assert len(runner.scheduler.get_jobs()) == 2
  runner.scheduler.shutdown(wait=False)


def test_scheduler_really_fires_a_job(tmp_path, history):
  """End to end with the real scheduler: run-on-start executes the task and records it."""
  f = tmp_path / "HEARTBEAT.md"
  f.write_text("## Every 1 hour\n- [ ] `echo fired` -- [local] x\n")
  runner = HeartbeatRunner(f, history, run_on_start=True)
  runner.scheduler = BackgroundScheduler()  # same jobs, non-blocking for the test
  runner.reload(first=True)
  runner.scheduler.start()
  try:
    deadline = time.time() + 10
    while time.time() < deadline and not history.recent():
      time.sleep(0.1)
  finally:
    runner.scheduler.shutdown(wait=True)
  rows = history.recent()
  assert rows and rows[0]["status"] == "ok" and "fired" in rows[0]["output_tail"]


def test_cli_list_once_history(tmp_path, capsys):
  f = tmp_path / "HEARTBEAT.md"
  f.write_text("## Every 1 hour\n- [ ] `echo cli` -- [local] x\n")
  db = str(tmp_path / "cli.db")
  assert cli_main(["-f", str(f), "--db", db, "list"]) == 0
  assert cli_main(["-f", str(f), "--db", db, "once"]) == 0
  assert cli_main(["-f", str(f), "--db", db, "history"]) == 0
  out = capsys.readouterr().out
  assert "echo cli" in out and "ok" in out and "catchup=600s" in out
  assert cli_main(["-f", str(tmp_path / "missing.md"), "list"]) == 2
