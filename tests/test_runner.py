import time
from datetime import datetime, timezone

import pytest
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from aura_core.heartbeat import runner as runner_module
from aura_core.heartbeat.cli import main as cli_main
from aura_core.heartbeat.parser import parse_text
from aura_core.heartbeat.runner import (
  HeartbeatRunner,
  RunHistory,
  execute_task,
  in_repeated_hour,
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
  assert "echo cli" in out and "ok" in out
  assert cli_main(["-f", str(tmp_path / "missing.md"), "list"]) == 2
