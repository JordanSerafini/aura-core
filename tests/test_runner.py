import time

import pytest
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from aura_core.heartbeat.cli import main as cli_main
from aura_core.heartbeat.parser import parse_text
from aura_core.heartbeat.runner import HeartbeatRunner, RunHistory, execute_task, run_once, sync_jobs


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
  assert sync_jobs(sched, tasks[:1], history) == (1, 2)  # file edited: two tasks removed
  assert len(sched.get_jobs()) == 1
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
