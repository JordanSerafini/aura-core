import time

import pytest

from aura_core.heartbeat.adaptive import ADAPTIVE_STREAK, base_period, multiplier, output_fingerprint
from aura_core.heartbeat.parser import parse_text
from aura_core.heartbeat.runner import RunHistory, execute_task


@pytest.fixture
def history(tmp_path):
  return RunHistory(tmp_path / "hb.db")


def test_fingerprint_ignores_timestamps_and_spacing_only():
  assert output_fingerprint("checked at 2026-11-04T18:30:12+01:00: 0 new files\n") == \
         output_fingerprint("checked at 2026-11-05 07:02:59:   0 new files")
  assert output_fingerprint("0 new files") != output_fingerprint("1 new file")
  # A date alone is ignored too, but never the number written after it.
  assert output_fingerprint("report of 2026-11-04") == output_fingerprint("report of 2026-11-05")
  assert output_fingerprint("2026-11-04 3 new files") != output_fingerprint("2026-11-05 47 new files")
  assert output_fingerprint("2026-11-04 12:00 3 new") != output_fingerprint("2026-11-04 13:00 47 new")


def test_multiplier_doubles_every_streak_up_to_the_cap():
  assert [multiplier(n, 8) for n in (0, ADAPTIVE_STREAK - 1, ADAPTIVE_STREAK, 2 * ADAPTIVE_STREAK)] == [1, 1, 2, 4]
  assert multiplier(10 * ADAPTIVE_STREAK, 8) == 8 and multiplier(3 * ADAPTIVE_STREAK, 5) == 5


def test_base_period_per_schedule_type():
  tasks = parse_text("""## Every 15 minutes
- [ ] `echo a` -- [adaptive:4] x
## Daily at 06:00
- [ ] `echo b` -- [adaptive:4] x
## Weekly on sunday at 06:00
- [ ] `echo c` -- [adaptive:4] x
""")
  assert [base_period(t) for t in tasks] == [900, 86400, 7 * 86400]


def test_adaptive_annotation_bounds():
  tasks = parse_text("""## Every 1 hour
- [ ] `echo a` -- [adaptive:2] x
- [ ] `echo b` -- [adaptive:32] x
- [ ] `echo c` -- [adaptive:1] x
- [ ] `echo d` -- [adaptive:33] x
- [ ] `echo e` -- [adaptive:often] x
- [ ] `echo f` -- x
""")
  assert [t.adaptive for t in tasks] == [2, 32, None, None, None, None]
  assert len(tasks.errors) == 3 and all("invalid adaptive" in e for e in tasks.errors)


def _watcher(tmp_path, cap=4):
  state = tmp_path / "state.txt"
  state.write_text("nothing new\n")
  # Prints the file, and fails while a `broken` file sits next to it.
  command = f"sh -c 'cat {state}; test ! -e {tmp_path / 'broken'}'"
  task = parse_text(f"## Every 1 hour\n- [ ] `{command}` -- [local] [adaptive:{cap}] watches a file\n")[0]
  return task, state


def test_a_task_whose_output_never_changes_backs_off(tmp_path, history):
  task, _ = _watcher(tmp_path)
  # The first run sets the reference, the next ADAPTIVE_STREAK repeat it: all of them run.
  statuses = [execute_task(task, history, scheduled=True).status for _ in range(ADAPTIVE_STREAK + 1)]
  assert statuses == ["ok"] * (ADAPTIVE_STREAK + 1)
  _, unchanged, skip_until = history.adaptive_state(task.task_id)
  assert unchanged == ADAPTIVE_STREAK and skip_until == pytest.approx(time.time() + 3600, abs=60)
  skipped = execute_task(task, history, scheduled=True)
  assert skipped.status == "skipped_adaptive" and "x2" in skipped.output_tail
  assert history.adaptive_state(task.task_id)[1] == ADAPTIVE_STREAK  # a skip leaves the streak alone
  assert history.last_status(task.task_id) == "ok"  # a skip is never a dependency status
  assert execute_task(task, history).status == "ok"  # a run started by hand is never skipped


def test_the_first_change_or_failure_restores_the_cadence(tmp_path, history):
  task, state = _watcher(tmp_path)
  for _ in range(ADAPTIVE_STREAK + 1):
    execute_task(task, history, scheduled=True)
  assert execute_task(task, history, scheduled=True).status == "skipped_adaptive"
  state.write_text("one new file\n")
  assert execute_task(task, history).status == "ok"  # the change is seen at the next real run
  assert history.adaptive_state(task.task_id)[1:] == (0, 0.0)
  assert execute_task(task, history, scheduled=True).status == "ok"
  for _ in range(ADAPTIVE_STREAK):
    execute_task(task, history, scheduled=True)
  assert execute_task(task, history, scheduled=True).status == "skipped_adaptive"
  (tmp_path / "broken").write_text("")  # same output as before, but the check now fails
  assert execute_task(task, history).status == "failed"
  assert history.adaptive_state(task.task_id)[1:] == (0, 0.0)
  assert execute_task(task, history, scheduled=True).status == "failed"  # runs again at once


def test_the_back_off_ends_on_its_own(tmp_path, history):
  task, _ = _watcher(tmp_path)
  for _ in range(ADAPTIVE_STREAK + 1):
    execute_task(task, history, scheduled=True)
  fingerprint, unchanged, _ = history.adaptive_state(task.task_id)
  history.save_adaptive(task.task_id, fingerprint, unchanged, time.time() - 1)
  assert execute_task(task, history, scheduled=True).status == "ok"


def test_a_task_that_prints_nothing_never_backs_off(history):
  # `rsync -a`, `find ... -delete`: silent when they succeed, yet they work at every run.
  task = parse_text("## Every 1 hour\n- [ ] `true` -- [local] [adaptive:8] silent worker\n")[0]
  statuses = {execute_task(task, history, scheduled=True).status for _ in range(ADAPTIVE_STREAK + 2)}
  assert statuses == {"ok"} and history.adaptive_state(task.task_id)[1:] == (0, 0.0)


def test_without_the_annotation_nothing_is_tracked(tmp_path, history):
  state = tmp_path / "state.txt"
  state.write_text("same\n")
  task = parse_text(f"## Every 1 hour\n- [ ] `cat {state}` -- [local] x\n")[0]
  statuses = {execute_task(task, history, scheduled=True).status for _ in range(ADAPTIVE_STREAK + 2)}
  assert statuses == {"ok"} and history.adaptive_state(task.task_id) is None
