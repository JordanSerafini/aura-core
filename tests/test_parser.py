from aura_core.heartbeat.models import annotation_prefix, command_stem, stable_task_id
from aura_core.heartbeat.parser import parse_heartbeat, parse_text

SAMPLE = """
# Title

## Every 5 minutes
- [ ] `echo a` -- [local] [timeout:30] first
- [x] `python3 /opt/jobs/backup.py --full` -- [tokens] [depends:echo] second

## Every 2 hours
- [ ] `echo hourly` -- hourly task

## Daily at 08:05
- [ ] `echo daily` -- [id:my-daily] daily task

## Weekly on Friday at 17:45
- [ ] `echo weekly` -- weekly task
"""


def test_schedules_and_annotations():
  tasks = parse_text(SAMPLE)
  assert not tasks.errors
  by = {t.command: t for t in tasks}
  a = by["echo a"]
  assert (a.schedule_type, a.interval_seconds, a.timeout, a.local, a.uses_tokens) == ("interval", 300, 30, True, False)
  b = by["python3 /opt/jobs/backup.py --full"]
  assert b.uses_tokens and b.depends_on == "echo" and b.timeout == 120
  assert by["echo hourly"].interval_seconds == 7200
  d = by["echo daily"]
  assert (d.schedule_type, d.daily_hour, d.daily_minute, d.task_id) == ("daily", 8, 5, "my-daily")
  w = by["echo weekly"]
  assert (w.schedule_type, w.weekly_day, w.daily_hour, w.daily_minute) == ("weekly", 4, 17, 45)


def test_annotations_only_count_before_the_prose():
  assert annotation_prefix("[local] [timeout:5] hello [tokens]") == "[local] [timeout:5] "
  task = parse_text("## Every 1 hour\n- [ ] `echo x` -- runs and mentions [tokens] in prose\n")[0]
  assert not task.uses_tokens


def test_bad_input_is_reported_not_raised():
  text = """
## Every 0 minutes
- [ ] `echo zero` -- never scheduled

## Daily at 25:00
- [ ] `echo late` -- bad hour

## Every banana
- [ ] `echo fruit` -- unknown heading

## On event: deploy
- [ ] `echo event` -- unsupported here

## Every 1 hour
- [ ] `echo no-separator` missing the double dash
- [ ] `echo ok` -- fine
- [ ] `echo self` -- [depends:echo] depends on itself
"""
  tasks = parse_text(text)
  assert [t.command for t in tasks] == ["echo ok"]
  joined = "\n".join(tasks.errors)
  for needle in ("invalid interval", "invalid time", "unknown schedule heading",
                 "event schedules", "missing ` -- `", "self-dependency"):
    assert needle in joined, needle


def test_task_before_any_heading_and_after_a_plain_heading():
  tasks = parse_text("- [ ] `echo orphan` -- nope\n## Every 1 hour\n## Notes\n- [ ] `echo x` -- nope\n")
  assert len(tasks) == 0 and len(tasks.errors) == 2


def test_disabled_task_syntax_is_silent():
  tasks = parse_text("## Every 1 hour\n- [ ] ~~`echo off`~~ -- disabled on purpose\n")
  assert len(tasks) == 0 and not tasks.errors


def test_duplicate_commands_get_distinct_ids():
  tasks = parse_text("## Every 1 hour\n- [ ] `echo a` -- one\n## Daily at 01:00\n- [ ] `echo a` -- two\n")
  assert len({t.task_id for t in tasks}) == 2


def test_missing_file(tmp_path):
  tasks = parse_heartbeat(tmp_path / "nope.md")
  assert len(tasks) == 0 and tasks.errors


def test_command_stem_and_ids_are_stable():
  assert command_stem("python3 /x/backup.py --full") == "backup"
  assert command_stem("python -m pkg.mod --x") == "mod"
  assert command_stem("bash -c 'echo hi'") == "bash"
  assert command_stem("echo hi") == "echo"
  assert command_stem("") == "task"
  assert stable_task_id("echo hi") == stable_task_id("echo hi") != stable_task_id("echo ho")


def test_bundled_example_parses_cleanly():
  from pathlib import Path

  example = Path(__file__).resolve().parent.parent / "examples" / "HEARTBEAT.md"
  tasks = parse_heartbeat(example)
  assert not tasks.errors and len(tasks) == 6
