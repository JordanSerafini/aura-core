"""HEARTBEAT.md parser.

Format:

    ## Every 5 minutes          (units: m, min, minute(s), h, hour(s), d, day(s))
    ## Daily at 08:30
    ## Weekly on friday at 17:45

    - [ ] `shell command` -- [local] [timeout:30] [depends:other_stem] description

A `##` schedule heading applies to the task lines below it, until the next
heading. Annotations are only honoured when they sit before the prose:
`[local]`, `[tokens]`, `[timeout:N]` (seconds), `[depends:stem_or_id]`,
`[id:custom-id]`. `## On event: name` headings are recognised but not
supported by this runner (a warning is recorded).

The parser never raises on a bad line. It records a message in
`ParsedTasks.errors` and keeps the tasks it could read, so one typo does not
silence the whole file. A line that looks like a task but does not match
(missing ` -- ` separator, for instance) is reported instead of silently dropped.
"""

import re
from pathlib import Path

from aura_core.heartbeat.models import (
  DEFAULT_TIMEOUT,
  ScheduledTask,
  annotation_prefix,
  command_stem,
  stable_task_id,
)

_INTERVAL = re.compile(r"^##\s+Every\s+(\d+)\s+(hour|hours|minute|minutes|min|h|m|d|day|days)\s*$", re.I)
_DAILY = re.compile(r"^##\s+Daily\s+at\s+(\d{1,2}):(\d{2})\s*$", re.I)
_WEEKLY = re.compile(r"^##\s+Weekly\s+on\s+(\w+)\s+at\s+(\d{1,2}):(\d{2})\s*$", re.I)
_EVENT = re.compile(r"^##\s+On\s+event:\s+(\w+)\s*$", re.I)
_TASK = re.compile(r"^-\s+\[[xX\s]*\]\s+`([^`]+)`\s+--\s+(.+)$")
_TASK_CANDIDATE = re.compile(r"^-\s+\[[xX\s]*\]\s+`")
_HEADING = re.compile(r"^#{1,6}(?:\s|$)")
_SCHEDULE_HEADING = re.compile(r"^#{1,6}\s+(Every|Daily|Weekly|On)\b", re.I)
_TIMEOUT = re.compile(r"\[timeout:(\d+)\]")
_DEPENDS = re.compile(r"\[depends:([^\]]+)\]")
_ID = re.compile(r"\[id:([A-Za-z0-9_.-]+)\]")

_UNIT_SECONDS = {"h": 3600, "hour": 3600, "hours": 3600, "m": 60, "min": 60, "minute": 60,
                 "minutes": 60, "d": 86400, "day": 86400, "days": 86400}
_DAYS = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6}


class ParsedTasks(list):
  """A list of ScheduledTask that also carries `.errors` (messages about what was skipped)."""

  def __init__(self) -> None:
    super().__init__()
    self.errors: list[str] = []


def parse_heartbeat(filepath: Path | str) -> ParsedTasks:
  """Parse a HEARTBEAT.md file. A missing file yields an empty list plus an error."""
  path = Path(filepath)
  if not path.exists():
    tasks = ParsedTasks()
    tasks.errors.append(f"{path} not found")
    return tasks
  return parse_text(path.read_text(encoding="utf-8"))


def parse_text(text: str) -> ParsedTasks:
  tasks = ParsedTasks()
  schedule: dict | None = None

  for raw in text.splitlines():
    line = raw.strip()

    if m := _INTERVAL.match(line):
      n, unit = int(m.group(1)), m.group(2).lower()
      if n <= 0:
        tasks.errors.append(f"invalid interval: {line}")
        schedule = None
      else:
        schedule = {"type": "interval", "value": f"{n}{unit[0]}", "interval_seconds": n * _UNIT_SECONDS[unit]}
      continue
    if m := _DAILY.match(line):
      hour, minute = int(m.group(1)), int(m.group(2))
      if hour > 23 or minute > 59:
        tasks.errors.append(f"invalid time: {line}")
        schedule = None
      else:
        schedule = {"type": "daily", "value": f"{hour:02d}:{minute:02d}", "hour": hour, "minute": minute}
      continue
    if m := _WEEKLY.match(line):
      day, hour, minute = m.group(1).lower(), int(m.group(2)), int(m.group(3))
      if day not in _DAYS or hour > 23 or minute > 59:
        tasks.errors.append(f"invalid weekly schedule: {line}")
        schedule = None
      else:
        schedule = {"type": "weekly", "value": f"{day} {hour:02d}:{minute:02d}",
                    "weekday": _DAYS[day], "hour": hour, "minute": minute}
      continue
    if m := _EVENT.match(line):
      tasks.errors.append(f"event schedules are not supported by this runner: {line}")
      schedule = None
      continue
    if _HEADING.match(line):
      schedule = None
      if _SCHEDULE_HEADING.match(line):
        tasks.errors.append(f"unknown schedule heading: {line}")
      continue

    m = _TASK.match(line)
    if not m:
      if _TASK_CANDIDATE.match(line):
        tasks.errors.append(f"task line not parsed (missing ` -- ` separator?): {line[:100]}")
      continue
    command, description = m.group(1), m.group(2).strip()
    if schedule is None:
      tasks.errors.append(f"task without a valid schedule: {command}")
      continue

    flags = annotation_prefix(description)
    timeout = _TIMEOUT.search(flags)
    depends = _DEPENDS.search(flags)
    custom_id = _ID.search(flags)
    task_id = custom_id.group(1) if custom_id else stable_task_id(command)
    depends_on = depends.group(1).strip() if depends else None
    if depends_on in {task_id, command_stem(command)}:
      tasks.errors.append(f"self-dependency refused: {command}")
      continue
    tasks.append(ScheduledTask(
      command=command,
      description=description,
      schedule_type=schedule["type"],
      schedule_value=schedule["value"],
      uses_tokens="[tokens]" in flags,
      local="[local]" in flags,
      interval_seconds=schedule.get("interval_seconds"),
      daily_hour=schedule.get("hour"),
      daily_minute=schedule.get("minute"),
      weekly_day=schedule.get("weekday"),
      timeout=int(timeout.group(1)) if timeout else DEFAULT_TIMEOUT,
      depends_on=depends_on,
      task_id=task_id,
    ))

  # The same command may appear under two schedules: keep ids stable but jobs distinct.
  counts: dict[str, int] = {}
  for task in tasks:
    counts[task.task_id] = counts.get(task.task_id, 0) + 1
    if counts[task.task_id] > 1:
      task.task_id = f"{task.task_id}-{counts[task.task_id]}"
  return tasks
