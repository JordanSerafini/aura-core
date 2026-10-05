"""Task model and small helpers for the HEARTBEAT.md format."""

import hashlib
import re
import shlex
from dataclasses import dataclass
from pathlib import Path

DEFAULT_TIMEOUT = 120

_INTERPRETERS = {"bash", "sh", "zsh", "node", "ruby", "perl"}


def annotation_prefix(description: str) -> str:
  """Only the contiguous `[annotations]` BEFORE the prose drive execution.

  `[tokens]` written later in a sentence is prose, not a flag.
  """
  return re.match(r"(?:\[[^\]\r\n]+\]\s*)*", description.lstrip()).group(0)


def command_stem(command: str) -> str:
  """Name of the script or module a command runs, not of its interpreter.

  `python3 /x/backup.py --full` -> `backup`; `python -m pkg.mod` -> `mod`;
  `echo hi` -> `echo`.
  """
  try:
    parts = shlex.split(command)
  except ValueError:
    parts = command.split()
  if not parts:
    return "task"
  runner = Path(parts[0]).stem
  if not (runner.startswith("python") or runner in _INTERPRETERS):
    return runner
  for index, argument in enumerate(parts[1:], start=1):
    if argument in {"-c", "-"}:
      return runner
    if argument == "-m" and index + 1 < len(parts):
      return parts[index + 1].rsplit(".", 1)[-1]
    if not argument.startswith("-"):
      return Path(argument).stem
  return runner


def stable_task_id(command: str) -> str:
  """Task id derived from the command only, so it survives a change of schedule."""
  digest = hashlib.sha1(command.encode("utf-8"), usedforsecurity=False).hexdigest()[:10]
  return f"{command_stem(command)}-{digest}"


@dataclass
class ScheduledTask:
  """A task parsed from HEARTBEAT.md."""
  command: str
  description: str
  schedule_type: str            # "interval" | "daily" | "weekly"
  schedule_value: str           # human-readable, e.g. "5m", "08:30", "friday 17:45"
  uses_tokens: bool = False     # [tokens]: the task consumes LLM tokens
  local: bool = False           # [local]: explicitly zero-token, safe at any time
  interval_seconds: int | None = None
  daily_hour: int | None = None
  daily_minute: int | None = None
  weekly_day: int | None = None  # 0 = Monday
  timeout: int = DEFAULT_TIMEOUT
  depends_on: str | None = None
  task_id: str = ""
