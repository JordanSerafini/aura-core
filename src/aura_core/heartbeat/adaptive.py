"""`[adaptive:N]`: a task whose output never changes runs less often.

After ADAPTIVE_STREAK successful runs in a row that repeat the output of the
run before, scheduled runs are skipped so that the task runs at half its
cadence; after the next streak a quarter, and so on, never less than 1/N of
it. The first run whose output differs, and any run that fails, bring it back
to its normal cadence. A run started by hand is never skipped.

"The same output" ignores ISO timestamps and spacing: a line that only
carries the current time is not news. An empty output never counts: a
command that is silent when it succeeds (`rsync -a`) does work at every run.

A change is only seen at the next real run: backed off to 1/16, an hourly task
can stay blind to it for 15 hours. Meant for watchers whose silence is the
normal case (a folder that rarely changes, a page checked for a new release),
not for checks whose failure must be seen within the period.
"""

import hashlib
import re

from aura_core.heartbeat.models import ScheduledTask

ADAPTIVE_STREAK = 20
MIN_ADAPTIVE, MAX_ADAPTIVE = 2, 32
_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}[T ][0-9:.+-]+")


def output_fingerprint(output: str) -> str:
  normalized = " ".join(_TIMESTAMP.sub("<ts>", output).split())
  return hashlib.sha256(normalized.encode("utf-8", errors="replace")).hexdigest()


def multiplier(unchanged: int, cap: int) -> int:
  """How many periods apart runs are after `unchanged` repeats: 1, then 2, 4, 8... at most `cap`."""
  return min(cap, 2 ** (unchanged // ADAPTIVE_STREAK))


def base_period(task: ScheduledTask) -> int:
  if task.schedule_type == "interval":
    return task.interval_seconds
  return 7 * 86400 if task.schedule_type == "weekly" else 86400


def next_state(task: ScheduledTask, previous: tuple | None, ok: bool, output: str,
               finished_at: float) -> tuple[str | None, int, float]:
  """(fingerprint, unchanged, skip_until) after a real run. A failure or an empty output starts over."""
  if not ok or not output.strip():
    return None, 0, 0.0
  fingerprint = output_fingerprint(output)
  unchanged = previous[1] + 1 if previous and previous[0] == fingerprint else 0
  times = multiplier(unchanged, task.adaptive)
  skip_until = finished_at + base_period(task) * (times - 1) if times > 1 else 0.0
  return fingerprint, unchanged, skip_until
