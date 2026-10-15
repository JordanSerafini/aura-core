"""Starter HEARTBEAT.md files for `aura-core-heartbeat init`.

Every template must parse without warning (tests/test_init.py enforces it) and
contains only harmless commands: the user replaces them with real ones.
"""

TEMPLATES: dict[str, str] = {
  "minimal": """# Heartbeat

Syntax: see `src/aura_core/heartbeat/parser.py`. The file is re-read live
(within 30 seconds), so edit it while the runner is up.

## Every 5 minutes

- [ ] `date +%FT%T` -- [local] [id:clock] [timeout:10] proof that the runner is alive
""",
  "watchdog": """# Heartbeat

Local, zero-token probes. Replace the commands by your own checks.

## Every 5 minutes

- [ ] `date +%FT%T` -- [local] [id:clock] [timeout:10] proof that the runner is alive

## Every 15 minutes

- [ ] `df -h /` -- [local] [id:disk] [timeout:20] root filesystem usage
- [ ] `uptime` -- [local] [id:load] [depends:disk] [timeout:10] load average, only after the disk probe succeeded

## Daily at 08:30

- [ ] `echo "good morning"` -- [local] [id:morning] daily marker
""",
  "nightly": """# Heartbeat

Local probes by day, one LLM-backed task by night. Tasks marked [tokens] are
skipped unless the runner is started with --allow-tokens.

## Every 5 minutes

- [ ] `date +%FT%T` -- [local] [id:clock] [timeout:10] proof that the runner is alive

## Daily at 03:00

- [ ] `echo "replace me with a command that calls your LLM"` -- [tokens] [id:night-review] [timeout:600] nightly review

## Weekly on sunday at 04:00

- [ ] `echo "weekly cleanup"` -- [local] [id:weekly-cleanup] [timeout:120] weekly housekeeping
""",
}
