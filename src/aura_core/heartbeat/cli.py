"""Command line: list, once, run, history."""

import argparse
import sys
from pathlib import Path

from aura_core.heartbeat.parser import parse_heartbeat
from aura_core.heartbeat.runner import HeartbeatRunner, RunHistory, run_once


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(prog="aura-core-heartbeat", description=__doc__)
  parser.add_argument("--file", "-f", default="HEARTBEAT.md", help="task file (default: ./HEARTBEAT.md)")
  parser.add_argument("--db", help="history database (default: $AURA_CORE_HOME/heartbeat.db)")
  parser.add_argument("--allow-tokens", action="store_true", help="also run tasks marked [tokens]")
  sub = parser.add_subparsers(dest="cmd", required=True)
  sub.add_parser("list", help="show the parsed tasks and parse warnings")
  sub.add_parser("once", help="run every task once now, in file order, then exit")
  run = sub.add_parser("run", help="start the scheduler (Ctrl-C to stop)")
  run.add_argument("--run-on-start", action="store_true", help="fire every task once right after start")
  hist = sub.add_parser("history", help="show recent runs")
  hist.add_argument("-n", type=int, default=20)
  args = parser.parse_args(argv)

  if args.cmd == "history":
    for r in RunHistory(args.db).recent(args.n):
      print(f"{r['started_at']}  {r['status']:<18} {r['duration_s']:>7.2f}s  {r['task_id']}")
    return 0

  if not Path(args.file).exists():
    print(f"no such file: {args.file}", file=sys.stderr)
    return 2

  if args.cmd == "list":
    tasks = parse_heartbeat(args.file)
    for t in tasks:
      flags = " ".join(f for f, on in (("local", t.local), ("tokens", t.uses_tokens)) if on)
      dep = f" depends:{t.depends_on}" if t.depends_on else ""
      print(f"{t.task_id:<28} {t.schedule_type:<8} {t.schedule_value:<14} "
            f"timeout={t.timeout}s {flags}{dep}  `{t.command}`")
    for message in tasks.errors:
      print(f"warning: {message}", file=sys.stderr)
    return 0 if not tasks.errors else 1

  if args.cmd == "once":
    tasks = parse_heartbeat(args.file)
    results = run_once(tasks, RunHistory(args.db), allow_tokens=args.allow_tokens)
    for r in results:
      print(f"{r.status:<18} {r.duration_s:>7.2f}s  {r.task_id}")
    return 0 if all(r.status in {"ok", "skipped_tokens"} for r in results) else 1

  HeartbeatRunner(args.file, RunHistory(args.db), allow_tokens=args.allow_tokens,
                  run_on_start=args.run_on_start).run()
  return 0


if __name__ == "__main__":
  sys.exit(main())
