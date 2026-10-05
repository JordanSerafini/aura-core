# The nightly loop: autonomous work with mandatory review

Case study. The code described here is not in this repository; it stays in the private instance.

Convention: every figure below comes from the origin private instance and was measured on 2026-10-05, over the 30 days from 2026-09-05 to 2026-10-05, unless another date is given. Costs are the per-session amounts reported by the model CLI at list-price equivalent. They are not an invoice: the instance runs on a flat plan.

## What it does

Between 00:00 and 04:45 a loop launches coding-agent sessions ("missions") on the owner's own machine. Most missions are reactive (an agent is failing, a log shows an error, a backlog task is waiting). A few are background work (an audit of one project per night, idea generation, extraction of reusable procedures). Code changes are made in an isolated git worktree on a throwaway branch, never in the working copy.

In the morning, a zero-token step decides whether each branch is merged. Nothing is merged by the agent that wrote it.

I am deliberately not calling this self-improving. The measures below show an autonomous loop with a review gate and a scorer that reorders its own priorities. They do not show that the system gets better over time, and for almost half of the missions the effect cannot be measured at all.

## Shape

```
 00:00                                                         04:45   05:00         05:10
   |  loop, one iteration at a time (5 min between active iterations)    |             |
   v                                                                     v             v
 local phases (no tokens)           missions (model sessions)        report      merge tick   scorer
 - memory consolidation             - backlog task  -> worktree
 - log scan, cleanup                - fix failing agents
 - agent dormancy                   - targeted audit, idea generation
        |                           - code review, procedure extraction
        |                                   |
        |                                   v
        |                    branch night/<date>-<task>  (commits, never merged by the writer)
        |                                   |
        |                                   v
        |        gate (zero tokens): 1. working tree clean on the files the branch touches
        |                            2. HEAD is the base branch
        |                            3. every changed .py file parses (ast)
        |                            4. dry-run merge has no conflict
        |                                   |
        |                       pass -> merge --no-ff + push if base exists on the remote
        |                       fail -> stays for manual review; SLA: reminder at 48 h, closed at 7 days
        v
 05:10: mission scorer credits only effects observed by someone else (commits, closed tasks, probes)
 05:20: "live" oracle tests (one test = one claim already written down); 23:45: full test run
```

## Design decisions

- **Hard caps in code, not in prompts.** Per night: 40 model missions, 20 backlog tasks, 1 procedure-extraction mission, an 80 USD spend cap, 2 sterile iterations before the loop exits, 60 minutes of budget for one iteration, and a loop deadline at 04:45 inside a scheduler timeout of 18,600 s. Several of these were raised on purpose by the owner on 2026-08-12 and 2026-08-13 (missions 6 to 15 to 40, spend cap 5 to 30 to 80 USD) because the lower caps stopped nights early. At the current values the mission cap and the spend cap do not bind in normal operation. The deadline does.
- **The writer never merges.** A mission commits to a branch in a worktree. A separate step, with no model call, decides. Reason: an agent that grades its own work reports success whatever it did. Earlier in the project the loop reported "all phases OK" for weeks while measuring only execution.
- **A gate that is conservative on purpose.** The four checks above are cheap and deterministic. Anything doubtful falls back to manual review instead of a blind push. The gate checks that the code parses and merges, not that it is correct.
- **Score the effect, not the volume.** The scorer runs after the fact and reads facts produced by others: commits, backlog status, probe output, usage counts. A mission cannot award itself points. Yield per mission type sets the next night's priority, with an exploration floor so a type that stopped paying can be seen paying again.
- **Review SLA.** A branch nobody decided on is a reminder after 48 hours and is closed after 7 days. The worktree is removed, the branch is deleted with `-d` (git refuses to destroy unmerged work), and the entry is marked abandoned.
- **Append-only journal.** Every branch has a creation line and a closing line in a journal that is never rewritten, because a single-slot status file lost evidence the scorer needed.
- **Peak-hours rule.** Automatic model calls are forbidden between 09:00 and 22:00 except for a short list of named tasks. One of them is a day-time "tick" (3 per day) that skips its turn if an interactive session is open.

## What was measured

Night missions, origin private instance, measured 2026-10-05, window 2026-09-05 to 2026-10-05, from the per-night mission logs:

| Measure | Value |
|---|---|
| Heartbeat runs of the loop | 26 success, 3 skipped (mean 5,235 s per successful run) |
| Nights with at least one mission | 28 of 31 |
| Missions launched | 218 |
| Process-level success flag | 186 of 218 (85 %) |
| Effect measured as verified | 85 (39 %) |
| Effect measured as none | 33 (15 %) |
| Effect unmeasurable | 100 (46 %) |
| Reported cost, total | 421.85 USD (mean 1.94 per mission) |
| Reported cost per night | median 14.42 USD, maximum 57.20 USD (night of 2026-09-21), cap 80 USD never reached |

Per mission type, same window:

| Type | Runs | Process success | Reported cost (USD) |
|---|---|---|---|
| fix failing agents | 87 | 80 | 103.2 |
| backlog task (code in a worktree) | 34 | 31 | 80.4 |
| targeted audit | 26 | 21 | 140.2 |
| procedure extraction | 26 | 23 | 26.3 |
| idea generation | 24 | 15 | 28.5 |
| log investigation | 11 | 10 | 16.4 |
| targeted code review | 10 | 6 | 26.8 |

Review gate, origin private instance, measured 2026-10-05, same window:

| Measure | Value | Source |
|---|---|---|
| Night branches journaled (backlog tasks that produced commits) | 22 | branch journal |
| Outcome | 16 auto-merged, 1 merged by hand, 4 "landed" (the work reached the base branch another way), 1 pending, 0 dropped, 0 abandoned | branch journal |
| Size of those branches | 34 commits, 4,861 lines added, 775 removed | diff stats in the journal |
| Merge ticks at 05:00 | 30 runs: 18 merged, 5 refused, 5 nothing to do, 2 missed | scheduler run history |
| Of the 18 merges | 10 pushed, 8 not pushed because the base branch exists only locally | tick output |
| Difference with the journal | the tick log counts 18 merges, the journal 16 auto-merged; I did not reconcile the two | both sources |
| The 5 refusals | 3 dirty working tree on a file the branch touches, 1 merge conflict, 1 HEAD not on the base branch | tick output |
| Syntax errors caught by the parse check | 0 in the window | tick output |
| Reverts in the instance repository since 2026-09-05 | 0 (308 commits over 28 days) | `git log --grep=revert` |

Day-time autonomous ticks (3 slots per day), same window: 76 successful runs, 4 errors, 9 skipped by the tick's own guards. Their branches go through an in-session review instead of the 05:00 tick: 11 were merged, with a lag of 0 to 12 days (median 2), 1 was rejected (2026-10-01) and 4 were still pending on 2026-10-05, aged 0 to 2 days.

Outcome ledger (one record per autonomous action, with who decided and whether the promised effect was observed), measured 2026-10-05, since it was created on 2026-09-05: 67 records, 55 of them from the day-time ticks. Effect: 13 verified, 27 none, 11 unmeasured, 16 expected. Decision: 8 accepted, 15 pending, 44 not required.

Scorer, origin private instance, measured 2026-10-05, 14-day window, 69 missions evaluated. Points per run: targeted audit 6.2, procedure extraction 5.7, targeted code review 3.5, idea generation 2.5, backlog task 2.3, log investigation 2.0, fix failing agents 0.0 (6 runs, none with a verified effect).

Task backlog on 2026-10-05: 993 tasks (813 done, 87 abandoned, 46 blocked, 44 pending, 2 in progress, 1 awaiting review), 423 of them created in the window. 18 tasks carry an auto-merge result.

## Limits and what failed

- **Half of the missions have no measurable effect.** 100 of 218 are "unmeasured" and 33 are "none". Only 85 (39 %) have an effect someone else could observe. Process success (85 %) is a much weaker number and I do not use it as a quality figure.
- **The most frequent mission is the least productive.** The error-fixing mission accounts for 87 of 218 runs and 103 USD of reported cost, with no verified effect in the scorer's window. On three nights (2026-09-14, 2026-09-15, 2026-09-21) it ran 23 to 27 times in one night (those three nights cost 43.5, 40.3 and 57.2 USD in reported cost, all missions included). The night report of 2026-09-14 shows at least one of those runs concluding that the target agent was not failing and changing nothing. I did not trace what re-triggered it.
- **Tasks that never produced a commit.** Of 34 backlog-task missions, 21 produced at least one commit and 13 produced none (3 of them failed outright). Four tasks were retried two or three times with zero commits each.
- **The merge rate measures the gate, not the quality.** 17 of 22 branches were merged through this path and 21 of 22 reached the base branch. The gate checks that files parse and merge cleanly. It does not run the tests of the changed code before merging. The live oracle suite runs 20 minutes after the merge tick, and the full suite at 23:45.
- **The tests were red often after the merges.** The full suite (about 3,090 tests passing in the runs of 2026-10-03 and 2026-10-04) ended red on 10 of 28 nightly runs (1 to 3 failing tests, 1 run skipped). The live oracle suite (about 170 tests, each one a claim already written down) was red on 10 of 28 runs, 2 to 5 failing. Those are not all regressions caused by a merge: some are documentation claims that stopped being true. I did not attribute each failure to a branch.
- **For a long period the gate merged almost nothing.** Measured on 2026-09-02 from the logs of 2026-08-14 to 2026-09-02 (documented in the code, not re-run): 20 ticks, 2 merges, 1 nothing to do, 17 refusals, all for the same reason, a dirty working tree anywhere in the repository. Two tracked state files rewritten at 03:00 were enough to block it every day. The check was narrowed to the files the branch touches on 2026-09-02. In the day-time pipeline, no branch was merged between 2026-09-06 and 2026-09-24 (18 days), then four were merged in one catch-up on 2026-09-24.
- **Infrastructure refusals look like mission failures.** Five missions between 2026-10-03 and 2026-10-05 were refused by the model provider before doing any work, and one hit a monthly spend limit on 2026-10-03. They count as failures in the table above although the mission logic did nothing wrong. A rule of thumb from my notes: failures with near-identical durations of 11 to 15 seconds and zero cost are the platform, not the mission.
- **Two nights without any mission** (2026-09-16 and 2026-09-17) because the scheduler was down for about 32 hours (0 to 14 runs per hour against roughly 50 normally). No alert fired: the check that should have reported it was itself scheduled by the scheduler. An external timer now watches it.
- **A throttle on its own output.** From 2026-10-03, the audit and idea-generation missions skip their turn when more than 20 suggestions are waiting for a human decision (69 and 68 were waiting on 2026-10-04 and 2026-10-05). Reason: findings were being produced faster than they were triaged.
- **Cost is a reported number.** The CLI's figure is a list-price equivalent for each session. I have no invoice-level cost for the window.
- **Single owner, single machine, one 30-day window.** Nothing here is a benchmark.

## What I would change

Give every mission type a verification step that is not the mission itself before it is allowed to run more than a few times a night, and cap the reactive missions by distinct targets, not by count. Run the changed code's own tests inside the gate rather than after the merge.
