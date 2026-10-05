# Proactive signals

Case study. The code described here is not in this repository; it stays in the private instance.

Convention: every figure below comes from the origin private instance and was measured on 2026-10-05, over the 30 days from 2026-09-05 to 2026-10-05, unless another date is given.

## What it does

A personal assistant that only answers when asked wastes the one thing it can do while nobody is looking: notice. The proactive layer has two halves that share a rule (say less, say it once, stop when ignored):

1. **A suggestion engine** (a scheduled Python agent, no model call). About twenty detectors read system state, repositories, backups, dependency advisories, goals with deadlines and the previous night's findings. Each detector emits a suggestion with a stable key, a priority and an optional action.
2. **Desktop and phone bubbles** (a rules module in the companion app server, no model call). It turns presence, calendar and health data into a short message: a meeting in 10 minutes, a summary when the owner comes back after 20 minutes away, an end-of-day list of commitments, a full system disk.

Neither half calls a language model to decide whether to speak. The model is used later, when the owner reacts or when the nightly loop picks up a finding (see [night-loop.md](night-loop.md)).

## Shape

```
 detectors (19)            state file (dedup by key)           channels
 ----------------          ---------------------------         ------------------
 git, disk, backups  --->  new -> pushed -> acknowledged  ---> chat message (max 3 per run)
 deps, probes, goals       snoozed 7 d / dismissed / resolved   digest for low priority
 night findings            reopened when the cause returns      companion app bubbles
                                   |
                                   v
                      engagement per category (last 12 decisions)
                      >= 80 % ignored  ->  category goes quiet in push
```

```
 presence (5 s) + health (5 min) + work context (15 min)
        |
        v
 5 local rules  ->  one key = one bubble, never twice
 agenda / return / start of day / end of day / health
        |
        +-- anti-spam: at most one non-urgent bubble per 15 min
        +-- the owner's work hours decide which machine is expected to be active
```

## Design decisions

- **Dedup by key, with a memory.** A suggestion that was acknowledged, snoozed or dismissed does not come back on its own. The state file keeps decisions for a year and detections for 30 days. Reason: a warning that was already decided on should not cost attention a second time.
- **A chronic problem is reminded slowly, not silenced.** After three reminders a still-true suggestion is repeated every 7 days (high) or 14 days (medium). A stale-project warning had been detected 122 times and had pushed nothing for 12 days, which reads as "fixed". That was the failure this rule replaces.
- **Ignored categories go quiet.** Each reaction is recorded per category. When at least 6 of the last 12 decisions exist and 80 % or more were "ignore", the category stops pushing (it stays visible in a weekly digest and a list). Reason: a sliding window, because an all-time counter let one old "act" click keep a category loud forever.
- **Push cap.** At most 3 messages per run. Low priority goes to a digest, not to unit messages.
- **Rules, not a model, decide to speak.** The bubble rules are plain code with thresholds in constants, tested with a fake clock. Reason: a rule can be tested and explained in one line; a token budget policy also forbids automatic model calls during the owner's working hours, except for a short list of named tasks.
- **Take the owner's schedule as an input.** Work hours and travel days are data. A machine that is idle outside its expected slot is normal and triggers nothing.

## What was measured

Engine, origin private instance, measured 2026-10-05, window 2026-09-05 to 2026-10-05:

| Measure | Value | Source |
|---|---|---|
| Analysis runs | 214 success, 14 skipped, mean 7.2 s | scheduler run history (`task_runs`) |
| Detectors in the engine | 19 `analyze_*` functions | static count in the agent |
| Distinct suggestions first seen | 341 (169 medium, 89 low, 83 high) | engine state file |
| Messages pushed to the chat channel | 371, covering 177 of the 341 suggestions | `pushes` field of the state file |
| Status at measurement | 139 acknowledged, 122 resolved because the condition disappeared, 41 pushed with no reaction, 22 new, 17 explicitly dismissed | state file |
| Re-opened after being closed | 31 events | suggestion event log |
| Suggestions flagged for automatic execution | 0 of 374 creation events | event log, `auto_execute` field |

Reading these: 122 of 341 (36 %) closed themselves, which says the detectors mostly describe transient states. 17 of 341 (5 %) were explicitly dismissed, and 41 (12 %) were pushed and never touched. "Acknowledged" (139) is ambiguous: the same status is set when a suggestion is sent to the task backlog, so it is not a pure "useful" signal.

Companion app, origin private instance, measured 2026-10-05:

| Measure | Value | Source |
|---|---|---|
| Notification history kept by the server | 300 entries, 2026-10-02 to 2026-10-05 | server notification file |
| Of those, routed to "quiet" (no bubble, no push, no badge) | 116 | `quiet` flag |
| Explicitly visible | 40 (144 carry no flag) | `quiet` flag |
| Reminders created since the feature shipped | 17 (12 done, 4 cancelled, 1 pending), first on 2026-09-28 | reminders file |

## Limits and what failed

- **The autonomous execution path never ran in the window.** The engine can auto-run a whitelist of safe actions. None of the 374 suggestions created in 30 days was flagged for it; two suggestions overall carry the status "executed". The feature exists and is tested, but I cannot claim it does anything in practice.
- **The engagement counter is not a live measure.** The per-category file reports 98 "acted" against 12 "ignored" across 9 categories, but every category has the same last-update timestamp (2026-09-24 15:43), which looks like a one-time rebuild. I do not use it as evidence that suggestions were useful.
- **"What the owner did with it" is only partly measurable.** The status field tells whether a suggestion was dismissed or resolved, not whether acting on it was worthwhile. I have no time-to-action measure.
- **Bubble volume is not measurable over 30 days.** The bubble module keeps its sent keys for 2 days on purpose (9 keys on 2026-10-05), and the notification file is capped at 300 entries.
- **Notification spam was real.** Before a triage step was added on 2026-09-25, the desktop app mirrored the chat channel one for one: 60 to 75 notifications a day, which the owner called noise. The triage (suppress duplicates and cancelled meetings, send diagnostics and test results to a quiet tab, shorten the text) measured 34 notifications in 17 hours reduced to 2 or 3 visible, on that day's history. That figure is from my notes of 2026-09-25, not re-run.
- **A cancelled meeting produced three reminders** before cancelled events were excluded from the reminder rules.
- **The scheduler itself went quiet for about 32 hours** (2026-09-16 to 2026-09-17, 0 to 14 runs per hour against roughly 50 normally) and nothing alerted, because the watchdog that should have reported it was scheduled by the thing it watched. The fix was an external timer. A proactive system is only as good as the check that it is running.
- **Single user.** Every number is one person's behaviour on one setup. Nothing here is a generalisation.

## What I would do differently

Record the reaction (open, act, dismiss, ignore) as one event type from the start, with a timestamp, instead of overloading "acknowledged". That would make "what did the owner do with it" a query instead of an estimate.
