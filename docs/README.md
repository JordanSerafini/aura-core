# Documentation: the parts of AURA that are not in this repository

This repository contains three extracted components (hybrid memory, retrieval benchmark, markdown scheduler). The pages below describe four other parts of the private system they came from. They are case studies: no code, a diagram, the design decisions, what was measured, and what did not work.

| Page | Subject |
|---|---|
| [proactive.md](proactive.md) | Suggestions and bubbles that speak first, and the rules that make them stop |
| [night-loop.md](night-loop.md) | An autonomous overnight loop with a mandatory, model-free review step |
| [vault.md](vault.md) | A Markdown notes vault that scripts and sessions keep up to date |
| [mobile-and-mascot.md](mobile-and-mascot.md) | An Android app, a watch app and a desktop mascot on one server |

## How to read the numbers

- Every figure comes from the origin private instance (one owner, one Linux machine) and carries its measurement date. Most were measured on 2026-10-05 over the 30 days from 2026-09-05 to 2026-10-05.
- When a figure could not be measured, the page says so. It is not estimated.
- Sources are named in the tables (a run-history table, a journal, a count over files). They are not reproducible from this repository, because the data and the code stay private.
- I do not call the overnight loop self-improving. The measures support "autonomous loop with review and a scorer", not "gets better by itself". [night-loop.md](night-loop.md) shows the numbers behind that choice.

## The scheduler these pages rely on

The scheduler component in this repository is the small public version of the one that runs the whole instance. Origin private instance, measured 2026-10-05, over the 30 days from 2026-09-05 to 2026-10-05, from the scheduler's run history:

| Measure | Value |
|---|---|
| Active task lines in the task file | 125 (108 marked local, 17 allowed to call a model, 5 of those allowed during working hours); 6 disabled. Jobs registered in the scheduler database: 123. (The root README quotes 65 tasks, an earlier count.) |
| Run records | 63,540 over 135 distinct task keys: 60,464 success, 146 error, 11 timeout, 2,919 skipped |
| Error and timeout rate among executed runs | 0.26 % (157 of 60,621) |
| Skipped share | 4.6 % |

What these numbers do not say: "success" means exit code 0, which is not the same as "useful". A probe that fails open also exits 0, and I fixed several of those during the period. The errors are concentrated in a few tasks (a few jobs that call external services, one probe that started receiving authorization errors, and the daily test runs). About 32 hours of the window (2026-09-16 to 2026-09-17) have almost no runs because the scheduler itself was down.

## Honest summary

| Claim | Evidence in the pages | Verdict |
|---|---|---|
| The system acts proactively | 341 suggestions in 30 days, 371 pushes; no measure of what was worth it | Proven that it speaks, not that it helps |
| It works at night | 218 missions over 28 nights, 39 % with an effect someone else verified | Real work, half of it unmeasurable |
| It is reviewed before merge | 22 branches, 17 merged by the gate, 0 dropped; the gate checks parse and merge only | Review is mechanical, not semantic |
| It keeps a vault up to date | 875 notes, 87 with generated sections, 1 finding in 187 checked claims | Generated parts yes, the hand-written rule is unmeasured |
| It reaches the owner on the phone and desktop | 193 messages in 5 days from one user | Used, by one person |
