# A notes vault maintained by the assistant

Case study. The code described here is not in this repository; it stays in the private instance.

Convention: every figure below comes from the origin private instance and was measured on 2026-10-05 unless another date is given. The "last 30 days" window is 2026-09-05 to 2026-10-05.

## What it does

The owner keeps a vault of Markdown notes (opened with Obsidian) that is written to and read by the assistant far more than by the owner. The notes are the assistant's picture of "the state of each project". The rule that shapes everything else is a reciprocity rule: **if a session opens a note to get informed, it must correct that note before it finishes**. A note that was read and left wrong is worse than a missing note, because the next session will believe it.

"Self-rewriting" here means two different things, and they have different reliability:

1. **Generated sections**, rewritten by scheduled scripts with no model call. They sit between marker comments of the form `<!-- aura:<kind>:start -->` and `<!-- aura:<kind>:end -->`. Anything a person or a session writes inside a marker pair is overwritten at the next run.
2. **Hand-written sections**, rewritten by a session when it finds a statement to be false. This one is a convention enforced by an instruction file, not by code. I cannot measure how often it is followed.

## Shape

```
 sources                         scheduled scripts (no model call)            vault
 ------------------------        --------------------------------            ----------------------
 conversation transcripts  --->  session capture (on session end)     --->    Sessions/  (1 note per session)
 git commits of the day    --->  daily maintenance (every 4 h)        --->    Daily/     (marker sections)
 night report, morning     --->  ingest scripts                       --->    Daily/, Logs/
 autonomous action ledger  --->  hourly sync                          --->    Outcomes/  (1 note per action)
                                 nightly roll-up (23:47)              --->    Projects/  (journal, max 5 entries)
                                                                              Todos/     (open and closed)
                                                                              Journal/   (older roll-ups)
        claims in notes  --->  claim check (04:30, reports only)      --->    health report
        whole vault      --->  health check, encrypted backup, verify
```

## Design decisions

- **Frontmatter and a one-paragraph preamble on every note.** Notes are meant to be found alone, through search, by a model. Each carries a type, dates, tags and a short summary for the next reader. Views are built on `type`, never on tags.
- **Markers delimit what a script owns.** A script may rewrite only what is between its own markers. Reason: two writers on one file need a boundary that survives a crash and a human edit.
- **Writes are atomic and detect most concurrent edits.** A shared helper takes a cooperative lock, writes through a temporary file, re-reads the target just before replacing it and, if it changed, keeps the refused version aside instead of overwriting. An external editor or a sync tool does not take the lock, so this narrows the race window, it is not a transaction. Added during a reliability pass on 2026-09-13.
- **The roll-up keeps a note short.** A project note keeps its 5 most recent consolidations. The rest moves to a monthly journal note linked from the project. Reason: one project note had grown to 85 KB, 36 of them journal.
- **Filter secrets at the entrance.** Notes are in a git repository and are synced to a phone, so a secret written once is versioned and replicated. Credential-looking values are masked before text is written, and again as a safety net when the backup is made. Names of variables are allowed, values are not.
- **The claim check reports, it never corrects.** It confronts only mechanically decidable claims (a scheduled job exists, a cited path exists, a cited service is not failing, a port described as local is listening). Rewriting a note on the strength of a regular expression would produce the error it is supposed to find. The first version flagged 22 divergences, about 20 of them false.
- **Counts are not written in notes.** The manual forbids copying a volume figure into prose and says how to count instead, after a note count was wrong four times.
- **A backup that is restored.** A weekly verification (Sunday) actually restores sample files and a git history from an encrypted, deduplicated repository, instead of trusting a green exit code.

## What was measured

Origin private instance, measured 2026-10-05:

| Measure | Value | Source |
|---|---|---|
| Notes | 875 Markdown files, 4.09 MB (excluding the sync tool's version folder, editor configuration and a symlinked memory folder). The health check counts 870. | file walk; health report |
| By folder | Sessions 539, Outcomes 68, Daily 62, Logs 62, Projects 61, Todos 33, Reviews 17, root 13, Journal 12, Inbox 4, Templates 4 | file walk |
| Notes touched | 34 in the last 24 h, 247 in 7 days, 625 in 30 days | file modification times |
| Notes with generated sections | 87 notes, 908 marker comments (start and end counted separately) | regular expression over the vault |
| Marker kinds (comments) | source 169, conversations 117, day summary 108, news watch 108, night works 108, commits 94, night report 58, batch 54, morning summary 51, roll-up 37, other 4 | same |
| Health score | 75 of 100, 0 broken links, 100 % of notes with frontmatter, 1 orphan note, 3.8 links per note on average | health check report |
| Automatic commits in the last 30 days | 194 vault commits, 185 of them automatic backups (about 6 a day) | `git log` on the vault |
| Encrypted backup | 5,031 files, 39.5 MB, last verified on 2026-10-04 by restoring 3 sample files and the git history | backup report |
| Claim check | 179 notes scanned, 187 checkable claims, 1 finding (2026-10-05 04:30) | claim check report |

Generated sections are the bulk of the "rewriting": they are refreshed every few hours, and the per-action outcome notes are regenerated whole by an hourly script. I did not split the 87 notes by folder.

## Limits and what failed

The vault's own manual records its past failures with dates. I re-read it on 2026-10-05 and did not re-run these checks.

- **Real-time capture was dead for days.** The session-end hook started the capture with the system Python interpreter instead of the virtual environment. The import of a native extension failed, the exception was swallowed, and no note was written. Only the weekly backfill fed the session folder, so there was no session note from 2026-09-21 to 2026-09-23. Found and fixed on 2026-09-23.
- **Project attribution was wrong for about five weeks (2026-08-15 to 2026-09-23).** Sessions were attributed to a project by searching keywords in the raw transcript, which includes the instruction files injected at every session start. 143 of 175 sessions attributed to one project did not talk about it. The search now covers the conversation only.
- **A small local model invented tasks.** From 2026-09-06 to 2026-09-23 the roll-up was routed to a 3-billion-parameter local model. Its consolidations invented to-do items in every project note. The route was removed. The roll-ups of that period are to be read with suspicion; I have not rewritten them.
- **Secrets did leak.** On 2026-08-15 a password dictated in conversation had been copied into four notes, along with two API keys, through the capture chain. The filter at the entrance dates from then. The latest secret scan of the vault's git history (2026-10-05) still reports 4 findings: the working notes are filtered, the history is not clean.
- **Counts in notes drifted.** The "number of notes" written in the manual was wrong four times, hence the rule above.
- **A false line can look fresh.** The claim check exists because a note kept stating that a job ran every 30 minutes after that job had been disabled, with a recent `updated:` date. `updated:` dates the last write, not the last verification. The check covers 4 kinds of claim. Everything else (a disk percentage, a priority) is still unchecked.
- **Phone sync is the weak point.** The health check reported degraded phone sync on 2026-10-05. The vault is a single copy plus a sync tool plus a backup, and the manual lists that as a single point of failure.
- **Sessions are 539 of 875 notes.** Most of the vault is session notes, mostly archived ones, which an editor setting excludes from search and the graph. The curated part (projects, to-dos, reviews, journal) is about 120 notes.
- **Not measured:** how often the read-then-correct rule is followed, how many of the 625 recently touched notes were changed by a person rather than a script, and whether the notes actually improved the answers of later sessions. The memory-retrieval benchmark in this repository measures retrieval, not this.
