# Phone app, watch app and desktop mascot

Case study. The code described here is not in this repository; it stays in private repositories. No screenshots are included.

Convention: every figure below comes from the origin private instance and was measured on 2026-10-05 unless another date is given. Line counts are physical lines in tracked files, node dependencies and build output excluded. Test counts are static counts of test annotations or `test_` functions, not a test run.

## What it does

The assistant runs on one Linux machine. Three thin clients reach it:

- **A desktop mascot** (Qt with QML) on two computers: a small animated robot, a chat window, an on-demand screen capture with text recognition, and a few local rules that raise short bubbles (see [proactive.md](proactive.md)).
- **An Android app** (React Native with a local Kotlin module). Its main tab is the same web client the desktop serves, in a WebView. The native part adds what a web page cannot do: a foreground service that keeps the connection alive, a hands-free voice conversation, a default-assistant overlay, a share target, call and notification events, and a small set of phone actions the assistant may request.
- **A Wear OS watch app**, a voice remote that talks only to the phone.

The brain does not move. Model sessions run on the Linux machine, behind a small WebSocket server (the "bridge"). A request made on a laptop or a phone executes on that machine. If the machine is off, the clients show "offline".

## Shape

```
  watch (Kotlin)  --data layer-->  phone (RN + Kotlin)  --wss, private VPN mesh-->  bridge (Python, WebSocket + HTTP)
                                   desktop mascot (Qt)  --wss------------------->        |
                                   web client (PWA)     --wss------------------->        |
                                                                                         v
                                                          per-device token, one session per conversation
                                                                                         |
                                                          model CLI session  <---  tools, files, local scripts
                                                                                         |
                          phone actions (request / confirm / result) <-----------------+
```

## Design decisions

- **One protocol, three clients.** A single document describes every message (about 1,100 lines, started the day the phone app was). The existing web client stayed the reference client, and messages were added, not changed. Every message carries an id so a refusal can free a client that would otherwise stay "busy" forever.
- **The server is the authority on risky actions.** Six outgoing actions (send a text, place a call, reply to a notification, send a chat message, add a calendar entry, compose an email) require a confirmation on the phone by default. The app may only make this stricter. Outgoing actions are capped at 30 per rolling hour and every action is written to a local log.
- **Phone content is untrusted data.** Text read from notifications or messages is wrapped as "untrusted data" before it reaches a model, with an instruction never to follow it.
- **Control of other apps is a whitelist.** Accessibility-based taps and typing work only in apps on a short list. Banking, payment, passwords, authenticators, settings and the app store are refused. A pause switch (15 minutes, 1 hour, or until resumed) cuts screen reading and app control, or everything, and applies locally even when offline.
- **Events are filtered at the source.** The phone sends an event type only if a rule is active for it. The rest stays visible in a local journal as "filtered".
- **Voice is generated on the server.** Text-to-speech runs on the Linux machine with the same voice and pronunciation fixes as the desktop, and is streamed to the requesting device in chunks. On 2026-09-24 I measured a first sound after 1.3 s and 14 s of audio generated in 2.9 s (my notes of that day, not re-run).
- **Usage counters, numbers only.** Each client reports counters whose names match a strict pattern. A report with an invalid name is refused whole, a client cannot invent devices, and the file keeps 90 days. Reason: on 2026-10-01 I had no way to tell whether the animated extras were used at all.
- **Restart only when idle.** The server carries the model session itself, so a restart in the middle of a turn kills it. Restarts wait for an "idle" health signal.
- **Heavy builds run outside the server's resource group**, with their own memory cap.

## What was measured

Code size, origin private instance, measured 2026-10-05:

| Piece | Measure | Source |
|---|---|---|
| Phone and watch repository | 194 tracked files, 55 commits (history was rewritten once locally to remove commit trailers, so dates are post-rewrite) | `git ls-files`, `git log` |
| Kotlin | 119 files, 16,899 lines; of which the phone's native module 75 files and 12,624 lines, the watch app 44 files and 4,275 lines | line count |
| Kotlin tests | 25 test files, 3,244 lines; 333 test annotations (phone 260, watch 73) | static count |
| TypeScript and JavaScript | 21 files, 2,245 lines (the phone UI is mostly the web client) | line count |
| Release APK | about 28 MB, version 1.1.0 built 2026-10-03 | file size |
| Desktop mascot | 41 non-test Python files and 9,624 lines, 11 QML files and 7,601 lines; 92 commits since 2026-09-24 | line count, `git log` |
| Mascot tests | 79 Python test files, 15,782 lines, 892 test functions | static count |
| Bridge server | 47 Python files, 14,485 lines; web client 7 files, 7,329 lines (JavaScript, HTML, CSS) | line count |
| Bridge tests | 54 test files, 1,118 test functions | static count |

Use, origin private instance, measured 2026-10-05:

| Measure | Value | Source |
|---|---|---|
| Usage counters | 5 days (2026-10-01 to 2026-10-05), 3 devices, 72 counter names, 1,195 increments | usage counter file |
| Conversations | chat opened 321 times, 193 messages sent (137 of them on 2026-10-02, none on 2026-10-04) | same |
| Decorative extras | animated "numbers" account for 96 of 1,195 increments (8 %) | same |
| Reminders created | 17 since 2026-09-28 (12 done, 4 cancelled, 1 pending) | reminders file |
| Phone actions logged | 4,523 between 2026-09-26 and 2026-10-05; 4,069 succeeded (90 %), 454 failed | action log |
| Of those, notification list | 4,432: automatic polling by the notification triage, not user requests | action log |
| Other actions | 91 (device status 41, battery of the watch 8, calendar entry 7, screen read 6, call log 5, others 24) | action log |
| Failures | 437 no device connected, 8 timeout, 4 disconnected, 2 refused by the owner, 3 other | action log |
| Median action latency | 125 ms | action log |
| Actions that required a confirmation | 12 | action log |

## Limits and what failed

- **One user, five days of counters.** The usage figures describe one person for five days, in the first week after the counters were added. They say which features were touched, not which are useful.
- **Most of the action log is polling.** 98 % of the logged phone actions are a notification list requested by a background job. Do not read 4,523 as "4,523 things the assistant did".
- **The phone is often unreachable.** 437 requests found no connected device. The assistant cannot act on a phone that is off the network, and a watch connects only through the phone.
- **A restart killed a session and a commit was lost** (2026-09-26 18:38): a session asked the server to restart itself from inside a conversation. The restart ran anyway. This is why restarts now wait for idle.
- **A build took the machine down** (2026-09-26): a Gradle build launched from a session ran inside the server's memory-limited group, filled the swap, pushed the load to about 40 and the machine rebooted at 23:37. Builds now run in their own transient unit with a memory cap.
- **Two agents edited the same files** (2026-09-24 17:19): the mascot's own session launched a second coding agent that modified the bridge and, through file sync, the client on two computers, while a terminal session worked on the same files. Rule since then: list running agents before editing these files. Three sessions edited them at once on 2026-09-25 and coordination through messages worked, but it was a habit, not a mechanism. The mascot can change its own code; I treat that as a risk, not a feature.
- **Bugs that desktop testing could not show.** The "listen" button was silent on the phone because audio started 1 to 2 s after the tap, which the mobile autoplay rule rejects. A pop-up closed immediately under a finger because the same tap produced a click on the "close" button under it. Both were invisible with a mouse. They were reproduced only with a strict autoplay policy and real touch events, and fixed in the 2026-09-24 pass.
- **An audit found real defects.** A three-way read-only audit on 2026-09-25 gave 30 findings, 23 reproduced; for example, protocol errors were sent without an id, so a client stayed "busy" for good on a text over 20,000 characters. The tests written for them were red on the old code (my notes of that day, not re-run).
- **An audit run could write to the real settings.** An audit that instantiated the real controller wrote the mascot's configuration file on the dev machine. Rule since then: redirect the settings path in every fixture that touches the controller.
- **The Windows computer is seen but not driven.** The assistant reads its context but cannot act on it. A command typed on that computer runs on the Linux machine.
- **Not measured:** the phone's battery cost (the protocol document estimates under 1 % a day for the reconnection watchdog, which I have not measured), crash rates, end-to-end latency of a model turn, and how many bubbles or suggestions the owner acted on.
