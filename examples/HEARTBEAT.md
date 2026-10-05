# Example task file

Syntax is documented in `src/aura_core/heartbeat/parser.py`. Every task here is
harmless (echo, date). The file is edited live: the runner picks up changes
within 30 seconds.

## Every 1 minute

- [ ] `echo "tick from the 1-minute list"` -- [local] [timeout:10] prints a line

## Every 5 minutes

- [ ] `date +%FT%T` -- [local] [id:clock] writes the current time
- [ ] `echo "clock ran fine"` -- [local] [depends:clock] only runs if the clock task last succeeded

## Daily at 08:30

- [ ] `echo "good morning"` -- [local] morning greeting

## Weekly on friday at 17:45

- [ ] `echo "weekly wrap-up"` -- [local] end of week marker

## Every 1 hour

- [ ] `echo "this one would call an LLM in a real setup"` -- [tokens] skipped unless --allow-tokens
