"""Generic secret masking, applied before text is written to the store.

Two layers, both conservative:

1. `mask_env_values`: exact-value masking. If `AURA_CORE_ENV_FILE` points to a
   dotenv-style file, every value of a credential-looking variable (name
   contains KEY, TOKEN, SECRET, PASSWORD, PASS, ...) is replaced wherever it
   appears, if it has 10 characters or more, or 8-9 characters mixing at least
   two kinds (lower case, upper case, digits, others). Zero false positives by
   construction: only exact values from the file are masked.
2. `redact`: pattern masking for text that was never in an env file: PEM
   private keys, well-known token prefixes, `name=value` assignments with a
   credential-looking name (quoted names too, as JSON prints them),
   `Authorization: Bearer ...`, credentials embedded in URLs, and the usual
   command lines that carry a password (`PGPASSWORD=... psql`, `sshpass -p`,
   `mysql -p...`, `curl -u <user>:<password>`, `echo ... | sudo -S`,
   `--password X`).

`clean` applies both. `clean_record` applies it to a JSON-like record value by
value and masks whole the fields whose name holds a credential (store
metadata). Mask a text whole, before it is cut into chunks or
extracts and before it is serialised: a token cut in two, or a `password=`
whose quotes json.dumps has escaped, no longer has a recognisable shape.

This is a safety net, not a guarantee. Pattern masking misses secrets with no
recognisable shape, so do not index material you would not want to leak.
"""

import json
import os
import re
from pathlib import Path

MARK = "[REDACTED]"
MIN_VALUE_LENGTH = 10
# Passwords of 8 or 9 characters are common and were left in clear text. Below 10, a value must mix at
# least two kinds of characters: an 8-letter word (`sentinel`) can be a setting and appear in plain prose.
MIN_MIXED_VALUE_LENGTH = 8

# `pass`, `psswd`, `apikey` and `jeton` (French for token) were real variable names that were never masked.
_CREDENTIAL_NAME = re.compile(
  r"(?i)(?:^|[_-])(?:key|apikey|token|jeton|secret|password|passwd|psswd|pwd|pass|credential|auth)(?:[_-]|$)"
)
# Field names that hold a credential, for records masked field by field (store metadata). Narrower than
# _ASSIGNMENT_KEY on purpose: in a record, `key`, `sort_key` or `auth` are ordinary settings. The credential
# word ends the name, plural included: `access_token` and `api_keys` hold one, `token_count` does not.
_CREDENTIAL_FIELD = re.compile(
  r"(?:^|_)(?:password|passwd|psswd|passphrase|pass|pwd|pw|secret|token|authorization|credential"
  r"|(?:api|secret|access|private)_?key)s?$"
)
# A NUMBER under `tokens` or a usage counter name (`max_tokens`, `input_tokens`, `cache_read_input_tokens`)
# counts a model's tokens. A closed list: `access_tokens: 123456789012` is a credential, not a counter.
_TOKEN_COUNTER = re.compile(
  r"^(?:(?:\w+_)?(?:max|min|input|output|prompt|completion|total|cached|reasoning|context|num|n|used|remaining)_)?"
  r"tokens$"
)
# Upper-case names where the credential word is glued to another one: PGPASSWORD, MYSQLTOKEN, SSHPASS.
_GLUED_NAME = r"[A-Z0-9_]{0,60}(?:PASSWORD|PASSWD|SECRET|TOKEN|API_?KEY)|SSHPASS"
_UNSET = frozenset({"none", "null", "true", "false"})
_PLACEHOLDERS = frozenset({"changeme", "change-me", "example", "placeholder", "your-key-here"})

_PEM = re.compile(
  r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----[\s\S]*?(?:-----END (?:[A-Z0-9]+ )*PRIVATE KEY-----|\Z)"
)
_PREFIXED = re.compile(
  r"\b(?:sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}"
  r"|xox[abprs]-[A-Za-z0-9-]{10,}|AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_-]{30,}"
  r"|eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,})\b"
)
_BEARER = re.compile(r"(?i)(?P<key>\bBearer\s+)(?P<val>[A-Za-z0-9._~+/=-]{12,})")
# The scheme starts where a name would (see _ASSIGNMENT_KEY): from every `\b` of `-a-a-a...`, it was quadratic.
_URL_CREDENTIAL = re.compile(
  r"(?P<key>(?<!\w)(?<![A-Za-z0-9][+.-])[A-Za-z][A-Za-z0-9+.-]*://[^\s:/@'\"]+:)(?P<val>[^\s@/'\"]+)(?=@)"
)
# The credential word must be a whole name segment (`api_key`, `DB_PASSWORD`,
# `apiKey`), so "monkey" or "tokenizer" do not trigger it. Same words as
# _CREDENTIAL_NAME: `auth=` and `credential:` used to slip through. `Auth` is not
# a camelCase suffix on purpose, so "oAuth: enabled" stays readable.
_CREDENTIAL_WORD = re.compile(r"(?i)key|token|secret|password|passwd|pwd|credentials?|auth")
_CREDENTIAL_CAMEL = re.compile(r"[a-z0-9]+(?:Key|Token|Secret|Password|Credentials?)")
# `name: value` or `name=value`, and a quoted name as JSON or a printed dict writes it (`"name": "value"`). The
# name (letters and digits, any alphabet, joined by `_`, `.` or `-`) is read whole, from where it starts (not after
# a word character, nor inside `a-b` or `a.b`), and checked segment by segment in Python; the value is read only
# after a credential name. A single regex looking for the credential word anywhere in the name took cubic time on
# `token-token-token...` (6 s for 3 KB, minutes for 100 KB).
_ASSIGNMENT_KEY = re.compile(
  r"(?<!\w)(?<![^\W_][.-])(?P<name>_*[^\W_]+(?:[_.-]+[^\W_]+)*)(?P<q>['\"])?\s*[:=]\s*"
)
_ASSIGNMENT_VALUE = re.compile(r"'[^'\n]+'|\"[^\"\n]+\"|[^\s'\";,]+")
# The value of a quoted name: a quoted string (escaped quotes included) or a bare scalar, which ends at `}` or `]`.
_FIELD_VALUE = re.compile(r"'(?:[^'\\\n]|\\.)*'|\"(?:[^\"\\\n]|\\.)*\"|[^\s'\",;}\]]+")
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")

# Command lines, where the credential word is glued to another one or the value follows a flag:
#   PGPASSWORD=... psql   sshpass -p '...' ssh   mysql -uroot -p...   curl -u <user>:<password>   echo '...' | sudo -S
#   any-cli --password '...'   any-cli --token ...
# Each form is anchored on an explicit keyword or command, so a bare `-p` or `-u` elsewhere is left alone.
_CMD_VALUE = r"""(?P<val>'[^'\n]*'|"[^"\n]*"|[^\s'"`;|&<>)]+)"""
_GLUED_ENV = re.compile(r"(?<![A-Za-z0-9_])(?P<key>" + _GLUED_NAME + r")(?P<sep>\s*=\s*)" + _CMD_VALUE)
_COMMAND_FORMS = (
  # A long option followed by its value, `--password X` (`--password=X` is an assignment).
  re.compile(r"(?P<key>(?<![\w-])--(?:password|passwd|pass|secret|token|api-?key|access-token|auth-token)\s+)"
             + _CMD_VALUE),
  re.compile(r"(?P<key>\bsshpass\s+(?:-[A-Za-oq-z]\S*\s+)*-p\s*)" + _CMD_VALUE),
  re.compile(r"(?P<key>\b(?:mysql(?:dump|admin|import|check|show)?|mariadb(?:-dump)?)\b[^\n|;&]{0,300}?\s-p)"
             + _CMD_VALUE),
  # `-u`, grouped (`-su`) or long; a quoted `user:password` is masked up to its closing quote.
  re.compile(r"(?P<key>\bcurl\b[^\n|;&]{0,300}?\s(?:-[A-Za-z]*u|--user)[=\s]\s*(?P<q>['\"])?[^\s:'\"]+:)"
             r"(?P<val>(?(q)[^'\"\n]+|[^\s'\"]+))"),
  re.compile(r"(?P<key>\b(?:echo(?:\s+-[neE]+)*|printf(?:\s+'[^'\n]*'|\s+\"[^\"\n]*\")?)\s+)"
             r"(?P<val>'[^'\n]*'|\"[^\"\n]*\"|[^\s|'\"]+)(?=\s*\|\s*sudo\s+(?:-S|--stdin)\b)"),
  re.compile(r"(?P<key>\bsudo\s[^\n|;&<]{0,300}?(?:-S|--stdin)\b[^\n|;&<]{0,300}?<<<\s*)" + _CMD_VALUE),
)
# What a script writes instead of a value, taken whole: `$PASS`, `${PASS}`, `<password>`, `%PASS%`, `{{ pass }}`.
_REFERENCE = re.compile(r"\$\{?[A-Za-z_][A-Za-z0-9_]*\}?|<[^<>\n]*>|%[A-Za-z_][A-Za-z0-9_]*%|\{\{[^{}\n]*\}\}")


def _env_path() -> Path | None:
  raw = os.environ.get("AURA_CORE_ENV_FILE", "").strip()
  return Path(raw).expanduser() if raw else None


def _long_enough(value: str) -> bool:
  if len(value) >= MIN_VALUE_LENGTH:
    return True
  if len(value) < MIN_MIXED_VALUE_LENGTH:
    return False
  kinds = (r"[a-z]", r"[A-Z]", r"[0-9]", r"[^A-Za-z0-9]")
  return sum(1 for kind in kinds if re.search(kind, value)) >= 2


def _known_values() -> tuple[str, ...]:
  """Credential values from the optional env file, longest first. Never logged."""
  path = _env_path()
  if path is None or not path.is_file():
    return ()
  values = set()
  for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
      continue
    name, _, value = line.partition("=")
    name = name.strip().removeprefix("export ").strip()
    value = value.strip().strip("'\"")
    if _CREDENTIAL_NAME.search(name) and _long_enough(value) and value.lower() not in _PLACEHOLDERS:
      values.add(value)
  return tuple(sorted(values, key=len, reverse=True))


def mask_env_values(text: str) -> tuple[str, int]:
  """Replace exact credential values from `AURA_CORE_ENV_FILE`. Returns (text, count)."""
  count = 0
  for value in _known_values():
    if value in text:
      count += text.count(value)
      text = text.replace(value, MARK)
  return text, count


def _credential_name(name: str) -> bool:
  return any(_CREDENTIAL_WORD.fullmatch(part) or _CREDENTIAL_CAMEL.fullmatch(part) for part in re.split(r"[_.-]+", name))


def _json_string(raw: str) -> str:
  # Escapes decoded, so `" $DB_PASS\\n"` is read as the reference it is.
  try:
    value = json.loads(raw)
  except ValueError:
    return raw[1:-1]
  return value if isinstance(value, str) else raw[1:-1]


def _masked_field(text: str, key: re.Match) -> re.Match | None:
  """The value to mask after a quoted name (`"password": "..."`), or None.

  A quoted name is a record field, written by json.dumps or a printed dict, so it is judged like one
  (`secret_field`): `"sort_key"` or `"auth"` stay readable, a number under `"password"` does not, a token counter
  does. A nested object or list (`"credentials": {...}`) is left to `clean_record`, which reads it whole.
  """
  quote = key.group("q")
  if text[key.start() - 1:key.start()] != quote or text[key.end():key.end() + 1] in ("{", "["):
    return None
  if not (found := _FIELD_VALUE.match(text, key.end())):
    return None
  raw = found.group()
  if raw[0] == '"':
    value = _json_string(raw)
  elif raw[0] == "'":
    value = raw[1:-1]
  elif _NUMBER.fullmatch(raw):
    value = float(raw)
  else:
    value = None if raw.lower() in _UNSET else raw
  return found if secret_field(key.group("name"), value) else None


def _mask_assignments(text: str) -> str:
  out, done, start = [], 0, 0
  while key := _ASSIGNMENT_KEY.search(text, start):
    start = key.end()
    if key.group("q"):
      # Masked inside quotes, so the JSON a store writes stays valid: `"password": "[REDACTED]"`.
      if found := _masked_field(text, key):
        quote = found.group()[0] if found.group()[0] in "'\"" else key.group("q")
        out += [text[done:start], quote + MARK + quote]
        done = start = found.end()
      continue
    if not _credential_name(key.group("name")):
      continue
    # Leave references and obvious placeholders alone: `token=$TOKEN`, `key=<your key>`. A value left in clear
    # is still read for names, so `<api_key:...` is not a way around the mask. A reference is told by its first
    # character, without reading it to its end: `key:<key:<...` would be quadratic.
    if text[start:start + 2].lstrip("'\"")[:1] in ("$", "<", "{", "%"):
      continue
    if not (found := _ASSIGNMENT_VALUE.match(text, start)):
      continue
    value = found.group().strip("'\"")
    if value.startswith(("$", "<", "{", "%")) or value.lower() in _PLACEHOLDERS or len(value) < 6:
      continue
    out += [text[done:start], MARK]
    done = start = found.end()
  return "".join(out) + text[done:]


def _is_reference(value: str) -> bool:
  """A reference, a placeholder or a value already masked: nothing to hide. `$uper$ecret` is not a reference."""
  return (not value or MARK in value or value.lower() in _PLACEHOLDERS
          or _REFERENCE.fullmatch(value) is not None)


def _mask_command(match: re.Match) -> str:
  raw = match.group("val")
  value = raw.strip("'\"")
  # The shell does not expand `$` between single quotes: there, `'$name'` is the password itself.
  if _is_reference(value) and not (raw.startswith("'") and value.startswith("$")):
    return match.group(0)
  return match.group("key") + (match.groupdict().get("sep") or "") + MARK


def _mask_glued_env(match: re.Match) -> str:
  # Same length bar as _mask_assignment, which also leaves `None`, `null`, `true` and `false` alone.
  if len(match.group("val").strip("'\"")) < 6:
    return match.group(0)
  return _mask_command(match)


def redact(text: str) -> str:
  """Pattern-based masking. Idempotent."""
  text = _PEM.sub(MARK, text)
  text = _PREFIXED.sub(MARK, text)
  text = _BEARER.sub(lambda m: m.group("key") + MARK, text)
  text = _URL_CREDENTIAL.sub(lambda m: m.group("key") + MARK, text)
  text = _GLUED_ENV.sub(_mask_glued_env, text)
  for form in _COMMAND_FORMS:
    text = form.sub(_mask_command, text)
  return _mask_assignments(text)


def clean(text: str) -> str:
  """Both layers: exact values from `AURA_CORE_ENV_FILE`, then patterns."""
  text, _ = mask_env_values(text)
  return redact(text)


def _snake(name: str) -> str:
  return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name).lower().replace("-", "_").replace(".", "_")


def credential_field(name: str) -> bool:
  """True for a field name that holds a credential: `password`, `DB_PASSWORD`, `apiKey`, `PGPASSWORD`,
  and their plurals (`passwords`, `api_keys`)."""
  if re.fullmatch(_GLUED_NAME, name):
    return True
  return bool(_CREDENTIAL_FIELD.search(_snake(name)))


def secret_field(name: str, value: str | int | float | bool | None) -> bool:
  """True when a record field is to be masked whole.

  Under a credential name the value is the secret: a string, short or with spaces, unless it is a
  reference, a placeholder or an unset marker (`None`, `null`); a number too (`password: 12345678`),
  except under a name that ends with `tokens`, a model's token counter (`max_tokens: 4096`). A bool or
  None is never a secret.
  """
  if value is None or isinstance(value, bool) or not credential_field(name):
    return False
  if isinstance(value, (int, float)):
    return not _TOKEN_COUNTER.search(_snake(name))
  value = value.strip()
  return not _is_reference(value) and value.lower() not in _UNSET


def clean_record(value, name: str = ""):
  """Mask a JSON-like record (dicts, lists, plain values) field by field, before it is serialised.

  Dumped first, quotes and newlines come out escaped and `{"api_key": v}` becomes `"api_key": "v"`: the
  patterns no longer see a `name=value`. So each string is masked on its own (`clean`), then whole when its
  field name holds a credential (`secret_field`). A credential name carries down: below `credentials` or
  `api_keys`, every value is judged under that name, so `{"credentials": {"user": "bob"}}` is masked and
  `{"tokens": {"input": 12}}` stays a counter. Below any other name, a value is judged under its own key, and
  a list passes its name to its items.
  """
  if isinstance(value, dict):
    inherited = name if credential_field(name) else ""
    return {k: clean_record(v, inherited or str(k)) for k, v in value.items()}
  if isinstance(value, list):
    return [clean_record(v, name) for v in value]
  if isinstance(value, str):
    value = clean(value)
  elif not isinstance(value, (int, float)):  # None, or a type json.loads does not produce: left as is
    return value
  return MARK if secret_field(name, value) else value


def has_secret(text: str) -> bool:
  return redact(text) != text
