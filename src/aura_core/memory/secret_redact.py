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
   credential-looking name, `Authorization: Bearer ...`, credentials
   embedded in URLs, and the usual command lines that carry a password
   (`PGPASSWORD=... psql`, `sshpass -p`, `mysql -p...`, `curl -u <user>:<password>`,
   `echo ... | sudo -S`).

`clean` applies both. Mask a text whole, before it is cut into chunks or
extracts and before it is serialised: a token cut in two, or a `password=`
whose quotes json.dumps has escaped, no longer has a recognisable shape.

This is a safety net, not a guarantee. Pattern masking misses secrets with no
recognisable shape, so do not index material you would not want to leak.
"""

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
# _ASSIGNMENT on purpose: in a record, `key`, `sort_key` or `auth` are ordinary settings. The credential
# word ends the name: `access_token` holds one, `token_count` does not.
_CREDENTIAL_FIELD = re.compile(
  r"(?:^|_)(?:password|passwd|psswd|passphrase|pass|pwd|pw|secret|token|authorization|credentials?"
  r"|(?:api|secret|access|private)_?key)$"
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
_URL_CREDENTIAL = re.compile(r"(?P<key>\b[A-Za-z][A-Za-z0-9+.-]*://[^\s:/@'\"]+:)(?P<val>[^\s@/'\"]+)(?=@)")
# The credential word must be a whole name segment (`api_key`, `DB_PASSWORD`,
# `apiKey`), so "monkey" or "tokenizer" do not trigger it. Same words as
# _CREDENTIAL_NAME: `auth=` and `credential:` used to slip through. `Auth` is not
# a camelCase suffix on purpose, so "oAuth: enabled" stays readable.
_ASSIGNMENT = re.compile(
  r"(?i)(?P<key>\b(?:[A-Za-z0-9]+[_.-])*"
  r"(?:(?:key|token|secret|password|passwd|pwd|credentials?|auth)|(?-i:[a-z0-9]+(?:Key|Token|Secret|Password|Credentials?)))"
  r"(?:[_.-][A-Za-z0-9]+)*\s*[:=]\s*)(?P<val>'[^'\n]+'|\"[^\"\n]+\"|[^\s'\";,]+)"
)

# Command lines, where the credential word is glued to another one or the value follows a flag:
#   PGPASSWORD=... psql   sshpass -p '...' ssh   mysql -uroot -p...   curl -u <user>:<password>   echo '...' | sudo -S
# Each form is anchored on an explicit keyword or command, so a bare `-p` or `-u` elsewhere is left alone.
_CMD_VALUE = r"""(?P<val>'[^'\n]*'|"[^"\n]*"|[^\s'"`;|&<>)]+)"""
_GLUED_ENV = re.compile(r"(?<![A-Za-z0-9_])(?P<key>" + _GLUED_NAME + r")(?P<sep>\s*=\s*)" + _CMD_VALUE)
_COMMAND_FORMS = (
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


def _mask_assignment(match: re.Match) -> str:
  value = match.group("val").strip("'\"")
  # Leave references and obvious placeholders alone: `token=$TOKEN`, `key=<your key>`.
  if value.startswith(("$", "<", "{", "%")) or value.lower() in _PLACEHOLDERS or len(value) < 6:
    return match.group(0)
  return match.group("key") + MARK


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
  text = _ASSIGNMENT.sub(_mask_assignment, text)
  return text


def clean(text: str) -> str:
  """Both layers: exact values from `AURA_CORE_ENV_FILE`, then patterns."""
  text, _ = mask_env_values(text)
  return redact(text)


def credential_field(name: str) -> bool:
  """True for a field name that holds a credential: `password`, `DB_PASSWORD`, `apiKey`, `PGPASSWORD`."""
  if re.fullmatch(_GLUED_NAME, name):
    return True
  snake = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name).lower().replace("-", "_").replace(".", "_")
  return bool(_CREDENTIAL_FIELD.search(snake))


def secret_field(name: str, value: str) -> bool:
  """True when a record field is to be masked whole: under a credential name, the value is the secret,
  short or with spaces, unless it is a reference, a placeholder or an unset marker (`None`, `null`)."""
  value = value.strip()
  return credential_field(name) and not _is_reference(value) and value.lower() not in _UNSET


def has_secret(text: str) -> bool:
  return redact(text) != text
