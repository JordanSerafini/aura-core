"""Generic secret masking, applied before text is written to the store.

Two layers, both conservative:

1. `mask_env_values`: exact-value masking. If `AURA_CORE_ENV_FILE` points to a
   dotenv-style file, every value of a credential-looking variable (name
   contains KEY, TOKEN, SECRET, PASSWORD, ...) that is long enough is replaced
   wherever it appears. Zero false positives by construction.
2. `redact`: pattern masking for text that was never in an env file: PEM
   private keys, well-known token prefixes, `name=value` assignments with a
   credential-looking name, `Authorization: Bearer ...`, and credentials
   embedded in URLs.

This is a safety net, not a guarantee. Pattern masking misses secrets with no
recognisable shape, so do not index material you would not want to leak.
"""

import os
import re
from pathlib import Path

MARK = "[REDACTED]"
MIN_VALUE_LENGTH = 10

_CREDENTIAL_NAME = re.compile(
  r"(?i)(?:^|[_-])(?:key|token|secret|password|passwd|pwd|credential|auth)(?:[_-]|$)"
)
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
# `apiKey`), so "monkey" or "tokenizer" do not trigger it.
_ASSIGNMENT = re.compile(
  r"(?i)(?P<key>\b(?:[A-Za-z0-9]+[_.-])*"
  r"(?:(?:key|token|secret|password|passwd|pwd)|(?-i:[a-z0-9]+(?:Key|Token|Secret|Password)))"
  r"(?:[_.-][A-Za-z0-9]+)*\s*[:=]\s*)(?P<val>'[^'\n]+'|\"[^\"\n]+\"|[^\s'\";,]+)"
)


def _env_path() -> Path | None:
  raw = os.environ.get("AURA_CORE_ENV_FILE", "").strip()
  return Path(raw).expanduser() if raw else None


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
    if (_CREDENTIAL_NAME.search(name) and len(value) >= MIN_VALUE_LENGTH
        and value.lower() not in _PLACEHOLDERS):
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


def redact(text: str) -> str:
  """Pattern-based masking. Idempotent."""
  text = _PEM.sub(MARK, text)
  text = _PREFIXED.sub(MARK, text)
  text = _BEARER.sub(lambda m: m.group("key") + MARK, text)
  text = _URL_CREDENTIAL.sub(lambda m: m.group("key") + MARK, text)
  text = _ASSIGNMENT.sub(_mask_assignment, text)
  return text


def has_secret(text: str) -> bool:
  return redact(text) != text
