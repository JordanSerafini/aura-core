"""Explicit dates, used to retrieve one specific session summary.

A query like "what happened in the session of 2026-03-14" should constrain the
candidate set to the summary whose header carries that date, instead of hoping
the ranker surfaces it. No relative dates ("yesterday") are interpreted.
"""

import re
from datetime import date

_MONTHS = {
  name: month
  for month, names in enumerate(
    [
      ("january", "janvier"), ("february", "fevrier", "février"), ("march", "mars"),
      ("april", "avril"), ("may", "mai"), ("june", "juin"), ("july", "juillet"),
      ("august", "aout", "août"), ("september", "septembre"), ("october", "octobre"),
      ("november", "novembre"), ("december", "decembre", "décembre"),
    ],
    1,
  )
  for name in names
}
_MONTH_ALT = "|".join(_MONTHS)
_DAY_FIRST = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th|er)?\s+(" + _MONTH_ALT + r")\s+(\d{4})\b", re.I)
_MONTH_FIRST = re.compile(r"\b(" + _MONTH_ALT + r")\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b", re.I)
_ISO = re.compile(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)")
_RANGE_WORDS = re.compile(
  r"\b(?:before|after|since|until|between|avant|apres|après|depuis|entre)\s+"
  r"(?:(?:the|le|la date du)\s+)?\d",
  re.I,
)
_HEADER = re.compile(r"^Session [a-f0-9-]+\s+-\s+(\d{4}-\d{2}-\d{2})\s+-", re.I)


def requested_session_date(query: str) -> str | None:
  """Return an ISO date when the query asks for ONE session on ONE explicit date."""
  if not re.search(r"\bsessions?\b", query, re.I) or _RANGE_WORDS.search(query):
    return None
  values = [(int(y), int(m), int(d)) for y, m, d in _ISO.findall(query)]
  values += [(int(y), _MONTHS[m.lower()], int(d)) for d, m, y in _DAY_FIRST.findall(query)]
  values += [(int(y), _MONTHS[m.lower()], int(d)) for m, d, y in _MONTH_FIRST.findall(query)]
  try:
    dates = {date(*parts).isoformat() for parts in values}
  except ValueError:
    return None
  return next(iter(dates)) if len(dates) == 1 else None


def session_header_date(kind: str, content: str) -> str | None:
  """Date in the summary header, never a date quoted in its body.

  Header format: `Session <hex-id> - YYYY-MM-DD - <title>`. Chunks of type
  "episode" may carry a leading `[session/...]` tag line before the header.
  """
  lines = content.splitlines()
  if kind == "episode":
    if not lines or not lines[0].startswith("[session/"):
      return None
    lines = lines[1:]
  elif kind != "session":
    return None
  match = _HEADER.match(lines[0]) if lines else None
  return match.group(1) if match else None
