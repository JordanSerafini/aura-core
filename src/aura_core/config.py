"""Paths and environment switches.

Everything the package writes lives under one directory, `AURA_CORE_HOME`
(default: `~/.aura_core`). Nothing is read from or written to any other place.
"""

import os
from pathlib import Path


def home() -> Path:
  """Root directory for local state (databases, scheduler history)."""
  return Path(os.environ.get("AURA_CORE_HOME", Path.home() / ".aura_core")).expanduser()


def default_memory_db() -> Path:
  return home() / "memory.db"


def default_heartbeat_db() -> Path:
  return home() / "heartbeat.db"
