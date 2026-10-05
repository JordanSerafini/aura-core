"""Memory data types: episodes, skills, knowledge triples and scoring.

Plain dataclasses with dict round-tripping. They carry no storage logic; they
describe what a multi-level agent memory might hold (episodic, procedural,
semantic/graph) and how a recalled item is scored.
"""

import hashlib
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any


class MemoryType(Enum):
  CORE = auto()        # identity, fundamental preferences
  EPISODIC = auto()    # past interactions with full context
  SEMANTIC = auto()    # facts, concepts
  PROCEDURAL = auto()  # learned skills and patterns
  WORKING = auto()     # current conversation context
  KNOWLEDGE = auto()   # knowledge graph (triples)


class MemoryPriority(Enum):
  CRITICAL = 5
  HIGH = 4
  NORMAL = 3
  LOW = 2
  EPHEMERAL = 1


class MemoryStatus(Enum):
  ACTIVE = auto()
  ARCHIVED = auto()
  CONSOLIDATED = auto()
  DEPRECATED = auto()


def _now() -> str:
  return datetime.now().isoformat()


def _short_hash(data: str) -> str:
  return hashlib.sha256(data.encode()).hexdigest()[:12]


@dataclass
class MemoryMetadata:
  created_at: str = field(default_factory=_now)
  updated_at: str = field(default_factory=_now)
  access_count: int = 0
  last_accessed: str | None = None
  source: str = "user"  # user, system, agent, consolidation
  tags: list[str] = field(default_factory=list)
  priority: int = MemoryPriority.NORMAL.value
  status: str = MemoryStatus.ACTIVE.name

  def to_dict(self) -> dict[str, Any]:
    return asdict(self)

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> "MemoryMetadata":
    return cls(**data)


class _Serializable:
  """Shared dict round-trip for dataclasses that carry a `metadata` field."""

  def to_dict(self) -> dict[str, Any]:
    d = asdict(self)  # type: ignore[call-overload]
    d["metadata"] = self.metadata.to_dict()  # type: ignore[attr-defined]
    return d

  @classmethod
  def from_dict(cls, data: dict[str, Any]):
    values = data.copy()
    if isinstance(values.get("metadata"), dict):
      values["metadata"] = MemoryMetadata.from_dict(values["metadata"])
    return cls(**values)


@dataclass
class Episode(_Serializable):
  """One interaction with its full context."""
  id: str
  timestamp: str
  context: str
  action: str
  outcome: str
  thought_process: str
  entities: list[str]
  emotional_valence: float = 0.0  # -1 (negative) to +1 (positive)
  importance: float = 0.5         # 0 to 1
  metadata: MemoryMetadata = field(default_factory=MemoryMetadata)

  def __post_init__(self):
    if not self.id:
      self.id = "ep_" + _short_hash(f"{self.timestamp}:{self.context[:50]}:{self.action[:50]}")


@dataclass
class Skill(_Serializable):
  """A learned pattern, typically distilled from successful episodes."""
  id: str
  name: str
  description: str
  pattern: str
  trigger_conditions: list[str]
  action_template: str
  success_rate: float = 0.0
  usage_count: int = 0
  source_episodes: list[str] = field(default_factory=list)
  metadata: MemoryMetadata = field(default_factory=MemoryMetadata)

  def __post_init__(self):
    if not self.id:
      self.id = "sk_" + _short_hash(f"{self.name}:{self.pattern[:50]}")


@dataclass
class KnowledgeTriple(_Serializable):
  """subject -> predicate -> object, for a knowledge graph."""
  id: str
  subject: str
  predicate: str
  object: str
  confidence: float = 1.0
  source_episode: str | None = None
  metadata: MemoryMetadata = field(default_factory=MemoryMetadata)

  def __post_init__(self):
    if not self.id:
      self.id = "kg_" + _short_hash(f"{self.subject}:{self.predicate}:{self.object}")

  def to_text(self) -> str:
    """Text form used for embedding."""
    return f"{self.subject} {self.predicate} {self.object}"


@dataclass
class MemoryScore:
  """Relevance of a recalled item: similarity, importance, recency, access frequency."""
  similarity: float
  importance: float
  recency: float
  access_frequency: float = 0.0

  @property
  def combined_score(self) -> float:
    """similarity x 0.4 + importance x 0.3 + recency x 0.2 + frequency x 0.1."""
    return (self.similarity * 0.4 + self.importance * 0.3
            + self.recency * 0.2 + self.access_frequency * 0.1)

  def to_dict(self) -> dict[str, Any]:
    return {
      "similarity": self.similarity,
      "importance": self.importance,
      "recency": self.recency,
      "access_frequency": self.access_frequency,
      "combined": self.combined_score,
    }


@dataclass
class ConsolidationResult:
  episodes_processed: int
  skills_created: int
  skills_updated: int
  triples_extracted: int
  episodes_archived: int
  timestamp: str = field(default_factory=_now)
  details: dict[str, Any] = field(default_factory=dict)


def calculate_recency_score(timestamp: str, decay_days: int = 30, now: datetime | None = None) -> float:
  """Exponential decay exp(-age / decay_days): 1.0 now, about 0.37 after `decay_days`.

  `decay_days` is the e-folding time, not a half-life. Unparseable timestamps
  score a neutral 0.5.
  """
  try:
    then = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    ref = now or (datetime.now(then.tzinfo) if then.tzinfo else datetime.now())
    age_days = (ref - then).total_seconds() / 86400
    return math.exp(-age_days / decay_days)
  except (ValueError, AttributeError, TypeError):
    return 0.5


def generate_memory_id(prefix: str, content: str) -> str:
  """Time-salted ID: the same content stored twice gets two IDs."""
  return f"{prefix}_{_short_hash(f'{_now()}:{content[:100]}')}"
