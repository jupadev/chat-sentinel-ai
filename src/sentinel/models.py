from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Protocol

Verdict = Literal["ok", "flag", "hide"]
Stage = Literal["rules", "encoders", "llm"]

SEVERITY: dict[str, int] = {"ok": 0, "flag": 1, "hide": 2}


@dataclass(frozen=True)
class ChatMessage:
    key: str  # internal UUID
    client_id: str
    message_id: str
    conversation_id: str
    author_id: str
    text: str
    created_at: datetime


@dataclass(frozen=True)
class RuleHit:
    rule: str
    action: Literal["flag", "hide", "ignore"]
    category: str
    match: str


@dataclass
class Decision:
    verdict: Verdict
    category: str | None
    reason: str
    stage: Stage
    risk_score: float | None = None
    scores: dict[str, float] = field(default_factory=dict)
    rule_hits: list[RuleHit] = field(default_factory=list)
    # Risk fell in the [llm_min, llm_max] band. Kept after the LLM answers, to measure how many there are.
    doubtful: bool = False
    latency_ms: dict[str, float] = field(default_factory=dict)


class History(Protocol):
    """Read access to earlier messages, used by the repeat rule and the LLM context."""

    def recent_texts_by_author(self, message: ChatMessage, since: datetime) -> list[str]: ...

    def conversation_context(self, message: ChatMessage, limit: int) -> list[ChatMessage]: ...
