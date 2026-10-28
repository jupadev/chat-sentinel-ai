import json
from dataclasses import dataclass

import httpx

from sentinel.config import LlmConfig
from sentinel.models import ChatMessage, RuleHit

CATEGORIES = ["none", "insult", "harassment", "hate", "threat", "sexual", "spam", "contact_sharing", "other"]

SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["ok", "flag", "hide"]},
        "category": {"type": "string", "enum": CATEGORIES},
        "reason": {"type": "string"},
    },
    "required": ["verdict", "category", "reason"],
}

SYSTEM_PROMPT = """You moderate a private chat inside {platform}. Messages are short and informal \
Rioplatense Spanish, with slang, emojis, abbreviations and typos.

Decide about the TARGET message only. Earlier messages are context.

- "ok": normal talk, coordination, hard but legitimate feedback, criticism of work or ideas, jokes and \
friendly banter between people who clearly get along. Words like "boludo" or "che, sos un genio \
pelotudo" are usually affectionate in Argentina.
- "flag": a human should look at it: insults aimed at the other person, humiliation, harassment, \
sexual comments or advances, pressure, threats, discrimination, spam.
- "hide": only for extreme content: slurs, explicit threats of violence, explicit sexual content.

When unsure between "ok" and "flag", choose "flag". When unsure between "flag" and "hide", choose "flag".
Answer with JSON: verdict, category and a short reason in Spanish (max 20 words)."""


@dataclass(frozen=True)
class LlmVerdict:
    verdict: str
    category: str
    reason: str


def build_user_prompt(message: ChatMessage, context: list[ChatMessage], rule_hits: list[RuleHit]) -> str:
    aliases: dict[str, str] = {}

    def alias(author_id: str) -> str:
        return aliases.setdefault(author_id, f"Persona {chr(ord('A') + len(aliases))}")

    lines = ["Earlier messages:"] if context else ["No earlier messages."]
    lines += [f"{alias(m.author_id)}: {m.text}" for m in context]
    lines.append("")
    lines.append(f"TARGET message, {alias(message.author_id)}: {message.text}")
    if rule_hits:
        lines.append("")
        lines.append("Keyword hints (may be friendly): " + ", ".join(sorted({h.match for h in rule_hits})))
    return "\n".join(lines)


class OllamaJudge:
    def __init__(self, config: LlmConfig, client: httpx.Client | None = None):
        self._config = config
        self._client = client or httpx.Client(base_url=config.base_url, timeout=config.timeout_seconds)

    def judge(self, message: ChatMessage, context: list[ChatMessage], rule_hits: list[RuleHit]) -> LlmVerdict:
        response = self._client.post(
            "/api/chat",
            json={
                "model": self._config.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT.format(platform=self._config.platform)},
                    {"role": "user", "content": build_user_prompt(message, context, rule_hits)},
                ],
                "format": SCHEMA,
                "think": self._config.think,
                "stream": False,
                "keep_alive": self._config.keep_alive,
                "options": {"temperature": 0, "num_ctx": self._config.num_ctx, "num_thread": self._config.num_thread},
            },
        )
        response.raise_for_status()
        data = json.loads(response.json()["message"]["content"])
        return LlmVerdict(data["verdict"], data["category"], data["reason"])

    def warm_up(self) -> None:
        """Load the model into memory so the first real message does not pay for it."""
        self._client.post(
            "/api/generate", json={"model": self._config.model, "keep_alive": self._config.keep_alive}
        ).raise_for_status()
